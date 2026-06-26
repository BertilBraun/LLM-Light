from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn

from llm_lite.config.models import (
    InferenceConfiguration,
    PythonCompletionEvaluationConfiguration,
)
from llm_lite.inference.engine import generate_text
from llm_lite.tokenizer.loading import TextTokenizer

CHECKS_MARKER = "LLM_LITE_CHECKS"
CHECKS_PATTERN = re.compile(r"^LLM_LITE_CHECKS (?P<passed>\d+) (?P<total>\d+)$")


class PythonCompletionTaskRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    prompt: str | None = None
    task_description: str | None = None
    checks: tuple[str, ...]
    task_family: str | None = None

    @model_validator(mode="after")
    def require_exactly_one_prompt_kind(self) -> PythonCompletionTaskRecord:
        prompt_count = sum(value is not None for value in (self.prompt, self.task_description))
        if prompt_count != 1:
            raise ValueError("Python completion task requires exactly one prompt kind.")
        return self

    @property
    def inference_prompt(self) -> str:
        if self.prompt is not None:
            return self.prompt.rstrip("\n")
        assert self.task_description is not None
        return f"{self.task_description.strip()}\n\n"


class PythonCompletionTaskResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    task_family: str | None
    prompt: str
    generated_completion: str
    parsed: bool
    passed_checks: int
    total_checks: int
    timed_out: bool
    error: str | None
    stdout: str
    stderr: str


class PythonCompletionFamilyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_family: str
    tasks: int
    parsed_tasks: int
    executed_tasks: int
    passed_checks: int
    total_checks: int
    pass_rate: float


class PythonCompletionEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tasks: tuple[PythonCompletionTaskResult, ...]
    families: tuple[PythonCompletionFamilyResult, ...]
    parsed_tasks: int
    executed_tasks: int
    passed_checks: int
    total_checks: int
    pass_rate: float


@dataclass(frozen=True)
class SourceParseResult:
    parsed: bool
    error: str | None


@dataclass(frozen=True)
class SubprocessExecutionResult:
    timed_out: bool
    return_code: int | None
    stdout: str
    stderr: str
    error: str | None


@dataclass(frozen=True)
class CheckMarkerResult:
    found: bool
    passed_checks: int
    total_checks: int


def evaluate_python_completion(
    model: nn.Module,
    tokenizer: TextTokenizer,
    evaluation_configuration: PythonCompletionEvaluationConfiguration,
    inference_configuration: InferenceConfiguration,
) -> PythonCompletionEvaluationResult:
    tasks = load_python_completion_tasks(
        tasks_path=evaluation_configuration.tasks_path,
        maximum_tasks=evaluation_configuration.maximum_tasks,
    )
    task_results = tuple(
        evaluate_python_completion_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            evaluation_configuration=evaluation_configuration,
            inference_configuration=inference_configuration,
        )
        for task in tasks
    )
    return aggregate_python_completion_results(task_results=task_results)


def load_python_completion_tasks(
    tasks_path: Path,
    maximum_tasks: int | None,
) -> tuple[PythonCompletionTaskRecord, ...]:
    records: list[PythonCompletionTaskRecord] = []
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        records.append(PythonCompletionTaskRecord.model_validate_json(line))
        if maximum_tasks is not None and len(records) == maximum_tasks:
            break
    return tuple(records)


def evaluate_python_completion_task(
    model: nn.Module,
    tokenizer: TextTokenizer,
    task: PythonCompletionTaskRecord,
    evaluation_configuration: PythonCompletionEvaluationConfiguration,
    inference_configuration: InferenceConfiguration,
) -> PythonCompletionTaskResult:
    prompt = task.inference_prompt
    generated_completion = truncate_at_stop_sequence(
        text=completion_from_generated_text(
            generated_text=generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                inference_configuration=InferenceConfiguration(
                    engine=inference_configuration.engine,
                    precision=inference_configuration.precision,
                    quantization=inference_configuration.quantization,
                    decoding=inference_configuration.decoding,
                    maximum_new_tokens=evaluation_configuration.maximum_new_tokens,
                ),
            ),
            prompt=prompt,
        ),
        stop_sequences=evaluation_configuration.stop_sequences,
    )
    source = source_from_completion(task=task, generated_completion=generated_completion)
    parse_result = parse_python_source(source=source)
    if not parse_result.parsed:
        return PythonCompletionTaskResult(
            task_id=task.task_id,
            task_family=task.task_family,
            prompt=prompt,
            generated_completion=generated_completion,
            parsed=False,
            passed_checks=0,
            total_checks=len(task.checks),
            timed_out=False,
            error=parse_result.error,
            stdout="",
            stderr="",
        )
    harness_source = source + build_check_counting_harness(checks=task.checks)
    execution_result = run_python_source_in_subprocess(
        source=harness_source,
        timeout_seconds=evaluation_configuration.execution_timeout_seconds,
    )
    marker_result = parse_check_marker(stdout=execution_result.stdout)
    passed_checks = _passed_checks_from_execution(
        execution_result=execution_result,
        marker_result=marker_result,
    )
    error = _error_from_execution(
        execution_result=execution_result,
        marker_result=marker_result,
    )
    return PythonCompletionTaskResult(
        task_id=task.task_id,
        task_family=task.task_family,
        prompt=prompt,
        generated_completion=generated_completion,
        parsed=True,
        passed_checks=passed_checks,
        total_checks=len(task.checks),
        timed_out=execution_result.timed_out,
        error=error,
        stdout=execution_result.stdout,
        stderr=execution_result.stderr,
    )


def completion_from_generated_text(generated_text: str, prompt: str) -> str:
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :]
    return generated_text


def source_from_completion(
    task: PythonCompletionTaskRecord,
    generated_completion: str,
) -> str:
    if task.prompt is not None:
        return append_completion_to_prompt(
            prompt=executable_prompt_suffix(prompt=task.inference_prompt),
            generated_completion=generated_completion,
        )
    return generated_completion


def append_completion_to_prompt(prompt: str, generated_completion: str) -> str:
    if (
        prompt
        and generated_completion
        and prompt.rstrip().endswith(":")
        and not generated_completion.startswith("\n")
    ):
        return prompt + "\n" + generated_completion
    return prompt + generated_completion


def executable_prompt_suffix(prompt: str) -> str:
    return prompt.rsplit("\n\n", maxsplit=1)[-1]


def truncate_at_stop_sequence(text: str, stop_sequences: tuple[str, ...]) -> str:
    stop_indexes = tuple(
        stop_index
        for stop_sequence in stop_sequences
        if (stop_index := text.find(stop_sequence)) >= 0
    )
    if len(stop_indexes) == 0:
        return text
    return text[: min(stop_indexes)]


def parse_python_source(source: str) -> SourceParseResult:
    try:
        ast.parse(source)
    except SyntaxError as syntax_error:
        return SourceParseResult(parsed=False, error=str(syntax_error))
    return SourceParseResult(parsed=True, error=None)


def build_check_counting_harness(checks: tuple[str, ...]) -> str:
    harness_lines = [
        "",
        f"{CHECKS_MARKER.lower()}_passed_checks = 0",
        f"{CHECKS_MARKER.lower()}_total_checks = {len(checks)}",
    ]
    for check in checks:
        ast.parse(check, mode="eval")
        harness_lines.append(f"if {check}:")
        harness_lines.append(f"    {CHECKS_MARKER.lower()}_passed_checks += 1")
    harness_lines.append(
        f'print(f"{CHECKS_MARKER} '
        f"{{{CHECKS_MARKER.lower()}_passed_checks}} "
        f'{{{CHECKS_MARKER.lower()}_total_checks}}")',
    )
    harness_lines.append("")
    return "\n".join(harness_lines)


def run_python_source_in_subprocess(
    source: str,
    timeout_seconds: float,
) -> SubprocessExecutionResult:
    with tempfile.TemporaryDirectory() as temporary_directory:
        source_path = Path(temporary_directory) / "completion_task.py"
        source_path.write_text(source, encoding="utf-8")
        try:
            completed_process = subprocess.run(
                [sys.executable, str(source_path)],
                cwd=temporary_directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as timeout_error:
            stdout = _timeout_output_to_text(output=timeout_error.stdout)
            stderr = _timeout_output_to_text(output=timeout_error.stderr)
            return SubprocessExecutionResult(
                timed_out=True,
                return_code=None,
                stdout=stdout,
                stderr=stderr,
                error=f"Execution timed out after {timeout_seconds} seconds.",
            )
    return SubprocessExecutionResult(
        timed_out=False,
        return_code=completed_process.returncode,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
        error=None,
    )


def parse_check_marker(stdout: str) -> CheckMarkerResult:
    for line in stdout.splitlines():
        match_result = CHECKS_PATTERN.match(line.strip())
        if match_result is not None:
            return CheckMarkerResult(
                found=True,
                passed_checks=int(match_result.group("passed")),
                total_checks=int(match_result.group("total")),
            )
    return CheckMarkerResult(found=False, passed_checks=0, total_checks=0)


def aggregate_python_completion_results(
    task_results: tuple[PythonCompletionTaskResult, ...],
) -> PythonCompletionEvaluationResult:
    parsed_tasks = sum(1 for task_result in task_results if task_result.parsed)
    executed_tasks = sum(1 for task_result in task_results if _task_executed(task_result))
    passed_checks = sum(task_result.passed_checks for task_result in task_results)
    total_checks = sum(task_result.total_checks for task_result in task_results)
    pass_rate = 0.0
    if total_checks > 0:
        pass_rate = passed_checks / total_checks
    return PythonCompletionEvaluationResult(
        tasks=task_results,
        families=_aggregate_python_completion_family_results(task_results=task_results),
        parsed_tasks=parsed_tasks,
        executed_tasks=executed_tasks,
        passed_checks=passed_checks,
        total_checks=total_checks,
        pass_rate=pass_rate,
    )


def _aggregate_python_completion_family_results(
    task_results: tuple[PythonCompletionTaskResult, ...],
) -> tuple[PythonCompletionFamilyResult, ...]:
    task_families = sorted(
        {
            task_result.task_family
            for task_result in task_results
            if task_result.task_family is not None
        },
    )
    family_results: list[PythonCompletionFamilyResult] = []
    for task_family in task_families:
        family_task_results = tuple(
            task_result for task_result in task_results if task_result.task_family == task_family
        )
        family_results.append(
            _aggregate_python_completion_family_result(
                task_family=task_family,
                task_results=family_task_results,
            ),
        )
    return tuple(family_results)


def _aggregate_python_completion_family_result(
    task_family: str,
    task_results: tuple[PythonCompletionTaskResult, ...],
) -> PythonCompletionFamilyResult:
    parsed_tasks = sum(1 for task_result in task_results if task_result.parsed)
    executed_tasks = sum(1 for task_result in task_results if _task_executed(task_result))
    passed_checks = sum(task_result.passed_checks for task_result in task_results)
    total_checks = sum(task_result.total_checks for task_result in task_results)
    pass_rate = 0.0
    if total_checks > 0:
        pass_rate = passed_checks / total_checks
    return PythonCompletionFamilyResult(
        task_family=task_family,
        tasks=len(task_results),
        parsed_tasks=parsed_tasks,
        executed_tasks=executed_tasks,
        passed_checks=passed_checks,
        total_checks=total_checks,
        pass_rate=pass_rate,
    )


def _timeout_output_to_text(output: bytes | str | None) -> str:
    match output:
        case None:
            return ""
        case bytes():
            return output.decode("utf-8", errors="replace")
        case str():
            return output


def _passed_checks_from_execution(
    execution_result: SubprocessExecutionResult,
    marker_result: CheckMarkerResult,
) -> int:
    if execution_result.timed_out:
        return 0
    if execution_result.return_code != 0:
        return 0
    if not marker_result.found:
        return 0
    return marker_result.passed_checks


def _error_from_execution(
    execution_result: SubprocessExecutionResult,
    marker_result: CheckMarkerResult,
) -> str | None:
    if execution_result.error is not None:
        return execution_result.error
    if execution_result.return_code != 0:
        return f"Execution failed with return code {execution_result.return_code}."
    if not marker_result.found:
        return f"Missing {CHECKS_MARKER} marker in subprocess output."
    return None


def _task_executed(task_result: PythonCompletionTaskResult) -> bool:
    marker_result = parse_check_marker(stdout=task_result.stdout)
    return (
        task_result.parsed
        and not task_result.timed_out
        and task_result.error is None
        and marker_result.found
    )
