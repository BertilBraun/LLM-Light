from pydantic import BaseModel, ConfigDict

from llm_lite.evaluation.python_completion import (
    PythonCompletionTaskRecord,
    build_check_counting_harness,
    parse_check_marker,
    parse_python_source,
    run_python_source_in_subprocess,
    source_from_completion,
)
from llm_lite.inference.candidates import GeneratedCandidateRecord


class PythonCandidateScoreRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    prompt: str
    sample_index: int
    generated_text: str
    score: float
    parsed: bool
    executed: bool
    passed_checks: int
    total_checks: int
    perfect: bool
    timed_out: bool
    error: str | None
    stdout: str
    stderr: str


class PythonCandidateScoringResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    scores: tuple[PythonCandidateScoreRecord, ...]


def score_python_candidates(
    candidates: tuple[GeneratedCandidateRecord, ...],
    tasks: tuple[PythonCompletionTaskRecord, ...],
    execution_timeout_seconds: float,
) -> PythonCandidateScoringResult:
    task_by_id = {task.task_id: task for task in tasks}
    scores: list[PythonCandidateScoreRecord] = []
    for candidate in candidates:
        task = task_by_id.get(candidate.task_id)
        if task is None:
            raise ValueError(f"Missing Python completion task for candidate {candidate.task_id!r}.")
        scores.append(
            score_python_candidate(
                candidate=candidate,
                task=task,
                execution_timeout_seconds=execution_timeout_seconds,
            ),
        )
    return PythonCandidateScoringResult(scores=tuple(scores))


def score_python_candidate(
    candidate: GeneratedCandidateRecord,
    task: PythonCompletionTaskRecord,
    execution_timeout_seconds: float,
) -> PythonCandidateScoreRecord:
    prompt = task.inference_prompt
    source = source_from_completion(task=task, generated_completion=candidate.generated_text)
    parse_result = parse_python_source(source=source)
    if not parse_result.parsed:
        return PythonCandidateScoreRecord(
            task_id=candidate.task_id,
            prompt=prompt,
            sample_index=candidate.sample_index,
            generated_text=candidate.generated_text,
            score=0.0,
            parsed=False,
            executed=False,
            passed_checks=0,
            total_checks=len(task.checks),
            perfect=False,
            timed_out=False,
            error=parse_result.error,
            stdout="",
            stderr="",
        )
    harness_source = source + build_check_counting_harness(checks=task.checks)
    execution_result = run_python_source_in_subprocess(
        source=harness_source,
        timeout_seconds=execution_timeout_seconds,
    )
    marker_result = parse_check_marker(stdout=execution_result.stdout)
    passed_checks = _passed_checks(
        execution_timed_out=execution_result.timed_out,
        execution_return_code=execution_result.return_code,
        marker_found=marker_result.found,
        marker_passed_checks=marker_result.passed_checks,
    )
    total_checks = len(task.checks)
    perfect = (
        not execution_result.timed_out
        and execution_result.return_code == 0
        and marker_result.found
        and passed_checks == total_checks
    )
    return PythonCandidateScoreRecord(
        task_id=candidate.task_id,
        prompt=prompt,
        sample_index=candidate.sample_index,
        generated_text=candidate.generated_text,
        score=1.0 + passed_checks,
        parsed=True,
        executed=_executed_successfully(
            execution_timed_out=execution_result.timed_out,
            execution_return_code=execution_result.return_code,
            marker_found=marker_result.found,
        ),
        passed_checks=passed_checks,
        total_checks=total_checks,
        perfect=perfect,
        timed_out=execution_result.timed_out,
        error=_score_error(
            execution_error=execution_result.error,
            execution_return_code=execution_result.return_code,
            marker_found=marker_result.found,
        ),
        stdout=execution_result.stdout,
        stderr=execution_result.stderr,
    )


def _passed_checks(
    execution_timed_out: bool,
    execution_return_code: int | None,
    marker_found: bool,
    marker_passed_checks: int,
) -> int:
    if execution_timed_out:
        return 0
    if execution_return_code != 0:
        return 0
    if not marker_found:
        return 0
    return marker_passed_checks


def _executed_successfully(
    execution_timed_out: bool,
    execution_return_code: int | None,
    marker_found: bool,
) -> bool:
    return not execution_timed_out and execution_return_code == 0 and marker_found


def _score_error(
    execution_error: str | None,
    execution_return_code: int | None,
    marker_found: bool,
) -> str | None:
    if execution_error is not None:
        return execution_error
    if execution_return_code != 0:
        return f"Execution failed with return code {execution_return_code}."
    if not marker_found:
        return "Missing check marker in subprocess output."
    return None
