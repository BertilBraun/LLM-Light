from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pytest import MonkeyPatch
from torch import nn

from llm_lite.config.models import (
    DecodingStrategy,
    EvaluationConfiguration,
    GreedyDecodingConfiguration,
    InferenceConfiguration,
    InferenceEngine,
    PackingConfiguration,
    Precision,
    PythonCompletionEvaluationConfiguration,
    QuantizationType,
)
from llm_lite.evaluation import python_completion
from llm_lite.evaluation.python_completion import (
    PythonCompletionTaskRecord,
    build_check_counting_harness,
    completion_from_generated_text,
    evaluate_python_completion,
    load_python_completion_tasks,
    parse_python_source,
    run_python_source_in_subprocess,
    source_from_completion,
    truncate_at_stop_sequence,
)
from llm_lite.evaluation.runner import run_configured_evaluators


@dataclass(frozen=True)
class CapturedGenerationCall:
    prompt: str
    inference_configuration: InferenceConfiguration


class UnusedTokenizer:
    @property
    def vocabulary_size(self) -> int:
        return 1

    @property
    def pad_token_id(self) -> int | None:
        return None

    @property
    def eos_token_id(self) -> int:
        return 0

    def encode(self, text: str, add_bos: bool, add_eos: bool) -> list[int]:
        return []

    def decode(self, token_ids: list[int]) -> str:
        return ""

    def save(self, directory: Path) -> None:
        return None


def test_load_python_completion_tasks_reads_typed_jsonl_records() -> None:
    tasks = load_python_completion_tasks(
        tasks_path=Path("tests/fixtures/python_completion/tasks.jsonl"),
        maximum_tasks=2,
    )

    assert len(tasks) == 2
    assert tasks[0] == PythonCompletionTaskRecord(
        task_id="add",
        prompt="def add(a: int, b: int) -> int:\n",
        checks=(
            "add(1, 2) == 3",
            "add(-1, 1) == 0",
            "add(0, 0) == 0",
            "add(10, -3) == 7",
            "add(5, 6) == 11",
        ),
    )


def test_truncate_at_stop_sequence_uses_first_configured_match() -> None:
    text = "    return 1\n\nclass Extra:\n    pass\n\ndef next_function() -> int:\n    return 2"

    truncated_text = truncate_at_stop_sequence(
        text=text,
        stop_sequences=("\n\ndef ", "\nclass "),
    )

    assert truncated_text == "    return 1\n"


def test_completion_from_generated_text_strips_full_prompt_echo() -> None:
    completion = completion_from_generated_text(
        generated_text="Return an int.\n\ndef answer() -> int:\n    return 1\n",
        prompt="Return an int.\n\n",
    )

    assert completion == "def answer() -> int:\n    return 1\n"


def test_source_from_completion_appends_generated_body_to_signature_prompt_code() -> None:
    prompt = (
        "Return a list containing each non-empty string uppercased and reversed, "
        "preserving order.\n\n"
        "def reverse_uppercase_strings(strings: list[str]) -> list[str]:\n"
    )
    task = PythonCompletionTaskRecord(
        task_id="reverse_uppercase_strings",
        prompt=prompt,
        checks=("reverse_uppercase_strings(['ab']) == ['BA']",),
    )

    source = source_from_completion(
        task=task,
        generated_completion="    return [value.upper()[::-1] for value in strings]\n",
    )

    assert source == (
        "def reverse_uppercase_strings(strings: list[str]) -> list[str]:\n"
        "    return [value.upper()[::-1] for value in strings]\n"
    )


def test_signature_prompt_generation_trims_trailing_newline() -> None:
    task = PythonCompletionTaskRecord(
        task_id="multiply_lists",
        prompt=(
            "Given two lists of integers, return pairwise products.\n\n"
            "def multiply_lists(list1: list[int], list2: list[int]) -> list[int]:\n"
        ),
        checks=("multiply_lists([2], [3]) == [6]",),
    )

    assert task.inference_prompt == (
        "Given two lists of integers, return pairwise products.\n\n"
        "def multiply_lists(list1: list[int], list2: list[int]) -> list[int]:"
    )
    assert source_from_completion(
        task=task,
        generated_completion=(
            "\n"
            "    result: list[int] = []\n"
            "    for a, b in zip(list1, list2):\n"
            "        result.append(a * b)\n"
            "    return result\n"
        ),
    ) == (
        "def multiply_lists(list1: list[int], list2: list[int]) -> list[int]:\n"
        "    result: list[int] = []\n"
        "    for a, b in zip(list1, list2):\n"
        "        result.append(a * b)\n"
        "    return result\n"
    )


def test_task_description_records_keep_full_function_generation_prompt() -> None:
    task = PythonCompletionTaskRecord(
        task_id="count_positive",
        task_description="Define a function named count_positive that returns positives.",
        checks=("count_positive([1, -1]) == 1",),
    )

    assert task.inference_prompt == (
        "Define a function named count_positive that returns positives.\n\n"
    )
    assert (
        source_from_completion(
            task=task,
            generated_completion="def count_positive(values: list[int]) -> int:\n    return 1\n",
        )
        == "def count_positive(values: list[int]) -> int:\n    return 1\n"
    )


def test_parse_python_source_reports_success_and_failure() -> None:
    successful_parse = parse_python_source(
        source="def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    failed_parse = parse_python_source(source="def add(a: int, b: int) -> int:\nreturn a + b\n")

    assert successful_parse.parsed is True
    assert successful_parse.error is None
    assert failed_parse.parsed is False
    assert failed_parse.error is not None


def test_generated_check_counting_harness_counts_partial_pass_rate() -> None:
    source = "def add(a: int, b: int) -> int:\n    return a + b\n" + build_check_counting_harness(
        checks=(
            "add(1, 2) == 3",
            "add(1, 2) == 4",
            "add(-1, 1) == 0",
        ),
    )

    execution_result = run_python_source_in_subprocess(source=source, timeout_seconds=2.0)
    marker_result = python_completion.parse_check_marker(stdout=execution_result.stdout)

    assert execution_result.timed_out is False
    assert execution_result.return_code == 0
    assert marker_result.found is True
    assert marker_result.passed_checks == 2
    assert marker_result.total_checks == 3


def test_subprocess_timeout_reports_timed_out() -> None:
    execution_result = run_python_source_in_subprocess(
        source="import time\ntime.sleep(1.0)\n",
        timeout_seconds=0.01,
    )

    assert execution_result.timed_out is True
    assert execution_result.return_code is None
    assert execution_result.error is not None


def test_evaluator_uses_generate_text_with_configured_inference_settings(
    monkeypatch: MonkeyPatch,
) -> None:
    captured_calls: list[CapturedGenerationCall] = []

    def generate_completion(
        model: nn.Module,
        tokenizer: UnusedTokenizer,
        prompt: str,
        inference_configuration: InferenceConfiguration,
    ) -> str:
        captured_calls.append(
            CapturedGenerationCall(
                prompt=prompt,
                inference_configuration=inference_configuration,
            ),
        )
        return "    return a + b\n\nclass StopHere:\n    pass\n"

    monkeypatch.setattr(python_completion, "generate_text", generate_completion)

    result = evaluate_python_completion(
        model=nn.Identity(),
        tokenizer=UnusedTokenizer(),
        evaluation_configuration=PythonCompletionEvaluationConfiguration(
            tasks_path=Path("tests/fixtures/python_completion/tasks.jsonl"),
            maximum_tasks=1,
            maximum_new_tokens=7,
            execution_timeout_seconds=2.0,
            stop_sequences=("\n\nclass ",),
        ),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=99,
        ),
    )

    assert len(captured_calls) == 1
    assert captured_calls[0].prompt == "def add(a: int, b: int) -> int:"
    assert captured_calls[0].inference_configuration.engine is InferenceEngine.NAIVE
    assert captured_calls[0].inference_configuration.maximum_new_tokens == 7
    assert result.tasks[0].generated_completion == "    return a + b"
    assert result.parsed_tasks == 1
    assert result.executed_tasks == 1
    assert result.passed_checks == 5
    assert result.total_checks == 5
    assert result.pass_rate == 1.0


def test_evaluator_runs_signature_prefix_records(
    monkeypatch: MonkeyPatch,
) -> None:
    captured_calls: list[CapturedGenerationCall] = []

    def generate_completion(
        model: nn.Module,
        tokenizer: UnusedTokenizer,
        prompt: str,
        inference_configuration: InferenceConfiguration,
    ) -> str:
        captured_calls.append(
            CapturedGenerationCall(
                prompt=prompt,
                inference_configuration=inference_configuration,
            ),
        )
        return (
            prompt
            + "\n"
            + "    user_info = profiles.get(id)\n"
            + "    if user_info and user_info.get('name'):\n"
            + "        return user_info['name']\n"
            + "    return None\n"
        )

    monkeypatch.setattr(python_completion, "generate_text", generate_completion)

    result = evaluate_python_completion(
        model=nn.Identity(),
        tokenizer=UnusedTokenizer(),
        evaluation_configuration=PythonCompletionEvaluationConfiguration(
            tasks_path=Path("tests/fixtures/tinypython_completion/tasks.jsonl"),
            maximum_tasks=1,
            maximum_new_tokens=32,
            execution_timeout_seconds=2.0,
            stop_sequences=("\n\nReturn ",),
        ),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=99,
        ),
    )

    assert len(captured_calls) == 1
    assert captured_calls[0].prompt == (
        "Given a dictionary `profiles` and an integer `id`, return the user's display "
        "name if it exists and is nonempty. Otherwise, return `None`.\n\n"
        "def get_user_display_name(profiles: dict[int, dict[str, str]], id: int) "
        "-> str | None:"
    )
    assert result.tasks[0].generated_completion == (
        "\n"
        "    user_info = profiles.get(id)\n"
        "    if user_info and user_info.get('name'):\n"
        "        return user_info['name']\n"
        "    return None\n"
    )
    assert result.parsed_tasks == 1
    assert result.executed_tasks == 1
    assert result.passed_checks == 5
    assert result.total_checks == 5
    assert result.pass_rate == 1.0


def test_runner_includes_python_completion_report_and_metrics(monkeypatch: MonkeyPatch) -> None:
    def generate_completion(
        model: nn.Module,
        tokenizer: UnusedTokenizer,
        prompt: str,
        inference_configuration: InferenceConfiguration,
    ) -> str:
        return "    return a + b\n"

    monkeypatch.setattr(python_completion, "generate_text", generate_completion)

    result = run_configured_evaluators(
        model=nn.Identity(),
        tokenizer=UnusedTokenizer(),
        registry=None,
        evaluation_configuration=EvaluationConfiguration(
            python_completion=PythonCompletionEvaluationConfiguration(
                tasks_path=Path("tests/fixtures/python_completion/tasks.jsonl"),
                maximum_tasks=1,
                maximum_new_tokens=5,
                execution_timeout_seconds=2.0,
                stop_sequences=("\n\ndef ", "\nclass "),
            ),
        ),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=10,
        ),
        packing_configuration=PackingConfiguration(context_length=8),
    )

    assert "python_completion" in result.report
    assert result.metrics["python_completion_tasks"] == 1
    assert result.metrics["python_completion_parsed_tasks"] == 1
    assert result.metrics["python_completion_executed_tasks"] == 1
    assert result.metrics["python_completion_passed_checks"] == 5
    assert result.metrics["python_completion_total_checks"] == 5
    assert result.metrics["python_completion_pass_rate"] == 1.0
