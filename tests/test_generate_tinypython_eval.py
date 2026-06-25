import pytest

from llm_lite.evaluation.python_completion import PythonCompletionTaskRecord
from llm_lite.scripts.generate_tinypython import generate_seeds, parse_generation
from llm_lite.scripts.generate_tinypython_eval import (
    CHECKS_SYSTEM_PROMPT,
    build_argument_parser,
    build_eval_record,
    checks_user_prompt,
    parse_checks_generation,
    validate_reference_checks,
)


def test_parse_checks_generation_accepts_json_object() -> None:
    checks = parse_checks_generation(
        '{"checks":["add(1, 2) == 3","add(-1, 1) == 0","add(0, 0) == 0","add(5, 7) == 12"]}',
        minimum_checks=4,
        maximum_checks=6,
    )

    assert checks == (
        "add(1, 2) == 3",
        "add(-1, 1) == 0",
        "add(0, 0) == 0",
        "add(5, 7) == 12",
    )


def test_parse_checks_generation_accepts_json_fence() -> None:
    checks = parse_checks_generation(
        """
```json
{"checks":["value() == 1","value() != 2","isinstance(value(), int)","value() > 0"]}
```
""",
        minimum_checks=4,
        maximum_checks=6,
    )

    assert checks[0] == "value() == 1"


@pytest.mark.parametrize(
    ("generation", "reason"),
    [
        ("not json", "invalid_checks_json"),
        ('{"checks":["x() == 1"]}', "wrong_check_count"),
        ('{"checks":["x() = 1","x() == 1","x() == 1","x() == 1"]}', "invalid_check_expression"),
    ],
)
def test_parse_checks_generation_rejects_invalid_output(generation: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_checks_generation(generation, minimum_checks=4, maximum_checks=6)


def test_build_eval_record_writes_completion_task_shape() -> None:
    seed = generate_seeds(count=1, rng_seed=123)[0]
    parsed = parse_generation(
        """
<task>Return the sum of a and b.</task>
<code>
def add(a: int, b: int) -> int:
    return a + b
</code>
""",
    )

    record = build_eval_record(
        model="teacher",
        seed=seed,
        parsed=parsed,
        checks=(
            "add(1, 2) == 3",
            "add(-1, 1) == 0",
            "add(0, 0) == 0",
            "add(5, 7) == 12",
        ),
    )

    task = PythonCompletionTaskRecord.model_validate(record)
    assert task.task_id.endswith("_add")
    assert task.prompt == "Return the sum of a and b.\n\ndef add(a: int, b: int) -> int:\n"
    assert task.checks[0] == "add(1, 2) == 3"
    assert record["reference_code"].startswith("def add")


def test_validate_reference_checks_requires_all_checks_to_pass() -> None:
    validate_reference_checks(
        code="def add(a: int, b: int) -> int:\n    return a + b",
        checks=(
            "add(1, 2) == 3",
            "add(-1, 1) == 0",
            "add(0, 0) == 0",
            "add(5, 7) == 12",
        ),
        timeout_seconds=2.0,
    )

    with pytest.raises(ValueError, match="checks_do_not_pass_reference"):
        validate_reference_checks(
            code="def add(a: int, b: int) -> int:\n    return a + b",
            checks=(
                "add(1, 2) == 4",
                "add(-1, 1) == 0",
                "add(0, 0) == 0",
                "add(5, 7) == 12",
            ),
            timeout_seconds=2.0,
        )


def test_checks_user_prompt_includes_task_and_code() -> None:
    prompt = checks_user_prompt(
        task_description="Return zero.",
        code="def zero() -> int:\n    return 0",
        minimum_checks=4,
        maximum_checks=6,
    )

    assert "Return zero." in prompt
    assert "def zero() -> int:" in prompt
    assert "Create 4 to 6" in prompt


def test_checks_system_prompt_shows_truthy_expression_examples() -> None:
    assert "truthy check expressions" in CHECKS_SYSTEM_PROMPT
    assert "count_positive([]) == 0" in CHECKS_SYSTEM_PROMPT
    assert "first_even([1, 3, 5]) is None" in CHECKS_SYSTEM_PROMPT


def test_defaults_match_heldout_plan() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args(["--model", "teacher", "--output", "eval.jsonl"])

    assert arguments.num_tasks == 200
    assert not hasattr(arguments, "seed")
    assert not hasattr(arguments, "exclude_seed")
    assert not hasattr(arguments, "exclude_num_seeds")
