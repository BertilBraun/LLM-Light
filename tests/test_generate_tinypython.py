import json
from pathlib import Path

import pytest

from llm_lite.scripts.generate_tinypython import (
    build_argument_parser,
    completed_seed_attempts,
    generate_seeds,
    invalid_output_path,
    parse_generation,
    seed_space_warning,
    unique_compatible_seed_count,
)


def test_generate_seeds_assigns_stable_requested_ids() -> None:
    seeds = generate_seeds(count=5, rng_seed=123)

    assert [seed.seed_id for seed in seeds] == [0, 1, 2, 3, 4]
    assert len({seed.input_kind for seed in seeds}) >= 1
    assert all(seed.description_style for seed in seeds)
    assert all(seed.naming_style for seed in seeds)


def test_unique_seed_space_supports_main_generation_run() -> None:
    assert unique_compatible_seed_count() >= 50_000


def test_seed_space_warning_when_request_cycles() -> None:
    warning = seed_space_warning(requested_seed_count=101, unique_seed_count=100)

    assert warning is not None
    assert "requested=101" in warning
    assert "unique=100" in warning


def test_seed_space_warning_omitted_within_unique_space() -> None:
    assert seed_space_warning(requested_seed_count=100, unique_seed_count=100) is None


def test_parse_generation_extracts_task_and_code() -> None:
    parsed = parse_generation(
        """
<task>
Return the number of positive integers in values.
</task>
<code>
def count_positive(values: list[int]) -> int:
    count = 0
    for value in values:
        if value > 0:
            count += 1
    return count
</code>
""",
    )

    assert parsed.task_description == "Return the number of positive integers in values."
    assert parsed.code.startswith("def count_positive")


def test_parse_generation_strips_top_level_imports() -> None:
    parsed = parse_generation(
        """
<task>Return item counts.</task>
<code>
from collections import Counter

def count_items(items: list[str]) -> dict[str, int]:
    return dict(Counter(items))
</code>
""",
    )

    assert parsed.code == (
        "def count_items(items: list[str]) -> dict[str, int]:\n"
        "    return dict(Counter(items))"
    )


@pytest.mark.parametrize(
    ("generation", "reason"),
    [
        (
            "def count_positive(values: list[int]) -> int:\n    return 0",
            "missing_or_malformed_tags",
        ),
        (
            """
<task>Return zero.</task>
<code>
x = 1
def value() -> int:
    return x
</code>
""",
            "not_exactly_one_top_level_function",
        ),
        (
            """
<task>Return the first value.</task>
<code>
def first(values):
    return values[0]
</code>
""",
            "missing_return_annotation",
        ),
    ],
)
def test_parse_generation_rejects_invalid_outputs(generation: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_generation(generation)


def test_completed_seed_attempts_reads_valid_and_invalid_outputs(tmp_path: Path) -> None:
    valid_path = tmp_path / "teacher.jsonl"
    invalid_path = tmp_path / "teacher.invalid.jsonl"
    valid_path.write_text(
        json.dumps({"seed": {"seed_id": 3}, "sample_index": 0}) + "\n",
        encoding="utf-8",
    )
    invalid_path.write_text(
        json.dumps({"seed": {"seed_id": 3}, "sample_index": 1}) + "\nnot json\n",
        encoding="utf-8",
    )

    assert completed_seed_attempts([valid_path, invalid_path]) == {(3, 0), (3, 1)}


def test_defaults_match_training_plan() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args(["--model", "teacher", "--output", "teacher.jsonl"])

    assert arguments.num_seeds == 50_000
    assert arguments.batch_size == 512
    assert arguments.max_tokens == 512
    assert arguments.dtype == "bfloat16"


def test_invalid_output_path_uses_invalid_suffix() -> None:
    assert invalid_output_path(Path("data/teacher_a.jsonl")) == Path("data/teacher_a.invalid.jsonl")
