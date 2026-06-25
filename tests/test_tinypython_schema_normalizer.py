import json

from llm_lite.scripts.normalize_tinypython_jsonl_schema import (
    normalize_jsonl_file,
    normalize_record,
)


def test_normalize_record_adds_current_metadata_to_legacy_record() -> None:
    normalized = normalize_record(
        {
            "model": "teacher",
            "seed": {
                "seed_id": 7,
                "input_kind": "a list of integers",
                "operation": "count matching elements",
                "condition": "positive",
                "output_kind": "an integer",
                "edge_behavior": "handle an empty input naturally",
                "implementation_style": "use an explicit loop",
                "extra_constraint": "use no imports",
                "description_style": "use a terse direct instruction",
                "naming_style": "use descriptive names",
            },
            "sample_index": 1,
            "task_description": "Return the number of positive values.",
            "code": "def count_positive(values: list[int]) -> int:\n    return 0",
        },
    )

    assert list(normalized) == [
        "model",
        "seed",
        "sample_index",
        "task_family",
        "operation_tags",
        "task_detail",
        "signature",
        "normalized_description",
        "task_description",
        "code",
    ]
    assert normalized["task_family"] == "legacy"
    assert normalized["operation_tags"] == ["legacy"]
    assert normalized["signature"] == "def count_positive(values: list[int]) -> int:"
    assert normalized["normalized_description"] == "return the number of positive values."
    assert normalized["seed"]["task_family"] == "legacy"
    assert normalized["seed"]["task_detail"] == ""
    assert normalized["seed"]["operation_tags"] == ["legacy"]


def test_normalize_jsonl_file_writes_normalized_records(tmp_path) -> None:
    input_path = tmp_path / "old.jsonl"
    output_path = tmp_path / "normalized" / "old.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "model": "teacher",
                "seed": {"seed_id": 3},
                "sample_index": 0,
                "task_description": "Return zero.",
                "code": "def zero() -> int:\n    return 0",
            },
        )
        + "\n",
        encoding="utf-8",
    )

    count = normalize_jsonl_file(input_path=input_path, output_path=output_path)

    assert count == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["signature"] == "def zero() -> int:"
    assert record["seed"]["seed_id"] == 3
