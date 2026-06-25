"""Normalize TinyPython JSONL records to the current Hugging Face schema."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CANONICAL_SEED_STRING_FIELDS = (
    "task_family",
    "input_kind",
    "operation",
    "condition",
    "output_kind",
    "edge_behavior",
    "implementation_style",
    "extra_constraint",
    "task_detail",
    "description_style",
    "naming_style",
)


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    seed = _mapping_value(record.get("seed"))
    code = _string_value(record.get("code"))
    task_description = _string_value(record.get("task_description"))
    task_family = _first_nonempty_string(
        record.get("task_family"),
        seed.get("task_family"),
        "legacy",
    )
    operation_tags = _operation_tags(record.get("operation_tags"), seed.get("operation_tags"))
    task_detail = _first_nonempty_string(
        record.get("task_detail"),
        seed.get("task_detail"),
        "",
    )

    normalized_seed = {
        "seed_id": _integer_value(seed.get("seed_id"), default=-1),
        "task_family": task_family,
        "input_kind": _string_value(seed.get("input_kind")),
        "operation": _string_value(seed.get("operation")),
        "condition": _string_value(seed.get("condition")),
        "output_kind": _string_value(seed.get("output_kind")),
        "edge_behavior": _string_value(seed.get("edge_behavior")),
        "implementation_style": _string_value(seed.get("implementation_style")),
        "extra_constraint": _string_value(seed.get("extra_constraint")),
        "task_detail": task_detail,
        "description_style": _string_value(seed.get("description_style")),
        "naming_style": _string_value(seed.get("naming_style")),
        "operation_tags": operation_tags,
    }

    return {
        "model": _string_value(record.get("model")),
        "seed": normalized_seed,
        "sample_index": _integer_value(record.get("sample_index"), default=0),
        "task_family": task_family,
        "operation_tags": operation_tags,
        "task_detail": task_detail,
        "signature": _first_nonempty_string(
            record.get("signature"),
            _signature_from_code(code),
            "",
        ),
        "normalized_description": _first_nonempty_string(
            record.get("normalized_description"),
            _normalize_description(task_description),
            "",
        ),
        "task_description": task_description,
        "code": code,
    }


def normalize_jsonl_file(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as input_handle:
        with output_path.open("w", encoding="utf-8") as output_handle:
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError(f"JSONL record must be an object at {input_path}:{line_number}")
                json.dump(normalize_record(record), output_handle, ensure_ascii=False)
                output_handle.write("\n")
                count += 1
    return count


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    for input_path in arguments.inputs:
        output_path = arguments.output_dir / input_path.name
        count = normalize_jsonl_file(input_path=input_path, output_path=output_path)
        print(f"{input_path} -> {output_path} records={count:,}")
    return 0


def _mapping_value(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _string_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _integer_value(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    return default


def _operation_tags(*values: object) -> list[str]:
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, str):
            tags = [item for item in value if isinstance(item, str) and item]
            if tags:
                return tags
    return ["legacy"]


def _first_nonempty_string(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _signature_from_code(code: str) -> str:
    for line in code.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _normalize_description(task_description: str) -> str:
    return " ".join(task_description.casefold().split())


if __name__ == "__main__":
    raise SystemExit(main())
