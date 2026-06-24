"""Report overlap between TinyPython training JSONL records and eval tasks."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def signature_from_code(code: str) -> str | None:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return None
    first_line = code.strip().splitlines()[0].strip()
    return first_line if first_line.startswith("def ") else None


def signature_from_prompt(prompt: str) -> str | None:
    first_line = prompt.strip().splitlines()[0].strip()
    return first_line if first_line.startswith("def ") else None


def report_contamination(
    *,
    training_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    training_descriptions = {
        value
        for record in training_records
        if isinstance(value := record.get("task_description"), str)
    }
    normalized_training_descriptions = {
        normalize_text(description) for description in training_descriptions
    }
    training_codes = {
        value for record in training_records if isinstance(value := record.get("code"), str)
    }
    training_signatures = {
        signature
        for record in training_records
        if isinstance(
            signature := record.get("signature")
            or (
                signature_from_code(code)
                if isinstance(code := record.get("code"), str)
                else None
            ),
            str,
        )
    }
    training_families = {
        value for record in training_records if isinstance(value := record.get("task_family"), str)
    }

    exact_prompt_or_description_matches: list[str] = []
    normalized_prompt_or_description_matches: list[str] = []
    exact_code_matches: list[str] = []
    signature_overlaps: list[str] = []
    task_family_counter: Counter[str] = Counter()

    for eval_record in eval_records:
        task_id = str(eval_record.get("task_id", "<unknown>"))
        prompt_or_description = _eval_prompt_or_description(eval_record)
        if prompt_or_description is not None:
            if prompt_or_description in training_descriptions:
                exact_prompt_or_description_matches.append(task_id)
            if normalize_text(prompt_or_description) in normalized_training_descriptions:
                normalized_prompt_or_description_matches.append(task_id)

        eval_code = eval_record.get("code") or eval_record.get("expected_code")
        if isinstance(eval_code, str) and eval_code in training_codes:
            exact_code_matches.append(task_id)

        eval_signature = _eval_signature(eval_record)
        if eval_signature is not None and eval_signature in training_signatures:
            signature_overlaps.append(task_id)

        eval_family = eval_record.get("task_family")
        if isinstance(eval_family, str) and eval_family in training_families:
            task_family_counter[eval_family] += 1

    return {
        "training_records": len(training_records),
        "eval_records": len(eval_records),
        "exact_prompt_or_description_matches": sorted(exact_prompt_or_description_matches),
        "normalized_prompt_or_description_matches": sorted(
            normalized_prompt_or_description_matches,
        ),
        "exact_code_matches": sorted(exact_code_matches),
        "signature_overlaps": sorted(signature_overlaps),
        "task_family_overlap_counts": dict(sorted(task_family_counter.items())),
    }


def load_jsonl_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
                records.append(record)
    return records


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    report = report_contamination(
        training_records=load_jsonl_records(arguments.training_jsonl),
        eval_records=load_jsonl_records([arguments.eval_jsonl]),
    )
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(report_text)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(report_text + "\n", encoding="utf-8")
    return 0


def _eval_prompt_or_description(record: Mapping[str, Any]) -> str | None:
    for key in ("prompt", "task_description", "description", "normalized_description"):
        value = record.get(key)
        if isinstance(value, str):
            return value.strip()
    return None


def _eval_signature(record: Mapping[str, Any]) -> str | None:
    signature = record.get("signature")
    if isinstance(signature, str):
        return signature.strip()
    prompt = record.get("prompt")
    if isinstance(prompt, str):
        return signature_from_prompt(prompt=prompt)
    code = record.get("code") or record.get("expected_code")
    if isinstance(code, str):
        return signature_from_code(code=code)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
