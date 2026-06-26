"""Generate held-out TinyPython completion evaluation tasks with a teacher model."""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_lite.evaluation.python_completion import (
    build_check_counting_harness,
    parse_check_marker,
    run_python_source_in_subprocess,
)
from llm_lite.scripts.generate_tinypython import (
    SYSTEM_PROMPT,
    ParsedGeneration,
    TaskSeed,
    batches,
    build_valid_record,
    excluded_training_seed_keys,
    generate_seeds,
    parse_generation,
    user_prompt,
)

CHECKS_SYSTEM_PROMPT = """
You generate executable Python check expressions for one standalone function.

Return exactly one JSON object and nothing else:

{"checks":["expression 1","expression 2","expression 3","expression 4"]}

Requirements:
- Write 4 to 6 check expressions.
- Each check must be a single Python expression that evaluates to True for the
  supplied function.
- Use direct function calls with concrete literal arguments.
- Cover empty inputs, singleton inputs, typical cases, and edge cases when
  applicable.
- Do not use imports, variables, assignments, loops, helper functions, comments,
  assertions, print(), input(), files, randomness, time, or third-party libraries.
- Do not wrap the JSON in Markdown.

Examples of valid truthy check expressions:
- count_positive([]) == 0
- count_positive([-1, 0, 2, 3]) == 2
- first_even([1, 3, 4]) == 4
- first_even([1, 3, 5]) is None
- group_names([{"team": "a", "name": "Ada"}]) == {"a": ["Ada"]}
""".strip()

FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)
HELDOUT_EVAL_SEED = 9001
TRAINING_GENERATION_SEED = 42
TRAINING_GENERATION_SEEDS = 500_000


def checks_user_prompt(
    *,
    task_description: str,
    code: str,
    minimum_checks: int,
    maximum_checks: int,
) -> str:
    return (
        f"Create {minimum_checks} to {maximum_checks} executable check expressions "
        f"""for this task and reference implementation.

Task:
{task_description.strip()}

Reference implementation:
{code.strip()}

Return only the JSON object."""
    )


def parse_checks_generation(
    text: str,
    *,
    minimum_checks: int,
    maximum_checks: int,
) -> tuple[str, ...]:
    payload = _strip_json_fence(text.strip())
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("invalid_checks_json") from error
    if not isinstance(value, dict):
        raise ValueError("checks_json_not_object")
    checks = value.get("checks")
    if not isinstance(checks, list):
        raise ValueError("checks_missing_or_not_list")
    if not minimum_checks <= len(checks) <= maximum_checks:
        raise ValueError("wrong_check_count")
    parsed_checks: list[str] = []
    for check in checks:
        if not isinstance(check, str) or not check.strip():
            raise ValueError("empty_or_non_string_check")
        parsed_check = check.strip()
        try:
            ast.parse(parsed_check, mode="eval")
        except SyntaxError as error:
            raise ValueError("invalid_check_expression") from error
        parsed_checks.append(parsed_check)
    return tuple(parsed_checks)


def build_eval_record(
    *,
    model: str,
    seed: TaskSeed,
    parsed: ParsedGeneration,
    checks: tuple[str, ...],
) -> dict[str, Any]:
    reference_record = build_valid_record(
        model=model,
        seed=seed,
        sample_index=0,
        parsed=parsed,
    )
    return {
        "task_id": f"heldout_{seed.seed_id:06d}_{_function_name(parsed.code)}",
        "prompt": f"{parsed.task_description.strip()}\n\n{_signature_line(parsed.code)}\n",
        "checks": list(checks),
        "task_family": seed.task_family,
        "operation_tags": list(seed.operation_tags),
        "task_detail": seed.task_detail,
        "reference_code": parsed.code,
        "reference_task_description": parsed.task_description,
        "reference_record": reference_record,
    }


def validate_reference_checks(
    *,
    code: str,
    checks: tuple[str, ...],
    timeout_seconds: float,
) -> None:
    source = code.strip() + "\n" + build_check_counting_harness(checks=checks)
    result = run_python_source_in_subprocess(source=source, timeout_seconds=timeout_seconds)
    marker = parse_check_marker(stdout=result.stdout)
    if result.timed_out:
        raise ValueError("checks_timed_out")
    if result.return_code != 0:
        raise ValueError("checks_execution_failed")
    if not marker.found:
        raise ValueError("checks_marker_missing")
    if marker.passed_checks != marker.total_checks:
        raise ValueError("checks_do_not_pass_reference")


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invalid-output", type=Path)
    parser.add_argument("--num-tasks", type=int, default=200)
    parser.add_argument("--candidate-seeds", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--minimum-checks", type=int, default=4)
    parser.add_argument("--maximum-checks", type=int, default=6)
    parser.add_argument("--execution-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--code-max-tokens", type=int, default=512)
    parser.add_argument("--checks-max-tokens", type=int, default=384)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.minimum_checks > args.maximum_checks:
        raise ValueError("--minimum-checks must not be greater than --maximum-checks.")
    candidate_seed_count = args.candidate_seeds or args.num_tasks * 2
    excluded_keys = excluded_training_seed_keys(
        count=TRAINING_GENERATION_SEEDS,
        rng_seed=TRAINING_GENERATION_SEED,
    )
    seeds = generate_seeds(
        count=candidate_seed_count,
        rng_seed=HELDOUT_EVAL_SEED,
        excluded_semantic_keys=excluded_keys,
    )

    invalid_output = args.invalid_output or args.output.with_name(
        f"{args.output.stem}.invalid{args.output.suffix}",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    invalid_output.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.chat_template is None:
        raise RuntimeError("The selected model tokenizer has no chat template.")

    llm_args = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": args.prefix_caching,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.quantization != "auto":
        llm_args["quantization"] = args.quantization
    llm = LLM(**llm_args)

    code_sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.code_max_tokens,
        min_tokens=40,
        repetition_penalty=1.03,
        seed=HELDOUT_EVAL_SEED,
    )
    checks_sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.checks_max_tokens,
        min_tokens=40,
        repetition_penalty=1.03,
        seed=HELDOUT_EVAL_SEED + 1,
    )

    valid_count = 0
    attempted_seeds = 0
    invalid_counts: Counter[str] = Counter()
    started = time.perf_counter()
    for batch in batches(seeds, args.batch_size):
        code_results = _generate_code_batch(
            llm=llm,
            tokenizer=tokenizer,
            seeds=batch,
            sampling=code_sampling,
        )
        parsed_items: list[tuple[TaskSeed, ParsedGeneration]] = []
        for seed, generation, finish_reason in code_results:
            attempted_seeds += 1
            if finish_reason != "stop":
                invalid_counts["code_finish_reason_not_stop"] += 1
                _write_invalid(
                    invalid_output,
                    seed=seed,
                    stage="code",
                    reason="finish_reason_not_stop",
                    payload=generation,
                )
                continue
            try:
                parsed = parse_generation(generation)
            except ValueError as error:
                reason = f"code_{error}"
                invalid_counts[reason] += 1
                _write_invalid(
                    invalid_output,
                    seed=seed,
                    stage="code",
                    reason=reason,
                    payload=generation,
                )
                continue
            parsed_items.append((seed, parsed))

        if parsed_items:
            check_results = _generate_checks_batch(
                llm=llm,
                tokenizer=tokenizer,
                parsed_items=parsed_items,
                sampling=checks_sampling,
                minimum_checks=args.minimum_checks,
                maximum_checks=args.maximum_checks,
            )
            for seed, parsed, generation, finish_reason in check_results:
                if finish_reason != "stop":
                    invalid_counts["checks_finish_reason_not_stop"] += 1
                    _write_invalid(
                        invalid_output,
                        seed=seed,
                        stage="checks",
                        reason="finish_reason_not_stop",
                        payload=generation,
                    )
                    continue
                try:
                    checks = parse_checks_generation(
                        generation,
                        minimum_checks=args.minimum_checks,
                        maximum_checks=args.maximum_checks,
                    )
                    validate_reference_checks(
                        code=parsed.code,
                        checks=checks,
                        timeout_seconds=args.execution_timeout_seconds,
                    )
                except ValueError as error:
                    reason = f"checks_{error}"
                    invalid_counts[reason] += 1
                    _write_invalid(
                        invalid_output,
                        seed=seed,
                        stage="checks",
                        reason=reason,
                        payload=generation,
                    )
                    continue
                write_jsonl_record(
                    args.output,
                    build_eval_record(
                        model=args.model,
                        seed=seed,
                        parsed=parsed,
                        checks=checks,
                    ),
                )
                valid_count += 1
                if valid_count >= args.num_tasks:
                    break
        elapsed = time.perf_counter() - started
        print(
            f"attempted_seeds={attempted_seeds:,}/{len(seeds):,} "
            f"valid_tasks={valid_count:,}/{args.num_tasks:,} "
            f"invalid={sum(invalid_counts.values()):,} "
            f"rate={attempted_seeds / max(elapsed, 1e-9):.2f} seeds/s",
            flush=True,
        )
        if valid_count >= args.num_tasks:
            break

    if invalid_counts:
        print("invalid_reasons=" + json.dumps(dict(invalid_counts), sort_keys=True))
    if valid_count < args.num_tasks:
        print(
            f"[warning] generated only {valid_count:,}/{args.num_tasks:,} requested eval tasks. "
            "Increase --candidate-seeds if needed.",
            flush=True,
        )
    return 0


def _generate_code_batch(
    *,
    llm: Any,
    tokenizer: Any,
    seeds: Sequence[TaskSeed],
    sampling: Any,
) -> list[tuple[TaskSeed, str, str | None]]:
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(seed)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for seed in seeds
    ]
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    return [
        (seed, output.outputs[0].text.strip(), output.outputs[0].finish_reason)
        for seed, output in zip(seeds, outputs, strict=True)
    ]


def _generate_checks_batch(
    *,
    llm: Any,
    tokenizer: Any,
    parsed_items: Sequence[tuple[TaskSeed, ParsedGeneration]],
    sampling: Any,
    minimum_checks: int,
    maximum_checks: int,
) -> list[tuple[TaskSeed, ParsedGeneration, str, str | None]]:
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": CHECKS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": checks_user_prompt(
                        task_description=parsed.task_description,
                        code=parsed.code,
                        minimum_checks=minimum_checks,
                        maximum_checks=maximum_checks,
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for _, parsed in parsed_items
    ]
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    return [
        (seed, parsed, output.outputs[0].text.strip(), output.outputs[0].finish_reason)
        for (seed, parsed), output in zip(parsed_items, outputs, strict=True)
    ]


def _write_invalid(
    path: Path,
    *,
    seed: TaskSeed,
    stage: str,
    reason: str,
    payload: str,
) -> None:
    write_jsonl_record(
        path,
        {
            "seed": asdict(seed),
            "stage": stage,
            "reason": reason,
            "payload": payload,
        },
    )


def _strip_json_fence(text: str) -> str:
    match = FENCE_PATTERN.match(text)
    if match is None:
        return text
    return match.group("body").strip()


def _signature_line(code: str) -> str:
    return code.strip().splitlines()[0].strip()


def _function_name(code: str) -> str:
    module = ast.parse(code)
    function = module.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise ValueError("reference_code_not_function")
    return function.name


if __name__ == "__main__":
    raise SystemExit(main())
