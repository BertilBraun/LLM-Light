"""Generate TinyPython task/code pairs with one local vLLM teacher."""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import random
import re
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """
You generate high-quality training examples for a very small Python language model.

Return exactly this format and nothing else:

<task>
One concise task description in one or two sentences.
</task>
<code>
One complete standalone Python function.
</code>

Requirements:
- Python 3 only.
- Exactly one top-level function and no other top-level statements.
- Add type annotations to every parameter and the return value.
- Use only built-in Python types and operations unless the seed explicitly allows
  a standard-library module.
- Do not import anything. Do not emit import statements or from-import
  statements. If a type name appears in an annotation, assume it is already
  available or use built-in generic forms such as list[int], dict[str, int],
  tuple[int, ...], and int | None.
- If a standard-library helper would normally require an import, assume it is
  already available by name. This includes typing names and common modules or
  helpers from collections, functools, itertools, math, operator, and statistics.
  Still do not emit any import statement.
- The task description must fully specify what the function computes.
- Respect every semantic field in the supplied seed.
- Use meaningful function, argument, and local-variable names.
- Use 3 to 30 non-empty lines.
- Return the result; never use input(), print(), files, global state, classes,
  decorators, third-party libraries, tests, assertions, comments, docstrings,
  Markdown fences, or explanatory prose.
- Prefer simple readable code over code golf.
- Do not mutate input collections unless explicitly requested.
- Resolve minor ambiguity in the simplest sensible way.
- Generate a correct implementation matching the description.
- Vary wording and implementation structure rather than copying the examples.

Examples:

<task>
Return the number of integers in values that are strictly greater than zero.
</task>
<code>
def count_positive(values: list[int]) -> int:
    count = 0
    for value in values:
        if value > 0:
            count += 1
    return count
</code>

<task>
Return a new list containing the lowercase forms of all nonempty strings in names,
while preserving their original order.
</task>
<code>
def lowercase_nonempty(names: list[str]) -> list[str]:
    result: list[str] = []
    for name in names:
        if name:
            result.append(name.lower())
    return result
</code>

<task>
Return the first integer in values that is divisible by divisor. Return None if no
such integer exists.
</task>
<code>
def first_divisible(values: list[int], divisor: int) -> int | None:
    for value in values:
        if value % divisor == 0:
            return value
    return None
</code>

<task>
Return a dictionary mapping each word to the number of times it occurs.
</task>
<code>
def count_words(words: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts
</code>
""".strip()

OUTPUT_PATTERN = re.compile(
    r"^\s*<task>\s*(?P<task>.*?)\s*</task>\s*<code>\s*(?P<code>.*?)\s*</code>\s*$",
    re.DOTALL,
)


@dataclass(frozen=True)
class TaskSeed:
    seed_id: int
    input_kind: str
    operation: str
    condition: str
    output_kind: str
    edge_behavior: str
    implementation_style: str
    extra_constraint: str
    description_style: str
    naming_style: str


@dataclass(frozen=True)
class ParsedGeneration:
    task_description: str
    code: str


DESCRIPTION_STYLES = [
    "use a terse direct instruction",
    "use a precise declarative sentence",
    "mention the edge behavior naturally",
    "use alternate wording from the operation name",
]

NAMING_STYLES = [
    "use generic names such as values, items, result, and mapping",
    "use descriptive domain-neutral names",
    "use short readable names when the function is simple",
    "use parameter names that match the input kind",
]


FAMILIES = [
    {
        "input": "a list of integers",
        "operations": {
            "count matching elements": "an integer",
            "sum matching elements": "an integer",
            "compute the product of matching elements": "an integer",
            "compute the minimum matching element": "an integer or None",
            "compute the maximum matching element": "an integer or None",
            "return both count and sum for matching elements": (
                "a tuple of an integer count and an integer sum"
            ),
            "filter matching elements": "a list of integers",
            "find the first matching element": "an integer or None",
            "find the last matching element": "an integer or None",
            "check whether any element matches": "a boolean",
            "check whether every element matches": "a boolean",
            "transform matching elements": "a list of integers",
            "clamp matching elements to a lower and upper bound": "a list of integers",
            "partition elements into two groups": "a tuple of two integer lists",
            "find the index of the first matching element": "an integer or None",
        },
        "conditions": [
            "positive",
            "negative",
            "zero",
            "even",
            "odd",
            "greater than a threshold parameter",
            "less than a threshold parameter",
            "equal to a target parameter",
            "divisible by a positive divisor parameter",
            "inside an inclusive lower and upper bound",
            "outside an inclusive lower and upper bound",
            "absolute value greater than a threshold parameter",
            "index is even",
            "index is odd",
        ],
        "edges": [
            "handle an empty input naturally",
            "preserve original order",
            "return None when no match exists",
            "return zero when no match contributes to a numeric result",
            "keep the original value when no transform applies",
        ],
        "styles": [
            "use an explicit loop",
            "use a comprehension when readable",
            "use an early return when appropriate",
            "use an accumulator variable",
            "use helper local variables for clarity",
        ],
        "extras": [
            "do not mutate the input list",
            "keep duplicate values",
            "use no imports",
            "avoid clever one-line implementations",
        ],
    },
    {
        "input": "a list of strings",
        "operations": {
            "count matching strings": "an integer",
            "filter matching strings": "a list of strings",
            "transform every string": "a list of strings",
            "transform matching strings": "a list of strings",
            "find the first matching string": "a string or None",
            "find the last matching string": "a string or None",
            "find the longest matching string": "a string or None",
            "find the shortest matching string": "a string or None",
            "build a frequency dictionary": "a dictionary from strings to integers",
            "remove duplicate strings": "a list of strings",
            "join selected strings": "a string",
            "check whether all strings match": "a boolean",
            "group strings by their first character": (
                "a dictionary from strings to lists of strings"
            ),
        },
        "conditions": [
            "nonempty",
            "empty",
            "starts with a prefix parameter",
            "ends with a suffix parameter",
            "contains a substring parameter",
            "has length greater than a limit parameter",
            "is entirely lowercase",
            "is entirely uppercase",
            "contains at least one digit",
            "equals a target string ignoring case",
            "contains only alphabetic characters",
            "contains no whitespace",
            "has length equal to a limit parameter",
        ],
        "edges": [
            "handle an empty input naturally",
            "preserve original order",
            "return None when no match exists",
            "resolve ties by first occurrence",
            "resolve ties by last occurrence",
            "ignore empty strings",
        ],
        "styles": [
            "use an explicit loop",
            "use a comprehension when readable",
            "use an early return when appropriate",
            "use a dictionary accumulator when appropriate",
            "build the result incrementally",
        ],
        "extras": [
            "do not mutate the input list",
            "keep duplicates unless the operation removes them",
            "use no imports",
            "perform case-insensitive comparisons only when requested",
        ],
    },
    {
        "input": "a string",
        "operations": {
            "count matching characters": "an integer",
            "filter characters": "a string",
            "replace matching characters": "a string",
            "find the first matching character": "a string or None",
            "find the last matching character": "a string or None",
            "split into runs": "a list of strings",
            "normalize whitespace": "a string",
            "build a character frequency dictionary": "a dictionary from strings to integers",
            "check whether the string matches": "a boolean",
            "extract a bounded substring": "a string",
            "remove repeated adjacent characters": "a string",
            "return the indexes of matching characters": "a list of integers",
        },
        "conditions": [
            "is a digit",
            "is alphabetic",
            "is whitespace",
            "is uppercase",
            "is lowercase",
            "equals a target character",
            "belongs to a supplied set of characters",
            "occurs more than once",
            "is a vowel",
            "is not whitespace",
            "appears before a limit index",
        ],
        "edges": [
            "handle an empty string naturally",
            "preserve character order",
            "return None when no match exists",
            "return an empty string when no characters match",
        ],
        "styles": [
            "use an explicit loop",
            "use string methods when readable",
            "use an early return when appropriate",
            "build the result incrementally",
            "use indexes when the condition depends on position",
        ],
        "extras": [
            "use no regular expressions",
            "use no imports",
            "avoid changing character case unless requested",
        ],
    },
    {
        "input": "a dictionary from strings to integers",
        "operations": {
            "select matching entries": "a dictionary from strings to integers",
            "sum matching values": "an integer",
            "count matching entries": "an integer",
            "find the key with the largest matching value": "a string or None",
            "find the key with the smallest matching value": "a string or None",
            "invert the mapping into grouped keys": (
                "a dictionary from integers to lists of strings"
            ),
            "merge with a second dictionary": "a dictionary from strings to integers",
            "return keys ordered by their values": "a list of strings",
            "return values ordered by their keys": "a list of integers",
            "check whether any entry matches": "a boolean",
            "transform matching values": "a dictionary from strings to integers",
            "rename matching keys with a prefix parameter": "a dictionary from strings to integers",
        },
        "conditions": [
            "positive value",
            "negative value",
            "zero value",
            "value greater than a threshold parameter",
            "value less than a threshold parameter",
            "even value",
            "odd value",
            "key starts with a prefix parameter",
            "key contains a substring parameter",
            "key ends with a suffix parameter",
            "value inside an inclusive lower and upper bound",
        ],
        "edges": [
            "handle an empty dictionary naturally",
            "return None when no match exists",
            "resolve ties by insertion order",
            "preserve insertion order where possible",
            "leave unmatched entries unchanged for transforms",
        ],
        "styles": [
            "use an explicit loop",
            "use a dictionary comprehension when readable",
            "use an early return when appropriate",
            "use an accumulator variable",
            "use items() iteration",
        ],
        "extras": [
            "do not mutate input dictionaries",
            "preserve insertion order where relevant",
            "use no imports",
            "avoid relying on sorted order unless requested",
        ],
    },
    {
        "input": "two lists of integers",
        "operations": {
            "compute elementwise sums": "a list of integers",
            "compute pairwise differences": "a list of integers",
            "compute elementwise products": "a list of integers",
            "return values appearing in both": "a list of integers",
            "return values unique to either list": "a list of integers",
            "interleave their elements": "a list of integers",
            "compare corresponding elements": "a list of booleans",
            "combine them without duplicates": "a list of integers",
            "find common values with counts": "a dictionary from integers to integers",
            "return pairs whose sum matches a target parameter": "a list of integer pairs",
            "return indexes where corresponding elements match": "a list of integers",
        },
        "conditions": [
            "process only positions available in both lists",
            "continue until both lists are exhausted",
            "preserve order of first appearance",
            "treat duplicate values as distinct occurrences",
            "ignore duplicate values",
            "keep pairs where the first value is greater",
            "keep pairs where both values are even",
        ],
        "edges": [
            "handle empty lists naturally",
            "preserve original relative order",
            "stop at the shorter list for position-wise operations",
            "include remaining elements when interleaving",
            "return an empty list when there are no matching pairs",
        ],
        "styles": [
            "use an explicit loop",
            "use zip when appropriate",
            "use index-based iteration",
            "use a set only when ordering remains correct",
            "avoid nested loops unless necessary",
        ],
        "extras": [
            "do not mutate either input list",
            "use no imports",
            "keep duplicate values only when requested",
        ],
    },
]


def compatible(seed: TaskSeed) -> bool:
    if "return None" in seed.edge_behavior and "or None" not in seed.output_kind:
        return False
    if (
        "dictionary accumulator" in seed.implementation_style
        and "dictionary" not in seed.output_kind
    ):
        return False
    if "comprehension" in seed.implementation_style and seed.operation.startswith("find the first"):
        return False
    if "early return" in seed.implementation_style and not (
        seed.output_kind == "a boolean"
        or "or None" in seed.output_kind
        or seed.operation.startswith("find")
        or seed.operation.startswith("check")
    ):
        return False
    if "duplicate" in seed.condition and "duplicate" not in seed.operation:
        return False
    return True


def compatible_seed_candidates() -> list[TaskSeed]:
    candidates: list[TaskSeed] = []
    seed_id = 0
    for family in FAMILIES:
        for operation, output_kind in family["operations"].items():
            for condition, edge, style, extra, description_style, naming_style in itertools.product(
                family["conditions"],
                family["edges"],
                family["styles"],
                family["extras"],
                DESCRIPTION_STYLES,
                NAMING_STYLES,
            ):
                item = TaskSeed(
                    seed_id=seed_id,
                    input_kind=family["input"],
                    operation=operation,
                    condition=condition,
                    output_kind=output_kind,
                    edge_behavior=edge,
                    implementation_style=style,
                    extra_constraint=extra,
                    description_style=description_style,
                    naming_style=naming_style,
                )
                if compatible(item):
                    candidates.append(item)
                    seed_id += 1
    return candidates


def unique_compatible_seed_count() -> int:
    return len(compatible_seed_candidates())


def seed_space_warning(requested_seed_count: int, unique_seed_count: int) -> str | None:
    if requested_seed_count <= unique_seed_count:
        return None
    return (
        "[warning] requested num_seeds exceeds the unique compatible seed space: "
        f"requested={requested_seed_count:,} unique={unique_seed_count:,}. "
        "Generation will cycle through semantic seeds and rely on stochastic "
        "sampling for additional variants."
    )


def generate_seeds(count: int, rng_seed: int) -> list[TaskSeed]:
    candidates = compatible_seed_candidates()

    rng = random.Random(rng_seed)
    rng.shuffle(candidates)
    if count <= len(candidates):
        chosen = candidates[:count]
    else:
        chosen = []
        while len(chosen) < count:
            rng.shuffle(candidates)
            chosen.extend(candidates[: count - len(chosen)])

    return [
        TaskSeed(
            seed_id=i,
            input_kind=item.input_kind,
            operation=item.operation,
            condition=item.condition,
            output_kind=item.output_kind,
            edge_behavior=item.edge_behavior,
            implementation_style=item.implementation_style,
            extra_constraint=item.extra_constraint,
            description_style=item.description_style,
            naming_style=item.naming_style,
        )
        for i, item in enumerate(chosen)
    ]


def user_prompt(seed: TaskSeed) -> str:
    return f"""Create one training example from this semantic seed.

Input: {seed.input_kind}
Operation: {seed.operation}
Condition or relation: {seed.condition}
Required output: {seed.output_kind}
Edge behavior: {seed.edge_behavior}
Implementation style: {seed.implementation_style}
Additional constraint: {seed.extra_constraint}
Description style: {seed.description_style}
Naming style: {seed.naming_style}

Resolve minor ambiguity in the simplest sensible way. Return only <task> and <code>."""


def batches(items: Sequence[TaskSeed], size: int) -> Iterator[Sequence[TaskSeed]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def invalid_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.invalid{output_path.suffix}")


def completed_seed_attempts(paths: Sequence[Path]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    result.add((int(record["seed"]["seed_id"]), int(record["sample_index"])))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    return result


def parse_generation(text: str) -> ParsedGeneration:
    match = OUTPUT_PATTERN.match(text.strip())
    if match is None:
        raise ValueError("missing_or_malformed_tags")
    task_description = match.group("task").strip()
    code = match.group("code").strip()
    if not task_description:
        raise ValueError("empty_task")
    if not code:
        raise ValueError("empty_code")
    code = _strip_top_level_imports(code=code)
    _validate_code(code=code)
    return ParsedGeneration(task_description=task_description, code=code)


def _strip_top_level_imports(code: str) -> str:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code
    if not module.body:
        return code

    non_import_nodes = [
        node for node in module.body if not isinstance(node, ast.Import | ast.ImportFrom)
    ]
    if len(non_import_nodes) != 1 or not isinstance(non_import_nodes[0], ast.FunctionDef):
        return code
    if len(non_import_nodes) == len(module.body):
        return code

    import_lines: set[int] = set()
    for node in module.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            start_line = node.lineno
            end_line = node.end_lineno or node.lineno
            import_lines.update(range(start_line, end_line + 1))

    lines = code.splitlines()
    cleaned_lines = [
        line for line_number, line in enumerate(lines, 1) if line_number not in import_lines
    ]
    return "\n".join(cleaned_lines).strip()


def _validate_code(code: str) -> None:
    try:
        module = ast.parse(code)
    except SyntaxError as error:
        raise ValueError("invalid_python") from error
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ValueError("not_exactly_one_top_level_function")
    function = module.body[0]
    if function.decorator_list:
        raise ValueError("function_has_decorators")
    if function.returns is None:
        raise ValueError("missing_return_annotation")
    for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        if argument.annotation is None:
            raise ValueError("missing_parameter_annotation")


def build_valid_record(
    *,
    model: str,
    seed: TaskSeed,
    sample_index: int,
    parsed: ParsedGeneration,
) -> dict[str, Any]:
    return {
        "model": model,
        "seed": asdict(seed),
        "sample_index": sample_index,
        "task_description": parsed.task_description,
        "code": parsed.code,
    }


def build_invalid_record(
    *,
    model: str,
    seed: TaskSeed,
    sample_index: int,
    generation: str,
    finish_reason: str | None,
    rejection_reason: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "seed": asdict(seed),
        "sample_index": sample_index,
        "generation": generation,
        "finish_reason": finish_reason,
        "rejection_reason": rejection_reason,
    }


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invalid-output", type=Path)
    parser.add_argument("--num-seeds", type=int, default=50_000)
    parser.add_argument("--samples-per-seed", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    invalid_path = args.invalid_output or invalid_output_path(args.output)
    unique_seed_count = unique_compatible_seed_count()
    warning = seed_space_warning(
        requested_seed_count=args.num_seeds,
        unique_seed_count=unique_seed_count,
    )
    if warning is not None:
        print(warning, flush=True)
    seeds = generate_seeds(args.num_seeds, args.seed)
    completed = (
        completed_seed_attempts([args.output, invalid_path])
        if args.resume
        else set()
    )
    pending = [
        seed
        for seed in seeds
        if any(
            (seed.seed_id, sample_index) not in completed
            for sample_index in range(args.samples_per_seed)
        )
    ]

    print(f"model={args.model}")
    print(
        f"total_seeds={len(seeds):,} completed_attempts={len(completed):,} "
        f"pending_seeds={len(pending):,}"
    )
    if not pending:
        return 0

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

    sampling = SamplingParams(
        n=args.samples_per_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated_seeds = 0
    valid_count = 0
    invalid_counts: Counter[str] = Counter()

    for batch_no, batch in enumerate(batches(pending, args.batch_size), 1):
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(seed)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for seed in batch
        ]
        outputs = llm.generate(prompts, sampling, use_tqdm=True)

        for seed, output in zip(batch, outputs, strict=True):
            for sample_index, completion in enumerate(output.outputs):
                if (seed.seed_id, sample_index) in completed:
                    continue
                generation = completion.text.strip()
                finish_reason = completion.finish_reason
                if finish_reason != "stop":
                    rejection_reason = "finish_reason_not_stop"
                else:
                    try:
                        parsed = parse_generation(generation)
                    except ValueError as error:
                        rejection_reason = str(error)
                    else:
                        write_jsonl_record(
                            args.output,
                            build_valid_record(
                                model=args.model,
                                seed=seed,
                                sample_index=sample_index,
                                parsed=parsed,
                            ),
                        )
                        valid_count += 1
                        continue

                invalid_counts[rejection_reason] += 1
                write_jsonl_record(
                    invalid_path,
                    build_invalid_record(
                        model=args.model,
                        seed=seed,
                        sample_index=sample_index,
                        generation=generation,
                        finish_reason=finish_reason,
                        rejection_reason=rejection_reason,
                    ),
                )

        generated_seeds += len(batch)
        elapsed = time.perf_counter() - started
        invalid_total = sum(invalid_counts.values())
        print(
            f"batch={batch_no} seeds={generated_seeds:,}/{len(pending):,} "
            f"valid={valid_count:,} invalid={invalid_total:,} "
            f"rate={generated_seeds / max(elapsed, 1e-9):.2f} seeds/s"
        )

    if invalid_counts:
        print("invalid_reasons=" + json.dumps(dict(invalid_counts), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
