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
    task_family: str
    input_kind: str
    operation: str
    condition: str
    output_kind: str
    edge_behavior: str
    implementation_style: str
    extra_constraint: str
    task_detail: str
    description_style: str
    naming_style: str
    operation_tags: tuple[str, ...]


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

DEFAULT_TASK_DETAILS = [
    "choose a concrete parameter contract implied by the seed and specify it clearly",
]


FAMILIES = [
    {
        "name": "integer_list_atomic",
        "tags": ("list", "integer", "atomic"),
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
        "name": "string_list_atomic",
        "tags": ("list", "string", "atomic"),
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
        "name": "string_atomic",
        "tags": ("string", "character", "atomic"),
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
        "name": "integer_mapping_atomic",
        "tags": ("dict", "integer", "atomic"),
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
        "name": "two_integer_lists_atomic",
        "tags": ("list", "integer", "two-input", "atomic"),
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
    {
        "name": "compositional_list_transform",
        "tags": ("list", "string", "composition", "filter-map"),
        "input": "a list of strings",
        "operations": {
            "filter selected strings, then uppercase and reverse each kept string": (
                "a list of strings"
            ),
            "strip whitespace, drop empty results, then lowercase the remaining strings": (
                "a list of strings"
            ),
            "keep strings matching a predicate, normalize spacing, then sort by length": (
                "a list of strings"
            ),
            "remove duplicates after case normalization while preserving first occurrence": (
                "a list of strings"
            ),
            "return cleaned strings paired with their original indexes": (
                "a list of tuples containing an integer and a string"
            ),
        },
        "conditions": [
            "nonempty after stripping whitespace",
            "contains a substring parameter after case normalization",
            "starts with a prefix parameter ignoring surrounding whitespace",
            "has length inside an inclusive lower and upper bound after stripping",
            "contains at least one alphabetic character and no digits",
            "matches when lowercased value is not already present",
        ],
        "edges": [
            "handle empty and singleton inputs naturally",
            "preserve duplicate transformed values unless the operation removes duplicates",
            "preserve first-occurrence order where possible",
            "return an empty list when no strings match",
            "ignore strings that become empty after normalization",
        ],
        "styles": [
            "use an explicit loop with two or three clear steps",
            "use helper local variables for each transformation stage",
            "use a comprehension only for the final simple projection",
            "build the result incrementally",
        ],
        "extras": [
            "do not mutate the input list",
            "use no imports",
            "avoid clever one-line implementations",
            "make string normalization explicit",
        ],
    },
    {
        "name": "grouped_aggregation",
        "tags": ("dict", "list", "aggregation", "grouping"),
        "input": "a list of dictionaries with string keys and simple values",
        "operations": {
            "group records by a string field and count records in each group": (
                "a dictionary from strings to integers"
            ),
            "group records by a category field and sum an integer amount field": (
                "a dictionary from strings to integers"
            ),
            "group records by a string field and collect selected values into lists": (
                "a dictionary from strings to lists of strings"
            ),
            "find the largest integer value for each group": (
                "a dictionary from strings to integers"
            ),
            "return groups whose aggregate count or sum crosses a threshold": (
                "a dictionary from strings to integers"
            ),
        },
        "conditions": [
            "ignore records missing the required group key",
            "ignore records whose amount value is not an integer",
            "use a default group name parameter when the group value is empty",
            "include only records whose enabled field is true",
            "include only records whose score is nonnegative",
        ],
        "edges": [
            "handle an empty list naturally",
            "handle singleton groups",
            "preserve first-seen group insertion order",
            "keep negative numbers when the operation allows them",
            "return an empty dictionary when no records contribute",
        ],
        "styles": [
            "use an explicit loop over records",
            "use dictionary get for accumulator updates",
            "use setdefault when collecting grouped lists",
            "use clear local variable names for extracted fields",
        ],
        "extras": [
            "do not mutate input dictionaries",
            "use no imports",
            "avoid relying on sorted order unless requested",
        ],
    },
    {
        "name": "nested_data_transform",
        "tags": ("dict", "list", "nested", "optional"),
        "input": "a nested dictionary or list structure using built-in Python values",
        "operations": {
            "extract nested values from records and return only valid values": (
                "a list of strings"
            ),
            "flatten lists stored under dictionary keys while skipping missing keys": (
                "a list of integers"
            ),
            "build a dictionary mapping ids to cleaned nested names": (
                "a dictionary from integers to strings"
            ),
            "return the first record whose nested field satisfies the predicate": (
                "a dictionary or None"
            ),
            "summarize nested item counts per outer key": (
                "a dictionary from strings to integers"
            ),
        },
        "conditions": [
            "nested value exists and is not None",
            "nested list is nonempty",
            "nested string is nonempty after stripping",
            "nested integer is greater than a threshold parameter",
            "nested tag list contains a target tag parameter",
        ],
        "edges": [
            "handle empty outer containers naturally",
            "skip malformed nested entries instead of failing",
            "return None when no nested match exists",
            "preserve outer input order",
            "keep duplicate nested values",
        ],
        "styles": [
            "use explicit isinstance checks where needed",
            "use nested loops when the data shape requires them",
            "use early return for first-match operations",
            "use local variables for intermediate nested values",
        ],
        "extras": [
            "do not mutate nested input structures",
            "use no imports",
            "avoid classes and helper functions",
        ],
    },
    {
        "name": "multi_condition_predicate",
        "tags": ("predicate", "multi-condition", "optional", "edge-case"),
        "input": "a list of integers or strings plus one or two threshold parameters",
        "operations": {
            "return the first value satisfying two conditions": "an integer or None",
            "return whether every value satisfies a compound condition": "a boolean",
            "partition values into accepted and rejected groups": "a tuple of two lists",
            "count values satisfying at least two of three conditions": "an integer",
            "return accepted values after applying a simple transformation": "a list",
        },
        "conditions": [
            "value is positive and inside an inclusive lower and upper bound",
            "value is even and not equal to an excluded parameter",
            "string is nonempty after stripping and contains no whitespace",
            "string starts with a prefix parameter and has length at most a limit parameter",
            "index is odd and value is not a duplicate of a previous value",
        ],
        "edges": [
            "handle empty, singleton, negative, and duplicate inputs",
            "return None when no value passes all required conditions",
            "keep original order in both partition groups",
            "return true for an empty input only when that follows Python all semantics",
            "return zero when no values satisfy the count condition",
        ],
        "styles": [
            "use readable boolean helper variables inside the loop",
            "use an explicit loop",
            "use early return when appropriate",
            "build outputs incrementally",
        ],
        "extras": [
            "do not mutate inputs",
            "use no imports",
            "avoid nested conditional expressions",
        ],
    },
    {
        "name": "small_algorithm",
        "tags": ("algorithm", "list", "string", "multi-step"),
        "input": "a short list or string plus simple scalar parameters",
        "operations": {
            "compute running totals after filtering invalid values": "a list of integers",
            "return the longest increasing contiguous run": "a list of integers",
            "collapse adjacent duplicate values, then count remaining values": "an integer",
            "rotate a list by a nonnegative offset and then drop repeated values": "a list",
            "normalize words, remove stop words, then count frequencies": (
                "a dictionary from strings to integers"
            ),
            "scan characters and return balanced bracket depth after validation": (
                "an integer or None"
            ),
        },
        "conditions": [
            "ignore negative numbers",
            "treat duplicate values as adjacent only when consecutive",
            "use modulo behavior for offsets larger than the list length",
            "ignore empty words after stripping punctuation-like edge characters",
            "return None when validation fails before completing the scan",
        ],
        "edges": [
            "handle empty and singleton inputs naturally",
            "handle negative numbers and duplicates explicitly",
            "return an empty list when no values remain",
            "return zero for empty valid depth computations",
            "preserve stable first-occurrence order after transformations",
        ],
        "styles": [
            "use two to four straightforward processing steps",
            "use explicit loops and local accumulators",
            "avoid clever slicing when a loop is clearer",
            "use helper local variables for each stage",
        ],
        "extras": [
            "do not mutate inputs",
            "use no imports",
            "avoid recursion",
        ],
    },
    {
        "name": "record_list_concrete",
        "tags": ("dict", "list", "records", "aggregation", "concrete"),
        "input": "a list of dictionaries representing small records",
        "operations": {
            "compute a derived dictionary from selected records": (
                "a dictionary from strings to integers"
            ),
            "return cleaned records with selected fields": (
                "a list of dictionaries with string keys and simple values"
            ),
            "find the best matching record by a numeric field": "a dictionary or None",
            "group selected record names by a categorical field": (
                "a dictionary from strings to lists of strings"
            ),
            "return ids of records that pass validation": "a list of integers",
            "merge duplicate records by id using an integer total": (
                "a dictionary from integers to integers"
            ),
        },
        "conditions": [
            "record has an active flag set to true",
            "record status equals a target status parameter",
            "record amount is an integer inside an inclusive range",
            "record name is a nonempty string after stripping",
            "record tags list contains a requested tag",
            "record priority is lower than or equal to a limit parameter",
        ],
        "edges": [
            "skip records with missing or malformed fields",
            "handle empty and singleton record lists naturally",
            "preserve first-seen order for returned names or ids",
            "resolve ties by keeping the first matching record",
            "return None when no record satisfies the required fields",
        ],
        "styles": [
            "use explicit isinstance checks for untrusted record values",
            "use dictionary get and clear local variables",
            "use an explicit loop over records",
            "use setdefault when building grouped lists",
        ],
        "extras": [
            "do not mutate input records",
            "use no imports",
            "avoid sorted unless the task detail explicitly asks for sorting",
        ],
        "details": [
            "records use keys id, name, status, amount, and active; ignore inactive records and sum amount by status",
            "records use keys id, category, score, and tags; return ids whose tags include target_tag and score is nonnegative",
            "records use keys owner, item, and quantity; group item names by owner after stripping whitespace",
            "records use keys id and points; combine duplicate ids by summing integer points",
            "records use keys name, priority, and done; return the unfinished name with the lowest priority",
            "records use keys code, region, and count; return total count per region for codes starting with prefix",
            "records use keys user, enabled, and quota; return users whose enabled flag is true and quota is at least minimum",
            "records use keys team, member, and active; group active members by team while skipping blank names",
        ],
    },
    {
        "name": "grid_matrix_concrete",
        "tags": ("list", "nested", "grid", "matrix", "concrete"),
        "input": "a two-dimensional list of integers or strings",
        "operations": {
            "summarize each row": "a list of integers",
            "summarize each column": "a list of integers",
            "return coordinates that satisfy a predicate": (
                "a list of tuples containing two integers"
            ),
            "replace selected cells in a copied grid": "a two-dimensional list",
            "flatten selected cells while preserving row-major order": "a list",
            "find the first coordinate matching a condition": (
                "a tuple of two integers or None"
            ),
        },
        "conditions": [
            "cell is positive",
            "cell is negative",
            "cell equals a target parameter",
            "cell is a nonempty string after stripping",
            "cell is on the main diagonal",
            "cell has no equal orthogonal neighbor",
        ],
        "edges": [
            "handle an empty grid naturally",
            "handle ragged rows by processing only cells that exist",
            "handle singleton rows and singleton columns",
            "return None when no coordinate matches",
            "do not fail on empty inner rows",
        ],
        "styles": [
            "use nested loops with row and column indexes",
            "build result rows without mutating the input grid",
            "use clear coordinate tuple names",
            "use local variables for row and cell values",
        ],
        "extras": [
            "do not mutate the input grid",
            "use no imports",
            "avoid assuming rectangular rows unless the detail says so",
        ],
        "details": [
            "sum positive integers in each row and return one total per row",
            "count nonempty stripped strings in each column of a rectangular grid",
            "return coordinates of negative integers in ragged row-major order",
            "copy the grid and replace cells equal to target with replacement",
            "return the first coordinate whose value is strictly greater than threshold",
            "flatten diagonal cells from a square integer grid",
            "count cells in each row that differ from their left and right neighbors",
            "return column totals for rows shorter than the widest row by treating missing cells as zero",
        ],
    },
    {
        "name": "string_parsing_concrete",
        "tags": ("string", "parsing", "normalization", "concrete"),
        "input": "a string containing small structured text",
        "operations": {
            "parse tokens into a dictionary": "a dictionary from strings to strings",
            "normalize separated words": "a string",
            "extract valid numeric fields": "a list of integers",
            "count categorized tokens": "a dictionary from strings to integers",
            "return the first valid parsed value": "a string or None",
            "redact selected text segments": "a string",
        },
        "conditions": [
            "token contains an equals sign with nonempty key and value",
            "token is an integer with an optional leading minus sign",
            "token starts with a supplied prefix",
            "token contains only alphabetic characters after stripping",
            "segment is inside square brackets",
            "word is not present in a stop word list",
        ],
        "edges": [
            "handle an empty string naturally",
            "ignore malformed tokens",
            "preserve first occurrence when duplicate keys appear",
            "strip surrounding whitespace from parsed pieces",
            "return None when no valid value exists",
        ],
        "styles": [
            "use split and explicit loops",
            "use simple string methods only",
            "use clear local names for tokens and pieces",
            "avoid regular expressions",
        ],
        "extras": [
            "use no imports",
            "do not use eval or exec",
            "avoid changing case unless the detail requests normalization",
        ],
        "details": [
            "parse comma-separated key=value tokens into a dictionary, keeping the first value for each key",
            "convert words separated by spaces, underscores, or hyphens into a lowercase hyphen slug",
            "extract signed integers from comma-separated tokens, skipping malformed tokens",
            "count lowercase words after stripping periods and commas from their ends",
            "return the first bracketed segment that is nonempty after stripping",
            "redact the local part of an email-like string before the first at sign",
            "parse semicolon-separated name:score pairs and keep scores that are valid integers",
            "normalize repeated whitespace to single spaces and trim the final string",
        ],
    },
    {
        "name": "optional_lookup_concrete",
        "tags": ("dict", "list", "optional", "lookup", "concrete"),
        "input": "one or two dictionaries plus simple lookup parameters",
        "operations": {
            "return a looked-up value after validation": "a string or None",
            "return a derived integer from optional fields": "an integer or None",
            "overlay two mappings without mutating either input": "a dictionary",
            "select keys whose mapped values satisfy a predicate": "a list of strings",
            "fill missing values from fallback data": "a dictionary",
            "compare two mappings and report changed keys": "a list of strings",
        },
        "conditions": [
            "key exists in the primary mapping",
            "value is not None and not an empty string",
            "value is an integer greater than a threshold parameter",
            "fallback value is used only when primary value is missing or None",
            "keys start with a prefix parameter",
            "values differ between two dictionaries",
        ],
        "edges": [
            "handle empty dictionaries naturally",
            "return None when the lookup cannot be completed",
            "preserve insertion order of primary keys first",
            "do not include keys whose final value is None",
            "handle duplicate key choices through normal dictionary behavior",
        ],
        "styles": [
            "use explicit membership checks",
            "copy dictionaries before adding or replacing keys",
            "use dictionary get where it does not hide required None handling",
            "build key lists incrementally",
        ],
        "extras": [
            "do not mutate input dictionaries",
            "use no imports",
            "avoid broad exception handling",
        ],
        "details": [
            "return user display name from profiles[id]['name'] when id exists and the name is nonempty",
            "return the sum of two optional integer fields only when both are present and integers",
            "merge default settings with override settings, skipping override values that are None",
            "return keys whose values are nonempty strings after stripping whitespace",
            "fill missing inventory counts from fallback counts while dropping negative final counts",
            "return changed keys sorted by their first appearance in the primary mapping then the secondary mapping",
            "return a lowercase email value for a user id when it contains exactly one at sign",
            "build a mapping of requested keys to values found in primary or fallback dictionaries",
        ],
    },
    {
        "name": "sequence_algorithm_concrete",
        "tags": ("list", "algorithm", "sequence", "concrete"),
        "input": "a list of integers or strings",
        "operations": {
            "find contiguous segments": "a list of lists",
            "compute adjacent differences or transitions": "a list",
            "summarize windows of fixed size": "a list of integers",
            "remove or collapse repeated values": "a list",
            "return indexes of structural positions": "a list of integers",
            "choose a best segment by length or total": "a list",
        },
        "conditions": [
            "value changes from the previous value",
            "window sum is at least a threshold parameter",
            "segment contains no negative numbers",
            "string value changes after case normalization",
            "value is a strict local peak",
            "run length is at least a minimum parameter",
        ],
        "edges": [
            "handle empty and singleton lists naturally",
            "handle duplicate and negative values explicitly",
            "resolve ties by keeping the earliest segment",
            "return an empty list when no segment qualifies",
            "avoid indexing past either end of the list",
        ],
        "styles": [
            "use one pass when practical",
            "use clear start and end index variables",
            "use explicit loops instead of recursion",
            "use local accumulators for current and best segments",
        ],
        "extras": [
            "do not mutate input lists",
            "use no imports",
            "avoid clever one-line implementations",
        ],
        "details": [
            "return lengths of consecutive equal-value runs",
            "return adjacent integer differences as current minus previous",
            "return indexes of strict local peaks excluding endpoints",
            "return the longest contiguous segment containing only nonnegative values",
            "collapse case-insensitive adjacent duplicate strings while preserving original spelling of the first item",
            "return sums of all complete windows of size width",
            "return segments separated by zero values, excluding the zero separators",
            "return values that are larger than every value seen before them",
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
    return True


def compatible_seed_candidates() -> list[TaskSeed]:
    candidates: list[TaskSeed] = []
    seed_id = 0
    for family in FAMILIES:
        details = family.get("details", DEFAULT_TASK_DETAILS)
        for operation, output_kind in family["operations"].items():
            for (
                condition,
                edge,
                style,
                extra,
                task_detail,
                description_style,
                naming_style,
            ) in itertools.product(
                family["conditions"],
                family["edges"],
                family["styles"],
                family["extras"],
                details,
                DESCRIPTION_STYLES,
                NAMING_STYLES,
            ):
                item = TaskSeed(
                    seed_id=seed_id,
                    task_family=family["name"],
                    input_kind=family["input"],
                    operation=operation,
                    condition=condition,
                    output_kind=output_kind,
                    edge_behavior=edge,
                    implementation_style=style,
                    extra_constraint=extra,
                    task_detail=task_detail,
                    description_style=description_style,
                    naming_style=naming_style,
                    operation_tags=tuple(family["tags"]),
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


def generate_seeds(
    count: int,
    rng_seed: int,
    excluded_semantic_keys: set[tuple[object, ...]] | None = None,
) -> list[TaskSeed]:
    candidates = compatible_seed_candidates()
    if excluded_semantic_keys is not None:
        candidates = [
            candidate
            for candidate in candidates
            if semantic_seed_key(candidate) not in excluded_semantic_keys
        ]
    if not candidates:
        raise ValueError("No compatible seed candidates remain after exclusions.")

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
            task_family=item.task_family,
            input_kind=item.input_kind,
            operation=item.operation,
            condition=item.condition,
            output_kind=item.output_kind,
            edge_behavior=item.edge_behavior,
            implementation_style=item.implementation_style,
            extra_constraint=item.extra_constraint,
            task_detail=item.task_detail,
            description_style=item.description_style,
            naming_style=item.naming_style,
            operation_tags=item.operation_tags,
        )
        for i, item in enumerate(chosen)
    ]


def semantic_seed_key(seed: TaskSeed) -> tuple[object, ...]:
    return (
        seed.task_family,
        seed.input_kind,
        seed.operation,
        seed.condition,
        seed.output_kind,
        seed.edge_behavior,
        seed.implementation_style,
        seed.extra_constraint,
        seed.task_detail,
        seed.description_style,
        seed.naming_style,
        seed.operation_tags,
    )


def excluded_training_seed_keys(
    *,
    count: int,
    rng_seed: int,
) -> set[tuple[object, ...]]:
    return {
        semantic_seed_key(seed)
        for seed in generate_seeds(count=count, rng_seed=rng_seed)
    }


def user_prompt(seed: TaskSeed) -> str:
    return f"""Create one training example from this semantic seed.

Input: {seed.input_kind}
Task family: {seed.task_family}
Operation tags: {", ".join(seed.operation_tags)}
Operation: {seed.operation}
Condition or relation: {seed.condition}
Required output: {seed.output_kind}
Edge behavior: {seed.edge_behavior}
Implementation style: {seed.implementation_style}
Additional constraint: {seed.extra_constraint}
Concrete task detail: {seed.task_detail}
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
        "task_family": seed.task_family,
        "operation_tags": list(seed.operation_tags),
        "task_detail": seed.task_detail,
        "signature": _function_signature_line(code=parsed.code),
        "normalized_description": _normalize_description(parsed.task_description),
        "task_description": parsed.task_description,
        "code": parsed.code,
    }


def _function_signature_line(code: str) -> str:
    return code.strip().splitlines()[0].strip()


def _normalize_description(task_description: str) -> str:
    return " ".join(task_description.casefold().split())


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
    parser.add_argument("--num-seeds", type=int, default=500_000)
    parser.add_argument("--samples-per-seed", type=int, default=2)
    parser.add_argument("--exclude-num-seeds", type=int, default=0)
    parser.add_argument("--exclude-seed", type=int, default=42)
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
    excluded_keys = (
        excluded_training_seed_keys(
            count=args.exclude_num_seeds,
            rng_seed=args.exclude_seed,
        )
        if args.exclude_num_seeds > 0
        else None
    )
    if excluded_keys is not None:
        print(
            f"excluded_semantic_seeds={len(excluded_keys):,} "
            f"exclude_seed={args.exclude_seed}",
            flush=True,
        )
    seeds = generate_seeds(
        count=args.num_seeds,
        rng_seed=args.seed,
        excluded_semantic_keys=excluded_keys,
    )
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
