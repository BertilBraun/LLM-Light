import glob
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from datasets import load_dataset

from llm_lite.config.models import (
    ExperimentFile,
    HuggingFaceDatasetConfiguration,
    HuggingFaceDatasetSplitConfiguration,
    InlineTextDatasetConfiguration,
    LocalTextDatasetConfiguration,
    TinyPythonJsonlDatasetConfiguration,
)
from llm_lite.data.document import Document


def iter_dataset_documents(experiment_configuration: ExperimentFile) -> Iterator[Document]:
    match experiment_configuration.dataset:
        case InlineTextDatasetConfiguration():
            yield from iter_inline_documents(
                dataset_configuration=experiment_configuration.dataset,
            )
        case LocalTextDatasetConfiguration():
            yield from iter_local_text_documents(
                dataset_configuration=experiment_configuration.dataset,
            )
        case TinyPythonJsonlDatasetConfiguration():
            yield from iter_tinypython_jsonl_documents(
                dataset_configuration=experiment_configuration.dataset,
            )
        case HuggingFaceDatasetConfiguration():
            yield from iter_huggingface_documents(
                dataset_configuration=experiment_configuration.dataset,
            )


def iter_inline_documents(
    dataset_configuration: InlineTextDatasetConfiguration,
) -> Iterator[Document]:
    for document_index, document_text in enumerate(dataset_configuration.documents):
        yield Document(
            document_id=f"inline-{document_index:06d}",
            text=document_text,
            split=None,
        )


def iter_local_text_documents(
    dataset_configuration: LocalTextDatasetConfiguration,
) -> Iterator[Document]:
    for text_path in resolve_local_text_paths(dataset_configuration=dataset_configuration):
        content_bytes = text_path.read_bytes()
        content_hash = _content_hash(content_bytes=content_bytes)
        normalized_path = _normalized_path(path=text_path)
        yield Document(
            document_id=_document_id(path=normalized_path, content_hash=content_hash),
            text=text_path.read_text(encoding="utf-8"),
            split=None,
        )


def iter_tinypython_jsonl_documents(
    dataset_configuration: TinyPythonJsonlDatasetConfiguration,
) -> Iterator[Document]:
    for jsonl_path in resolve_tinypython_jsonl_paths(dataset_configuration=dataset_configuration):
        normalized_path = _normalized_path(path=jsonl_path)
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = _tinypython_jsonl_record(
                    line=line,
                    path=jsonl_path,
                    line_index=line_index,
                )
                text = _tinypython_training_text(record=record)
                document_id = _tinypython_document_id(
                    path=normalized_path,
                    line_index=line_index,
                    text=text,
                )
                yield Document(
                    document_id=document_id,
                    text=text,
                    split=_assigned_split(
                        document_id=document_id,
                        train_probability=dataset_configuration.train_probability,
                        validation_probability=dataset_configuration.validation_probability,
                    ),
                )


def resolve_tinypython_jsonl_paths(
    dataset_configuration: TinyPythonJsonlDatasetConfiguration,
) -> tuple[Path, ...]:
    resolved_paths: set[Path] = set()
    for configured_path in dataset_configuration.paths:
        resolved_path = configured_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise ValueError(f"TinyPython JSONL path is not a file: {configured_path}")
        resolved_paths.add(resolved_path)
    for glob_pattern in dataset_configuration.glob_patterns:
        resolved_paths.update(_resolve_glob_pattern(glob_pattern=glob_pattern))
    return tuple(sorted(resolved_paths, key=_path_sort_key))


def iter_huggingface_documents(
    dataset_configuration: HuggingFaceDatasetConfiguration,
) -> Iterator[Document]:
    for split_configuration in dataset_configuration.splits:
        yield from _iter_huggingface_split_documents(
            dataset_configuration=dataset_configuration,
            split_configuration=split_configuration,
        )


def resolve_local_text_paths(
    dataset_configuration: LocalTextDatasetConfiguration,
) -> tuple[Path, ...]:
    resolved_paths: set[Path] = set()
    for configured_path in dataset_configuration.paths:
        resolved_path = configured_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise ValueError(f"Local text path is not a file: {configured_path}")
        resolved_paths.add(resolved_path)
    for glob_pattern in dataset_configuration.glob_patterns:
        glob_paths = _resolve_glob_pattern(glob_pattern=glob_pattern)
        resolved_paths.update(glob_paths)
    return tuple(sorted(resolved_paths, key=_path_sort_key))


def _resolve_glob_pattern(glob_pattern: str) -> Iterable[Path]:
    matched_paths = glob.glob(glob_pattern, recursive=True)
    for matched_path in matched_paths:
        resolved_path = Path(matched_path).expanduser().resolve()
        if resolved_path.is_file():
            yield resolved_path


def _path_sort_key(path: Path) -> tuple[str, str]:
    normalized_path = _normalized_path(path=path)
    return (normalized_path.casefold(), normalized_path)


def _normalized_path(path: Path) -> str:
    return path.resolve().as_posix()


def _content_hash(content_bytes: bytes) -> str:
    return f"sha256:{hashlib.sha256(content_bytes).hexdigest()}"


def _document_id(path: str, content_hash: str) -> str:
    identifier_hash = hashlib.sha256(f"{path}\n{content_hash}".encode()).hexdigest()
    return f"local-text-{identifier_hash[:24]}"


def _tinypython_jsonl_record(
    line: str,
    path: Path,
    line_index: int,
) -> Mapping[str, object]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid TinyPython JSONL at {path}:{line_index}") from error
    if not isinstance(record, Mapping):
        raise ValueError(f"TinyPython JSONL record must be an object at {path}:{line_index}")
    return record


def _tinypython_training_text(record: Mapping[str, object]) -> str:
    task_description = _record_text(row=record, text_column="task_description").strip()
    code = _record_text(row=record, text_column="code").strip()
    if not task_description:
        raise ValueError("TinyPython JSONL task_description must not be empty.")
    if not code:
        raise ValueError("TinyPython JSONL code must not be empty.")
    return f"{task_description}\n\n{code}\n"


def _tinypython_document_id(path: str, line_index: int, text: str) -> str:
    content_hash = _content_hash(content_bytes=text.encode("utf-8"))
    identifier_hash = hashlib.sha256(f"{path}\n{line_index}\n{content_hash}".encode()).hexdigest()
    return f"tinypython-{identifier_hash[:24]}"


def _assigned_split(
    document_id: str,
    train_probability: float,
    validation_probability: float,
) -> str:
    hash_value = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:16], 16)
    normalized_value = hash_value / float(16**16 - 1)
    validation_threshold = train_probability + validation_probability
    if normalized_value < train_probability:
        return "train"
    if normalized_value < validation_threshold:
        return "validation"
    return "test"


def _iter_huggingface_split_documents(
    dataset_configuration: HuggingFaceDatasetConfiguration,
    split_configuration: HuggingFaceDatasetSplitConfiguration,
) -> Iterator[Document]:
    dataset = load_dataset(
        path=dataset_configuration.name,
        split=split_configuration.source_split,
        streaming=dataset_configuration.streaming,
    )
    accepted_rows = 0
    emitted_rows = 0
    for row in dataset:
        if not _row_matches_filters(row=row, dataset_configuration=dataset_configuration):
            continue
        if accepted_rows < split_configuration.skip_documents:
            accepted_rows += 1
            continue
        if (
            split_configuration.max_documents is not None
            and emitted_rows >= split_configuration.max_documents
        ):
            break
        text = _record_text(row=row, text_column=dataset_configuration.text_column)
        yield Document(
            document_id=f"{split_configuration.split}-{emitted_rows:08d}",
            text=text,
            split=split_configuration.split,
        )
        accepted_rows += 1
        emitted_rows += 1


def _record_text(row: Mapping[str, object], text_column: str) -> str:
    text_value = row[text_column]
    match text_value:
        case str():
            return text_value
        case _:
            raise ValueError("Hugging Face text column must contain strings.")


def _row_matches_filters(
    row: Mapping[str, object],
    dataset_configuration: HuggingFaceDatasetConfiguration,
) -> bool:
    if dataset_configuration.languages:
        if dataset_configuration.language_column is None:
            raise ValueError("Hugging Face language filters require language_column.")
        language_value = _optional_record_text(
            row=row,
            column=dataset_configuration.language_column,
        )
        if language_value not in dataset_configuration.languages:
            return False
    if dataset_configuration.licenses:
        if dataset_configuration.license_column is None:
            raise ValueError("Hugging Face license filters require license_column.")
        license_value = _optional_record_text(
            row=row,
            column=dataset_configuration.license_column,
        )
        if license_value not in dataset_configuration.licenses:
            return False
    return True


def _optional_record_text(row: Mapping[str, object], column: str) -> str:
    text_value = row[column]
    match text_value:
        case str():
            return text_value
        case _:
            raise ValueError("Hugging Face filter columns must contain strings.")
