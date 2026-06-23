import glob
import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

from llm_lite.config.models import (
    ExperimentFile,
    InlineTextDatasetConfiguration,
    LocalTextDatasetConfiguration,
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


def iter_inline_documents(
    dataset_configuration: InlineTextDatasetConfiguration,
) -> Iterator[Document]:
    for document_index, document_text in enumerate(dataset_configuration.documents):
        text_bytes = document_text.encode("utf-8")
        yield Document(
            document_id=f"inline-{document_index:06d}",
            text=document_text,
            metadata={
                "source": "inline_text",
                "index": document_index,
                "byte_size": len(text_bytes),
                "content_hash": _content_hash(content_bytes=text_bytes),
            },
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
            text=content_bytes.decode("utf-8"),
            metadata={
                "source": "local_text",
                "path": normalized_path,
                "byte_size": len(content_bytes),
                "content_hash": content_hash,
            },
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
