import hashlib
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from llm_lite.config.models import (
    AssignSplitTransformConfiguration,
    ExactDeduplicationTransformConfiguration,
    LowerCaseTransformConfiguration,
    MaxLengthTransformConfiguration,
    MinLengthTransformConfiguration,
    NormalizeLineEndingsTransformConfiguration,
    NormalizeUnicodeTransformConfiguration,
    PreprocessingTransformConfiguration,
)
from llm_lite.data.document import Document


@dataclass(frozen=True)
class PreprocessingResult:
    documents: Iterator[Document]
    counters: "PreprocessingCounters"


@dataclass
class PreprocessingCounters:
    input_documents: int = 0
    output_documents: int = 0
    rejected_documents: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    input_characters: int = 0
    output_characters: int = 0
    unicode_normalized_documents: int = 0
    line_endings_normalized_documents: int = 0
    lower_cased_documents: int = 0
    deduplicated_documents: int = 0
    split_assigned_documents: int = 0


def preprocess_documents(
    documents: Iterable[Document],
    transforms: tuple[PreprocessingTransformConfiguration, ...],
) -> PreprocessingResult:
    counters = PreprocessingCounters()
    seen_document_hashes: set[str] = set()

    def iter_processed_documents() -> Iterator[Document]:
        for document in documents:
            counters.input_documents += 1
            counters.input_bytes += len(document.text.encode("utf-8"))
            counters.input_characters += len(document.text)
            transformed_document = _apply_transforms(
                document=document,
                transforms=transforms,
                counters=counters,
                seen_document_hashes=seen_document_hashes,
            )
            if transformed_document is None:
                counters.rejected_documents += 1
                continue
            counters.output_documents += 1
            counters.output_bytes += len(transformed_document.text.encode("utf-8"))
            counters.output_characters += len(transformed_document.text)
            yield transformed_document

    return PreprocessingResult(
        documents=iter_processed_documents(),
        counters=counters,
    )


def _apply_transforms(
    document: Document,
    transforms: tuple[PreprocessingTransformConfiguration, ...],
    counters: PreprocessingCounters,
    seen_document_hashes: set[str],
) -> Document | None:
    current_document: Document | None = document
    for transform in transforms:
        if current_document is None:
            return None
        current_document = _apply_transform(
            document=current_document,
            transform=transform,
            counters=counters,
            seen_document_hashes=seen_document_hashes,
        )
    return current_document


def _apply_transform(
    document: Document,
    transform: PreprocessingTransformConfiguration,
    counters: PreprocessingCounters,
    seen_document_hashes: set[str],
) -> Document | None:
    match transform:
        case NormalizeUnicodeTransformConfiguration(form=form):
            normalized_text = unicodedata.normalize(form, document.text)
            if normalized_text != document.text:
                counters.unicode_normalized_documents += 1
            return Document(
                document_id=document.document_id,
                text=normalized_text,
                metadata=document.metadata,
            )
        case NormalizeLineEndingsTransformConfiguration():
            normalized_text = document.text.replace("\r\n", "\n").replace("\r", "\n")
            if normalized_text != document.text:
                counters.line_endings_normalized_documents += 1
            return Document(
                document_id=document.document_id,
                text=normalized_text,
                metadata=document.metadata,
            )
        case LowerCaseTransformConfiguration():
            lowered_text = document.text.lower()
            if lowered_text != document.text:
                counters.lower_cased_documents += 1
            return Document(
                document_id=document.document_id,
                text=lowered_text,
                metadata=document.metadata,
            )
        case ExactDeduplicationTransformConfiguration():
            document_hash = _text_hash(text=document.text)
            if document_hash in seen_document_hashes:
                counters.deduplicated_documents += 1
                return None
            seen_document_hashes.add(document_hash)
            return Document(
                document_id=document.document_id,
                text=document.text,
                metadata={**document.metadata, "processed_content_hash": document_hash},
            )
        case MinLengthTransformConfiguration(min_characters=min_characters):
            if len(document.text) < min_characters:
                return None
            return document
        case MaxLengthTransformConfiguration(max_characters=max_characters):
            if len(document.text) > max_characters:
                return None
            return document
        case AssignSplitTransformConfiguration(
            train_probability=train_probability,
            validation_probability=validation_probability,
        ):
            counters.split_assigned_documents += 1
            split_name = _assigned_split(
                document_id=document.document_id,
                train_probability=train_probability,
                validation_probability=validation_probability,
            )
            return Document(
                document_id=document.document_id,
                text=document.text,
                metadata={**document.metadata, "split": split_name},
            )


def _text_hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


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
