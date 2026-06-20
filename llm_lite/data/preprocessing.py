from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from llm_lite.config.models import (
    MaxLengthTransformConfiguration,
    MinLengthTransformConfiguration,
    NormalizeLineEndingsTransformConfiguration,
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


def preprocess_documents(
    documents: Iterable[Document],
    transforms: tuple[PreprocessingTransformConfiguration, ...],
) -> PreprocessingResult:
    counters = PreprocessingCounters()

    def iter_processed_documents() -> Iterator[Document]:
        for document in documents:
            counters.input_documents += 1
            counters.input_bytes += len(document.text.encode("utf-8"))
            transformed_document = _apply_transforms(document=document, transforms=transforms)
            if transformed_document is None:
                counters.rejected_documents += 1
                continue
            counters.output_documents += 1
            counters.output_bytes += len(transformed_document.text.encode("utf-8"))
            yield transformed_document

    return PreprocessingResult(
        documents=iter_processed_documents(),
        counters=counters,
    )


def _apply_transforms(
    document: Document,
    transforms: tuple[PreprocessingTransformConfiguration, ...],
) -> Document | None:
    current_document: Document | None = document
    for transform in transforms:
        if current_document is None:
            return None
        current_document = _apply_transform(document=current_document, transform=transform)
    return current_document


def _apply_transform(
    document: Document,
    transform: PreprocessingTransformConfiguration,
) -> Document | None:
    match transform:
        case NormalizeLineEndingsTransformConfiguration():
            return Document(
                document_id=document.document_id,
                text=document.text.replace("\r\n", "\n").replace("\r", "\n"),
                metadata=document.metadata,
            )
        case MinLengthTransformConfiguration(min_characters=min_characters):
            if len(document.text) < min_characters:
                return None
            return document
        case MaxLengthTransformConfiguration(max_characters=max_characters):
            if len(document.text) > max_characters:
                return None
            return document
