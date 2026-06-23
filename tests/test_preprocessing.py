import pytest

from llm_lite.config.models import (
    AssignSplitTransformConfiguration,
    ExactDeduplicationTransformConfiguration,
    LowerCaseTransformConfiguration,
    MaxLengthTransformConfiguration,
    MinLengthTransformConfiguration,
    NormalizeLineEndingsTransformConfiguration,
    NormalizeUnicodeTransformConfiguration,
    PreprocessingTransformType,
)
from llm_lite.data.document import Document
from llm_lite.data.preprocessing import preprocess_documents


def test_preprocess_documents_normalizes_line_endings() -> None:
    result = preprocess_documents(
        documents=[
            Document(
                document_id="document-1",
                text="hello\r\nworld\r",
                split=None,
            ),
        ],
        transforms=(
            NormalizeLineEndingsTransformConfiguration(
                type=PreprocessingTransformType.NORMALIZE_LINE_ENDINGS,
            ),
        ),
    )

    documents = list(result.documents)

    assert documents[0].text == "hello\nworld\n"
    assert result.counters.input_documents == 1
    assert result.counters.output_documents == 1


def test_preprocess_documents_filters_by_length() -> None:
    result = preprocess_documents(
        documents=[
            Document(document_id="short", text="hi", split=None),
            Document(document_id="long", text="hello", split=None),
        ],
        transforms=(
            MinLengthTransformConfiguration(
                type=PreprocessingTransformType.MIN_LENGTH,
                min_characters=3,
            ),
            MaxLengthTransformConfiguration(
                type=PreprocessingTransformType.MAX_LENGTH,
                max_characters=5,
            ),
        ),
    )

    documents = list(result.documents)

    assert [document.document_id for document in documents] == ["long"]
    assert result.counters.input_documents == 2
    assert result.counters.output_documents == 1
    assert result.counters.rejected_documents == 1


def test_preprocess_documents_normalizes_unicode() -> None:
    result = preprocess_documents(
        documents=[
            Document(
                document_id="document-1",
                text="Cafe\u0301",
                split=None,
            ),
        ],
        transforms=(
            NormalizeUnicodeTransformConfiguration(
                type=PreprocessingTransformType.NORMALIZE_UNICODE,
                form="NFC",
            ),
        ),
    )

    documents = list(result.documents)

    assert documents[0].text == "Caf\u00e9"
    assert result.counters.unicode_normalized_documents == 1


def test_preprocess_documents_lower_case_is_optional_transform() -> None:
    result = preprocess_documents(
        documents=[
            Document(
                document_id="document-1",
                text="Hello",
                split=None,
            ),
        ],
        transforms=(LowerCaseTransformConfiguration(type=PreprocessingTransformType.LOWER_CASE),),
    )

    documents = list(result.documents)

    assert documents[0].text == "hello"
    assert result.counters.lower_cased_documents == 1


def test_preprocess_documents_exact_deduplication() -> None:
    result = preprocess_documents(
        documents=[
            Document(document_id="document-1", text="same", split=None),
            Document(document_id="document-2", text="same", split=None),
            Document(document_id="document-3", text="different", split=None),
        ],
        transforms=(
            ExactDeduplicationTransformConfiguration(
                type=PreprocessingTransformType.EXACT_DEDUPLICATION,
            ),
        ),
    )

    documents = list(result.documents)

    assert [document.document_id for document in documents] == ["document-1", "document-3"]
    assert result.counters.deduplicated_documents == 1
    assert result.counters.rejected_documents == 1


def test_preprocess_documents_assigns_split_deterministically() -> None:
    transform = AssignSplitTransformConfiguration(
        type=PreprocessingTransformType.ASSIGN_SPLIT,
        train_probability=0.5,
        validation_probability=0.25,
        test_probability=0.25,
    )
    documents = [Document(document_id="document-1", text="text", split=None)]

    first_result = preprocess_documents(documents=documents, transforms=(transform,))
    second_result = preprocess_documents(documents=documents, transforms=(transform,))

    first_document = list(first_result.documents)[0]
    second_document = list(second_result.documents)[0]
    assert first_document.split == second_document.split


def test_preprocess_documents_rejects_reassigning_existing_split() -> None:
    transform = AssignSplitTransformConfiguration(
        type=PreprocessingTransformType.ASSIGN_SPLIT,
        train_probability=0.5,
        validation_probability=0.25,
        test_probability=0.25,
    )
    result = preprocess_documents(
        documents=[Document(document_id="document-1", text="text", split="validation")],
        transforms=(transform,),
    )

    with pytest.raises(ValueError, match="existing document split"):
        list(result.documents)
