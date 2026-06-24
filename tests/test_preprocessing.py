import pytest

from llm_lite.config.models import (
    AssignSplitTransformConfiguration,
    ExactDeduplicationTransformConfiguration,
    ExtractPythonFunctionsTransformConfiguration,
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


def test_preprocess_documents_extracts_top_level_python_functions() -> None:
    result = preprocess_documents(
        documents=[
            Document(
                document_id="module",
                text=(
                    "import os\n\n"
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n\n"
                    "def _private() -> int:\n"
                    "    return 1\n\n"
                    "class Calculator:\n"
                    "    def multiply(self, left: int, right: int) -> int:\n"
                    "        return left * right\n"
                ),
                split="train",
            ),
        ],
        transforms=(
            ExtractPythonFunctionsTransformConfiguration(
                type=PreprocessingTransformType.EXTRACT_PYTHON_FUNCTIONS,
                include_async_functions=False,
                include_private_functions=False,
                include_methods=False,
            ),
        ),
    )

    documents = list(result.documents)

    assert len(documents) == 1
    assert documents[0].document_id == "module__function_0000_add"
    assert documents[0].split == "train"
    assert documents[0].text == "def add(a: int, b: int) -> int:\n    return a + b\n"
    assert "import os" not in documents[0].text
    assert result.counters.python_extracted_functions == 1


def test_preprocess_documents_can_extract_python_methods() -> None:
    result = preprocess_documents(
        documents=[
            Document(
                document_id="module",
                text=(
                    "class Calculator:\n"
                    "    def multiply(self, left: int, right: int) -> int:\n"
                    "        return left * right\n"
                ),
                split=None,
            ),
        ],
        transforms=(
            ExtractPythonFunctionsTransformConfiguration(
                type=PreprocessingTransformType.EXTRACT_PYTHON_FUNCTIONS,
                include_async_functions=False,
                include_private_functions=False,
                include_methods=True,
            ),
        ),
    )

    documents = list(result.documents)

    assert len(documents) == 1
    assert documents[0].text.startswith("def multiply")


def test_preprocess_documents_rejects_unparseable_python_for_function_extraction() -> None:
    result = preprocess_documents(
        documents=[Document(document_id="bad", text="def broken(:\n", split=None)],
        transforms=(
            ExtractPythonFunctionsTransformConfiguration(
                type=PreprocessingTransformType.EXTRACT_PYTHON_FUNCTIONS,
                include_async_functions=False,
                include_private_functions=False,
                include_methods=False,
            ),
        ),
    )

    assert list(result.documents) == []
    assert result.counters.python_parse_failed_documents == 1
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
