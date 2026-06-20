from llm_lite.config.models import (
    MaxLengthTransformConfiguration,
    MinLengthTransformConfiguration,
    NormalizeLineEndingsTransformConfiguration,
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
                metadata={},
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
            Document(document_id="short", text="hi", metadata={}),
            Document(document_id="long", text="hello", metadata={}),
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
