from collections.abc import Iterator

import pytest

from llm_lite.config.models import (
    DatasetType,
    HuggingFaceDatasetConfiguration,
    HuggingFaceDatasetSplitConfiguration,
)
from llm_lite.data import sources


def test_iter_huggingface_documents_maps_source_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def load_dataset_stub(path: str, split: str, streaming: bool) -> Iterator[dict[str, str]]:
        calls.append(f"{path}:{split}:{streaming}")
        yield {"text": f"{split} story 1"}
        yield {"text": f"{split} story 2"}

    monkeypatch.setattr(sources, "load_dataset", load_dataset_stub)
    dataset_configuration = HuggingFaceDatasetConfiguration(
        type=DatasetType.HUGGINGFACE,
        name="roneneldan/TinyStories",
        text_column="text",
        streaming=True,
        splits=(
            HuggingFaceDatasetSplitConfiguration(
                source_split="train",
                split="train",
                max_documents=1,
            ),
            HuggingFaceDatasetSplitConfiguration(
                source_split="validation",
                split="validation_small",
                max_documents=2,
            ),
        ),
    )

    documents = list(
        sources.iter_huggingface_documents(dataset_configuration=dataset_configuration),
    )

    assert calls == [
        "roneneldan/TinyStories:train:True",
        "roneneldan/TinyStories:validation:True",
    ]
    assert [document.document_id for document in documents] == [
        "train-00000000",
        "validation_small-00000000",
        "validation_small-00000001",
    ]
    assert [document.split for document in documents] == [
        "train",
        "validation_small",
        "validation_small",
    ]


def test_iter_huggingface_documents_filters_and_offsets_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_dataset_stub(path: str, split: str, streaming: bool) -> Iterator[dict[str, str]]:
        assert path == "codeparrot/github-code"
        assert split == "train"
        assert streaming is True
        yield {"code": "print('js')", "language": "JavaScript", "license": "mit"}
        yield {"code": "print('skip')", "language": "Python", "license": "mit"}
        yield {"code": "print('emit')", "language": "Python", "license": "mit"}
        yield {"code": "print('apache')", "language": "Python", "license": "apache-2.0"}

    monkeypatch.setattr(sources, "load_dataset", load_dataset_stub)
    dataset_configuration = HuggingFaceDatasetConfiguration(
        type=DatasetType.HUGGINGFACE,
        name="codeparrot/github-code",
        text_column="code",
        language_column="language",
        languages=("Python",),
        license_column="license",
        licenses=("mit",),
        streaming=True,
        splits=(
            HuggingFaceDatasetSplitConfiguration(
                source_split="train",
                split="validation",
                skip_documents=1,
                max_documents=1,
            ),
        ),
    )

    documents = list(
        sources.iter_huggingface_documents(dataset_configuration=dataset_configuration),
    )

    assert len(documents) == 1
    assert documents[0].document_id == "validation-00000000"
    assert documents[0].text == "print('emit')"
    assert documents[0].split == "validation"
