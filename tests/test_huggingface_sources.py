from collections.abc import Iterator

import pytest
from pydantic import ValidationError

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


def test_iter_huggingface_documents_formats_template_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_dataset_stub(path: str, split: str, streaming: bool) -> Iterator[dict[str, str]]:
        assert path == "BertilBraun/TinyPython"
        assert split == "train"
        assert streaming is True
        yield {
            "task_description": "Return the number of values.",
            "code": "def count_values(values: list[int]) -> int:\n    return len(values)",
        }

    monkeypatch.setattr(sources, "load_dataset", load_dataset_stub)
    dataset_configuration = HuggingFaceDatasetConfiguration(
        type=DatasetType.HUGGINGFACE,
        name="BertilBraun/TinyPython",
        text_template="{task_description}\n\n{code}\n",
        streaming=True,
        splits=(
            HuggingFaceDatasetSplitConfiguration(
                source_split="train",
                split="train",
            ),
        ),
    )

    documents = list(
        sources.iter_huggingface_documents(dataset_configuration=dataset_configuration),
    )

    assert len(documents) == 1
    assert documents[0].document_id == "train-00000000"
    assert documents[0].split == "train"
    assert documents[0].text == (
        "Return the number of values.\n\n"
        "def count_values(values: list[int]) -> int:\n"
        "    return len(values)\n"
    )


def test_huggingface_configuration_rejects_ambiguous_text_sources() -> None:
    split_configuration = HuggingFaceDatasetSplitConfiguration(
        source_split="train",
        split="train",
    )

    with pytest.raises(ValidationError, match="exactly one text source"):
        HuggingFaceDatasetConfiguration(
            type=DatasetType.HUGGINGFACE,
            name="BertilBraun/TinyPython",
            text_column="code",
            text_template="{task_description}\n\n{code}\n",
            splits=(split_configuration,),
        )

    with pytest.raises(ValidationError, match="exactly one text source"):
        HuggingFaceDatasetConfiguration(
            type=DatasetType.HUGGINGFACE,
            name="BertilBraun/TinyPython",
            splits=(split_configuration,),
        )
