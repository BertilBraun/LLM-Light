import json
from pathlib import Path

from llm_lite.config.models import (
    DatasetType,
    LocalTextDatasetConfiguration,
    TinyPythonJsonlDatasetConfiguration,
)
from llm_lite.data.sources import (
    iter_local_text_documents,
    iter_tinypython_jsonl_documents,
    resolve_local_text_paths,
)


def test_resolve_local_text_paths_deduplicates_and_sorts() -> None:
    dataset_configuration = LocalTextDatasetConfiguration(
        type=DatasetType.LOCAL_TEXT,
        paths=(Path("tests/fixtures/local_text/nested/c_story.txt"),),
        glob_patterns=("tests/fixtures/local_text/*.txt",),
    )

    resolved_paths = resolve_local_text_paths(dataset_configuration=dataset_configuration)
    path_keys = [
        (resolved_path.as_posix().casefold(), resolved_path.as_posix())
        for resolved_path in resolved_paths
    ]

    assert path_keys == sorted(path_keys)
    assert len(resolved_paths) == len(set(resolved_paths))


def test_iter_local_text_documents_records_stable_document() -> None:
    dataset_configuration = LocalTextDatasetConfiguration(
        type=DatasetType.LOCAL_TEXT,
        paths=(Path("tests/fixtures/local_text/hello_world.txt"),),
        glob_patterns=(),
    )

    documents = list(iter_local_text_documents(dataset_configuration=dataset_configuration))

    assert len(documents) == 1
    assert documents[0].document_id.startswith("local-text-")
    assert documents[0].text == "hello world\n"
    assert documents[0].split is None


def test_iter_tinypython_jsonl_documents_formats_task_and_code(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "teacher.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "task_description": "Return the number of values.",
                "code": "def count_values(values: list[int]) -> int:\n    return len(values)",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_configuration = TinyPythonJsonlDatasetConfiguration(
        type=DatasetType.TINYPYTHON_JSONL,
        paths=(jsonl_path,),
        train_probability=1.0,
        validation_probability=0.0,
        test_probability=0.0,
    )

    documents = list(iter_tinypython_jsonl_documents(dataset_configuration=dataset_configuration))

    assert len(documents) == 1
    assert documents[0].document_id.startswith("tinypython-")
    assert documents[0].split == "train"
    assert documents[0].text == (
        "Return the number of values.\n\n"
        "def count_values(values: list[int]) -> int:\n"
        "    return len(values)\n"
    )
