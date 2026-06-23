from pathlib import Path

from llm_lite.config.models import DatasetType, LocalTextDatasetConfiguration
from llm_lite.data.sources import iter_local_text_documents, resolve_local_text_paths


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


def test_iter_local_text_documents_records_stable_metadata() -> None:
    dataset_configuration = LocalTextDatasetConfiguration(
        type=DatasetType.LOCAL_TEXT,
        paths=(Path("tests/fixtures/local_text/hello_world.txt"),),
        glob_patterns=(),
    )

    documents = list(iter_local_text_documents(dataset_configuration=dataset_configuration))

    assert len(documents) == 1
    assert documents[0].document_id.startswith("local-text-")
    assert documents[0].text == "hello world\n"
    assert documents[0].metadata["source"] == "local_text"
    assert documents[0].metadata["path"].endswith("/tests/fixtures/local_text/hello_world.txt")
    assert documents[0].metadata["byte_size"] == 12
    assert str(documents[0].metadata["content_hash"]).startswith("sha256:")
