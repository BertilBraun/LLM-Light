from pathlib import Path

from llm_lite.data.document import Document
from llm_lite.data.text_shards import iter_text_shard_documents, write_text_shards


def test_write_and_read_split_text_shards(tmp_path: Path) -> None:
    corpus_manifest = write_text_shards(
        documents=iter(
            [
                Document(document_id="train-1", text="hello", split="train"),
                Document(document_id="validation-1", text="world", split="validation"),
            ],
        ),
        artifact_directory=tmp_path,
        shard_document_limit=1,
    )

    train_documents = list(iter_text_shard_documents(artifact_directory=tmp_path, split="train"))
    validation_documents = list(
        iter_text_shard_documents(artifact_directory=tmp_path, split="validation"),
    )

    assert (tmp_path / "corpus.json").exists()
    assert (tmp_path / "train" / "shard_000000.tar.gz").exists()
    assert (tmp_path / "validation" / "shard_000000.tar.gz").exists()
    assert corpus_manifest.splits[0].documents == 1
    assert train_documents == [Document(document_id="train-1", text="hello", split="train")]
    assert validation_documents == [
        Document(document_id="validation-1", text="world", split="validation"),
    ]
