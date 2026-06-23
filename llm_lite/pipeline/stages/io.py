from collections.abc import Iterator

from llm_lite.data.document import Document
from llm_lite.data.text_shards import (
    iter_text_shard_documents,
    iter_text_shard_texts,
    load_text_shard_corpus_manifest,
)
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName

TRAIN_SPLIT_NAME = "train"


def iter_raw_documents(registry: ArtifactRegistry) -> Iterator[Document]:
    artifact_directory = registry.artifact_directory(StageName.RAW_DATASET.value)
    yield from iter_text_shard_documents(artifact_directory=artifact_directory, split=None)


def iter_processed_document_texts(
    registry: ArtifactRegistry,
    split: str | None,
) -> Iterator[str]:
    artifact_directory = registry.artifact_directory(StageName.PROCESSED_DATASET.value)
    yield from iter_text_shard_texts(artifact_directory=artifact_directory, split=split)


def tokenizer_training_split(registry: ArtifactRegistry) -> str | None:
    return _preferred_training_split(
        registry=registry,
        stage_name=StageName.PROCESSED_DATASET,
    )


def packing_split(registry: ArtifactRegistry) -> str | None:
    return _preferred_training_split(
        registry=registry,
        stage_name=StageName.PROCESSED_DATASET,
    )


def _preferred_training_split(registry: ArtifactRegistry, stage_name: StageName) -> str | None:
    corpus_manifest = load_text_shard_corpus_manifest(
        artifact_directory=registry.artifact_directory(stage_name.value),
    )
    split_names = {split_manifest.split for split_manifest in corpus_manifest.splits}
    if TRAIN_SPLIT_NAME in split_names:
        return TRAIN_SPLIT_NAME
    return None
