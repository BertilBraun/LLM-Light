from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.data.sources import iter_dataset_documents
from llm_lite.data.text_shards import TextShardCorpusManifest, write_text_shards
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import compatible_skip_action

RAW_SHARD_DOCUMENT_LIMIT = 10000


@dataclass(frozen=True)
class RawDatasetStage:
    name: StageName = StageName.RAW_DATASET
    parents: tuple[StageName, ...] = ()

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_json_value(
            value={
                "dataset": experiment_configuration.dataset.model_dump(mode="json"),
                "raw_dataset_format_version": 2,
            },
        )

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        corpus_manifest = write_text_shards(
            documents=iter_dataset_documents(
                experiment_configuration=experiment_configuration,
            ),
            artifact_directory=artifact_directory,
            shard_document_limit=RAW_SHARD_DOCUMENT_LIMIT,
        )
        return StageOutput(
            files={"corpus": "corpus.json"},
            metrics=_raw_metrics(corpus_manifest=corpus_manifest),
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        return compatible_skip_action(registry=registry)


def _raw_metrics(corpus_manifest: TextShardCorpusManifest) -> dict[str, int]:
    document_count = sum(split.documents for split in corpus_manifest.splits)
    total_characters = sum(split.characters for split in corpus_manifest.splits)
    total_bytes = sum(split.bytes for split in corpus_manifest.splits)
    shard_count = sum(split.shards for split in corpus_manifest.splits)
    return {
        "raw_documents": document_count,
        "processed_documents": document_count,
        "rejected_documents": 0,
        "total_characters": total_characters,
        "total_bytes": total_bytes,
        "shards": shard_count,
    }
