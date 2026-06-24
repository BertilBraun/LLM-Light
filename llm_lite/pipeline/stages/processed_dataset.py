from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.data.preprocessing import preprocess_text_shards
from llm_lite.data.text_shards import TextShardCorpusManifest
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import BasePipelineStage


class ProcessedDatasetStage(BasePipelineStage):
    name: StageName = StageName.PROCESSED_DATASET
    parents: tuple[StageName, ...] = (StageName.RAW_DATASET,)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_json_value(
            value={
                "preprocessing": experiment_configuration.preprocessing.model_dump(mode="json"),
                "processed_dataset_format_version": 2,
            },
        )

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        preprocessing_result = preprocess_text_shards(
            input_artifact_directory=registry.artifact_directory(StageName.RAW_DATASET.value),
            output_artifact_directory=artifact_directory,
            transforms=experiment_configuration.preprocessing.transforms,
            output_shard_documents=experiment_configuration.preprocessing.output_shard_documents,
            workers=experiment_configuration.preprocessing.workers,
        )
        counters = preprocessing_result.counters
        metrics = {
            "raw_documents": counters.input_documents,
            "processed_documents": counters.output_documents,
            "rejected_documents": counters.rejected_documents,
            "total_characters": counters.output_characters,
            "total_bytes": counters.output_bytes,
            "input_characters": counters.input_characters,
            "input_bytes": counters.input_bytes,
            "unicode_normalized_documents": counters.unicode_normalized_documents,
            "line_endings_normalized_documents": counters.line_endings_normalized_documents,
            "lower_cased_documents": counters.lower_cased_documents,
            "deduplicated_documents": counters.deduplicated_documents,
            "split_assigned_documents": counters.split_assigned_documents,
            "python_extracted_functions": counters.python_extracted_functions,
            "python_parse_failed_documents": counters.python_parse_failed_documents,
            "shards": _shard_count(corpus_manifest=preprocessing_result.corpus_manifest),
            "workers": preprocessing_result.worker_count,
        }
        for split_manifest in preprocessing_result.corpus_manifest.splits:
            metrics[f"split_{split_manifest.split}_documents"] = split_manifest.documents
            metrics[f"split_{split_manifest.split}_bytes"] = split_manifest.bytes
        return StageOutput(
            files={"corpus": "corpus.json"},
            metrics=metrics,
        )


def _shard_count(corpus_manifest: TextShardCorpusManifest) -> int:
    return sum(split.shards for split in corpus_manifest.splits)
