from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.data.datasets import write_packed_sequence_stream
from llm_lite.data.packing import pack_token_sequences
from llm_lite.pipeline.hashing import hash_model
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import BasePipelineStage
from llm_lite.pipeline.stages.io import iter_processed_document_texts, packing_split
from llm_lite.tokenizer.loading import load_tokenizer


@dataclass(frozen=True)
class PackedDatasetStage(BasePipelineStage):
    name: StageName = StageName.PACKED_DATASET
    parents: tuple[StageName, ...] = (StageName.PROCESSED_DATASET, StageName.TOKENIZER)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_model(model=experiment_configuration.packing)

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        tokenizer = load_tokenizer(
            directory=registry.artifact_directory(StageName.TOKENIZER.value),
            tokenizer_configuration=experiment_configuration.tokenizer,
        )
        if tokenizer.pad_token_id is None:
            raise ValueError("Packing requires a configured pad token.")
        split = packing_split(registry=registry)
        tokenized_document_stream = (
            tokenizer.encode(
                text=document_text,
                add_bos=experiment_configuration.packing.add_bos,
                add_eos=experiment_configuration.packing.add_eos,
            )
            for document_text in iter_processed_document_texts(registry=registry, split=split)
        )
        sequences = pack_token_sequences(
            tokenized_document_stream=tokenized_document_stream,
            context_length=experiment_configuration.packing.context_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        index = write_packed_sequence_stream(
            sequences=sequences,
            artifact_directory=artifact_directory,
            row_length=experiment_configuration.packing.context_length + 1,
            maximum_shard_tokens=experiment_configuration.packing.maximum_shard_tokens,
        )
        return StageOutput(
            files={"index": "index.json", "shards": "shards"},
            metrics={
                "sequences": index.total_sequences,
                "total_tokens": index.total_tokens,
                "row_length": index.row_length,
                "shards": len(index.shards),
                "training_split": "all" if split is None else split,
            },
        )
