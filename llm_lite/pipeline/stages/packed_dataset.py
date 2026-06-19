from dataclasses import dataclass
from pathlib import Path

import torch

from llm_lite.config.models import ExperimentFile
from llm_lite.data.packing import pack_token_sequences
from llm_lite.pipeline.hashing import hash_model
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import compatible_skip_action
from llm_lite.pipeline.stages.io import read_tokenized_documents
from llm_lite.tokenizer.character import CharacterTokenizer


@dataclass(frozen=True)
class PackedDatasetStage:
    name: StageName = StageName.PACKED_DATASET
    parents: tuple[StageName, ...] = (StageName.TOKENIZED_DATASET,)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_model(model=experiment_configuration.packing)

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        tokenizer = CharacterTokenizer.load(
            directory=registry.artifact_directory(StageName.TOKENIZER.value),
        )
        if tokenizer.pad_token_id is None:
            raise ValueError("Packing requires a configured pad token.")
        tokenized_documents = read_tokenized_documents(registry=registry)
        sequences = pack_token_sequences(
            tokenized_documents=tokenized_documents,
            context_length=experiment_configuration.packing.context_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        torch.save(
            [sequence.token_ids for sequence in sequences], artifact_directory / "sequences.pt"
        )
        return StageOutput(
            files={"sequences": "sequences.pt"},
            metrics={"sequences": len(sequences)},
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        return compatible_skip_action(registry=registry)
