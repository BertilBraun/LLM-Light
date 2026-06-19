from dataclasses import dataclass
from pathlib import Path

import torch

from llm_lite.config.models import ExperimentFile
from llm_lite.data.datasets import PackedSequence, PackedSequenceDataset
from llm_lite.model.gpt import DenseGpt
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.tokenizer.character import CharacterTokenizer
from llm_lite.training.checkpoint import latest_checkpoint
from llm_lite.training.trainer import train_model


@dataclass(frozen=True)
class PretrainingStage:
    name: StageName = StageName.PRETRAINING
    parents: tuple[StageName, ...] = (StageName.PACKED_DATASET, StageName.TOKENIZER)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_json_value(
            value={
                "logging_schema_version": 1,
                "model": experiment_configuration.model.model_dump(mode="json"),
                "training": experiment_configuration.training.model_dump(mode="json"),
            },
        )

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        tokenizer = CharacterTokenizer.load(
            directory=registry.artifact_directory(StageName.TOKENIZER.value),
        )
        sequences = torch.load(
            registry.artifact_directory(StageName.PACKED_DATASET.value) / "sequences.pt",
            weights_only=False,
        )
        dataset = PackedSequenceDataset(
            sequences=[PackedSequence(token_ids=tuple(token_ids)) for token_ids in sequences],
        )
        model = DenseGpt(
            model_configuration=experiment_configuration.model,
            vocabulary_size=tokenizer.vocabulary_size,
        )
        result = train_model(
            model=model,
            dataset=dataset,
            training_configuration=experiment_configuration.training,
            artifact_directory=artifact_directory,
        )
        return StageOutput(
            files={
                "checkpoint": str(result.checkpoint_path.relative_to(artifact_directory)),
                "metrics": "metrics.jsonl",
                "tensorboard": "tensorboard",
            },
            metrics={
                "final_step": result.final_step,
                "final_loss": result.final_loss,
                "resumed_from_step": result.resumed_from_step,
            },
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        checkpoint_state = latest_checkpoint(
            checkpoint_directory=registry.artifact_directory(StageName.PRETRAINING.value)
            / "checkpoints",
        )
        if checkpoint_state is not None:
            return f"complete at step {checkpoint_state.step}, skip"
        return "compatible, skip"
