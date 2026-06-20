from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.pipeline.hashing import hash_model
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import compatible_skip_action
from llm_lite.pipeline.stages.io import iter_document_texts
from llm_lite.tokenizer.character import train_character_tokenizer


@dataclass(frozen=True)
class TokenizerStage:
    name: StageName = StageName.TOKENIZER
    parents: tuple[StageName, ...] = (StageName.RAW_DATASET,)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_model(model=experiment_configuration.tokenizer)

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        tokenizer = train_character_tokenizer(
            texts=iter_document_texts(registry=registry),
            add_bos_token=experiment_configuration.tokenizer.add_bos_token,
            add_eos_token=experiment_configuration.tokenizer.add_eos_token,
            add_pad_token=experiment_configuration.tokenizer.add_pad_token,
        )
        tokenizer.save(directory=artifact_directory)
        return StageOutput(
            files={"tokenizer": "tokenizer.json"},
            metrics={"vocabulary_size": tokenizer.vocabulary_size},
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        return compatible_skip_action(registry=registry)
