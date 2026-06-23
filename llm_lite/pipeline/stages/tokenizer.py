from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.pipeline.hashing import hash_model
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import BasePipelineStage
from llm_lite.pipeline.stages.io import iter_processed_document_texts, tokenizer_training_split
from llm_lite.tokenizer.loading import train_tokenizer


class TokenizerStage(BasePipelineStage):
    name: StageName = StageName.TOKENIZER
    parents: tuple[StageName, ...] = (StageName.PROCESSED_DATASET,)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_model(model=experiment_configuration.tokenizer)

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        trained_tokenizer = train_tokenizer(
            texts=iter_processed_document_texts(
                registry=registry,
                split=tokenizer_training_split(registry=registry),
            ),
            tokenizer_configuration=experiment_configuration.tokenizer,
        )
        trained_tokenizer.tokenizer.save(directory=artifact_directory)
        return StageOutput(
            files={"tokenizer": "tokenizer.json"},
            metrics=trained_tokenizer.metrics,
        )
