from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from llm_lite.config.models import ExperimentFile
from llm_lite.pipeline.registry import ArtifactRegistry


class StageName(str, Enum):
    RAW_DATASET = "raw_dataset"
    TOKENIZER = "tokenizer"
    TOKENIZED_DATASET = "tokenized_dataset"
    PACKED_DATASET = "packed_dataset"
    PRETRAINING = "pretraining"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class StageOutput:
    files: dict[str, str]
    metrics: dict[str, int | float | str | bool]


class PipelineStage(Protocol):
    name: StageName
    parents: tuple[StageName, ...]

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str: ...

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput: ...

    def compatible_action(self, registry: ArtifactRegistry) -> str: ...
