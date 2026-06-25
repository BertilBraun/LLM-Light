from llm_lite.config.models import ExperimentFile
from llm_lite.pipeline.registry import ArtifactRegistry


class BasePipelineStage:
    def compatible_action(self, registry: ArtifactRegistry) -> str:
        return "compatible, skip"

    def continuation_action(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
    ) -> str | None:
        return None

    def interrupted_action(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
    ) -> str | None:
        return None
