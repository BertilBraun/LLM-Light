from llm_lite.config.models import ExperimentFile
from llm_lite.pipeline.registry import ArtifactRegistry


def compatible_skip_action(registry: ArtifactRegistry) -> str:
    return "compatible, skip"


def no_continuation_action(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
) -> str | None:
    return None
