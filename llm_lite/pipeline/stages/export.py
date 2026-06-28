import shutil
from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import BasePipelineStage
from llm_lite.scripts.export_run_bundle import write_bundle

EXPORT_BUNDLE_FILENAME = "bundle.zip"
EXPORT_BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"


class ExportStage(BasePipelineStage):
    name: StageName = StageName.EXPORT
    parents: tuple[StageName, ...] = (StageName.EVALUATION,)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_json_value(
            value={
                "experiment": experiment_configuration.experiment.model_dump(mode="json"),
                "export": experiment_configuration.export.model_dump(mode="json"),
            },
        )

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        bundle_path = artifact_directory / EXPORT_BUNDLE_FILENAME
        bundle_manifest_path = artifact_directory / EXPORT_BUNDLE_MANIFEST_FILENAME
        bundle_manifest = write_bundle(
            run_directory=registry.run_directory,
            output_path=bundle_path,
            manifest_output_path=bundle_manifest_path,
            include_all_checkpoints=experiment_configuration.export.include_all_checkpoints,
            include_tensorboard=experiment_configuration.export.include_tensorboard,
        )
        configured_bundle_path = _configured_bundle_path(
            experiment_configuration=experiment_configuration,
        )
        configured_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_path, configured_bundle_path)
        return StageOutput(
            files={
                "bundle": EXPORT_BUNDLE_FILENAME,
                "bundle_manifest": EXPORT_BUNDLE_MANIFEST_FILENAME,
            },
            metrics={
                "bundle_file_count": bundle_manifest.file_count,
                "bundle_size_bytes": bundle_path.stat().st_size,
                "configured_bundle_path": str(configured_bundle_path),
            },
        )


def _configured_bundle_path(experiment_configuration: ExperimentFile) -> Path:
    configured_path = experiment_configuration.export.bundle_path
    if configured_path.is_absolute():
        return configured_path
    return experiment_configuration.experiment.output_dir / configured_path
