from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llm_lite.config.models import ExperimentFile
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.stage import PipelineStage, StageName


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ArtifactFingerprint(OrchestrationModel):
    value: str

    @classmethod
    def compute(
        cls,
        stage_name: StageName,
        configuration_hash: str,
        parent_fingerprints: tuple[StageArtifactReference, ...],
        contract_version: int,
    ) -> ArtifactFingerprint:
        fingerprint_value = hash_json_value(
            value={
                "stage_name": stage_name.value,
                "configuration_hash": configuration_hash,
                "parent_artifact_fingerprints": {
                    parent.stage_name.value: parent.fingerprint.value
                    for parent in parent_fingerprints
                },
                "contract_version": contract_version,
            },
        )
        return cls(value=fingerprint_value)


class StageArtifactReference(OrchestrationModel):
    stage_name: StageName
    fingerprint: ArtifactFingerprint


class PlannedArtifact(OrchestrationModel):
    stage_name: StageName
    configuration_hash: str
    contract_version: int
    parent_fingerprints: tuple[StageArtifactReference, ...]
    fingerprint: ArtifactFingerprint


class ArtifactStorePaths(OrchestrationModel):
    root_directory: Path

    @classmethod
    def for_run_directory(cls, run_directory: Path) -> ArtifactStorePaths:
        if run_directory.parent.name == "runs":
            return cls(root_directory=run_directory.parent.parent / "artifact_store")
        return cls(root_directory=run_directory.parent / "artifact_store")

    def stage_directory(self, stage_name: StageName) -> Path:
        return self.root_directory / stage_name.value

    def artifact_directory(self, stage_name: StageName, fingerprint: ArtifactFingerprint) -> Path:
        return self.stage_directory(stage_name=stage_name) / fingerprint.value

    def manifest_path(self, stage_name: StageName, fingerprint: ArtifactFingerprint) -> Path:
        return self.artifact_directory(stage_name=stage_name, fingerprint=fingerprint) / (
            "manifest.json"
        )

    def tensorboard_directory(
        self, stage_name: StageName, fingerprint: ArtifactFingerprint
    ) -> Path:
        return self.artifact_directory(stage_name=stage_name, fingerprint=fingerprint) / (
            "tensorboard"
        )


class RunManifest(OrchestrationModel):
    experiment: str
    artifacts: dict[str, str]


class ResolvedRun(OrchestrationModel):
    experiment_configuration: ExperimentFile
    run_directory: Path
    resolved_configuration_path: Path
    run_manifest_path: Path
    artifact_store_paths: ArtifactStorePaths
    artifacts: tuple[PlannedArtifact, ...]

    def artifact_for_stage(self, stage_name: StageName) -> PlannedArtifact:
        for artifact in self.artifacts:
            if artifact.stage_name is stage_name:
                return artifact
        raise ValueError(f"No planned artifact for stage {stage_name.value}.")

    def run_manifest(self, completed_stage_names: tuple[StageName, ...]) -> RunManifest:
        completed_artifacts = {
            artifact.stage_name.value: artifact.fingerprint.value
            for artifact in self.artifacts
            if artifact.stage_name in completed_stage_names
        }
        return RunManifest(
            experiment=self.experiment_configuration.experiment.name,
            artifacts=completed_artifacts,
        )


def resolve_run(
    experiment_configuration: ExperimentFile,
    stages: tuple[PipelineStage, ...],
) -> ResolvedRun:
    planned_artifacts: list[PlannedArtifact] = []
    for stage in stages:
        parent_fingerprints = _parent_fingerprints(
            planned_artifacts=tuple(planned_artifacts),
            stage=stage,
        )
        configuration_hash = stage.configuration_hash(
            experiment_configuration=experiment_configuration,
        )
        contract_version = _stage_contract_version(stage=stage)
        fingerprint = ArtifactFingerprint.compute(
            stage_name=stage.name,
            configuration_hash=configuration_hash,
            parent_fingerprints=parent_fingerprints,
            contract_version=contract_version,
        )
        planned_artifacts.append(
            PlannedArtifact(
                stage_name=stage.name,
                configuration_hash=configuration_hash,
                contract_version=contract_version,
                parent_fingerprints=parent_fingerprints,
                fingerprint=fingerprint,
            ),
        )
    run_directory = experiment_configuration.experiment.output_dir
    return ResolvedRun(
        experiment_configuration=experiment_configuration,
        run_directory=run_directory,
        resolved_configuration_path=run_directory / "resolved_config.json",
        run_manifest_path=run_directory / "run_manifest.json",
        artifact_store_paths=ArtifactStorePaths.for_run_directory(run_directory=run_directory),
        artifacts=tuple(planned_artifacts),
    )


def _parent_fingerprints(
    planned_artifacts: tuple[PlannedArtifact, ...],
    stage: PipelineStage,
) -> tuple[StageArtifactReference, ...]:
    parent_references: list[StageArtifactReference] = []
    for parent_stage_name in stage.parents:
        parent_artifact = _planned_artifact_for_stage(
            planned_artifacts=planned_artifacts,
            stage_name=parent_stage_name,
        )
        parent_references.append(
            StageArtifactReference(
                stage_name=parent_stage_name,
                fingerprint=parent_artifact.fingerprint,
            ),
        )
    return tuple(parent_references)


def _planned_artifact_for_stage(
    planned_artifacts: tuple[PlannedArtifact, ...],
    stage_name: StageName,
) -> PlannedArtifact:
    for artifact in planned_artifacts:
        if artifact.stage_name is stage_name:
            return artifact
    raise ValueError(f"Parent stage {stage_name.value} must be planned before dependents.")


def _stage_contract_version(stage: PipelineStage) -> int:
    return stage.contract_version
