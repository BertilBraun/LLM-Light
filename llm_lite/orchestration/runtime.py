from __future__ import annotations

import os
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llm_lite.config.models import ExperimentFile
from llm_lite.orchestration.models import PlannedArtifact, ResolvedRun, resolve_run
from llm_lite.pipeline.artifact import ArtifactManifest, ArtifactStatus
from llm_lite.pipeline.registry import ArtifactDirectory, ArtifactRegistry
from llm_lite.pipeline.stage import PipelineStage, StageName, StageOutput
from llm_lite.pipeline.stages import ORDERED_PIPELINE_STAGES


class LockRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    fingerprint: str
    process_id: int
    hostname: str
    command: str
    started_at: str
    heartbeat_at: str


@dataclass(frozen=True)
class StageExecutionResult:
    stage_name: StageName
    fingerprint: str
    stage_output: StageOutput


def load_resolved_configuration(resolved_configuration_path: Path) -> ExperimentFile:
    return ExperimentFile.model_validate_json(
        resolved_configuration_path.read_text(encoding="utf-8"),
    )


def load_resolved_run(resolved_configuration_path: Path) -> ResolvedRun:
    return resolve_run(
        experiment_configuration=load_resolved_configuration(
            resolved_configuration_path=resolved_configuration_path,
        ),
        stages=ORDERED_PIPELINE_STAGES,
    )


def write_resolved_configuration(resolved_run: ResolvedRun) -> None:
    resolved_run.run_directory.mkdir(parents=True, exist_ok=True)
    resolved_run.resolved_configuration_path.write_text(
        resolved_run.experiment_configuration.model_dump_json(indent=2),
        encoding="utf-8",
    )


def write_run_manifest(
    resolved_run: ResolvedRun,
    completed_stage_names: tuple[StageName, ...],
) -> None:
    resolved_run.run_manifest_path.write_text(
        resolved_run.run_manifest(
            completed_stage_names=completed_stage_names,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )


def artifact_registry_for_resolved_run(resolved_run: ResolvedRun) -> ArtifactRegistry:
    return ArtifactRegistry(
        run_directory=resolved_run.run_directory,
        artifact_directories=tuple(
            ArtifactDirectory(
                artifact_type=artifact.stage_name.value,
                directory=resolved_run.artifact_store_paths.artifact_directory(
                    stage_name=artifact.stage_name,
                    fingerprint=artifact.fingerprint,
                ),
            )
            for artifact in resolved_run.artifacts
        ),
    )


def stage_by_name(stage_name: StageName) -> PipelineStage:
    for stage in ORDERED_PIPELINE_STAGES:
        if stage.name is stage_name:
            return stage
    raise ValueError(f"Unknown stage {stage_name.value}.")


def parent_fingerprints(planned_artifact: PlannedArtifact) -> dict[str, str]:
    return {
        parent.stage_name.value: parent.fingerprint.value
        for parent in planned_artifact.parent_fingerprints
    }


def complete_manifest_matches(
    manifest: ArtifactManifest | None,
    planned_artifact: PlannedArtifact,
) -> bool:
    if manifest is None:
        return False
    if manifest.status is not ArtifactStatus.COMPLETE:
        return False
    if manifest.fingerprint != planned_artifact.fingerprint.value:
        return False
    if manifest.configuration_hash != planned_artifact.configuration_hash:
        return False
    if manifest.contract_version != planned_artifact.contract_version:
        return False
    return manifest.parents == parent_fingerprints(planned_artifact=planned_artifact)


def run_stage_job(
    resolved_run: ResolvedRun,
    stage_name: StageName,
    expected_fingerprint: str,
) -> StageExecutionResult:
    planned_artifact = resolved_run.artifact_for_stage(stage_name=stage_name)
    if planned_artifact.fingerprint.value != expected_fingerprint:
        raise ValueError(
            f"Expected fingerprint {expected_fingerprint} does not match planned "
            f"fingerprint {planned_artifact.fingerprint.value} for {stage_name.value}.",
        )
    registry = artifact_registry_for_resolved_run(resolved_run=resolved_run)
    stage = stage_by_name(stage_name=stage_name)
    artifact_directory = registry.artifact_directory(artifact_type=stage_name.value)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    registry.write_running_manifest(
        artifact_type=stage_name.value,
        fingerprint=planned_artifact.fingerprint.value,
        configuration_hash=planned_artifact.configuration_hash,
        parent_hashes=parent_fingerprints(planned_artifact=planned_artifact),
        contract_version=planned_artifact.contract_version,
    )
    try:
        stage_output = stage.run(
            experiment_configuration=resolved_run.experiment_configuration,
            registry=registry,
            artifact_directory=artifact_directory,
        )
    except Exception:
        registry.write_failed_manifest(
            artifact_type=stage_name.value,
            fingerprint=planned_artifact.fingerprint.value,
            configuration_hash=planned_artifact.configuration_hash,
            parent_hashes=parent_fingerprints(planned_artifact=planned_artifact),
            contract_version=planned_artifact.contract_version,
        )
        raise
    registry.write_complete_manifest(
        artifact_type=stage_name.value,
        fingerprint=planned_artifact.fingerprint.value,
        configuration_hash=planned_artifact.configuration_hash,
        parent_hashes=parent_fingerprints(planned_artifact=planned_artifact),
        contract_version=planned_artifact.contract_version,
        files=stage_output.files,
        metrics=stage_output.metrics,
    )
    return StageExecutionResult(
        stage_name=stage_name,
        fingerprint=planned_artifact.fingerprint.value,
        stage_output=stage_output,
    )


def copy_stage_tensorboard_to_run_view(resolved_run: ResolvedRun, stage_name: StageName) -> None:
    planned_artifact = resolved_run.artifact_for_stage(stage_name=stage_name)
    source_directory = resolved_run.artifact_store_paths.tensorboard_directory(
        stage_name=stage_name,
        fingerprint=planned_artifact.fingerprint,
    )
    if not source_directory.exists():
        return
    destination_directory = resolved_run.run_directory / "tensorboard" / stage_name.value
    if destination_directory.exists():
        shutil.rmtree(destination_directory)
    shutil.copytree(source_directory, destination_directory)


def acquire_artifact_lock(
    artifact_directory: Path,
    fingerprint: str,
    command: str,
) -> bool:
    lock_directory = artifact_directory / ".lock"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    try:
        lock_directory.mkdir()
    except FileExistsError:
        if lock_is_stale(lock_directory=lock_directory):
            shutil.rmtree(lock_directory)
            lock_directory.mkdir()
        else:
            return False
    write_lock_heartbeat(
        lock_directory=lock_directory,
        fingerprint=fingerprint,
        command=command,
    )
    return True


def release_artifact_lock(artifact_directory: Path) -> None:
    lock_directory = artifact_directory / ".lock"
    if lock_directory.exists():
        shutil.rmtree(lock_directory)


def write_lock_heartbeat(lock_directory: Path, fingerprint: str, command: str) -> None:
    timestamp = _utc_now()
    lock_record = LockRecord(
        fingerprint=fingerprint,
        process_id=os.getpid(),
        hostname=socket.gethostname(),
        command=command,
        started_at=timestamp,
        heartbeat_at=timestamp,
    )
    (lock_directory / "lock.json").write_text(
        lock_record.model_dump_json(indent=2),
        encoding="utf-8",
    )


def lock_is_stale(lock_directory: Path) -> bool:
    lock_path = lock_directory / "lock.json"
    if not lock_path.exists():
        return True
    lock_record = LockRecord.model_validate_json(lock_path.read_text(encoding="utf-8"))
    if lock_record.hostname != socket.gethostname():
        return False
    return not _process_is_running(process_id=lock_record.process_id)


def _process_is_running(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
