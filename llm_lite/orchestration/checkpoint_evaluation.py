from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError

from llm_lite.config.models import DistributedStrategy
from llm_lite.orchestration.models import (
    ArtifactFingerprint,
    PlannedArtifact,
    ResolvedRun,
    StageArtifactReference,
)
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.stage import StageName
from llm_lite.pipeline.stages.evaluation import EvaluationStage
from llm_lite.training.checkpoint import (
    CheckpointEvent,
    CheckpointKind,
    CheckpointManifest,
    ShardedCheckpointManifest,
)

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class CheckpointEvaluationTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoint_manifest_path: Path
    checkpoint_path: Path
    checkpoint_step: int
    producing_stage_name: StageName
    producing_artifact_fingerprint: str


def checkpoint_evaluation_target_from_manifest(
    resolved_run: ResolvedRun,
    checkpoint_manifest_path: Path,
) -> CheckpointEvaluationTarget:
    manifest_text = checkpoint_manifest_path.read_text(encoding="utf-8")
    try:
        manifest = CheckpointManifest.model_validate_json(manifest_text)
    except ValidationError:
        sharded_manifest = ShardedCheckpointManifest.model_validate_json(manifest_text)
        producing_stage_name = _producing_stage_name(
            resolved_run=resolved_run,
            producing_artifact_fingerprint=sharded_manifest.producing_artifact_fingerprint,
        )
        return _ddp_sharded_checkpoint_evaluation_target(
            checkpoint_manifest_path=checkpoint_manifest_path,
            sharded_manifest=sharded_manifest,
            producing_stage_name=producing_stage_name,
        )
    producing_stage_name = _producing_stage_name(
        resolved_run=resolved_run,
        producing_artifact_fingerprint=manifest.producing_artifact_fingerprint,
    )
    checkpoint_path = checkpoint_manifest_path.parent / manifest.checkpoint_path
    return CheckpointEvaluationTarget(
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_path=checkpoint_path.resolve(),
        checkpoint_step=manifest.step,
        producing_stage_name=producing_stage_name,
        producing_artifact_fingerprint=manifest.producing_artifact_fingerprint,
    )


def checkpoint_evaluation_target_from_event(
    resolved_run: ResolvedRun,
    event_path: Path,
) -> CheckpointEvaluationTarget:
    event = CheckpointEvent.model_validate_json(event_path.read_text(encoding="utf-8"))
    producing_stage_name = _producing_stage_name(
        resolved_run=resolved_run,
        producing_artifact_fingerprint=event.producing_artifact_fingerprint,
    )
    producing_artifact = resolved_run.artifact_for_stage(stage_name=producing_stage_name)
    producing_artifact_directory = resolved_run.artifact_store_paths.artifact_directory(
        stage_name=producing_stage_name,
        fingerprint=producing_artifact.fingerprint,
    )
    checkpoint_manifest_path = producing_artifact_directory / event.checkpoint_manifest_path
    match event.checkpoint_kind:
        case CheckpointKind.FULL:
            return checkpoint_evaluation_target_from_manifest(
                resolved_run=resolved_run,
                checkpoint_manifest_path=checkpoint_manifest_path,
            )
        case CheckpointKind.SHARDED:
            return _ddp_sharded_checkpoint_evaluation_target(
                checkpoint_manifest_path=checkpoint_manifest_path,
                sharded_manifest=ShardedCheckpointManifest.model_validate_json(
                    checkpoint_manifest_path.read_text(encoding="utf-8"),
                ),
                producing_stage_name=producing_stage_name,
            )


def checkpoint_evaluation_artifact(
    resolved_run: ResolvedRun,
    target: CheckpointEvaluationTarget,
) -> PlannedArtifact:
    evaluation_stage = EvaluationStage()
    configuration_hash = hash_json_value(
        value={
            "evaluation": _checkpoint_evaluation_configuration_json(resolved_run=resolved_run),
            "inference": resolved_run.experiment_configuration.inference.model_dump(mode="json"),
            "checkpoint_step": target.checkpoint_step,
            "checkpoint_manifest_path": target.checkpoint_manifest_path.as_posix(),
            "producing_artifact_fingerprint": target.producing_artifact_fingerprint,
        },
    )
    parent_fingerprints = _checkpoint_evaluation_parents(
        resolved_run=resolved_run,
        producing_stage_name=target.producing_stage_name,
    )
    fingerprint = ArtifactFingerprint.compute(
        stage_name=StageName.EVALUATION,
        configuration_hash=configuration_hash,
        parent_fingerprints=parent_fingerprints,
        contract_version=evaluation_stage.contract_version,
    )
    return PlannedArtifact(
        stage_name=StageName.EVALUATION,
        configuration_hash=configuration_hash,
        contract_version=evaluation_stage.contract_version,
        parent_fingerprints=parent_fingerprints,
        fingerprint=fingerprint,
    )


def _checkpoint_evaluation_parents(
    resolved_run: ResolvedRun,
    producing_stage_name: StageName,
) -> tuple[StageArtifactReference, ...]:
    producing_artifact = resolved_run.artifact_for_stage(stage_name=producing_stage_name)
    tokenizer_artifact = resolved_run.artifact_for_stage(stage_name=StageName.TOKENIZER)
    return (
        StageArtifactReference(
            stage_name=producing_stage_name,
            fingerprint=producing_artifact.fingerprint,
        ),
        StageArtifactReference(
            stage_name=StageName.TOKENIZER,
            fingerprint=tokenizer_artifact.fingerprint,
        ),
    )


def _checkpoint_evaluation_configuration_json(resolved_run: ResolvedRun) -> dict[str, JsonValue]:
    training_evaluation_configuration = resolved_run.experiment_configuration.training.evaluation
    if training_evaluation_configuration is None:
        raise ValueError("Checkpoint evaluation requires training.evaluation.")
    return training_evaluation_configuration.evaluators.model_dump(mode="json")


def checkpoint_event_is_supported_for_evaluation(event_path: Path) -> bool:
    event = CheckpointEvent.model_validate_json(event_path.read_text(encoding="utf-8"))
    match event.checkpoint_kind:
        case CheckpointKind.FULL:
            return True
        case CheckpointKind.SHARDED:
            checkpoint_manifest_path = event_path.parent.parent / event.checkpoint_manifest_path
            return _sharded_checkpoint_event_uses_ddp_full_bridge(
                checkpoint_manifest_path=checkpoint_manifest_path,
            )


def _ddp_sharded_checkpoint_evaluation_target(
    checkpoint_manifest_path: Path,
    sharded_manifest: ShardedCheckpointManifest,
    producing_stage_name: StageName,
) -> CheckpointEvaluationTarget:
    if sharded_manifest.strategy != DistributedStrategy.DATA_PARALLEL.value:
        raise ValueError("Only DDP sharded checkpoints can be evaluated via a full bridge.")
    if sharded_manifest.rank_zero_full_checkpoint_path is None:
        raise ValueError("DDP sharded checkpoint manifest is missing rank-zero full checkpoint.")
    checkpoint_path = (
        checkpoint_manifest_path.parent / sharded_manifest.rank_zero_full_checkpoint_path
    )
    if not checkpoint_path.exists():
        raise ValueError("DDP sharded checkpoint rank-zero full checkpoint is missing.")
    return CheckpointEvaluationTarget(
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_path=checkpoint_path.resolve(),
        checkpoint_step=sharded_manifest.step,
        producing_stage_name=producing_stage_name,
        producing_artifact_fingerprint=sharded_manifest.producing_artifact_fingerprint,
    )


def _sharded_checkpoint_event_uses_ddp_full_bridge(checkpoint_manifest_path: Path) -> bool:
    if not checkpoint_manifest_path.exists():
        return False
    manifest = ShardedCheckpointManifest.model_validate_json(
        checkpoint_manifest_path.read_text(encoding="utf-8"),
    )
    return (
        manifest.strategy == DistributedStrategy.DATA_PARALLEL.value
        and manifest.rank_zero_full_checkpoint_path is not None
    )


def _producing_stage_name(
    resolved_run: ResolvedRun,
    producing_artifact_fingerprint: str,
) -> StageName:
    pretraining_artifact = resolved_run.artifact_for_stage(stage_name=StageName.PRETRAINING)
    if pretraining_artifact.fingerprint.value == producing_artifact_fingerprint:
        return StageName.PRETRAINING
    post_training_artifact = resolved_run.artifact_for_stage(stage_name=StageName.POST_TRAINING)
    if post_training_artifact.fingerprint.value == producing_artifact_fingerprint:
        return StageName.POST_TRAINING
    raise ValueError("Checkpoint event does not match a planned training artifact.")
