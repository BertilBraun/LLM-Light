import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import ExperimentFile
from llm_lite.orchestration.models import PlannedArtifact, ResolvedRun, resolve_run
from llm_lite.pipeline.logging import (
    PipelineEventLogger,
    PipelineEventRecord,
    PipelineEventType,
)
from llm_lite.pipeline.performance import (
    PipelinePerformanceLogger,
)
from llm_lite.pipeline.progress import console_log
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import PipelineStage, StageName, StageOutput
from llm_lite.pipeline.stages import ORDERED_PIPELINE_STAGES, ORDERED_STAGE_NAMES
from llm_lite.scripts.run_plan import run_plan
from llm_lite.utilities.random import seed_everything


@dataclass(frozen=True)
class StageReview:
    stage_name: StageName
    action: str


def run_pipeline(
    configuration_path: Path,
    dry_run: bool,
    force_stages: tuple[StageName, ...],
    from_stage: StageName | None = None,
    to_stage: StageName | None = None,
) -> int:
    experiment_configuration = load_experiment_configuration(configuration_path=configuration_path)
    seed_everything(seed=experiment_configuration.experiment.seed)
    selected_stages = _selected_stages(
        stages=ORDERED_PIPELINE_STAGES,
        from_stage=from_stage,
        to_stage=to_stage,
    )
    resolved_run = resolve_run(
        experiment_configuration=experiment_configuration,
        stages=ORDERED_PIPELINE_STAGES,
    )
    if not dry_run:
        _write_resolved_configuration(resolved_run=resolved_run)
    registry = ArtifactRegistry(run_directory=resolved_run.run_directory)
    distributed_rank = _environment_rank()
    if distributed_rank is not None and distributed_rank != 0:
        return _run_distributed_worker_stage(
            experiment_configuration=experiment_configuration,
            registry=registry,
            selected_stages=selected_stages,
            dry_run=dry_run,
            rank=distributed_rank,
        )
    force_stage_names = _expanded_force_stages(
        stages=selected_stages,
        force_stages=force_stages,
    )
    review = _review_pipeline(
        resolved_run=resolved_run,
        registry=registry,
        stages=selected_stages,
        force_stage_names=force_stage_names,
    )
    _print_review(review=review)
    if dry_run:
        return 0
    if force_stages:
        console_log(
            "--force is not supported by run_plan execution; recompute by removing artifacts.",
        )
        return 1
    return run_plan(
        configuration_paths=(configuration_path,),
        max_parallel_jobs=1,
        gpus=None,
        from_stage=from_stage,
        to_stage=to_stage,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", required=True, type=Path)
    argument_parser.add_argument("--dry-run", action="store_true")
    argument_parser.add_argument(
        "--from",
        dest="from_stage",
        choices=[stage_name.value for stage_name in ORDERED_STAGE_NAMES],
    )
    argument_parser.add_argument(
        "--to",
        dest="to_stage",
        choices=[stage_name.value for stage_name in ORDERED_STAGE_NAMES],
    )
    argument_parser.add_argument(
        "--force",
        action="append",
        choices=[stage_name.value for stage_name in ORDERED_STAGE_NAMES],
        const=StageName.RAW_DATASET.value,
        nargs="?",
    )
    return argument_parser


def main() -> int:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    force_values = tuple(arguments.force) if arguments.force is not None else ()
    force_stages = tuple(StageName(force_value) for force_value in force_values)
    return run_pipeline(
        configuration_path=arguments.config,
        dry_run=arguments.dry_run,
        force_stages=force_stages,
        from_stage=None if arguments.from_stage is None else StageName(arguments.from_stage),
        to_stage=None if arguments.to_stage is None else StageName(arguments.to_stage),
    )


def _selected_stages(
    stages: tuple[PipelineStage, ...],
    from_stage: StageName | None,
    to_stage: StageName | None,
) -> tuple[PipelineStage, ...]:
    ordered_stage_names = tuple(stage.name for stage in stages)
    start_index = 0 if from_stage is None else ordered_stage_names.index(from_stage)
    stop_index = len(stages) - 1 if to_stage is None else ordered_stage_names.index(to_stage)
    if start_index > stop_index:
        raise ValueError("--from stage must not come after --to stage.")
    return stages[start_index : stop_index + 1]


def _environment_rank() -> int | None:
    rank = os.environ.get("RANK")
    if rank is None:
        return None
    return int(rank)


def _run_distributed_worker_stage(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    selected_stages: tuple[PipelineStage, ...],
    dry_run: bool,
    rank: int,
) -> int:
    if dry_run:
        return 0
    if not experiment_configuration.distributed.enabled:
        raise ValueError("RANK is set, but distributed training is disabled.")
    pretraining_stage = next(
        (stage for stage in selected_stages if stage.name is StageName.PRETRAINING),
        None,
    )
    if pretraining_stage is None:
        return 0
    artifact_directory = registry.artifacts_directory / StageName.PRETRAINING.value
    artifact_directory.mkdir(parents=True, exist_ok=True)
    console_log(f"[rank {rank}] entering distributed pretraining worker")
    pretraining_stage.run(
        experiment_configuration=experiment_configuration,
        registry=registry,
        artifact_directory=artifact_directory,
    )
    return 0


def _review_pipeline(
    resolved_run: ResolvedRun,
    registry: ArtifactRegistry,
    stages: tuple[PipelineStage, ...],
    force_stage_names: set[StageName],
) -> list[StageReview]:
    review: list[StageReview] = []
    for stage in stages:
        planned_artifact = resolved_run.artifact_for_stage(stage_name=stage.name)
        parent_hashes = _parent_fingerprints(planned_artifact=planned_artifact)
        if stage.name in force_stage_names:
            review.append(StageReview(stage_name=stage.name, action="force recompute"))
        elif registry.is_compatible(
            artifact_type=stage.name.value,
            fingerprint=planned_artifact.fingerprint.value,
            configuration_hash=planned_artifact.configuration_hash,
            parent_hashes=parent_hashes,
            contract_version=planned_artifact.contract_version,
        ):
            continuation_action = stage.continuation_action(
                experiment_configuration=resolved_run.experiment_configuration,
                registry=registry,
            )
            if continuation_action is not None:
                review.append(
                    StageReview(
                        stage_name=stage.name,
                        action=continuation_action,
                    ),
                )
                continue
            review.append(
                StageReview(
                    stage_name=stage.name,
                    action=stage.compatible_action(registry=registry),
                ),
            )
        elif registry.has_matching_fingerprint(
            artifact_type=stage.name.value,
            fingerprint=planned_artifact.fingerprint.value,
            configuration_hash=planned_artifact.configuration_hash,
            parent_hashes=parent_hashes,
            contract_version=planned_artifact.contract_version,
        ):
            interrupted_action = stage.interrupted_action(
                experiment_configuration=resolved_run.experiment_configuration,
                registry=registry,
            )
            if interrupted_action is not None:
                review.append(
                    StageReview(
                        stage_name=stage.name,
                        action=interrupted_action,
                    ),
                )
            else:
                review.append(StageReview(stage_name=stage.name, action="execute"))
        else:
            review.append(StageReview(stage_name=stage.name, action="execute"))
    return review


def _execute_pipeline(
    resolved_run: ResolvedRun,
    registry: ArtifactRegistry,
    event_logger: PipelineEventLogger,
    performance_logger: PipelinePerformanceLogger,
    stages: tuple[PipelineStage, ...],
    force_stage_names: set[StageName],
) -> None:
    completed_stage_names: list[StageName] = []
    for stage in stages:
        planned_artifact = resolved_run.artifact_for_stage(stage_name=stage.name)
        parent_hashes = _parent_fingerprints(planned_artifact=planned_artifact)
        compatible = registry.is_compatible(
            artifact_type=stage.name.value,
            fingerprint=planned_artifact.fingerprint.value,
            configuration_hash=planned_artifact.configuration_hash,
            parent_hashes=parent_hashes,
            contract_version=planned_artifact.contract_version,
        )
        continuation_action = stage.continuation_action(
            experiment_configuration=resolved_run.experiment_configuration,
            registry=registry,
        )
        matching_fingerprint = registry.has_matching_fingerprint(
            artifact_type=stage.name.value,
            fingerprint=planned_artifact.fingerprint.value,
            configuration_hash=planned_artifact.configuration_hash,
            parent_hashes=parent_hashes,
            contract_version=planned_artifact.contract_version,
        )
        interrupted_action = (
            stage.interrupted_action(
                experiment_configuration=resolved_run.experiment_configuration,
                registry=registry,
            )
            if matching_fingerprint and not compatible
            else None
        )
        continue_compatible_stage = stage.name not in force_stage_names and (
            (compatible and continuation_action is not None) or interrupted_action is not None
        )
        if compatible and stage.name not in force_stage_names and not continue_compatible_stage:
            console_log(f"[skip] {stage.name.value}: compatible artifact found")
            _log_stage_event(
                event_logger=event_logger,
                event_type=PipelineEventType.STAGE_SKIP,
                stage_name=stage.name,
                message="compatible artifact found",
            )
            completed_stage_names.append(stage.name)
            _write_run_manifest(
                resolved_run=resolved_run,
                completed_stage_names=tuple(completed_stage_names),
            )
            continue
        artifact_directory = registry.artifacts_directory / stage.name.value
        if artifact_directory.exists() and (
            stage.name in force_stage_names or not continue_compatible_stage
        ):
            shutil.rmtree(artifact_directory)
        artifact_directory.mkdir(parents=True, exist_ok=True)
        registry.write_running_manifest(
            artifact_type=stage.name.value,
            fingerprint=planned_artifact.fingerprint.value,
            configuration_hash=planned_artifact.configuration_hash,
            parent_hashes=parent_hashes,
            contract_version=planned_artifact.contract_version,
        )
        _log_stage_event(
            event_logger=event_logger,
            event_type=PipelineEventType.STAGE_START,
            stage_name=stage.name,
            message="stage execution started",
        )
        console_log(f"[start] {stage.name.value}")
        with performance_logger.measure_stage(stage_name=stage.name.value) as performance_timer:
            stage_output = stage.run(
                experiment_configuration=resolved_run.experiment_configuration,
                registry=registry,
                artifact_directory=artifact_directory,
            )
        registry.write_complete_manifest(
            artifact_type=stage.name.value,
            fingerprint=planned_artifact.fingerprint.value,
            configuration_hash=planned_artifact.configuration_hash,
            parent_hashes=parent_hashes,
            contract_version=planned_artifact.contract_version,
            files=stage_output.files,
            metrics=stage_output.metrics,
        )
        completed_stage_names.append(stage.name)
        _write_run_manifest(
            resolved_run=resolved_run,
            completed_stage_names=tuple(completed_stage_names),
        )
        _print_stage_output(stage_name=stage.name, stage_output=stage_output)
        performance_logger.write_stage_timing(
            timing=performance_timer.timing(),
            metrics=stage_output.metrics,
        )
        _log_stage_event(
            event_logger=event_logger,
            event_type=PipelineEventType.STAGE_COMPLETE,
            stage_name=stage.name,
            message="stage execution completed",
        )


def _parent_fingerprints(planned_artifact: PlannedArtifact) -> dict[str, str]:
    return {
        parent.stage_name.value: parent.fingerprint.value
        for parent in planned_artifact.parent_fingerprints
    }


def _write_resolved_configuration(resolved_run: ResolvedRun) -> None:
    resolved_run.run_directory.mkdir(parents=True, exist_ok=True)
    resolved_run.resolved_configuration_path.write_text(
        resolved_run.experiment_configuration.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _write_run_manifest(
    resolved_run: ResolvedRun,
    completed_stage_names: tuple[StageName, ...],
) -> None:
    resolved_run.run_manifest_path.write_text(
        resolved_run.run_manifest(
            completed_stage_names=completed_stage_names,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )


def _expanded_force_stages(
    stages: tuple[PipelineStage, ...],
    force_stages: tuple[StageName, ...],
) -> set[StageName]:
    if not force_stages:
        return set()
    ordered_stage_names = tuple(stage.name for stage in stages)
    first_forced_index = min(ordered_stage_names.index(stage_name) for stage_name in force_stages)
    return set(ordered_stage_names[first_forced_index:])


def _print_review(review: list[StageReview]) -> None:
    console_log("Pipeline review:")
    for review_item in review:
        console_log(f"{review_item.stage_name.value:18} {review_item.action}")
    console_log("")


def _print_stage_output(stage_name: StageName, stage_output: StageOutput) -> None:
    console_log(f"[done]  {stage_name.value}")
    if stage_output.files:
        files = ", ".join(
            f"{file_name}={relative_path}"
            for file_name, relative_path in sorted(stage_output.files.items())
        )
        console_log(f"        files: {files}")
    if stage_output.metrics:
        metrics = ", ".join(
            f"{metric_name}={metric_value}"
            for metric_name, metric_value in sorted(stage_output.metrics.items())
        )
        console_log(f"        metrics: {metrics}")


def _log_review(review: list[StageReview], event_logger: PipelineEventLogger) -> None:
    for review_item in review:
        _log_stage_event(
            event_logger=event_logger,
            event_type=PipelineEventType.REVIEW,
            stage_name=review_item.stage_name,
            message=review_item.action,
        )


def _log_stage_event(
    event_logger: PipelineEventLogger,
    event_type: PipelineEventType,
    stage_name: StageName,
    message: str,
) -> None:
    event_logger.write(
        event_record=PipelineEventRecord(
            event_type=event_type,
            stage_name=stage_name,
            message=message,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
