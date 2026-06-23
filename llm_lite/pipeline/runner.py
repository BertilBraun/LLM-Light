import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import ExperimentFile
from llm_lite.pipeline.logging import (
    PipelineEventLogger,
    PipelineEventRecord,
    PipelineEventType,
)
from llm_lite.pipeline.performance import (
    PipelinePerformanceLogger,
)
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import PipelineStage, StageName, StageOutput
from llm_lite.pipeline.stages import ORDERED_PIPELINE_STAGES, ORDERED_STAGE_NAMES
from llm_lite.utilities.random import seed_everything


@dataclass(frozen=True)
class StageReview:
    stage_name: StageName
    action: str


def run_pipeline(
    configuration_path: Path, dry_run: bool, force_stages: tuple[StageName, ...]
) -> int:
    experiment_configuration = load_experiment_configuration(configuration_path=configuration_path)
    seed_everything(seed=experiment_configuration.experiment.seed)
    registry = ArtifactRegistry(run_directory=experiment_configuration.experiment.output_dir)
    event_logger = PipelineEventLogger(run_directory=experiment_configuration.experiment.output_dir)
    performance_logger = PipelinePerformanceLogger(
        run_directory=experiment_configuration.experiment.output_dir,
    )
    force_stage_names = _expanded_force_stages(
        stages=ORDERED_PIPELINE_STAGES,
        force_stages=force_stages,
    )
    review = _review_pipeline(
        experiment_configuration=experiment_configuration,
        registry=registry,
        stages=ORDERED_PIPELINE_STAGES,
        force_stage_names=force_stage_names,
    )
    _print_review(review=review)
    _log_review(review=review, event_logger=event_logger)
    if dry_run:
        return 0
    experiment_configuration.experiment.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_configuration_path = (
        experiment_configuration.experiment.output_dir / "resolved_config.json"
    )
    resolved_configuration_path.write_text(
        experiment_configuration.model_dump_json(indent=2),
        encoding="utf-8",
    )
    try:
        _execute_pipeline(
            experiment_configuration=experiment_configuration,
            registry=registry,
            event_logger=event_logger,
            performance_logger=performance_logger,
            stages=ORDERED_PIPELINE_STAGES,
            force_stage_names=force_stage_names,
        )
    finally:
        performance_logger.close()
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", required=True, type=Path)
    argument_parser.add_argument("--dry-run", action="store_true")
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
    )


def _review_pipeline(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    stages: tuple[PipelineStage, ...],
    force_stage_names: set[StageName],
) -> list[StageReview]:
    review: list[StageReview] = []
    for stage in stages:
        configuration_hash = stage.configuration_hash(
            experiment_configuration=experiment_configuration,
        )
        parent_hashes = _parent_hashes(registry=registry, stage=stage)
        if stage.name in force_stage_names:
            review.append(StageReview(stage_name=stage.name, action="force recompute"))
        elif registry.is_compatible(
            artifact_type=stage.name.value,
            configuration_hash=configuration_hash,
            parent_hashes=parent_hashes,
        ):
            continuation_action = stage.continuation_action(
                experiment_configuration=experiment_configuration,
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
        else:
            review.append(StageReview(stage_name=stage.name, action="execute"))
    return review


def _execute_pipeline(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    event_logger: PipelineEventLogger,
    performance_logger: PipelinePerformanceLogger,
    stages: tuple[PipelineStage, ...],
    force_stage_names: set[StageName],
) -> None:
    for stage in stages:
        configuration_hash = stage.configuration_hash(
            experiment_configuration=experiment_configuration,
        )
        parent_hashes = _parent_hashes(registry=registry, stage=stage)
        compatible = registry.is_compatible(
            artifact_type=stage.name.value,
            configuration_hash=configuration_hash,
            parent_hashes=parent_hashes,
        )
        continuation_action = stage.continuation_action(
            experiment_configuration=experiment_configuration,
            registry=registry,
        )
        continue_compatible_stage = (
            compatible and stage.name not in force_stage_names and continuation_action is not None
        )
        if compatible and stage.name not in force_stage_names and not continue_compatible_stage:
            print(f"[skip] {stage.name.value}: compatible artifact found", flush=True)
            _log_stage_event(
                event_logger=event_logger,
                event_type=PipelineEventType.STAGE_SKIP,
                stage_name=stage.name,
                message="compatible artifact found",
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
            configuration_hash=configuration_hash,
            parent_hashes=parent_hashes,
        )
        _log_stage_event(
            event_logger=event_logger,
            event_type=PipelineEventType.STAGE_START,
            stage_name=stage.name,
            message="stage execution started",
        )
        print(f"[start] {stage.name.value}", flush=True)
        with performance_logger.measure_stage(stage_name=stage.name.value) as performance_timer:
            stage_output = stage.run(
                experiment_configuration=experiment_configuration,
                registry=registry,
                artifact_directory=artifact_directory,
            )
        registry.write_complete_manifest(
            artifact_type=stage.name.value,
            configuration_hash=configuration_hash,
            parent_hashes=parent_hashes,
            files=stage_output.files,
            metrics=stage_output.metrics,
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


def _parent_hashes(registry: ArtifactRegistry, stage: PipelineStage) -> dict[str, str]:
    parent_hashes: dict[str, str] = {}
    for parent_stage_name in stage.parents:
        manifest = registry.read_manifest(artifact_type=parent_stage_name.value)
        if manifest is not None:
            parent_hashes[parent_stage_name.value] = registry.artifact_identifier(
                artifact_type=parent_stage_name.value,
            )
    return parent_hashes


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
    print("Pipeline review:", flush=True)
    for review_item in review:
        print(f"{review_item.stage_name.value:18} {review_item.action}", flush=True)
    print(flush=True)


def _print_stage_output(stage_name: StageName, stage_output: StageOutput) -> None:
    print(f"[done]  {stage_name.value}", flush=True)
    if stage_output.files:
        files = ", ".join(
            f"{file_name}={relative_path}"
            for file_name, relative_path in sorted(stage_output.files.items())
        )
        print(f"        files: {files}", flush=True)
    if stage_output.metrics:
        metrics = ", ".join(
            f"{metric_name}={metric_value}"
            for metric_name, metric_value in sorted(stage_output.metrics.items())
        )
        print(f"        metrics: {metrics}", flush=True)


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
