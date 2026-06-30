import argparse
import concurrent.futures
import glob
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition, Event, Thread
from typing import TextIO

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.orchestration.checkpoint_evaluation import (
    checkpoint_evaluation_artifact,
    checkpoint_evaluation_target_from_event,
    checkpoint_event_is_supported_for_evaluation,
)
from llm_lite.orchestration.logging import (
    OrchestrationEventLogger,
    OrchestrationEventRecord,
    OrchestrationEventType,
    utc_timestamp,
)
from llm_lite.orchestration.models import PlannedArtifact, ResolvedRun, resolve_run
from llm_lite.orchestration.runtime import (
    acquire_artifact_lock,
    artifact_registry_for_resolved_run,
    complete_manifest_matches,
    copy_stage_tensorboard_to_run_view,
    parent_fingerprints,
    release_artifact_lock,
    write_resolved_configuration,
    write_run_manifest,
)
from llm_lite.pipeline.artifact import ArtifactManifest, ArtifactStatus
from llm_lite.pipeline.logging import (
    PipelineEventLogger,
    PipelineEventRecord,
    PipelineEventType,
)
from llm_lite.pipeline.progress import console_log
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName
from llm_lite.pipeline.stages import ORDERED_PIPELINE_STAGES, ORDERED_STAGE_NAMES
from llm_lite.training.checkpoint import latest_checkpoint

LOCK_WAIT_SECONDS = 5.0
CHECKPOINT_EVENT_POLL_SECONDS = 1.0
JOB_LOG_NAME = "job.log"
ARCHIVED_JOB_LOG_DIRECTORY_NAME = ".job_logs"
ORCHESTRATION_LOG_NAME = "orchestration.jsonl"


class RunPlanCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class RunFailure:
    run_name: str
    error: Exception


class RunPlanFailed(RuntimeError):
    def __init__(self, failures: tuple[RunFailure, ...]) -> None:
        self.failures = failures
        failed_run_names = ", ".join(failure.run_name for failure in failures)
        super().__init__(
            f"Run plan completed with {len(failures)} failed run(s): {failed_run_names}"
        )


@dataclass(frozen=True)
class GpuAllocation:
    visible_devices: tuple[str, ...]

    def environment_value(self) -> str | None:
        if not self.visible_devices:
            return None
        return ",".join(self.visible_devices)

    def first_device(self) -> "GpuAllocation":
        if not self.visible_devices:
            return self
        return GpuAllocation(visible_devices=(self.visible_devices[0],))


class GpuPool:
    def __init__(self, visible_devices: tuple[str, ...]) -> None:
        self.visible_devices = visible_devices
        self._available_devices = list(visible_devices)
        self._condition = Condition()

    @classmethod
    def from_argument(cls, gpus: str | None) -> "GpuPool":
        if gpus is None or gpus == "":
            return cls(visible_devices=())
        return cls(visible_devices=tuple(gpu.strip() for gpu in gpus.split(",") if gpu.strip()))

    def acquire(
        self,
        gpu_count: int,
        cancellation_event: Event | None = None,
    ) -> GpuAllocation:
        if cancellation_event is not None and cancellation_event.is_set():
            raise RunPlanCancelled("Run plan cancelled after another job failed.")
        if gpu_count == 0:
            return GpuAllocation(visible_devices=())
        if not self.visible_devices:
            return GpuAllocation(visible_devices=())
        if gpu_count > len(self.visible_devices):
            raise ValueError(
                f"Stage requires {gpu_count} GPU(s), but only "
                f"{len(self.visible_devices)} device(s) were provided by --gpus.",
            )
        with self._condition:
            while len(self._available_devices) < gpu_count:
                if cancellation_event is not None and cancellation_event.is_set():
                    raise RunPlanCancelled("Run plan cancelled after another job failed.")
                self._condition.wait(timeout=1.0)
            if cancellation_event is not None and cancellation_event.is_set():
                raise RunPlanCancelled("Run plan cancelled after another job failed.")
            allocated_devices = tuple(self._available_devices[:gpu_count])
            del self._available_devices[:gpu_count]
            return GpuAllocation(visible_devices=allocated_devices)

    def release(self, gpu_allocation: GpuAllocation) -> None:
        if not gpu_allocation.visible_devices:
            return
        with self._condition:
            for visible_device in gpu_allocation.visible_devices:
                assert visible_device in self.visible_devices
                assert visible_device not in self._available_devices
            self._available_devices.extend(gpu_allocation.visible_devices)
            self._available_devices.sort(key=self.visible_devices.index)
            self._condition.notify_all()

    def notify_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()


@dataclass(frozen=True)
class ResourceRequest:
    cpu_workers: int
    gpu_count: int
    exclusive_gpus: bool


@dataclass(frozen=True)
class AsyncEvaluationSubmission:
    planned_artifact: PlannedArtifact
    event_path: Path
    checkpoint_manifest_path: Path


@dataclass(frozen=True)
class OrchestrationStageContext:
    resolved_run: ResolvedRun
    planned_artifact: PlannedArtifact
    event_logger: OrchestrationEventLogger


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", required=True, nargs="+")
    argument_parser.add_argument("--max-parallel-jobs", type=int, default=1)
    argument_parser.add_argument("--fail-fast", action="store_true")
    argument_parser.add_argument("--gpus")
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
    return argument_parser


def main() -> int:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    return run_plan(
        configuration_paths=_expand_configuration_paths(configurations=tuple(arguments.config)),
        max_parallel_jobs=arguments.max_parallel_jobs,
        gpus=arguments.gpus,
        from_stage=None if arguments.from_stage is None else StageName(arguments.from_stage),
        to_stage=None if arguments.to_stage is None else StageName(arguments.to_stage),
        fail_fast=arguments.fail_fast,
    )


def run_plan(
    configuration_paths: tuple[Path, ...],
    max_parallel_jobs: int,
    gpus: str | None,
    from_stage: StageName | None = None,
    to_stage: StageName | None = None,
    fail_fast: bool = False,
) -> int:
    if max_parallel_jobs < 1:
        raise ValueError("--max-parallel-jobs must be at least 1.")
    resolved_runs = tuple(
        resolve_run(
            experiment_configuration=load_experiment_configuration(
                configuration_path=configuration_path,
            ),
            stages=ORDERED_PIPELINE_STAGES,
        )
        for configuration_path in configuration_paths
    )
    orchestration_logger = OrchestrationEventLogger(
        events_path=_orchestration_log_path(resolved_runs=resolved_runs),
    )
    console_log(f"[orchestration] writing {orchestration_logger.events_path}")
    for resolved_run in resolved_runs:
        _log_run_event(
            event_logger=orchestration_logger,
            event_type=OrchestrationEventType.RUN_QUEUED,
            resolved_run=resolved_run,
            message="run queued for execution",
        )
    gpu_pool = GpuPool.from_argument(gpus=gpus)
    selected_stage_names = _selected_stage_names(
        from_stage=from_stage,
        to_stage=to_stage,
    )
    completed_run_count = 0
    failures: list[RunFailure] = []
    cancelled_run_count = 0
    try:
        if max_parallel_jobs == 1 or len(resolved_runs) == 1:
            cancellation_event = Event()
            for resolved_run in resolved_runs:
                try:
                    _execute_resolved_run(
                        resolved_run=resolved_run,
                        gpu_pool=gpu_pool,
                        selected_stage_names=selected_stage_names,
                        max_parallel_jobs=max_parallel_jobs,
                        cancellation_event=cancellation_event,
                        orchestration_logger=orchestration_logger,
                    )
                    completed_run_count += 1
                except RunPlanCancelled as error:
                    cancelled_run_count += 1
                    _log_run_event(
                        event_logger=orchestration_logger,
                        event_type=OrchestrationEventType.RUN_CANCELLED,
                        resolved_run=resolved_run,
                        message=str(error),
                    )
                    raise
                except Exception as error:
                    failures.append(_run_failure(resolved_run=resolved_run, error=error))
                    _log_run_failure(
                        event_logger=orchestration_logger,
                        resolved_run=resolved_run,
                        error=error,
                    )
                    if fail_fast:
                        cancellation_event.set()
                        gpu_pool.notify_waiters()
                        raise
            if failures:
                raise RunPlanFailed(failures=tuple(failures))
            _log_plan_summary(
                event_logger=orchestration_logger,
                event_type=OrchestrationEventType.PLAN_COMPLETED,
                resolved_runs=resolved_runs,
                message="run plan completed",
                completed_run_count=completed_run_count,
                failed_run_count=len(failures),
                cancelled_run_count=cancelled_run_count,
            )
            return 0
        cancellation_event = Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
            future_runs = {
                executor.submit(
                    _execute_resolved_run,
                    resolved_run=resolved_run,
                    gpu_pool=gpu_pool,
                    selected_stage_names=selected_stage_names,
                    max_parallel_jobs=max_parallel_jobs,
                    cancellation_event=cancellation_event,
                    orchestration_logger=orchestration_logger,
                ): resolved_run
                for resolved_run in resolved_runs
            }
            cancelled_without_error: RunPlanCancelled | None = None
            for future in concurrent.futures.as_completed(tuple(future_runs)):
                resolved_run = future_runs[future]
                try:
                    future.result()
                    completed_run_count += 1
                except concurrent.futures.CancelledError:
                    cancelled_run_count += 1
                    _log_run_event(
                        event_logger=orchestration_logger,
                        event_type=OrchestrationEventType.RUN_CANCELLED,
                        resolved_run=resolved_run,
                        message="run future cancelled before execution",
                    )
                except RunPlanCancelled as error:
                    cancelled_run_count += 1
                    _log_run_event(
                        event_logger=orchestration_logger,
                        event_type=OrchestrationEventType.RUN_CANCELLED,
                        resolved_run=resolved_run,
                        message=str(error),
                    )
                    if cancelled_without_error is None:
                        cancelled_without_error = error
                except Exception as error:
                    failures.append(_run_failure(resolved_run=resolved_run, error=error))
                    _log_run_failure(
                        event_logger=orchestration_logger,
                        resolved_run=resolved_run,
                        error=error,
                    )
                    if fail_fast:
                        cancellation_event.set()
                        gpu_pool.notify_waiters()
                        for pending_future in future_runs:
                            if not pending_future.done():
                                pending_future.cancel()
        if failures:
            if fail_fast:
                raise failures[0].error
            raise RunPlanFailed(failures=tuple(failures))
        if cancelled_without_error is not None:
            raise cancelled_without_error
        _log_plan_summary(
            event_logger=orchestration_logger,
            event_type=OrchestrationEventType.PLAN_COMPLETED,
            resolved_runs=resolved_runs,
            message="run plan completed",
            completed_run_count=completed_run_count,
            failed_run_count=len(failures),
            cancelled_run_count=cancelled_run_count,
        )
        return 0
    except Exception as error:
        _log_plan_summary(
            event_logger=orchestration_logger,
            event_type=OrchestrationEventType.PLAN_FAILED,
            resolved_runs=resolved_runs,
            message=f"{error.__class__.__name__}: {error}",
            completed_run_count=completed_run_count,
            failed_run_count=len(failures),
            cancelled_run_count=cancelled_run_count,
        )
        raise


def _run_failure(resolved_run: ResolvedRun, error: Exception) -> RunFailure:
    run_name = resolved_run.experiment_configuration.experiment.name
    console_log(f"[failed] {run_name}: {error.__class__.__name__}: {error}")
    return RunFailure(
        run_name=run_name,
        error=error,
    )


def _expand_configuration_paths(configurations: tuple[str, ...]) -> tuple[Path, ...]:
    configuration_paths: list[Path] = []
    for configuration in configurations:
        matches = tuple(Path(match) for match in glob.glob(configuration))
        if matches:
            configuration_paths.extend(sorted(matches))
        else:
            configuration_paths.append(Path(configuration))
    if not configuration_paths:
        raise ValueError("At least one configuration path is required.")
    return tuple(configuration_paths)


def _selected_stage_names(
    from_stage: StageName | None,
    to_stage: StageName | None,
) -> tuple[StageName, ...]:
    start_index = 0 if from_stage is None else ORDERED_STAGE_NAMES.index(from_stage)
    stop_index = (
        len(ORDERED_STAGE_NAMES) - 1 if to_stage is None else ORDERED_STAGE_NAMES.index(to_stage)
    )
    if start_index > stop_index:
        raise ValueError("--from stage must not come after --to stage.")
    return ORDERED_STAGE_NAMES[start_index : stop_index + 1]


def _execute_resolved_run(
    resolved_run: ResolvedRun,
    gpu_pool: GpuPool,
    selected_stage_names: tuple[StageName, ...],
    max_parallel_jobs: int,
    cancellation_event: Event,
    orchestration_logger: OrchestrationEventLogger,
) -> None:
    write_resolved_configuration(resolved_run=resolved_run)
    registry = artifact_registry_for_resolved_run(resolved_run=resolved_run)
    pipeline_event_logger = PipelineEventLogger(run_directory=resolved_run.run_directory)
    completed_stage_names = list(
        _complete_stage_names_from_artifact_store(
            resolved_run=resolved_run,
            registry=registry,
        ),
    )
    write_run_manifest(
        resolved_run=resolved_run,
        completed_stage_names=tuple(completed_stage_names),
    )
    for planned_artifact in resolved_run.artifacts:
        if cancellation_event.is_set():
            raise RunPlanCancelled("Run plan cancelled after another job failed.")
        if planned_artifact.stage_name not in selected_stage_names:
            continue
        manifest = registry.read_manifest(artifact_type=planned_artifact.stage_name.value)
        if complete_manifest_matches(manifest=manifest, planned_artifact=planned_artifact):
            console_log(f"[cache] {planned_artifact.stage_name.value}: compatible artifact found")
            _log_stage_event(
                event_logger=pipeline_event_logger,
                event_type=PipelineEventType.STAGE_SKIP,
                stage_name=planned_artifact.stage_name,
                message="compatible artifact found",
            )
            _log_orchestration_stage_event(
                event_logger=orchestration_logger,
                event_type=OrchestrationEventType.STAGE_CACHE_HIT,
                resolved_run=resolved_run,
                planned_artifact=planned_artifact,
                message="compatible artifact found",
            )
            _append_missing_stage_name(
                completed_stage_names=completed_stage_names,
                stage_name=planned_artifact.stage_name,
            )
            copy_stage_tensorboard_to_run_view(
                resolved_run=resolved_run,
                stage_name=planned_artifact.stage_name,
            )
            write_run_manifest(
                resolved_run=resolved_run,
                completed_stage_names=tuple(completed_stage_names),
            )
            continue
        _run_missing_artifact(
            resolved_run=resolved_run,
            planned_artifact=planned_artifact,
            gpu_pool=gpu_pool,
            pipeline_event_logger=pipeline_event_logger,
            orchestration_logger=orchestration_logger,
            max_parallel_jobs=max_parallel_jobs,
            cancellation_event=cancellation_event,
        )
        _append_missing_stage_name(
            completed_stage_names=completed_stage_names,
            stage_name=planned_artifact.stage_name,
        )
        copy_stage_tensorboard_to_run_view(
            resolved_run=resolved_run,
            stage_name=planned_artifact.stage_name,
        )
        write_run_manifest(
            resolved_run=resolved_run,
            completed_stage_names=tuple(completed_stage_names),
        )
    _log_run_event(
        event_logger=orchestration_logger,
        event_type=OrchestrationEventType.RUN_COMPLETED,
        resolved_run=resolved_run,
        message="run completed",
    )


def _complete_stage_names_from_artifact_store(
    resolved_run: ResolvedRun,
    registry: ArtifactRegistry,
) -> tuple[StageName, ...]:
    completed_stage_names: list[StageName] = []
    for planned_artifact in resolved_run.artifacts:
        manifest = registry.read_manifest(artifact_type=planned_artifact.stage_name.value)
        if complete_manifest_matches(manifest=manifest, planned_artifact=planned_artifact):
            completed_stage_names.append(planned_artifact.stage_name)
    return tuple(completed_stage_names)


def _append_missing_stage_name(
    completed_stage_names: list[StageName],
    stage_name: StageName,
) -> None:
    if stage_name not in completed_stage_names:
        completed_stage_names.append(stage_name)


def _run_missing_artifact(
    resolved_run: ResolvedRun,
    planned_artifact: PlannedArtifact,
    gpu_pool: GpuPool,
    pipeline_event_logger: PipelineEventLogger,
    orchestration_logger: OrchestrationEventLogger,
    max_parallel_jobs: int,
    cancellation_event: Event,
) -> None:
    registry = artifact_registry_for_resolved_run(resolved_run=resolved_run)
    artifact_directory = registry.artifact_directory(
        artifact_type=planned_artifact.stage_name.value,
    )
    command = _job_command(
        resolved_run=resolved_run,
        planned_artifact=planned_artifact,
    )
    while not acquire_artifact_lock(
        artifact_directory=artifact_directory,
        fingerprint=planned_artifact.fingerprint.value,
        command=" ".join(command),
    ):
        _log_orchestration_stage_event(
            event_logger=orchestration_logger,
            event_type=OrchestrationEventType.STAGE_WAITING_FOR_ARTIFACT_LOCK,
            resolved_run=resolved_run,
            planned_artifact=planned_artifact,
            message=f"waiting for artifact lock at {artifact_directory / '.lock'}",
        )
        if cancellation_event.is_set():
            raise RunPlanCancelled("Run plan cancelled after another job failed.")
        time.sleep(LOCK_WAIT_SECONDS)
        manifest = registry.read_manifest(artifact_type=planned_artifact.stage_name.value)
        if complete_manifest_matches(manifest=manifest, planned_artifact=planned_artifact):
            _log_orchestration_stage_event(
                event_logger=orchestration_logger,
                event_type=OrchestrationEventType.STAGE_CACHE_HIT,
                resolved_run=resolved_run,
                planned_artifact=planned_artifact,
                message="compatible artifact found after waiting for artifact lock",
            )
            return
    try:
        stage_label = _stage_label(
            resolved_run=resolved_run,
            planned_artifact=planned_artifact,
        )
        manifest = registry.read_manifest(artifact_type=planned_artifact.stage_name.value)
        can_reuse_incomplete_payload = _can_reuse_incomplete_payload(
            artifact_directory=artifact_directory,
            manifest=manifest,
            planned_artifact=planned_artifact,
        )
        try:
            resource_request = _resource_request(
                resolved_run=resolved_run,
                planned_artifact=planned_artifact,
            )
            if resource_request.gpu_count > 0:
                _log_orchestration_stage_event(
                    event_logger=orchestration_logger,
                    event_type=OrchestrationEventType.STAGE_WAITING_FOR_GPU_ALLOCATION,
                    resolved_run=resolved_run,
                    planned_artifact=planned_artifact,
                    message=f"waiting for {resource_request.gpu_count} GPU allocation",
                )
            gpu_allocation = gpu_pool.acquire(
                gpu_count=resource_request.gpu_count,
                cancellation_event=cancellation_event,
            )
            environment = _job_environment(
                gpu_allocation=gpu_allocation,
            )
            try:
                if not can_reuse_incomplete_payload:
                    _clear_incomplete_artifact_payload(
                        artifact_directory=artifact_directory,
                        artifact_store_root=resolved_run.artifact_store_paths.root_directory,
                    )
                _log_stage_event(
                    event_logger=pipeline_event_logger,
                    event_type=PipelineEventType.STAGE_START,
                    stage_name=planned_artifact.stage_name,
                    message="stage job started",
                )
                _log_orchestration_stage_event(
                    event_logger=orchestration_logger,
                    event_type=OrchestrationEventType.STAGE_STARTED,
                    resolved_run=resolved_run,
                    planned_artifact=planned_artifact,
                    message="stage job started",
                )
                console_log(f"[start] {stage_label}")
                if _supports_async_checkpoint_evaluation(
                    resolved_run=resolved_run,
                    planned_artifact=planned_artifact,
                    max_parallel_jobs=max_parallel_jobs,
                ):
                    _run_training_job_with_async_evaluations(
                        resolved_run=resolved_run,
                        planned_artifact=planned_artifact,
                        command=command,
                        artifact_directory=artifact_directory,
                        environment=environment,
                        gpu_allocation=gpu_allocation,
                        orchestration_logger=orchestration_logger,
                        max_parallel_jobs=max_parallel_jobs,
                        cancellation_event=cancellation_event,
                    )
                else:
                    _run_subprocess_job(
                        command=command,
                        artifact_directory=artifact_directory,
                        environment=environment,
                        cancellation_event=cancellation_event,
                        stage_context=OrchestrationStageContext(
                            resolved_run=resolved_run,
                            planned_artifact=planned_artifact,
                            event_logger=orchestration_logger,
                        ),
                    )
            finally:
                gpu_pool.release(gpu_allocation=gpu_allocation)
            completed_manifest = registry.read_manifest(
                artifact_type=planned_artifact.stage_name.value,
            )
            if not complete_manifest_matches(
                manifest=completed_manifest,
                planned_artifact=planned_artifact,
            ):
                raise ValueError(
                    f"Stage {planned_artifact.stage_name.value} exited without a complete "
                    "manifest.",
                )
        except RunPlanCancelled:
            console_log(f"[cancel] {stage_label}")
            raise
        except Exception as error:
            _log_stage_event(
                event_logger=pipeline_event_logger,
                event_type=PipelineEventType.STAGE_FAILURE,
                stage_name=planned_artifact.stage_name,
                message=_failure_message(error=error, artifact_directory=artifact_directory),
            )
            _log_orchestration_stage_event(
                event_logger=orchestration_logger,
                event_type=OrchestrationEventType.STAGE_FAILED,
                resolved_run=resolved_run,
                planned_artifact=planned_artifact,
                message=_failure_message(error=error, artifact_directory=artifact_directory),
            )
            console_log(
                f"[fail]  {stage_label}; see {artifact_directory / JOB_LOG_NAME}",
            )
            raise
        _log_stage_event(
            event_logger=pipeline_event_logger,
            event_type=PipelineEventType.STAGE_COMPLETE,
            stage_name=planned_artifact.stage_name,
            message="stage job completed",
        )
        _log_orchestration_stage_event(
            event_logger=orchestration_logger,
            event_type=OrchestrationEventType.STAGE_COMPLETED,
            resolved_run=resolved_run,
            planned_artifact=planned_artifact,
            message="stage job completed",
        )
        console_log(f"[done]  {stage_label}")
    finally:
        release_artifact_lock(artifact_directory=artifact_directory)


def _job_command(resolved_run: ResolvedRun, planned_artifact: PlannedArtifact) -> list[str]:
    if (
        planned_artifact.stage_name is StageName.PRETRAINING
        and resolved_run.experiment_configuration.distributed.enabled
    ):
        return [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={resolved_run.experiment_configuration.distributed.world_size}",
            "-m",
            "llm_lite.scripts.run_job",
            "--resolved-config",
            str(resolved_run.resolved_configuration_path),
            "--stage",
            planned_artifact.stage_name.value,
            "--fingerprint",
            planned_artifact.fingerprint.value,
        ]
    return [
        sys.executable,
        "-m",
        "llm_lite.scripts.run_job",
        "--resolved-config",
        str(resolved_run.resolved_configuration_path),
        "--stage",
        planned_artifact.stage_name.value,
        "--fingerprint",
        planned_artifact.fingerprint.value,
    ]


def _stage_label(resolved_run: ResolvedRun, planned_artifact: PlannedArtifact) -> str:
    return (
        f"{resolved_run.experiment_configuration.experiment.name}/"
        f"{planned_artifact.stage_name.value}"
    )


def _job_environment(gpu_allocation: GpuAllocation) -> dict[str, str]:
    environment = os.environ.copy()
    gpu_environment = gpu_allocation.environment_value()
    if gpu_environment is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_environment
    return environment


def _resource_request(
    resolved_run: ResolvedRun,
    planned_artifact: PlannedArtifact,
) -> ResourceRequest:
    match planned_artifact.stage_name:
        case StageName.PRETRAINING:
            return ResourceRequest(
                cpu_workers=resolved_run.experiment_configuration.training.dataloader.num_workers,
                gpu_count=(
                    resolved_run.experiment_configuration.distributed.world_size
                    if resolved_run.experiment_configuration.distributed.enabled
                    else 1
                ),
                exclusive_gpus=True,
            )
        case StageName.POST_TRAINING:
            return ResourceRequest(cpu_workers=1, gpu_count=1, exclusive_gpus=True)
        case StageName.EVALUATION:
            return ResourceRequest(cpu_workers=1, gpu_count=1, exclusive_gpus=False)
        case _:
            return ResourceRequest(cpu_workers=1, gpu_count=0, exclusive_gpus=False)


def _run_subprocess_job(
    command: list[str],
    artifact_directory: Path,
    environment: dict[str, str],
    cancellation_event: Event,
    stage_context: OrchestrationStageContext,
) -> None:
    log_path = artifact_directory / JOB_LOG_NAME
    with log_path.open("a", encoding="utf-8") as log_file:
        _log_orchestration_stage_event(
            event_logger=stage_context.event_logger,
            event_type=OrchestrationEventType.SUBPROCESS_COMMAND_STARTED,
            resolved_run=stage_context.resolved_run,
            planned_artifact=stage_context.planned_artifact,
            message="subprocess command started",
            command=tuple(command),
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        if process.stdout is None:
            raise ValueError("Subprocess stdout was not captured.")
        output_thread = Thread(
            target=_stream_process_output,
            kwargs={"process": process, "log_file": log_file},
            daemon=True,
        )
        output_thread.start()
        while process.poll() is None:
            if cancellation_event.is_set():
                _terminate_process(process=process)
                output_thread.join()
                raise RunPlanCancelled("Run plan cancelled after another job failed.")
            time.sleep(CHECKPOINT_EVENT_POLL_SECONDS)
        output_thread.join()
        exit_code = process.wait()
    if exit_code != 0:
        raise subprocess.CalledProcessError(returncode=exit_code, cmd=command)


def _supports_async_checkpoint_evaluation(
    resolved_run: ResolvedRun,
    planned_artifact: PlannedArtifact,
    max_parallel_jobs: int,
) -> bool:
    return (
        planned_artifact.stage_name is StageName.PRETRAINING
        and resolved_run.experiment_configuration.training.evaluation is not None
        and max_parallel_jobs > 1
    )


def _run_training_job_with_async_evaluations(
    resolved_run: ResolvedRun,
    planned_artifact: PlannedArtifact,
    command: list[str],
    artifact_directory: Path,
    environment: dict[str, str],
    gpu_allocation: GpuAllocation,
    orchestration_logger: OrchestrationEventLogger,
    max_parallel_jobs: int,
    cancellation_event: Event,
) -> None:
    seen_event_paths: set[Path] = set()
    evaluation_futures: list[concurrent.futures.Future[None]] = []
    log_path = artifact_directory / JOB_LOG_NAME
    with (
        log_path.open("a", encoding="utf-8") as log_file,
        concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_jobs - 1) as executor,
    ):
        _log_orchestration_stage_event(
            event_logger=orchestration_logger,
            event_type=OrchestrationEventType.SUBPROCESS_COMMAND_STARTED,
            resolved_run=resolved_run,
            planned_artifact=planned_artifact,
            message="subprocess command started",
            command=tuple(command),
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        if process.stdout is None:
            raise ValueError("Subprocess stdout was not captured.")
        try:
            output_thread = Thread(
                target=_stream_process_output,
                kwargs={"process": process, "log_file": log_file},
                daemon=True,
            )
            output_thread.start()
            while process.poll() is None:
                if cancellation_event.is_set():
                    _terminate_process(process=process)
                    output_thread.join()
                    _cancel_evaluation_futures(evaluation_futures=evaluation_futures)
                    raise RunPlanCancelled("Run plan cancelled after another job failed.")
                _submit_ready_checkpoint_evaluations(
                    resolved_run=resolved_run,
                    training_artifact_directory=artifact_directory,
                    seen_event_paths=seen_event_paths,
                    evaluation_futures=evaluation_futures,
                    executor=executor,
                    gpu_allocation=gpu_allocation,
                    orchestration_logger=orchestration_logger,
                    cancellation_event=cancellation_event,
                )
                _raise_completed_evaluation_failures(evaluation_futures=evaluation_futures)
                time.sleep(CHECKPOINT_EVENT_POLL_SECONDS)
            output_thread.join()
            if cancellation_event.is_set():
                _cancel_evaluation_futures(evaluation_futures=evaluation_futures)
                raise RunPlanCancelled("Run plan cancelled after another job failed.")
            _submit_ready_checkpoint_evaluations(
                resolved_run=resolved_run,
                training_artifact_directory=artifact_directory,
                seen_event_paths=seen_event_paths,
                evaluation_futures=evaluation_futures,
                executor=executor,
                gpu_allocation=gpu_allocation,
                orchestration_logger=orchestration_logger,
                cancellation_event=cancellation_event,
            )
            exit_code = process.wait()
            _wait_for_async_evaluations(evaluation_futures=evaluation_futures)
        except Exception:
            _terminate_process(process=process)
            output_thread.join()
            raise
    if exit_code != 0:
        raise subprocess.CalledProcessError(returncode=exit_code, cmd=command)


def _stream_process_output(process: subprocess.Popen[str], log_file: TextIO) -> None:
    if process.stdout is None:
        raise ValueError("Subprocess stdout was not captured.")
    for line in process.stdout:
        print(line, end="")
        log_file.write(line)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _submit_ready_checkpoint_evaluations(
    resolved_run: ResolvedRun,
    training_artifact_directory: Path,
    seen_event_paths: set[Path],
    evaluation_futures: list[concurrent.futures.Future[None]],
    executor: concurrent.futures.ThreadPoolExecutor,
    gpu_allocation: GpuAllocation,
    orchestration_logger: OrchestrationEventLogger,
    cancellation_event: Event,
) -> None:
    if cancellation_event.is_set():
        return
    for submission in _checkpoint_evaluation_submissions(
        resolved_run=resolved_run,
        training_artifact_directory=training_artifact_directory,
        seen_event_paths=seen_event_paths,
    ):
        seen_event_paths.add(submission.event_path)
        evaluation_futures.append(
            executor.submit(
                _run_checkpoint_evaluation_job,
                resolved_run,
                submission,
                gpu_allocation.first_device(),
                orchestration_logger,
                cancellation_event,
            ),
        )


def _checkpoint_evaluation_submissions(
    resolved_run: ResolvedRun,
    training_artifact_directory: Path,
    seen_event_paths: set[Path],
) -> tuple[AsyncEvaluationSubmission, ...]:
    submissions: list[AsyncEvaluationSubmission] = []
    events_directory = training_artifact_directory / "events"
    if not events_directory.exists():
        return ()
    training_evaluation_configuration = resolved_run.experiment_configuration.training.evaluation
    if training_evaluation_configuration is None:
        return ()
    for event_path in sorted(events_directory.glob("checkpoint_*.json")):
        if event_path in seen_event_paths:
            continue
        try:
            if not checkpoint_event_is_supported_for_evaluation(event_path=event_path):
                seen_event_paths.add(event_path)
                continue
            target = checkpoint_evaluation_target_from_event(
                resolved_run=resolved_run,
                event_path=event_path,
            )
        except FileNotFoundError:
            seen_event_paths.add(event_path)
            continue
        if target.checkpoint_step % training_evaluation_configuration.interval_steps != 0:
            continue
        planned_artifact = checkpoint_evaluation_artifact(
            resolved_run=resolved_run,
            target=target,
        )
        submissions.append(
            AsyncEvaluationSubmission(
                planned_artifact=planned_artifact,
                event_path=event_path,
                checkpoint_manifest_path=target.checkpoint_manifest_path,
            ),
        )
    return tuple(submissions)


def _run_checkpoint_evaluation_job(
    resolved_run: ResolvedRun,
    submission: AsyncEvaluationSubmission,
    gpu_allocation: GpuAllocation,
    orchestration_logger: OrchestrationEventLogger,
    cancellation_event: Event,
) -> None:
    if cancellation_event.is_set():
        raise RunPlanCancelled("Run plan cancelled after another job failed.")
    registry = artifact_registry_for_resolved_run(
        resolved_run=resolved_run,
        override_artifact=submission.planned_artifact,
    )
    manifest = registry.read_manifest(artifact_type=StageName.EVALUATION.value)
    if complete_manifest_matches(
        manifest=manifest,
        planned_artifact=submission.planned_artifact,
    ):
        return
    if not submission.checkpoint_manifest_path.exists():
        _log_pruned_checkpoint_evaluation(submission=submission)
        return
    artifact_directory = registry.artifact_directory(artifact_type=StageName.EVALUATION.value)
    command = _checkpoint_evaluation_command(
        resolved_run=resolved_run,
        submission=submission,
    )
    while not acquire_artifact_lock(
        artifact_directory=artifact_directory,
        fingerprint=submission.planned_artifact.fingerprint.value,
        command=" ".join(command),
    ):
        if cancellation_event.is_set():
            raise RunPlanCancelled("Run plan cancelled after another job failed.")
        time.sleep(LOCK_WAIT_SECONDS)
        manifest = registry.read_manifest(artifact_type=StageName.EVALUATION.value)
        if complete_manifest_matches(
            manifest=manifest,
            planned_artifact=submission.planned_artifact,
        ):
            return
        if not submission.checkpoint_manifest_path.exists():
            _log_pruned_checkpoint_evaluation(submission=submission)
            return
    try:
        if not submission.checkpoint_manifest_path.exists():
            _log_pruned_checkpoint_evaluation(submission=submission)
            return
        _clear_incomplete_artifact_payload(
            artifact_directory=artifact_directory,
            artifact_store_root=resolved_run.artifact_store_paths.root_directory,
        )
        console_log(
            f"[start] evaluation checkpoint={submission.checkpoint_manifest_path.parent.name}",
        )
        _run_subprocess_job(
            command=command,
            artifact_directory=artifact_directory,
            environment=_job_environment(gpu_allocation=gpu_allocation),
            cancellation_event=cancellation_event,
            stage_context=OrchestrationStageContext(
                resolved_run=resolved_run,
                planned_artifact=submission.planned_artifact,
                event_logger=orchestration_logger,
            ),
        )
        completed_manifest = registry.read_manifest(artifact_type=StageName.EVALUATION.value)
        if not complete_manifest_matches(
            manifest=completed_manifest,
            planned_artifact=submission.planned_artifact,
        ):
            raise ValueError("Checkpoint evaluation exited without a complete manifest.")
        copy_stage_tensorboard_to_run_view(
            resolved_run=resolved_run,
            stage_name=StageName.EVALUATION,
            planned_artifact_override=submission.planned_artifact,
        )
        console_log(
            f"[done]  evaluation checkpoint={submission.checkpoint_manifest_path.parent.name}",
        )
    except subprocess.CalledProcessError:
        if not submission.checkpoint_manifest_path.exists():
            _log_pruned_checkpoint_evaluation(submission=submission)
            return
        raise
    finally:
        release_artifact_lock(artifact_directory=artifact_directory)


def _log_pruned_checkpoint_evaluation(submission: AsyncEvaluationSubmission) -> None:
    console_log(
        f"[skip] evaluation checkpoint={submission.checkpoint_manifest_path.parent.name} "
        "was pruned before evaluation",
    )


def _checkpoint_evaluation_command(
    resolved_run: ResolvedRun,
    submission: AsyncEvaluationSubmission,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_lite.scripts.run_job",
        "--resolved-config",
        str(resolved_run.resolved_configuration_path),
        "--stage",
        StageName.EVALUATION.value,
        "--fingerprint",
        submission.planned_artifact.fingerprint.value,
        "--checkpoint-manifest",
        str(submission.checkpoint_manifest_path),
    ]


def _raise_completed_evaluation_failures(
    evaluation_futures: list[concurrent.futures.Future[None]],
) -> None:
    for future in tuple(evaluation_futures):
        if future.done():
            future.result()
            evaluation_futures.remove(future)


def _wait_for_async_evaluations(
    evaluation_futures: list[concurrent.futures.Future[None]],
) -> None:
    for future in concurrent.futures.as_completed(tuple(evaluation_futures)):
        future.result()


def _cancel_evaluation_futures(
    evaluation_futures: list[concurrent.futures.Future[None]],
) -> None:
    for future in evaluation_futures:
        future.cancel()


def _failure_message(error: Exception, artifact_directory: Path) -> str:
    log_path = artifact_directory / JOB_LOG_NAME
    return f"{error.__class__.__name__}: {error}; log={log_path}"


def _clear_incomplete_artifact_payload(artifact_directory: Path, artifact_store_root: Path) -> None:
    resolved_artifact_directory = artifact_directory.resolve()
    resolved_store_root = artifact_store_root.resolve()
    if resolved_store_root not in resolved_artifact_directory.parents:
        raise ValueError(f"Refusing to clear artifact outside store: {artifact_directory}")
    _archive_existing_job_log(artifact_directory=artifact_directory)
    for child_path in artifact_directory.iterdir():
        if child_path.name in (".lock", ARCHIVED_JOB_LOG_DIRECTORY_NAME):
            continue
        if child_path.is_dir():
            shutil.rmtree(child_path)
        else:
            child_path.unlink()


def _archive_existing_job_log(artifact_directory: Path) -> Path | None:
    log_path = artifact_directory / JOB_LOG_NAME
    if not log_path.exists():
        return None
    archive_directory = artifact_directory / ARCHIVED_JOB_LOG_DIRECTORY_NAME
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive_path = _available_archived_job_log_path(archive_directory=archive_directory)
    shutil.copy2(log_path, archive_path)
    return archive_path


def _available_archived_job_log_path(archive_directory: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_directory / f"job_{timestamp}.log"
    duplicate_index = 1
    while archive_path.exists():
        archive_path = archive_directory / f"job_{timestamp}_{duplicate_index}.log"
        duplicate_index += 1
    return archive_path


def _can_reuse_incomplete_payload(
    artifact_directory: Path,
    manifest: ArtifactManifest | None,
    planned_artifact: PlannedArtifact,
) -> bool:
    if manifest is None:
        return False
    if manifest.status is ArtifactStatus.COMPLETE:
        return False
    if manifest.fingerprint != planned_artifact.fingerprint.value:
        return False
    if manifest.configuration_hash != planned_artifact.configuration_hash:
        return False
    if manifest.contract_version != planned_artifact.contract_version:
        return False
    if manifest.parents != parent_fingerprints(planned_artifact=planned_artifact):
        return False
    if planned_artifact.stage_name is not StageName.PRETRAINING:
        return False
    return latest_checkpoint(checkpoint_directory=artifact_directory / "checkpoints") is not None


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


def _orchestration_log_path(resolved_runs: tuple[ResolvedRun, ...]) -> Path:
    if not resolved_runs:
        raise ValueError("At least one resolved run is required.")
    run_parent_directories = tuple(
        resolved_run.run_directory.parent for resolved_run in resolved_runs
    )
    common_directory = Path(
        os.path.commonpath(tuple(str(directory) for directory in run_parent_directories)),
    )
    return common_directory / ORCHESTRATION_LOG_NAME


def _log_orchestration_stage_event(
    event_logger: OrchestrationEventLogger,
    event_type: OrchestrationEventType,
    resolved_run: ResolvedRun,
    planned_artifact: PlannedArtifact,
    message: str,
    command: tuple[str, ...] | None = None,
) -> None:
    event_logger.write(
        event_record=OrchestrationEventRecord(
            timestamp_utc=utc_timestamp(),
            event_type=event_type,
            experiment_name=resolved_run.experiment_configuration.experiment.name,
            run_directory=resolved_run.run_directory,
            stage_name=planned_artifact.stage_name,
            artifact_fingerprint=planned_artifact.fingerprint.value,
            resolved_config_path=resolved_run.resolved_configuration_path,
            message=message,
            command=command,
        ),
    )


def _log_run_event(
    event_logger: OrchestrationEventLogger,
    event_type: OrchestrationEventType,
    resolved_run: ResolvedRun,
    message: str,
) -> None:
    event_logger.write(
        event_record=OrchestrationEventRecord(
            timestamp_utc=utc_timestamp(),
            event_type=event_type,
            experiment_name=resolved_run.experiment_configuration.experiment.name,
            run_directory=resolved_run.run_directory,
            resolved_config_path=resolved_run.resolved_configuration_path,
            message=message,
        ),
    )


def _log_run_failure(
    event_logger: OrchestrationEventLogger,
    resolved_run: ResolvedRun,
    error: Exception,
) -> None:
    _log_run_event(
        event_logger=event_logger,
        event_type=OrchestrationEventType.RUN_FAILED,
        resolved_run=resolved_run,
        message=f"{error.__class__.__name__}: {error}",
    )


def _log_plan_summary(
    event_logger: OrchestrationEventLogger,
    event_type: OrchestrationEventType,
    resolved_runs: tuple[ResolvedRun, ...],
    message: str,
    completed_run_count: int,
    failed_run_count: int,
    cancelled_run_count: int,
) -> None:
    event_logger.write(
        event_record=OrchestrationEventRecord(
            timestamp_utc=utc_timestamp(),
            event_type=event_type,
            experiment_name="run_plan",
            run_directory=_orchestration_log_path(resolved_runs=resolved_runs).parent,
            message=message,
            planned_run_count=len(resolved_runs),
            completed_run_count=completed_run_count,
            failed_run_count=failed_run_count,
            cancelled_run_count=cancelled_run_count,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
