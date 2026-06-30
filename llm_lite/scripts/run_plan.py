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
from threading import Condition, Thread
from typing import TextIO

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.orchestration.checkpoint_evaluation import (
    checkpoint_evaluation_artifact,
    checkpoint_evaluation_target_from_event,
    checkpoint_event_is_supported_for_evaluation,
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

    def acquire(self, gpu_count: int) -> GpuAllocation:
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
                self._condition.wait()
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


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", required=True, nargs="+")
    argument_parser.add_argument("--max-parallel-jobs", type=int, default=1)
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
    )


def run_plan(
    configuration_paths: tuple[Path, ...],
    max_parallel_jobs: int,
    gpus: str | None,
    from_stage: StageName | None = None,
    to_stage: StageName | None = None,
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
    gpu_pool = GpuPool.from_argument(gpus=gpus)
    selected_stage_names = _selected_stage_names(
        from_stage=from_stage,
        to_stage=to_stage,
    )
    if max_parallel_jobs == 1 or len(resolved_runs) == 1:
        for resolved_run in resolved_runs:
            _execute_resolved_run(
                resolved_run=resolved_run,
                gpu_pool=gpu_pool,
                selected_stage_names=selected_stage_names,
                max_parallel_jobs=max_parallel_jobs,
            )
        return 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = tuple(
            executor.submit(
                _execute_resolved_run,
                resolved_run,
                gpu_pool,
                selected_stage_names,
                max_parallel_jobs,
            )
            for resolved_run in resolved_runs
        )
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return 0


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
) -> None:
    write_resolved_configuration(resolved_run=resolved_run)
    registry = artifact_registry_for_resolved_run(resolved_run=resolved_run)
    event_logger = PipelineEventLogger(run_directory=resolved_run.run_directory)
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
        if planned_artifact.stage_name not in selected_stage_names:
            continue
        manifest = registry.read_manifest(artifact_type=planned_artifact.stage_name.value)
        if complete_manifest_matches(manifest=manifest, planned_artifact=planned_artifact):
            console_log(f"[cache] {planned_artifact.stage_name.value}: compatible artifact found")
            _log_stage_event(
                event_logger=event_logger,
                event_type=PipelineEventType.STAGE_SKIP,
                stage_name=planned_artifact.stage_name,
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
            event_logger=event_logger,
            max_parallel_jobs=max_parallel_jobs,
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
    event_logger: PipelineEventLogger,
    max_parallel_jobs: int,
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
        time.sleep(LOCK_WAIT_SECONDS)
        manifest = registry.read_manifest(artifact_type=planned_artifact.stage_name.value)
        if complete_manifest_matches(manifest=manifest, planned_artifact=planned_artifact):
            return
    try:
        manifest = registry.read_manifest(artifact_type=planned_artifact.stage_name.value)
        if not _can_reuse_incomplete_payload(
            artifact_directory=artifact_directory,
            manifest=manifest,
            planned_artifact=planned_artifact,
        ):
            _clear_incomplete_artifact_payload(
                artifact_directory=artifact_directory,
                artifact_store_root=resolved_run.artifact_store_paths.root_directory,
            )
        _log_stage_event(
            event_logger=event_logger,
            event_type=PipelineEventType.STAGE_START,
            stage_name=planned_artifact.stage_name,
            message="stage job started",
        )
        console_log(f"[start] {planned_artifact.stage_name.value}")
        try:
            resource_request = _resource_request(
                resolved_run=resolved_run,
                planned_artifact=planned_artifact,
            )
            gpu_allocation = gpu_pool.acquire(gpu_count=resource_request.gpu_count)
            environment = _job_environment(
                gpu_allocation=gpu_allocation,
            )
            try:
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
                        max_parallel_jobs=max_parallel_jobs,
                    )
                else:
                    _run_subprocess_job(
                        command=command,
                        artifact_directory=artifact_directory,
                        environment=environment,
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
        except Exception as error:
            _log_stage_event(
                event_logger=event_logger,
                event_type=PipelineEventType.STAGE_FAILURE,
                stage_name=planned_artifact.stage_name,
                message=_failure_message(error=error, artifact_directory=artifact_directory),
            )
            console_log(
                f"[fail]  {planned_artifact.stage_name.value}; see "
                f"{artifact_directory / JOB_LOG_NAME}",
            )
            raise
        _log_stage_event(
            event_logger=event_logger,
            event_type=PipelineEventType.STAGE_COMPLETE,
            stage_name=planned_artifact.stage_name,
            message="stage job completed",
        )
        console_log(f"[done]  {planned_artifact.stage_name.value}")
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
) -> None:
    log_path = artifact_directory / JOB_LOG_NAME
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        if process.stdout is None:
            raise ValueError("Subprocess stdout was not captured.")
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
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
    max_parallel_jobs: int,
) -> None:
    seen_event_paths: set[Path] = set()
    evaluation_futures: list[concurrent.futures.Future[None]] = []
    log_path = artifact_directory / JOB_LOG_NAME
    with (
        log_path.open("a", encoding="utf-8") as log_file,
        concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_jobs - 1) as executor,
    ):
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
                _submit_ready_checkpoint_evaluations(
                    resolved_run=resolved_run,
                    training_artifact_directory=artifact_directory,
                    seen_event_paths=seen_event_paths,
                    evaluation_futures=evaluation_futures,
                    executor=executor,
                    gpu_allocation=gpu_allocation,
                )
                _raise_completed_evaluation_failures(evaluation_futures=evaluation_futures)
                time.sleep(CHECKPOINT_EVENT_POLL_SECONDS)
            output_thread.join()
            _submit_ready_checkpoint_evaluations(
                resolved_run=resolved_run,
                training_artifact_directory=artifact_directory,
                seen_event_paths=seen_event_paths,
                evaluation_futures=evaluation_futures,
                executor=executor,
                gpu_allocation=gpu_allocation,
            )
            exit_code = process.wait()
            _wait_for_async_evaluations(evaluation_futures=evaluation_futures)
        except Exception:
            _terminate_process(process=process)
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
) -> None:
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
) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
