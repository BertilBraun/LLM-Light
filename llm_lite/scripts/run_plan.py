import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.orchestration.models import PlannedArtifact, ResolvedRun, resolve_run
from llm_lite.orchestration.runtime import (
    acquire_artifact_lock,
    artifact_registry_for_resolved_run,
    complete_manifest_matches,
    copy_stage_tensorboard_to_run_view,
    release_artifact_lock,
    write_resolved_configuration,
    write_run_manifest,
)
from llm_lite.pipeline.logging import (
    PipelineEventLogger,
    PipelineEventRecord,
    PipelineEventType,
)
from llm_lite.pipeline.progress import console_log
from llm_lite.pipeline.stage import StageName
from llm_lite.pipeline.stages import ORDERED_PIPELINE_STAGES, ORDERED_STAGE_NAMES

LOCK_WAIT_SECONDS = 5.0


@dataclass(frozen=True)
class GpuPool:
    visible_devices: tuple[str, ...]

    @classmethod
    def from_argument(cls, gpus: str | None) -> "GpuPool":
        if gpus is None or gpus == "":
            return cls(visible_devices=())
        return cls(visible_devices=tuple(gpu.strip() for gpu in gpus.split(",") if gpu.strip()))

    def environment_value(self, gpu_count: int) -> str | None:
        if gpu_count == 0:
            return None
        if not self.visible_devices:
            return None
        return ",".join(self.visible_devices[:gpu_count])


@dataclass(frozen=True)
class ResourceRequest:
    cpu_workers: int
    gpu_count: int
    exclusive_gpus: bool


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
    configuration_paths = _expand_configuration_paths(configurations=tuple(arguments.config))
    resolved_runs = tuple(
        resolve_run(
            experiment_configuration=load_experiment_configuration(
                configuration_path=configuration_path,
            ),
            stages=ORDERED_PIPELINE_STAGES,
        )
        for configuration_path in configuration_paths
    )
    gpu_pool = GpuPool.from_argument(gpus=arguments.gpus)
    selected_stage_names = _selected_stage_names(
        from_stage=None if arguments.from_stage is None else StageName(arguments.from_stage),
        to_stage=None if arguments.to_stage is None else StageName(arguments.to_stage),
    )
    for resolved_run in resolved_runs:
        _execute_resolved_run(
            resolved_run=resolved_run,
            gpu_pool=gpu_pool,
            selected_stage_names=selected_stage_names,
        )
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
) -> None:
    write_resolved_configuration(resolved_run=resolved_run)
    registry = artifact_registry_for_resolved_run(resolved_run=resolved_run)
    event_logger = PipelineEventLogger(run_directory=resolved_run.run_directory)
    completed_stage_names: list[StageName] = []
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
            completed_stage_names.append(planned_artifact.stage_name)
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
        )
        completed_stage_names.append(planned_artifact.stage_name)
        copy_stage_tensorboard_to_run_view(
            resolved_run=resolved_run,
            stage_name=planned_artifact.stage_name,
        )
        write_run_manifest(
            resolved_run=resolved_run,
            completed_stage_names=tuple(completed_stage_names),
        )


def _run_missing_artifact(
    resolved_run: ResolvedRun,
    planned_artifact: PlannedArtifact,
    gpu_pool: GpuPool,
    event_logger: PipelineEventLogger,
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
        _run_subprocess_job(
            command=command,
            artifact_directory=artifact_directory,
            environment=_job_environment(
                resolved_run=resolved_run,
                planned_artifact=planned_artifact,
                gpu_pool=gpu_pool,
            ),
        )
        completed_manifest = registry.read_manifest(artifact_type=planned_artifact.stage_name.value)
        if not complete_manifest_matches(
            manifest=completed_manifest,
            planned_artifact=planned_artifact,
        ):
            raise ValueError(
                f"Stage {planned_artifact.stage_name.value} exited without a complete manifest.",
            )
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


def _job_environment(
    resolved_run: ResolvedRun,
    planned_artifact: PlannedArtifact,
    gpu_pool: GpuPool,
) -> dict[str, str]:
    environment = os.environ.copy()
    gpu_environment = gpu_pool.environment_value(
        gpu_count=_resource_request(
            resolved_run=resolved_run,
            planned_artifact=planned_artifact,
        ).gpu_count,
    )
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
                    else 0
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
    log_path = artifact_directory / "job.log"
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


def _clear_incomplete_artifact_payload(artifact_directory: Path, artifact_store_root: Path) -> None:
    resolved_artifact_directory = artifact_directory.resolve()
    resolved_store_root = artifact_store_root.resolve()
    if resolved_store_root not in resolved_artifact_directory.parents:
        raise ValueError(f"Refusing to clear artifact outside store: {artifact_directory}")
    for child_path in artifact_directory.iterdir():
        if child_path.name == ".lock":
            continue
        if child_path.is_dir():
            shutil.rmtree(child_path)
        else:
            child_path.unlink()


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
