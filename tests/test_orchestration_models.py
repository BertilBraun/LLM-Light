import json
import subprocess
import sys
from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import TrainingEvaluationConfiguration
from llm_lite.orchestration.checkpoint_evaluation import (
    checkpoint_evaluation_artifact,
    checkpoint_evaluation_target_from_event,
    checkpoint_evaluation_target_from_manifest,
    checkpoint_event_is_supported_for_evaluation,
)
from llm_lite.orchestration.models import ArtifactStorePaths, ResolvedRun, resolve_run
from llm_lite.orchestration.runtime import (
    artifact_registry_for_resolved_run,
    parent_fingerprints,
    write_run_manifest,
)
from llm_lite.pipeline.stage import StageName
from llm_lite.pipeline.stages import ORDERED_PIPELINE_STAGES
from llm_lite.scripts.run_plan import (
    AsyncEvaluationSubmission,
    GpuAllocation,
    GpuPool,
    _checkpoint_evaluation_submissions,
    _complete_stage_names_from_artifact_store,
    _job_environment,
    _run_checkpoint_evaluation_job,
    run_plan,
)


def test_gpu_pool_allocates_non_overlapping_devices() -> None:
    gpu_pool = GpuPool.from_argument(gpus="0,1,2,3")

    first_allocation = gpu_pool.acquire(gpu_count=2)
    second_allocation = gpu_pool.acquire(gpu_count=2)

    assert first_allocation.environment_value() == "0,1"
    assert second_allocation.environment_value() == "2,3"


def test_gpu_pool_reuses_released_devices_in_pool_order() -> None:
    gpu_pool = GpuPool.from_argument(gpus="0,1,2,3")
    first_allocation = gpu_pool.acquire(gpu_count=2)
    second_allocation = gpu_pool.acquire(gpu_count=2)

    gpu_pool.release(gpu_allocation=first_allocation)
    third_allocation = gpu_pool.acquire(gpu_count=1)

    assert second_allocation.environment_value() == "2,3"
    assert third_allocation.environment_value() == "0"


def test_gpu_pool_rejects_requests_larger_than_configured_pool() -> None:
    gpu_pool = GpuPool.from_argument(gpus="0,1")

    try:
        gpu_pool.acquire(gpu_count=3)
    except ValueError as error:
        assert "requires 3 GPU(s)" in str(error)
        assert "only 2 device(s)" in str(error)
    else:
        raise AssertionError("Expected oversized GPU request to fail.")


def test_checkpoint_evaluation_uses_first_training_gpu() -> None:
    training_allocation = GpuAllocation(visible_devices=("2", "3"))

    evaluation_environment = _job_environment(
        gpu_allocation=training_allocation.first_device(),
    )

    assert evaluation_environment["CUDA_VISIBLE_DEVICES"] == "2"


def test_resolved_run_uses_semantic_parent_fingerprints() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )

    resolved_run = resolve_run(
        experiment_configuration=experiment_configuration,
        stages=ORDERED_PIPELINE_STAGES,
    )

    raw_artifact = resolved_run.artifact_for_stage(stage_name=StageName.RAW_DATASET)
    processed_artifact = resolved_run.artifact_for_stage(stage_name=StageName.PROCESSED_DATASET)
    packed_artifact = resolved_run.artifact_for_stage(stage_name=StageName.PACKED_DATASET)
    tokenizer_artifact = resolved_run.artifact_for_stage(stage_name=StageName.TOKENIZER)

    assert processed_artifact.parent_fingerprints[0].fingerprint == raw_artifact.fingerprint
    assert {
        parent.stage_name: parent.fingerprint for parent in packed_artifact.parent_fingerprints
    } == {
        StageName.PROCESSED_DATASET: processed_artifact.fingerprint,
        StageName.TOKENIZER: tokenizer_artifact.fingerprint,
    }


def test_artifact_store_paths_resolve_from_run_directory() -> None:
    run_directory = Path("runs") / "example"
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    resolved_run = resolve_run(
        experiment_configuration=experiment_configuration,
        stages=ORDERED_PIPELINE_STAGES,
    )
    raw_artifact = resolved_run.artifact_for_stage(stage_name=StageName.RAW_DATASET)
    store_paths = ArtifactStorePaths.for_run_directory(run_directory=run_directory)

    assert store_paths.root_directory == Path("artifact_store")
    assert (
        store_paths.artifact_directory(
            stage_name=StageName.RAW_DATASET,
            fingerprint=raw_artifact.fingerprint,
        ).parent
        == Path("artifact_store") / "raw_dataset"
    )
    assert store_paths.artifact_directory(
        stage_name=StageName.RAW_DATASET,
        fingerprint=raw_artifact.fingerprint,
    ).name == raw_artifact.fingerprint.value.replace(":", "_")


def test_pipeline_writes_resolved_run_and_semantic_manifest(tmp_path: Path) -> None:
    run_directory = tmp_path / "verify_one_sentence"
    configuration_path = tmp_path / "verify_one_sentence.yaml"
    configuration_path.write_text(
        Path("configs/verify_one_sentence.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "output_dir: runs/verify_one_sentence",
            f"output_dir: {str(run_directory).replace(chr(92), '/')}",
        ),
        encoding="utf-8",
    )

    exit_code = run_plan(
        configuration_paths=(configuration_path,),
        max_parallel_jobs=1,
        gpus=None,
        to_stage=StageName.RAW_DATASET,
    )

    run_manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    raw_fingerprint = run_manifest["artifacts"]["raw_dataset"]
    raw_manifest = json.loads(
        (
            tmp_path
            / "artifact_store"
            / "raw_dataset"
            / raw_fingerprint.replace(":", "_")
            / "manifest.json"
        ).read_text(encoding="utf-8"),
    )

    assert exit_code == 0
    assert (run_directory / "resolved_config.json").exists()
    assert raw_manifest["stage_name"] == "raw_dataset"
    assert raw_manifest["fingerprint"].startswith("sha256:")
    assert raw_manifest["parents"] == {}
    assert run_manifest["artifacts"]["raw_dataset"] == raw_manifest["fingerprint"]


def test_run_plan_writes_raw_dataset_to_artifact_store(tmp_path: Path) -> None:
    run_directory = tmp_path / "runs" / "verify_one_sentence"
    configuration_path = tmp_path / "verify_one_sentence.yaml"
    configuration_path.write_text(
        Path("configs/verify_one_sentence.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "output_dir: runs/verify_one_sentence",
            f"output_dir: {str(run_directory).replace(chr(92), '/')}",
        ),
        encoding="utf-8",
    )

    completed_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_lite.scripts.run_plan",
            "--config",
            str(configuration_path),
            "--to",
            StageName.RAW_DATASET.value,
        ],
        check=False,
        cwd=Path.cwd(),
    )

    run_manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    raw_fingerprint = run_manifest["artifacts"]["raw_dataset"]
    artifact_manifest_path = (
        tmp_path
        / "artifact_store"
        / "raw_dataset"
        / raw_fingerprint.replace(":", "_")
        / "manifest.json"
    )
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))

    assert completed_process.returncode == 0
    assert (run_directory / "resolved_config.json").exists()
    assert artifact_manifest["fingerprint"] == raw_fingerprint
    assert artifact_manifest["status"] == "complete"


def test_run_plan_accepts_parallel_configurations(tmp_path: Path) -> None:
    first_run_directory = tmp_path / "runs" / "first"
    second_run_directory = tmp_path / "runs" / "second"
    first_configuration_path = tmp_path / "first.yaml"
    second_configuration_path = tmp_path / "second.yaml"
    base_configuration_text = Path("configs/verify_one_sentence.yaml").read_text(encoding="utf-8")
    first_configuration_path.write_text(
        base_configuration_text.replace(
            "name: verify_one_sentence",
            "name: first",
        ).replace(
            "output_dir: runs/verify_one_sentence",
            f"output_dir: {str(first_run_directory).replace(chr(92), '/')}",
        ),
        encoding="utf-8",
    )
    second_configuration_path.write_text(
        base_configuration_text.replace(
            "name: verify_one_sentence",
            "name: second",
        ).replace(
            "output_dir: runs/verify_one_sentence",
            f"output_dir: {str(second_run_directory).replace(chr(92), '/')}",
        ),
        encoding="utf-8",
    )

    completed_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_lite.scripts.run_plan",
            "--config",
            str(first_configuration_path),
            str(second_configuration_path),
            "--to",
            StageName.RAW_DATASET.value,
            "--max-parallel-jobs",
            "2",
        ],
        check=False,
        cwd=Path.cwd(),
    )

    first_manifest = json.loads(
        (first_run_directory / "run_manifest.json").read_text(encoding="utf-8"),
    )
    second_manifest = json.loads(
        (second_run_directory / "run_manifest.json").read_text(encoding="utf-8"),
    )

    assert completed_process.returncode == 0
    assert first_manifest["artifacts"]["raw_dataset"] == second_manifest["artifacts"]["raw_dataset"]


def test_run_manifest_preserves_complete_artifacts_for_partial_stage_run(
    tmp_path: Path,
) -> None:
    base_experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    experiment_configuration = base_experiment_configuration.model_copy(
        update={
            "experiment": base_experiment_configuration.experiment.model_copy(
                update={"output_dir": tmp_path / "run"},
            ),
        },
    )
    resolved_run = resolve_run(
        experiment_configuration=experiment_configuration,
        stages=ORDERED_PIPELINE_STAGES,
    )
    registry = artifact_registry_for_resolved_run(resolved_run=resolved_run)
    for stage_name in (StageName.PRETRAINING, StageName.EVALUATION):
        planned_artifact = resolved_run.artifact_for_stage(stage_name=stage_name)
        registry.write_complete_manifest(
            artifact_type=stage_name.value,
            fingerprint=planned_artifact.fingerprint.value,
            configuration_hash=planned_artifact.configuration_hash,
            parent_hashes=parent_fingerprints(planned_artifact=planned_artifact),
            contract_version=planned_artifact.contract_version,
            files={},
            metrics={},
        )
    resolved_run.run_directory.mkdir(parents=True, exist_ok=True)
    resolved_run.run_manifest_path.write_text(
        json.dumps({"experiment": "verify_one_sentence", "artifacts": {"export": "stale"}}),
        encoding="utf-8",
    )

    completed_stage_names = _complete_stage_names_from_artifact_store(
        resolved_run=resolved_run,
        registry=registry,
    )
    write_run_manifest(
        resolved_run=resolved_run,
        completed_stage_names=completed_stage_names,
    )

    run_manifest = json.loads(resolved_run.run_manifest_path.read_text(encoding="utf-8"))

    assert run_manifest["artifacts"] == {
        "pretraining": resolved_run.artifact_for_stage(
            stage_name=StageName.PRETRAINING,
        ).fingerprint.value,
        "evaluation": resolved_run.artifact_for_stage(
            stage_name=StageName.EVALUATION,
        ).fingerprint.value,
    }


def test_checkpoint_evaluation_artifact_uses_checkpoint_step_fingerprint(
    tmp_path: Path,
) -> None:
    base_experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    experiment_configuration = base_experiment_configuration.model_copy(
        update={
            "experiment": base_experiment_configuration.experiment.model_copy(
                update={"output_dir": tmp_path / "run"},
            ),
            "training": base_experiment_configuration.training.model_copy(
                update={
                    "evaluation": TrainingEvaluationConfiguration(
                        interval_steps=2,
                        evaluators=base_experiment_configuration.evaluation,
                    ),
                },
            ),
        },
    )
    resolved_run = resolve_run(
        experiment_configuration=experiment_configuration,
        stages=ORDERED_PIPELINE_STAGES,
    )
    pretraining_artifact = resolved_run.artifact_for_stage(stage_name=StageName.PRETRAINING)
    pretraining_artifact_directory = resolved_run.artifact_store_paths.artifact_directory(
        stage_name=StageName.PRETRAINING,
        fingerprint=pretraining_artifact.fingerprint,
    )
    first_event_path = _write_checkpoint_event(
        artifact_directory=pretraining_artifact_directory,
        producing_artifact_fingerprint=pretraining_artifact.fingerprint.value,
        checkpoint_step=2,
    )
    second_event_path = _write_checkpoint_event(
        artifact_directory=pretraining_artifact_directory,
        producing_artifact_fingerprint=pretraining_artifact.fingerprint.value,
        checkpoint_step=4,
    )

    first_target = checkpoint_evaluation_target_from_event(
        resolved_run=resolved_run,
        event_path=first_event_path,
    )
    second_target = checkpoint_evaluation_target_from_event(
        resolved_run=resolved_run,
        event_path=second_event_path,
    )
    first_artifact = checkpoint_evaluation_artifact(
        resolved_run=resolved_run,
        target=first_target,
    )
    second_artifact = checkpoint_evaluation_artifact(
        resolved_run=resolved_run,
        target=second_target,
    )

    assert first_artifact.stage_name is StageName.EVALUATION
    assert first_artifact.fingerprint != second_artifact.fingerprint
    assert {
        parent.stage_name: parent.fingerprint for parent in first_artifact.parent_fingerprints
    } == {
        StageName.PRETRAINING: pretraining_artifact.fingerprint,
        StageName.TOKENIZER: resolved_run.artifact_for_stage(
            stage_name=StageName.TOKENIZER,
        ).fingerprint,
    }


def test_ddp_sharded_checkpoint_event_resolves_to_rank_zero_full_checkpoint(
    tmp_path: Path,
) -> None:
    base_experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    experiment_configuration = base_experiment_configuration.model_copy(
        update={
            "experiment": base_experiment_configuration.experiment.model_copy(
                update={"output_dir": tmp_path / "run"},
            ),
            "training": base_experiment_configuration.training.model_copy(
                update={
                    "evaluation": TrainingEvaluationConfiguration(
                        interval_steps=2,
                        evaluators=base_experiment_configuration.evaluation,
                    ),
                },
            ),
        },
    )
    resolved_run = resolve_run(
        experiment_configuration=experiment_configuration,
        stages=ORDERED_PIPELINE_STAGES,
    )
    pretraining_artifact = resolved_run.artifact_for_stage(stage_name=StageName.PRETRAINING)
    pretraining_artifact_directory = resolved_run.artifact_store_paths.artifact_directory(
        stage_name=StageName.PRETRAINING,
        fingerprint=pretraining_artifact.fingerprint,
    )
    event_path = _write_sharded_checkpoint_event(
        artifact_directory=pretraining_artifact_directory,
        producing_artifact_fingerprint=pretraining_artifact.fingerprint.value,
        checkpoint_step=2,
    )

    target_from_event = checkpoint_evaluation_target_from_event(
        resolved_run=resolved_run,
        event_path=event_path,
    )
    target_from_manifest = checkpoint_evaluation_target_from_manifest(
        resolved_run=resolved_run,
        checkpoint_manifest_path=target_from_event.checkpoint_manifest_path,
    )

    assert checkpoint_event_is_supported_for_evaluation(event_path=event_path)
    assert target_from_event.checkpoint_path.name == "rank_zero_full.pt"
    assert target_from_event.checkpoint_path == target_from_manifest.checkpoint_path
    assert target_from_event.checkpoint_step == 2


def test_pruned_checkpoint_event_is_not_submitted_for_evaluation(tmp_path: Path) -> None:
    resolved_run, pretraining_artifact_directory = _resolved_run_with_training_evaluation(
        tmp_path=tmp_path,
    )
    event_path = _write_checkpoint_event(
        artifact_directory=pretraining_artifact_directory,
        producing_artifact_fingerprint=resolved_run.artifact_for_stage(
            stage_name=StageName.PRETRAINING,
        ).fingerprint.value,
        checkpoint_step=2,
    )
    target = checkpoint_evaluation_target_from_event(
        resolved_run=resolved_run,
        event_path=event_path,
    )
    target.checkpoint_manifest_path.unlink()

    submissions = _checkpoint_evaluation_submissions(
        resolved_run=resolved_run,
        training_artifact_directory=pretraining_artifact_directory,
        seen_event_paths=set(),
    )

    assert submissions == ()


def test_pruned_queued_checkpoint_evaluation_is_skipped(tmp_path: Path) -> None:
    resolved_run, pretraining_artifact_directory = _resolved_run_with_training_evaluation(
        tmp_path=tmp_path,
    )
    event_path = _write_checkpoint_event(
        artifact_directory=pretraining_artifact_directory,
        producing_artifact_fingerprint=resolved_run.artifact_for_stage(
            stage_name=StageName.PRETRAINING,
        ).fingerprint.value,
        checkpoint_step=2,
    )
    target = checkpoint_evaluation_target_from_event(
        resolved_run=resolved_run,
        event_path=event_path,
    )
    planned_artifact = checkpoint_evaluation_artifact(
        resolved_run=resolved_run,
        target=target,
    )
    submission = AsyncEvaluationSubmission(
        planned_artifact=planned_artifact,
        event_path=event_path,
        checkpoint_manifest_path=target.checkpoint_manifest_path,
    )
    target.checkpoint_manifest_path.unlink()

    _run_checkpoint_evaluation_job(
        resolved_run=resolved_run,
        submission=submission,
        gpu_allocation=GpuAllocation(visible_devices=()),
    )


def _resolved_run_with_training_evaluation(tmp_path: Path) -> tuple[ResolvedRun, Path]:
    base_experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    experiment_configuration = base_experiment_configuration.model_copy(
        update={
            "experiment": base_experiment_configuration.experiment.model_copy(
                update={"output_dir": tmp_path / "run"},
            ),
            "training": base_experiment_configuration.training.model_copy(
                update={
                    "evaluation": TrainingEvaluationConfiguration(
                        interval_steps=2,
                        evaluators=base_experiment_configuration.evaluation,
                    ),
                },
            ),
        },
    )
    resolved_run = resolve_run(
        experiment_configuration=experiment_configuration,
        stages=ORDERED_PIPELINE_STAGES,
    )
    pretraining_artifact = resolved_run.artifact_for_stage(stage_name=StageName.PRETRAINING)
    pretraining_artifact_directory = resolved_run.artifact_store_paths.artifact_directory(
        stage_name=StageName.PRETRAINING,
        fingerprint=pretraining_artifact.fingerprint,
    )
    return resolved_run, pretraining_artifact_directory


def _write_checkpoint_event(
    artifact_directory: Path,
    producing_artifact_fingerprint: str,
    checkpoint_step: int,
) -> Path:
    checkpoint_directory = artifact_directory / "checkpoints"
    step_directory = checkpoint_directory / f"step_{checkpoint_step:08d}"
    step_directory.mkdir(parents=True)
    checkpoint_path = checkpoint_directory / f"step_{checkpoint_step:08d}.pt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    (step_directory / "manifest.json").write_text(
        json.dumps(
            {
                "step": checkpoint_step,
                "producing_artifact_fingerprint": producing_artifact_fingerprint,
                "checkpoint_kind": "full",
                "checkpoint_path": f"../step_{checkpoint_step:08d}.pt",
                "completion_status": "complete",
                "created_at": "2026-06-28T00:00:00Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    event_directory = artifact_directory / "events"
    event_directory.mkdir(parents=True, exist_ok=True)
    event_path = event_directory / f"checkpoint_{checkpoint_step:08d}.json"
    event_path.write_text(
        json.dumps(
            {
                "producing_artifact_fingerprint": producing_artifact_fingerprint,
                "checkpoint_step": checkpoint_step,
                "checkpoint_manifest_path": (
                    f"checkpoints/step_{checkpoint_step:08d}/manifest.json"
                ),
                "checkpoint_kind": "full",
                "created_at": "2026-06-28T00:00:00Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return event_path


def _write_sharded_checkpoint_event(
    artifact_directory: Path,
    producing_artifact_fingerprint: str,
    checkpoint_step: int,
) -> Path:
    checkpoint_directory = artifact_directory / "checkpoints"
    step_directory = checkpoint_directory / f"step_{checkpoint_step:08d}"
    step_directory.mkdir(parents=True)
    (step_directory / "rank_zero_full.pt").write_text("checkpoint", encoding="utf-8")
    (step_directory / "manifest.json").write_text(
        json.dumps(
            {
                "step": checkpoint_step,
                "producing_artifact_fingerprint": producing_artifact_fingerprint,
                "world_size": 2,
                "backend": "gloo",
                "strategy": "data_parallel",
                "checkpoint_type": "sharded",
                "topology": {},
                "model_configuration_hash": "model-hash",
                "optimizer_present": True,
                "expected_rank_shards": ["rank_00000", "rank_00001"],
                "completion_status": "complete",
                "created_at": "2026-06-28T00:00:00Z",
                "rank_zero_full_checkpoint_path": "rank_zero_full.pt",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    event_directory = artifact_directory / "events"
    event_directory.mkdir(parents=True, exist_ok=True)
    event_path = event_directory / f"checkpoint_{checkpoint_step:08d}.json"
    event_path.write_text(
        json.dumps(
            {
                "producing_artifact_fingerprint": producing_artifact_fingerprint,
                "checkpoint_step": checkpoint_step,
                "checkpoint_manifest_path": (
                    f"checkpoints/step_{checkpoint_step:08d}/manifest.json"
                ),
                "checkpoint_kind": "sharded",
                "created_at": "2026-06-28T00:00:00Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return event_path
