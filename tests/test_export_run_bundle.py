import json
import zipfile
from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.orchestration.models import resolve_run
from llm_lite.pipeline.export_bundle import collect_bundle_entries, write_bundle
from llm_lite.pipeline.registry import ArtifactDirectory, ArtifactRegistry
from llm_lite.pipeline.stage import StageName
from llm_lite.pipeline.stages import ORDERED_PIPELINE_STAGES
from llm_lite.pipeline.stages.export import ExportStage


def test_collect_bundle_entries_includes_latest_sharded_checkpoint_only(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    artifact_store_directory = tmp_path / "artifact_store"
    pretraining_directory = artifact_store_directory / "pretraining" / "sha256_pretraining"
    tokenizer_directory = artifact_store_directory / "tokenizer" / "sha256_tokenizer"
    _write(run_directory / "resolved_config.json", "{}")
    _write(
        run_directory / "run_manifest.json",
        json.dumps(
            {
                "experiment": "run",
                "artifacts": {
                    "pretraining": "sha256:pretraining",
                    "tokenizer": "sha256:tokenizer",
                },
            },
        ),
    )
    _write(run_directory / "pipeline.jsonl", "{}\n")
    _write(tokenizer_directory / "tokenizer.json", "{}")
    _write(tokenizer_directory / "manifest.json", "{}")
    _write(pretraining_directory / "metrics.jsonl", "{}\n")
    _write(pretraining_directory / "manifest.json", "{}")
    _write(
        pretraining_directory / "checkpoints" / "latest.json",
        json.dumps({"step": 20, "checkpoint": "step_00000020"}),
    )
    _write(
        pretraining_directory / "checkpoints" / "step_00000010" / "rank_00000" / "state.pt",
        "old",
    )
    _write(
        pretraining_directory / "checkpoints" / "step_00000020" / "rank_00000" / "state.pt",
        "latest",
    )

    entries = collect_bundle_entries(run_directory=run_directory)
    archive_paths = {entry.archive_path.as_posix() for entry in entries}

    assert "resolved_config.json" in archive_paths
    assert "artifacts/pretraining/manifest.json" in archive_paths
    assert "artifacts/tokenizer/manifest.json" in archive_paths
    assert "artifacts/tokenizer/tokenizer.json" in archive_paths
    assert "artifacts/pretraining/checkpoints/latest.json" in archive_paths
    assert "artifacts/pretraining/checkpoints/step_00000020/rank_00000/state.pt" in archive_paths
    assert (
        "artifacts/pretraining/checkpoints/step_00000010/rank_00000/state.pt" not in archive_paths
    )


def test_write_bundle_creates_zip_with_manifest(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    artifact_store_directory = tmp_path / "artifact_store"
    pretraining_directory = artifact_store_directory / "pretraining" / "sha256_pretraining"
    output_path = tmp_path / "bundle.zip"
    _write(run_directory / "resolved_config.json", "{}")
    _write(
        run_directory / "run_manifest.json",
        json.dumps(
            {
                "experiment": "run",
                "artifacts": {"pretraining": "sha256:pretraining"},
            },
        ),
    )
    _write(pretraining_directory / "checkpoints" / "latest.pt", "state")

    write_bundle(run_directory=run_directory, output_path=output_path)

    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("bundle_manifest.json"))

    assert "resolved_config.json" in names
    assert "run_manifest.json" in names
    assert "artifacts/pretraining/checkpoints/latest.pt" in names
    assert manifest["include_all_checkpoints"] is False
    assert manifest["file_count"] == 3
    assert manifest["experiment"] == "run"
    assert manifest["artifacts"][0]["stage_name"] == "pretraining"


def test_collect_bundle_entries_prefers_latest_post_training_checkpoint(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    artifact_store_directory = tmp_path / "artifact_store"
    pretraining_directory = artifact_store_directory / "pretraining" / "sha256_pretraining"
    post_training_directory = artifact_store_directory / "post_training" / "sha256_post_training"
    _write(
        run_directory / "run_manifest.json",
        json.dumps(
            {
                "experiment": "run",
                "artifacts": {
                    "pretraining": "sha256:pretraining",
                    "post_training": "sha256:post_training",
                },
            },
        ),
    )
    _write(pretraining_directory / "checkpoints" / "latest.pt", "pretraining")
    _write(post_training_directory / "checkpoints" / "latest.pt", "post-training")

    entries = collect_bundle_entries(run_directory=run_directory)
    archive_paths = {entry.archive_path.as_posix() for entry in entries}

    assert "artifacts/post_training/checkpoints/latest.pt" in archive_paths
    assert "artifacts/pretraining/checkpoints/latest.pt" not in archive_paths


def test_export_stage_writes_cacheable_artifact_and_configured_bundle(tmp_path: Path) -> None:
    run_directory = tmp_path / "runs" / "verify_one_sentence"
    base_experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    experiment_configuration = base_experiment_configuration.model_copy(
        update={
            "experiment": base_experiment_configuration.experiment.model_copy(
                update={"output_dir": run_directory},
            ),
        },
    )
    resolved_run = resolve_run(
        experiment_configuration=experiment_configuration,
        stages=ORDERED_PIPELINE_STAGES,
    )
    pretraining_artifact = resolved_run.artifact_for_stage(stage_name=StageName.PRETRAINING)
    evaluation_artifact = resolved_run.artifact_for_stage(stage_name=StageName.EVALUATION)
    export_artifact = resolved_run.artifact_for_stage(stage_name=StageName.EXPORT)
    pretraining_directory = resolved_run.artifact_store_paths.artifact_directory(
        stage_name=StageName.PRETRAINING,
        fingerprint=pretraining_artifact.fingerprint,
    )
    evaluation_directory = resolved_run.artifact_store_paths.artifact_directory(
        stage_name=StageName.EVALUATION,
        fingerprint=evaluation_artifact.fingerprint,
    )
    export_directory = resolved_run.artifact_store_paths.artifact_directory(
        stage_name=StageName.EXPORT,
        fingerprint=export_artifact.fingerprint,
    )
    _write(run_directory / "resolved_config.json", "{}")
    _write(
        run_directory / "run_manifest.json",
        json.dumps(
            {
                "experiment": "verify_one_sentence",
                "artifacts": {
                    "pretraining": pretraining_artifact.fingerprint.value,
                    "evaluation": evaluation_artifact.fingerprint.value,
                },
            },
        ),
    )
    _write(pretraining_directory / "manifest.json", "{}")
    _write(pretraining_directory / "metrics.jsonl", "{}\n")
    _write(pretraining_directory / "checkpoints" / "latest.pt", "state")
    _write(evaluation_directory / "manifest.json", "{}")
    _write(evaluation_directory / "report.json", "{}")
    registry = ArtifactRegistry(
        run_directory=run_directory,
        artifact_directories=(
            ArtifactDirectory(
                artifact_type=StageName.EXPORT.value,
                directory=export_directory,
            ),
        ),
    )

    stage_output = ExportStage().run(
        experiment_configuration=experiment_configuration,
        registry=registry,
        artifact_directory=export_directory,
    )

    configured_bundle_path = run_directory / "export" / "bundle.zip"
    assert stage_output.files == {
        "bundle": "bundle.zip",
        "bundle_manifest": "bundle_manifest.json",
    }
    assert (export_directory / "bundle.zip").exists()
    assert (export_directory / "bundle_manifest.json").exists()
    assert configured_bundle_path.exists()
    with zipfile.ZipFile(export_directory / "bundle.zip") as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("bundle_manifest.json"))
    assert "artifacts/pretraining/checkpoints/latest.pt" in names
    assert "artifacts/evaluation/report.json" in names
    assert manifest["experiment"] == "verify_one_sentence"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
