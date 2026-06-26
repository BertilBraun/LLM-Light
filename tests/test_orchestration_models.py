import json
from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.orchestration.models import ArtifactStorePaths, resolve_run
from llm_lite.pipeline.runner import run_pipeline
from llm_lite.pipeline.stage import StageName
from llm_lite.pipeline.stages import ORDERED_PIPELINE_STAGES


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

    exit_code = run_pipeline(
        configuration_path=configuration_path,
        dry_run=False,
        force_stages=(),
        to_stage=StageName.RAW_DATASET,
    )

    raw_manifest = json.loads(
        (run_directory / "artifacts" / "raw_dataset" / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    run_manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert (run_directory / "resolved_config.json").exists()
    assert raw_manifest["stage_name"] == "raw_dataset"
    assert raw_manifest["fingerprint"].startswith("sha256:")
    assert raw_manifest["parents"] == {}
    assert run_manifest["artifacts"]["raw_dataset"] == raw_manifest["fingerprint"]
