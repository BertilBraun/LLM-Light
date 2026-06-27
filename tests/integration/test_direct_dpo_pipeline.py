import json
from pathlib import Path

from llm_lite.pipeline.runner import run_pipeline
from tests.artifact_helpers import stage_artifact_directory


def test_direct_dpo_pipeline_runs_post_training(tmp_path: Path) -> None:
    run_directory = tmp_path / "direct_dpo_smoke"
    configuration_path = tmp_path / "direct_dpo_smoke.yaml"
    configuration_text = Path("tests/configs/direct_dpo_smoke.yaml").read_text(
        encoding="utf-8",
    )
    configuration_path.write_text(
        configuration_text.replace(
            "output_dir: runs/direct_dpo_smoke",
            f"output_dir: {str(run_directory).replace(chr(92), '/')}",
        ),
        encoding="utf-8",
    )

    exit_code = run_pipeline(
        configuration_path=configuration_path,
        dry_run=False,
        force_stages=(),
    )
    post_training_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="post_training",
    )
    post_training_manifest = json.loads(
        (post_training_artifact_directory / "manifest.json").read_text(encoding="utf-8"),
    )
    evaluation_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="evaluation",
    )
    evaluation_report = json.loads(
        (evaluation_artifact_directory / "report.json").read_text(encoding="utf-8"),
    )

    assert exit_code == 0
    assert post_training_manifest["metrics"]["post_training_enabled"] is True
    assert post_training_manifest["metrics"]["preference_pairs"] == 1
    assert post_training_manifest["metrics"]["final_step"] == 5
    assert (post_training_artifact_directory / "checkpoints" / "latest.pt").exists()
    assert evaluation_report == {}
