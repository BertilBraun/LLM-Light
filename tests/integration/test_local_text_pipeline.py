import json
from pathlib import Path

from llm_lite.pipeline.runner import run_pipeline
from tests.artifact_helpers import stage_artifact_directory


def test_local_text_pipeline_produces_packed_dataset(tmp_path: Path) -> None:
    run_directory = tmp_path / "verify_local_text"
    configuration_path = tmp_path / "verify_local_text.yaml"
    configuration_text = Path("tests/configs/verify_local_text.yaml").read_text(
        encoding="utf-8",
    )
    configuration_path.write_text(
        configuration_text.replace(
            "output_dir: runs/verify_local_text",
            f"output_dir: {str(run_directory).replace(chr(92), '/')}",
        ),
        encoding="utf-8",
    )

    exit_code = run_pipeline(
        configuration_path=configuration_path,
        dry_run=False,
        force_stages=(),
    )
    raw_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="raw_dataset",
    )
    processed_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="processed_dataset",
    )
    packed_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="packed_dataset",
    )
    raw_manifest = json.loads(
        (raw_artifact_directory / "manifest.json").read_text(encoding="utf-8"),
    )
    processed_manifest = json.loads(
        (processed_artifact_directory / "manifest.json").read_text(encoding="utf-8"),
    )
    packed_index = json.loads((packed_artifact_directory / "index.json").read_text("utf-8"))

    assert exit_code == 0
    assert raw_manifest["metrics"]["raw_documents"] == 1
    assert processed_manifest["metrics"]["processed_documents"] == 1
    assert processed_manifest["metrics"]["total_characters"] == 12
    assert (raw_artifact_directory / "unsplit").is_dir()
    assert (processed_artifact_directory / "train").is_dir()
    assert list((processed_artifact_directory / "train").glob("*.tar.gz"))
    assert packed_index["total_sequences"] >= 1
