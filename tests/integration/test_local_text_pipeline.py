import json
from pathlib import Path

from llm_lite.pipeline.runner import run_pipeline


def test_local_text_pipeline_produces_packed_dataset(tmp_path: Path) -> None:
    run_directory = tmp_path / "verify_local_text"
    configuration_path = tmp_path / "verify_local_text.yaml"
    configuration_text = Path("configs/verify_local_text.yaml").read_text(encoding="utf-8")
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
    raw_manifest = json.loads(
        (run_directory / "artifacts" / "raw_dataset" / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    processed_manifest = json.loads(
        (run_directory / "artifacts" / "processed_dataset" / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    packed_index = json.loads(
        (run_directory / "artifacts" / "packed_dataset" / "index.json").read_text(
            encoding="utf-8",
        ),
    )

    assert exit_code == 0
    assert raw_manifest["metrics"]["raw_documents"] == 1
    assert processed_manifest["metrics"]["processed_documents"] == 1
    assert processed_manifest["metrics"]["total_characters"] == 12
    assert packed_index["total_sequences"] >= 1
