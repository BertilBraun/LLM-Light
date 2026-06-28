import json
from pathlib import Path

from llm_lite.scripts.run_plan import run_plan
from tests.artifact_helpers import stage_artifact_directory


def test_byte_bpe_pipeline_produces_packed_dataset(tmp_path: Path) -> None:
    run_directory = tmp_path / "verify_byte_bpe"
    configuration_path = tmp_path / "verify_byte_bpe.yaml"
    configuration_text = Path("tests/configs/verify_byte_bpe.yaml").read_text(
        encoding="utf-8",
    )
    configuration_path.write_text(
        configuration_text.replace(
            "output_dir: runs/verify_byte_bpe",
            f"output_dir: {str(run_directory).replace(chr(92), '/')}",
        ),
        encoding="utf-8",
    )

    exit_code = run_plan(configuration_paths=(configuration_path,), max_parallel_jobs=1, gpus=None)
    tokenizer_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="tokenizer",
    )
    packed_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="packed_dataset",
    )
    tokenizer_manifest = json.loads(
        (tokenizer_artifact_directory / "manifest.json").read_text(encoding="utf-8"),
    )
    packed_index = json.loads((packed_artifact_directory / "index.json").read_text("utf-8"))

    assert exit_code == 0
    assert tokenizer_manifest["metrics"]["vocabulary_size"] == 260
    assert tokenizer_manifest["metrics"]["merge_count"] == 1
    assert tokenizer_manifest["metrics"]["training_documents"] == 1
    assert packed_index["total_sequences"] >= 1
