import json
from pathlib import Path

from llm_lite.pipeline.runner import run_pipeline


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

    exit_code = run_pipeline(
        configuration_path=configuration_path,
        dry_run=False,
        force_stages=(),
    )
    tokenizer_manifest = json.loads(
        (run_directory / "artifacts" / "tokenizer" / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    packed_index = json.loads(
        (run_directory / "artifacts" / "packed_dataset" / "index.json").read_text(
            encoding="utf-8",
        ),
    )

    assert exit_code == 0
    assert tokenizer_manifest["metrics"]["vocabulary_size"] == 260
    assert tokenizer_manifest["metrics"]["merge_count"] == 1
    assert tokenizer_manifest["metrics"]["training_documents"] == 1
    assert packed_index["total_sequences"] >= 1
