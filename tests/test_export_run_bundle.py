import json
import zipfile
from pathlib import Path

from llm_lite.scripts.export_run_bundle import collect_bundle_entries, write_bundle


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
    _write(pretraining_directory / "metrics.jsonl", "{}\n")
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


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
