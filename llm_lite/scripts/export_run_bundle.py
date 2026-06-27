"""Export a compact, download-friendly bundle from a pipeline run."""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm_lite.orchestration.models import (
    ArtifactFingerprint,
    ArtifactStorePaths,
    RunManifest,
)
from llm_lite.pipeline.stage import StageName


@dataclass(frozen=True)
class BundleEntry:
    source_path: Path
    archive_path: Path


def collect_bundle_entries(
    *,
    run_directory: Path,
    include_all_checkpoints: bool = False,
    include_tensorboard: bool = False,
) -> list[BundleEntry]:
    run_directory = run_directory.resolve()
    entries: list[BundleEntry] = []
    run_manifest = _read_run_manifest(run_directory=run_directory)
    artifact_store_paths = ArtifactStorePaths.for_run_directory(run_directory=run_directory)

    _add_existing_run_files(
        entries=entries,
        run_directory=run_directory,
        relative_paths=(
            Path("resolved_config.json"),
            Path("run_manifest.json"),
            Path("pipeline.jsonl"),
            Path("performance.jsonl"),
        ),
    )
    _add_stage_file(
        entries=entries,
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        stage_name=StageName.EVALUATION,
        relative_path=Path("manifest.json"),
    )
    _add_stage_file(
        entries=entries,
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        stage_name=StageName.EVALUATION,
        relative_path=Path("report.json"),
    )
    _add_stage_file(
        entries=entries,
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        stage_name=StageName.PRETRAINING,
        relative_path=Path("manifest.json"),
    )
    _add_stage_file(
        entries=entries,
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        stage_name=StageName.PRETRAINING,
        relative_path=Path("metrics.jsonl"),
    )
    _add_stage_file(
        entries=entries,
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        stage_name=StageName.PRETRAINING,
        relative_path=Path("training_evaluations.jsonl"),
    )
    _add_stage_tree(
        entries=entries,
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        stage_name=StageName.TOKENIZER,
        relative_directory=Path("."),
    )

    checkpoint_directory = (
        _stage_artifact_directory(
            run_manifest=run_manifest,
            artifact_store_paths=artifact_store_paths,
            stage_name=StageName.PRETRAINING,
        )
        / "checkpoints"
    )
    if include_all_checkpoints:
        _add_stage_tree(
            entries=entries,
            run_manifest=run_manifest,
            artifact_store_paths=artifact_store_paths,
            stage_name=StageName.PRETRAINING,
            relative_directory=Path("checkpoints"),
        )
    else:
        _add_latest_checkpoint(entries=entries, checkpoint_directory=checkpoint_directory)

    if include_tensorboard:
        _add_existing_run_tree(
            entries=entries,
            run_directory=run_directory,
            relative_directory=Path("tensorboard"),
        )
        for stage_name in StageName:
            _add_stage_tree(
                entries=entries,
                run_manifest=run_manifest,
                artifact_store_paths=artifact_store_paths,
                stage_name=stage_name,
                relative_directory=Path("tensorboard"),
            )

    return sorted(_deduplicate_entries(entries), key=lambda entry: entry.archive_path.as_posix())


def write_bundle(
    *,
    run_directory: Path,
    output_path: Path,
    include_all_checkpoints: bool = False,
    include_tensorboard: bool = False,
) -> None:
    run_directory = run_directory.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = collect_bundle_entries(
        run_directory=run_directory,
        include_all_checkpoints=include_all_checkpoints,
        include_tensorboard=include_tensorboard,
    )
    manifest = {
        "created_at": _utc_now(),
        "source_run_directory": str(run_directory),
        "include_all_checkpoints": include_all_checkpoints,
        "include_tensorboard": include_tensorboard,
        "file_count": len(entries),
        "files": [entry.archive_path.as_posix() for entry in entries],
    }
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        )
        for entry in entries:
            archive.write(entry.source_path, entry.archive_path.as_posix())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-all-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include every pretraining checkpoint instead of only the latest one.",
    )
    parser.add_argument(
        "--include-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include TensorBoard event files.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    write_bundle(
        run_directory=args.run_dir,
        output_path=args.output,
        include_all_checkpoints=args.include_all_checkpoints,
        include_tensorboard=args.include_tensorboard,
    )
    print(f"wrote {args.output}")
    return 0


def _add_latest_checkpoint(entries: list[BundleEntry], checkpoint_directory: Path) -> None:
    latest_full_checkpoint = checkpoint_directory / "latest.pt"
    if latest_full_checkpoint.exists():
        _add_artifact_file(
            entries=entries,
            artifact_directory=checkpoint_directory.parent,
            path=latest_full_checkpoint,
            stage_name=StageName.PRETRAINING,
        )
        return

    latest_sharded_manifest = checkpoint_directory / "latest.json"
    if not latest_sharded_manifest.exists():
        return
    _add_artifact_file(
        entries=entries,
        artifact_directory=checkpoint_directory.parent,
        path=latest_sharded_manifest,
        stage_name=StageName.PRETRAINING,
    )
    latest_data = json.loads(latest_sharded_manifest.read_text(encoding="utf-8"))
    checkpoint_name = str(latest_data["checkpoint"])
    checkpoint_path = checkpoint_directory / checkpoint_name
    _add_artifact_tree(
        entries=entries,
        artifact_directory=checkpoint_directory.parent,
        directory=checkpoint_path,
        stage_name=StageName.PRETRAINING,
    )


def _add_existing_run_files(
    entries: list[BundleEntry],
    run_directory: Path,
    relative_paths: tuple[Path, ...],
) -> None:
    for relative_path in relative_paths:
        path = run_directory / relative_path
        if path.exists() and path.is_file():
            entries.append(
                BundleEntry(
                    source_path=path.resolve(),
                    archive_path=relative_path,
                ),
            )


def _add_existing_run_tree(
    entries: list[BundleEntry],
    run_directory: Path,
    relative_directory: Path,
) -> None:
    directory = run_directory / relative_directory
    if not directory.exists() or not directory.is_dir():
        return
    for path in directory.rglob("*"):
        if path.is_file():
            entries.append(
                BundleEntry(
                    source_path=path.resolve(),
                    archive_path=path.relative_to(run_directory),
                ),
            )


def _add_stage_file(
    entries: list[BundleEntry],
    run_manifest: RunManifest,
    artifact_store_paths: ArtifactStorePaths,
    stage_name: StageName,
    relative_path: Path,
) -> None:
    artifact_directory = _stage_artifact_directory(
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        stage_name=stage_name,
    )
    path = artifact_directory / relative_path
    if path.exists() and path.is_file():
        _add_artifact_file(
            entries=entries,
            artifact_directory=artifact_directory,
            path=path,
            stage_name=stage_name,
        )


def _add_stage_tree(
    entries: list[BundleEntry],
    run_manifest: RunManifest,
    artifact_store_paths: ArtifactStorePaths,
    stage_name: StageName,
    relative_directory: Path,
) -> None:
    artifact_directory = _stage_artifact_directory(
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        stage_name=stage_name,
    )
    directory = artifact_directory / relative_directory
    if directory.exists() and directory.is_dir():
        _add_artifact_tree(
            entries=entries,
            artifact_directory=artifact_directory,
            directory=directory,
            stage_name=stage_name,
        )


def _add_artifact_tree(
    entries: list[BundleEntry],
    artifact_directory: Path,
    directory: Path,
    stage_name: StageName,
) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            _add_artifact_file(
                entries=entries,
                artifact_directory=artifact_directory,
                path=path,
                stage_name=stage_name,
            )


def _add_artifact_file(
    entries: list[BundleEntry],
    artifact_directory: Path,
    path: Path,
    stage_name: StageName,
) -> None:
    entries.append(
        BundleEntry(
            source_path=path.resolve(),
            archive_path=Path("artifacts")
            / stage_name.value
            / path.relative_to(artifact_directory),
        ),
    )


def _stage_artifact_directory(
    run_manifest: RunManifest,
    artifact_store_paths: ArtifactStorePaths,
    stage_name: StageName,
) -> Path:
    fingerprint_value = run_manifest.artifacts.get(stage_name.value)
    if fingerprint_value is None:
        return Path("__missing_artifact__")
    return artifact_store_paths.artifact_directory(
        stage_name=stage_name,
        fingerprint=ArtifactFingerprint(value=fingerprint_value),
    )


def _read_run_manifest(run_directory: Path) -> RunManifest:
    run_manifest_path = run_directory / "run_manifest.json"
    if not run_manifest_path.exists():
        return RunManifest(experiment=run_directory.name, artifacts={})
    return RunManifest.model_validate_json(run_manifest_path.read_text(encoding="utf-8"))


def _deduplicate_entries(entries: list[BundleEntry]) -> list[BundleEntry]:
    deduplicated: dict[Path, BundleEntry] = {}
    for entry in entries:
        deduplicated[entry.archive_path] = entry
    return list(deduplicated.values())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
