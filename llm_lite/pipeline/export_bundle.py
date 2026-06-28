"""Build compact, download-friendly bundles from completed runs."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

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
    artifact_stage: str | None = None
    artifact_fingerprint: str | None = None


class BundleFileRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    archive_path: str
    source_path: str
    artifact_stage: str | None = None
    artifact_fingerprint: str | None = None


class BundleArtifactRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage_name: str
    fingerprint: str
    files: tuple[str, ...]


class BundleManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: str
    source_run_directory: str
    experiment: str
    run_artifacts: dict[str, str]
    include_all_checkpoints: bool
    include_tensorboard: bool
    file_count: int
    files: tuple[BundleFileRecord, ...]
    artifacts: tuple[BundleArtifactRecord, ...]


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
    _add_all_stage_files(
        entries=entries,
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        relative_path=Path("manifest.json"),
    )
    _add_all_stage_files(
        entries=entries,
        run_manifest=run_manifest,
        artifact_store_paths=artifact_store_paths,
        relative_path=Path("job.log"),
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

    if include_all_checkpoints:
        for checkpoint_stage_name in (StageName.PRETRAINING, StageName.POST_TRAINING):
            _add_stage_tree(
                entries=entries,
                run_manifest=run_manifest,
                artifact_store_paths=artifact_store_paths,
                stage_name=checkpoint_stage_name,
                relative_directory=Path("checkpoints"),
            )
    else:
        checkpoint_stage_name = _latest_checkpoint_stage(
            run_manifest=run_manifest,
            artifact_store_paths=artifact_store_paths,
        )
        checkpoint_directory = (
            _stage_artifact_directory(
                run_manifest=run_manifest,
                artifact_store_paths=artifact_store_paths,
                stage_name=checkpoint_stage_name,
            )
            / "checkpoints"
        )
        _add_latest_checkpoint(
            entries=entries,
            checkpoint_directory=checkpoint_directory,
            artifact_fingerprint=_stage_fingerprint(
                run_manifest=run_manifest,
                stage_name=checkpoint_stage_name,
            ),
            stage_name=checkpoint_stage_name,
        )

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
    manifest_output_path: Path | None = None,
    include_all_checkpoints: bool = False,
    include_tensorboard: bool = False,
) -> BundleManifest:
    run_directory = run_directory.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = collect_bundle_entries(
        run_directory=run_directory,
        include_all_checkpoints=include_all_checkpoints,
        include_tensorboard=include_tensorboard,
    )
    run_manifest = _read_run_manifest(run_directory=run_directory)
    manifest = _bundle_manifest(
        run_directory=run_directory,
        run_manifest=run_manifest,
        entries=entries,
        include_all_checkpoints=include_all_checkpoints,
        include_tensorboard=include_tensorboard,
    )
    manifest_text = manifest.model_dump_json(indent=2) + "\n"
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bundle_manifest.json", manifest_text)
        for entry in entries:
            archive.write(entry.source_path, entry.archive_path.as_posix())
    if manifest_output_path is not None:
        manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_output_path.write_text(manifest_text, encoding="utf-8")
    return manifest


def _add_latest_checkpoint(
    entries: list[BundleEntry],
    checkpoint_directory: Path,
    artifact_fingerprint: str | None,
    stage_name: StageName,
) -> None:
    latest_full_checkpoint = checkpoint_directory / "latest.pt"
    if latest_full_checkpoint.exists():
        _add_artifact_file(
            entries=entries,
            artifact_directory=checkpoint_directory.parent,
            path=latest_full_checkpoint,
            stage_name=stage_name,
            artifact_fingerprint=artifact_fingerprint,
        )
        return

    latest_sharded_manifest = checkpoint_directory / "latest.json"
    if not latest_sharded_manifest.exists():
        return
    _add_artifact_file(
        entries=entries,
        artifact_directory=checkpoint_directory.parent,
        path=latest_sharded_manifest,
        stage_name=stage_name,
        artifact_fingerprint=artifact_fingerprint,
    )
    latest_data = json.loads(latest_sharded_manifest.read_text(encoding="utf-8"))
    checkpoint_name = str(latest_data["checkpoint"])
    checkpoint_path = checkpoint_directory / checkpoint_name
    _add_artifact_tree(
        entries=entries,
        artifact_directory=checkpoint_directory.parent,
        directory=checkpoint_path,
        stage_name=stage_name,
        artifact_fingerprint=artifact_fingerprint,
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
            artifact_fingerprint=_stage_fingerprint(
                run_manifest=run_manifest,
                stage_name=stage_name,
            ),
        )


def _add_all_stage_files(
    entries: list[BundleEntry],
    run_manifest: RunManifest,
    artifact_store_paths: ArtifactStorePaths,
    relative_path: Path,
) -> None:
    for stage_name_value in sorted(run_manifest.artifacts):
        stage_name = StageName(stage_name_value)
        if stage_name is StageName.EXPORT:
            continue
        _add_stage_file(
            entries=entries,
            run_manifest=run_manifest,
            artifact_store_paths=artifact_store_paths,
            stage_name=stage_name,
            relative_path=relative_path,
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
            artifact_fingerprint=_stage_fingerprint(
                run_manifest=run_manifest,
                stage_name=stage_name,
            ),
        )


def _add_artifact_tree(
    entries: list[BundleEntry],
    artifact_directory: Path,
    directory: Path,
    stage_name: StageName,
    artifact_fingerprint: str | None = None,
) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            _add_artifact_file(
                entries=entries,
                artifact_directory=artifact_directory,
                path=path,
                stage_name=stage_name,
                artifact_fingerprint=artifact_fingerprint,
            )


def _add_artifact_file(
    entries: list[BundleEntry],
    artifact_directory: Path,
    path: Path,
    stage_name: StageName,
    artifact_fingerprint: str | None = None,
) -> None:
    entries.append(
        BundleEntry(
            source_path=path.resolve(),
            archive_path=Path("artifacts")
            / stage_name.value
            / path.relative_to(artifact_directory),
            artifact_stage=stage_name.value,
            artifact_fingerprint=artifact_fingerprint,
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


def _stage_fingerprint(run_manifest: RunManifest, stage_name: StageName) -> str | None:
    return run_manifest.artifacts.get(stage_name.value)


def _latest_checkpoint_stage(
    run_manifest: RunManifest,
    artifact_store_paths: ArtifactStorePaths,
) -> StageName:
    post_training_directory = (
        _stage_artifact_directory(
            run_manifest=run_manifest,
            artifact_store_paths=artifact_store_paths,
            stage_name=StageName.POST_TRAINING,
        )
        / "checkpoints"
    )
    if (post_training_directory / "latest.pt").exists() or (
        post_training_directory / "latest.json"
    ).exists():
        return StageName.POST_TRAINING
    return StageName.PRETRAINING


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


def _bundle_manifest(
    run_directory: Path,
    run_manifest: RunManifest,
    entries: list[BundleEntry],
    include_all_checkpoints: bool,
    include_tensorboard: bool,
) -> BundleManifest:
    return BundleManifest(
        created_at=_utc_now(),
        source_run_directory=str(run_directory),
        experiment=run_manifest.experiment,
        run_artifacts=run_manifest.artifacts,
        include_all_checkpoints=include_all_checkpoints,
        include_tensorboard=include_tensorboard,
        file_count=len(entries),
        files=tuple(
            BundleFileRecord(
                archive_path=entry.archive_path.as_posix(),
                source_path=str(entry.source_path),
                artifact_stage=entry.artifact_stage,
                artifact_fingerprint=entry.artifact_fingerprint,
            )
            for entry in entries
        ),
        artifacts=_bundle_artifact_records(entries=entries),
    )


def _bundle_artifact_records(entries: list[BundleEntry]) -> tuple[BundleArtifactRecord, ...]:
    grouped_files: dict[tuple[str, str], list[str]] = {}
    for entry in entries:
        if entry.artifact_stage is None or entry.artifact_fingerprint is None:
            continue
        key = (entry.artifact_stage, entry.artifact_fingerprint)
        grouped_files.setdefault(key, []).append(entry.archive_path.as_posix())
    return tuple(
        BundleArtifactRecord(
            stage_name=stage_name,
            fingerprint=fingerprint,
            files=tuple(sorted(files)),
        )
        for (stage_name, fingerprint), files in sorted(grouped_files.items())
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
