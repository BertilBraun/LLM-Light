"""Export a compact, download-friendly bundle from a pipeline run."""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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

    _add_existing_files(
        entries=entries,
        run_directory=run_directory,
        relative_paths=[
            Path('resolved_config.json'),
            Path('pipeline.jsonl'),
            Path('performance.jsonl'),
            Path('artifacts/evaluation/manifest.json'),
            Path('artifacts/evaluation/report.json'),
            Path('artifacts/pretraining/manifest.json'),
            Path('artifacts/pretraining/metrics.jsonl'),
            Path('artifacts/pretraining/training_evaluations.jsonl'),
        ],
    )
    _add_existing_tree(
        entries=entries,
        run_directory=run_directory,
        relative_directory=Path('artifacts/tokenizer'),
    )

    checkpoint_directory = run_directory / 'artifacts' / 'pretraining' / 'checkpoints'
    if include_all_checkpoints:
        _add_existing_tree(
            entries=entries,
            run_directory=run_directory,
            relative_directory=Path('artifacts/pretraining/checkpoints'),
        )
    else:
        _add_latest_checkpoint(
            entries=entries,
            run_directory=run_directory,
            checkpoint_directory=checkpoint_directory,
        )

    if include_tensorboard:
        _add_existing_tree(
            entries=entries,
            run_directory=run_directory,
            relative_directory=Path('tensorboard'),
        )
        _add_existing_tree(
            entries=entries,
            run_directory=run_directory,
            relative_directory=Path('artifacts/pretraining/tensorboard'),
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
        'created_at': _utc_now(),
        'source_run_directory': str(run_directory),
        'include_all_checkpoints': include_all_checkpoints,
        'include_tensorboard': include_tensorboard,
        'file_count': len(entries),
        'files': [entry.archive_path.as_posix() for entry in entries],
    }
    with zipfile.ZipFile(output_path, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            'bundle_manifest.json',
            json.dumps(manifest, sort_keys=True, indent=2) + '\n',
        )
        for entry in entries:
            archive.write(entry.source_path, entry.archive_path.as_posix())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--include-all-checkpoints',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Include every pretraining checkpoint instead of only the latest one.',
    )
    parser.add_argument(
        '--include-tensorboard',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Include TensorBoard event files.',
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
    print(f'wrote {args.output}')
    return 0


def _add_latest_checkpoint(
    *,
    entries: list[BundleEntry],
    run_directory: Path,
    checkpoint_directory: Path,
) -> None:
    latest_full_checkpoint = checkpoint_directory / 'latest.pt'
    if latest_full_checkpoint.exists():
        _add_file(entries=entries, run_directory=run_directory, path=latest_full_checkpoint)
        return

    latest_sharded_manifest = checkpoint_directory / 'latest.json'
    if not latest_sharded_manifest.exists():
        return
    _add_file(entries=entries, run_directory=run_directory, path=latest_sharded_manifest)
    latest_data = json.loads(latest_sharded_manifest.read_text(encoding='utf-8'))
    checkpoint_name = str(latest_data['checkpoint'])
    checkpoint_path = checkpoint_directory / checkpoint_name
    _add_tree(entries=entries, run_directory=run_directory, directory=checkpoint_path)


def _add_existing_files(
    *,
    entries: list[BundleEntry],
    run_directory: Path,
    relative_paths: list[Path],
) -> None:
    for relative_path in relative_paths:
        path = run_directory / relative_path
        if path.exists() and path.is_file():
            _add_file(entries=entries, run_directory=run_directory, path=path)


def _add_existing_tree(
    *,
    entries: list[BundleEntry],
    run_directory: Path,
    relative_directory: Path,
) -> None:
    directory = run_directory / relative_directory
    if directory.exists() and directory.is_dir():
        _add_tree(entries=entries, run_directory=run_directory, directory=directory)


def _add_tree(*, entries: list[BundleEntry], run_directory: Path, directory: Path) -> None:
    for path in directory.rglob('*'):
        if path.is_file():
            _add_file(entries=entries, run_directory=run_directory, path=path)


def _add_file(*, entries: list[BundleEntry], run_directory: Path, path: Path) -> None:
    source_path = path.resolve()
    archive_path = _relative_to_run(path=source_path, run_directory=run_directory)
    entries.append(BundleEntry(source_path=source_path, archive_path=archive_path))


def _relative_to_run(*, path: Path, run_directory: Path) -> Path:
    try:
        return path.relative_to(run_directory)
    except ValueError as error:
        raise ValueError(f'Refusing to export path outside run directory: {path}') from error


def _deduplicate_entries(entries: list[BundleEntry]) -> list[BundleEntry]:
    deduplicated: dict[Path, BundleEntry] = {}
    for entry in entries:
        deduplicated[entry.archive_path] = entry
    return list(deduplicated.values())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


if __name__ == '__main__':
    raise SystemExit(main())
