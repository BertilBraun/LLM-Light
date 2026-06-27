import json
from pathlib import Path


def stage_artifact_directory(run_directory: Path, stage_name: str) -> Path:
    run_manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    fingerprint = str(run_manifest["artifacts"][stage_name])
    return run_directory.parent / "artifact_store" / stage_name / fingerprint.replace(":", "_")
