import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from llm_lite.pipeline.artifact import ArtifactManifest, ArtifactStatus


class ArtifactRegistry:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory
        self.artifacts_directory = run_directory / "artifacts"
        self.artifacts_directory.mkdir(parents=True, exist_ok=True)

    def artifact_directory(self, artifact_type: str) -> Path:
        artifact_directory = self.artifacts_directory / artifact_type
        artifact_directory.mkdir(parents=True, exist_ok=True)
        return artifact_directory

    def manifest_path(self, artifact_type: str) -> Path:
        return self.artifacts_directory / artifact_type / "manifest.json"

    def read_manifest(self, artifact_type: str) -> ArtifactManifest | None:
        manifest_path = self.manifest_path(artifact_type=artifact_type)
        if not manifest_path.exists():
            return None
        try:
            return ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except ValidationError:
            return None

    def write_running_manifest(
        self,
        artifact_type: str,
        fingerprint: str,
        configuration_hash: str,
        parent_hashes: dict[str, str],
        contract_version: int,
    ) -> None:
        manifest = ArtifactManifest(
            stage_name=artifact_type,
            fingerprint=fingerprint,
            artifact_version=1,
            status=ArtifactStatus.RUNNING,
            created_at=_utc_now(),
            configuration_hash=configuration_hash,
            contract_version=contract_version,
            parents=parent_hashes,
            files={},
            metrics={},
        )
        self._write_manifest_atomically(artifact_type=artifact_type, manifest=manifest)

    def write_complete_manifest(
        self,
        artifact_type: str,
        fingerprint: str,
        configuration_hash: str,
        parent_hashes: dict[str, str],
        contract_version: int,
        files: dict[str, str],
        metrics: dict[str, int | float | str | bool],
    ) -> ArtifactManifest:
        manifest = ArtifactManifest(
            stage_name=artifact_type,
            fingerprint=fingerprint,
            artifact_version=1,
            status=ArtifactStatus.COMPLETE,
            created_at=_utc_now(),
            completed_at=_utc_now(),
            configuration_hash=configuration_hash,
            contract_version=contract_version,
            parents=parent_hashes,
            files=files,
            metrics=metrics,
        )
        self._write_manifest_atomically(artifact_type=artifact_type, manifest=manifest)
        return manifest

    def is_compatible(
        self,
        artifact_type: str,
        fingerprint: str,
        configuration_hash: str,
        parent_hashes: dict[str, str],
        contract_version: int,
    ) -> bool:
        manifest = self.read_manifest(artifact_type=artifact_type)
        if manifest is None:
            return False
        if manifest.status != ArtifactStatus.COMPLETE:
            return False
        if manifest.fingerprint != fingerprint:
            return False
        if manifest.configuration_hash != configuration_hash:
            return False
        if manifest.contract_version != contract_version:
            return False
        if manifest.parents != parent_hashes:
            return False
        artifact_directory = self.artifact_directory(artifact_type=artifact_type)
        return all(
            (artifact_directory / relative_path).exists()
            for relative_path in manifest.files.values()
        )

    def has_matching_fingerprint(
        self,
        artifact_type: str,
        fingerprint: str,
        configuration_hash: str,
        parent_hashes: dict[str, str],
        contract_version: int,
    ) -> bool:
        manifest = self.read_manifest(artifact_type=artifact_type)
        if manifest is None:
            return False
        return (
            manifest.fingerprint == fingerprint
            and manifest.configuration_hash == configuration_hash
            and manifest.parents == parent_hashes
            and manifest.contract_version == contract_version
        )

    def artifact_identifier(self, artifact_type: str) -> str:
        manifest = self.read_manifest(artifact_type=artifact_type)
        if manifest is None:
            raise ValueError(f"No manifest exists for artifact {artifact_type}.")
        return manifest.fingerprint

    def _write_manifest_atomically(self, artifact_type: str, manifest: ArtifactManifest) -> None:
        artifact_directory = self.artifact_directory(artifact_type=artifact_type)
        temporary_path = artifact_directory / "manifest.json.pending"
        final_path = artifact_directory / "manifest.json"
        temporary_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary_path, final_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
