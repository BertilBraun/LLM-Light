from enum import Enum

from pydantic import BaseModel, ConfigDict


class ArtifactStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_name: str
    fingerprint: str
    artifact_version: int
    status: ArtifactStatus
    created_at: str
    completed_at: str | None = None
    configuration_hash: str
    contract_version: int
    parents: dict[str, str]
    files: dict[str, str]
    metrics: dict[str, int | float | str | bool]
