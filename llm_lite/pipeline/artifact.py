from enum import Enum

from pydantic import BaseModel, ConfigDict


class ArtifactStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    FAILED = "failed"


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_type: str
    artifact_version: int
    status: ArtifactStatus
    created_at: str
    configuration_hash: str
    implementation_version: str
    parents: dict[str, str]
    files: dict[str, str]
    metrics: dict[str, int | float | str | bool]
