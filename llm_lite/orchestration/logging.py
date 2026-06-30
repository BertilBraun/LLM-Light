from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, ConfigDict

from llm_lite.pipeline.stage import StageName


class OrchestrationEventType(str, Enum):
    RUN_QUEUED = "run_queued"
    STAGE_CACHE_HIT = "stage_cache_hit"
    STAGE_WAITING_FOR_ARTIFACT_LOCK = "stage_waiting_for_artifact_lock"
    STAGE_WAITING_FOR_GPU_ALLOCATION = "stage_waiting_for_gpu_allocation"
    STAGE_STARTED = "stage_started"
    SUBPROCESS_COMMAND_STARTED = "subprocess_command_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    PLAN_COMPLETED = "plan_completed"
    PLAN_FAILED = "plan_failed"


class OrchestrationEventRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp_utc: str
    event_type: OrchestrationEventType
    experiment_name: str
    run_directory: Path
    stage_name: StageName | None = None
    artifact_fingerprint: str | None = None
    resolved_config_path: Path | None = None
    message: str
    command: tuple[str, ...] | None = None
    planned_run_count: int | None = None
    completed_run_count: int | None = None
    failed_run_count: int | None = None
    cancelled_run_count: int | None = None


class OrchestrationEventLogger:
    def __init__(self, events_path: Path) -> None:
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, event_record: OrchestrationEventRecord) -> None:
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as events_file:
                events_file.write(event_record.model_dump_json() + "\n")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
