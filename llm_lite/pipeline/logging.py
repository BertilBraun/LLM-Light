from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llm_lite.pipeline.stage import StageName


class PipelineEventType(str, Enum):
    REVIEW = "review"
    STAGE_START = "stage_start"
    STAGE_SKIP = "stage_skip"
    STAGE_COMPLETE = "stage_complete"


class PipelineEventRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: PipelineEventType
    stage_name: StageName
    message: str


class PipelineEventLogger:
    def __init__(self, run_directory: Path) -> None:
        self.events_path = run_directory / "pipeline.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_record: PipelineEventRecord) -> None:
        with self.events_path.open("a", encoding="utf-8") as events_file:
            events_file.write(event_record.model_dump_json() + "\n")
