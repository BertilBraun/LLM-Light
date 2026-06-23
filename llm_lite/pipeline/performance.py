import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel, ConfigDict, Field


class StagePerformanceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage_name: str
    started_at: str
    ended_at: str
    duration_seconds: float = Field(ge=0.0)
    worker_count: int = Field(ge=0)
    metrics: dict[str, int | float | str | bool]


@dataclass(frozen=True)
class StagePerformanceTiming:
    stage_name: str
    started_at: str
    ended_at: str
    duration_seconds: float


class StagePerformanceTimer:
    def __init__(self, stage_name: str) -> None:
        self.stage_name = stage_name
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.start_seconds: float | None = None
        self.duration_seconds: float | None = None

    def __enter__(self) -> "StagePerformanceTimer":
        self.started_at = _utc_now()
        self.start_seconds = time.perf_counter()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.start_seconds is None:
            raise ValueError("Stage performance timer was not started.")
        self.duration_seconds = time.perf_counter() - self.start_seconds
        self.ended_at = _utc_now()

    def timing(self) -> StagePerformanceTiming:
        if self.started_at is None or self.ended_at is None or self.duration_seconds is None:
            raise ValueError("Stage performance timer has not completed.")
        return StagePerformanceTiming(
            stage_name=self.stage_name,
            started_at=self.started_at,
            ended_at=self.ended_at,
            duration_seconds=self.duration_seconds,
        )


class PipelinePerformanceLogger:
    def __init__(self, run_directory: Path) -> None:
        self.performance_path = run_directory / "performance.jsonl"
        run_directory.mkdir(parents=True, exist_ok=True)

    def measure_stage(self, stage_name: str) -> StagePerformanceTimer:
        return StagePerformanceTimer(stage_name=stage_name)

    def write_stage_timing(
        self,
        timing: StagePerformanceTiming,
        metrics: dict[str, int | float | str | bool],
    ) -> None:
        self.write(
            performance_record=StagePerformanceRecord(
                stage_name=timing.stage_name,
                started_at=timing.started_at,
                ended_at=timing.ended_at,
                duration_seconds=timing.duration_seconds,
                worker_count=_worker_count(metrics=metrics),
                metrics=metrics,
            ),
        )

    def write(self, performance_record: StagePerformanceRecord) -> None:
        with self.performance_path.open("a", encoding="utf-8") as performance_file:
            performance_file.write(performance_record.model_dump_json())
            performance_file.write("\n")


def _worker_count(metrics: dict[str, int | float | str | bool]) -> int:
    worker_count_metric = metrics.get("workers")
    match worker_count_metric:
        case int():
            return worker_count_metric
        case _:
            return 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
