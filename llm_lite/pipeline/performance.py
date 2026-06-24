import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel, ConfigDict, Field
from torch.utils.tensorboard import SummaryWriter

PIPELINE_TENSORBOARD_DIRECTORY_NAME = "tensorboard"


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
        self.tensorboard_directory = run_directory / PIPELINE_TENSORBOARD_DIRECTORY_NAME
        self.stage_index = 0
        run_directory.mkdir(parents=True, exist_ok=True)
        self.tensorboard_directory.mkdir(parents=True, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=str(self.tensorboard_directory))

    def measure_stage(self, stage_name: str) -> StagePerformanceTimer:
        return StagePerformanceTimer(stage_name=stage_name)

    def write_stage_timing(
        self,
        timing: StagePerformanceTiming,
        metrics: dict[str, int | float | str | bool],
    ) -> None:
        worker_count = _worker_count(metrics=metrics)
        self.write(
            performance_record=StagePerformanceRecord(
                stage_name=timing.stage_name,
                started_at=timing.started_at,
                ended_at=timing.ended_at,
                duration_seconds=timing.duration_seconds,
                worker_count=worker_count,
                metrics=metrics,
            ),
        )
        self._write_tensorboard_text(
            timing=timing,
            worker_count=worker_count,
            metrics=metrics,
        )
        self.stage_index += 1

    def write(self, performance_record: StagePerformanceRecord) -> None:
        with self.performance_path.open("a", encoding="utf-8") as performance_file:
            performance_file.write(performance_record.model_dump_json())
            performance_file.write("\n")

    def close(self) -> None:
        self.summary_writer.close()

    def _write_tensorboard_text(
        self,
        timing: StagePerformanceTiming,
        worker_count: int,
        metrics: dict[str, int | float | str | bool],
    ) -> None:
        self.summary_writer.add_text(
            f"pipeline/{timing.stage_name}",
            _stage_summary_text(
                timing=timing,
                worker_count=worker_count,
                metrics=metrics,
            ),
            self.stage_index,
        )
        self.summary_writer.flush()


def _worker_count(metrics: dict[str, int | float | str | bool]) -> int:
    worker_count_metric = metrics.get("workers")
    match worker_count_metric:
        case int():
            return worker_count_metric
        case _:
            return 1


def _stage_summary_text(
    timing: StagePerformanceTiming,
    worker_count: int,
    metrics: dict[str, int | float | str | bool],
) -> str:
    rows = [
        ("started_at", timing.started_at),
        ("ended_at", timing.ended_at),
        ("duration_seconds", f"{timing.duration_seconds:.6f}"),
        ("worker_count", str(worker_count)),
    ]
    rows.extend(
        (metric_name, _format_metric_value(metric_value))
        for metric_name, metric_value in sorted(metrics.items())
    )
    row_text = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return f"| metric | value |\n| --- | --- |\n{row_text}"


def _format_metric_value(metric_value: int | float | str | bool) -> str:
    match metric_value:
        case bool():
            return str(metric_value).lower()
        case float():
            return f"{metric_value:.6g}"
        case _:
            return str(metric_value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
