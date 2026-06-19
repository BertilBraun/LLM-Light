import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter


class TrainingScalar(str, Enum):
    LOSS = "train/loss"
    LEARNING_RATE = "train/learning_rate"
    GRADIENT_NORM = "train/gradient_norm"
    TOKENS_PER_SECOND = "train/tokens_per_second"


class TrainingMetricRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    loss: float
    learning_rate: float
    gradient_norm: float
    elapsed_seconds: float
    tokens_per_second: float


class TrainingMetricLogger:
    def __init__(self, artifact_directory: Path) -> None:
        self.metrics_path = artifact_directory / "metrics.jsonl"
        self.tensorboard_directory = artifact_directory / "tensorboard"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.tensorboard_directory.mkdir(parents=True, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=str(self.tensorboard_directory))

    def write(self, metric_record: TrainingMetricRecord) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(metric_record.model_dump_json() + "\n")
        self.summary_writer.add_scalar(
            TrainingScalar.LOSS.value,
            metric_record.loss,
            metric_record.step,
        )
        self.summary_writer.add_scalar(
            TrainingScalar.LEARNING_RATE.value,
            metric_record.learning_rate,
            metric_record.step,
        )
        self.summary_writer.add_scalar(
            TrainingScalar.GRADIENT_NORM.value,
            metric_record.gradient_norm,
            metric_record.step,
        )
        self.summary_writer.add_scalar(
            TrainingScalar.TOKENS_PER_SECOND.value,
            metric_record.tokens_per_second,
            metric_record.step,
        )
        self.summary_writer.flush()

    def close(self) -> None:
        self.summary_writer.close()


def create_training_metric_record(
    step: int,
    loss: float,
    learning_rate: float,
    gradient_norm: float,
    started_at_seconds: float,
    tokens_processed: int,
) -> TrainingMetricRecord:
    elapsed_seconds = time.perf_counter() - started_at_seconds
    tokens_per_second = tokens_processed / max(elapsed_seconds, 1e-9)
    return TrainingMetricRecord(
        step=step,
        loss=loss,
        learning_rate=learning_rate,
        gradient_norm=gradient_norm,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_per_second,
    )
