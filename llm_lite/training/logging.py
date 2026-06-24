import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

from llm_lite.model.routing import RouterUsageSummary


class TrainingScalar(str, Enum):
    LOSS = "train/loss"
    LEARNING_RATE = "train/learning_rate"
    GRADIENT_NORM = "train/gradient_norm"
    TOKENS_PER_SECOND = "train/tokens_per_second"
    DISTRIBUTED_WORLD_SIZE = "distributed/world_size"
    DISTRIBUTED_GLOBAL_TOKENS_PER_SECOND = "distributed/global_tokens_per_second"
    DISTRIBUTED_RANK_TOKENS_PER_SECOND = "distributed/rank_tokens_per_second"
    DISTRIBUTED_CHECKPOINT_TIME = "distributed/checkpoint_time"


class TrainingMetricRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    loss: float
    learning_rate: float
    gradient_norm: float
    elapsed_seconds: float
    tokens_per_second: float
    distributed_world_size: int | None = None
    distributed_global_tokens_per_second: float | None = None
    distributed_rank_tokens_per_second: float | None = None
    distributed_checkpoint_time: float | None = None
    distributed_strategy: str | None = None


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
        print(
            f"[{datetime.now().strftime('%H:%M')}] [train] "
            f"step={metric_record.step} "
            f"loss={metric_record.loss:.6f} "
            f"learning_rate={metric_record.learning_rate:.6g} "
            f"gradient_norm={metric_record.gradient_norm:.4f} "
            f"tokens_per_second={metric_record.tokens_per_second:.2f}",
            flush=True,
        )
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
        if metric_record.distributed_world_size is not None:
            self.summary_writer.add_scalar(
                TrainingScalar.DISTRIBUTED_WORLD_SIZE.value,
                metric_record.distributed_world_size,
                metric_record.step,
            )
        if metric_record.distributed_global_tokens_per_second is not None:
            self.summary_writer.add_scalar(
                TrainingScalar.DISTRIBUTED_GLOBAL_TOKENS_PER_SECOND.value,
                metric_record.distributed_global_tokens_per_second,
                metric_record.step,
            )
        if metric_record.distributed_rank_tokens_per_second is not None:
            self.summary_writer.add_scalar(
                TrainingScalar.DISTRIBUTED_RANK_TOKENS_PER_SECOND.value,
                metric_record.distributed_rank_tokens_per_second,
                metric_record.step,
            )
        if metric_record.distributed_checkpoint_time is not None:
            self.summary_writer.add_scalar(
                TrainingScalar.DISTRIBUTED_CHECKPOINT_TIME.value,
                metric_record.distributed_checkpoint_time,
                metric_record.step,
            )
        self.summary_writer.flush()

    def write_router_usage(
        self,
        step: int,
        router_usage_summaries: tuple[RouterUsageSummary, ...],
    ) -> None:
        for router_usage_summary in router_usage_summaries:
            histogram_values = _expert_index_histogram_values(
                expert_counts=router_usage_summary.expert_counts,
            )
            tag_prefix = f"moe/router_layer_{router_usage_summary.layer_index:02d}"
            self.summary_writer.add_histogram(
                f"{tag_prefix}/selected_expert",
                histogram_values,
                step,
            )
            total_count = max(float(router_usage_summary.expert_counts.sum().item()), 1.0)
            for expert_index, expert_count in enumerate(router_usage_summary.expert_counts):
                self.summary_writer.add_scalar(
                    f"{tag_prefix}/expert_{expert_index:02d}_fraction",
                    float(expert_count.item()) / total_count,
                    step,
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
    distributed_world_size: int | None = None,
    distributed_global_tokens_per_second: float | None = None,
    distributed_rank_tokens_per_second: float | None = None,
    distributed_checkpoint_time: float | None = None,
    distributed_strategy: str | None = None,
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
        distributed_world_size=distributed_world_size,
        distributed_global_tokens_per_second=distributed_global_tokens_per_second,
        distributed_rank_tokens_per_second=distributed_rank_tokens_per_second,
        distributed_checkpoint_time=distributed_checkpoint_time,
        distributed_strategy=distributed_strategy,
    )


def _expert_index_histogram_values(expert_counts: torch.Tensor) -> torch.Tensor:
    expert_indices = torch.arange(expert_counts.shape[0], dtype=torch.float32)
    repeated_counts = expert_counts.to(dtype=torch.long)
    if int(repeated_counts.sum().item()) == 0:
        return expert_indices
    return torch.repeat_interleave(expert_indices, repeated_counts)
