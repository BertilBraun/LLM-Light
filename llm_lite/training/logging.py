import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

from llm_lite.model.routing import RouterUsageSummary
from llm_lite.pipeline.progress import console_log
from llm_lite.pipeline.tensorboard import configured_run_tensorboard_directory


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


@dataclass(frozen=True)
class RouterLayerMetricSummary:
    layer_index: int
    usage_mean: float
    usage_std: float
    usage_min: float
    usage_max: float
    entropy: float
    imbalance: float
    dominance: float


class TrainingMetricLogger:
    def __init__(self, artifact_directory: Path) -> None:
        self.metrics_path = artifact_directory / "metrics.jsonl"
        self.tensorboard_directory = artifact_directory / "tensorboard"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.tensorboard_directory.mkdir(parents=True, exist_ok=True)
        run_tensorboard_directory = configured_run_tensorboard_directory()
        if run_tensorboard_directory is not None:
            run_tensorboard_directory.mkdir(parents=True, exist_ok=True)
            self.summary_writers = (
                SummaryWriter(log_dir=str(self.tensorboard_directory)),
                SummaryWriter(log_dir=str(run_tensorboard_directory)),
            )
        else:
            self.summary_writers = (SummaryWriter(log_dir=str(self.tensorboard_directory)),)

    def write(self, metric_record: TrainingMetricRecord) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(metric_record.model_dump_json() + "\n")
        message = (
            "[train] "
            f"step={metric_record.step} "
            f"loss={metric_record.loss:.6f} "
            f"learning_rate={metric_record.learning_rate:.6g} "
            f"gradient_norm={metric_record.gradient_norm:.4f} "
        )
        if metric_record.distributed_world_size is None:
            message += f"tokens_per_second={metric_record.tokens_per_second:.2f}"
        else:
            message += (
                f"rank_tokens_per_second={metric_record.tokens_per_second:.2f} "
                f"global_tokens_per_second="
                f"{metric_record.distributed_global_tokens_per_second or 0.0:.2f} "
                f"world_size={metric_record.distributed_world_size}"
            )
        console_log(message)
        self._add_scalar(TrainingScalar.LOSS.value, metric_record.loss, metric_record.step)
        self._add_scalar(
            TrainingScalar.LEARNING_RATE.value,
            metric_record.learning_rate,
            metric_record.step,
        )
        self._add_scalar(
            TrainingScalar.GRADIENT_NORM.value,
            metric_record.gradient_norm,
            metric_record.step,
        )
        self._add_scalar(
            TrainingScalar.TOKENS_PER_SECOND.value,
            metric_record.tokens_per_second,
            metric_record.step,
        )
        if metric_record.distributed_world_size is not None:
            self._add_scalar(
                TrainingScalar.DISTRIBUTED_WORLD_SIZE.value,
                metric_record.distributed_world_size,
                metric_record.step,
            )
        if metric_record.distributed_global_tokens_per_second is not None:
            self._add_scalar(
                TrainingScalar.DISTRIBUTED_GLOBAL_TOKENS_PER_SECOND.value,
                metric_record.distributed_global_tokens_per_second,
                metric_record.step,
            )
        if metric_record.distributed_rank_tokens_per_second is not None:
            self._add_scalar(
                TrainingScalar.DISTRIBUTED_RANK_TOKENS_PER_SECOND.value,
                metric_record.distributed_rank_tokens_per_second,
                metric_record.step,
            )
        if metric_record.distributed_checkpoint_time is not None:
            self._add_scalar(
                TrainingScalar.DISTRIBUTED_CHECKPOINT_TIME.value,
                metric_record.distributed_checkpoint_time,
                metric_record.step,
            )
        self._flush()

    def write_router_usage(
        self,
        step: int,
        router_usage_summaries: tuple[RouterUsageSummary, ...],
    ) -> None:
        layer_metric_summaries: list[RouterLayerMetricSummary] = []
        for router_usage_summary in router_usage_summaries:
            histogram_values = _expert_index_histogram_values(
                expert_counts=router_usage_summary.expert_counts,
            )
            tag_prefix = f"moe/router_layer_{router_usage_summary.layer_index:02d}"
            self._add_histogram(
                f"{tag_prefix}/selected_expert",
                histogram_values,
                step,
            )
            layer_metric_summary = _router_layer_metric_summary(
                router_usage_summary=router_usage_summary,
            )
            layer_metric_summaries.append(layer_metric_summary)
            self._add_scalar(
                f"{tag_prefix}/usage_mean",
                layer_metric_summary.usage_mean,
                step,
            )
            self._add_scalar(
                f"{tag_prefix}/usage_std",
                layer_metric_summary.usage_std,
                step,
            )
            self._add_scalar(
                f"{tag_prefix}/usage_min",
                layer_metric_summary.usage_min,
                step,
            )
            self._add_scalar(
                f"{tag_prefix}/usage_max",
                layer_metric_summary.usage_max,
                step,
            )
            self._add_scalar(
                f"{tag_prefix}/entropy",
                layer_metric_summary.entropy,
                step,
            )
            self._add_scalar(
                f"{tag_prefix}/imbalance",
                layer_metric_summary.imbalance,
                step,
            )
            self._add_scalar(
                f"{tag_prefix}/dominance",
                layer_metric_summary.dominance,
                step,
            )
        if len(layer_metric_summaries) > 0:
            self._add_scalar(
                "moe/summary/worst_layer_imbalance",
                max(summary.imbalance for summary in layer_metric_summaries),
                step,
            )
            self._add_scalar(
                "moe/summary/worst_layer_dominance",
                max(summary.dominance for summary in layer_metric_summaries),
                step,
            )
            self._add_scalar(
                "moe/summary/worst_layer_entropy",
                min(summary.entropy for summary in layer_metric_summaries),
                step,
            )
        self._flush()

    def close(self) -> None:
        for summary_writer in self.summary_writers:
            summary_writer.close()

    def _add_scalar(self, tag: str, scalar_value: float, step: int) -> None:
        for summary_writer in self.summary_writers:
            summary_writer.add_scalar(tag, scalar_value, step)

    def _add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        for summary_writer in self.summary_writers:
            summary_writer.add_histogram(tag, values, step)

    def _flush(self) -> None:
        for summary_writer in self.summary_writers:
            summary_writer.flush()


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


def _router_layer_metric_summary(
    router_usage_summary: RouterUsageSummary,
) -> RouterLayerMetricSummary:
    expert_counts = router_usage_summary.expert_counts.to(dtype=torch.float32)
    total_count = float(expert_counts.sum().item())
    if total_count == 0.0:
        usage_fractions = torch.zeros_like(expert_counts)
    else:
        usage_fractions = expert_counts / total_count
    usage_mean = float(usage_fractions.mean().item())
    usage_std = float(usage_fractions.std(unbiased=False).item())
    usage_min = float(usage_fractions.min().item())
    usage_max = float(usage_fractions.max().item())
    return RouterLayerMetricSummary(
        layer_index=router_usage_summary.layer_index,
        usage_mean=usage_mean,
        usage_std=usage_std,
        usage_min=usage_min,
        usage_max=usage_max,
        entropy=_normalized_entropy(probabilities=usage_fractions),
        imbalance=usage_max - usage_min,
        dominance=usage_max,
    )


def _normalized_entropy(probabilities: torch.Tensor) -> float:
    if probabilities.numel() <= 1:
        return 0.0
    nonzero_probabilities = probabilities[probabilities > 0.0]
    if nonzero_probabilities.numel() == 0:
        return 0.0
    entropy = -torch.sum(nonzero_probabilities * torch.log(nonzero_probabilities))
    normalizer = torch.log(torch.tensor(float(probabilities.numel())))
    return float((entropy / normalizer).item())
