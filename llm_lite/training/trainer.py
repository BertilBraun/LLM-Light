from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, IterableDataset

from llm_lite.config.models import DistributedConfiguration, Precision, TrainingConfiguration
from llm_lite.model.router_usage import collect_router_usage_summaries, reset_router_usage
from llm_lite.pipeline.progress import console_log
from llm_lite.training.checkpoint import (
    finalize_sharded_checkpoint,
    load_latest_checkpoint,
    load_latest_sharded_checkpoint,
    save_checkpoint,
    save_rank_zero_full_checkpoint_bridge,
    save_sharded_rank_checkpoint,
)
from llm_lite.training.data import (
    DistributedDataAssignment,
    InfiniteDataIterator,
    create_training_data_iterator,
)
from llm_lite.training.distributed import (
    DistributedRuntime,
    initialize_distributed_runtime,
    prepare_model_for_distributed_training,
    unwrap_distributed_model,
)
from llm_lite.training.logging import TrainingMetricLogger, create_training_metric_record
from llm_lite.training.objectives import TrainingBatch, TrainingObjectiveRunner


@dataclass(frozen=True)
class TrainingResult:
    final_step: int
    final_loss: float
    checkpoint_path: Path
    resumed_from_step: int
    evaluation_path: Path | None


class TrainingEvaluationCallback(Protocol):
    def __call__(self, step: int, model: nn.Module) -> Path: ...


def train_model(
    model: nn.Module,
    dataset: Dataset[TrainingBatch] | IterableDataset[TrainingBatch],
    training_configuration: TrainingConfiguration,
    artifact_directory: Path,
    seed: int,
    evaluation_callback: TrainingEvaluationCallback | None,
    objective_runner: TrainingObjectiveRunner,
) -> TrainingResult:
    device = _single_process_training_device()
    _apply_training_precision(model=model, precision=training_configuration.precision)
    model = model.to(device)
    console_log(
        "[train] single_process_device "
        f"device={device} precision={training_configuration.precision.value}"
    )
    optimizer = AdamW(
        model.parameters(),
        lr=training_configuration.optimizer.learning_rate,
        weight_decay=training_configuration.optimizer.weight_decay,
    )
    checkpoint_directory = artifact_directory / "checkpoints"
    loaded_checkpoint_step = load_latest_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=model,
        optimizer=optimizer,
    )
    _apply_current_optimizer_configuration(
        optimizer=optimizer,
        training_configuration=training_configuration,
    )
    start_step = 0 if loaded_checkpoint_step is None else loaded_checkpoint_step
    data_iterator = create_training_data_iterator(
        dataset=dataset,
        batch_size_sequences=training_configuration.batch_size_sequences,
        dataloader_configuration=training_configuration.dataloader,
        seed=seed,
    )
    _log_epoch_plan(
        prefix="[train] epoch_plan",
        data_iterator=data_iterator,
        maximum_steps=training_configuration.maximum_steps,
        world_size=1,
    )
    metrics_logger = TrainingMetricLogger(artifact_directory=artifact_directory)
    final_loss = float("inf")
    checkpoint_path = checkpoint_directory / "latest.pt"
    evaluation_path: Path | None = None
    started_at_seconds = perf_counter()
    tokens_processed = 0
    model.train()
    try:
        for step in range(start_step + 1, training_configuration.maximum_steps + 1):
            token_batch = data_iterator.next_batch()
            prepared_batch = objective_runner.prepare_batch(
                batch=token_batch,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = objective_runner.loss(model=model, batch=prepared_batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training_configuration.gradient_clip_norm,
            )
            optimizer.step()
            tokens_processed += _batch_token_count(batch=prepared_batch)
            final_loss = float(loss.detach().cpu().item())
            if step % training_configuration.log_interval_steps == 0:
                metrics_logger.write(
                    metric_record=create_training_metric_record(
                        step=step,
                        loss=final_loss,
                        learning_rate=training_configuration.optimizer.learning_rate,
                        gradient_norm=float(gradient_norm.detach().cpu().item()),
                        started_at_seconds=started_at_seconds,
                        tokens_processed=tokens_processed,
                    ),
                )
                metrics_logger.write_router_usage(
                    step=step,
                    router_usage_summaries=collect_router_usage_summaries(model=model),
                )
                reset_router_usage(model=model)
            training_evaluation_configuration = training_configuration.evaluation
            if (
                training_evaluation_configuration is not None
                and step % training_evaluation_configuration.interval_steps == 0
            ):
                if evaluation_callback is None:
                    raise ValueError("Training evaluation requires an evaluation callback.")
                evaluation_path = evaluation_callback(step=step, model=model)
                model.train()
            if step % training_configuration.checkpoint_interval_steps == 0:
                checkpoint_path = save_checkpoint(
                    checkpoint_directory=checkpoint_directory,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                )
        if not checkpoint_path.exists() or training_configuration.maximum_steps != start_step:
            checkpoint_path = save_checkpoint(
                checkpoint_directory=checkpoint_directory,
                model=model,
                optimizer=optimizer,
                step=training_configuration.maximum_steps,
            )
    finally:
        metrics_logger.close()
    _log_epoch_summary(prefix="[train] epoch_summary", data_iterator=data_iterator)
    return TrainingResult(
        final_step=training_configuration.maximum_steps,
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
        resumed_from_step=start_step,
        evaluation_path=evaluation_path,
    )


def train_model_distributed(
    model: nn.Module,
    dataset: Dataset[TrainingBatch] | IterableDataset[TrainingBatch],
    training_configuration: TrainingConfiguration,
    distributed_configuration: DistributedConfiguration,
    artifact_directory: Path,
    seed: int,
    evaluation_callback: TrainingEvaluationCallback | None,
    model_configuration_hash: str,
    objective_runner: TrainingObjectiveRunner,
) -> TrainingResult:
    distributed_runtime = initialize_distributed_runtime(
        distributed_configuration=distributed_configuration,
        artifact_directory=artifact_directory,
    )
    try:
        return _train_model_distributed_initialized(
            model=model,
            dataset=dataset,
            training_configuration=training_configuration,
            distributed_configuration=distributed_configuration,
            distributed_runtime=distributed_runtime,
            artifact_directory=artifact_directory,
            seed=seed,
            evaluation_callback=evaluation_callback,
            model_configuration_hash=model_configuration_hash,
            objective_runner=objective_runner,
        )
    finally:
        distributed_runtime.close()


def _train_model_distributed_initialized(
    model: nn.Module,
    dataset: Dataset[TrainingBatch] | IterableDataset[TrainingBatch],
    training_configuration: TrainingConfiguration,
    distributed_configuration: DistributedConfiguration,
    distributed_runtime: DistributedRuntime,
    artifact_directory: Path,
    seed: int,
    evaluation_callback: TrainingEvaluationCallback | None,
    model_configuration_hash: str,
    objective_runner: TrainingObjectiveRunner,
) -> TrainingResult:
    _apply_training_precision(model=model, precision=training_configuration.precision)
    model = prepare_model_for_distributed_training(
        model=model,
        distributed_runtime=distributed_runtime,
    )
    console_log(
        "[train] distributed_device "
        f"rank={distributed_runtime.rank} "
        f"device={distributed_runtime.device} "
        f"precision={training_configuration.precision.value}"
    )
    optimizer = AdamW(
        model.parameters(),
        lr=training_configuration.optimizer.learning_rate,
        weight_decay=training_configuration.optimizer.weight_decay,
    )
    checkpoint_directory = artifact_directory / "checkpoints"
    loaded_checkpoint_step = load_latest_sharded_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=model,
        optimizer=optimizer,
        rank=distributed_runtime.rank,
    )
    _apply_current_optimizer_configuration(
        optimizer=optimizer,
        training_configuration=training_configuration,
    )
    start_step = 0 if loaded_checkpoint_step is None else loaded_checkpoint_step
    data_iterator = create_training_data_iterator(
        dataset=dataset,
        batch_size_sequences=training_configuration.batch_size_sequences,
        dataloader_configuration=training_configuration.dataloader,
        seed=seed,
        distributed_data_assignment=DistributedDataAssignment(
            rank=distributed_runtime.rank,
            world_size=distributed_runtime.world_size,
        ),
    )
    if distributed_runtime.is_coordinator:
        _log_epoch_plan(
            prefix="[train] epoch_plan",
            data_iterator=data_iterator,
            maximum_steps=training_configuration.maximum_steps,
            world_size=distributed_runtime.world_size,
        )
    metrics_logger = (
        TrainingMetricLogger(artifact_directory=artifact_directory)
        if distributed_runtime.is_coordinator
        else None
    )
    final_loss = float("inf")
    checkpoint_path = checkpoint_directory / "latest.json"
    evaluation_path: Path | None = None
    started_at_seconds = perf_counter()
    tokens_processed = 0
    latest_checkpoint_seconds = 0.0
    model.train()
    try:
        for step in range(start_step + 1, training_configuration.maximum_steps + 1):
            token_batch = data_iterator.next_batch()
            prepared_batch = objective_runner.prepare_batch(
                batch=token_batch,
                device=distributed_runtime.device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = objective_runner.loss(model=model, batch=prepared_batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training_configuration.gradient_clip_norm,
            )
            optimizer.step()
            tokens_processed += _batch_token_count(batch=prepared_batch)
            final_loss = distributed_runtime.reduce_mean(float(loss.detach().cpu().item()))
            global_tokens_processed = distributed_runtime.reduce_sum(float(tokens_processed))
            global_gradient_norm = distributed_runtime.reduce_mean(
                float(gradient_norm.detach().cpu().item()),
            )
            if step % training_configuration.checkpoint_interval_steps == 0:
                checkpoint_started_seconds = perf_counter()
                checkpoint_path = _save_distributed_checkpoint(
                    checkpoint_directory=checkpoint_directory,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    distributed_configuration=distributed_configuration,
                    distributed_runtime=distributed_runtime,
                    model_configuration_hash=model_configuration_hash,
                )
                latest_checkpoint_seconds = perf_counter() - checkpoint_started_seconds
            if step % training_configuration.log_interval_steps == 0 and metrics_logger is not None:
                elapsed_seconds = perf_counter() - started_at_seconds
                metrics_logger.write(
                    metric_record=create_training_metric_record(
                        step=step,
                        loss=final_loss,
                        learning_rate=training_configuration.optimizer.learning_rate,
                        gradient_norm=global_gradient_norm,
                        started_at_seconds=started_at_seconds,
                        tokens_processed=tokens_processed,
                        distributed_world_size=distributed_runtime.world_size,
                        distributed_global_tokens_per_second=global_tokens_processed
                        / max(elapsed_seconds, 1e-9),
                        distributed_rank_tokens_per_second=tokens_processed
                        / max(elapsed_seconds, 1e-9),
                        distributed_checkpoint_time=latest_checkpoint_seconds,
                        distributed_strategy=distributed_configuration.strategy.value,
                    ),
                )
                unwrapped_model = unwrap_distributed_model(model=model)
                metrics_logger.write_router_usage(
                    step=step,
                    router_usage_summaries=collect_router_usage_summaries(
                        model=unwrapped_model,
                    ),
                )
                reset_router_usage(model=unwrapped_model)
            training_evaluation_configuration = training_configuration.evaluation
            should_run_training_evaluation = (
                training_evaluation_configuration is not None
                and step % training_evaluation_configuration.interval_steps == 0
            )
            if should_run_training_evaluation:
                if distributed_runtime.is_coordinator:
                    if evaluation_callback is None:
                        raise ValueError("Training evaluation requires an evaluation callback.")
                    evaluation_path = evaluation_callback(
                        step=step,
                        model=unwrap_distributed_model(model=model),
                    )
                    model.train()
                distributed_runtime.barrier()
        if training_configuration.maximum_steps != start_step:
            checkpoint_started_seconds = perf_counter()
            checkpoint_path = _save_distributed_checkpoint(
                checkpoint_directory=checkpoint_directory,
                model=model,
                optimizer=optimizer,
                step=training_configuration.maximum_steps,
                distributed_configuration=distributed_configuration,
                distributed_runtime=distributed_runtime,
                model_configuration_hash=model_configuration_hash,
            )
            latest_checkpoint_seconds = perf_counter() - checkpoint_started_seconds
        distributed_runtime.barrier()
    finally:
        if metrics_logger is not None:
            metrics_logger.close()
    if distributed_runtime.is_coordinator:
        _log_epoch_summary(prefix="[train] epoch_summary", data_iterator=data_iterator)
    return TrainingResult(
        final_step=training_configuration.maximum_steps,
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
        resumed_from_step=start_step,
        evaluation_path=evaluation_path,
    )


def _save_distributed_checkpoint(
    checkpoint_directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    distributed_configuration: DistributedConfiguration,
    distributed_runtime: DistributedRuntime,
    model_configuration_hash: str,
) -> Path:
    save_sharded_rank_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=model,
        optimizer=optimizer,
        step=step,
        rank=distributed_runtime.rank,
        world_size=distributed_runtime.world_size,
    )
    distributed_runtime.barrier()
    checkpoint_path = checkpoint_directory / f"step_{step:08d}"
    if distributed_runtime.is_coordinator:
        checkpoint_path = finalize_sharded_checkpoint(
            checkpoint_directory=checkpoint_directory,
            step=step,
            world_size=distributed_runtime.world_size,
            backend=distributed_configuration.backend,
            strategy=distributed_configuration.strategy,
            topology=distributed_runtime.topology,
            model_configuration_hash=model_configuration_hash,
        )
        save_rank_zero_full_checkpoint_bridge(
            checkpoint_directory=checkpoint_directory,
            model=unwrap_distributed_model(model=model),
            optimizer=optimizer,
            step=step,
        )
    distributed_runtime.barrier()
    return checkpoint_path


def _apply_current_optimizer_configuration(
    optimizer: Optimizer,
    training_configuration: TrainingConfiguration,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = training_configuration.optimizer.learning_rate
        parameter_group["weight_decay"] = training_configuration.optimizer.weight_decay


def _single_process_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _apply_training_precision(model: nn.Module, precision: Precision) -> None:
    match precision:
        case Precision.FP32:
            model.float()
        case Precision.FP16:
            model.half()
        case Precision.BF16:
            model.bfloat16()


def _batch_token_count(batch: TrainingBatch) -> int:
    match batch:
        case torch.Tensor():
            return int(batch.numel())
        case _:
            return int(batch.chosen_token_ids.numel() + batch.rejected_token_ids.numel())


def _log_epoch_plan(
    prefix: str,
    data_iterator: InfiniteDataIterator,
    maximum_steps: int,
    world_size: int,
) -> None:
    batches_per_epoch = data_iterator.batches_per_epoch
    if batches_per_epoch is None:
        console_log(f"{prefix} batches_per_epoch=unknown requested_steps={maximum_steps}")
        return
    estimated_epochs = maximum_steps / max(batches_per_epoch, 1)
    console_log(
        f"{prefix} batches_per_epoch_per_rank={batches_per_epoch} "
        f"requested_steps={maximum_steps} estimated_epochs={estimated_epochs:.4f} "
        f"world_size={world_size}"
    )


def _log_epoch_summary(prefix: str, data_iterator: InfiniteDataIterator) -> None:
    epoch_progress = data_iterator.epoch_progress
    if epoch_progress is None:
        console_log(f"{prefix} completed_epochs=unknown batches_seen={data_iterator.batches_seen}")
        return
    console_log(
        f"{prefix} completed_epochs={epoch_progress:.4f} "
        f"batches_seen={data_iterator.batches_seen}"
    )
