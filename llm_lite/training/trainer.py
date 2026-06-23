from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, IterableDataset

from llm_lite.config.models import TrainingConfiguration
from llm_lite.training.checkpoint import load_latest_checkpoint, save_checkpoint
from llm_lite.training.data import create_training_data_iterator
from llm_lite.training.logging import TrainingMetricLogger, create_training_metric_record
from llm_lite.training.objectives import causal_language_modeling_loss


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
    dataset: Dataset[torch.Tensor] | IterableDataset[torch.Tensor],
    training_configuration: TrainingConfiguration,
    artifact_directory: Path,
    seed: int,
    evaluation_callback: TrainingEvaluationCallback | None,
) -> TrainingResult:
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
            optimizer.zero_grad(set_to_none=True)
            model_output = model(token_batch)
            loss = causal_language_modeling_loss(logits=model_output.logits, token_ids=token_batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training_configuration.gradient_clip_norm,
            )
            optimizer.step()
            tokens_processed += int(token_batch.numel())
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
    return TrainingResult(
        final_step=training_configuration.maximum_steps,
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
        resumed_from_step=start_step,
        evaluation_path=evaluation_path,
    )


def _apply_current_optimizer_configuration(
    optimizer: Optimizer,
    training_configuration: TrainingConfiguration,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = training_configuration.optimizer.learning_rate
        parameter_group["weight_decay"] = training_configuration.optimizer.weight_decay
