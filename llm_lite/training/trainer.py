from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm_lite.config.models import TrainingConfiguration
from llm_lite.data.datasets import PackedSequenceDataset
from llm_lite.training.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from llm_lite.training.logging import JsonlMetricLogger
from llm_lite.training.objectives import causal_language_modeling_loss


@dataclass(frozen=True)
class TrainingResult:
    final_step: int
    final_loss: float
    checkpoint_path: Path
    resumed_from_step: int


def train_model(
    model: nn.Module,
    dataset: PackedSequenceDataset,
    training_configuration: TrainingConfiguration,
    artifact_directory: Path,
) -> TrainingResult:
    optimizer = AdamW(
        model.parameters(),
        lr=training_configuration.optimizer.learning_rate,
        weight_decay=training_configuration.optimizer.weight_decay,
    )
    checkpoint_directory = artifact_directory / "checkpoints"
    checkpoint_state = latest_checkpoint(checkpoint_directory=checkpoint_directory)
    start_step = 0
    if checkpoint_state is not None:
        start_step = load_checkpoint(
            checkpoint_path=checkpoint_state.checkpoint_path,
            model=model,
            optimizer=optimizer,
        )
    data_loader = DataLoader(
        dataset,
        batch_size=training_configuration.batch_size_sequences,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    data_iterator = iter(data_loader)
    metrics_logger = JsonlMetricLogger(metrics_path=artifact_directory / "metrics.jsonl")
    final_loss = float("inf")
    checkpoint_path = checkpoint_directory / "latest.pt"
    model.train()
    for step in range(start_step + 1, training_configuration.maximum_steps + 1):
        try:
            token_batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            token_batch = next(data_iterator)
        optimizer.zero_grad(set_to_none=True)
        model_output = model(token_batch)
        loss = causal_language_modeling_loss(logits=model_output.logits, token_ids=token_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=training_configuration.gradient_clip_norm,
        )
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        if step % training_configuration.log_interval_steps == 0:
            metrics_logger.write(step=step, loss=final_loss)
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
    return TrainingResult(
        final_step=training_configuration.maximum_steps,
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
        resumed_from_step=start_step,
    )
