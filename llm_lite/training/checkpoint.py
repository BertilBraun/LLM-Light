from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class CheckpointState:
    step: int
    checkpoint_path: Path


def latest_checkpoint(checkpoint_directory: Path) -> CheckpointState | None:
    latest_path = checkpoint_directory / "latest.pt"
    if not latest_path.exists():
        return None
    checkpoint_data = torch.load(latest_path, map_location="cpu", weights_only=False)
    return CheckpointState(step=int(checkpoint_data["step"]), checkpoint_path=latest_path)


def save_checkpoint(
    checkpoint_directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
) -> Path:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / f"step_{step:08d}.pt"
    checkpoint_data = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint_data, checkpoint_path)
    torch.save(checkpoint_data, checkpoint_directory / "latest.pt")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer | None,
) -> int:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data["optimizer"])
    return int(checkpoint_data["step"])
