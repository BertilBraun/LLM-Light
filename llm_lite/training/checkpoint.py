import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TypeAlias

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.optim import Optimizer

from llm_lite.config.models import (
    DistributedBackend,
    DistributedCheckpointType,
    DistributedStrategy,
)
from llm_lite.pipeline.artifact import ArtifactManifest
from llm_lite.training.topology import DistributedTopology

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class CheckpointState:
    step: int
    checkpoint_path: Path


class CheckpointKind(str, Enum):
    FULL = "full"
    SHARDED = "sharded"


class CheckpointManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    producing_artifact_fingerprint: str
    checkpoint_kind: CheckpointKind
    checkpoint_path: str
    completion_status: str
    created_at: str


class CheckpointEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    producing_artifact_fingerprint: str
    checkpoint_step: int
    checkpoint_manifest_path: str
    checkpoint_kind: CheckpointKind
    created_at: str


def latest_checkpoint(checkpoint_directory: Path) -> CheckpointState | None:
    latest_path = checkpoint_directory / "latest.pt"
    if not latest_path.exists():
        latest_sharded_path = checkpoint_directory / "latest.json"
        if not latest_sharded_path.exists():
            return None
        latest_data = json.loads(latest_sharded_path.read_text(encoding="utf-8"))
        step = int(latest_data["step"])
        checkpoint_name = str(latest_data["checkpoint"])
        return CheckpointState(
            step=step,
            checkpoint_path=checkpoint_directory / checkpoint_name,
        )
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
    _write_checkpoint_completion(
        checkpoint_directory=checkpoint_directory,
        step=step,
        checkpoint_path=checkpoint_path,
        checkpoint_kind=CheckpointKind.FULL,
    )
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


def load_latest_checkpoint(
    checkpoint_directory: Path,
    model: nn.Module,
    optimizer: Optimizer | None,
) -> int | None:
    checkpoint_state = latest_checkpoint(checkpoint_directory=checkpoint_directory)
    if checkpoint_state is None:
        return None
    return load_checkpoint(
        checkpoint_path=checkpoint_state.checkpoint_path,
        model=model,
        optimizer=optimizer,
    )


def save_sharded_rank_checkpoint(
    checkpoint_directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    rank: int,
    world_size: int,
) -> Path:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    step_directory = checkpoint_directory / f"step_{step:08d}"
    rank_directory = step_directory / f"rank_{rank:05d}"
    rank_directory.mkdir(parents=True, exist_ok=True)
    shard_path = rank_directory / "state.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rank": rank,
            "world_size": world_size,
        },
        shard_path,
    )
    complete_path = rank_directory / "complete.json"
    complete_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "step": step,
                "state": "complete",
                "state_file": "state.pt",
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return step_directory


def finalize_sharded_checkpoint(
    checkpoint_directory: Path,
    step: int,
    world_size: int,
    backend: DistributedBackend,
    strategy: DistributedStrategy,
    topology: DistributedTopology,
    model_configuration_hash: str,
) -> Path:
    step_directory = checkpoint_directory / f"step_{step:08d}"
    _validate_rank_shards(
        step_directory=step_directory,
        world_size=world_size,
    )
    manifest = {
        "step": step,
        "world_size": world_size,
        "backend": backend.value,
        "strategy": strategy.value,
        "checkpoint_type": DistributedCheckpointType.SHARDED.value,
        "topology": topology.manifest_json(),
        "model_configuration_hash": model_configuration_hash,
        "optimizer_present": True,
        "expected_rank_shards": [f"rank_{rank_index:05d}" for rank_index in range(world_size)],
        "completion_status": "complete",
        "created_at": _utc_now(),
    }
    _write_json_atomically(path=step_directory / "manifest.json", value=manifest)
    _write_json_atomically(
        path=checkpoint_directory / "latest.json",
        value={
            "step": step,
            "checkpoint": step_directory.name,
            "manifest": f"{step_directory.name}/manifest.json",
        },
    )
    _write_checkpoint_event(
        checkpoint_directory=checkpoint_directory,
        step=step,
        checkpoint_kind=CheckpointKind.SHARDED,
        checkpoint_manifest_path=step_directory / "manifest.json",
    )
    return step_directory


def save_rank_zero_full_checkpoint_bridge(
    checkpoint_directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
) -> Path:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    bridge_path = checkpoint_directory / "latest.pt"
    torch.save(checkpoint_data, bridge_path)
    return bridge_path


def load_latest_sharded_checkpoint(
    checkpoint_directory: Path,
    model: nn.Module,
    optimizer: Optimizer | None,
    rank: int,
) -> int | None:
    latest_path = checkpoint_directory / "latest.json"
    if not latest_path.exists():
        return None
    latest_data = json.loads(latest_path.read_text(encoding="utf-8"))
    step = int(latest_data["step"])
    checkpoint_name = str(latest_data["checkpoint"])
    shard_path = checkpoint_directory / checkpoint_name / f"rank_{rank:05d}" / "state.pt"
    if not shard_path.exists():
        raise ValueError("Expected rank-local checkpoint shard is missing.")
    checkpoint_data = torch.load(shard_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data["optimizer"])
    return step


def validate_sharded_checkpoint(checkpoint_path: Path, world_size: int) -> None:
    manifest_path = checkpoint_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Sharded checkpoint manifest is missing.")
    _validate_rank_shards(step_directory=checkpoint_path, world_size=world_size)


def _validate_rank_shards(step_directory: Path, world_size: int) -> None:
    missing_ranks: list[int] = []
    for rank in range(world_size):
        rank_directory = step_directory / f"rank_{rank:05d}"
        if (
            not (rank_directory / "state.pt").exists()
            or not (rank_directory / "complete.json").exists()
        ):
            missing_ranks.append(rank)
    if missing_ranks:
        raise ValueError(f"Missing sharded checkpoint ranks: {missing_ranks}")


def _write_json_atomically(path: Path, value: dict[str, JsonValue]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".pending")
    temporary_path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def _write_checkpoint_completion(
    checkpoint_directory: Path,
    step: int,
    checkpoint_path: Path,
    checkpoint_kind: CheckpointKind,
) -> None:
    artifact_directory = checkpoint_directory.parent
    step_directory = checkpoint_directory / f"step_{step:08d}"
    step_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = step_directory / "manifest.json"
    manifest = CheckpointManifest(
        step=step,
        producing_artifact_fingerprint=_artifact_fingerprint(
            artifact_directory=artifact_directory,
        ),
        checkpoint_kind=checkpoint_kind,
        checkpoint_path=Path(os.path.relpath(checkpoint_path, step_directory)).as_posix(),
        completion_status="complete",
        created_at=_utc_now(),
    )
    temporary_path = manifest_path.with_suffix(".json.pending")
    temporary_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary_path, manifest_path)
    _write_checkpoint_event(
        checkpoint_directory=checkpoint_directory,
        step=step,
        checkpoint_kind=checkpoint_kind,
        checkpoint_manifest_path=manifest_path,
    )


def _write_checkpoint_event(
    checkpoint_directory: Path,
    step: int,
    checkpoint_kind: CheckpointKind,
    checkpoint_manifest_path: Path,
) -> None:
    artifact_directory = checkpoint_directory.parent
    event_directory = artifact_directory / "events"
    event_directory.mkdir(parents=True, exist_ok=True)
    event = CheckpointEvent(
        producing_artifact_fingerprint=_artifact_fingerprint(
            artifact_directory=artifact_directory,
        ),
        checkpoint_step=step,
        checkpoint_manifest_path=checkpoint_manifest_path.relative_to(
            artifact_directory,
        ).as_posix(),
        checkpoint_kind=checkpoint_kind,
        created_at=_utc_now(),
    )
    event_path = event_directory / f"checkpoint_{step:08d}.json"
    temporary_path = event_path.with_suffix(".json.pending")
    temporary_path.write_text(event.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary_path, event_path)


def _artifact_fingerprint(artifact_directory: Path) -> str:
    manifest_path = artifact_directory / "manifest.json"
    if not manifest_path.exists():
        return "unknown"
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return manifest.fingerprint


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
