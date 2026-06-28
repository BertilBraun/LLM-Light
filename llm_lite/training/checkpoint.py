import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, ValidationError
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
TopologyManifestValue: TypeAlias = int | str | list[dict[str, int | str]] | list[list[int]]


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


class ShardedCheckpointManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    producing_artifact_fingerprint: str
    world_size: int
    backend: str
    strategy: str
    checkpoint_type: str
    topology: dict[str, TopologyManifestValue]
    model_configuration_hash: str
    optimizer_present: bool
    expected_rank_shards: tuple[str, ...]
    completion_status: str
    created_at: str
    rank_zero_full_checkpoint_path: str | None = None


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
    latest_path = checkpoint_directory / "latest.pt"
    pending_latest_path = checkpoint_directory / "latest.pt.pending"
    torch.save(checkpoint_data, pending_latest_path)
    os.replace(pending_latest_path, latest_path)
    _write_checkpoint_completion(
        checkpoint_directory=checkpoint_directory,
        step=step,
        checkpoint_path=checkpoint_path,
        checkpoint_kind=CheckpointKind.FULL,
    )
    return checkpoint_path


def retain_recent_checkpoints(checkpoint_directory: Path, max_checkpoints: int | None) -> None:
    if max_checkpoints is None:
        return
    latest_state = latest_checkpoint(checkpoint_directory=checkpoint_directory)
    if latest_state is None:
        return
    completed_steps = _completed_checkpoint_steps(checkpoint_directory=checkpoint_directory)
    older_completed_steps = tuple(step for step in completed_steps if step != latest_state.step)
    retained_interval_steps = set(older_completed_steps[-max_checkpoints:])
    retained_steps = retained_interval_steps | {latest_state.step}
    for step in completed_steps:
        if step not in retained_steps:
            _delete_checkpoint_step(checkpoint_directory=checkpoint_directory, step=step)


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
    rank_zero_full_checkpoint_path: Path | None = None,
) -> Path:
    step_directory = checkpoint_directory / f"step_{step:08d}"
    _validate_rank_shards(
        step_directory=step_directory,
        world_size=world_size,
    )
    manifest = {
        "step": step,
        "producing_artifact_fingerprint": _artifact_fingerprint(
            artifact_directory=checkpoint_directory.parent,
        ),
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
        "rank_zero_full_checkpoint_path": (
            None
            if rank_zero_full_checkpoint_path is None
            else Path(os.path.relpath(rank_zero_full_checkpoint_path, step_directory)).as_posix()
        ),
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
    step_directory = checkpoint_directory / f"step_{step:08d}"
    step_directory.mkdir(parents=True, exist_ok=True)
    step_bridge_path = step_directory / "rank_zero_full.pt"
    pending_step_bridge_path = step_directory / "rank_zero_full.pt.pending"
    torch.save(checkpoint_data, pending_step_bridge_path)
    os.replace(pending_step_bridge_path, step_bridge_path)
    bridge_path = checkpoint_directory / "latest.pt"
    pending_bridge_path = checkpoint_directory / "latest.pt.pending"
    torch.save(checkpoint_data, pending_bridge_path)
    os.replace(pending_bridge_path, bridge_path)
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


def _completed_checkpoint_steps(checkpoint_directory: Path) -> tuple[int, ...]:
    completed_steps: list[int] = []
    for manifest_path in checkpoint_directory.glob("step_*/manifest.json"):
        step_directory = manifest_path.parent
        step = _checkpoint_step_from_name(step_directory.name)
        if step is None:
            continue
        if _checkpoint_manifest_is_complete(manifest_path=manifest_path):
            completed_steps.append(step)
    return tuple(sorted(completed_steps))


def _checkpoint_manifest_is_complete(manifest_path: Path) -> bool:
    manifest_text = manifest_path.read_text(encoding="utf-8")
    try:
        manifest = CheckpointManifest.model_validate_json(manifest_text)
        return manifest.completion_status == "complete"
    except ValidationError:
        sharded_manifest = ShardedCheckpointManifest.model_validate_json(manifest_text)
        return sharded_manifest.completion_status == "complete"


def _checkpoint_step_from_name(name: str) -> int | None:
    prefix = "step_"
    if not name.startswith(prefix):
        return None
    step_text = name[len(prefix) :]
    if not step_text.isdecimal():
        return None
    return int(step_text)


def _delete_checkpoint_step(checkpoint_directory: Path, step: int) -> None:
    checkpoint_name = f"step_{step:08d}"
    step_directory = checkpoint_directory / checkpoint_name
    checkpoint_file = checkpoint_directory / f"{checkpoint_name}.pt"
    event_path = checkpoint_directory.parent / "events" / f"checkpoint_{step:08d}.json"
    if step_directory.exists():
        shutil.rmtree(step_directory)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    if event_path.exists():
        event_path.unlink()


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
