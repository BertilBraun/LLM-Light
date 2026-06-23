import json
from pathlib import Path

import pytest
from torch import nn
from torch.optim import AdamW

from llm_lite.config.models import DistributedConfiguration
from llm_lite.data.datasets import (
    IterablePackedSequenceDataset,
    PackedDatasetIndex,
    PackedShardIndex,
)
from llm_lite.training.checkpoint import (
    finalize_sharded_checkpoint,
    load_latest_sharded_checkpoint,
    save_sharded_rank_checkpoint,
    validate_sharded_checkpoint,
)
from llm_lite.training.topology import build_distributed_topology


def test_iterable_packed_dataset_partitions_shards_by_rank_and_worker(tmp_path: Path) -> None:
    index = PackedDatasetIndex(
        format_version=1,
        dtype="uint16",
        row_length=4,
        total_sequences=8,
        total_tokens=32,
        shards=tuple(
            PackedShardIndex(
                shard_index=shard_index,
                path=f"shards/shard_{shard_index:06d}.bin",
                sequence_count=1,
                token_count=4,
                first_sequence_index=shard_index,
            )
            for shard_index in range(8)
        ),
    )
    rank_zero_dataset = IterablePackedSequenceDataset(
        artifact_directory=tmp_path,
        index=index,
        seed=0,
        distributed_rank=0,
        distributed_world_size=2,
    )
    rank_one_dataset = IterablePackedSequenceDataset(
        artifact_directory=tmp_path,
        index=index,
        seed=0,
        distributed_rank=1,
        distributed_world_size=2,
    )

    rank_zero_positions = set(
        rank_zero_dataset.shard_positions_for_worker(worker_id=0, worker_count=1)
    )
    rank_one_positions = set(
        rank_one_dataset.shard_positions_for_worker(worker_id=0, worker_count=1)
    )

    assert rank_zero_positions == {0, 2, 4, 6}
    assert rank_one_positions == {1, 3, 5, 7}
    assert not rank_zero_positions.intersection(rank_one_positions)


def test_sharded_checkpoint_manifest_latest_and_missing_shard_rejection(tmp_path: Path) -> None:
    distributed_configuration = DistributedConfiguration.model_validate(
        {
            "enabled": True,
            "strategy": "data_parallel",
            "world_size": 2,
            "simulated_nodes": {"count": 1, "processes_per_node": 2},
            "parallelism": {"data": 2},
            "checkpoint": {"type": "sharded"},
        },
    )
    topology = build_distributed_topology(
        distributed_configuration=distributed_configuration,
        artifact_directory=tmp_path,
    )
    checkpoint_directory = tmp_path / "checkpoints"
    for rank in range(2):
        model = nn.Linear(2, 2)
        optimizer = AdamW(model.parameters(), lr=0.01)
        save_sharded_rank_checkpoint(
            checkpoint_directory=checkpoint_directory,
            model=model,
            optimizer=optimizer,
            step=3,
            rank=rank,
            world_size=2,
        )

    checkpoint_path = finalize_sharded_checkpoint(
        checkpoint_directory=checkpoint_directory,
        step=3,
        world_size=2,
        backend=distributed_configuration.backend,
        strategy=distributed_configuration.strategy,
        topology=topology,
        model_configuration_hash="model-hash",
    )

    manifest = json.loads((checkpoint_path / "manifest.json").read_text(encoding="utf-8"))
    latest = json.loads((checkpoint_directory / "latest.json").read_text(encoding="utf-8"))
    assert manifest["step"] == 3
    assert manifest["world_size"] == 2
    assert manifest["backend"] == "gloo"
    assert manifest["strategy"] == "data_parallel"
    assert manifest["expected_rank_shards"] == ["rank_00000", "rank_00001"]
    assert manifest["completion_status"] == "complete"
    assert latest["checkpoint"] == "step_00000003"

    (checkpoint_path / "rank_00001" / "complete.json").unlink()
    with pytest.raises(ValueError, match="Missing sharded checkpoint ranks"):
        validate_sharded_checkpoint(checkpoint_path=checkpoint_path, world_size=2)


def test_load_latest_sharded_checkpoint_restores_rank_state(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "checkpoints"
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters(), lr=0.01)
    save_sharded_rank_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=model,
        optimizer=optimizer,
        step=5,
        rank=0,
        world_size=1,
    )
    distributed_configuration = DistributedConfiguration.model_validate(
        {
            "enabled": True,
            "strategy": "data_parallel",
            "world_size": 1,
            "parallelism": {"data": 1},
        },
    )
    topology = build_distributed_topology(
        distributed_configuration=distributed_configuration,
        artifact_directory=tmp_path,
    )
    finalize_sharded_checkpoint(
        checkpoint_directory=checkpoint_directory,
        step=5,
        world_size=1,
        backend=distributed_configuration.backend,
        strategy=distributed_configuration.strategy,
        topology=topology,
        model_configuration_hash="model-hash",
    )
    loaded_model = nn.Linear(2, 2)
    loaded_optimizer = AdamW(loaded_model.parameters(), lr=0.01)

    loaded_step = load_latest_sharded_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=loaded_model,
        optimizer=loaded_optimizer,
        rank=0,
    )

    assert loaded_step == 5
