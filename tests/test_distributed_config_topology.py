from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import DistributedConfiguration
from llm_lite.pipeline.stages.pretraining import PretrainingStage
from llm_lite.training.topology import build_distributed_topology


def test_distributed_configuration_loads_defaults() -> None:
    distributed_configuration = DistributedConfiguration.model_validate({})

    assert distributed_configuration.enabled is False
    assert distributed_configuration.backend.value == "gloo"
    assert distributed_configuration.strategy.value == "single_process"
    assert distributed_configuration.world_size == 1
    assert distributed_configuration.checkpoint.type.value == "full"


def test_distributed_configuration_accepts_data_parallel_layout() -> None:
    distributed_configuration = DistributedConfiguration.model_validate(
        {
            "enabled": True,
            "backend": "gloo",
            "strategy": "data_parallel",
            "world_size": 4,
            "simulated_nodes": {"count": 2, "processes_per_node": 2},
            "parallelism": {
                "data": 4,
                "tensor": 1,
                "pipeline": 1,
                "context": 1,
                "expert": 1,
            },
            "checkpoint": {"type": "sharded", "save_rank_local_state": True},
        },
    )

    assert distributed_configuration.enabled is True
    assert distributed_configuration.parallelism.data == 4


def test_distributed_configuration_rejects_invalid_node_shape() -> None:
    with pytest.raises(ValidationError, match="simulated_nodes"):
        DistributedConfiguration.model_validate(
            {
                "enabled": True,
                "strategy": "data_parallel",
                "world_size": 4,
                "simulated_nodes": {"count": 3, "processes_per_node": 2},
                "parallelism": {"data": 4},
            },
        )


def test_distributed_configuration_rejects_unimplemented_parallelism() -> None:
    with pytest.raises(ValidationError, match="Tensor parallelism"):
        DistributedConfiguration.model_validate(
            {
                "enabled": True,
                "strategy": "data_parallel",
                "world_size": 4,
                "simulated_nodes": {"count": 2, "processes_per_node": 2},
                "parallelism": {"data": 2, "tensor": 2},
            },
        )


def test_topology_constructs_rank_paths_and_groups(tmp_path: Path) -> None:
    distributed_configuration = DistributedConfiguration.model_validate(
        {
            "enabled": True,
            "strategy": "data_parallel",
            "world_size": 4,
            "simulated_nodes": {"count": 2, "processes_per_node": 2},
            "parallelism": {"data": 4},
        },
    )

    topology = build_distributed_topology(
        distributed_configuration=distributed_configuration,
        artifact_directory=tmp_path,
    )

    assert topology.world_size == 4
    assert topology.data_parallel_groups == ((0, 1, 2, 3),)
    assert topology.rank_topology(rank=2).node_id == 1
    assert topology.rank_topology(rank=2).node_local_rank == 0
    assert topology.rank_topology(rank=2).rank_work_directory == (
        tmp_path / "work" / "node_001" / "rank_002"
    )
    assert topology.rank_topology(rank=3).node_shared_directory == (
        tmp_path / "work" / "node_001" / "shared"
    )


def test_pretraining_hash_includes_distributed_reconstruction_contract() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
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
    distributed_experiment_configuration = experiment_configuration.model_copy(
        update={"distributed": distributed_configuration},
    )
    stage = PretrainingStage()

    assert stage.configuration_hash(
        experiment_configuration=experiment_configuration,
    ) != stage.configuration_hash(
        experiment_configuration=distributed_experiment_configuration,
    )
