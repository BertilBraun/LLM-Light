import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as torch_multiprocessing
from torch import nn

import llm_lite.training.distributed as distributed_training
from llm_lite.config.models import (
    DataLoaderConfiguration,
    DenseGptConfiguration,
    DistributedConfiguration,
    ModelType,
    MoeGptConfiguration,
    TrainingConfiguration,
)
from llm_lite.data.datasets import (
    PackedSequence,
    load_packed_sequence_dataset,
    write_packed_sequence_stream,
)
from llm_lite.model.gpt import DenseGpt
from llm_lite.model.moe import MoeGpt
from llm_lite.training.distributed import DistributedRuntime, prepare_model_for_distributed_training
from llm_lite.training.objectives import CausalLanguageModelingObjectiveRunner
from llm_lite.training.trainer import train_model_distributed


def test_two_process_gloo_data_parallel_tiny_training_and_resume(tmp_path: Path) -> None:
    packed_directory = tmp_path / "packed"
    packed_directory.mkdir()
    write_packed_sequence_stream(
        sequences=[
            PackedSequence(token_ids=(1, 2, 3, 4, 5)),
            PackedSequence(token_ids=(2, 3, 4, 5, 6)),
            PackedSequence(token_ids=(3, 4, 5, 6, 7)),
            PackedSequence(token_ids=(4, 5, 6, 7, 8)),
        ],
        artifact_directory=packed_directory,
        row_length=5,
        maximum_shard_tokens=5,
    )
    artifact_directory = tmp_path / "artifact"
    port = _free_tcp_port()

    torch_multiprocessing.spawn(
        _distributed_training_worker,
        args=(2, port, str(packed_directory), str(artifact_directory), 2),
        nprocs=2,
        join=True,
    )
    torch_multiprocessing.spawn(
        _distributed_training_worker,
        args=(2, port + 1, str(packed_directory), str(artifact_directory), 3),
        nprocs=2,
        join=True,
    )

    latest = json.loads((artifact_directory / "checkpoints" / "latest.json").read_text("utf-8"))
    manifest_path = artifact_directory / "checkpoints" / latest["manifest"]
    manifest = json.loads(manifest_path.read_text("utf-8"))
    metrics = [
        json.loads(metric_line)
        for metric_line in (artifact_directory / "metrics.jsonl").read_text("utf-8").splitlines()
    ]

    assert latest["step"] == 3
    assert manifest["strategy"] == "data_parallel"
    assert manifest["world_size"] == 2
    assert (artifact_directory / "checkpoints" / "latest.pt").exists()
    assert metrics[-1]["step"] == 3
    assert metrics[-1]["distributed_world_size"] == 2
    assert metrics[-1]["distributed_strategy"] == "data_parallel"
    assert metrics[-1]["distributed_global_tokens_per_second"] > 0


def test_data_parallel_wrap_enables_unused_parameter_detection_for_sparse_moe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ddp_calls: list[dict[str, object]] = []

    class CapturingDistributedDataParallel(nn.Module):
        def __init__(self, module: nn.Module, **kwargs: object) -> None:
            super().__init__()
            self.module = module
            ddp_calls.append(kwargs)

    monkeypatch.setattr(
        distributed_training,
        "DistributedDataParallel",
        CapturingDistributedDataParallel,
    )
    runtime = _runtime_for_wrapper_test(tmp_path=tmp_path)

    prepare_model_for_distributed_training(
        model=DenseGpt(model_configuration=_tiny_model_configuration(), vocabulary_size=16),
        distributed_runtime=runtime,
    )
    prepare_model_for_distributed_training(
        model=MoeGpt(model_configuration=_tiny_sparse_moe_configuration(), vocabulary_size=16),
        distributed_runtime=runtime,
    )

    assert ddp_calls[0]["find_unused_parameters"] is False
    assert ddp_calls[1]["find_unused_parameters"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FSDP smoke requires an accelerator in this installed PyTorch build.",
)
def test_two_process_fsdp_smoke_is_available_for_accelerator_builds(tmp_path: Path) -> None:
    assert tmp_path.exists()


def _distributed_training_worker(
    rank: int,
    world_size: int,
    port: int,
    packed_directory: str,
    artifact_directory: str,
    maximum_steps: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dataset = load_packed_sequence_dataset(artifact_directory=Path(packed_directory))
    model = DenseGpt(
        model_configuration=_tiny_model_configuration(),
        vocabulary_size=16,
    )
    result = train_model_distributed(
        model=model,
        dataset=dataset,
        training_configuration=_tiny_training_configuration(maximum_steps=maximum_steps),
        distributed_configuration=_data_parallel_configuration(world_size=world_size),
        artifact_directory=Path(artifact_directory),
        seed=7,
        evaluation_callback=None,
        model_configuration_hash="tiny-model",
        objective_runner=CausalLanguageModelingObjectiveRunner(auxiliary_loss_weight=0.0),
    )
    result_path = Path(artifact_directory) / f"rank_{rank}_result.json"
    result_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "final_step": result.final_step,
                "resumed_from_step": result.resumed_from_step,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _tiny_model_configuration() -> DenseGptConfiguration:
    return DenseGptConfiguration(
        type=ModelType.DENSE_GPT,
        dimension=8,
        layers=1,
        attention_heads=2,
        feed_forward_dimension=16,
        dropout=0.0,
        tie_embeddings=True,
    )


def _tiny_sparse_moe_configuration() -> MoeGptConfiguration:
    return MoeGptConfiguration(
        type=ModelType.MOE_GPT,
        dimension=8,
        layers=1,
        attention_heads=2,
        expert_feed_forward_dimension=16,
        expert_count=4,
        router_top_k=1,
        dropout=0.0,
        tie_embeddings=False,
    )


def _tiny_training_configuration(maximum_steps: int) -> TrainingConfiguration:
    return TrainingConfiguration(
        maximum_steps=maximum_steps,
        batch_size_sequences=1,
        dataloader=DataLoaderConfiguration(
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
        ),
        checkpoint_interval_steps=1,
        log_interval_steps=1,
    )


def _data_parallel_configuration(world_size: int) -> DistributedConfiguration:
    return DistributedConfiguration.model_validate(
        {
            "enabled": True,
            "backend": "gloo",
            "strategy": "data_parallel",
            "world_size": world_size,
            "simulated_nodes": {"count": 1, "processes_per_node": world_size},
            "parallelism": {"data": world_size},
            "checkpoint": {"type": "sharded"},
        },
    )


def _runtime_for_wrapper_test(tmp_path: Path) -> DistributedRuntime:
    distributed_configuration = _data_parallel_configuration(world_size=1)
    return DistributedRuntime(
        distributed_configuration=distributed_configuration,
        topology=None,  # type: ignore[arg-type]
        rank_topology=None,  # type: ignore[arg-type]
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        is_coordinator=True,
        initialized_process_group=False,
    )


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        return int(port_socket.getsockname()[1])
