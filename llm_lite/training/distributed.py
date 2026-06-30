from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as torch_distributed
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel

from llm_lite.config.models import DistributedConfiguration, DistributedStrategy
from llm_lite.model.modern import ModernMoeGpt
from llm_lite.model.moe import MoeGpt
from llm_lite.training.topology import DistributedTopology, RankTopology, build_distributed_topology


@dataclass(frozen=True)
class DistributedRuntime:
    distributed_configuration: DistributedConfiguration
    topology: DistributedTopology
    rank_topology: RankTopology
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_coordinator: bool
    initialized_process_group: bool

    def barrier(self) -> None:
        if torch_distributed.is_initialized():
            torch_distributed.barrier()

    def reduce_mean(self, value: float) -> float:
        if not torch_distributed.is_initialized():
            return value
        metric_tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        torch_distributed.all_reduce(metric_tensor, op=torch_distributed.ReduceOp.SUM)
        return float((metric_tensor / self.world_size).detach().cpu().item())

    def reduce_sum(self, value: float) -> float:
        if not torch_distributed.is_initialized():
            return value
        metric_tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        torch_distributed.all_reduce(metric_tensor, op=torch_distributed.ReduceOp.SUM)
        return float(metric_tensor.detach().cpu().item())

    def close(self) -> None:
        if self.initialized_process_group and torch_distributed.is_initialized():
            torch_distributed.destroy_process_group()


def initialize_distributed_runtime(
    distributed_configuration: DistributedConfiguration,
    artifact_directory: Path,
) -> DistributedRuntime:
    topology = build_distributed_topology(
        distributed_configuration=distributed_configuration,
        artifact_directory=artifact_directory,
    )
    rank = _environment_integer(name="RANK")
    local_rank = _environment_integer(name="LOCAL_RANK")
    world_size = _environment_integer(name="WORLD_SIZE")
    if world_size != distributed_configuration.world_size:
        raise ValueError("Runtime WORLD_SIZE must match distributed.world_size.")
    device = _runtime_device(
        distributed_configuration=distributed_configuration,
        local_rank=local_rank,
    )
    if not torch_distributed.is_initialized():
        torch_distributed.init_process_group(
            backend=distributed_configuration.backend.value,
            rank=rank,
            world_size=world_size,
        )
        initialized_process_group = True
    else:
        initialized_process_group = False
    rank_topology = topology.rank_topology(rank=rank)
    rank_topology.rank_work_directory.mkdir(parents=True, exist_ok=True)
    rank_topology.node_shared_directory.mkdir(parents=True, exist_ok=True)
    return DistributedRuntime(
        distributed_configuration=distributed_configuration,
        topology=topology,
        rank_topology=rank_topology,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_coordinator=rank == 0,
        initialized_process_group=initialized_process_group,
    )


def prepare_model_for_distributed_training(
    model: nn.Module,
    distributed_runtime: DistributedRuntime,
) -> nn.Module:
    model = model.to(distributed_runtime.device)
    find_unused_parameters = _needs_unused_parameter_detection(model=model)
    match distributed_runtime.distributed_configuration.strategy:
        case DistributedStrategy.DATA_PARALLEL:
            if distributed_runtime.device.type == "cuda":
                return DistributedDataParallel(
                    model,
                    device_ids=(distributed_runtime.local_rank,),
                    output_device=distributed_runtime.local_rank,
                    find_unused_parameters=find_unused_parameters,
                )
            return DistributedDataParallel(
                model,
                find_unused_parameters=find_unused_parameters,
            )
        case DistributedStrategy.FULLY_SHARDED_DATA_PARALLEL:
            return FullyShardedDataParallel(model)
        case DistributedStrategy.SINGLE_PROCESS:
            raise ValueError("Distributed runtime cannot use single_process strategy.")


def unwrap_distributed_model(model: nn.Module) -> nn.Module:
    match model:
        case DistributedDataParallel(module=module):
            return module
        case _:
            return model


def _needs_unused_parameter_detection(model: nn.Module) -> bool:
    match model:
        case (
            MoeGpt(model_configuration=model_configuration)
            | ModernMoeGpt(
                model_configuration=model_configuration,
            )
        ):
            return model_configuration.router_top_k < model_configuration.expert_count
        case _:
            return False


def _runtime_device(
    distributed_configuration: DistributedConfiguration,
    local_rank: int,
) -> torch.device:
    if distributed_configuration.backend.value == "nccl":
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _environment_integer(name: str) -> int:
    environment_value = os.environ.get(name)
    if environment_value is None:
        raise ValueError(f"{name} must be set for distributed training.")
    return int(environment_value)
