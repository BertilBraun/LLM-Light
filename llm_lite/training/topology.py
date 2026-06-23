from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.models import DistributedConfiguration, DistributedStrategy


@dataclass(frozen=True)
class RankTopology:
    rank: int
    local_rank: int
    node_id: int
    node_local_rank: int
    rank_work_directory: Path
    node_shared_directory: Path


@dataclass(frozen=True)
class ParallelismTopology:
    data: int
    tensor: int
    pipeline: int
    context: int
    expert: int


@dataclass(frozen=True)
class DistributedTopology:
    world_size: int
    node_count: int
    processes_per_node: int
    global_artifact_directory: Path
    work_directory: Path
    ranks: tuple[RankTopology, ...]
    data_parallel_groups: tuple[tuple[int, ...], ...]
    parallelism: ParallelismTopology

    def rank_topology(self, rank: int) -> RankTopology:
        if rank < 0 or rank >= self.world_size:
            raise ValueError("Rank must be inside the topology world size.")
        return self.ranks[rank]

    def rank_topology_json(self, rank: int) -> dict[str, int | str]:
        rank_topology = self.rank_topology(rank=rank)
        return {
            "rank": rank_topology.rank,
            "local_rank": rank_topology.local_rank,
            "node_id": rank_topology.node_id,
            "node_local_rank": rank_topology.node_local_rank,
            "rank_work_directory": str(rank_topology.rank_work_directory),
            "node_shared_directory": str(rank_topology.node_shared_directory),
        }

    def manifest_json(self) -> dict[str, int | str | list[dict[str, int | str]] | list[list[int]]]:
        return {
            "world_size": self.world_size,
            "node_count": self.node_count,
            "processes_per_node": self.processes_per_node,
            "global_artifact_directory": str(self.global_artifact_directory),
            "work_directory": str(self.work_directory),
            "ranks": [
                self.rank_topology_json(rank=rank_topology.rank) for rank_topology in self.ranks
            ],
            "data_parallel_groups": [list(group) for group in self.data_parallel_groups],
        }


def build_distributed_topology(
    distributed_configuration: DistributedConfiguration,
    artifact_directory: Path,
) -> DistributedTopology:
    world_size = distributed_configuration.world_size
    node_count = distributed_configuration.simulated_nodes.count
    processes_per_node = distributed_configuration.simulated_nodes.processes_per_node
    if node_count * processes_per_node != world_size:
        raise ValueError("Logical node shape must multiply to world size.")
    if distributed_configuration.parallelism.data != world_size:
        raise ValueError("Only full-width data parallel groups are executable.")
    if distributed_configuration.strategy is DistributedStrategy.SINGLE_PROCESS:
        raise ValueError("Distributed topology requires a distributed execution strategy.")
    work_directory = artifact_directory / "work"
    ranks: list[RankTopology] = []
    for rank in range(world_size):
        node_id = rank // processes_per_node
        node_local_rank = rank % processes_per_node
        ranks.append(
            RankTopology(
                rank=rank,
                local_rank=node_local_rank,
                node_id=node_id,
                node_local_rank=node_local_rank,
                rank_work_directory=work_directory / f"node_{node_id:03d}" / f"rank_{rank:03d}",
                node_shared_directory=work_directory / f"node_{node_id:03d}" / "shared",
            ),
        )
    return DistributedTopology(
        world_size=world_size,
        node_count=node_count,
        processes_per_node=processes_per_node,
        global_artifact_directory=artifact_directory,
        work_directory=work_directory,
        ranks=tuple(ranks),
        data_parallel_groups=(tuple(range(world_size)),),
        parallelism=ParallelismTopology(
            data=distributed_configuration.parallelism.data,
            tensor=distributed_configuration.parallelism.tensor,
            pipeline=distributed_configuration.parallelism.pipeline,
            context=distributed_configuration.parallelism.context,
            expert=distributed_configuration.parallelism.expert,
        ),
    )
