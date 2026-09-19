from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from sparseengine.distributed.topology import ParallelTopology, parallel_group_ranks


@dataclass(frozen=True)
class ParallelGroup:
    process_group: dist.ProcessGroup | None
    ranks: tuple[int, ...]
    rank: int
    size: int

    def __post_init__(self) -> None:
        if self.size != len(self.ranks):
            raise ValueError(
                f"ParallelGroup size={self.size} does not match ranks={self.ranks}."
            )
        if not 0 <= self.rank < self.size:
            raise ValueError(
                f"ParallelGroup rank must be in [0, {self.size}), got {self.rank}."
            )

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        if self.size > 1:
            dist.all_reduce(tensor, op=op, group=self.process_group)
        return tensor

    def broadcast(self, tensor: torch.Tensor, *, src_rank: int = 0) -> torch.Tensor:
        if not 0 <= src_rank < self.size:
            raise ValueError(
                f"Broadcast source must be in [0, {self.size}), got {src_rank}."
            )
        if self.size > 1:
            dist.broadcast(tensor, src=self.ranks[src_rank], group=self.process_group)
        return tensor

    def gather(
        self, tensor: torch.Tensor, dst_rank: int = 0
    ) -> list[torch.Tensor] | None:
        if not 0 <= dst_rank < self.size:
            raise ValueError(
                f"Gather destination must be in [0, {self.size}), got {dst_rank}."
            )
        if self.size == 1:
            return [tensor]
        gather_list = (
            [torch.empty_like(tensor) for _ in range(self.size)]
            if self.rank == dst_rank
            else None
        )
        dist.gather(
            tensor,
            gather_list=gather_list,
            dst=self.ranks[dst_rank],
            group=self.process_group,
        )
        return gather_list

    def barrier(self, *, device_ids: list[int] | None = None) -> None:
        if self.size > 1:
            dist.barrier(group=self.process_group, device_ids=device_ids)


@dataclass(frozen=True)
class ParallelContext:
    world: ParallelGroup
    attn_tp: ParallelGroup
    attn_dp: ParallelGroup
    moe_tp: ParallelGroup
    moe_ep: ParallelGroup

    @property
    def world_rank(self) -> int:
        return self.world.rank

    @property
    def world_size(self) -> int:
        return self.world.size

    @property
    def attn_tp_rank(self) -> int:
        return self.attn_tp.rank

    @property
    def attn_tp_size(self) -> int:
        return self.attn_tp.size

    @property
    def attn_dp_rank(self) -> int:
        return self.attn_dp.rank

    @property
    def attn_dp_size(self) -> int:
        return self.attn_dp.size

    @property
    def moe_tp_rank(self) -> int:
        return self.moe_tp.rank

    @property
    def moe_tp_size(self) -> int:
        return self.moe_tp.size

    @property
    def moe_ep_rank(self) -> int:
        return self.moe_ep.rank

    @property
    def moe_ep_size(self) -> int:
        return self.moe_ep.size


_PARALLEL_CONTEXT: ParallelContext | None = None


def _local_group(
    groups: tuple[tuple[int, ...], ...],
    process_groups: dict[tuple[int, ...], dist.ProcessGroup | None],
    world_rank: int,
) -> ParallelGroup:
    for ranks in groups:
        if world_rank in ranks:
            return ParallelGroup(
                process_group=process_groups[ranks],
                ranks=ranks,
                rank=ranks.index(world_rank),
                size=len(ranks),
            )
    raise RuntimeError(f"No parallel group contains world rank {world_rank}.")


def init_parallel_context(
    *,
    topology: ParallelTopology,
) -> ParallelContext:
    global _PARALLEL_CONTEXT
    if _PARALLEL_CONTEXT is not None:
        raise RuntimeError("ParallelContext is already initialized.")
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before ParallelContext."
        )

    expected_world_size = topology.world_size
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    if world_size != expected_world_size:
        raise ValueError(
            "Distributed world size does not match parallel configuration: "
            f"world_size={world_size}, attention DP={topology.attn_dp_size}, "
            f"TP={topology.attn_tp_size}, MoE EP={topology.moe_ep_size}, "
            f"TP={topology.moe_tp_size}."
        )

    ranks_by_dimension = parallel_group_ranks(topology)
    world_ranks = tuple(range(world_size))
    process_groups: dict[tuple[int, ...], dist.ProcessGroup | None] = {
        world_ranks: dist.group.WORLD,
    }
    for groups in ranks_by_dimension.values():
        for ranks in groups:
            if ranks in process_groups:
                continue
            process_groups[ranks] = (
                None if len(ranks) == 1 else dist.new_group(list(ranks))
            )

    context = ParallelContext(
        world=ParallelGroup(dist.group.WORLD, world_ranks, world_rank, world_size),
        **{
            dimension: _local_group(groups, process_groups, world_rank)
            for dimension, groups in ranks_by_dimension.items()
        },
    )
    _PARALLEL_CONTEXT = context
    return _PARALLEL_CONTEXT


def get_parallel_context() -> ParallelContext:
    if _PARALLEL_CONTEXT is None:
        raise RuntimeError("ParallelContext is not initialized.")
    return _PARALLEL_CONTEXT


def reset_parallel_context() -> None:
    global _PARALLEL_CONTEXT
    _PARALLEL_CONTEXT = None
