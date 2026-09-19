from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelTopology:
    """Two factorizations of one world: attention DP x TP and MoE EP x TP."""

    attn_tp_size: int
    moe_ep_size: int
    attn_dp_size: int

    def __post_init__(self) -> None:
        sizes = (self.attn_tp_size, self.moe_ep_size, self.attn_dp_size)
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in sizes
        ):
            raise ValueError("Parallel sizes must be positive integers.")
        if self.world_size % self.moe_ep_size:
            raise ValueError(
                f"World size DP*TP={self.world_size} must be divisible by "
                f"MoE EP={self.moe_ep_size}."
            )

    @property
    def world_size(self) -> int:
        return self.attn_dp_size * self.attn_tp_size

    @property
    def moe_tp_size(self) -> int:
        return self.world_size // self.moe_ep_size

    def attn_ranks(self, world_rank: int) -> tuple[int, int]:
        """Return (attention DP rank, attention TP rank)."""
        self._validate_world_rank(world_rank)
        return divmod(world_rank, self.attn_tp_size)

    def moe_ranks(self, world_rank: int) -> tuple[int, int]:
        """Return (MoE EP rank, MoE TP rank)."""
        self._validate_world_rank(world_rank)
        return divmod(world_rank, self.moe_tp_size)

    def _validate_world_rank(self, world_rank: int) -> None:
        if not 0 <= world_rank < self.world_size:
            raise ValueError(
                f"world_rank must be in [0, {self.world_size}), got {world_rank}."
            )


def parallel_group_ranks(
    topology: ParallelTopology,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    world_size = topology.world_size
    attn_tp = topology.attn_tp_size
    moe_tp = topology.moe_tp_size
    return {
        "attn_tp": tuple(
            tuple(range(start, start + attn_tp))
            for start in range(0, world_size, attn_tp)
        ),
        "attn_dp": tuple(
            tuple(range(offset, world_size, attn_tp)) for offset in range(attn_tp)
        ),
        "moe_tp": tuple(
            tuple(range(start, start + moe_tp))
            for start in range(0, world_size, moe_tp)
        ),
        "moe_ep": tuple(
            tuple(range(offset, world_size, moe_tp)) for offset in range(moe_tp)
        ),
    }
