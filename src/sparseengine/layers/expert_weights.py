from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnquantizedExpertTpShard:
    """Rank-local intermediate-dimension slice for an unquantized expert."""

    global_size: int
    tp_rank: int
    tp_size: int

    def __post_init__(self) -> None:
        if self.global_size <= 0 or self.tp_size <= 0:
            raise ValueError("Expert TP dimensions must be positive.")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(
                f"Expert TP rank {self.tp_rank} is outside [0, {self.tp_size})."
            )
        if self.global_size % self.tp_size:
            raise ValueError(
                "Expert intermediate size must be divisible by MoE TP size, "
                f"got {self.global_size} and {self.tp_size}."
            )

    @property
    def logical_size(self) -> int:
        return self.global_size // self.tp_size

    def checkpoint_slice(
        self,
        source_shape: tuple[int, ...],
        *,
        hidden_size: int,
        down_projection: bool,
        is_scale: bool,
    ) -> tuple[slice, ...] | None:
        if self.tp_size == 1:
            return None
        if is_scale:
            raise ValueError("Unquantized expert checkpoint has a weight scale.")
        global_shape = (
            (hidden_size, self.global_size)
            if down_projection
            else (self.global_size, hidden_size)
        )
        if tuple(source_shape) != global_shape:
            raise ValueError(
                "Expert checkpoint shape mismatch: "
                f"expected={global_shape}, got={source_shape}."
            )
        start = self.tp_rank * self.logical_size
        stop = start + self.logical_size
        return (
            (slice(None), slice(start, stop))
            if down_projection
            else (slice(start, stop), slice(None))
        )


class PackedExpertWeightLoader:
    """Shared rank-local loading semantics for packed expert modules."""

    checkpoint_projection_map: dict[str, str]
    checkpoint_tp_shard: object
    hidden_size: int

    def rank_local_weight_slice(
        self,
        source_shape: tuple[int, ...],
        *,
        loaded_shard_id: tuple[int, str],
        is_scale: bool = False,
    ) -> tuple[slice, ...] | None:
        if not isinstance(loaded_shard_id, tuple) or len(loaded_shard_id) != 2:
            raise TypeError("Packed expert shard id must be (expert_id, projection).")
        _, checkpoint_projection = loaded_shard_id
        logical_projection = self.checkpoint_projection_map.get(checkpoint_projection)
        if logical_projection is None:
            raise ValueError(
                f"Unsupported expert projection {checkpoint_projection!r}."
            )
        return self.checkpoint_tp_shard.checkpoint_slice(
            source_shape,
            hidden_size=self.hidden_size,
            down_projection=logical_projection == "down",
            is_scale=is_scale,
        )
