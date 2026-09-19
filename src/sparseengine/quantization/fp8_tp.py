from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Fp8ExpertTpShard:
    """Logical TP shard backed by a block-aligned physical FP8 shard."""

    global_size: int
    tp_rank: int
    tp_size: int
    block_size: int = 128

    def __post_init__(self) -> None:
        if self.global_size <= 0 or self.tp_size <= 0 or self.block_size <= 0:
            raise ValueError("FP8 expert TP dimensions must be positive.")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(
                f"FP8 expert TP rank {self.tp_rank} is outside [0, {self.tp_size})."
            )
        if self.global_size % self.tp_size:
            raise ValueError(
                "FP8 expert intermediate size must be divisible by MoE TP size, "
                f"got {self.global_size} and {self.tp_size}."
            )
        if self.global_size % self.block_size:
            raise ValueError(
                "FP8 expert intermediate size must be block aligned, "
                f"got size={self.global_size}, block={self.block_size}."
            )

    @property
    def logical_size(self) -> int:
        return self.global_size // self.tp_size

    @property
    def logical_start(self) -> int:
        return self.tp_rank * self.logical_size

    @property
    def logical_stop(self) -> int:
        return self.logical_start + self.logical_size

    @property
    def aligned_start(self) -> int:
        return self.logical_start // self.block_size * self.block_size

    @property
    def aligned_stop(self) -> int:
        return (
            (self.logical_stop + self.block_size - 1)
            // self.block_size
            * self.block_size
        )

    @property
    def physical_size(self) -> int:
        return self.aligned_stop - self.aligned_start

    @property
    def local_logical_start(self) -> int:
        return self.logical_start - self.aligned_start

    @property
    def local_logical_stop(self) -> int:
        return self.local_logical_start + self.logical_size

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
            global_shape = (
                (hidden_size // self.block_size, self.global_size // self.block_size)
                if down_projection
                else (
                    self.global_size // self.block_size,
                    hidden_size // self.block_size,
                )
            )
            start = self.aligned_start // self.block_size
            stop = self.aligned_stop // self.block_size
        else:
            global_shape = (
                (hidden_size, self.global_size)
                if down_projection
                else (self.global_size, hidden_size)
            )
            start, stop = self.aligned_start, self.aligned_stop
        if tuple(source_shape) != global_shape:
            kind = "scale" if is_scale else "weight"
            raise ValueError(
                f"FP8 expert checkpoint {kind} shape mismatch: "
                f"expected={global_shape}, got={source_shape}."
            )
        return (
            (slice(None), slice(start, stop))
            if down_projection
            else (slice(start, stop), slice(None))
        )

    def prepare_projection(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor,
        *,
        hidden_size: int,
        down_projection: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_weight_shape = (
            (hidden_size, self.physical_size)
            if down_projection
            else (self.physical_size, hidden_size)
        )
        global_weight_shape = (
            (hidden_size, self.global_size)
            if down_projection
            else (self.global_size, hidden_size)
        )
        if tuple(loaded_weight.shape) == global_weight_shape:
            loaded_weight = (
                loaded_weight[:, self.aligned_start : self.aligned_stop]
                if down_projection
                else loaded_weight[self.aligned_start : self.aligned_stop, :]
            )
        elif tuple(loaded_weight.shape) != local_weight_shape:
            raise ValueError(
                "FP8 expert weight shape mismatch: "
                f"expected local={local_weight_shape} or global={global_weight_shape}, "
                f"got={tuple(loaded_weight.shape)}."
            )

        block = self.block_size
        local_scale_shape = (
            (hidden_size // block, self.physical_size // block)
            if down_projection
            else (self.physical_size // block, hidden_size // block)
        )
        global_scale_shape = (
            (hidden_size // block, self.global_size // block)
            if down_projection
            else (self.global_size // block, hidden_size // block)
        )
        if tuple(loaded_scale.shape) == global_scale_shape:
            start, stop = self.aligned_start // block, self.aligned_stop // block
            loaded_scale = (
                loaded_scale[:, start:stop]
                if down_projection
                else loaded_scale[start:stop, :]
            )
        elif tuple(loaded_scale.shape) != local_scale_shape:
            raise ValueError(
                "FP8 expert scale shape mismatch: "
                f"expected local={local_scale_shape} or global={global_scale_shape}, "
                f"got={tuple(loaded_scale.shape)}."
            )

        # Gate/up keep complete overlapping blocks so dynamic activation scales
        # match the unsharded FP8 block. Down columns alone define ownership.
        if self.logical_size != self.physical_size and down_projection:
            loaded_weight = loaded_weight.clone()
            prefix = self.local_logical_start
            suffix = self.local_logical_stop
            loaded_weight[:, :prefix] = 0
            loaded_weight[:, suffix:] = 0
        return loaded_weight, loaded_scale
