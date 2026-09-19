"""Physical transfer rows; logical prefix and residency policy live with owners."""

from dataclasses import dataclass
from math import prod
from typing import Literal, Protocol

import torch


@dataclass(frozen=True)
class CacheComponentSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    index_unit: Literal["token", "page"] = "token"

    @property
    def row_bytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize

    def rows_per_block(self, block_size: int) -> int:
        return block_size if self.index_unit == "token" else 1

    def block_bytes(self, block_size: int) -> int:
        return self.rows_per_block(block_size) * self.row_bytes


class CacheTransferStorage(Protocol):
    """Optional physical-copy contract for storage layouts supported by offload."""

    def component_specs(self, layer_idx: int) -> tuple[CacheComponentSpec, ...]: ...

    def component_tensors(self, layer_idx: int) -> tuple[torch.Tensor, ...]: ...


PrefixComponents = tuple[tuple[tuple[CacheComponentSpec, torch.Tensor], ...], ...]
