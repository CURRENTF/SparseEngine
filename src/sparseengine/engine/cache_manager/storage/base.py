from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import torch

from ..base import AttentionCacheWrite, AttentionPayload


class CacheLayout(str, Enum):
    EXPLICIT_KV = "explicit_kv"
    MLA_LATENT = "mla_latent"


@runtime_checkable
class AttentionCacheStorage(Protocol):
    """Physical attention-cache storage owned by a cache manager."""

    layout: CacheLayout

    def allocate(
        self,
        *,
        num_layers: int,
        num_slots: int,
        device: torch.device,
    ) -> None: ...

    def layer_payload(self, layer_idx: int) -> AttentionPayload: ...

    def validate_slot_mapping(self, slot_mapping: torch.Tensor) -> None: ...

    def validate_slot_mappings(
        self,
        slot_mappings: tuple[torch.Tensor, ...],
    ) -> None: ...

    def store(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: AttentionCacheWrite,
    ) -> None: ...

    def copy_slots(
        self,
        layer_idx: int,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> None: ...

    def slot_capacity(self) -> int: ...

    def bytes_per_slot_per_layer(self) -> int: ...

    def accounting_tensors(self) -> tuple[torch.Tensor, ...]: ...
