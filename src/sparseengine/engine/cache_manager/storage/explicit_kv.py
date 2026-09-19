from __future__ import annotations

import torch

from sparseengine.kernels.triton.store_kvcache import (
    store_kvcache,
    store_kvcache_with_quest_metadata,
    store_prefill_kvcache_with_quest_metadata,
)

from ..base import AttentionCacheWrite, ExplicitKVPayload, ExplicitKVWrite
from .base import CacheLayout
from .components import CacheComponentSpec


class ExplicitKVStorage:
    """The ordinary two-tensor K/V cache layout."""

    layout = CacheLayout.EXPLICIT_KV

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        if self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError(
                "Explicit KV dimensions must be positive, got "
                f"num_kv_heads={self.num_kv_heads} head_dim={self.head_dim}."
            )
        self.kv_cache: torch.Tensor | None = None

    def allocate(
        self,
        *,
        num_layers: int,
        num_slots: int,
        device: torch.device,
    ) -> None:
        num_layers = int(num_layers)
        num_slots = int(num_slots)
        if num_layers <= 0 or num_slots <= 0:
            raise ValueError(
                "Explicit KV allocation requires positive layers and slots, got "
                f"num_layers={num_layers} num_slots={num_slots}."
            )
        self.kv_cache = torch.empty(
            2,
            num_layers,
            num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=device,
        )

    def _require_cache(self) -> torch.Tensor:
        if self.kv_cache is None:
            raise RuntimeError("Explicit KV storage has not been allocated.")
        return self.kv_cache

    @property
    def cache(self) -> torch.Tensor:
        return self._require_cache()

    def component_specs(self, layer_idx: int) -> tuple[CacheComponentSpec, ...]:
        return tuple(
            CacheComponentSpec(name, (self.num_kv_heads, self.head_dim), self.dtype)
            for name in ("key", "value")
        )

    def component_tensors(self, layer_idx: int) -> tuple[torch.Tensor, ...]:
        payload = self.layer_payload(layer_idx)
        return payload.k_cache, payload.v_cache

    def layer_payload(self, layer_idx: int) -> ExplicitKVPayload:
        cache = self._require_cache()
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < int(cache.shape[1]):
            raise IndexError(
                f"Explicit KV layer index {layer_idx} is outside [0, {int(cache.shape[1])})."
            )
        return ExplicitKVPayload(
            k_cache=cache[0, layer_idx],
            v_cache=cache[1, layer_idx],
        )

    def validate_slot_mapping(self, slot_mapping: torch.Tensor) -> None:
        cache = self._require_cache()
        if slot_mapping.ndim != 1:
            raise ValueError(
                f"Explicit KV slot_mapping must be 1D, got {tuple(slot_mapping.shape)}."
            )
        if slot_mapping.dtype != torch.int32:
            raise TypeError(
                f"Explicit KV slot_mapping must use torch.int32, got {slot_mapping.dtype}."
            )
        if slot_mapping.device != cache.device:
            raise ValueError(
                "Explicit KV slot_mapping must share the cache device, got "
                f"slots={slot_mapping.device} cache={cache.device}."
            )

    def validate_slot_mappings(
        self,
        slot_mappings: tuple[torch.Tensor, ...],
    ) -> None:
        for slot_mapping in slot_mappings:
            self.validate_slot_mapping(slot_mapping)

    def store(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: AttentionCacheWrite,
    ) -> None:
        destination = self._validate_store(layer_idx, slot_mapping, payload)
        assert isinstance(payload, ExplicitKVWrite)
        store_kvcache(
            payload.key,
            payload.value,
            destination.k_cache,
            destination.v_cache,
            slot_mapping,
        )

    def _validate_store(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: AttentionCacheWrite,
    ) -> ExplicitKVPayload:
        if not isinstance(payload, ExplicitKVWrite):
            raise TypeError(
                "ExplicitKVStorage.store requires ExplicitKVWrite, got "
                f"{type(payload).__name__}."
            )
        destination = self.layer_payload(layer_idx)
        if payload.key.shape != payload.value.shape:
            raise ValueError(
                "Explicit KV store tensors must have equal shapes, got "
                f"k={tuple(payload.key.shape)} v={tuple(payload.value.shape)}."
            )
        expected_tail = (self.num_kv_heads, self.head_dim)
        if payload.key.ndim != 3 or tuple(payload.key.shape[1:]) != expected_tail:
            raise ValueError(
                "Explicit KV store tensors must have shape [tokens, "
                f"{self.num_kv_heads}, {self.head_dim}], got "
                f"{tuple(payload.key.shape)}."
            )
        if slot_mapping.shape != (int(payload.key.shape[0]),):
            raise ValueError(
                "Explicit KV slot_mapping must match the token dimension, got "
                f"slots={tuple(slot_mapping.shape)} tokens={int(payload.key.shape[0])}."
            )
        if payload.key.dtype != self.dtype or payload.value.dtype != self.dtype:
            raise TypeError(
                f"Explicit KV store requires dtype={self.dtype}, got "
                f"k={payload.key.dtype} v={payload.value.dtype}."
            )
        destination_device = destination.k_cache.device
        if (
            payload.key.device != destination_device
            or payload.value.device != destination_device
            or slot_mapping.device != destination_device
        ):
            raise ValueError(
                "Explicit KV store tensors must share the destination device, got "
                f"destination={destination_device} key={payload.key.device} "
                f"value={payload.value.device} slots={slot_mapping.device}."
            )
        self.validate_slot_mapping(slot_mapping)
        return destination

    def store_with_quest_metadata(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: AttentionCacheWrite,
        page_max: torch.Tensor,
        page_min: torch.Tensor,
        *,
        page_size: int,
    ) -> None:
        destination = self._validate_store(layer_idx, slot_mapping, payload)
        assert isinstance(payload, ExplicitKVWrite)
        store_kvcache_with_quest_metadata(
            payload.key,
            payload.value,
            destination.k_cache,
            destination.v_cache,
            slot_mapping,
            page_max,
            page_min,
            page_size=page_size,
        )

    def store_prefill_with_quest_metadata(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: AttentionCacheWrite,
        page_segments: torch.Tensor,
        page_max: torch.Tensor,
        page_min: torch.Tensor,
        *,
        page_size: int,
    ) -> None:
        destination = self._validate_store(layer_idx, slot_mapping, payload)
        assert isinstance(payload, ExplicitKVWrite)
        store_prefill_kvcache_with_quest_metadata(
            payload.key,
            payload.value,
            destination.k_cache,
            destination.v_cache,
            slot_mapping,
            page_segments,
            page_max,
            page_min,
            page_size=page_size,
        )

    def bytes_per_slot_per_layer(self) -> int:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return int(2 * self.num_kv_heads * self.head_dim * element_size)

    def slot_capacity(self) -> int:
        return int(self._require_cache().shape[2])

    @torch.no_grad()
    def copy_slots(
        self,
        layer_idx: int,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> None:
        payload = self.layer_payload(layer_idx)
        source_slots = source_slots.to(
            device=payload.k_cache.device,
            dtype=torch.long,
        ).reshape(-1)
        destination_slots = destination_slots.to(
            device=payload.k_cache.device,
            dtype=torch.long,
        ).reshape(-1)
        if source_slots.shape != destination_slots.shape:
            raise ValueError(
                "Explicit KV slot copy requires equal source/destination shapes, "
                f"got {tuple(source_slots.shape)} and "
                f"{tuple(destination_slots.shape)}."
            )
        if source_slots.numel() == 0:
            return
        slot_count = int(payload.k_cache.shape[0])
        in_bounds = (
            (source_slots >= 0)
            & (source_slots < slot_count)
            & (destination_slots >= 0)
            & (destination_slots < slot_count)
        ).all()
        if in_bounds.is_cuda:
            torch._assert_async(in_bounds)
        elif not bool(in_bounds.item()):
            raise ValueError(
                f"Explicit KV slot copy indices must be in [0, {slot_count})."
            )
        k_selected = payload.k_cache.index_select(0, source_slots)
        v_selected = payload.v_cache.index_select(0, source_slots)
        payload.k_cache.index_copy_(0, destination_slots, k_selected)
        payload.v_cache.index_copy_(0, destination_slots, v_selected)

    def accounting_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self._require_cache(),)
