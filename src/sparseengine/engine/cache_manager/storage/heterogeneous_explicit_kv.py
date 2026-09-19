from __future__ import annotations

import torch

from sparseengine.kernels.triton.store_kvcache import store_kvcache

from ..base import AttentionCacheWrite, ExplicitKVPayload, ExplicitKVWrite
from .base import CacheLayout
from .components import CacheComponentSpec


class HeterogeneousExplicitKVStorage:
    """Explicit K/V tensors whose head layout may differ by layer."""

    layout = CacheLayout.EXPLICIT_KV

    def __init__(self, *, layer_shapes: tuple[tuple[int, int], ...], dtype: torch.dtype) -> None:
        self.layer_shapes = tuple((int(heads), int(dim)) for heads, dim in layer_shapes)
        if not self.layer_shapes or any(heads <= 0 or dim <= 0 for heads, dim in self.layer_shapes):
            raise ValueError(f"Heterogeneous KV layer shapes must be positive, got {self.layer_shapes}.")
        self.dtype = dtype
        self.kv_cache: list[torch.Tensor] = []

    def allocate(self, *, num_layers: int, num_slots: int, device: torch.device) -> None:
        if int(num_layers) != len(self.layer_shapes) or int(num_slots) <= 0:
            raise ValueError(
                "Heterogeneous KV allocation does not match its layout: "
                f"layers={num_layers}/{len(self.layer_shapes)} slots={num_slots}."
            )
        self.kv_cache = [
            torch.empty(2, int(num_slots), heads, dim, dtype=self.dtype, device=device)
            for heads, dim in self.layer_shapes
        ]

    def _layer_cache(self, layer_idx: int) -> torch.Tensor:
        if not self.kv_cache:
            raise RuntimeError("Heterogeneous KV storage has not been allocated.")
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < len(self.kv_cache):
            raise IndexError(f"KV layer index {layer_idx} is outside [0, {len(self.kv_cache)}).")
        return self.kv_cache[layer_idx]

    @property
    def cache(self) -> list[torch.Tensor]:
        if not self.kv_cache:
            raise RuntimeError("Heterogeneous KV storage has not been allocated.")
        return self.kv_cache

    def component_specs(self, layer_idx: int) -> tuple[CacheComponentSpec, ...]:
        return tuple(
            CacheComponentSpec(name, self.layer_shapes[layer_idx], self.dtype)
            for name in ("key", "value")
        )

    def component_tensors(self, layer_idx: int) -> tuple[torch.Tensor, ...]:
        payload = self.layer_payload(layer_idx)
        return payload.k_cache, payload.v_cache

    def layer_payload(self, layer_idx: int) -> ExplicitKVPayload:
        cache = self._layer_cache(layer_idx)
        return ExplicitKVPayload(k_cache=cache[0], v_cache=cache[1])

    def validate_slot_mapping(self, slot_mapping: torch.Tensor) -> None:
        cache = self._layer_cache(0)
        if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.int32:
            raise ValueError(
                "Heterogeneous KV slot_mapping must be 1D int32, "
                f"got shape={tuple(slot_mapping.shape)} dtype={slot_mapping.dtype}."
            )
        if slot_mapping.device != cache.device:
            raise ValueError(
                f"KV slot_mapping device {slot_mapping.device} does not match cache {cache.device}."
            )

    def validate_slot_mappings(self, slot_mappings: tuple[torch.Tensor, ...]) -> None:
        for slot_mapping in slot_mappings:
            self.validate_slot_mapping(slot_mapping)

    def store(self, layer_idx: int, slot_mapping: torch.Tensor, payload: AttentionCacheWrite) -> None:
        if not isinstance(payload, ExplicitKVWrite):
            raise TypeError(f"Heterogeneous KV storage requires ExplicitKVWrite, got {type(payload).__name__}.")
        destination = self.layer_payload(layer_idx)
        expected = (int(payload.key.shape[0]), *self.layer_shapes[int(layer_idx)])
        if tuple(payload.key.shape) != expected or tuple(payload.value.shape) != expected:
            raise ValueError(
                f"KV payload for layer {layer_idx} must have shape {expected}, "
                f"got K={tuple(payload.key.shape)} V={tuple(payload.value.shape)}."
            )
        if payload.key.dtype != self.dtype or payload.value.dtype != self.dtype:
            raise TypeError(
                f"KV payload requires dtype={self.dtype}, got K={payload.key.dtype} V={payload.value.dtype}."
            )
        if slot_mapping.shape != (int(payload.key.shape[0]),):
            raise ValueError(
                "Heterogeneous KV slot_mapping must match the token dimension, "
                f"got slots={tuple(slot_mapping.shape)} tokens={payload.key.shape[0]}."
            )
        if payload.key.device != destination.k_cache.device or payload.value.device != destination.v_cache.device:
            raise ValueError(
                "Heterogeneous KV payload must share the cache device, got "
                f"K={payload.key.device} V={payload.value.device} cache={destination.k_cache.device}."
            )
        self.validate_slot_mapping(slot_mapping)
        if payload.key.is_cuda:
            store_kvcache(payload.key, payload.value, destination.k_cache, destination.v_cache, slot_mapping)
        else:
            slots = slot_mapping.to(torch.long)
            destination.k_cache.index_copy_(0, slots, payload.key)
            destination.v_cache.index_copy_(0, slots, payload.value)

    def bytes_per_slot_per_layer(self) -> int:
        total = self.bytes_per_slot()
        return (total + len(self.layer_shapes) - 1) // len(self.layer_shapes)

    def bytes_per_slot(self) -> int:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return sum(2 * heads * dim * element_size for heads, dim in self.layer_shapes)

    def slot_capacity(self) -> int:
        return int(self._layer_cache(0).shape[1])

    @torch.no_grad()
    def copy_slots(
        self,
        layer_idx: int,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> None:
        payload = self.layer_payload(layer_idx)
        source = source_slots.to(device=payload.k_cache.device, dtype=torch.long).reshape(-1)
        destination = destination_slots.to(device=payload.k_cache.device, dtype=torch.long).reshape(-1)
        if source.shape != destination.shape:
            raise ValueError(
                f"KV slot copy requires equal shapes, got {tuple(source.shape)} and {tuple(destination.shape)}."
            )
        if source.numel() == 0:
            return
        payload.k_cache.index_copy_(0, destination, payload.k_cache.index_select(0, source))
        payload.v_cache.index_copy_(0, destination, payload.v_cache.index_select(0, source))

    def accounting_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.kv_cache)
