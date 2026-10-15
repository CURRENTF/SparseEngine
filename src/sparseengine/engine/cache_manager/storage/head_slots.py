from __future__ import annotations

import torch

from ..base import ExplicitKVPayload, ExplicitKVWrite
from .base import CacheLayout


class HeadSlotStorage:
    """Each physical slot holds one KV head, with no padded head dimension."""

    layout = CacheLayout.HEAD_SLOTS

    def __init__(self, *, num_kv_heads, head_dim, dtype):
        self.num_kv_heads, self.head_dim, self.dtype = num_kv_heads, head_dim, dtype
        self.kv_cache = None

    def allocate(self, *, num_layers, num_slots, device):
        self.kv_cache = torch.empty((2, num_layers, num_slots, self.head_dim),
                                    dtype=self.dtype, device=device)

    @property
    def cache(self):
        if self.kv_cache is None:
            raise RuntimeError("Head-slot storage has not been allocated.")
        return self.kv_cache

    def layer_payload(self, layer_idx):
        return ExplicitKVPayload(self.cache[0, layer_idx], self.cache[1, layer_idx],
                                 backend="head_slots")

    def validate_slot_mapping(self, slots):
        if slots.ndim != 2 or slots.shape[1] != self.num_kv_heads:
            raise ValueError("Head write slots must have shape [tokens, KV heads].")
        if slots.dtype != torch.int32 or slots.device != self.cache.device:
            raise TypeError("Head write slots must be int32 on the cache device.")

    def validate_slot_mappings(self, mappings):
        for slots in mappings:
            self.validate_slot_mapping(slots)

    def store(self, layer_idx, slot_mapping, payload):
        if not isinstance(payload, ExplicitKVWrite):
            raise TypeError("Head slots require ExplicitKVWrite.")
        self.validate_slot_mapping(slot_mapping)
        k, v = payload.key, payload.value
        expected = (*slot_mapping.shape, self.head_dim)
        if k.shape != expected or v.shape != expected:
            raise ValueError("Head-slot K/V shapes must match write slots and head_dim.")
        if k.dtype != self.dtype or v.dtype != self.dtype:
            raise TypeError("Head-slot K/V must match the storage dtype.")
        if k.device != self.cache.device or v.device != self.cache.device:
            raise ValueError("Head-slot K/V must share the cache device.")
        from sparseengine.kernels.triton.head_slots import store_head_slots

        dst = self.layer_payload(layer_idx)
        store_head_slots(k, v, dst.k_cache, dst.v_cache, slot_mapping)

    def slot_capacity(self):
        return self.cache.shape[2]

    def bytes_per_slot_per_layer(self):
        return 2 * self.head_dim * torch.empty((), dtype=self.dtype).element_size()

    def copy_slots(self, layer_idx, source_slots, destination_slots):
        raise NotImplementedError("Head-slot compaction changes indices; KV copying is unsupported.")

    def accounting_tensors(self):
        return (self.cache,)
