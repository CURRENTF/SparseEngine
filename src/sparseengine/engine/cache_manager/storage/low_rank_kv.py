from __future__ import annotations

import torch

from ..base import LowRankKVPayload, LowRankKVWrite
from .base import CacheLayout
from .components import CacheComponentSpec


class LowRankKVStorage:
    layout = CacheLayout.LOW_RANK_KV

    def __init__(self, manifest, *, dtype):
        self.groups = manifest["model"]["num_key_value_heads"] // manifest["group_size"]
        self.ranks = tuple(tuple(pair) for pair in manifest["ranks"])
        self.dtype = dtype
        self.layers = []

    def allocate(self, *, num_layers, num_slots, device):
        if num_layers != len(self.ranks) or num_slots <= 0:
            raise ValueError("Low-rank KV allocation does not match its layer/slot contract.")
        self.layers = [LowRankKVPayload(
            torch.empty(num_slots, self.groups, rk, dtype=self.dtype, device=device),
            torch.empty(num_slots, self.groups, rv, dtype=self.dtype, device=device),
            torch.empty(num_slots, dtype=torch.int32, device=device),
        ) for rk, rv in self.ranks]

    def layer_payload(self, layer_idx):
        return self.layers[layer_idx]

    def allocate_shared_history(self, *, num_slots, device):
        """Startup synthetic history shares tensors only across identical ranks."""
        by_rank = {}
        for rk, rv in self.ranks:
            if (rk, rv) not in by_rank:
                by_rank[rk, rv] = LowRankKVPayload(
                    torch.zeros(num_slots, self.groups, rk, dtype=self.dtype, device=device),
                    torch.zeros(num_slots, self.groups, rv, dtype=self.dtype, device=device),
                    torch.zeros(num_slots, dtype=torch.int32, device=device),
                )
        self.layers = [by_rank[pair] for pair in self.ranks]

    def bytes_per_slot(self):
        size = torch.empty((), dtype=self.dtype).element_size()
        return sum(size * self.groups * (rk + rv) + 4 for rk, rv in self.ranks)

    def bytes_per_slot_per_layer(self):
        # Conservative average for generic diagnostics; allocation uses the exact sum.
        return (self.bytes_per_slot() + len(self.ranks) - 1) // len(self.ranks)

    def component_specs(self, layer_idx):
        rk, rv = self.ranks[layer_idx]
        return (CacheComponentSpec("key_latent", (self.groups, rk), self.dtype),
                CacheComponentSpec("value_latent", (self.groups, rv), self.dtype),
                CacheComponentSpec("positions", (), torch.int32))

    def component_tensors(self, layer_idx):
        p = self.layer_payload(layer_idx)
        return p.key_latent, p.value_latent, p.positions

    def validate_slot_mapping(self, slots):
        if slots.ndim != 1 or slots.dtype != torch.int32 or not slots.is_contiguous():
            raise ValueError("Low-rank KV write slots must be a one-dimensional int32 tensor.")
        if slots.device != self.layers[0].positions.device:
            raise ValueError("Low-rank KV slots must be on the cache device.")

    def validate_slot_mappings(self, mappings):
        for slots in mappings:
            self.validate_slot_mapping(slots)

    def store(self, layer_idx, slot_mapping, payload):
        if not isinstance(payload, LowRankKVWrite):
            raise TypeError("Low-rank storage requires LowRankKVWrite.")
        self.validate_slot_mapping(slot_mapping)
        count = slot_mapping.numel()
        rk, rv = self.ranks[layer_idx]
        for x, rank in ((payload.key_latent, rk), (payload.value_latent, rv)):
            if (x.shape != (count, self.groups, rank) or x.dtype != self.dtype
                    or x.device != slot_mapping.device or x.stride(-1) != 1 or x.stride(-2) != rank):
                raise ValueError("Low-rank KV write shape/dtype/device mismatch.")
        if (payload.positions.shape != (count,) or payload.positions.device != slot_mapping.device
                or payload.positions.dtype not in (torch.int32, torch.int64) or not payload.positions.is_contiguous()):
            raise ValueError("Low-rank KV positions must match write tokens and device.")
        from sparseengine.kernels.triton.palu import store_latent

        store_latent(payload, slot_mapping, self.layer_payload(layer_idx))

    def copy_slots(self, layer_idx, source_slots, destination_slots):
        if source_slots.shape != destination_slots.shape:
            raise ValueError("Low-rank slot copy requires equal source/destination lengths.")
        for tensor in self.component_tensors(layer_idx):
            tensor.index_copy_(0, destination_slots.long(), tensor.index_select(0, source_slots.long()))

    def slot_capacity(self):
        return int(self.layers[0].positions.numel())

    def accounting_tensors(self):
        return tuple(tensor for i in range(len(self.layers)) for tensor in self.component_tensors(i))
