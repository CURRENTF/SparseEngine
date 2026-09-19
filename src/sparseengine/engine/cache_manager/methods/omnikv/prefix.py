"""Prefix backing for split-resident OmniKV, including MLA latent storage."""

from __future__ import annotations

from sparseengine.operators.indexed_host_copy import make_pointer_table, transfer_rows

from ...offload.host_pool import HostTensorPool
from ...prefix_offload import PinnedPrefixBlockPool, PrefixOffloadController


class OmniKVPrefixPool(PinnedPrefixBlockPool):
    def __init__(self, storage, capacity_blocks, block_size, device):
        super().__init__(
            capacity_blocks=capacity_blocks,
            block_size=block_size,
            num_layers=len(storage.layers),
        )
        count = capacity_blocks * block_size
        self.layers = []
        self.pointers = []
        for layer, parts in enumerate(storage.layers):
            if layer in storage.full_layers:
                parts = HostTensorPool(
                    [(count, *shape) for shape in storage.shapes],
                    dtype=storage.dtype,
                ).tensors
            else:
                # This region becomes the sole valid sparse prefix backing.
                parts = tuple(x[storage.num_slots :] for x in parts)
            self.layers.append(parts)
            self.pointers.append(make_pointer_table(parts, device=device))


class OmniKVPrefixOffloadController(PrefixOffloadController):
    def __init__(self, *, prefix_cache, storage, host_pool, block_size, device):
        self.storage = storage
        super().__init__(
            prefix_cache=prefix_cache,
            host_pool=host_pool,
            block_size=block_size,
            device=device,
        )

    def _submit_d2h_payload(self, device_slots, host_token_indices, auxiliary_tensors):
        for layer in range(self.host_pool.num_layers):
            for component, (heads, dim) in enumerate(self.storage.shapes):
                transfer_rows(
                    self.storage.pointers[layer],
                    self.host_pool.pointers[layer],
                    device_slots,
                    host_token_indices,
                    width=heads * dim,
                    dtype=self.storage.dtype,
                    component=component,
                    slot_map=self.storage.host_slot_map
                    if layer not in self.storage.full_layers
                    else None,
                )

    def _finish_d2h(self, operation):
        # Publish remapping only after both components of every layer are ready.
        self.storage.host_slot_map.index_copy_(
            0,
            operation.device_token_indices.long(),
            (operation.host_token_indices + self.storage.num_slots).int(),
        )
        super()._finish_d2h(operation)

    def _submit_h2d_layer(
        self, layer_index, host_token_indices, device_slots, auxiliary_tensors
    ):
        if layer_index not in self.storage.full_layers:
            self.storage.host_slot_map.index_copy_(
                0,
                device_slots.long(),
                (host_token_indices + self.storage.num_slots).int(),
            )
            return
        for component, (heads, dim) in enumerate(self.storage.shapes):
            transfer_rows(
                self.host_pool.pointers[layer_index],
                self.storage.pointers[layer_index],
                host_token_indices,
                device_slots,
                width=heads * dim,
                dtype=self.storage.dtype,
                component=component,
            )

    def _transfer_token_byte_count(self, token_count):
        return (
            token_count
            * self.host_pool.num_layers
            * self.storage.bytes_per_slot_per_layer()
        )

    def _h2d_token_byte_count(self, token_count):
        # Sparse prefix restore changes the map; only full layers cross PCIe.
        return (
            token_count
            * len(self.storage.full_layers)
            * self.storage.bytes_per_slot_per_layer()
        )

    def stats(self):
        result = super().stats()
        # D2H publication also reads sparse backing through the GPU before
        # writing its permanent host location. Report that read separately.
        result["prefix_cache_sparse_rehome_h2d_bytes"] = (
            self.d2h_bytes
            * (self.host_pool.num_layers - len(self.storage.full_layers))
            // self.host_pool.num_layers
        )
        return result
