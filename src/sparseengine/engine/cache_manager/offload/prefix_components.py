"""Component-backed prefix transfers for layouts without the paired-KV fast path."""

from math import prod

import torch

from sparseengine.operators.indexed_host_copy import (
    make_pointer_table,
    transfer_components,
)

from ..prefix_offload import (
    PinnedPrefixBlockPool,
    PrefixOffloadController,
    _payload_device_page,
)
from ..storage.components import CacheTransferStorage, PrefixComponents
from .host_pool import HostTensorPool


def storage_prefix_components(
    storage: CacheTransferStorage, num_layers: int
) -> PrefixComponents:
    return tuple(
        tuple(
            zip(
                storage.component_specs(layer),
                storage.component_tensors(layer),
                strict=True,
            )
        )
        for layer in range(num_layers)
    )


def prefix_block_bytes(components: PrefixComponents, block_size: int) -> int:
    return sum(
        spec.block_bytes(block_size) for layer in components for spec, _ in layer
    )


class ComponentPrefixPool(PinnedPrefixBlockPool):
    def __init__(self, *, components, capacity_blocks, block_size):
        super().__init__(
            capacity_blocks=capacity_blocks,
            num_layers=len(components),
            block_size=block_size,
        )
        # Packing only combines components whose lifetime is the complete pool.
        shapes_by_dtype = {}
        for layer in components:
            for spec, _ in layer:
                shapes_by_dtype.setdefault(spec.dtype, []).append(
                    (capacity_blocks * spec.rows_per_block(block_size), *spec.shape)
                )
        self.allocations = {
            dtype: HostTensorPool(shapes, dtype=dtype, allow_packing=True)
            for dtype, shapes in shapes_by_dtype.items()
        }
        parts = {dtype: iter(pool.tensors) for dtype, pool in self.allocations.items()}
        self.layers = tuple(
            tuple(next(parts[spec.dtype]) for spec, _ in layer) for layer in components
        )


class ComponentPrefixOffloadController(PrefixOffloadController):
    def __init__(self, *, components, prefix_cache, host_pool, block_size, device):
        self.components = components  # Retain all device backing for in-flight copies.
        self._token_bytes = 0
        self._page_bytes = 0
        layers = []
        for layer, host_tensors in zip(components, host_pool.layers, strict=True):
            bound = []
            for (spec, tensor), host in zip(layer, host_tensors, strict=True):
                if tuple(tensor.shape[1:]) != spec.shape or tensor.dtype != spec.dtype:
                    raise ValueError(
                        f"Prefix component {spec.name} does not match storage description."
                    )
                bound.append((spec, tensor, host))
                if spec.index_unit == "token":
                    self._token_bytes += spec.row_bytes
                else:
                    self._page_bytes += spec.row_bytes
            layers.append(bound)
        self._layer_transfers = tuple(
            self._bind_transfers(layer, device) for layer in layers
        )
        # Backup has one completion boundary; restore retains per-layer readiness.
        self._backup_transfers = self._bind_transfers(
            [component for layer in layers for component in layer], device
        )
        super().__init__(
            prefix_cache=prefix_cache,
            host_pool=host_pool,
            block_size=block_size,
            device=device,
        )

    @staticmethod
    def _bind_transfers(components, device):
        groups = {}
        for spec, tensor, host in components:
            groups.setdefault(
                (spec.dtype, spec.index_unit, prod(spec.shape)), []
            ).append((tensor, host))
        return tuple(
            (
                make_pointer_table((tensor for tensor, _ in group), device=device),
                make_pointer_table((host for _, host in group), device=device),
                dtype,
                unit,
                width,
            )
            for (dtype, unit, width), group in groups.items()
        )

    def _prepare_d2h_auxiliary(self, blocks, host_indices):
        if not self._page_bytes:
            return ()
        return (
            torch.tensor(
                [_payload_device_page(block) for block in blocks],
                dtype=torch.long,
                device=self.device,
            ),
            torch.tensor(host_indices, dtype=torch.long, device=self.device),
        )

    def _prepare_h2d_auxiliary(self, blocks, host_indices):
        return self._prepare_d2h_auxiliary(blocks, host_indices)

    def _copy_transfers(
        self, transfers, device_slots, host_slots, auxiliary, *, to_host
    ):
        for device_ptrs, host_ptrs, dtype, unit, width in transfers:
            src_slots, dst_slots = (
                (device_slots, host_slots) if unit == "token" else auxiliary
            )
            source, destination = device_ptrs, host_ptrs
            if not to_host:
                source, destination = destination, source
                src_slots, dst_slots = dst_slots, src_slots
            transfer_components(
                source,
                destination,
                src_slots,
                dst_slots,
                width=width,
                dtype=dtype,
            )

    def _submit_d2h_payload(self, device_slots, host_token_indices, auxiliary_tensors):
        self._copy_transfers(
            self._backup_transfers,
            device_slots,
            host_token_indices,
            auxiliary_tensors,
            to_host=True,
        )

    def _submit_h2d_layer(
        self, layer_index, host_token_indices, device_slots, auxiliary_tensors
    ):
        # The base records layer readiness only after every component is submitted.
        self._copy_transfers(
            self._layer_transfers[layer_index],
            device_slots,
            host_token_indices,
            auxiliary_tensors,
            to_host=False,
        )

    def _transfer_token_byte_count(self, token_count):
        return (
            token_count * self._token_bytes
            + (token_count // self.block_size) * self._page_bytes
        )
