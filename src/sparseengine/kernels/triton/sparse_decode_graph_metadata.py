from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _publish_uniform_sparse_decode_slots_kernel(
    free_slots,
    slot_table,
    layer_slot_mapping,
    public_slot_mapping,
    context_lens,
    request_indices,
    layer_free_starts,
    active_count,
    transformer_layer_ids,
    free_slots_stride_layer: tl.constexpr,
    slot_table_stride_layer: tl.constexpr,
    slot_table_stride_row: tl.constexpr,
    slot_table_stride_token: tl.constexpr,
    layer_slot_stride: tl.constexpr,
    batch_capacity: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    kv_layer = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < batch_capacity
    active = offsets < tl.load(active_count)
    transformer_layer = tl.load(transformer_layer_ids + kv_layer)
    free_start = tl.load(layer_free_starts + transformer_layer)
    slots = tl.load(
        free_slots + kv_layer * free_slots_stride_layer + free_start + offsets,
        mask=in_bounds & active,
        other=-1,
    )
    tl.store(
        layer_slot_mapping + transformer_layer * layer_slot_stride + offsets,
        slots,
        mask=in_bounds,
    )
    rows = tl.load(request_indices + offsets, mask=in_bounds, other=0)
    context = tl.load(context_lens + offsets, mask=in_bounds, other=1)
    table_offsets = (
        kv_layer * slot_table_stride_layer
        + rows * slot_table_stride_row
        + (context - 1) * slot_table_stride_token
    )
    tl.store(slot_table + table_offsets, slots, mask=in_bounds & active)
    tl.store(
        public_slot_mapping + offsets,
        slots,
        mask=in_bounds & (kv_layer == 0),
    )


def publish_uniform_sparse_decode_graph_slots(
    free_slots: torch.Tensor,
    slot_table: torch.Tensor,
    layer_slot_mapping: torch.Tensor,
    public_slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    request_indices: torch.Tensor,
    layer_free_starts: torch.Tensor,
    active_count: torch.Tensor,
    transformer_layer_ids: torch.Tensor,
) -> None:
    """Publish physical layer slots while sharing uniform row/length metadata."""

    batch_capacity = int(public_slot_mapping.numel())
    if layer_slot_mapping.ndim != 2 or layer_slot_mapping.shape[1] != batch_capacity:
        raise ValueError("Uniform sparse layer slots must match public batch capacity.")
    num_layers = int(layer_slot_mapping.shape[0])
    if context_lens.shape != (batch_capacity,) or context_lens.dtype != torch.int32:
        raise ValueError("Uniform sparse context lengths must match batch capacity.")
    if request_indices.shape != (batch_capacity,) or request_indices.dtype != torch.int32:
        raise ValueError("Uniform sparse request indices must match batch capacity.")
    if public_slot_mapping.dtype != torch.int32 or layer_slot_mapping.dtype != torch.int32:
        raise TypeError("Uniform sparse slot mappings must use int32.")
    if free_slots.ndim != 2 or free_slots.dtype != torch.int32:
        raise TypeError("Uniform sparse free slots must be rank-2 int32.")
    if slot_table.ndim != 3 or slot_table.dtype != torch.int32:
        raise TypeError("Uniform sparse slot table must be rank-3 int32.")
    if layer_free_starts.shape != (num_layers,) or layer_free_starts.dtype != torch.int32:
        raise ValueError("Uniform sparse free starts must contain one int32 per layer.")
    if active_count.shape != (1,) or active_count.dtype != torch.int32:
        raise ValueError("Uniform sparse active count must be one int32 scalar tensor.")
    if transformer_layer_ids.ndim != 1 or transformer_layer_ids.dtype != torch.int32:
        raise TypeError("Uniform sparse transformer layer ids must be 1D int32.")
    if transformer_layer_ids.numel() != free_slots.shape[0] or (
        transformer_layer_ids.numel() != slot_table.shape[0]
    ):
        raise ValueError("Uniform sparse KV and transformer layer mappings must match.")
    tensors = (
        slot_table,
        layer_slot_mapping,
        public_slot_mapping,
        context_lens,
        request_indices,
        layer_free_starts,
        active_count,
        transformer_layer_ids,
    )
    if any(tensor.device != free_slots.device for tensor in tensors):
        raise ValueError("Uniform sparse graph metadata must share one device.")

    active_rows = torch.arange(int(active_count[0])) if free_slots.device.type == "cpu" else None
    if active_rows is not None:
        for kv_layer, transformer_layer in enumerate(transformer_layer_ids.tolist()):
            start = int(layer_free_starts[transformer_layer])
            slots = torch.full((batch_capacity,), -1, dtype=torch.int32)
            slots[active_rows] = free_slots[kv_layer, start + active_rows]
            layer_slot_mapping[transformer_layer].copy_(slots)
            rows = request_indices[active_rows].long()
            columns = context_lens[active_rows].long() - 1
            slot_table[kv_layer, rows, columns] = slots[active_rows]
            if kv_layer == 0:
                public_slot_mapping.copy_(slots)
        return

    block_size = 128
    grid = (
        int(transformer_layer_ids.numel()),
        triton.cdiv(batch_capacity, block_size),
    )
    _publish_uniform_sparse_decode_slots_kernel[grid](
        free_slots,
        slot_table,
        layer_slot_mapping,
        public_slot_mapping,
        context_lens,
        request_indices,
        layer_free_starts,
        active_count,
        transformer_layer_ids,
        free_slots.stride(0),
        slot_table.stride(0),
        slot_table.stride(1),
        slot_table.stride(2),
        layer_slot_mapping.stride(0),
        batch_capacity=batch_capacity,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


__all__ = ["publish_uniform_sparse_decode_graph_slots"]
