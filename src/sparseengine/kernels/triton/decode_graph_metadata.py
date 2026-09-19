from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _publish_decode_slots_kernel(
    slot_table,
    request_indices,
    context_lens,
    write_slots,
    active_mask,
    slot_table_stride_row: tl.constexpr,
    slot_table_stride_token: tl.constexpr,
    batch_capacity,
    HAS_ACTIVE_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < batch_capacity
    if HAS_ACTIVE_MASK:
        active = tl.load(active_mask + offsets, mask=in_bounds, other=False)
    else:
        active = in_bounds
    rows = tl.load(request_indices + offsets, mask=in_bounds, other=0)
    context = tl.load(context_lens + offsets, mask=in_bounds, other=1)
    slots = tl.load(write_slots + offsets, mask=in_bounds, other=0)
    table_offsets = (
        rows * slot_table_stride_row
        + (context - 1) * slot_table_stride_token
    )
    tl.store(slot_table + table_offsets, slots, mask=in_bounds & active)


def publish_decode_graph_slots(
    slot_table: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    write_slots: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> None:
    """Publish successful decode reservations to a fixed-address slot table."""

    if slot_table.ndim != 2:
        raise ValueError(
            f"Decode slot table must be two-dimensional, got {slot_table.shape}."
        )
    if slot_table.dtype != torch.int32:
        raise TypeError(
            f"Decode slot table must use int32 entries, got {slot_table.dtype}."
        )
    inputs = (request_indices, context_lens, write_slots)
    batch_capacity = int(request_indices.numel())
    if any(tensor.ndim != 1 or tensor.numel() != batch_capacity for tensor in inputs):
        raise ValueError("Decode slot publish inputs must share one 1D capacity.")
    if tuple(tensor.dtype for tensor in inputs) != (torch.int32,) * 3:
        raise TypeError(
            "Decode slot publish expects int32 rows/lengths/slots."
        )
    if active_mask is not None and (
        active_mask.ndim != 1
        or active_mask.numel() != batch_capacity
        or active_mask.dtype != torch.bool
    ):
        raise ValueError(
            "Decode slot publish active mask must be bool with the shared 1D capacity."
        )
    device = slot_table.device
    if any(tensor.device != device for tensor in inputs) or (
        active_mask is not None and active_mask.device != device
    ):
        raise ValueError("Decode slot publish inputs and table must share one device.")

    if device.type == "cpu":
        active_rows = (
            torch.arange(batch_capacity)
            if active_mask is None
            else active_mask.nonzero(as_tuple=False).flatten()
        )
        rows = request_indices.index_select(0, active_rows).to(torch.long)
        columns = context_lens.index_select(0, active_rows).to(torch.long) - 1
        slots = write_slots.index_select(0, active_rows)
        slot_table[rows, columns] = slots
        return

    block_size = 128
    grid = (triton.cdiv(batch_capacity, block_size),)
    _publish_decode_slots_kernel[grid](
        slot_table,
        request_indices,
        context_lens,
        write_slots,
        write_slots if active_mask is None else active_mask,
        slot_table.stride(0),
        slot_table.stride(1),
        batch_capacity,
        HAS_ACTIVE_MASK=active_mask is not None,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )
