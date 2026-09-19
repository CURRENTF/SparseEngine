"""Launch GPU-indexed physical KV transfers and cached miss gathers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sparseengine.kernels.triton.indexed_host_copy import (
    _append,
    _copy_rows,
    _gather,
    _gather_cached,
    _gather_prefill,
    _gather_prefill_history,
    _transfer,
)


def make_pointer_table(tensors, *, device) -> torch.Tensor:
    """Bind contiguous FP16/BF16 components for GPU-indexed host transfers.

    Call during storage preparation, outside graph capture. The storage owner
    must retain every tensor while a transfer or captured graph can use the
    table. Component widths may differ (for example MLA latent and RoPE).
    """
    tensors = tuple(tensors)
    if not tensors or tensors[0].dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Indexed host transfers require FP16/BF16 components.")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Indexed host pointer tables require a CUDA device.")
    for tensor in tensors:
        if tensor.dtype != tensors[0].dtype or not tensor.is_contiguous():
            raise ValueError(
                "Indexed transfer components must be contiguous and share a dtype."
            )
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError("Indexed transfer components must be on CPU or CUDA.")
        if tensor.device.type == "cpu" and not tensor.is_pinned():
            raise ValueError("GPU-indexed host transfers require pinned CPU components.")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if any(tensor.is_cuda and tensor.device.index != device_index for tensor in tensors):
        raise ValueError("Indexed transfer components must belong to the target CUDA device.")
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors], dtype=torch.uint64, device=device
    )


def store_rows(
    source: torch.Tensor,
    destination_ptrs: torch.Tensor,
    slots: torch.Tensor,
    component: int,
    slot_map=None,
) -> None:
    """Scatter current rows; negative slots are inactive.

    If supplied, slot_map is reset to identity at written slots by component
    zero. This fused mapping update is opt-in, not a generic copy side effect.
    The caller orders publication after all component writes before any reader.
    """
    width = source.shape[-2] * source.shape[-1]
    _copy_rows[(source.shape[0], triton.cdiv(width, 256))](
        source,
        destination_ptrs,
        slots,
        slot_map,
        width,
        source.stride(0),
        component,
        256,
    )


def gather_rows(
    source_ptrs: torch.Tensor,
    destination: torch.Tensor,
    table: torch.Tensor,
    rows: torch.Tensor,
    lengths: torch.Tensor,
    *,
    capacity: int,
    component: int,
    skip_last: bool = False,
    slot_map=None,
    block_budget: int = 0,
    exclude_slots=None,
    plan=None,
    miss_tokens=None,
    miss_counts=None,
) -> None:
    """Gather a complete view or fill cache misses within a fixed launch budget.

    Zero leaves the grid unbounded. Plain gathers retain at least one block
    per request; cached gathers share the budget across all request misses.
    A cached plan encodes a miss destination as -slot-1 and a hit as slot;
    miss_tokens/miss_counts describe compact selected-token indices per row.
    The caller owns selection and replacement policy. All dynamic metadata
    stays in fixed-address GPU tensors for replay. No allocation occurs here.
    """
    width = destination.shape[-2] * destination.shape[-1]
    blocks = triton.cdiv(capacity * width, 4096)
    batch = rows.numel()
    if plan is not None:
        _gather_cached[(block_budget or batch * blocks,)](
            source_ptrs,
            destination,
            table,
            rows,
            lengths,
            slot_map,
            exclude_slots,
            plan,
            miss_tokens,
            miss_counts,
            table.stride(0),
            width,
            capacity,
            component,
            skip_last,
            batch,
            triton.next_power_of_2(batch),
            4096,
        )
        return
    if block_budget:
        blocks = min(blocks, max(1, block_budget // batch))
    _gather[(rows.numel(), blocks)](
        source_ptrs,
        destination,
        table,
        rows,
        lengths,
        slot_map,
        exclude_slots,
        table.stride(0),
        width,
        capacity,
        component,
        skip_last,
        4096,
    )


def gather_prefill_rows(
    source_ptrs,
    current,
    destination,
    table,
    rows,
    lengths,
    cu_query,
    slot_map,
    *,
    capacity,
    component,
):
    """Restore full history while reading the current chunk directly from GPU."""
    width = destination.shape[-2] * destination.shape[-1]
    _gather_prefill[(rows.numel(), triton.cdiv(capacity * width, 4096))](
        source_ptrs, current, destination, table, rows, lengths, cu_query, slot_map,
        table.stride(0), current.stride(0), width, capacity, component, 4096,
    )


def append_rows(
    source: torch.Tensor,
    destination: torch.Tensor,
    lengths: torch.Tensor,
    write_slots: torch.Tensor,
    capacity: int,
    *,
    table=None,
    rows=None,
    plan=None,
) -> None:
    """Write current GPU rows into a contiguous view or encoded cache plan.

    A supplied table locates current tokens when they are not necessarily the
    last selected entry. Negative write slots denote inactive graph padding.
    """
    width = source.shape[-2] * source.shape[-1]
    _append[(source.shape[0], triton.cdiv(width, 256))](
        source,
        destination,
        lengths,
        write_slots,
        table,
        rows,
        plan,
        0 if table is None else table.stride(0),
        triton.next_power_of_2(capacity),
        width,
        source.stride(0),
        capacity,
        256,
    )


def transfer_rows(
    source_ptrs,
    destination_ptrs,
    source_slots,
    destination_slots,
    *,
    width,
    dtype,
    component,
    slot_map=None,
):
    if source_slots.numel():
        _transfer[(source_slots.numel(), triton.cdiv(width, 256))](
            source_ptrs,
            destination_ptrs,
            source_slots,
            destination_slots,
            slot_map,
            width,
            component,
            tl.bfloat16 if dtype == torch.bfloat16 else tl.float16,
            256,
        )


def transfer_components(
    source_ptrs,
    destination_ptrs,
    source_slots,
    destination_slots,
    *,
    width,
    dtype,
):
    """Copy equally shaped components sharing the same row index vectors."""
    if source_slots.numel():
        _transfer[(source_slots.numel(), triton.cdiv(width, 256), source_ptrs.numel())](
            source_ptrs,
            destination_ptrs,
            source_slots,
            destination_slots,
            None,
            width,
            None,
            tl.bfloat16 if dtype == torch.bfloat16 else tl.float16,
            256,
        )


def gather_prefill_history(
    source_ptrs, destination, table, rows, lengths, cu_query, slot_map, *, component
):
    """Restore only historical rows, sharing a bounded grid across requests."""
    width = destination.shape[-2] * destination.shape[-1]
    _gather_prefill_history[(32,)](
        source_ptrs, destination, table, rows, lengths, cu_query, slot_map,
        table.stride(0), width, component, rows.numel(),
        triton.next_power_of_2(rows.numel()), 4096,
    )


def scatter_prefill_current(source, destination, slots):
    """Write explicit or MLA current KV into the physical prefill view."""
    width = destination.shape[-2] * destination.shape[-1]
    _copy_rows[(source.shape[0], triton.cdiv(width, 256))](
        source, destination, slots, None, width,
        source.stride(0), 0, 256, DIRECT=True,
    )
