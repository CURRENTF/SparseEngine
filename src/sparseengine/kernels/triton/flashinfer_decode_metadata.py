from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_page_indices_kernel(
    active_slots,
    request_indices,
    context_lens,
    packed_indices,
    active_slots_stride_0: tl.constexpr,
    active_slots_stride_1: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_block_idx = tl.program_id(1)

    batch_offsets = tl.arange(0, BATCH_BLOCK)
    lengths = tl.load(context_lens + batch_offsets, mask=batch_offsets < BATCH_SIZE)
    page_counts = (lengths + PAGE_SIZE - 1) // PAGE_SIZE
    packed_start = tl.sum(
        tl.where(batch_offsets < batch_idx, page_counts, 0)
    )

    page_offsets = token_block_idx * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
    context_len = tl.load(context_lens + batch_idx)
    page_count = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    request_idx = tl.load(request_indices + batch_idx)
    valid = (page_offsets < page_count) & (page_offsets < PAGE_CAPACITY)
    slots = tl.load(
        active_slots
        + request_idx * active_slots_stride_0
        + page_offsets * active_slots_stride_1,
        mask=valid,
    )
    tl.store(packed_indices + packed_start + page_offsets, slots, mask=valid)


def pack_flashinfer_page_indices(
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    packed_indices: torch.Tensor,
    *,
    context_capacity: int,
    page_size: int = 1,
) -> None:
    """Pack a layer's canonical page table into graph-stable storage."""

    if active_slots.ndim != 2 or active_slots.dtype != torch.int32:
        raise TypeError("FlashInfer graph decode requires a rank-2 int32 slot table.")
    if request_indices.ndim != 1 or request_indices.dtype != torch.int32:
        raise TypeError("FlashInfer graph decode requires int32 request indices.")
    if context_lens.ndim != 1 or context_lens.dtype != torch.int32:
        raise TypeError("FlashInfer graph decode requires int32 context lengths.")
    if request_indices.shape != context_lens.shape:
        raise ValueError("FlashInfer graph request indices and context lengths must match.")

    batch_size = int(context_lens.numel())
    context_capacity = int(context_capacity)
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError(f"FlashInfer graph page_size must be positive, got {page_size}.")
    page_capacity = (context_capacity + page_size - 1) // page_size
    if page_capacity <= 0 or page_capacity > int(active_slots.shape[1]):
        raise ValueError(
            "FlashInfer graph context capacity is outside the slot table: "
            f"tokens={context_capacity} pages={page_capacity} "
            f"width={int(active_slots.shape[1])}."
        )
    if packed_indices.ndim != 1 or packed_indices.dtype != torch.int32:
        raise TypeError("FlashInfer graph packed indices must be a 1D int32 tensor.")
    required = batch_size * page_capacity
    if int(packed_indices.numel()) < required:
        raise ValueError(
            "FlashInfer graph packed-index buffer is too small: "
            f"required={required} actual={int(packed_indices.numel())}."
        )

    page_block = 128
    _pack_page_indices_kernel[
        (batch_size, triton.cdiv(page_capacity, page_block))
    ](
        active_slots,
        request_indices,
        context_lens,
        packed_indices,
        active_slots.stride(0),
        active_slots.stride(1),
        BATCH_SIZE=batch_size,
        BATCH_BLOCK=triton.next_power_of_2(batch_size),
        PAGE_CAPACITY=page_capacity,
        PAGE_SIZE=page_size,
        PAGE_BLOCK=page_block,
    )


__all__ = ["pack_flashinfer_page_indices"]
