from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _exact_select_kernel(
    scores,
    page_table,
    lengths,
    output,
    score_stride,
    page_stride,
    output_stride,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    length = tl.load(lengths + row)
    valid = offsets < length
    scores_row = tl.load(
        scores + row * score_stride + offsets,
        mask=valid,
        other=-float("inf"),
    )
    scores_row = tl.where(scores_row == 0.0, 0.0, scores_row)
    score_bits = scores_row.to(tl.uint16, bitcast=True).to(tl.uint32)
    ordered_score = tl.where(
        (score_bits & 0x8000) != 0,
        (~score_bits) & 0xFFFF,
        score_bits | 0x8000,
    )
    inverse_index = 0xFFFFFFFF - offsets.to(tl.uint32)
    keys = (ordered_score.to(tl.uint64) << 32) | inverse_index.to(tl.uint64)
    ranked_keys = tl.sort(tl.where(valid, keys, 0), dim=0, descending=True)
    ranked_indices = 0xFFFFFFFF - (ranked_keys & 0xFFFFFFFF).to(tl.uint32)
    selected_indices = tl.where(offsets < K, ranked_indices, BLOCK_N)
    selected_indices = tl.sort(selected_indices, dim=0, descending=False)
    selected_valid = (offsets < K) & (selected_indices < length)
    selected_pages = tl.load(
        page_table + row * page_stride + selected_indices.to(tl.int32),
        mask=selected_valid,
        other=-1,
    )
    tl.store(
        output + row * output_stride + offsets,
        selected_pages,
        mask=offsets < K,
    )


@triton.jit
def _fused_exact_select_paged_view_kernel(
    scores,
    row_page_slots,
    previous_page_counts,
    context_lens,
    output_page_table,
    output_req_indices,
    output_context_lens,
    output_page_counts,
    output_last_page_lens,
    score_stride,
    page_stride,
    output_stride,
    K: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    TOKEN_BUDGET: tl.constexpr,
    USE_DENSE_FALLBACK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    previous_count = tl.load(previous_page_counts + row)
    valid = offsets < previous_count
    scores_row = tl.load(
        scores + row * score_stride + offsets,
        mask=valid,
        other=-float("inf"),
    )

    # Compose a sortable key so equal BF16 scores prefer the smaller logical
    # page index. Signed zero is normalized before constructing the key.
    scores_row = tl.where(scores_row == 0.0, 0.0, scores_row)
    score_bits = scores_row.to(tl.uint16, bitcast=True).to(tl.uint32)
    ordered_score = tl.where(
        (score_bits & 0x8000) != 0,
        (~score_bits) & 0xFFFF,
        score_bits | 0x8000,
    )
    inverse_index = 0xFFFFFFFF - offsets.to(tl.uint32)
    keys = (ordered_score.to(tl.uint64) << 32) | inverse_index.to(tl.uint64)
    ranked_keys = tl.sort(tl.where(valid, keys, 0), dim=0, descending=True)

    ranked_indices = 0xFFFFFFFF - (ranked_keys & 0xFFFFFFFF).to(tl.uint32)
    selected_indices = tl.where(offsets < K, ranked_indices, BLOCK_N)
    # The attention view follows logical page order, not descending score order.
    selected_indices = tl.sort(selected_indices, dim=0, descending=False)
    selected_pages = tl.load(
        row_page_slots + row * page_stride + selected_indices.to(tl.int32),
        mask=(offsets < K) & (selected_indices < previous_count),
        other=0,
    )

    context_len = tl.load(context_lens + row)
    num_pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    last_page = tl.load(
        row_page_slots + row * page_stride + tl.maximum(num_pages - 1, 0),
        mask=num_pages > 0, other=0,
    )
    last_page_len = tl.where(num_pages > 0, context_len - (num_pages - 1) * PAGE_SIZE, 0)
    output_pages = tl.where(offsets < K, selected_pages, last_page)
    output_count = K + 1
    output_len = K * PAGE_SIZE + last_page_len
    if USE_DENSE_FALLBACK:
        use_dense = (context_len <= TOKEN_BUDGET) | (num_pages <= K + 1)
        dense_pages = tl.load(
            row_page_slots + row * page_stride + offsets,
            mask=(offsets < OUTPUT_WIDTH) & (offsets < num_pages), other=0,
        )
        output_pages = tl.where(use_dense, dense_pages, output_pages)
        output_count = tl.where(use_dense, num_pages, K + 1)
        output_len = tl.where(use_dense, context_len, output_len)
    tl.store(
        output_page_table + row * output_stride + offsets,
        tl.where(offsets < output_count, output_pages, 0),
        mask=offsets < OUTPUT_WIDTH,
    )
    tl.store(output_req_indices + row, row)
    tl.store(output_context_lens + row, output_len)
    tl.store(output_page_counts + row, output_count)
    tl.store(output_last_page_lens + row, last_page_len)


def fused_exact_select_quest_paged_view(
    scores: torch.Tensor,
    row_page_slots: torch.Tensor,
    previous_page_counts: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    k: int,
    page_size: int,
    output_page_table: torch.Tensor,
    output_req_indices: torch.Tensor,
    output_context_lens: torch.Tensor,
    output_page_counts: torch.Tensor,
    output_last_page_lens: torch.Tensor,
    token_budget: int = 0,
    use_dense_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact BF16 page selection fused with sparse paged-view finalization."""

    batch_size, width = map(int, scores.shape)
    k = int(k)
    outputs = (
        output_page_table,
        output_req_indices,
        output_context_lens,
        output_page_counts,
        output_last_page_lens,
    )
    if not scores.is_cuda or scores.dtype != torch.bfloat16:
        raise TypeError("Fused QuEST selection requires CUDA BF16 scores.")
    if scores.ndim != 2 or not scores.is_contiguous():
        raise ValueError("Fused QuEST scores must be contiguous and rank 2.")
    if (
        row_page_slots.shape != scores.shape
        or row_page_slots.dtype != torch.int32
        or not row_page_slots.is_contiguous()
        or row_page_slots.device != scores.device
    ):
        raise TypeError(
            "Fused QuEST page table must be contiguous int32 and match scores."
        )
    if not 0 < k < width:
        raise ValueError(f"Fused QuEST selection requires 0 < k < {width}, got {k}.")
    if int(page_size) <= 0:
        raise ValueError("Fused QuEST selection requires a positive page size.")
    if width > 512:
        raise ValueError(
            f"Profiled fused QuEST selection supports width <= 512, got {width}."
        )
    output_width = int(output_page_table.shape[1])
    required_width = max(k + 1, triton.cdiv(int(token_budget), int(page_size))) if use_dense_fallback else k + 1
    if output_page_table.shape != (batch_size, output_width) or output_width < required_width:
        raise ValueError("Fused QuEST output page table has insufficient capacity.")
    for name, tensor in {
        "previous_page_counts": previous_page_counts,
        "context_lens": context_lens,
        "output_req_indices": output_req_indices,
        "output_context_lens": output_context_lens,
        "output_page_counts": output_page_counts,
        "output_last_page_lens": output_last_page_lens,
    }.items():
        if tensor.shape != (batch_size,) or tensor.dtype != torch.int32:
            raise TypeError(
                f"{name} must be contiguous int32 with shape [{batch_size}]."
            )
        if not tensor.is_contiguous() or tensor.device != scores.device:
            raise ValueError(f"{name} must be contiguous and share the score device.")
    if output_page_table.dtype != torch.int32 or not output_page_table.is_contiguous():
        raise TypeError("Fused QuEST output page table must be contiguous int32.")
    if output_page_table.device != scores.device:
        raise ValueError("Fused QuEST outputs must share the score device.")

    block_n = triton.next_power_of_2(max(width, output_width))
    _fused_exact_select_paged_view_kernel[(batch_size,)](
        scores,
        row_page_slots,
        previous_page_counts,
        context_lens,
        *outputs,
        scores.stride(0),
        row_page_slots.stride(0),
        output_page_table.stride(0),
        K=k,
        PAGE_SIZE=int(page_size),
        OUTPUT_WIDTH=output_width,
        TOKEN_BUDGET=int(token_budget),
        USE_DENSE_FALLBACK=bool(use_dense_fallback),
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=1,
    )
    return outputs


def exact_select_quest_pages(
    scores: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Select exact BF16 top-k pages with stable small-index ties."""

    batch_size, width = map(int, scores.shape)
    k = int(k)
    if not scores.is_cuda or scores.dtype != torch.bfloat16:
        raise TypeError("Triton exact QuEST selection requires CUDA BF16 scores.")
    if scores.ndim != 2 or not scores.is_contiguous():
        raise ValueError("Triton exact QuEST scores must be contiguous and rank 2.")
    if (
        page_table.shape != scores.shape
        or page_table.dtype != torch.int32
        or not page_table.is_contiguous()
        or page_table.device != scores.device
    ):
        raise TypeError("Triton exact QuEST page table must be contiguous int32.")
    if lengths.shape != (batch_size,) or lengths.dtype != torch.int32:
        raise TypeError(
            "Triton exact QuEST lengths must be int32 with one value per row."
        )
    if not lengths.is_contiguous() or lengths.device != scores.device:
        raise ValueError(
            "Triton exact QuEST lengths must be contiguous on the score device."
        )
    if not 0 < k <= width or width > 512:
        raise ValueError(
            f"Triton exact QuEST selection requires 0 < k <= width <= 512, got "
            f"k={k}, width={width}."
        )
    output = torch.empty((batch_size, k), dtype=torch.int32, device=scores.device)
    block_n = triton.next_power_of_2(width)
    _exact_select_kernel[(batch_size,)](
        scores,
        page_table,
        lengths,
        output,
        scores.stride(0),
        page_table.stride(0),
        output.stride(0),
        K=k,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=1,
    )
    return output


__all__ = ["exact_select_quest_pages", "fused_exact_select_quest_paged_view"]
