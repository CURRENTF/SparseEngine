from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _compare_swap(value_a, id_a, value_b, id_b, direction: tl.constexpr):
    a_is_nan = value_a != value_a
    b_is_nan = value_b != value_b
    a_greater = (a_is_nan & ~b_is_nan) | (value_a > value_b)
    swap = a_greater if direction else ~a_greater
    return (
        tl.where(swap, value_b, value_a),
        tl.where(swap, id_b, id_a),
        tl.where(swap, value_a, value_b),
        tl.where(swap, id_a, id_b),
    )


@triton.jit
def _descending_merge8(
    v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7
):
    v0, i0, v4, i4 = _compare_swap(v0, i0, v4, i4, False)
    v1, i1, v5, i5 = _compare_swap(v1, i1, v5, i5, False)
    v2, i2, v6, i6 = _compare_swap(v2, i2, v6, i6, False)
    v3, i3, v7, i7 = _compare_swap(v3, i3, v7, i7, False)
    v0, i0, v2, i2 = _compare_swap(v0, i0, v2, i2, False)
    v1, i1, v3, i3 = _compare_swap(v1, i1, v3, i3, False)
    v4, i4, v6, i6 = _compare_swap(v4, i4, v6, i6, False)
    v5, i5, v7, i7 = _compare_swap(v5, i5, v7, i7, False)
    v0, i0, v1, i1 = _compare_swap(v0, i0, v1, i1, False)
    v2, i2, v3, i3 = _compare_swap(v2, i2, v3, i3, False)
    v4, i4, v5, i5 = _compare_swap(v4, i4, v5, i5, False)
    v6, i6, v7, i7 = _compare_swap(v6, i6, v7, i7, False)
    return v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7


@triton.jit
def _torch_sort8(
    v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7
):
    # Mirrors SortUtils.cuh bitonicSort<32>, reduced to its eight valid slots.
    v0, i0, v1, i1 = _compare_swap(v0, i0, v1, i1, False)
    v2, i2, v3, i3 = _compare_swap(v2, i2, v3, i3, True)
    v4, i4, v5, i5 = _compare_swap(v4, i4, v5, i5, False)
    v6, i6, v7, i7 = _compare_swap(v6, i6, v7, i7, True)
    v0, i0, v2, i2 = _compare_swap(v0, i0, v2, i2, False)
    v1, i1, v3, i3 = _compare_swap(v1, i1, v3, i3, False)
    v4, i4, v6, i6 = _compare_swap(v4, i4, v6, i6, True)
    v5, i5, v7, i7 = _compare_swap(v5, i5, v7, i7, True)
    v0, i0, v1, i1 = _compare_swap(v0, i0, v1, i1, False)
    v2, i2, v3, i3 = _compare_swap(v2, i2, v3, i3, False)
    v4, i4, v5, i5 = _compare_swap(v4, i4, v5, i5, True)
    v6, i6, v7, i7 = _compare_swap(v6, i6, v7, i7, True)

    values_and_ids = (
        v0, i0, v1, i1, v2, i2, v3, i3,
        v4, i4, v5, i5, v6, i6, v7, i7,
    )
    values_and_ids = _descending_merge8(*values_and_ids)
    values_and_ids = _descending_merge8(*values_and_ids)
    return _descending_merge8(*values_and_ids)


@triton.jit
def _gather_candidate(
    values,
    offsets,
    greater_mask,
    equal_mask,
    greater_rank,
    equal_rank,
    num_greater,
    slot: tl.constexpr,
    num_experts: tl.constexpr,
):
    use_greater = slot < num_greater
    rank = tl.where(use_greater, slot, slot - num_greater)
    mask = tl.where(
        use_greater,
        greater_mask & (greater_rank == rank),
        equal_mask & (equal_rank == rank),
    )
    expert_id = tl.min(tl.where(mask, offsets, num_experts), axis=0)
    value = tl.sum(tl.where(mask, values, 0.0), axis=0)
    return value, expert_id


@triton.jit
def _topk_softmax_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    stride_logits_m,
    stride_weights_m,
    stride_ids_m,
    NUM_EXPERTS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, NUM_EXPERTS)
    logits = tl.load(logits_ptr + row * stride_logits_m + offsets).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    probabilities = libdevice.exp(logits - row_max)
    probabilities /= tl.sum(probabilities, axis=0)

    # PyTorch applies topk to FP32 softmax output. NaN probabilities compare
    # ahead of finite values in its descending topk implementation.
    selection_values = tl.where(
        probabilities == probabilities, probabilities, float("inf")
    )
    threshold = tl.min(tl.topk(selection_values, 8), axis=0)
    greater_mask = selection_values > threshold
    equal_mask = selection_values == threshold
    greater_rank = tl.cumsum(greater_mask.to(tl.int32), axis=0) - 1
    equal_rank = tl.cumsum(equal_mask.to(tl.int32), axis=0) - 1
    num_greater = tl.sum(greater_mask.to(tl.int32), axis=0)

    v0, i0 = _gather_candidate(
        probabilities, offsets, greater_mask, equal_mask,
        greater_rank, equal_rank, num_greater, 0, NUM_EXPERTS,
    )
    v1, i1 = _gather_candidate(
        probabilities, offsets, greater_mask, equal_mask,
        greater_rank, equal_rank, num_greater, 1, NUM_EXPERTS,
    )
    v2, i2 = _gather_candidate(
        probabilities, offsets, greater_mask, equal_mask,
        greater_rank, equal_rank, num_greater, 2, NUM_EXPERTS,
    )
    v3, i3 = _gather_candidate(
        probabilities, offsets, greater_mask, equal_mask,
        greater_rank, equal_rank, num_greater, 3, NUM_EXPERTS,
    )
    v4, i4 = _gather_candidate(
        probabilities, offsets, greater_mask, equal_mask,
        greater_rank, equal_rank, num_greater, 4, NUM_EXPERTS,
    )
    v5, i5 = _gather_candidate(
        probabilities, offsets, greater_mask, equal_mask,
        greater_rank, equal_rank, num_greater, 5, NUM_EXPERTS,
    )
    v6, i6 = _gather_candidate(
        probabilities, offsets, greater_mask, equal_mask,
        greater_rank, equal_rank, num_greater, 6, NUM_EXPERTS,
    )
    v7, i7 = _gather_candidate(
        probabilities, offsets, greater_mask, equal_mask,
        greater_rank, equal_rank, num_greater, 7, NUM_EXPERTS,
    )
    values_and_ids = _torch_sort8(
        v0, i0, v1, i1, v2, i2, v3, i3,
        v4, i4, v5, i5, v6, i6, v7, i7,
    )
    v0, i0, v1, i1, v2, i2, v3, i3 = values_and_ids[:8]
    v4, i4, v5, i5, v6, i6, v7, i7 = values_and_ids[8:]

    denominator = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7

    weights_base = weights_ptr + row * stride_weights_m
    ids_base = ids_ptr + row * stride_ids_m
    tl.store(weights_base + 0, v0 / denominator)
    tl.store(weights_base + 1, v1 / denominator)
    tl.store(weights_base + 2, v2 / denominator)
    tl.store(weights_base + 3, v3 / denominator)
    tl.store(weights_base + 4, v4 / denominator)
    tl.store(weights_base + 5, v5 / denominator)
    tl.store(weights_base + 6, v6 / denominator)
    tl.store(weights_base + 7, v7 / denominator)
    tl.store(ids_base + 0, i0)
    tl.store(ids_base + 1, i1)
    tl.store(ids_base + 2, i2)
    tl.store(ids_base + 3, i3)
    tl.store(ids_base + 4, i4)
    tl.store(ids_base + 5, i5)
    tl.store(ids_base + 6, i6)
    tl.store(ids_base + 7, i7)


def topk_softmax(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not router_logits.is_cuda:
        raise ValueError("Triton topk_softmax requires CUDA router_logits.")
    if router_logits.ndim != 2:
        raise ValueError(
            "router_logits must have shape [tokens, experts], got "
            f"{tuple(router_logits.shape)}."
        )
    if router_logits.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "Triton topk_softmax supports BF16 and FP16 logits, got "
            f"{router_logits.dtype}."
        )
    if router_logits.stride(1) != 1:
        raise ValueError("Triton topk_softmax requires a contiguous expert dimension.")
    if int(router_logits.shape[0]) <= 0:
        raise ValueError("Triton topk_softmax requires at least one token.")
    num_experts = int(router_logits.shape[1])
    if num_experts not in {128, 256} or int(top_k) != 8:
        raise ValueError(
            "Triton topk_softmax currently requires num_experts=128 or "
            "num_experts=256 and top_k=8, "
            f"got num_experts={router_logits.shape[1]}, top_k={top_k}."
        )

    if not norm_topk_prob:
        probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(probabilities, top_k, dim=-1)
        return weights.to(router_logits.dtype), ids

    num_tokens = int(router_logits.shape[0])
    weights = torch.empty(
        (num_tokens, 8), dtype=router_logits.dtype, device=router_logits.device
    )
    ids = torch.empty((num_tokens, 8), dtype=torch.int64, device=router_logits.device)
    _topk_softmax_kernel[(num_tokens,)](
        router_logits,
        weights,
        ids,
        router_logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        NUM_EXPERTS=num_experts,
        num_warps=1,
    )
    return weights, ids
