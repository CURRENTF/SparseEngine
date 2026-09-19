from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_context_attention_kernel(
    q,
    k,
    v,
    output,
    q_start,
    context_lens,
    cached_prefix_lens,
    active_slots,
    req_indices,
    attn_score,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_ot,
    stride_oh,
    stride_sb,
    stride_ss,
    stride_asb,
    stride_ash,
    stride_asl,
    group_size,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW: tl.constexpr,
    SCORE_MODE: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // NUM_HEADS
    query_head = batch_head % NUM_HEADS
    kv_head = query_head // group_size
    query_start = tl.load(q_start + batch)
    prefix_len = tl.load(cached_prefix_lens + batch)
    query_len = tl.load(context_lens + batch) - prefix_len
    request = tl.load(req_indices + batch)
    query_positions = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    query = tl.load(
        q
        + (query_start + query_positions[:, None]) * stride_qt
        + query_head * stride_qh
        + dims[None, :],
        mask=query_positions[:, None] < query_len,
        other=0.0,
    )
    max_logit = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    max_key = tl.minimum(
        prefix_len + (query_block + 1) * BLOCK_M, prefix_len + query_len
    )
    for key_start in range(0, max_key, BLOCK_N):
        key_positions = key_start + tl.arange(0, BLOCK_N)
        slots = tl.load(
            active_slots + request * stride_sb + key_positions * stride_ss,
            mask=key_positions < max_key,
            other=0,
        )
        key = tl.load(
            k + slots[None, :] * stride_kt + kv_head * stride_kh + dims[:, None],
            mask=key_positions[None, :] < max_key,
            other=0.0,
        )
        logits = tl.dot(query, key) * 1.4426950408889634
        absolute_queries = prefix_len + query_positions[:, None]
        visible = key_positions[None, :] <= absolute_queries
        if WINDOW > 0:
            visible &= key_positions[None, :] > absolute_queries - WINDOW
        if SCORE_MODE == 3:
            score = tl.sum(
                tl.where(visible, logits * 0.6931471805599453, 0.0),
                axis=0,
            )
            tl.atomic_add(
                attn_score
                + batch * stride_asb
                + query_head * stride_ash
                + key_positions * stride_asl,
                score,
                mask=key_positions < max_key,
            )
        elif SCORE_MODE == 2:
            score = (
                tl.sum(
                    tl.where(visible, logits * 0.6931471805599453, 0.0),
                    axis=0,
                )
                / query_len
            )
            tl.atomic_max(
                attn_score + batch * stride_asb + key_positions * stride_asl,
                score,
                mask=key_positions < max_key,
            )
        logits = tl.where(visible, logits, -float("inf"))
        has_visible_key = tl.max(visible.to(tl.int32), axis=1) > 0
        block_max = tl.max(logits, axis=1)
        new_max = tl.where(
            has_visible_key, tl.maximum(max_logit, block_max), max_logit
        )
        probabilities = tl.where(
            visible, tl.exp2(logits - new_max[:, None]), 0.0
        )
        correction = tl.where(
            has_visible_key, tl.exp2(max_logit - new_max), 1.0
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        value = tl.load(
            v + slots[:, None] * stride_vt + kv_head * stride_vh + dims[None, :],
            mask=key_positions[:, None] < max_key,
            other=0.0,
        )
        accumulator = tl.dot(probabilities.to(value.dtype), value, accumulator)
        max_logit = new_max
    output_positions = query_start + query_positions
    tl.store(
        output
        + output_positions[:, None] * stride_ot
        + query_head * stride_oh
        + dims[None, :],
        accumulator / denominator[:, None],
        mask=query_positions[:, None] < query_len,
    )


@torch.no_grad()
def gemma4_context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    req_indices: torch.Tensor,
    q_start: torch.Tensor,
    context_lens: torch.Tensor,
    cached_prefix_lens: torch.Tensor,
    max_query_len: int,
    active_slots: torch.Tensor,
    *,
    sliding_window: int | None,
    attn_score: torch.Tensor | None = None,
) -> None:
    head_dim = int(q.shape[-1])
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or output.shape != q.shape:
        raise ValueError("Gemma 4 attention requires matching rank-3 Q/K/V/output.")
    if head_dim not in {256, 512} or k.shape[-1] != head_dim:
        raise ValueError(
            f"Gemma 4 attention requires head_dim 256 or 512, got {head_dim}."
        )
    if not all(t.is_cuda for t in (q, k, v, output)):
        raise TypeError("Gemma 4 attention requires CUDA tensors.")
    if q.dtype not in {torch.float16, torch.bfloat16} or any(
        t.dtype != q.dtype for t in (k, v, output)
    ):
        raise TypeError("Gemma 4 attention requires matching FP16 or BF16 tensors.")
    if any(t.stride(-1) != 1 for t in (q, k, v, output)):
        raise ValueError(
            "Gemma 4 attention requires contiguous BF16/FP16 head dimensions."
        )
    if int(q.shape[1]) % int(k.shape[1]):
        raise ValueError("Gemma 4 attention requires divisible Q and KV heads.")
    block_m = 32 if head_dim == 256 else 16
    block_n = block_m
    if attn_score is not None and attn_score.dim() not in {2, 3}:
        raise ValueError(
            "Gemma 4 prefill attention scores must be [B, L] or [B, H, L], "
            f"got {tuple(attn_score.shape)}."
        )
    score = context_lens if attn_score is None else attn_score
    score_head_stride = score.stride(1) if score.dim() == 3 else 0
    score_length_stride = score.stride(-1)
    batch, num_heads = int(context_lens.numel()), int(q.shape[1])
    _gemma4_context_attention_kernel[
        (triton.cdiv(int(max_query_len), block_m), batch * num_heads)
    ](
        q,
        k,
        v,
        output,
        q_start,
        context_lens,
        cached_prefix_lens,
        active_slots,
        req_indices,
        score,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        output.stride(0),
        output.stride(1),
        active_slots.stride(0),
        active_slots.stride(1),
        score.stride(0),
        score_head_stride,
        score_length_stride,
        int(q.shape[1]) // int(k.shape[1]),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        WINDOW=int(sliding_window or 0),
        SCORE_MODE=0 if attn_score is None else attn_score.dim(),
        num_warps=8,
        num_stages=1,
    )
