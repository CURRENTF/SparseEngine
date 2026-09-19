from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _h2o_headwise_probability_from_lse_kernel(
    Raw, Lse, Lengths, Score,
    stride_rb, stride_rh, stride_rt, stride_lh, stride_lb, stride_len,
    stride_sb, stride_st,
    HEADS: tl.constexpr, CAPACITY: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr,
):
    batch = tl.program_id(0)
    start = tl.program_id(1) * BLOCK_T
    tokens = start + tl.arange(0, BLOCK_T)
    heads = tl.arange(0, BLOCK_H)
    length = tl.load(Lengths + batch * stride_len)
    if start < length:
        valid = ((heads[:, None] < HEADS) & (tokens[None, :] < length)
                 & (tokens[None, :] < CAPACITY))
        raw = tl.load(Raw + batch * stride_rb + heads[:, None] * stride_rh
                      + tokens[None, :] * stride_rt, mask=valid, other=0.0)
        lse = tl.load(Lse + heads * stride_lh + batch * stride_lb,
                      mask=heads < HEADS, other=0.0)
        probability = tl.where(valid, tl.exp(raw * SCALE - lse[:, None]), 0.0)
        mass = tl.sum(probability, axis=0)
    else:
        mass = tl.full((BLOCK_T,), 0.0, tl.float32)
    tl.store(Score + batch * stride_sb + tokens * stride_st, mass, mask=tokens < CAPACITY)


@torch.no_grad()
def h2o_headwise_probability_from_lse(
    raw_logits: torch.Tensor,
    attention_lse: torch.Tensor,
    context_lens: torch.Tensor,
    score: torch.Tensor,
    *,
    softmax_scale: float,
) -> None:
    """Consume shared [B,H,L] raw QK into per-layer sum_h softmax(QK) [B,L].

    The attention kernel supplies natural-log LSE of scaled QK. Token tiles
    beyond each device length skip logits/LSE reads and overwrite scores with
    zero, including inactive rows. Launch shape depends only on capacity.
    """
    if raw_logits.ndim != 3 or score.ndim != 2:
        raise ValueError("H2O shared logits/probability output must be rank 3/2.")
    batch, heads, capacity = map(int, raw_logits.shape)
    if min(batch, heads, capacity) <= 0 or tuple(score.shape) != (batch, capacity):
        raise ValueError("H2O shared logits and output disagree on positive batch/capacity.")
    if tuple(attention_lse.shape) != (heads, batch) or tuple(context_lens.shape) != (batch,):
        raise ValueError("H2O LSE/lengths must be [heads,batch]/[batch].")
    if any(t.dtype != torch.float32 for t in (raw_logits, attention_lse, score)):
        raise TypeError("H2O shared logits, LSE and reduced scores must be FP32.")
    if context_lens.dtype != torch.int32:
        raise TypeError("H2O context lengths must be int32.")
    if not raw_logits.is_cuda or any(t.device != raw_logits.device for t in (attention_lse, context_lens, score)):
        raise TypeError("H2O shared score tensors must be on the same CUDA device.")
    if not 0 < softmax_scale < float("inf"):
        raise ValueError("H2O softmax scale must be finite and positive.")
    _h2o_headwise_probability_from_lse_kernel[(batch, triton.cdiv(capacity, 128))](
        raw_logits, attention_lse, context_lens, score,
        *raw_logits.stride(), *attention_lse.stride(), context_lens.stride(0), *score.stride(),
        HEADS=heads, CAPACITY=capacity, SCALE=float(softmax_scale),
        BLOCK_H=triton.next_power_of_2(heads), BLOCK_T=128,
        num_warps=4, num_stages=1,
    )


@triton.jit
def _h2o_probability_from_lse_kernel(
    q,
    k,
    attention_lse,
    page_table,
    request_indices,
    context_lens,
    score,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_lseh,
    stride_lseb,
    stride_ptb,
    stride_pts,
    stride_sb,
    stride_ss,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    token_offsets = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, HEAD_DIM)
    context_len = tl.load(context_lens + batch)
    request = tl.load(request_indices + batch)
    token_valid = token_offsets < context_len
    slots = tl.load(
        page_table + request * stride_ptb + token_offsets * stride_pts,
        mask=token_valid,
        other=0,
    )
    query_heads = kv_head * GQA_GROUP + head_offsets
    head_valid = head_offsets < GQA_GROUP
    query = tl.load(
        q
        + batch * stride_qb
        + query_heads[:, None] * stride_qh
        + dim_offsets[None, :] * stride_qd,
        mask=head_valid[:, None],
        other=0.0,
    )
    keys = tl.load(
        k
        + slots[None, :] * stride_ks
        + kv_head * stride_kh
        + dim_offsets[:, None] * stride_kd,
        mask=token_valid[None, :],
        other=0.0,
    )
    row_lse = tl.load(
        attention_lse + query_heads * stride_lseh + batch * stride_lseb,
        mask=head_valid,
        other=0.0,
    )
    logits = tl.dot(query, keys) * SM_SCALE
    probabilities = tl.where(
        head_valid[:, None] & token_valid[None, :],
        tl.exp(logits - row_lse[:, None]),
        0.0,
    )
    token_score = tl.sum(probabilities, axis=0)
    tl.atomic_add(
        score + batch * stride_sb + token_offsets * stride_ss,
        token_score,
        mask=token_valid,
    )


@torch.no_grad()
def h2o_probability_from_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_lse: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    score: torch.Tensor,
    *,
    softmax_scale: float,
) -> None:
    """Reduce one layer's exact decode probabilities into [batch, context]."""

    if q.ndim != 3 or k.ndim != 3 or score.ndim != 2:
        raise ValueError(
            "H2O decode probability scoring expects Q/K/score ranks 3/3/2, got "
            f"{q.ndim}/{k.ndim}/{score.ndim}."
        )
    batch, query_heads, head_dim = map(int, q.shape)
    kv_heads = int(k.shape[1])
    if query_heads % kv_heads:
        raise ValueError(
            f"H2O decode probability scoring requires GQA divisibility: "
            f"{query_heads}/{kv_heads}."
        )
    if tuple(attention_lse.shape) != (query_heads, batch):
        raise ValueError(
            "H2O decode FA3 LSE must be [query_heads, batch], got "
            f"{tuple(attention_lse.shape)}."
        )
    if attention_lse.dtype != torch.float32 or attention_lse.device != q.device:
        raise TypeError("H2O decode FA3 LSE must be FP32 on the query device.")
    if tuple(score.shape[:1]) != (batch,) or score.dtype != torch.float32:
        raise ValueError(
            "H2O decode probability scoring requires FP32 [batch, width] output, "
            f"got shape={tuple(score.shape)} dtype={score.dtype}."
        )
    if q.dtype != k.dtype or q.stride(-1) != 1 or k.stride(-1) != 1:
        raise TypeError("H2O decode probability scoring requires matching contiguous Q/K.")
    if page_table.dtype != torch.int32 or request_indices.dtype != torch.int32:
        raise TypeError("H2O decode probability scoring requires int32 page metadata.")
    if context_lens.dtype != torch.int32:
        raise TypeError("H2O decode probability scoring requires int32 context lengths.")
    if softmax_scale <= 0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}.")

    score.zero_()
    group = query_heads // kv_heads
    block_n = 64
    _h2o_probability_from_lse_kernel[
        (batch, kv_heads, triton.cdiv(int(score.shape[1]), block_n))
    ](
        q,
        k,
        attention_lse,
        page_table,
        request_indices,
        context_lens,
        score,
        *q.stride(),
        *k.stride(),
        *attention_lse.stride(),
        *page_table.stride(),
        *score.stride(),
        GQA_GROUP=group,
        HEAD_DIM=head_dim,
        BLOCK_H=max(16, triton.next_power_of_2(group)),
        BLOCK_N=block_n,
        SM_SCALE=float(softmax_scale),
        num_warps=4,
        num_stages=2,
    )
