"""Graph-stable split-KV paged decode attention.

The stable decode kernels intentionally remain unchanged.  This variant fixes
the CUDA launch grid and workspace split dimension while deriving the effective
split ranges from the device-resident context lengths.  The split scheduling
follows the fixed-upper-bound design used by SGLang's Triton decode attention
(reference revision ed0a62e4), adapted to Sparse-Engine's slot-table layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_stage1(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Mid_O,
    Mid_Lse,
    Attn_Score,
    stride_req_b,
    stride_req_s,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_vb,
    stride_vh,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_score_b,
    stride_score_h,
    stride_score_s,
    GQA_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MAX_EFFECTIVE_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
    SCORE_MODE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    split_id = tl.program_id(2)
    kv_head_id = head_id // GQA_GROUP_SIZE

    seq_len = tl.load(B_Seqlen + batch_id)
    requested_splits = tl.cdiv(seq_len, TARGET_TOKENS_PER_SPLIT)
    num_splits = tl.maximum(1, tl.minimum(requested_splits, MAX_EFFECTIVE_SPLITS))
    split_tokens = tl.where(
        requested_splits <= MAX_EFFECTIVE_SPLITS,
        TARGET_TOKENS_PER_SPLIT,
        tl.cdiv(seq_len, num_splits),
    )
    split_start = split_id * split_tokens
    split_end = tl.minimum(split_start + split_tokens, seq_len)
    split_valid = (split_id < num_splits) & (split_start < split_end)
    if not split_valid:
        return

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + batch_id * stride_qb + head_id * stride_qh + offs_d)
    req_id = tl.load(B_req_idx + batch_id)

    max_logit = -float("inf")
    exp_sum = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    block_count = tl.where(split_valid, tl.cdiv(split_end - split_start, BLOCK_N), 0)

    for block_id in range(0, block_count):
        positions = split_start + block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        position_mask = positions < split_end
        slots = tl.load(
            Req_to_tokens + req_id * stride_req_b + positions * stride_req_s,
            mask=position_mask,
            other=0,
        ).to(tl.int64)
        k_offsets = slots[:, None] * stride_kb + kv_head_id * stride_kh + offs_d[None, :]
        v_offsets = slots[:, None] * stride_vb + kv_head_id * stride_vh + offs_d[None, :]
        k = tl.load(K + k_offsets, mask=position_mask[:, None], other=0.0)
        v = tl.load(V + v_offsets, mask=position_mask[:, None], other=0.0)
        logits = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1)
        logits = tl.where(position_mask, logits, -float("inf"))
        if SCORE_MODE == 3:
            score_offsets = (
                batch_id * stride_score_b
                + head_id * stride_score_h
                + positions * stride_score_s
            )
            tl.store(Attn_Score + score_offsets, logits, mask=position_mask)
        elif SCORE_MODE == 2:
            score_offsets = batch_id * stride_score_b + positions * stride_score_s
            tl.atomic_max(Attn_Score + score_offsets, logits, mask=position_mask)
        logits *= sm_scale

        block_max = tl.max(logits, axis=0)
        next_max = tl.maximum(max_logit, block_max)
        old_scale = tl.exp(max_logit - next_max)
        probs = tl.exp(logits - next_max)
        acc = acc * old_scale + tl.sum(probs[:, None] * v, axis=0)
        exp_sum = exp_sum * old_scale + tl.sum(probs, axis=0)
        max_logit = next_max

    mid_offset = (
        batch_id * stride_mid_b
        + head_id * stride_mid_h
        + split_id * stride_mid_s
        + offs_d
    )
    lse_offset = (
        batch_id * stride_lse_b + head_id * stride_lse_h + split_id * stride_lse_s
    )
    safe_sum = tl.where(split_valid, exp_sum, 1.0)
    tl.store(Mid_O + mid_offset, tl.where(split_valid, acc / safe_sum, 0.0))
    tl.store(
        Mid_Lse + lse_offset,
        tl.where(split_valid, max_logit + tl.log(safe_sum), -float("inf")),
    )


@triton.jit
def _paged_grouped_decode_stage1(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Mid_O,
    Mid_Lse,
    Attn_Score,
    stride_req_b,
    stride_req_s,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_vb,
    stride_vh,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_score_b,
    stride_score_h,
    stride_score_s,
    GQA_GROUP_SIZE: tl.constexpr,
    QUERY_HEAD_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MAX_EFFECTIVE_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
    SCORE_MODE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    split_id = tl.program_id(2)
    head_offsets = tl.arange(0, QUERY_HEAD_BLOCK)
    query_heads = kv_head_id * GQA_GROUP_SIZE + head_offsets
    head_mask = head_offsets < GQA_GROUP_SIZE

    seq_len = tl.load(B_Seqlen + batch_id)
    requested_splits = tl.cdiv(seq_len, TARGET_TOKENS_PER_SPLIT)
    num_splits = tl.maximum(1, tl.minimum(requested_splits, MAX_EFFECTIVE_SPLITS))
    split_tokens = tl.where(
        requested_splits <= MAX_EFFECTIVE_SPLITS,
        TARGET_TOKENS_PER_SPLIT,
        tl.cdiv(seq_len, num_splits),
    )
    split_start = split_id * split_tokens
    split_end = tl.minimum(split_start + split_tokens, seq_len)
    split_valid = (split_id < num_splits) & (split_start < split_end)
    if not split_valid:
        return

    offs_d = tl.arange(0, HEAD_DIM)
    q_offsets = batch_id * stride_qb + query_heads[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + q_offsets, mask=head_mask[:, None], other=0.0)
    req_id = tl.load(B_req_idx + batch_id)

    max_logit = tl.zeros([QUERY_HEAD_BLOCK], dtype=tl.float32) - float("inf")
    exp_sum = tl.zeros([QUERY_HEAD_BLOCK], dtype=tl.float32)
    acc = tl.zeros([QUERY_HEAD_BLOCK, HEAD_DIM], dtype=tl.float32)
    block_count = tl.where(split_valid, tl.cdiv(split_end - split_start, BLOCK_N), 0)

    for block_id in range(0, block_count):
        positions = split_start + block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        position_mask = positions < split_end
        slots = tl.load(
            Req_to_tokens + req_id * stride_req_b + positions * stride_req_s,
            mask=position_mask,
            other=0,
        ).to(tl.int64)
        k_offsets = slots[None, :] * stride_kb + kv_head_id * stride_kh + offs_d[:, None]
        v_offsets = slots[:, None] * stride_vb + kv_head_id * stride_vh + offs_d[None, :]
        k = tl.load(K + k_offsets, mask=position_mask[None, :], other=0.0)
        v = tl.load(V + v_offsets, mask=position_mask[:, None], other=0.0)
        logits = tl.dot(q, k)
        logits = tl.where(position_mask[None, :], logits, -float("inf"))
        if SCORE_MODE == 3:
            score_offsets = (
                batch_id * stride_score_b
                + query_heads[:, None] * stride_score_h
                + positions[None, :] * stride_score_s
            )
            tl.store(
                Attn_Score + score_offsets,
                logits,
                mask=head_mask[:, None] & position_mask[None, :],
            )
        elif SCORE_MODE == 2:
            score_offsets = batch_id * stride_score_b + positions * stride_score_s
            reduced_logits = tl.max(
                tl.where(head_mask[:, None], logits, -float("inf")),
                axis=0,
            )
            tl.atomic_max(
                Attn_Score + score_offsets,
                reduced_logits,
                mask=position_mask,
            )
        logits *= sm_scale

        block_max = tl.max(logits, axis=1)
        next_max = tl.maximum(max_logit, block_max)
        old_scale = tl.exp(max_logit - next_max)
        probs = tl.exp(logits - next_max[:, None])
        acc *= old_scale[:, None]
        acc += tl.dot(probs.to(v.dtype), v)
        exp_sum = exp_sum * old_scale + tl.sum(probs, axis=1)
        max_logit = next_max

    safe_sum = tl.where(split_valid, exp_sum, 1.0)
    mid_offsets = (
        batch_id * stride_mid_b
        + query_heads[:, None] * stride_mid_h
        + split_id * stride_mid_s
        + offs_d[None, :]
    )
    lse_offsets = (
        batch_id * stride_lse_b
        + query_heads * stride_lse_h
        + split_id * stride_lse_s
    )
    tl.store(
        Mid_O + mid_offsets,
        tl.where(split_valid, acc / safe_sum[:, None], 0.0),
        mask=head_mask[:, None],
    )
    tl.store(
        Mid_Lse + lse_offsets,
        tl.where(split_valid, max_logit + tl.log(safe_sum), -float("inf")),
        mask=head_mask,
    )


@triton.jit
def _paged_decode_stage2(
    B_Seqlen,
    Mid_O,
    Mid_Lse,
    O,
    Out_Lse,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_ob,
    stride_oh,
    stride_out_lse_h,
    stride_out_lse_b,
    HEAD_DIM: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MAX_EFFECTIVE_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_len = tl.load(B_Seqlen + batch_id)
    if seq_len <= 0:
        tl.store(O + batch_id * stride_ob + head_id * stride_oh + tl.arange(0, HEAD_DIM), 0.0)
        tl.store(Out_Lse + head_id * stride_out_lse_h + batch_id * stride_out_lse_b, -float("inf"))
        return
    num_splits = tl.maximum(
        1,
        tl.minimum(
            tl.cdiv(seq_len, TARGET_TOKENS_PER_SPLIT),
            MAX_EFFECTIVE_SPLITS,
        ),
    )

    offs_d = tl.arange(0, HEAD_DIM)
    mid_base = batch_id * stride_mid_b + head_id * stride_mid_h + offs_d
    lse_base = batch_id * stride_lse_b + head_id * stride_lse_h
    max_lse = -float("inf")
    exp_sum = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for split_id in range(0, num_splits):
        split_lse = tl.load(Mid_Lse + lse_base + split_id * stride_lse_s)
        split_o = tl.load(Mid_O + mid_base + split_id * stride_mid_s)
        next_max = tl.maximum(max_lse, split_lse)
        old_scale = tl.exp(max_lse - next_max)
        split_scale = tl.exp(split_lse - next_max)
        acc = acc * old_scale + split_scale * split_o
        exp_sum = exp_sum * old_scale + split_scale
        max_lse = next_max

    tl.store(O + batch_id * stride_ob + head_id * stride_oh + offs_d, acc / exp_sum)
    tl.store(
        Out_Lse + head_id * stride_out_lse_h + batch_id * stride_out_lse_b,
        max_lse + tl.log(exp_sum),
    )


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    mid_o: torch.Tensor,
    mid_lse: torch.Tensor,
    attn_score: torch.Tensor | None,
) -> None:
    head_dim = int(q.shape[-1])
    if head_dim not in {16, 32, 64, 128, 256}:
        raise ValueError(f"unsupported context-stable decode head_dim={head_dim}")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise TypeError("query, key, and value tensors must have the same dtype")
    if int(q.shape[1]) % int(k.shape[1]):
        raise ValueError("query head count must be divisible by KV head count")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("query, key, and value head dimensions must be contiguous")
    if k.stride() != v.stride():
        raise ValueError("key and value cache layouts must match")
    if active_slots.dim() != 2 or active_slots.stride(-1) != 1:
        raise ValueError("active_slots must be a contiguous 2D slot table")
    if req_indices.stride(0) != 1 or context_lens.stride(0) != 1:
        raise ValueError("request indices and context lengths must be contiguous")
    expected_workspace = (int(q.shape[0]), int(q.shape[1]))
    if tuple(mid_o.shape[:2]) != expected_workspace or tuple(mid_lse.shape[:2]) != expected_workspace:
        raise ValueError(
            "workspace batch/head dimensions do not match query: "
            f"q={tuple(q.shape)} mid_o={tuple(mid_o.shape)} mid_lse={tuple(mid_lse.shape)}"
        )
    if int(mid_o.shape[2]) != int(mid_lse.shape[2]):
        raise ValueError("workspace split dimensions must match")
    if attn_score is not None and attn_score.dim() not in {2, 3}:
        raise ValueError("attention score output must be 2D or 3D")


@torch.no_grad()
def fixed_grid_flash_decode_stage2(
    mid_o: torch.Tensor,
    mid_lse: torch.Tensor,
    context_lens: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor,
    *,
    target_tokens_per_split: int,
    num_warps: int | None = None,
    num_stages: int = 2,
) -> None:
    """Reduce a fixed split envelope using device-resident effective lengths."""
    if mid_o.dim() != 4 or mid_lse.dim() != 3:
        raise ValueError("fixed-grid decode workspaces must be rank 4 and rank 3")
    if tuple(mid_o.shape[:3]) != tuple(mid_lse.shape):
        raise ValueError(
            "fixed-grid decode workspace shapes do not match: "
            f"mid_o={tuple(mid_o.shape)} mid_lse={tuple(mid_lse.shape)}"
        )
    batch, num_heads, max_kv_splits, head_dim = map(int, mid_o.shape)
    if tuple(output.shape) != (batch, num_heads, head_dim):
        raise ValueError(
            "fixed-grid decode output shape does not match its workspace: "
            f"output={tuple(output.shape)} expected={(batch, num_heads, head_dim)}"
        )
    if tuple(output_lse.shape) != (num_heads, batch):
        raise ValueError(
            "fixed-grid decode LSE output must be [heads, batch], got "
            f"{tuple(output_lse.shape)}."
        )
    if context_lens.dtype != torch.int32 or context_lens.stride(0) != 1:
        raise TypeError("fixed-grid decode context_lens must be contiguous int32")
    if int(context_lens.numel()) != batch:
        raise ValueError("fixed-grid decode expects one context length per batch row")
    if max_kv_splits <= 0 or int(target_tokens_per_split) <= 0:
        raise ValueError("fixed-grid decode split envelope must be positive")
    if head_dim not in {16, 32, 64, 128, 256}:
        raise ValueError(f"unsupported fixed-grid decode head_dim={head_dim}")
    if output_lse.dtype != torch.float32 or output_lse.device != output.device:
        raise TypeError("fixed-grid decode LSE output must be FP32 on the output device")
    if num_warps is None:
        num_warps = 8 if head_dim == 256 else 4
    if int(num_warps) <= 0 or int(num_stages) <= 0:
        raise ValueError("fixed-grid decode stage2 warps/stages must be positive")

    _paged_decode_stage2[(batch, num_heads)](
        context_lens,
        mid_o,
        mid_lse,
        output,
        output_lse,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output_lse.stride(0),
        output_lse.stride(1),
        HEAD_DIM=head_dim,
        MAX_KV_SPLITS=max_kv_splits,
        MAX_EFFECTIVE_SPLITS=max_kv_splits,
        TARGET_TOKENS_PER_SPLIT=int(target_tokens_per_split),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )


@torch.no_grad()
def paged_flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    mid_o: torch.Tensor,
    mid_lse: torch.Tensor,
    *,
    attn_score: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    target_tokens_per_split: int,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    stage2_num_warps: int | None = None,
    stage2_num_stages: int = 2,
    return_softmax_lse: bool = False,
    output_lse: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-grid split-KV decode for MHA or GQA."""
    _check_inputs(
        q,
        k,
        v,
        active_slots,
        req_indices,
        context_lens,
        mid_o,
        mid_lse,
        attn_score,
    )
    max_kv_splits = int(mid_o.shape[2])
    if max_kv_splits <= 0 or target_tokens_per_split <= 0:
        raise ValueError("split count and target tokens per split must be positive")
    if block_n not in {16, 32, 64, 128}:
        raise ValueError(f"unsupported BLOCK_N={block_n}")
    if num_warps <= 0 or num_stages <= 0 or stage2_num_stages <= 0:
        raise ValueError("Triton launch warps/stages must be positive")

    batch, num_heads, head_dim = map(int, q.shape)
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    if softmax_scale <= 0:
        raise ValueError("softmax_scale must be positive")
    if stage2_num_warps is None:
        stage2_num_warps = 8 if head_dim == 256 else 4
    group_size = num_heads // int(k.shape[1])
    max_effective_splits = max_kv_splits
    score = mid_lse if attn_score is None else attn_score
    if attn_score is None:
        score_strides = (0, 0, 0)
    elif attn_score.dim() == 3:
        score_strides = tuple(int(stride) for stride in attn_score.stride())
    else:
        score_strides = (int(attn_score.stride(0)), 0, int(attn_score.stride(1)))
    stage1_args = (
        q,
        k,
        v,
        softmax_scale,
        active_slots,
        req_indices,
        context_lens,
        mid_o,
        mid_lse,
        score,
        active_slots.stride(0),
        active_slots.stride(1),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        *score_strides,
    )
    stage1_meta = dict(
        GQA_GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
        MAX_KV_SPLITS=max_kv_splits,
        MAX_EFFECTIVE_SPLITS=max_effective_splits,
        TARGET_TOKENS_PER_SPLIT=target_tokens_per_split,
        SCORE_MODE=0 if attn_score is None else attn_score.dim(),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if group_size > 1:
        _paged_grouped_decode_stage1[
            (batch, int(k.shape[1]), max_kv_splits)
        ](
            *stage1_args,
            QUERY_HEAD_BLOCK=max(16, triton.next_power_of_2(group_size)),
            **stage1_meta,
        )
    else:
        _paged_decode_stage1[(batch, num_heads, max_kv_splits)](
            *stage1_args,
            **stage1_meta,
        )

    if output is None:
        output = torch.empty_like(q)
    elif tuple(output.shape) != tuple(q.shape):
        raise ValueError(
            "decode output workspace must match Q shape, got "
            f"output={tuple(output.shape)} q={tuple(q.shape)}"
        )
    elif output.dtype != q.dtype or output.device != q.device:
        raise TypeError("decode output workspace must match Q dtype and device")
    if output_lse is None:
        output_lse = torch.empty(
            (num_heads, batch), dtype=torch.float32, device=q.device
        )
    elif tuple(output_lse.shape) != (num_heads, batch):
        raise ValueError(
            "softmax LSE workspace must be [query_heads, batch], got "
            f"{tuple(output_lse.shape)}."
        )
    if output_lse.dtype != torch.float32 or output_lse.device != q.device:
        raise TypeError("softmax LSE workspace must be FP32 on the query device")
    fixed_grid_flash_decode_stage2(
        mid_o,
        mid_lse,
        context_lens,
        output,
        output_lse,
        target_tokens_per_split=target_tokens_per_split,
        num_warps=stage2_num_warps,
        num_stages=stage2_num_stages,
    )
    return (output, output_lse) if return_softmax_lse else output
