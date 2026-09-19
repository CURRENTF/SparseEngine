# SPDX-License-Identifier: Apache-2.0
# Derived from ModelTC/lightllm at commit
# 65c174ee95ac6a6fd36b18b63d0b33d97e76b770:
# lightllm/common/basemodel/triton_kernel/mla_att/decode_att/
# gqa_flash_decoding_stage1.py
# Local changes: remove LightLLM runtime/device helpers, expose an explicit
# workspace API, restrict the layout to the GLM MLA contract, and preserve
# arbitrary tensor strides.

from __future__ import annotations

import torch
import triton
import triton.language as tl


MLA_LATENT_DIM = 512
MLA_ROPE_DIM = 64


@triton.jit
def _decode_stage1_kernel(
    q_latent,
    q_rope,
    latent_cache,
    rope_cache,
    softmax_scale,
    active_slots,
    request_indices,
    context_lens,
    mid_output,
    mid_logsumexp,
    attn_score,
    stride_slots_row,
    stride_slots_token,
    stride_q_latent_batch,
    stride_q_latent_head,
    stride_q_latent_dim,
    stride_q_rope_batch,
    stride_q_rope_head,
    stride_q_rope_dim,
    stride_latent_slot,
    stride_latent_head,
    stride_latent_dim,
    stride_rope_slot,
    stride_rope_head,
    stride_rope_dim,
    stride_mid_head,
    stride_mid_block,
    stride_mid_dim,
    stride_lse_head,
    stride_lse_block,
    stride_score_batch,
    stride_score_head,
    stride_score_token,
    block_size_ptr,
    cache_slot_count,
    program_count,
    head_group_count,
    head_count,
    batch_size,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_Q_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    MASK_HEADS: tl.constexpr,
    STORE_SCORE: tl.constexpr,
    REDUCE_SCORE_HEADS: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    output_batch_start = tl.cast(0, tl.int64)
    block_size = tl.load(block_size_ptr, eviction_policy="evict_last")

    head_offsets = tl.arange(0, BLOCK_Q_HEADS)
    latent_offsets = tl.arange(0, LATENT_DIM)
    rope_offsets = tl.arange(0, ROPE_DIM)

    for batch_index in range(batch_size):
        context_len = tl.load(
            context_lens + batch_index,
            eviction_policy="evict_last",
        )
        block_count = tl.cdiv(context_len, block_size)
        work_count = block_count * head_group_count
        request_index = tl.load(
            request_indices + batch_index,
            eviction_policy="evict_last",
        )
        slots_row = active_slots + request_index * stride_slots_row

        work_index = program_id
        while work_index < work_count:
            head_group_index = work_index % head_group_count
            sequence_block_index = work_index // head_group_count
            query_heads = head_group_index * BLOCK_Q_HEADS + head_offsets
            if MASK_HEADS:
                head_mask = query_heads < head_count

            block_start = block_size * sequence_block_index
            block_end = tl.minimum(context_len, block_start + block_size)

            q_latent_offsets = (
                batch_index * stride_q_latent_batch
                + query_heads[:, None] * stride_q_latent_head
                + latent_offsets[None, :] * stride_q_latent_dim
            )
            q_rope_offsets = (
                batch_index * stride_q_rope_batch
                + query_heads[:, None] * stride_q_rope_head
                + rope_offsets[None, :] * stride_q_rope_dim
            )
            if MASK_HEADS:
                query_latent = tl.load(
                    q_latent + q_latent_offsets,
                    mask=head_mask[:, None],
                    other=0.0,
                )
                query_rope = tl.load(
                    q_rope + q_rope_offsets,
                    mask=head_mask[:, None],
                    other=0.0,
                )
            else:
                query_latent = tl.load(q_latent + q_latent_offsets)
                query_rope = tl.load(q_rope + q_rope_offsets)

            loop_count = tl.cdiv(block_end - block_start, BLOCK_N)
            token_offsets = block_start + tl.arange(0, BLOCK_N)
            sum_exp = tl.zeros([BLOCK_Q_HEADS], dtype=tl.float32)
            max_logit = tl.full(
                [BLOCK_Q_HEADS],
                -float("inf"),
                dtype=tl.float32,
            )
            accumulator = tl.zeros(
                [BLOCK_Q_HEADS, LATENT_DIM],
                dtype=tl.float32,
            )

            for token_block in tl.range(
                0,
                loop_count,
                1,
                num_stages=PIPELINE_STAGES,
            ):
                token_indices = token_block * BLOCK_N + token_offsets
                token_mask = token_indices < block_end
                cache_slots = tl.load(
                    slots_row + token_indices * stride_slots_token,
                    mask=token_mask,
                    other=0,
                ).to(tl.int64)
                valid_cache_slot = (
                    token_mask
                    & (cache_slots >= 0)
                    & (cache_slots < cache_slot_count)
                )
                safe_cache_slots = tl.where(valid_cache_slot, cache_slots, 0)

                latent_cache_offsets = (
                    safe_cache_slots[None, :] * stride_latent_slot
                    + latent_offsets[:, None] * stride_latent_dim
                )
                cached_latent = tl.load(
                    latent_cache + latent_cache_offsets,
                    mask=valid_cache_slot[None, :],
                    other=0.0,
                )
                raw_logits = tl.dot(query_latent, cached_latent)

                rope_cache_offsets = (
                    safe_cache_slots[None, :] * stride_rope_slot
                    + rope_offsets[:, None] * stride_rope_dim
                )
                cached_rope = tl.load(
                    rope_cache + rope_cache_offsets,
                    mask=valid_cache_slot[None, :],
                    other=0.0,
                )
                raw_logits += tl.dot(query_rope, cached_rope)
                if STORE_SCORE:
                    score_mask = valid_cache_slot[None, :]
                    if MASK_HEADS:
                        score_mask &= head_mask[:, None]
                    if REDUCE_SCORE_HEADS:
                        reduced_score = tl.max(
                            tl.where(score_mask, raw_logits, -float("inf")),
                            axis=0,
                        )
                        score_offsets = (
                            batch_index * stride_score_batch
                            + token_indices * stride_score_token
                        )
                        tl.atomic_max(
                            attn_score + score_offsets,
                            reduced_score,
                            mask=valid_cache_slot,
                        )
                    else:
                        score_offsets = (
                            batch_index * stride_score_batch
                            + query_heads[:, None] * stride_score_head
                            + token_indices[None, :] * stride_score_token
                        )
                        tl.store(
                            attn_score + score_offsets,
                            raw_logits,
                            mask=score_mask,
                        )
                logits = raw_logits * softmax_scale
                logits = tl.where(
                    valid_cache_slot[None, :],
                    logits,
                    -float("inf"),
                )

                block_max = tl.max(logits, axis=1)
                new_max = tl.maximum(block_max, max_logit)
                exp_logits = tl.exp(logits - new_max[:, None])
                old_scale = tl.exp(max_logit - new_max)
                accumulator *= old_scale[:, None]
                accumulator += tl.dot(
                    exp_logits.to(cached_latent.dtype),
                    tl.trans(cached_latent),
                )
                sum_exp = sum_exp * old_scale + tl.sum(exp_logits, axis=1)
                max_logit = new_max

            output_block_index = output_batch_start + sequence_block_index
            mid_offsets = (
                query_heads[:, None] * stride_mid_head
                + output_block_index * stride_mid_block
                + latent_offsets[None, :] * stride_mid_dim
            )
            lse_offsets = (
                query_heads * stride_lse_head
                + output_block_index * stride_lse_block
            )
            normalized = accumulator / sum_exp[:, None]
            logsumexp = max_logit + tl.log(sum_exp)
            if MASK_HEADS:
                tl.store(
                    mid_output + mid_offsets,
                    normalized,
                    mask=head_mask[:, None],
                )
                tl.store(
                    mid_logsumexp + lse_offsets,
                    logsumexp,
                    mask=head_mask,
                )
            else:
                tl.store(mid_output + mid_offsets, normalized)
                tl.store(mid_logsumexp + lse_offsets, logsumexp)

            work_index += program_count

        output_batch_start += block_count


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")


def _require_dtype(
    name: str,
    tensor: torch.Tensor,
    expected: torch.dtype,
) -> None:
    if tensor.dtype != expected:
        raise TypeError(f"{name} must use {expected}, got {tensor.dtype}")


@torch.no_grad()
def decode_stage1(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: torch.Tensor,
    mid_output: torch.Tensor,
    mid_logsumexp: torch.Tensor,
    *,
    attn_score: torch.Tensor | None = None,
    max_context_len: int | None = None,
    softmax_scale: float,
    program_count: int,
    block_q_heads: int,
    block_n: int,
    pipeline_stages: int,
    num_warps: int,
) -> None:
    """Compute independently normalized MLA attention blocks.

    All scheduling tensors and workspaces are caller-owned. The function does
    not allocate, inspect device properties, or depend on model/runtime state.
    """

    tensors = {
        "q_latent": q_latent,
        "q_rope": q_rope,
        "latent_cache": latent_cache,
        "rope_cache": rope_cache,
        "active_slots": active_slots,
        "request_indices": request_indices,
        "context_lens": context_lens,
        "block_size": block_size,
        "mid_output": mid_output,
        "mid_logsumexp": mid_logsumexp,
    }
    if attn_score is not None:
        tensors["attn_score"] = attn_score
    for name, tensor in tensors.items():
        _require_cuda_tensor(name, tensor)
        if tensor.device != q_latent.device:
            raise ValueError(
                f"{name} is on {tensor.device}, expected {q_latent.device}"
            )

    for name in ("q_latent", "q_rope", "latent_cache", "rope_cache"):
        _require_dtype(name, tensors[name], torch.bfloat16)
    for name in ("active_slots", "request_indices", "context_lens"):
        _require_dtype(name, tensors[name], torch.int32)
    _require_dtype("block_size", block_size, torch.int32)
    _require_dtype("mid_output", mid_output, torch.float32)
    _require_dtype("mid_logsumexp", mid_logsumexp, torch.float32)

    if q_latent.ndim != 3 or q_latent.shape[-1] != MLA_LATENT_DIM:
        raise ValueError(
            "q_latent must have shape [batch, heads, 512], got "
            f"{tuple(q_latent.shape)}"
        )
    if q_rope.shape != (*q_latent.shape[:-1], MLA_ROPE_DIM):
        raise ValueError(
            "q_rope must have shape [batch, heads, 64], got "
            f"{tuple(q_rope.shape)}"
        )
    if latent_cache.ndim != 3 or latent_cache.shape[1:] != (
        1,
        MLA_LATENT_DIM,
    ):
        raise ValueError(
            "latent_cache must have shape [slots, 1, 512], got "
            f"{tuple(latent_cache.shape)}"
        )
    if rope_cache.ndim != 3 or rope_cache.shape[1:] != (1, MLA_ROPE_DIM):
        raise ValueError(
            "rope_cache must have shape [slots, 1, 64], got "
            f"{tuple(rope_cache.shape)}"
        )
    if latent_cache.shape[0] != rope_cache.shape[0]:
        raise ValueError("latent_cache and rope_cache must have equal slots")
    if active_slots.ndim != 2:
        raise ValueError("active_slots must have shape [rows, max_context_len]")

    batch_size, head_count = q_latent.shape[:2]
    if request_indices.shape != (batch_size,):
        raise ValueError(
            f"request_indices must have shape ({batch_size},), got "
            f"{tuple(request_indices.shape)}"
        )
    if context_lens.shape != (batch_size,):
        raise ValueError(
            f"context_lens must have shape ({batch_size},), got "
            f"{tuple(context_lens.shape)}"
        )
    if block_size.shape != (1,):
        raise ValueError("block_size must have shape [1]")
    if mid_output.ndim != 3 or mid_output.shape[2] != MLA_LATENT_DIM:
        raise ValueError("mid_output must have shape [heads, blocks, 512]")
    if mid_logsumexp.shape != mid_output.shape[:2]:
        raise ValueError(
            "mid_logsumexp must match the first two mid_output dimensions"
        )
    if mid_output.shape[0] < head_count:
        raise ValueError(
            f"mid_output has capacity for {mid_output.shape[0]} heads, "
            f"but {head_count} are required"
        )
    if attn_score is not None:
        min_width = (
            int(active_slots.shape[1])
            if max_context_len is None
            else int(max_context_len)
        )
        if not 0 < min_width <= int(active_slots.shape[1]):
            raise ValueError(
                "MLA attention-score context capacity must be within the "
                f"active-slot width: capacity={min_width} "
                f"active_slot_width={int(active_slots.shape[1])}."
            )
        if attn_score.dim() == 2:
            if attn_score.dtype != torch.float32:
                raise TypeError(
                    "Head-reduced MLA attention scores must use torch.float32, "
                    f"got {attn_score.dtype}."
                )
            if (
                int(attn_score.shape[0]) < batch_size
                or int(attn_score.shape[1]) < min_width
            ):
                raise ValueError(
                    "Head-reduced MLA attention scores must cover "
                    f"[batch, context]=[{batch_size}, {min_width}], got "
                    f"{tuple(attn_score.shape)}."
                )
        elif attn_score.dim() == 3:
            if attn_score.dtype not in {
                torch.float32,
                torch.bfloat16,
                torch.float16,
            }:
                raise TypeError(
                    "Per-head MLA attention scores must use a floating dtype, "
                    f"got {attn_score.dtype}."
                )
            if (
                int(attn_score.shape[0]) < batch_size
                or int(attn_score.shape[1]) < head_count
                or int(attn_score.shape[2]) < min_width
            ):
                raise ValueError(
                    "Per-head MLA attention scores must cover "
                    f"[batch, heads, context]=[{batch_size}, {head_count}, "
                    f"{min_width}], got {tuple(attn_score.shape)}."
                )
        else:
            raise ValueError(
                "MLA attention score must be [batch, context] or "
                f"[batch, heads, context], got {tuple(attn_score.shape)}."
            )
    if program_count <= 0:
        raise ValueError("program_count must be positive")
    if block_q_heads <= 0 or block_q_heads & (block_q_heads - 1):
        raise ValueError("block_q_heads must be a positive power of two")
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError("block_n must be a positive power of two")
    if pipeline_stages <= 0:
        raise ValueError("pipeline_stages must be positive")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")

    head_group_count = triton.cdiv(head_count, block_q_heads)
    mask_heads = head_count % block_q_heads != 0
    score_arg = attn_score if attn_score is not None else mid_logsumexp
    if attn_score is None:
        score_strides = (0, 0, 0)
    elif attn_score.dim() == 2:
        score_strides = (
            attn_score.stride(0),
            0,
            attn_score.stride(1),
        )
    else:
        score_strides = attn_score.stride()
    _decode_stage1_kernel[(program_count,)](
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        softmax_scale,
        active_slots,
        request_indices,
        context_lens,
        mid_output,
        mid_logsumexp,
        score_arg,
        *active_slots.stride(),
        *q_latent.stride(),
        *q_rope.stride(),
        *latent_cache.stride(),
        *rope_cache.stride(),
        *mid_output.stride(),
        *mid_logsumexp.stride(),
        *score_strides,
        block_size,
        cache_slot_count=latent_cache.shape[0],
        program_count=program_count,
        head_group_count=head_group_count,
        head_count=head_count,
        batch_size=batch_size,
        LATENT_DIM=MLA_LATENT_DIM,
        ROPE_DIM=MLA_ROPE_DIM,
        BLOCK_Q_HEADS=block_q_heads,
        BLOCK_N=block_n,
        PIPELINE_STAGES=pipeline_stages,
        MASK_HEADS=mask_heads,
        STORE_SCORE=attn_score is not None,
        REDUCE_SCORE_HEADS=(
            attn_score is not None and attn_score.dim() == 2
        ),
        num_warps=num_warps,
        num_stages=1,
    )
