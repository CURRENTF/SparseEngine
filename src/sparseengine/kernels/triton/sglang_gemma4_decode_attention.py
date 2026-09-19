# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGLang fixed-grid Triton decode adapted for Sparse-Engine Gemma 4.

Source: sglang/srt/layers/attention/triton_ops/decode_attention.py at
ed0a62e4dd006132a2c6434378962528f010c906.

The kernel topology and split scheduling follow SGLang.  The local changes are
limited to Sparse-Engine's two-dimensional slot table, Gemma 4 sliding-window
coordinates, and the optional raw-QK score output used by sparse methods.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MIN_BLOCK_KV = tl.constexpr(32)


@triton.jit
def _get_num_kv_splits(
    num_kv_splits,
    context_lens,
    num_seq,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_kv_splits: tl.constexpr,
    multi_processor_count: tl.constexpr,
    window: tl.constexpr,
    max_num_seq: tl.constexpr,
):
    offsets = tl.arange(0, max_num_seq)
    mask = offsets < num_seq
    seq_lens = tl.load(context_lens + offsets, mask=mask, other=0)
    if window > 0:
        seq_lens = tl.minimum(seq_lens, window)
    max_seq_len = tl.max(seq_lens)
    seq_lens_for_min = tl.load(
        context_lens + offsets, mask=mask, other=max_seq_len
    )
    if window > 0:
        seq_lens_for_min = tl.minimum(seq_lens_for_min, window)
    min_seq_len = tl.min(seq_lens_for_min)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len

    split_cap_by_lengths = tl.minimum(
        tl.cdiv(max_seq_len, min_seq_len), max_kv_splits
    )
    chunk_by_lengths = tl.cdiv(max_seq_len, split_cap_by_lengths)

    extended_len = tl.cast(max_seq_len, tl.float32) / 64.0
    extended_cores = tl.cast(
        multi_processor_count * tl.maximum(tl.log2(extended_len), 1.0), tl.int32
    )
    group_size: tl.constexpr = num_heads // num_kv_heads
    if group_size == 1:
        token_grid = num_seq * num_heads
    else:
        block_h: tl.constexpr = min(16, group_size)
        token_grid = num_seq * tl.cdiv(num_heads, block_h)
    split_cap_by_cores = tl.minimum(
        tl.cdiv(extended_cores, token_grid), max_kv_splits
    )
    chunk_by_cores = tl.cdiv(max_seq_len, split_cap_by_cores)
    splits = tl.maximum(
        tl.cdiv(seq_lens, chunk_by_lengths), tl.cdiv(seq_lens, chunk_by_cores)
    )
    # Every split consumed by stage2 must have at least one 32-token block.
    # Otherwise stage1 leaves that split's workspace uninitialized.
    splits = tl.maximum(
        1,
        tl.minimum(splits, tl.cdiv(seq_lens, _MIN_BLOCK_KV)),
    )
    tl.store(num_kv_splits + offsets, splits, mask=mask)


@triton.jit
def _decode_stage1_normal(
    q,
    k,
    v,
    active_slots,
    req_indices,
    context_lens,
    mid_output,
    mid_lse,
    num_kv_splits,
    attn_score,
    stride_qb,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_sb,
    stride_ss,
    stride_mob,
    stride_moh,
    stride_mos,
    stride_mlb,
    stride_mlh,
    stride_mls,
    stride_asb,
    stride_ash,
    stride_asl,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    block_n: tl.constexpr,
    window: tl.constexpr,
    score_mode: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    dims = tl.arange(0, block_dim)
    dim_mask = dims < head_dim
    sequence_len = tl.load(context_lens + batch)
    visible_len = sequence_len
    visible_start = 0
    if window > 0:
        visible_len = tl.minimum(sequence_len, window)
        visible_start = sequence_len - visible_len
    splits = tl.load(num_kv_splits + batch)
    tokens_per_split = (
        tl.cdiv(tl.cdiv(visible_len, splits), _MIN_BLOCK_KV) * _MIN_BLOCK_KV
    )
    split_start = split * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, visible_len)

    max_logit = -float("inf")
    exp_sum = 0.0
    accumulator = tl.zeros((block_dim,), tl.float32)
    if split_end > split_start:
        query = tl.load(
            q + batch * stride_qb + head * stride_qh + dims,
            mask=dim_mask,
            other=0.0,
        )
        request = tl.load(req_indices + batch)
        for start in range(split_start, split_end, block_n):
            local_positions = start + tl.arange(0, block_n)
            positions = visible_start + local_positions
            position_mask = local_positions < split_end
            slots = tl.load(
                active_slots + request * stride_sb + positions * stride_ss,
                mask=position_mask,
                other=0,
            )
            keys = tl.load(
                k + slots[:, None] * stride_kt + head * stride_kh + dims[None, :],
                mask=position_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            logits = tl.sum(query[None, :] * keys, axis=1)
            if score_mode == 3:
                tl.store(
                    attn_score
                    + batch * stride_asb
                    + head * stride_ash
                    + positions * stride_asl,
                    logits,
                    mask=position_mask,
                )
            elif score_mode == 2:
                tl.atomic_max(
                    attn_score + batch * stride_asb + positions * stride_asl,
                    logits,
                    mask=position_mask,
                )
            logits = tl.where(position_mask, logits, -float("inf"))
            values = tl.load(
                v + slots[:, None] * stride_vt + head * stride_vh + dims[None, :],
                mask=position_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            next_max = tl.maximum(tl.max(logits, axis=0), max_logit)
            old_scale = tl.exp(max_logit - next_max)
            probabilities = tl.exp(logits - next_max)
            accumulator *= old_scale
            accumulator += tl.sum(probabilities[:, None] * values, axis=0)
            exp_sum = exp_sum * old_scale + tl.sum(probabilities, axis=0)
            max_logit = next_max

        mid_offset = batch * stride_mob + head * stride_moh + split * stride_mos
        tl.store(
            mid_output + mid_offset + dims,
            accumulator / exp_sum,
            mask=dim_mask,
        )
        tl.store(
            mid_lse + batch * stride_mlb + head * stride_mlh + split * stride_mls,
            max_logit + tl.log(exp_sum),
        )


@triton.jit
def _decode_stage1_grouped(
    q,
    k,
    v,
    active_slots,
    req_indices,
    context_lens,
    mid_output,
    mid_lse,
    num_kv_splits,
    attn_score,
    stride_qb,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_sb,
    stride_ss,
    stride_mob,
    stride_moh,
    stride_mos,
    stride_mlb,
    stride_mlh,
    stride_mls,
    stride_asb,
    stride_ash,
    stride_asl,
    group_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    block_n: tl.constexpr,
    block_h: tl.constexpr,
    window: tl.constexpr,
    score_mode: tl.constexpr,
):
    batch = tl.program_id(0)
    head_group = tl.program_id(1)
    split = tl.program_id(2)
    valid_block_h: tl.constexpr = min(block_h, group_size)
    kv_head = head_group // tl.cdiv(group_size, block_h)
    heads = head_group * valid_block_h + tl.arange(0, block_h)
    head_mask = (heads < (head_group + 1) * valid_block_h) & (heads < num_heads)
    dims = tl.arange(0, block_dim)
    dim_mask = dims < head_dim

    sequence_len = tl.load(context_lens + batch)
    visible_len = sequence_len
    visible_start = 0
    if window > 0:
        visible_len = tl.minimum(sequence_len, window)
        visible_start = sequence_len - visible_len
    splits = tl.load(num_kv_splits + batch)
    tokens_per_split = (
        tl.cdiv(tl.cdiv(visible_len, splits), _MIN_BLOCK_KV) * _MIN_BLOCK_KV
    )
    split_start = split * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, visible_len)

    max_logit = tl.full((block_h,), -float("inf"), tl.float32)
    exp_sum = tl.zeros((block_h,), tl.float32)
    accumulator = tl.zeros((block_h, block_dim), tl.float32)
    if split_end > split_start:
        query = tl.load(
            q + batch * stride_qb + heads[:, None] * stride_qh + dims[None, :],
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        request = tl.load(req_indices + batch)
        key_base = kv_head * stride_kh + dims[:, None]
        value_base = kv_head * stride_vh + dims[None, :]
        for start in tl.range(split_start, split_end, block_n):
            local_positions = start + tl.arange(0, block_n)
            positions = visible_start + local_positions
            position_mask = local_positions < split_end
            slots = tl.load(
                active_slots + request * stride_sb + positions * stride_ss,
                mask=position_mask,
                other=0,
            )
            keys = tl.load(
                k + slots[None, :] * stride_kt + key_base,
                mask=dim_mask[:, None] & position_mask[None, :],
                other=0.0,
            )
            logits = tl.dot(query.to(k.dtype.element_ty), keys)
            if score_mode == 3:
                tl.store(
                    attn_score
                    + batch * stride_asb
                    + heads[:, None] * stride_ash
                    + positions[None, :] * stride_asl,
                    logits,
                    mask=head_mask[:, None] & position_mask[None, :],
                )
            elif score_mode == 2:
                reduced_logits = tl.max(
                    tl.where(head_mask[:, None], logits, -float("inf")), axis=0
                )
                tl.atomic_max(
                    attn_score + batch * stride_asb + positions * stride_asl,
                    reduced_logits,
                    mask=position_mask,
                )
            logits = tl.where(
                head_mask[:, None] & position_mask[None, :],
                logits,
                -float("inf"),
            )
            values = tl.load(
                v + slots[:, None] * stride_vt + value_base,
                mask=position_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            next_max = tl.maximum(tl.max(logits, axis=1), max_logit)
            old_scale = tl.exp(max_logit - next_max)
            probabilities = tl.exp(logits - next_max[:, None])
            accumulator *= old_scale[:, None]
            accumulator += tl.dot(probabilities.to(values.dtype), values)
            exp_sum = exp_sum * old_scale + tl.sum(probabilities, axis=1)
            max_logit = next_max

        mid_offsets = (
            batch * stride_mob
            + heads[:, None] * stride_moh
            + split * stride_mos
            + dims[None, :]
        )
        lse_offsets = batch * stride_mlb + heads * stride_mlh + split * stride_mls
        tl.store(
            mid_output + mid_offsets,
            accumulator / exp_sum[:, None],
            mask=head_mask[:, None] & dim_mask[None, :],
        )
        tl.store(mid_lse + lse_offsets, max_logit + tl.log(exp_sum), mask=head_mask)


@triton.jit
def _decode_stage2(
    mid_output,
    mid_lse,
    output,
    context_lens,
    num_kv_splits,
    stride_mob,
    stride_moh,
    stride_mos,
    stride_mlb,
    stride_mlh,
    stride_mls,
    stride_ob,
    stride_oh,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    max_kv_splits: tl.constexpr,
    window: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, block_dim)
    dim_mask = dims < head_dim
    sequence_len = tl.load(context_lens + batch)
    visible_len = sequence_len
    if window > 0:
        visible_len = tl.minimum(sequence_len, window)
    splits = tl.load(num_kv_splits + batch)
    tokens_per_split = (
        tl.cdiv(tl.cdiv(visible_len, splits), _MIN_BLOCK_KV) * _MIN_BLOCK_KV
    )

    max_lse = -float("inf")
    exp_sum = 0.0
    accumulator = tl.zeros((block_dim,), tl.float32)
    value_offset = batch * stride_mob + head * stride_moh + dims
    lse_offset = batch * stride_mlb + head * stride_mlh
    for split in tl.range(0, max_kv_splits, num_stages=2):
        split_start = tokens_per_split * split
        split_end = tl.minimum(split_start + tokens_per_split, visible_len)
        if split_end > split_start:
            value = tl.load(
                mid_output + value_offset + split * stride_mos,
                mask=dim_mask,
                other=0.0,
            )
            lse = tl.load(mid_lse + lse_offset + split * stride_mls)
            next_max = tl.maximum(lse, max_lse)
            old_scale = tl.exp(max_lse - next_max)
            split_scale = tl.exp(lse - next_max)
            accumulator = accumulator * old_scale + value * split_scale
            exp_sum = exp_sum * old_scale + split_scale
            max_lse = next_max
    tl.store(
        output + batch * stride_ob + head * stride_oh + dims,
        accumulator / exp_sum,
        mask=dim_mask,
    )


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    mid_output: torch.Tensor,
    mid_lse: torch.Tensor,
    num_kv_splits: torch.Tensor,
    attn_score: torch.Tensor | None,
) -> None:
    tensors = (
        q,
        k,
        v,
        active_slots,
        req_indices,
        context_lens,
        mid_output,
        mid_lse,
        num_kv_splits,
    )
    if not all(tensor.is_cuda and tensor.device == q.device for tensor in tensors):
        raise TypeError("Gemma 4 decode tensors must share one CUDA device.")
    if attn_score is not None and (
        not attn_score.is_cuda or attn_score.device != q.device
    ):
        raise TypeError("Gemma 4 attention scores must share the Q/K/V CUDA device.")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Gemma 4 decode requires matching rank-3 Q/K/V tensors.")
    head_dim = int(q.shape[-1])
    if head_dim not in {256, 512} or int(k.shape[-1]) != head_dim:
        raise ValueError(f"Gemma 4 decode requires head_dim 256 or 512, got {head_dim}.")
    if q.dtype not in {torch.float16, torch.bfloat16} or any(
        tensor.dtype != q.dtype for tensor in (k, v)
    ):
        raise TypeError("Gemma 4 decode requires matching FP16 or BF16 Q/K/V.")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
        raise ValueError("Gemma 4 Q/K/V head dimensions must be contiguous.")
    if int(q.shape[1]) % int(k.shape[1]):
        raise ValueError("Gemma 4 query heads must be divisible by KV heads.")
    if active_slots.ndim != 2 or active_slots.stride(-1) != 1:
        raise ValueError("Gemma 4 active_slots must be a contiguous 2D slot table.")
    if active_slots.dtype not in {torch.int32, torch.int64}:
        raise TypeError("Gemma 4 active_slots must use int32 or int64 indices.")
    batch, heads = int(q.shape[0]), int(q.shape[1])
    if req_indices.shape != (batch,) or context_lens.shape != (batch,):
        raise ValueError("Gemma 4 request indices and context lengths must match batch size.")
    if req_indices.dtype not in {torch.int32, torch.int64}:
        raise TypeError("Gemma 4 request indices must use int32 or int64.")
    if context_lens.dtype not in {torch.int32, torch.int64}:
        raise TypeError("Gemma 4 context lengths must use int32 or int64.")
    if mid_output.shape[:2] != (batch, heads) or mid_lse.shape[:2] != (batch, heads):
        raise ValueError("Gemma 4 workspace batch/head dimensions must match query.")
    if mid_output.dtype != torch.float32 or mid_lse.dtype != torch.float32:
        raise TypeError("Gemma 4 decode workspace must use FP32 tensors.")
    if mid_output.shape[2] != mid_lse.shape[2] or mid_output.shape[-1] != head_dim:
        raise ValueError("Gemma 4 workspace split/head dimensions do not match.")
    if num_kv_splits.dtype != torch.int32 or num_kv_splits.shape != (batch,):
        raise ValueError("Gemma 4 num_kv_splits must be a batch-sized int32 tensor.")
    if attn_score is not None and attn_score.dim() not in {2, 3}:
        raise ValueError("Gemma 4 attention scores must be rank 2 or 3.")
    if attn_score is not None:
        expected_prefix = (batch, heads) if attn_score.dim() == 3 else (batch,)
        if attn_score.shape[:-1] != expected_prefix:
            raise ValueError("Gemma 4 attention score batch/head dimensions do not match.")
        if attn_score.dtype != torch.float32:
            raise TypeError("Gemma 4 attention scores must use FP32.")


@torch.no_grad()
def sglang_gemma4_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    mid_output: torch.Tensor,
    mid_lse: torch.Tensor,
    num_kv_splits: torch.Tensor,
    *,
    sliding_window: int | None,
    multi_processor_count: int,
    attn_score: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run SGLang's context-stable fixed-grid Gemma 4 decode."""
    _check_inputs(
        q,
        k,
        v,
        active_slots,
        req_indices,
        context_lens,
        mid_output,
        mid_lse,
        num_kv_splits,
        attn_score,
    )
    batch, num_heads, head_dim = map(int, q.shape)
    num_kv_heads = int(k.shape[1])
    max_kv_splits = int(mid_output.shape[2])
    if max_kv_splits <= 0 or int(multi_processor_count) <= 0:
        raise ValueError(
            "Gemma 4 split count and multi-processor count must be positive."
        )
    max_num_seq = 256 if batch < 256 else triton.next_power_of_2(batch)
    window = int(sliding_window or 0)
    _get_num_kv_splits[(1,)](
        num_kv_splits,
        context_lens,
        batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_kv_splits=max_kv_splits,
        multi_processor_count=int(multi_processor_count),
        window=window,
        max_num_seq=max_num_seq,
    )

    if attn_score is None:
        score = mid_lse
        score_mode = 0
        score_strides = (0, 0, 0)
    else:
        score = attn_score
        score_mode = attn_score.dim()
        score_strides = (
            int(attn_score.stride(0)),
            int(attn_score.stride(1)) if score_mode == 3 else 0,
            int(attn_score.stride(-1)),
        )
    block_dim = triton.next_power_of_2(head_dim)
    common_args = (
        q,
        k,
        v,
        active_slots,
        req_indices,
        context_lens,
        mid_output,
        mid_lse,
        num_kv_splits,
        score,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        active_slots.stride(0),
        active_slots.stride(1),
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        *score_strides,
    )
    group_size = num_heads // num_kv_heads
    if group_size == 1:
        _decode_stage1_normal[(batch, num_heads, max_kv_splits)](
            *common_args,
            head_dim=head_dim,
            block_dim=block_dim,
            block_n=64,
            window=window,
            score_mode=score_mode,
            num_warps=4,
            num_stages=2,
        )
    else:
        block_h = 16
        _decode_stage1_grouped[
            (batch, triton.cdiv(num_heads, min(block_h, group_size)), max_kv_splits)
        ](
            *common_args,
            group_size=group_size,
            num_heads=num_heads,
            head_dim=head_dim,
            block_dim=block_dim,
            block_n=32,
            block_h=block_h,
            window=window,
            score_mode=score_mode,
            num_warps=4,
            num_stages=2,
        )

    output = torch.empty_like(q)
    _decode_stage2[(batch, num_heads)](
        mid_output,
        mid_lse,
        output,
        context_lens,
        num_kv_splits,
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        output.stride(0),
        output.stride(1),
        head_dim=head_dim,
        block_dim=block_dim,
        max_kv_splits=max_kv_splits,
        window=window,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["sglang_gemma4_decode"]
