# SPDX-License-Identifier: Apache-2.0
# Derived from ModelTC/lightllm at commit
# 65c174ee95ac6a6fd36b18b63d0b33d97e76b770:
# lightllm/models/deepseek2/triton_kernel/sample_kv.py
# Local changes: gather full ragged history without modulo sampling, support
# request indirection/non-contiguous strides/padded rows, and validate packed
# destinations explicitly.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .decode_stage1 import MLA_LATENT_DIM, MLA_ROPE_DIM


@triton.jit
def _gather_latent_kernel(
    latent_cache,
    rope_cache,
    active_slots,
    request_indices,
    context_lens,
    packed_start_locs,
    gathered_latent,
    gathered_rope,
    stride_latent_slot,
    stride_latent_head,
    stride_latent_dim,
    stride_rope_slot,
    stride_rope_head,
    stride_rope_dim,
    stride_slots_row,
    stride_slots_token,
    stride_request,
    stride_context,
    stride_packed_start,
    stride_output_latent_token,
    stride_output_latent_dim,
    stride_output_rope_token,
    stride_output_rope_dim,
    cache_slot_count,
    output_capacity,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    batch_index = tl.program_id(0)
    sequence_block = tl.program_id(1)
    context_len = tl.load(context_lens + batch_index * stride_context)
    request_index = tl.load(
        request_indices + batch_index * stride_request
    ).to(tl.int64)
    packed_start = tl.load(
        packed_start_locs + batch_index * stride_packed_start
    ).to(tl.int64)

    token_offsets = sequence_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    valid_row = (request_index >= 0) & (context_len > 0)
    valid_token = valid_row & (token_offsets < context_len)
    safe_request_index = tl.where(valid_row, request_index, 0)
    cache_slots = tl.load(
        active_slots
        + safe_request_index * stride_slots_row
        + token_offsets * stride_slots_token,
        mask=valid_token,
        other=0,
    ).to(tl.int64)
    valid_slot = valid_token & (cache_slots >= 0) & (
        cache_slots < cache_slot_count
    )
    safe_cache_slots = tl.where(valid_slot, cache_slots, 0)

    latent_offsets = tl.arange(0, LATENT_DIM)
    rope_offsets = tl.arange(0, ROPE_DIM)
    latent_cache_offsets = (
        safe_cache_slots[:, None] * stride_latent_slot
        + latent_offsets[None, :] * stride_latent_dim
    )
    rope_cache_offsets = (
        safe_cache_slots[:, None] * stride_rope_slot
        + rope_offsets[None, :] * stride_rope_dim
    )
    latent_values = tl.load(
        latent_cache + latent_cache_offsets,
        mask=valid_slot[:, None],
        other=0.0,
    )
    rope_values = tl.load(
        rope_cache + rope_cache_offsets,
        mask=valid_slot[:, None],
        other=0.0,
    )

    output_tokens = packed_start + token_offsets
    valid_output = (
        valid_token
        & (output_tokens >= 0)
        & (output_tokens < output_capacity)
    )
    output_latent_offsets = (
        output_tokens[:, None] * stride_output_latent_token
        + latent_offsets[None, :] * stride_output_latent_dim
    )
    output_rope_offsets = (
        output_tokens[:, None] * stride_output_rope_token
        + rope_offsets[None, :] * stride_output_rope_dim
    )
    tl.store(
        gathered_latent + output_latent_offsets,
        latent_values,
        mask=valid_output[:, None],
    )
    tl.store(
        gathered_rope + output_rope_offsets,
        rope_values,
        mask=valid_output[:, None],
    )


def _validate_gather_tensors(
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    packed_start_locs: torch.Tensor,
    gathered_latent: torch.Tensor,
    gathered_rope: torch.Tensor,
) -> None:
    tensors = {
        "latent_cache": latent_cache,
        "rope_cache": rope_cache,
        "active_slots": active_slots,
        "request_indices": request_indices,
        "context_lens": context_lens,
        "packed_start_locs": packed_start_locs,
        "gathered_latent": gathered_latent,
        "gathered_rope": gathered_rope,
    }
    for name, tensor in tensors.items():
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")
        if tensor.device != latent_cache.device:
            raise ValueError(
                f"{name} is on {tensor.device}, expected {latent_cache.device}"
            )
    for name in (
        "latent_cache",
        "rope_cache",
        "gathered_latent",
        "gathered_rope",
    ):
        if tensors[name].dtype != torch.bfloat16:
            raise TypeError(
                f"{name} must use {torch.bfloat16}, got {tensors[name].dtype}"
            )
    for name in (
        "active_slots",
        "request_indices",
        "context_lens",
        "packed_start_locs",
    ):
        if tensors[name].dtype != torch.int32:
            raise TypeError(
                f"{name} must use {torch.int32}, got {tensors[name].dtype}"
            )

    if latent_cache.ndim != 3 or latent_cache.shape[1:] != (
        1,
        MLA_LATENT_DIM,
    ):
        raise ValueError("latent_cache must have shape [slots, 1, 512]")
    if rope_cache.ndim != 3 or rope_cache.shape[1:] != (1, MLA_ROPE_DIM):
        raise ValueError("rope_cache must have shape [slots, 1, 64]")
    if latent_cache.shape[0] != rope_cache.shape[0]:
        raise ValueError("latent_cache and rope_cache must have equal slots")
    if active_slots.ndim != 2:
        raise ValueError("active_slots must have shape [rows, max_context_len]")
    if context_lens.ndim != 1 or context_lens.numel() == 0:
        raise ValueError("context_lens must be a non-empty one-dimensional tensor")
    batch_size = context_lens.numel()
    if request_indices.shape != (batch_size,):
        raise ValueError(
            f"request_indices must have shape ({batch_size},), got "
            f"{tuple(request_indices.shape)}"
        )
    if packed_start_locs.shape != (batch_size,):
        raise ValueError(
            f"packed_start_locs must have shape ({batch_size},), got "
            f"{tuple(packed_start_locs.shape)}"
        )
    if gathered_latent.ndim != 2 or gathered_latent.shape[1] != MLA_LATENT_DIM:
        raise ValueError("gathered_latent must have shape [capacity, 512]")
    if gathered_rope.ndim != 2 or gathered_rope.shape[1] != MLA_ROPE_DIM:
        raise ValueError("gathered_rope must have shape [capacity, 64]")
    if gathered_latent.shape[0] != gathered_rope.shape[0]:
        raise ValueError("gathered_latent and gathered_rope capacities must match")


def validate_gather_metadata(
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    packed_start_locs: torch.Tensor,
    *,
    cache_slot_count: int,
    output_capacity: int,
    max_context_len: int,
) -> None:
    """Synchronously validate one gather description before per-layer reuse."""

    if max_context_len < 0 or max_context_len > active_slots.shape[1]:
        raise ValueError(
            "max_context_len must be within active_slots capacity: "
            f"{max_context_len} > {active_slots.shape[1]}"
        )
    request_rows = request_indices.tolist()
    lengths = context_lens.tolist()
    packed_starts = packed_start_locs.tolist()
    intervals: list[tuple[int, int]] = []
    for batch_index, (request_row, length, packed_start) in enumerate(
        zip(request_rows, lengths, packed_starts)
    ):
        if length < 0 or length > max_context_len:
            raise ValueError(
                f"context_lens[{batch_index}]={length} is outside "
                f"[0, {max_context_len}]"
            )
        if request_row < 0:
            if length != 0:
                raise ValueError(
                    "padded request rows must have zero context length"
                )
            continue
        if request_row >= active_slots.shape[0]:
            raise ValueError(
                f"request_indices[{batch_index}]={request_row} is outside "
                f"[0, {active_slots.shape[0]})"
            )
        if packed_start < 0 or packed_start + length > output_capacity:
            raise ValueError(
                f"packed output for batch {batch_index} exceeds capacity "
                f"{output_capacity}"
            )
        if length == 0:
            continue
        slots = active_slots[request_row, :length]
        if bool(torch.any(slots < 0).item()) or bool(
            torch.any(slots >= cache_slot_count).item()
        ):
            raise ValueError(
                f"active_slots row {request_row} contains an invalid slot"
            )
        if slots.unique().numel() != slots.numel():
            raise ValueError(
                f"active_slots row {request_row} contains duplicate slots"
            )
        intervals.append((packed_start, packed_start + length))

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError("packed gather output ranges overlap")


@torch.no_grad()
def gather_latent_history(
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    packed_start_locs: torch.Tensor,
    gathered_latent: torch.Tensor,
    gathered_rope: torch.Tensor,
    *,
    max_context_len: int,
    validate_metadata: bool = True,
) -> None:
    """Gather complete ragged MLA history into caller-owned packed buffers."""

    _validate_gather_tensors(
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        packed_start_locs,
        gathered_latent,
        gathered_rope,
    )
    output_capacity = gathered_latent.shape[0]
    if validate_metadata:
        validate_gather_metadata(
            active_slots,
            request_indices,
            context_lens,
            packed_start_locs,
            cache_slot_count=latent_cache.shape[0],
            output_capacity=output_capacity,
            max_context_len=max_context_len,
        )
    if max_context_len == 0:
        return

    block_seq = 64
    batch_size = context_lens.numel()
    _gather_latent_kernel[
        (batch_size, triton.cdiv(max_context_len, block_seq))
    ](
        latent_cache,
        rope_cache,
        active_slots,
        request_indices,
        context_lens,
        packed_start_locs,
        gathered_latent,
        gathered_rope,
        *latent_cache.stride(),
        *rope_cache.stride(),
        *active_slots.stride(),
        request_indices.stride(0),
        context_lens.stride(0),
        packed_start_locs.stride(0),
        *gathered_latent.stride(),
        *gathered_rope.stride(),
        cache_slot_count=latent_cache.shape[0],
        output_capacity=output_capacity,
        LATENT_DIM=MLA_LATENT_DIM,
        ROPE_DIM=MLA_ROPE_DIM,
        BLOCK_SEQ=block_seq,
        num_warps=8,
        num_stages=1,
    )
