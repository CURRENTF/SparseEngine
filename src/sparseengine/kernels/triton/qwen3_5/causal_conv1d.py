# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
"""Varlen causal Conv1D for Qwen3.5/Qwen3.6 prefill.

Adapted from SGLang's Triton causal Conv1D implementation. This local kernel
intentionally implements only the contract used by Qwen Gated DeltaNet
prefill: packed varlen input, kernel width four, and an optional state cache.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_KERNEL_WIDTH = 4
_STATE_LENGTH = _KERNEL_WIDTH - 1


@triton.jit
def _causal_conv1d_varlen_fwd_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    state_ptr,
    query_start_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    dim,
    stride_x_dim,
    stride_x_token,
    stride_weight_dim,
    stride_weight_width,
    stride_state_batch,
    stride_state_dim,
    stride_state_token,
    stride_output_dim,
    stride_output_token,
    pad_slot_id,
    HAS_BIAS: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    sequence_id = tl.program_id(0)
    dim_offsets = tl.program_id(1) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < dim
    sequence_start = tl.load(query_start_ptr + sequence_id)
    sequence_end = tl.load(query_start_ptr + sequence_id + 1)
    sequence_length = sequence_end - sequence_start

    if HAS_CACHE_INDICES:
        state_index = tl.load(cache_indices_ptr + sequence_id).to(tl.int64)
    else:
        state_index = sequence_id
    if state_index == pad_slot_id:
        return

    use_initial_state = False
    if HAS_INITIAL_STATE:
        use_initial_state = tl.load(
            has_initial_state_ptr + sequence_id
        ).to(tl.int1)

    state_base = (
        state_ptr
        + state_index * stride_state_batch
        + dim_offsets * stride_state_dim
    )
    weight_base = weight_ptr + dim_offsets * stride_weight_dim
    weight0 = tl.load(
        weight_base + 0 * stride_weight_width,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    weight1 = tl.load(
        weight_base + 1 * stride_weight_width,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    weight2 = tl.load(
        weight_base + 2 * stride_weight_width,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    weight3 = tl.load(
        weight_base + 3 * stride_weight_width,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + dim_offsets, mask=dim_mask, other=0.0).to(
            tl.float32
        )
    else:
        bias = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

    block_start = 0
    while block_start < sequence_length:
        token_offsets = block_start + tl.arange(0, BLOCK_TOKENS)
        token_mask = token_offsets < sequence_length
        accumulator = bias[None, :]

        source0 = token_offsets - 3
        source1 = token_offsets - 2
        source2 = token_offsets - 1
        source3 = token_offsets

        x0 = tl.load(
            x_ptr
            + dim_offsets[None, :] * stride_x_dim
            + (sequence_start + source0)[:, None] * stride_x_token,
            mask=token_mask[:, None]
            & (source0 >= 0)[:, None]
            & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            x_ptr
            + dim_offsets[None, :] * stride_x_dim
            + (sequence_start + source1)[:, None] * stride_x_token,
            mask=token_mask[:, None]
            & (source1 >= 0)[:, None]
            & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            x_ptr
            + dim_offsets[None, :] * stride_x_dim
            + (sequence_start + source2)[:, None] * stride_x_token,
            mask=token_mask[:, None]
            & (source2 >= 0)[:, None]
            & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x3 = tl.load(
            x_ptr
            + dim_offsets[None, :] * stride_x_dim
            + (sequence_start + source3)[:, None] * stride_x_token,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        if use_initial_state:
            state0 = tl.load(
                state_base[None, :]
                + (source0 + STATE_LENGTH)[:, None] * stride_state_token,
                mask=token_mask[:, None]
                & (source0 < 0)[:, None]
                & (source0 >= -STATE_LENGTH)[:, None]
                & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            state1 = tl.load(
                state_base[None, :]
                + (source1 + STATE_LENGTH)[:, None] * stride_state_token,
                mask=token_mask[:, None]
                & (source1 < 0)[:, None]
                & (source1 >= -STATE_LENGTH)[:, None]
                & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            state2 = tl.load(
                state_base[None, :]
                + (source2 + STATE_LENGTH)[:, None] * stride_state_token,
                mask=token_mask[:, None]
                & (source2 < 0)[:, None]
                & (source2 >= -STATE_LENGTH)[:, None]
                & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x0 += state0
            x1 += state1
            x2 += state2

        accumulator += (
            x0 * weight0[None, :]
            + x1 * weight1[None, :]
            + x2 * weight2[None, :]
            + x3 * weight3[None, :]
        )
        if APPLY_SILU:
            accumulator = accumulator * tl.sigmoid(accumulator)
        tl.store(
            output_ptr
            + dim_offsets[None, :] * stride_output_dim
            + (sequence_start + token_offsets)[:, None] * stride_output_token,
            accumulator,
            mask=token_mask[:, None] & dim_mask[None, :],
        )
        block_start += BLOCK_TOKENS

    final_slots = tl.arange(0, 4)
    final_offsets = sequence_length - STATE_LENGTH + final_slots
    final_from_x = tl.load(
        x_ptr
        + dim_offsets[None, :] * stride_x_dim
        + (sequence_start + final_offsets)[:, None] * stride_x_token,
        mask=(final_slots < STATE_LENGTH)[:, None]
        & (final_offsets >= 0)[:, None]
        & dim_mask[None, :],
        other=0.0,
    )
    if use_initial_state:
        final_from_state = tl.load(
            state_base[None, :]
            + (final_offsets + STATE_LENGTH)[:, None] * stride_state_token,
            mask=(final_slots < STATE_LENGTH)[:, None]
            & (final_offsets < 0)[:, None]
            & (final_offsets >= -STATE_LENGTH)[:, None]
            & dim_mask[None, :],
            other=0.0,
        )
        final_from_x += final_from_state
    tl.store(
        state_base[None, :]
        + final_slots[:, None] * stride_state_token,
        final_from_x,
        mask=(final_slots < STATE_LENGTH)[:, None] & dim_mask[None, :],
    )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    conv_states: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = -1,
) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("Qwen causal Conv1D prefill requires CUDA tensors.")
    if x.ndim != 2:
        raise ValueError(
            f"Qwen causal Conv1D expects packed [dim, tokens] input, got {x.shape}."
        )
    if query_start_loc is None or conv_states is None:
        raise ValueError(
            "Qwen causal Conv1D requires query_start_loc and conv_states."
        )
    tensors = {
        "weight": weight,
        "query_start_loc": query_start_loc,
        "conv_states": conv_states,
    }
    if bias is not None:
        tensors["bias"] = bias
    if cache_indices is not None:
        tensors["cache_indices"] = cache_indices
    if has_initial_state is not None:
        tensors["has_initial_state"] = has_initial_state
    for name, tensor in tensors.items():
        if not tensor.is_cuda or tensor.device != x.device:
            raise ValueError(
                f"Qwen causal Conv1D {name} must be on device {x.device}."
            )
    if x.dtype not in _SUPPORTED_DTYPES or weight.dtype != x.dtype:
        raise TypeError(
            "Qwen causal Conv1D input and weight must share BF16/FP16 dtype, "
            f"got input={x.dtype}, weight={weight.dtype}."
        )
    if bias is not None and (bias.dtype != x.dtype or bias.shape != (x.shape[0],)):
        raise ValueError(
            f"Qwen causal Conv1D bias must have shape {(x.shape[0],)} "
            f"and dtype {x.dtype}."
        )
    if weight.ndim != 2 or tuple(weight.shape) != (x.shape[0], _KERNEL_WIDTH):
        raise ValueError(
            "Qwen causal Conv1D supports weight shape [dim, 4], got "
            f"{tuple(weight.shape)}."
        )
    batch_size = int(query_start_loc.numel()) - 1
    if query_start_loc.ndim != 1 or batch_size <= 0:
        raise ValueError("query_start_loc must have shape [batch + 1].")
    if query_start_loc.dtype not in (torch.int32, torch.int64):
        raise TypeError("query_start_loc must use int32 or int64.")
    if not query_start_loc.is_contiguous():
        raise ValueError("query_start_loc must be contiguous.")
    if tuple(conv_states.shape[1:]) != (x.shape[0], _STATE_LENGTH):
        raise ValueError(
            "Qwen causal Conv1D state shape mismatch: expected "
            f"[cache, {x.shape[0]}, {_STATE_LENGTH}], got {tuple(conv_states.shape)}."
        )
    if conv_states.dtype != x.dtype or not conv_states.is_contiguous():
        raise ValueError(
            "Qwen causal Conv1D states must be contiguous and match input dtype."
        )
    if cache_indices is not None and (
        cache_indices.ndim != 1
        or cache_indices.numel() != batch_size
        or cache_indices.dtype not in (torch.int32, torch.int64)
        or not cache_indices.is_contiguous()
    ):
        raise ValueError(
            "cache_indices must be contiguous int32/int64 with shape [batch]."
        )
    if has_initial_state is not None and (
        has_initial_state.dtype != torch.bool
        or has_initial_state.ndim != 1
        or has_initial_state.numel() != batch_size
        or not has_initial_state.is_contiguous()
    ):
        raise ValueError(
            "has_initial_state must be contiguous bool with shape [batch]."
        )
    if not weight.is_contiguous() or (bias is not None and not bias.is_contiguous()):
        raise ValueError("Qwen causal Conv1D weight and bias must be contiguous.")
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish.")

    # Padded sequences are intentionally skipped and retain their input values,
    # matching the SGLang kernel contract.
    output = x.clone()
    placeholder = x
    grid = (batch_size, triton.cdiv(int(x.shape[0]), 128))
    _causal_conv1d_varlen_fwd_kernel[grid](
        x,
        weight,
        bias if bias is not None else placeholder,
        conv_states,
        query_start_loc,
        cache_indices if cache_indices is not None else placeholder,
        has_initial_state if has_initial_state is not None else placeholder,
        output,
        int(x.shape[0]),
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        output.stride(0),
        output.stride(1),
        int(pad_slot_id),
        HAS_BIAS=bias is not None,
        HAS_CACHE_INDICES=cache_indices is not None,
        HAS_INITIAL_STATE=has_initial_state is not None,
        APPLY_SILU=activation in ("silu", "swish"),
        STATE_LENGTH=_STATE_LENGTH,
        BLOCK_TOKENS=8,
        BLOCK_DIM=128,
        num_warps=4,
        num_stages=2,
    )
    return output
