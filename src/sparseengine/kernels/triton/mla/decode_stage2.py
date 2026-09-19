# SPDX-License-Identifier: Apache-2.0
# Derived from ModelTC/lightllm at commit
# 65c174ee95ac6a6fd36b18b63d0b33d97e76b770:
# lightllm/common/basemodel/triton_kernel/mla_att/decode_att/
# gqa_flash_decoding_stage2.py
# Local changes: remove unused runtime imports, expose explicit scheduling and
# workspace tensors, preserve arbitrary strides, and restrict dimensions to
# the GLM MLA contract.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .decode_stage1 import MLA_LATENT_DIM


@triton.jit
def _decode_stage2_kernel(
    block_size_ptr,
    batch_start_indices,
    context_lens,
    mid_output,
    mid_logsumexp,
    output,
    stride_mid_head,
    stride_mid_block,
    stride_mid_dim,
    stride_lse_head,
    stride_lse_block,
    stride_output_batch,
    stride_output_head,
    stride_output_dim,
    LATENT_DIM: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    head_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    dim_offsets = tl.arange(0, LATENT_DIM)

    context_len = tl.load(context_lens + batch_index)
    batch_start = tl.load(batch_start_indices + batch_index)
    block_size = tl.load(block_size_ptr)
    block_count = tl.cdiv(context_len, block_size)

    sum_exp = 0.0
    max_logit = -float("inf")
    accumulator = tl.zeros([LATENT_DIM], dtype=tl.float32)
    mid_offsets = (
        head_index * stride_mid_head
        + batch_start * stride_mid_block
        + dim_offsets * stride_mid_dim
    )
    lse_offset = (
        head_index * stride_lse_head
        + batch_start * stride_lse_block
    )

    for block_index in tl.range(
        0,
        block_count,
        1,
        num_stages=PIPELINE_STAGES,
    ):
        block_output = tl.load(
            mid_output + mid_offsets + block_index * stride_mid_block
        )
        block_lse = tl.load(
            mid_logsumexp + lse_offset + block_index * stride_lse_block
        )
        new_max = tl.maximum(block_lse, max_logit)
        old_scale = tl.exp(max_logit - new_max)
        block_scale = tl.exp(block_lse - new_max)
        accumulator = accumulator * old_scale + block_scale * block_output
        sum_exp = sum_exp * old_scale + block_scale
        max_logit = new_max

    output_offsets = (
        batch_index * stride_output_batch
        + head_index * stride_output_head
        + dim_offsets * stride_output_dim
    )
    normalized = tl.where(block_count > 0, accumulator / sum_exp, 0.0)
    tl.store(output + output_offsets, normalized)


@torch.no_grad()
def decode_stage2(
    block_size: torch.Tensor,
    batch_start_indices: torch.Tensor,
    context_lens: torch.Tensor,
    mid_output: torch.Tensor,
    mid_logsumexp: torch.Tensor,
    output: torch.Tensor,
    *,
    pipeline_stages: int,
    num_warps: int,
) -> None:
    """Merge independently normalized stage-one blocks into final output."""

    tensors = {
        "block_size": block_size,
        "batch_start_indices": batch_start_indices,
        "context_lens": context_lens,
        "mid_output": mid_output,
        "mid_logsumexp": mid_logsumexp,
        "output": output,
    }
    for name, tensor in tensors.items():
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")
        if tensor.device != output.device:
            raise ValueError(
                f"{name} is on {tensor.device}, expected {output.device}"
            )

    for name in ("block_size", "batch_start_indices", "context_lens"):
        if tensors[name].dtype != torch.int32:
            raise TypeError(
                f"{name} must use {torch.int32}, got {tensors[name].dtype}"
            )
    if mid_output.dtype != torch.float32:
        raise TypeError("mid_output must use torch.float32")
    if mid_logsumexp.dtype != torch.float32:
        raise TypeError("mid_logsumexp must use torch.float32")
    if output.dtype != torch.bfloat16:
        raise TypeError("output must use torch.bfloat16")

    if block_size.shape != (1,):
        raise ValueError("block_size must have shape [1]")
    if output.ndim != 3 or output.shape[-1] != MLA_LATENT_DIM:
        raise ValueError("output must have shape [batch, heads, 512]")
    batch_size, head_count = output.shape[:2]
    if context_lens.shape != (batch_size,):
        raise ValueError(
            f"context_lens must have shape ({batch_size},), got "
            f"{tuple(context_lens.shape)}"
        )
    if batch_start_indices.ndim != 1 or batch_start_indices.numel() < batch_size:
        raise ValueError(
            "batch_start_indices must be one-dimensional with at least "
            f"{batch_size} entries"
        )
    if mid_output.ndim != 3 or mid_output.shape[-1] != MLA_LATENT_DIM:
        raise ValueError("mid_output must have shape [heads, blocks, 512]")
    if mid_output.shape[0] < head_count:
        raise ValueError(
            f"mid_output has capacity for {mid_output.shape[0]} heads, "
            f"but {head_count} are required"
        )
    if mid_logsumexp.shape != mid_output.shape[:2]:
        raise ValueError(
            "mid_logsumexp must match the first two mid_output dimensions"
        )
    if pipeline_stages <= 0:
        raise ValueError("pipeline_stages must be positive")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")

    _decode_stage2_kernel[(head_count, batch_size)](
        block_size,
        batch_start_indices,
        context_lens,
        mid_output,
        mid_logsumexp,
        output,
        *mid_output.stride(),
        *mid_logsumexp.stride(),
        *output.stride(),
        LATENT_DIM=MLA_LATENT_DIM,
        PIPELINE_STAGES=pipeline_stages,
        num_warps=num_warps,
        num_stages=1,
    )
