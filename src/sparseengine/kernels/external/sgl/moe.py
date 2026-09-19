from __future__ import annotations

import importlib
import inspect
from functools import lru_cache

import torch
import triton

from sparseengine.kernels.external.support import ExternalKernelContractError
from sparseengine.kernels.external.sgl.support import sgl_kernel_support
from sparseengine.kernels.moe import MoeAlignment

_FP8_GROUP_QUANT_ARGUMENTS = (
    "input",
    "output_q",
    "output_s",
    "group_size",
    "eps",
    "fp8_min",
    "fp8_max",
    "scale_ue8m0",
    "fuse_silu_and_mul",
    "masked_m",
    "enable_v2",
)


def sgl_moe_alignment_support() -> tuple[bool, str]:
    """Check the SGL expert-alignment API used by the Triton MoE provider."""

    supported, reason = sgl_kernel_support("MoE alignment")
    if not supported:
        return supported, reason
    try:
        alignment = importlib.import_module("sgl_kernel").moe_align_block_size
    except Exception as error:
        raise ExternalKernelContractError(
            "sglang-kernel",
            "MoE alignment",
            f"failed to load: {type(error).__name__}: {error}",
        ) from error
    if not callable(alignment):
        raise ExternalKernelContractError(
            "sglang-kernel",
            "MoE alignment",
            "moe_align_block_size is not callable",
        )
    return True, reason


@lru_cache(maxsize=1)
def _sgl_fp8_group_quant_op():
    _, reason = sgl_kernel_support("per-token FP8 group quantization")
    try:
        quantize = importlib.import_module(
            "sgl_kernel.gemm"
        ).sgl_per_token_group_quant_8bit
        argument_names = tuple(inspect.signature(quantize).parameters)
    except Exception as error:
        raise ExternalKernelContractError(
            "sglang-kernel",
            "per-token FP8 group quantization",
            f"failed to load: {type(error).__name__}: {error}",
        ) from error
    if argument_names != _FP8_GROUP_QUANT_ARGUMENTS:
        raise ExternalKernelContractError(
            "sglang-kernel",
            "per-token FP8 group quantization",
            f"unsupported schema: {argument_names}",
        )
    return quantize, reason


def sgl_fp8_group_quantization_support() -> tuple[bool, str]:
    """Validate the SGL activation-quantization contract used by Triton FP8 MoE."""

    _, reason = _sgl_fp8_group_quant_op()
    return True, reason


def sgl_per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    *,
    enable_v2: bool,
) -> None:
    quantize, _ = _sgl_fp8_group_quant_op()
    quantize(
        input,
        output_q,
        output_s,
        int(group_size),
        float(eps),
        float(fp8_min),
        float(fp8_max),
        enable_v2=bool(enable_v2),
    )


def sgl_moe_align_block_size(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
) -> MoeAlignment:
    """Group local expert assignments with the SGL CUDA kernel."""

    num_experts = int(num_experts)
    if num_experts <= 0:
        raise ValueError(f"SGL MoE alignment requires experts, got {num_experts}.")
    num_assignments = int(topk_ids.numel())
    if num_assignments < num_experts + 1:
        max_num_tokens_padded = num_assignments * int(block_size)
    else:
        max_num_tokens_padded = (
            num_assignments + (num_experts + 1) * (int(block_size) - 1)
        )
    sorted_token_ids = torch.empty(
        max_num_tokens_padded,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        triton.cdiv(max_num_tokens_padded, int(block_size)),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_padded = torch.empty(
        1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    from sparseengine.kernels.triton.sgl_moe_align import (
        SMALL_NUMEL_LIMIT,
        sgl_moe_align_small_numel,
    )

    if num_assignments <= SMALL_NUMEL_LIMIT and num_experts + 1 > 64:
        sgl_moe_align_small_numel(
            topk_ids,
            num_experts=num_experts,
            block_size=block_size,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
        )
        return MoeAlignment(
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            block_size=int(block_size),
            naive=False,
        )
    cumsum_buffer = torch.empty(
        num_experts + 2,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    from sgl_kernel import moe_align_block_size

    # The +1 bucket maps filtered expert -1 to a skipped expert block.
    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        int(block_size),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,
    )
    return MoeAlignment(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_size=int(block_size),
        naive=False,
    )


__all__ = [
    "sgl_fp8_group_quantization_support",
    "sgl_moe_align_block_size",
    "sgl_moe_alignment_support",
    "sgl_per_token_group_quant_8bit",
]
