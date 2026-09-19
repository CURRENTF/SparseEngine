from __future__ import annotations

import importlib
import re
from functools import lru_cache

import torch

from sparseengine.kernels.external.flashinfer.support import (
    flashinfer_kernel_health,
    flashinfer_kernel_support,
)
from sparseengine.kernels.external.support import ExternalKernelContractError


@lru_cache(maxsize=1)
def _cutlass_fp8_moe_op():
    feature = "SM90 CUTLASS FP8 MoE"
    _, reason = flashinfer_kernel_support(feature)
    try:
        module = importlib.import_module("flashinfer.fused_moe")
        function = getattr(module, "cutlass_fused_moe")
        workspace_size = getattr(module, "cutlass_fused_moe_workspace_size")
        activation_type = getattr(
            importlib.import_module("flashinfer.tllm_enums").ActivationType,
            "Swiglu",
        )
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"failed to load: {type(error).__name__}: {error}",
        ) from error
    if not callable(function):
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            "flashinfer.fused_moe.cutlass_fused_moe is not callable",
        )
    if not callable(workspace_size):
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            "flashinfer.fused_moe.cutlass_fused_moe_workspace_size is not callable",
        )
    return function, workspace_size, activation_type, reason


def flashinfer_cutlass_fp8_moe_support() -> tuple[bool, str]:
    health = flashinfer_kernel_health()
    if not health.ready:
        flashinfer_kernel_support("SM90 CUTLASS FP8 MoE")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(health.version))
    parsed = tuple(map(int, match.groups())) if match else None
    if parsed is None or parsed < (0, 6, 17):
        return False, (
            "reusable CUTLASS FP8 MoE workspace requires "
            f"flashinfer-python>=0.6.17, got {health.version}"
        )
    _, _, _, reason = _cutlass_fp8_moe_op()
    return True, reason


def flashinfer_cutlass_fused_moe_workspace_size(
    *,
    max_num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    activation_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
    device: torch.device,
) -> int:
    _, workspace_size, activation_type, _ = _cutlass_fp8_moe_op()
    return int(
        workspace_size(
            int(max_num_tokens),
            int(hidden_size),
            int(intermediate_size),
            int(num_experts),
            int(top_k),
            x_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            output_dtype=activation_dtype,
            activation_type=activation_type,
            tp_size=int(tp_size),
            tp_rank=int(tp_rank),
            ep_size=int(ep_size),
            ep_rank=int(ep_rank),
            use_deepseek_fp8_block_scale=True,
            use_fused_finalize=False,
            device=device,
        )
    )


def flashinfer_cutlass_fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale_inv: torch.Tensor,
    w2_scale_inv: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
    output: torch.Tensor,
    workspace_buffer: torch.Tensor,
) -> None:
    function, _, activation_type, _ = _cutlass_fp8_moe_op()
    function(
        hidden_states,
        topk_ids.to(dtype=torch.int32),
        topk_weights.to(dtype=torch.float32),
        w13_weight,
        w2_weight,
        hidden_states.dtype,
        quant_scales=[w13_scale_inv, w2_scale_inv],
        tp_size=int(tp_size),
        tp_rank=int(tp_rank),
        ep_size=int(ep_size),
        ep_rank=int(ep_rank),
        output=output,
        workspace_buffer=workspace_buffer,
        use_deepseek_fp8_block_scale=True,
        use_fused_finalize=False,
        enable_pdl=None,
        activation_type=activation_type,
    )


__all__ = [
    "flashinfer_cutlass_fp8_moe_support",
    "flashinfer_cutlass_fused_moe",
    "flashinfer_cutlass_fused_moe_workspace_size",
]
