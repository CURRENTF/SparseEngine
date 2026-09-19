from __future__ import annotations

import torch

from sparseengine.kernels.triton.gemma4_gelu_and_mul import gelu_tanh_and_mul_fwd
from sparseengine.kernels.triton.moe import (
    _prepare_expert_assignment,
    _routed_gemm,
    _validate_fused_moe_inputs,
    moe_sum,
)
from sparseengine.kernels.triton.moe_config import device_info, resolve_moe_gemm_config


def _gemma4_moe_config(
    num_tokens: int, large_token_config: dict[str, int] | None
) -> dict[str, int] | None:
    if int(num_tokens) <= 32:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4,
        }
    if int(num_tokens) < 512:
        return None
    return None if large_token_config is None else dict(large_token_config)


def fused_gemma4_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
    large_token_config: dict[str, int] | None = None,
) -> torch.Tensor:
    """Run Gemma 4 routed GEGLU experts without changing generic MoE kernels."""

    num_experts, local_expert_start = int(num_experts), int(local_expert_start)
    _validate_fused_moe_inputs(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts,
        local_expert_start,
    )
    num_tokens, top_k = int(hidden_states.shape[0]), int(topk_ids.shape[1])
    intermediate_size = int(w13_weight.shape[1]) // 2
    hidden_size = int(hidden_states.shape[1])
    local_expert_end = local_expert_start + int(w13_weight.shape[0])
    device_name, capability = device_info(
        hidden_states.device.type,
        int(hidden_states.device.index),
    )
    w13_config = _gemma4_moe_config(
        num_tokens, large_token_config
    ) or resolve_moe_gemm_config(
        dtype=hidden_states.dtype,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=int(w13_weight.shape[0]),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        stage="w13",
        device_name=device_name,
        device_capability=capability,
    ).as_triton_kwargs()
    alignment = _prepare_expert_assignment(
        topk_ids,
        block_size=w13_config["BLOCK_SIZE_M"],
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
    )
    w13_output = torch.empty(
        (num_tokens * top_k, 2 * intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _routed_gemm(
        hidden_states,
        w13_weight,
        w13_output,
        topk_weights,
        alignment,
        input_top_k=top_k,
        multiply_routing_weight=False,
        launch_config=w13_config,
    )
    activated = gelu_tanh_and_mul_fwd(w13_output)
    w2_config = _gemma4_moe_config(
        num_tokens, large_token_config
    ) or resolve_moe_gemm_config(
        dtype=hidden_states.dtype,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=int(w13_weight.shape[0]),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        stage="w2",
        device_name=device_name,
        device_capability=capability,
    ).as_triton_kwargs()
    w2_config["BLOCK_SIZE_M"] = alignment.block_size
    w2_output = torch.empty(
        (num_tokens * top_k, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _routed_gemm(
        activated,
        w2_weight,
        w2_output,
        topk_weights,
        alignment,
        input_top_k=1,
        multiply_routing_weight=True,
        launch_config=w2_config,
    )
    return moe_sum(
        w2_output.view(num_tokens, top_k, hidden_size),
        topk_ids,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
    )
