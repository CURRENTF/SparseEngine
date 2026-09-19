from __future__ import annotations

import torch
import triton
import triton.language as tl


_SUPPORTED_SHAPES = frozenset({(64, 4), (256, 8)})


@triton.jit
def _fused_topk_biased_sigmoid_kernel(
    logits_ptr,
    correction_bias_ptr,
    weights_ptr,
    ids_ptr,
    stride_logits_m,
    stride_weights_m,
    stride_ids_m,
    normalization_epsilon: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    expert_mask = offsets < NUM_EXPERTS
    logits = tl.load(
        logits_ptr + row * stride_logits_m + offsets,
        mask=expert_mask,
        other=-float("inf"),
    )
    routing_weights = tl.sigmoid(logits)
    correction_bias = tl.load(
        correction_bias_ptr + offsets,
        mask=expert_mask,
        other=0.0,
    )
    scores = routing_weights + correction_bias
    selection_values = tl.where(scores == scores, scores, float("inf"))
    threshold = tl.min(tl.topk(selection_values, TOP_K), axis=0)
    greater_mask = expert_mask & (selection_values > threshold)
    equal_mask = expert_mask & (selection_values == threshold)
    greater_rank = tl.cumsum(greater_mask.to(tl.int32), axis=0) - 1
    equal_rank = tl.cumsum(equal_mask.to(tl.int32), axis=0) - 1
    num_greater = tl.sum(greater_mask.to(tl.int32), axis=0)
    selected_equal = equal_mask & (equal_rank < TOP_K - num_greater)
    selected = greater_mask | selected_equal
    output_slot = tl.where(
        greater_mask,
        greater_rank,
        num_greater + equal_rank,
    )
    denominator = tl.sum(
        tl.where(selected, routing_weights, 0.0),
        axis=0,
    )
    weights_base = weights_ptr + row * stride_weights_m
    ids_base = ids_ptr + row * stride_ids_m
    tl.store(
        weights_base + output_slot,
        routing_weights
        / (denominator + normalization_epsilon)
        * routed_scaling_factor,
        mask=selected,
    )
    tl.store(ids_base + output_slot, offsets, mask=selected)


@triton.jit
def _topk_biased_sigmoid_kernel(
    routing_weights_ptr,
    correction_bias_ptr,
    ids_ptr,
    stride_routing_weights_m,
    stride_ids_m,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    expert_mask = offsets < NUM_EXPERTS
    routing_weights = tl.load(
        routing_weights_ptr + row * stride_routing_weights_m + offsets,
        mask=expert_mask,
        other=-float("inf"),
    )
    correction_bias = tl.load(
        correction_bias_ptr + offsets,
        mask=expert_mask,
        other=0.0,
    )
    scores = routing_weights + correction_bias

    # Match torch.topk(sorted=False): values strictly above the kth threshold
    # are emitted first, followed by first-seen threshold ties.
    selection_values = tl.where(scores == scores, scores, float("inf"))
    threshold = tl.min(tl.topk(selection_values, TOP_K), axis=0)
    greater_mask = expert_mask & (selection_values > threshold)
    equal_mask = expert_mask & (selection_values == threshold)
    greater_rank = tl.cumsum(greater_mask.to(tl.int32), axis=0) - 1
    equal_rank = tl.cumsum(equal_mask.to(tl.int32), axis=0) - 1
    num_greater = tl.sum(greater_mask.to(tl.int32), axis=0)
    selected_equal = equal_mask & (equal_rank < TOP_K - num_greater)
    selected = greater_mask | selected_equal
    output_slot = tl.where(greater_mask, greater_rank, num_greater + equal_rank)

    ids_base = ids_ptr + row * stride_ids_m
    tl.store(ids_base + output_slot, offsets, mask=selected)


def _validate_inputs(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
) -> tuple[int, int]:
    if not router_logits.is_cuda or not correction_bias.is_cuda:
        raise ValueError("Biased-sigmoid routing requires CUDA tensors.")
    if router_logits.device != correction_bias.device:
        raise ValueError("router_logits and correction_bias must be on one device.")
    if router_logits.ndim != 2:
        raise ValueError(
            "router_logits must have shape [tokens, experts], got "
            f"{tuple(router_logits.shape)}."
        )
    num_tokens, num_experts = map(int, router_logits.shape)
    if tuple(correction_bias.shape) != (num_experts,):
        raise ValueError(
            f"correction_bias must have shape [{num_experts}], got "
            f"{tuple(correction_bias.shape)}."
        )
    if router_logits.dtype != torch.float32 or correction_bias.dtype != torch.float32:
        raise TypeError(
            "Biased-sigmoid routing requires FP32 logits and correction_bias, "
            f"got {router_logits.dtype} and {correction_bias.dtype}."
        )
    if not router_logits.is_contiguous() or not correction_bias.is_contiguous():
        raise ValueError("Biased-sigmoid router inputs must be contiguous.")
    if num_tokens <= 0:
        raise ValueError("Biased-sigmoid routing requires at least one token.")
    shape = (num_experts, int(top_k))
    if shape not in _SUPPORTED_SHAPES:
        raise ValueError(
            "Unsupported biased-sigmoid router shape: "
            f"num_experts={num_experts}, top_k={top_k}; supported="
            f"{sorted(_SUPPORTED_SHAPES)}."
        )
    return num_tokens, num_experts


def fused_topk_biased_sigmoid(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    routed_scaling_factor: float,
    normalization_epsilon: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route the fixed GLM 64x4 shape in one graph-capturable kernel."""

    num_tokens, num_experts = _validate_inputs(
        router_logits,
        correction_bias,
        top_k=top_k,
    )
    if (num_experts, int(top_k)) != (64, 4):
        raise ValueError(
            "Fused biased-sigmoid routing requires 64 experts and top-k 4, "
            f"got {num_experts} and {top_k}."
        )
    weights = torch.empty(
        (num_tokens, int(top_k)),
        dtype=torch.float32,
        device=router_logits.device,
    )
    ids = torch.empty(
        (num_tokens, int(top_k)),
        dtype=torch.int32,
        device=router_logits.device,
    )
    _fused_topk_biased_sigmoid_kernel[(num_tokens,)](
        router_logits,
        correction_bias,
        weights,
        ids,
        router_logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        normalization_epsilon=float(normalization_epsilon),
        routed_scaling_factor=float(routed_scaling_factor),
        NUM_EXPERTS=num_experts,
        TOP_K=int(top_k),
        BLOCK_SIZE=triton.next_power_of_2(num_experts),
        num_warps=1,
    )
    return weights, ids


def topk_biased_sigmoid(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    normalization_epsilon: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts with correction bias and return unbiased sigmoid weights."""

    num_tokens, num_experts = _validate_inputs(
        router_logits,
        correction_bias,
        top_k=top_k,
    )
    routing_weights = torch.sigmoid(router_logits)
    ids = torch.empty(
        (num_tokens, int(top_k)),
        dtype=torch.int64,
        device=router_logits.device,
    )
    block_size = triton.next_power_of_2(num_experts)
    _topk_biased_sigmoid_kernel[(num_tokens,)](
        routing_weights,
        correction_bias,
        ids,
        routing_weights.stride(0),
        ids.stride(0),
        NUM_EXPERTS=num_experts,
        TOP_K=int(top_k),
        BLOCK_SIZE=block_size,
        num_warps=2 if num_tokens <= 256 else 1,
    )
    weights = routing_weights.gather(1, ids)
    weights = weights / (
        weights.sum(dim=-1, keepdim=True) + float(normalization_epsilon)
    )
    return weights, ids


__all__ = ["fused_topk_biased_sigmoid", "topk_biased_sigmoid"]
