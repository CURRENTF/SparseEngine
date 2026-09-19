from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_fused_router_kernel(
    logits_ptr,
    scale_ptr,
    weights_ptr,
    ids_ptr,
    stride_logits,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    row = tl.program_id(0)
    experts = tl.arange(0, BLOCK_EXPERTS)
    valid = experts < NUM_EXPERTS
    logits = tl.load(
        logits_ptr + row * stride_logits + experts,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    # Pack a descending-float sort key and the expert id into one int64 value.
    min_int32 = -2147483648
    bits = logits.to(tl.int32, bitcast=True)
    keys = tl.where(bits >> 31 == 0, bits ^ -1, bits ^ min_int32)
    keys = tl.where(valid, keys, 0x7FFFFFFF)
    packed = ((keys.to(tl.int64) & 0xFFFFFFFF) << 32) | experts.to(tl.int64)
    sorted_packed = tl.sort(packed, descending=False)
    sorted_keys = ((sorted_packed >> 32) & 0xFFFFFFFF).to(tl.int32)
    sorted_ids = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    sorted_bits = tl.where(
        sorted_keys >> 31 < 0,
        sorted_keys ^ -1,
        sorted_keys ^ min_int32,
    )
    sorted_logits = sorted_bits.to(tl.float32, bitcast=True)

    selected = experts < TOP_K
    selected_logits = tl.where(selected, sorted_logits, -float("inf"))
    selected_max = tl.max(selected_logits, axis=0)
    probabilities = tl.where(
        selected,
        tl.exp2((sorted_logits - selected_max) * 1.4426950408889634),
        0.0,
    )
    probabilities /= tl.sum(probabilities, axis=0)
    probabilities *= tl.load(
        scale_ptr + sorted_ids,
        mask=selected,
        other=1.0,
    ).to(tl.float32)
    output_offsets = row * TOP_K + experts
    tl.store(weights_ptr + output_offsets, probabilities, mask=selected)
    tl.store(ids_ptr + output_offsets, sorted_ids, mask=selected)


def gemma4_fused_router_topk(
    logits: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not logits.is_cuda or logits.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("Gemma 4 fused router requires CUDA FP16 or BF16 logits.")
    num_experts = int(logits.shape[-1])
    if (
        logits.ndim != 2
        or logits.stride(-1) != 1
        or per_expert_scale.shape != (num_experts,)
        or per_expert_scale.stride(0) != 1
    ):
        raise ValueError(
            "Gemma 4 fused router requires contiguous [tokens, experts] logits "
            "and matching contiguous expert scales."
        )
    if per_expert_scale.device != logits.device or per_expert_scale.dtype != logits.dtype:
        raise TypeError("Gemma 4 fused router scales must match logits dtype and device.")
    if not 0 < int(top_k) <= num_experts or num_experts > 1024:
        raise ValueError(
            f"Gemma 4 fused router requires 0 < top_k <= experts <= 1024, got "
            f"top_k={top_k}, experts={num_experts}."
        )
    weights = torch.empty(
        (logits.shape[0], int(top_k)), dtype=torch.float32, device=logits.device
    )
    ids = torch.empty(
        (logits.shape[0], int(top_k)), dtype=torch.int32, device=logits.device
    )
    _gemma4_fused_router_kernel[(int(logits.shape[0]),)](
        logits,
        per_expert_scale,
        weights,
        ids,
        logits.stride(0),
        NUM_EXPERTS=num_experts,
        TOP_K=int(top_k),
        BLOCK_EXPERTS=triton.next_power_of_2(num_experts),
        num_warps=1,
    )
    return weights, ids


__all__ = ["gemma4_fused_router_topk"]
