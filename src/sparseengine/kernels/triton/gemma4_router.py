from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _router_input_kernel(
    x_ptr,
    scale_ptr,
    output_ptr,
    stride,
    root_size: tl.constexpr,
    eps: tl.constexpr,
    hidden_size: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < hidden_size
    offsets = row * stride + cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / hidden_size
    element_dtype = x_ptr.dtype.element_ty
    x = (x * libdevice.pow(variance + eps, -0.5)).to(element_dtype).to(tl.float32)
    x = (
        (x * tl.load(scale_ptr + cols, mask=mask, other=0.0))
        .to(element_dtype)
        .to(tl.float32)
    )
    x = (x * root_size).to(element_dtype)
    tl.store(output_ptr + offsets, x, mask=mask)


@triton.jit
def _router_weights_kernel(
    probabilities_ptr,
    ids_ptr,
    scale_ptr,
    weights_ptr,
    probabilities_stride,
    top_k: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    routes = tl.arange(0, block)
    mask = routes < top_k
    experts = tl.load(ids_ptr + row * top_k + routes, mask=mask, other=0)
    values = tl.load(
        probabilities_ptr + row * probabilities_stride + experts,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values /= tl.sum(values, axis=0)
    values *= tl.load(scale_ptr + experts, mask=mask, other=0.0).to(tl.float32)
    tl.store(weights_ptr + row * top_k + routes, values, mask=mask)


def gemma4_router_input(
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    root_size: float,
    eps: float,
) -> torch.Tensor:
    if not hidden_states.is_cuda or hidden_states.dtype not in {
        torch.float16,
        torch.bfloat16,
    }:
        raise TypeError("Gemma 4 router input requires CUDA FP16 or BF16 tensors.")
    if hidden_states.stride(-1) != 1 or scale.shape != (hidden_states.shape[-1],):
        raise ValueError(
            "Gemma 4 router input requires contiguous features and matching scale."
        )
    if scale.device != hidden_states.device or scale.dtype != hidden_states.dtype:
        raise TypeError(
            "Gemma 4 router input scale must match activation dtype and device."
        )
    output = torch.empty_like(hidden_states)
    rows = hidden_states.reshape(-1, hidden_states.shape[-1])
    hidden_size = int(hidden_states.shape[-1])
    block = triton.next_power_of_2(hidden_size)
    _router_input_kernel[(rows.shape[0],)](
        hidden_states,
        scale,
        output,
        rows.stride(0),
        root_size=float(root_size),
        eps=float(eps),
        hidden_size=hidden_size,
        block=block,
        num_warps=min(max(block // 256, 1), 8),
    )
    return output


def gemma4_router_topk(
    logits: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts = int(logits.shape[-1])
    if not logits.is_cuda or logits.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("Gemma 4 router top-k requires CUDA FP16 or BF16 logits.")
    if (
        logits.ndim != 2
        or logits.stride(-1) != 1
        or per_expert_scale.shape != (num_experts,)
    ):
        raise ValueError(
            "Gemma 4 router top-k requires contiguous 2D logits and matching scales."
        )
    if (
        per_expert_scale.device != logits.device
        or per_expert_scale.dtype != logits.dtype
    ):
        raise TypeError(
            "Gemma 4 router expert scales must match logits dtype and device."
        )
    if not 0 < int(top_k) <= num_experts:
        raise ValueError(
            f"Invalid Gemma 4 router top-k {top_k} for {num_experts} experts."
        )
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    ids = probabilities.topk(int(top_k), dim=-1).indices
    weights = torch.empty(
        (logits.shape[0], top_k), dtype=torch.float32, device=logits.device
    )
    _router_weights_kernel[(int(logits.shape[0]),)](
        probabilities,
        ids,
        per_expert_scale,
        weights,
        probabilities.stride(0),
        top_k=int(top_k),
        block=triton.next_power_of_2(int(top_k)),
        num_warps=1,
    )
    return weights, ids


__all__ = ["gemma4_router_input", "gemma4_router_topk"]
