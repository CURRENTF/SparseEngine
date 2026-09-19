from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _norm_rope(
    x_ptr,
    weight_ptr,
    rope_ptr,
    positions_ptr,
    token,
    head,
    stride_token,
    stride_head,
    rope_stride,
    eps: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    cols = tl.arange(0, block)
    mask = cols < head_dim
    offset = token * stride_token + head * stride_head + cols
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / head_dim
    x *= libdevice.pow(variance + eps, -0.5)
    x *= tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = x.to(x_ptr.dtype.element_ty).to(tl.float32)
    half = head_dim // 2
    pair = (cols + half) % head_dim
    other = tl.load(x_ptr + token * stride_token + head * stride_head + pair).to(
        tl.float32
    )
    other *= libdevice.pow(variance + eps, -0.5)
    other *= tl.load(weight_ptr + pair).to(tl.float32)
    other = other.to(x_ptr.dtype.element_ty).to(tl.float32)
    position = tl.load(positions_ptr + token)
    cos = tl.load(rope_ptr + position * rope_stride + cols % half).to(tl.float32)
    sin = tl.load(rope_ptr + position * rope_stride + half + cols % half).to(tl.float32)
    rotated = tl.where(cols < half, x * cos - other * sin, x * cos + other * sin)
    tl.store(x_ptr + offset, rotated, mask=mask)


@triton.jit
def _norm(
    x_ptr,
    weight_ptr,
    token,
    head,
    stride_token,
    stride_head,
    eps: tl.constexpr,
    head_dim: tl.constexpr,
    has_weight: tl.constexpr,
    block: tl.constexpr,
):
    cols = tl.arange(0, block)
    mask = cols < head_dim
    offset = token * stride_token + head * stride_head + cols
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / head_dim
    x *= libdevice.pow(variance + eps, -0.5)
    if has_weight:
        x *= tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(x_ptr + offset, x, mask=mask)


@triton.jit
def _gemma4_qkv_norm_rope_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    rope_ptr,
    positions_ptr,
    q_stride_token,
    q_stride_head,
    k_stride_token,
    k_stride_head,
    v_stride_token,
    v_stride_head,
    rope_stride,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    has_kv: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    if head < num_q_heads:
        _norm_rope(
            q_ptr,
            q_weight_ptr,
            rope_ptr,
            positions_ptr,
            token,
            head,
            q_stride_token,
            q_stride_head,
            rope_stride,
            eps,
            head_dim,
            block,
        )
    elif has_kv and head < num_q_heads + num_kv_heads:
        kv_head = head - num_q_heads
        _norm_rope(
            k_ptr,
            k_weight_ptr,
            rope_ptr,
            positions_ptr,
            token,
            kv_head,
            k_stride_token,
            k_stride_head,
            rope_stride,
            eps,
            head_dim,
            block,
        )
    elif has_kv:
        kv_head = head - num_q_heads - num_kv_heads
        _norm(
            v_ptr,
            q_weight_ptr,
            token,
            kv_head,
            v_stride_token,
            v_stride_head,
            eps,
            head_dim,
            False,
            block,
        )


def gemma4_qkv_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor | None,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> None:
    if q.dtype not in (torch.float16, torch.bfloat16) or not q.is_cuda:
        raise TypeError(
            "Gemma 4 fused QKV norm-RoPE requires CUDA FP16 or BF16 tensors."
        )
    if q.ndim != 3 or q.stride(-1) != 1 or q.shape[-1] not in {256, 512}:
        raise ValueError(
            "Gemma 4 fused QKV norm-RoPE requires contiguous rank-3 heads of 256 or 512."
        )
    has_kv = k is not None and v is not None
    if (k is None) != (v is None):
        raise ValueError("Gemma 4 fused QKV norm-RoPE requires K and V together.")
    if has_kv != (k_weight is not None):
        raise ValueError(
            "Gemma 4 fused QKV norm-RoPE requires K weight with K/V tensors."
        )
    head_dim = int(q.shape[-1])
    if (
        q_weight.shape != (head_dim,)
        or q_weight.device != q.device
        or q_weight.dtype != q.dtype
    ):
        raise ValueError(
            "Gemma 4 Q norm weight must match Q head size, device, and dtype."
        )
    if has_kv and (
        k.shape[0] != q.shape[0]
        or k.shape[-1] != head_dim
        or v.shape != k.shape
        or any(
            t.device != q.device or t.dtype != q.dtype or t.stride(-1) != 1
            for t in (k, v)
        )
        or k_weight.shape != (head_dim,)
        or k_weight.device != q.device
        or k_weight.dtype != q.dtype
    ):
        raise ValueError(
            "Gemma 4 K/V tensors and K norm weight must match Q layout and dtype."
        )
    if (
        positions.shape != (q.shape[0],)
        or positions.device != q.device
        or rope_cache.device != q.device
        or rope_cache.shape[-1] != head_dim
    ):
        raise ValueError(
            "Gemma 4 positions and RoPE cache must match Q tokens, device, and head size."
        )
    num_kv_heads = int(k.shape[1]) if k is not None else 0
    _gemma4_qkv_norm_rope_kernel[(int(q.shape[0]), int(q.shape[1]) + 2 * num_kv_heads)](
        q,
        q if k is None else k,
        q if v is None else v,
        q_weight,
        q_weight if k_weight is None else k_weight,
        rope_cache,
        positions,
        q.stride(0),
        q.stride(1),
        0 if k is None else k.stride(0),
        0 if k is None else k.stride(1),
        0 if v is None else v.stride(0),
        0 if v is None else v.stride(1),
        rope_cache.stride(0),
        num_q_heads=int(q.shape[1]),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=float(eps),
        has_kv=has_kv,
        block=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


__all__ = ["gemma4_qkv_norm_rope"]
