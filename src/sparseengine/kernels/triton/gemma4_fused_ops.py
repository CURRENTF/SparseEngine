from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rmsnorm_residual_kernel(
    x_ptr,
    weight_ptr,
    residual_ptr,
    scalar_ptr,
    output_ptr,
    stride,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    apply_scalar: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < hidden_size
    offsets = row * stride + cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / hidden_size
    x *= libdevice.pow(variance + eps, -0.5)
    x *= tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = x.to(x_ptr.dtype.element_ty).to(tl.float32)
    output = x + tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = output.to(x_ptr.dtype.element_ty)
    if apply_scalar:
        output *= tl.load(scalar_ptr).to(tl.float32)
    tl.store(output_ptr + offsets, output, mask=mask)


def gemma4_rmsnorm_residual(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
    scalar: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.shape != residual.shape or x.stride(-1) != 1 or residual.stride(-1) != 1:
        raise ValueError(
            "Gemma 4 fused RMSNorm-residual requires matching contiguous features."
        )
    if not x.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise TypeError("Gemma 4 fused RMSNorm-residual requires CUDA tensors.")
    if x.dtype not in {torch.float16, torch.bfloat16} or residual.dtype != x.dtype:
        raise TypeError(
            "Gemma 4 fused RMSNorm-residual requires matching FP16 or BF16 tensors."
        )
    if weight.shape != (x.shape[-1],) or weight.device != x.device:
        raise ValueError(
            "Gemma 4 fused RMSNorm-residual requires a matching device-local weight."
        )
    if scalar is not None and (scalar.numel() != 1 or scalar.device != x.device):
        raise ValueError(
            "Gemma 4 fused RMSNorm-residual scalar must be device-local and scalar."
        )
    output = torch.empty_like(x)
    rows = x.reshape(-1, x.shape[-1])
    hidden_size = int(x.shape[-1])
    block = triton.next_power_of_2(hidden_size)
    _rmsnorm_residual_kernel[(rows.shape[0],)](
        x,
        weight,
        residual,
        weight if scalar is None else scalar,
        output,
        rows.stride(0),
        hidden_size=hidden_size,
        eps=float(eps),
        apply_scalar=scalar is not None,
        block=block,
        num_warps=min(max(block // 256, 1), 8),
    )
    return output


@triton.jit
def _gelu_mul_kernel(
    gate_ptr,
    input_ptr,
    rows,
    cols,
    gate_stride,
    input_stride,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < cols
    gate = tl.load(gate_ptr + row * gate_stride + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    gate = (gate * tl.sigmoid(2.0 * inner)).to(gate_ptr.dtype.element_ty)
    value = tl.load(input_ptr + row * input_stride + offsets, mask=mask, other=0.0)
    tl.store(gate_ptr + row * gate_stride + offsets, gate * value, mask=mask)


def gemma4_gelu_mul(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if gate.shape != value.shape or gate.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError(
            "Gemma 4 fused GELU-multiply requires matching contiguous features."
        )
    if not gate.is_cuda or not value.is_cuda:
        raise TypeError("Gemma 4 fused GELU-multiply requires CUDA tensors.")
    if gate.dtype not in {torch.float16, torch.bfloat16} or value.dtype != gate.dtype:
        raise TypeError(
            "Gemma 4 fused GELU-multiply requires matching FP16 or BF16 tensors."
        )
    rows, cols = gate.reshape(-1, gate.shape[-1]).shape
    block = triton.next_power_of_2(cols)
    _gelu_mul_kernel[(rows,)](
        gate,
        value,
        rows,
        cols,
        gate.stride(0),
        value.stride(0),
        block=block,
        num_warps=min(max(block // 256, 1), 8),
    )
    return gate


__all__ = ["gemma4_gelu_mul", "gemma4_rmsnorm_residual"]
