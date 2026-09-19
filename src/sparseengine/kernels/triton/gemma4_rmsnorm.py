from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gemma4_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    row_stride,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    has_weight: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    mask = cols < hidden_size
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / hidden_size
    output = x * libdevice.pow(variance + eps, -0.5)
    if has_weight:
        output *= tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + row * hidden_size + cols, output, mask=mask)


def gemma4_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Gemma 4 Triton RMSNorm requires CUDA FP16 or BF16 input.")
    if not x.is_contiguous():
        raise ValueError("Gemma 4 Triton RMSNorm requires contiguous input.")
    hidden_size = int(x.shape[-1])
    if weight is not None and (
        weight.shape != (hidden_size,)
        or weight.device != x.device
        or weight.dtype != x.dtype
    ):
        raise ValueError(
            "Gemma 4 RMSNorm weight must match the input feature dimension, device, and dtype."
        )
    rows = x.reshape(-1, hidden_size)
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(hidden_size)
    _gemma4_rmsnorm_kernel[(rows.shape[0],)](
        rows,
        rows if weight is None else weight,
        output,
        rows.stride(0),
        hidden_size=hidden_size,
        eps=float(eps),
        has_weight=weight is not None,
        block_size=block_size,
        num_warps=min(max(block_size // 256, 1), 8),
    )
    return output


__all__ = ["gemma4_rmsnorm"]
