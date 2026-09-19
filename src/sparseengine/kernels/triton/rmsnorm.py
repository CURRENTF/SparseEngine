"""Inference-only Triton RMSNorm kernels.

Adapted from SGLang's Apache-2.0 licensed Triton normalization kernel:
reference/sglang/python/sglang/jit_kernel/diffusion/triton/norm.py
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def _rms_norm_fwd_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    residual_out_ptr,
    stride_x_row,
    stride_residual_row,
    stride_output_row,
    stride_residual_out_row,
    hidden_size,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    ZERO_CENTERED_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    x = tl.load(
        x_ptr + row * stride_x_row + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(
            residual_ptr + row * stride_residual_row + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x += residual
        tl.store(
            residual_out_ptr + row * stride_residual_out_row + cols,
            x,
            mask=mask,
        )

    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / hidden_size
    normalized = x * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED_WEIGHT:
        weight += 1.0
    output = normalized * weight
    tl.store(
        output_ptr + row * stride_output_row + cols,
        output,
        mask=mask,
    )


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
) -> None:
    if not x.is_cuda:
        raise RuntimeError("Triton RMSNorm requires a CUDA input tensor.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "Triton RMSNorm supports FP16 and BF16 inputs, "
            f"got {x.dtype}."
        )
    if x.ndim == 0:
        raise ValueError("Triton RMSNorm requires at least one input dimension.")
    if x.shape[-1] <= 0:
        raise ValueError("Triton RMSNorm requires a non-empty feature dimension.")
    if x.stride(-1) != 1:
        raise ValueError(
            "Triton RMSNorm requires the input's last dimension to be contiguous."
        )
    if weight.shape != (x.shape[-1],):
        raise ValueError(
            f"RMSNorm weight must have shape ({x.shape[-1]},), "
            f"got {tuple(weight.shape)}."
        )
    if weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError(
            "RMSNorm input and weight must have the same device and dtype, "
            f"got x=({x.device}, {x.dtype}) and "
            f"weight=({weight.device}, {weight.dtype})."
        )
    if not weight.is_contiguous():
        raise ValueError("Triton RMSNorm requires a contiguous weight tensor.")
    if residual is None:
        return
    if residual.shape != x.shape:
        raise ValueError(
            "RMSNorm residual must match the input shape, "
            f"got x={tuple(x.shape)} and residual={tuple(residual.shape)}."
        )
    if residual.device != x.device or residual.dtype != x.dtype:
        raise ValueError(
            "RMSNorm input and residual must have the same device and dtype, "
            f"got x=({x.device}, {x.dtype}) and "
            f"residual=({residual.device}, {residual.dtype})."
        )
    if residual.stride(-1) != 1:
        raise ValueError(
            "Triton fused add RMSNorm requires the residual's last dimension "
            "to be contiguous."
        )
    if residual.data_ptr() == x.data_ptr():
        raise ValueError(
            "Triton fused add RMSNorm requires distinct input and residual tensors."
        )


def _block_size(hidden_size: int, element_size: int) -> int:
    max_fused_size = 65536 // element_size
    block_size = min(max_fused_size, triton.next_power_of_2(hidden_size))
    if hidden_size > block_size:
        raise RuntimeError(
            "Triton RMSNorm does not support feature dimensions occupying "
            "more than 64 KiB per row."
        )
    return block_size


def _view_rows_without_copy(x: torch.Tensor, name: str) -> torch.Tensor:
    try:
        rows = x.view(-1, x.shape[-1])
    except RuntimeError as exc:
        raise ValueError(
            f"Triton fused add RMSNorm requires {name} to flatten into rows "
            "without copying."
        ) from exc
    if rows.data_ptr() != x.data_ptr():
        raise ValueError(
            f"Triton fused add RMSNorm requires {name} to flatten into rows "
            "without copying."
        )
    return rows


def _launch(
    x_rows: torch.Tensor,
    weight: torch.Tensor,
    output_rows: torch.Tensor,
    eps: float,
    *,
    zero_centered_weight: bool,
    residual_rows: torch.Tensor | None = None,
    residual_out_rows: torch.Tensor | None = None,
) -> None:
    row_count, hidden_size = x_rows.shape
    block_size = _block_size(hidden_size, x_rows.element_size())
    num_warps = min(max(block_size // 256, 1), 8)
    has_residual = residual_rows is not None

    with torch.cuda.device(x_rows.device):
        _rms_norm_fwd_kernel[(row_count,)](
            x_rows,
            residual_rows,
            weight,
            output_rows,
            residual_out_rows,
            x_rows.stride(0),
            residual_rows.stride(0) if residual_rows is not None else 0,
            output_rows.stride(0),
            residual_out_rows.stride(0) if residual_out_rows is not None else 0,
            hidden_size,
            float(eps),
            HAS_RESIDUAL=has_residual,
            ZERO_CENTERED_WEIGHT=bool(zero_centered_weight),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )


def rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered_weight: bool = False,
) -> torch.Tensor:
    """Return RMSNorm without modifying ``x``."""

    _validate_inputs(x, weight, residual=None)
    hidden_size = x.shape[-1]
    x_rows = x.reshape(-1, hidden_size)
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    output_rows = output.view(-1, hidden_size)
    _launch(
        x_rows,
        weight,
        output_rows,
        eps,
        zero_centered_weight=zero_centered_weight,
    )
    return output


def fused_add_rmsnorm_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered_weight: bool = False,
) -> None:
    """Normalize ``x + residual`` into ``x`` and store the sum in ``residual``."""

    _validate_inputs(x, weight, residual)
    x_rows = _view_rows_without_copy(x, "input")
    residual_rows = _view_rows_without_copy(residual, "residual")
    _launch(
        x_rows,
        weight,
        x_rows,
        eps,
        zero_centered_weight=zero_centered_weight,
        residual_rows=residual_rows,
        residual_out_rows=residual_rows,
    )
