"""Fused CUDA kernels for paired column-parallel RMSNorm."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def _paired_square_sum_kernel(
    x_ptr,
    other_ptr,
    square_sums_ptr,
    stride_x_row,
    stride_other_row,
    stride_sums_row,
    x_hidden_size: tl.constexpr,
    other_hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_mask = cols < x_hidden_size
    other_mask = cols < other_hidden_size
    x = tl.load(
        x_ptr + row * stride_x_row + cols,
        mask=x_mask,
        other=0.0,
    ).to(tl.float32)
    other = tl.load(
        other_ptr + row * stride_other_row + cols,
        mask=other_mask,
        other=0.0,
    ).to(tl.float32)
    x_sum = tl.sum(tl.where(x_mask, x * x, 0.0), axis=0)
    other_sum = tl.sum(tl.where(other_mask, other * other, 0.0), axis=0)
    sums_row = square_sums_ptr + row * stride_sums_row
    tl.store(sums_row, x_sum)
    tl.store(sums_row + 1, other_sum)


@triton.jit
def _paired_rms_apply_kernel(
    x_ptr,
    other_ptr,
    square_sums_ptr,
    x_weight_ptr,
    other_weight_ptr,
    x_output_ptr,
    other_output_ptr,
    stride_x_row,
    stride_other_row,
    stride_sums_row,
    stride_x_output_row,
    stride_other_output_row,
    x_global_hidden_size: tl.constexpr,
    other_global_hidden_size: tl.constexpr,
    x_hidden_size: tl.constexpr,
    other_hidden_size: tl.constexpr,
    x_eps: tl.constexpr,
    other_eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_mask = cols < x_hidden_size
    other_mask = cols < other_hidden_size
    sums_row = square_sums_ptr + row * stride_sums_row
    x_inv_rms = tl.rsqrt(tl.load(sums_row) / x_global_hidden_size + x_eps)
    other_inv_rms = tl.rsqrt(
        tl.load(sums_row + 1) / other_global_hidden_size + other_eps
    )

    x = tl.load(
        x_ptr + row * stride_x_row + cols,
        mask=x_mask,
        other=0.0,
    ).to(tl.float32)
    other = tl.load(
        other_ptr + row * stride_other_row + cols,
        mask=other_mask,
        other=0.0,
    ).to(tl.float32)
    x_weight = tl.load(x_weight_ptr + cols, mask=x_mask, other=0.0)
    other_weight = tl.load(
        other_weight_ptr + cols,
        mask=other_mask,
        other=0.0,
    )
    x_dtype = x_output_ptr.dtype.element_ty
    other_dtype = other_output_ptr.dtype.element_ty
    x_normalized = (x * x_inv_rms).to(x_dtype)
    other_normalized = (other * other_inv_rms).to(other_dtype)
    tl.store(
        x_output_ptr + row * stride_x_output_row + cols,
        x_normalized * x_weight,
        mask=x_mask,
    )
    tl.store(
        other_output_ptr + row * stride_other_output_row + cols,
        other_normalized * other_weight,
        mask=other_mask,
    )


def _validate_pair(
    x: torch.Tensor,
    other: torch.Tensor,
    x_weight: torch.Tensor | None = None,
    other_weight: torch.Tensor | None = None,
) -> None:
    if not x.is_cuda or not other.is_cuda:
        raise ValueError("Paired column-parallel RMSNorm requires CUDA tensors.")
    if x.device != other.device or x.dtype != other.dtype:
        raise ValueError("Paired RMSNorm inputs must share a device and dtype.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("Paired RMSNorm supports only FP16 and BF16 inputs.")
    if x.ndim != 2 or other.ndim != 2 or x.shape[0] != other.shape[0]:
        raise ValueError("Paired RMSNorm inputs must have matching two-dimensional rows.")
    if x.shape[1] <= 0 or other.shape[1] <= 0:
        raise ValueError("Paired RMSNorm requires non-empty feature dimensions.")
    if x.stride(1) != 1 or other.stride(1) != 1:
        raise ValueError("Paired RMSNorm requires contiguous feature dimensions.")
    if x_weight is None or other_weight is None:
        return
    for name, weight, hidden_size in (
        ("x", x_weight, x.shape[1]),
        ("other", other_weight, other.shape[1]),
    ):
        if weight.shape != (hidden_size,):
            raise ValueError(
                f"Paired RMSNorm {name} weight must have shape ({hidden_size},)."
            )
        if weight.device != x.device or weight.dtype != x.dtype:
            raise ValueError(
                f"Paired RMSNorm {name} weight must match the inputs."
            )
        if not weight.is_contiguous():
            raise ValueError(f"Paired RMSNorm {name} weight must be contiguous.")


def _launch_config(x: torch.Tensor, other: torch.Tensor) -> tuple[int, int]:
    hidden_size = max(int(x.shape[1]), int(other.shape[1]))
    block_size = triton.next_power_of_2(hidden_size)
    if block_size * x.element_size() > 65536:
        raise RuntimeError(
            "Paired RMSNorm does not support feature dimensions occupying "
            "more than 64 KiB per row."
        )
    num_warps = min(max(block_size // 256, 1), 8)
    return block_size, num_warps


def paired_square_sums(x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Return one FP32 square-sum pair per input row."""
    _validate_pair(x, other)
    block_size, num_warps = _launch_config(x, other)
    square_sums = torch.empty((x.shape[0], 2), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _paired_square_sum_kernel[(x.shape[0],)](
            x,
            other,
            square_sums,
            x.stride(0),
            other.stride(0),
            square_sums.stride(0),
            x_hidden_size=int(x.shape[1]),
            other_hidden_size=int(other.shape[1]),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return square_sums


def paired_rms_apply(
    x: torch.Tensor,
    other: torch.Tensor,
    square_sums: torch.Tensor,
    x_weight: torch.Tensor,
    other_weight: torch.Tensor,
    *,
    x_global_hidden_size: int,
    other_global_hidden_size: int,
    x_eps: float,
    other_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply paired RMSNorm after the statistics have been reduced across TP."""
    _validate_pair(x, other, x_weight, other_weight)
    expected_sums_shape = (x.shape[0], 2)
    if (
        square_sums.shape != expected_sums_shape
        or square_sums.dtype != torch.float32
        or square_sums.device != x.device
        or not square_sums.is_contiguous()
    ):
        raise ValueError(
            "Paired RMSNorm square sums must be contiguous FP32 values with "
            f"shape {expected_sums_shape} on the input device."
        )
    block_size, num_warps = _launch_config(x, other)
    x_output = torch.empty_like(x, memory_format=torch.contiguous_format)
    other_output = torch.empty_like(other, memory_format=torch.contiguous_format)
    with torch.cuda.device(x.device):
        _paired_rms_apply_kernel[(x.shape[0],)](
            x,
            other,
            square_sums,
            x_weight,
            other_weight,
            x_output,
            other_output,
            x.stride(0),
            other.stride(0),
            square_sums.stride(0),
            x_output.stride(0),
            other_output.stride(0),
            x_global_hidden_size=int(x_global_hidden_size),
            other_global_hidden_size=int(other_global_hidden_size),
            x_hidden_size=int(x.shape[1]),
            other_hidden_size=int(other.shape[1]),
            x_eps=float(x_eps),
            other_eps=float(other_eps),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return x_output, other_output
