from __future__ import annotations

import torch
import triton
import triton.language as tl


FP8_BLOCK_SIZE = 128
_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16)


@triton.jit
def _fp8_blockwise_matmul_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        remaining_k = K - k_start * BLOCK_K
        x_raw = tl.load(
            x_ptr
            + offsets_m[:, None] * stride_xm
            + offsets_k[None, :] * stride_xk,
            mask=(offsets_m[:, None] < M)
            & (offsets_k[None, :] < remaining_k),
            other=0.0,
        ).to(tl.float32)
        x_scale = tl.max(tl.abs(x_raw), axis=1) / 448.0
        x_quant = (x_raw / tl.maximum(x_scale[:, None], 1.0e-12)).to(
            tl.float8e4nv
        )
        weight = tl.load(
            weight_ptr
            + offsets_n[None, :] * stride_wn
            + offsets_k[:, None] * stride_wk,
            mask=(offsets_n[None, :] < N)
            & (offsets_k[:, None] < remaining_k),
            other=0.0,
        )
        weight_scale = tl.load(
            scale_ptr + pid_n * stride_sn + k_start * stride_sk
        ).to(tl.float32)
        accumulator += (
            tl.dot(x_quant, weight)
            * x_scale[:, None]
            * weight_scale
        )
        x_ptr += BLOCK_K * stride_xk
        weight_ptr += BLOCK_K * stride_wk

    tl.store(
        output_ptr
        + offsets_m[:, None] * stride_om
        + offsets_n[None, :] * stride_on,
        accumulator,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )


def fp8_blockwise_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not x.is_cuda or not weight.is_cuda or not weight_scale_inv.is_cuda:
        raise ValueError("Triton block-FP8 matmul requires CUDA tensors.")
    if x.device != weight.device or x.device != weight_scale_inv.device:
        raise ValueError("Triton block-FP8 matmul tensors must share one device.")
    if x.dtype not in _ACTIVATION_DTYPES:
        raise TypeError(f"FP8 matmul activations must be BF16 or FP16, got {x.dtype}.")
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"FP8 matmul weight must be E4M3, got {weight.dtype}.")
    if weight_scale_inv.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            "FP8 matmul weight scales must be FP32 or BF16, got "
            f"{weight_scale_inv.dtype}."
        )
    if x.ndim != 2 or weight.ndim != 2 or weight_scale_inv.ndim != 2:
        raise ValueError("Triton block-FP8 matmul expects rank-2 tensors.")
    if int(x.shape[1]) != int(weight.shape[1]):
        raise ValueError(
            f"FP8 matmul K mismatch: x={tuple(x.shape)}, weight={tuple(weight.shape)}."
        )
    expected_scale_shape = (
        triton.cdiv(int(weight.shape[0]), FP8_BLOCK_SIZE),
        triton.cdiv(int(weight.shape[1]), FP8_BLOCK_SIZE),
    )
    if tuple(weight_scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            "FP8 matmul scale shape mismatch: "
            f"expected={expected_scale_shape}, got={tuple(weight_scale_inv.shape)}."
        )
    if not x.is_contiguous() or not weight.is_contiguous() or not weight_scale_inv.is_contiguous():
        raise ValueError("Triton block-FP8 matmul requires contiguous tensors.")

    if output_dtype is None:
        output_dtype = x.dtype
    if output_dtype not in _ACTIVATION_DTYPES:
        raise TypeError(f"FP8 matmul output must be BF16 or FP16, got {output_dtype}.")
    output = torch.empty(
        (int(x.shape[0]), int(weight.shape[0])),
        dtype=output_dtype,
        device=x.device,
    )
    block_m = 16
    grid = (
        triton.cdiv(int(x.shape[0]), block_m),
        triton.cdiv(int(weight.shape[0]), FP8_BLOCK_SIZE),
    )
    _fp8_blockwise_matmul_kernel[grid](
        x,
        weight,
        weight_scale_inv,
        output,
        int(x.shape[0]),
        N=int(weight.shape[0]),
        K=int(weight.shape[1]),
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
        stride_sn=weight_scale_inv.stride(0),
        stride_sk=weight_scale_inv.stride(1),
        stride_om=output.stride(0),
        stride_on=output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=FP8_BLOCK_SIZE,
        BLOCK_K=FP8_BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )
    return output
