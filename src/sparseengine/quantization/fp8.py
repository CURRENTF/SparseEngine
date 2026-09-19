from __future__ import annotations

import torch
import torch.nn.functional as F


def expand_fp8_tensor_scale(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Losslessly represent a checkpoint tensor scale in the block runtime layout."""
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("Per-tensor FP8 loading requires a rank-2 E4M3 weight.")
    if scale.numel() != 1 or scale.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Per-tensor FP8 weight_scale must be one BF16 or FP32 scalar.")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("Per-tensor FP8 weight_scale must be finite and positive.")
    return scale.reshape(1, 1).expand(
        (weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128
    )


def _validate_fp8_weight_and_scale(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor | None,
    block_size: tuple[int, int],
) -> None:
    if tuple(block_size) != (128, 128):
        raise ValueError(
            f"FP8 backend supports block_size=(128, 128), got {block_size}."
        )
    if weight.ndim != 2:
        raise RuntimeError(
            f"FP8 Linear weight must be rank-2, got shape={tuple(weight.shape)}."
        )
    if weight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"FP8 Linear weight must be torch.float8_e4m3fn, got {weight.dtype}."
        )
    if weight_scale_inv is None:
        raise RuntimeError("FP8 Linear requires weight_scale_inv.")
    if weight_scale_inv.dtype != torch.float32:
        raise RuntimeError(
            f"weight_scale_inv must be FP32, got dtype={weight_scale_inv.dtype}."
        )
    if weight_scale_inv.dim() != 2:
        raise RuntimeError(
            "weight_scale_inv must be rank-2, "
            f"got shape={tuple(weight_scale_inv.shape)}."
        )
    if weight_scale_inv.device != weight.device:
        raise RuntimeError(
            "FP8 weight and weight_scale_inv must be on the same device, got "
            f"weight={weight.device}, scale={weight_scale_inv.device}."
        )
    expected = (
        (int(weight.shape[0]) + block_size[0] - 1) // block_size[0],
        (int(weight.shape[1]) + block_size[1] - 1) // block_size[1],
    )
    if tuple(weight_scale_inv.shape) != expected:
        raise RuntimeError(
            "weight_scale_inv shape mismatch: "
            f"expected={expected}, got={tuple(weight_scale_inv.shape)} "
            f"for weight={tuple(weight.shape)}."
        )


def fp8_blockwise_dequantize(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Explicit block-wise FP8 dequantization for correctness oracles."""

    block_size = tuple(int(value) for value in block_size)
    _validate_fp8_weight_and_scale(weight, weight_scale_inv, block_size)
    if output_dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise TypeError(
            "FP8 reference dequantization output must be FP32, BF16, or FP16, "
            f"got {output_dtype}."
        )
    block_rows, block_cols = block_size
    scales = weight_scale_inv.repeat_interleave(block_rows, dim=0)
    scales = scales.repeat_interleave(block_cols, dim=1)
    scales = scales[: weight.shape[0], : weight.shape[1]]
    return weight.to(output_dtype) * scales.to(output_dtype)


def fp8_blockwise_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Explicit dynamic W8A8 Linear used only by the reference backend."""

    if x.device != weight.device:
        raise RuntimeError(
            f"FP8 reference input and weight must share a device, got {x.device} and "
            f"{weight.device}."
        )
    if x.shape[-1] != weight.shape[-1]:
        raise RuntimeError(
            f"FP8 Linear input feature mismatch: input={tuple(x.shape)} "
            f"weight={tuple(weight.shape)}."
        )
    output_dtype = (
        x.dtype
        if x.dtype in {torch.float32, torch.bfloat16, torch.float16}
        else torch.bfloat16
    )
    block_size = tuple(int(value) for value in block_size)
    _validate_fp8_weight_and_scale(weight, weight_scale_inv, block_size)
    block_rows, block_cols = block_size
    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.zeros(
        x_2d.shape[0],
        weight.shape[0],
        device=x.device,
        dtype=torch.float32,
    )
    weight_scales = weight_scale_inv.repeat_interleave(block_rows, dim=0)
    weight_scales = weight_scales[: weight.shape[0]]
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    for block_index, column_start in enumerate(
        range(0, weight.shape[1], block_cols)
    ):
        column_end = min(column_start + block_cols, weight.shape[1])
        x_block = x_2d[:, column_start:column_end].float()
        x_scale = (x_block.abs().amax(dim=-1) / fp8_max).clamp_min(1.0e-12)
        x_quantized = (x_block / x_scale[:, None]).to(torch.float8_e4m3fn)
        block_product = F.linear(
            x_quantized.float(),
            weight[:, column_start:column_end].float(),
        )
        block_product.mul_(x_scale[:, None])
        block_product.mul_(weight_scales[:, block_index][None, :])
        output.add_(block_product)

    output = output.to(output_dtype)
    if bias is not None:
        output.add_(bias)
    return output.reshape(*original_shape, weight.shape[0])
