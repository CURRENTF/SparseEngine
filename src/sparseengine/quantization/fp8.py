from __future__ import annotations

import torch


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
