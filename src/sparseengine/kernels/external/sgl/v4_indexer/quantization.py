# SPDX-License-Identifier: Apache-2.0
# SGLang pure-Torch FP4 reference; pinned provenance in NOTICE.
import torch

FP4_MAX = 6.0
FP4_BLOCK_SIZE = 32
FP4_AMAX_FLOOR = 6 * 2.0**-126


def ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for positive fp32 x, computed on the IEEE bits so the
    result is exact at powers of two."""
    bits = x.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF) != 0
    exponent = exponent + has_mantissa.to(torch.int32)
    return ((exponent + 127) << 23).view(torch.float32)


def block_scale(x: torch.Tensor, block_size: int, fmax: float, amax_floor: float):
    """Per-block ue8m0 scale, as fp32 powers of two, shape [..., N // block_size]."""
    amax = x.float().unflatten(-1, (-1, block_size)).abs().amax(dim=-1)
    amax = amax.clamp_min(amax_floor)
    # The kernel multiplies by the fp32 reciprocal rather than dividing. A Python
    # scalar keeps this free of host tensors, so it can run under CUDA graph capture.
    return ceil_pow2(amax * (1.0 / fmax))


def round_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values in [-6, 6] onto the e2m1 grid with round-to-nearest-even."""
    magnitude = x.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    return torch.round(magnitude / step) * step * torch.sign(x)


def fake_quant_fp4(x: torch.Tensor, block_size: int = FP4_BLOCK_SIZE) -> torch.Tensor:
    """Quantize to fp4 (per-block ue8m0 scale) and back, in x's dtype."""
    scale = block_scale(x, block_size, FP4_MAX, FP4_AMAX_FLOOR)
    scaled = x.float().unflatten(-1, (-1, block_size)) / scale.unsqueeze(-1)
    deq = round_fp4(scaled.clamp(-FP4_MAX, FP4_MAX)) * scale.unsqueeze(-1)
    return deq.flatten(-2).to(x.dtype)
