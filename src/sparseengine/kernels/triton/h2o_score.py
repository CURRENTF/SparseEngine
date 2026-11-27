from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["batch_size"])
def _h2o_headwise_softmax_accumulate_kernel(
    Logits, RowMetadata,
    stride_ll, stride_lb, stride_lh, stride_lw,
    batch_size, HEADS: tl.constexpr,
    SCALE: tl.constexpr, BLOCK: tl.constexpr, HEAD_BLOCK: tl.constexpr,
    NORMALIZE: tl.constexpr = True,
):
    layer = tl.program_id(0)
    batch = tl.program_id(1)
    row = layer * batch_size + batch
    length = tl.load(RowMetadata + row * 2 + 1).to(tl.int32)
    if length <= 0:
        return
    tokens = tl.arange(0, BLOCK)
    heads = tl.arange(0, HEAD_BLOCK)
    valid = tokens < length
    mass = tl.full((BLOCK,), 0, tl.float32)
    for start in range(tl.cdiv(HEADS, HEAD_BLOCK)):
        head = start * HEAD_BLOCK + heads
        values = tl.load(
            Logits + layer * stride_ll + batch * stride_lb
            + head[:, None] * stride_lh + tokens[None, :] * stride_lw,
            mask=(head[:, None] < HEADS) & valid[None, :],
            other=-float("inf"),
        ) * SCALE
        if NORMALIZE:
            maximum = tl.max(values, axis=1)
            maximum = tl.where(head < HEADS, maximum, 0.0)
            probability = tl.exp(values - maximum[:, None])
            denominator = tl.sum(probability, axis=1)
            probability /= tl.where(denominator > 0, denominator, 1.0)[:, None]
            # Normalize each query head over tokens BEFORE the head reduction.
            mass += tl.sum(probability, axis=0)
        else:
            mass += tl.sum(tl.where((head[:, None] < HEADS) & valid[None, :], values, 0.), axis=0)
    destination = tl.load(RowMetadata + row * 2).to(tl.pointer_type(tl.float32)) + tokens
    previous = tl.load(destination, mask=tokens < length - 1, other=0.0)
    tl.store(destination, previous + mass, mask=valid)


@torch.no_grad()
def h2o_headwise_softmax_accumulate_rows(
    raw_logits: torch.Tensor,
    row_metadata: torch.Tensor,
    *,
    softmax_scale: float,
    normalize_logits: bool = True,
) -> None:
    """Accumulate into independently owned rows using [address, length] metadata.

    The cache manager owns and retains the FP32 contiguous destination rows;
    each has capacity >= length and initialized history through length - 1.
    Rows must not alias. Zero lengths mask inactive rows before pointer access.
    """
    if raw_logits.ndim != 4 or min(raw_logits.shape) <= 0:
        raise ValueError("H2O row update requires positive [layers, batch, heads, width]")
    layers, batch, heads, capacity = raw_logits.shape
    if row_metadata.shape != (layers, batch, 2) or not row_metadata.is_contiguous():
        raise ValueError("H2O row metadata must be contiguous [layers, batch, 2]")
    if raw_logits.dtype != torch.float32 or row_metadata.dtype != torch.int64:
        raise TypeError("H2O row update requires FP32 logits and int64 metadata")
    if not raw_logits.is_cuda or row_metadata.device != raw_logits.device:
        raise TypeError("H2O row tensors must share a CUDA device")
    if not 0 < softmax_scale < float("inf"):
        raise ValueError("H2O softmax scale must be finite and positive")
    _h2o_headwise_softmax_accumulate_kernel[(layers, batch)](
        raw_logits, row_metadata,
        *raw_logits.stride(),
        batch_size=batch, HEADS=heads, SCALE=float(softmax_scale),
        BLOCK=triton.next_power_of_2(capacity), HEAD_BLOCK=4,
        NORMALIZE=normalize_logits, num_warps=8, num_stages=1,
    )


__all__ = ["h2o_headwise_softmax_accumulate_rows"]
