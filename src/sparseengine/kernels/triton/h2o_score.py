from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _h2o_softmax_accumulate_kernel(
    logits,
    cumulative,
    stride_ll,
    stride_lb,
    stride_lw,
    stride_cl,
    stride_cb,
    stride_cw,
    batch_size,
    width,
    previous_width,
    softmax_scale: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    layer = row // batch_size
    batch = row - layer * batch_size
    offsets = tl.arange(0, BLOCK)
    valid = offsets < width
    values = tl.load(
        logits
        + layer * stride_ll
        + batch * stride_lb
        + offsets * stride_lw,
        mask=valid,
        other=-float("inf"),
    )
    values = values * softmax_scale
    values = values - tl.max(values, axis=0)
    probabilities = tl.exp(values)
    probabilities = probabilities / tl.sum(probabilities, axis=0)
    previous = tl.load(
        cumulative
        + layer * stride_cl
        + batch * stride_cb
        + offsets * stride_cw,
        mask=offsets < previous_width,
        other=0.0,
    )
    tl.store(
        cumulative
        + layer * stride_cl
        + batch * stride_cb
        + offsets * stride_cw,
        previous + probabilities,
        mask=valid,
    )


@torch.no_grad()
def h2o_softmax_accumulate(
    raw_logits: torch.Tensor,
    cumulative_scores: torch.Tensor,
    *,
    width: int,
    previous_width: int,
    softmax_scale: float,
) -> None:
    """Normalize and accumulate every H2O layer/batch row in one launch."""

    if raw_logits.ndim != 3 or cumulative_scores.ndim != 3:
        raise ValueError(
            "H2O fused score update requires [layers, batch, width] tensors, "
            f"got {tuple(raw_logits.shape)} and {tuple(cumulative_scores.shape)}."
        )
    if tuple(raw_logits.shape[:2]) != tuple(cumulative_scores.shape[:2]):
        raise ValueError(
            "H2O fused score tensors disagree on layers or batch: "
            f"logits={tuple(raw_logits.shape)} cumulative={tuple(cumulative_scores.shape)}."
        )
    if raw_logits.dtype != torch.float32 or cumulative_scores.dtype != torch.float32:
        raise TypeError(
            "H2O fused score update requires FP32 logits and cumulative scores, "
            f"got {raw_logits.dtype} and {cumulative_scores.dtype}."
        )
    if not raw_logits.is_cuda or not cumulative_scores.is_cuda:
        raise TypeError("H2O fused score update requires CUDA tensors.")
    if raw_logits.device != cumulative_scores.device:
        raise ValueError(
            "H2O fused score tensors must share a device, got "
            f"{raw_logits.device} and {cumulative_scores.device}."
        )
    width = int(width)
    previous_width = int(previous_width)
    if not 0 <= previous_width <= width <= int(raw_logits.shape[2]):
        raise ValueError(
            "H2O fused score widths are invalid: "
            f"previous={previous_width} width={width} logits={int(raw_logits.shape[2])}."
        )
    if width > int(cumulative_scores.shape[2]):
        raise ValueError(
            "H2O cumulative score capacity is too small: "
            f"width={width} capacity={int(cumulative_scores.shape[2])}."
        )
    if width == 0:
        return

    rows = int(raw_logits.shape[0]) * int(raw_logits.shape[1])
    block = triton.next_power_of_2(int(raw_logits.shape[2]))
    _h2o_softmax_accumulate_kernel[(rows,)](
        raw_logits,
        cumulative_scores,
        *raw_logits.stride(),
        *cumulative_scores.stride(),
        int(raw_logits.shape[1]),
        width,
        previous_width,
        softmax_scale=float(softmax_scale),
        BLOCK=block,
        num_warps=8,
        num_stages=1,
    )


@triton.jit(do_not_specialize=["batch_size"])
def _h2o_headwise_softmax_accumulate_kernel(
    Logits, Cumulative, Lengths,
    stride_ll, stride_lb, stride_lh, stride_lw,
    stride_cl, stride_cb, stride_cw,
    batch_size, HEADS: tl.constexpr,
    SCALE: tl.constexpr, BLOCK: tl.constexpr, HEAD_BLOCK: tl.constexpr,
    INDIRECT: tl.constexpr = False,
    NORMALIZE: tl.constexpr = True,
):
    layer = tl.program_id(0)
    batch = tl.program_id(1)
    row = layer * batch_size + batch
    if INDIRECT:
        length = tl.load(Lengths + row * 2 + 1).to(tl.int32)
    else:
        length = tl.load(Lengths + row)
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
    if INDIRECT:
        destination = tl.load(Lengths + row * 2).to(tl.pointer_type(tl.float32)) + tokens
    else:
        destination = Cumulative + layer * stride_cl + batch * stride_cb + tokens * stride_cw
    previous = tl.load(destination, mask=tokens < length - 1, other=0.0)
    tl.store(destination, previous + mass, mask=valid)


@torch.no_grad()
def h2o_headwise_softmax_accumulate(
    raw_logits: torch.Tensor,
    cumulative_scores: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    softmax_scale: float,
) -> None:
    """Add sum_h softmax(scale * logits_h) to history; append one new token.

    Logits are [layers, batch, heads, capacity] in physical retained-KV order.
    Lengths are contiguous int32 [layers, batch], with zero for inactive rows.
    Invalid logits may be uninitialized; padding in cumulative is untouched.
    The caller validates device lengths against both capacities before launch.
    """
    if raw_logits.ndim != 4 or cumulative_scores.ndim != 3:
        raise ValueError("H2O headwise update expects rank-4 logits and rank-3 history")
    layers, batch, heads, capacity = map(int, raw_logits.shape)
    if min(layers, batch, heads, capacity) <= 0:
        raise ValueError("H2O headwise update requires positive tensor dimensions")
    if tuple(cumulative_scores.shape[:2]) != (layers, batch):
        raise ValueError("H2O logits and history disagree on layers/batch")
    if tuple(context_lens.shape) != (layers, batch) or not context_lens.is_contiguous():
        raise ValueError("H2O lengths must be contiguous [layers, batch]")
    if raw_logits.dtype != torch.float32 or cumulative_scores.dtype != torch.float32:
        raise TypeError("H2O logits and history must be FP32")
    if context_lens.dtype != torch.int32:
        raise TypeError("H2O lengths must be int32")
    if not raw_logits.is_cuda or any(
        tensor.device != raw_logits.device for tensor in (cumulative_scores, context_lens)
    ):
        raise TypeError("H2O tensors must share a CUDA device")
    if not 0 < softmax_scale < float("inf"):
        raise ValueError("H2O softmax scale must be finite and positive")
    _h2o_headwise_softmax_accumulate_kernel[(layers, batch)](
        raw_logits, cumulative_scores, context_lens,
        *raw_logits.stride(), *cumulative_scores.stride(),
        batch_size=batch, HEADS=heads, SCALE=float(softmax_scale),
        BLOCK=triton.next_power_of_2(capacity), HEAD_BLOCK=4,
        num_warps=8, num_stages=1,
    )


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
        raw_logits, raw_logits, row_metadata,
        *raw_logits.stride(), 0, 0, 0,
        batch_size=batch, HEADS=heads, SCALE=float(softmax_scale),
        BLOCK=triton.next_power_of_2(capacity), HEAD_BLOCK=4,
        INDIRECT=True, NORMALIZE=normalize_logits, num_warps=8, num_stages=1,
    )


__all__ = ["h2o_softmax_accumulate", "h2o_headwise_softmax_accumulate",
           "h2o_headwise_softmax_accumulate_rows"]
