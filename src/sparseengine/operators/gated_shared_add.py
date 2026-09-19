from __future__ import annotations

import torch


def gated_shared_add(
    routed: torch.Tensor,
    shared: torch.Tensor,
    gate_logits: torch.Tensor,
) -> torch.Tensor:
    if (
        routed.ndim != 2
        or routed.shape != shared.shape
        or gate_logits.shape != (routed.shape[0], 1)
    ):
        raise ValueError(
            "gated_shared_add expects routed/shared [tokens, hidden] and gate "
            f"[tokens, 1], got {tuple(routed.shape)}, {tuple(shared.shape)}, "
            f"{tuple(gate_logits.shape)}."
        )
    if (
        routed.dtype != torch.bfloat16
        or shared.dtype != routed.dtype
        or gate_logits.dtype != routed.dtype
    ):
        raise TypeError("gated_shared_add requires BF16 inputs with matching dtypes.")
    if (
        not routed.is_cuda
        or shared.device != routed.device
        or gate_logits.device != routed.device
    ):
        raise ValueError("gated_shared_add requires CUDA inputs on one device.")
    if (
        not routed.is_contiguous()
        or not shared.is_contiguous()
        or gate_logits.stride(1) != 1
    ):
        raise ValueError("gated_shared_add requires contiguous hidden dimensions.")

    from sparseengine.kernels.triton.qwen3_5.gated_shared_add import (
        triton_gated_shared_add,
    )

    return triton_gated_shared_add(routed, shared, gate_logits)
