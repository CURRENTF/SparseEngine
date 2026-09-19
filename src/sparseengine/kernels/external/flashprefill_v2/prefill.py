from __future__ import annotations

import importlib
import inspect

import torch

from sparseengine.kernels.external.flashprefill_v2.support import (
    flashprefill_v2_support,
)
from sparseengine.kernels.external.support import ExternalKernelContractError


def _require_parameters(
    callable_: object,
    required: frozenset[str],
    *,
    entrypoint: str,
) -> None:
    try:
        actual = frozenset(inspect.signature(callable_).parameters)
    except Exception as error:
        raise ExternalKernelContractError(
            "flashprefill",
            "V2 paged prefill",
            f"failed to inspect {entrypoint}: {type(error).__name__}: {error}",
        ) from error
    missing = sorted(required - actual)
    if missing:
        raise ExternalKernelContractError(
            "flashprefill",
            "V2 paged prefill",
            f"{entrypoint} is missing required parameters {missing}",
        )


def _flashprefill_type():
    flashprefill_v2_support()
    try:
        flashprefill_type = getattr(
            importlib.import_module("flashprefill"),
            "FlashPrefill",
        )
    except Exception as error:
        raise ExternalKernelContractError(
            "flashprefill",
            "V2 paged prefill",
            f"failed to load FlashPrefill: {type(error).__name__}: {error}",
        ) from error
    if not callable(flashprefill_type):
        raise ExternalKernelContractError(
            "flashprefill",
            "V2 paged prefill",
            "FlashPrefill is not callable",
        )
    _require_parameters(
        flashprefill_type,
        frozenset(
            {
                "k_block_m",
                "k_block_n",
                "abs_threshold",
                "attention_sink",
                "window_size",
                "last_n_blocks",
                "min_sparse_q_len",
                "causal",
                "softmax_scale",
                "num_splits",
                "use_mean_correction",
            }
        ),
        entrypoint="FlashPrefill",
    )
    _require_parameters(
        flashprefill_type.__call__,
        frozenset(
            {
                "q",
                "k_cache",
                "v_cache",
                "page_table",
                "cache_seqlens",
                "cu_seqlens_q",
                "q_lens",
                "max_cache_seqlen",
                "softmax_scale",
            }
        ),
        entrypoint="FlashPrefill.__call__",
    )
    return flashprefill_type


def make_flashprefill_v2(*, semantics, softmax_scale: float):
    flashprefill_type = _flashprefill_type()
    return flashprefill_type(
        k_block_m=int(semantics.k_block_m),
        k_block_n=int(semantics.k_block_n),
        abs_threshold=float(semantics.abs_threshold),
        attention_sink=int(semantics.attention_sink_blocks),
        window_size=int(semantics.window_blocks),
        last_n_blocks=int(semantics.last_query_blocks),
        min_sparse_q_len=int(semantics.min_sparse_q_len),
        causal=True,
        softmax_scale=float(softmax_scale),
        num_splits=1,
        use_mean_correction=bool(semantics.use_mean_correction),
    )


def build_flashprefill_v2_page_table(
    active_slots: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    max_context_len: int,
) -> torch.Tensor:
    if active_slots.dtype != torch.int32 or active_slots.ndim != 2:
        raise TypeError(
            "FlashPrefill V2 requires a 2D int32 physical-slot table, got "
            f"dtype={active_slots.dtype} shape={tuple(active_slots.shape)}."
        )
    batch_size = int(context_lens.numel())
    if batch_size <= 0 or int(req_indices.numel()) != batch_size:
        raise ValueError(
            "FlashPrefill V2 requires matched non-empty request indices and "
            "context lengths."
        )
    width = int(max_context_len)
    if width <= 0 or width > int(active_slots.shape[1]):
        raise ValueError(
            "FlashPrefill V2 max context is outside the physical-slot table: "
            f"max_context_len={width} width={int(active_slots.shape[1])}."
        )
    return active_slots.index_select(0, req_indices.to(torch.long))[
        :, :width
    ].contiguous()


__all__ = ["build_flashprefill_v2_page_table", "make_flashprefill_v2"]
