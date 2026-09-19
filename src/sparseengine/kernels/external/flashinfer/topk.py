from __future__ import annotations

import importlib
import inspect
from functools import lru_cache

import torch

from sparseengine.kernels.external.flashinfer.support import (
    flashinfer_kernel_support,
)
from sparseengine.kernels.external.support import ExternalKernelContractError


_FEATURE = "fused top-k page-table transform"
_REQUIRED_PARAMETERS = frozenset(
    {
        "input",
        "src_page_table",
        "lengths",
        "k",
        "deterministic",
        "tie_break",
        "dsa_graph_safe",
    }
)


@lru_cache(maxsize=1)
def _top_k_page_table_transform():
    _, reason = flashinfer_kernel_support(_FEATURE)
    try:
        module = importlib.import_module("flashinfer.topk")
        callable_ = getattr(module, "top_k_page_table_transform")
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            _FEATURE,
            f"failed to load public API: {type(error).__name__}: {error}",
        ) from error
    if not callable(callable_):
        raise ExternalKernelContractError(
            "flashinfer-python",
            _FEATURE,
            "flashinfer.topk.top_k_page_table_transform is not callable",
        )
    try:
        actual = frozenset(inspect.signature(callable_).parameters)
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            _FEATURE,
            f"failed to inspect public API: {type(error).__name__}: {error}",
        ) from error
    missing = sorted(_REQUIRED_PARAMETERS - actual)
    if missing:
        raise ExternalKernelContractError(
            "flashinfer-python",
            _FEATURE,
            f"public API is missing required parameters {missing}",
        )
    return callable_, reason


def flashinfer_top_k_page_table_transform_support(
    device_index: int | None = None,
) -> tuple[bool, str]:
    _, reason = _top_k_page_table_transform()
    # FlashInfer's stable tie-break uses FilteredTopK, whose two 16K int32
    # buffers require 128 KiB per SM (CanImplementFilteredTopK in topk.cuh).
    # Check the resource contract, not a GPU model or performance profile.
    props = torch.cuda.get_device_properties(device_index)
    if props.shared_memory_per_multiprocessor < 128 * 1024:
        return False, "stable FlashInfer Top-K requires 128 KiB shared memory per SM"
    return True, reason


def flashinfer_top_k_page_table_transform(
    scores: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    *,
    cuda_graph: bool,
) -> torch.Tensor:
    if scores.ndim != 2 or not scores.is_contiguous():
        raise ValueError("FlashInfer fused top-k requires contiguous rank-2 scores.")
    if scores.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise TypeError(
            "FlashInfer fused top-k requires FP32, FP16, or BF16 scores, got "
            f"{scores.dtype}."
        )
    if (
        page_table.shape != scores.shape
        or page_table.dtype != torch.int32
        or not page_table.is_contiguous()
    ):
        raise TypeError(
            "FlashInfer fused top-k requires a contiguous int32 page table "
            f"matching scores, got {tuple(page_table.shape)}/{page_table.dtype}."
        )
    if lengths.shape != (int(scores.shape[0]),) or lengths.dtype != torch.int32:
        raise TypeError(
            "FlashInfer fused top-k requires one int32 length per score row."
        )
    if not lengths.is_contiguous():
        raise ValueError("FlashInfer fused top-k lengths must be contiguous.")
    if (
        not scores.is_cuda
        or page_table.device != scores.device
        or lengths.device != scores.device
    ):
        raise ValueError("FlashInfer fused top-k inputs must share one CUDA device.")
    k = int(k)
    if not 0 < k <= int(scores.shape[1]):
        raise ValueError(
            f"FlashInfer fused top-k requires 0 < k <= {scores.shape[1]}, got {k}."
        )

    callable_, _ = _top_k_page_table_transform()
    output = callable_(
        scores,
        page_table,
        lengths,
        k,
        deterministic=True,
        tie_break=1,
        dsa_graph_safe=bool(cuda_graph),
    )
    if (
        output.shape != (int(scores.shape[0]), k)
        or output.dtype != torch.int32
        or output.device != scores.device
        or not output.is_contiguous()
    ):
        raise RuntimeError(
            "FlashInfer fused top-k returned an invalid page-table result: "
            f"shape={tuple(output.shape)} dtype={output.dtype}."
        )
    return output


__all__ = [
    "flashinfer_top_k_page_table_transform",
    "flashinfer_top_k_page_table_transform_support",
]


@lru_cache(maxsize=1)
def _ragged_topk_api():
    flashinfer_kernel_support('ragged top-k transform')
    module = importlib.import_module('flashinfer.topk')
    api = getattr(module, 'top_k_ragged_transform', None)
    required = {'input', 'offsets', 'lengths', 'k', 'deterministic',
                'tie_break', 'dsa_graph_safe', 'row_starts'}
    if not callable(api) or not required.issubset(inspect.signature(api).parameters):
        raise ExternalKernelContractError(
            'flashinfer-python', 'ragged top-k transform', 'public API contract is missing',
        )
    return api


def flashinfer_ragged_topk_support(device_index=None):
    from sparseengine.kernels.external.flashinfer.support import flashinfer_kernel_metadata_health
    from sparseengine.kernels.external.support import KernelFamilyState
    health = flashinfer_kernel_metadata_health()
    if health.state is KernelFamilyState.ABSENT:
        return False, health.reason
    _ragged_topk_api()
    del device_index
    return True, 'public length-aware deterministic radix top-k with scalar row-start indexing'


def flashinfer_ragged_topk(scores, lengths, k, *, sink):
    if not scores.is_contiguous():
        raise ValueError('FlashInfer ragged top-k requires contiguous scores')
    starts = torch.full_like(lengths, sink)
    output = _ragged_topk_api()(
        scores, starts, lengths, k, deterministic=True, tie_break=0,
        # dsa_graph_safe forces FilteredTopK (k <= 2048); it is not required
        # for radix graph capture. Non-null row_starts forces scalar loads,
        # and capacity fixes its CTA envelope across all replay lengths.
        dsa_graph_safe=False, row_starts=starts,
    )
    if output.shape != (scores.shape[0], k) or output.dtype != torch.int32 or output.device != scores.device:
        raise RuntimeError('FlashInfer ragged top-k returned an invalid index tensor')
    return output
