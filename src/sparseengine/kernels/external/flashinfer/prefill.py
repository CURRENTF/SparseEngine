from __future__ import annotations

import importlib
import inspect
from functools import lru_cache

import torch

from sparseengine.kernels.external.flashinfer.support import (
    flashinfer_kernel_support,
)
from sparseengine.kernels.external.support import ExternalKernelContractError


def _require_parameters(
    callable_: object,
    required: frozenset[str],
    *,
    feature: str,
    entrypoint: str,
) -> None:
    try:
        actual = frozenset(inspect.signature(callable_).parameters)
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"failed to inspect {entrypoint}: {type(error).__name__}: {error}",
        ) from error
    missing = sorted(required - actual)
    if missing:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"{entrypoint} is missing required parameters {missing}",
        )


@lru_cache(maxsize=2)
def _paged_prefill_wrapper_type(backend: str):
    if backend not in {"fa2", "fa3"}:
        raise ValueError(f"Unsupported FlashInfer prefill backend {backend!r}.")
    feature = f"{backend.upper()} paged prefill"
    _, reason = flashinfer_kernel_support(feature)
    try:
        wrapper_type = getattr(
            importlib.import_module("flashinfer.prefill"),
            "BatchPrefillWithPagedKVCacheWrapper",
        )
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"failed to load wrapper: {type(error).__name__}: {error}",
        ) from error
    if not callable(wrapper_type):
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            "BatchPrefillWithPagedKVCacheWrapper is not callable",
        )
    _require_parameters(
        wrapper_type,
        frozenset({"float_workspace_buffer", "kv_layout", "backend"}),
        feature=feature,
        entrypoint="BatchPrefillWithPagedKVCacheWrapper",
    )
    _require_parameters(
        wrapper_type.plan,
        frozenset(
            {
                "qo_indptr",
                "paged_kv_indptr",
                "paged_kv_indices",
                "paged_kv_last_page_len",
                "num_qo_heads",
                "num_kv_heads",
                "head_dim_qk",
                "page_size",
                "causal",
                "sm_scale",
                "q_data_type",
                "kv_data_type",
                "non_blocking",
            }
        ),
        feature=feature,
        entrypoint="BatchPrefillWithPagedKVCacheWrapper.plan",
    )
    if backend == "fa2":
        _require_parameters(
            getattr(wrapper_type, "workspace_size", None),
            frozenset(
                {
                    "qo_indptr",
                    "paged_kv_indptr",
                    "paged_kv_indices",
                    "paged_kv_last_page_len",
                    "num_qo_heads",
                    "num_kv_heads",
                    "head_dim_qk",
                    "page_size",
                    "causal",
                    "sm_scale",
                    "q_data_type",
                    "kv_data_type",
                }
            ),
            feature=feature,
            entrypoint="BatchPrefillWithPagedKVCacheWrapper.workspace_size",
        )
        _require_parameters(
            getattr(wrapper_type, "reset_workspace_buffer", None),
            frozenset({"float_workspace_buffer", "int_workspace_buffer"}),
            feature=feature,
            entrypoint="BatchPrefillWithPagedKVCacheWrapper.reset_workspace_buffer",
        )
    _require_parameters(
        wrapper_type.run,
        frozenset({"q", "paged_kv_cache", "out"}),
        feature=feature,
        entrypoint="BatchPrefillWithPagedKVCacheWrapper.run",
    )
    return wrapper_type, reason


def flashinfer_paged_prefill_support(backend: str) -> tuple[bool, str]:
    _, reason = _paged_prefill_wrapper_type(backend)
    return True, reason


def make_flashinfer_paged_prefill_wrapper(
    workspace: torch.Tensor,
    *,
    backend: str,
):
    wrapper_type, _ = _paged_prefill_wrapper_type(backend)
    return wrapper_type(
        workspace,
        kv_layout="NHD",
        backend=backend,
    )


__all__ = [
    "flashinfer_paged_prefill_support",
    "make_flashinfer_paged_prefill_wrapper",
]
