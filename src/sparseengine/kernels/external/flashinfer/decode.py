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
    allow_var_keyword: bool = False,
) -> None:
    try:
        signature = inspect.signature(callable_)
        actual = frozenset(signature.parameters)
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"failed to inspect {entrypoint}: {type(error).__name__}: {error}",
        ) from error
    accepts_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    missing = sorted(required - actual)
    if allow_var_keyword and accepts_var_keyword:
        missing = []
    if missing:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"{entrypoint} is missing required parameters {missing}",
        )


@lru_cache(maxsize=1)
def _paged_decode_wrapper_type():
    feature = "paged decode with softmax LSE"
    _, reason = flashinfer_kernel_support(feature)
    try:
        wrapper_type = getattr(
            importlib.import_module("flashinfer.decode"),
            "BatchDecodeWithPagedKVCacheWrapper",
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
            "BatchDecodeWithPagedKVCacheWrapper is not callable",
        )
    _require_parameters(
        wrapper_type,
        frozenset(
            {
                "float_workspace_buffer",
                "kv_layout",
                "use_cuda_graph",
                "paged_kv_indptr_buffer",
                "paged_kv_indices_buffer",
                "paged_kv_last_page_len_buffer",
                "backend",
            }
        ),
        feature=feature,
        entrypoint="BatchDecodeWithPagedKVCacheWrapper",
    )
    _require_parameters(
        wrapper_type.plan,
        frozenset(
            {
                "indptr",
                "indices",
                "last_page_len",
                "num_qo_heads",
                "num_kv_heads",
                "head_dim",
                "page_size",
            }
        ),
        feature=feature,
        entrypoint="BatchDecodeWithPagedKVCacheWrapper.plan",
    )
    _require_parameters(
        wrapper_type.plan,
        frozenset(
            {"sm_scale", "q_data_type", "kv_data_type", "non_blocking"}
        ),
        feature=feature,
        entrypoint="BatchDecodeWithPagedKVCacheWrapper.plan",
        allow_var_keyword=True,
    )
    _require_parameters(
        wrapper_type.run,
        frozenset({"q", "paged_kv_cache", "out", "return_lse"}),
        feature=feature,
        entrypoint="BatchDecodeWithPagedKVCacheWrapper.run",
    )
    return wrapper_type, reason


def flashinfer_paged_decode_support() -> tuple[bool, str]:
    _, reason = _paged_decode_wrapper_type()
    return True, reason


def make_flashinfer_paged_decode_wrapper(
    workspace: torch.Tensor,
    *,
    use_cuda_graph: bool = False,
    paged_kv_indptr_buffer: torch.Tensor | None = None,
    paged_kv_indices_buffer: torch.Tensor | None = None,
    paged_kv_last_page_len_buffer: torch.Tensor | None = None,
):
    wrapper_type, _ = _paged_decode_wrapper_type()
    return wrapper_type(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=use_cuda_graph,
        paged_kv_indptr_buffer=paged_kv_indptr_buffer,
        paged_kv_indices_buffer=paged_kv_indices_buffer,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
        backend="auto",
    )


__all__ = [
    "flashinfer_paged_decode_support",
    "make_flashinfer_paged_decode_wrapper",
]
