from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sparseengine.kernels.external.flashinfer.prefill import (
    _paged_prefill_wrapper_type,
    flashinfer_paged_prefill_support,
)
from sparseengine.kernels.external.support import ExternalKernelContractError


class _PublicPagedPrefillWrapper:
    def __init__(
        self,
        float_workspace_buffer,
        kv_layout="NHD",
        backend="auto",
    ):
        pass

    def plan(
        self,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        page_size,
        causal=False,
        sm_scale=None,
        q_data_type=None,
        kv_data_type=None,
        non_blocking=True,
    ):
        pass

    def workspace_size(
        self,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        page_size,
        causal=False,
        sm_scale=None,
        q_data_type=None,
        kv_data_type=None,
    ):
        return 0, 0

    def reset_workspace_buffer(
        self,
        float_workspace_buffer,
        int_workspace_buffer,
    ):
        pass

    def run(self, q, paged_kv_cache, *, out=None):
        pass


def test_flashinfer_paged_prefill_accepts_public_wrapper_contract():
    module = SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=_PublicPagedPrefillWrapper
    )
    _paged_prefill_wrapper_type.cache_clear()
    try:
        with (
            patch(
                "sparseengine.kernels.external.flashinfer.prefill.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch("importlib.import_module", return_value=module),
        ):
            assert flashinfer_paged_prefill_support("fa2") == (True, "available")
    finally:
        _paged_prefill_wrapper_type.cache_clear()


def test_flashinfer_paged_prefill_rejects_wrapper_without_backend_contract():
    class WrapperWithoutBackend(_PublicPagedPrefillWrapper):
        def __init__(self, float_workspace_buffer, kv_layout="NHD"):
            pass

    module = SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=WrapperWithoutBackend
    )
    _paged_prefill_wrapper_type.cache_clear()
    try:
        with (
            patch(
                "sparseengine.kernels.external.flashinfer.prefill.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch("importlib.import_module", return_value=module),
            pytest.raises(ExternalKernelContractError, match="backend"),
        ):
            flashinfer_paged_prefill_support("fa3")
    finally:
        _paged_prefill_wrapper_type.cache_clear()


def test_flashinfer_paged_prefill_rejects_wrapper_without_workspace_sizing():
    class WrapperWithoutWorkspaceSizing(_PublicPagedPrefillWrapper):
        workspace_size = None

    module = SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=WrapperWithoutWorkspaceSizing
    )
    _paged_prefill_wrapper_type.cache_clear()
    try:
        with (
            patch(
                "sparseengine.kernels.external.flashinfer.prefill.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch("importlib.import_module", return_value=module),
            pytest.raises(ExternalKernelContractError, match="workspace_size"),
        ):
            flashinfer_paged_prefill_support("fa2")
    finally:
        _paged_prefill_wrapper_type.cache_clear()


def test_flashinfer_fa3_accepts_wrapper_without_workspace_sizing():
    class WrapperWithoutWorkspaceSizing(_PublicPagedPrefillWrapper):
        workspace_size = None
        reset_workspace_buffer = None

    module = SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=WrapperWithoutWorkspaceSizing
    )
    _paged_prefill_wrapper_type.cache_clear()
    try:
        with (
            patch(
                "sparseengine.kernels.external.flashinfer.prefill.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch("importlib.import_module", return_value=module),
        ):
            assert flashinfer_paged_prefill_support("fa3") == (True, "available")
    finally:
        _paged_prefill_wrapper_type.cache_clear()
