from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sparseengine.kernels.external.flashinfer.decode import (
    _paged_decode_wrapper_type,
    flashinfer_paged_decode_support,
)
from sparseengine.kernels.external.support import ExternalKernelContractError


class _PublicPagedDecodeWrapper:
    def __init__(
        self,
        float_workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=False,
        paged_kv_indptr_buffer=None,
        paged_kv_indices_buffer=None,
        paged_kv_last_page_len_buffer=None,
        backend="auto",
    ):
        pass

    def plan(
        self,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        sm_scale=None,
        q_data_type=None,
        kv_data_type=None,
        non_blocking=True,
    ):
        pass

    def run(self, q, paged_kv_cache, *, out=None, return_lse=False):
        pass


def test_flashinfer_paged_decode_accepts_public_lse_contract():
    module = SimpleNamespace(
        BatchDecodeWithPagedKVCacheWrapper=_PublicPagedDecodeWrapper
    )
    _paged_decode_wrapper_type.cache_clear()
    try:
        with (
            patch(
                "sparseengine.kernels.external.flashinfer.decode.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch("importlib.import_module", return_value=module),
        ):
            assert flashinfer_paged_decode_support() == (True, "available")
    finally:
        _paged_decode_wrapper_type.cache_clear()


def test_flashinfer_paged_decode_rejects_wrapper_without_lse_contract():
    class WrapperWithoutLse(_PublicPagedDecodeWrapper):
        def run(self, q, paged_kv_cache, *, out=None):
            pass

    module = SimpleNamespace(
        BatchDecodeWithPagedKVCacheWrapper=WrapperWithoutLse
    )
    _paged_decode_wrapper_type.cache_clear()
    try:
        with (
            patch(
                "sparseengine.kernels.external.flashinfer.decode.flashinfer_kernel_support",
                return_value=(True, "available"),
            ),
            patch("importlib.import_module", return_value=module),
            pytest.raises(ExternalKernelContractError, match="return_lse"),
        ):
            flashinfer_paged_decode_support()
    finally:
        _paged_decode_wrapper_type.cache_clear()
