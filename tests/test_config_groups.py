from types import SimpleNamespace

import torch
import pytest

from sparseengine.config import Config as PublicConfig
from sparseengine.configs import Config
from sparseengine.configs.model import _normalize_hf_config_dtype
from sparseengine.configs.groups import SparseMethodConfig
from sparseengine.configs.sparse import _normalize_h2o, _normalize_sparse_prefill_score


def test_public_config_import_remains_compatible():
    assert PublicConfig is Config


def test_model_dtype_normalization_uses_current_transformers_field():
    class ModernConfig:
        dtype = torch.bfloat16

        @property
        def torch_dtype(self):
            raise AssertionError("deprecated dtype field was accessed")

    config = ModernConfig()

    assert _normalize_hf_config_dtype(config) is torch.bfloat16
    assert config.dtype is torch.bfloat16


def test_model_dtype_normalization_accepts_legacy_field():
    config = SimpleNamespace(torch_dtype=torch.float16)

    assert _normalize_hf_config_dtype(config) is torch.float16
    assert config.dtype is torch.float16


@pytest.mark.parametrize("mode", [None, "logits", "probability"])
@pytest.mark.parametrize("window", [0, 64, 128])
def test_h2o_online_eviction_normalizes_score_units_without_changing_window(
    mode, window
):
    config = SparseMethodConfig(
        sparse_method="h2o",
        h2o_decode_eviction=True,
        sparse_prefill_score_mode=mode,
        h2o_prefill_score_window=window,
    )
    config.prefill_sparse_method = "h2o_prefill"
    _normalize_sparse_prefill_score(config)
    _normalize_h2o(config)
    assert config.sparse_prefill_score_mode == "probability"
    assert config.h2o_prefill_score_window == window


@pytest.mark.parametrize(
    ("method", "layout", "error", "message"),
    [("", "explicit_kv", ValueError, "requires sparse_method")],
)
def test_h2o_online_eviction_rejects_incompatible_execution_contract(
    method, layout, error, message
):
    config = SparseMethodConfig(sparse_method=method, h2o_decode_eviction=True)
    config.prefill_sparse_method = "h2o_prefill"
    config.attention_cache_layout = layout
    with pytest.raises(error, match=message):
        _normalize_sparse_prefill_score(config)


def test_h2o_online_eviction_still_validates_probability_window():
    config = SparseMethodConfig(
        sparse_method="h2o",
        h2o_decode_eviction=True,
        sparse_prefill_score_mode="logits",
        h2o_prefill_score_window=256,
    )
    config.prefill_sparse_method = "h2o_prefill"
    _normalize_sparse_prefill_score(config)
    with pytest.raises(ValueError, match="probability mode"):
        _normalize_h2o(config)


def test_mla_h2o_approximation_warns_once_without_changing_prefill_window(monkeypatch):
    from sparseengine.utils import log

    monkeypatch.setattr(log, "_seen_messages", set())
    messages = []
    sink = log.logger.add(lambda message: messages.append(message.record))
    try:
        for layout, enabled in [
            ("explicit_kv", True),
            ("mla_latent", False),
            ("mla_latent", True),
            ("mla_latent", True),
        ]:
            config = SparseMethodConfig(
                sparse_method="h2o",
                h2o_decode_eviction=enabled,
                sparse_prefill_score_mode="probability",
                h2o_prefill_score_window=64,
            )
            config.prefill_sparse_method = "h2o_prefill"
            config.attention_cache_layout = layout
            _normalize_sparse_prefill_score(config)
            _normalize_h2o(config)
            assert config.h2o_prefill_score_window == 64
            approximation = [
                m for m in messages if "TODO(h2o-mla-parity)" in m["message"]
            ]
            assert len(approximation) == int(layout == "mla_latent" and enabled)
        assert approximation[0]["level"].name == "WARNING"
    finally:
        log.logger.remove(sink)


def test_omnikv_offload_rejects_other_methods():
    from sparseengine.configs.sparse import normalize_sparse_method_name

    config = SparseMethodConfig(sparse_method="h2o", enable_omnikv_offload=True)
    with pytest.raises(ValueError, match="enable_omnikv_offload requires"):
        normalize_sparse_method_name(config)


@pytest.mark.parametrize("tokens", [-1, True, 1.5])
def test_omnikv_cache_rejects_invalid_capacity(tokens):
    from sparseengine.configs.sparse import normalize_sparse_method_name

    config = SparseMethodConfig(
        sparse_method="omnikv",
        enable_omnikv_offload=True,
        omnikv_offload_cache_tokens=tokens,
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        normalize_sparse_method_name(config)


def test_omnikv_cache_requires_host_backing():
    from sparseengine.configs.sparse import normalize_sparse_method_name

    config = SparseMethodConfig(
        sparse_method="omnikv", omnikv_offload_cache_tokens=1024
    )
    with pytest.raises(ValueError, match="requires enable_omnikv_offload"):
        normalize_sparse_method_name(config)
