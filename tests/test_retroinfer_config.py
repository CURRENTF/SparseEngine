"""Reject lifecycle modes that cannot preserve a request's GPU wave index."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sparseengine.configs.sparse import _normalize_retroinfer
from sparseengine.models.layout import RuntimeLayout


def _config(**overrides):
    values = dict(
        sparse_method="retroinfer",
        attention_cache_layout="explicit_kv",
        hf_config=SimpleNamespace(dtype=torch.bfloat16),
        runtime_layout=RuntimeLayout.dense(2),
        enable_prefix_caching=False,
        enable_prefix_cache_offload=False,
        decode_graph=False,
        async_scheduling=None,
        retroinfer_sink_tokens=4,
        retroinfer_recent_tokens=64,
        retroinfer_retrieval_ratio=0.018,
        retroinfer_estimation_ratio=0.232,
        retroinfer_avg_cluster_size=16,
        retroinfer_min_index_tokens=16384,
        retroinfer_update_tokens=1024,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"decode_graph": True}, "decode_graph=False"),
        ({"enable_prefix_caching": True}, "prefix caching"),
        ({"async_scheduling": True}, "async_scheduling=False"),
        ({"attention_cache_layout": "mla_latent"}, "explicit KV"),
        ({"retroinfer_retrieval_ratio": 0.9, "retroinfer_estimation_ratio": 0.2}, "sum"),
    ],
)
def test_retroinfer_rejects_unsupported_runtime_contracts(override, error):
    with pytest.raises(ValueError, match=error):
        _normalize_retroinfer(_config(**override))


def test_retroinfer_disables_automatic_async_scheduling():
    config = _config()
    _normalize_retroinfer(config)
    assert config.async_scheduling is False


@pytest.mark.parametrize(
    ("layout", "layer_types", "error"),
    [
        (replace(RuntimeLayout.dense(2), linear_attention_layer_indices=(1,)), (), "uniform explicit KV layers"),
        (replace(RuntimeLayout.dense(2), kv_num_heads=(1, 2), kv_head_dims=(128, 128)), (), "uniform explicit KV layers"),
        (replace(RuntimeLayout.dense(2), layer_idx_to_kv_idx=(0, 0)), (), "independent per-layer KV caches"),
        (RuntimeLayout.dense(2), ("full_attention", "sliding_attention"), "full causal attention"),
    ],
)
def test_retroinfer_rejects_incompatible_attention_layouts(layout, layer_types, error):
    config = _config(runtime_layout=layout, hf_config=SimpleNamespace(dtype=torch.bfloat16, layer_types=layer_types))
    with pytest.raises(ValueError, match=error):
        _normalize_retroinfer(config)
