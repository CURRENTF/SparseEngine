from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from sparseengine.layers.attention import Attention
from sparseengine.models.attention_runtime import build_mha_full_attention_provider
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.decode_attention import DecodeAttentionOpSpec
from sparseengine.operators.full_attention import (
    FullAttentionOpSpec,
    FullAttentionProvider,
    prepare_full_attention_provider,
)
from sparseengine.operators.prefill_attention import PrefillAttentionOpSpec


def _specs(**decode_overrides) -> tuple[PrefillAttentionOpSpec, DecodeAttentionOpSpec]:
    shared = {
        "num_query_heads": 4,
        "num_kv_heads": 1,
        "head_dim": 128,
        "activation_dtype": torch.bfloat16,
        "softmax_scale": 128**-0.5,
        "causal": True,
        "page_size": 1,
        "layer_varying_page_table": False,
    }
    prefill = PrefillAttentionOpSpec(
        **shared,
        score_output=AttentionScoreKind.NONE,
    )
    decode = DecodeAttentionOpSpec(
        **shared,
        max_batch_size=16,
        cuda_graph=True,
        **decode_overrides,
    )
    return prefill, decode


def _prepared(spec, name: str):
    prepared = Mock(name=name)
    prepared.spec = spec
    prepared.name = name
    return prepared


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_query_heads", 8),
        ("num_kv_heads", 2),
        ("head_dim", 256),
        ("activation_dtype", torch.float16),
        ("softmax_scale", 0.5),
        ("causal", False),
    ],
)
def test_full_attention_spec_rejects_incompatible_phase_contract(field, value):
    prefill, decode = _specs()
    values = dict(decode.__dict__)
    values[field] = value

    with pytest.raises(ValueError, match=field):
        FullAttentionOpSpec(prefill=prefill, decode=DecodeAttentionOpSpec(**values))


def test_full_attention_spec_accepts_phase_specific_execution_page_sizes():
    prefill, decode = _specs()
    decode_values = dict(decode.__dict__)
    decode_values["page_size"] = 16

    spec = FullAttentionOpSpec(
        prefill=prefill,
        decode=DecodeAttentionOpSpec(**decode_values),
    )

    assert spec.prefill.page_size == 1
    assert spec.decode.page_size == 16


def test_full_attention_spec_accepts_phase_specific_page_table_variation():
    prefill, decode = _specs()
    decode_values = dict(decode.__dict__)
    decode_values["layer_varying_page_table"] = True

    spec = FullAttentionOpSpec(
        prefill=prefill,
        decode=DecodeAttentionOpSpec(**decode_values),
    )

    assert not spec.prefill.layer_varying_page_table
    assert spec.decode.layer_varying_page_table


def test_full_attention_provider_binds_both_phases_and_closes_once():
    prefill_spec, decode_spec = _specs()
    prefill_op = _prepared(prefill_spec, "flex_prefill")
    decode_op = _prepared(decode_spec, "flashinfer_decode")
    provider = FullAttentionProvider(
        FullAttentionOpSpec(prefill_spec, decode_spec),
        prefill_op=prefill_op,
        decode_op=decode_op,
    )
    model = nn.Sequential(
        Attention(4, 128, 128**-0.5, 1),
        Attention(4, 128, 128**-0.5, 1),
    )

    assert provider.bind(model) == 2
    assert all(layer.full_attention_provider is provider for layer in model)
    assert all(layer.prefill_op is prefill_op for layer in model)
    assert all(layer.decode_op is decode_op for layer in model)
    assert provider.binding_metadata() == {
        "implementation_kind": "composite_provider",
        "semantic_operator": "full_attention",
        "prefill_provider": "flex_prefill",
        "decode_provider": "flashinfer_decode",
    }

    provider.close()
    provider.close()
    prefill_op.close.assert_called_once_with()
    decode_op.close.assert_called_once_with()


@pytest.mark.parametrize("method", ["snapkv", "pyramidkv"])
def test_snapkv_family_prepares_only_the_selected_decode_provider(method):
    """Unused eager binding must not block score-free graph startup."""
    config = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    runtime_config = SimpleNamespace(snapkv_decode_eviction=True)
    decode_provider = Mock(name="decode_provider")
    decode_provider.name = "decode_provider"

    def make_prefill(spec, *, device_index):
        prefill = Mock(name="prefill_op")
        prefill.spec = spec
        prefill.name = "prefill_op"
        return prefill

    with (
        patch(
            "sparseengine.operators.full_attention.prepare_prefill_attention_op",
            side_effect=make_prefill,
        ),
        patch("sparseengine.operators.decode_attention.OpResolver") as resolver,
        patch("sparseengine.operators.decode_attention.platforms.current_platform") as platform,
    ):
        resolver.return_value.resolve.return_value = SimpleNamespace(
            provider=decode_provider, rejected={},
        )
        prepared = build_mha_full_attention_provider(
            config,
            sparse_method=method,
            attention_tp_size=1,
            device=torch.device("cpu"),
            max_batch_size=8,
            cuda_graph=True,
            runtime_config=runtime_config,
        )
        spec = prepared.spec.decode
        assert spec.cuda_graph and not spec.may_require_attention_scores
        resolver.return_value.resolve.assert_called_once_with(
            spec, platform.get_device_caps.return_value,
        )
        decode_provider.prepare.assert_called_once_with(spec, device_index=0)
        assert prepared.decode_op.provider is decode_provider

    prepared.close()
    decode_provider.close.assert_called_once_with()


def test_full_attention_provider_binding_is_atomic_across_layers():
    prefill_spec, decode_spec = _specs()
    provider = FullAttentionProvider(
        FullAttentionOpSpec(prefill_spec, decode_spec),
        prefill_op=_prepared(prefill_spec, "prefill"),
        decode_op=_prepared(decode_spec, "decode"),
    )
    valid = Attention(4, 128, 128**-0.5, 1)
    incompatible = Attention(8, 128, 128**-0.5, 1)
    model = nn.Sequential(valid, incompatible)

    with pytest.raises(ValueError, match="does not match model layer contract"):
        provider.bind(model)

    assert valid.full_attention_provider is None
    assert valid.prefill_op is None
    assert valid.decode_op is None


def test_full_attention_provider_forbids_rebinding():
    prefill_spec, decode_spec = _specs()
    provider = FullAttentionProvider(
        FullAttentionOpSpec(prefill_spec, decode_spec),
        prefill_op=_prepared(prefill_spec, "prefill"),
        decode_op=_prepared(decode_spec, "decode"),
    )
    model = nn.Sequential(Attention(4, 128, 128**-0.5, 1))
    provider.bind(model)

    with pytest.raises(RuntimeError, match="rebind"):
        provider.bind(model)


def test_full_attention_provider_refuses_partial_phase_binding():
    prefill_spec, decode_spec = _specs()
    provider = FullAttentionProvider(
        FullAttentionOpSpec(prefill_spec, decode_spec),
        prefill_op=_prepared(prefill_spec, "prefill"),
        decode_op=_prepared(decode_spec, "decode"),
    )
    model = nn.Sequential(
        Attention(
            4,
            128,
            128**-0.5,
            1,
            prefill_op=Mock(name="legacy_prefill"),
        )
    )

    with pytest.raises(RuntimeError, match="partial binding"):
        provider.bind(model)


def test_full_attention_prepare_cleans_prefill_when_decode_prepare_fails():
    prefill_spec, decode_spec = _specs()
    spec = FullAttentionOpSpec(prefill_spec, decode_spec)
    prefill_op = _prepared(prefill_spec, "prefill")

    with (
        patch(
            "sparseengine.operators.full_attention.prepare_prefill_attention_op",
            return_value=prefill_op,
        ),
        patch(
            "sparseengine.operators.full_attention.prepare_decode_attention_op",
            side_effect=RuntimeError("decode prepare failed"),
        ),
        pytest.raises(RuntimeError, match="decode prepare failed"),
    ):
        prepare_full_attention_provider(spec, device_index=0)

    prefill_op.close.assert_called_once_with()


def test_full_attention_prepare_cleans_both_phases_when_composition_fails():
    prefill_spec, decode_spec = _specs()
    spec = FullAttentionOpSpec(prefill_spec, decode_spec)
    prefill_op = _prepared(prefill_spec, "prefill")
    incompatible_decode = _prepared(
        DecodeAttentionOpSpec(
            **{
                **decode_spec.__dict__,
                "max_batch_size": decode_spec.max_batch_size + 1,
            }
        ),
        "decode",
    )

    with (
        patch(
            "sparseengine.operators.full_attention.prepare_prefill_attention_op",
            return_value=prefill_op,
        ),
        patch(
            "sparseengine.operators.full_attention.prepare_decode_attention_op",
            return_value=incompatible_decode,
        ),
        pytest.raises(ValueError, match="Prepared decode operator"),
    ):
        prepare_full_attention_provider(spec, device_index=0)

    prefill_op.close.assert_called_once_with()
    incompatible_decode.close.assert_called_once_with()
