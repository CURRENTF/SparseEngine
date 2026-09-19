from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.config import Config
from sparseengine.configs.sparse import normalize_prefill_sparse_method
from sparseengine.engine.cache_manager.base import (
    AttentionViewMeta,
    ExplicitKVPayload,
    PrefillComputeView,
)
from sparseengine.kernels.external.flashprefill_v2.prefill import (
    build_flashprefill_v2_page_table,
)
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.prefill_attention import (
    FlashPrefillV2Provider,
    FlashPrefillV2Semantics,
    PrefillAttentionOpSpec,
    _resolve_prefill_attention_provider,
)
from sparseengine.platforms import DeviceCaps, PlatformEnum


def _config(**overrides):
    values = {
        "sparse_method": "",
        "prefill_sparse_method": None,
        "flashprefill_v2_k_block_m": 128,
        "flashprefill_v2_k_block_n": 128,
        "flashprefill_v2_abs_threshold": 0.1,
        "flashprefill_v2_attention_sink_blocks": 2,
        "flashprefill_v2_window_blocks": 4,
        "flashprefill_v2_last_query_blocks": 8,
        "flashprefill_v2_min_sparse_q_len": 4096,
        "flashprefill_v2_use_mean_correction": True,
        "enable_prefix_caching": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _h100_caps():
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability=(9, 0),
        runtime_version="13.0",
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
        supports_native_fp8=True,
    )


def _runtime_config(tmp_path, **kwargs):
    hf_config = SimpleNamespace(
        model_type="qwen2",
        torch_dtype=torch.bfloat16,
        max_position_embeddings=32768,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    with patch(
        "sparseengine.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        return Config(model=str(tmp_path), **kwargs)


def _flashprefill_spec(**overrides):
    values = {
        "num_query_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 128,
        "activation_dtype": torch.bfloat16,
        "softmax_scale": 128**-0.5,
        "causal": True,
        "page_size": 1,
        "score_output": AttentionScoreKind.NONE,
        "layer_varying_page_table": True,
        "prefill_sparse_method": "flashprefill_v2",
        "flashprefill_v2": FlashPrefillV2Semantics(abs_threshold=0.1),
    }
    values.update(overrides)
    return PrefillAttentionOpSpec(**values)


@pytest.mark.parametrize("alias", ["flashprefill-v2", "flash-prefill-v2"])
def test_prefill_sparse_method_normalizes_independently_from_sparse_method(alias):
    config = _config(prefill_sparse_method=alias)

    normalize_prefill_sparse_method(config)

    assert config.prefill_sparse_method == "flashprefill_v2"


def test_flashprefill_mean_correction_parses_explicit_false_string():
    config = _config(flashprefill_v2_use_mean_correction="false")

    normalize_prefill_sparse_method(config)

    assert config.flashprefill_v2_use_mean_correction is False


def test_flashprefill_mean_correction_rejects_ambiguous_string():
    config = _config(flashprefill_v2_use_mean_correction="disabled")

    with pytest.raises(ValueError, match="must be a boolean"):
        normalize_prefill_sparse_method(config)


def test_h2o_defaults_to_its_owned_prefill_method():
    config = _config(
        sparse_method="h2o",
        prefill_sparse_method=None,
        flashprefill_v2_abs_threshold=None,
    )

    normalize_prefill_sparse_method(config)

    assert config.prefill_sparse_method == "h2o_prefill"
    assert config.resolved_cache_sparse_method == "h2o"


def test_explicit_empty_prefill_method_enables_decode_only_h2o():
    config = _config(
        sparse_method="h2o",
        prefill_sparse_method="",
        flashprefill_v2_abs_threshold=None,
    )

    normalize_prefill_sparse_method(config)

    assert config.prefill_sparse_method == ""
    assert config.resolved_cache_sparse_method == "h2o"


def test_h2o_prefill_can_run_with_vanilla_decode():
    config = _config(
        sparse_method="",
        prefill_sparse_method="h2o_prefill",
        flashprefill_v2_abs_threshold=None,
    )

    normalize_prefill_sparse_method(config)

    assert config.prefill_sparse_method == "h2o_prefill"
    assert config.resolved_cache_sparse_method == "h2o"


@pytest.mark.parametrize("sparse_method", ["", "omnikv", "quest", "snapkv", "h2o"])
def test_flashprefill_supports_declared_cache_decode_combinations(sparse_method):
    config = _config(
        sparse_method=sparse_method,
        prefill_sparse_method="flashprefill_v2",
        enable_prefix_caching=True,
    )

    normalize_prefill_sparse_method(config)

    assert config.prefill_sparse_method == "flashprefill_v2"


@pytest.mark.parametrize(
    ("sparse_method", "prefill_sparse_method", "expected_prefill", "prefix_mode"),
    [
        ("", "flashprefill_v2", "flashprefill_v2", "radix"),
        ("", "h2o_prefill", "h2o_prefill", "chain"),
        ("h2o", None, "h2o_prefill", "chain"),
        ("h2o", "", "", "chain"),
        ("h2o", "flashprefill_v2", "flashprefill_v2", "chain"),
    ],
)
def test_runtime_config_resolves_prefill_method_and_prefix_mode(
    tmp_path,
    sparse_method,
    prefill_sparse_method,
    expected_prefill,
    prefix_mode,
):
    config = _runtime_config(
        tmp_path,
        sparse_method=sparse_method,
        prefill_sparse_method=prefill_sparse_method,
        flashprefill_v2_abs_threshold=(
            0.1 if prefill_sparse_method == "flashprefill_v2" else None
        ),
        enable_prefix_caching=True,
    )

    assert config.prefill_sparse_method == expected_prefill
    assert config.resolved_prefix_cache_mode == prefix_mode


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"prefill_sparse_method": "unknown"}, "Unsupported prefill_sparse_method"),
        (
            {
                "prefill_sparse_method": "flashprefill_v2",
                "flashprefill_v2_abs_threshold": None,
            },
            "requires an explicit flashprefill_v2_abs_threshold",
        ),
        ({"flashprefill_v2_k_block_m": 63}, "multiple of 16"),
        ({"flashprefill_v2_k_block_n": 192}, "power-of-two multiple of 64"),
        ({"flashprefill_v2_abs_threshold": 1.1}, r"must be in \[0, 1\]"),
        ({"flashprefill_v2_min_sparse_q_len": -1}, "must be non-negative"),
        (
            {
                "sparse_method": "streamingllm",
                "prefill_sparse_method": "flashprefill_v2",
            },
            "is incompatible with sparse_method",
        ),
        (
            {"sparse_method": "snapkv", "prefill_sparse_method": "h2o_prefill"},
            "is incompatible with sparse_method",
        ),
    ],
)
def test_prefill_sparse_method_rejects_invalid_or_unvalidated_contracts(
    overrides,
    reason,
):
    with pytest.raises(ValueError, match=reason):
        normalize_prefill_sparse_method(_config(**overrides))


def test_flashprefill_page_table_gathers_request_rows_without_reordering_tokens():
    active_slots = torch.tensor(
        [
            [101, 102, 103, 104],
            [201, 202, 203, 204],
            [301, 302, 303, 304],
        ],
        dtype=torch.int32,
    )

    actual = build_flashprefill_v2_page_table(
        active_slots,
        torch.tensor([2, 0], dtype=torch.int32),
        torch.tensor([3, 2], dtype=torch.int32),
        max_context_len=3,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [[301, 302, 303], [101, 102, 103]],
            dtype=torch.int32,
        ),
    )
    assert actual.is_contiguous()


def test_flashprefill_semantics_exclude_dense_providers_and_resolve_explicitly():
    platform = SimpleNamespace(get_device_caps=lambda _index: _h100_caps())
    with (
        patch(
            "sparseengine.platforms.get_current_platform",
            return_value=platform,
        ),
        patch(
            "sparseengine.operators.prefill_attention.flashprefill_v2_support",
            return_value=(True, "flashprefill available"),
        ),
    ):
        provider, execution_spec = _resolve_prefill_attention_provider(
            _flashprefill_spec(),
            device_index=0,
        )

    assert isinstance(provider, FlashPrefillV2Provider)
    assert execution_spec.prefill_sparse_method == "flashprefill_v2"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"activation_dtype": torch.float16}, "activation dtype"),
        ({"head_dim": 256, "softmax_scale": 256**-0.5}, "head_dim"),
        ({"score_output": AttentionScoreKind.RAW_QK_REDUCED}, "RAW_QK_REDUCED"),
        ({"return_softmax_lse": True}, "softmax LSE"),
    ],
)
def test_flashprefill_provider_rejects_unvalidated_kernel_contracts(overrides, reason):
    with patch(
        "sparseengine.operators.prefill_attention.flashprefill_v2_support",
        return_value=(True, "flashprefill available"),
    ):
        result = FlashPrefillV2Provider.supports(
            _flashprefill_spec(**overrides),
            _h100_caps(),
        )

    assert not result.supported
    assert reason in result.reason


def test_flashprefill_provider_maps_prefix_aware_varlen_view_to_upstream_call():
    provider = FlashPrefillV2Provider()
    pipeline = Mock(side_effect=lambda q, *_args, **_kwargs: q.clone())
    provider._pipeline = pipeline
    q = torch.randn(3, 32, 128, dtype=torch.bfloat16)
    k_cache = torch.randn(400, 4, 128, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    view = PrefillComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor(
                [
                    [101, 102, 103, 104],
                    [201, 202, 203, 204],
                    [301, 302, 303, 304],
                ],
                dtype=torch.int32,
            ),
            req_indices=torch.tensor([2, 0], dtype=torch.int32),
            context_lens=torch.tensor([3, 2], dtype=torch.int32),
            attn_score=None,
            max_context_len=3,
            temp_slots=None,
        ),
        payload=ExplicitKVPayload(k_cache=k_cache, v_cache=v_cache),
    )
    scope = object()

    with patch(
        "sparseengine.utils.context.get_context",
        return_value=SimpleNamespace(attention_validation_scope=scope),
    ):
        actual = provider.run(
            _flashprefill_spec(),
            q,
            view,
            qo_indptr=torch.tensor([0, 2, 3], dtype=torch.int32),
            chunk_lens=torch.tensor([2, 1], dtype=torch.int32),
            max_context_len=3,
            layer_idx=7,
        )

    torch.testing.assert_close(actual, q)
    args = pipeline.call_args.args
    assert tuple(args[1].shape) == (400, 1, 4, 128)
    assert tuple(args[2].shape) == (400, 1, 4, 128)
    torch.testing.assert_close(
        args[3],
        torch.tensor(
            [[301, 302, 303], [101, 102, 103]],
            dtype=torch.int32,
        ),
    )
    assert pipeline.call_args.kwargs["q_lens"] == (2, 1)
    assert pipeline.call_args.kwargs["max_cache_seqlen"] == 3
    torch.testing.assert_close(
        args[4],
        torch.tensor([3, 2], dtype=torch.int32),
    )
    assert tuple(args[4].tolist()) != pipeline.call_args.kwargs["q_lens"]
