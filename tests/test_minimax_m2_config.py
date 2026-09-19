import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparseengine.config import Config


def _quantization_config(**overrides):
    values = {
        "quant_method": "fp8",
        "fmt": "float8_e4m3fn",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "gate",
            "e_score_correction_bias",
            "lm_head",
        ],
    }
    values.update(overrides)
    return values


def _official_config(**overrides):
    values = {
        "architectures": ["MiniMaxM2ForCausalLM"],
        "model_type": "minimax_m2",
        "vocab_size": 200064,
        "hidden_size": 3072,
        "intermediate_size": 1536,
        "num_hidden_layers": 62,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rotary_dim": 64,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "max_position_embeddings": 204800,
        "shared_intermediate_size": 0,
        "mtp_transformer_layers": 1,
        "num_mtp_modules": 3,
        "hidden_act": "silu",
        "qk_norm_type": "per_layer",
        "scoring_func": "sigmoid",
        "use_qk_norm": True,
        "use_routing_bias": True,
        "use_mtp": True,
        "tie_word_embeddings": False,
        "dtype": torch.bfloat16,
        "quantization_config": _quantization_config(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_config(tmp_path, hf_config=None, **kwargs):
    if hf_config is None:
        hf_config = _official_config()
    with patch(
        "sparseengine.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        return Config(model=str(tmp_path), **kwargs)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("qk_norm_type", "per_head"),
        ("scoring_func", "softmax"),
        ("use_mtp", False),
    ],
)
def test_minimax_config_rejects_unsupported_semantics(
    tmp_path,
    field_name,
    invalid_value,
):
    hf_config = _official_config(**{field_name: invalid_value})
    with pytest.raises(ValueError, match=field_name):
        _make_config(tmp_path, hf_config=hf_config)


def test_minimax_config_requires_all_fp8_exclusions(tmp_path):
    hf_config = _official_config(
        quantization_config=_quantization_config(
            modules_to_not_convert=["gate", "lm_head"],
        )
    )
    with pytest.raises(ValueError, match="e_score_correction_bias"):
        _make_config(tmp_path, hf_config=hf_config)


def test_minimax_config_supports_quantized_tiny_random(tmp_path):
    tiny_config = tmp_path / "tiny.json"
    tiny_config.write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "hidden_size": 3072,
                "intermediate_size": 1536,
                "num_attention_heads": 48,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 256,
                "max_position_embeddings": 512,
            }
        ),
        encoding="utf-8",
    )

    config = _make_config(
        tmp_path,
        tiny_random=True,
        tiny_random_config=str(tiny_config),
        max_model_len=512,
    )

    assert config.tiny_random
    assert config.quantization_config.enabled
    assert config.hf_config.hidden_size == 3072


@pytest.mark.parametrize(
    "parallel_kwargs",
    [
        {"data_parallel_size": 2},
        {"expert_parallel_size": 3},
        {"tensor_parallel_size": 4, "expert_parallel_size": 3},
    ],
)
def test_minimax_config_rejects_unvalidated_parallel_layout(
    tmp_path,
    parallel_kwargs,
):
    with pytest.raises(ValueError, match="MoE TP=1|must be divisible by MoE EP"):
        _make_config(tmp_path, **parallel_kwargs)


def test_minimax_snapkv_tp_ep_supports_chain_cache_with_decode_graph(tmp_path):
    config = _make_config(
        tmp_path,
        sparse_method="snapkv",
        tensor_parallel_size=4,
        expert_parallel_size=4,
        enable_prefix_caching=True,
        prefix_cache_mode="chain",
        decode_graph=True,
        decode_graph_capture_sampling=False,
    )

    assert config.resolved_prefix_cache_mode == "chain"
    assert config.enable_prefix_caching is True


def test_snapkv_rejects_full_attention_layers(tmp_path):
    with pytest.raises(ValueError, match="snapkv_num_full_layers.*must be 0"):
        _make_config(
            tmp_path,
            sparse_method="snapkv",
            snapkv_num_full_layers=1,
        )
