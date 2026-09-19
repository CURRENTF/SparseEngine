import json
import os
import subprocess
import sys

import pytest
import torch
from transformers import Qwen3Config

from sparseengine.debug.tiny_random import (
    apply_tiny_random_overrides,
    build_tiny_random_hf_model,
    initialize_sparse_model,
    load_tiny_random_overrides,
    resolve_tiny_random_settings,
)


def _write_overrides(tmp_path, **updates):
    values = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "max_position_embeddings": 128,
    }
    values.update(updates)
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def _qwen3_config():
    return Qwen3Config(
        num_hidden_layers=4,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=256,
        dtype=torch.bfloat16,
    )


def test_tiny_random_overrides_are_applied(tmp_path):
    path = _write_overrides(tmp_path)
    config = _qwen3_config()

    overrides = apply_tiny_random_overrides(config, str(path))

    assert overrides["num_hidden_layers"] == 2
    assert config.hidden_size == 64
    assert config.num_attention_heads == 4
    assert config.head_dim == 16
    assert len(config.layer_types) == 2


def test_tiny_random_rejects_unknown_override(tmp_path):
    path = _write_overrides(tmp_path, model_type="llama")

    with pytest.raises(ValueError, match="Unsupported tiny random config keys"):
        load_tiny_random_overrides(str(path))


def test_tiny_random_requires_config_when_enabled(monkeypatch):
    monkeypatch.delenv("SPARSEENGINE_TINY_RANDOM_CONFIG", raising=False)

    with pytest.raises(ValueError, match="requires a JSON override file"):
        resolve_tiny_random_settings(enabled=True, config_path=None, seed=0)


def test_tiny_random_hf_initialization_is_deterministic(tmp_path):
    path = _write_overrides(tmp_path)
    config = _qwen3_config()
    apply_tiny_random_overrides(config, str(path))

    first = build_tiny_random_hf_model(config, seed=17)
    second = build_tiny_random_hf_model(config, seed=17)
    third = build_tiny_random_hf_model(config, seed=18)
    first_state = first.state_dict()
    second_state = second.state_dict()
    third_state = third.state_dict()

    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert any(not torch.equal(first_state[name], third_state[name]) for name in first_state)


def test_normal_config_import_does_not_import_tiny_random_module():
    env = dict(os.environ)
    env.pop("SPARSEENGINE_TINY_RANDOM", None)
    env.pop("SPARSEENGINE_TINY_RANDOM_CONFIG", None)
    env.pop("SPARSEENGINE_TINY_RANDOM_SEED", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import sparseengine.config; "
                "assert 'sparseengine.debug.tiny_random' not in sys.modules"
            ),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_quantized_tiny_random_initializes_fp8_weights_and_scales():
    class QuantizedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.empty(128, 128, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.register_buffer(
                "weight_scale_inv",
                torch.empty(1, 1, dtype=torch.float32),
            )

    first = QuantizedModel()
    second = QuantizedModel()
    initialize_sparse_model(first, None, seed=23, quantized=True)
    initialize_sparse_model(second, None, seed=23, quantized=True)

    assert torch.equal(first.weight, second.weight)
    assert torch.equal(first.weight_scale_inv, torch.ones_like(first.weight_scale_inv))
