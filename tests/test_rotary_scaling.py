from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, ROPE_INIT_FUNCTIONS

from sparseengine.layers.rotary_embedding import (
    RotaryEmbedding,
    apply_rotary_emb,
    reverse_rotary_emb,
)
from sparseengine.models.rope import (
    resolve_rope_max_position,
    resolve_rope_scaling,
    resolve_rope_theta,
)


@pytest.mark.parametrize(
    "rope_parameters",
    [
        {"rope_type": "linear", "factor": 4.0},
        {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 16,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 0.75,
            "mscale_all_dim": 0.5,
        },
        {
            "rope_type": "llama3",
            "factor": 4.0,
            "original_max_position_embeddings": 16,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
    ],
)
def test_static_rope_cache_matches_transformers_oracle(rope_parameters) -> None:
    declared_max_position = 64 if rope_parameters["rope_type"] == "llama3" else 16
    config = LlamaConfig(
        hidden_size=64,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=declared_max_position,
        rope_parameters={"rope_theta": 10_000.0, **rope_parameters},
    )
    scaling = resolve_rope_scaling(config, model_name="test")
    max_position = resolve_rope_max_position(config, model_name="test")
    actual = RotaryEmbedding(
        64,
        rotary_dim=64,
        max_position_embeddings=max_position,
        base=resolve_rope_theta(config),
        rope_scaling=scaling,
        backend="torch",
    )

    rope_type = config.rope_parameters["rope_type"]
    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, "cpu")
    positions = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    expected = torch.cat(
        (freqs.cos() * attention_scaling, freqs.sin() * attention_scaling),
        dim=-1,
    ).unsqueeze(1)

    torch.testing.assert_close(actual.cos_sin_cache, expected)


@pytest.mark.parametrize("rope_type", ["dynamic", "longrope"])
def test_sequence_dependent_rope_is_rejected_before_cache_construction(
    rope_type: str,
) -> None:
    config = SimpleNamespace(
        max_position_embeddings=32,
        rope_parameters={"rope_type": rope_type, "factor": 2.0},
    )

    with pytest.raises(NotImplementedError, match="sequence-length dependent"):
        resolve_rope_scaling(config, model_name="test")


def test_yarn_rejects_nonintegral_scaled_context_length() -> None:
    config = SimpleNamespace(
        max_position_embeddings=3,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 1.1,
            "original_max_position_embeddings": 3,
        },
    )

    with pytest.raises(ValueError, match="context length must be an integer"):
        resolve_rope_max_position(config, model_name="test")


def test_reverse_rotary_emb_handles_yarn_cache_amplitude() -> None:
    rotary = RotaryEmbedding(
        64,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10_000.0,
        rope_scaling=tuple(
            sorted(
                {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 8,
                }.items()
            )
        ),
        backend="torch",
    )
    positions = torch.tensor([0, 1, 7, 31])
    cos, sin = rotary.cos_sin_cache[positions].chunk(2, dim=-1)
    source = torch.randn(4, 3, 64)

    rotated = apply_rotary_emb(source, cos, sin)
    restored = reverse_rotary_emb(rotated, cos, sin)

    torch.testing.assert_close(restored, source)
