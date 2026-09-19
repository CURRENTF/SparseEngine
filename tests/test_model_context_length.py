from types import MethodType, SimpleNamespace

import pytest

from sparseengine.configs.model import (
    _finalize_model_config,
    _model_context_length,
    _validate_sparse_method_rope_compatibility,
)
from sparseengine.configs.runtime import Config
from sparseengine.engine.cache_manager.standard import StandardCacheManager
from sparseengine.models.rope import resolve_rope_max_position


def test_extended_linear_rope_uses_declared_length_for_admission_and_cache():
    hf_config = SimpleNamespace(
        max_position_embeddings=202752,
        rope_parameters={
            "rope_type": "linear",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    )

    assert _model_context_length(hf_config) == 202752
    assert resolve_rope_max_position(hf_config, model_name="test") == 202752


def test_model_context_length_preserves_legacy_linear_scaling():
    hf_config = SimpleNamespace(
        max_position_embeddings=32768,
        rope_parameters={"rope_type": "linear", "factor": 4.0},
    )

    assert _model_context_length(hf_config) == 131072


@pytest.mark.parametrize("declared", [32768, 131072])
def test_yarn_context_length_is_shared_by_admission_and_rope_cache(declared):
    hf_config = SimpleNamespace(
        max_position_embeddings=declared,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    )

    assert _model_context_length(hf_config) == 131072
    assert resolve_rope_max_position(hf_config, model_name="test") == 131072


def test_yarn_rejects_inconsistent_declared_and_scaled_lengths():
    hf_config = SimpleNamespace(
        max_position_embeddings=202752,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    )

    with pytest.raises(ValueError, match="yarn context lengths are inconsistent"):
        _model_context_length(hf_config)


@pytest.mark.parametrize("max_sequence_length", [65536, 131073])
def test_yarn_explicit_context_cap_cannot_exceed_effective_limit(
    max_sequence_length,
):
    hf_config = SimpleNamespace(
        max_sequence_length=max_sequence_length,
        max_position_embeddings=32768,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    )

    if max_sequence_length <= 131072:
        assert _model_context_length(hf_config) == max_sequence_length
    else:
        with pytest.raises(ValueError, match="exceeds the YaRN context length"):
            _model_context_length(hf_config)


def test_model_context_length_accepts_per_layer_rope_parameters():
    hf_config = SimpleNamespace(
        max_position_embeddings=262144,
        rope_parameters={
            "full_attention": {"rope_type": "proportional"},
            "sliding_attention": {"rope_type": "default"},
        },
    )

    assert _model_context_length(hf_config) == 262144


@pytest.mark.parametrize("rope_type", ["linear", "yarn"])
def test_deltakv_rejects_unvalidated_rope_reconstruction(rope_type):
    config = SimpleNamespace(
        sparse_method="deltakv",
        hf_config=SimpleNamespace(
            rope_parameters={"rope_type": rope_type},
        ),
    )

    with pytest.raises(NotImplementedError, match="DeltaKV reconstruction"):
        _validate_sparse_method_rope_compatibility(config)


@pytest.mark.parametrize(
    "rope_scaling",
    [
        {"rope_type": "llama3", "factor": 8.0},
        {"factor": 4.0, "original_max_position_embeddings": 32768},
    ],
)
def test_model_context_length_does_not_double_apply_rope_scaling(rope_scaling):
    hf_config = SimpleNamespace(
        max_position_embeddings=131072,
        rope_scaling=rope_scaling,
    )

    assert _model_context_length(hf_config) == 131072


def test_auto_max_model_len_uses_model_context_length(monkeypatch):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(max_position_embeddings=32768),
        max_model_len=None,
        max_num_seqs_in_batch=32,
    )
    monkeypatch.setattr(
        "sparseengine.configs.model.RuntimeLayout.from_config",
        lambda *_args, **_kwargs: object(),
    )

    _finalize_model_config(config, SimpleNamespace(mixed_attention=False))

    assert config.max_model_len == 32768
    assert config.max_model_len_auto is True


def test_auto_max_model_len_uses_effective_yarn_context_length(monkeypatch):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            max_position_embeddings=32768,
            rope_parameters={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        ),
        max_model_len=None,
        max_num_seqs_in_batch=32,
    )
    monkeypatch.setattr(
        "sparseengine.configs.model.RuntimeLayout.from_config",
        lambda *_args, **_kwargs: object(),
    )

    _finalize_model_config(config, SimpleNamespace(mixed_attention=False))

    assert config.max_model_len == 131072
    assert config.max_model_len_auto is True


def test_explicit_max_model_len_cannot_exceed_model_context(monkeypatch):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(max_position_embeddings=32768),
        max_model_len=32769,
        max_num_seqs_in_batch=32,
    )
    monkeypatch.setattr(
        "sparseengine.configs.model.RuntimeLayout.from_config",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(ValueError, match="exceeds the model context length"):
        _finalize_model_config(config, SimpleNamespace(mixed_attention=False))


class _Storage:
    def allocate(self, **kwargs):
        self.allocation = kwargs


def _standard_manager(*, max_model_len, auto):
    config = SimpleNamespace(
        max_model_len=max_model_len,
        max_model_len_auto=auto,
        decode_graph=False,
        prefix_cache_max_blocks=None,
    )
    config.limit_auto_max_model_len = MethodType(Config.limit_auto_max_model_len, config)
    manager = object.__new__(StandardCacheManager)
    manager.config = config
    manager.num_kv_layers = 2
    manager.max_buffer_rows = 2
    manager.max_model_len = max_model_len
    manager.device = "cpu"
    manager.attention_cache_storage = _Storage()
    manager._get_available_slots_info = lambda: (1000, 10)
    return manager


def test_standard_cache_limits_auto_length_to_runtime_capacity():
    manager = _standard_manager(max_model_len=100, auto=True)

    manager.allocate_kv_cache()

    assert manager.max_model_len == 31
    assert manager.config.num_kvcache_slots == 31
    assert manager.attention_cache_storage.allocation["num_slots"] == 31


def test_runtime_capacity_limits_model_length_without_graph_buckets():
    config = SimpleNamespace(
        max_model_len=262144,
        max_model_len_auto=True,
        decode_graph=True,
    )

    Config.limit_auto_max_model_len(config, 9000)

    assert config.max_model_len == 9000
    assert not hasattr(config, "decode_graph_context_sizes")


def test_standard_cache_rejects_explicit_length_above_runtime_capacity():
    manager = _standard_manager(max_model_len=100, auto=False)

    with pytest.raises(RuntimeError, match="capacity is smaller than max_model_len"):
        manager.allocate_kv_cache()
