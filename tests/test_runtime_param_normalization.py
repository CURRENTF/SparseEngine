import unittest
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from sparseengine.config import Config
from sparseengine.configs.groups import SparseMethodConfig


def test_observation_and_eviction_parameter_renames_reject_legacy_names():
    for legacy in ("snapkv_window_size", "snapkv_decode_eviction_interval"):
        with pytest.raises(TypeError, match=legacy):
            SparseMethodConfig(**{legacy: 32})


@pytest.fixture
def prefill_config(tmp_path, monkeypatch):
    from sparseengine import platforms

    # Keep estimates below the automatic ceiling so device/model sensitivity
    # is observable without freezing a tuning constant in the assertions.
    memory = {0: 1 << 30, 1: 1 << 29}
    probes = []

    def total_memory(rank):
        probes.append(rank)
        return memory[rank]

    monkeypatch.setattr(platforms.current_platform, "get_total_memory", total_memory)

    def make(**kwargs):
        hf = SimpleNamespace(
            model_type="qwen2", dtype=torch.float16, max_position_embeddings=131072,
            hidden_size=1024, intermediate_size=4096, num_hidden_layers=2,
            num_attention_heads=16, num_key_value_heads=8,
        )
        monkeypatch.setattr("sparseengine.configs.runtime.AutoConfig.from_pretrained", lambda *a, **k: hf)
        return Config(model=str(tmp_path), decode_graph=False, **kwargs)

    return make, memory, probes


def test_auto_prefill_budget_uses_smallest_participating_device(prefill_config):
    """A larger rank must not select a budget that exceeds its smaller peer."""
    make, memory, probes = prefill_config
    mixed = make(tensor_parallel_size=2)
    assert probes == [0, 1]
    memory[0] = memory[1]
    uniform = make(tensor_parallel_size=2)
    assert mixed.max_num_batched_tokens == uniform.max_num_batched_tokens
    assert mixed.engine_prefill_chunk_size == mixed.max_num_batched_tokens
    assert mixed.max_num_batched_tokens_auto and mixed.engine_prefill_chunk_size_auto


def test_auto_prefill_budget_responds_to_memory_headroom_and_tp(prefill_config):
    """The estimate must use loaded model metadata and the actual TP shard."""
    make, memory, _ = prefill_config
    baseline = make()
    memory[0] *= 2
    larger = make()
    memory[0] //= 2
    memory[1] = memory[0]
    sharded = make(tensor_parallel_size=2)
    less_headroom = make(gpu_memory_utilization=0.95)
    assert larger.max_num_batched_tokens > baseline.max_num_batched_tokens > less_headroom.max_num_batched_tokens
    assert sharded.max_num_batched_tokens > baseline.max_num_batched_tokens


@pytest.mark.parametrize("chunk", ["auto", None])
def test_auto_chunk_follows_explicit_budget_without_device_probe(prefill_config, chunk):
    """Explicit budgets remain usable without a memory probe, including CLI input."""
    from sparseengine.entrypoints.openai.api_server import _parse_engine_kwargs

    make, _, probes = prefill_config
    args = _parse_engine_kwargs(["--max-num-batched-tokens", "512"])
    config = make(engine_prefill_chunk_size=chunk, **args)
    assert config.engine_prefill_chunk_size == config.max_num_batched_tokens == 512
    assert not probes


def test_explicit_prefill_limits_are_not_reestimated(prefill_config):
    make, memory, probes = prefill_config
    memory[0] = 1
    config = make(max_num_batched_tokens=1024, engine_prefill_chunk_size=256)
    assert (config.max_num_batched_tokens, config.engine_prefill_chunk_size) == (1024, 256)
    assert not probes


def test_auto_budget_preserves_explicit_chunk_or_reports_infeasibility(prefill_config):
    make, _, _ = prefill_config
    config = make(engine_prefill_chunk_size=64)
    assert config.engine_prefill_chunk_size == 64
    assert config.max_num_batched_tokens >= config.engine_prefill_chunk_size
    with pytest.raises(ValueError, match="below the required"):
        make(engine_prefill_chunk_size=config.max_num_batched_tokens * 2)


def test_auto_long_prefill_keeps_atomic_threshold(prefill_config):
    """Auto must not silently change the full/offload split to fit memory."""
    make, memory, _ = prefill_config
    config = make(sparse_method="pyramidkv", long_prefill_offload_threshold=256)
    assert config.engine_prefill_chunk_size == config.max_num_batched_tokens == config.long_prefill_offload_threshold
    memory[0] //= 128
    with pytest.raises(ValueError, match="offload threshold is not changed"):
        make(sparse_method="pyramidkv", long_prefill_offload_threshold=256)


def test_auto_chunk_uses_final_atomic_budget_and_score_window(prefill_config):
    make, _, probes = prefill_config
    config = make(sparse_method="pyramidkv", max_num_batched_tokens=64,
                  long_prefill_offload_threshold=128, observation_window_size=32)
    assert config.engine_prefill_chunk_size == config.max_num_batched_tokens == 128
    assert not probes
    with pytest.raises(ValueError, match="observation_window_size"):
        make(sparse_method="snapkv", max_num_batched_tokens=16, observation_window_size=32)


def test_explicit_batch_budget_must_fit_final_score_window(prefill_config):
    # Chunk-only validation misses the independent scheduler-wide token cap.
    make, _, probes = prefill_config
    with pytest.raises(ValueError, match="effective prefill step budget"):
        make(
            sparse_method="snapkv",
            engine_prefill_chunk_size=64,
            max_num_batched_tokens=32,
            observation_window_size=64,
        )
    assert not probes


def test_auto_budget_rejects_missing_activation_headroom(prefill_config):
    make, _, _ = prefill_config
    with pytest.raises(ValueError, match="capacity is zero"):
        make(gpu_memory_utilization=1.0)


def test_auto_prefill_resolves_before_cache_fingerprint_and_spawn(prefill_config):
    """Workers and prefix identities must receive the resolved numeric contract."""
    import pickle
    from sparseengine.method_registry import prefill_sparse_method_fingerprint

    make, _, _ = prefill_config
    config = make(prefill_sparse_method="omnikv_prefill", omnikv_prefill_full_attention_layers=[0])
    restored = pickle.loads(pickle.dumps(config))
    fingerprint = prefill_sparse_method_fingerprint(restored)
    assert fingerprint["engine_prefill_chunk_size"] == restored.max_num_batched_tokens
    assert fingerprint["max_num_batched_tokens"] == restored.max_num_batched_tokens
    assert isinstance(restored.max_num_batched_tokens, int)


class RuntimeParamNamingTest(unittest.TestCase):
    def test_config_uses_public_runtime_parameter_names(self):
        config_fields = {field.name for field in fields(Config) if field.init}
        canonical = {
            "sparse_method",
            "prefill_sparse_method",
            "deltakv_checkpoint_path",
            "decode_keep_tokens",
            "sink_keep_tokens",
            "recent_keep_tokens",
            "full_attention_layers",
            "deltakv_neighbor_count",
            "deltakv_center_ratio",
            "deltakv_latent_dim",
            "deltakv_latent_quant_bits",
            "deltakv_latent_quant_group_size",
            "engine_prefill_chunk_size",
            "gpu_memory_utilization",
            "decode_graph",
            "decode_graph_capture_sampling",
            "decode_graph_capture_sizes",
        }
        self.assertLessEqual(canonical, config_fields)

    def test_legacy_runtime_parameter_names_are_not_config_fields(self):
        config_fields = {field.name for field in fields(Config) if field.init}
        legacy = {
            "model_cls",
            "vllm_sparse_method",
            "compressor_path",
            "deltakv_path",
            "num_top_tokens",
            "num_sink_tokens",
            "num_recent_tokens",
            "full_attn_layers",
            "deltakv_k_neighbors",
            "cluster_ratio",
            "kv_compressed_size",
            "kv_quant_bits",
            "kv_quant_group_size",
            "chunk_prefill_size",
            "decode_cuda_graph",
            "decode_cuda_graph_capture_sampling",
            "decode_cuda_graph_capture_sizes",
            "device_memory_utilization",
            "allow_unknown_config_keys",
        }
        self.assertTrue(config_fields.isdisjoint(legacy))

    def test_unknown_runtime_parameter_fails_at_engine_boundary(self):
        from sparseengine import LLM

        with self.assertRaisesRegex(ValueError, "Unknown SparseEngine config keys"):
            LLM("/tmp/unused-model", vllm_sparse_method="omnikv")

    def test_keep_token_budgets_reject_ratio_values(self):
        from sparseengine import LLM

        with self.assertRaisesRegex(ValueError, "integer token count"):
            LLM("/tmp/unused-model", decode_keep_tokens=0.17)

    def test_internal_derived_fields_are_not_public_inputs(self):
        from sparseengine import LLM

        for key in (
            "quest_token_budget",
            "observation_layers",
            "obs_layer_ids",
            "resolved_cache_sparse_method",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Unknown SparseEngine config keys"):
                    LLM("/tmp/unused-model", **{key: 1})


if __name__ == "__main__":
    unittest.main()
