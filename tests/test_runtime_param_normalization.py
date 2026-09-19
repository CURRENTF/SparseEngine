import unittest
from dataclasses import fields

from sparseengine.config import Config


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

        with self.assertRaisesRegex(ValueError, "Unknown Sparse-Engine config keys"):
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
                with self.assertRaisesRegex(ValueError, "Unknown Sparse-Engine config keys"):
                    LLM("/tmp/unused-model", **{key: 1})


if __name__ == "__main__":
    unittest.main()
