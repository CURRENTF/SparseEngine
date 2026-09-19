import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from benchmark.model_adapters.quantization import build_model_load_kwargs, restore_modules_to_dtype
from sparseengine.utils.loader import load_model


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
        self.compress_down = torch.nn.Sequential(
            torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
        )
        self.cluster = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
        self.nested = torch.nn.Module()
        self.nested.v_compress_up = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)


class _PackedBiasModel(torch.nn.Module):
    packed_modules_mapping = {"k_proj": ("qkv_proj", "k")}

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].self_attn = torch.nn.Module()
        self.model.layers[0].self_attn.qkv_proj = torch.nn.Linear(2, 2, bias=True)
        bias = self.model.layers[0].self_attn.qkv_proj.bias

        def weight_loader(param, loaded_weight, shard_id):
            self.loaded_shard_id = shard_id
            param.data.copy_(loaded_weight)

        bias.weight_loader = weight_loader


class QuantizationHelperTests(unittest.TestCase):
    def test_load_model_accepts_bias_for_a_packed_projection(self):
        model = _PackedBiasModel()
        expected = torch.tensor([1.5, -2.0])
        with tempfile.TemporaryDirectory() as tmp:
            save_file(
                {"model.layers.0.self_attn.k_proj.bias": expected},
                str(Path(tmp) / "model.safetensors"),
            )
            load_model(model, tmp)

        self.assertEqual(model.loaded_shard_id, "k")
        torch.testing.assert_close(model.model.layers[0].self_attn.qkv_proj.bias, expected)

    def test_build_model_load_kwargs_for_4bit(self):
        runtime_cfg, load_kwargs, target_dtype = build_model_load_kwargs(
            {
                "load_in_4bit": True,
                "torch_dtype": "fp16",
                "bnb_4bit_compute_dtype": "bf16",
                "quant_skip_modules": ["custom_head"],
                "engine_prefill_chunk_size": 4096,
            },
            default_torch_dtype=torch.bfloat16,
        )

        self.assertEqual(runtime_cfg, {"engine_prefill_chunk_size": 4096})
        self.assertEqual(target_dtype, torch.float16)
        self.assertIn("quantization_config", load_kwargs)
        quant_cfg = load_kwargs["quantization_config"]
        self.assertTrue(quant_cfg.load_in_4bit)
        self.assertEqual(quant_cfg.bnb_4bit_compute_dtype, torch.bfloat16)
        self.assertIn("compress_down", quant_cfg.llm_int8_skip_modules)
        self.assertIn("custom_head", quant_cfg.llm_int8_skip_modules)

    def test_preserves_chunk_prefill_size(self):
        runtime_cfg, load_kwargs, target_dtype = build_model_load_kwargs(
            {
                "load_in_4bit": True,
                "engine_prefill_chunk_size": 204800000,
            },
            default_torch_dtype=torch.bfloat16,
        )

        self.assertEqual(runtime_cfg["engine_prefill_chunk_size"], 204800000)
        self.assertIn("quantization_config", load_kwargs)
        self.assertEqual(target_dtype, torch.bfloat16)

    def test_restore_modules_to_dtype_skips_transformer(self):
        model = _DummyModel()

        restored = restore_modules_to_dtype(model, torch.bfloat16)

        self.assertIn("compress_down", restored)
        self.assertIn("cluster", restored)
        self.assertIn("nested.v_compress_up", restored)
        self.assertEqual(model.compress_down[0].weight.dtype, torch.bfloat16)
        self.assertEqual(model.cluster.weight.dtype, torch.bfloat16)
        self.assertEqual(model.nested.v_compress_up.weight.dtype, torch.bfloat16)
        self.assertEqual(model.transformer.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
