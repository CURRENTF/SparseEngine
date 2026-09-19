from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

from sparseengine.config import QuantizationConfig
from sparseengine.models.qwen3 import Qwen3ForCausalLM
from sparseengine.quantization.fp8 import fp8_blockwise_linear_reference
from sparseengine.utils.loader import load_model


def _parallel_context():
    return SimpleNamespace(
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp=SimpleNamespace(all_reduce=lambda tensor: tensor),
    )


def _config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    config.quantization_config = QuantizationConfig(
        enabled=True,
        quant_method="fp8",
        weight_dtype="e4m3",
        activation_scheme="dynamic",
        weight_block_size=(128, 128),
        model_name="Qwen3",
    )
    return config


def _fp8_weight(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape).clamp(-4.0, 4.0).to(torch.float8_e4m3fn)


def _checkpoint(model: Qwen3ForCausalLM) -> dict[str, torch.Tensor]:
    checkpoint: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".self_attn.qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            for projection in ("q", "k", "v"):
                checkpoint[prefix + f"{projection}_proj.weight"] = _fp8_weight(
                    (128, 128)
                )
                checkpoint[
                    prefix + f"{projection}_proj.weight_scale_inv"
                ] = torch.rand(1, 1, dtype=torch.bfloat16) + 0.25
        elif name.endswith(".mlp.gate_up_proj.weight"):
            prefix = name[: -len("gate_up_proj.weight")]
            for projection in ("gate", "up"):
                checkpoint[prefix + f"{projection}_proj.weight"] = _fp8_weight(
                    (128, 128)
                )
                checkpoint[
                    prefix + f"{projection}_proj.weight_scale_inv"
                ] = torch.rand(1, 1, dtype=torch.bfloat16) + 0.25
        elif parameter.dtype == torch.float8_e4m3fn:
            checkpoint[name] = _fp8_weight(tuple(parameter.shape))
            checkpoint[name[: -len(".weight")] + ".weight_scale_inv"] = (
                torch.rand(
                    (parameter.shape[0] + 127) // 128,
                    (parameter.shape[1] + 127) // 128,
                    dtype=torch.bfloat16,
                )
                + 0.25
            )
        else:
            checkpoint[name] = torch.randn(parameter.shape, dtype=parameter.dtype)
    return checkpoint


def test_qwen3_dense_loads_official_block_fp8_projection_layout(tmp_path):
    context = _parallel_context()
    with (
        patch("sparseengine.models.qwen3.get_parallel_context", return_value=context),
        patch("sparseengine.layers.linear.get_parallel_context", return_value=context),
        patch("sparseengine.layers.embed_head.get_parallel_context", return_value=context),
        patch(
            "sparseengine.layers.linear.QuantizationRegistry.resolve_linear_provider",
            return_value=Mock(side_effect=fp8_blockwise_linear_reference),
        ),
    ):
        model = Qwen3ForCausalLM(_config())

    checkpoint = _checkpoint(model)
    save_file(checkpoint, tmp_path / "model.safetensors")
    load_model(model, str(tmp_path))

    layer = model.model.layers[0]
    qkv_prefix = "model.layers.0.self_attn."
    expected_qkv_weight = torch.cat(
        [checkpoint[qkv_prefix + f"{name}_proj.weight"] for name in ("q", "k", "v")]
    )
    expected_qkv_scale = torch.cat(
        [
            checkpoint[qkv_prefix + f"{name}_proj.weight_scale_inv"]
            for name in ("q", "k", "v")
        ]
    )
    assert torch.equal(layer.self_attn.qkv_proj.weight, expected_qkv_weight)
    assert torch.equal(
        layer.self_attn.qkv_proj.weight_scale_inv,
        expected_qkv_scale.float(),
    )

    mlp_prefix = "model.layers.0.mlp."
    expected_gate_up_weight = torch.cat(
        [
            checkpoint[mlp_prefix + f"{name}_proj.weight"]
            for name in ("gate", "up")
        ]
    )
    expected_gate_up_scale = torch.cat(
        [
            checkpoint[mlp_prefix + f"{name}_proj.weight_scale_inv"]
            for name in ("gate", "up")
        ]
    )
    assert torch.equal(layer.mlp.gate_up_proj.weight, expected_gate_up_weight)
    assert torch.equal(
        layer.mlp.gate_up_proj.weight_scale_inv,
        expected_gate_up_scale.float(),
    )


def test_qwen3_dense_fp8_forward_uses_loaded_scales(tmp_path):
    context = _parallel_context()
    with (
        patch("sparseengine.models.qwen3.get_parallel_context", return_value=context),
        patch("sparseengine.layers.linear.get_parallel_context", return_value=context),
        patch("sparseengine.layers.embed_head.get_parallel_context", return_value=context),
        patch(
            "sparseengine.layers.linear.QuantizationRegistry.resolve_linear_provider",
            return_value=Mock(side_effect=fp8_blockwise_linear_reference),
        ),
    ):
        model = Qwen3ForCausalLM(_config())

    checkpoint = _checkpoint(model)
    save_file(checkpoint, tmp_path / "model.safetensors")
    load_model(model, str(tmp_path))
    projection = model.model.layers[0].self_attn.o_proj
    inputs = torch.randn(3, 128, dtype=torch.bfloat16)

    actual = projection(inputs)
    expected = fp8_blockwise_linear_reference(
        inputs,
        projection.weight,
        projection.weight_scale_inv,
    )
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
