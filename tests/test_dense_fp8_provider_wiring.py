from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch
from safetensors.torch import save_file

from sparseengine.models.llama import LlamaForCausalLM
from sparseengine.models.qwen2 import Qwen2ForCausalLM
from sparseengine.models.qwen3 import Qwen3ForCausalLM
from sparseengine.models.attention_runtime import resolve_mha_head_dim
from sparseengine.quantization.config import QuantizationConfig
from sparseengine.quantization.fp8 import fp8_blockwise_linear_reference
from sparseengine.utils.loader import load_model


def _parallel_context() -> SimpleNamespace:
    return SimpleNamespace(
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp=SimpleNamespace(all_reduce=lambda tensor: tensor),
    )


def _dense_config(model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        rms_norm_eps=1.0e-6,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        rope_theta=10_000.0,
        rope_scaling=None,
        quantization_config=QuantizationConfig(
            enabled=True,
            quant_method="fp8",
            weight_dtype="e4m3",
            activation_scheme="dynamic",
            weight_block_size=(128, 128),
            model_name=model_name,
        ),
    )


def test_qwen2_style_config_infers_missing_head_dim():
    config = SimpleNamespace(hidden_size=3584, num_attention_heads=28)

    assert resolve_mha_head_dim(config) == 128


def test_explicit_head_dim_remains_authoritative():
    config = SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=32,
        head_dim=256,
    )

    assert resolve_mha_head_dim(config) == 256


def _fp8_weight(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape).clamp(-4.0, 4.0).to(torch.float8_e4m3fn)


def _block_fp8_checkpoint(model) -> dict[str, torch.Tensor]:
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


@pytest.mark.parametrize(
    ("module_name", "model_type", "model_name"),
    [
        ("llama", LlamaForCausalLM, "Llama"),
        ("qwen2", Qwen2ForCausalLM, "Qwen2"),
        ("qwen3", Qwen3ForCausalLM, "Qwen3"),
    ],
)
def test_dense_models_bind_all_fp8_projections_through_shared_registry(
    module_name,
    model_type,
    model_name,
):
    config = _dense_config(model_name)
    context = _parallel_context()
    provider = Mock(name="bound_fp8_linear")

    with (
        patch(
            f"sparseengine.models.{module_name}.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.layers.linear.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.layers.embed_head.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.layers.linear.QuantizationRegistry.resolve_linear_provider",
            return_value=provider,
        ) as resolve_provider,
    ):
        model = model_type(config)

    layer = model.model.layers[0]
    projections = (
        layer.self_attn.qkv_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_up_proj,
        layer.mlp.down_proj,
    )
    assert all(projection.quantized for projection in projections)
    assert all(projection.quant_provider is provider for projection in projections)
    assert all(
        projection.weight.dtype == torch.float8_e4m3fn
        for projection in projections
    )
    assert resolve_provider.call_args_list == [
        call(config.quantization_config, input_features=128, output_features=384),
        call(config.quantization_config, input_features=128, output_features=128),
        call(config.quantization_config, input_features=128, output_features=256),
        call(config.quantization_config, input_features=128, output_features=128),
    ]


@pytest.mark.parametrize(
    ("module_name", "model_type", "model_name"),
    [
        ("llama", LlamaForCausalLM, "Llama"),
        ("qwen2", Qwen2ForCausalLM, "Qwen2"),
    ],
)
def test_new_dense_fp8_models_load_shared_packed_weight_contract(
    tmp_path,
    module_name,
    model_type,
    model_name,
):
    config = _dense_config(model_name)
    context = _parallel_context()
    with (
        patch(
            f"sparseengine.models.{module_name}.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.layers.linear.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.layers.embed_head.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.layers.linear.QuantizationRegistry.resolve_linear_provider",
            return_value=Mock(side_effect=fp8_blockwise_linear_reference),
        ),
    ):
        model = model_type(config)

    checkpoint = _block_fp8_checkpoint(model)
    save_file(checkpoint, tmp_path / "model.safetensors")
    load_model(model, str(tmp_path))

    layer = model.model.layers[0]
    qkv_prefix = "model.layers.0.self_attn."
    expected_qkv_weight = torch.cat(
        [
            checkpoint[qkv_prefix + f"{projection}_proj.weight"]
            for projection in ("q", "k", "v")
        ]
    )
    expected_qkv_scale = torch.cat(
        [
            checkpoint[qkv_prefix + f"{projection}_proj.weight_scale_inv"]
            for projection in ("q", "k", "v")
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
            checkpoint[mlp_prefix + f"{projection}_proj.weight"]
            for projection in ("gate", "up")
        ]
    )
    expected_gate_up_scale = torch.cat(
        [
            checkpoint[mlp_prefix + f"{projection}_proj.weight_scale_inv"]
            for projection in ("gate", "up")
        ]
    )
    assert torch.equal(layer.mlp.gate_up_proj.weight, expected_gate_up_weight)
    assert torch.equal(
        layer.mlp.gate_up_proj.weight_scale_inv,
        expected_gate_up_scale.float(),
    )


@pytest.mark.parametrize(
    ("module_name", "model_type", "model_name"),
    [
        ("llama", LlamaForCausalLM, "Llama"),
        ("qwen2", Qwen2ForCausalLM, "Qwen2"),
    ],
)
def test_dense_mha_models_build_and_bind_full_attention_provider(
    module_name,
    model_type,
    model_name,
):
    config = _dense_config(model_name)
    context = _parallel_context()
    context.attn_tp_size = 1
    engine_config = SimpleNamespace(sparse_method="vanilla")
    engine_config.max_decoding_seqs = 64
    engine_config.decode_graph = True
    prepared_prefill = Mock(name="prepared_prefill")
    prepared_decode = Mock(name="prepared_decode")
    full_attention = Mock(name="full_attention")
    full_attention.prefill_op = prepared_prefill
    full_attention.decode_op = prepared_decode
    full_attention.prefill_name = "shared_prefill"
    full_attention.decode_name = "shared_decode"

    def bind_full_attention(model):
        for layer in model.layers:
            layer.self_attn.attn.full_attention_provider = full_attention
            layer.self_attn.attn.prefill_op = prepared_prefill
            layer.self_attn.attn.decode_op = prepared_decode
        return len(model.layers)

    full_attention.bind.side_effect = bind_full_attention

    with patch(
        f"sparseengine.models.{module_name}.build_mha_full_attention_provider",
        return_value=full_attention,
    ) as build_full_attention:
        kwargs = model_type.build_runtime_kwargs(
            config,
            engine_config=engine_config,
            parallel_context=context,
            device=torch.device("cpu"),
        )

    assert kwargs == {"full_attention_provider": full_attention}
    build_full_attention.assert_called_once_with(
        config,
        sparse_method="vanilla",
        attention_tp_size=1,
        device=torch.device("cpu"),
        max_batch_size=64,
        cuda_graph=True,
        runtime_config=engine_config,
    )

    with (
        patch(
            f"sparseengine.models.{module_name}.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.layers.linear.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.layers.embed_head.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.layers.linear.QuantizationRegistry.resolve_linear_provider",
            return_value=Mock(name="bound_fp8_linear"),
        ),
    ):
        model = model_type(
            config,
            full_attention_provider=full_attention,
        )

    assert all(
        layer.self_attn.attn.prefill_op is prepared_prefill
        for layer in model.model.layers
    )
    assert all(
        layer.self_attn.attn.decode_op is prepared_decode
        for layer in model.model.layers
    )
    assert all(
        layer.self_attn.attn.full_attention_provider is full_attention
        for layer in model.model.layers
    )
    model.close_runtime_operators()
    full_attention.close.assert_called_once_with()
