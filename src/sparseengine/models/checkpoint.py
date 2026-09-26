from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from sparseengine.distributed.topology import ParallelTopology
from sparseengine.quantization.config import QuantizationConfig
from sparseengine.utils.config import config_get


def _validate_architecture(model_name: str, config: Any, expected: str) -> None:
    architectures = tuple(config_get(config, "architectures", ()) or ())
    if architectures != (expected,):
        raise ValueError(
            f"{model_name} requires architectures=[{expected!r}], "
            f"got {list(architectures)}."
        )


def _validate_bf16(model_name: str, config: Any, description: str) -> None:
    dtype = config.dtype
    if dtype not in {torch.bfloat16, "bfloat16"}:
        raise ValueError(f"{model_name} requires {description}, got dtype={dtype!r}.")


def _validate_fields(
    model_name: str,
    config: Any,
    expected_fields: Mapping[str, Any],
) -> None:
    for field, expected in expected_fields.items():
        actual = config_get(config, field, None)
        if actual != expected:
            raise ValueError(
                f"{model_name} requires {field}={expected!r}, got {actual!r}."
            )


def _validate_exclusions(
    model_name: str,
    raw_quantization_config: Any,
    required: set[str],
    description: str,
) -> None:
    excluded = {
        str(name)
        for name in (
            config_get(raw_quantization_config, "modules_to_not_convert", ()) or ()
        )
    }
    missing = sorted(required - excluded)
    if missing:
        raise ValueError(
            f"{model_name} quantization_config must exclude {description}; "
            f"missing {missing[:8]}."
        )


def _validate_dense_fp8(
    config: Any,
    topology: ParallelTopology,
    *,
    model_name: str,
    architecture: str,
) -> None:
    _validate_architecture(model_name, config, architecture)
    _validate_bf16(model_name, config, "BF16 non-quantized parameters")
    num_heads = int(config_get(config, "num_attention_heads", 0) or 0)
    head_dim = int(config_get(config, "head_dim", 0) or 0)
    if head_dim <= 0 and num_heads > 0:
        head_dim = int(config_get(config, "hidden_size", 0) or 0) // num_heads
    dimensions = {
        "hidden_size": int(config_get(config, "hidden_size", 0) or 0),
        "intermediate_size": int(config_get(config, "intermediate_size", 0) or 0),
        "query_size": num_heads * head_dim,
        "key_value_size": int(
            config_get(config, "num_key_value_heads", 0) or 0
        )
        * head_dim,
    }
    alignment = 128 * topology.attn_tp_size
    invalid = {
        name: size
        for name, size in dimensions.items()
        if size <= 0 or size % alignment
    }
    if invalid:
        raise ValueError(
            f"{model_name} requires every TP-local dense projection dimension to be "
            f"128-aligned; TP={topology.attn_tp_size}, invalid={invalid}."
        )


def _validate_qwen3_fp8(config: Any, topology: ParallelTopology) -> None:
    _validate_dense_fp8(
        config,
        topology,
        model_name="Qwen3 FP8",
        architecture="Qwen3ForCausalLM",
    )


def _validate_qwen3_moe_fp8(config: Any, raw_quantization_config: Any) -> None:
    _validate_architecture("Qwen3MoE FP8", config, "Qwen3MoeForCausalLM")
    _validate_bf16("Qwen3MoE FP8", config, "BF16 non-quantized parameters")
    dimensions = {
        "hidden_size": int(config_get(config, "hidden_size", 0) or 0),
        "moe_intermediate_size": int(
            config_get(config, "moe_intermediate_size", 0) or 0
        ),
    }
    invalid = {
        name: size for name, size in dimensions.items() if size <= 0 or size % 128
    }
    if invalid:
        raise ValueError(
            "Qwen3MoE FP8 requires hidden_size and moe_intermediate_size aligned "
            f"to 128, invalid={invalid}."
        )
    required = {"lm_head"}
    required.update(
        f"model.layers.{layer_idx}.mlp.gate"
        for layer_idx in range(int(config_get(config, "num_hidden_layers", 0) or 0))
    )
    _validate_exclusions(
        "Qwen3MoE FP8",
        raw_quantization_config,
        required,
        "lm_head and every router gate",
    )


def _validate_qwen3_moe(
    config: Any,
    raw_quantization_config: Any,
    quantization: QuantizationConfig,
    topology: ParallelTopology,
) -> None:
    decoder_sparse_step = int(config_get(config, "decoder_sparse_step", 1))
    mlp_only_layers = tuple(
        int(layer_idx)
        for layer_idx in (config_get(config, "mlp_only_layers", ()) or ())
    )
    if decoder_sparse_step != 1 or mlp_only_layers:
        raise NotImplementedError(
            "Qwen3MoE v1 requires every decoder layer to be MoE, got "
            f"decoder_sparse_step={decoder_sparse_step}, "
            f"mlp_only_layers={list(mlp_only_layers)}."
        )
    shared_intermediate_size = int(
        config_get(config, "shared_expert_intermediate_size", 0) or 0
    )
    if shared_intermediate_size:
        raise NotImplementedError(
            "Qwen3MoE v1 does not support shared experts, got "
            f"shared_expert_intermediate_size={shared_intermediate_size}."
        )
    dtype = config.dtype
    if dtype not in {torch.bfloat16, torch.float16}:
        raise NotImplementedError(
            "Qwen3MoE v1 supports BF16/FP16 checkpoints only, "
            f"got dtype={dtype}."
        )
    if topology.attn_tp_size > 1 and dtype != torch.bfloat16:
        raise NotImplementedError(
            "Qwen3MoE attention TP supports BF16 checkpoints only, "
            f"got dtype={dtype}."
        )
    if quantization.enabled:
        _validate_qwen3_moe_fp8(config, raw_quantization_config)


def _validate_qwen35_moe(
    outer_config: Any,
    config: Any,
    quantization: QuantizationConfig,
    topology: ParallelTopology,
) -> None:
    _validate_architecture(
        "Qwen3.6 MoE", outer_config, "Qwen3_5MoeForConditionalGeneration"
    )
    _validate_bf16(
        "Qwen3.6 MoE",
        config,
        "BF16 activations with either BF16 or block-FP8 language-model weights",
    )
    _validate_fields(
        "Qwen3.6 MoE",
        config,
        {
            "hidden_act": "silu",
            "attn_output_gate": True,
            "attention_bias": False,
            "partial_rotary_factor": 0.25,
            "mamba_ssm_dtype": "float32",
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": False,
        },
    )
    if quantization.enabled:
        dimensions = {
            "hidden_size": int(config_get(config, "hidden_size", 0) or 0),
            "shared_expert_intermediate_size": int(
                config_get(config, "shared_expert_intermediate_size", 0) or 0
            )
            // topology.attn_tp_size,
        }
        invalid = {
            name: size
            for name, size in dimensions.items()
            if size <= 0 or size % 128
        }
        if invalid:
            raise ValueError(
                "Qwen3.6 MoE FP8 local Linear dimensions must be 128-aligned, "
                f"got TP={topology.attn_tp_size}, invalid={invalid}."
            )


def _validate_minimax(config: Any, raw_quantization_config: Any) -> None:
    _validate_architecture("MiniMax M2.7", config, "MiniMaxM2ForCausalLM")
    _validate_fields(
        "MiniMax M2.7",
        config,
        {
            "hidden_act": "silu",
            "qk_norm_type": "per_layer",
            "scoring_func": "sigmoid",
            "use_qk_norm": True,
            "use_routing_bias": True,
            "use_mtp": True,
            "tie_word_embeddings": False,
        },
    )
    _validate_bf16("MiniMax M2.7", config, "BF16 non-quantized parameters")
    _validate_exclusions(
        "MiniMax M2.7",
        raw_quantization_config,
        {"gate", "e_score_correction_bias", "lm_head"},
        "gate, e_score_correction_bias, and lm_head",
    )


def validate_checkpoint(
    model_type: str,
    *,
    outer_config: Any,
    config: Any,
    raw_quantization_config: Any,
    quantization: QuantizationConfig,
    topology: ParallelTopology,
) -> None:
    validator = CHECKPOINT_VALIDATORS.get(model_type)
    if validator is not None:
        validator(
            outer_config,
            config,
            raw_quantization_config,
            quantization,
            topology,
        )


def _qwen35_checkpoint(_outer, config, _raw, quantization, _topology) -> None:
    if not quantization.enabled:
        _validate_bf16("Qwen3.5", config, "BF16 weights")


def _qwen35_moe_checkpoint(outer, config, _raw, quantization, topology) -> None:
    _validate_qwen35_moe(outer, config, quantization, topology)


def _minimax_checkpoint(_outer, config, raw, _quantization, _topology) -> None:
    _validate_minimax(config, raw)


def _qwen3_checkpoint(_outer, config, _raw, quantization, topology) -> None:
    if quantization.enabled:
        _validate_qwen3_fp8(config, topology)


def _llama_checkpoint(_outer, config, _raw, quantization, topology) -> None:
    if quantization.enabled:
        _validate_dense_fp8(
            config,
            topology,
            model_name="Llama FP8",
            architecture="LlamaForCausalLM",
        )


def _qwen2_checkpoint(_outer, config, _raw, quantization, topology) -> None:
    if quantization.enabled:
        _validate_dense_fp8(
            config,
            topology,
            model_name="Qwen2 FP8",
            architecture="Qwen2ForCausalLM",
        )


def _qwen3_moe_checkpoint(_outer, config, raw, quantization, topology) -> None:
    _validate_qwen3_moe(config, raw, quantization, topology)


def _gemma4_checkpoint(outer, config, _raw, quantization, topology) -> None:
    _validate_architecture("Gemma 4", outer, "Gemma4ForConditionalGeneration")
    _validate_bf16("Gemma 4", config, "BF16 weights")
    _validate_fields(
        "Gemma 4",
        config,
        {
            "attention_bias": False,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": True,
        },
    )
    if quantization.enabled:
        raise NotImplementedError("Gemma 4 currently supports unquantized BF16 checkpoints only.")
    enable_moe = bool(config_get(config, "enable_moe_block", False))
    if topology.moe_ep_size > 1 and not enable_moe:
        raise ValueError("Gemma 4 dense checkpoints require expert_parallel_size=1.")
    if enable_moe and not int(config_get(config, "num_experts", 0) or 0):
        raise ValueError("Gemma 4 MoE requires a positive num_experts.")


def _deepseek_v4_checkpoint(outer, config, raw, quantization, topology):
    _validate_architecture("DeepSeek V4", outer, "DeepseekV4ForCausalLM")
    _validate_bf16("DeepSeek V4", config, "BF16 activations")
    _validate_fields("DeepSeek V4 Flash", config, {
        "hidden_size": 4096, "head_dim": 512, "num_attention_heads": 64,
        "qk_rope_head_dim": 64, "q_lora_rank": 1024, "o_lora_rank": 1024,
        "o_groups": 8, "index_head_dim": 128, "index_n_heads": 64, "index_topk": 512,
        "expert_dtype": "fp4", "scoring_func": "sqrtsoftplus", "sliding_window": 128,
        "hc_mult": 4, "swiglu_limit": 10.0,
    })
    if topology.moe_tp_size != 1 or not quantization.enabled:
        raise ValueError("DeepSeek V4 requires packed MXFP4 experts with MoE TP=1 and block-FP8 dense projections")
    if tuple(config_get(raw, "weight_block_size", ())) != (128, 128) or config_get(raw, "scale_fmt") != "ue8m0":
        raise ValueError("DeepSeek V4 requires original block-128 FP8/UE8M0 checkpoint scales")
    ratios = tuple(config.compress_ratios[:config.num_hidden_layers])
    if len(ratios) != config.num_hidden_layers or any(r not in (0, 4, 128) for r in ratios):
        raise ValueError("DeepSeek V4 requires valid native compression ratios")


CHECKPOINT_VALIDATORS = {
    "deepseek_v4": _deepseek_v4_checkpoint,
    "llama": _llama_checkpoint,
    "qwen2": _qwen2_checkpoint,
    "qwen3": _qwen3_checkpoint,
    "qwen3_moe": _qwen3_moe_checkpoint,
    "qwen3_5": _qwen35_checkpoint,
    "qwen3_5_moe": _qwen35_moe_checkpoint,
    "minimax_m2": _minimax_checkpoint,
    "gemma4": _gemma4_checkpoint,
}
