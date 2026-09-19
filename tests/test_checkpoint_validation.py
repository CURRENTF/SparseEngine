from types import SimpleNamespace

import pytest
import torch

from sparseengine.distributed.topology import ParallelTopology
from sparseengine.models.checkpoint import validate_checkpoint
from sparseengine.quantization.config import QuantizationConfig


def _fp8_quantization(model_name: str) -> QuantizationConfig:
    return QuantizationConfig(
        enabled=True,
        quant_method="fp8",
        weight_dtype="e4m3",
        activation_scheme="dynamic",
        weight_block_size=(128, 128),
        model_name=model_name,
        activation_dtype="bfloat16",
    )


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("llama", "LlamaForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
    ],
)
def test_dense_fp8_checkpoint_boundary_accepts_aligned_models(
    model_type,
    architecture,
):
    config = SimpleNamespace(
        architectures=[architecture],
        dtype=torch.bfloat16,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )

    validate_checkpoint(
        model_type,
        outer_config=config,
        config=config,
        raw_quantization_config={},
        quantization=_fp8_quantization(model_type),
        topology=ParallelTopology(2, 1, 1),
    )


def test_dense_fp8_checkpoint_boundary_rejects_unaligned_tp_projection():
    config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        dtype=torch.bfloat16,
        hidden_size=4096,
        intermediate_size=11072,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )

    with pytest.raises(ValueError, match="128-aligned"):
        validate_checkpoint(
            "qwen2",
            outer_config=config,
            config=config,
            raw_quantization_config={},
            quantization=_fp8_quantization("qwen2"),
            topology=ParallelTopology(2, 1, 1),
        )
