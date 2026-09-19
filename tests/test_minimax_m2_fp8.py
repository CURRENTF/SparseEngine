from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from sparseengine.layers.linear import ReplicatedLinear
from sparseengine.layers.rotary_embedding import (
    RotaryEmbedding,
    apply_partial_rotary_emb,
    apply_rotary_emb,
)
from sparseengine.quantization.fp8 import (
    fp8_blockwise_dequantize,
    fp8_blockwise_linear_reference,
)
from sparseengine.utils.loader import load_model


def _reference_quantization():
    return SimpleNamespace(
        enabled=True,
        quant_method="fp8",
        weight_block_size=(128, 128),
    )


def test_block_fp8_dequant_handles_non_square_boundary_tiles():
    rows, columns = 129, 257
    weight = torch.ones(rows, columns).to(torch.float8_e4m3fn)
    scale = torch.tensor(
        [[1.0, 2.0, 3.0], [5.0, 7.0, 11.0]],
        dtype=torch.float32,
    )

    actual = fp8_blockwise_dequantize(weight, scale)

    expected = torch.empty(rows, columns)
    expected[:128, :128] = 1.0
    expected[:128, 128:256] = 2.0
    expected[:128, 256:] = 3.0
    expected[128:, :128] = 5.0
    expected[128:, 128:256] = 7.0
    expected[128:, 256:] = 11.0
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_block_fp8_linear_reference_matches_dynamic_w8a8_oracle():
    torch.manual_seed(0)
    weight = torch.randn(130, 129).clamp(-4, 4).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 2, dtype=torch.float32) + 0.25
    inputs = torch.randn(3, 129)

    actual = fp8_blockwise_linear_reference(inputs, weight, scale)
    expected = torch.zeros(3, 130, dtype=torch.float32)
    for column_block, column_start in enumerate(range(0, 129, 128)):
        column_end = min(column_start + 128, 129)
        input_block = inputs[:, column_start:column_end]
        input_scale = (
            input_block.abs().amax(dim=-1) / torch.finfo(torch.float8_e4m3fn).max
        ).clamp_min(1.0e-12)
        input_quantized = (input_block / input_scale[:, None]).to(
            torch.float8_e4m3fn
        )
        for row_block, row_start in enumerate(range(0, 130, 128)):
            row_end = min(row_start + 128, 130)
            partial = F.linear(
                input_quantized.float(),
                weight[
                    row_start:row_end,
                    column_start:column_end,
                ].float(),
            )
            expected[:, row_start:row_end].add_(
                partial * input_scale[:, None] * scale[row_block, column_block]
            )

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_block_fp8_dequant_rejects_wrong_scale_axis():
    weight = torch.ones(129, 257).to(torch.float8_e4m3fn)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        fp8_blockwise_dequantize(weight, torch.ones(3, 2))


def test_partial_rope_preserves_pass_through_features_bitwise():
    torch.manual_seed(1)
    rotary = RotaryEmbedding(
        head_size=4,
        rotary_dim=4,
        max_position_embeddings=16,
        base=10_000.0,
        backend="torch",
    )
    positions = torch.tensor([1, 3, 7])
    query = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    key = torch.randn(3, 1, 8, dtype=torch.bfloat16)

    actual_query, actual_key = apply_partial_rotary_emb(
        rotary,
        positions,
        query,
        key,
        rotary_dim=4,
    )

    cos, sin = rotary.cos_sin_cache[positions].chunk(2, dim=-1)
    expected_query_prefix = apply_rotary_emb(query[..., :4], cos, sin)
    expected_key_prefix = apply_rotary_emb(key[..., :4], cos, sin)
    assert torch.equal(actual_query[..., :4], expected_query_prefix)
    assert torch.equal(actual_key[..., :4], expected_key_prefix)
    assert torch.equal(actual_query[..., 4:], query[..., 4:])
    assert torch.equal(actual_key[..., 4:], key[..., 4:])


class _StrictQuantizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        context = SimpleNamespace(tp_rank=0, tp_size=1)
        with patch(
            "sparseengine.layers.linear.get_parallel_context",
            return_value=context,
        ):
            self.proj = ReplicatedLinear(
                128,
                128,
                quantization=_reference_quantization(),
            )
        self.validated_names = None

    def validate_loaded_weights(self, loaded_parameter_names):
        self.validated_names = set(loaded_parameter_names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_loader_reports_grouped_quantized_dense_parameter_as_loaded(tmp_path):
    weight = torch.randn(128, 128).clamp(-4, 4).to(torch.float8_e4m3fn)
    scale = torch.rand(1, 1, dtype=torch.float32)
    save_file(
        {
            "proj.weight": weight,
            "proj.weight_scale_inv": scale,
        },
        tmp_path / "model.safetensors",
    )
    model = _StrictQuantizedModel()

    load_model(model, str(tmp_path))

    assert model.validated_names == {"proj.weight"}
    assert torch.equal(model.proj.weight, weight)
    assert torch.equal(model.proj.weight_scale_inv, scale)


def test_loader_rejects_duplicate_tensor_keys_across_shards(tmp_path):
    save_file({"weight": torch.ones(1)}, tmp_path / "model-00001.safetensors")
    save_file({"weight": torch.zeros(1)}, tmp_path / "model-00002.safetensors")
    model = torch.nn.Linear(1, 1, bias=False)

    with pytest.raises(ValueError, match="duplicate tensor keys"):
        load_model(model, str(tmp_path))
