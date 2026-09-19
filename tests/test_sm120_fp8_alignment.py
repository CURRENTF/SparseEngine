from unittest.mock import patch

import torch

from sparseengine.operators.fp8_linear import (
    FlashInferGroupwiseSm120Fp8LinearProvider,
    Fp8LinearSpec,
)


def test_flashinfer_sm120_groupwise_pads_unaligned_rows_before_gemm():
    spec = Fp8LinearSpec(
        block_shape=(128, 128),
        input_features=128,
        output_features=128,
    )
    provider = FlashInferGroupwiseSm120Fp8LinearProvider(spec=spec)
    inputs = torch.arange(3 * 128, dtype=torch.bfloat16).reshape(3, 128)
    weight = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones((1, 1), dtype=torch.float32)
    observed = {}

    def quantize(source, quantized, scales, *_args, **_kwargs):
        observed["quantize_shape"] = tuple(source.shape)
        observed["padding"] = source[3].clone()
        quantized.zero_()
        scales.fill_(1.0)

    def gemm(activation, _weight, activation_scale, _weight_scale, **_kwargs):
        observed["gemm_shape"] = tuple(activation.shape)
        assert tuple(activation_scale.shape) == (4, 1)
        return torch.arange(4 * 128, dtype=torch.bfloat16).reshape(4, 128)

    with (
        patch(
            "sparseengine.kernels.external.sgl.moe.sgl_per_token_group_quant_8bit",
            side_effect=quantize,
        ),
        patch(
            "sparseengine.kernels.external.flashinfer.fp8_linear.flashinfer_fp8_nt_groupwise_sm120",
            side_effect=gemm,
        ),
    ):
        output = provider(inputs, weight, weight_scale)

    assert observed["quantize_shape"] == (4, 128)
    assert observed["gemm_shape"] == (4, 128)
    assert torch.count_nonzero(observed["padding"]) == 0
    assert output.shape == (3, 128)
    torch.testing.assert_close(
        output,
        torch.arange(3 * 128, dtype=torch.bfloat16).reshape(3, 128),
    )
