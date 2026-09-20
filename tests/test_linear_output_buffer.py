from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sparseengine.layers.linear import ColumnParallelLinear


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_column_projection_writes_chunk_into_destination(device, dtype, bias):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(419)
    layer = SimpleNamespace(
        weight=torch.randn(128, 64, device=device, dtype=dtype),
        bias=torch.randn(128, device=device, dtype=dtype) if bias else None,
        quantized=False,
    )
    x = torch.randn(19, 64, device=device, dtype=dtype)
    destination = torch.full((23, 128), -17, device=device, dtype=dtype)
    with torch.inference_mode():
        expected = F.linear(x, layer.weight, layer.bias)
        for start in range(0, 19, 7):
            end = min(start + 7, 19)
            result = ColumnParallelLinear.forward(layer, x[start:end], out=destination[start:end])
            assert result.data_ptr() == destination[start:end].data_ptr()
    torch.testing.assert_close(destination[:19], expected, atol=0.125 if dtype == torch.bfloat16 else 1e-5, rtol=0.01 if dtype == torch.bfloat16 else 1e-5)
    assert (destination[19:] == -17).all()


def test_quantized_projection_keeps_provider_output_contract():
    expected = torch.randn(7, 13)
    layer = SimpleNamespace(quantized=True, weight=None, weight_scale_inv=None, bias=None,
                            quant_provider=lambda *args: expected)
    destination = torch.empty_like(expected)
    actual = ColumnParallelLinear.forward(layer, torch.empty(7, 11), out=destination)
    assert actual is destination
    torch.testing.assert_close(actual, expected)
