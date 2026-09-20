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


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("collective", ["none", "inplace", "out_of_place"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_row_projection_output_buffer_preserves_bias_and_collective(rank, collective, device):
    from sparseengine.layers.linear import RowParallelLinear
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(857)
    x = torch.randn(19, 64, device=device)
    weight = torch.randn(128, 64, device=device)
    bias = torch.randn(128, device=device)
    other_rank = torch.randn(19, 128, device=device)
    calls = []

    def reduce(y):
        calls.append(y.data_ptr())
        return y.add_(other_rank) if collective == "inplace" else y + other_rank

    layer = SimpleNamespace(
        weight=weight, bias=bias, tp_rank=rank, quantized=False,
        reduce_results=collective != "none",
        parallel_context=SimpleNamespace(attn_tp=SimpleNamespace(all_reduce=reduce)),
    )
    destination = torch.full((23, 128), -17., device=device)
    with torch.inference_mode():
        actual = RowParallelLinear.forward(layer, x, out=destination[:19])
        expected = F.linear(x, weight, bias if rank == 0 else None)
        if collective != "none":
            expected += other_rank
    assert actual.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(actual, expected)
    assert (destination[19:] == -17).all()
    assert len(calls) == int(collective != "none")
    if calls:
        assert calls[0] == destination.data_ptr()


def test_row_quantized_output_is_reduced_before_copy():
    from sparseengine.layers.linear import RowParallelLinear
    provider_output = torch.randn(7, 13)
    original = provider_output.clone()
    layer = SimpleNamespace(
        tp_rank=1, bias=torch.zeros(13), weight=None, weight_scale_inv=None,
        quantized=True, reduce_results=True,
        quant_provider=lambda x, w, s, bias: provider_output,
        parallel_context=SimpleNamespace(attn_tp=SimpleNamespace(all_reduce=lambda y: y * 2)),
    )
    destination = torch.empty_like(provider_output)
    actual = RowParallelLinear.forward(layer, torch.empty(7, 11), out=destination)
    assert actual is destination
    torch.testing.assert_close(actual, original * 2)
    torch.testing.assert_close(provider_output, original)
