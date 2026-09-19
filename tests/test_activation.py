import pytest
import torch
import torch.nn.functional as F

from sparseengine.layers.activation import SiluAndMul
from sparseengine.operators.activation import (
    SiluAndMulSpec,
    TorchSiluAndMulProvider,
    TritonSiluAndMulProvider,
)


def _reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def test_silu_and_mul_cpu_matches_reference():
    x = torch.randn(3, 16, dtype=torch.float32)

    torch.testing.assert_close(
        SiluAndMul(provider=TorchSiluAndMulProvider())(x.clone()),
        _reference(x),
    )


def test_silu_and_mul_rejects_odd_width():
    with pytest.raises(ValueError, match="even final dimension"):
        SiluAndMul(provider=TorchSiluAndMulProvider())(torch.randn(2, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [8, 256, 257])
def test_silu_and_mul_cuda_matches_reference_and_aliases_input(
    dtype: torch.dtype,
    rows: int,
):
    torch.manual_seed(20260810)
    x = torch.randn(rows, 3072, device="cuda", dtype=dtype)
    expected = _reference(x)
    actual_input = x.clone()
    up_before = actual_input[:, 1536:].clone()
    actual = SiluAndMul(
        provider=TritonSiluAndMulProvider(
            op_spec=SiluAndMulSpec(activation_dtype=dtype),
        )
    )(actual_input)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    assert actual.data_ptr() == actual_input.data_ptr()
    torch.testing.assert_close(actual_input[:, :1536], actual)
    torch.testing.assert_close(actual_input[:, 1536:], up_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bound_triton_silu_and_mul_rejects_contract_mismatch():
    provider = TritonSiluAndMulProvider(
        op_spec=SiluAndMulSpec(activation_dtype=torch.bfloat16),
    )

    with pytest.raises(TypeError, match="requires dtype"):
        provider(torch.randn(4, 16, dtype=torch.float32, device="cuda"))
    with pytest.raises(ValueError, match="contiguous"):
        provider(
            torch.randn(4, 16, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
        )
