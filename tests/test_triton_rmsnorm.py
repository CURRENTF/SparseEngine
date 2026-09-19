from unittest.mock import patch

import pytest
import torch

import sparseengine.layers.layernorm as layernorm
from sparseengine.layers.layernorm import GemmaRMSNorm, RMSNorm


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered_weight: bool,
) -> torch.Tensor:
    x_float = x.float()
    normalized = x_float * torch.rsqrt(
        x_float.square().mean(dim=-1, keepdim=True) + eps
    )
    effective_weight = weight.float() + float(zero_centered_weight)
    return (normalized * effective_weight).to(x.dtype)


@pytest.fixture
def force_triton_rmsnorm():
    layernorm._resolve_rmsnorm_ops.cache_clear()
    with patch(
        "sparseengine.layers.layernorm.find_spec",
        return_value=None,
    ):
        yield
    layernorm._resolve_rmsnorm_ops.cache_clear()


def test_rmsnorm_rejects_unknown_explicit_provider(monkeypatch):
    monkeypatch.setenv("SPARSEENGINE_RMSNORM_PROVIDER", "unknown")
    layernorm._resolve_rmsnorm_ops.cache_clear()
    try:
        with pytest.raises(
            ValueError,
            match="SPARSEENGINE_RMSNORM_PROVIDER",
        ):
            RMSNorm(128)
    finally:
        layernorm._resolve_rmsnorm_ops.cache_clear()


def test_rmsnorm_does_not_mask_broken_flashinfer_installation():
    layernorm._resolve_rmsnorm_ops.cache_clear()
    try:
        with (
            patch("sparseengine.layers.layernorm.find_spec", return_value=object()),
            patch(
                "sparseengine.layers.layernorm.import_module",
                side_effect=ImportError("broken flashinfer installation"),
            ),
            pytest.raises(ImportError, match="broken flashinfer installation"),
        ):
            RMSNorm(128)
    finally:
        layernorm._resolve_rmsnorm_ops.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("norm_cls", [RMSNorm, GemmaRMSNorm])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 3072, 6144])
def test_triton_rmsnorm_matches_fp32_reference(
    force_triton_rmsnorm,
    norm_cls,
    dtype,
    hidden_size,
):
    torch.manual_seed(61)
    norm = norm_cls(hidden_size, eps=1.0e-6).cuda().to(dtype)
    mean = 0.0 if norm.zero_centered_weight else 1.0
    norm.weight.data.normal_(mean=mean, std=0.2)
    x = torch.randn(7, hidden_size, device="cuda", dtype=dtype)
    original = x.clone()

    actual = norm(x)
    expected = _reference(
        x,
        norm.weight,
        norm.eps,
        zero_centered_weight=norm.zero_centered_weight,
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=3.0e-2)
    assert torch.equal(x, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("norm_cls", [RMSNorm, GemmaRMSNorm])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_fused_add_rmsnorm_matches_flashinfer_mutation_semantics(
    force_triton_rmsnorm,
    norm_cls,
    dtype,
):
    torch.manual_seed(67)
    norm = norm_cls(128, eps=1.0e-6).cuda().to(dtype)
    mean = 0.0 if norm.zero_centered_weight else 1.0
    norm.weight.data.normal_(mean=mean, std=0.2)
    x = torch.randn(7, 128, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    merged = x.float() + residual.float()
    expected_residual = merged.to(dtype)
    expected = _reference(
        merged,
        norm.weight,
        norm.eps,
        zero_centered_weight=norm.zero_centered_weight,
    ).to(dtype)

    actual, actual_residual = norm(x, residual)

    assert actual is x
    assert actual_residual is residual
    assert torch.equal(actual_residual, expected_residual)
    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=3.0e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rmsnorm_supports_strided_qkv_views(
    force_triton_rmsnorm,
    dtype,
):
    torch.manual_seed(71)
    norm = GemmaRMSNorm(128, eps=1.0e-6).cuda().to(dtype)
    norm.weight.data.normal_(mean=0.0, std=0.2)
    projection = torch.randn(7, 14336, device="cuda", dtype=dtype)
    query = projection[:, : 48 * 128].view(7, 48, 128)

    actual = norm(query)
    expected = _reference(
        query,
        norm.weight,
        norm.eps,
        zero_centered_weight=True,
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=3.0e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_fused_add_rmsnorm_rejects_non_flattenable_views(
    force_triton_rmsnorm,
):
    norm = RMSNorm(128).cuda().to(torch.bfloat16)
    storage = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    x = storage[:, :256].view(7, 2, 128)
    residual = torch.randn_like(x)

    with pytest.raises(ValueError, match="without copying"):
        norm(x, residual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("with_residual", [False, True])
def test_triton_rmsnorm_supports_cuda_graph_capture(
    force_triton_rmsnorm,
    with_residual,
):
    torch.manual_seed(73)
    norm = GemmaRMSNorm(128).cuda().to(torch.bfloat16)
    norm.weight.data.normal_(mean=0.0, std=0.2)
    x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    original_x = x.clone()
    original_residual = residual.clone() if residual is not None else None

    norm(x.clone(), residual.clone() if residual is not None else None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = norm(x, residual)
    x.copy_(original_x)
    if residual is not None:
        residual.copy_(original_residual)
    graph.replay()

    actual = output[0] if isinstance(output, tuple) else output
    reference_input = (
        original_x.float() + original_residual.float()
        if original_residual is not None
        else original_x
    )
    expected = _reference(
        reference_input,
        norm.weight,
        norm.eps,
        zero_centered_weight=True,
    ).to(actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=3.0e-2)
    if residual is not None:
        assert torch.equal(residual, reference_input.to(residual.dtype))
