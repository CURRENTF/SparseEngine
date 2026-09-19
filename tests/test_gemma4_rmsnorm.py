import pytest
import torch

from sparseengine.layers.gemma4_rmsnorm import Gemma4RMSNorm
from sparseengine.operators.gemma4 import (
    GEMMA4_REGISTRY,
    Gemma4OpSpec,
    TorchGemma4OperatorProvider,
    TritonGemma4OperatorProvider,
)
from sparseengine.operators.registry import OpResolver
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


def _reference(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    output = x.float()
    output *= torch.pow(output.square().mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        output *= weight.float()
    return output.to(x.dtype)


@pytest.mark.parametrize("with_scale", [False, True])
def test_gemma4_rmsnorm_matches_torch(with_scale):
    torch.manual_seed(7)
    layer = Gemma4RMSNorm(
        32, with_scale=with_scale, provider=TorchGemma4OperatorProvider()
    )
    x = torch.randn(5, 32)
    torch.testing.assert_close(layer(x), _reference(x, layer.weight, layer.eps))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gemma4_rmsnorm_cuda_graph_matches_torch():
    torch.manual_seed(11)
    layer = Gemma4RMSNorm(
        2816, provider=TritonGemma4OperatorProvider()
    ).cuda().to(torch.bfloat16)
    x = torch.randn(4, 2816, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        layer(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(x)
    graph.replay()
    torch.testing.assert_close(output, _reference(x, layer.weight, layer.eps), rtol=0, atol=0)


def test_gemma4_provider_requires_supported_cuda_profile():
    caps = DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="test",
        supports_graph_capture=True,
        supports_triton=True,
    )
    with pytest.raises(RuntimeError, match="requires attention head dimensions"):
        OpResolver(GEMMA4_REGISTRY).resolve(
            Gemma4OpSpec(
                torch.bfloat16,
                (128,),
                cuda_graph=True,
                context_capacity=4096,
            ),
            caps,
        )
