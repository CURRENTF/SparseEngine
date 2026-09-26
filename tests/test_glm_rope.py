from dataclasses import replace

import pytest
import torch

from sparseengine.layers.rotary_embedding import RotaryEmbedding
from sparseengine.operators.glm_rope import (
    GLM_ROPE_REGISTRY, GlmRopeSpec, TritonGlmRopeProvider,
)
from sparseengine.operators.registry import OpResolver
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


def _caps():
    return DeviceCaps(
        platform=PlatformEnum.CUDA, device_type="cuda", device_index=0,
        device_name="unprofiled CUDA device", supports_triton=True,
        supports_bfloat16=True,
    )


@pytest.mark.parametrize("change", [
    {"supports_triton": False}, {"supports_bfloat16": False},
    {"platform": PlatformEnum.UNSPECIFIED, "device_type": "cpu"},
])
def test_glm_rope_resolver_preserves_fallback(change):
    spec = GlmRopeSpec(torch.bfloat16, 7, 48)
    provider = OpResolver(GLM_ROPE_REGISTRY).resolve(
        spec, replace(_caps(), **change), op_spec=spec,
    ).provider
    assert provider.name == "torch"


def test_glm_rope_support_is_not_a_measured_shape_whitelist():
    assert TritonGlmRopeProvider.supports(GlmRopeSpec(torch.bfloat16, 7, 48), _caps()).supported
    assert not TritonGlmRopeProvider.supports(GlmRopeSpec(torch.float32, 20, 64), _caps()).supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tokens,heads,dim", [
    (0, 1, 64), (1, 1, 32), (3, 10, 64), (17, 20, 64),
    (33, 7, 48), (257, 20, 64), (2048, 20, 64), (16384, 20, 64),
])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_glm_rope_strided_cuda_matches_reference(tokens, heads, dim, position_dtype):
    torch.manual_seed(936)
    rope = RotaryEmbedding(dim, dim, 4096, 10000, backend="torch", interleaved=True).cuda()
    rope.cos_sin_cache.mul_(1.1)
    q_storage = torch.randn(tokens, heads, dim + 192, device="cuda", dtype=torch.bfloat16)
    k_storage = torch.randn(tokens, 2048 + dim, device="cuda", dtype=torch.bfloat16)
    q = q_storage[..., 192:]
    k = k_storage[:, -dim:].unsqueeze(1)
    q_before, k_before = q_storage.clone(), k_storage.clone()
    positions = torch.randint(0, 4096, (tokens,), device="cuda", dtype=position_dtype)
    cos, sin = rope.cos_sin_cache[positions].chunk(2, -1)

    def reference(x):
        even, odd = x.float()[..., ::2], x.float()[..., 1::2]
        return torch.cat((even * cos - odd * sin, odd * cos + even * sin), -1).to(x.dtype)

    provider = TritonGlmRopeProvider(op_spec=GlmRopeSpec(torch.bfloat16, heads, dim))
    actual = provider(rope, positions, q, k)
    for result, expected in zip(actual, (reference(q), reference(k))):
        torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(q_storage, q_before, atol=0, rtol=0)
    torch.testing.assert_close(k_storage, k_before, atol=0, rtol=0)


def test_glm_rope_rejects_noncontiguous_positions_before_launch():
    provider = TritonGlmRopeProvider(op_spec=GlmRopeSpec(torch.bfloat16, 2, 64))
    rope = RotaryEmbedding(64, 64, 16, 10000, backend="torch", interleaved=True)
    with pytest.raises(ValueError, match="contiguous positions"):
        provider(rope, torch.arange(6)[::2], torch.empty(3, 2, 64, dtype=torch.bfloat16),
                 torch.empty(3, 1, 64, dtype=torch.bfloat16))
