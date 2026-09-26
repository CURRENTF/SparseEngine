"""Prepared GLM MLA interleaved-to-split-half Q/K rotation."""

from dataclasses import dataclass

import torch

import sparseengine.platforms as platforms
from sparseengine.operators.registry import (
    OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult,
)
from sparseengine.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class GlmRopeSpec:
    activation_dtype: torch.dtype
    heads: int
    rotary_dim: int

    def __post_init__(self):
        if self.heads <= 0 or self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError("GLM RoPE requires positive heads and a positive even rotary dimension")


GLM_ROPE_REGISTRY = OpRegistry(
    "GLM MLA RoPE",
    portfolio=PortfolioPolicy(repo_portable=("triton", "torch")),
)


class GlmRopeProvider:
    def __init__(self, *, op_spec):
        self.spec = op_spec

    def _validate(self, rotary_emb, positions, query, key):
        spec = self.spec
        if (query.ndim != 3 or query.shape[1:] != (spec.heads, spec.rotary_dim)
                or key.shape != (query.shape[0], 1, spec.rotary_dim)):
            raise ValueError("GLM RoPE requires token-aligned Q heads and one K head")
        if query.dtype != spec.activation_dtype or key.dtype != query.dtype:
            raise TypeError("GLM RoPE inputs differ from the prepared activation dtype")
        if not rotary_emb.interleaved:
            raise ValueError("GLM RoPE requires interleaved input and split-half output")
        if positions.shape != (query.shape[0],) or not positions.is_contiguous():
            raise ValueError("GLM RoPE requires contiguous positions, one per token")


@GLM_ROPE_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TorchGlmRopeProvider(GlmRopeProvider):
    name = "torch"

    @classmethod
    def supports(cls, spec, caps):
        return SupportResult.yes()

    def __call__(self, rotary_emb, positions, query, key):
        self._validate(rotary_emb, positions, query, key)
        return rotary_emb.compiled_forward(positions, query, key)


@GLM_ROPE_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonGlmRopeProvider(GlmRopeProvider):
    name = "triton"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA and Triton")
        if spec.activation_dtype != torch.bfloat16 or not caps.supports_bfloat16:
            return SupportResult.unsupported("requires BF16 activations and hardware")
        return SupportResult.yes()

    def __call__(self, rotary_emb, positions, query, key):
        self._validate(rotary_emb, positions, query, key)
        from sparseengine.kernels.triton.glm_mla_decode import fuse_glm_mla_decode_rope

        # The shared kernel rotates Q/K and optionally prepends a non-RoPE
        # segment. A zero-width segment selects rotation alone, including the
        # GLM-specific split-half output layout, without copying the full Q.
        q, k = fuse_glm_mla_decode_rope(
            query[..., :0], query, key.squeeze(1), positions,
            rotary_emb.cos_sin_cache,
        )
        return q, k.unsqueeze(1)


def resolve_glm_rope_provider(*, activation_dtype, heads, rotary_dim, device_index=None):
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    spec = GlmRopeSpec(activation_dtype, int(heads), int(rotary_dim))
    return OpResolver(GLM_ROPE_REGISTRY).resolve(
        spec, platform.get_device_caps(int(device_index)), op_spec=spec,
    ).provider
