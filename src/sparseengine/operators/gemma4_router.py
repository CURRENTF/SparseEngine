from __future__ import annotations

from dataclasses import dataclass

import torch

import sparseengine.platforms as platforms
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProfileMatch,
    ProviderRole,
    SupportResult,
    runtime_version_at_least,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.utils.device_name import device_name_contains


@dataclass(frozen=True)
class Gemma4RouterOpSpec:
    activation_dtype: torch.dtype
    num_experts: int
    top_k: int
    cuda_graph: bool

    def __post_init__(self) -> None:
        if self.num_experts <= 0 or not 0 < self.top_k <= self.num_experts:
            raise ValueError(
                "Gemma 4 router requires 0 < top_k <= num_experts, got "
                f"top_k={self.top_k}, num_experts={self.num_experts}."
            )


class Gemma4RouterProvider:
    name = ""

    def topk(
        self,
        logits: torch.Tensor,
        per_expert_scale: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


GEMMA4_ROUTER_REGISTRY: OpRegistry[Gemma4RouterOpSpec, Gemma4RouterProvider] = (
    OpRegistry(
        "Gemma 4 router",
        portfolio=PortfolioPolicy(repo_nonstandard=("triton",)),
        profile_order=("gemma4_h20_profile",),
    )
)


@GEMMA4_ROUTER_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonGemma4RouterProvider(Gemma4RouterProvider):
    name = "triton"

    @classmethod
    def supports(cls, spec: Gemma4RouterOpSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.unsupported("requires BF16 or FP16 activations")
        return SupportResult.yes()

    def topk(self, logits, per_expert_scale, top_k):
        from sparseengine.kernels.triton.gemma4_router import gemma4_router_topk

        return gemma4_router_topk(logits, per_expert_scale, top_k)


@GEMMA4_ROUTER_REGISTRY.register_atomic(
    ProviderRole.REPO_NONSTANDARD,
    profile_only=True,
)
class H20Gemma4RouterProvider(TritonGemma4RouterProvider):
    name = "gemma4_h20"

    @classmethod
    def supports(cls, spec: Gemma4RouterOpSpec, caps: DeviceCaps) -> SupportResult:
        generic = super().supports(spec, caps)
        if not generic.supported:
            return generic
        if caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.compute_capability}"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.unsupported(
                f"requires CUDA runtime >= 12.8, got {caps.runtime_version or 'unknown'}"
            )
        if spec.num_experts > 1024:
            return SupportResult.unsupported("fused router requires at most 1024 experts")
        return SupportResult.yes()

    def topk(self, logits, per_expert_scale, top_k):
        from sparseengine.kernels.triton.gemma4_fused_router import (
            gemma4_fused_router_topk,
        )

        return gemma4_fused_router_topk(logits, per_expert_scale, top_k)


@GEMMA4_ROUTER_REGISTRY.register_profile
class H20Gemma4RouterProfile:
    name = "gemma4_h20_profile"

    @classmethod
    def atomic_provider_names(cls, spec: Gemma4RouterOpSpec) -> tuple[str, ...]:
        del spec
        return ("gemma4_h20",)

    @classmethod
    def matches(cls, spec: Gemma4RouterOpSpec, caps: DeviceCaps) -> ProfileMatch:
        del spec
        if not device_name_contains(caps.device_name, "H20"):
            return ProfileMatch.no(
                f"requires profiled H20 hardware, got {caps.device_name}"
            )
        return ProfileMatch.yes("matched H20 Gemma 4 router profile")

    @classmethod
    def bind(cls, spec: Gemma4RouterOpSpec, caps: DeviceCaps, **kwargs):
        del spec, caps
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        return H20Gemma4RouterProvider()


class TorchGemma4RouterProvider(Gemma4RouterProvider):
    """Explicit correctness oracle; never selected for production inference."""

    name = "torch_oracle"

    def topk(self, logits, per_expert_scale, top_k):
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        weights, ids = probabilities.topk(top_k, dim=-1)
        weights.div_(weights.sum(-1, keepdim=True)).mul_(per_expert_scale[ids])
        return weights, ids


def resolve_gemma4_router_provider(
    spec: Gemma4RouterOpSpec, *, device_index: int | None = None
) -> Gemma4RouterProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(GEMMA4_ROUTER_REGISTRY).resolve(spec, caps).provider


__all__ = [
    "GEMMA4_ROUTER_REGISTRY",
    "Gemma4RouterOpSpec",
    "Gemma4RouterProvider",
    "H20Gemma4RouterProvider",
    "TorchGemma4RouterProvider",
    "TritonGemma4RouterProvider",
    "resolve_gemma4_router_provider",
]
