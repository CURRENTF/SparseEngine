from __future__ import annotations

from dataclasses import dataclass

import torch

import sparseengine.platforms as platforms
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class MoeRouterOpSpec:
    num_experts: int
    top_k: int
    activation_dtype: torch.dtype
    norm_topk_prob: bool
    cuda_graph: bool
    routing_method: str = "softmax"

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("MoE router num_experts must be positive.")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"MoE router top_k must be in [1, {self.num_experts}], "
                f"got {self.top_k}."
            )
        if not self.activation_dtype.is_floating_point:
            raise TypeError(
                "MoE router activations must be floating point, "
                f"got {self.activation_dtype}."
            )
        if self.routing_method not in {"softmax", "biased_sigmoid"}:
            raise ValueError(f"Unsupported MoE routing method {self.routing_method!r}.")


class MoeRouterProvider:
    name = ""

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "kernel_path": self.name,
        }

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


MOE_ROUTER_REGISTRY: OpRegistry[MoeRouterOpSpec, MoeRouterProvider] = OpRegistry(
    "MoE router",
    portfolio=PortfolioPolicy(
        repo_nonstandard=(
            "triton_glm_biased_sigmoid",
            "triton_minimax_biased_sigmoid",
            "triton",
        )
    ),
)


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonMoeRouterProvider(MoeRouterProvider):
    name = "triton"

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "kernel_path": "triton.moe_topk.topk_softmax",
        }

    @classmethod
    def supports(
        cls,
        spec: MoeRouterOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if spec.routing_method != "softmax":
            return SupportResult.unsupported("requires softmax routing")
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.unsupported(
                f"requires BF16 or FP16 logits, got {spec.activation_dtype}"
            )
        if spec.num_experts not in {128, 256} or spec.top_k != 8:
            return SupportResult.unsupported(
                "requires num_experts in {128, 256} and top_k=8"
            )
        return SupportResult.yes()

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if correction_bias is not None or routed_scaling_factor != 1.0:
            raise ValueError("Softmax routing does not accept bias or route scaling.")
        from sparseengine.kernels.triton.moe_topk import topk_softmax

        return topk_softmax(
            router_logits,
            top_k=spec.top_k,
            norm_topk_prob=spec.norm_topk_prob,
        )


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class GlmBiasedSigmoidRouterProvider(MoeRouterProvider):
    name = "triton_glm_biased_sigmoid"

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "kernel_path": (
                "triton.moe_biased_sigmoid.fused_topk_biased_sigmoid"
            ),
            "routing_contract": "glm_group_limited_biased_sigmoid",
        }

    @classmethod
    def supports(cls, spec: MoeRouterOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.routing_method != "biased_sigmoid":
            return SupportResult.unsupported("requires biased-sigmoid routing")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if (spec.num_experts, spec.top_k) != (64, 4):
            return SupportResult.unsupported("requires 64 experts and top-k 4")
        return SupportResult.yes()

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if correction_bias is None:
            raise ValueError("Biased-sigmoid routing requires correction_bias.")
        from sparseengine.kernels.triton.moe_biased_sigmoid import (
            fused_topk_biased_sigmoid,
        )

        return fused_topk_biased_sigmoid(
            router_logits,
            correction_bias,
            top_k=spec.top_k,
            routed_scaling_factor=routed_scaling_factor,
        )


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class MiniMaxBiasedSigmoidRouterProvider(MoeRouterProvider):
    """MiniMax M2's exact FP32 biased-sigmoid routing contract."""

    name = "triton_minimax_biased_sigmoid"

    @classmethod
    def supports(cls, spec: MoeRouterOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.routing_method != "biased_sigmoid":
            return SupportResult.unsupported("requires biased-sigmoid routing")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if (spec.num_experts, spec.top_k) != (256, 8):
            return SupportResult.unsupported("requires 256 experts and top-k 8")
        if spec.activation_dtype != torch.float32:
            return SupportResult.unsupported(
                f"requires FP32 logits, got {spec.activation_dtype}"
            )
        if not spec.norm_topk_prob:
            return SupportResult.unsupported("requires normalized top-k probabilities")
        return SupportResult.yes("MiniMax M2 FP32 biased-sigmoid router")

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "kernel_path": "triton.minimax_m2_router.topk_biased_sigmoid",
            "routing_contract": "minimax_m2_biased_sigmoid",
        }

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if correction_bias is None:
            raise ValueError("MiniMax biased-sigmoid routing requires correction_bias.")
        if routed_scaling_factor != 1.0:
            raise ValueError("MiniMax biased-sigmoid routing does not accept route scaling.")
        from sparseengine.kernels.triton.minimax_m2_router import (
            topk_biased_sigmoid,
        )

        return topk_biased_sigmoid(
            router_logits,
            correction_bias,
            top_k=spec.top_k,
        )


def resolve_moe_router_provider(
    spec: MoeRouterOpSpec,
    *,
    device_index: int | None = None,
) -> MoeRouterProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(MOE_ROUTER_REGISTRY).resolve(spec, caps).provider
