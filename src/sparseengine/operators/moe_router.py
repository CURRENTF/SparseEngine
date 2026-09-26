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
        if self.routing_method not in {"softmax", "biased_sigmoid", "sqrtsoftplus", "hash_sqrtsoftplus"}:
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
        upstream_standard=("flashinfer_hash_sqrtsoftplus", "torch_sqrtsoftplus"),
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


@torch.compile(fullgraph=True, dynamic=True)
def _sqrtsoftplus_route(logits, bias, top_k, scale):
    scores = torch.nn.functional.softplus(logits).sqrt()
    ids = (scores+bias).topk(top_k, dim=-1, sorted=False).indices
    selected = scores.gather(1, ids)
    normalized = selected / selected.sum(-1, keepdim=True)
    # Mathematical log-space limit when all FP32 scores underflow.
    chosen_logits = logits.gather(1, ids)
    log_scores = .5*torch.where(chosen_logits < -20, chosen_logits,
                               torch.nn.functional.softplus(chosen_logits).log())
    normalized = torch.where(torch.isfinite(normalized), normalized, log_scores.softmax(-1))
    return normalized*scale, ids.to(torch.int32)


@torch.compile(fullgraph=True, dynamic=True)
def _finish_hash_weights(logits, ids, weights, scale):
    chosen = logits.gather(1, ids.long())
    log_scores = .5*torch.where(chosen < -20, chosen, torch.nn.functional.softplus(chosen).log())
    return torch.where(torch.isfinite(weights), weights, log_scores.softmax(-1))*scale


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferHashSqrtSoftplusRouterProvider(MoeRouterProvider):
    name = "flashinfer_hash_sqrtsoftplus"

    @classmethod
    def supports(cls, spec, caps):
        if spec.routing_method != "hash_sqrtsoftplus":
            return SupportResult.unsupported("requires hash sqrt-softplus routing")
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported("hash routing requires SM90")
        from flashinfer.fused_moe import hash_topk
        if not callable(hash_topk):
            return SupportResult.dependency_broken("FlashInfer lacks hash_topk")
        return SupportResult.yes()

    @staticmethod
    def project(hidden_states, weight):
        return torch.mm(hidden_states, weight.T, out_dtype=torch.float32)

    def run(self, spec, router_logits, correction_bias=None, *, routed_scaling_factor=1.,
            input_ids=None, tid2eid=None):
        from flashinfer.fused_moe import hash_topk
        if correction_bias is not None or input_ids is None or tid2eid is None:
            raise ValueError("Hash routing requires input IDs and expert table, without correction bias")
        weights, ids = hash_topk(router_logits, input_ids, tid2eid,
                                num_fused_shared_experts=0, launch_with_pdl=False)
        return _finish_hash_weights(router_logits, ids, weights, routed_scaling_factor), ids


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class TorchSqrtSoftplusRouterProvider(MoeRouterProvider):
    name = "torch_sqrtsoftplus"

    @classmethod
    def supports(cls, spec, caps):
        if spec.routing_method != "sqrtsoftplus":
            return SupportResult.unsupported("requires sqrt-softplus routing")
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported("sqrt-softplus router requires CUDA")
        return SupportResult.yes()

    @staticmethod
    def project(hidden_states, weight):
        return torch.mm(hidden_states, weight.T, out_dtype=torch.float32)

    def run(self, spec, router_logits, correction_bias=None, *, routed_scaling_factor=1.):
        if correction_bias is None or not spec.norm_topk_prob:
            raise ValueError("sqrt-softplus routing requires bias and normalized top-k weights")
        return _sqrtsoftplus_route(router_logits, correction_bias, spec.top_k, routed_scaling_factor)
