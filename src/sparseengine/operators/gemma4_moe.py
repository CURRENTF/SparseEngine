from __future__ import annotations

import torch
import torch.nn.functional as F

from sparseengine import platforms
from sparseengine.distributed import get_parallel_context
from sparseengine.layers.packed_moe import PackedMoeExperts
from sparseengine.operators.moe import MoeOpSpec, MoeProvider, model_activation_dtype
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


class Gemma4MoeProvider(MoeProvider):
    def load_packed_projection(
        self,
        spec: MoeOpSpec,
        *,
        projection: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        tp_size: int,
        local_expert_start: int,
        local_expert_end: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> tuple[str, ...]:
        if loaded_weight.shape[0] == spec.num_experts:
            loaded_weight = loaded_weight[local_expert_start:local_expert_end]
        if projection == "gate_up_proj":
            gate, up = loaded_weight.chunk(2, 1)
            gate = gate.chunk(tp_size, 1)[tp_rank]
            up = up.chunk(tp_size, 1)[tp_rank]
            w13_weight.copy_(torch.cat((gate, up), 1))
            return "gate_proj", "up_proj"
        if projection == "down_proj":
            w2_weight.copy_(loaded_weight.chunk(tp_size, 2)[tp_rank])
            return ("down_proj",)
        raise ValueError(f"Unsupported Gemma 4 projection {projection!r}.")


GEMMA4_MOE_REGISTRY: OpRegistry[MoeOpSpec, Gemma4MoeProvider] = OpRegistry(
    "Gemma 4 routed GEGLU MoE",
    portfolio=PortfolioPolicy(
        repo_nonstandard=("triton_gemma4_geglu",),
    ),
    profile_order=("triton_gemma4_geglu_h20_profile",),
)


@GEMMA4_MOE_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonGemma4MoeProvider(Gemma4MoeProvider):
    name = "triton_gemma4_geglu"
    gate_up_order = "gate_up"
    _large_token_config = None

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.activation != "gelu_tanh" or spec.routing_method != "softmax":
            return SupportResult.unsupported("requires Gemma 4 GELU-tanh and softmax routing")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.unsupported("requires BF16 or FP16 activations")
        if spec.weight_dtype != spec.activation_dtype or spec.block_shape is not None:
            return SupportResult.unsupported(
                "requires unquantized experts matching activation dtype"
            )
        return SupportResult.yes()

    def run(
        self,
        spec,
        hidden_states,
        topk_ids,
        topk_weights,
        w13_weight,
        w2_weight,
        w13_scale_inv,
        w2_scale_inv,
        *,
        local_expert_start,
        tp_rank,
        ep_rank,
    ):
        del tp_rank, ep_rank
        if w13_scale_inv is not None or w2_scale_inv is not None:
            raise RuntimeError("Gemma 4 BF16 MoE does not accept expert scales.")
        from sparseengine.kernels.triton.gemma4_moe import fused_gemma4_moe

        return fused_gemma4_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
            large_token_config=self._large_token_config,
        )


@GEMMA4_MOE_REGISTRY.register_atomic(
    ProviderRole.REPO_NONSTANDARD,
    profile_only=True,
)
class H20Gemma4MoeProvider(TritonGemma4MoeProvider):
    name = "triton_gemma4_geglu_h20"
    _large_token_config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        triton = super().supports(spec, caps)
        if not triton.supported:
            return triton
        if caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.compute_capability}"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.unsupported("requires CUDA runtime >= 12.8")
        return SupportResult.yes()


@GEMMA4_MOE_REGISTRY.register_profile
class H20Gemma4MoeProfile:
    name = "triton_gemma4_geglu_h20_profile"

    @classmethod
    def atomic_provider_names(cls, spec: MoeOpSpec) -> tuple[str, ...]:
        del spec
        return ("triton_gemma4_geglu_h20",)

    @classmethod
    def matches(cls, spec: MoeOpSpec, caps: DeviceCaps) -> ProfileMatch:
        del spec
        if not device_name_contains(caps.device_name, "H20"):
            return ProfileMatch.no(
                f"requires profiled H20 hardware, got {caps.device_name}"
            )
        return ProfileMatch.yes("matched H20 Gemma 4 MoE profile")

    @classmethod
    def bind(cls, spec: MoeOpSpec, caps: DeviceCaps, **kwargs):
        del spec, caps
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        return H20Gemma4MoeProvider()

class TorchGemma4MoeProvider(Gemma4MoeProvider):
    """Explicit correctness oracle; never selected for production inference."""

    name = "torch_gemma4_geglu"

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        del caps
        if spec.activation != "gelu_tanh" or spec.routing_method != "softmax":
            return SupportResult.unsupported("requires Gemma 4 GELU-tanh and softmax routing")
        if spec.weight_dtype != spec.activation_dtype or spec.block_shape is not None:
            return SupportResult.unsupported("requires unquantized Gemma 4 GELU-tanh experts")
        return SupportResult.yes()

    def run(
        self,
        spec,
        hidden_states,
        topk_ids,
        topk_weights,
        w13_weight,
        w2_weight,
        w13_scale_inv,
        w2_scale_inv,
        *,
        local_expert_start,
        tp_rank,
        ep_rank,
    ):
        del tp_rank, ep_rank
        if w13_scale_inv is not None or w2_scale_inv is not None:
            raise RuntimeError("Gemma 4 Torch MoE does not accept expert scales.")
        output = torch.zeros_like(hidden_states)
        for local_id in range(spec.num_local_experts):
            global_id = int(local_expert_start) + local_id
            token_ids, routes = torch.where(topk_ids == global_id)
            if token_ids.numel() == 0:
                continue
            gate, up = F.linear(hidden_states[token_ids], w13_weight[local_id]).chunk(
                2, -1
            )
            routed = F.linear(
                F.gelu(gate, approximate="tanh") * up, w2_weight[local_id]
            )
            output.index_add_(
                0, token_ids, routed * topk_weights[token_ids, routes, None]
            )
        return output


def resolve_gemma4_moe_provider(
    spec: MoeOpSpec,
    *,
    device_index: int | None = None,
) -> Gemma4MoeProvider:
    if spec.activation != "gelu_tanh":
        raise ValueError(
            "Gemma 4 MoE resolver requires activation='gelu_tanh', "
            f"got {spec.activation!r}."
        )
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(GEMMA4_MOE_REGISTRY).resolve(spec, caps).provider


class Gemma4PackedExperts(PackedMoeExperts):
    def __init__(self, config) -> None:
        super().__init__(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            top_k=config.top_k_experts,
            activation_dtype=model_activation_dtype(config),
            fp8_enabled=False,
            cuda_graph=bool(getattr(config, "decode_graph", False)),
            activation="gelu_tanh",
            model_label="Gemma4MoE",
            provider_resolver=resolve_gemma4_moe_provider,
            parallel_context=get_parallel_context(),
        )

    def rank_local_weight_slice(
        self,
        source_shape: tuple[int, ...],
        *,
        loaded_shard_id: str,
        is_scale: bool = False,
    ) -> tuple[slice, ...] | None:
        if is_scale:
            raise ValueError("Gemma 4 BF16 experts do not use weight scales.")
        if len(source_shape) != 3 or int(source_shape[0]) != self.num_experts:
            raise ValueError(f"Invalid Gemma 4 packed expert shape {source_shape}.")
        if self.ep_size == 1:
            return None
        return (
            slice(self.local_expert_start, self.local_expert_end),
            slice(None),
            slice(None),
        )

    def load_packed_weight(self, projection: str, loaded_weight: torch.Tensor) -> None:
        projections = self.provider.load_packed_projection(
            self.op_spec,
            projection=projection,
            loaded_weight=loaded_weight,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            local_expert_start=self.local_expert_start,
            local_expert_end=self.local_expert_end,
            w13_weight=self.w13_weight.data,
            w2_weight=self.w2_weight.data,
        )
        self._loaded_expert_shards.update(
            (expert_id, name)
            for expert_id in range(self.local_expert_start, self.local_expert_end)
            for name in projections
        )


__all__ = [
    "GEMMA4_MOE_REGISTRY",
    "Gemma4MoeProvider",
    "Gemma4PackedExperts",
    "H20Gemma4MoeProvider",
    "TorchGemma4MoeProvider",
    "TritonGemma4MoeProvider",
    "resolve_gemma4_moe_provider",
]
