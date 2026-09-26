from __future__ import annotations

from dataclasses import dataclass

import torch

import sparseengine.platforms as platforms
from sparseengine.platforms import device_runtime
from sparseengine.kernels.moe import MoeAlignment
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
class MoeOpSpec:
    num_experts: int
    num_local_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    activation_dtype: torch.dtype
    weight_dtype: torch.dtype
    block_shape: tuple[int, int] | None
    ep_size: int
    cuda_graph: bool
    tp_size: int = 1
    routing_method: str = "softmax"
    scale_dtype: torch.dtype | None = None
    activation: str = "silu"
    max_num_tokens: int = 1
    activation_limit: float | None = None

    def __post_init__(self) -> None:
        if self.num_experts <= 0 or self.num_local_experts <= 0:
            raise ValueError("MoE expert counts must be positive.")
        if self.ep_size <= 0:
            raise ValueError("MoE ep_size must be positive.")
        if self.tp_size <= 0:
            raise ValueError("MoE tp_size must be positive.")
        if self.num_local_experts * self.ep_size != self.num_experts:
            raise ValueError(
                "MoE local expert topology is inconsistent: "
                f"{self.num_local_experts} * {self.ep_size} != {self.num_experts}."
            )
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("MoE hidden/intermediate sizes must be positive.")
        if self.max_num_tokens <= 0:
            raise ValueError("MoE max_num_tokens must be positive.")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"MoE top_k must be in [1, {self.num_experts}], got {self.top_k}."
            )
        if self.block_shape is not None and (
            len(self.block_shape) != 2 or any(value <= 0 for value in self.block_shape)
        ):
            raise ValueError(
                f"MoE block_shape must contain two positive values, got {self.block_shape}."
            )
        if self.routing_method not in {"softmax", "biased_sigmoid", "sqrtsoftplus", "hash_sqrtsoftplus"}:
            raise ValueError(
                "Unsupported MoE routing method: "
                f"got {self.routing_method!r}."
            )
        if self.activation not in {"silu", "gelu_tanh", "clipped_silu"}:
            raise ValueError(
                "Unsupported MoE activation: "
                f"got {self.activation!r}."
            )
        if self.activation == "clipped_silu":
            if self.activation_limit is None or self.activation_limit <= 0:
                raise ValueError("Clipped SwiGLU requires a positive activation limit")
        elif self.activation_limit is not None:
            raise ValueError("Activation limit requires clipped SwiGLU")


def model_activation_dtype(config) -> torch.dtype:
    value = config.dtype
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value or "").lower().replace("torch.", "")
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    return torch.bfloat16


def _silu_activation_support(spec: MoeOpSpec) -> SupportResult | None:
    if spec.activation != "silu":
        return SupportResult.unsupported(
            f"requires SiLU activation, got {spec.activation}"
        )
    return None


class MoeProvider:
    name = ""
    gate_up_order = "gate_up"

    def prepare_prefill_alignment(self, spec, caps) -> None:
        """Prepare optional eager routing outside the model forward path."""

    def bind_workspace_lane(self, lane: str) -> None:
        """Bind standalone shared projections; routed scratch stays on its lane."""
        for provider, _, _ in getattr(self, "_shared_projections", ()):
            provider.bind_workspace_lane(lane)

    def prepare_shared_expert(self, spec, w13, w2, scale13, scale2):
        """Bind standalone projections over a packed tensor-FP8 expert's views."""
        if spec.weight_dtype != torch.float8_e4m3fn or spec.block_shape is not None:
            raise ValueError("Standalone packed FP8 shared projection requires tensor scales")
        from sparseengine.operators.fp8_linear import resolve_fp8_linear_provider
        providers = []
        for weight, scale in ((w13, scale13), (w2, scale2)):
            provider = resolve_fp8_linear_provider(
                (128, 128), input_features=weight.shape[1], output_features=weight.shape[0],
                activation_dtype=spec.activation_dtype, cuda_graph=spec.cuda_graph,
                weight_layout_id="tensor_scales_in_block_grid")
            provider.prepare_weights(weight, scale)
            providers.append((provider, weight, scale))
        self._shared_projections = tuple(providers)

    def run_shared_expert(self, hidden_states):
        from sparseengine.kernels.triton.silu_and_mul import silu_and_mul_fwd
        first, second = self._shared_projections
        packed = first[0](hidden_states, first[1], first[2])
        activated = silu_and_mul_fwd(packed, gate_up_order=self.gate_up_order)
        return second[0](activated, second[1], second[2])

    @property
    def weight_layout_id(self) -> str:
        return f"packed_{self.gate_up_order}_v1"

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "weight_layout_id": self.weight_layout_id,
        }

    def _packed_projection_offset(
        self,
        projection: str,
        intermediate_size: int,
    ) -> int:
        if projection not in {"gate", "up"}:
            raise ValueError(f"Unknown packed MoE projection {projection!r}.")
        first = "gate" if self.gate_up_order == "gate_up" else "up"
        return 0 if projection == first else int(intermediate_size)

    def load_expert_projection(
        self,
        spec: MoeOpSpec,
        *,
        local_expert_id: int,
        projection: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale_inv: torch.Tensor | None,
        w2_scale_inv: torch.Tensor | None,
    ) -> None:
        if not 0 <= local_expert_id < spec.num_local_experts:
            raise ValueError(
                f"Local expert id {local_expert_id} is outside "
                f"[0, {spec.num_local_experts})."
            )
        if projection == "down":
            weight_target = w2_weight[local_expert_id]
            scale_target = (
                None if w2_scale_inv is None else w2_scale_inv[local_expert_id]
            )
        elif projection in {"gate", "up"}:
            weight_offset = self._packed_projection_offset(
                projection,
                spec.intermediate_size,
            )
            weight_target = w13_weight[
                local_expert_id,
                weight_offset : weight_offset + spec.intermediate_size,
            ]
            if w13_scale_inv is None:
                scale_target = None
            else:
                scale_rows = w13_scale_inv.shape[1] // 2
                scale_offset = self._packed_projection_offset(
                    projection,
                    scale_rows,
                )
                scale_target = w13_scale_inv[
                    local_expert_id,
                    scale_offset : scale_offset + scale_rows,
                ]
        else:
            raise ValueError(f"Unknown logical MoE projection {projection!r}.")

        if tuple(loaded_weight.shape) != tuple(weight_target.shape):
            raise ValueError(
                "MoE expert weight shape mismatch: "
                f"expected={tuple(weight_target.shape)}, "
                f"got={tuple(loaded_weight.shape)}."
            )
        if (loaded_scale is None) != (scale_target is None):
            raise ValueError(
                "MoE expert scale presence does not match provider storage."
            )
        if loaded_scale is not None and scale_target is not None:
            if tuple(loaded_scale.shape) != tuple(scale_target.shape):
                raise ValueError(
                    "MoE expert scale shape mismatch: "
                    f"expected={tuple(scale_target.shape)}, "
                    f"got={tuple(loaded_scale.shape)}."
                )
            scale_target.copy_(loaded_scale)
        weight_target.copy_(loaded_weight)

    def prepare(
        self,
        spec: MoeOpSpec,
        *,
        device: torch.device,
        tp_rank: int,
        ep_rank: int,
        max_num_tokens: int | None = None,
    ) -> None:
        del spec, device, tp_rank, ep_rank, max_num_tokens

    def run(
        self,
        spec: MoeOpSpec,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale_inv: torch.Tensor | None,
        w2_scale_inv: torch.Tensor | None,
        *,
        local_expert_start: int,
        tp_rank: int,
        ep_rank: int,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class MoeDispatchRoute:
    min_tokens: int
    max_tokens: int | None
    provider: MoeProvider
    kernel_path: str

    def matches(self, num_tokens: int) -> bool:
        return num_tokens >= self.min_tokens and (
            self.max_tokens is None or num_tokens <= self.max_tokens
        )


class MoeDispatchPlan(MoeProvider):
    """Prepared token-range dispatch over layout-compatible atomic providers."""

    def __init__(self, spec: MoeOpSpec) -> None:
        self.spec = spec
        self.routes = tuple(self._build_routes(spec))
        self._validate_routes()
        self._runtime_kernel_path_counts: dict[str, dict[str, int]] = {}

    @classmethod
    def bind(cls, spec: MoeOpSpec, caps: DeviceCaps, **kwargs) -> MoeDispatchPlan:
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        plan = cls(spec)
        for route in plan.routes:
            route.provider.prepare_prefill_alignment(spec, caps)
        return plan

    def _build_routes(self, spec: MoeOpSpec) -> tuple[MoeDispatchRoute, ...]:
        raise NotImplementedError

    def _validate_routes(self) -> None:
        if not self.routes:
            raise ValueError(f"{self.name} must contain at least one dispatch route.")
        expected_min = 0
        for index, route in enumerate(self.routes):
            if route.min_tokens != expected_min:
                raise ValueError(
                    f"{self.name} has a token-range gap or overlap before "
                    f"M={route.min_tokens}; expected M={expected_min}."
                )
            if route.max_tokens is not None and route.max_tokens < route.min_tokens:
                raise ValueError(
                    f"{self.name} has an invalid token range "
                    f"[{route.min_tokens}, {route.max_tokens}]."
                )
            if route.max_tokens is None and index != len(self.routes) - 1:
                raise ValueError(
                    f"{self.name} has routes after an unbounded token range."
                )
            if route.provider.weight_layout_id != self.weight_layout_id:
                raise ValueError(
                    f"{self.name} route {route.provider.name!r} requires layout "
                    f"{route.provider.weight_layout_id!r}, but the plan owns "
                    f"{self.weight_layout_id!r}."
                )
            expected_min = (
                route.max_tokens + 1
                if route.max_tokens is not None
                else expected_min
            )
        if self.routes[-1].max_tokens is not None:
            raise ValueError(f"{self.name} must cover all token counts.")

    def _route(self, num_tokens: int) -> MoeDispatchRoute:
        for route in self.routes:
            if route.matches(num_tokens):
                return route
        raise RuntimeError(f"{self.name} has no prepared route for M={num_tokens}.")

    def _record_runtime_kernel_path(self, path: str) -> None:
        counts = self._runtime_kernel_path_counts.setdefault(
            str(path),
            {"eager_dispatches": 0, "cuda_graph_capture_dispatches": 0},
        )
        key = (
            "cuda_graph_capture_dispatches"
            if device_runtime.is_stream_capturing()
            else "eager_dispatches"
        )
        counts[key] += 1

    def prepare(
        self,
        spec: MoeOpSpec,
        *,
        device: torch.device,
        tp_rank: int,
        ep_rank: int,
        max_num_tokens: int | None = None,
    ) -> None:
        if spec is not self.spec:
            raise RuntimeError(
                f"{self.name} was bound for {self.spec!r}, got {spec!r}."
            )
        requested_max = (
            int(spec.max_num_tokens)
            if max_num_tokens is None
            else int(max_num_tokens)
        )
        if requested_max <= 0:
            raise ValueError(
                f"MoE prepared max_num_tokens must be positive, got {requested_max}."
            )
        prepared_max = min(int(spec.max_num_tokens), requested_max)
        for route in self.routes:
            if route.min_tokens > prepared_max:
                continue
            route_max = (
                prepared_max
                if route.max_tokens is None
                else min(prepared_max, route.max_tokens)
            )
            route.provider.prepare(
                spec,
                device=device,
                tp_rank=tp_rank,
                ep_rank=ep_rank,
                max_num_tokens=route_max,
            )

    def runtime_kernel_stats(self) -> dict[str, object]:
        return {
            "kernel_paths": {
                path: {key: int(value) for key, value in sorted(counts.items())}
                for path, counts in sorted(self._runtime_kernel_path_counts.items())
            },
            "fallback_reasons": {},
        }

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "dispatch_plan",
            "weight_layout_id": self.weight_layout_id,
            "routes": [
                {
                    "min_tokens": route.min_tokens,
                    "max_tokens": route.max_tokens,
                    "provider": route.provider.name,
                    "kernel_path": route.kernel_path,
                    "provider_metadata": route.provider.binding_metadata(),
                }
                for route in self.routes
            ],
        }

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
        if spec is not self.spec:
            raise RuntimeError(
                f"{self.name} was bound for {self.spec!r}, got {spec!r}."
            )
        route = self._route(int(hidden_states.shape[0]))
        self._record_runtime_kernel_path(route.kernel_path)
        return route.provider.run(
            spec,
            hidden_states,
            topk_ids,
            topk_weights,
            w13_weight,
            w2_weight,
            w13_scale_inv,
            w2_scale_inv,
            local_expert_start=local_expert_start,
            tp_rank=tp_rank,
            ep_rank=ep_rank,
        )

MOE_REGISTRY: OpRegistry[MoeOpSpec, MoeProvider] = OpRegistry(
    "routed MoE",
    portfolio=PortfolioPolicy(
        upstream_standard=("flashinfer_cutlass_fp8_sm90", "flashinfer_cutlass_mxfp4_sm90"),
        repo_portable=("triton_minimax_m2_fused", "triton"),
    ),
    profile_order=(
        "h20_qwen36_fp8_dispatch_plan",
        "hopper_qwen36_fp8_dispatch_plan",
        "triton_minimax_m2_profile",
        "qwen3_fp8_dispatch_plan",
        "sgl_triton_glm_tp1_bf16_profile",
        "glm_h100_bf16_decode_dispatch_plan",
        "sgl_aligned_triton_glm_bf16_profile",
        "h20_qwen36_fused_bf16_profile",
        "hopper_fused_bf16_profile",
        "qwen3_bf16_dispatch_plan",
    ),
)

_PACKED_SHARED_EXPERT_PROFILES = frozenset(
    {
        (64, 1, 4, 2048, 1536, 1, 1),
        (64, 1, 4, 2048, 1536, 2, 1),
    }
)
_PACKED_SHARED_PREFILL_PROFILES = frozenset(
    {(64, 1, 4, 2048, 1536, 1, 1)}
)


def _tensor_fp8_shared_decode_profile(profile, cuda_graph):
    if not cuda_graph or profile != (64, 1, 4, 2048, 1536, 1, 1):
        return False
    platform = platforms.current_platform
    if not platform.is_cuda_alike():
        return False
    caps = platform.get_device_caps(torch.cuda.current_device())
    if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0) or not device_name_contains(caps.device_name, "H100"):
        return False
    import triton
    return (str(torch.__version__).split("+")[0].startswith("2.11.")
            and str(triton.__version__).startswith("3.6.")
            and str(caps.runtime_version) == "13.0")


def packed_shared_expert_token_limit(weight_dtype):
    """The tensor-FP8 decode packing profile covers batches through four."""
    return 4 if weight_dtype == torch.float8_e4m3fn else None


def use_packed_shared_experts(
    *,
    num_routed_experts: int,
    num_shared_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    tp_size: int,
    ep_size: int,
    cuda_graph: bool,
    weight_dtype: torch.dtype = torch.bfloat16,
    fp8_tensor_scales: bool = False,
) -> bool:
    """Return whether a profiled decode path packs shared experts as routes."""

    profile = (
        int(num_routed_experts),
        int(num_shared_experts),
        int(top_k),
        int(hidden_size),
        int(intermediate_size),
        int(tp_size),
        int(ep_size),
    )
    if weight_dtype == torch.float8_e4m3fn and fp8_tensor_scales:
        return _tensor_fp8_shared_decode_profile(profile, cuda_graph)
    return weight_dtype == torch.bfloat16 and bool(cuda_graph) and profile in _PACKED_SHARED_EXPERT_PROFILES


def use_packed_shared_experts_in_prefill(
    *,
    num_routed_experts: int,
    num_shared_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    tp_size: int,
    ep_size: int,
    cuda_graph: bool,
    weight_dtype: torch.dtype = torch.bfloat16,
) -> bool:
    profile = (
        int(num_routed_experts),
        int(num_shared_experts),
        int(top_k),
        int(hidden_size),
        int(intermediate_size),
        int(tp_size),
        int(ep_size),
    )
    return weight_dtype == torch.bfloat16 and bool(cuda_graph) and profile in _PACKED_SHARED_PREFILL_PROFILES


def append_shared_expert_route(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    shared_expert_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sparseengine.kernels.triton.moe import append_shared_expert_route as run

    return run(
        topk_ids,
        topk_weights,
        shared_expert_id=shared_expert_id,
    )


def _sgl_moe_align_block_size(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> MoeAlignment:
    from sparseengine.kernels.external.sgl.moe import sgl_moe_align_block_size
    from sparseengine.kernels.triton.moe import localize_expert_ids

    num_local_experts = int(local_expert_end) - int(local_expert_start)
    has_remote_experts = num_local_experts != int(num_experts)
    if has_remote_experts:
        topk_ids = localize_expert_ids(
            topk_ids,
            local_expert_start=local_expert_start,
            local_expert_end=local_expert_end,
            remote_expert_id=-1,
        )
    return sgl_moe_align_block_size(
        topk_ids,
        block_size=block_size,
        num_experts=num_local_experts,
    )


def _prefill_moe_align_block_size(topk_ids, **kwargs) -> MoeAlignment:
    # Tiny assignments already need just one launch and no sorting. Retain
    # that algorithm; otherwise use the maintained fused alignment adapter.
    assignments = topk_ids.numel()
    sharded = kwargs["local_expert_end"] - kwargs["local_expert_start"] != kwargs["num_experts"]
    # Large sharded batches also need localization and an ignored-expert
    # bucket. Measurements favor the original skip-remote algorithm beyond
    # this range; unsharded batches retain the SGL path.
    if assignments * 4 <= kwargs["num_experts"] or (sharded and assignments > 32768):
        from sparseengine.kernels.triton.moe import _prepare_expert_assignment

        return _prepare_expert_assignment(topk_ids, **kwargs)
    return _sgl_moe_align_block_size(topk_ids, **kwargs)


@MOE_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class SglAlignedTritonGlmMoeProvider(MoeProvider):
    name = "sgl_aligned_triton_glm"
    gate_up_order = "gate_up"
    fuse_gate_up_swiglu = False

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        activation = _silu_activation_support(spec)
        if activation is not None:
            return activation
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported("requires CUDA SM90")
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported("requires BF16 activations")
        if spec.weight_dtype != torch.bfloat16 or spec.block_shape is not None:
            return SupportResult.unsupported("requires unquantized BF16 expert weights")
        from sparseengine.kernels.external.sgl.moe import sgl_moe_alignment_support

        supported, reason = sgl_moe_alignment_support()
        return SupportResult.yes() if supported else SupportResult.unsupported(reason)

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
            raise RuntimeError("SGL-aligned BF16 MoE does not accept expert scales.")
        from sparseengine.kernels.triton.moe import fused_moe

        alignment_impl = None
        num_tokens = int(hidden_states.shape[0])
        if (
            num_tokens <= 64
            and (
                int(spec.ep_size) > 1
                or num_tokens * int(spec.top_k) * 4 > int(spec.num_experts)
            )
        ):
            alignment_impl = _sgl_moe_align_block_size
        return fused_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
            _fuse_gate_up_swiglu=self.fuse_gate_up_swiglu,
            alignment_impl=alignment_impl,
        )


@MOE_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class SglAlignedTritonGlmFusedMoeProvider(SglAlignedTritonGlmMoeProvider):
    name = "sgl_aligned_triton_glm_fused"
    fuse_gate_up_swiglu = True


@MOE_REGISTRY.register_profile
class GlmH100Bf16DecodeMoeDispatchPlan(MoeDispatchPlan):
    """Prepared H100 GLM decode fusion with the established prefill path."""

    name = "glm_h100_bf16_decode_dispatch_plan"
    gate_up_order = "gate_up"
    MAX_FUSED_TOKENS = 32
    PROFILED_SHAPES = frozenset(
        {
            (64, 64, 2048, 768, 4, 2, 1, "biased_sigmoid"),
            (65, 65, 2048, 768, 5, 2, 1, "biased_sigmoid"),
            (64, 32, 2048, 1536, 4, 1, 2, "biased_sigmoid"),
        }
    )

    @classmethod
    def atomic_provider_names(cls, spec: MoeOpSpec) -> tuple[str, ...]:
        del spec
        return (
            "sgl_aligned_triton_glm_fused",
            "sgl_aligned_triton_glm",
        )

    @classmethod
    def matches(cls, spec: MoeOpSpec, caps: DeviceCaps) -> ProfileMatch:
        actual = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
            spec.routing_method,
        )
        if not device_name_contains(caps.device_name, "H100"):
            return ProfileMatch.no("requires profiled H100 hardware")
        if actual not in cls.PROFILED_SHAPES:
            return ProfileMatch.no(
                f"requires a profiled GLM H100 BF16 shape, got {actual}"
            )
        return ProfileMatch.yes("matched GLM H100 BF16 decode fusion profile")

    def _build_routes(self, spec: MoeOpSpec) -> tuple[MoeDispatchRoute, ...]:
        if spec.ep_size == 2:
            # The EP-local 1536-wide experts favor fusion only at tiny batches.
            # Larger decode and prefill retain the established unfused path.
            return (
                MoeDispatchRoute(
                    0, 2, SglAlignedTritonGlmMoeProvider(), "triton_routed_gemm_silu",
                ),
                MoeDispatchRoute(
                    3, 4, SglAlignedTritonGlmFusedMoeProvider(),
                    "triton_routed_gate_up_swiglu",
                ),
                MoeDispatchRoute(
                    5, None, SglAlignedTritonGlmMoeProvider(), "triton_routed_gemm_silu",
                ),
            )
        del spec
        return (
            MoeDispatchRoute(
                min_tokens=0,
                max_tokens=self.MAX_FUSED_TOKENS,
                provider=SglAlignedTritonGlmFusedMoeProvider(),
                kernel_path="triton_routed_gate_up_swiglu",
            ),
            MoeDispatchRoute(
                min_tokens=self.MAX_FUSED_TOKENS + 1,
                max_tokens=None,
                provider=SglAlignedTritonGlmMoeProvider(),
                kernel_path="triton_routed_gemm_silu",
            ),
        )


@MOE_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonMinimaxM2FusedMoeProvider(MoeProvider):
    name = "triton_minimax_m2_fused"
    gate_up_order = "gate_up"

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        activation = _silu_activation_support(spec)
        if activation is not None:
            return activation
        if spec.routing_method != "biased_sigmoid":
            return SupportResult.unsupported("requires biased-sigmoid routing")
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if not caps.supports_bfloat16:
            return SupportResult.unsupported("device does not support BF16")
        if not caps.supports_native_fp8:
            return SupportResult.unsupported("device does not provide native FP8 tensor cores")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if spec.weight_dtype != torch.float8_e4m3fn:
            return SupportResult.unsupported(
                f"requires FP8 E4M3 weights, got {spec.weight_dtype}"
            )
        if spec.block_shape != (128, 128):
            return SupportResult.unsupported(
                f"requires block_shape=(128, 128), got {spec.block_shape}"
            )
        if spec.scale_dtype != torch.float32:
            return SupportResult.unsupported(
                f"requires FP32 expert scales, got {spec.scale_dtype}"
            )
        if spec.tp_size not in {1, 2, 4}:
            return SupportResult.unsupported(
                f"requires MoE TP size 1, 2, or 4, got {spec.tp_size}"
            )
        expected_shape = (
            256,
            256 // spec.ep_size,
            3072,
            1536 // spec.tp_size,
            8,
        )
        actual_shape = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
        )
        if actual_shape != expected_shape:
            return SupportResult.unsupported(
                "requires MiniMax M2.7 expert shape "
                f"{expected_shape}, got {actual_shape}"
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
        if w13_scale_inv is None or w2_scale_inv is None:
            raise RuntimeError("MiniMax M2.7 fused MoE requires expert scales.")
        from sparseengine.kernels.triton.minimax_m2_moe import (
            fused_minimax_m2_moe_fp8,
        )

        return fused_minimax_m2_moe_fp8(
            hidden_states,
            w13_weight,
            w2_weight,
            w13_scale_inv,
            w2_scale_inv,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
        )


@MOE_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferCutlassFp8MoeProvider(MoeProvider):
    name = "flashinfer_cutlass_fp8_sm90"
    gate_up_order = "up_gate"
    workspace_lane = "flashinfer_cutlass_moe"

    def __init__(self) -> None:
        self._workspace_lease = None
        self._workspace_num_tokens = 0
        self._workspace_contract: tuple[object, ...] | None = None

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        activation = _silu_activation_support(spec)
        if activation is not None:
            return activation
        if spec.weight_dtype != torch.float8_e4m3fn:
            return SupportResult.unsupported(f"requires FP8 E4M3 weights, got {spec.weight_dtype}")
        if spec.block_shape != (128, 128):
            return SupportResult.unsupported(f"requires block_shape=(128, 128), got {spec.block_shape}")
        if spec.hidden_size % 128 or spec.intermediate_size % 128:
            return SupportResult.unsupported("FP8 hidden/intermediate sizes must be 128-aligned")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if not caps.supports_native_fp8:
            return SupportResult.unsupported(
                "device does not provide native FP8 tensor cores"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.unsupported(
                "requires CUDA runtime >= 12.8, "
                f"got {caps.runtime_version or 'unknown'}"
            )
        from sparseengine.kernels.external.flashinfer.moe import (
            flashinfer_cutlass_fp8_moe_support,
        )

        supported, reason = flashinfer_cutlass_fp8_moe_support()
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "workspace": "shared_reusable_byte_buffer",
            "workspace_lane": self.workspace_lane,
            "workspace_sized_by": "flashinfer.cutlass_fused_moe_workspace_size",
        }

    def prepare(
        self,
        spec: MoeOpSpec,
        *,
        device: torch.device,
        tp_rank: int,
        ep_rank: int,
        max_num_tokens: int | None = None,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            return
        tp_rank, ep_rank = int(tp_rank), int(ep_rank)
        if not 0 <= tp_rank < int(spec.tp_size):
            raise ValueError(
                f"MoE tp_rank must be in [0, {spec.tp_size}), got {tp_rank}."
            )
        if not 0 <= ep_rank < int(spec.ep_size):
            raise ValueError(
                f"MoE ep_rank must be in [0, {spec.ep_size}), got {ep_rank}."
            )
        contract = (spec, device, tp_rank, ep_rank)
        if self._workspace_contract is not None and self._workspace_contract != contract:
            raise RuntimeError(
                "FlashInfer CUTLASS MoE provider cannot change its prepared "
                "shape, device, or TP/EP rank contract."
            )
        self._workspace_contract = contract
        requested_max = (
            int(spec.max_num_tokens)
            if max_num_tokens is None
            else int(max_num_tokens)
        )
        if requested_max <= 0:
            raise ValueError(
                f"MoE prepared max_num_tokens must be positive, got {requested_max}."
            )
        self._reserve_workspace(
            spec,
            device,
            tp_rank,
            ep_rank,
            min(int(spec.max_num_tokens), requested_max),
        )

    def _reserve_workspace(
        self,
        spec: MoeOpSpec,
        device: torch.device,
        tp_rank: int,
        ep_rank: int,
        num_tokens: int,
    ) -> None:
        num_tokens = int(num_tokens)
        if num_tokens <= self._workspace_num_tokens:
            return
        from sparseengine.kernels.external.flashinfer.moe import (
            flashinfer_cutlass_fused_moe_workspace_size,
        )
        from sparseengine.operators.workspace import get_workspace_manager

        required_bytes = flashinfer_cutlass_fused_moe_workspace_size(
            max_num_tokens=num_tokens,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_experts=spec.num_experts,
            top_k=spec.top_k,
            activation_dtype=spec.activation_dtype,
            weight_dtype=spec.weight_dtype,
            tp_size=spec.tp_size,
            tp_rank=tp_rank,
            ep_size=spec.ep_size,
            ep_rank=ep_rank,
            device=device,
        )
        manager = get_workspace_manager(device, create=True)
        if self._workspace_lease is None:
            self._workspace_lease = manager.reserve_bytes(
                required_bytes,
                label=self.name,
                lane=self.workspace_lane,
            )
        else:
            self._workspace_lease.ensure_bytes(required_bytes)
        self._workspace_num_tokens = num_tokens

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
        del local_expert_start
        if w13_scale_inv is None or w2_scale_inv is None:
            raise RuntimeError("FlashInfer FP8 MoE requires expert scales.")
        from sparseengine.kernels.external.flashinfer.moe import (
            flashinfer_cutlass_fused_moe,
        )

        device = hidden_states.device
        contract = (spec, device, int(tp_rank), int(ep_rank))
        if self._workspace_contract is None:
            self._workspace_contract = contract
        elif self._workspace_contract != contract:
            raise RuntimeError(
                "FlashInfer CUTLASS MoE execution does not match its prepared "
                "shape, device, or TP/EP rank contract."
            )
        self._reserve_workspace(
            spec,
            device,
            int(tp_rank),
            int(ep_rank),
            int(hidden_states.shape[0]),
        )
        if self._workspace_lease is None:
            raise RuntimeError("FlashInfer CUTLASS MoE workspace was not prepared.")

        output = torch.empty_like(hidden_states)
        flashinfer_cutlass_fused_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            w13_weight,
            w2_weight,
            w13_scale_inv,
            w2_scale_inv,
            tp_size=int(spec.tp_size),
            tp_rank=int(tp_rank),
            ep_size=int(spec.ep_size),
            ep_rank=int(ep_rank),
            output=output,
            workspace_buffer=self._workspace_lease.buffer,
        )
        return output


@MOE_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class TritonHopperFusedMoeProvider(MoeProvider):
    name = "triton_hopper_fused"
    gate_up_order = "gate_up"
    PROFILED_SHAPES = (
        (128, 64, 2048, 384, 8, 2, 2),
        (256, 256, 2048, 512, 8, 1, 1),
        (256, 256, 2048, 256, 8, 2, 1),
        (256, 128, 2048, 512, 8, 1, 2),
    )

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        activation = _silu_activation_support(spec)
        if activation is not None:
            return activation
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if spec.weight_dtype != torch.bfloat16 or spec.block_shape is not None:
            return SupportResult.unsupported("requires unquantized BF16 expert weights")
        if not caps.supports_bfloat16:
            return SupportResult.unsupported("device does not support BF16")
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
            raise RuntimeError("Fused Hopper BF16 MoE does not accept expert scales.")
        from sparseengine.kernels.triton.moe import fused_moe_gate_up_swiglu

        return fused_moe_gate_up_swiglu(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
        )


@MOE_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class SglDerivedTritonMoeProvider(MoeProvider):
    """Repository-owned BF16 Triton MoE port using SGL expert alignment."""

    name = "sgl_derived_triton_bf16"
    gate_up_order = "gate_up"
    PROFILED_SHAPES = frozenset(
        {
            (128, 128, 2048, 768, 8, 1, 1),
            (128, 128, 2048, 384, 8, 2, 1),
        }
    )

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        activation = _silu_activation_support(spec)
        if activation is not None:
            return activation
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} "
                f"{caps.compute_capability}"
            )
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if spec.weight_dtype != spec.activation_dtype or spec.block_shape is not None:
            return SupportResult.unsupported(
                "requires unquantized expert weights matching the activation dtype"
            )
        if not caps.supports_bfloat16:
            return SupportResult.unsupported("device does not support BF16")
        from sparseengine.kernels.external.sgl.moe import sgl_moe_alignment_support

        supported, reason = sgl_moe_alignment_support()
        if not supported:
            return SupportResult.unsupported(reason)
        return SupportResult.yes(reason)

    def binding_metadata(self) -> dict[str, object]:
        from sparseengine.kernels.triton.sgl_fused_moe import (
            sgl_moe_profile_metadata,
        )

        return {**super().binding_metadata(), **sgl_moe_profile_metadata()}

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
            raise RuntimeError("SGL-derived BF16 Triton MoE does not accept scales.")
        from sparseengine.kernels.triton.sgl_fused_moe import sgl_fused_moe

        return sgl_fused_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
            alignment_impl=_sgl_moe_align_block_size,
        )


@MOE_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class SglTritonGlmMoeProvider(SglDerivedTritonMoeProvider):
    """Exact-profile GLM BF16 provider using the SGL Triton pipeline."""

    name = "sgl_triton_glm_bf16"

    def binding_metadata(self) -> dict[str, object]:
        from sparseengine.kernels.triton.sgl_fused_moe import (
            sgl_glm47_moe_profile_metadata,
        )

        return {
            **MoeProvider.binding_metadata(self),
            **sgl_glm47_moe_profile_metadata(),
        }


@MOE_REGISTRY.register_profile
class Qwen3Bf16MoeDispatchPlan(MoeDispatchPlan):
    """Prepared Qwen3 BF16 token ranges over two atomic Triton providers."""

    name = "qwen3_bf16_dispatch_plan"
    gate_up_order = "gate_up"
    MIN_SGL_TOKENS_BY_TP_SIZE = {1: 64, 2: 64}

    @classmethod
    def atomic_provider_names(cls, spec: MoeOpSpec) -> tuple[str, ...]:
        del spec
        return ("sgl_derived_triton_bf16", "triton")

    @classmethod
    def matches(cls, spec: MoeOpSpec, caps: DeviceCaps) -> ProfileMatch:
        actual = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
        )
        if not device_name_contains(caps.device_name, "H100"):
            return ProfileMatch.no("requires profiled H100 hardware")
        if actual not in SglDerivedTritonMoeProvider.PROFILED_SHAPES:
            return ProfileMatch.no(
                "requires a profiled Qwen3 BF16 shape in "
                f"{sorted(SglDerivedTritonMoeProvider.PROFILED_SHAPES)}, "
                f"got {actual}"
            )
        from sparseengine.kernels.triton.sgl_fused_moe import (
            sgl_moe_profile_support,
        )

        profile_supported, profile_reason = sgl_moe_profile_support()
        if not profile_supported:
            return ProfileMatch.no(profile_reason)
        return ProfileMatch.yes("matched Qwen3 BF16 token dispatch profile")

    def _build_routes(self, spec: MoeOpSpec) -> tuple[MoeDispatchRoute, ...]:
        min_sgl_tokens = self.MIN_SGL_TOKENS_BY_TP_SIZE[int(spec.tp_size)]
        return (
            MoeDispatchRoute(
                min_tokens=0,
                max_tokens=min_sgl_tokens - 1,
                provider=TritonMoeProvider(),
                kernel_path="triton_fused_moe",
            ),
            MoeDispatchRoute(
                min_tokens=min_sgl_tokens,
                max_tokens=None,
                provider=SglDerivedTritonMoeProvider(),
                kernel_path="sgl_fused_moe",
            ),
        )


@MOE_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class H20Qwen36FusedMoeProvider(TritonHopperFusedMoeProvider):
    name = "h20_qwen36_fused_bf16"
    PROFILED_SHAPES = (
        (256, 256, 2048, 512, 8, 1, 1),
        (256, 256, 2048, 256, 8, 2, 1),
        (256, 128, 2048, 512, 8, 1, 2),
    )


class _SingleAtomicMoeProfile:
    atomic_provider_name = ""

    @classmethod
    def atomic_provider_names(cls, spec: MoeOpSpec) -> tuple[str, ...]:
        del spec
        return (cls.atomic_provider_name,)

    @classmethod
    def bind(cls, spec: MoeOpSpec, caps: DeviceCaps, **kwargs):
        del spec, caps
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        provider_type = MOE_REGISTRY.atomic_registry.registration(
            cls.atomic_provider_name
        ).provider
        return provider_type()


@MOE_REGISTRY.register_profile
class SglAlignedTritonGlmMoeProfile(_SingleAtomicMoeProfile):
    name = "sgl_aligned_triton_glm_bf16_profile"
    atomic_provider_name = "sgl_aligned_triton_glm"

    @classmethod
    def matches(cls, spec: MoeOpSpec, caps: DeviceCaps) -> ProfileMatch:
        expected = {
            (64, 64, 2048, 768, 4, 2, 1, "biased_sigmoid"),
            (65, 65, 2048, 768, 5, 2, 1, "biased_sigmoid"),
            (64, 32, 2048, 1536, 4, 1, 2, "biased_sigmoid"),
        }
        actual = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
            spec.routing_method,
        )
        if not any(
            device_name_contains(caps.device_name, name)
            for name in ("H100", "H20")
        ):
            return ProfileMatch.no("requires profiled H100 or H20 hardware")
        if actual not in expected:
            return ProfileMatch.no(
                f"requires a profiled GLM TP2 MoE shape {expected}, got {actual}"
            )
        return ProfileMatch.yes("matched SGL-aligned GLM BF16 profile")


@MOE_REGISTRY.register_profile
class SglTritonGlmTp1MoeProfile(_SingleAtomicMoeProfile):
    name = "sgl_triton_glm_tp1_bf16_profile"
    atomic_provider_name = "sgl_triton_glm_bf16"

    @classmethod
    def matches(cls, spec: MoeOpSpec, caps: DeviceCaps) -> ProfileMatch:
        expected = (65, 65, 2048, 1536, 5, 1, 1, "biased_sigmoid")
        actual = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
            spec.routing_method,
        )
        if not device_name_contains(caps.device_name, "H100"):
            return ProfileMatch.no("requires profiled H100 hardware")
        if actual != expected:
            return ProfileMatch.no(
                f"requires the GLM TP1 fused-shared shape {expected}, got {actual}"
            )
        from sparseengine.kernels.triton.sgl_fused_moe import (
            sgl_glm47_moe_profile_support,
        )

        supported, reason = sgl_glm47_moe_profile_support()
        return ProfileMatch.yes(reason) if supported else ProfileMatch.no(reason)


@MOE_REGISTRY.register_profile
class TritonMinimaxM2MoeProfile(_SingleAtomicMoeProfile):
    name = "triton_minimax_m2_profile"
    atomic_provider_name = "triton_minimax_m2_fused"
    profiled_shape = (256, 256, 3072, 384, 8, 4, 1)

    @classmethod
    def matches(cls, spec: MoeOpSpec, caps: DeviceCaps) -> ProfileMatch:
        if not device_name_contains(caps.device_name, "H100"):
            return ProfileMatch.no("requires profiled H100 hardware")
        actual = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
        )
        if actual != cls.profiled_shape:
            return ProfileMatch.no(
                f"requires profiled MiniMax M2 TP4/EP1 shape {cls.profiled_shape}, "
                f"got {actual}"
            )
        return ProfileMatch.yes("matched MiniMax M2 FP8 MoE profile")


@MOE_REGISTRY.register_profile
class HopperFusedBf16MoeProfile(_SingleAtomicMoeProfile):
    name = "hopper_fused_bf16_profile"
    atomic_provider_name = "triton_hopper_fused"
    profiled_device_name = "H100"
    profiled_shapes = TritonHopperFusedMoeProvider.PROFILED_SHAPES

    @classmethod
    def matches(cls, spec: MoeOpSpec, caps: DeviceCaps) -> ProfileMatch:
        actual = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
        )
        if not device_name_contains(caps.device_name, cls.profiled_device_name):
            return ProfileMatch.no(
                f"requires profiled {cls.profiled_device_name} hardware, "
                f"got {caps.device_name}"
            )
        if actual not in cls.profiled_shapes:
            return ProfileMatch.no(
                f"requires a profiled MoE shape in {cls.profiled_shapes}, got {actual}"
            )
        return ProfileMatch.yes("matched Hopper BF16 fused MoE profile")


@MOE_REGISTRY.register_profile
class H20Qwen36FusedBf16MoeProfile(HopperFusedBf16MoeProfile):
    name = "h20_qwen36_fused_bf16_profile"
    atomic_provider_name = "h20_qwen36_fused_bf16"
    profiled_device_name = "H20"
    profiled_shapes = H20Qwen36FusedMoeProvider.PROFILED_SHAPES


@MOE_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonMoeProvider(MoeProvider):
    name = "triton"
    gate_up_order = "gate_up"
    _prefill_alignment = None
    _prefill_alignment_reason = "not prepared"

    @classmethod
    def bind(cls, spec, caps, **kwargs):
        provider = cls(**kwargs)
        provider.prepare_prefill_alignment(spec, caps)
        return provider

    def prepare_prefill_alignment(self, spec, caps) -> None:
        self._prefill_alignment = None
        if (caps.platform != PlatformEnum.CUDA
                or spec.weight_dtype != spec.activation_dtype
                or spec.activation_dtype not in (torch.float16, torch.bfloat16)):
            self._prefill_alignment_reason = "requires unquantized CUDA experts"
            return
        from sparseengine.kernels.external.sgl.moe import sgl_moe_alignment_support

        supported, reason = sgl_moe_alignment_support()
        self._prefill_alignment_reason = reason
        if supported:
            self._prefill_alignment = _prefill_moe_align_block_size

    def binding_metadata(self):
        return {
            **super().binding_metadata(),
            "prefill_alignment": "sgl_with_naive_small_inputs" if self._prefill_alignment else "triton",
            "prefill_alignment_reason": self._prefill_alignment_reason,
        }

    @classmethod
    def supports(cls, spec: MoeOpSpec, caps: DeviceCaps) -> SupportResult:
        activation = _silu_activation_support(spec)
        if activation is not None:
            return activation
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.activation_dtype not in (torch.bfloat16, torch.float16):
            return SupportResult.unsupported(
                f"requires BF16 or FP16 activations, got {spec.activation_dtype}"
            )
        if spec.weight_dtype == torch.float8_e4m3fn:
            if spec.block_shape not in (None, (128, 128)):
                return SupportResult.unsupported(
                    f"FP8 requires tensor scales or block_shape=(128, 128), got {spec.block_shape}"
                )
            if not caps.supports_native_fp8:
                return SupportResult.unsupported("device does not provide native FP8 tensor cores")
            if spec.hidden_size % 128 or spec.intermediate_size % 128:
                return SupportResult.unsupported("FP8 hidden/intermediate sizes must be 128-aligned")
            if spec.block_shape is None:
                from sparseengine.kernels.external.sgl.fp8_linear import tensor_fp8_ops
                tensor_fp8_ops()
                return SupportResult.yes("per-token activations and per-logical-tensor expert scales")
            from sparseengine.kernels.external.sgl.moe import (
                sgl_fp8_group_quantization_support,
            )

            supported, reason = sgl_fp8_group_quantization_support()
            return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)
        if spec.weight_dtype != spec.activation_dtype:
            return SupportResult.unsupported(
                "unquantized weights must match the activation dtype"
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
        if spec.weight_dtype == torch.float8_e4m3fn:
            if w13_scale_inv is None or w2_scale_inv is None:
                raise RuntimeError("Triton FP8 MoE requires expert scales.")
            from sparseengine.kernels.triton.moe import fused_moe_fp8

            return fused_moe_fp8(
                hidden_states,
                w13_weight,
                w2_weight,
                w13_scale_inv,
                w2_scale_inv,
                topk_ids,
                topk_weights,
                num_experts=spec.num_experts,
                local_expert_start=local_expert_start,
                gate_up_order=self.gate_up_order,
                tensor_scales=spec.block_shape is None,
            )
        from sparseengine.kernels.triton.moe import fused_moe
        from sparseengine.utils.context import get_context

        return fused_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            num_experts=spec.num_experts,
            local_expert_start=local_expert_start,
            alignment_impl=self._prefill_alignment if get_context().is_prefill else None,
        )


@MOE_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class TritonUpGateFp8MoeProvider(TritonMoeProvider):
    """Atomic Triton FP8 MoE over FlashInfer-compatible packed weights."""

    name = "triton_fp8_up_gate"
    gate_up_order = "up_gate"


@MOE_REGISTRY.register_profile
class Qwen3Fp8MoeDispatchPlan(MoeDispatchPlan):
    """Prepared Qwen3 FP8 token ranges over layout-compatible providers."""

    name = "qwen3_fp8_dispatch_plan"
    gate_up_order = "up_gate"
    FLASHINFER_MAX_TOKENS = 128
    PROFILED_SHAPES = frozenset(
        {
            (128, 128, 2048, 768, 8, 1, 1),
            (128, 128, 2048, 384, 8, 2, 1),
        }
    )

    @classmethod
    def atomic_provider_names(cls, spec: MoeOpSpec) -> tuple[str, ...]:
        names = ["triton_fp8_up_gate"]
        if spec.tp_size == 1:
            names.append("flashinfer_cutlass_fp8_sm90")
        return tuple(names)

    @classmethod
    def matches(cls, spec: MoeOpSpec, caps: DeviceCaps) -> ProfileMatch:
        if not device_name_contains(caps.device_name, "H100"):
            return ProfileMatch.no("requires profiled H100 hardware")
        if spec.weight_dtype != torch.float8_e4m3fn:
            return ProfileMatch.no("requires FP8 E4M3 weights")
        if spec.block_shape != (128, 128):
            return ProfileMatch.no("requires block_shape=(128, 128)")
        actual_shape = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
        )
        if actual_shape not in cls.PROFILED_SHAPES:
            return ProfileMatch.no(
                f"requires a profiled Qwen3 FP8 shape, got {actual_shape}"
            )
        return ProfileMatch.yes("matched Qwen3 FP8 token dispatch profile")

    def _build_routes(self, spec: MoeOpSpec) -> tuple[MoeDispatchRoute, ...]:
        from importlib.metadata import version

        triton = TritonUpGateFp8MoeProvider()
        # Keep the existing profile's toolchain and graph domain. Within it,
        # use one packed-weight route for both single- and multi-request work.
        profiled_graph = (
            spec.tp_size == 1
            and spec.cuda_graph
            and spec.activation_dtype == torch.bfloat16
            and spec.scale_dtype == torch.float32
            and torch.__version__ == "2.11.0+cu130"
            and version("triton") == "3.6.0"
            and version("flashinfer-python") == "0.6.17"
        )
        if spec.tp_size != 1 or profiled_graph:
            return (
                MoeDispatchRoute(
                    min_tokens=0,
                    max_tokens=None,
                    provider=triton,
                    kernel_path=triton.name,
                ),
            )
        flashinfer = FlashInferCutlassFp8MoeProvider()
        return (
            MoeDispatchRoute(
                min_tokens=0,
                max_tokens=self.FLASHINFER_MAX_TOKENS,
                provider=flashinfer,
                kernel_path=flashinfer.name,
            ),
            MoeDispatchRoute(
                min_tokens=self.FLASHINFER_MAX_TOKENS + 1,
                max_tokens=None,
                provider=triton,
                kernel_path=triton.name,
            ),
        )


@MOE_REGISTRY.register_profile
class HopperQwen36Fp8MoeDispatchPlan(MoeDispatchPlan):
    """Prepared Qwen3.6 FP8 token ranges over layout-compatible providers."""

    name = "hopper_qwen36_fp8_dispatch_plan"
    gate_up_order = "up_gate"
    PROFILED_DEVICE_NAME = "H100"
    PROFILED_SHAPES = frozenset(
        {
            (256, 256, 2048, 512, 8, 1, 1),
            (256, 128, 2048, 512, 8, 1, 2),
        }
    )
    TRITON_MAX_TOKENS_BY_EP_SIZE = {1: 8, 2: 4}

    @classmethod
    def atomic_provider_names(cls, spec: MoeOpSpec) -> tuple[str, ...]:
        del spec
        return ("triton_fp8_up_gate", "flashinfer_cutlass_fp8_sm90")

    @classmethod
    def matches(cls, spec: MoeOpSpec, caps: DeviceCaps) -> ProfileMatch:
        if spec.cuda_graph and not caps.supports_graph_capture:
            return ProfileMatch.no("device does not support CUDA Graph capture")
        if not device_name_contains(caps.device_name, cls.PROFILED_DEVICE_NAME):
            return ProfileMatch.no(
                f"requires profiled {cls.PROFILED_DEVICE_NAME} hardware, "
                f"got {caps.device_name}"
            )
        actual_shape = (
            spec.num_experts,
            spec.num_local_experts,
            spec.hidden_size,
            spec.intermediate_size,
            spec.top_k,
            spec.tp_size,
            spec.ep_size,
        )
        if actual_shape not in cls.PROFILED_SHAPES:
            return ProfileMatch.no(
                "requires profiled Qwen3.6 TP1/EP1 or "
                "global-TP2/MoE-TP1xEP2 shape "
                f"{sorted(cls.PROFILED_SHAPES)}, "
                f"got {actual_shape}"
            )
        return ProfileMatch.yes("matched Qwen3.6 FP8 token dispatch profile")

    def _build_routes(self, spec: MoeOpSpec) -> tuple[MoeDispatchRoute, ...]:
        triton_max_tokens = self.TRITON_MAX_TOKENS_BY_EP_SIZE[int(spec.ep_size)]
        triton = TritonUpGateFp8MoeProvider()
        flashinfer = FlashInferCutlassFp8MoeProvider()
        return (
            MoeDispatchRoute(
                min_tokens=0,
                max_tokens=triton_max_tokens,
                provider=triton,
                kernel_path=triton.name,
            ),
            MoeDispatchRoute(
                min_tokens=triton_max_tokens + 1,
                max_tokens=None,
                provider=flashinfer,
                kernel_path=flashinfer.name,
            ),
        )


@MOE_REGISTRY.register_profile
class H20Qwen36Fp8MoeDispatchPlan(HopperQwen36Fp8MoeDispatchPlan):
    name = "h20_qwen36_fp8_dispatch_plan"
    PROFILED_DEVICE_NAME = "H20"
    TRITON_MAX_TOKENS_BY_EP_SIZE = {1: 8, 2: 1}


@MOE_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferCutlassMxfp4MoeProvider(MoeProvider):
    """SM90 W4A16 experts with checkpoint UE8M0 groups and clipped SwiGLU."""

    name = "flashinfer_cutlass_mxfp4_sm90"
    gate_up_order = "up_gate"

    @property
    def weight_layout_id(self):
        return "sm90_interleaved_mxfp4_up_gate_ue8m0_g32"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported("MXFP4 W4A16 requires CUDA SM90")
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.unsupported("FlashInfer W4A16 requires CUDA >= 12.8")
        if spec.weight_dtype != torch.uint8 or spec.block_shape != (1, 32):
            return SupportResult.unsupported("requires packed E2M1 weights and per-32 UE8M0 scales")
        if spec.scale_dtype not in (torch.uint8, torch.float8_e8m0fnu):
            return SupportResult.unsupported("requires UE8M0 scale bytes")
        if spec.activation_dtype != torch.bfloat16 or spec.activation != "clipped_silu":
            return SupportResult.unsupported("requires BF16 input and clipped SwiGLU")
        if spec.hidden_size % 128 or spec.intermediate_size % 128 or spec.tp_size != 1:
            return SupportResult.unsupported("requires 128-aligned dimensions and MoE TP1")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph")
        from sparseengine.kernels.external.flashinfer.moe import flashinfer_cutlass_fp8_moe_support
        supported, reason = flashinfer_cutlass_fp8_moe_support()
        if not supported:
            return SupportResult.unsupported(reason)
        import flashinfer.fused_moe as fi
        for symbol in ("interleave_moe_weights_for_sm90_mixed_gemm",
                       "interleave_moe_scales_for_sm90_mixed_gemm"):
            if not callable(getattr(fi, symbol, None)):
                return SupportResult.dependency_broken(f"FlashInfer lacks {symbol}")
        return SupportResult.yes("FlashInfer SM90 MXFP4 W4A16 with fused clipped SwiGLU")

    def prepare(self, spec, *, device, tp_rank, ep_rank, max_num_tokens=None):
        import flashinfer.fused_moe as fi
        from sparseengine.operators.workspace import get_workspace_manager

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("MXFP4 experts must be prepared on CUDA")
        if tp_rank != 0 or not 0 <= ep_rank < spec.ep_size:
            raise ValueError("MXFP4 prepared rank does not match the expert topology")
        if hasattr(self, "spec"):
            raise RuntimeError("MXFP4 provider must be prepared once before model execution")
        self.spec, self.device, self.ep_rank = spec, device, ep_rank
        self.max_num_tokens = min(spec.max_num_tokens, max_num_tokens or spec.max_num_tokens)
        size = fi.cutlass_fused_moe_workspace_size(
            self.max_num_tokens, spec.hidden_size, spec.intermediate_size,
            spec.num_experts, spec.top_k, x_dtype=torch.bfloat16,
            weight_dtype=torch.uint8, ep_size=spec.ep_size, ep_rank=ep_rank,
            use_w4_group_scaling=True, use_packed_weights=False,
            use_fused_finalize=False, device=device,
        )
        self.workspace = get_workspace_manager(device, create=True).reserve_bytes(
            size, label=self.name, lane="flashinfer_cutlass_mxfp4_moe",
        )
        self.alpha = torch.ones(spec.num_local_experts, device=device, dtype=torch.float32)
        self.beta = torch.zeros_like(self.alpha)
        self.limit = torch.full_like(self.alpha, spec.activation_limit)
        self._run = fi.cutlass_fused_moe

    def prepare_weights(self, w13, w2, scale13, scale2):
        """Reorder loaded storage once in place, avoiding a second model copy."""
        import flashinfer.fused_moe as fi

        if not hasattr(self, "spec") or hasattr(self, "weights"):
            raise RuntimeError("Prepare MXFP4 topology before loading weights exactly once")
        spec = self.spec
        shapes = ((spec.num_local_experts, 2 * spec.intermediate_size, spec.hidden_size // 2),
                  (spec.num_local_experts, spec.hidden_size, spec.intermediate_size // 2),
                  (spec.num_local_experts, 2 * spec.intermediate_size, spec.hidden_size // 32),
                  (spec.num_local_experts, spec.hidden_size, spec.intermediate_size // 32))
        for tensor, shape in zip((w13, w2, scale13, scale2), shapes):
            if tuple(tensor.shape) != shape or tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError("MXFP4 loaded weight/scale storage differs from its prepared contract")
        if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
            raise TypeError("MXFP4 weights must contain packed E2M1 bytes")
        if scale13.dtype != spec.scale_dtype or scale2.dtype != spec.scale_dtype:
            raise TypeError("MXFP4 scales differ from the bound UE8M0 dtype")
        for weight in (w13, w2):
            weight.copy_(fi.interleave_moe_weights_for_sm90_mixed_gemm(weight))
        folded = []
        for scales in (scale13, scale2):
            raw = scales.view(torch.uint8)
            prepared = fi.interleave_moe_scales_for_sm90_mixed_gemm(raw, group_size=32)
            raw.copy_(prepared.reshape_as(raw))
            folded.append(raw.view(prepared.shape))
        self.weights = (w13, w2)
        self.scales = folded

    def run(self, spec, hidden_states, topk_ids, topk_weights, w13_weight,
            w2_weight, w13_scale_inv, w2_scale_inv, *, local_expert_start,
            tp_rank, ep_rank):
        if not hasattr(self, "weights"):
            raise RuntimeError("MXFP4 physical weights must be prepared before execution")
        if (spec != self.spec or tp_rank != 0 or ep_rank != self.ep_rank or
                local_expert_start != self.ep_rank * spec.num_local_experts):
            raise ValueError("MXFP4 execution differs from the prepared expert topology")
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError("MXFP4 batch exceeds the prepared workspace capacity")
        if hidden_states.dtype != torch.bfloat16 or hidden_states.device != self.device:
            raise ValueError("MXFP4 input differs from the prepared activation contract")
        if any(x.data_ptr() != y.data_ptr() for x, y in
               zip((w13_weight, w2_weight), self.weights)):
            raise ValueError("MXFP4 execution must use prepared weight storage")
        output = torch.empty_like(hidden_states)
        if not hidden_states.shape[0]:
            return output
        self._run(hidden_states, topk_ids.to(torch.int32), topk_weights.float(),
                  *self.weights, torch.bfloat16, self.scales,
                  swiglu_alpha=self.alpha, swiglu_beta=self.beta, swiglu_limit=self.limit,
                  ep_size=spec.ep_size, ep_rank=self.ep_rank, output=output,
                  use_w4_group_scaling=True, use_packed_weights=False,
                  use_fused_finalize=False, enable_pdl=False,
                  workspace_buffer=self.workspace.buffer)
        return output


def resolve_moe_provider(
    spec: MoeOpSpec,
    *,
    device_index: int | None = None,
    force_atomic_provider: str | None = None,
) -> MoeProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(MOE_REGISTRY).resolve(
        spec,
        caps,
        force_atomic_provider=force_atomic_provider,
    ).provider
