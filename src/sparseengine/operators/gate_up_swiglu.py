from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

import sparseengine.platforms as platforms
from sparseengine.platforms import device_runtime
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProfileMatch,
    ProviderRole,
    SupportResult,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.utils.device_name import device_name_contains


@dataclass(frozen=True)
class GateUpSwiGLUOpSpec:
    hidden_size: int
    intermediate_size: int
    tp_size: int
    activation_dtype: torch.dtype
    weight_dtype: torch.dtype
    cuda_graph: bool

    def __post_init__(self) -> None:
        if min(self.hidden_size, self.intermediate_size, self.tp_size) <= 0:
            raise ValueError("Gate/up SwiGLU dimensions and TP size must be positive.")
        if self.intermediate_size % self.tp_size:
            raise ValueError(
                "Gate/up SwiGLU intermediate size must be divisible by TP size."
            )
        if not (
            self.activation_dtype.is_floating_point
            and self.weight_dtype.is_floating_point
        ):
            raise TypeError(
                "Gate/up SwiGLU activations and weights must be floating point."
            )


class GateUpSwiGLUProvider:
    name = ""

    def binding_metadata(self) -> dict[str, object]:
        return {"implementation_kind": "atomic_provider"}

    def run(
        self,
        spec: GateUpSwiGLUOpSpec,
        inputs: torch.Tensor,
        projection,
    ) -> torch.Tensor:
        raise NotImplementedError


GATE_UP_SWIGLU_REGISTRY: OpRegistry[
    GateUpSwiGLUOpSpec, GateUpSwiGLUProvider
] = OpRegistry(
    "gate/up SwiGLU",
    portfolio=PortfolioPolicy(repo_portable=("native",)),
    profile_order=("h20_gate_up_dispatch_plan",),
)


@GATE_UP_SWIGLU_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class NativeGateUpSwiGLUProvider(GateUpSwiGLUProvider):
    name = "native"

    @classmethod
    def supports(
        cls,
        spec: GateUpSwiGLUOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        del spec, caps
        return SupportResult.yes()

    def run(
        self,
        spec: GateUpSwiGLUOpSpec,
        inputs: torch.Tensor,
        projection,
    ) -> torch.Tensor:
        del spec
        gate, up = projection(inputs).chunk(2, dim=-1)
        return F.silu(gate, inplace=True).mul_(up)


@GATE_UP_SWIGLU_REGISTRY.register_atomic(
    ProviderRole.REPO_PORTABLE,
    profile_only=True,
)
class H20DecodeGateUpSwiGLUProvider(GateUpSwiGLUProvider):
    name = "h20_triton_decode"

    @classmethod
    def supports(
        cls,
        spec: GateUpSwiGLUOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} "
                f"{caps.compute_capability}"
            )
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported(
                "device does not support CUDA Graph capture"
            )
        if spec.activation_dtype != torch.bfloat16 or spec.weight_dtype != torch.bfloat16:
            return SupportResult.unsupported(
                "requires BF16 activations and weights, got "
                f"{spec.activation_dtype} and {spec.weight_dtype}"
            )
        if (spec.hidden_size, spec.intermediate_size, spec.tp_size) not in {
            (2048, 512, 1),
            (2048, 512, 2),
        }:
            return SupportResult.unsupported(
                "kernel contract requires (hidden, intermediate, TP) in "
                "{(2048, 512, 1), (2048, 512, 2)}"
            )
        return SupportResult.yes()

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "implementation_source": "repo_triton",
            "kernel_path": "triton.gate_up_swiglu.h20_gate_up_swiglu",
        }

    def run(
        self,
        spec: GateUpSwiGLUOpSpec,
        inputs: torch.Tensor,
        projection,
    ) -> torch.Tensor:
        del spec
        if int(inputs.shape[0]) != 1:
            raise ValueError(
                "H20 decode gate/up SwiGLU requires exactly one token, got "
                f"M={inputs.shape[0]}."
            )
        from sparseengine.kernels.triton.gate_up_swiglu import h20_gate_up_swiglu

        return h20_gate_up_swiglu(inputs, projection.weight)


@dataclass(frozen=True, slots=True)
class GateUpSwiGLUDispatchRoute:
    min_tokens: int
    max_tokens: int | None
    provider: GateUpSwiGLUProvider
    kernel_path: str

    def matches(self, num_tokens: int) -> bool:
        return num_tokens >= self.min_tokens and (
            self.max_tokens is None or num_tokens <= self.max_tokens
        )


@GATE_UP_SWIGLU_REGISTRY.register_profile
class H20GateUpSwiGLUDispatchPlan(GateUpSwiGLUProvider):
    name = "h20_gate_up_dispatch_plan"

    @classmethod
    def atomic_provider_names(
        cls,
        spec: GateUpSwiGLUOpSpec,
    ) -> tuple[str, ...]:
        del spec
        return ("h20_triton_decode", "native")

    @classmethod
    def matches(
        cls,
        spec: GateUpSwiGLUOpSpec,
        caps: DeviceCaps,
    ) -> ProfileMatch:
        if not device_name_contains(caps.device_name, "H20"):
            return ProfileMatch.no(
                f"requires profiled H20 hardware, got {caps.device_name}"
            )
        if (spec.hidden_size, spec.intermediate_size, spec.tp_size) not in {
            (2048, 512, 1),
            (2048, 512, 2),
        }:
            return ProfileMatch.no(
                "requires profiled (hidden, intermediate, TP) shape in "
                "{(2048, 512, 1), (2048, 512, 2)}"
            )
        return ProfileMatch.yes("matched H20 gate/up token-range profile")

    @classmethod
    def bind(
        cls,
        spec: GateUpSwiGLUOpSpec,
        caps: DeviceCaps,
        **kwargs,
    ) -> H20GateUpSwiGLUDispatchPlan:
        del caps
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        return cls(spec)

    def __init__(self, spec: GateUpSwiGLUOpSpec) -> None:
        self.spec = spec
        self.routes = (
            GateUpSwiGLUDispatchRoute(
                1,
                1,
                H20DecodeGateUpSwiGLUProvider(),
                "triton.gate_up_swiglu.h20_gate_up_swiglu",
            ),
            GateUpSwiGLUDispatchRoute(
                2,
                None,
                NativeGateUpSwiGLUProvider(),
                "native.projection+silu_mul",
            ),
        )
        self._runtime_kernel_path_counts: dict[str, dict[str, int]] = {}

    def _route(self, num_tokens: int) -> GateUpSwiGLUDispatchRoute:
        if num_tokens <= 0:
            raise ValueError("Gate/up SwiGLU requires at least one token.")
        for route in self.routes:
            if route.matches(num_tokens):
                return route
        raise RuntimeError(f"{self.name} has no prepared route for M={num_tokens}.")

    def _record_runtime_kernel_path(self, path: str) -> None:
        counts = self._runtime_kernel_path_counts.setdefault(
            path,
            {"eager_dispatches": 0, "cuda_graph_capture_dispatches": 0},
        )
        key = (
            "cuda_graph_capture_dispatches"
            if device_runtime.is_stream_capturing()
            else "eager_dispatches"
        )
        counts[key] += 1

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "dispatch_plan",
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

    def runtime_kernel_stats(self) -> dict[str, object]:
        return {
            "kernel_paths": {
                path: dict(sorted(counts.items()))
                for path, counts in sorted(self._runtime_kernel_path_counts.items())
            },
            "fallback_reasons": {},
        }

    def run(
        self,
        spec: GateUpSwiGLUOpSpec,
        inputs: torch.Tensor,
        projection,
    ) -> torch.Tensor:
        if spec is not self.spec:
            raise RuntimeError(
                f"{self.name} was bound for {self.spec!r}, got {spec!r}."
            )
        route = self._route(int(inputs.shape[0]))
        self._record_runtime_kernel_path(route.kernel_path)
        return route.provider.run(spec, inputs, projection)


def resolve_gate_up_swiglu_provider(
    spec: GateUpSwiGLUOpSpec,
    *,
    device_index: int | None = None,
) -> GateUpSwiGLUProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(GATE_UP_SWIGLU_REGISTRY).resolve(spec, caps).provider
