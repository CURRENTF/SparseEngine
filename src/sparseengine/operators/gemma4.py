from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import sparseengine.platforms as platforms
from sparseengine.kernels.external.flashinfer.prefill import (
    flashinfer_paged_prefill_support,
)
from sparseengine.layers.rotary_embedding import apply_rotary_emb
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
class Gemma4OpSpec:
    activation_dtype: torch.dtype
    head_dims: tuple[int, ...]
    cuda_graph: bool
    attention_contracts: tuple[tuple[int, int, int, int], ...] = ()
    max_batch_size: int = 1
    context_capacity: int | None = None

    def __post_init__(self) -> None:
        if not self.head_dims or any(int(value) <= 0 for value in self.head_dims):
            raise ValueError("Gemma 4 head dimensions must be positive.")
        if self.max_batch_size <= 0:
            raise ValueError("Gemma 4 max_batch_size must be positive.")
        if self.context_capacity is not None and self.context_capacity <= 0:
            raise ValueError("Gemma 4 context_capacity must be positive.")
        if self.cuda_graph and self.context_capacity is None:
            raise ValueError("Gemma 4 Decode Graph requires context_capacity.")


class Gemma4OperatorProvider:
    name = ""

    def __init__(self) -> None:
        self._attention_backends: list[object] = []

    def _register_attention_backend(self, backend):
        self._attention_backends.append(backend)
        return backend

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "composite_provider",
            "implementation_source": "repo_triton",
            "attention_dispatch": {
                "prefill_routes": [
                    "triton_multimodal_context",
                    "triton_context",
                ],
                "decode_routes": [
                    "sglang_fixed_grid",
                ],
            },
        }

    def runtime_kernel_stats(self) -> dict[str, object]:
        kernel_paths: dict[str, dict[str, int]] = {}
        for backend in self._attention_backends:
            stats_fn = getattr(backend, "runtime_kernel_stats", None)
            if not callable(stats_fn):
                continue
            for path, counts in stats_fn().get("kernel_paths", {}).items():
                aggregate = kernel_paths.setdefault(str(path), {})
                for key, count in counts.items():
                    aggregate[str(key)] = int(aggregate.get(str(key), 0)) + int(count)
        return {
            "kernel_paths": {
                path: dict(sorted(counts.items()))
                for path, counts in sorted(kernel_paths.items())
            },
            "fallback_reasons": {},
        }

    def attention_backend(self, *, sliding_window: int | None):
        raise NotImplementedError

    def close(self) -> None:
        for backend in self._attention_backends:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
        self._attention_backends.clear()

    def rmsnorm(
        self, x: torch.Tensor, weight: torch.Tensor | None, eps: float
    ) -> torch.Tensor:
        raise NotImplementedError

    def qkv_norm_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor | None,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        raise NotImplementedError

    def router_input(
        self,
        hidden_states: torch.Tensor,
        scale: torch.Tensor,
        root_size: float,
        eps: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def gelu_tanh_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gelu_mul(self, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def rmsnorm_residual(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        eps: float,
        scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


GEMMA4_REGISTRY: OpRegistry[Gemma4OpSpec, Gemma4OperatorProvider] = OpRegistry(
    "Gemma 4 model operations",
    portfolio=PortfolioPolicy(
        repo_nonstandard=("triton",)
    ),
    profile_order=("gemma4_h20_profile",),
)


@GEMMA4_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonGemma4OperatorProvider(Gemma4OperatorProvider):
    name = "triton"

    def __init__(
        self,
        *,
        spec: Gemma4OpSpec | None = None,
        caps: DeviceCaps | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.device = None if caps is None else torch.device("cuda", caps.device_index)
        self.multi_processor_count = (
            None if caps is None else int(caps.multi_processor_count or 0)
        )
        if caps is not None and self.multi_processor_count <= 0:
            raise ValueError("Gemma 4 requires a positive multi-processor count.")
        self._decode_workspaces: dict[tuple[int, int, int], object] = {}

    @classmethod
    def supports(cls, spec: Gemma4OpSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.unsupported("requires BF16 or FP16 activations")
        if any(head_dim not in {256, 512} for head_dim in spec.head_dims):
            return SupportResult.unsupported("requires attention head dimensions 256 or 512")
        if (
            caps.multi_processor_count is None
            or int(caps.multi_processor_count) <= 0
        ):
            return SupportResult.unsupported(
                "requires a positive multi-processor count"
            )
        return SupportResult.yes()

    @classmethod
    def bind(
        cls,
        spec: Gemma4OpSpec,
        caps: DeviceCaps,
        **kwargs,
    ) -> TritonGemma4OperatorProvider:
        if kwargs:
            raise TypeError(f"Unexpected Gemma 4 bind arguments: {sorted(kwargs)}.")
        return cls(spec=spec, caps=caps)

    def attention_backend(self, *, sliding_window: int | None):
        from sparseengine.operators.gemma4_attention import (
            Gemma4AttentionBackend,
            Gemma4DecodeWorkspace,
        )

        if (
            self.spec is None
            or self.device is None
            or self.multi_processor_count is None
        ):
            raise RuntimeError(
                "Gemma 4 attention requires a provider bound from Gemma4OpSpec."
            )
        window_left = -1 if sliding_window is None else int(sliding_window) - 1
        matching = [
            contract
            for contract in self.spec.attention_contracts
            if int(contract[3]) == window_left
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "Gemma 4 provider requires one attention contract for "
                f"window_left={window_left}, got {matching}."
            )
        query_heads, _, head_dim, _ = matching[0]
        max_kv_splits = 8
        signature = (int(query_heads), int(head_dim), max_kv_splits)
        workspace = self._decode_workspaces.get(signature)
        if workspace is None:
            workspace = Gemma4DecodeWorkspace(
                mid_output=torch.empty(
                    (
                        self.spec.max_batch_size,
                        signature[0],
                        signature[2],
                        signature[1],
                    ),
                    dtype=torch.float32,
                    device=self.device,
                ),
                mid_lse=torch.empty(
                    (self.spec.max_batch_size, signature[0], signature[2]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                num_kv_splits=torch.empty(
                    (self.spec.max_batch_size,),
                    dtype=torch.int32,
                    device=self.device,
                ),
            )
            self._decode_workspaces[signature] = workspace

        return self._register_attention_backend(
            Gemma4AttentionBackend(
                sliding_window=sliding_window,
                decode_workspace=workspace,
                multi_processor_count=self.multi_processor_count,
            )
        )

    def close(self) -> None:
        super().close()
        self._decode_workspaces.clear()

    def rmsnorm(self, x, weight, eps):
        from sparseengine.kernels.triton.gemma4_rmsnorm import gemma4_rmsnorm

        return gemma4_rmsnorm(x, weight, eps)

    def qkv_norm_rope(
        self, q, k, v, q_weight, k_weight, rope_cache, positions, eps
    ):
        from sparseengine.kernels.triton.gemma4_qkv_norm_rope import (
            gemma4_qkv_norm_rope,
        )

        gemma4_qkv_norm_rope(
            q, k, v, q_weight, k_weight, rope_cache, positions, eps
        )
        return q, k, v

    def router_input(self, hidden_states, scale, root_size, eps):
        from sparseengine.kernels.triton.gemma4_router import gemma4_router_input

        return gemma4_router_input(hidden_states, scale, root_size, eps)

    def gelu_tanh_and_mul(self, x):
        from sparseengine.kernels.triton.gemma4_gelu_and_mul import (
            gelu_tanh_and_mul_fwd,
        )

        return gelu_tanh_and_mul_fwd(x)

    def gelu_mul(self, gate, value):
        from sparseengine.kernels.triton.gemma4_fused_ops import gemma4_gelu_mul

        return gemma4_gelu_mul(gate, value)

    def rmsnorm_residual(self, x, weight, residual, eps, scalar=None):
        from sparseengine.kernels.triton.gemma4_fused_ops import (
            gemma4_rmsnorm_residual,
        )

        return gemma4_rmsnorm_residual(x, weight, residual, eps, scalar)


@GEMMA4_REGISTRY.register_atomic(
    ProviderRole.REPO_NONSTANDARD,
    profile_only=True,
)
class H20Gemma4OperatorProvider(TritonGemma4OperatorProvider):
    """Profiled H20 provider; generic Gemma kernels remain unchanged."""

    name = "gemma4_h20"

    @classmethod
    def supports(cls, spec: Gemma4OpSpec, caps: DeviceCaps) -> SupportResult:
        triton = super().supports(spec, caps)
        if not triton.supported:
            return triton
        if caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.compute_capability}"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.unsupported(
                f"requires CUDA runtime >= 12.8, got {caps.runtime_version or 'unknown'}"
            )
        supported, reason = flashinfer_paged_prefill_support("fa2")
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    @classmethod
    def bind(
        cls,
        spec: Gemma4OpSpec,
        caps: DeviceCaps,
        **kwargs,
    ) -> H20Gemma4OperatorProvider:
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        return cls(
            spec=spec,
            caps=caps,
            device_index=caps.device_index,
            max_prefill_contracts=(
                len(spec.attention_contracts) or len(spec.head_dims)
            ),
        )

    def __init__(
        self,
        *,
        spec: Gemma4OpSpec,
        caps: DeviceCaps,
        device_index: int | None = None,
        max_prefill_contracts: int = 2,
    ) -> None:
        super().__init__(spec=spec, caps=caps)
        from sparseengine.operators.gemma4_attention import Gemma4FlashInferPrefill

        self._prefill = Gemma4FlashInferPrefill()
        self._prefill.prepare(
            device_index=device_index,
            max_contracts=max_prefill_contracts,
        )

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "implementation_source": "flashinfer-python+repo_triton",
            "attention_dispatch": {
                "prefill_routes": [
                    "triton_multimodal_context",
                    "flashinfer_paged_prefill_fa2",
                    "triton_context",
                ],
                "decode_routes": [
                    "sglang_fixed_grid",
                ],
            },
            "flashinfer_backend": "fa2",
            "flashinfer_prepared_during_bind": True,
        }

    def close(self) -> None:
        try:
            self._prefill.close()
        finally:
            super().close()

    def attention_backend(self, *, sliding_window: int | None):
        backend = super().attention_backend(sliding_window=sliding_window)
        backend.flashinfer_prefill = self._prefill
        return backend


@GEMMA4_REGISTRY.register_profile
class H20Gemma4Profile:
    name = "gemma4_h20_profile"

    @classmethod
    def atomic_provider_names(cls, spec: Gemma4OpSpec) -> tuple[str, ...]:
        del spec
        return ("gemma4_h20",)

    @classmethod
    def matches(cls, spec: Gemma4OpSpec, caps: DeviceCaps) -> ProfileMatch:
        del spec
        if not device_name_contains(caps.device_name, "H20"):
            return ProfileMatch.no(
                f"requires profiled H20 hardware, got {caps.device_name}"
            )
        return ProfileMatch.yes("matched H20 Gemma 4 composite profile")

    @classmethod
    def bind(cls, spec: Gemma4OpSpec, caps: DeviceCaps, **kwargs):
        return H20Gemma4OperatorProvider.bind(spec, caps, **kwargs)


class TorchGemma4OperatorProvider(Gemma4OperatorProvider):
    """Explicit correctness oracle; never selected for production inference."""

    name = "torch_oracle"

    def attention_backend(self, *, sliding_window: int | None):
        from sparseengine.operators.gemma4_attention import Gemma4AttentionBackend

        return self._register_attention_backend(
            Gemma4AttentionBackend(sliding_window=sliding_window)
        )

    def rmsnorm(self, x, weight, eps):
        output = x.float()
        output *= torch.rsqrt(output.square().mean(-1, keepdim=True) + eps)
        if weight is not None:
            output *= weight.float()
        return output.to(x.dtype)

    def qkv_norm_rope(
        self, q, k, v, q_weight, k_weight, rope_cache, positions, eps
    ):
        q = self.rmsnorm(q, q_weight, eps)
        cos, sin = rope_cache[positions].chunk(2, -1)
        q = apply_rotary_emb(q, cos, sin)
        if k is not None:
            k = apply_rotary_emb(self.rmsnorm(k, k_weight, eps), cos, sin)
            v = self.rmsnorm(v, None, eps)
        return q, k, v

    def router_input(self, hidden_states, scale, root_size, eps):
        return self.rmsnorm(hidden_states, None, eps) * scale * root_size

    def gelu_tanh_and_mul(self, x):
        gate, up = x.chunk(2, -1)
        return F.gelu(gate, approximate="tanh") * up

    def gelu_mul(self, gate, value):
        return F.gelu(gate, approximate="tanh") * value

    def rmsnorm_residual(self, x, weight, residual, eps, scalar=None):
        output = self.rmsnorm(x, weight, eps) + residual
        return output if scalar is None else output * scalar


def resolve_gemma4_provider(
    spec: Gemma4OpSpec, *, device_index: int | None = None
) -> Gemma4OperatorProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(GEMMA4_REGISTRY).resolve(spec, caps).provider


__all__ = [
    "GEMMA4_REGISTRY",
    "H20Gemma4OperatorProvider",
    "Gemma4OperatorProvider",
    "Gemma4OpSpec",
    "TorchGemma4OperatorProvider",
    "TritonGemma4OperatorProvider",
    "resolve_gemma4_provider",
]
