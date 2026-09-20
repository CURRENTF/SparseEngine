from __future__ import annotations

from dataclasses import dataclass

import torch

import sparseengine.platforms as platforms
from sparseengine.kernels.external.flashinfer.gdn import (
    FLASHINFER_GDN_PREFILL_CAPABILITIES,
    flashinfer_chunk_gated_delta_rule,
    flashinfer_gdn_prefill_support,
)
from sparseengine.kernels.triton.qwen3_5.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from sparseengine.kernels.triton.qwen3_5.fla.ops.l2norm import l2norm_fwd
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
    runtime_version_at_least,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.utils.log import logger


@dataclass(frozen=True)
class GatedDeltaRuleOpSpec:
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    activation_dtype: torch.dtype
    recurrent_state_dtype: torch.dtype
    state_layout_id: str = "k_major_hkv"
    varlen_prefill: bool = True
    cuda_graph_decode: bool = True

    def __post_init__(self) -> None:
        if self.num_key_heads <= 0 or self.num_value_heads <= 0:
            raise ValueError("GDN head counts must be positive.")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError("GDN value heads must be divisible by key heads.")
        if self.key_head_dim <= 0 or self.value_head_dim <= 0:
            raise ValueError("GDN head dimensions must be positive.")
        if self.state_layout_id != "k_major_hkv":
            raise ValueError(
                f"Unsupported GDN recurrent-state layout {self.state_layout_id!r}."
            )


class GatedDeltaRuleProvider:
    name = ""

    def prepare(
        self,
        spec: GatedDeltaRuleOpSpec,
        *,
        device_index: int | None = None,
    ) -> None:
        del spec, device_index

    def close(self) -> None:
        pass

    def run_gating(
        self,
        *,
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sparseengine.kernels.triton.qwen3_5.fused_gdn_gating import (
            fused_gdn_gating,
        )

        return fused_gdn_gating(A_log, a, b, dt_bias)

    def run_prefill_conv(
        self,
        *,
        mixed_qkv: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        conv_states: torch.Tensor,
        activation: str,
    ) -> torch.Tensor:
        from sparseengine.kernels.triton.qwen3_5.causal_conv1d import (
            causal_conv1d_fn,
        )

        return causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            weight,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            conv_states=conv_states,
            activation=activation,
        ).transpose(0, 1)

    def prepare_decode_inputs(self, **kwargs) -> tuple[torch.Tensor, ...]:
        from sparseengine.kernels.triton.qwen3_5.gdn_decode_pack import (
            conv_pack_gdn_decode_inputs,
        )

        return conv_pack_gdn_decode_inputs(**kwargs)

    def run_gated_rmsnorm(
        self,
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        from sparseengine.kernels.triton.qwen3_5.gated_rmsnorm import (
            gated_rmsnorm_forward,
        )

        return gated_rmsnorm_forward(
            x=x.contiguous(),
            weight=weight,
            bias=None,
            eps=eps,
            z=gate.contiguous(),
        )

    def run_prefill(
        self,
        spec: GatedDeltaRuleOpSpec,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def run_decode(
        self,
        spec: GatedDeltaRuleOpSpec,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        initial_state: torch.Tensor,
        state_indices: torch.Tensor,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        del spec
        output, _ = fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            initial_state=initial_state,
            inplace_final_state=True,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            a_raw=a,
            b_raw=b,
        )
        return output


GATED_DELTA_RULE_REGISTRY: OpRegistry[
    GatedDeltaRuleOpSpec, GatedDeltaRuleProvider
] = OpRegistry(
    "gated delta rule",
    portfolio=PortfolioPolicy(
        upstream_standard=(
            "flashinfer_gdn_prefill_triton_decode",
        ),
        repo_nonstandard=("triton_gated_delta_rule",),
    ),
)


@GATED_DELTA_RULE_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferGatedDeltaRuleProvider(GatedDeltaRuleProvider):
    """Fixed FlashInfer-prefill/repo-decode GDN implementation plan."""

    name = "flashinfer_gdn_prefill_triton_decode"

    @classmethod
    def supports(
        cls,
        spec: GatedDeltaRuleOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if (
            caps.platform != PlatformEnum.CUDA
            or caps.compute_capability not in FLASHINFER_GDN_PREFILL_CAPABILITIES
        ):
            return SupportResult.unsupported(
                "requires CUDA SM90, SM100, SM103, SM120, or SM121, got "
                f"{caps.platform.name} {caps.compute_capability}"
            )
        if not caps.supports_triton:
            return SupportResult.unsupported("decode implementation requires Triton")
        minimum_runtime = (
            (13, 0) if caps.compute_capability[0] == 10 else (12, 8)
        )
        if not runtime_version_at_least(caps.runtime_version, minimum_runtime):
            return SupportResult.unsupported(
                "requires CUDA runtime >= "
                f"{minimum_runtime[0]}.{minimum_runtime[1]}, "
                f"got {caps.runtime_version or 'unknown'}"
            )
        if spec.activation_dtype not in (torch.bfloat16, torch.float16):
            return SupportResult.unsupported(
                "requires BF16 or FP16 activations, got "
                f"{spec.activation_dtype}"
            )
        if spec.recurrent_state_dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            return SupportResult.unsupported(
                "FlashInfer GDN requires BF16/FP32 runtime recurrent state, got "
                f"{spec.recurrent_state_dtype}"
            )
        if caps.compute_capability[0] in {10, 12} and (
            spec.key_head_dim != 128 or spec.value_head_dim != 128
        ):
            return SupportResult.unsupported(
                "FlashInfer Blackwell GDN prefill requires key/value head dim 128, got "
                f"{spec.key_head_dim}/{spec.value_head_dim}"
            )
        if spec.key_head_dim != spec.value_head_dim:
            return SupportResult.unsupported(
                "FlashInfer GDN requires equal key/value head dimensions, got "
                f"{spec.key_head_dim}/{spec.value_head_dim}"
            )
        if not spec.varlen_prefill:
            return SupportResult.unsupported("requires varlen prefill")
        if spec.cuda_graph_decode and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        supported, reason = flashinfer_gdn_prefill_support(
            caps.compute_capability
        )
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "flashinfer-python+repo_triton",
            "prefill_kernel_path": "flashinfer.gdn_prefill.chunk_gated_delta_rule",
            "decode_kernel_path": "triton.fused_recurrent_gated_delta_rule",
            "runtime_state_layout": "k_major_hkv",
            "prefill_state_layout": "v_major_hvk",
            "state_layout_adapter": "transpose_last_two_dims",
            "profile_source": "flashinfer-native",
            "auxiliary_kernel_paths": [
                "triton.qwen3_5.fused_gdn_gating",
                "triton.qwen3_5.causal_conv1d",
                "triton.qwen3_5.gdn_decode_pack",
                "triton.qwen3_5.gated_rmsnorm",
            ],
        }

    def run_prefill(
        self,
        spec: GatedDeltaRuleOpSpec,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del spec
        # Q/K are column views of the packed Conv1D output. The repo L2-norm
        # kernel requires a packed token/head tail, and FlashInfer requires
        # contiguous 3D inputs, so the adapter owns this materialization.
        normalized_q = l2norm_fwd(q.contiguous()).squeeze(0)
        normalized_k = l2norm_fwd(k.contiguous()).squeeze(0)
        # SparseEngine keeps recurrent state in the repo decode kernel's
        # K-major [N, H, K, V] layout. FlashInfer's public prefill contract is
        # V-major [N, H, V, K], so this adapter owns both conversions. Qwen's
        # K/V dimensions are both 128, making a shape-only check insufficient.
        flashinfer_initial_state = (
            initial_state.to(torch.float32).transpose(-1, -2).contiguous()
        )
        output, final_state = flashinfer_chunk_gated_delta_rule(
            normalized_q,
            normalized_k,
            v.squeeze(0).contiguous(),
            torch.exp(g.squeeze(0)).contiguous(),
            beta.squeeze(0).to(torch.float32).contiguous(),
            flashinfer_initial_state,
            cu_seqlens,
        )
        repo_final_state = final_state.transpose(-1, -2).contiguous()
        return output.unsqueeze(0), repo_final_state


@GATED_DELTA_RULE_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonGatedDeltaRuleProvider(GatedDeltaRuleProvider):
    name = "triton_gated_delta_rule"

    @classmethod
    def supports(
        cls,
        spec: GatedDeltaRuleOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.activation_dtype not in (torch.bfloat16, torch.float16):
            return SupportResult.unsupported(
                f"requires BF16/FP16 activations, got {spec.activation_dtype}"
            )
        if spec.recurrent_state_dtype not in (torch.bfloat16, torch.float32):
            return SupportResult.unsupported(
                "requires BF16/FP32 recurrent state, got "
                f"{spec.recurrent_state_dtype}"
            )
        if spec.cuda_graph_decode and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        return SupportResult.yes("generic repo Triton GDN implementation")

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "prefill_kernel_path": "triton.chunk_gated_delta_rule",
            "decode_kernel_path": "triton.fused_recurrent_gated_delta_rule",
            "runtime_state_layout": "k_major_hkv",
            "prefill_state_layout": "k_major_hkv",
            "profile_source": "repo-static",
            "auxiliary_kernel_paths": [
                "triton.qwen3_5.fused_gdn_gating",
                "triton.qwen3_5.causal_conv1d",
                "triton.qwen3_5.gdn_decode_pack",
                "triton.qwen3_5.gated_rmsnorm",
            ],
        }

    def run_prefill(
        self,
        spec: GatedDeltaRuleOpSpec,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repeats = spec.num_value_heads // spec.num_key_heads
        if repeats > 1:
            q = q.repeat_interleave(repeats, dim=2)
            k = k.repeat_interleave(repeats, dim=2)
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )


class PreparedGatedDeltaRuleOp:
    def __init__(
        self,
        spec: GatedDeltaRuleOpSpec,
        provider: GatedDeltaRuleProvider,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self._closed = False

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def supports_decode_graph(self) -> bool:
        return bool(self.spec.cuda_graph_decode)

    def run_prefill(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("GDN operator is closed.")
        return self.provider.run_prefill(self.spec, **kwargs)

    def run_decode(self, **kwargs) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("GDN operator is closed.")
        return self.provider.run_decode(self.spec, **kwargs)

    def run_gating(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("GDN operator is closed.")
        return self.provider.run_gating(**kwargs)

    def run_prefill_conv(self, **kwargs) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("GDN operator is closed.")
        return self.provider.run_prefill_conv(**kwargs)

    def prepare_decode_inputs(self, **kwargs) -> tuple[torch.Tensor, ...]:
        if self._closed:
            raise RuntimeError("GDN operator is closed.")
        return self.provider.prepare_decode_inputs(**kwargs)

    def run_gated_rmsnorm(self, **kwargs) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("GDN operator is closed.")
        return self.provider.run_gated_rmsnorm(**kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self.provider.close()
        self._closed = True


def prepare_gated_delta_rule_op(
    spec: GatedDeltaRuleOpSpec,
    *,
    device_index: int | None = None,
) -> PreparedGatedDeltaRuleOp:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    resolved = OpResolver(GATED_DELTA_RULE_REGISTRY).resolve(spec, caps)
    logger.info(
        "Resolved GDN provider={} rejected={}",
        resolved.provider.name,
        dict(resolved.rejected),
    )
    resolved.provider.prepare(spec, device_index=device_index)
    return PreparedGatedDeltaRuleOp(spec, resolved.provider)


__all__ = [
    "FlashInferGatedDeltaRuleProvider",
    "GATED_DELTA_RULE_REGISTRY",
    "GatedDeltaRuleOpSpec",
    "PreparedGatedDeltaRuleOp",
    "TritonGatedDeltaRuleProvider",
    "prepare_gated_delta_rule_op",
]
