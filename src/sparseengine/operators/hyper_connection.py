from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import torch

import sparseengine.platforms as platforms
from sparseengine.operators.registry import (
    OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class HyperConnectionOpSpec:
    hidden_size: int
    residual_streams: int
    activation_dtype: torch.dtype
    rms_eps: float
    mixing_eps: float
    sinkhorn_iterations: int
    post_multiplier: float = 2.0
    cuda_graph: bool = True

    def __post_init__(self):
        if self.hidden_size <= 0 or self.residual_streams <= 0:
            raise ValueError("Hyperconnection dimensions must be positive.")
        if self.rms_eps <= 0 or self.mixing_eps <= 0:
            raise ValueError("Hyperconnection epsilons must be positive.")
        if self.sinkhorn_iterations <= 0:
            raise ValueError("Sinkhorn requires at least one iteration.")


class HyperConnectionProvider:
    """Residual mixing semantics; projection and kernel lifetimes belong here."""

    name = ""

    def pre(self, residual, projection, scale, base):
        raise NotImplementedError

    def post(self, output, residual, post_mix, residual_mix):
        raise NotImplementedError


HYPER_CONNECTION_REGISTRY = OpRegistry(
    "hyperconnection",
    portfolio=PortfolioPolicy(upstream_standard=("flashinfer",)),
)


@HYPER_CONNECTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferHyperConnectionProvider(HyperConnectionProvider):
    name = "flashinfer"

    @classmethod
    def supports(cls, spec: HyperConnectionOpSpec, caps: DeviceCaps):
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported("FlashInfer mHC requires CUDA")
        if spec.residual_streams != 4 or spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported("FlashInfer mHC requires four BF16 streams")
        if spec.hidden_size % 256:
            return SupportResult.unsupported("FlashInfer mHC requires hidden size divisible by 256")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph")
        if importlib.util.find_spec("flashinfer") is None:
            return SupportResult.dependency_absent("install FlashInfer >= 0.6.18")
        # Validate the installed API before binding; execution failures propagate.
        from flashinfer import mhc
        for name in ("mhc_pre_big_fuse_with_prenorm", "mhc_post"):
            if not callable(getattr(mhc, name, None)):
                return SupportResult.dependency_broken(f"FlashInfer lacks mhc.{name}")
        return SupportResult.yes()

    def __init__(self, *, op_spec: HyperConnectionOpSpec):
        from flashinfer.mhc import mhc_post, mhc_pre_big_fuse_with_prenorm

        self.spec = op_spec
        self._pre = mhc_pre_big_fuse_with_prenorm
        self._post = mhc_post

    def pre(self, residual, projection, scale, base):
        spec = self.spec
        streams, hidden = spec.residual_streams, spec.hidden_size
        mix_size = streams * (streams + 2)
        if residual.ndim != 3 or residual.shape[1:] != (streams, hidden):
            raise ValueError("mHC residual must have shape [tokens, streams, hidden]")
        if projection.shape != (mix_size, streams * hidden):
            raise ValueError("mHC projection must have shape [mix, streams * hidden]")
        if scale.shape != (3,) or base.shape != (mix_size,):
            raise ValueError("mHC requires three scales and one bias per mix")
        if residual.dtype != spec.activation_dtype:
            raise TypeError("mHC residual dtype differs from the bound contract")
        if any(t.dtype != torch.float32 for t in (projection, scale, base)):
            raise TypeError("mHC projection, scale and bias must be FP32")
        if not residual.is_cuda or any(t.device != residual.device for t in (projection, scale, base)):
            raise ValueError("mHC inputs must share a CUDA device")
        # The checkpoint projection is FP32. Keep this baseline separate from
        # the external mix/Sinkhorn kernel so projection can later be fused.
        dot = torch.nn.functional.linear(residual.flatten(1).float(), projection)
        return self._pre(
            dot, residual, scale, base,
            rms_eps=spec.rms_eps, mhc_pre_eps=spec.mixing_eps,
            mhc_sinkhorn_eps=spec.mixing_eps,
            mhc_post_mult_value=spec.post_multiplier,
            sinkhorn_repeat=spec.sinkhorn_iterations,
        )

    def post(self, output, residual, post_mix, residual_mix):
        return self._post(output, residual, post_mix, residual_mix)

    def head(self, residual, projection, scale, base):
        streams, hidden = self.spec.residual_streams, self.spec.hidden_size
        if residual.shape[1:] != (streams, hidden) or projection.shape != (streams, streams*hidden):
            raise ValueError("mHC head residual/projection differs from the bound contract")
        if scale.numel() != 1 or base.shape != (streams,):
            raise ValueError("mHC head requires one scale and one bias per stream")
        return _head_reference(residual, projection, scale, base,
                               self.spec.rms_eps, self.spec.mixing_eps)


def resolve_hyper_connection_provider(spec: HyperConnectionOpSpec, *, device_index: int):
    caps = platforms.current_platform.get_device_caps(device_index)
    return OpResolver(HYPER_CONNECTION_REGISTRY).resolve(
        spec, caps, op_spec=spec,
    ).provider


@torch.compile(fullgraph=True, dynamic=True)
def _head_reference(residual, projection, scale, base, rms_eps, mixing_eps):
    flat = residual.flatten(1).float()
    mixes = torch.nn.functional.linear(flat, projection)
    mixes = mixes * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes * scale + base) + mixing_eps
    return (pre[..., None] * residual.float()).sum(1).to(residual.dtype)
