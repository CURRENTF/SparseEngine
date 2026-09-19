"""Prepared repository MLA partial attention, independent of decode selection."""

from dataclasses import dataclass
from functools import partial
from importlib import import_module

import torch

from sparseengine.kernels.triton.mla.prefill_plan import (
    select_prefill_splits,
    split_workspace_bytes,
)
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
)
from sparseengine.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class MlaPrefillSpec:
    heads: int
    qk_dim: int
    value_dim: int
    dtype: torch.dtype
    kv_aligned: bool


MLA_PREFILL_REGISTRY = OpRegistry(
    "Triton MLA partial attention",
    portfolio=PortfolioPolicy(repo_portable=("hopper", "pipelined", "baseline")),
)


@MLA_PREFILL_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class BaselineMlaPrefill:
    name = "baseline"
    module = "prefill"
    kernel_path = "triton_mla_prefill"
    split_kv = False

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA and Triton")
        if spec.dtype not in (torch.float16, torch.bfloat16):
            return SupportResult.unsupported("requires FP16 or BF16")
        if spec.dtype == torch.bfloat16 and not caps.supports_bfloat16:
            return SupportResult.unsupported("requires BF16 hardware support")
        if (spec.qk_dim != spec.value_dim or spec.qk_dim < 16
                or spec.qk_dim & (spec.qk_dim - 1)):
            return SupportResult.unsupported("requires equal power-of-two QK/V dimensions >= 16")
        if spec.heads <= 0:
            return SupportResult.unsupported("requires positive local head count")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps, **kwargs):
        return cls(spec, caps)

    def __init__(self, spec, caps):
        self.spec = spec
        self.device = torch.device(caps.device_type, caps.device_index)
        self.sm_count = caps.multi_processor_count
        kernel = import_module(f"sparseengine.kernels.triton.mla.{self.module}").attention_partial
        self.kernel = (
            partial(kernel, split_kv=True, sm_count=self.sm_count) if self.split_kv else kernel
        )

    def __call__(self, q, k, v, cu_q, cu_k, max_q, max_k, *, scale, causal):
        if q.shape[1:] != (self.spec.heads, self.spec.qk_dim) or q.dtype != self.spec.dtype:
            raise ValueError("MLA prefill input differs from its prepared head/dtype contract")
        if any(t.device != self.device for t in (q, k, v)):
            raise ValueError("MLA prefill tensors must use the prepared device")
        return self.kernel(q, k, v, cu_q, cu_k, max_q, max_k, scale=scale, causal=causal)

    def workspace_bytes(self, *, tokens, batch, max_q, max_k):
        if not self.split_kv:
            return 0
        splits = select_prefill_splits(
            heads=self.spec.heads, batch=batch, max_q=max_q, max_k=max_k,
            sm_count=self.sm_count,
        )
        return split_workspace_bytes(
            tokens=tokens, heads=self.spec.heads, splits=splits, value_dim=self.spec.value_dim,
        )

    def binding_metadata(self):
        return {"kernel_path": self.kernel_path, "split_kv": self.split_kv}


@MLA_PREFILL_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class PipelinedMlaPrefill(BaselineMlaPrefill):
    name = "pipelined"
    module = "prefill_pipelined"
    kernel_path = "triton_mla_prefill_pipelined"
    split_kv = True

    @classmethod
    def supports(cls, spec, caps):
        base = super().supports(spec, caps)
        if not base.supported:
            return base
        if spec.dtype != torch.bfloat16 or spec.qk_dim != 256:
            return SupportResult.unsupported("requires BF16 QK/V dimension 256")
        if caps.compute_capability is None or caps.compute_capability < (8, 0):
            return SupportResult.unsupported("requires Ampere-or-newer asynchronous copies and MMA")
        if caps.multi_processor_count is None or caps.multi_processor_count <= 0:
            return SupportResult.unsupported("requires a positive SM count for split-KV planning")
        return SupportResult.yes()


@MLA_PREFILL_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class HopperMlaPrefill(PipelinedMlaPrefill):
    name = "hopper"
    module = "prefill_hopper"
    kernel_path = "triton_mla_prefill_hopper"

    @classmethod
    def supports(cls, spec, caps):
        base = super().supports(spec, caps)
        if not base.supported:
            return base
        if caps.compute_capability != (9, 0):
            return SupportResult.unsupported("requires Hopper WGMMA")
        if not spec.kv_aligned:
            return SupportResult.unsupported("TMA requires aligned K/V projection storage")
        return SupportResult.yes()


def resolve_mla_prefill(spec, caps):
    # expand() owns contiguous K and the head-strided projection view for V.
    element = spec.activation_dtype.itemsize
    nope = spec.qk_head_dim - spec.rope_dim
    partial_spec = MlaPrefillSpec(
        heads=spec.local_q_heads,
        qk_dim=spec.qk_head_dim,
        value_dim=spec.value_head_dim,
        dtype=spec.activation_dtype,
        kv_aligned=all(n * element % 16 == 0 for n in (
            spec.qk_head_dim, nope, nope + spec.value_head_dim,
        )),
    )
    return OpResolver(MLA_PREFILL_REGISTRY).resolve(partial_spec, caps).provider
