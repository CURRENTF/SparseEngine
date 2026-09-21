"""Independent compressed MLA prefill portfolio; decode scores are unrelated."""

import torch

from sparseengine.kernels.external.sgl.fa3 import SglFa3DecodeKernel, sgl_fa3_device_support
from sparseengine.kernels.external.sgl.support import sgl_kernel_metadata_health
from sparseengine.kernels.external.support import KernelFamilyState
from sparseengine.operators.registry import (
    OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult,
)
from sparseengine.platforms.interface import PlatformEnum


MLA_COMPRESSED_PREFILL_REGISTRY = OpRegistry(
    "MLA compressed prefill",
    portfolio=PortfolioPolicy(upstream_standard=("sgl_fa3_latent",),
                              repo_portable=("triton_latent",)),
)


def _contract(spec, caps):
    if caps.platform != PlatformEnum.CUDA:
        return SupportResult.unsupported("requires CUDA")
    if (spec.kv_lora_rank != 512 or spec.rope_dim != 64
            or spec.activation_dtype not in (torch.bfloat16, torch.float16)
            or spec.cache_dtype != spec.activation_dtype):
        return SupportResult.unsupported("requires matching FP16/BF16 latent 512 and RoPE 64")
    if spec.activation_dtype == torch.bfloat16 and not caps.supports_bfloat16:
        return SupportResult.unsupported("requires BF16 hardware support")
    return SupportResult.yes()


@MLA_COMPRESSED_PREFILL_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class SglLatentPrefill:
    name = "sgl_fa3_latent"
    kernel_path = "sgl_fa3_prefill_latent"

    @classmethod
    def supports(cls, spec, caps):
        base = _contract(spec, caps)
        if not base.supported:
            return base
        health = sgl_kernel_metadata_health()
        if health.state is KernelFamilyState.ABSENT:
            return SupportResult.dependency_absent(health.reason)
        supported, reason = sgl_fa3_device_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    @classmethod
    def bind(cls, spec, caps, *, max_batch_size):
        return cls(spec, caps, max_batch_size)

    def __init__(self, spec, caps, max_batch_size):
        self.kernel = SglFa3DecodeKernel(
            device=torch.device(caps.device_type, caps.device_index),
            max_batch_size=max_batch_size, softmax_scale=spec.softmax_scale,
        )

    def run(self, q, q_rope, view, plan):
        output = torch.empty_like(q, memory_format=torch.contiguous_format)
        return self.kernel.run_varlen(
            q_rope, q, view.payload.rope_cache, view.payload.latent_cache,
            view.meta.active_slots, view.meta.req_indices, view.meta.context_lens,
            output, cu_seqlens_q=plan.cu_q,
            max_seqlen_q=max(b - a for a, b in zip(plan.query_starts, plan.query_starts[1:])),
            validation_scope=plan.scope, return_softmax_lse=True,
        )

    def workspace_bytes(self, plan):
        # Opaque upstream scratch remains covered by startup memory profiling.
        return 0

    def binding_metadata(self):
        return {"kernel_path": self.kernel_path, "lse": "natural_log"}


@MLA_COMPRESSED_PREFILL_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonLatentPrefill:
    name = "triton_latent"
    kernel_path = "triton_mla_prefill_latent"

    @classmethod
    def supports(cls, spec, caps):
        base = _contract(spec, caps)
        if not base.supported:
            return base
        if not caps.supports_triton:
            return SupportResult.unsupported("requires Triton")
        if caps.compute_capability is None or caps.compute_capability < (8, 0):
            return SupportResult.unsupported("requires Ampere-or-newer MMA v2")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps, **kwargs):
        return cls(spec, caps)

    def __init__(self, spec, caps):
        from sparseengine.kernels.triton.mla.prefill_latent import attention_latent

        self.spec, self.sm_count = spec, caps.multi_processor_count or 1
        self.device = torch.device(caps.device_type, caps.device_index)
        self.kernel = attention_latent

    def splits(self, plan):
        # Supply roughly two CTA waves without splitting already wide batches.
        programs = sum((self.spec.local_q_heads * (b - a) + 15) // 16
                       for a, b in zip(plan.query_starts, plan.query_starts[1:]))
        waves = max(1, (2 * self.sm_count + programs - 1) // programs)
        chunks = max(1, max(plan.contexts) // 256)
        return min(32, 1 << (waves - 1).bit_length(), 1 << (chunks.bit_length() - 1))

    def workspace_bytes(self, plan):
        splits = self.splits(plan)
        return (0 if splits == 1 else splits * self.spec.local_q_heads
                * plan.query_starts[-1] * (self.spec.kv_lora_rank + 1) * 4)

    def run(self, q, q_rope, view, plan):
        if (q.device != self.device or q.dtype != self.spec.activation_dtype
                or q.shape[:2] != (plan.query_starts[-1], self.spec.local_q_heads)):
            raise ValueError("Latent prefill input differs from its prepared contract")
        return self.kernel(
            q, q_rope, view.payload.latent_cache, view.payload.rope_cache,
            view.meta.active_slots, view.meta.req_indices, view.meta.context_lens, plan.cu_q,
            max_q=max(b - a for a, b in zip(plan.query_starts, plan.query_starts[1:])),
            scale=self.spec.softmax_scale, splits=self.splits(plan),
        )

    def binding_metadata(self):
        return {"kernel_path": self.kernel_path, "lse": "natural_log",
                "mma": "v2", "query_head_tile": 16, "key_tile": 64,
                "num_warps": 8, "split_policy": "two_cta_waves"}


def resolve_mla_compressed_prefill(spec, caps, *, max_batch_size):
    return OpResolver(MLA_COMPRESSED_PREFILL_REGISTRY).resolve(
        spec, caps, max_batch_size=max_batch_size,
    ).provider
