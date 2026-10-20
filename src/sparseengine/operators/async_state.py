"""Prepared operator for request-indexed async feedback and penalty state."""
from dataclasses import dataclass

import torch

from sparseengine import platforms
from sparseengine.operators.registry import (
    OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult,
)
from sparseengine.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class AsyncStateSpec:
    """Int64 metadata rows: feedback slot, seed, ready, penalty slot, publish."""


REGISTRY = OpRegistry(
    "Async request feedback and sampling state",
    portfolio=PortfolioPolicy(repo_nonstandard=("triton",)),
)


@REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonAsyncState:
    name = "triton"

    def __init__(self, *, op_spec):
        from sparseengine.kernels.triton import async_state
        self.kernels = async_state

    @classmethod
    def supports(cls, spec, caps):
        del spec
        if caps.platform not in (PlatformEnum.CUDA, PlatformEnum.ROCM) or not caps.supports_triton:
            return SupportResult.unsupported("requires a GPU with Triton")
        return SupportResult.yes("request-mapped feedback and fused FP32 sampling penalties")

    def gather(self, tokens, metadata, inputs, selected):
        n = selected.numel()
        if (metadata.shape != (5, n) or metadata.dtype != torch.int64
                or not metadata.is_contiguous() or inputs.dtype != torch.int64
                or selected.dtype != torch.int64 or not inputs.is_contiguous()
                or not selected.is_contiguous() or not 0 < n <= inputs.numel()):
            raise ValueError("Async feedback requires contiguous int64 inputs and [5, batch] metadata")
        self.kernels.gather_feedback[((inputs.numel() + 255) // 256,)](
            tokens, metadata, inputs, selected, n, inputs.numel(), 256,
        )

    def commit(self, sampled, metadata, feedback, flags):
        n = sampled.numel()
        self.kernels.commit_feedback[((n + 255) // 256,)](
            sampled, metadata, feedback, flags, n, flags.shape[1], 256,
        )

    def penalize(self, logits, flags, metadata, parameters, output):
        n, vocab = logits.shape
        if (logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or output.dtype != torch.float32 or output.shape != logits.shape
                or not output.is_contiguous() or flags.dtype != torch.uint8
                or flags.shape[1] != vocab or metadata.shape != (5, n)
                or parameters.shape != (2, flags.shape[0])):
            raise ValueError("Async penalties require floating logits, byte token flags and FP32 output")
        self.kernels.apply_penalties[(n, (vocab + 1023) // 1024)](
            logits, flags, metadata, parameters, output,
            logits.stride(0), logits.stride(1), n, vocab, flags.shape[0], 1024,
            enable_fp_fusion=False,
        )


def prepare_async_state_provider(device):
    caps = platforms.current_platform.get_device_caps(device.index or 0)
    spec = AsyncStateSpec()
    return OpResolver(REGISTRY).resolve(spec, caps, op_spec=spec).provider
