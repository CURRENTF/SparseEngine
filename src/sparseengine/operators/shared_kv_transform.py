from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import shutil

import torch

import sparseengine.platforms as platforms
from sparseengine.operators.registry import (
    OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class SharedKVTransformOpSpec:
    page_size: int
    activation_dtype: torch.dtype
    rms_eps: float
    cuda_graph: bool = True

    def __post_init__(self):
        if self.page_size <= 0 or self.page_size & (self.page_size - 1):
            raise ValueError("Shared KV page size must be a power of two")
        if self.rms_eps <= 0:
            raise ValueError("Shared KV RMS epsilon must be positive")


SHARED_KV_TRANSFORM_REGISTRY = OpRegistry(
    "shared KV normalization, rotation and storage",
    portfolio=PortfolioPolicy(repo_nonstandard=("sgl_v4",)),
)


@SHARED_KV_TRANSFORM_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class SGLSharedKVTransformProvider:
    name = "sgl_v4"

    @classmethod
    def supports(cls, spec: SharedKVTransformOpSpec, caps: DeviceCaps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported("V4 shared KV transforms require SM90")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported("V4 shared KV transforms require BF16")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph")
        if importlib.util.find_spec("tvm_ffi") is None:
            return SupportResult.dependency_absent("install apache-tvm-ffi 0.1.10")
        if shutil.which("nvcc") is None or shutil.which("ninja") is None:
            return SupportResult.dependency_absent("V4 JIT requires nvcc and ninja on PATH")
        return SupportResult.yes()

    def __init__(self, *, op_spec, architecture):
        from sparseengine.kernels.external.sgl.v4_compression import _load
        from sparseengine.kernels.external.sgl.v4_compression.gather import gather_packed_kv

        self.spec = op_spec
        self._query = _load("query", 512, torch.bfloat16, torch.bfloat16, architecture)
        self._stores = {
            dtype: _load(f"store_{op_spec.page_size}", 512, dtype, dtype, architecture)
            for dtype in (torch.bfloat16, torch.float32)
        }
        self._gather = gather_packed_kv

    def normalize_rotate_query(self, query, positions, freqs, *, out=None):
        if query.ndim != 3 or query.shape[-1] != 512 or query.dtype != torch.bfloat16:
            raise ValueError("Query must be BF16 [tokens, heads, 512]")
        out = torch.empty_like(query) if out is None else out
        self._query.forward(query, out, freqs, positions, self.spec.rms_eps)
        return out

    def store(self, values, byte_storage, slots):
        """Store already normalized/rotated values; every slot must be valid.

        Cache managers supply distinct reserved scratch slots for graph padding
        and incomplete compression groups. Negative slots are not supported by
        the upstream store kernel.
        """
        if values.dtype not in self._stores or values.shape != (slots.numel(), 512):
            raise ValueError("Packed KV store requires BF16/FP32 [tokens, 512]")
        if slots.dtype != torch.int32 or slots.ndim != 1:
            raise ValueError("Packed KV slots must be INT32 [tokens]")
        self._stores[values.dtype].forward(values, byte_storage, slots)

    def gather(self, byte_storage, slots, *, out=None):
        if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64):
            raise ValueError("Packed KV gather slots must be a flat integer tensor")
        if out is None:
            out = torch.empty(slots.numel(), 1, 512, dtype=torch.bfloat16,
                              device=byte_storage.device)
        if out.shape != (slots.numel(), 1, 512) or out.dtype != torch.bfloat16:
            raise ValueError("Packed KV gather output must be BF16 [tokens, 1, 512]")
        return self._gather(byte_storage, slots, self.spec.page_size, out)


def resolve_shared_kv_transform_provider(spec, *, device_index):
    caps = platforms.current_platform.get_device_caps(device_index)
    return OpResolver(SHARED_KV_TRANSFORM_REGISTRY).resolve(
        spec, caps, op_spec=spec, architecture=caps.compute_capability,
    ).provider


@torch.compile(fullgraph=True, dynamic=True)
def inverse_shared_kv_rope(values, positions, freqs):
    """Torch baseline for the grouped output projection's inverse rotation."""
    tail = values[..., -64:].float().reshape(*values.shape[:-1], 32, 2)
    cs = freqs[positions].reshape(-1, 1, 32, 2)
    real, imag = tail.unbind(-1)
    cosine, sine = cs.unbind(-1)
    rotated = torch.stack((real*cosine + imag*sine, imag*cosine - real*sine), -1)
    return torch.cat((values[..., :-64], rotated.flatten(-2).to(values.dtype)), -1)


class GroupedSharedKVProjection:
    """Prepared BF16 cuBLAS baseline for block-FP8 grouped output weights.

    The checkpoint's 128x128 scales apply before grouping. This object owns
    only the converted physical weight; the loader releases raw FP8 storage.
    """

    def __init__(self, *, num_groups, input_size_per_group, output_size_per_group):
        if min(num_groups, input_size_per_group, output_size_per_group) <= 0:
            raise ValueError("Grouped projection dimensions must be positive")
        self.groups = num_groups
        self.input_size = input_size_per_group
        self.output_size = output_size_per_group

    def prepare_weights(self, weight, scales):
        if hasattr(self, "weight"):
            raise RuntimeError("Grouped projection weights can only be prepared once")
        if weight.shape != (self.groups*self.output_size, self.input_size):
            raise ValueError("Grouped output checkpoint weight has the wrong shape")
        if weight.dtype != torch.float8_e4m3fn or not weight.is_cuda:
            raise ValueError("Grouped projection requires CUDA FP8 checkpoint weights")
        if scales.shape != ((weight.shape[0]+127)//128, (weight.shape[1]+127)//128):
            raise ValueError("Grouped output checkpoint scales have the wrong shape")
        # Dequantize in FP32, then cast once. No scale broadcast is retained.
        rows = torch.arange(weight.shape[0], device=weight.device) // 128
        cols = torch.arange(weight.shape[1], device=weight.device) // 128
        self.weight = (weight.float() * scales.float()[rows[:, None], cols[None, :]]).bfloat16()
        self.weight = self.weight.view(self.groups, self.output_size, self.input_size)

    def forward(self, values):
        if not hasattr(self, "weight"):
            raise RuntimeError("Grouped projection weights have not been prepared")
        if values.shape[1:] != (self.groups, self.input_size) or values.dtype != torch.bfloat16:
            raise ValueError("Grouped output input differs from its prepared shape/dtype")
        return torch.bmm(values.transpose(0, 1), self.weight.transpose(1, 2)).transpose(0, 1).flatten(1)
