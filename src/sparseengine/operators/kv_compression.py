from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import shutil

import torch

import sparseengine.platforms as platforms
from sparseengine.engine.cache_manager.base import CompressionComputeView
from sparseengine.operators.registry import (
    OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult,
    runtime_version_at_least,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class KVCompressionOpSpec:
    ratio: int
    head_dim: int
    hidden_size: int
    activation_dtype: torch.dtype
    rms_eps: float
    cuda_graph: bool

    def __post_init__(self):
        if self.ratio not in (4, 128) or self.head_dim not in (128, 512):
            raise ValueError("V4 compression requires ratio 4/128 and D128/D512")
        if self.hidden_size <= 0 or self.rms_eps <= 0:
            raise ValueError("Compression hidden size and RMS epsilon must be positive")

    @property
    def projection_width(self):
        return self.head_dim * (4 if self.ratio == 4 else 2)


KV_COMPRESSION_REGISTRY = OpRegistry(
    "KV compression",
    portfolio=PortfolioPolicy(repo_nonstandard=("sgl_v4",)),
)


@KV_COMPRESSION_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class SGLKVCompressionProvider:
    name = "sgl_v4"

    @classmethod
    def supports(cls, spec: KVCompressionOpSpec, caps: DeviceCaps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability is None:
            return SupportResult.unsupported("SGL compression requires a CUDA target")
        if not runtime_version_at_least(caps.runtime_version, (12, 0)):
            return SupportResult.unsupported("SGL compression requires CUDA runtime >= 12.0")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.unsupported("SGL compression projection requires BF16 activations")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph")
        if importlib.util.find_spec("tvm_ffi") is None:
            return SupportResult.dependency_absent("install apache-tvm-ffi 0.1.10")
        if shutil.which("nvcc") is None or shutil.which("ninja") is None:
            return SupportResult.dependency_absent("SGL compression JIT requires nvcc and ninja on PATH")
        return SupportResult.yes("pinned SGL compression computation with TVM FFI")

    def __init__(self, *, op_spec, architecture):
        from sparseengine.kernels.external.sgl.v4_compression import SGLCompressionKernels

        self.spec = op_spec
        self.kernels = SGLCompressionKernels(ratio=op_spec.ratio, head_dim=op_spec.head_dim,
                                             architecture=architecture)

    def prepare_weights(self, kv_weight, gate_weight, ape, norm_weight):
        if hasattr(self, "projection"):
            raise RuntimeError("Compression weights can only be prepared once")
        spec = self.spec
        width = spec.projection_width // 2
        if kv_weight.shape != (width, spec.hidden_size) or gate_weight.shape != kv_weight.shape:
            raise ValueError("Compression projection weights differ from the bound contract")
        if kv_weight.dtype != torch.bfloat16 or gate_weight.dtype != torch.bfloat16:
            raise TypeError("Compression projection checkpoint weights must be BF16")
        if ape.shape != (spec.ratio, width) or norm_weight.shape != (spec.head_dim,):
            raise ValueError("Compression APE or normalization weight has the wrong shape")
        if not kv_weight.is_cuda or any(t.device != kv_weight.device
                                       for t in (gate_weight, ape, norm_weight)):
            raise ValueError("Compression weights must share a CUDA device")
        if ape.dtype != torch.float32 or norm_weight.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("Compression requires FP32 APE and BF16/FP32 norm weights")
        # Physical merged projection and overlap APE order are provider-owned.
        self.projection = torch.cat((kv_weight, gate_weight)).contiguous()
        self.ape = (torch.cat((ape[:, :spec.head_dim], ape[:, spec.head_dim:]))
                    if spec.ratio == 4 else ape).float().contiguous()
        self.norm_weight = norm_weight.float().contiguous()

    def project(self, hidden_states):
        if not hasattr(self, "projection"):
            raise RuntimeError("Compression weights have not been prepared")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.spec.hidden_size:
            raise ValueError("Compression input differs from the bound hidden size")
        if hidden_states.dtype != self.spec.activation_dtype:
            raise TypeError("Compression input differs from the bound activation dtype")
        if hidden_states.device != self.projection.device:
            raise ValueError("Compression input must be on the projection device")
        # This is SGL's cuBLAS linear_bf16_fp32 path through a public Torch API.
        return torch.mm(hidden_states, self.projection.T, out_dtype=torch.float32)

    def compute(self, projected, view: CompressionComputeView, freqs):
        if view.prefill_plan is None:
            self.kernels.decode(view.carry, projected, view.output, self.ape,
                                view.rows, view.seq_lens)
            handle = view.seq_lens
        else:
            self.kernels.prefill(view.carry, projected, view.output, self.ape,
                                 view.rows, view.prefill_plan)
            handle = view.prefill_plan[0]
        self.kernels.normalize_rotate(view.output, self.norm_weight, handle, freqs,
                                      self.spec.rms_eps, is_decode=view.prefill_plan is None)
        if view.prefill_plan is not None:
            # Plan rows are [ragged token, request, position, window length].
            # Cache store slots describe only completed compression groups.
            completed_rows = handle.view(torch.int32)[:, 0].to(torch.int64)
            return view.output[completed_rows]
        return view.output


def resolve_kv_compression_provider(spec: KVCompressionOpSpec, *, device_index: int):
    caps = platforms.current_platform.get_device_caps(device_index)
    return OpResolver(KV_COMPRESSION_REGISTRY).resolve(
        spec, caps, op_spec=spec, architecture=caps.compute_capability,
    ).provider
