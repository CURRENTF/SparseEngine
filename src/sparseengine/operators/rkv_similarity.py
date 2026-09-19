"""Prepared R-KV similarity postprocessing; no quadratic index/cast scratch."""
from dataclasses import dataclass

import torch

from sparseengine import platforms
from sparseengine.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class RKVSimilaritySpec:
    dtype: torch.dtype

    def __post_init__(self):
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("R-KV similarity requires FP16, BF16 or FP32")


class RKVSimilarityProvider:
    def __init__(self, *, op_spec):
        self.spec = op_spec

    def validate(self, sim, row_start):
        if sim.ndim != 3 or min(sim.shape) <= 0 or not sim.is_contiguous():
            raise ValueError("R-KV similarity requires contiguous nonempty [units, rows, length]")
        if sim.dtype != self.spec.dtype:
            raise TypeError("R-KV similarity dtype differs from the bound provider")
        if row_start < 0 or row_start + sim.shape[1] > sim.shape[2]:
            raise ValueError("R-KV similarity rows are outside the resident domain")


RKV_SIMILARITY_REGISTRY = OpRegistry(
    "R-KV similarity reduction",
    portfolio=PortfolioPolicy(repo_nonstandard=("triton",), repo_portable=("torch_cpu",)),
)


@RKV_SIMILARITY_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonRKVSimilarityProvider(RKVSimilarityProvider):
    name = "triton"
    row_block = 128

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform not in (PlatformEnum.CUDA, PlatformEnum.ROCM) or not caps.supports_triton:
            return SupportResult.unsupported("requires a Triton GPU platform")
        if spec.dtype == torch.bfloat16 and not caps.supports_bfloat16:
            return SupportResult.unsupported("requires BF16 support")
        return SupportResult.yes("fused representative selection and FP32 column reduction")

    def __init__(self, *, op_spec):
        super().__init__(op_spec=op_spec)
        # Import when binding, not while executing or importing a CPU fixture.
        from sparseengine.kernels.triton.rkv_similarity import similarity_column_sums
        self._run = similarity_column_sums

    def column_sums(self, sim, row_start):
        self.validate(sim, row_start)
        if sim.device.type != "cuda":
            raise ValueError("Triton R-KV similarity requires GPU tensors")
        units, rows, length = sim.shape
        representatives = torch.empty((units, rows), device=sim.device, dtype=torch.int32)
        partial = torch.empty((units, (rows + self.row_block - 1) // self.row_block, length),
                              device=sim.device, dtype=torch.float32)
        output = torch.empty((units, length), device=sim.device, dtype=torch.float32)
        return self._run(sim, row_start, representatives, partial, output, row_block=self.row_block)


@RKV_SIMILARITY_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TorchRKVSimilarityProvider(RKVSimilarityProvider):
    """CPU implementation; explicitly usable as the diagnostic GPU baseline."""
    name = "torch_cpu"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CPU:
            return SupportResult.unsupported("Torch reference is CPU-only in production")
        return SupportResult.yes()

    def column_sums(self, sim, row_start):
        self.validate(sim, row_start)
        rows, length = sim.shape[-2:]
        # Do not mutate the matmul result, matching the fused operator contract.
        values = sim.clone()
        idx = torch.arange(rows, device=sim.device)
        values[:, idx, idx + row_start] = 0
        columns = torch.arange(length, device=sim.device)
        representatives = torch.where(values > 0.5, columns, 0).amax(dim=-1)
        values.scatter_(-1, representatives.unsqueeze(-1), 0)
        return values.float().sum(dim=-2)


def prepare_rkv_similarity_provider(dtype, *, device):
    device = torch.device(device)
    if device.type == "cpu":
        caps = DeviceCaps(platform=PlatformEnum.CPU, device_type="cpu", device_index=0, device_name="cpu")
    else:
        caps = platforms.current_platform.get_device_caps(device.index or 0)
    spec = RKVSimilaritySpec(dtype)
    return OpResolver(RKV_SIMILARITY_REGISTRY).resolve(spec, caps, op_spec=spec).provider
