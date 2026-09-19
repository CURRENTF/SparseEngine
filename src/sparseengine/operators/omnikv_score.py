"""Prepared candidate-domain score reduction for OmniKV decode."""
from dataclasses import dataclass
import math

import torch

from sparseengine import platforms
from sparseengine.platforms import device_runtime
from sparseengine.platforms.interface import PlatformEnum
from sparseengine.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult


@dataclass(frozen=True)
class OmniKVScoreSpec:
    sink: int
    recent: int
    scale: float
    output_dtype: torch.dtype

    def __post_init__(self):
        if self.sink < 0 or self.recent < 0 or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('OmniKV scores require nonnegative budgets and a positive finite scale')
        if self.output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError('OmniKV token scores require FP32, FP16 or BF16')


@dataclass
class ScoreWorkspace:
    partial: torch.Tensor
    stats: torch.Tensor
    output: torch.Tensor

    def tensors(self):
        return (self.partial, self.stats, self.output)


class OmniKVScoreProvider:
    def __init__(self, *, op_spec):
        self.spec = op_spec
        self.workspaces: dict[int, ScoreWorkspace] = {}

    def clear(self):
        self.workspaces.clear()

    def keepalive_tensors(self):
        return [tensor for workspace in self.workspaces.values() for tensor in workspace.tensors()]

    def prepare(self, scores, *, slot):
        if scores.ndim != 3 or min(scores.shape) <= 0:
            raise ValueError('OmniKV raw scores require nonempty [batch, heads, capacity]')
        if scores.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError('OmniKV raw scores require a floating-point score dtype')
        if self.spec.sink > scores.shape[-1]:
            raise ValueError('OmniKV sink exceeds score capacity')


OMNIKV_SCORE_REGISTRY = OpRegistry(
    'OmniKV decode score reduction',
    portfolio=PortfolioPolicy(repo_nonstandard=('triton_candidate_softmax',), repo_portable=('torch_cpu',)),
)


@OMNIKV_SCORE_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonOmniKVScoreProvider(OmniKVScoreProvider):
    name = 'triton_candidate_softmax'
    block = 2048
    output_block = 256

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform not in (PlatformEnum.CUDA, PlatformEnum.ROCM) or not caps.supports_triton:
            return SupportResult.unsupported('requires a Triton GPU platform')
        if spec.output_dtype == torch.bfloat16 and not caps.supports_bfloat16:
            return SupportResult.unsupported('device does not support BF16')
        return SupportResult.yes('candidate softmax plus head-max with device-side length masking')

    def prepare(self, scores, *, slot):
        super().prepare(scores, slot=slot)
        if scores.device.type != 'cuda':
            raise ValueError('Triton OmniKV score inputs must be GPU tensors')
        batch, heads, capacity = scores.shape
        splits = max(1, (capacity - self.spec.sink + self.block - 1) // self.block)
        sizes = (batch * heads * splits * 2, batch * heads * 2, batch * capacity)
        workspace = self.workspaces.get(slot)
        if workspace is None or any(
            tensor.device != scores.device or tensor.numel() < size
            for tensor, size in zip(workspace.tensors(), sizes)
        ):
            if device_runtime.is_stream_capturing():
                raise RuntimeError('OmniKV score workspace must be prepared before Graph capture')
            # Flat grow-only storage avoids retaining one allocation per eager
            # context length. Captured graphs retain older allocations via keepalive.
            workspace = ScoreWorkspace(*(
                torch.empty(size, device=scores.device, dtype=dtype)
                for size, dtype in zip(
                    sizes, (torch.float32, torch.float32, self.spec.output_dtype)
                )
            ))
            # Selection-only execution leaves short rows and unused tails unread.
            workspace.output.fill_(torch.finfo(self.spec.output_dtype).min)
            self.workspaces[slot] = workspace

    def run(self, scores, lengths, *, slot, selection_keep=-1):
        from sparseengine.kernels.triton.omnikv_score import launch_omnikv_decode_scores

        workspace = self.workspaces[slot]
        batch, _, capacity = scores.shape
        if (
            lengths.shape != (batch,)
            or lengths.dtype not in (torch.int32, torch.int64)
            or lengths.device != scores.device
        ):
            raise ValueError('OmniKV requires one integer context length per score row on the same device')
        output = workspace.output[:batch * capacity].view(batch, capacity)
        if capacity == self.spec.sink:
            output.fill_(torch.finfo(output.dtype).min)
            return output
        launch_omnikv_decode_scores(
            scores, lengths, workspace.partial, workspace.stats, output,
            sink=self.spec.sink, recent=self.spec.recent, scale=self.spec.scale,
            min_score=torch.finfo(output.dtype).min,
            block=self.block, output_block=self.output_block, selection_keep=selection_keep,
        )
        return output

    def binding_metadata(self):
        return {
            'implementation_source': 'repository_triton',
            'algorithm': 'split_candidate_softmax_head_max',
            'block': self.block,
            'output_block': self.output_block,
            'dynamic_lengths': 'device_only',
        }


@OMNIKV_SCORE_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TorchCPUOmniKVScoreProvider(OmniKVScoreProvider):
    """CPU semantic reference; never selected as a GPU execution fallback."""
    name = 'torch_cpu'

    @classmethod
    def supports(cls, spec, caps):
        del spec
        if caps.platform != PlatformEnum.CPU:
            return SupportResult.unsupported('CPU reference only')
        return SupportResult.yes('CPU reference for sparse-runtime lifecycle tests')

    def run(self, scores, lengths, *, slot, selection_keep=-1):
        del slot
        candidate = scores[:, :, self.spec.sink:]
        lens = (lengths.to(torch.long) - self.spec.recent - self.spec.sink).clamp(0, candidate.shape[-1])
        valid = torch.arange(candidate.shape[-1], device=scores.device)[None,:] < lens[:,None]
        logits = (candidate.float() * self.spec.scale).masked_fill(~valid[:,None,:], torch.finfo(torch.float32).min)
        reduced = logits.softmax(-1).amax(1).to(self.spec.output_dtype)
        minimum = torch.finfo(reduced.dtype).min
        output = torch.full((scores.shape[0],scores.shape[2]),minimum,device=scores.device,dtype=reduced.dtype)
        output[:,self.spec.sink:] = reduced.masked_fill(~valid,minimum)
        return output


def prepare_omnikv_score_provider(spec, *, device):
    # Explicit CPU fixtures can coexist with a CUDA-capable host.
    if device.type == 'cpu':
        from sparseengine.platforms.interface import DeviceCaps
        caps = DeviceCaps(platform=PlatformEnum.CPU, device_type='cpu', device_index=0, device_name='cpu')
    else:
        caps = platforms.current_platform.get_device_caps(device.index or 0)
    return OpResolver(OMNIKV_SCORE_REGISTRY).resolve(spec, caps, op_spec=spec).provider
