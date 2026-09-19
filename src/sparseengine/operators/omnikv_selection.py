"""Length-aware OmniKV history selection with a fixed graph launch contract."""
from dataclasses import dataclass

import torch

from sparseengine.operators.registry import (
    OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult,
)
from sparseengine.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class OmniKVSelectionSpec:
    sink: int
    keep: int

    def __post_init__(self):
        if self.sink < 0 or self.keep < 0:
            raise ValueError('OmniKV selection budgets must be nonnegative')


class OmniKVSelectionProvider:
    def __init__(self, *, op_spec):
        self.spec = op_spec

    def select(self, scores, lengths, k):
        if scores.ndim != 2 or scores.shape[1] < self.spec.sink:
            raise ValueError('OmniKV selection requires [batch, capacity >= sink] scores')
        if scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError('OmniKV selection requires floating-point scores')
        if lengths.shape != scores.shape[:1] or lengths.device != scores.device:
            raise ValueError('OmniKV selection requires same-device per-row lengths')
        if lengths.dtype != torch.int32 or not lengths.is_contiguous():
            raise TypeError('OmniKV selection requires contiguous int32 lengths')
        if not 0 < k <= min(self.spec.keep, scores.shape[1] - self.spec.sink):
            raise ValueError('OmniKV selection k exceeds its prepared contract')
        return self._select(scores, lengths, k)


REGISTRY = OpRegistry(
    'OmniKV history selection',
    portfolio=PortfolioPolicy(upstream_standard=('flashinfer',), repo_portable=('triton', 'torch_cpu')),
)


@REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferOmniKVSelection(OmniKVSelectionProvider):
    name = 'flashinfer'

    @classmethod
    def supports(cls, spec, caps):
        del spec
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported('requires CUDA')
        from sparseengine.kernels.external.flashinfer.topk import flashinfer_ragged_topk_support
        supported, reason = flashinfer_ragged_topk_support(caps.device_index)
        return SupportResult.yes(reason) if supported else SupportResult.dependency_absent(reason)

    def _select(self, scores, lengths, k):
        from sparseengine.kernels.external.flashinfer.topk import flashinfer_ragged_topk
        return flashinfer_ragged_topk(scores, lengths, k, sink=self.spec.sink)


@REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonOmniKVSelection(OmniKVSelectionProvider):
    name = 'triton'

    @classmethod
    def supports(cls, spec, caps):
        del spec
        if caps.platform not in (PlatformEnum.CUDA, PlatformEnum.ROCM) or not caps.supports_triton:
            return SupportResult.unsupported('requires a Triton GPU platform')
        return SupportResult.yes('radix history selection bounded by device lengths')

    def _select(self, scores, lengths, k):
        from sparseengine.kernels.triton.omnikv_score import select_omnikv_history
        return select_omnikv_history(scores, lengths, k, sink=self.spec.sink)


@REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TorchCPUOmniKVSelection(OmniKVSelectionProvider):
    name = 'torch_cpu'

    @classmethod
    def supports(cls, spec, caps):
        del spec
        return (SupportResult.yes('CPU reference') if caps.platform == PlatformEnum.CPU
                else SupportResult.unsupported('CPU reference only'))

    def _select(self, scores, lengths, k):
        output = torch.full((scores.shape[0], k), -1, dtype=torch.int32, device=scores.device)
        for row, length in enumerate(lengths.tolist()):
            if not 0 <= length <= scores.shape[1] - self.spec.sink:
                raise ValueError('OmniKV candidate length exceeds score capacity')
            if length <= k:
                indices = torch.arange(length, device=scores.device)
            else:
                indices = scores[row, self.spec.sink:self.spec.sink + length].argsort(
                    descending=True, stable=True,
                )[:k]
            output[row, :indices.numel()] = indices.to(torch.int32) + self.spec.sink
        return output


def prepare_omnikv_selection(spec, *, device):
    if device.type == 'cpu':
        from sparseengine.platforms.interface import DeviceCaps
        caps = DeviceCaps(platform=PlatformEnum.CPU, device_type='cpu', device_index=0, device_name='cpu')
    else:
        from sparseengine import platforms
        caps = platforms.current_platform.get_device_caps(device.index or 0)
    return OpResolver(REGISTRY).resolve(spec, caps, op_spec=spec).provider
