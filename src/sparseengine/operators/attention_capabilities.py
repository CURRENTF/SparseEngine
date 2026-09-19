from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch

from sparseengine.operators.registry import SupportResult, runtime_version_at_least
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


class AttentionScoreKind(Enum):
    NONE = auto()
    RAW_QK_PER_HEAD = auto()
    RAW_QK_REDUCED = auto()
    ATTENTION_PROBABILITY_REDUCED = auto()


@dataclass(frozen=True, slots=True)
class AttentionKernelRequest:
    activation_dtype: torch.dtype
    head_dim: int
    page_size: int | None = None
    score_output: AttentionScoreKind | None = None
    requires_softmax_lse: bool = False
    layer_varying_page_table: bool = False
    varlen: bool = True
    cuda_graph: bool = False


@dataclass(frozen=True, slots=True)
class AttentionKernelCapabilities:
    platforms: frozenset[PlatformEnum]
    activation_dtypes: frozenset[torch.dtype]
    compute_capabilities: frozenset[tuple[int, int]] | None = None
    head_dims: frozenset[int] | None = None
    page_sizes: frozenset[int] | None = None
    score_outputs: frozenset[AttentionScoreKind] = frozenset(
        {AttentionScoreKind.NONE}
    )
    returns_softmax_lse: bool = False
    layer_varying_page_table: bool = False
    varlen: bool = True
    cuda_graph: bool = False
    requires_triton: bool = False
    minimum_runtime_version: tuple[int, int] | None = None


def match_attention_capabilities(
    request: AttentionKernelRequest,
    caps: DeviceCaps,
    capabilities: AttentionKernelCapabilities,
) -> SupportResult:
    if caps.platform not in capabilities.platforms:
        allowed = ", ".join(
            sorted(item.name for item in capabilities.platforms)
        )
        return SupportResult.unsupported(
            f"requires platform in {{{allowed}}}, got {caps.platform.name}"
        )
    if (
        capabilities.compute_capabilities is not None
        and caps.compute_capability not in capabilities.compute_capabilities
    ):
        return SupportResult.unsupported(
            "requires compute capability in "
            f"{sorted(capabilities.compute_capabilities)}, got {caps.compute_capability}"
        )
    if capabilities.minimum_runtime_version is not None and not runtime_version_at_least(
        caps.runtime_version,
        capabilities.minimum_runtime_version,
    ):
        minimum = ".".join(map(str, capabilities.minimum_runtime_version))
        return SupportResult.unsupported(
            f"requires runtime >= {minimum}, got {caps.runtime_version or 'unknown'}"
        )
    if request.activation_dtype not in capabilities.activation_dtypes:
        return SupportResult.unsupported(
            f"unsupported activation dtype {request.activation_dtype}; "
            f"supported={sorted(map(str, capabilities.activation_dtypes))}"
        )
    if request.activation_dtype == torch.bfloat16 and not caps.supports_bfloat16:
        return SupportResult.unsupported("device does not support BF16")
    if capabilities.head_dims is not None and request.head_dim not in capabilities.head_dims:
        return SupportResult.unsupported(
            f"unsupported head_dim={request.head_dim}; "
            f"supported={sorted(capabilities.head_dims)}"
        )
    if (
        request.page_size is not None
        and capabilities.page_sizes is not None
        and request.page_size not in capabilities.page_sizes
    ):
        return SupportResult.unsupported(
            f"unsupported page_size={request.page_size}; "
            f"supported={sorted(capabilities.page_sizes)}"
        )
    if (
        request.score_output is not None
        and request.score_output not in capabilities.score_outputs
    ):
        return SupportResult.unsupported(
            f"does not produce attention score {request.score_output.name}"
        )
    if request.requires_softmax_lse and not capabilities.returns_softmax_lse:
        return SupportResult.unsupported("does not return softmax LSE")
    if request.layer_varying_page_table and not capabilities.layer_varying_page_table:
        return SupportResult.unsupported("does not support layer-varying page tables")
    if request.varlen and not capabilities.varlen:
        return SupportResult.unsupported("does not support variable-length batches")
    if request.cuda_graph and not capabilities.cuda_graph:
        return SupportResult.unsupported("does not support CUDA Graph execution")
    if capabilities.requires_triton and not caps.supports_triton:
        return SupportResult.unsupported("platform does not support Triton")
    return SupportResult.yes()


__all__ = [
    "AttentionKernelCapabilities",
    "AttentionKernelRequest",
    "AttentionScoreKind",
    "match_attention_capabilities",
]
