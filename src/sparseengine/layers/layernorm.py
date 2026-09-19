from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Callable

import torch
from torch import nn

from sparseengine.operators.registry import record_operator_binding


RMSNormFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
FusedAddRMSNormFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    None,
]


@dataclass(frozen=True)
class _RMSNormOps:
    provider_name: str
    rmsnorm: RMSNormFn
    fused_add_rmsnorm: FusedAddRMSNormFn


def _load_flashinfer_ops(*, zero_centered_weight: bool) -> _RMSNormOps:
    module = import_module("flashinfer.norm")
    if zero_centered_weight:
        rmsnorm = getattr(module, "gemma_rmsnorm")
        fused_add_rmsnorm = getattr(module, "gemma_fused_add_rmsnorm")
    else:
        rmsnorm = getattr(module, "rmsnorm")
        fused_add_rmsnorm = getattr(module, "fused_add_rmsnorm")

    def run_rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return rmsnorm(x, weight, eps=eps)

    def run_fused_add_rmsnorm(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> None:
        fused_add_rmsnorm(x, residual, weight, eps=eps)

    return _RMSNormOps(
        provider_name="flashinfer",
        rmsnorm=run_rmsnorm,
        fused_add_rmsnorm=run_fused_add_rmsnorm,
    )


def _load_triton_ops(*, zero_centered_weight: bool) -> _RMSNormOps:
    from sparseengine.kernels.triton.rmsnorm import (
        fused_add_rmsnorm_forward,
        rmsnorm_forward,
    )

    def run_rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return rmsnorm_forward(
            x,
            weight,
            eps,
            zero_centered_weight=zero_centered_weight,
        )

    def run_fused_add_rmsnorm(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> None:
        fused_add_rmsnorm_forward(
            x,
            residual,
            weight,
            eps,
            zero_centered_weight=zero_centered_weight,
        )

    return _RMSNormOps(
        provider_name="triton",
        rmsnorm=run_rmsnorm,
        fused_add_rmsnorm=run_fused_add_rmsnorm,
    )


@lru_cache(maxsize=6)
def _resolve_rmsnorm_ops(
    *,
    zero_centered_weight: bool,
    provider: str = "auto",
) -> _RMSNormOps:
    if provider not in {"auto", "flashinfer", "triton"}:
        raise ValueError(
            "SPARSEENGINE_RMSNORM_PROVIDER must be one of "
            f"'auto', 'flashinfer', or 'triton', got {provider!r}."
        )
    if provider == "triton":
        return _load_triton_ops(zero_centered_weight=zero_centered_weight)
    if provider == "flashinfer" or find_spec("flashinfer") is not None:
        return _load_flashinfer_ops(
            zero_centered_weight=zero_centered_weight,
        )
    return _load_triton_ops(zero_centered_weight=zero_centered_weight)


class RMSNorm(nn.Module):
    """CUDA RMSNorm with FlashInfer preferred over a local Triton baseline."""

    zero_centered_weight = False

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        provider = (
            os.environ.get(
                "SPARSEENGINE_RMSNORM_PROVIDER",
                "auto",
            )
            .strip()
            .lower()
        )
        self._ops = _resolve_rmsnorm_ops(
            zero_centered_weight=self.zero_centered_weight,
            provider=provider,
        )
        record_operator_binding("RMSNorm", self)

    @property
    def provider_name(self) -> str:
        return self._ops.provider_name

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._ops.rmsnorm(x, self.weight, self.eps)
        self._ops.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class ColumnParallelRMSNorm(RMSNorm):
    """RMSNorm over a feature dimension sharded across attention TP ranks."""

    def __init__(
        self,
        global_hidden_size: int,
        eps: float = 1e-6,
        *,
        parallel_context=None,
    ) -> None:
        if parallel_context is None:
            from sparseengine.distributed import get_parallel_context

            parallel_context = get_parallel_context()
        self.parallel_context = parallel_context
        self.global_hidden_size = int(global_hidden_size)
        self.tp_rank = int(self.parallel_context.attn_tp_rank)
        self.tp_size = int(self.parallel_context.attn_tp_size)
        if self.global_hidden_size % self.tp_size:
            raise ValueError(
                "Column-parallel RMSNorm size must be divisible by attention TP, "
                f"got {self.global_hidden_size} and {self.tp_size}."
            )
        super().__init__(self.global_hidden_size // self.tp_size, eps=eps)

    def rank_local_weight_slice(
        self,
        source_shape: tuple[int, ...],
        **_: object,
    ) -> tuple[slice, ...] | None:
        expected = (self.global_hidden_size,)
        if tuple(source_shape) != expected:
            raise ValueError(
                "Column-parallel RMSNorm checkpoint shape mismatch: "
                f"expected={expected}, got={source_shape}."
            )
        if self.tp_size == 1:
            return None
        local_size = self.global_hidden_size // self.tp_size
        start = self.tp_rank * local_size
        return (slice(start, start + local_size),)

    def _apply_global_rms(
        self,
        x: torch.Tensor,
        global_square_sum: torch.Tensor,
    ) -> torch.Tensor:
        inv_rms = torch.rsqrt(global_square_sum / self.global_hidden_size + self.eps)
        return (x.float() * inv_rms.unsqueeze(-1)).to(x.dtype) * self.weight.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_sum = x.float().square().sum(dim=-1)
        self.parallel_context.attn_tp.all_reduce(square_sum)
        return self._apply_global_rms(x, square_sum)

    def forward_pair(
        self,
        x: torch.Tensor,
        other: torch.Tensor,
        other_norm: "ColumnParallelRMSNorm",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.parallel_context is not other_norm.parallel_context:
            raise ValueError("Paired column-parallel RMSNorms must share a context.")
        if x.is_cuda or other.is_cuda:
            from sparseengine.kernels.triton.column_parallel_rmsnorm import (
                paired_rms_apply,
                paired_square_sums,
            )

            square_sums = paired_square_sums(x, other)
            self.parallel_context.attn_tp.all_reduce(square_sums)
            return paired_rms_apply(
                x,
                other,
                square_sums,
                self.weight,
                other_norm.weight,
                x_global_hidden_size=self.global_hidden_size,
                other_global_hidden_size=other_norm.global_hidden_size,
                x_eps=self.eps,
                other_eps=other_norm.eps,
            )
        square_sums = torch.stack(
            (
                x.float().square().sum(dim=-1),
                other.float().square().sum(dim=-1),
            ),
            dim=-1,
        )
        self.parallel_context.attn_tp.all_reduce(square_sums)
        return (
            self._apply_global_rms(x, square_sums[..., 0]),
            other_norm._apply_global_rms(other, square_sums[..., 1]),
        )


class GemmaRMSNorm(RMSNorm):
    """RMSNorm with the Hugging Face ``1 + weight`` convention."""

    zero_centered_weight = True

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(hidden_size, eps=eps)
        nn.init.zeros_(self.weight)
