from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import sparseengine.platforms as platforms
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class SiluAndMulSpec:
    activation_dtype: torch.dtype
    input_ndim: int = 2
    contiguous: bool = True

    def __post_init__(self) -> None:
        if int(self.input_ndim) <= 0:
            raise ValueError("SiluAndMul input_ndim must be positive.")


def _validate_input(x: torch.Tensor) -> None:
    if int(x.shape[-1]) % 2:
        raise ValueError(
            "SiluAndMul requires an even final dimension, got "
            f"{int(x.shape[-1])}."
        )


def _validate_bound_input(x: torch.Tensor, spec: SiluAndMulSpec) -> None:
    _validate_input(x)
    if x.dtype != spec.activation_dtype:
        raise TypeError(
            "Bound SiluAndMul provider requires "
            f"dtype={spec.activation_dtype}, got {x.dtype}."
        )
    if x.ndim != int(spec.input_ndim):
        raise ValueError(
            "Bound SiluAndMul provider requires "
            f"ndim={spec.input_ndim}, got {x.ndim}."
        )
    if spec.contiguous and not x.is_contiguous():
        raise ValueError("Bound SiluAndMul provider requires contiguous input.")


class SiluAndMulProvider:
    name = ""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


SILU_AND_MUL_REGISTRY: OpRegistry[SiluAndMulSpec, SiluAndMulProvider] = OpRegistry(
    "SiLU-and-multiply",
    portfolio=PortfolioPolicy(repo_portable=("triton", "torch")),
)


@SILU_AND_MUL_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TritonSiluAndMulProvider(SiluAndMulProvider):
    name = "triton"

    def __init__(self, *, op_spec: SiluAndMulSpec) -> None:
        self.spec = op_spec

    @classmethod
    def supports(cls, spec: SiluAndMulSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.activation_dtype not in (torch.float16, torch.bfloat16):
            return SupportResult.unsupported(
                "requires FP16 or BF16 activations, "
                f"got {spec.activation_dtype}"
            )
        if int(spec.input_ndim) != 2 or not spec.contiguous:
            return SupportResult.unsupported("requires contiguous rank-2 inputs")
        return SupportResult.yes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _validate_bound_input(x, self.spec)
        if not x.is_cuda:
            raise ValueError("Triton SiluAndMul provider requires a CUDA input.")
        from sparseengine.kernels.triton.silu_and_mul import silu_and_mul_fwd

        return silu_and_mul_fwd(x)


@SILU_AND_MUL_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TorchSiluAndMulProvider(SiluAndMulProvider):
    name = "torch"

    def __init__(self, *, op_spec: SiluAndMulSpec | None = None) -> None:
        self.spec = op_spec

    @classmethod
    def supports(cls, spec: SiluAndMulSpec, caps: DeviceCaps) -> SupportResult:
        del spec, caps
        return SupportResult.yes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec is None:
            _validate_input(x)
        else:
            _validate_bound_input(x, self.spec)
        gate, up = x.chunk(2, -1)
        F.silu(gate, inplace=True)
        gate.mul_(up)
        return gate


def resolve_silu_and_mul_provider(
    *,
    activation_dtype: torch.dtype,
    input_ndim: int = 2,
    contiguous: bool = True,
    device_index: int | None = None,
) -> SiluAndMulProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    spec = SiluAndMulSpec(
        activation_dtype=activation_dtype,
        input_ndim=int(input_ndim),
        contiguous=bool(contiguous),
    )
    return OpResolver(SILU_AND_MUL_REGISTRY).resolve(
        spec,
        caps,
        op_spec=spec,
    ).provider


__all__ = [
    "SILU_AND_MUL_REGISTRY",
    "SiluAndMulProvider",
    "SiluAndMulSpec",
    "TorchSiluAndMulProvider",
    "TritonSiluAndMulProvider",
    "resolve_silu_and_mul_provider",
]
