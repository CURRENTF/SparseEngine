from __future__ import annotations

import importlib
import inspect
from functools import lru_cache

import torch

from sparseengine.kernels.external.flashinfer.support import (
    flashinfer_kernel_support,
)
from sparseengine.kernels.external.support import ExternalKernelContractError

_SM90_ARGUMENTS = (
    "input",
    "weight",
    "input_scale",
    "weight_scale",
    "out",
    "out_dtype",
)
_GROUPWISE_ARGUMENTS = (
    "a",
    "b",
    "a_scale",
    "b_scale",
    "scale_major_mode",
    "mma_sm",
    "scale_granularity_mnk",
    "out",
    "out_dtype",
    "backend",
)


def _require_callable(name: str, arguments: tuple[str, ...], feature: str):
    try:
        function = getattr(importlib.import_module("flashinfer.gemm"), name)
        actual_arguments = tuple(inspect.signature(function).parameters)
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"failed to load: {type(error).__name__}: {error}",
        ) from error
    if not callable(function):
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"flashinfer.gemm.{name} is not callable",
        )
    if actual_arguments != arguments:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"unsupported schema: {actual_arguments}",
        )
    return function


@lru_cache(maxsize=1)
def _sm90_fp8_linear_op():
    feature = "SM90 block-scale FP8 Linear"
    _, reason = flashinfer_kernel_support(feature)
    return (
        _require_callable("fp8_blockscale_gemm_sm90", _SM90_ARGUMENTS, feature),
        reason,
    )


@lru_cache(maxsize=1)
def _sm120_groupwise_fp8_linear_op():
    feature = "SM120 groupwise FP8 Linear"
    _, reason = flashinfer_kernel_support(feature)
    return (
        _require_callable("gemm_fp8_nt_groupwise", _GROUPWISE_ARGUMENTS, feature),
        reason,
    )


def flashinfer_sm90_fp8_linear_support() -> tuple[bool, str]:
    _, reason = _sm90_fp8_linear_op()
    return True, reason


def flashinfer_sm120_groupwise_fp8_linear_support() -> tuple[bool, str]:
    _, reason = _sm120_groupwise_fp8_linear_op()
    return True, reason


def flashinfer_fp8_blockscale_gemm_sm90(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    function, _ = _sm90_fp8_linear_op()
    return function(
        input,
        weight,
        weight_scale=weight_scale,
        out_dtype=out_dtype,
    )


def flashinfer_fp8_nt_groupwise_sm120(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    function, _ = _sm120_groupwise_fp8_linear_op()
    return function(
        activation,
        weight,
        activation_scale,
        weight_scale,
        scale_major_mode="K",
        scale_granularity_mnk=(1, 128, 128),
        out_dtype=out_dtype,
        backend="cutlass",
    )


__all__ = [
    "flashinfer_fp8_blockscale_gemm_sm90",
    "flashinfer_fp8_nt_groupwise_sm120",
    "flashinfer_sm90_fp8_linear_support",
    "flashinfer_sm120_groupwise_fp8_linear_support",
]
