"""Public SGL per-token quantization and channel-scaled CUTLASS GEMM adapter."""
from functools import lru_cache
import inspect

from sparseengine.kernels.external.sgl.support import sgl_kernel_support
from sparseengine.kernels.external.support import ExternalKernelContractError


@lru_cache(maxsize=1)
def tensor_fp8_ops():
    sgl_kernel_support("tensor-scaled FP8 Linear")
    from sgl_kernel.gemm import fp8_scaled_mm, sgl_per_token_quant_fp8

    for function, expected in (
        (fp8_scaled_mm, ("mat_a", "mat_b", "scales_a", "scales_b", "out_dtype", "bias")),
        (sgl_per_token_quant_fp8, ("input", "output_q", "output_s")),
    ):
        if tuple(inspect.signature(function).parameters) != expected:
            raise ExternalKernelContractError(
                "sglang-kernel", "tensor-scaled FP8 Linear",
                f"unsupported schema for {function.__name__}: {inspect.signature(function)}",
            )
    return sgl_per_token_quant_fp8, fp8_scaled_mm
