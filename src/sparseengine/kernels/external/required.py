from __future__ import annotations

from sparseengine.kernels.external.flashinfer.support import (
    flashinfer_kernel_health,
    flashinfer_kernel_metadata_health,
)
from sparseengine.kernels.external.sgl.support import (
    sgl_kernel_health,
    sgl_kernel_metadata_health,
)
from sparseengine.kernels.external.support import (
    CUDA_DEPENDENCY_INSTALL_HINT,
    KernelFamilyHealth,
)


def _raise_for_unhealthy(
    health_by_family: tuple[KernelFamilyHealth, ...],
    *,
    stage: str,
) -> None:
    unhealthy = tuple(health for health in health_by_family if not health.ready)
    if not unhealthy:
        return
    details = "; ".join(
        f"{health.family} is {health.state.value}: {health.reason}"
        for health in unhealthy
    )
    raise RuntimeError(
        "SparseEngine CUDA engine requires healthy FlashInfer and SGL kernel "
        f"dependencies during {stage}, but {details}. "
        f"{CUDA_DEPENDENCY_INSTALL_HINT}"
    )


def validate_required_cuda_kernel_metadata() -> None:
    """Validate required packages before workers start without importing them."""

    _raise_for_unhealthy(
        (
            flashinfer_kernel_metadata_health(),
            sgl_kernel_metadata_health(),
        ),
        stage="startup metadata validation",
    )


def validate_required_cuda_kernel_families() -> None:
    """Import required kernel packages after the rank's CUDA device is selected."""

    _raise_for_unhealthy(
        (
            flashinfer_kernel_health(),
            sgl_kernel_health(),
        ),
        stage="device-bound binary validation",
    )


__all__ = [
    "validate_required_cuda_kernel_families",
    "validate_required_cuda_kernel_metadata",
]
