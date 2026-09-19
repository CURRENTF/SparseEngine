from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util

from sparseengine.kernels.external.support import (
    KernelFamilyHealth,
    KernelFamilyState,
    RequiredExternalKernelFamilyError,
)

_DISTRIBUTION = "sglang-kernel"
_REQUIRED_VERSION = "0.4.5"


def sgl_kernel_metadata_health() -> KernelFamilyHealth:
    """Inspect SGL discovery and metadata without importing device-bound ops."""
    try:
        package_spec = importlib.util.find_spec("sgl_kernel")
    except (ImportError, ValueError) as error:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            None,
            f"package discovery failed: {type(error).__name__}: {error}",
        )
    if package_spec is None:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.ABSENT,
            None,
            f"{_DISTRIBUTION} is not installed",
        )
    try:
        version = importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            None,
            f"{_DISTRIBUTION} package metadata is unavailable",
        )
    if version != _REQUIRED_VERSION:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            version,
            f"requires {_DISTRIBUTION}=={_REQUIRED_VERSION}, got {version}",
        )
    return KernelFamilyHealth(
        _DISTRIBUTION,
        KernelFamilyState.READY,
        version,
        f"{_DISTRIBUTION} {version} package metadata is ready",
    )


def sgl_kernel_health() -> KernelFamilyHealth:
    """Inspect SGL metadata and import ops for the already-selected device."""

    metadata_health = sgl_kernel_metadata_health()
    if not metadata_health.ready:
        return metadata_health
    version = metadata_health.version
    try:
        importlib.import_module("sgl_kernel")
    except Exception as error:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            version,
            f"binary failed to load: {type(error).__name__}: {error}",
        )
    return KernelFamilyHealth(
        _DISTRIBUTION,
        KernelFamilyState.READY,
        version,
        f"{_DISTRIBUTION} {version} binary family is ready",
    )


def sgl_kernel_support(feature: str) -> tuple[bool, str]:
    """Require a healthy SGL binary family before checking a feature contract."""

    health = sgl_kernel_health()
    if not health.ready:
        raise RequiredExternalKernelFamilyError(health, feature=feature)
    return True, f"{_DISTRIBUTION} {health.version} {feature} is available"


__all__ = [
    "sgl_kernel_health",
    "sgl_kernel_metadata_health",
    "sgl_kernel_support",
]
