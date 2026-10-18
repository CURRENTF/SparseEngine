from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from sparseengine.kernels.external.support import (
    KernelFamilyHealth,
    KernelFamilyState,
    RequiredExternalKernelFamilyError,
)

_VERSION_REQUIREMENT = ">=0.6.18.post1,<0.7"
_SUPPORTED_VERSIONS = SpecifierSet(_VERSION_REQUIREMENT)
_DISTRIBUTION = "flashinfer-python"


def flashinfer_kernel_metadata_health() -> KernelFamilyHealth:
    """Inspect FlashInfer discovery and metadata without importing the package."""
    try:
        package_spec = importlib.util.find_spec("flashinfer")
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
    try:
        parsed = Version(version)
    except InvalidVersion:
        parsed = None
    if parsed is None or parsed not in _SUPPORTED_VERSIONS:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            version,
            f"requires {_DISTRIBUTION}{_VERSION_REQUIREMENT}, got {version}",
        )
    return KernelFamilyHealth(
        _DISTRIBUTION,
        KernelFamilyState.READY,
        version,
        f"{_DISTRIBUTION} {version} package metadata is ready",
    )


def flashinfer_kernel_health() -> KernelFamilyHealth:
    """Inspect FlashInfer metadata and its device-bound top-level import."""

    metadata_health = flashinfer_kernel_metadata_health()
    if not metadata_health.ready:
        return metadata_health
    version = metadata_health.version
    try:
        importlib.import_module("flashinfer")
    except Exception as error:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            version,
            f"package failed to load: {type(error).__name__}: {error}",
        )
    return KernelFamilyHealth(
        _DISTRIBUTION,
        KernelFamilyState.READY,
        version,
        f"{_DISTRIBUTION} {version} package family is ready",
    )


def flashinfer_kernel_support(feature: str) -> tuple[bool, str]:
    """Require a healthy FlashInfer family before probing one feature."""

    health = flashinfer_kernel_health()
    if not health.ready:
        raise RequiredExternalKernelFamilyError(health, feature=feature)
    return True, f"{_DISTRIBUTION} {health.version} {feature} is available"


__all__ = [
    "flashinfer_kernel_health",
    "flashinfer_kernel_metadata_health",
    "flashinfer_kernel_support",
]
