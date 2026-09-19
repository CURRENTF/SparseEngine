from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util

from sparseengine.kernels.external.support import (
    ExternalKernelFamilyError,
    KernelFamilyHealth,
    KernelFamilyState,
)

_DISTRIBUTION = "flashprefill"
_REQUIRED_VERSION = "3.0.0"


def flashprefill_v2_health() -> KernelFamilyHealth:
    try:
        package_spec = importlib.util.find_spec("flashprefill")
        index_spec = importlib.util.find_spec("flash_block_sparse_index_triton")
    except (ImportError, ValueError) as error:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            None,
            f"package discovery failed: {type(error).__name__}: {error}",
        )
    if package_spec is None or index_spec is None:
        missing = []
        if package_spec is None:
            missing.append("flashprefill")
        if index_spec is None:
            missing.append("flash_block_sparse_index_triton")
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.ABSENT,
            None,
            f"required modules are not installed: {', '.join(missing)}",
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
    try:
        importlib.import_module("flashprefill")
        importlib.import_module("flash_block_sparse_index_triton")
    except Exception as error:
        return KernelFamilyHealth(
            _DISTRIBUTION,
            KernelFamilyState.BROKEN,
            version,
            f"binary family failed to load: {type(error).__name__}: {error}",
        )
    return KernelFamilyHealth(
        _DISTRIBUTION,
        KernelFamilyState.READY,
        version,
        f"{_DISTRIBUTION} {version} binary family is ready",
    )


def flashprefill_v2_support(feature: str = "V2 paged prefill") -> tuple[bool, str]:
    health = flashprefill_v2_health()
    if not health.ready:
        raise ExternalKernelFamilyError(health, feature=feature)
    return True, f"{_DISTRIBUTION} {health.version} {feature} is available"


__all__ = ["flashprefill_v2_health", "flashprefill_v2_support"]
