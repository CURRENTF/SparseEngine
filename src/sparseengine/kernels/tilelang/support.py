"""Lightweight compatibility checks for repository-owned TileLang kernels."""

from __future__ import annotations

from importlib import metadata

from sparseengine.kernels.external.support import (
    ExternalKernelFamilyError,
    KernelFamilyHealth,
    KernelFamilyState,
)


_SUPPORTED_VERSION_PAIR = ("0.1.9", "0.1.10")


def tilelang_dependency_support() -> tuple[bool, str]:
    """Check the validated dependency pair without importing the compiler."""

    try:
        tilelang_version = metadata.version("tilelang")
    except metadata.PackageNotFoundError:
        return False, "tilelang is not installed"
    try:
        tvm_ffi_version = metadata.version("apache-tvm-ffi")
    except metadata.PackageNotFoundError:
        raise ExternalKernelFamilyError(
            KernelFamilyHealth(
                family="tilelang",
                state=KernelFamilyState.BROKEN,
                version=tilelang_version,
                reason="tilelang is installed but apache-tvm-ffi is not installed",
            ),
            feature="repository-owned kernels",
        )
    installed_pair = (tilelang_version, tvm_ffi_version)
    if installed_pair != _SUPPORTED_VERSION_PAIR:
        raise ExternalKernelFamilyError(
            KernelFamilyHealth(
                family="tilelang",
                state=KernelFamilyState.BROKEN,
                version=tilelang_version,
                reason=(
                    "requires the validated dependency pair "
                    f"tilelang=={_SUPPORTED_VERSION_PAIR[0]}, "
                    f"apache-tvm-ffi=={_SUPPORTED_VERSION_PAIR[1]}; "
                    f"got tilelang=={tilelang_version}, "
                    f"apache-tvm-ffi=={tvm_ffi_version}"
                ),
            ),
            feature="repository-owned kernels",
        )

    return True, (
        f"tilelang {tilelang_version}, apache-tvm-ffi {tvm_ffi_version}"
    )


__all__ = ["tilelang_dependency_support"]
