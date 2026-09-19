# SPDX-License-Identifier: Apache-2.0
"""Lightweight dependency checks for the portable TileLang GQA kernels."""

from sparseengine.kernels.tilelang.support import tilelang_dependency_support


def tilelang_gqa_device_support(device_index: int | None = None) -> tuple[bool, str]:
    """Check compatible dependencies without importing the TileLang compiler."""

    del device_index
    return tilelang_dependency_support()


__all__ = ["tilelang_gqa_device_support"]
