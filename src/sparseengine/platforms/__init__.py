from __future__ import annotations

import os
import pkgutil
from importlib.metadata import entry_points
from typing import Any

from sparseengine.platforms.cpu import CpuPlatform
from sparseengine.platforms.cuda import CudaPlatform
from sparseengine.platforms.interface import (
    AllocatorStats,
    DeviceCaps,
    Platform,
    PlatformEnum,
)
from sparseengine.platforms.rocm import RocmPlatform

PLATFORM_ENTRYPOINT_GROUP = "sparseengine.platforms"

_current_platform: Platform | None = None


def _entry_points_for_group(group: str):
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=group))
    return list(discovered.get(group, ()))


def _coerce_platform(result: Any) -> Platform:
    if isinstance(result, Platform):
        return result
    if isinstance(result, type) and issubclass(result, Platform):
        return result()
    if isinstance(result, str):
        cls = pkgutil.resolve_name(result)
        if not isinstance(cls, type) or not issubclass(cls, Platform):
            raise TypeError(f"Platform entry point returned non-Platform class: {result!r}.")
        return cls()
    raise TypeError(f"Platform entry point returned unsupported value: {result!r}.")


def _load_plugin_platform(selected: str) -> Platform:
    eps = {ep.name: ep for ep in _entry_points_for_group(PLATFORM_ENTRYPOINT_GROUP)}
    if selected not in eps:
        available = ", ".join(sorted(eps)) if eps else "none"
        raise RuntimeError(
            f"SPARSEENGINE_PLATFORM={selected!r} was not found in {PLATFORM_ENTRYPOINT_GROUP!r} "
            f"entry points. Available plugins: {available}."
        )
    activate = eps[selected].load()
    result = activate()
    if result is None:
        raise RuntimeError(f"Platform plugin {selected!r} returned None and cannot be used.")
    platform = _coerce_platform(result)
    platform.validate_environment()
    return platform


def _resolve_builtin(selected: str) -> Platform | None:
    if selected == "cuda":
        platform = CudaPlatform()
    elif selected == "rocm":
        platform = RocmPlatform()
    elif selected == "cpu":
        platform = CpuPlatform()
    else:
        return None
    platform.validate_environment()
    return platform


def _resolve_platform() -> Platform:
    selected = os.getenv("SPARSEENGINE_PLATFORM", "").strip().lower()
    if selected:
        platform = _resolve_builtin(selected)
        if platform is not None:
            return platform
        return _load_plugin_platform(selected)

    cuda = CudaPlatform()
    if cuda.check_available():
        cuda.validate_environment()
        return cuda

    rocm = RocmPlatform()
    if rocm.check_available():
        rocm.validate_environment()
        return rocm

    raise RuntimeError(
        "No supported SparseEngine platform was detected. Set SPARSEENGINE_PLATFORM=cpu only for "
        "import/unit-test paths; real inference currently requires CUDA."
    )


def get_current_platform() -> Platform:
    global _current_platform
    if _current_platform is None:
        _current_platform = _resolve_platform()
    return _current_platform


def _set_current_platform_for_tests(platform: Platform | None) -> None:
    global _current_platform
    _current_platform = platform


def __getattr__(name: str):
    if name == "current_platform":
        return get_current_platform()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AllocatorStats",
    "CpuPlatform",
    "CudaPlatform",
    "DeviceCaps",
    "Platform",
    "PlatformEnum",
    "RocmPlatform",
    "current_platform",
    "get_current_platform",
]
