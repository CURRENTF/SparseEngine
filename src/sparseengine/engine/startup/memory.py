from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from sparseengine.platforms.interface import Platform


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    free_bytes: int
    total_bytes: int
    allocated_bytes: int
    peak_allocated_bytes: int

    @classmethod
    def capture(
        cls,
        platform: Platform,
        device: torch.device,
    ) -> "DeviceMemorySnapshot":
        free_bytes, total_bytes = platform.get_available_memory(device.index or 0)
        allocator = platform.get_allocator_stats(device)
        return cls(
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            allocated_bytes=int(allocator.current_allocated_bytes),
            peak_allocated_bytes=int(allocator.peak_allocated_bytes),
        )


@dataclass(frozen=True)
class MemoryProfileMeasurement:
    consumed_bytes: int
    transient_peak_bytes: int

    @classmethod
    def from_snapshots(
        cls,
        before: DeviceMemorySnapshot,
        after: DeviceMemorySnapshot,
    ) -> "MemoryProfileMeasurement":
        if before.total_bytes != after.total_bytes:
            raise RuntimeError(
                "GPU total memory changed during startup profiling: "
                f"before={before.total_bytes} after={after.total_bytes}."
            )
        return cls(
            consumed_bytes=max(0, before.free_bytes - after.free_bytes),
            transient_peak_bytes=max(
                0,
                after.peak_allocated_bytes - after.allocated_bytes,
            ),
        )


@dataclass(frozen=True)
class CacheRuntimeBuildMeasurement:
    manager_consumed_bytes: int
    manager_budget_overflow_bytes: int
    auxiliary_consumed_bytes: int

    @classmethod
    def from_snapshots(
        cls,
        before_manager: DeviceMemorySnapshot,
        after_manager: DeviceMemorySnapshot,
        after_runtime: DeviceMemorySnapshot,
        *,
        allocation_budget_bytes: int,
    ) -> "CacheRuntimeBuildMeasurement":
        totals = {
            int(before_manager.total_bytes),
            int(after_manager.total_bytes),
            int(after_runtime.total_bytes),
        }
        if len(totals) != 1:
            raise RuntimeError(
                "GPU total memory changed while building the cache runtime: "
                f"before={before_manager.total_bytes} "
                f"manager={after_manager.total_bytes} runtime={after_runtime.total_bytes}."
            )
        manager_consumed = max(
            0,
            int(before_manager.free_bytes) - int(after_manager.free_bytes),
        )
        return cls(
            manager_consumed_bytes=manager_consumed,
            manager_budget_overflow_bytes=max(
                0,
                manager_consumed - int(allocation_budget_bytes),
            ),
            auxiliary_consumed_bytes=max(
                0,
                int(after_manager.free_bytes) - int(after_runtime.free_bytes),
            ),
        )


def release_unused_device_memory(platform: Platform) -> None:
    platform.synchronize()
    gc.collect()
    platform.empty_cache()
    platform.synchronize()


__all__ = [
    "CacheRuntimeBuildMeasurement",
    "DeviceMemorySnapshot",
    "MemoryProfileMeasurement",
    "release_unused_device_memory",
]
