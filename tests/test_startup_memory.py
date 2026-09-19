from __future__ import annotations

import torch

from sparseengine.engine.startup import (
    CacheRuntimeBuildMeasurement,
    DeviceMemorySnapshot,
    MemoryProfileMeasurement,
    release_unused_device_memory,
)
from sparseengine.platforms.interface import AllocatorStats, Platform


class _RecordingPlatform(Platform):
    def __init__(self):
        self.calls: list[str] = []

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def empty_cache(self) -> None:
        self.calls.append("empty_cache")

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        assert device_id == 0
        return 700, 1000

    def get_allocator_stats(self, device=None) -> AllocatorStats:
        assert device == torch.device("cuda", 0)
        return AllocatorStats(
            current_allocated_bytes=200,
            peak_allocated_bytes=350,
        )


def test_release_unused_device_memory_synchronizes_around_empty_cache(monkeypatch):
    platform = _RecordingPlatform()
    collected = []
    monkeypatch.setattr(
        "sparseengine.engine.startup.memory.gc.collect",
        lambda: collected.append(True),
    )

    release_unused_device_memory(platform)

    assert collected == [True]
    assert platform.calls == ["synchronize", "empty_cache", "synchronize"]


def test_device_memory_snapshot_reads_driver_and_allocator_state():
    snapshot = DeviceMemorySnapshot.capture(
        _RecordingPlatform(),
        torch.device("cuda", 0),
    )

    assert snapshot == DeviceMemorySnapshot(
        free_bytes=700,
        total_bytes=1000,
        allocated_bytes=200,
        peak_allocated_bytes=350,
    )


def test_memory_profile_separates_consumed_and_transient_bytes():
    before = DeviceMemorySnapshot(
        free_bytes=700,
        total_bytes=1000,
        allocated_bytes=200,
        peak_allocated_bytes=200,
    )
    after = DeviceMemorySnapshot(
        free_bytes=650,
        total_bytes=1000,
        allocated_bytes=230,
        peak_allocated_bytes=410,
    )

    measurement = MemoryProfileMeasurement.from_snapshots(before, after)

    assert measurement.consumed_bytes == 50
    assert measurement.transient_peak_bytes == 180


def test_cache_runtime_build_separates_budget_overflow_and_auxiliary_state():
    measurement = CacheRuntimeBuildMeasurement.from_snapshots(
        DeviceMemorySnapshot(900, 1000, 100, 100),
        DeviceMemorySnapshot(680, 1000, 320, 320),
        DeviceMemorySnapshot(650, 1000, 350, 350),
        allocation_budget_bytes=200,
    )

    assert measurement.manager_consumed_bytes == 220
    assert measurement.manager_budget_overflow_bytes == 20
    assert measurement.auxiliary_consumed_bytes == 30


def test_memory_profile_rejects_changed_total_memory():
    before = DeviceMemorySnapshot(700, 1000, 200, 200)
    after = DeviceMemorySnapshot(650, 900, 230, 410)

    try:
        MemoryProfileMeasurement.from_snapshots(before, after)
    except RuntimeError as exc:
        assert "total memory changed" in str(exc)
    else:
        raise AssertionError("Expected changed total memory to fail startup profiling.")
