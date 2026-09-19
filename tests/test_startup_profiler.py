from __future__ import annotations

import torch

from sparseengine.engine.startup import StartupMemoryProfiler
from sparseengine.platforms.interface import AllocatorStats, Platform


class _Platform(Platform):
    def __init__(self):
        self.free_bytes = 800
        self.allocated_bytes = 100
        self.peak_bytes = 100
        self.reset_count = 0

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return self.free_bytes, 1000

    def get_allocator_stats(self, device=None) -> AllocatorStats:
        return AllocatorStats(
            current_allocated_bytes=self.allocated_bytes,
            peak_allocated_bytes=self.peak_bytes,
        )

    def reset_peak_memory_stats(self, device=None) -> None:
        self.reset_count += 1
        self.peak_bytes = self.allocated_bytes


def test_startup_profiler_measures_one_named_phase(monkeypatch):
    platform = _Platform()
    monkeypatch.setattr(
        "sparseengine.engine.startup.profiling.release_unused_device_memory",
        lambda current: None,
    )
    profiler = StartupMemoryProfiler(platform, torch.device("cuda", 0))

    profiler.begin("prefill")
    platform.free_bytes = 750
    platform.allocated_bytes = 130
    platform.peak_bytes = 310
    result = profiler.finish("prefill")

    assert platform.reset_count == 1
    assert result.measurement.consumed_bytes == 50
    assert result.measurement.transient_peak_bytes == 180


def test_startup_profiler_rejects_overlapping_or_mismatched_phases(monkeypatch):
    platform = _Platform()
    monkeypatch.setattr(
        "sparseengine.engine.startup.profiling.release_unused_device_memory",
        lambda current: None,
    )
    profiler = StartupMemoryProfiler(platform, torch.device("cuda", 0))

    profiler.begin("prefill")
    try:
        profiler.begin("decode")
    except RuntimeError as exc:
        assert "cannot overlap" in str(exc)
    else:
        raise AssertionError("Expected overlapping startup profiles to fail.")

    try:
        profiler.finish("decode")
    except RuntimeError as exc:
        assert "phase mismatch" in str(exc)
    else:
        raise AssertionError("Expected mismatched startup profile phase to fail.")
