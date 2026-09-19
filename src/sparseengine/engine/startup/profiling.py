from __future__ import annotations

from dataclasses import dataclass

import torch

from sparseengine.platforms.interface import Platform

from .memory import (
    DeviceMemorySnapshot,
    MemoryProfileMeasurement,
    release_unused_device_memory,
)


@dataclass(frozen=True)
class CompletedMemoryProfile:
    before: DeviceMemorySnapshot
    after: DeviceMemorySnapshot
    measurement: MemoryProfileMeasurement


class StartupMemoryProfiler:
    def __init__(self, platform: Platform, device: torch.device):
        self.platform = platform
        self.device = device
        self._active: tuple[str, DeviceMemorySnapshot] | None = None

    def begin(self, phase: str) -> None:
        phase = str(phase).strip()
        if not phase:
            raise ValueError("Startup memory profile phase must be non-empty.")
        if self._active is not None:
            raise RuntimeError(
                "Startup memory profiles cannot overlap: "
                f"active={self._active[0]!r} requested={phase!r}."
            )
        release_unused_device_memory(self.platform)
        self.platform.reset_peak_memory_stats(self.device)
        self._active = (
            phase,
            DeviceMemorySnapshot.capture(self.platform, self.device),
        )

    def finish(self, phase: str) -> CompletedMemoryProfile:
        phase = str(phase).strip()
        if self._active is None:
            raise RuntimeError(
                f"Startup memory profile {phase!r} finished without begin()."
            )
        active_phase, before = self._active
        if active_phase != phase:
            raise RuntimeError(
                "Startup memory profile phase mismatch: "
                f"active={active_phase!r} finished={phase!r}."
            )
        release_unused_device_memory(self.platform)
        after = DeviceMemorySnapshot.capture(self.platform, self.device)
        self._active = None
        return CompletedMemoryProfile(
            before=before,
            after=after,
            measurement=MemoryProfileMeasurement.from_snapshots(before, after),
        )


__all__ = ["CompletedMemoryProfile", "StartupMemoryProfiler"]
