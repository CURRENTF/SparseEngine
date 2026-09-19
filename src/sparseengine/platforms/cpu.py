from __future__ import annotations

import os
from functools import lru_cache

import torch

from sparseengine.platforms.interface import DeviceCaps, Platform, PlatformEnum


class CpuPlatform(Platform):
    name = "cpu"
    device_type = "cpu"
    enum = PlatformEnum.CPU

    def check_available(self) -> bool:
        return True

    def get_device(self, local_rank: int = 0) -> torch.device:
        del local_rank
        return torch.device("cpu")

    def set_device(self, device: torch.device | int | str) -> None:
        del device
        return None

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        del device_id
        if hasattr(os, "sysconf"):
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return available_pages * page_size, total_pages * page_size
        raise NotImplementedError("CPU memory probing requires os.sysconf support.")

    @lru_cache(maxsize=None)
    def get_device_caps(self, device_index: int = 0) -> DeviceCaps:
        return DeviceCaps(
            platform=self.enum,
            device_type=self.device_type,
            device_index=int(device_index),
            device_name="cpu",
            supports_bfloat16=True,
        )
