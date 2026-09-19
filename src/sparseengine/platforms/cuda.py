from __future__ import annotations

from functools import lru_cache

import torch

from sparseengine.platforms.interface import (
    AllocatorStats,
    DeviceCaps,
    Platform,
    PlatformEnum,
)


class CudaPlatform(Platform):
    name = "cuda"
    device_type = "cuda"
    enum = PlatformEnum.CUDA

    def check_available(self) -> bool:
        return bool(torch.cuda.is_available() and torch.version.hip is None)

    def validate_environment(self) -> None:
        if not self.check_available():
            raise RuntimeError("CUDA platform was selected, but torch.cuda is unavailable or is backed by ROCm.")

    def supports_inference(self) -> bool:
        return True

    def get_device(self, local_rank: int = 0) -> torch.device:
        return torch.device(self.device_type, int(local_rank))

    def set_device(self, device: torch.device | int | str) -> None:
        torch.cuda.set_device(device)

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.cuda.mem_get_info(int(device_id))

    def get_allocator_stats(self, device: torch.device | None = None) -> AllocatorStats:
        stats = torch.cuda.memory_stats(device)
        return AllocatorStats(
            peak_allocated_bytes=int(stats.get("allocated_bytes.all.peak", 0)),
            current_allocated_bytes=int(stats.get("allocated_bytes.all.current", 0)),
        )

    def reset_peak_memory_stats(self, device: torch.device | None = None) -> None:
        torch.cuda.reset_peak_memory_stats(device)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_stream_capturing(self) -> bool:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())

    def get_distributed_backend(self) -> str:
        return "nccl"

    def barrier_device_ids(self, rank: int) -> list[int] | None:
        return [int(rank)]

    def supports_nvlink_group(self, device_indices: tuple[int, ...]) -> bool:
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("NVLink validation requires optional nvidia-ml-py.") from exc
        try:
            pynvml.nvmlInit()
            try:
                handles = [
                    pynvml.nvmlDeviceGetHandleByUUID(str(torch.cuda.get_device_properties(i).uuid))
                    for i in device_indices
                ]
                return all(
                    torch.cuda.can_device_access_peer(device_indices[i], device_indices[j])
                    and pynvml.nvmlDeviceGetP2PStatus(
                        left, right, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    ) == pynvml.NVML_P2P_STATUS_OK
                    for i, left in enumerate(handles)
                    for j, right in enumerate(handles)
                    if i != j
                )
            finally:
                pynvml.nvmlShutdown()
        except pynvml.NVMLError as exc:
            raise RuntimeError(f"NVLink topology validation failed: {exc}") from exc

    def supports_multicast(self, device_index: int) -> bool:
        # Called only by providers whose cuda-python dependency is available.
        from cuda.bindings import driver

        status, supported = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
            int(device_index),
        )
        if status != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA multicast capability query failed: {status}")
        return bool(supported)

    def supports_peer_atomics(self, device_index: int, peer_index: int) -> bool:
        from cuda.bindings import driver

        for attribute in (
            driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED,
            driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
        ):
            status, supported = driver.cuDeviceGetP2PAttribute(
                attribute, int(device_index), int(peer_index),
            )
            if status != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA peer capability query failed: {status}")
            if not supported:
                return False
        return True

    @lru_cache(maxsize=None)
    def get_device_caps(self, device_index: int = 0) -> DeviceCaps:
        device_index = int(device_index)
        major, minor = torch.cuda.get_device_capability(device_index)
        try:
            multi_processor_count = int(
                torch.cuda.get_device_properties(device_index).multi_processor_count
            )
        except (AssertionError, RuntimeError):
            # Capability-only unit tests may stub the public capability probes
            # without initializing a CUDA driver. This optional performance
            # fact is resolved on real devices and may remain unknown otherwise.
            multi_processor_count = None
        return DeviceCaps(
            platform=self.enum,
            device_type=self.device_type,
            device_index=device_index,
            device_name=str(torch.cuda.get_device_name(device_index)),
            compute_capability=(int(major), int(minor)),
            runtime_version=torch.version.cuda,
            supports_graph_capture=True,
            supports_torch_compile=True,
            supports_triton=True,
            supports_pin_memory=True,
            supports_bfloat16=(int(major), int(minor)) >= (8, 0),
            # Ada (SM89), Hopper and Blackwell provide native FP8 tensor cores.
            supports_native_fp8=(int(major), int(minor)) >= (8, 9),
            multi_processor_count=multi_processor_count,
        )

    def get_default_attention_backend(self) -> str:
        return "cuda_triton"

    def get_decode_graph_runner_cls(self):
        from sparseengine.engine.decode_cuda_graph import DecodeCudaGraphRunner

        return DecodeCudaGraphRunner
