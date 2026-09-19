from __future__ import annotations

import ctypes
import importlib.util
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from sparseengine import platforms
from sparseengine.kernels.external.flashinfer.support import flashinfer_kernel_support
from sparseengine.kernels.external.support import (
    KernelFamilyHealth,
    KernelFamilyState,
    RequiredExternalKernelFamilyError,
)
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProfileMatch,
    ProviderRole,
    SupportResult,
    runtime_version_at_least,
)
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.utils.device_name import device_name_contains

if TYPE_CHECKING:
    from sparseengine.distributed.parallel_context import ParallelGroup


@dataclass(frozen=True)
class AllReduceOpSpec:
    world_size: int
    ranks: tuple[int, ...]
    max_rows: int
    hidden_size: int
    dtype: torch.dtype
    cuda_graph: bool
    backend: str
    device_ordinals: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0 or self.max_rows <= 0 or self.hidden_size <= 0:
            raise ValueError("All-reduce dimensions must be positive.")
        if (
            len(self.ranks) != self.world_size
            or len(set(self.ranks)) != self.world_size
        ):
            raise ValueError(
                f"All-reduce ranks must contain world_size unique ranks, got {self.ranks}."
            )
        if (
            self.device_ordinals is not None
            and len(self.device_ordinals) != self.world_size
        ):
            raise ValueError(
                "All-reduce CUDA ordinal mapping must contain world_size entries, "
                f"got {self.device_ordinals}."
            )


@dataclass(frozen=True)
class AllReduceGraphBufferMetadata:
    handles: tuple[int, ...]
    offsets: tuple[int, ...]


class AllReduceProvider:
    name = ""

    def prepare(
        self,
        spec: AllReduceOpSpec,
        *,
        group: dist.ProcessGroup | None,
        rank: int,
        device_index: int | None = None,
    ) -> None:
        del spec, group, rank, device_index

    def close(self) -> None:
        pass

    def collect_local_cuda_graph_metadata(
        self,
        spec: AllReduceOpSpec,
        *,
        group: dist.ProcessGroup | None,
    ) -> AllReduceGraphBufferMetadata | None:
        del spec, group
        return None

    def graph_metadata_summary(
        self,
        metadata: AllReduceGraphBufferMetadata | None,
    ) -> tuple[int, int]:
        if metadata is not None:
            raise TypeError(
                f"All-reduce provider {self.name} returned unexpected graph metadata."
            )
        return (0, 0)

    def register_cuda_graph_buffers(
        self,
        spec: AllReduceOpSpec,
        gathered_metadata: list[AllReduceGraphBufferMetadata | None],
        *,
        group: dist.ProcessGroup | None,
    ) -> None:
        del spec, gathered_metadata, group

    def run(
        self,
        spec: AllReduceOpSpec,
        tensor: torch.Tensor,
        *,
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        """Return the reduced tensor, which may or may not alias the input."""
        raise NotImplementedError


ALL_REDUCE_REGISTRY: OpRegistry[AllReduceOpSpec, AllReduceProvider] = OpRegistry(
    "all-reduce",
    portfolio=PortfolioPolicy(repo_portable=("torch_distributed",)),
    profile_order=(
        "flashinfer_vllm_sm90_profile",
        "flashinfer_trtllm_sm90_profile",
    ),
)


@dataclass(frozen=True)
class _FlashInferTrtllmProfile:
    max_rows: int
    launch_with_pdl: bool = False
    completion_row_threshold: int | None = None
    provider_output_buffer: bool = True


_FLASHINFER_TRTLLM_PROFILES = {
    ("h100", 2, 2048): _FlashInferTrtllmProfile(
        max_rows=256,
        launch_with_pdl=True,
        completion_row_threshold=16,
        provider_output_buffer=False,
    ),
    ("h100", 4, 3072): _FlashInferTrtllmProfile(max_rows=32),
}


def _flashinfer_trtllm_profile(
    spec: AllReduceOpSpec,
    caps: DeviceCaps,
) -> _FlashInferTrtllmProfile | None:
    if not device_name_contains(caps.device_name, "H100"):
        return None
    return _FLASHINFER_TRTLLM_PROFILES.get(
        ("h100", spec.world_size, spec.hidden_size)
    )


def _flashinfer_dependency_support() -> SupportResult:
    _, reason = flashinfer_kernel_support("communication")
    return SupportResult.yes(reason)


def _expandable_segments_enabled() -> bool:
    """Return whether PyTorch's CUDA allocator uses expandable segments."""

    for variable in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        for setting in os.getenv(variable, "").split(","):
            key, separator, value = setting.partition(":")
            if (
                separator
                and key.strip() == "expandable_segments"
                and value.strip().lower() in {"1", "true", "yes", "on"}
            ):
                return True
    return False


def _loaded_cuda_runtime_path() -> str:
    """Return a loaded CUDA runtime library, excluding compiler stub libraries."""

    candidates: list[str] = []
    with open("/proc/self/maps", encoding="utf-8") as mappings:
        for mapping in mappings:
            path_start = mapping.find("/")
            if path_start < 0:
                continue
            path = mapping[path_start:].strip()
            filename = os.path.basename(path)
            if (
                filename.startswith("libcudart")
                and ".so" in filename
                and "stub" not in filename
                and " (deleted)" not in path
                and path not in candidates
            ):
                candidates.append(path)
    if not candidates:
        raise RuntimeError(
            "FlashInfer all-reduce requires a loaded CUDA runtime library, but "
            "only compiler stubs or no libcudart mapping was found."
        )
    return candidates[0]


def _prepare_flashinfer_cuda_runtime():
    from flashinfer.comm import CudaRTLibrary
    from flashinfer.comm.cuda_ipc import cudart

    runtime = CudaRTLibrary(_loaded_cuda_runtime_path())
    # Keep the shared lazy proxy: TRT-LLM and IPC helpers retain references to it.
    # FlashInfer 0.6.x has no public API for configuring this shared runtime.
    cudart._library = runtime
    return runtime


@ALL_REDUCE_REGISTRY.register_atomic(
    ProviderRole.UPSTREAM_STANDARD,
    profile_only=True,
)
class FlashInferTrtllmAllReduceProvider(AllReduceProvider):
    name = "flashinfer_trtllm_sm90"

    @classmethod
    def supports(cls, spec: AllReduceOpSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.unsupported(
                "requires CUDA runtime >= 12.8, "
                f"got {caps.runtime_version or 'unknown'}"
            )
        if not caps.supports_graph_capture or not spec.cuda_graph:
            return SupportResult.unsupported("requires CUDA Graph execution")
        if spec.backend != "nccl":
            return SupportResult.unsupported(f"requires NCCL, got {spec.backend}")
        if spec.dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 tensors, got {spec.dtype}"
            )
        return _flashinfer_dependency_support()

    def __init__(self) -> None:
        self.workspace = None
        self._output_buffer: torch.Tensor | None = None
        self._profile: _FlashInferTrtllmProfile | None = None

    def prepare(self, spec, *, group, rank, device_index=None) -> None:
        from flashinfer.comm import create_allreduce_fusion_workspace

        if self.workspace is not None or self._output_buffer is not None:
            raise RuntimeError("FlashInfer all-reduce provider is already prepared.")
        if group is None:
            raise RuntimeError("FlashInfer all-reduce requires a distributed process group.")
        if dist.get_backend(group) != dist.Backend.NCCL:
            raise RuntimeError("FlashInfer all-reduce requires an NCCL process group.")
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "FlashInfer all-reduce must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        caps = platforms.current_platform.get_device_caps(current_device)
        profile = _flashinfer_trtllm_profile(spec, caps)
        if profile is None:
            raise RuntimeError(
                "Prepared FlashInfer TRT-LLM profile no longer matches the active "
                f"device {caps.device_name}."
            )
        _prepare_flashinfer_cuda_runtime()
        workspace = create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=spec.world_size,
            rank=rank,
            max_token_num=profile.max_rows,
            hidden_dim=spec.hidden_size,
            dtype=spec.dtype,
            group=group,
        )
        try:
            output_buffer = (
                torch.empty(
                    (spec.max_rows, spec.hidden_size),
                    dtype=spec.dtype,
                    device=torch.device("cuda", int(device_index)),
                )
                if profile.provider_output_buffer
                else None
            )
        except Exception:
            workspace.destroy()
            raise
        self.workspace = workspace
        self._output_buffer = output_buffer
        self._profile = profile

    def close(self) -> None:
        workspace = self.workspace
        self.workspace = None
        self._output_buffer = None
        self._profile = None
        if workspace is not None:
            workspace.destroy()

    def run(self, spec, tensor, *, group) -> torch.Tensor:
        del group
        if self.workspace is None or self._profile is None:
            raise RuntimeError("FlashInfer all-reduce provider was not prepared.")
        if not tensor.is_cuda:
            raise ValueError(
                f"FlashInfer all-reduce requires a CUDA tensor, got {tensor.device}."
            )
        from flashinfer.comm import allreduce_fusion
        from flashinfer.comm.trtllm_ar import AllReduceFusionPattern

        flattened = tensor.view(-1, spec.hidden_size)
        output = (
            None
            if self._output_buffer is None
            else self._output_buffer[: int(flattened.shape[0])]
        )
        result = allreduce_fusion(
            input=flattened,
            workspace=self.workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=self._profile.launch_with_pdl,
            trigger_completion_at_end=(
                self._profile.completion_row_threshold is None
                or int(flattened.shape[0]) > self._profile.completion_row_threshold
            ),
            output=output,
        )
        return result.view_as(tensor)


@ALL_REDUCE_REGISTRY.register_atomic(
    ProviderRole.UPSTREAM_STANDARD,
    profile_only=True,
)
class FlashInferVllmAllReduceProvider(AllReduceProvider):
    name = "flashinfer_vllm_sm90"
    max_rows = 1024
    num_ctas = 32

    @classmethod
    def supports(cls, spec: AllReduceOpSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("requires CUDA Graph capture support")
        if spec.world_size not in (2, 4, 6, 8):
            return SupportResult.unsupported(
                "upstream custom all-reduce requires 2, 4, 6 or 8 ranks"
            )
        if spec.backend != "nccl":
            return SupportResult.unsupported(f"requires NCCL, got {spec.backend}")
        if spec.dtype != torch.bfloat16:
            return SupportResult.unsupported(
                f"requires BF16 tensors, got {spec.dtype}"
            )
        if spec.device_ordinals is None:
            return SupportResult.unsupported(
                "requires an explicit process-rank to CUDA-ordinal mapping"
            )
        if len(set(spec.device_ordinals)) != spec.world_size:
            return SupportResult.unsupported(
                "requires one rank per CUDA device on a single host; "
                f"device_ordinals={spec.device_ordinals}"
            )
        dependency = _flashinfer_dependency_support()
        if not dependency.supported:
            return dependency
        try:
            from flashinfer import comm
        except (ImportError, OSError, RuntimeError) as exc:
            raise RequiredExternalKernelFamilyError(
                KernelFamilyHealth(
                    family="flashinfer-python",
                    state=KernelFamilyState.BROKEN,
                    version=None,
                    reason=f"communication APIs are unavailable: {exc}",
                ),
                feature="communication",
            )
        required = (
            "CudaRTLibrary",
            "create_shared_buffer",
            "vllm_all_reduce",
            "vllm_dispose",
            "vllm_get_graph_buffer_ipc_meta",
            "vllm_init_custom_ar",
            "vllm_meta_size",
            "vllm_register_buffer",
            "vllm_register_graph_buffers",
        )
        missing = [name for name in required if not hasattr(comm, name)]
        if missing:
            raise RequiredExternalKernelFamilyError(
                KernelFamilyHealth(
                    family="flashinfer-python",
                    state=KernelFamilyState.BROKEN,
                    version=None,
                    reason="communication APIs are missing: " + ", ".join(missing),
                ),
                feature="communication",
            )
        return dependency

    def __init__(self) -> None:
        self._group: dist.ProcessGroup | None = None
        self._rank = -1
        self._max_size_bytes = 0
        self._rank_data: torch.Tensor | None = None
        self._meta_ptrs: list[int] = []
        self._buffer_ptrs: list[int] = []
        self._handle = None
        self._cudart = None
        self._graph_buffers_registered = False

    def binding_metadata(self) -> dict[str, object]:
        return {"cuda_graph_input_mode": "direct_ipc"}

    def prepare(self, spec, *, group, rank, device_index=None) -> None:
        if spec.cuda_graph and _expandable_segments_enabled():
            raise RuntimeError(
                "FlashInfer vLLM all-reduce CUDA Graph capture requires "
                "IPC-exportable graph allocations, but PyTorch expandable "
                "segments are enabled. Unset expandable_segments in "
                "PYTORCH_ALLOC_CONF/PYTORCH_CUDA_ALLOC_CONF before starting "
                "the process."
            )
        from flashinfer.comm import (
            create_shared_buffer,
            vllm_init_custom_ar,
            vllm_meta_size,
            vllm_register_buffer,
        )

        if self._handle is not None:
            raise RuntimeError("FlashInfer all-reduce provider is already prepared.")
        if group is None:
            raise RuntimeError(
                "FlashInfer all-reduce requires a distributed process group."
            )
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        device_ordinals = spec.device_ordinals
        if device_ordinals is None:
            raise RuntimeError(
                "FlashInfer vLLM all-reduce requires an explicit CUDA-ordinal mapping."
            )
        if (
            int(device_index) != current_device
            or current_device != device_ordinals[rank]
        ):
            raise RuntimeError(
                "FlashInfer vLLM all-reduce requires rank-to-device mapping: "
                f"ranks={spec.ranks} device_ordinals={device_ordinals} rank={rank} "
                f"selected={device_index} "
                f"current={current_device}."
            )
        if any(
            peer != current_device
            and not torch.cuda.can_device_access_peer(current_device, peer)
            for peer in device_ordinals
        ):
            raise RuntimeError("FlashInfer vLLM all-reduce requires CUDA peer access.")
        # The upstream kernel for >2 ranks launches only in full-NVLink mode.
        full_nvlink = (
            spec.world_size > 2
            and platforms.current_platform.supports_nvlink_group(device_ordinals)
        )
        if spec.world_size > 2 and not full_nvlink:
            raise RuntimeError(
                "FlashInfer vLLM all-reduce with more than two ranks requires "
                "fully connected NVLink."
            )
        cudart = _prepare_flashinfer_cuda_runtime()
        max_size_bytes = spec.max_rows * spec.hidden_size * spec.dtype.itemsize
        meta_ptrs = create_shared_buffer(vllm_meta_size() + max_size_bytes, group)
        buffer_ptrs = create_shared_buffer(max_size_bytes, group)
        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        handle = vllm_init_custom_ar(meta_ptrs, rank_data, rank, full_nvlink)
        vllm_register_buffer(handle, buffer_ptrs)
        self._group = group
        self._rank = rank
        self._max_size_bytes = max_size_bytes
        self._rank_data = rank_data
        self._meta_ptrs = meta_ptrs
        self._buffer_ptrs = buffer_ptrs
        self._handle = handle
        self._cudart = cudart

    def run(self, spec, tensor, *, group) -> torch.Tensor:
        del group
        if self._handle is None:
            raise RuntimeError("FlashInfer all-reduce provider was not prepared.")
        from flashinfer.comm import vllm_all_reduce

        output = torch.empty_like(tensor)
        capturing = bool(
            spec.cuda_graph and torch.cuda.is_current_stream_capturing()
        )
        vllm_all_reduce(
            self._handle,
            tensor,
            output,
            0 if capturing else self._buffer_ptrs[self._rank],
            0 if capturing else self._max_size_bytes,
            self.num_ctas,
        )
        return output

    def collect_local_cuda_graph_metadata(self, spec, *, group):
        if not spec.cuda_graph:
            return None
        if self._handle is None or self._group is None:
            raise RuntimeError("FlashInfer all-reduce provider was not prepared.")
        if group is not self._group:
            raise RuntimeError("FlashInfer all-reduce graph registration group changed.")
        if self._graph_buffers_registered:
            raise RuntimeError(
                "FlashInfer all-reduce CUDA Graph buffers are already registered."
            )
        from flashinfer.comm import vllm_get_graph_buffer_ipc_meta

        local_handles, local_offsets = vllm_get_graph_buffer_ipc_meta(self._handle)
        local_handles = list(local_handles)
        local_offsets = list(local_offsets)
        if not local_handles or not local_offsets:
            raise RuntimeError(
                "FlashInfer all-reduce captured no CUDA Graph buffer addresses."
            )
        return AllReduceGraphBufferMetadata(
            handles=tuple(local_handles),
            offsets=tuple(local_offsets),
        )

    def graph_metadata_summary(self, metadata):
        if not isinstance(metadata, AllReduceGraphBufferMetadata):
            raise TypeError("FlashInfer all-reduce graph metadata is missing.")
        count = len(metadata.offsets)
        if count <= 0 or len(metadata.handles) % count:
            raise RuntimeError(
                "FlashInfer all-reduce graph metadata has incompatible handles and "
                f"offsets: handles={len(metadata.handles)} offsets={count}."
            )
        return (count, len(metadata.handles) // count)

    def register_cuda_graph_buffers(self, spec, gathered_metadata, *, group) -> None:
        if not spec.cuda_graph:
            return
        if self._handle is None or self._group is None:
            raise RuntimeError("FlashInfer all-reduce provider was not prepared.")
        if group is not self._group:
            raise RuntimeError("FlashInfer all-reduce graph registration group changed.")
        if self._graph_buffers_registered:
            raise RuntimeError(
                "FlashInfer all-reduce CUDA Graph buffers are already registered."
            )
        if len(gathered_metadata) != spec.world_size or any(
            not isinstance(item, AllReduceGraphBufferMetadata)
            for item in gathered_metadata
        ):
            raise RuntimeError("FlashInfer all-reduce graph metadata is incomplete.")
        metadata = [
            item for item in gathered_metadata
            if isinstance(item, AllReduceGraphBufferMetadata)
        ]
        offsets = [list(item.offsets) for item in metadata]
        if len({len(item) for item in offsets}) != 1:
            raise RuntimeError(
                "FlashInfer all-reduce ranks captured different graph buffer counts: "
                f"{[len(item) for item in offsets]}."
            )
        from flashinfer.comm import vllm_register_graph_buffers

        handles = [list(item.handles) for item in metadata]
        vllm_register_graph_buffers(self._handle, handles, offsets)
        self._graph_buffers_registered = True

    @staticmethod
    def _close_shared_buffer(pointers, group, rank, cudart) -> None:
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        close = cudart.lib.cudaIpcCloseMemHandle
        close.restype = ctypes.c_int
        close.argtypes = [ctypes.c_void_p]
        for peer_rank, pointer in enumerate(pointers):
            if peer_rank != rank:
                result = int(close(ctypes.c_void_p(pointer)))
                if result != 0:
                    raise RuntimeError(
                        f"cudaIpcCloseMemHandle failed: {cudart.cudaGetErrorString(result)}"
                    )
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        cudart.cudaFree(ctypes.c_void_p(pointers[rank]))
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])

    def close(self) -> None:
        if self._handle is None:
            return
        from flashinfer.comm import vllm_dispose

        vllm_dispose(self._handle)
        self._handle = None
        self._graph_buffers_registered = False
        self._close_shared_buffer(
            self._buffer_ptrs, self._group, self._rank, self._cudart
        )
        self._close_shared_buffer(
            self._meta_ptrs, self._group, self._rank, self._cudart
        )
        self._rank_data = None


class _FlashInferAllReduceProfile:
    atomic_provider_name = ""

    @classmethod
    def atomic_provider_names(cls, spec: AllReduceOpSpec) -> tuple[str, ...]:
        del spec
        return (cls.atomic_provider_name,)

    @classmethod
    def bind(cls, spec: AllReduceOpSpec, caps: DeviceCaps, **kwargs):
        del spec, caps
        if kwargs:
            raise TypeError(
                f"{cls.name} does not accept provider arguments: {sorted(kwargs)}"
            )
        provider_type = ALL_REDUCE_REGISTRY.atomic_registry.registration(
            cls.atomic_provider_name
        ).provider
        return provider_type()


@ALL_REDUCE_REGISTRY.register_profile
class FlashInferTrtllmAllReduceProfile(_FlashInferAllReduceProfile):
    name = "flashinfer_trtllm_sm90_profile"
    atomic_provider_name = "flashinfer_trtllm_sm90"

    @classmethod
    def matches(cls, spec: AllReduceOpSpec, caps: DeviceCaps) -> ProfileMatch:
        profile = _flashinfer_trtllm_profile(spec, caps)
        if profile is None or spec.dtype != torch.bfloat16:
            return ProfileMatch.no(
                "requires an exact H100 BF16 topology/shape profile, got "
                f"device={caps.device_name} world_size={spec.world_size} "
                f"hidden_size={spec.hidden_size} dtype={spec.dtype}"
            )
        if spec.max_rows > profile.max_rows:
            return ProfileMatch.no(
                f"profile requires max_rows <= {profile.max_rows}, "
                f"got {spec.max_rows}"
            )
        return ProfileMatch.yes("matched exact FlashInfer TRT-LLM profile")


@ALL_REDUCE_REGISTRY.register_profile
class FlashInferVllmAllReduceProfile(_FlashInferAllReduceProfile):
    max_rows_without_nvlink = 256
    name = "flashinfer_vllm_sm90_profile"
    atomic_provider_name = "flashinfer_vllm_sm90"

    @classmethod
    def matches(cls, spec: AllReduceOpSpec, caps: DeviceCaps) -> ProfileMatch:
        if not any(
            device_name_contains(caps.device_name, name) for name in ("H100", "H20")
        ):
            return ProfileMatch.no(
                f"requires profiled H100 or H20 hardware, got {caps.device_name}"
            )
        if (
            spec.world_size != 2
            or spec.max_rows > FlashInferVllmAllReduceProvider.max_rows
            or spec.hidden_size != 2048
            or spec.dtype != torch.bfloat16
        ):
            return ProfileMatch.no(
                "requires profiled TP2 BF16 [..., 2048] with "
                f"max_rows <= {FlashInferVllmAllReduceProvider.max_rows}, "
                f"got world_size={spec.world_size} max_rows={spec.max_rows} "
                f"hidden_size={spec.hidden_size} dtype={spec.dtype}"
            )
        if spec.max_rows > cls.max_rows_without_nvlink:
            if not spec.cuda_graph:
                return ProfileMatch.no("extended row range is profiled with CUDA Graphs")
            if (
                spec.device_ordinals is None
                or len(set(spec.device_ordinals)) != spec.world_size
            ):
                return ProfileMatch.no("extended row range requires distinct CUDA ordinals")
            if importlib.util.find_spec("pynvml") is None:
                return ProfileMatch.no(
                    "extended row range needs nvidia-ml-py for NVLink validation"
                )
            if not platforms.current_platform.supports_nvlink_group(
                spec.device_ordinals
            ):
                return ProfileMatch.no(
                    "extended row range is profiled on fully connected NVLink"
                )
        return ProfileMatch.yes("matched exact FlashInfer vLLM profile")


@ALL_REDUCE_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TorchDistributedAllReduceProvider(AllReduceProvider):
    name = "torch_distributed"

    @classmethod
    def supports(cls, spec: AllReduceOpSpec, caps: DeviceCaps) -> SupportResult:
        del spec, caps
        return SupportResult.yes()

    def run(self, spec, tensor, *, group) -> torch.Tensor:
        if spec.world_size > 1:
            if group is None:
                raise RuntimeError(
                    "Torch distributed all-reduce requires a process group when world_size > 1."
                )
            dist.all_reduce(tensor, group=group)
        return tensor


def _validate_tensor_contract(spec: AllReduceOpSpec, tensor: torch.Tensor) -> None:
    if tensor.dtype != spec.dtype:
        raise TypeError(
            f"All-reduce expected dtype={spec.dtype}, got {tensor.dtype}."
        )
    if tensor.ndim < 2:
        raise ValueError(
            f"All-reduce expects [..., hidden], got shape={tuple(tensor.shape)}."
        )
    row_count = tensor.numel() // int(tensor.shape[-1])
    hidden_size = int(tensor.shape[-1])
    if not 0 < row_count <= spec.max_rows:
        raise ValueError(
            "All-reduce row count is outside the prepared range: "
            f"rows={row_count} max_rows={spec.max_rows}."
        )
    if hidden_size != spec.hidden_size:
        raise ValueError(
            "All-reduce hidden size does not match the prepared operator: "
            f"hidden={hidden_size} expected={spec.hidden_size}."
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"All-reduce requires a contiguous tensor, got stride={tensor.stride()}."
        )


class PreparedAllReduceOp:
    """A pre-bound all-reduce with no execution-time fallback."""

    def __init__(
        self,
        spec: AllReduceOpSpec,
        provider: AllReduceProvider,
        *,
        group: dist.ProcessGroup | None,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.group = group
        self._closed = False

    @property
    def name(self) -> str:
        return self.provider.name

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("All-reduce operator is closed.")
        _validate_tensor_contract(self.spec, tensor)
        output = self.provider.run(self.spec, tensor, group=self.group)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"All-reduce provider {self.provider.name} returned "
                f"{type(output).__name__}, expected torch.Tensor."
            )
        _validate_tensor_contract(self.spec, output)
        if output.shape != tensor.shape or output.device != tensor.device:
            raise ValueError(
                f"All-reduce provider {self.provider.name} returned an incompatible tensor: "
                f"input_shape={tuple(tensor.shape)} output_shape={tuple(output.shape)} "
                f"input_device={tensor.device} output_device={output.device}."
            )
        return output

    def collect_local_cuda_graph_metadata(
        self,
    ) -> AllReduceGraphBufferMetadata | None:
        if self._closed:
            raise RuntimeError("All-reduce operator is closed.")
        return self.provider.collect_local_cuda_graph_metadata(
            self.spec,
            group=self.group,
        )

    def graph_metadata_summary(
        self,
        metadata: AllReduceGraphBufferMetadata | None,
    ) -> tuple[int, int]:
        return self.provider.graph_metadata_summary(metadata)

    def register_cuda_graph_buffers(
        self,
        gathered_metadata: list[AllReduceGraphBufferMetadata | None],
    ) -> None:
        if self._closed:
            raise RuntimeError("All-reduce operator is closed.")
        self.provider.register_cuda_graph_buffers(
            self.spec,
            gathered_metadata,
            group=self.group,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.provider.close()
        self._closed = True


def resolve_all_reduce_provider(
    spec: AllReduceOpSpec,
    *,
    device_index: int | None = None,
) -> AllReduceProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(ALL_REDUCE_REGISTRY).resolve(spec, caps).provider


def prepare_all_reduce_op(
    spec: AllReduceOpSpec,
    *,
    group: dist.ProcessGroup | None,
    rank: int,
    device_index: int | None = None,
    provider: AllReduceProvider | None = None,
) -> PreparedAllReduceOp:
    rank = int(rank)
    if not 0 <= rank < spec.world_size:
        raise ValueError(
            f"All-reduce rank must be in [0, {spec.world_size}), got {rank}."
        )
    if spec.world_size > 1 and group is None:
        raise RuntimeError(
            "All-reduce requires a process group when world_size > 1."
        )
    if provider is None:
        provider = resolve_all_reduce_provider(spec, device_index=device_index)
    provider.prepare(
        spec,
        group=group,
        rank=rank,
        device_index=device_index,
    )
    return PreparedAllReduceOp(spec, provider, group=group)


def prepare_parallel_all_reduce(
    group: ParallelGroup,
    *,
    max_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    cuda_graph: bool,
    device_index: int,
) -> PreparedAllReduceOp:
    """Bind an all-reduce operator to an initialized parallel group."""

    device_ordinals: tuple[int, ...] | None = None
    if group.process_group is not None and str(
        dist.get_backend(group.process_group)
    ) == "nccl":
        current_device = torch.cuda.current_device()
        if int(device_index) != current_device:
            raise RuntimeError(
                "All-reduce CUDA-ordinal discovery must run on the selected device: "
                f"selected={device_index} current={current_device}."
            )
        local_ordinal = torch.tensor(
            [current_device],
            dtype=torch.int32,
            device=torch.device("cuda", current_device),
        )
        gathered_ordinals = [torch.empty_like(local_ordinal) for _ in range(group.size)]
        dist.all_gather(
            gathered_ordinals,
            local_ordinal,
            group=group.process_group,
        )
        device_ordinals = tuple(int(item.item()) for item in gathered_ordinals)

    return prepare_all_reduce_op(
        AllReduceOpSpec(
            world_size=group.size,
            ranks=group.ranks,
            max_rows=max_rows,
            hidden_size=hidden_size,
            dtype=dtype,
            cuda_graph=cuda_graph,
            backend=(
                "none"
                if group.process_group is None
                else str(dist.get_backend(group.process_group))
            ),
            device_ordinals=device_ordinals,
        ),
        group=group.process_group,
        rank=group.rank,
        device_index=device_index,
    )
