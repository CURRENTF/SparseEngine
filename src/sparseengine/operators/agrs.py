"""Prepared AG/RS primitives; token ownership and padding belong to transport."""

from __future__ import annotations

import importlib.util
import socket
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from sparseengine import platforms
from sparseengine.kernels.external.flashinfer.support import flashinfer_kernel_support
from sparseengine.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
)
from sparseengine.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class AgRsOpSpec:
    world_size: int
    hidden_size: int
    dtype: torch.dtype
    max_rows: int
    backend: str
    full_local_world: bool
    multicast_and_peer_access: bool

    def __post_init__(self):
        if min(self.world_size, self.hidden_size, self.max_rows) <= 0:
            raise ValueError("AG/RS dimensions must be positive.")


class TorchAgRsProvider:
    name = "torch_distributed_agrs"
    supports_variable_sizes = True

    @classmethod
    def supports(cls, spec, caps):
        return SupportResult.yes()

    def prepare(self, spec, *, group, rank, device_index):
        self.group = group
        self.rank = rank
        self.size = spec.world_size
        self.backend = spec.backend

    def all_gather(self, output, local):
        if self.size == 1:
            output.copy_(local)
        else:
            dist.all_gather_into_tensor(output, local, group=self.group)

    def reduce_scatter(self, output, partial):
        if self.size == 1:
            output.copy_(partial)
        else:
            dist.reduce_scatter_tensor(output, partial, group=self.group)

    def all_gatherv(self, output, local, sizes):
        if self.size == 1:
            output.copy_(local)
            return
        offset = 0
        if self.backend != "nccl":
            for root, rows in enumerate(sizes):
                rows = int(rows)
                if rows:
                    segment = output[offset : offset + rows]
                    if self.rank == root:
                        segment.copy_(local)
                    global_root = (
                        root
                        if self.group is None
                        else dist.get_global_rank(self.group, root)
                    )
                    dist.broadcast(segment, src=global_root, group=self.group)
                offset += rows
            return
        segments = []
        for rows in sizes:
            rows = int(rows)
            segments.append(output[offset : offset + rows])
            offset += rows
        dist.all_gather(segments, local, group=self.group)

    def reduce_scatterv(self, output, partial, sizes):
        if self.size == 1:
            output.copy_(partial)
            return
        dist.all_reduce(partial, group=self.group)
        offset = sum(int(rows) for rows in sizes[: self.rank])
        output.copy_(partial[offset : offset + int(sizes[self.rank])])

    def close(self):
        pass


AGRS_REGISTRY = OpRegistry(
    "all-gather reduce-scatter",
    portfolio=PortfolioPolicy(
        upstream_standard=("flashinfer_mixed_comm_uc",),
        repo_portable=(TorchAgRsProvider.name,),
    ),
)
AGRS_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)(TorchAgRsProvider)


def _mixed_comm_available():
    # Older supported FlashInfer releases need not contain this optional API.
    # Import failures from an installed module propagate instead of disabling it.
    package = importlib.util.find_spec("flashinfer")
    if package is None:
        return False
    if not any(
        (Path(root) / "comm" / "mixed_comm.py").is_file()
        for root in package.submodule_search_locations or ()
    ):
        return False
    for parent, child in (("cuda", "cuda.bindings"), ("nvidia", "nvidia.nvshmem")):
        if (
            importlib.util.find_spec(parent) is None
            or importlib.util.find_spec(child) is None
        ):
            return False
    return True


@AGRS_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferMixedAgRsProvider:
    name = "flashinfer_mixed_comm_uc"
    supports_variable_sizes = False

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability not in (
            (9, 0),
            (10, 0),
        ):
            return SupportResult.unsupported(
                "upstream mixed communication requires SM90 or SM100"
            )
        if spec.dtype not in (torch.float16, torch.bfloat16) or spec.hidden_size % 8:
            return SupportResult.unsupported(
                "requires FP16/BF16 rows aligned to 16 bytes"
            )
        if (
            spec.world_size not in (2, 4, 8)
            or spec.backend != "nccl"
            or not spec.full_local_world
        ):
            return SupportResult.unsupported(
                "requires a complete single-host NCCL world of 2, 4 or 8 ranks"
            )
        if not spec.multicast_and_peer_access:
            return SupportResult.unsupported(
                "requires multicast and peer access on every rank"
            )
        if not _mixed_comm_available():
            return SupportResult.unsupported(
                "optional FlashInfer mixed-comm dependencies are absent"
            )
        return SupportResult.yes()

    def prepare(self, spec, *, group, rank, device_index):
        flashinfer_kernel_support("mixed communication")
        from flashinfer.comm.mixed_comm import (
            MixedCommHandler,
            MixedCommMode,
            MixedCommOp,
            run_mixed_comm,
        )

        self.handler = MixedCommHandler(
            world_rank=rank,
            world_size=spec.world_size,
            local_rank=rank,
            local_size=spec.world_size,
            inter_rank=0,
            inter_size=1,
            local_tp_size=1,
            local_dp_size=spec.world_size,
            inter_tp_size=1,
            inter_dp_size=1,
            dtype=spec.dtype,
            device=torch.device("cuda", device_index),
            use_autotune=False,
        )
        self.run = run_mixed_comm
        self.mode = MixedCommMode.FUSED_OPT_WAITS_UC
        self.ag = MixedCommOp.ALLGATHER
        self.rs = MixedCommOp.REDUCESCATTER

    def all_gather(self, output, local):
        self.run(self.ag, self.handler, local, output, self.mode)

    def reduce_scatter(self, output, partial):
        self.run(self.rs, self.handler, partial, output, self.mode)

    def close(self):
        self.handler.shutdown()


class PreparedAgRsOp:
    def __init__(self, spec, provider):
        self.spec = spec
        self.provider = provider
        self.closed = False

    @property
    def name(self):
        return self.provider.name

    @property
    def supports_variable_sizes(self):
        return bool(self.provider.supports_variable_sizes)

    def _validate(self, local, global_tensor):
        if self.closed:
            raise RuntimeError("AG/RS operator is closed.")
        if (
            local.ndim != 2
            or global_tensor.ndim != 2
            or not 0 < local.shape[0] <= self.spec.max_rows
            or local.shape[1] != self.spec.hidden_size
            or global_tensor.shape
            != (local.shape[0] * self.spec.world_size, self.spec.hidden_size)
            or local.dtype != self.spec.dtype
            or global_tensor.dtype != local.dtype
            or local.device != global_tensor.device
            or not local.is_contiguous()
            or not global_tensor.is_contiguous()
        ):
            raise ValueError(
                "AG/RS tensors do not match the prepared shape, dtype, device and layout."
            )

    def all_gather(self, output, local):
        self._validate(local, output)
        self.provider.all_gather(output, local)

    def reduce_scatter(self, output, partial):
        self._validate(output, partial)
        self.provider.reduce_scatter(output, partial)

    def _validate_variable(self, local, global_tensor, sizes):
        if self.closed:
            raise RuntimeError("AG/RS operator is closed.")
        sizes = tuple(int(value) for value in sizes)
        if (
            len(sizes) != self.spec.world_size
            or any(value < 0 for value in sizes)
            or local.ndim != 2
            or global_tensor.ndim != 2
            or local.shape[0] != sizes[self.provider.rank]
            or global_tensor.shape != (sum(sizes), self.spec.hidden_size)
            or local.shape[1] != self.spec.hidden_size
            or local.dtype != self.spec.dtype
            or global_tensor.dtype != local.dtype
            or local.device != global_tensor.device
            or not local.is_contiguous()
            or not global_tensor.is_contiguous()
        ):
            raise ValueError(
                "Variable AG/RS tensors do not match the prepared sizes, dtype, device and layout."
            )
        return sizes

    def all_gatherv(self, output, local, sizes):
        if not self.supports_variable_sizes:
            raise RuntimeError(
                f"AG/RS provider {self.name!r} does not support variable token sizes."
            )
        sizes = self._validate_variable(local, output, sizes)
        self.provider.all_gatherv(output, local, sizes)

    def reduce_scatterv(self, output, partial, sizes):
        if not self.supports_variable_sizes:
            raise RuntimeError(
                f"AG/RS provider {self.name!r} does not support variable token sizes."
            )
        sizes = self._validate_variable(output, partial, sizes)
        self.provider.reduce_scatterv(output, partial, sizes)

    def close(self):
        if not self.closed:
            self.provider.close()
            self.closed = True


def prepare_parallel_agrs(group, *, max_rows, hidden_size, dtype, device_index):
    platform = platforms.current_platform
    caps = platform.get_device_caps(device_index)
    backend = (
        "none"
        if group.process_group is None
        else str(dist.get_backend(group.process_group))
    )
    multicast = False
    if (
        caps.platform == PlatformEnum.CUDA
        and caps.compute_capability in ((9, 0), (10, 0))
        and _mixed_comm_available()
    ):
        multicast = platform.supports_multicast(device_index)
    info = (socket.gethostname(), device_index, multicast)
    peers = [info]
    if group.size > 1:
        peers = [None] * group.size
        dist.all_gather_object(peers, info, group=group.process_group)
    full_local = (
        len({p[0] for p in peers}) == 1
        and len({p[1] for p in peers}) == group.size
        and dist.is_initialized()
        and group.size == dist.get_world_size()
        and group.ranks == tuple(range(group.size))
    )
    peer_access = full_local and all(p[2] for p in peers)
    if peer_access:
        peer_access = all(
            p[1] == device_index or (platform.supports_peer_atomics(device_index, p[1]))
            for p in peers
        )
        flags = [None] * group.size
        dist.all_gather_object(flags, peer_access, group=group.process_group)
        peer_access = all(flags)
    spec = AgRsOpSpec(
        group.size,
        hidden_size,
        dtype,
        max_rows,
        backend,
        full_local,
        peer_access,
    )
    provider = OpResolver(AGRS_REGISTRY).resolve(spec, caps).provider
    provider.prepare(
        spec, group=group.process_group, rank=group.rank, device_index=device_index
    )
    return PreparedAgRsOp(spec, provider)
