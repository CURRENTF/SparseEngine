from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.distributed as dist

from sparseengine.distributed.moe_communication import (
    AllGatherReduceScatterMoeCommunication,
    MoeCommunication,
)
from sparseengine.distributed.parallel_context import ParallelContext, ParallelGroup
from sparseengine.operators.all_reduce import (
    AllReduceGraphBufferMetadata,
    PreparedAllReduceOp,
    prepare_parallel_all_reduce,
)
from sparseengine.utils.context import get_context


class ParallelCollectiveState(str, Enum):
    OPEN = "open"
    PREPARED = "prepared"
    CAPTURING = "capturing"
    CAPTURED = "captured"
    EXCHANGED = "exchanged"
    REGISTERED = "registered"
    REPLAYABLE = "replayable"
    CLOSED = "closed"


@dataclass(frozen=True)
class _AllReduceKey:
    process_group_id: int
    ranks: tuple[int, ...]
    hidden_size: int
    dtype: torch.dtype


@dataclass
class _AllReduceBinding:
    group: ParallelGroup
    hidden_size: int
    dtype: torch.dtype
    max_rows: int
    roles: set[str] = field(default_factory=set)
    op: PreparedAllReduceOp | None = None
    local_metadata: AllReduceGraphBufferMetadata | None = None
    local_metadata_summary: tuple[int, int] | None = None
    gathered_metadata: list[AllReduceGraphBufferMetadata | None] | None = None


class ParallelAllReduceHandle:
    """Model-facing all-reduce handle backed by the shared parallel runtime."""

    def __init__(
        self,
        group: ParallelGroup,
        binding: _AllReduceBinding | None = None,
    ) -> None:
        self._group = group
        self._binding = binding

    @property
    def name(self) -> str:
        if self._binding is None:
            return "identity"
        op = self._binding.op
        return "unprepared" if op is None else op.name

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        binding = self._binding
        if binding is None:
            return tensor
        if get_context().is_prefill:
            return self._group.all_reduce(tensor)
        if binding.op is None:
            raise RuntimeError("Parallel all-reduce handle is not prepared.")
        return binding.op.run(tensor)


@dataclass(frozen=True)
class DecodeParallelCollectives:
    attention: ParallelAllReduceHandle
    moe: ParallelAllReduceHandle
    moe_transport: MoeCommunication | None = None


class ParallelCollectiveRuntime:
    """Own shared collective resources and their CUDA Graph lifecycle."""

    def __init__(
        self,
        parallel_context: ParallelContext,
        *,
        cuda_graph: bool,
        device_index: int,
    ) -> None:
        self.parallel_context = parallel_context
        self.cuda_graph = bool(cuda_graph)
        self.device_index = int(device_index)
        self.state = ParallelCollectiveState.OPEN
        self._bindings: list[_AllReduceBinding] = []
        self._handles: dict[_AllReduceKey, ParallelAllReduceHandle] = {}
        self._moe_transport: MoeCommunication | None = None

    @property
    def has_graph_collectives(self) -> bool:
        return self.cuda_graph and bool(self._bindings)

    def moe_transport_stats(self):
        transport = self._moe_transport
        return {
            "transport": "allreduce" if transport is None else type(transport).__name__,
            "provider": None if transport is None else transport.name,
            "version": getattr(getattr(transport, "op", None), "version", None),
        }

    def _request_all_reduce(
        self,
        role: str,
        group: ParallelGroup,
        *,
        max_rows: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> ParallelAllReduceHandle:
        if self.state is not ParallelCollectiveState.OPEN:
            raise RuntimeError(
                "All-reduce requests must be declared before the parallel runtime "
                f"is prepared, got state={self.state.value}."
            )
        role = str(role).strip()
        max_rows = int(max_rows)
        hidden_size = int(hidden_size)
        if not role or max_rows <= 0 or hidden_size <= 0:
            raise ValueError(
                "All-reduce role and dimensions must be positive, got "
                f"role={role!r} max_rows={max_rows} hidden_size={hidden_size}."
            )
        key = _AllReduceKey(
            process_group_id=id(group.process_group),
            ranks=group.ranks,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        handle = self._handles.get(key)
        if handle is None:
            binding = None
            if group.size > 1:
                binding = _AllReduceBinding(
                    group=group,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    max_rows=max_rows,
                    roles={role},
                )
                self._bindings.append(binding)
            handle = ParallelAllReduceHandle(group, binding)
            self._handles[key] = handle
        else:
            binding = handle._binding
            if binding is not None:
                binding.max_rows = max(binding.max_rows, max_rows)
                binding.roles.add(role)
        return handle

    def request_decode_collectives(
        self,
        *,
        attention_max_rows: int,
        moe_max_rows: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> DecodeParallelCollectives:
        attention = self._request_all_reduce(
            "attention",
            self.parallel_context.attn_tp,
            max_rows=attention_max_rows,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        moe = self._request_all_reduce(
            "moe",
            self.parallel_context.world,
            max_rows=moe_max_rows,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        return DecodeParallelCollectives(attention=attention, moe=moe)

    def request_moe_collectives(
        self, *, attention_max_rows, moe_max_rows, max_local_tokens,
        hidden_size, dtype, backend, num_experts, top_k,
    ):
        """Prepare transport from token ownership; models supply tensor contracts."""
        if self.parallel_context.attn_dp_size > 1:
            return self.request_dp_collectives(
                max_rows=attention_max_rows,
                max_local_tokens=max(attention_max_rows, max_local_tokens),
                hidden_size=hidden_size, dtype=dtype, backend=backend,
                num_experts=num_experts, top_k=top_k,
            )
        return self.request_decode_collectives(
            attention_max_rows=attention_max_rows, moe_max_rows=moe_max_rows,
            hidden_size=hidden_size, dtype=dtype,
        )

    def request_dp_collectives(
        self, *, max_rows, hidden_size, dtype, backend="agrs",
        max_local_tokens=None, num_experts=None, top_k=None,
    ):
        if self.state is not ParallelCollectiveState.OPEN or self._moe_transport is not None:
            raise RuntimeError("DP collectives must be requested once before preparation.")
        if self.parallel_context.attn_dp_size <= 1:
            raise ValueError("DP collectives require the DP attention topology.")
        attention = self._request_all_reduce(
            "attention", self.parallel_context.attn_tp,
            max_rows=max_rows, hidden_size=hidden_size, dtype=dtype,
        )
        moe = self._request_all_reduce(
            "moe", self.parallel_context.attn_tp,
            max_rows=max_rows, hidden_size=hidden_size, dtype=dtype,
        )
        if backend == "agrs":
            self._moe_transport = AllGatherReduceScatterMoeCommunication(
                self.parallel_context,
                max_rows=max(max_rows, max_local_tokens or max_rows),
                hidden_size=hidden_size,
                dtype=dtype,
                reduce=moe.run,
            )
        elif backend == "deepepv1":
            from sparseengine.distributed.moe_all2all import AllToAllMoeCommunication
            from sparseengine.operators.all2all import AllToAllOpSpec

            spec = AllToAllOpSpec(
                self.parallel_context.moe_ep_size, hidden_size, num_experts, top_k,
                max_local_tokens, dtype, self.cuda_graph,
            )
            self._moe_transport = AllToAllMoeCommunication(self.parallel_context, spec)
        else:
            raise ValueError(f"Unsupported DP MoE transport: {backend}")
        return DecodeParallelCollectives(attention, moe, self._moe_transport)

    def prepare(self) -> None:
        if self.state is not ParallelCollectiveState.OPEN:
            raise RuntimeError(
                f"Parallel collective runtime cannot prepare from {self.state.value}."
            )
        prepared: list[PreparedAllReduceOp] = []
        try:
            if self._moe_transport is not None and self._moe_transport.op is None:
                self._moe_transport.prepare(
                    device_index=self.device_index, cuda_graph=self.cuda_graph
                )
            for binding in self._bindings:
                binding.op = prepare_parallel_all_reduce(
                    binding.group,
                    max_rows=binding.max_rows,
                    hidden_size=binding.hidden_size,
                    dtype=binding.dtype,
                    cuda_graph=self.cuda_graph,
                    device_index=self.device_index,
                )
                prepared.append(binding.op)
        except BaseException:
            for op in reversed(prepared):
                op.close()
            if self._moe_transport is not None:
                self._moe_transport.close()
            self.state = ParallelCollectiveState.CLOSED
            raise
        self.state = ParallelCollectiveState.PREPARED

    def begin_cuda_graph_capture(self) -> None:
        if not self.cuda_graph:
            raise RuntimeError("CUDA Graph collective capture requires cuda_graph=True.")
        if self.state is not ParallelCollectiveState.PREPARED:
            raise RuntimeError(
                "Parallel collective capture must begin after prepare, got "
                f"state={self.state.value}."
            )
        self.state = ParallelCollectiveState.CAPTURING

    def assert_can_capture(self) -> None:
        if self.has_graph_collectives and self.state is not ParallelCollectiveState.CAPTURING:
            raise RuntimeError(
                "CUDA Graph capture is closed for parallel collectives, got "
                f"state={self.state.value}."
            )

    def collect_local_cuda_graph_metadata(self) -> None:
        if self.state is not ParallelCollectiveState.CAPTURING:
            raise RuntimeError(
                "Local CUDA Graph metadata must be collected after capture, got "
                f"state={self.state.value}."
            )
        for binding in self._bindings:
            op = binding.op
            if op is None:
                raise RuntimeError("Parallel collective runtime was not fully prepared.")
            metadata = op.collect_local_cuda_graph_metadata()
            binding.local_metadata_summary = op.graph_metadata_summary(metadata)
            binding.local_metadata = metadata
        self.state = ParallelCollectiveState.CAPTURED

    def _local_schedule(self) -> tuple[tuple[object, ...], ...]:
        schedule = []
        for binding in self._bindings:
            op = binding.op
            if op is None:
                raise RuntimeError("Parallel collective runtime was not fully prepared.")
            schedule.append(
                (
                    tuple(sorted(binding.roles)),
                    binding.group.ranks,
                    binding.max_rows,
                    binding.hidden_size,
                    str(binding.dtype),
                    op.name,
                )
            )
        return tuple(schedule)

    def _validate_world_schedules(
        self,
        schedules: list[tuple[tuple[object, ...], ...]],
    ) -> None:
        world_ranks = self.parallel_context.world.ranks
        binding_count = len(self._bindings)
        if len(schedules) != len(world_ranks) or any(
            len(schedule) != binding_count for schedule in schedules
        ):
            raise RuntimeError(
                "Parallel ranks prepared different all-reduce binding counts: "
                f"{[len(schedule) for schedule in schedules]}."
            )

        for binding_idx in range(binding_count):
            entries = [schedule[binding_idx] for schedule in schedules]
            definitions = {(entry[0], *entry[2:]) for entry in entries}
            rank_groups = {
                world_rank: tuple(int(rank) for rank in entry[1])
                for world_rank, entry in zip(world_ranks, entries)
            }
            groups_agree = all(
                group
                and len(set(group)) == len(group)
                and rank in group
                and all(rank_groups.get(peer) == group for peer in group)
                for rank, group in rank_groups.items()
            )
            if len(definitions) != 1 or not groups_agree:
                raise RuntimeError(
                    "Parallel ranks prepared incompatible all-reduce schedules: "
                    f"binding={binding_idx} entries={entries}."
                )

    @staticmethod
    def _all_gather_object(local, group: ParallelGroup):
        if group.size == 1:
            return [local]
        gathered = [None for _ in range(group.size)]
        dist.all_gather_object(gathered, local, group=group.process_group)
        return gathered

    def exchange_cuda_graph_metadata(self) -> None:
        if self.state is not ParallelCollectiveState.CAPTURED:
            raise RuntimeError(
                "CUDA Graph metadata exchange requires completed local capture, got "
                f"state={self.state.value}."
            )

        schedule = self._local_schedule()
        world_schedules = self._all_gather_object(schedule, self.parallel_context.world)
        self._validate_world_schedules(world_schedules)

        errors: list[str] = []
        for binding in self._bindings:
            op = binding.op
            assert op is not None
            if binding.local_metadata_summary is None:
                raise RuntimeError("Parallel collective graph metadata was not checked.")
            local_record = (
                (
                    tuple(sorted(binding.roles)),
                    binding.group.ranks,
                    binding.max_rows,
                    binding.hidden_size,
                    str(binding.dtype),
                    op.name,
                    binding.local_metadata_summary,
                ),
                binding.local_metadata,
            )
            gathered = self._all_gather_object(local_record, binding.group)
            fingerprints = [record[0] for record in gathered]
            if any(fingerprint != fingerprints[0] for fingerprint in fingerprints[1:]):
                errors.append(
                    f"group={binding.group.ranks} metadata={fingerprints}"
                )
            binding.gathered_metadata = [record[1] for record in gathered]

        if errors:
            raise RuntimeError(
                "Parallel ranks captured incompatible all-reduce CUDA Graph buffers: "
                + "; ".join(errors)
            )
        self.state = ParallelCollectiveState.EXCHANGED

    def register_cuda_graph_buffers(self) -> None:
        if self.state is not ParallelCollectiveState.EXCHANGED:
            raise RuntimeError(
                "CUDA Graph buffer registration requires exchanged metadata, got "
                f"state={self.state.value}."
            )
        for binding in self._bindings:
            op = binding.op
            if op is None or binding.gathered_metadata is None:
                raise RuntimeError("Parallel collective graph metadata is incomplete.")
            op.register_cuda_graph_buffers(binding.gathered_metadata)
        self.state = ParallelCollectiveState.REGISTERED

    def mark_cuda_graph_replayable(self) -> None:
        if self.state is not ParallelCollectiveState.REGISTERED:
            raise RuntimeError(
                "CUDA Graph replay requires registered collective buffers, got "
                f"state={self.state.value}."
            )
        self.state = ParallelCollectiveState.REPLAYABLE

    def assert_cuda_graph_replayable(self) -> None:
        if self.has_graph_collectives and self.state is not ParallelCollectiveState.REPLAYABLE:
            raise RuntimeError(
                "CUDA Graph replay is blocked until parallel collective buffers are "
                f"registered, got state={self.state.value}."
            )

    def reset_for_cuda_graph_recapture(self) -> None:
        if self.state is ParallelCollectiveState.PREPARED:
            return
        if self.state is not ParallelCollectiveState.REPLAYABLE:
            raise RuntimeError(
                "Parallel collective runtime can reset only after a complete CUDA "
                f"Graph lifecycle, got state={self.state.value}."
            )
        for binding in reversed(self._bindings):
            if binding.op is not None:
                binding.op.close()
                binding.op = None
            binding.local_metadata = None
            binding.local_metadata_summary = None
            binding.gathered_metadata = None
        # AG/RS uses persistent transport workspace, independent of graph input
        # addresses. Keep it alive when replacing active and idle graphs.
        self.state = ParallelCollectiveState.OPEN
        self.prepare()

    def close(self) -> None:
        if self.state is ParallelCollectiveState.CLOSED:
            return
        for binding in reversed(self._bindings):
            if binding.op is not None:
                binding.op.close()
                binding.op = None
        if self._moe_transport is not None:
            self._moe_transport.close()
        self.state = ParallelCollectiveState.CLOSED


__all__ = [
    "DecodeParallelCollectives",
    "ParallelAllReduceHandle",
    "ParallelCollectiveRuntime",
    "ParallelCollectiveState",
]
