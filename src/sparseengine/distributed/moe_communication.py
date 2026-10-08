"""Token transport around local expert execution.

Transport owns token layout and reduction; expert providers keep their existing
token-major input and local-expert weight contracts. Routing after an AG avoids
two small collectives for route IDs and weights. An expert-directed transport
can instead route before dispatch without changing the local expert callable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from sparseengine.distributed.parallel_context import ParallelContext
from sparseengine.operators.agrs import prepare_parallel_agrs
from sparseengine.utils.context import get_context


@dataclass(frozen=True)
class MoeDispatch:
    hidden_states: torch.Tensor
    local_rows: int
    capacity: int
    token_sizes: tuple[int, ...] | None = None


class MoeCommunication:
    """Execution boundary that lets transport choose where routing happens.

    Replicated dispatch routes after dispatch. An expert-directed backend can
    override this operation to route before dispatch, while retaining the
    model's router and compatible local expert provider.
    """

    def _run(self, hidden_states, *, route, experts, chunk_size, capacity):
        dispatch = self.dispatch(hidden_states, capacity=capacity)
        outputs = []
        for chunk in dispatch.hidden_states.split(chunk_size, dim=0):
            ids, weights = route(chunk)
            outputs.append(experts(chunk, ids, weights))
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        return self.combine(output, dispatch)

    def routing_token_metadata(self, input_ids, *, capacity=None):
        """Expert-directed or replicated transports route their local tokens."""
        return input_ids

    def _finish(self, output):
        return output

    def run(self, hidden_states, *, shared_experts=None, capacity=None, **kwargs):
        if capacity is None:
            capacity = get_context().moe_token_capacity
            if capacity is None:
                # Startup warmup/capture uses identical shapes on every replica.
                capacity = hidden_states.shape[0]
        output = self._run(hidden_states, capacity=capacity, **kwargs)
        if shared_experts is not None and len(hidden_states):
            output = output + shared_experts(hidden_states)
        return self._finish(output)

    def run_with_shared_experts(self, hidden_states, *, shared_experts, **kwargs):
        """Sum routed and shared partials before completing the TP reduction."""
        return self.run(hidden_states, shared_experts=shared_experts, **kwargs)


class AllReduceMoeCommunication(MoeCommunication):
    """Replicated tokens, partial expert outputs, replicated reduced output."""

    def __init__(self, reduce: Callable[[torch.Tensor], torch.Tensor]):
        self._reduce = reduce

    def dispatch(
        self, hidden_states: torch.Tensor, *, capacity: int | None = None
    ) -> MoeDispatch:
        return MoeDispatch(
            hidden_states, hidden_states.shape[0], hidden_states.shape[0]
        )

    def combine(
        self, output: torch.Tensor, dispatch: MoeDispatch | None = None
    ) -> torch.Tensor:
        return self._reduce(output)

    def combine_local_branches(self, routed, shared, *, reduce_separately=False):
        """Compose token-aligned world partials without changing BF16 order."""
        if reduce_separately:
            partials = self.combine(torch.stack((routed, shared), dim=0))
            return partials[0] + partials[1]
        return self.combine(routed + shared)


class AllGatherReduceScatterMoeCommunication(MoeCommunication):
    """Gather DP tokens per TP lane, scatter partials, then complete the TP sum.

    The runner agrees on capacity once per step, outside the layer loop. Equal
    padded rank segments make NCCL AG/RS capturable; slicing restores the local
    token order. Padding never enters attention or persistent request state.
    """

    def __init__(
        self, parallel_context: ParallelContext, *, max_rows=None, hidden_size=None,
        dtype=None, reduce=None,
    ):
        if parallel_context.moe_tp_size != 1:
            raise ValueError("AG/RS currently requires MoE TP=1 (EP=world size).")
        self.group = parallel_context.attn_dp
        self._reduce = reduce or parallel_context.attn_tp.all_reduce
        self.max_rows = max_rows
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.op = None
        self.closed = False

    def _finish(self, output):
        # RS sums experts on one TP lane; complete the other lanes locally.
        # Idle replicas have no output rows and need no TP collective.
        return self._reduce(output) if len(output) else output

    @property
    def name(self):
        return "torch_distributed_agrs" if self.op is None else self.op.name

    def prepare(self, *, device_index, cuda_graph):
        if self.closed or self.op is not None:
            raise RuntimeError("AG/RS transport can only be prepared once.")
        self.op = prepare_parallel_agrs(
            self.group, max_rows=self.max_rows, hidden_size=self.hidden_size,
            dtype=self.dtype, device_index=device_index,
        )

    def close(self):
        if self.op is not None:
            self.op.close()
            self.op = None
        self.closed = True

    def _use_prepared(self, capacity):
        if self.closed:
            raise RuntimeError("AG/RS transport is closed.")
        if self.max_rows is not None and self.op is None:
            raise RuntimeError("AG/RS transport is not prepared.")
        # Only the agreed capacity can select the collective. Local phases may
        # differ (e.g. a one-token prefill opposite a decode or idle rank).
        return (
            self.op is not None
            and capacity <= self.max_rows
        )

    def dispatch(self, hidden_states: torch.Tensor, *, capacity: int) -> MoeDispatch:
        rows, hidden = hidden_states.shape
        if capacity < rows or capacity <= 0:
            raise ValueError(
                f"MoE capacity {capacity} cannot hold {rows} local tokens."
            )
        token_sizes = get_context().moe_token_sizes
        if token_sizes is not None:
            token_sizes = tuple(int(value) for value in token_sizes)
            if len(token_sizes) != self.group.size:
                raise ValueError(
                    "DP token-size vector does not match the AG/RS group: "
                    f"sizes={token_sizes} group_size={self.group.size}."
                )
            if token_sizes[self.group.rank] != rows:
                raise ValueError(
                    "DP token-size vector does not match the local model rows: "
                    f"sizes={token_sizes} rank={self.group.rank} rows={rows}."
                )
            if max(token_sizes) != capacity:
                raise ValueError(
                    "DP token-size vector does not match the agreed capacity: "
                    f"sizes={token_sizes} capacity={capacity}."
                )
            if (
                len(set(token_sizes)) > 1
                and self.op is not None
                and self.op.supports_variable_sizes
            ):
                gathered = hidden_states.new_empty((sum(token_sizes), hidden))
                self.op.all_gatherv(gathered, hidden_states.contiguous(), token_sizes)
                return MoeDispatch(
                    gathered, rows, capacity, token_sizes=token_sizes
                )
        if rows == capacity:
            local = hidden_states.contiguous()
        else:
            local = hidden_states.new_zeros((capacity, hidden))
            local[:rows].copy_(hidden_states)
        gathered = hidden_states.new_empty((capacity * self.group.size, hidden))
        if self._use_prepared(capacity):
            self.op.all_gather(gathered, local)
        elif self.group.size == 1:
            gathered.copy_(local)
        else:
            dist.all_gather_into_tensor(gathered, local, group=self.group.process_group)
        return MoeDispatch(gathered, rows, capacity)

    def routing_token_metadata(self, input_ids, *, capacity=None):
        """Gather IDs once per model step in the same layout as hidden dispatch."""
        if capacity is None:
            capacity = get_context().moe_token_capacity or input_ids.numel()
        local = torch.zeros(capacity, dtype=torch.int64, device=input_ids.device)
        local[:input_ids.numel()].copy_(input_ids)
        gathered = torch.empty(capacity*self.group.size, dtype=torch.int64, device=input_ids.device)
        if self.group.size == 1:
            gathered.copy_(local)
        else:
            dist.all_gather_into_tensor(gathered, local, group=self.group.process_group)
        sizes = get_context().moe_token_sizes
        if (sizes is not None and len(set(sizes)) > 1 and self.op is not None
                and self.op.supports_variable_sizes):
            gathered = torch.cat([gathered[i*capacity:i*capacity+size] for i, size in enumerate(sizes)])
        return gathered

    def combine(self, output: torch.Tensor, dispatch: MoeDispatch) -> torch.Tensor:
        if dispatch.token_sizes is not None:
            if output.shape[0] != sum(dispatch.token_sizes):
                raise ValueError(
                    "Variable AG/RS expert output does not match the dispatched token layout."
                )
            local = output.new_empty((dispatch.local_rows, output.shape[1]))
            self.op.reduce_scatterv(
                local, output.contiguous(), dispatch.token_sizes
            )
            return local
        if output.shape[0] != dispatch.capacity * self.group.size:
            raise ValueError(
                "Local expert output does not match the dispatched token layout."
            )
        local = output.new_empty((dispatch.capacity, output.shape[1]))
        if self._use_prepared(dispatch.capacity):
            self.op.reduce_scatter(local, output.contiguous())
        elif self.group.size == 1:
            local.copy_(output)
        else:
            dist.reduce_scatter_tensor(
                local, output.contiguous(), group=self.group.process_group
            )
        return local[: dispatch.local_rows]


def prepare_moe_communication(parallel_context: ParallelContext, collectives=None):
    if parallel_context.attn_dp_size > 1:
        if collectives is not None:
            return collectives.moe_transport
        return AllGatherReduceScatterMoeCommunication(parallel_context)
    return AllReduceMoeCommunication(
        collectives.moe.run
        if collectives is not None
        else parallel_context.world.all_reduce
    )
