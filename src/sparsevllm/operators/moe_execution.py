"""Prepared local MoE branch execution, independent of model and expert kernels.

The caller supplies read-only-input branches producing token-aligned local
partials. Transport retains ownership of the addition/reduction order. A shared
branch must not issue collectives or mutate workspace owned by the routed branch.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from sparsevllm.operators.workspace import bind_module_workspace_lane
from sparsevllm.platforms import device_runtime


class MoeExecutionPlan:
    def __init__(
        self, *, routed: Callable, shared: Callable, communication,
        chunk_size: int | None, fused: Callable | None = None,
        fuse_prefill: bool = False, fuse_decode: bool = False,
        fusion_token_limit: int | None = None,
        reduce_decode_branches_separately: bool = False,
        shared_modules: tuple[torch.nn.Module, ...] = (),
        finish: Callable | None = None,
    ):
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("MoE chunk_size must be positive")
        if (fuse_prefill or fuse_decode) and fused is None:
            raise ValueError("Fused MoE execution requires a fused callable")
        self.finish = finish
        self.routed = routed
        self.shared = shared
        self.communication = communication
        self.chunk_size = chunk_size
        self.fused = fused
        self.fuse_prefill = fuse_prefill
        self.fuse_decode = fuse_decode
        self.fusion_token_limit = fusion_token_limit
        self.reduce_decode_branches_separately = reduce_decode_branches_separately
        self.shared_modules = shared_modules
        self._prepared = False
        self.stream = None
        self.input_ready = None
        self.shared_ready = None

    def prepare(self, stream):
        """Bind once before warmup/capture; None selects the serial reference."""
        if self._prepared:
            raise RuntimeError("MoE execution must be prepared once before graph capture")
        self._prepared = True
        self.stream = stream
        # One lane for all sequential shared branches, not one allocation per layer.
        if stream is not None:
            for module in self.shared_modules:
                bind_module_workspace_lane(module, "moe_shared")
        self.input_ready = device_runtime.new_event() if stream is not None else None
        self.shared_ready = device_runtime.new_event() if stream is not None else None

    def _chunked(self, branch, hidden_states):
        if self.chunk_size is None or hidden_states.shape[0] <= self.chunk_size:
            return branch(hidden_states)
        parts = [branch(x) for x in hidden_states.split(self.chunk_size)]
        if isinstance(parts[0], tuple):
            return tuple(torch.cat(items, dim=0) for items in zip(*parts))
        return torch.cat(parts, dim=0)

    def __call__(self, hidden_states, *, is_prefill: bool):
        fused = self.fuse_prefill if is_prefill else self.fuse_decode
        if fused and (self.fusion_token_limit is None
                      or hidden_states.shape[0] <= self.fusion_token_limit):
            return self.communication.combine(self._chunked(self.fused, hidden_states))
        # Decode graph captures this fork/join, including its cross-stream
        # dependencies. Prefill retains serial execution: large GEMMs compete
        # for the same device resources and need a separate measured policy.
        if self.stream is not None and not is_prefill and hidden_states.shape[0]:
            device_runtime.record_event(self.input_ready, hidden_states.device)
            with device_runtime.stream_context(self.stream):
                device_runtime.wait_event(self.input_ready, hidden_states.device)
                shared = self.shared(hidden_states)
                device_runtime.record_event(self.shared_ready, hidden_states.device)
            routed = self._chunked(self.routed, hidden_states)
            device_runtime.wait_event(self.shared_ready, hidden_states.device)
            # shared was allocated on the auxiliary stream and is consumed on
            # the caller stream. Tell the allocator its last-use stream. Input
            # is safe to reuse on the caller stream after the join above.
            device_runtime.record_tensor_stream(shared, device_runtime.current_stream(hidden_states.device))
        else:
            routed = self._chunked(self.routed, hidden_states)
            shared = self.shared(hidden_states)
        if self.finish is not None:
            # Model semantics may require gates or normalization after reduction.
            # This callback runs only after both local branches have joined.
            return self.finish(routed, shared)
        return self.communication.combine_local_branches(
            routed, shared,
            reduce_separately=(not is_prefill and self.reduce_decode_branches_separately),
        )


def prepare_model_moe_execution(model, device):
    """Share one auxiliary stream across a worker's sequential MoE layers.

    Layer-owned events are captured with each graph. Every branch rejoins the
    calling stream before returning; this does not introduce concurrent graph
    replay or new per-step storage. Existing ordered replay ownership applies.
    """
    plans = [m.moe_execution for m in model.modules()
             if isinstance(getattr(m, "moe_execution", None), MoeExecutionPlan)]
    # An unconditional fused decode never executes the independent shared
    # branch. Keep its providers on their original workspace lane and avoid
    # allocating unused events. A bounded fusion route still needs fork/join
    # resources for batches above its token limit, even on a single GPU.
    parallel_plans = {plan for plan in plans
                      if not plan.fuse_decode or plan.fusion_token_limit is not None}
    stream = (device_runtime.new_stream(device)
              if parallel_plans and device_runtime.supports_streams(device) else None)
    for plan in plans:
        plan.prepare(stream if plan in parallel_plans else None)
    return stream
