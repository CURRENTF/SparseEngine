"""Expert-directed dispatch around unchanged token-major expert compute."""

import torch

from sparseengine.distributed.moe_communication import MoeCommunication
from sparseengine.operators.all2all import prepare_parallel_all2all


class AllToAllMoeCommunication(MoeCommunication):
    def __init__(self, parallel_context, spec):
        if (parallel_context.attn_dp_size <= 1
                or parallel_context.attn_tp_size != 1
                or parallel_context.moe_tp_size != 1):
            raise ValueError("All-to-all currently requires TP=1, EP=DP attention.")
        self.group = parallel_context.moe_ep
        self.spec = spec
        self.op = None

    @property
    def name(self):
        return "unprepared_all2all" if self.op is None else self.op.name

    def prepare(self, *, device_index, cuda_graph):
        self.op = prepare_parallel_all2all(
            self.spec, group=self.group, device_index=device_index
        )

    def _run(self, hidden_states, *, route, experts, chunk_size, capacity):
        if self.op is None:
            raise RuntimeError("All-to-all transport is not prepared.")
        # Even an idle owner participates. Received rows may belong to any
        # other rank, and local expert chunking never inserts more collectives.
        if len(hidden_states):
            ids, weights = route(hidden_states)
        else:
            ids = hidden_states.new_empty((0, self.spec.top_k), dtype=torch.int64)
            weights = hidden_states.new_empty((0, self.spec.top_k), dtype=torch.float32)
        dispatch = self.op.dispatch(hidden_states, ids, weights, capacity=capacity)
        outputs = [
            experts(x, ids, weights)
            for x, ids, weights in zip(
                dispatch.hidden_states.split(chunk_size),
                dispatch.topk_ids.split(chunk_size),
                dispatch.topk_weights.split(chunk_size),
            )
        ]
        if len(outputs) == 1:
            output = outputs[0]
        else:
            output = torch.cat(outputs)
        return self.op.combine(output, dispatch)

    def close(self):
        if self.op is not None:
            self.op.close()
            self.op = None
