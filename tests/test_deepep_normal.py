"""Real NVLink transport oracle: routing changes, idle owners and graph reuse."""

from datetime import timedelta
from importlib.util import find_spec

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from sparseengine.distributed import ParallelTopology
from sparseengine.distributed.collective_runtime import ParallelCollectiveRuntime
from sparseengine.distributed.parallel_context import (
    init_parallel_context,
    reset_parallel_context,
)


def _worker(rank, rendezvous, real_experts):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    parallel = init_parallel_context(
        topology=ParallelTopology(1, 2, 2)
    )
    runtime = ParallelCollectiveRuntime(parallel, cuda_graph=True, device_index=rank)
    comm = runtime.request_moe_collectives(
        attention_max_rows=8,
        moe_max_rows=16,
        hidden_size=2048,
        dtype=torch.bfloat16,
        backend="deepepv1",
        max_local_tokens=8,
        num_experts=64 if real_experts else 8,
        top_k=4 if real_experts else 2,
    ).moe_transport
    try:
        runtime.prepare()
        if real_experts:
            _check_real_experts(comm, parallel)
            return
        op = comm.op
        for sizes in ((3, 1), (0, 5), (5, 0)):
            x = torch.full(
                (sizes[rank], 2048), 0.5 + rank / 4, device="cuda", dtype=torch.bfloat16
            )
            ids = torch.zeros((sizes[rank], 2), device="cuda", dtype=torch.int64)
            weights = torch.full(ids.shape, 0.25, device="cuda", dtype=torch.float32)

            def experts(tokens, ids, weights):
                # Independent expert family f_e(x)=(e+1)*x, exact binary inputs.
                local = (ids >= rank * 4) & (ids < (rank + 1) * 4)
                scale = torch.where(local, (ids + 1) * weights, 0).sum(-1, keepdim=True)
                safe = torch.where(local.any(-1, keepdim=True), tokens, 0)
                return (safe.float() * scale).to(tokens.dtype)

            capacity = max(sizes)

            def forward(
                x=x, ids=ids, weights=weights, capacity=capacity, experts=experts
            ):
                return comm.run_with_shared_experts(
                    x,
                    route=lambda _: (ids, weights),
                    experts=experts,
                    shared_experts=lambda t: t.square(),
                    chunk_size=3,
                    capacity=capacity,
                )

            ids[:, 1] = 4
            forward()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = forward()
            # Both local-only and cross-rank routes; a cached dispatch handle
            # would replay obsolete destinations and fail these transitions.
            for route in ((0, 4), (1, 2), (5, 7), (3, 6), (0, 4)):
                ids.copy_(torch.tensor(route, device="cuda").expand_as(ids))
                x.add_(0.125)
                graph.replay()
                expected = (x.float() * ((sum(route) + 2) / 4)).to(x.dtype) + x.square()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
            del graph
            assert comm.op is op
        runtime.close()
        runtime.close()
        with pytest.raises(RuntimeError, match="not prepared"):
            comm.run(x, route=None, experts=None, chunk_size=1, capacity=1)
    finally:
        torch.cuda.synchronize()
        runtime.close()
        reset_parallel_context()
        dist.destroy_process_group()


def _check_real_experts(comm, parallel):
    # The linear transport oracle cannot catch a mismatch with real expert
    # alignment, nonlinear activation or route weighting. Use GLM dimensions
    # and an independent FP32 Torch expert sum with fixed routes.
    from sparseengine.operators.moe import MoeOpSpec, resolve_moe_provider

    rank = parallel.moe_ep.rank
    spec = MoeOpSpec(
        64,
        32,
        2048,
        1536,
        4,
        torch.bfloat16,
        torch.bfloat16,
        None,
        2,
        True,
        routing_method="biased_sigmoid",
        max_num_tokens=16,
    )
    provider = resolve_moe_provider(spec)
    provider.prepare(spec, device=torch.device("cuda", rank), tp_rank=0, ep_rank=rank)
    torch.manual_seed(173 + rank)
    w13 = torch.randn(32, 3072, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(32, 2048, 1536, device="cuda", dtype=torch.bfloat16) * 0.02

    def experts(x, ids, weights):
        return provider.run(
            spec,
            x,
            ids,
            weights,
            w13,
            w2,
            None,
            None,
            local_expert_start=rank * 32,
            tp_rank=0,
            ep_rank=rank,
        )

    for sizes in ((3, 1), (0, 5), (1, 0)):
        capacity = max(sizes)
        x = torch.randn(sizes[rank], 2048, device="cuda", dtype=torch.bfloat16)
        ids = (
            torch.arange(sizes[rank], device="cuda")[:, None] * 17
            + rank * 7
            + torch.tensor([0, 11, 32, 43], device="cuda")
        ) % 64
        weights = (
            torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda")
            .expand_as(ids)
            .contiguous()
        )

        def forward(x=x, ids=ids, weights=weights, capacity=capacity):
            return comm.run(
                x,
                route=lambda _: (ids, weights),
                experts=experts,
                chunk_size=16,
                capacity=capacity,
            )

        forward()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = forward()
        x.mul_(0.75)
        graph.replay()
        # Gather only to construct the independent reference; production uses
        # DeepEP for dispatch and combine, including the idle-owner cases.
        gathered = []
        for tensor, fill in ((x, 0), (ids, -1), (weights, 0)):
            local = tensor.new_full((capacity, *tensor.shape[1:]), fill)
            local[: len(tensor)].copy_(tensor)
            global_tensor = tensor.new_empty((capacity * 2, *tensor.shape[1:]))
            dist.all_gather_into_tensor(
                global_tensor, local, group=parallel.moe_ep.process_group
            )
            gathered.append(global_tensor)
        global_x, global_ids, global_weights = gathered
        expected = torch.zeros(capacity * 2, 2048, device="cuda")
        for local_expert in range(32):
            rows, choices = torch.where(global_ids == local_expert + rank * 32)
            if rows.numel():
                gate, up = F.linear(
                    global_x[rows].float(), w13[local_expert].float()
                ).chunk(2, -1)
                partial = F.linear(F.silu(gate) * up, w2[local_expert].float())
                expected.index_add_(
                    0, rows, partial * global_weights[rows, choices, None]
                )
        dist.all_reduce(expected, group=parallel.moe_ep.process_group)
        reference = expected[rank * capacity : rank * capacity + sizes[rank]]
        torch.testing.assert_close(result.float(), reference, atol=0.004, rtol=0.02)
        torch.cuda.synchronize()
        del graph


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or find_spec("deep_ep") is None,
    reason="requires DeepEP V1 and two idle NVLink CUDA devices",
)
@pytest.mark.parametrize(
    "real_experts", [False, True], ids=["transport", "glm-experts"]
)
def test_deepep_normal_dynamic_routing_idle_and_replaced_graphs(tmp_path, real_experts):
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", real_experts),
        nprocs=2,
        join=True,
    )
