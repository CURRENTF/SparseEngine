"""AG/RS must preserve token ownership, including idle and graph-padded ranks."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparseengine.distributed import ParallelTopology
from sparseengine.distributed.moe_communication import (
    AllGatherReduceScatterMoeCommunication,
)
from sparseengine.distributed.parallel_context import (
    init_parallel_context,
    reset_parallel_context,
)
from sparseengine.distributed.topology import parallel_group_ranks


def test_dp_attention_groups_share_experts_without_replicating_attention():
    topology = ParallelTopology(1, 4, 4)
    groups = parallel_group_ranks(topology)
    assert topology.world_size == 4
    assert all(len(group) == 1 for group in groups["attn_tp"])
    assert groups["moe_ep"] == groups["attn_dp"]
    assert set(groups["moe_ep"][0]) == set(range(topology.world_size))
    topology = ParallelTopology(1, 2, 4)
    assert topology.moe_tp_size == 2


def _agrs_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    parallel = init_parallel_context(
        topology=ParallelTopology(1, 2, 2)
    )
    communication = AllGatherReduceScatterMoeCommunication(parallel)
    try:
        for sizes in ((3, 1), (0, 5), (5, 0)):
            capacity = max(sizes)
            source = (
                torch.arange(sizes[rank] * 32, device="cuda", dtype=torch.float32).view(
                    -1, 32
                )
                / 32
                + rank
            )

            def forward(source=source, capacity=capacity):
                # Independent additive expert contributions; the owner should
                # get (1+2)*x regardless of remote/padded token rows. Chunking
                # must stay inside dispatch/combine so idle ranks cannot skip
                # or reorder a collective.
                return communication.run(
                    source,
                    route=lambda x: (
                        torch.zeros((len(x), 1), dtype=torch.int64, device=x.device),
                        torch.ones((len(x), 1), device=x.device),
                    ),
                    experts=lambda x, ids, weights: x * (rank + 1) * weights,
                    chunk_size=2,
                    capacity=capacity,
                )

            torch.testing.assert_close(forward(), source * 3)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = forward()
            for _ in range(3):
                source.add_(1)
                graph.replay()
                torch.testing.assert_close(result, source * 3)
            del graph
    finally:
        torch.cuda.synchronize()
        reset_parallel_context()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two idle CUDA devices"
)
def test_agrs_uneven_idle_and_graph_replay(tmp_path: Path):
    mp.spawn(
        _agrs_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True
    )


def _prepared_agrs_worker(
    rank, rendezvous, force_torch, world_size, hidden_size, dtype
):
    from sparseengine.distributed.collective_runtime import ParallelCollectiveRuntime
    from sparseengine.operators.agrs import FlashInferMixedAgRsProvider
    from sparseengine.operators.registry import SupportResult
    from sparseengine.utils.context import get_context, reset_context, set_context

    if force_torch:
        FlashInferMixedAgRsProvider.supports = classmethod(
            lambda cls, spec, caps: SupportResult.unsupported("test selects NCCL")
        )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size
    )
    parallel = init_parallel_context(
        topology=ParallelTopology(1, world_size, world_size)
    )
    runtime = ParallelCollectiveRuntime(parallel, cuda_graph=True, device_index=rank)
    collectives = runtime.request_moe_collectives(
        attention_max_rows=1024, moe_max_rows=2048, max_local_tokens=1024,
        hidden_size=hidden_size, dtype=dtype, backend="agrs",
        num_experts=8, top_k=2,
    )
    runtime.prepare()
    communication = collectives.moe_transport
    set_context(False)
    try:
        for sizes in (
            (3,) + (1,) * (world_size - 1),
            (0,) * (world_size - 1) + (5,),
            (5,) + (0,) * (world_size - 1),
            (513,) * (world_size - 1) + (0,),
        ):
            capacity = max(sizes)
            sources = [
                (
                    torch.arange(rows * hidden_size, device="cuda")
                    .remainder(16)
                    .reshape(rows, hidden_size)
                    / 16
                    + peer
                ).to(dtype)
                for peer, rows in enumerate(sizes)
            ]
            source = sources[rank]
            set_context(rank == 0)
            if force_torch:
                get_context().moe_token_sizes = sizes
            dispatched = communication.dispatch(source, capacity=capacity)
            dispatched_rows = sum(sizes) if force_torch else capacity * world_size
            expected = torch.zeros(dispatched_rows, hidden_size, device="cuda", dtype=dtype)
            offset = 0
            for peer, value in enumerate(sources):
                start = offset if force_torch else peer * capacity
                expected[start : start + len(value)].copy_(value)
                offset += len(value)
            # Check AG ordering independently: paired inverse permutations in
            # AG/RS could otherwise pass an end-to-end additive expert oracle.
            torch.testing.assert_close(
                dispatched.hidden_states, expected, atol=0, rtol=0
            )
            partial = (
                torch.arange(dispatched_rows * hidden_size, device="cuda")
                .remainder(16)
                .reshape(dispatched_rows, hidden_size)
                .to(dtype)
            )
            result = communication.combine(partial * (rank + 1), dispatched)
            local_start = sum(sizes[:rank]) if force_torch else rank * capacity
            reference = partial[local_start : local_start + sizes[rank]] * (
                world_size * (world_size + 1) // 2
            )
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
            set_context(False)
            get_context().moe_token_sizes = None

            def forward(source=source, capacity=capacity):
                # A nonlinear owner-local branch catches missing/doubled shared
                # outputs and stale graph inputs after capture or replacement.
                return communication.run_with_shared_experts(
                    source,
                    shared_experts=lambda x: x.square(),
                    route=lambda x: (None, None),
                    experts=lambda x, ids, weights: x,
                    chunk_size=256,
                    capacity=capacity,
                )

            torch.testing.assert_close(
                forward(), source * world_size + source.square(), atol=0, rtol=0
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = forward()
            for _ in range(3):
                source.add_(0.125)
                graph.replay()
                torch.testing.assert_close(
                    result, source * world_size + source.square(), atol=0, rtol=0
                )
            torch.cuda.synchronize()
            del graph
    finally:
        torch.cuda.synchronize()
        runtime.close()
        reset_context()
        reset_parallel_context()
        dist.destroy_process_group()


@pytest.mark.parametrize("force_torch", [False, True], ids=["default", "torch"])
@pytest.mark.parametrize(
    "world_size,hidden_size,dtype",
    [(2, 2048, torch.bfloat16), (4, 4096, torch.float16)],
    ids=["two-rank-bf16", "four-rank-fp16"],
)
def test_prepared_agrs_rank_order_idle_and_replaced_graphs(
    tmp_path, force_torch, world_size, hidden_size, dtype
):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} idle CUDA devices")
    mp.spawn(
        _prepared_agrs_worker,
        args=(
            f"file://{tmp_path / 'prepared-rendezvous'}",
            force_torch,
            world_size,
            hidden_size,
            dtype,
        ),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("step_capacity,expected", [(None, 3), (8, 8)])
def test_shared_expert_execution_uses_the_agreed_step_capacity(monkeypatch, step_capacity, expected):
    from unittest.mock import Mock

    from sparseengine.distributed.moe_communication import MoeCommunication
    from sparseengine.utils.context import get_context

    # A replica's local row count must not override the capacity agreed with peers.
    monkeypatch.setattr(get_context(), "moe_token_capacity", step_capacity)
    transport = MoeCommunication()
    x = torch.ones(3, 4)
    transport._run = Mock(return_value=x)
    transport.run_with_shared_experts(x, shared_experts=lambda t: t)
    assert transport._run.call_args.kwargs["capacity"] == expected


def _hybrid_agrs_worker(rank, rendezvous, cuda):
    from sparseengine.distributed.collective_runtime import ParallelCollectiveRuntime
    from sparseengine.utils.context import get_context, reset_context, set_context

    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if cuda else "gloo", init_method=rendezvous, rank=rank, world_size=4
    )
    parallel = init_parallel_context(topology=ParallelTopology(2, 4, 2))
    runtime = ParallelCollectiveRuntime(parallel, cuda_graph=cuda, device_index=rank)
    transport = runtime.request_moe_collectives(
        attention_max_rows=8, moe_max_rows=8, max_local_tokens=8,
        hidden_size=256, dtype=torch.bfloat16 if cuda else torch.float32,
        backend="agrs", num_experts=8, top_k=2,
    ).moe_transport
    runtime.prepare()
    set_context(False)
    graphs = []
    try:
        if cuda:
            runtime.begin_cuda_graph_capture()
        for sizes in ((3, 1), (0, 5), (5, 0)):
            source = torch.full(
                (sizes[parallel.attn_dp_rank], 256),
                parallel.attn_dp_rank + 1.0, device=device,
                dtype=torch.bfloat16 if cuda else torch.float32,
            )

            capacity = max(sizes)
            if not cuda:
                get_context().moe_token_sizes = sizes

            def forward(source=source, capacity=capacity):
                return transport.run_with_shared_experts(
                    source,
                    route=lambda x: (torch.zeros(len(x), 1, device=device, dtype=torch.int64),
                                     torch.ones(len(x), 1, device=device)),
                    experts=lambda x, ids, weights: x * (rank + 1),
                    shared_experts=lambda x: x * (parallel.attn_tp_rank + 1),
                    chunk_size=2, capacity=capacity,
                )

            # Four expert partitions contribute 1+2+3+4; the two shared
            # shards contribute 1+2 once per replica, regardless of padding.
            torch.testing.assert_close(forward(), source * 13)
            if not cuda:
                get_context().moe_token_sizes = None
            if cuda:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    forward()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = forward()
                graphs.append((graph, source, output))
        if cuda:
            runtime.collect_local_cuda_graph_metadata()
            runtime.exchange_cuda_graph_metadata()
            runtime.register_cuda_graph_buffers()
            runtime.mark_cuda_graph_replayable()
            for graph, source, output in graphs:
                for _ in range(3):
                    source.add_(1)
                    graph.replay()
                    torch.testing.assert_close(output, source * 13)
            torch.cuda.synchronize()
    finally:
        graphs.clear()
        runtime.close()
        reset_context()
        reset_parallel_context()
        dist.destroy_process_group()


def test_hybrid_agrs_preserves_replica_ownership_and_sums_shared_shards(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARSEENGINE_PLATFORM", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.spawn(_hybrid_agrs_worker,
             args=(f"file://{tmp_path / 'hybrid'}", False), nprocs=4, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four idle CUDA devices")
def test_hybrid_agrs_prepared_collectives_and_graph_replay(tmp_path):
    mp.spawn(_hybrid_agrs_worker,
             args=(f"file://{tmp_path / 'hybrid-cuda'}", True), nprocs=4, join=True)
