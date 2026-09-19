"""Exact reuse must survive eviction, row turnover, and captured replay."""

from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.cache_manager.methods.omnikv.lru import OmniKVLRU
from sparseengine.operators.indexed_host_copy import append_rows, gather_rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape,dtype",
    [((8, 128), torch.float16), ((1, 512), torch.bfloat16), ((1, 64), torch.bfloat16)],
)
def test_lru_exact_replay_eviction_and_request_turnover(shape, dtype):
    # The plain-copy tests cannot catch stale hits or eviction of still-selected
    # KV: replay changes selected order, request row, and physical slot contents.
    torch.manual_seed(42)
    storage = SimpleNamespace(num_slots=32, shapes=[shape], dtype=dtype)
    view = torch.empty(2, 8, dtype=torch.int32, device="cuda")
    lru = OmniKVLRU(storage, {1: 0, 2: 0}, 2, 6, 4, "cuda", view=view)
    hosts = [torch.randn(32, *shape, dtype=dtype).pin_memory() for _ in range(2)]
    pointers = [
        torch.tensor([h.data_ptr()], dtype=torch.uint64, device="cuda") for h in hosts
    ]
    table = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    rows = torch.arange(2, dtype=torch.int32, device="cuda")
    owners = rows.clone()
    lengths = torch.full((2,), 4, dtype=torch.int32, device="cuda")
    writes = torch.full((2,), 31, dtype=torch.int32, device="cuda")
    sources = [torch.randn(2, *shape, dtype=dtype, device="cuda") for _ in hosts]
    table.copy_(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device="cuda"))

    def run():
        lru.planned.clear()
        for layer, ptr, source in zip((1, 2), pointers, sources):
            plan = lru.prepare(layer, table, rows, owners, lengths, writes)
            gather_rows(
                ptr,
                lru.parts[layer][0],
                table,
                rows,
                lengths,
                capacity=4,
                component=0,
                exclude_slots=writes,
                plan=plan,
                miss_tokens=lru.misses(layer)[0],
                miss_counts=lru.misses(layer)[1],
            )
            append_rows(
                source,
                lru.parts[layer][0],
                lengths,
                writes,
                4,
                table=table,
                rows=rows,
                plan=plan,
            )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    cases = [
        ([[3, 0, 1, 4], [5, 8, 6, 9]], [0, 1], [31, 31]),
        ([[5, 0, 6, 1], [6, 7, 10, 11]], [0, 1], [31, 31]),
        ([[0, 1, 7, 8], [4, 6, 10, 12]], [1, 0], [7, 31]),
        ([[12, 4, 13, 6], [12, 4, 13, 6]], [0, 0], [13, -1]),
        ([[0, 1, 2, 3], [0, 1, 2, 3]], [1, 0], [31, 31]),
    ]
    for index, (selected, request_rows, current) in enumerate(cases):
        if index == 4:
            # Reuse the same physical slots for a different request, including
            # identical token IDs. Releasing its private row must invalidate hits.
            lru.invalidate(1)
            for host in hosts:
                host.add_(10)
            lru.invalidate(0)
        table.copy_(torch.tensor(selected, device="cuda", dtype=torch.int32))
        owners.copy_(torch.tensor(request_rows, device="cuda", dtype=torch.int32))
        writes.copy_(torch.tensor(current, device="cuda", dtype=torch.int32))
        old_keys = lru.metadata[0][1].cpu().tolist()
        old_ages = lru.metadata[0][2].cpu().tolist()
        old_clock = lru.metadata[0][3].cpu().tolist()
        graph.replay()
        torch.cuda.synchronize()
        outputs = [
            lru.parts[layer][0][view[:, :4].reshape(-1).long()]
            for layer in (1, 2)
        ]
        for batch, owner in enumerate(request_rows):
            if current[batch] < 0:
                for output in outputs:
                    assert torch.count_nonzero(output.view(2, 4, *shape)[batch]) == 0
                assert lru.metadata[0][3][1].item() == old_clock[1]
                continue
            # Independently check LRU victims: no selected item is evicted,
            # and any evicted unselected item is no newer than any retained one.
            keys = lru.metadata[0][1][owner].cpu().tolist()
            assert len({k for k in keys if k >= 0}) == sum(k >= 0 for k in keys)
            assert set(selected[batch]) <= set(keys)
            evicted = [
                old_ages[owner][i]
                for i, k in enumerate(old_keys[owner])
                if k >= 0 and k not in keys
            ]
            retained = [
                old_ages[owner][i]
                for i, k in enumerate(old_keys[owner])
                if k >= 0 and k in keys and k not in selected[batch]
            ]
            if evicted and retained:
                assert max(evicted) <= min(retained)
            for host, source, output in zip(hosts, sources, outputs):
                expected = host[selected[batch]].clone()
                if current[batch] in selected[batch]:
                    expected[selected[batch].index(current[batch])] = source[
                        batch
                    ].cpu()
                torch.testing.assert_close(
                    output.view(2, 4, *shape)[batch].cpu(), expected, atol=0, rtol=0
                )
                # Model write-through makes this new token valid history next step.
                host[current[batch]].copy_(source[batch].cpu())
