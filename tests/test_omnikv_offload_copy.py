"""Pinned-memory copies must follow replay metadata and protect padding rows."""

import pytest
import torch

from sparseengine.operators.indexed_host_copy import (
    append_rows,
    gather_rows,
    store_rows,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((4, 128), torch.bfloat16),
        ((4, 128), torch.float16),
        ((1, 512), torch.bfloat16),
        ((1, 64), torch.bfloat16),
    ],
)
def test_indexed_host_copy_replay(shape, dtype):
    torch.manual_seed(17)
    host = torch.randn(23, *shape, dtype=dtype).pin_memory()
    ptr = torch.tensor([host.data_ptr()], dtype=torch.uint64, device="cuda")
    table = torch.tensor(
        [[8, 2, 19, 4], [10, 3, 5, 7]], dtype=torch.int32, device="cuda"
    )
    rows = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([4, 3], dtype=torch.int32, device="cuda")
    slots = torch.tensor([7, -1], dtype=torch.int32, device="cuda")
    source = torch.randn(2, *shape, dtype=dtype, device="cuda")
    output = torch.zeros(8, *shape, dtype=dtype, device="cuda")

    def run():
        store_rows(source, ptr, slots, 0)
        gather_rows(
            ptr,
            output,
            table,
            rows,
            lengths,
            capacity=4,
            component=0,
            skip_last=True,
            block_budget=4,
            exclude_slots=slots,
        )
        append_rows(source, output, lengths, slots, 4)

    run()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for order in ([1, 0], [0, 1]):
        rows.copy_(torch.tensor(order, device="cuda", dtype=torch.int32))
        source.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        actual = output.cpu().view(2, 4, *shape)
        for batch, row in enumerate(order):
            n = int(lengths[batch]) - 1
            expected = (
                host[table[row, :n].cpu().long()]
                if batch == 0
                else torch.zeros_like(actual[batch, :n])
            )
            torch.testing.assert_close(actual[batch, :n], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[0, 3], source[0].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(host[7], source[0].cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_current_token_follows_selection_when_recent_budget_is_zero():
    # Top-K is unsorted and may exclude the newest token entirely. The old
    # last-position append corrupted the selected history in both situations.
    host = torch.randn(11, 1, 64, dtype=torch.bfloat16).pin_memory()
    pointers = torch.tensor([host.data_ptr()], dtype=torch.uint64, device="cuda")
    table = torch.tensor([[2, 5, 8], [6, 1, 9]], dtype=torch.int32, device="cuda")
    rows = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
    write_slots = torch.tensor([5, 7], dtype=torch.int32, device="cuda")
    current = torch.randn(2, 1, 64, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(6, 1, 64, dtype=torch.bfloat16, device="cuda")

    def run():
        gather_rows(
            pointers,
            output,
            table,
            rows,
            lengths,
            capacity=3,
            component=0,
            exclude_slots=write_slots,
        )
        append_rows(current, output, lengths, write_slots, 3, table=table, rows=rows)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for selected in ([[2, 5, 8], [6, 1, 9]], [[8, 2, 9], [7, 1, 6]]):
        table.copy_(torch.tensor(selected, dtype=torch.int32, device="cuda"))
        current.add_(1)
        graph.replay()
        actual = output.cpu().view(2, 3, 1, 64)
        for batch, slots in enumerate(selected):
            reference = host[slots].clone()
            for index, slot in enumerate(slots):
                if slot == int(write_slots[batch]):
                    reference[index] = current[batch].cpu()
            torch.testing.assert_close(actual[batch], reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "batch,shape,dtype",
    [
        (3, (8, 128), torch.bfloat16),
        (5, (1, 512), torch.bfloat16),
        (3, (1, 64), torch.float16),
    ],
)
def test_cached_gather_rebalances_dynamic_compacted_misses(batch, shape, dtype):
    # Per-request copy tests miss errors in the global tile-to-request mapping,
    # particularly empty requests and partial tiles in non-power-of-two batches.
    torch.manual_seed(19)
    capacity, slots = 9, 32
    host = torch.randn(slots, *shape, dtype=dtype).pin_memory()
    pointers = torch.tensor([host.data_ptr()], dtype=torch.uint64, device="cuda")
    table_cpu = torch.stack(
        [torch.randperm(slots)[: capacity + 2] for _ in range(batch)]
    ).int()
    rows_cpu = torch.randperm(batch).int()
    slot_map_cpu = torch.randperm(slots).int()
    lengths_cpu = torch.full((batch,), capacity, dtype=torch.int32)
    lengths_cpu[-1] = capacity - 2
    writes_cpu = table_cpu[rows_cpu.long(), capacity - 1].clone()
    positions = torch.randperm(batch * capacity).reshape(batch, capacity).int()
    plan_cpu = -positions - 1
    plan_cpu[:, ::3] = positions[:, ::3]
    misses_cpu = torch.stack(
        [torch.randperm(capacity) for _ in range(batch)]
    ).int()
    table, rows, slot_map, lengths, writes, plan, misses = [
        tensor.cuda()
        for tensor in (
            table_cpu, rows_cpu, slot_map_cpu, lengths_cpu,
            writes_cpu, plan_cpu, misses_cpu,
        )
    ]
    counts = torch.zeros(batch, dtype=torch.int32, device="cuda")
    cache = torch.full((batch * capacity, *shape), -17, dtype=dtype, device="cuda")

    def run():
        gather_rows(
            pointers,
            cache,
            table,
            rows,
            lengths,
            capacity=capacity,
            component=0,
            block_budget=2,
            slot_map=slot_map,
            exclude_slots=writes,
            plan=plan,
            miss_tokens=misses,
            miss_counts=counts,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for values in (
        [0] * batch,
        [0, 1, capacity, 3, capacity - 1][:batch],
        [capacity] * batch,
    ):
        counts.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        misses_cpu = misses_cpu.flip(1)
        misses.copy_(misses_cpu)
        cache.fill_(-17)
        graph.replay()
        expected = torch.full_like(cache.cpu(), -17)
        for request, count in enumerate(values):
            for token in misses_cpu[request, :count].tolist():
                slot = int(table_cpu[rows_cpu[request], token])
                entry = int(plan_cpu[request, token])
                if (
                    token < lengths_cpu[request]
                    and slot != writes_cpu[request]
                    and entry < 0
                ):
                    expected[-entry - 1] = host[slot_map_cpu[slot]]
        torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
