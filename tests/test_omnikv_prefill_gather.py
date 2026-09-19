"""Full prefill views must combine shared history and private GPU chunk KV."""

import pytest
import torch

from sparseengine.operators.indexed_host_copy import (
    gather_prefill_rows, gather_prefill_history, scatter_prefill_current,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize(
    "width,dtype,query_lengths,history_lengths,latent",
    [
        (1024, torch.bfloat16, [17, 9, 33], [0, 0, 0], False),
        (1024, torch.float16, [0, 1, 64], [0, 32, 1], False),
        (512, torch.bfloat16, [1, 9, 5], [32, 32, 17], True),
        (64, torch.bfloat16, [17, 9, 33], [32, 32, 1], True),
        (128, torch.bfloat16, [1] * 8, [0, 4097, 1, 0, 7, 11, 2, 0], False),
    ],
)
def test_full_prefill_view_uses_gpu_current_chunk(
    width, dtype, query_lengths, history_lengths, latent, prefetch
):
    # Existing host-only gather tests cannot catch a current-chunk round trip,
    # incorrect query offsets, or damage to shared prefix/private suffix views.
    torch.manual_seed(42)
    batch = len(query_lengths)
    lengths_cpu = torch.tensor(
        [q + h for q, h in zip(query_lengths, history_lengths)], dtype=torch.int32
    )
    capacity = int(lengths_cpu.max())
    slots = batch * (capacity + 3)
    table_cpu = torch.randperm(slots).reshape(batch, capacity + 3).int()
    rows_cpu = torch.randperm(batch).int()
    if history_lengths[0] and history_lengths[1]:
        shared = min(history_lengths[0], history_lengths[1])
        table_cpu[rows_cpu[1], :shared] = table_cpu[rows_cpu[0], :shared]
    map_cpu = torch.randperm(slots * 2)[:slots].int()
    hosts = [torch.randn(slots * 2, 1, width, dtype=dtype) for _ in range(2)]
    for request, (history, query) in enumerate(zip(history_lengths, query_lengths)):
        current_slots = table_cpu[rows_cpu[request], history : history + query].long()
        for host in hosts:
            host[map_cpu[current_slots].long()] = float("nan")
    pinned = [host.pin_memory() for host in hosts]
    pointers = torch.tensor(
        [host.data_ptr() for host in pinned], dtype=torch.uint64, device="cuda"
    )
    cu_cpu = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    table, rows, lengths, slot_map, cu_query = [
        x.cuda() for x in (table_cpu, rows_cpu, lengths_cpu, map_cpu, cu_cpu)
    ]
    for component in range(2):
        backing = torch.randn(sum(query_lengths), 2 * width, dtype=dtype)
        current_cpu = backing[:, :width]
        current = backing.cuda()[:, :width]
        if not latent:
            current = current.view(-1, 8, width // 8)
        destination = torch.full(
            (slots, 1, width), -17, dtype=dtype, device="cuda"
        )
        expected = torch.full((slots, 1, width), -17, dtype=dtype)
        for request, (history, query) in enumerate(zip(history_lengths, query_lengths)):
            row = rows_cpu[request]
            old_slots = table_cpu[row, :history].long()
            expected[old_slots] = hosts[component][map_cpu[old_slots].long()]
            new_slots = table_cpu[row, history : history + query].long()
            expected[new_slots] = current_cpu[
                cu_cpu[request] : cu_cpu[request + 1]
            ].unsqueeze(1)
        if prefetch:
            # A remapped shared prefix must be visible to the transfer stream;
            # the current CPU chunk is poisoned, so it cannot be read early.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                gather_prefill_history(
                    pointers, destination, table, rows, lengths, cu_query, slot_map,
                    component=component,
                )
            torch.cuda.current_stream().wait_stream(stream)
            new_slots = torch.cat([
                table_cpu[rows_cpu[i], h : h + q]
                for i, (h, q) in enumerate(zip(history_lengths, query_lengths))
            ]).cuda()
            scatter_prefill_current(current, destination, new_slots)
        else:
            gather_prefill_rows(
                pointers, current, destination, table, rows, lengths, cu_query, slot_map,
                capacity=capacity, component=component,
            )
        torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
