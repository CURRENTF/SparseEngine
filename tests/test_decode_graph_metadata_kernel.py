import pytest
import torch

from sparseengine.kernels.triton.decode_graph_metadata import (
    publish_decode_graph_slots,
)


def test_publish_decode_graph_slots_ignores_padding_rows() -> None:
    table = torch.zeros((4, 8), dtype=torch.int32)
    request_indices = torch.tensor([2, 1, 3, 3], dtype=torch.int32)
    context_lens = torch.tensor([4, 6, 4, 4], dtype=torch.int32)
    write_slots = torch.tensor([101, 202, -1, -1], dtype=torch.int32)
    active_mask = torch.tensor([True, True, False, False])

    publish_decode_graph_slots(
        table,
        request_indices,
        context_lens,
        write_slots,
        active_mask,
    )

    expected = torch.zeros_like(table)
    expected[2, 3] = 101
    expected[1, 5] = 202
    torch.testing.assert_close(table, expected)

    write_slots[:2] = torch.tensor([303, 404], dtype=torch.int32)
    context_lens[:2] = torch.tensor([5, 7], dtype=torch.int32)
    publish_decode_graph_slots(
        table,
        request_indices,
        context_lens,
        write_slots,
        active_mask,
    )
    expected[2, 4] = 303
    expected[1, 6] = 404
    torch.testing.assert_close(table, expected)


def test_publish_decode_graph_slots_without_mask_publishes_every_row() -> None:
    table = torch.zeros((3, 6), dtype=torch.int32)
    publish_decode_graph_slots(
        table,
        torch.tensor([2, 0], dtype=torch.int32),
        torch.tensor([4, 5], dtype=torch.int32),
        torch.tensor([101, 202], dtype=torch.int32),
    )

    expected = torch.zeros_like(table)
    expected[2, 3] = 101
    expected[0, 4] = 202
    torch.testing.assert_close(table, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_publish_decode_graph_slots_replays_with_updated_inputs() -> None:
    device = torch.device("cuda")
    table = torch.zeros((4, 8), dtype=torch.int32, device=device)
    request_indices = torch.tensor([2, 1, 3, 3], dtype=torch.int32, device=device)
    context_lens = torch.tensor([4, 6, 4, 4], dtype=torch.int32, device=device)
    write_slots = torch.tensor([101, 202, -1, -1], dtype=torch.int32, device=device)
    active_mask = torch.tensor([True, True, False, False], device=device)

    direct_table = torch.zeros_like(table)
    publish_decode_graph_slots(
        direct_table,
        request_indices[:2],
        context_lens[:2],
        write_slots[:2],
    )
    direct_expected = torch.zeros_like(table)
    direct_expected[2, 3] = 101
    direct_expected[1, 5] = 202
    torch.testing.assert_close(direct_table, direct_expected)

    publish_decode_graph_slots(
        table,
        request_indices,
        context_lens,
        write_slots,
        active_mask,
    )
    torch.cuda.synchronize()
    table.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        publish_decode_graph_slots(
            table,
            request_indices,
            context_lens,
            write_slots,
            active_mask,
        )

    table.zero_()
    context_lens.copy_(torch.tensor([5, 7, 4, 4], dtype=torch.int32, device=device))
    write_slots.copy_(torch.tensor([303, 404, -1, -1], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.zeros_like(table)
    expected[2, 4] = 303
    expected[1, 6] = 404
    torch.testing.assert_close(table, expected)
