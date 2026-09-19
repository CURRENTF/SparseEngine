import pytest
import torch

from sparseengine.kernels.triton.sparse_decode_graph_metadata import (
    publish_uniform_sparse_decode_graph_slots,
)


def test_publish_uniform_sparse_decode_graph_slots_shares_rows_not_physical_slots():
    free_slots = torch.stack(
        (
            torch.arange(100, 132, dtype=torch.int32),
            torch.arange(200, 232, dtype=torch.int32),
        )
    )
    table = torch.zeros((2, 4, 8), dtype=torch.int32)
    layer_slots = torch.empty((4, 3), dtype=torch.int32)
    public_slots = torch.empty(3, dtype=torch.int32)
    context_lens = torch.tensor([4, 6, 4], dtype=torch.int32)
    request_indices = torch.tensor([2, 1, 2], dtype=torch.int32)
    free_starts = torch.tensor([5, 0, 11, 0], dtype=torch.int32)
    active_count = torch.tensor([2], dtype=torch.int32)
    transformer_layers = torch.tensor([0, 2], dtype=torch.int32)

    publish_uniform_sparse_decode_graph_slots(
        free_slots,
        table,
        layer_slots,
        public_slots,
        context_lens,
        request_indices,
        free_starts,
        active_count,
        transformer_layers,
    )

    assert layer_slots[0].tolist() == [105, 106, -1]
    assert layer_slots[2].tolist() == [211, 212, -1]
    assert public_slots.tolist() == [105, 106, -1]
    assert table[0, 2, 3] == 105
    assert table[0, 1, 5] == 106
    assert table[1, 2, 3] == 211
    assert table[1, 1, 5] == 212


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_publish_uniform_sparse_decode_graph_slots_replays_updated_reservations():
    device = torch.device("cuda")
    free_slots = torch.arange(64, dtype=torch.int32, device=device).view(2, 32)
    table = torch.zeros((2, 4, 8), dtype=torch.int32, device=device)
    layer_slots = torch.empty((4, 3), dtype=torch.int32, device=device)
    public_slots = torch.empty(3, dtype=torch.int32, device=device)
    context_lens = torch.tensor([4, 6, 4], dtype=torch.int32, device=device)
    request_indices = torch.tensor([2, 1, 2], dtype=torch.int32, device=device)
    free_starts = torch.tensor([5, 0, 11, 0], dtype=torch.int32, device=device)
    active_count = torch.tensor([2], dtype=torch.int32, device=device)
    transformer_layers = torch.tensor([0, 2], dtype=torch.int32, device=device)
    args = (
        free_slots,
        table,
        layer_slots,
        public_slots,
        context_lens,
        request_indices,
        free_starts,
        active_count,
        transformer_layers,
    )
    publish_uniform_sparse_decode_graph_slots(*args)
    torch.cuda.synchronize()
    table.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        publish_uniform_sparse_decode_graph_slots(*args)

    graph.replay()
    torch.cuda.synchronize()
    expected = torch.zeros_like(table)
    expected[0, 2, 3] = 5
    expected[0, 1, 5] = 6
    expected[1, 2, 3] = 43
    expected[1, 1, 5] = 44
    torch.testing.assert_close(table, expected)

    table.zero_()
    free_starts[0] = 7
    free_starts[2] = 13
    context_lens[:2] = torch.tensor([5, 7], dtype=torch.int32, device=device)
    graph.replay()
    torch.cuda.synchronize()

    expected.zero_()
    expected[0, 2, 4] = 7
    expected[0, 1, 6] = 8
    expected[1, 2, 4] = 45
    expected[1, 1, 6] = 46
    torch.testing.assert_close(table, expected)
    assert public_slots.tolist() == [7, 8, -1]
