"""Independent numerical contracts for a future RetroInfer kernel/provider."""

import math

import torch

from benchmark.retroinfer.reference import (
    WaveIndexReference,
    build_wave_index_reference,
    wave_attention_reference,
)


def test_all_clusters_retrieved_equals_dense_attention():
    torch.manual_seed(7)
    keys = torch.randn(2, 24, 4, dtype=torch.float64)
    values = torch.randn_like(keys)
    steady_keys = torch.randn(2, 3, 4, dtype=torch.float64)
    steady_values = torch.randn_like(steady_keys)
    queries = torch.randn(4, 4, dtype=torch.float64)
    index = build_wave_index_reference(
        keys, values, num_clusters=6, num_segments=2, iterations=4
    )
    output, _ = wave_attention_reference(
        queries, keys, values, steady_keys, steady_values, index,
        retrieval_clusters=6, estimation_clusters=0,
    )
    expected = []
    for query_head, query in enumerate(queries):
        kv_head = query_head // 2
        all_keys = torch.cat((steady_keys[kv_head], keys[kv_head]))
        all_values = torch.cat((steady_values[kv_head], values[kv_head]))
        weights = torch.softmax((all_keys @ query) / math.sqrt(4), dim=0)
        expected.append(weights @ all_values)
    torch.testing.assert_close(output, torch.stack(expected), atol=1e-12, rtol=1e-12)


def test_estimation_uses_cluster_size_and_value_sum_once():
    keys = torch.tensor([[[2.0], [1.0], [-1.0], [-2.0]]])
    values = torch.tensor([[[2.0], [4.0], [5.0], [7.0]]])
    index = WaveIndexReference(
        centroids=torch.tensor([[[1.5], [-1.5]]], dtype=torch.float64),
        value_sums=torch.tensor([[[6.0], [12.0]]], dtype=torch.float64),
        sizes=torch.tensor([[2, 2]]),
        assignments=torch.tensor([[0, 0, 1, 1]]),
    )
    output, ranked = wave_attention_reference(
        torch.tensor([[1.0]]), keys, values,
        torch.tensor([[[0.0]]]), torch.tensor([[[3.0]]]), index,
        retrieval_clusters=1, estimation_clusters=1,
    )
    assert ranked.tolist() == [[0, 1]]
    denominator = 1 + math.exp(2) + math.exp(1) + 2 * math.exp(-1.5)
    numerator = 3 + 2 * math.exp(2) + 4 * math.exp(1) + 12 * math.exp(-1.5)
    torch.testing.assert_close(output, torch.tensor([[numerator / denominator]]))


def test_index_preserves_token_partition_and_cluster_means():
    torch.manual_seed(11)
    keys = torch.randn(2, 17, 8, dtype=torch.float64)
    values = torch.randn_like(keys)
    index = build_wave_index_reference(
        keys, values, num_clusters=4, num_segments=2, iterations=3
    )
    assert torch.equal(index.sizes.sum(dim=1), torch.full((2,), 17))
    torch.testing.assert_close(index.value_sums.sum(dim=1), values.sum(dim=1))
    for head in range(2):
        for cluster in range(4):
            members = index.assignments[head] == cluster
            if members.any():
                torch.testing.assert_close(index.centroids[head, cluster], keys[head, members].mean(dim=0))
                torch.testing.assert_close(index.value_sums[head, cluster], values[head, members].sum(dim=0))
