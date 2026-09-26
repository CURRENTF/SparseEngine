"""Index partition and physical-slot attention contracts for RetroInfer."""

import pytest
import torch

from benchmark.retroinfer.reference import WaveIndexReference, wave_attention_reference
from sparseengine.engine.cache_manager.methods.retroinfer_index import (
    RetroInferIndex,
    append_retroinfer_index,
)


def _small_index(start: int, length: int) -> RetroInferIndex:
    sizes = torch.tensor([[length]], dtype=torch.int32)
    return RetroInferIndex(
        centroids=torch.zeros(1, 1, 2),
        value_sums=torch.zeros(1, 1, 2),
        sizes=sizes,
        offsets=torch.tensor([[0, length]], dtype=torch.int32),
        sorted_positions=torch.arange(start, start + length, dtype=torch.int32)[None],
        indexed_start=start,
        indexed_end=start + length,
        retrieval_clusters=1,
        estimation_clusters=0,
    )


def test_retroinfer_index_append_preserves_partition_and_rejects_a_gap():
    combined = append_retroinfer_index(_small_index(4, 7), _small_index(11, 5))
    assert combined.indexed_start == 4 and combined.indexed_end == 16
    assert combined.offsets.tolist() == [[0, 7, 12]]
    assert combined.sorted_positions.tolist() == [list(range(4, 16))]
    with pytest.raises(ValueError, match="adjacent"):
        append_retroinfer_index(combined, _small_index(17, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_retroinfer_paged_decode_matches_oracle_with_shuffled_slots(dtype, dim):
    from sparseengine.kernels.triton.retroinfer_attention import retroinfer_paged_decode

    torch.manual_seed(73)
    device = torch.device("cuda")
    kv_heads, group = 2, 2
    sink, indexed, recent = 2, 11, 3
    length = sink + indexed + recent
    keys = torch.randn(kv_heads, length, dim, device=device, dtype=dtype) * 0.2
    values = torch.randn_like(keys)
    queries = torch.randn(kv_heads * group, dim, device=device, dtype=dtype)
    assignment = torch.tensor(
        [[0, 1, 0, 2, 1, 2, 0, 1, 2, 0, 1],
         [2, 0, 1, 1, 2, 0, 1, 2, 0, 1, 2]],
        device=device,
    )
    clusters = 4  # Cluster 3 is empty and must not contribute.
    sizes = torch.stack([(assignment == cluster).sum(dim=1) for cluster in range(clusters)], dim=1).to(torch.int32)
    centers = torch.zeros(kv_heads, clusters, dim, device=device, dtype=dtype)
    value_sums = torch.zeros(kv_heads, clusters, dim, device=device, dtype=torch.float32)
    for head in range(kv_heads):
        for cluster in range(clusters - 1):
            members = assignment[head] == cluster
            centers[head, cluster] = keys[head, sink:sink + indexed][members].float().mean(dim=0)
            value_sums[head, cluster] = values[head, sink:sink + indexed][members].float().sum(dim=0)
    oracle_index = WaveIndexReference(centers, value_sums, sizes, assignment)
    steady_keys = torch.cat((keys[:, :sink], keys[:, sink + indexed:]), dim=1)
    steady_values = torch.cat((values[:, :sink], values[:, sink + indexed:]), dim=1)
    expected, ranked = wave_attention_reference(
        queries, keys[:, sink:sink + indexed], values[:, sink:sink + indexed],
        steady_keys, steady_values, oracle_index,
        retrieval_clusters=2, estimation_clusters=2,
    )
    slot_table = torch.randperm(length, device=device).to(torch.int32)
    key_cache = torch.empty(length, kv_heads, dim, device=device, dtype=dtype)
    value_cache = torch.empty_like(key_cache)
    key_cache[slot_table.long()] = keys.transpose(0, 1)
    value_cache[slot_table.long()] = values.transpose(0, 1)
    sorted_positions = torch.argsort(assignment, dim=1, stable=True).to(torch.int32) + sink
    offsets = torch.nn.functional.pad(torch.cumsum(sizes, dim=1, dtype=torch.int32), (1, 0))
    actual = torch.empty_like(queries)
    retroinfer_paged_decode(
        queries.contiguous(), key_cache, value_cache, slot_table,
        centers.contiguous(), value_sums.contiguous(), sizes.contiguous(),
        offsets.contiguous(), sorted_positions.contiguous(),
        ranked[:, :4].to(torch.int32).contiguous(),
        sink_end=sink, recent_start=sink + indexed, recent_end=length,
        retrieval_clusters=2, estimation_clusters=2, output=actual,
    )
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.007
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_retroinfer_gpu_index_partitions_every_indexed_token():
    from sparseengine.engine.cache_manager.methods.retroinfer_index import build_retroinfer_segment

    torch.manual_seed(79)
    keys = torch.randn(2, 37, 64, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    index = build_retroinfer_segment(
        keys, values, start_position=4, avg_cluster_size=4,
        num_segments=2, retrieval_ratio=0.2, estimation_ratio=0.3,
        iterations=4,
    )
    for head in range(2):
        assert sorted(index.sorted_positions[head].cpu().tolist()) == list(range(4, 41))
        assert int(index.offsets[head, -1]) == 37
        for cluster in range(index.sizes.shape[1]):
            begin = int(index.offsets[head, cluster])
            end = int(index.offsets[head, cluster + 1])
            if begin == end:
                continue
            positions = index.sorted_positions[head, begin:end].long() - 4
            torch.testing.assert_close(
                index.centroids[head, cluster].float(),
                keys[head, positions].float().mean(dim=0),
                atol=0.003, rtol=0.003,
            )
            torch.testing.assert_close(
                index.value_sums[head, cluster],
                values[head, positions].float().sum(dim=0),
                atol=0.003, rtol=0.003,
            )
