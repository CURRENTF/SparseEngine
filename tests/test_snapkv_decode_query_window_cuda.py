from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager.base import CacheManager
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.sequence import Sequence


def _manager(method: str, score_mode: str) -> SnapKVCacheManager:
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=1, num_key_value_heads=2,
            num_attention_heads=4, hidden_size=64, head_dim=16,
            dtype=torch.bfloat16,
        ),
        runtime_layout=RuntimeLayout.dense(1),
        attention_cache_layout="explicit_kv",
        max_model_len=8,
        max_num_batched_tokens=8,
        max_num_seqs_in_gpu=2,
        sparse_method=method,
        pyramid_layer_ratios=[1.0] if method == "pyramidkv" else None,
        prefill_schedule_policy="long_bs1full_short_batch",
        num_kvcache_slots=None,
        sink_keep_tokens=1,
        decode_keep_tokens=2,
        recent_keep_tokens=1,
        decode_reservation_tokens=2,
        decode_eviction_interval=2,
        observation_window_size=3,
        snapkv_decode_eviction=True,
        sparse_prefill_score_mode=score_mode,
    )
    parallel = SimpleNamespace(
        world_rank=0, world_size=1, attn_tp_rank=0, attn_tp_size=1,
        moe_ep_rank=0, moe_ep_size=1, attn_dp_rank=0, attn_dp_size=1,
    )
    with patch.object(CacheManager, "_get_available_slots_info", return_value=(1_000_000, 128)):
        return SnapKVCacheManager(config, parallel)


def _reference(keys: torch.Tensor, queries: torch.Tensor, *, mode: str) -> torch.Tensor:
    length, kv_heads, dim = keys.shape
    window, query_heads, _ = queries.shape
    groups = query_heads // kv_heads
    result = (torch.zeros(length, dtype=torch.float32) if mode == "probability"
              else torch.full((length,), -torch.inf))
    for head in range(query_heads):
        head_score = torch.zeros(length) if mode == "probability" else torch.full((length,), -torch.inf)
        for query_index in range(window):
            position = length - window + query_index
            candidate_positions = torch.arange(1, min(position + 1, length - 1))
            logits = torch.mv(
                keys[candidate_positions, head // groups].float(),
                queries[query_index, head].float(),
            )
            if mode == "probability":
                values = torch.softmax(logits / dim**0.5, dim=0) / window
                head_score[candidate_positions] += values
            else:
                head_score[candidate_positions] = torch.maximum(
                    head_score[candidate_positions], logits,
                )
        result[1:-1] = torch.maximum(result[1:-1], head_score[1:-1])
    return result


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("method", "score_mode"),
    [("snapkv", "logits"), ("snapkv", "probability"), ("pyramidkv", "probability")],
)
def test_graph_replay_query_window_scores_match_independent_reference(method, score_mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    manager = _manager(method, score_mode)
    assert manager.device.type == "cuda"
    seq = Sequence([1])
    slots = manager._allocate(0, seq.seq_id, 6)
    row = manager.seq_id_to_row[0][seq.seq_id]
    generator = torch.Generator(device="cpu").manual_seed(17)
    keys = torch.randn((6, 2, 16), generator=generator).to(torch.bfloat16)
    queries = torch.randn((3, 4, 16), generator=generator).to(torch.bfloat16)
    k_cache, _ = manager.get_layer_kv_cache(0)
    k_cache[slots.long()] = keys.to(manager.device)

    state = manager.get_layer_batch_states(0)
    state.req_indices = torch.tensor([row, row], dtype=torch.int32, device=manager.device)
    state.context_lens = torch.tensor([4, 4], dtype=torch.int32, device=manager.device)
    manager._decode_query_active_mask = torch.tensor(
        [True, False], dtype=torch.bool, device=manager.device,
    )
    static_query = torch.zeros((2, 4, 16), dtype=torch.bfloat16, device=manager.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        manager.record_decode_query(0, static_query)
    _, stamps = manager._decode_query_cache_layer(0)
    stamps[row].fill_(-1)

    for position, query in zip((3, 4, 5), queries):
        state.context_lens.fill_(position + 1)
        static_query[0].copy_(query.to(manager.device))
        static_query[1].fill_(100)
        graph.replay()
    torch.cuda.synchronize()
    cached_queries, _ = manager._decode_query_cache_layer(0)
    torch.testing.assert_close(
        cached_queries[row, torch.tensor([0, 1, 2], device=manager.device)].cpu(),
        queries,
    )
    actual = manager.decode_query_scores(0, seq, 6).cpu()
    expected = _reference(keys, queries, mode=score_mode)
    torch.testing.assert_close(actual[1:-1], expected[1:-1], atol=3e-2, rtol=3e-2)

    second = Sequence([2])
    manager._allocate(0, second.seq_id, 6)
    second_row = manager.seq_id_to_row[0][second.seq_id]
    assert second_row != row
    state.req_indices[1] = second_row
    manager._decode_query_active_mask[1] = True
    static_query[1].fill_(7)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        cached_queries[second_row, 5 % 3],
        torch.full((4, 16), 7, dtype=torch.bfloat16, device=manager.device),
    )

    manager.free_part_slots(0, seq, torch.tensor([0, 2, 4, 5], device=manager.device))
    assert bool((stamps[row] == -1).all())
