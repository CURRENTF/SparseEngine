"""Independent R-KV numerical and global-eviction lifecycle contracts."""
from types import SimpleNamespace

import pytest
import torch

from sparseengine.config import RuntimeLayout
from sparseengine.distributed.parallel_context import ParallelGroup
from sparseengine.engine.cache_manager.base import LayerBatchStates
from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
from sparseengine.engine.cache_manager.methods.rkv_scoring import rkv_head_scores
from sparseengine.engine.sparse_methods.base import SparseStepContext
from sparseengine.engine.sparse_methods.rkv import RKVRuntime
from tests.test_static_eviction_compaction import _page_table_manager


def reference_scores(keys, queries, window, kernel_size, alpha):
    # Deliberately scalar/head-wise: checks GQA reduction order, column-wise
    # redundancy, below-threshold mass, and local pooling independently.
    batch, heads, length, dim = keys.shape
    groups = queries.shape[1] // heads
    outputs = []
    for b in range(batch):
        head_outputs = []
        for h in range(heads):
            k = keys[b, h]
            qs = queries[b, h * groups:(h + 1) * groups]
            logits = torch.stack([q @ k.T / dim**0.5 for q in qs]).amax(0)
            importance = logits[:, :-window].softmax(-1, dtype=torch.float32).mean(0).to(k.dtype)
            radius = kernel_size // 2
            importance = torch.stack([importance[max(0, t-radius):min(length-window, t+radius+1)].max()
                                      for t in range(length-window)])
            norm = (k.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(k.dtype)
                    if k.dtype == torch.float16 else k.norm(dim=-1, keepdim=True) + 1e-8)
            normalized = k / norm
            sim = normalized @ normalized.T
            sim.fill_diagonal_(0)
            for row in range(length):
                matches = torch.where(sim[row] > 0.5)[0]
                sim[row, int(matches[-1]) if len(matches) else 0] = 0
            redundancy = sim.mean(0).softmax(0)[:-window]
            head_outputs.append(alpha * importance - (1-alpha) * redundancy)
        outputs.append(torch.stack(head_outputs))
    return torch.stack(outputs)


@pytest.mark.parametrize("gqa", [1, 3])
@pytest.mark.parametrize("tiled", [False, True])
def test_scores_match_independent_oracle_with_correlated_keys(gqa, tiled):
    torch.manual_seed(41)
    k = torch.randn(2, 2, 19, 4)
    k[..., 8:12, :] = k[..., 2:6, :] + 0.01
    q = torch.randn(2, 2*gqa, 3, 4)
    cap = 12000 if tiled else 1024**2
    actual = rkv_head_scores(k, q, window=3, kernel_size=7, alpha=0.1, workspace_bytes=cap)
    expected = reference_scores(k, q, 3, 7, 0.1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    assert actual.mean(1).topk(5).indices.tolist() == expected.mean(1).topk(5).indices.tolist()


def test_workspace_failure_is_explicit():
    with pytest.raises(RuntimeError, match="workspace"):
        rkv_head_scores(torch.ones(1, 1, 8, 4), torch.ones(1, 1, 2, 4),
                        window=2, kernel_size=3, alpha=0.1, workspace_bytes=1)


def runtime_fixture(lengths=(8, 10), decoded=2):
    manager, seqs = _page_table_manager([list(lengths), list(lengths)], manager_cls=RKVCacheManager)
    manager._rkv_observation_tokens = 2
    manager._rkv_query_cache = [torch.zeros(len(seqs), 2, 1, 4) for _ in range(2)]
    manager._rkv_query_positions = [torch.tensor([[l-2, l-1] for l in lengths], dtype=torch.int32) for _ in range(2)]
    manager.parallel_context = SimpleNamespace(attn_tp=ParallelGroup(None, (0,), 0, 1))
    for seq in seqs:
        seq.num_prompt_tokens = seq.num_tokens - decoded
    runtime = object.__new__(RKVRuntime)
    runtime.cache_manager = manager
    runtime.config = SimpleNamespace(rkv_compression_interval=2, rkv_observation_tokens=2)
    runtime.num_sink, runtime.decode_keep_tokens, runtime.num_recent = 1, 2, 1
    runtime.debug_dynamic_selection = {}
    # Layer 0 favors token 0, layer 1 strongly favors tokens 2/3. A layer-local
    # choice or sink retention would disagree with the global oracle.
    def score(layer, requests, length):
        result = torch.zeros(len(requests), length-2)
        result[:, 0 if layer == 0 else 2] = 4 if layer == 0 else 9
        result[:, 3] = 6
        return result
    manager.rkv_joint_scores = score
    return runtime, manager, seqs


def test_global_selection_compacts_all_layers_and_returns_actual_slots():
    runtime, manager, seqs = runtime_fixture()
    originals = [x.clone() for x in manager.buffer_req_to_token_slots]
    runtime.finish_step(SparseStepContext(seqs, False, None))
    for layer in range(2):
        for row, length in enumerate((8, 10)):
            expected = originals[layer][row, [2, 3, length-2, length-1]]
            torch.testing.assert_close(manager.buffer_req_to_token_slots[layer][row, :4], expected)
            assert manager.row_seq_lens[layer][row] == 4
        assert manager._num_free_slots[layer] == 2 + 4 + 6
        assert (manager._rkv_query_positions[layer] == -1).all()
    assert runtime.debug_dynamic_selection["rkv_compactions"] == 2


def test_long_prompt_waits_for_decode_buffer_boundary():
    runtime, manager, seqs = runtime_fixture(decoded=1)
    runtime.finish_step(SparseStepContext(seqs, False, None))
    assert manager.row_seq_lens[0].tolist() == [8, 10]
    for seq in seqs:
        seq.num_prompt_tokens -= 1
    runtime.finish_step(SparseStepContext(seqs, False, None))
    assert manager.row_seq_lens[0].tolist() == [4, 4]


def test_tp_sum_changes_global_decision_before_mutation():
    runtime, manager, seqs = runtime_fixture()
    reductions = []
    def reduce(scores, op=torch.distributed.ReduceOp.SUM):
        reductions.append(op)
        if op == torch.distributed.ReduceOp.SUM:
            scores[:, 1] += 100
        return scores
    manager.parallel_context = SimpleNamespace(attn_tp=SimpleNamespace(all_reduce=reduce))
    runtime.finish_step(SparseStepContext(seqs, False, None))
    assert manager.buffer_req_to_token_slots[0][0, :4].tolist() == [1, 3, 6, 7]
    assert torch.distributed.ReduceOp.SUM in reductions


def test_nonfinite_later_group_never_partially_evicts_earlier_requests():
    runtime, manager, seqs = runtime_fixture()
    old = manager.rkv_joint_scores
    def corrupt(layer, requests, length):
        result = old(layer, requests, length)
        if length == 10:
            result.fill_(float("nan"))
        return result
    manager.rkv_joint_scores = corrupt
    with pytest.raises(RuntimeError, match="non-finite"):
        runtime.finish_step(SparseStepContext(seqs, False, None))
    assert manager.row_seq_lens[0].tolist() == [8, 10]
    assert manager._num_free_slots == [2, 2]


def test_missing_window_after_eviction_fails_without_mutation():
    runtime, manager, seqs = runtime_fixture()
    seqs[0].num_tokens += 2
    manager._rkv_query_positions[0].fill_(-1)
    with pytest.raises(RuntimeError, match="missing decode observations"):
        runtime.finish_step(SparseStepContext(seqs, False, None))
    assert manager.row_seq_lens[0].tolist() == [8, 10]


def test_layer_domain_mismatch_fails_before_scoring():
    runtime, manager, seqs = runtime_fixture()
    manager.row_seq_lens[1][0] -= 1
    with pytest.raises(RuntimeError, match="matching token domains"):
        runtime.finish_step(SparseStepContext(seqs, False, None))
    assert manager._num_free_slots == [2, 2]


def make_ring(device="cpu", dtype=torch.float32):
    manager = object.__new__(RKVCacheManager)
    manager.runtime_layout = RuntimeLayout.dense(1)
    manager.device = torch.device(device)
    manager.config = SimpleNamespace(rkv_kernel_size=3, rkv_alpha=0.1, rkv_score_chunk_mb=1)
    manager._rkv_observation_tokens = 3
    manager._rkv_query_cache = [torch.zeros(2, 3, 4, 8, device=device, dtype=dtype)]
    manager._rkv_query_positions = [torch.full((2, 3), -1, device=device, dtype=torch.int32)]
    manager.layer_batch_states = [LayerBatchStates(req_indices=torch.tensor([0, 1], device=device),
                                                 context_lens=torch.tensor([11, 1], device=device))]
    manager.seq_id_to_row = [{0: 0}]
    manager.buffer_req_to_token_slots = [torch.randperm(16, device=device).int()[None]]
    manager.kv_cache = [(torch.randn(16, 2, 8, device=device, dtype=dtype), torch.zeros(16, 2, 8, device=device, dtype=dtype))]
    return manager


def test_prefill_invalidates_queries_then_decode_refills_ring():
    manager = make_ring()
    manager._rkv_query_positions[0][0].fill_(123)
    view = SimpleNamespace(meta=SimpleNamespace(req_indices=torch.tensor([0])))
    manager.record_prefill_query(0, None, view, b_start_loc=None, chunk_lens=None)
    assert manager._rkv_query_positions[0][0].tolist() == [-1]*3
    q = torch.randn(2, 4, 8)
    for length in (11, 12, 13):
        manager.layer_batch_states[0].context_lens[0] = length
        manager.record_decode_query(0, q)
    assert manager.rkv_observation_ready(0, [SimpleNamespace(seq_id=0)], 13).tolist() == [True]
    slots = manager.buffer_req_to_token_slots[0][0, :13].long()
    keys = manager.kv_cache[0][0][slots].transpose(0, 1)[None]
    queries = q[0].unsqueeze(1).expand(4, 3, 8)[None]
    expected = reference_scores(keys, queries, 3, 3, 0.1).mean(1)
    torch.testing.assert_close(manager.rkv_joint_scores(0, [SimpleNamespace(seq_id=0)], 13), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_query_ring_graph_replay_growing_lengths_and_row_reuse():
    manager = make_ring("cuda")
    q = torch.randn(2, 4, 8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        manager.record_decode_query(0, q)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        manager.record_decode_query(0, q)
    pointer = manager._rkv_query_cache[0].data_ptr()
    for length in (11, 12, 13):
        manager.layer_batch_states[0].context_lens[0] = length
        q.fill_(length)
        graph.replay()
    assert manager.rkv_observation_ready(0, [SimpleNamespace(seq_id=0)], 13).item()
    manager._clear_rkv_query_cache_row(0, 0)
    assert not manager.rkv_observation_ready(0, [SimpleNamespace(seq_id=0)], 13).item()
    manager.layer_batch_states[0].context_lens[0] = 2
    q.fill_(99)
    graph.replay()
    assert manager._rkv_query_cache[0].data_ptr() == pointer
    assert manager._rkv_query_positions[0][0].tolist() == [-1, 1, -1]
    torch.testing.assert_close(manager._rkv_query_cache[0][0, 1], q[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_scores_match_independent_oracle(dtype):
    torch.manual_seed(12)
    keys = torch.randn(1, 2, 37, 16, device="cuda", dtype=dtype)
    keys[:, :, 9:13] = keys[:, :, 2:6]
    queries = torch.randn(1, 6, 4, 16, device="cuda", dtype=dtype)
    actual = rkv_head_scores(keys, queries, window=4, kernel_size=7, alpha=0.1, workspace_bytes=40000)
    expected = reference_scores(keys, queries, 4, 7, 0.1)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0.02)
    assert actual.mean(1).topk(8).indices.tolist() == expected.mean(1).topk(8).indices.tolist()


def test_legacy_approximation_config_is_rejected():
    from sparseengine.configs.groups import SparseMethodConfig
    from sparseengine.configs.sparse import _normalize_rkv
    config = SparseMethodConfig(sparse_method="rkv", rkv_redundancy_window=64)
    with pytest.raises(ValueError, match="full resident domain"):
        _normalize_rkv(config)
    config = SparseMethodConfig(sparse_method="rkv", rkv_observation_tokens=9, rkv_compression_interval=8)
    with pytest.raises(ValueError, match="rkv_compression_interval"):
        _normalize_rkv(config)


def test_long_prompt_admission_reserves_until_buffer_boundary():
    from collections import deque
    runtime, manager, seqs = runtime_fixture(lengths=(100,))
    manager.config = SimpleNamespace(sparse_method="rkv", sink_keep_tokens=0,
                                    recent_keep_tokens=2, decode_keep_tokens=2,
                                    rkv_compression_interval=8)
    manager.free_rows = [deque([1]), deque([1])]
    required, _, _, _ = manager.chain_capacity_deficits(
        suffix_tokens=0, generation_tokens=10, existing_slots_by_layer=(100, 100),
        needs_resident_row=False,
    )
    assert required == (8, 8)  # First eviction cannot occur on decode step 1.


def test_decode_workspace_does_not_consume_small_prefill_profiling_pool(monkeypatch):
    from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
    manager = object.__new__(RKVCacheManager)
    manager.config = SimpleNamespace(startup_cache_phase="profiling", rkv_score_chunk_mb=64)
    manager._rkv_query_cache_bytes = lambda: 1024**2
    monkeypatch.setattr(SnapKVCacheManager, "_get_available_slots_info", lambda self: (40 * 1024**2, 128))
    assert manager._get_available_slots_info() == (39 * 1024**2, 128)
    manager.config.startup_cache_phase = "serving"
    manager.config.max_model_len = 128
    manager.config.rkv_observation_tokens = 8
    manager.num_kv_heads, manager.head_dim, manager.tp_size = 2, 16, 1
    manager.hf_config = SimpleNamespace(dtype=torch.float32, num_attention_heads=4)
    with pytest.raises(RuntimeError, match="scoring workspace"):
        manager._get_available_slots_info()


def test_long_prompt_workspace_rejected_before_cache_allocation(monkeypatch):
    """A small retention budget must not hide an unscorable first long prompt."""
    from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
    manager = object.__new__(RKVCacheManager)
    manager.config = SimpleNamespace(startup_cache_phase="serving", max_model_len=4096,
                                    rkv_observation_tokens=8, rkv_score_chunk_mb=1)
    manager.num_kv_heads, manager.head_dim, manager.tp_size = 2, 32, 1
    manager.hf_config = SimpleNamespace(dtype=torch.float32, num_attention_heads=4)
    monkeypatch.setattr(SnapKVCacheManager, "_get_available_slots_info",
                        lambda self: pytest.fail("must reject before allocating the cache pool"))
    with pytest.raises(RuntimeError, match="length=4096.*retention budget"):
        manager._get_available_slots_info()


def test_workspace_plan_acceptance_executes_against_independent_oracle():
    """The startup planner and execution must agree even at a tight tile budget."""
    from sparseengine.engine.cache_manager.methods.rkv_scoring import rkv_score_tiles
    torch.manual_seed(7)
    keys = torch.randn(1, 2, 23, 8)
    queries = torch.randn(1, 4, 3, 8)
    cap = 16384
    units, rows = rkv_score_tiles(batch=1, heads=2, length=23, dim=8, groups=2,
                                 window=3, element_size=4, workspace_bytes=cap)
    assert 1 <= units <= 2 and 1 <= rows <= 23
    actual = rkv_head_scores(keys, queries, window=3, kernel_size=3,
                             alpha=0.1, workspace_bytes=cap)
    torch.testing.assert_close(actual, reference_scores(keys, queries, 3, 3, 0.1))


def test_recompute_discards_observations_and_reserves_uncompressed_growth():
    runtime, manager, seqs = runtime_fixture()
    for seq in seqs:
        seq.recompute_replay_cursor = 0
    runtime.finish_step(SparseStepContext(seqs, False, None))
    assert manager.row_seq_lens[0].tolist() == [8, 10]
    assert (manager._rkv_query_positions[0] == -1).all()
    assert manager.decode_window_costs(seqs[0], 32) == {"layer_0": 32, "layer_1": 32}
