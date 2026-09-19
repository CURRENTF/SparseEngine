from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sparseengine.platforms as platforms
from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager.base import CacheManager, ExplicitKVPayload
from sparseengine.engine.cache_manager.methods.kvzip import KVzipCacheManager
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_methods.base import SparseStepContext
from sparseengine.engine.sparse_methods.kvzip import KVzipRuntime
from sparseengine.platforms.cpu import CpuPlatform


def _manager(*, profiling=False, **overrides):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hidden_layers=2, num_key_value_heads=1,
                                  num_attention_heads=2, hidden_size=32,
                                  head_dim=16, dtype=torch.float32),
        runtime_layout=RuntimeLayout.dense(2), attention_cache_layout="explicit_kv",
        max_model_len=128, max_num_batched_tokens=32, max_num_seqs_in_gpu=3,
        engine_prefill_chunk_size=32, sparse_method="kvzip", pyramid_layer_ratios=None,
        prefill_schedule_policy="all_chunked", num_kvcache_slots=None,
        sink_keep_tokens=0, decode_keep_tokens=4, recent_keep_tokens=0,
        kvzip_token_budget=4, kvzip_score_chunk_size=5, kvzip_prev_postfix_size=2,
        sparse_attn_score_dtype="float32", obs_layer_ids=[], full_attention_layers=[],
        parallel_topology=SimpleNamespace(attn_tp_size=1),
        max_num_seqs_in_batch=1, max_decoding_seqs=1,
        decode_graph_startup_capture=False,
    )
    vars(config).update(overrides)
    context = SimpleNamespace(
        world_rank=0, world_size=1, attn_tp_rank=0, attn_tp_size=1,
        moe_ep_rank=0, moe_ep_size=1, attn_dp_rank=0, attn_dp_size=1,
        attn_tp=SimpleNamespace(all_reduce=lambda tensor, **kwargs: tensor),
    )
    with patch.object(platforms, "_current_platform", CpuPlatform()):
        if profiling:
            from sparseengine.engine.startup.capacity import profiling_kv_budget_bytes, profiling_kv_slots

            config.startup_cache_phase = "profiling"
            manager = KVzipCacheManager(config, context, allocation_budget_bytes=
                profiling_kv_budget_bytes(config, profiling_kv_slots(config)))
        else:
            with patch.object(CacheManager, "_get_available_slots_info", return_value=(1_000_000, 128)):
                manager = KVzipCacheManager(config, context)
    manager.set_reconstruction_prompt([90, 91])
    return manager


def _seed(manager, seq):
    for layer in manager.kv_transformer_layer_indices():
        manager._allocate(layer, seq.seq_id, seq.num_prompt_tokens)


@pytest.mark.parametrize("budget,graph", [(4, False), (4, True), (128, False)])
def test_profiling_capacity_admits_warmup_without_consuming_replay_reserve(monkeypatch, budget, graph):
    """Small profiling pools used to have zero usable slots after reserving replay KV."""
    from sparseengine.engine.runtime_state import RuntimeState
    from sparseengine.engine.startup import capacity

    graph_batch = 7
    monkeypatch.setattr(capacity, "build_decode_cuda_graph_startup_plan",
                        lambda config: [(graph_batch, 0)])
    manager = _manager(profiling=True, max_num_batched_tokens=12,
                       max_num_seqs_in_gpu=graph_batch, kvzip_token_budget=budget,
                       decode_graph_startup_capture=graph)
    config = manager.config
    runtime = RuntimeState(config, manager)
    lengths = capacity.profiling_prefill_chunk_lengths(config)
    assert runtime.startup_batch_fits(lengths, max_tokens=2)
    if graph:
        assert runtime.startup_batch_fits((1,) * graph_batch, max_tokens=2)
        assert manager.num_free_slots >= graph_batch * 3

    # The temporary cache needs both its payload and its allocator metadata;
    # doubling KV bytes is not a substitute for this physical allocation.
    tensors = (manager.kv_cache, manager.free_slots_stack_tensor,
               manager.buffer_req_to_token_slots_tensor)
    assert sum(t.numel() * t.element_size() for t in tensors) == manager.allocation_budget_bytes
    seq = Sequence(list(range(lengths[0])))
    _seed(manager, seq)
    before = manager._num_free_slots.copy()
    if seq.num_prompt_tokens > budget:
        replay = len(manager.reconstruction_prompt_ids) + config.kvzip_score_chunk_size
        with manager.reconstruction_chunk(seq, replay, torch.zeros(seq.num_prompt_tokens)):
            for layer in manager.kv_transformer_layer_indices():
                manager._allocate(layer, seq.seq_id, replay)
                manager._reconstruction.visited_layers.add(layer)
        assert manager._num_free_slots == before
    manager.free_seq(seq.seq_id)
    assert manager._num_free_slots == manager.layer_num_slots


def test_reconstruction_failure_releases_only_temporary_slots():
    """A failed layer must neither leak replay slots nor release another request."""
    manager = _manager()
    seq, other = Sequence(list(range(10))), Sequence(list(range(7)))
    _seed(manager, seq)
    _seed(manager, other)
    original = [x.clone() for x in manager.buffer_req_to_token_slots]
    counts = manager._num_free_slots.copy()
    with pytest.raises(RuntimeError, match="injected"):
        with manager.reconstruction_chunk(seq, 6, torch.zeros(10)):
            # Partial allocation failure: only the first layer acquired scratch.
            manager._allocate(0, seq.seq_id, 6)
            raise RuntimeError("injected")
    assert manager._reconstruction is None
    assert manager._num_free_slots == counts
    for before, after in zip(original, manager.buffer_req_to_token_slots):
        torch.testing.assert_close(before, after)
    manager.free_seq(seq.seq_id)
    manager.free_seq(other.seq_id)
    assert manager._num_free_slots == manager.layer_num_slots


def test_reconstruction_requires_every_layer_and_preserves_headroom():
    """Silent missing scores must fail; scratch remains unavailable to admission."""
    manager = _manager()
    seq = Sequence(list(range(10)))
    _seed(manager, seq)
    with pytest.raises(RuntimeError, match="every KV layer"):
        with manager.reconstruction_chunk(seq, 5, torch.zeros(10)):
            for layer in manager.kv_transformer_layer_indices():
                manager._allocate(layer, seq.seq_id, 5)
    assert manager.num_free_slots + manager.reconstruction_slot_reserve == min(manager._num_free_slots)
    for layer, free in enumerate(manager._num_free_slots):
        assert manager.decode_window_budgets()[f"layer_{layer}"] == free - manager.reconstruction_slot_reserve
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        manager.prompt_admission_costs(Sequence(list(range(124))))


def test_tp_chunk_only_ipc_retains_identical_reconstruction_source():
    """Follower IPC drops earlier chunks; reconstruction must retain them itself."""
    import pickle
    from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager

    leader, follower = _manager(), _manager()
    seq = Sequence(list(range(11)))
    with patch.object(SnapKVCacheManager, "prepare_step", return_value=None):
        for start, count in ((0, 5), (5, 5), (10, 1)):
            seq.num_prefilled_tokens, seq.current_chunk_size = start, count
            remote = pickle.loads(pickle.dumps(seq))
            assert len(remote.token_ids) == count
            leader.prepare_step([seq], True)
            follower.prepare_step([remote], True)
    assert leader.reconstruction_source(seq) == follower.reconstruction_source(remote) == list(range(11))
    for manager in (leader, follower):
        _seed(manager, seq)
        manager.free_seq(seq.seq_id)
        assert not manager._reconstruction_source


def test_runtime_compacts_only_completed_prompts_with_deterministic_global_selection():
    """Mixed prefill batches must not compact incomplete rows or retain replay KV."""
    manager = _manager()
    runtime = KVzipRuntime(manager.config, manager)
    completed, partial, short = (Sequence(list(range(n))) for n in (11, 8, 3))
    for seq in (completed, partial, short):
        _seed(manager, seq)
        seq.current_chunk_size = seq.num_prompt_tokens
    partial.current_chunk_size = 4
    original_slots = [table[manager.seq_id_to_row[i][completed.seq_id], :11].clone()
                      for i, table in enumerate(manager.buffer_req_to_token_slots)]
    calls = []

    def replay(request):
        calls.append(request)
        state = manager._reconstruction
        for layer in manager.kv_transformer_layer_indices():
            manager._allocate(layer, request.seq.seq_id, len(request.token_ids))
            state.visited_layers.add(layer)
        # Independent oracle: choose the four largest with earliest-index ties.
        state.scores.copy_(torch.tensor([0, 2, 8, 3, 8, 1, 7, 7, 7, 2, 0.]))

    runtime.bind_auxiliary_prefill(replay)
    manager._reconstruction_source[completed.seq_id] = completed.token_ids.copy()
    runtime.finish_step(SparseStepContext([completed, partial, short], True, None))
    assert len(calls) == 3
    assert calls[1].token_ids == (90, 91, 3, 4, 5, 6, 7, 8, 9)
    for layer in manager.kv_transformer_layer_indices():
        assert manager.decode_kv_lens_for_layer(layer, [completed, partial, short]) == [4, 8, 3]
        row = manager.seq_id_to_row[layer][completed.seq_id]
        torch.testing.assert_close(manager.buffer_req_to_token_slots[layer][row, :4],
                                   original_slots[layer][torch.tensor([2, 4, 6, 7])])
    assert completed.token_ids == list(range(11))
    for seq in (completed, partial, short):
        manager.free_seq(seq.seq_id)
    assert manager._num_free_slots == manager.layer_num_slots


def test_auxiliary_forward_restores_context_and_does_not_mutate_request_on_failure():
    """Replay failure must leave business token/sampling state and context intact."""
    from sparseengine.engine.model_runner import ModelRunner
    from sparseengine.engine.sparse_methods.base import AuxiliaryPrefillRequest
    from sparseengine.utils.context import get_context, reset_context

    seq = Sequence([1, 2, 3])
    seq.current_chunk_size = 3
    runner = object.__new__(ModelRunner)
    context = get_context()
    context.seqs = [seq]
    snapshot = vars(context).copy()

    def failing_prepare(seqs, is_prefill):
        replay = seqs[0]
        assert replay is not seq
        assert replay.token_ids == [4, 5]
        assert replay.num_tokens == 5
        assert replay.num_prefilled_tokens == 3
        context.seqs = seqs
        context.is_prefill = is_prefill
        raise RuntimeError("injected forward failure")

    runner.prepare_step = failing_prepare
    try:
        with pytest.raises(RuntimeError, match="injected"):
            runner._run_auxiliary_prefill(AuxiliaryPrefillRequest(seq, (4, 5), 3))
        assert vars(context) == snapshot
        assert seq.token_ids == [1, 2, 3]
        assert seq.num_prefilled_tokens == 0
        assert seq.num_prompt_tokens == seq.num_tokens == 3
    finally:
        reset_context()


@pytest.mark.parametrize("field,value", [
    ("kvzip_token_budget", 0), ("kvzip_token_budget", True),
    ("kvzip_score_chunk_size", 0), ("kvzip_prev_postfix_size", -1),
    ("sparse_attn_score_dtype", "float16"),
])
def test_invalid_reconstruction_config_fails_before_execution(field, value):
    from sparseengine.configs.sparse import _normalize_kvzip

    config = _manager().config
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        _normalize_kvzip(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("replay,heads", [(17, 6), (137, 14)])
def test_reconstruction_scores_match_independent_probability_oracle(replay, heads):
    """Verify GQA head/query reduction and global layer max through the CUDA scorer."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    length, kv_heads, dim = 19, 2, 32
    manager = object.__new__(KVzipCacheManager)
    manager.config = SimpleNamespace(sparse_attn_score_dtype="float32")
    manager.device = device
    manager._prefill_step_score_buffers = {}
    from sparseengine.kernels.triton.prefill_score import PrefillScoreWorkspace
    from sparseengine.engine.cache_manager.methods.kvzip import ReconstructionChunk
    manager._prefill_score_workspace = PrefillScoreWorkspace()
    manager.seq_id_to_row = [{3: 0}, {3: 0}]
    manager.row_seq_lens = [[length + replay], [length + replay]]
    scores = torch.zeros(length, device=device)
    start = torch.tensor([length], dtype=torch.int32, device=device)
    end = torch.tensor([length + replay], dtype=torch.int32, device=device)
    manager._reconstruction = ReconstructionChunk(3, length, replay, scores, start, end)
    slots = torch.randperm(length + replay, device=device).to(torch.int32)[None]
    expected = torch.zeros_like(scores)
    for layer in range(2):
        q = torch.randn(replay, heads, dim, dtype=torch.float16, device=device)
        k = torch.randn(length + replay, kv_heads, dim, dtype=q.dtype, device=device)
        keys = k[slots[0, :length].long()].repeat_interleave(heads // kv_heads, dim=1)
        logits = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) / dim**0.5
        oracle = logits.softmax(-1).mean(1).amax(0)
        expected = torch.maximum(expected, oracle)
        view = SimpleNamespace(
            payload=ExplicitKVPayload(k_cache=k, v_cache=k),
            meta=SimpleNamespace(active_slots=slots, req_indices=torch.zeros(1, dtype=torch.int32, device=device),
                                 context_lens=end),
        )
        manager.collect_prefill_attention_score(
            layer, q, view, b_start_loc=torch.zeros(1, dtype=torch.int32, device=device),
            chunk_lens=torch.tensor([replay], dtype=torch.int32, device=device),
        )
    torch.testing.assert_close(scores, expected, rtol=2e-3, atol=2e-4)
