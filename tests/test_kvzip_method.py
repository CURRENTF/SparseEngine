from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sparseengine.platforms as platforms
from sparseengine.config import RuntimeLayout
from sparseengine.distributed.parallel_context import ParallelGroup
from sparseengine.engine.cache_manager.base import CacheManager, ExplicitKVPayload
from sparseengine.engine.cache_manager.methods.kvzip import KVzipCacheManager, KVzipHeadSlotCacheManager
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_methods.base import SparseStepContext
from sparseengine.engine.sparse_methods.kvzip import KVzipRuntime, select_kvzip_layers
from sparseengine.platforms.cpu import CpuPlatform


def _manager(*, profiling=False, head_slots=False, cuda=False, **overrides):
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
        kv_cache_layout="head_slots" if head_slots else "token",
    )
    vars(config).update(overrides)
    manager_cls = KVzipHeadSlotCacheManager if head_slots else KVzipCacheManager
    context = SimpleNamespace(
        world_rank=0, world_size=1, attn_tp_rank=0, attn_tp_size=1,
        moe_ep_rank=0, moe_ep_size=1, attn_dp_rank=0, attn_dp_size=1,
        attn_tp=ParallelGroup(None, (0,), 0, 1),
    )
    if cuda:
        from sparseengine.platforms.cuda import CudaPlatform
        platform = CudaPlatform()
    else:
        platform = CpuPlatform()
    with patch.object(platforms, "_current_platform", platform):
        if profiling:
            from sparseengine.engine.startup.capacity import profiling_kv_budget_bytes, profiling_kv_slots

            config.startup_cache_phase = "profiling"
            manager = manager_cls(config, context, allocation_budget_bytes=
                profiling_kv_budget_bytes(config, profiling_kv_slots(config)))
        else:
            with patch.object(CacheManager, "_get_available_slots_info", return_value=(1_000_000, 128)):
                manager = manager_cls(config, context)
    manager.set_reconstruction_prompt({"first": [90], "continuation": [91], "suffix": [92]})
    return manager


def _context(length):
    return Sequence(list(range(length)) + [99], context_length=length, context_prefix_length=1)


def _seed(manager, seq):
    for layer in manager.kv_transformer_layer_indices():
        manager._allocate(layer, seq.seq_id, seq.context_length or seq.num_prompt_tokens)


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
    manager.free_seq(seq.seq_id)
    assert manager._num_free_slots == manager.layer_num_slots


def test_reconstruction_failure_releases_only_temporary_slots():
    """A failed layer must neither leak replay slots nor release another request."""
    manager = _manager()
    seq, other = _context(10), _context(7)
    _seed(manager, seq)
    _seed(manager, other)
    original = [x.clone() for x in manager.buffer_req_to_token_slots]
    counts = manager._num_free_slots.copy()
    with pytest.raises(RuntimeError, match="injected"):
        with manager.reconstruction_chunk(seq, 6, torch.zeros(2, 1, 10), chunk_start=1, chunk_end=6):
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
    seq = _context(10)
    _seed(manager, seq)
    with pytest.raises(RuntimeError, match="every KV layer"):
        with manager.reconstruction_chunk(seq, 5, torch.zeros(2, 1, 10), chunk_start=1, chunk_end=6):
            for layer in manager.kv_transformer_layer_indices():
                manager._allocate(layer, seq.seq_id, 5)
    assert manager.num_free_slots + manager.reconstruction_slot_reserve == min(manager._num_free_slots)
    for layer, free in enumerate(manager._num_free_slots):
        assert manager.decode_window_budgets()[f"layer_{layer}"] == free - manager.reconstruction_slot_reserve
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        manager.prompt_admission_costs(_context(124))


def test_tp_chunk_only_ipc_retains_identical_reconstruction_source():
    """Follower IPC drops earlier chunks; reconstruction must retain them itself."""
    import pickle
    from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager

    leader, follower = _manager(), _manager()
    seq = _context(11)
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


def test_runtime_compacts_context_before_question_with_independent_layer_selection():
    """Previously layer collapse and post-question eviction selected identical rows."""
    manager = _manager()
    runtime = KVzipRuntime(manager.config, manager)
    completed, partial, short = (_context(n) for n in (11, 8, 3))
    for seq in (completed, partial, short):
        _seed(manager, seq)
        seq.current_chunk_size = seq.context_length
    partial.current_chunk_size = 4
    original = [table[manager.seq_id_to_row[i][completed.seq_id], :11].clone()
                for i, table in enumerate(manager.buffer_req_to_token_slots)]
    calls = []
    # Global rank 8 has score 3; strict threshold retains 7 across both layers.
    oracle_scores = torch.tensor([[0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                                  [0, 0, 1, 2, 3, 4, 2, 1, 0, 0, 0.]])

    def replay(request):
        calls.append(request)
        state = manager._reconstruction
        for layer in manager.kv_transformer_layer_indices():
            manager._allocate(layer, request.seq.seq_id, len(request.token_ids))
            state.visited_layers.add(layer)
        state.scores[:, 0, state.chunk_start:state.chunk_end] = oracle_scores[:, state.chunk_start:state.chunk_end]

    runtime.bind_auxiliary_prefill(replay)
    manager._reconstruction_source[completed.seq_id] = list(range(11))
    runtime.finish_step(SparseStepContext([completed, partial, short], True, None))
    assert len(calls) == 2
    assert calls[0].token_ids == (90, 92, 1, 2, 3, 4, 5)
    assert calls[1].token_ids == (91, 4, 5, 92, 6, 7, 8, 9, 10)
    expected = ([0, 1, 2, 3, 4, 5, 6], [0, 5])
    for layer, positions in enumerate(expected):
        assert manager.decode_kv_lens_for_layer(layer, [completed, partial, short]) == [len(positions), 8, 3]
        row = manager.seq_id_to_row[layer][completed.seq_id]
        torch.testing.assert_close(manager.buffer_req_to_token_slots[layer][row, :len(positions)],
                                   original[layer][torch.tensor(positions)])
    assert completed.token_ids == list(range(11)) + [99]
    # Question prefill cannot trigger reconstruction or collect question tokens.
    completed.num_prefilled_tokens = 11
    completed.current_chunk_size = 1
    input_ids, positions, *_ = manager.prepare_step([completed], True)
    assert input_ids.tolist() == [99] and positions.tolist() == [11]
    assert [manager.decode_kv_lens_for_layer(i, [completed])[0] for i in range(2)] == [8, 3]
    runtime.finish_step(SparseStepContext([completed], True, None))
    assert len(calls) == 2 and not manager._reconstruction_source
    for seq in (completed, partial, short):
        manager.free_seq(seq.seq_id)
    assert manager._num_free_slots == manager.layer_num_slots


@pytest.mark.parametrize("other_compressed", [True, False])
def test_ragged_question_prefill_uses_physical_rows_after_context_compaction(other_compressed):
    """Single-request coverage missed the slow allocator's logical-length check."""
    manager = _manager(max_num_seqs_in_batch=2)
    question = Sequence(list(range(10)) + [90], context_length=10, context_prefix_length=1)
    other = Sequence(list(range(12)) + [91, 92], context_length=12, context_prefix_length=1)
    selected = [torch.tensor([0, 1, 2]), torch.tensor([0, 2])]
    _seed(manager, question)
    manager.compact_reconstruction(question, selected)
    question.num_prefilled_tokens, question.current_chunk_size = 10, 1
    if other_compressed:
        _seed(manager, other)
        manager.compact_reconstruction(other, selected)
        other.num_prefilled_tokens = 12
    else:
        for layer in manager.kv_transformer_layer_indices():
            manager._allocate(layer, other.seq_id, 4)
        manager._reconstruction_source[other.seq_id] = other.token_ids[:4]
        other.num_prefilled_tokens = 4
    other.current_chunk_size = 2
    seqs = [question, other]
    before = [[manager.buffer_req_to_token_slots[layer][
        manager.seq_id_to_row[layer][seq.seq_id],
        :manager.row_seq_lens[layer][manager.seq_id_to_row[layer][seq.seq_id]],
    ].clone() for seq in seqs] for layer in range(2)]
    input_ids, positions, *_ = manager.prepare_step(seqs, True)
    start = other.num_prefilled_tokens
    assert input_ids.tolist() == [90, *other.token_ids[start:start + 2]]
    assert positions.tolist() == [10, start, start + 1]
    for layer in range(2):
        expected_lens = [len(old) + seq.current_chunk_size for old, seq in zip(before[layer], seqs)]
        assert manager.layer_batch_states[layer].context_lens.tolist() == expected_lens
        for seq, old in zip(seqs, before[layer]):
            row = manager.seq_id_to_row[layer][seq.seq_id]
            torch.testing.assert_close(manager.buffer_req_to_token_slots[layer][row, :len(old)], old)
    if not other_compressed:
        assert manager._reconstruction_source[other.seq_id] == other.token_ids[:6]
    for seq in seqs:
        manager.free_seq(seq.seq_id)
    assert manager._num_free_slots == manager.layer_num_slots


def test_global_threshold_ties_protect_prefix_and_never_exceed_budget():
    """An exact top-k or per-layer budget would violate upstream's strict threshold."""
    scores = torch.tensor([[[0, 8, 7, 6, 5]], [[0, 1, 1, 1, 1.]]])
    keep = select_kvzip_layers(scores, 2, 1)
    assert [x.tolist() for x in keep] == [[0, 1, 2, 3], [0]]
    ties = select_kvzip_layers(torch.ones(2, 1, 5), 2, 1)
    assert [x.tolist() for x in ties] == [[0], [0]]
    assert sum(len(x) - 1 for x in keep) < 2 * 2


def _head_union_fixture():
    return torch.tensor([[[0, 9, 0, 7, 0], [0, 0, 8, 0, 6]],
                         [[0, 5, 4, 3, 2], [0, 1, 0.9, 0.8, 0.7]]])


def _original_head_union_oracle(scores, budget, prefix):
    """Direct original pair masks, with Python ranking independent of selection."""
    layers, heads, length = scores.shape
    population = sorted([float(scores[l, h, t]) for l in range(layers)
                         for h in range(heads) for t in range(prefix, length)], reverse=True)
    threshold = population[layers * heads * budget - 1]
    return [list(range(prefix)) + [t for t in range(prefix, length)
            if any(float(scores[l, h, t]) > threshold for h in range(heads))]
            for l in range(layers)]


def test_head_union_threshold_preserves_disjoint_original_selections():
    """Early head MAX lost independently important tokens and changed layer budgets."""
    scores = _head_union_fixture()
    keep = select_kvzip_layers(scores, 1, 1)
    assert [x.tolist() for x in keep] == _original_head_union_oracle(scores, 1, 1)
    # Distinct per-head positions need more shared slots than the pair-average
    # allowance. A physical-budget cap here would reintroduce the observed loss.
    assert sum(len(x) - 1 for x in keep) > scores.shape[0]
    assert [x.tolist() for x in select_kvzip_layers(torch.ones_like(scores), 1, 1)] == [[0], [0]]


def test_independent_head_selection_keeps_original_pairs_without_union_inflation():
    """Shared positions cannot represent independently selected head supports."""
    scores = _head_union_fixture()
    selected = select_kvzip_layers(scores, 1, 1, share_heads=False)
    threshold = sorted(scores[..., 1:].flatten().tolist(), reverse=True)[3]
    for layer, selection in enumerate(selected):
        for head, indices in enumerate(selection.indices):
            expected = [0] + [t for t in range(1, scores.shape[-1]) if scores[layer, head, t] > threshold]
            assert indices.tolist() == expected
    assert sum(indices.numel() - 1 for s in selected for indices in s.indices) < 4


def test_head_slot_replay_failure_rolls_back_only_owned_suffix():
    """Reconstruction must release every head's scratch without freeing other owners."""
    import numpy as np
    from sparseengine.engine.cache_manager.base import HeadSlotSelection

    manager = _manager(head_slots=True)
    seq, other = _context(10), _context(7)
    manager.allocator.apply_append(manager.allocator.plan_append([
        (seq.seq_id, list(range(10))), (other.seq_id, list(range(7)))]))
    for current in (seq, other):
        manager.publish_seq_metadata(current.seq_id)
    before = manager.allocator.slots.copy()
    counts = manager.allocator.free_counts.copy()
    with pytest.raises(RuntimeError, match='injected'):
        with manager.reconstruction_chunk(seq, 6, torch.zeros(2, 1, 10), chunk_start=1, chunk_end=6):
            manager.allocator.apply_append(manager.allocator.plan_append([(seq.seq_id, list(range(10, 16)))]))
            manager.publish_seq_metadata(seq.seq_id)
            raise RuntimeError('injected')
    np.testing.assert_array_equal(manager.allocator.slots, before)
    np.testing.assert_array_equal(manager.allocator.free_counts, counts)
    manager.allocator.assert_invariants()
    manager.compact_reconstruction(seq, [HeadSlotSelection((torch.tensor([0, 2]),)),
                                        HeadSlotSelection((torch.tensor([0, 4, 8]),))])
    manager.allocator.assert_invariants()
    for current in (seq, other):
        manager.free_seq(current.seq_id)
    np.testing.assert_array_equal(manager.allocator.free_counts, manager.allocator.capacities)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_head_slot_reconstruction_scorer_matches_independent_head_oracle():
    """Virtual MQA packing must retain the original head identity and denominator."""
    from sparseengine.engine.cache_manager.methods.kvzip import ReconstructionChunk
    from sparseengine.kernels.triton.prefill_score import PrefillScoreWorkspace

    torch.manual_seed(47)
    length, replay, heads, groups, dim = 19, 137, 2, 7, 128
    sink, start, end = 3, 6, 15
    device = torch.device('cuda')
    manager = object.__new__(KVzipHeadSlotCacheManager)
    manager.num_kv_heads, manager.device = heads, device
    manager.kv_layer_index = lambda layer: layer
    manager.allocator = SimpleNamespace(seq_rows={3: 0},
        lengths=torch.full((1, 1, heads), length + replay).numpy())
    manager._prefill_score_workspace = PrefillScoreWorkspace()
    scores = torch.full((1, heads, length), -7.0, device=device)
    cached = torch.tensor([length], dtype=torch.int32, device=device)
    manager._reconstruction = ReconstructionChunk(3, length, replay, scores,
        cached, cached + replay, sink, start, end)
    slots = torch.randperm(heads * (length + replay), device=device).reshape(1, heads, length + replay).int()
    keys = torch.randn(heads * (length + replay), dim, dtype=torch.bfloat16, device=device)
    q = torch.randn(replay, heads * groups, dim, dtype=keys.dtype, device=device)
    view = SimpleNamespace(payload=ExplicitKVPayload(keys, keys, backend='head_slots'),
        meta=SimpleNamespace(active_slots=slots))
    manager.collect_prefill_attention_score(0, q, view, b_start_loc=torch.zeros_like(cached),
        chunk_lens=torch.tensor([replay], dtype=torch.int32, device=device))
    positions = torch.cat((torch.arange(sink), torch.arange(start, end),
                           torch.arange(length, length + replay))).to(device)
    expected = torch.full_like(scores, -7.0)
    for head in range(heads):
        k = keys[slots[0, head, positions].long()].float()
        logits = torch.einsum('qgd,kd->gqk', q[:, head * groups:(head + 1) * groups].float(), k) / dim**0.5
        future = positions[None, :] > (length + torch.arange(replay, device=device))[:, None]
        logits.masked_fill_(future[None], -torch.inf)
        expected[0, head, start:end] = logits.softmax(-1)[:, :, sink:sink + end - start].amax((0, 1))
    torch.testing.assert_close(scores, expected, atol=2e-4, rtol=2e-3)


def test_head_slot_profiling_capacity_preserves_replay_reserve():
    """Startup must count physical head slots and metadata before allocation."""
    from sparseengine.engine.runtime_state import RuntimeState
    from sparseengine.engine.startup.capacity import profiling_prefill_chunk_lengths

    manager = _manager(head_slots=True, profiling=True)
    runtime = RuntimeState(manager.config, manager)
    assert runtime.startup_batch_fits(profiling_prefill_chunk_lengths(manager.config), max_tokens=2)
    assert manager.num_free_slots > 0
    manager.allocator.assert_invariants()


def test_head_slot_partial_startup_prefill_does_not_reserve_scratch_per_row():
    """Compile warmup stalled after 112/128 tokens despite free profiling slots."""
    from sparseengine.engine.runtime_state import RuntimeState
    from sparseengine.engine.scheduler import Scheduler
    from sparseengine.sampling_params import SamplingParams

    manager = _manager(head_slots=True, profiling=True, max_model_len=624,
                       engine_prefill_chunk_size=512, max_num_batched_tokens=256,
                       max_num_seqs_in_batch=4, max_decoding_seqs=4,
                       max_num_seqs_in_gpu=4, kvzip_token_budget=32,
                       decode_reservation_tokens=8, eos=-1)
    runtime = RuntimeState(manager.config, manager)
    seq = Sequence(list(range(128)), SamplingParams(max_tokens=1))
    # Reproduce the live failure point using the physical ownership ledger.
    manager.allocator.apply_append(manager.allocator.plan_append([
        (seq.seq_id, list(range(112)))]))
    seq.num_prefilled_tokens = 112
    scheduler = Scheduler(manager.config, runtime)
    scheduler.waiting.append(seq)
    scheduled, is_prefill, preempted = scheduler.schedule()
    assert scheduled == [seq] and is_prefill and not preempted
    assert seq.current_chunk_size == 16
    manager.allocator.apply_append(manager.allocator.plan_append([
        (seq.seq_id, list(range(112, 128)))]))
    scheduler.postprocess(scheduled, [0], True)
    assert seq.num_prefilled_tokens == 128 and seq.is_finished
    assert manager.allocator.free_counts.min() >= (
        manager.scratch_slot_reserve * manager.num_kv_heads)
    manager.allocator.assert_invariants()
    manager.free_seq(seq.seq_id)
    manager.allocator.assert_invariants()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_head_slot_append_after_compaction_preserves_owners_and_graph_padding():
    """Unequal head histories must append at their own columns with stable padded writes."""
    import numpy as np
    from sparseengine.engine.cache_manager.base import HeadSlotSelection, ExplicitKVWrite
    from sparseengine.engine.decode_graph_contract import DecodeGraphContract, DecodeGraphInputs

    hf = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2,
                         num_attention_heads=4, hidden_size=64, head_dim=16, dtype=torch.float32)
    manager = _manager(head_slots=True, cuda=True, hf_config=hf)
    seq, other = _context(10), _context(7)
    seq.current_chunk_size, other.current_chunk_size = 10, 7
    manager.prepare_step([seq, other], True)
    manager.allocator.assert_invariants()
    for layer in manager.kv_transformer_layer_indices():
        view = manager.build_prefill_compute_view(layer, None, None, None)
        row = manager.allocator.seq_rows[seq.seq_id]
        np.testing.assert_array_equal(view.meta.active_slots[row, :, :10].cpu(),
                                      manager.allocator.slots[layer, row, :, :10])
    keep = [HeadSlotSelection((torch.tensor([0, 2]), torch.tensor([0, 4, 8]))),
            HeadSlotSelection((torch.tensor([0, 3, 5, 9]), torch.tensor([0])))]
    manager.compact_reconstruction(seq, keep)
    other_row = manager.allocator.seq_rows[other.seq_id]
    other_slots = manager.allocator.slots[:, other_row].copy()
    seq.num_prefilled_tokens = 10
    seq.current_chunk_size = 1
    manager.prepare_step([seq], True)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(33)
    contract = DecodeGraphContract(method='kvzip', topology_path_id='unified',
                                  batch_capacity=3, context_capacity=manager.max_model_len)
    inputs = DecodeGraphInputs.allocate(contract, device=manager.device, pin_memory=False)
    state = manager.init_decode_graph_state(contract, inputs)
    pointers = inputs.data_ptrs(), state.writes.data_ptr(), state.lengths.data_ptr()
    for _ in range(2):
        manager.prepare_decode_graph_step([seq], state)
        assert pointers == (inputs.data_ptrs(), state.writes.data_ptr(), state.lengths.data_ptr())
        assert state.writes[:, 1:].eq(-1).all()
        assert state.lengths[:, 1:].eq(0).all()
        for layer in manager.kv_transformer_layer_indices():
            k = torch.randn(3, 2, 16, device=manager.device)
            v = torch.randn_like(k)
            before = manager.get_layer_kv_cache(layer)[0].clone()
            manager.store_attention_payload(layer, ExplicitKVWrite(k, v))
            actual, _ = manager.get_layer_kv_cache(layer)
            writes = state.writes[layer, 0].long()
            torch.testing.assert_close(actual[writes], k[0], atol=0, rtol=0)
            unchanged = torch.ones(actual.shape[0], device=manager.device, dtype=torch.bool)
            unchanged[writes] = False
            torch.testing.assert_close(actual[unchanged], before[unchanged], atol=0, rtol=0, equal_nan=True)
        manager.allocator.assert_invariants()
        seq.append_token(34)
    np.testing.assert_array_equal(manager.allocator.slots[:, other_row], other_slots)
    for current in (seq, other):
        manager.free_seq(current.seq_id)
    np.testing.assert_array_equal(manager.allocator.free_counts, manager.allocator.capacities)


def _tp_head_union_worker(rank, scores, init_path, output_dir, backend):
    import json
    from pathlib import Path
    import torch.distributed as dist

    if backend == 'nccl':
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
    else:
        device = torch.device('cpu')
    dist.init_process_group(backend, init_method=f'file://{init_path}', rank=rank, world_size=2)
    try:
        group = ParallelGroup(dist.group.WORLD, (0, 1), rank, 2)
        # Noncontiguous head slices must be collected as distinct score populations.
        keep = select_kvzip_layers(scores[:, rank:rank + 1].to(device), 1, 1,
                                   parallel_group=group)
        Path(output_dir, f'{rank}.json').write_text(json.dumps([x.tolist() for x in keep]))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('backend', ['gloo', 'nccl'])
def test_tp_threshold_and_union_match_global_head_population(tmp_path, backend):
    """MAX-reducing different TP heads before thresholding changes the population."""
    import json
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if backend == 'gloo' and not dist.is_gloo_available():
        pytest.skip('Gloo required')
    if backend == 'nccl' and (not dist.is_nccl_available() or torch.cuda.device_count() < 2):
        pytest.skip('Two CUDA devices and NCCL required')
    scores = _head_union_fixture()
    mp.spawn(_tp_head_union_worker, args=(scores, str(tmp_path / 'init'), str(tmp_path), backend),
             nprocs=2, join=True)
    expected = _original_head_union_oracle(scores, 1, 1)
    for rank in range(2):
        assert json.loads((tmp_path / f'{rank}.json').read_text()) == expected


def test_context_boundary_survives_ipc_and_recompute_without_early_sampling():
    """A chunk spanning the boundary would make scores depend on the question."""
    import pickle
    from test_prefill_schedule_policy import make_scheduler
    scheduler = make_scheduler("all_chunked", chunk=8, max_tokens=8)
    seq = _context(6)
    scheduler.add(seq)
    batch, prefill, _ = scheduler.schedule()
    assert batch == [seq] and prefill and seq.current_chunk_size == 6
    remote = pickle.loads(pickle.dumps(seq))
    assert remote.context_length == 6 and remote.context_prefix_length == 1
    assert remote.is_context_boundary_prefill and not remote.is_last_chunk_prefill
    scheduler.postprocess(batch, [123], True)
    assert seq.num_completion_tokens == 0 and seq.token_ids[-1] == 99
    batch, prefill, _ = scheduler.schedule()
    assert batch == [seq] and prefill and seq.current_chunk_size == 1
    scheduler.postprocess(batch, [123], True)
    assert seq.num_completion_tokens == 1
    seq.start_recompute_replay()
    assert seq.prefill_stage_remaining == 6
    seq.current_chunk_size = 6
    remote = pickle.loads(pickle.dumps(seq))
    assert remote.is_context_boundary_prefill and remote.context_prefix_length == 1


def test_missing_context_boundary_and_unprotected_prefix_fail_explicitly():
    manager = _manager()
    with pytest.raises(ValueError, match="context_length"):
        manager.prompt_admission_costs(Sequence(list(range(10))))
    with pytest.raises(ValueError, match="system prefix"):
        manager.prompt_admission_costs(Sequence(list(range(10)), context_length=8))


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
@pytest.mark.parametrize("replay,heads,dim,dtype,length,sink,chunk_start,chunk_end", [
    (17, 6, 32, torch.float16, 19, 3, 6, 15),
    (137, 14, 128, torch.bfloat16, 19, 3, 6, 15),
    (137, 26, 32, torch.float16, 19, 3, 6, 15),
    # Disjoint tiles, shared boundary tiles, and an absent sink exercise the
    # compact reconstruction grid; the original tiny cases share one key tile.
    (137, 14, 128, torch.bfloat16, 4097, 65, 257, 2057),
    (65, 6, 64, torch.float16, 1023, 0, 129, 513),
    (137, 14, 128, torch.bfloat16, 129, 65, 65, 129),
])
def test_reconstruction_scores_match_independent_probability_oracle(
    replay, heads, dim, dtype, length, sink, chunk_start, chunk_end,
):
    """Oracle concatenates sink/chunk/replay keys and causally masks only replay."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    kv_heads = 2
    manager = object.__new__(KVzipCacheManager)
    manager.config = SimpleNamespace(sparse_attn_score_dtype="float32")
    manager.device = device
    manager._prefill_step_score_buffers = {}
    manager.kv_layer_index = lambda layer: layer
    from sparseengine.kernels.triton.prefill_score import PrefillScoreWorkspace
    from sparseengine.engine.cache_manager.methods.kvzip import ReconstructionChunk
    manager._prefill_score_workspace = PrefillScoreWorkspace()
    manager.num_kv_heads = kv_heads
    manager.seq_id_to_row = [{3: 0}, {3: 0}]
    manager.row_seq_lens = [[length + replay], [length + replay]]
    scores = torch.full((2, kv_heads, length), -7.0, device=device)
    start = torch.tensor([length], dtype=torch.int32, device=device)
    end = torch.tensor([length + replay], dtype=torch.int32, device=device)
    manager._reconstruction = ReconstructionChunk(3, length, replay, scores, start, end,
                                                   sink, chunk_start, chunk_end)
    slots = torch.randperm(length + replay, device=device).to(torch.int32)[None]
    expected = scores.clone()
    for layer in range(2):
        q = torch.randn(replay, heads, dim, dtype=dtype, device=device)
        k = torch.randn(length + replay, kv_heads, dim, dtype=q.dtype, device=device)
        positions = torch.cat((torch.arange(sink), torch.arange(chunk_start, chunk_end),
                               torch.arange(length, length + replay))).to(device)
        keys = k[slots[0, positions].long()].repeat_interleave(heads // kv_heads, dim=1)
        logits = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) / dim**0.5
        # The last R keys are replay: query i may attend replay positions <= i.
        causal = torch.arange(replay, device=device)[None, :] > torch.arange(replay, device=device)[:, None]
        logits[:, :, -replay:].masked_fill_(causal, -torch.inf)
        probabilities = logits.softmax(-1)[:, :, sink:sink + chunk_end - chunk_start]
        oracle = probabilities.reshape(kv_heads, heads // kv_heads, replay, -1).amax((1, 2))
        expected[layer, :, chunk_start:chunk_end] = oracle
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
