"""Cross-step regressions for scheduler progress and failed-forward ownership."""

from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.cache_manager.methods.streamingllm import StreamingLLMCacheManager
from sparseengine.engine.chain_cache import ChainCacheCoordinator, ChainGoneError
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.scheduler import Scheduler
from sparseengine.engine.sequence import Sequence, SequenceStatus
from sparseengine.sampling_params import SamplingParams
from test_chain_offload import make_manager
from test_prefill_schedule_policy import FakeMemoryOracle, make_scheduler


def make_runtime(*, capacity=16, decode_limit=1, manager_cls=SnapKVCacheManager):
    manager = make_manager(manager_cls, rows=4, capacity=capacity)
    config = manager.config
    config.max_num_seqs_in_batch = 4
    config.max_decoding_seqs = decode_limit
    config.max_num_batched_tokens = 8
    config.engine_prefill_chunk_size = 4
    config.prefill_schedule_policy = "all_chunked"
    config.eos = -1
    config.decode_reservation_tokens = 1
    config.decode_keep_tokens = 2
    config.snapkv_window_size = 2
    manager.validate_runtime_invariants = True
    runtime = RuntimeState(config, manager)
    return manager, runtime, Scheduler(config, runtime)


def request(prompt_tokens, output_tokens=4):
    return Sequence([1] * prompt_tokens, SamplingParams(
        max_tokens=output_tokens, ignore_eos=True,
    ))


def allocate_step(manager, seqs, is_prefill):
    for seq in seqs:
        for layer in manager.kv_transformer_layer_indices():
            manager._allocate(layer, seq.seq_id, seq.current_chunk_size if is_prefill else 1)


@pytest.mark.parametrize("window", [3, 1024])
def test_streamingllm_bounded_residency_keeps_both_decoders_running(window):
    # Two rows fit at their simultaneous peak, including append-before-evict.
    # Existing SnapKV tests cannot catch this method-specific over-reservation.
    manager, runtime, scheduler = make_runtime(
        capacity=16, decode_limit=2, manager_cls=StreamingLLMCacheManager,
    )
    manager.config.sparse_method = "streamingllm"
    manager.config.sink_keep_tokens = 1
    manager.config.recent_keep_tokens = 3
    runtime.decode_reservations.window = window
    requests = [request(4, 18), request(4, 18)]
    for seq in requests:
        seq.num_prefilled_tokens = 4
        seq.append_token(7)
        for layer in manager.kv_transformer_layer_indices():
            manager._allocate(layer, seq.seq_id, 4)
    scheduler.decoding.extend(requests)
    for _ in range(17):
        seqs, is_prefill, preempted = scheduler.schedule()
        assert len(seqs) == 2 and not is_prefill and not preempted
        allocate_step(manager, seqs, is_prefill)
        # Independent CPU eviction oracle; real allocator and scheduler run.
        for seq in seqs:
            for layer in manager.kv_transformer_layer_indices():
                row = manager.seq_id_to_row[layer][seq.seq_id]
                if manager.row_seq_lens[layer][row] >= 8:
                    manager.free_part_slots(layer, seq, torch.tensor([0, 5, 6, 7]))
        scheduler.postprocess(seqs, [7, 7], is_prefill)
        for seq in seqs:
            if seq.is_finished:
                runtime.free_seq(seq.seq_id)
    assert scheduler.is_finished()
    assert scheduler.total_preemptions == 0
    assert manager.num_free_slots == 16
    assert not runtime.decode_reservations.requests


@pytest.mark.parametrize("favor", [0, 1])
def test_partial_prefill_finishes_and_frees_capacity_at_decode_limit(favor):
    # Both prompts fit, but decode cannot spend B's reserved eight slots.
    # B must finish and compact before A can publish another token.
    manager, runtime, scheduler = make_runtime()
    scheduler.favor_min_decoding_seqs = favor
    a, b = request(4), request(12)
    scheduler.add(a)
    scheduler.add(b)
    for _ in range(12):
        seqs, is_prefill, preempted = scheduler.schedule()
        assert seqs and not preempted
        allocate_step(manager, seqs, is_prefill)
        if is_prefill:
            for seq in seqs:
                if seq.is_last_chunk_prefill and seq.num_prompt_tokens > 4:
                    for layer in manager.kv_transformer_layer_indices():
                        manager.free_part_slots(layer, seq, torch.arange(4))
        scheduler.postprocess(seqs, [7] * len(seqs), is_prefill)
        for seq in seqs:
            if seq.is_finished:
                runtime.free_seq(seq.seq_id)
        if scheduler.is_finished():
            break
    assert scheduler.is_finished()
    assert a.num_completion_tokens == b.num_completion_tokens == 4
    assert manager.num_free_slots == 16
    assert not runtime._resident_seq_ids
    assert not runtime.decode_reservations.requests


def test_decode_limit_allows_partial_prefill_but_keeps_fresh_prompt_queued():
    scheduler = make_scheduler("all_chunked", oracle=FakeMemoryOracle(), chunk=4, max_tokens=8)
    scheduler.max_decoding_seqs = 1
    active, fresh, partial = request(4), request(4), request(8)
    active.num_prefilled_tokens = 4
    active.append_token(7)
    partial.num_prefilled_tokens = 4
    scheduler.decoding.append(active)
    scheduler.waiting.extend([fresh, partial])
    seqs, is_prefill, preempted = scheduler.schedule()
    assert seqs == [partial] and is_prefill and not preempted
    assert list(scheduler.waiting) == [fresh]


@pytest.mark.parametrize("window", [1, 1024])
def test_pyramidkv_last_decode_reclaims_idle_chain_before_capacity_failure(window):
    # The final output needs one append even when the row is above its retained
    # budget. A zero reservation used to bypass the idle-chain reclamation hook.
    manager, runtime, scheduler = make_runtime(capacity=8)
    manager.config.sparse_method = "pyramidkv"
    manager.config.pyramid_layer_ratios = [1.0, 1.0]
    runtime.decode_reservations.window = window
    coordinator = ChainCacheCoordinator(manager.config, manager)
    runtime.chain_cache_coordinator = coordinator
    idle = request(3)
    plan = coordinator.index.plan_admission(
        chain_id="idle", seq_id=idle.seq_id, token_ids=idle.token_ids,
        fingerprint=coordinator.fingerprint,
    )
    coordinator.index.apply_admission(plan, fingerprint=coordinator.fingerprint)
    active = request(4, 3)
    active.num_prefilled_tokens = 4
    active.append_token(7)
    active.append_token(7)
    for layer in manager.kv_transformer_layer_indices():
        manager._allocate(layer, idle.seq_id, 3)
        manager._allocate(layer, active.seq_id, 5)
    coordinator.index.finish(
        "idle", token_ids=idle.token_ids, processed_token_count=3,
        physical_slots_by_layer=(3, 3),
    )
    runtime._resident_seq_ids.update((idle.seq_id, active.seq_id))
    scheduler.decoding.append(active)
    engine = object.__new__(LLMEngine)
    engine.scheduler = scheduler
    engine.model_runner = SimpleNamespace(
        runtime_state=runtime,
        call=lambda method, *args: getattr(runtime, method)(*args),
    )
    scheduler.decode_capacity_reclaimer = engine._reclaim_idle_chains_for_decode

    seqs, is_prefill, preempted = scheduler.schedule()

    assert seqs == [active] and not is_prefill and not preempted
    assert not manager.chain_has_residency(idle.seq_id)
    with pytest.raises(ChainGoneError):
        coordinator.index.lookup("idle")
    allocate_step(manager, seqs, is_prefill)
    scheduler.postprocess(seqs, [7], is_prefill)
    runtime.free_seq(active.seq_id)
    assert scheduler.is_finished()
    assert scheduler.total_preemptions == 0
    assert manager.num_free_slots == 8
    assert not runtime.decode_reservations.requests
    assert not runtime._resident_seq_ids


@pytest.mark.parametrize("reservation_failure", [False, True])
def test_rejected_sole_decode_preemption_keeps_request_abortable(reservation_failure):
    # Cover both the reservation gate and the legacy per-step capacity gate.
    oracle = FakeMemoryOracle(free_slots=0)
    scheduler = make_scheduler("all_chunked", oracle=oracle)
    seq = request(4)
    seq.num_prefilled_tokens = 4
    seq.append_token(7)
    scheduler.decoding.append(seq)
    if reservation_failure:
        oracle.reserve_decode_windows = lambda decoding, waiting: seq
    with pytest.raises(RuntimeError, match="sole remaining decode"):
        scheduler.schedule()
    assert not scheduler.is_finished()
    assert list(scheduler.decoding) == [seq]
    assert scheduler.total_preemptions == 0
    assert not seq.is_recompute_replay
    assert scheduler.abort(seq.seq_id)
    assert scheduler.is_finished()


@pytest.mark.parametrize("phase", ["first_prefill", "partial_prefill", "decode"])
@pytest.mark.parametrize("cleanup_fails_once", [False, True])
@pytest.mark.parametrize("chain_mode", [False, True])
def test_forward_failure_reclaims_owned_slots_or_preserves_abort_owner(phase, cleanup_fails_once, chain_mode):
    manager, runtime, scheduler = make_runtime(decode_limit=4)
    seq = request(8)
    if chain_mode:
        coordinator = ChainCacheCoordinator(manager.config, manager)
        plan = coordinator.index.plan_admission(
            chain_id="test-chain", seq_id=seq.seq_id,
            token_ids=seq.token_ids, fingerprint=coordinator.fingerprint,
        )
        coordinator.index.apply_admission(plan, fingerprint=coordinator.fingerprint)
        runtime.chain_cache_coordinator = coordinator
        seq.chain_id = plan.chain_id
    if phase != "first_prefill":
        seq.num_prefilled_tokens = 4 if phase == "partial_prefill" else 8
        for layer in manager.kv_transformer_layer_indices():
            manager._allocate(layer, seq.seq_id, seq.num_prefilled_tokens)
        runtime._resident_seq_ids.add(seq.seq_id)
    if phase == "decode":
        seq.append_token(7)
        seq.status = SequenceStatus.RUNNING
        scheduler.decoding.append(seq)
    else:
        scheduler.add(seq)

    forward_error = RuntimeError("injected forward failure after KV allocation")
    calls = []
    release_method = "chain_invalidate" if chain_mode else "free_slots"

    def call(method, *args):
        calls.append(method)
        if method == "run":
            allocate_step(manager, *args)
            raise forward_error
        assert method == release_method
        if cleanup_fails_once and calls.count(release_method) == 1:
            raise RuntimeError("injected release failure before mutation")
        if chain_mode:
            return runtime.chain_invalidate(args[0], expected_seq_id=args[1])
        runtime.free_seq(args[0])

    engine = object.__new__(LLMEngine)
    engine.scheduler = scheduler
    engine._active_chain_sequences = {seq.seq_id: seq} if chain_mode else {}
    engine.model_runner = SimpleNamespace(call=call, runtime_state=runtime)
    with pytest.raises(RuntimeError) as raised:
        engine.step()
    assert raised.value is forward_error
    assert scheduler.is_finished() is not cleanup_fails_once
    engine.abort_request(seq.seq_id)
    assert scheduler.is_finished()
    assert seq.is_finished
    assert manager.num_free_slots == 16
    assert not runtime._resident_seq_ids
    assert not runtime.decode_reservations.requests
    assert not engine._active_chain_sequences
    if chain_mode:
        with pytest.raises(ChainGoneError):
            coordinator.index.lookup(seq.chain_id)
    # A later serving cancellation must not free already reclaimed rows twice.
    engine.abort_request(seq.seq_id)
    assert calls.count(release_method) == (2 if cleanup_fails_once else 1)
