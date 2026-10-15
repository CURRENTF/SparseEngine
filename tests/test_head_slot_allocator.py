"""Regressions for ownership, atomic batch admission, and ragged reclamation."""

import numpy as np
import pytest

from sparseengine.engine.cache_manager.head_slot_allocator import HeadSlotAllocator


def test_ragged_compaction_preserves_addresses_and_recycles_once():
    a = HeadSlotAllocator((47, 53), heads=3, width=20, rows=2)
    a.apply_append(a.plan_append([(4, list(range(8))), (9, [0, 1])]))
    original = a.slots[:, a.seq_rows[4]].copy()
    positions = [[np.array([0, 4]), np.array([1, 3, 5]), np.array([0, 2, 4, 6])],
                 [np.array([2]), np.array([0, 3, 7]), np.array([1, 3, 4, 6, 7])]]
    released = a.compact(4, positions)
    assert released == (24 - 9, 24 - 9)
    for l, heads in enumerate(positions):
        for h, selected in enumerate(heads):
            np.testing.assert_array_equal(a.slots[l, a.seq_rows[4], h, :len(selected)],
                                          original[l, h, selected])
    a.assert_invariants()
    a.apply_append(a.plan_append([(4, [8, 9]), (9, [2])]))
    a.assert_invariants()
    for sid in (4, 9, 4, 9):
        a.free_seq(sid)
        a.assert_invariants()
    np.testing.assert_array_equal(a.free_counts, a.capacities)


def test_multilayer_bottleneck_reservation_and_failed_batch_are_atomic():
    a = HeadSlotAllocator((60, 25), heads=3, width=20, rows=3)
    assert a.append_capacity(1, (0, 7)) == 6
    a.apply_append(a.plan_append([(1, [0, 1, 2])]))
    before = (a.slots.copy(), a.free_counts.copy(), a.generation, dict(a.seq_rows))
    # Each individual request fits; their total exceeds layer 1's free pool.
    with pytest.raises(RuntimeError):
        a.plan_append([(1, [3, 4, 5]), (2, [0, 1, 2])])
    np.testing.assert_array_equal(a.slots, before[0])
    np.testing.assert_array_equal(a.free_counts, before[1])
    assert (a.generation, a.seq_rows) == (before[2], before[3])
    a.assert_invariants((12, 6))


def test_stale_plan_and_invalid_selection_cannot_mutate_live_owners():
    a = HeadSlotAllocator((32, 40), heads=2, width=8, rows=2)
    p = a.plan_append([(1, [0, 1, 2])])
    a.apply_append(p)
    with pytest.raises(RuntimeError, match="Stale"):
        a.apply_append(p)
    before = a.slots.copy()
    with pytest.raises(ValueError, match="ordered unique"):
        a.compact(1, [[np.array([0]), np.array([1])],
                      [np.array([0]), np.array([2, 2])]])
    np.testing.assert_array_equal(a.slots, before)
    a.assert_invariants()


def test_largest_retained_head_limits_metadata_append_capacity():
    a = HeadSlotAllocator((100, 100), heads=2, width=8, rows=2)
    a.apply_append(a.plan_append([(1, list(range(7)))]))
    a.compact(1, [[np.array([0]), np.array([0, 2, 4, 6])]] * 2)
    assert a.append_capacity(1) == 4
    a.assert_invariants()


def test_chain_reservations_share_units_with_append_and_cancel_once():
    from collections import deque
    from types import SimpleNamespace
    from sparseengine.engine.cache_manager.head_slots import HeadSlotCacheManager
    from sparseengine.engine.chain_cache import ChainCacheCoordinator, ChainCapacityError, ChainGoneError
    from sparseengine.engine.runtime_state import RuntimeState
    from sparseengine.engine.sequence import Sequence
    from sparseengine.sampling_params import SamplingParams

    config = SimpleNamespace(sparse_method="snapkv", model="test-model",
                             hf_config=SimpleNamespace(model_type="llama", dtype="bfloat16"),
                             tensor_parallel_size=1, max_model_len=32, max_num_seqs_in_gpu=3,
                             full_attention_layers=[], kv_cache_layout="head_slots",
                             decode_reservation_tokens=3, engine_prefill_chunk_size=8)
    manager = object.__new__(HeadSlotCacheManager)
    manager.config = config
    manager.runtime_layout = SimpleNamespace(kv_idx_to_layer_idx=(0, 1))
    manager.num_kv_layers, manager.num_kv_heads = 2, 2
    manager.allocator = HeadSlotAllocator((40, 24), 2, 32, 3)
    coordinator = ChainCacheCoordinator(config, manager)
    runtime = RuntimeState(config, manager, chain_cache_coordinator=coordinator)
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=3))
    seq.seq_id, seq.chain_id = 1, "a"
    first = coordinator.plan_admission(chain_id="a", seq_id=1, token_ids=seq.token_ids)
    coordinator.apply_admission(first)
    assert coordinator._outstanding_active_reservations() == ((8, 8), 1)
    manager.allocator.apply_append(manager.allocator.plan_append([(1, [0, 1, 2, 3])]))
    assert coordinator._outstanding_active_reservations() == ((0, 0), 0)
    assert runtime.reserve_decode_windows([seq], deque()) is None
    # A three-token window needs six physical head slots in each layer. Chain
    # admission and the scalar prefill oracle must reserve the same capacity.
    assert runtime.decode_reservations.outstanding() == {"layer_0": 6, "layer_1": 6}
    manager.allocator.assert_invariants(manager._outstanding())
    assert runtime.prompt_admission_free_slots() == 5
    assert runtime.prefill_step_free_slots() == 5
    with pytest.raises(ChainCapacityError):
        coordinator.plan_admission(chain_id="b", seq_id=2, token_ids=[8] * 6)
    second = coordinator.plan_admission(chain_id="b", seq_id=2, token_ids=[8] * 5)
    coordinator.apply_admission(second)
    assert runtime.prompt_admission_free_slots() == 0
    # Renew while the other chain owns an unmaterialized prompt promise. That
    # promise must be charged once, and its head-slot units must match decode.
    runtime.decode_reservations.release(seq.seq_id)
    assert runtime.reserve_decode_windows([seq], deque()) is None
    assert runtime.decode_reservations.outstanding() == {"layer_0": 6, "layer_1": 6}
    # Cancelling an admitted but unallocated chain returns its reservation only.
    coordinator.invalidate("b")
    assert runtime.prompt_admission_free_slots() == 5
    with pytest.raises(ChainGoneError):
        coordinator.invalidate("b")
    assert runtime.prompt_admission_free_slots() == 5
    manager.allocator.apply_append(manager.allocator.plan_append([(1, [4])]))
    seq.append_token(9)
    assert runtime.decode_reservations.outstanding() == {"layer_0": 4, "layer_1": 4}
    assert runtime.prompt_admission_free_slots() == 5
    runtime.decode_reservations.release(seq.seq_id)
    assert runtime.prompt_admission_free_slots() == 7
    manager.allocator.assert_invariants(manager._outstanding())


def test_decode_renewal_preserves_partial_chain_prefill_and_forward_progress():
    """A partial chain was omitted from both pending-prefill reservation owners."""
    from collections import deque
    from types import SimpleNamespace
    from sparseengine.engine.cache_manager.head_slots import HeadSlotCacheManager
    from sparseengine.engine.chain_cache import ChainCacheCoordinator
    from sparseengine.engine.runtime_state import RuntimeState
    from sparseengine.engine.scheduler import Scheduler
    from sparseengine.engine.sequence import Sequence
    from sparseengine.sampling_params import SamplingParams

    config = SimpleNamespace(sparse_method="snapkv", model="test-model",
        hf_config=SimpleNamespace(model_type="llama", dtype="bfloat16"),
        tensor_parallel_size=1, max_model_len=32, max_num_seqs_in_gpu=3,
        full_attention_layers=[], kv_cache_layout="head_slots", decode_reservation_tokens=3,
        engine_prefill_chunk_size=2, max_num_seqs_in_batch=3, max_num_batched_tokens=8,
        max_decoding_seqs=3, prefill_schedule_policy="all_chunked", eos=-1)
    manager = object.__new__(HeadSlotCacheManager)
    manager.config, manager.runtime_layout = config, SimpleNamespace(kv_idx_to_layer_idx=(0, 1))
    manager.num_kv_layers, manager.num_kv_heads = 2, 2
    manager.allocator = HeadSlotAllocator((24, 24), 2, 32, 3)
    coordinator = ChainCacheCoordinator(config, manager)
    runtime = RuntimeState(config, manager, chain_cache_coordinator=coordinator)
    a = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=9))
    a.chain_id = "a"
    coordinator.apply_admission(coordinator.plan_admission(
        chain_id=a.chain_id, seq_id=a.seq_id, token_ids=a.token_ids))
    manager.allocator.apply_append(manager.allocator.plan_append([(a.seq_id, list(range(4)))]))
    a.num_prefilled_tokens = 4
    a.append_token(9)
    assert runtime.reserve_decode_windows([a], deque()) is None
    manager.allocator.apply_append(manager.allocator.plan_append([(a.seq_id, [4, 5, 6])]))
    for token in (10, 11, 12):
        a.append_token(token)
    b = Sequence([8] * 5, SamplingParams(max_tokens=1))
    b.chain_id = "b"
    coordinator.apply_admission(coordinator.plan_admission(
        chain_id=b.chain_id, seq_id=b.seq_id, token_ids=b.token_ids))
    scheduler = Scheduler(config, runtime)
    scheduler.decoding.append(a)
    scheduler.waiting.append(b)
    for _ in range(b.num_prompt_tokens):
        if b.num_prefilled_tokens == b.num_prompt_tokens:
            break
        if b.num_prefilled_tokens:
            assert runtime.reserve_decode_windows([a], scheduler.waiting) is a
        chain_pending = coordinator._outstanding_active_reservations()[0]
        decode_pending = runtime.decode_reservations.outstanding()
        for layer, free in enumerate(manager.allocator.free_counts):
            assert chain_pending[layer] + decode_pending.get(f"layer_{layer}", 0) <= free
        seqs, is_prefill, preempted = scheduler.schedule()
        assert seqs == [b] and is_prefill and not preempted
        start, size = b.num_prefilled_tokens, b.current_chunk_size
        manager.allocator.apply_append(manager.allocator.plan_append([
            (b.seq_id, list(range(start, start + size)))]))
        scheduler.postprocess(seqs, [0], True)
    assert b.num_prefilled_tokens == b.num_prompt_tokens
    coordinator.invalidate(b.chain_id)
    manager.allocator.free_seq(b.seq_id)
    assert runtime.reserve_decode_windows([a], scheduler.waiting) is None
    seqs, is_prefill, preempted = scheduler.schedule()
    assert seqs == [a] and not is_prefill and not preempted


def test_idle_reclaim_must_cover_each_layer_without_spending_active_reservations():
    from sparseengine.engine.chain_cache import ChainCacheIndex, ChainCapacityError

    index = ChainCacheIndex(max_tombstones=8)
    for sid, chain, physical in [(1, "a", (30, 1)), (2, "b", (1, 7))]:
        plan = index.plan_admission(chain_id=chain, seq_id=sid, token_ids=[sid], fingerprint=b"same")
        index.apply_admission(plan, fingerprint=b"same")
        index.finish(chain, token_ids=[sid, sid], processed_token_count=1,
                     physical_slots_by_layer=physical)
    p = index.plan_admission(chain_id="new", seq_id=3, token_ids=[3], fingerprint=b"same",
                              required_slots_by_layer=(2, 6))
    assert p.victim_chain_ids == ("a", "b")
    with pytest.raises(ChainCapacityError):
        index.plan_admission(chain_id="new", seq_id=3, token_ids=[3], fingerprint=b"same",
                              required_slots_by_layer=(1, 9))
