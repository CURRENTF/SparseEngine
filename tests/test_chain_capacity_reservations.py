"""CPU lifecycle regressions for unnecessary H2O chain eviction."""
from collections import deque
from types import SimpleNamespace

import pytest

from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
from sparseengine.engine.chain_cache import ChainCacheCoordinator
from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams
from test_chain_offload import make_manager
from test_prefill_schedule_policy import make_scheduler_with_oracle, PREFILL_POLICY_ALL_CHUNKED


def runtime():
    manager = make_manager(cls=H2OCacheManager, rows=4, capacity=24)
    manager.config.sparse_method = "h2o"
    manager.config.h2o_prefill_budget = 8
    manager.config.h2o_decode_budget = 4
    manager.config.h2o_decode_eviction = False
    manager.config.engine_prefill_chunk_size = 4
    manager.config.decode_reservation_tokens = 1
    coordinator = ChainCacheCoordinator(manager.config, manager)
    state = RuntimeState(manager.config, manager, chain_cache_coordinator=coordinator)
    return manager, coordinator, state


def allocate(manager, state, seq_id, length):
    for layer in range(2):
        manager._allocate(layer, seq_id, length)
    state._resident_seq_ids.add(seq_id)


def test_active_reservations_follow_finish_resume_and_invalidate():
    manager, coordinator, state = runtime()
    index = coordinator.index
    for name, sid, reserve in [('a', 1, (3, 5)), ('b', 2, (7, 11))]:
        plan = index.plan_admission(chain_id=name, seq_id=sid, token_ids=[1],
            fingerprint=coordinator.fingerprint, reserved_slots_by_layer=reserve, reserved_rows=1)
        index.apply_admission(plan, fingerprint=coordinator.fingerprint)
    assert coordinator._outstanding_active_reservations() == ((10, 16), 2)
    allocate(manager, state, 1, 1)
    assert coordinator._outstanding_active_reservations() == ((9, 15), 1)
    index.finish('a', token_ids=[1], processed_token_count=1, physical_slots_by_layer=(1, 1))
    assert coordinator._outstanding_active_reservations() == ((7, 11), 1)
    assert set(index.active_records) == {'b'}
    plan = index.plan_admission(chain_id='a', seq_id=1, token_ids=[1, 2],
        fingerprint=coordinator.fingerprint, reserved_slots_by_layer=(4, 6))
    index.apply_admission(plan, fingerprint=coordinator.fingerprint)
    assert coordinator._outstanding_active_reservations() == ((11, 17), 1)
    assert coordinator._outstanding_active_reservations(exclude_seq_ids={1}) == ((7, 11), 1)
    index.invalidate('b')
    assert coordinator._outstanding_active_reservations() == ((4, 6), 0)
    index.invalidate('a')
    manager.free_seq(1)
    assert not index.active_records
    assert coordinator._outstanding_active_reservations() == ((), 0)
    plan = index.plan_admission(chain_id='reset', seq_id=3, token_ids=[1],
        fingerprint=coordinator.fingerprint, reserved_slots_by_layer=(2, 2), reserved_rows=1)
    index.apply_admission(plan, fingerprint=coordinator.fingerprint)
    index.reset()
    assert not index.active_records
    assert coordinator._outstanding_active_reservations() == ((), 0)


def test_active_chain_without_prefill_promises_does_not_read_physical_layers(monkeypatch):
    """Completed prefills owe zero; live row-only and slot promises still count."""
    manager, coordinator, state = runtime()
    index = coordinator.index
    for name, sid, slots, rows in [('done', 1, (), 0), ('row', 2, (), 1), ('slots', 3, (5, 7), 1)]:
        plan = index.plan_admission(chain_id=name, seq_id=sid, token_ids=[1],
            fingerprint=coordinator.fingerprint, reserved_slots_by_layer=slots, reserved_rows=rows)
        index.apply_admission(plan, fingerprint=coordinator.fingerprint)
    allocate(manager, state, 1, 1)
    allocate(manager, state, 3, 2)
    read = manager.chain_physical_residency
    def checked(seq_id):
        assert seq_id != 1, 'A completed prefill has no outstanding promise to evaluate'
        return read(seq_id)
    monkeypatch.setattr(manager, 'chain_physical_residency', checked)
    assert coordinator._outstanding_active_reservations() == ((3, 5), 1)
    assert coordinator._outstanding_active_reservations(exclude_seq_ids={3}) == ((), 1)


def test_final_prefill_releases_peak_before_next_chain_admission(monkeypatch):
    # Old releases only at turn finish caused needless LRU eviction during
    # decode. Supply post-compaction storage; this tests lifecycle, not kernels.
    manager, coordinator, state = runtime()
    for name, seq_id, prompt_len in [("idle", 1, 4), ("active", 2, 20)]:
        plan = coordinator.plan_admission(chain_id=name, seq_id=seq_id,
                                          token_ids=[1] * prompt_len)
        state.chain_apply_admission(plan)
        allocate(manager, state, seq_id, 4)
        if name == "idle":
            coordinator.index.finish(name, token_ids=[1] * 4, processed_token_count=4,
                                     physical_slots_by_layer=(4, 4))
    seq = Sequence([1] * 20, SamplingParams(max_tokens=4))
    seq.seq_id, seq.chain_id = 2, "active"
    seq.num_prefilled_tokens, seq.current_chunk_size = 4, 4
    monkeypatch.setattr(manager, "on_forward_end", lambda seqs, prefill: None)
    state.on_forward_end([seq], True)
    assert coordinator._outstanding_active_reservations()[0] == (8, 8)
    assert manager.num_free_slots == 16

    seq.num_prefilled_tokens = 16
    state.on_forward_end([seq], True)
    assert coordinator._outstanding_active_reservations() == ((), 0)
    seq.num_prefilled_tokens = 20
    seq.append_token(2)
    assert state.reserve_decode_windows(deque([seq]), deque()) is None
    assert state.decode_reservations.outstanding() == {"layer_0": 1, "layer_1": 1}
    plan = coordinator.plan_admission(chain_id="next", seq_id=3, token_ids=[1] * 20)
    # 12 slots for the next prefill plus one decode slot fit within 16 free.
    assert plan.victim_chain_ids == ()
    assert plan.demote_chain_ids == ()
    assert coordinator.index.lookup("idle").state.value == "idle"


def test_final_inflight_decode_does_not_evict_idle_chain():
    # Reproduce through the real scheduler/reclaimer: A has submitted its final
    # output and B needs the one physically free slot. A owes no further KV.
    manager, coordinator, state = runtime()
    decoding = []
    for name, sid, length in [("A", 1, 4), ("B", 2, 4), ("idle", 3, 14)]:
        plan = coordinator.index.plan_admission(chain_id=name, seq_id=sid,
            token_ids=[1] * length, fingerprint=coordinator.fingerprint)
        coordinator.index.apply_admission(plan, fingerprint=coordinator.fingerprint)
        allocate(manager, state, sid, length)
        if name == "idle":
            coordinator.index.finish(name, token_ids=[1] * length,
                processed_token_count=length, physical_slots_by_layer=(length, length))
        else:
            seq = Sequence([1] * length, SamplingParams(max_tokens=2 if name == "A" else 8))
            seq.seq_id, seq.chain_id = sid, name
            seq.num_prefilled_tokens = length
            seq.append_token(2)
            decoding.append(seq)
    a, b = decoding
    assert state.decode_reservations.acquire(a)
    allocate(manager, state, a.seq_id, 1)
    a.num_pending_outputs = 1
    assert manager.num_free_slots == 1
    engine = object.__new__(LLMEngine)
    engine.scheduler = make_scheduler_with_oracle(PREFILL_POLICY_ALL_CHUNKED, state,
                                                  method="h2o", chunk=4, max_tokens=32)
    engine.scheduler.decoding = deque(decoding)
    engine.scheduler.decode_capacity_reclaimer = engine._reclaim_idle_chains_for_decode
    engine.model_runner = SimpleNamespace(runtime_state=state,
        call=lambda name, *args: getattr(state, name)(*args))
    selected, prefill, preempted = engine.scheduler.schedule()
    assert selected == [b] and not prefill and not preempted
    assert coordinator.index.stats()["chain_cache_evicted"] == 0
    assert coordinator.index.lookup("idle").physical_slots_by_layer == (14, 14)
    assert manager.num_free_slots == 1


def test_multi_pool_reclaim_uses_resource_units_not_layer_token_counts():
    """P1: packed residual capacity must not be credited from raw KV tokens."""
    from sparseengine.engine.chain_cache import ChainCacheIndex
    index = ChainCacheIndex()
    for name, sid, pools in (("raw", 1, (9, 0)), ("packed", 2, (0, 7))):
        plan = index.plan_admission(chain_id=name, seq_id=sid, token_ids=[1], fingerprint=b"pools")
        index.apply_admission(plan, fingerprint=b"pools")
        index.finish(name, token_ids=[1], processed_token_count=1,
                     physical_slots_by_layer=(100, 100), capacity_slots=pools)
    plan = index.plan_admission(chain_id="next", seq_id=3, token_ids=[1], fingerprint=b"pools",
                                required_slots_by_layer=(0, 6))
    assert plan.victim_chain_ids == ("raw", "packed")
    assert set(index.records) == {"raw", "packed"}, "Planning must remain read-only"


@pytest.mark.parametrize("kivi", [False, True])
def test_deltakv_partial_chain_reserves_raw_capacity_once(kivi):
    """P1: double charging partial chain promises causes needless decode preemption.

    Existing chain tests do not combine DeltaKV partial prefill and live decode
    at the raw-pool limit. GPU payloads are irrelevant to this CPU-owned ledger.
    """
    from sparseengine.engine.cache_manager.decode_reservation import DecodeReservations
    from sparseengine.engine.cache_manager.methods.deltakv_runtime import DeltaKVCacheManager

    manager = object.__new__(DeltaKVCacheManager)
    manager.config = SimpleNamespace(
        sink_keep_tokens=4, recent_keep_tokens=32, engine_prefill_chunk_size=4,
        full_layer_kv_quant_bits=4 if kivi else 0, enable_full_layer_kivi_quant=kivi,
        full_layer_kivi_group_size=32, full_layer_kivi_residual_length=32,
    )
    manager.head_dim = 128
    manager._num_free_slots_full = 50
    manager._num_free_slots_deltakv_full = 8
    manager._num_free_slots_deltakv_latent = 50
    manager._num_free_slots_full_layer_kivi = 50
    manager._deltakv_centers_capacity = 100
    manager._deltakv_centers_reserved_total = 0
    manager._deltakv_latent_reserved_by_seq = {}
    manager._full_layer_kivi_reserved_by_seq = {}
    manager.seq_id_to_row = {1: 0, 2: 1}
    manager.row_seq_lens = [20, 20]
    manager.row_deltakv_compressed_lens = [0, 0]
    manager.row_full_layer_kivi_quantized_lens = [4, 4]

    decoding = []
    for seq_id in (1, 2):
        seq = Sequence([1] * 20, SamplingParams(max_tokens=10))
        seq.seq_id, seq.num_prefilled_tokens = seq_id, 20
        seq.append_token(2)
        decoding.append(seq)
    partial = Sequence([1] * 104, SamplingParams(max_tokens=10))
    partial.seq_id, partial.num_prefilled_tokens = 3, 100
    partial.chain_id, partial.chain_status = "partial", "resumed"
    ordinary = Sequence([1] * 22, SamplingParams(max_tokens=10))
    ordinary.seq_id, ordinary.num_prefilled_tokens = 4, 20
    waiting = deque([partial, ordinary])

    state = object.__new__(RuntimeState)
    state.config, state.cache_manager = manager.config, manager
    state.prefix_cache_coordinator = None
    # Supply the existing logical promise; budget and decode acquisition remain
    # real implementations. Neither request has allocated this future raw KV.
    state.chain_cache_coordinator = SimpleNamespace(
        _outstanding_active_reservations=lambda **kwargs: ((4, 4, 0, 0), 0),
    )
    state.decode_reservations = DecodeReservations(manager, 1)

    # Four chain slots + two ordinary-prefill slots + two decode slots exactly
    # fit. A false rejection here reaches the IDLE reclaimer/preemption path.
    assert state.reserve_decode_windows(deque(decoding), waiting) is None
    assert state.decode_reservations.outstanding()["deltakv_raw"] == 2

    for seq in decoding:
        state.decode_reservations.release(seq.seq_id)
    manager._num_free_slots_deltakv_full -= 1
    # The ordinary partial request must still be charged: 4 + 2 + 2 > 7.
    assert state.reserve_decode_windows(deque(decoding), waiting) is decoding[1]


@pytest.fixture(params=[False, True], ids=["raw", "kivi"])
def deltakv_offload_manager(request):
    import torch
    from sparseengine.engine.cache_manager.methods.deltakv_runtime import DeltaKVCacheManager
    from test_prefill_schedule_policy import DeltaKVFullPrefillStagingTest

    original = DeltaKVFullPrefillStagingTest()._make_raw_deltakv_prefill_manager()
    manager = object.__new__(DeltaKVCacheManager)
    manager.__dict__.update(original.__dict__)
    manager.config.full_layer_kv_quant_bits = 4 if request.param else 0
    manager.config.enable_full_layer_kivi_quant = request.param
    manager.config.full_layer_kivi_group_size = 8
    manager.config.full_layer_kivi_residual_length = 1
    manager.config.max_model_len = 16
    manager.config.max_num_seqs_in_gpu = 2
    manager.config.hf_config = SimpleNamespace(dtype=torch.float32)
    manager.num_layers = manager.num_kv_layers = 2
    manager.head_dim = 8
    manager.runtime_layout = SimpleNamespace(kv_idx_to_layer_idx=(0, 1))
    manager.free_rows = deque([0, 1])
    for name in ("full_layer_slots_map", "sparse_layer_raw_slots_map", "sparse_layer_latent_slots_map"):
        setattr(manager, name, getattr(manager, name).repeat(2, 1))
    for name in ("row_seq_lens", "row_deltakv_compressed_lens", "row_deltakv_compressed_lens_gpu"):
        setattr(manager, name, getattr(manager, name).repeat(2))
    manager.row_deltakv_center_slots = [[None, None], [None, None]]
    for prefix in ("_deltakv_centers", "_full_layers", "_deltakv_latent", "_full_layer_kivi"):
        setattr(manager, prefix + "_reserved_by_seq", {})
        setattr(manager, prefix + "_reserved_total", 0)
    manager._deltakv_centers_capacity = 16
    manager._full_layer_kivi_full_prefill_plans = {}
    manager._full_layer_kivi_full_prefill_materialized_layers = set()
    manager.full_layer_slot_to_pos = torch.full((128,), -1, dtype=torch.int32)
    manager.full_layer_kivi_block_slots_map = torch.full((2, 16), -1, dtype=torch.int32)
    manager.full_layer_kivi_block_start_pos = torch.full((16,), -1, dtype=torch.int32)
    manager.free_slots_stack_full_layer_kivi = torch.arange(16, dtype=torch.int32)
    manager._num_free_slots_full_layer_kivi = 16 if request.param else 0
    manager.row_full_layer_kivi_quantized_lens = manager.row_seq_lens.copy()
    manager.row_full_layer_kivi_quantized_lens_gpu = torch.zeros(2, dtype=torch.int32)
    return manager


def test_deltakv_offload_cancel_returns_future_allocations(deltakv_offload_manager):
    """P1: cancellation must return every planned slot, not only computed KV.

    The model E2E cancellation prompts never cross the raw-offload threshold.
    Reuse its real CPU allocator/planning hooks and verify exact pool ownership.
    """
    manager = deltakv_offload_manager
    pools = ("full", "deltakv_full", "deltakv_latent", "full_layer_kivi")
    def free_slots():
        return {name: sorted(getattr(manager, "free_slots_stack_" + name)[
            :getattr(manager, "_num_free_slots_" + name)].tolist()) for name in pools}
    baseline = free_slots()
    for _ in range(3):
        seq = Sequence(list(range(12)))
        seq.chain_id = "cancel-me"
        seq.current_chunk_size = 2
        manager.prepare_step([seq], is_prefill=True)
        row = manager.seq_id_to_row[seq.seq_id]
        assert int(manager.row_seq_lens[row]) < seq.num_prompt_tokens
        manager.free_seq(seq.seq_id)
        manager.free_seq(seq.seq_id)
        assert free_slots() == baseline
        assert not manager.seq_id_to_row
        assert not manager._deltakv_full_prefill_plans
        assert not manager._full_layer_kivi_full_prefill_plans
        assert manager._deltakv_long_prefill_offload_row_idx is None


def test_deltakv_offload_admission_counts_future_allocations(deltakv_offload_manager):
    """P1: allocated suffix payloads must discharge promises before next admission."""
    manager = deltakv_offload_manager
    coordinator = ChainCacheCoordinator(manager.config, manager)
    seq = Sequence(list(range(12)))
    seq.chain_id = "long"
    seq.current_chunk_size = 2
    plan = coordinator.plan_admission(chain_id=seq.chain_id, seq_id=seq.seq_id, token_ids=seq.token_ids)
    coordinator.apply_admission(plan)
    before = manager.decode_window_budgets()
    manager.prepare_step([seq], is_prefill=True)
    after = manager.decode_window_budgets()
    allocated = tuple(before.get(key, 0) - after.get(key, 0) for key in manager.chain_capacity_keys())
    assert manager.chain_capacity_residency(seq.seq_id) == allocated
    remaining, _ = coordinator._outstanding_active_reservations()
    assert remaining == tuple(max(0, promised - used)
                              for promised, used in zip(plan.reserved_slots_by_layer, allocated))
    next_plan = coordinator.plan_admission(chain_id="one-token", seq_id=seq.seq_id + 1, token_ids=[1])
    assert not next_plan.victim_chain_ids
