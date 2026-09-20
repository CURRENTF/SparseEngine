"""CPU lifecycle regressions for unnecessary H2O chain eviction."""
from collections import deque
from types import SimpleNamespace

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
