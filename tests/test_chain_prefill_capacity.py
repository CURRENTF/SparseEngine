"""Protect admitted prompts from decode growth and stranded IDLE residency."""
from collections import deque
from types import SimpleNamespace

import pytest

from sparseengine.engine.chain_cache import ChainCacheCoordinator, ChainGoneError
from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams
from test_chain_offload import make_manager
from test_prefill_schedule_policy import make_scheduler_with_oracle, PREFILL_POLICY_ALL_CHUNKED


def runtime(capacity=32):
    manager = make_manager(rows=4, capacity=capacity)
    manager.config.decode_reservation_tokens = 1
    manager.config.engine_prefill_chunk_size = 4
    manager.config.observation_window_size = 2
    coordinator = ChainCacheCoordinator(manager.config, manager)
    state = RuntimeState(manager.config, manager, chain_cache_coordinator=coordinator)
    return manager, coordinator, state


def admit(state, name, length):
    seq = Sequence([1] * length, SamplingParams(max_tokens=8, ignore_eos=True))
    seq.chain_id = name
    plan = state.chain_admission_plan(name, seq.seq_id, seq.token_ids)
    state.chain_apply_admission(plan)
    seq.chain_status = plan.status
    return seq


def allocate(manager, state, seq, length):
    for layer in range(2):
        manager._allocate(layer, seq.seq_id, length)
    state._resident_seq_ids.add(seq.seq_id)


def scheduler(state):
    return make_scheduler_with_oracle(PREFILL_POLICY_ALL_CHUNKED, state,
                                     method="snapkv", chunk=4, max_tokens=32)


def test_decode_renewal_preserves_cold_chain_admission_and_prefill_progress():
    # Previously B reserved A's last slot before the first prefill chunk,
    # stranding A when B finished with retained chain KV.
    manager, coordinator, state = runtime()
    b = admit(state, "B", 6)
    allocate(manager, state, b, 6)
    b.num_prefilled_tokens = 6
    b.append_token(2)
    record = coordinator.index.lookup("B")
    record.reserved_slots_by_layer = ()
    record.reserved_rows = 0
    a = admit(state, "A", 26)
    sched = scheduler(state)
    sched.add(a)
    sched.decoding.append(b)
    assert state.reserve_decode_windows(sched.decoding, sched.waiting) is b
    assert not state.decode_reservations.requests
    selected, prefill, preempted = sched.schedule()
    assert selected == [a] and prefill and not preempted
    assert manager.num_free_slots == 26  # Scheduling itself allocates no KV.


def test_partial_chain_prefill_reservation_is_not_counted_twice():
    manager, coordinator, state = runtime(capacity=40)
    b = admit(state, "B", 6)
    allocate(manager, state, b, 6)
    b.num_prefilled_tokens = 6
    b.append_token(2)
    record = coordinator.index.lookup("B")
    record.reserved_slots_by_layer = ()
    record.reserved_rows = 0
    a = admit(state, "A", 26)
    allocate(manager, state, a, 4)
    a.num_prefilled_tokens = 4
    assert manager.num_free_slots == 30
    assert coordinator._outstanding_active_reservations()[0] == (22, 22)
    assert state.reserve_decode_windows(deque([b]), deque([a])) is None
    assert state.decode_reservations.outstanding() == {"layer_0": 1, "layer_1": 1}


def stranded_rank(*, full=False):
    manager, coordinator, state = runtime()
    for name, seq_id, length in [("old", 10, 29 if full else 7), ("keep", 11, 3)]:
        seq = Sequence([1] * length)
        seq.seq_id = seq_id
        plan = coordinator.plan_admission(chain_id=name, seq_id=seq_id, token_ids=seq.token_ids)
        state.chain_apply_admission(plan)
        allocate(manager, state, seq, length)
        coordinator.index.finish(name, token_ids=seq.token_ids,
                                 processed_token_count=length,
                                 physical_slots_by_layer=(length, length))
    # A has lost physical residency after preemption, while completed peers
    # remain IDLE. Recompute does not run chain admission again.
    a = Sequence([2] * 26, SamplingParams(max_tokens=8, ignore_eos=True))
    a.seq_id = 100
    a.chain_id = "A"
    a.num_prefilled_tokens = 26
    a.append_token(3)
    a.start_recompute_replay()
    plan = coordinator.index.plan_admission(chain_id="A", seq_id=a.seq_id,
        token_ids=list(a.prompt_token_ids), fingerprint=coordinator.fingerprint)
    coordinator.index.apply_admission(plan, fingerprint=coordinator.fingerprint)
    return manager, coordinator, state, a


def engine(ranks):
    state = ranks[0][2]
    calls = []
    runners = []
    for rank in ranks:
        runner = object.__new__(ModelRunner)
        runner.runtime_state = rank[2]
        runners.append(runner)
    def call(method, *args):
        calls.append((method, args))
        results = [getattr(runner, method)(*args) for runner in runners]
        assert all(result == results[0] for result in results)
        return results[0]
    result = object.__new__(LLMEngine)
    result.model_runner = SimpleNamespace(runtime_state=state, call=call)
    result.scheduler = scheduler(state)
    result.scheduler.prefill_capacity_reclaimer = result._reclaim_idle_chains_for_prefill
    result.scheduler.add(ranks[0][3])
    return result, calls


@pytest.mark.parametrize("full", [False, True])
def test_replay_reclaims_only_needed_idle_chain_on_every_rank_then_prefills(full):
    # A full pool skips the admission scan; reclaim must still run.
    ranks = [stranded_rank(full=full), stranded_rank(full=full)]
    instance, calls = engine(ranks)
    selected, prefill, preempted = instance.scheduler.schedule()
    assert selected == [ranks[0][3]] and prefill and not preempted
    assert calls == [("chain_reclaim_idle", ("old", 10, False))]
    for manager, coordinator, state, a in ranks:
        with pytest.raises(ChainGoneError):
            coordinator.index.lookup("old")
        assert coordinator.index.lookup("keep").physical_slots_by_layer == (3, 3)
        assert coordinator.index.lookup("A").state.value == "active"
        assert manager.num_free_slots == 29
        assert 10 not in state._resident_seq_ids and 11 in state._resident_seq_ids


@pytest.mark.parametrize("full", [False, True])
def test_prefill_reclaim_drains_async_users_before_mutating_ownership(full):
    from sparseengine.engine.async_scheduling.execution import AsyncDrainRequired
    ranks = [stranded_rank(full=full)]
    instance, calls = engine(ranks)
    instance.scheduler._async_inflight = 1
    with pytest.raises(AsyncDrainRequired):
        instance.scheduler.schedule()
    assert not calls
    assert ranks[0][0].num_free_slots == (0 if full else 22)
    assert list(instance.scheduler.waiting) == [ranks[0][3]]
    instance.scheduler._async_inflight = 0
    selected, prefill, preempted = instance.scheduler.schedule()
    assert selected == [ranks[0][3]] and prefill and not preempted


def test_no_idle_victim_keeps_explicit_failure_and_active_ownership():
    ranks = [stranded_rank()]
    instance, calls = engine(ranks)
    coordinator = ranks[0][1]
    # An ACTIVE owner must never be discarded just to satisfy another prompt.
    from sparseengine.engine.chain_cache import ChainState
    for name in ("old", "keep"):
        coordinator.index._set_record_state(coordinator.index.lookup(name), ChainState.ACTIVE)
    with pytest.raises(RuntimeError, match="All prompt admissions were deferred"):
        instance.scheduler.schedule()
    assert not calls
    assert ranks[0][0].num_free_slots == 22
    assert list(instance.scheduler.waiting) == [ranks[0][3]]
