"""Regressions for reclaiming retained KV before preempting active decode."""
from collections import deque
from types import SimpleNamespace

import pytest

from sparseengine.engine.chain_cache import ChainBusyError, ChainCacheCoordinator, ChainGoneError, ChainOwnerMismatchError
from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams
from test_chain_offload import make_manager
from test_prefill_schedule_policy import FakeMemoryOracle, make_scheduler_with_oracle, PREFILL_POLICY_ALL_CHUNKED


def make_rank(window=6):
    manager = make_manager(rows=4, capacity=32)
    manager.config.decode_reservation_tokens = window
    manager.config.engine_prefill_chunk_size = 4
    coordinator = ChainCacheCoordinator(manager.config, manager)
    runtime = RuntimeState(manager.config, manager, chain_cache_coordinator=coordinator)
    sequences = []
    for seq_id, lengths, idle in [(1, (3, 5), True), (2, (4, 4), True),
                                  (10, (7, 7), False), (11, (7, 7), False)]:
        tokens = [1] * 7
        plan = coordinator.index.plan_admission(chain_id=str(seq_id), seq_id=seq_id,
            token_ids=tokens, fingerprint=coordinator.fingerprint)
        coordinator.index.apply_admission(plan, fingerprint=coordinator.fingerprint)
        for layer, length in enumerate(lengths):
            manager._allocate(layer, seq_id, length)
        runtime._resident_seq_ids.add(seq_id)
        if idle:
            coordinator.index.finish(str(seq_id), token_ids=tokens,
                processed_token_count=7, physical_slots_by_layer=lengths)
        else:
            seq = Sequence(tokens, SamplingParams(max_tokens=32, ignore_eos=True))
            seq.seq_id = seq_id
            seq.chain_id = str(seq_id)
            seq.num_prefilled_tokens = len(tokens)
            seq.append_token(9)
            sequences.append(seq)
    return manager, coordinator, runtime, sequences


def engine_with_ranks(ranks):
    driver = ranks[0][2]
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
    engine = object.__new__(LLMEngine)
    engine.model_runner = SimpleNamespace(runtime_state=driver, call=call)
    oracle = FakeMemoryOracle()
    oracle.reserve_decode_windows = driver.reserve_decode_windows
    engine.scheduler = make_scheduler_with_oracle(PREFILL_POLICY_ALL_CHUNKED, oracle,
        method="snapkv", chunk=4, max_tokens=64)
    engine.scheduler.decoding = deque(ranks[0][3])
    engine.scheduler.decode_capacity_reclaimer = engine._reclaim_idle_chains_for_decode
    return engine, calls


def assert_partition(manager):
    for layer in range(2):
        free = manager.free_slots_stack[layer][:manager._num_free_slots[layer]].tolist()
        live = []
        for row in manager.seq_id_to_row[layer].values():
            length = int(manager.row_seq_lens[layer][row])
            live.extend(manager.buffer_req_to_token_slots[layer][row, :length].tolist())
        assert sorted(free + live) == list(range(32))


def test_scheduler_reclaims_only_needed_idle_owner_before_preemption_on_both_ranks():
    ranks = [make_rank(), make_rank()]
    engine, calls = engine_with_ranks(ranks)
    selected, prefill, preempted = engine.scheduler.schedule()
    assert not prefill and not preempted
    assert {seq.seq_id for seq in selected} == {10, 11}
    assert engine.scheduler.total_preemptions == 0
    assert calls == [("chain_reclaim_idle", ("1", 1, False))]
    for manager, coordinator, runtime, _ in ranks:
        assert manager._num_free_slots == [14, 14]
        assert runtime._resident_seq_ids == {2, 10, 11}
        with pytest.raises(ChainGoneError):
            coordinator.index.lookup("1")
        assert coordinator.index.lookup("2").resident_rows == 1
        assert_partition(manager)


def test_insufficient_idle_capacity_still_preempts_after_bounded_reclaim():
    ranks = [make_rank(window=12)]
    engine, calls = engine_with_ranks(ranks)
    selected, _, preempted = engine.scheduler.schedule()
    assert not selected and len(preempted) == 1
    assert [args[0] for _, args in calls] == ["1", "2"]
    assert engine.scheduler.total_preemptions == 1
    assert_partition(ranks[0][0])


def test_sufficient_decode_capacity_does_not_evict_idle_cache():
    ranks = [make_rank(window=2)]
    engine, calls = engine_with_ranks(ranks)
    selected, _, preempted = engine.scheduler.schedule()
    assert len(selected) == 2 and not preempted and not calls
    assert_partition(ranks[0][0])


@pytest.mark.parametrize("chain_id,seq_id,error", [("10", 10, ChainBusyError),
                                                  ("1", 99, ChainOwnerMismatchError)])
def test_reclaim_rejects_active_or_stale_owner_without_freeing(chain_id, seq_id, error):
    manager, coordinator, runtime, _ = make_rank()
    before = list(manager._num_free_slots)
    with pytest.raises(error):
        runtime.chain_reclaim_idle(chain_id, seq_id, False)
    assert manager._num_free_slots == before
    assert len(coordinator.index.records) == 4
    assert_partition(manager)
