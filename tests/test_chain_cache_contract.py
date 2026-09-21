"""One behavioral contract for every supported Chain configuration."""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from cache_contracts.cases import CHAIN_CASES, RADIX_CASES, make_chain, allocate_chain
from cache_contracts.ownership import ChainHarness, check_cost_contract

# Reuse the repository's transport-only CPU fixture. Allocation and state hooks
# remain the real implementations for the selected method.
from test_chain_offload import cpu_transfers


@pytest.mark.parametrize('case', CHAIN_CASES, ids=lambda c: c.id)
def test_chain_binding_and_exact_physical_conservation(case):
    from sparseengine.engine.cache_manager.base import CacheManager
    from sparseengine.engine.cache_manager.chain_contract import validate_chain_manager
    m = make_chain(case)
    validate_chain_manager(m, CacheManager, offload=not case.shared)
    h = ChainHarness(m, shared_slots=case.shared)
    original = h.observe()
    allocate_chain(m, case, 71, (3, 3) if case.shared else (3, 5))
    h.observe()
    allocate_chain(m, case, 82, (2, 2))
    h.observe()
    m.free_seq(71)
    h.assert_released(71)
    m.free_seq(71)
    m.free_seq(82)
    h.assert_released(82)
    assert [set(p.free) for p in h.observe()] == [set(p.free) for p in original]


@pytest.mark.parametrize('case', CHAIN_CASES, ids=lambda c: c.id)
def test_chain_index_exclusive_owner_and_speculative_plan(case):
    from sparseengine.engine.chain_cache import ChainCacheIndex, ChainBusyError, ChainOwnerMismatchError
    m = make_chain(case)
    index = ChainCacheIndex()
    fp = b'contract-fixture'
    before = pickle.dumps(index)
    plan = index.plan_admission(chain_id='turn', seq_id=71,
                                token_ids=[1, 2, 3], fingerprint=fp)
    assert pickle.dumps(index) == before
    index.apply_admission(plan, fingerprint=fp)
    with pytest.raises((ChainBusyError, ChainOwnerMismatchError)):
        index.plan_admission(chain_id='turn', seq_id=82,
                             token_ids=[1, 2, 3, 4], fingerprint=fp)
    lengths = (3, 3) if case.shared else (2, 3)
    allocate_chain(m, case, 71, lengths)
    index.finish('turn', token_ids=[1, 2, 3, 4], processed_token_count=3,
                 physical_slots_by_layer=lengths)
    h = ChainHarness(m, shared_slots=case.shared)
    h.check(SimpleNamespace(index=index, offload=None))
    resumed = index.plan_admission(chain_id='turn', seq_id=71,
                                   token_ids=[1, 2, 3, 4, 5], fingerprint=fp)
    assert resumed.reused_tokens == 3  # Not the physical row length or last output.
    index.apply_admission(resumed, fingerprint=fp)
    h.check(SimpleNamespace(index=index, offload=None))
    index.invalidate('turn')
    m.free_seq(71)
    h.assert_released(71)


@pytest.mark.parametrize('case', [c for c in CHAIN_CASES if not c.shared], ids=lambda c: c.id)
@pytest.mark.parametrize('bad', [(-1, 1), (1,), (1, 33), (1, 17), (True, 1)])
def test_restore_preflight_does_not_claim_any_layer(case, bad):
    m = make_chain(case)
    h = ChainHarness(m)
    before = h.observe()
    with pytest.raises((ValueError, RuntimeError)):
        m.allocate_chain_restore(71, bad)
    assert h.observe() == before


@pytest.mark.parametrize('case', [c for c in CHAIN_CASES if not c.shared], ids=lambda c: c.id)
def test_restore_partial_allocation_failure_then_retry(case, monkeypatch):
    m = make_chain(case)
    original = m._allocate
    def fail_second(layer, sid, size):
        if layer == 1:
            raise RuntimeError('injected second-layer allocation failure')
        return original(layer, sid, size)
    with monkeypatch.context() as patch:
        patch.setattr(m, '_allocate', fail_second)
        with pytest.raises(RuntimeError, match='injected'):
            m.allocate_chain_restore(71, (3, 5))
    h = ChainHarness(m)
    h.assert_released(71)
    allocated = m.allocate_chain_restore(71, (3, 5))
    assert tuple(len(allocated[i]) for i in (0, 1)) == (3, 5)
    h.observe()
    m.free_seq(71)
    m.free_seq(71)
    h.assert_released(71)


@pytest.mark.parametrize('case', [c for c in CHAIN_CASES if not c.shared], ids=lambda c: c.id)
def test_chain_slot_and_method_state_survive_relocated_restore(case, cpu_transfers):
    from test_chain_offload import round_trip
    m = make_chain(case)
    round_trip(m)
    ChainHarness(m).observe()


@pytest.mark.parametrize('case', [c for c in CHAIN_CASES if not c.shared], ids=lambda c: c.id)
def test_host_census_and_no_early_snapshot_publication(case, cpu_transfers):
    from test_chain_offload import populate
    from sparseengine.engine.cache_manager.chain_offload import ChainOffloadController
    m = make_chain(case)
    populate(m)
    controller = ChainOffloadController(m, 16384)
    controller.save(1, m.snapshot_chain_method_state(1))
    ChainHarness.check_host(controller)
    assert not controller.snapshots[1].valid
    controller.wait(1)
    ChainHarness.check_host(controller)
    controller.drop(1)
    ChainHarness.check_host(controller)
    m.free_seq(1)
    ChainHarness(m).assert_released(1)


@pytest.mark.parametrize('case', CHAIN_CASES, ids=lambda c: c.id)
def test_small_decode_promise_bounds_observed_allocation_peak(case):
    from sparseengine.engine.sequence import Sequence
    from sparseengine.sampling_params import SamplingParams
    m = make_chain(case)
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=8))
    allocate_chain(m, case, seq.seq_id, (3, 3))
    check_cost_contract(m, seq, (1, 2))
    advertised = m.decode_window_costs(seq, 2)
    h = ChainHarness(m, shared_slots=case.shared)
    initial_free = [len(p.free) for p in h.observe()]
    # This trace stays below all compression thresholds: no selection algorithm
    # is substituted. The high-water mark is measured from real slot identities.
    for _ in range(2):
        allocate_chain(m, case, seq.seq_id, (1, 1))
        pools = h.observe()
        for i, pool in enumerate(pools):
            name = 'slots' if case.shared else f'layer_{i}'
            assert initial_free[i] - len(pool.free) <= advertised[name]
    m.free_seq(seq.seq_id)
    h.assert_released(seq.seq_id)


def test_matrix_covers_every_supported_mode_without_cross_mode_unification():
    from sparseengine.engine.chain_cache import (
        CHAIN_PREFIX_METHODS, RADIX_PREFIX_METHODS, normalize_prefix_cache_mode,
    )
    from sparseengine.method_registry import PREFIX_CACHE_SUPPORTED_METHODS
    assert {c.method for c in CHAIN_CASES if not c.shared} == set(CHAIN_PREFIX_METHODS)
    assert set(RADIX_CASES) == set(RADIX_PREFIX_METHODS)
    assert set(PREFIX_CACHE_SUPPORTED_METHODS) == set(RADIX_CASES) | set(CHAIN_PREFIX_METHODS)
    for case in CHAIN_CASES:
        assert normalize_prefix_cache_mode('auto', enabled=True, method=case.method,
                                           prefill_method=case.prefill) == 'chain'


def test_terminal_cleanup_releases_nonresident_promise(cpu_transfers):
    from test_chain_offload import make_runtime
    m, coordinator, runtime = make_runtime(cpu_transfers)
    plan = coordinator.plan_admission(chain_id='queued', seq_id=71, token_ids=[1, 2])
    runtime.chain_apply_admission(plan)
    # Exercise actual terminal release; the reservation's contents are opaque
    # here because this path must remove it regardless of any physical row.
    runtime.decode_reservations.requests[71] = object()
    runtime.chain_invalidate('queued', expected_seq_id=71)
    ChainHarness(m).assert_released(71, runtime)


def test_wrong_owner_cannot_run_method_finalization(cpu_transfers, monkeypatch):
    from test_chain_offload import make_runtime, populate
    from sparseengine.engine.chain_cache import ChainOwnerMismatchError
    m, coordinator, runtime = make_runtime(cpu_transfers)
    plan = coordinator.plan_admission(chain_id='owner', seq_id=71, token_ids=[1, 2, 3])
    runtime.chain_apply_admission(plan)
    populate(m, 71)
    called = []
    monkeypatch.setattr(m, 'on_chain_turn_finished', lambda *a: called.append(a))
    before = ChainHarness(m).observe()
    with pytest.raises(ChainOwnerMismatchError):
        runtime.chain_finish('owner', 82, b'wrong', 3)
    assert not called
    assert ChainHarness(m).observe() == before
    runtime.chain_invalidate('owner', expected_seq_id=71)


def test_method_finalization_failure_invalidates_whole_turn(cpu_transfers, monkeypatch):
    from test_chain_offload import make_runtime, populate
    m, coordinator, runtime = make_runtime(cpu_transfers)
    plan = coordinator.plan_admission(chain_id='failed', seq_id=71, token_ids=[1, 2, 3])
    runtime.chain_apply_admission(plan)
    populate(m, 71)
    def fail(*args):
        raise RuntimeError('injected method finalization failure')
    monkeypatch.setattr(m, 'on_chain_turn_finished', fail)
    with pytest.raises(RuntimeError, match='injected'):
        runtime.chain_finish('failed', 71, b'not-published', 3)
    assert 'failed' not in coordinator.index.records
    ChainHarness(m).assert_released(71, runtime)


def test_pyramid_independent_pool_boundary_is_not_hidden_by_first_layer(cpu_transfers):
    import torch
    from sparseengine.engine.cache_manager.chain_offload import ChainOffloadController
    case = next(c for c in CHAIN_CASES if c.method == 'pyramidkv')
    m = make_chain(case)
    original_storage = m.kv_cache
    capacities = (16, 8)
    m.kv_cache = [(original_storage[0, layer, :n].clone(),
                   original_storage[1, layer, :n].clone())
                  for layer, n in enumerate(capacities)]
    m.free_slots_stack = [torch.arange(n, dtype=torch.int32) for n in capacities]
    m._num_free_slots = list(capacities)
    m.config.num_kvcache_slots = list(capacities)
    h = ChainHarness(m)
    before = h.observe()
    with pytest.raises(RuntimeError):
        m.allocate_chain_restore(71, (2, 9))
    assert h.observe() == before
    m.allocate_chain_restore(71, (3, 5))
    h.observe()
    controller = ChainOffloadController(m, 16384)
    controller.save(71, m.snapshot_chain_method_state(71))
    controller.wait(71)
    saved = {layer: tuple(t.clone() for t in pair)
             for layer, pair in controller.snapshots[71].kv.items()}
    m.free_seq(71)
    m.allocate_chain_restore(82, (1, 1))
    controller.restore(71)
    h.observe()
    for layer, expected in saved.items():
        slots = m.chain_token_slots(layer, 71).long()
        for live, reference in zip(m.chain_storage_tensors(layer), expected):
            torch.testing.assert_close(live[slots], reference, rtol=0, atol=0)
    controller.reset()
    m.free_seq(71)
    m.free_seq(82)
    assert [len(pool.free) for pool in h.observe()] == list(capacities)
