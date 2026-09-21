"""Execution ownership at control boundaries, without a GPU or another ledger."""
from __future__ import annotations

from types import SimpleNamespace
import pytest


def execution():
    from sparseengine.engine.async_scheduling.execution import AsyncExecution
    value = object.__new__(AsyncExecution)
    value.results = {}
    value.last_tokens = {}
    value.penalties = {}
    return value


def test_completed_but_unretired_result_still_owns_cache():
    from sparseengine.engine.async_scheduling.execution import AsyncDrainRequired
    value = execution()
    # Event completion is intentionally not used as permission to release.
    value.results[9] = SimpleNamespace(
        seqs=[SimpleNamespace(seq_id=71)], event=SimpleNamespace(query=lambda: True),
    )
    with pytest.raises(AsyncDrainRequired):
        value.assert_releasable((71,))
    value.assert_releasable((82,))
    value.results.pop(9)
    value.assert_releasable((71,))


def test_release_checks_all_interleaved_results_without_waiting():
    from sparseengine.engine.async_scheduling.execution import AsyncDrainRequired
    value = execution()
    class NoWaitEvent:
        def query(self):
            raise AssertionError('release guard must not poll GPU events')
        def synchronize(self):
            raise AssertionError('release guard must not synchronize')
    for ticket, owners in enumerate(((71, 82), (82, 93), (71,))):
        value.results[ticket] = SimpleNamespace(
            seqs=[SimpleNamespace(seq_id=sid) for sid in owners], event=NoWaitEvent(),
        )
    value.results.pop(0)
    with pytest.raises(AsyncDrainRequired):
        value.assert_releasable((71,))
    value.results.pop(2)
    value.assert_releasable((71,))
    with pytest.raises(AsyncDrainRequired):
        value.assert_releasable((93, 104))


def test_runner_rejects_release_during_capture_before_any_free(monkeypatch):
    from sparseengine.engine.model_runner import ModelRunner
    from sparseengine.platforms import device_runtime
    runner = object.__new__(ModelRunner)
    monkeypatch.setattr(device_runtime, 'is_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='capture'):
        runner._assert_cache_release((71,))


def test_terminal_cleanup_attempts_every_owner_after_post_detach_failure():
    from sparseengine.engine.runtime_state import RuntimeState
    runtime = object.__new__(RuntimeState)
    calls = []
    def detached_then_failed(sid):
        calls.append(('cache', sid))
        raise RuntimeError('host offload submission failed after detach')
    runtime.decode_reservations = SimpleNamespace(release=lambda sid: calls.append(('window', sid)))
    runtime.cache_manager = SimpleNamespace(free_seq=detached_then_failed)
    runtime.prefix_cache_coordinator = SimpleNamespace(release_seq=lambda sid: calls.append(('prefix', sid)))
    runtime.recurrent_state_manager = SimpleNamespace(free_seq=lambda sid: calls.append(('recurrent', sid)))
    runtime._resident_seq_ids = {71}
    with pytest.raises(RuntimeError, match='host offload'):
        runtime.free_seq(71)
    assert calls == [('window', 71), ('cache', 71), ('prefix', 71), ('recurrent', 71)]
    assert not runtime._resident_seq_ids


@pytest.mark.parametrize('window', [0, -1, True, 1.5])
def test_invalid_reservation_window_rejected_before_scheduling(window):
    from sparseengine.engine.cache_manager.decode_reservation import DecodeReservations
    with pytest.raises(ValueError):
        DecodeReservations(object(), window)


def test_partial_acquisition_is_explicit_and_cancel_releases_only_its_window():
    from sparseengine.engine.cache_manager.decode_reservation import DecodeReservations
    from sparseengine.engine.sequence import Sequence
    from sparseengine.sampling_params import SamplingParams
    from cache_contracts.cases import CHAIN_CASES, make_chain, allocate_chain
    from cache_contracts.ownership import ChainHarness
    m = make_chain(next(c for c in CHAIN_CASES if c.method == 'snapkv'), capacity=7)
    a = Sequence([1, 2, 3], SamplingParams(max_tokens=8))
    b = Sequence([4, 5, 6], SamplingParams(max_tokens=8))
    for seq in (a, b):
        allocate_chain(m, CHAIN_CASES[1], seq.seq_id, (3, 3))
    windows = DecodeReservations(m, 1)
    assert windows.acquire_many((a, b)) is b
    assert set(windows.requests) == {a.seq_id}
    before = ChainHarness(m).observe()
    # A promise is not a physical allocation, and releasing it frees no pages.
    windows.release(a.seq_id)
    windows.release(a.seq_id)
    assert ChainHarness(m).observe() == before
    assert windows.acquire(b)
    windows.release(b.seq_id)
    for seq in (a, b):
        m.free_seq(seq.seq_id)
    ChainHarness(m).observe()
