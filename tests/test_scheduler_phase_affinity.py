"""Phase policy regressions use a fake clock and independent request histories."""
import pytest

from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams
from test_prefill_schedule_policy import FakeMemoryOracle, make_scheduler


def request(n=4):
    return Sequence([1] * n, SamplingParams(max_tokens=100, ignore_eos=True))


def setup(monkeypatch, *, threshold=2, oracle=None):
    clock = [0.]
    monkeypatch.setattr("sparseengine.engine.scheduler.time.monotonic", lambda: clock[0])
    scheduler = make_scheduler("all_chunked", oracle=oracle or FakeMemoryOracle(),
                               chunk=4, max_tokens=8)
    scheduler.favor_min_decoding_seqs = threshold
    return scheduler, clock


def decode(scheduler, count=2):
    seqs = [request() for _ in range(count)]
    for seq in seqs:
        seq.num_prefilled_tokens = seq.num_prompt_tokens
        seq.append_token(7)
        scheduler.decoding.append(seq)
    scheduler.schedule()  # no waiting work starts decode even below threshold
    return seqs


def test_decode_holds_then_switches_below_threshold(monkeypatch):
    s, _ = setup(monkeypatch)
    active = decode(s)
    fresh = request()
    s.add(fresh)
    for _ in range(3):
        batch, prefill, _ = s.schedule()
        assert batch == active and not prefill
    s.abort(active[0].seq_id)
    assert s.schedule()[:2] == ([fresh], True)


def test_decode_affinity_skips_prefill_queries_but_timeout_restores_them(monkeypatch):
    # Existing phase tests checked the selected batch but missed the expensive
    # prefill-capacity traversal performed before returning a decode batch.
    from unittest.mock import Mock

    oracle = FakeMemoryOracle()
    s, clock = setup(monkeypatch, oracle=oracle)
    active = decode(s)
    fresh = request()
    s.add(fresh)
    oracle.prompt_admission_free_slots = Mock(wraps=oracle.prompt_admission_free_slots)
    oracle.prefill_step_free_slots = Mock(wraps=oracle.prefill_step_free_slots)
    oracle.prompt_admission_budgets = Mock(wraps=oracle.prompt_admission_budgets)
    for _ in range(3):
        assert s.schedule()[:2] == (active, False)
    oracle.prompt_admission_free_slots.assert_not_called()
    oracle.prefill_step_free_slots.assert_not_called()
    oracle.prompt_admission_budgets.assert_not_called()
    clock[0] = 60
    assert s.schedule()[:2] == ([fresh], True)
    assert oracle.prompt_admission_free_slots.called
    assert oracle.prefill_step_free_slots.called
    assert oracle.prompt_admission_budgets.called


def test_timeout_survives_scans_and_partial_gets_new_wait(monkeypatch):
    s, clock = setup(monkeypatch)
    decode(s)
    fresh = request(12)
    s.add(fresh)
    for now in (20, 59.9):
        clock[0] = now
        assert not s.schedule()[1]
    clock[0] = 60
    batch, prefill, _ = s.schedule()
    assert batch == [fresh] and prefill
    assert s.last_phase_decision["reason"] == "prefill_wait_timeout"
    assert s.last_phase_decision["prefill_waits"][0]["seconds"] == 60
    clock[0] = 62
    s.postprocess(batch, [7], prefill)
    assert s._prefill_wait_since[fresh.seq_id] == 62
    assert s.schedule()[1]  # high decode count cannot interrupt remaining chunks


def test_saturated_decode_skips_prefix_rpc_until_admission_is_possible(monkeypatch):
    # Aged queued prompts formerly issued TP prefix RPCs every decode step even
    # though the decode limit made every fresh prompt ineligible.
    from unittest.mock import Mock

    s, clock = setup(monkeypatch)
    active = decode(s)
    s.max_decoding_seqs = len(active)
    fresh = request()
    s.add(fresh)
    refresh = Mock(wraps=s.prefix_cache_hit_refresher)
    s.prefix_cache_hit_refresher = refresh
    clock[0] = 61
    for _ in range(3):
        assert s.schedule()[:2] == (active, False)
    refresh.assert_not_called()
    assert list(s.waiting) == [fresh]
    assert s._prefill_wait_since[fresh.seq_id] == 0
    s.abort(active[0].seq_id)
    assert s.schedule()[:2] == ([fresh], True)
    refresh.assert_called_once_with(fresh)


def test_prefill_diagnostic_counts_do_not_refresh_or_mutate_hits(monkeypatch):
    # LLMEngine calls these counters after every step; telemetry must not issue
    # distributed cache operations or alter future admission state.
    from unittest.mock import Mock

    oracle = FakeMemoryOracle(prefix_hit_len=2)
    s, _ = setup(monkeypatch, oracle=oracle)
    fresh = request()
    s.add(fresh)
    refresh = Mock(wraps=s.prefix_cache_hit_refresher)
    s.prefix_cache_hit_refresher = refresh
    before = fresh.prefix_cache_hit_len
    counts = s.prefill_execution_mode_counts()
    assert sum(counts.values()) == 1
    refresh.assert_not_called()
    assert fresh.prefix_cache_hit_len == before


def test_prefill_accepts_arrivals_and_partial_at_admission_limit(monkeypatch):
    s, _ = setup(monkeypatch, threshold=1)
    partial = request(12)
    s.add(partial)
    batch, pf, _ = s.schedule()
    s.postprocess(batch, [7], pf)
    fresh = request()
    s.add(fresh)
    batch, pf, _ = s.schedule()
    assert pf and set(batch) == {partial, fresh}
    s.postprocess(batch, [7] * len(batch), pf)
    s.max_decoding_seqs = 1
    blocked = request()
    s.add(blocked)
    assert s.schedule()[:2] == ([partial], True)
    assert blocked in s.waiting


def test_count_uses_executable_group_and_capacity(monkeypatch):
    s, _ = setup(monkeypatch)
    active = decode(s, 3)
    active[0].append_token(3)
    fresh = request()
    s.add(fresh)
    assert s.schedule()[:2] == (active, False)

    oracle = FakeMemoryOracle()
    s, _ = setup(monkeypatch, oracle=oracle)
    active = decode(s, 3)
    oracle.decode_step_free_slots_for = lambda seq: 1 if seq is active[0] else 0
    fresh = request()
    s.add(fresh)
    assert s.schedule()[:2] == ([fresh], True)


def test_unavailable_prefill_falls_back_without_resetting_age(monkeypatch):
    oracle = FakeMemoryOracle(step_free_slots=0)
    s, clock = setup(monkeypatch, oracle=oracle)
    active = decode(s, 1)
    fresh = request()
    s.add(fresh)
    for now in (10, 60, 70):
        clock[0] = now
        assert s.schedule()[:2] == (active, False)
        assert s._prefill_wait_since[fresh.seq_id] == 0
    oracle._step_free_slots = 100
    assert s.schedule()[:2] == ([fresh], True)


def test_cancel_removes_timeout_and_hook_failure_retains_owner(monkeypatch):
    s, clock = setup(monkeypatch)
    active = decode(s)
    fresh = request()
    s.add(fresh)
    clock[0] = 61
    s.abort(fresh.seq_id)
    later = request()
    s.add(later)
    assert s.schedule()[:2] == (active, False)
    assert fresh.seq_id not in s._prefill_wait_since
    s.phase = "prefill"
    def fail(seq):
        raise RuntimeError("admission hook failed")
    s.memory_oracle.prompt_admission_costs = fail
    with pytest.raises(RuntimeError, match="admission hook failed"):
        s.schedule()
    assert later in s.waiting
    s.abort(later.seq_id)
    assert not s._prefill_wait_since


def test_disabled_policy_keeps_prefill_priority(monkeypatch):
    s, _ = setup(monkeypatch, threshold=0)
    decode(s)
    fresh = request()
    s.add(fresh)
    assert s.schedule()[:2] == ([fresh], True)


@pytest.mark.parametrize("favor", [-1, 9, True, 1.5, "4"])
def test_threshold_rejects_invalid_public_input(favor):
    from types import SimpleNamespace
    from sparseengine.configs.scheduling import normalize_scheduling
    config = SimpleNamespace(decode_reservation_tokens=1, max_num_seqs_in_batch=8,
                             max_decoding_seqs=8, favor_min_decoding_seqs=favor)
    with pytest.raises(ValueError, match="favor_min_decoding_seqs"):
        normalize_scheduling(config)


def test_cli_threshold_drives_phase_policy(monkeypatch):
    from sparseengine.entrypoints.openai.api_server import _parse_engine_kwargs
    kwargs = _parse_engine_kwargs(["--favor-min-decoding-seqs", "2"])
    s, _ = setup(monkeypatch, threshold=kwargs["favor_min_decoding_seqs"])
    active = decode(s)
    s.add(request())
    assert s.schedule()[:2] == (active, False)


def test_replay_recovery_still_drains_surviving_decode(monkeypatch):
    s, clock = setup(monkeypatch)
    active = decode(s, 1)
    victim = request()
    victim.num_prefilled_tokens = 4
    victim.append_token(7)
    victim.start_recompute_replay()
    s.add(victim)
    clock[0] = 65
    assert s.schedule()[:2] == (active, False)
    s.abort(active[0].seq_id)
    assert s.schedule()[:2] == ([victim], True)


def test_phase_metrics_include_last_run_and_actual_batches():
    from benchmark.efficiency.metrics import scheduler_phase_metrics
    result = scheduler_phase_metrics([
        {"elapsed_s": 1., "phase": "prefill", "reason": "prefill_continue",
         "decode_batch": 0, "prefill_waits": [{"partial": False, "seconds": .5}]},
        {"elapsed_s": 3., "phase": "decode", "reason": "no_prefill",
         "decode_batch": 3, "prefill_waits": []},
        {"elapsed_s": 6., "phase": "decode", "reason": "decode_affinity",
         "decode_batch": 1, "prefill_waits": []},
    ])
    assert result["phase_switches"] == 1
    assert result["phase_run_seconds"]["decode"]["mean"] == 5
    assert result["decode_batch"]["mean"] == 2
    assert result["prefill_wait_seconds"]["new"]["mean"] == .5


def test_failed_admission_hook_remains_owned_for_kv_cleanup(monkeypatch):
    s, _ = setup(monkeypatch)
    fresh = request()
    s.add(fresh)
    def fail(seq, costs):
        raise RuntimeError("failed after acquiring residency")
    s.memory_oracle.on_prompt_admitted = fail
    with pytest.raises(RuntimeError, match="acquiring residency"):
        s.schedule()
    assert fresh in s.waiting
    assert s.abort(fresh.seq_id)  # instructs engine to free the acquired resources
    assert s.is_finished() and not s._prefill_wait_since
