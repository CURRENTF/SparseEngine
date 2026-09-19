"""Scheduling and ownership contracts; CUDA equivalence is tested separately."""
from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.async_scheduling.execution import AsyncDrainRequired, AsyncExecution
from sparseengine.engine.async_scheduling.scheduler import AsyncScheduler, execution_snapshot
from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams
from test_prefill_schedule_policy import FakeMemoryOracle, make_scheduler


class Executor:
    def __init__(self):
        self.pending = {}
        self.calls = []
        self.retired = set()

    def call(self, method, *args):
        self.calls.append((method, args))
        if method == 'submit_async':
            ticket, seqs, is_prefill = args
            self.pending[ticket] = (seqs, is_prefill)
        elif method == 'collect_async':
            seqs, prefill = self.pending.pop(args[0])
            return [1000 + (s.num_prompt_tokens if prefill else s.num_tokens) for s in seqs], None
        elif method == 'retire_async':
            referenced = {s.seq_id for seqs, _ in self.pending.values() for s in seqs}
            assert not referenced.intersection(args[0]), 'freed in-flight storage'
            self.retired.update(args[0])
        else:
            raise AssertionError(method)


def engine(depth=2, chunk=3):
    scheduler = make_scheduler('all_chunked', chunk=chunk, max_tokens=12)
    executor = Executor()
    def no_sync():
        raise AssertionError('unexpected synchronous execution')
    result = SimpleNamespace(
        config=SimpleNamespace(async_max_inflight=depth), scheduler=scheduler,
        model_runner=executor, _release_preempted_sequences=lambda _: None,
        _step_sync=no_sync,
        _throughput_logger=SimpleNamespace(record_step=lambda _: None, record_state=lambda *args: None),
    )
    return result, AsyncScheduler(result)


def request(length, outputs, **kwargs):
    return Sequence(list(range(1, length+1)), SamplingParams(max_tokens=outputs, **kwargs))


@pytest.mark.parametrize('depth', [2, 3, 4])
@pytest.mark.parametrize('outputs', [1, 2, 7])
def test_chunked_prefill_and_output_limits_without_placeholder_tokens(depth, outputs):
    e, driver = engine(depth)
    seqs = [request(5, outputs, ignore_eos=True), request(8, outputs, ignore_eos=True)]
    for seq in seqs:
        e.scheduler.add(seq)
    finished = []
    for _ in range(40):
        done, _ = driver.step()
        finished.extend(done)
        if not driver.pending and e.scheduler.is_finished():
            break
    else:
        pytest.fail('scheduler did not drain')
    assert len(finished) == len(seqs)
    for seq in seqs:
        assert seq.completion_token_ids == list(range(1000+seq.num_prompt_tokens, 1000+seq.num_prompt_tokens+outputs))
        assert seq.num_pending_outputs == 0
    assert e.model_runner.retired == {s.seq_id for s in seqs}
    methods = [m for m, _ in e.model_runner.calls]
    assert methods[:2] == ['submit_async', 'submit_async']


def test_eos_discards_lookahead_and_retires_only_after_completion():
    e, driver = engine(depth=3, chunk=16)
    seq = request(5, 20, eos_token_ids=[1005])
    e.scheduler.add(seq)
    done, _ = driver.step()
    assert done[0][1] == [1005]
    assert seq.num_pending_outputs == 0
    assert not driver.pending
    assert seq.seq_id in e.model_runner.retired


def test_abort_last_request_drains_gpu_owners_without_publishing_more_tokens():
    e, driver = engine(chunk=16)
    seq = request(5, 10, ignore_eos=True)
    e.scheduler.add(seq)
    driver.step()
    assert driver.pending
    before = list(seq.completion_token_ids)
    driver.abort(seq.seq_id)
    assert seq.completion_token_ids == before
    assert not driver.pending
    assert e.scheduler.is_finished()
    assert e.model_runner.retired == {seq.seq_id}


def test_new_request_can_join_while_prior_output_is_in_flight():
    e, driver = engine(chunk=16)
    first = request(5, 8, ignore_eos=True)
    e.scheduler.add(first)
    driver.step()
    later = request(7, 3, ignore_eos=True)
    e.scheduler.add(later)
    done = []
    for _ in range(30):
        rows, _ = driver.step()
        done.extend(rows)
        if e.scheduler.is_finished() and not driver.pending:
            break
    assert {x[0] for x in done} == {first.seq_id, later.seq_id}
    assert len(first.completion_token_ids) == 8
    assert len(later.completion_token_ids) == 3


def test_execution_snapshot_advances_position_without_fabricating_history():
    seq = request(5, 8, ignore_eos=True)
    seq.num_pending_outputs = 2
    snapshot = execution_snapshot(seq)
    assert snapshot.decode_input_position == seq.decode_input_position + 2
    assert list(snapshot.token_ids) == seq.token_ids
    seq.append_token(17)
    assert list(snapshot.token_ids) == [1, 2, 3, 4, 5]
    assert snapshot.token_ids[-1] == 5
    assert snapshot.token_ids[:] == [1, 2, 3, 4, 5]
    assert snapshot.token_ids[::-1] == [5, 4, 3, 2, 1]
    assert snapshot.token_ids[3:100] == [4, 5]
    with pytest.raises(IndexError):
        snapshot.token_ids[5]


def test_device_penalty_state_matches_independent_value_formula():
    state = object.__new__(AsyncExecution)
    state.penalties = {}
    seq = Sequence([1, 3], SamplingParams(repetition_penalty=1.3, presence_penalty=.7))
    seq.append_token(2)
    logits = torch.tensor([[.1, -2., 3., 4., -1.]])
    result = state.apply_penalties(logits, [seq])
    expected = logits.clone()
    for token in [1, 2, 3]:
        expected[0, token] = expected[0, token] / 1.3 if expected[0, token] > 0 else expected[0, token] * 1.3
    expected[0, 2] -= .7
    torch.testing.assert_close(result, expected)
    assert torch.equal(logits, torch.tensor([[.1, -2., 3., 4., -1.]]))


def test_sync_recovery_invalidates_feedback_only_after_all_results_retire():
    state = object.__new__(AsyncExecution)
    state.results = {7: object()}
    state.last_tokens = {1: torch.tensor(17)}
    state.penalties = {1: object()}
    with pytest.raises(RuntimeError, match="all asynchronous results"):
        state.prepare_synchronous_execution()
    assert 1 in state.last_tokens
    state.results.clear()
    state.prepare_synchronous_execution()
    assert not state.last_tokens
    assert not state.penalties


@pytest.mark.parametrize('reservation_failure', [False, True])
@pytest.mark.parametrize('count', [1, 2])
def test_inflight_preemption_preserves_every_request_for_drain_and_abort(reservation_failure, count):
    oracle = FakeMemoryOracle(free_slots=0)
    scheduler = make_scheduler('all_chunked', oracle=oracle)
    seqs = [request(4, 16, ignore_eos=True) for _ in range(count)]
    for seq in seqs:
        seq.num_prefilled_tokens = 4
        seq.append_token(7)
        seq.num_pending_outputs = 1
        scheduler.decoding.append(seq)
    if reservation_failure:
        oracle.reserve_decode_windows = lambda decoding, waiting: seqs[0]
    scheduler._async_inflight = 1
    with pytest.raises(AsyncDrainRequired):
        scheduler.schedule()
    assert list(scheduler.decoding) == seqs
    assert scheduler.total_preemptions == 0
    assert not any(seq.is_recompute_replay for seq in seqs)
    for seq in seqs:
        assert scheduler.abort(seq.seq_id)
    assert scheduler.is_finished()


def test_cpu_token_dependency_retires_before_snapshotting_next_input():
    e, driver = engine(depth=3)
    driver.requires_committed_token_history = True
    seq = request(2, 4, ignore_eos=True)
    e.scheduler.add(seq)
    for _ in range(4):
        driver.step()
        assert not driver.pending
    submissions = [args for method, args in e.model_runner.calls if method == 'submit_async']
    # The next input is the actually committed predecessor, not a placeholder
    # plus an advanced position. All steps still use the same submit/collect API.
    for previous, current in zip(submissions, submissions[1:]):
        prev = previous[1][0]
        expected = 1000 + (prev.num_prompt_tokens if previous[2] else prev.num_tokens)
        assert current[1][0].decode_input_token == expected


def test_async_submission_runs_state_transitions_before_context_is_reset():
    from sparseengine.engine.model_runner import ModelRunner
    from sparseengine.utils.context import get_context, reset_context, set_context
    calls = []
    runner = SimpleNamespace(
        _async_submitting=True,
        sparse_controller=SimpleNamespace(post_forward=lambda seqs, pf: calls.append(
            ('sparse', get_context().seqs, pf))),
        runtime_state=SimpleNamespace(on_forward_end=lambda seqs, pf: calls.append(
            ('cache', get_context().seqs, pf))),
    )
    seqs = [request(3, 2)]
    try:
        set_context(True, seqs=seqs)
        ModelRunner._post_sparse_forward(runner, seqs, True)
        assert calls == [('sparse', seqs, True), ('cache', seqs, True)]
    finally:
        reset_context()


def test_cpu_history_boundary_allows_nonpublishing_prefill_chunks():
    e, driver = engine(depth=3, chunk=2)
    driver.requires_committed_token_history = True
    seq = request(9, 2, ignore_eos=True)
    e.scheduler.add(seq)
    driver.step()
    submits = [args for method, args in e.model_runner.calls if method == 'submit_async']
    assert len(submits) == 3
    assert all(not args[1][0].is_last_chunk_prefill for args in submits)


@pytest.mark.parametrize('requested,streams,expected', [
    (None, True, True), (None, False, False), (False, True, False), (True, True, True),
])
def test_async_configuration_resolves_automatic_and_explicit_execution(tmp_path, monkeypatch,
                                                                       requested, streams, expected):
    from transformers import Qwen3Config
    from sparseengine.config import Config
    from sparseengine.platforms import device_runtime
    Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                vocab_size=256, max_position_embeddings=1024).save_pretrained(tmp_path)
    monkeypatch.setattr(device_runtime, 'supports_streams', lambda *args: streams)
    config = Config(str(tmp_path), async_scheduling=requested, decode_graph=False,
                    max_model_len=512, max_num_seqs_in_batch=2)
    assert config.async_scheduling is expected
