"""Protect capture preparation when prompt peaks exceed compacted residency."""

from collections import deque
from types import SimpleNamespace

import pytest

from sparseengine.engine.decode_cuda_graph import DecodeCudaGraphKey
from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.sampling_params import SamplingParams


def _engine(*, capacity=25, remote_capacity=None, fail_prefill=False, fail_capture=False):
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        sparse_method="h2o", decode_graph_startup_capture=True,
        decode_graph_capture_sizes=[3], decode_graph_startup_capture_limit=8,
        max_model_len=24, sink_keep_tokens=2, decode_keep_tokens=6,
        recent_keep_tokens=2, hf_config=SimpleNamespace(vocab_size=128),
    )
    engine.scheduler = SimpleNamespace(
        waiting=deque(), decoding=deque(),
    )
    live = {}
    costs = {}
    events = []
    graph = SimpleNamespace(_graphs={}, capture_count=0, replay_count=0)

    def add_request(tokens, _params):
        seq = SimpleNamespace(seq_id=tokens[0], num_tokens=len(tokens) + 1,
                              prompt_len=len(tokens))
        live[seq.seq_id] = seq
        engine.scheduler.waiting.append(seq)
        return seq.seq_id

    def step():
        seq = engine.scheduler.waiting.popleft()
        assert sum(costs.values()) + seq.prompt_len <= capacity
        costs[seq.seq_id] = seq.prompt_len
        engine.scheduler.decoding.append(seq)
        if fail_prefill:
            raise RuntimeError("prefill kernel failed")
        costs[seq.seq_id] = min(seq.prompt_len, 7)
        events.append(("prefill", seq.seq_id, sum(costs.values())))

    def abort(seq_id):
        seq = live.pop(seq_id)
        costs.pop(seq_id, None)
        for queue in (engine.scheduler.waiting, engine.scheduler.decoding):
            if seq in queue:
                queue.remove(seq)

    def call(method, *args):
        if method == "resolve_startup_decode_graph_plan":
            plan = args[0]
            return [{"feasible": [entry for entry in plan
                                   if entry[0] * 3 <= capacity]}]
        if method == "startup_capture_prefill_fits":
            peak = sum(costs.values()) + args[0]
            return [{"fits": peak <= limit}
                    for limit in (capacity, remote_capacity or capacity)]
        if method == "startup_capture_decode_fits":
            peak = sum(costs.values()) + len(args[0])
            return [{"world_rank": rank, "fits": peak <= limit}
                    for rank, limit in enumerate((capacity, remote_capacity or capacity))]
        if method == "capture_decode_cuda_graph_warmup":
            seqs = args[0]
            assert len(live) == len(seqs)
            assert sum(costs.values()) + len(seqs) <= capacity
            if fail_capture:
                raise RuntimeError("capture kernel failed")
            key = DecodeCudaGraphKey(
                method="h2o", batch_size=len(seqs), capture_sampling=False,
                graph_path_id="unified",
            )
            graph._graphs[key] = SimpleNamespace(
                graph=object(), capture_context_capacity=24,
            )
            graph.capture_count += 1
            events.append(("capture", key.graph_path_id, len(seqs)))

    engine.add_request = add_request
    engine.step = step
    engine.abort_request = abort
    engine.model_runner = SimpleNamespace(call=call, decode_graph_runner=graph)
    return engine, live, costs, events


def test_capture_uses_short_seeds_for_full_context_capacity():
    engine, live, costs, events = _engine()
    assert engine._capture_startup_decode_graphs(0, respect_runtime_capacity=True) == 3
    assert ("capture", "unified", 3) in events
    assert not live and not costs
    assert not engine.scheduler.waiting and not engine.scheduler.decoding


def test_capture_reaches_full_batch_after_prefill_compaction():
    engine, live, costs, events = _engine()
    parked, offset = engine._prepare_startup_capture_batch(
        SamplingParams(max_tokens=2), 0, batch_size=3, prompt_len=10,
        sequential_prefill=True,
    )
    assert offset == 3 and len(parked) == 3
    assert max(event[2] for event in events if event[0] == "prefill") <= 25
    for seq in parked:
        engine.abort_request(seq.seq_id)
    assert not live and not costs


def test_serial_capture_checks_every_rank_and_releases_partial_batch():
    engine, live, costs, events = _engine(remote_capacity=20)
    parked, offset = engine._prepare_startup_capture_batch(
        SamplingParams(max_tokens=2), 0, batch_size=3, prompt_len=10,
        sequential_prefill=True,
    )
    assert parked is None and offset == 2
    assert not live and not costs
    assert not engine.scheduler.waiting and not engine.scheduler.decoding


def test_capture_prefill_failure_propagates_and_releases_requests():
    engine, live, costs, _ = _engine(fail_prefill=True)

    with pytest.raises(RuntimeError, match="prefill kernel failed"):
        engine._capture_startup_decode_graphs(0, respect_runtime_capacity=True)

    assert not live and not costs
    assert not engine.scheduler.waiting and not engine.scheduler.decoding


@pytest.mark.parametrize("capacity,remote_capacity,fits", [
    (31, None, False),
    (32, None, True),
    (32, 31, False),
])
def test_capture_checks_full_decode_append_after_serial_prefill(
    capacity, remote_capacity, fits,
):
    # Four prompts peak at 31 slots with serial prefill, retain 28, and need
    # 32 for the captured decode step. Existing B=3 coverage misses this gap.
    engine, live, costs, events = _engine(
        capacity=capacity, remote_capacity=remote_capacity,
    )
    engine.config.decode_graph_capture_sizes = [4]

    parked, offset = engine._prepare_startup_capture_batch(
        SamplingParams(max_tokens=2), 0, batch_size=4, prompt_len=10,
        sequential_prefill=True,
    )
    assert offset == 4
    assert (parked is not None) == fits
    for seq in parked or []:
        engine.abort_request(seq.seq_id)
    assert not live and not costs
    assert not engine.scheduler.waiting and not engine.scheduler.decoding


def test_capture_rejects_empty_plan_after_decode_capacity_checks():
    # The short prompts fit together, but even their first decode cannot fit.
    engine, live, costs, events = _engine(capacity=4)
    engine.config.decode_graph_capture_sizes = [4]

    with pytest.raises(RuntimeError, match="cannot capture any configured"):
        engine._capture_startup_decode_graphs(0, respect_runtime_capacity=True)

    assert not any(event[0] == "capture" for event in events)
    assert not live and not costs
    assert not engine.scheduler.waiting and not engine.scheduler.decoding


def test_capture_kernel_failure_is_not_treated_as_capacity_skip():
    engine, live, costs, _ = _engine(fail_capture=True)

    with pytest.raises(RuntimeError, match="capture kernel failed"):
        engine._capture_startup_decode_graphs(0, respect_runtime_capacity=True)

    assert not live and not costs
    assert not engine.scheduler.waiting and not engine.scheduler.decoding
