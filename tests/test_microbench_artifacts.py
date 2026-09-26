import json
from types import SimpleNamespace

import pytest

from benchmark.microbench import (
    _benchmark_sparse_method,
    _completed_output_records,
    _decode_cuda_graph_status,
    _record_child_exit_failure,
    _write_output_dir,
)


def test_partial_output_artifact_does_not_label_short_generation_success():
    """A failed strict run must retain a failure status on its short raw output."""
    outputs = [(12, [1, 2, 3]), (13, [4])]
    rows = _completed_output_records(outputs, output_len=3)
    assert [row["status"] for row in rows] == ["success", "model_failed"]
    assert [(row["request_id"], row["token_ids"]) for row in rows] == outputs


@pytest.mark.parametrize("admitted,preempt", [(3, False), (4, False), (4, True)])
def test_native_loop_distinguishes_unadmitted_requests_from_short_window(monkeypatch, tmp_path, admitted, preempt):
    """BS4 admission at B3 must reach capacity search, not abort as bad timing.

    Exercise the actual loop; collector tests cannot catch post-loop error order.
    A fully admitted but too-short output must still be a window error.
    """
    import sparseengine
    from benchmark import microbench

    class Sequence:
        def __init__(self, seq_id):
            self.seq_id, self.length = seq_id, 3

        def __len__(self):
            return self.length

    class Engine:
        def __init__(self, *args, **kwargs):
            self.scheduler = SimpleNamespace(waiting=[], decoding=[], total_preemptions=0)
            self.iteration = 0
            self.last_step_token_outputs = []
            self.exited = False

        def add_request(self, *args):
            seq_id = len(self.scheduler.waiting)
            self.scheduler.waiting.append(Sequence(seq_id))
            return seq_id

        def is_finished(self):
            return self.iteration == 3

        def step(self):
            self.iteration += 1
            if self.iteration == 1:
                self.scheduler.decoding = self.scheduler.waiting[:admitted]
                self.scheduler.waiting = self.scheduler.waiting[admitted:]
                return [], 2 * admitted
            self.last_step_token_outputs = [(s.seq_id, [7]) for s in self.scheduler.decoding]
            for seq in self.scheduler.decoding:
                seq.length += 1
            outputs = []
            if self.iteration == 3:
                if preempt:
                    self.scheduler.total_preemptions += 1
                    return [], -(admitted - 1)
                outputs = [(s.seq_id, [7] * 3) for s in self.scheduler.decoding]
                self.scheduler.decoding = []
            return outputs, -admitted

        def debug_sparse_state_summaries(self, **kwargs):
            return [dict(decode_graph=dict(capture_count=1, replay_count=self.iteration - 1,
                                           eager_static_count=0, force_eager_count=0))]

        def exit(self):
            self.exited = True

    engine = Engine()
    monkeypatch.setitem(sparseengine.__dict__, "LLM", lambda *a, **kw: engine)
    for name in ("reset_peak_memory_stats", "empty_cache", "synchronize"):
        monkeypatch.setattr(microbench.torch.cuda, name, lambda: None)
    args = SimpleNamespace(hyper_params_dict={}, output_len=3,
        max_model_len_override=None, model_path="fixture", temperature=0.0, top_p=1.0,
        output_dir=str(tmp_path), require_full_decode_batch=True,
        decode_window_steps=4, decode_warmup_steps_after_full=1)
    results = {}
    microbench.benchmark_task("vanilla", 2, 4, args, results)
    row = results[("vanilla", 2, 4)]
    expected = "Full decode batch capacity exceeded" if admitted < 4 or preempt else "without the requested full-residency"
    assert expected in row["error"]
    assert row["actual_decode_peak"] == admitted
    assert row["completed_requests"] == (0 if preempt else admitted)
    assert engine.exited
    raw = [json.loads(line) for line in (tmp_path / "vanilla-2-4/raw_outputs.jsonl").read_text().splitlines()]
    assert len(raw) == (0 if preempt else admitted) and all(len(r["token_ids"]) == 3 for r in raw)


def test_decode_cuda_graph_status_records_execution_counters():
    graph = object()
    runner = SimpleNamespace(
        _graphs={"bs4": SimpleNamespace(graph=graph)},
        last_state_key="bs4",
        capture_count=2,
        replay_count=17,
        eager_static_count=3,
        force_eager_count=1,
    )
    llm = SimpleNamespace(
        model_runner=SimpleNamespace(decode_graph_runner=runner),
        config=SimpleNamespace(decode_graph=True),
    )

    assert _decode_cuda_graph_status(llm) == {
        "decode_graph_configured": True,
        "decode_graph_runner_initialized": True,
        "decode_graph_state_count": 1,
        "decode_graph_graph_count": 1,
        "decode_graph_capture_count": 2,
        "decode_graph_replay_count": 17,
        "decode_graph_eager_static_count": 3,
        "decode_graph_force_eager_count": 1,
        "decode_graph_last_state_key": "bs4",
        "decode_graph_active": True,
    }


def test_benchmark_sparse_method_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported benchmark sparse method"):
        _benchmark_sparse_method("typo")


def test_nonzero_child_exit_overrides_partial_success_row():
    results = {
        ("h2o", 16, 1): {
            "method": "h2o",
            "length": 16,
            "batch_size": 1,
            "status": "SUCCESS",
            "prefill_tp": 123.0,
        }
    }

    _record_child_exit_failure(
        results,
        method="h2o",
        length=16,
        batch_size=1,
        exitcode=7,
        synchronize_step_timing=True,
    )

    row = results[("h2o", 16, 1)]
    assert row["status"] == "FAILED"
    assert row["child_exitcode"] == 7
    assert row["child_partial_status"] == "SUCCESS"
    assert row["prefill_tp"] == 123.0


def test_output_metadata_records_step_timing_mode(tmp_path, monkeypatch):
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_len=8,
        temperature=0.0,
        top_p=1.0,
        synchronize_step_timing=True,
        model_path="test-model",
        methods="h2o",
        lengths="16",
        batch_sizes="1",
        hyper_params_dict={},
    )
    monkeypatch.setattr(
        "benchmark.microbench._git_metadata",
        lambda: {"git_commit": "test", "git_branch": "test", "git_dirty": False},
    )

    _write_output_dir(args, [{"status": "SUCCESS", "length": 16}])

    run_info = json.loads((tmp_path / "run_info.json").read_text(encoding="utf-8"))
    aggregate = json.loads(
        (tmp_path / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    performance = json.loads(
        (tmp_path / "performance.jsonl").read_text(encoding="utf-8")
    )
    assert run_info["synchronize_step_timing"] is True
    assert aggregate["synchronize_step_timing"] is True
    assert aggregate["records"][0]["synchronize_step_timing"] is True
    assert performance["synchronize_step_timing"] is True
