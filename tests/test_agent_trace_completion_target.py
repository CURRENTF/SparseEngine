"""Protect the server-log boundary for partial agent-trace results."""

import hashlib
import json
import sys

import pytest

from scripts.official_experiments.agent_trace_sparse_methods import (
    summarize_completion_target,
)


def _summarize(tmp_path, monkeypatch, events):
    trace = tmp_path / "manifest.json"
    trace.write_text("{}")
    forced = tmp_path / "forced.json"
    forced.write_text(json.dumps({
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "forced_workload_sha256": "frozen-inputs",
    }))
    server_log = tmp_path / "server.log"
    server_log.write_text("\n".join(events) + "\n")
    output = tmp_path / "summary.json"
    stopped = []
    monkeypatch.setattr(summarize_completion_target.os, "kill", lambda pid, sig: stopped.append(pid))
    monkeypatch.setattr(sys, "argv", [
        "summarize_completion_target.py",
        "--server-log", str(server_log),
        "--client-pid", "12345",
        "--target", "1",
        "--method", "vanilla-prefix",
        "--concurrency", "1",
        "--trace-manifest", str(trace),
        "--forced-workload", str(forced),
        "--output", str(output),
    ])
    return output, stopped


def test_prefill_preemption_without_recompute_rejects_target(tmp_path, monkeypatch):
    output, stopped = _summarize(tmp_path, monkeypatch, [
        "2026-09-26 10:00:00 request_start id=7",
        "2026-09-26 10:00:01 驱逐请求 id = 7 | slots={}",
        "2026-09-26 10:00:02 request_finish id=7 prompt_tokens=3 completion_tokens=2",
        '2026-09-26 10:00:02 POST /v1/chat/completions HTTP/1.1" 200',
    ])
    with pytest.raises(ValueError, match="preemptions=1 recompute_replays=0"):
        summarize_completion_target.main()
    assert not output.exists()
    assert stopped == []


def test_events_after_target_do_not_invalidate_completed_prefix(tmp_path, monkeypatch):
    output, stopped = _summarize(tmp_path, monkeypatch, [
        "2026-09-26 10:00:00 request_start id=7",
        "2026-09-26 10:00:02 request_finish id=7 prompt_tokens=3 completion_tokens=2",
        "2026-09-26 10:00:03 驱逐请求 id = 8 | slots={}",
        '2026-09-26 10:00:03 POST /v1/chat/completions HTTP/1.1" 200',
    ])
    summarize_completion_target.main()
    result = json.loads(output.read_text())
    assert result["elapsed_s"] == 2
    assert result["preemptions_before_target"] == 0
    assert result["recompute_replays_before_target"] == 0
    assert stopped == [12345]
