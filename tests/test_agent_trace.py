"""Real HTTP payload/correlation, closed-loop pacing, and artifact trust contracts."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace

import httpx
import pytest

from benchmark.swe_bench_lite.agent_trace import AgentTrace, CURRENT_TRACE, install_http_recorder
from benchmark.sparseengine_regression.agent_trace import (
    export_legacy, grade, load_trace, parse_recording, replay_agent, replay_body,
    run_replay, write,
)
from benchmark.efficiency.metrics import http_trace_summary


def response(rid="r", count=2):
    return {"id": rid, "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"completion_tokens": count}}


def turn(index=0, gap=None):
    return {"turn": index, "think_time_s": gap, "completion_tokens": 2,
            "request": {"model": "original", "messages": [{"role": "user", "content": "hello"}],
                        "stop": ["end"], "max_completion_tokens": 9}, "response": response()}


def test_http_recorder_preserves_payloads_and_thread_identity(tmp_path, monkeypatch):
    # MiniSWE executes instances concurrently; another instance must never inherit its context.
    original = httpx.Client.send
    monkeypatch.setattr(httpx.Client, "send", original)
    install_http_recorder()

    def worker(iid):
        trace = AgentTrace(tmp_path / f"{iid}.jsonl", iid)
        token = CURRENT_TRACE.set(trace)
        try:
            with httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=response(iid)))) as client:
                for _ in range(2):
                    result = client.post("http://test/v1/chat/completions", json=turn()["request"],
                                         headers={"Authorization": "Bearer do-not-record"})
                    assert result.json()["id"] == iid
        finally:
            CURRENT_TRACE.reset(token)
            trace.finish(None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, ["a", "b"]))
    for iid in ("a", "b"):
        path = tmp_path / f"{iid}.jsonl"
        rows = parse_recording(path, iid)
        assert [row["response"]["id"] for row in rows] == [iid, iid]
        assert rows[0]["request"] == turn()["request"]
        assert rows[0]["think_time_s"] is None and rows[1]["think_time_s"] >= 0
        assert "do-not-record" not in path.read_text()


def test_failed_attempt_and_killed_worker_are_not_exported_as_success(tmp_path):
    path = tmp_path / "raw.jsonl"
    trace = AgentTrace(path, "a")
    request = httpx.Request("POST", "http://test/v1/chat/completions", json=turn()["request"])

    def fail(*args, **kwargs):
        raise httpx.ReadTimeout("broken transport")
    with pytest.raises(httpx.ReadTimeout):
        trace.send(fail, None, request)
    with pytest.raises(ValueError, match="Unfinished"):
        parse_recording(path, "a")
    trace.finish(None)
    with pytest.raises(ValueError, match="Failed HTTP"):
        parse_recording(path, "a")


def test_replay_waits_after_response_and_never_uses_new_generated_history():
    now = [0.0]
    arrivals, prompts = [], []
    agent = {"instance_id": "a", "turns": [turn(), turn(1, 3.5)]}
    before = copy.deepcopy(agent)

    def sleep(delay):
        now[0] += delay

    def send(item):
        arrivals.append(now[0])
        prompts.append(replay_body(item, "target"))
        now[0] += 2
        return response()

    rows = replay_agent(agent, send, sleep=sleep, clock=lambda: now[0])
    assert arrivals == [0, 5.5]
    assert all(row["latency_s"] == 2 for row in rows)
    assert prompts[1]["messages"] == before["turns"][1]["request"]["messages"]
    assert prompts[0]["max_tokens"] == 2 and prompts[0]["ignore_eos"]
    assert "stop" not in prompts[0] and "max_completion_tokens" not in prompts[0]
    assert agent == before


def test_short_decode_fails_and_blocks_dependent_turns():
    calls = []
    def send(item):
        calls.append(item)
        return response(count=1)
    rows = replay_agent({"instance_id": "a", "turns": [turn(), turn(1, .1)]}, send, sleep=lambda _: None)
    assert [row["status"] for row in rows] == ["model_failed", "skipped_by_policy"]
    assert len(calls) == 1
    summary = http_trace_summary(rows, 1)
    assert summary["status"] == "failed" and summary["output_token_throughput_tps"] is None


def legacy_fixture(tmp_path):
    run, logs = tmp_path / "run", tmp_path / "logs"
    run.mkdir(); logs.mkdir()
    (run / "instances.txt").write_text("a\nb\n")
    write(run / "run_config.json", {"instance_ids": ["a", "b"]})
    write(run / "run_manifest.json", {"provenance": "test"})
    directory = run / "batches/batch_000/a"
    directory.mkdir(parents=True)
    write(directory / "a.traj.json", {"info": {"exit_status": "LimitsExceeded", "model_stats": {"api_calls": 2}},
                                      "messages": [{"extra": {"response": response("r1")}},
                                                   {"extra": {"response": response("r2")}}]})
    for rid, stamp in (("r1", 1789629000000), ("r2", 1789629003500)):
        write(logs / f"{stamp}_id.json", {"request_id": rid, "status": "success", "elapsed_s": 2,
                                          "request": turn()["request"], "response": response(rid)})
    return run, logs


def test_legacy_join_preserves_terminal_failures_and_marks_estimated_timing(tmp_path):
    run, logs = legacy_fixture(tmp_path)
    out = tmp_path / "export"
    manifest = export_legacy(run, [logs], out, 1)
    assert manifest["timing_quality"] == "estimated"
    assert manifest["agents"][0]["exit_status"] == "LimitsExceeded"
    data = json.loads((out / "agent_000.json").read_text())
    assert data["turns"][1]["think_time_s"] == 1.5
    assert all("recorded_latency_s" not in item for item in data["turns"])
    assert load_trace(out)["request_count"] == 2
    (out / "agent_000.json").write_text("{}")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        load_trace(out)


def test_legacy_import_rejects_missing_response_and_duplicate_ids(tmp_path):
    run, logs = legacy_fixture(tmp_path)
    files = sorted(logs.glob("*.json"))
    second = json.loads(files[1].read_text())
    files[1].unlink()
    with pytest.raises(ValueError, match="Missing 1"):
        export_legacy(run, [logs], tmp_path / "missing", 1)
    write(files[1], second)
    write(logs / "duplicate.json", second)
    with pytest.raises(ValueError, match="Ambiguous"):
        export_legacy(run, [logs], tmp_path / "duplicate", 1)


def test_legacy_replay_requires_explicit_timing_acceptance(tmp_path):
    run, logs = legacy_fixture(tmp_path)
    out = tmp_path / "export"
    export_legacy(run, [logs], out, 1)
    args = SimpleNamespace(agent_trace=out, agent_api_base="http://test/v1", agent_server_manifest="unused",
                           agent_concurrency=2, agent_request_timeout=1, agent_allow_estimated_timing=False)
    with pytest.raises(ValueError, match="estimated"):
        run_replay(args)


def test_grading_cannot_pass_different_workload_or_failed_samples():
    base = {"status": "success", "comparison_contract": {"trace": "a"}, "latency_s_p95": 2, "elapsed_s": 10}
    assert grade({**base, "elapsed_s": 12}, base)["status"] == "failed"
    with pytest.raises(ValueError, match="Baseline mismatch"):
        grade({**base, "comparison_contract": {"trace": "b"}}, base)
    with pytest.raises(ValueError, match="incomplete/failed"):
        grade({**base, "status": "failed"}, base)


def test_real_http_replay_writes_results_and_detects_regression(tmp_path):
    run, logs = legacy_fixture(tmp_path)
    trace = tmp_path / "export"
    export_legacy(run, [logs], trace, 1)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            data = json.dumps(response(count=body["max_tokens"])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = tmp_path / "server.json"
    write(manifest, {"model_path": "test-model", "served_model_name": "target", "engine_kwargs": {"sparse_method": "vanilla"},
                     "hardware": {"gpus": "GPU-1234-abcd"}})
    args = SimpleNamespace(agent_trace=trace, agent_api_base=f"http://127.0.0.1:{server.server_port}/v1",
                           agent_server_manifest=manifest, agent_concurrency=1, agent_request_timeout=2,
                           agent_allow_estimated_timing=True, output_root=tmp_path, run_id="replay",
                           dry_run=False, agent_api_key_env="UNUSED_TEST_KEY", agent_baseline=None,
                           agent_max_slowdown=1.1)
    try:
        assert run_replay(args) == 0
    finally:
        server.shutdown(); thread.join(); server.server_close()
    result = json.loads((tmp_path / "sparseengine_regression/replay/agent_trace.json").read_text())
    assert result["status"] == "success" and result["request_count"] == 2
    assert result["elapsed_s"] >= 1.5
    assert all(body["model"] == "target" and body["ignore_eos"] for body in requests)
