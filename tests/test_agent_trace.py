"""Real HTTP payload/correlation, closed-loop pacing, and artifact trust contracts."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace

import httpx
import pytest

from benchmark.swe_bench_lite.agent_trace import AgentTrace, CURRENT_TRACE, install_http_recorder
from benchmark.sparseengine_regression.agent_trace import (
    digest, export_legacy, grade, load_trace, parse_recording, replay_agent, replay_body,
    post_with_chain_recovery, run_replay, synthetic_think_time, write,
)
from benchmark.efficiency.metrics import http_trace_summary


def response(rid="r", count=2):
    return {"id": rid, "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"completion_tokens": count}}


def turn(index=0, gap=None):
    return {"turn": index, "think_time_s": gap, "completion_tokens": 2,
            "request": {"model": "original", "messages": [{"role": "user", "content": "hello"}],
                        "stop": ["end"], "max_completion_tokens": 9}, "response": response()}


def test_synthetic_wait_is_stable_and_bounded_across_replays():
    first = [synthetic_think_time("agent-a", turn, 42, 2.0) for turn in range(80)]
    assert first[0] == 0
    assert all(0 <= delay < 2.0 for delay in first)
    assert first == [synthetic_think_time("agent-a", turn, 42, 2.0) for turn in range(80)]
    assert first[1:] != [synthetic_think_time("agent-b", turn, 42, 2.0) for turn in range(1, 80)]


def test_evicted_chain_retries_once_with_the_full_recorded_prompt():
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return httpx.Response(410, json={"detail": {"code": "chain_gone"}})
        return httpx.Response(200, json={"chain_id": "new-chain", "chain_status": "created"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result, recovered = post_with_chain_recovery(
            client, "http://test/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}],
             "chain_id": "old-chain", "chain_append_start": 2},
        )
    assert recovered and result["chain_id"] == "new-chain"
    assert len(sent) == 2 and "chain_id" not in sent[1] and "chain_append_start" not in sent[1]
    assert sent[1]["messages"] == sent[0]["messages"]


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


def test_legacy_selection_ranks_complete_trajectories_and_keeps_source_order_on_ties(tmp_path):
    from benchmark.sparseengine_regression.agent_trace import select_legacy_ids

    run = tmp_path / "run"
    for iid, prompt_counts in (("short", [20]), ("long_a", [10, 30]),
                               ("long_b", [25, 25])):
        directory = run / "batches" / "batch_000" / iid
        directory.mkdir(parents=True)
        messages = [{"extra": {"response": {"usage": {"prompt_tokens": count}}}}
                    for count in prompt_counts]
        write(directory / f"{iid}.traj.json", {
            "info": {"model_stats": {"api_calls": len(messages)}}, "messages": messages,
        })
    ids = ["short", "long_b", "long_a"]
    assert select_legacy_ids(run, ids, 2, "longest_turns") == ["long_b", "long_a"]
    assert select_legacy_ids(run, ids, 2, "total_prompt_tokens") == ["long_b", "long_a"]
    assert select_legacy_ids(run, ids, 1, "max_prompt_tokens") == ["long_a"]
    assert select_legacy_ids(run, ids, 2, "longest_turns", ("long_b",)) == ["long_a", "short"]
    with pytest.raises(ValueError, match="absent from source order"):
        select_legacy_ids(run, ids, 2, "longest_turns", ("missing",))


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


@pytest.mark.parametrize(
    ("synthetic_wait", "think_scale", "keep_ratio", "expected_protocol"),
    [
        (None, 1.0, None, "closed_loop_forced_recorded_answers_v1"),
        (0.1, 1.0, None, "closed_loop_forced_recorded_answers_synthetic_wait_v1"),
        (None, 0.5, None, "closed_loop_forced_recorded_answers_v2"),
        (0.1, 1.0, 0.5, "closed_loop_forced_recorded_answers_synthetic_wait_v2"),
    ],
)
def test_forced_replay_manifest_names_its_actual_protocol(
    tmp_path, synthetic_wait, think_scale, keep_ratio, expected_protocol,
):
    run, logs = legacy_fixture(tmp_path)
    trace = tmp_path / "export"
    manifest = export_legacy(run, [logs], trace, 1)
    server_path = tmp_path / "server.json"
    write(server_path, {
        "model_path": "test-model", "served_model_name": "target",
        "model_config": {"vocab_size": 128},
        "engine_kwargs": {"max_model_len": 128},
        "hardware": {"gpus": "GPU-1234-abcd"},
    })
    prepared = {}
    forced_digest = hashlib.sha256()
    for entry in manifest["agents"]:
        agent = json.loads((trace / entry["file"]).read_text())
        prepared[entry["file"]] = [
            {
                "token_ids": [11] * turn["completion_tokens"],
                "chain_append_start": len(turn["request"]["messages"]) + 1,
            }
            for turn in agent["turns"]
        ]
        for turn, spec in zip(agent["turns"], prepared[entry["file"]]):
            ids = spec["token_ids"]
            forced_digest.update(json.dumps(
                [entry["instance_id"], turn["turn"], ids, len(ids)],
                separators=(",", ":"),
            ).encode())
    forced_path = tmp_path / "forced.json"
    write(forced_path, {
        "schema": "agent_forced_workload_v1",
        "trace_sha256": digest(trace / "manifest.json"),
        "model_path": "test-model", "agents": prepared,
        "forced_workload_sha256": forced_digest.hexdigest(),
    })
    args = SimpleNamespace(
        agent_trace=trace, agent_api_base="http://test/v1",
        agent_server_manifest=server_path, agent_concurrency=1,
        agent_request_timeout=2, agent_allow_estimated_timing=True,
        output_root=tmp_path, run_id="replay", dry_run=True,
        agent_force_recorded_responses=True, agent_forced_workload=forced_path,
        agent_synthetic_think_time_max_s=synthetic_wait,
        agent_think_time_scale=think_scale,
        agent_prefix_prune_keep_ratio=keep_ratio,
        agent_prefix_prune_tokenizer="unused" if keep_ratio is not None else None,
    )
    assert run_replay(args) == 0
    resolved = json.loads((tmp_path / "sparseengine_regression/replay/resolved_manifest.json").read_text())
    assert resolved["comparison_contract"]["protocol"] == expected_protocol


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
            answer = response(count=body["max_tokens"])
            answer["usage"]["prompt_tokens_details"] = {"cached_tokens": int(len(requests) > 1)}
            data = json.dumps(answer).encode()
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
                           agent_max_slowdown=1.1, agent_require_cache_hit=True)
    try:
        assert run_replay(args) == 0
    finally:
        server.shutdown(); thread.join(); server.server_close()
    result = json.loads((tmp_path / "sparseengine_regression/replay/agent_trace.json").read_text())
    assert result["status"] == "success" and result["request_count"] == 2
    assert result["cache_reuse"]["successful_requests_with_cached_tokens"] == 1
    assert result["elapsed_s"] >= 1.5
    assert all(body["model"] == "target" and body["ignore_eos"] for body in requests)


def test_replay_pruning_finishes_before_next_turn_and_excludes_request_latency():
    # A replay that starts the next turn before pruning cannot test physical reuse.
    now = [0.0]
    events = []
    agent = {"instance_id": "a", "turns": [turn(), turn(1, 90)]}
    def send(item):
        events.append(("send", item["turn"], now[0]))
        now[0] += 2
        return response()
    def prune(item):
        events.append(("prune", item["turn"], now[0]))
        now[0] += 3
    rows = replay_agent(agent, send, after_turn=prune, think_time_scale=0,
                        sleep=lambda delay: now.__setitem__(0, now[0] + delay), clock=lambda: now[0])
    assert events == [("send", 0, 0), ("prune", 0, 2), ("send", 1, 5), ("prune", 1, 7)]
    assert [r["latency_s"] for r in rows] == [2, 2]
    assert [r["prefix_prune_elapsed_s"] for r in rows] == [3, 3]


def test_replay_prune_failure_stops_trajectory_without_hiding_model_response():
    def prune(item):
        raise RuntimeError("physical reuse failed")
    rows = replay_agent({"instance_id": "a", "turns": [turn(), turn(1, 0)]},
                        lambda item: response(), after_turn=prune)
    assert rows[0]["status"] == "model_failed"
    assert rows[0]["response"]["id"] == "r"
    assert "physical reuse failed" in rows[0]["error"]
    assert rows[1]["status"] == "skipped_by_policy"
