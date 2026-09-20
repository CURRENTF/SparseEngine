"""Export MiniSWE HTTP traces and replay closed-loop agent workloads.

One worker owns a trajectory until it ends. Recorded prompts are immutable;
generated text is measured, not executed or substituted into later prompts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from benchmark.efficiency.metrics import http_trace_summary
from benchmark.swe_bench_lite.agent_trace import SCHEMA


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def positive_count(value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"Expected positive integer token count, got {value!r}")
    return value


def parse_recording(path, iid):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(row.get("schema") != SCHEMA or row.get("instance_id") != iid for row in rows):
        raise ValueError(f"Invalid trace identity: {path}")
    if rows[0].get("event") != "instance_start" or rows[-1].get("event") != "instance_end":
        raise ValueError(f"Unfinished instance trace: {path}")
    if rows[-1].get("uncaught_exception") is not None:
        raise ValueError(f"Instance failed outside agent handling: {path}")
    middle = rows[1:-1]
    if not middle or len(middle) % 2 or rows[-1]["request_count"] != len(middle) // 2:
        raise ValueError(f"Missing HTTP turns: {path}")
    turns = []
    previous_end = None
    for index in range(0, len(middle), 2):
        request, response = middle[index:index + 2]
        turn = index // 2
        if (request.get("event"), response.get("event"), request.get("turn"), response.get("turn")) != (
                "request", "response", turn, turn):
            raise ValueError(f"Nonsequential HTTP turns: {path}")
        if response.get("status") != "success" or response.get("http_status") != 200:
            raise ValueError(f"Failed HTTP attempt in {path}, turn {turn}; do not silently omit retries")
        body = request["request"]
        output = response["response"]
        if not isinstance(body.get("messages"), list) or not body["messages"] or body.get("stream"):
            raise ValueError(f"Expected non-streaming chat request: {path}")
        if body.get("chain_id") or "chain_append_start" in body:
            raise ValueError("Capture a vanilla/radix trace, not server-specific chain handles")
        if not isinstance(output, dict) or not output.get("choices"):
            raise ValueError(f"Missing model response: {path}")
        tokens = positive_count(output["usage"]["completion_tokens"])
        start, end = response["started_monotonic_s"], response["ended_monotonic_s"]
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)) or end < start:
            raise ValueError(f"Invalid HTTP timestamps: {path}")
        gap = response["think_time_s"]
        if previous_end is None:
            if gap is not None:
                raise ValueError("First turn cannot have a predecessor delay")
        elif (not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap < 0
              or not math.isclose(gap, start - previous_end, abs_tol=1e-6)):
            raise ValueError(f"Invalid inter-turn delay: {path}")
        previous_end = end
        turns.append({"turn": turn, "request": body, "response": output,
                      "think_time_s": gap,
                      "completion_tokens": tokens})
    return turns


def export_trace(run_dir, output, expected_instances=100):
    run_dir, output = Path(run_dir), Path(output)
    ids = (run_dir / "instances.txt").read_text().splitlines()
    if len(ids) != expected_instances or len(set(ids)) != len(ids):
        raise ValueError(f"Expected exactly {expected_instances} unique selected instance IDs")
    config = read(run_dir / "run_config.json")
    if config.get("instance_ids") != ids:
        raise ValueError("Selected IDs disagree with the frozen run config")
    output.mkdir(parents=True, exist_ok=False)
    agents = []
    for index, iid in enumerate(ids):
        candidates = list((run_dir / "batches").glob(f"batch_*/{iid}/agent_trace.jsonl"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one recorded trajectory for {iid}, found {len(candidates)}")
        path = candidates[0]
        turns = parse_recording(path, iid)
        trajectory = read(path.parent / f"{iid}.traj.json")
        exit_status = trajectory["info"]["exit_status"]
        if not isinstance(exit_status, str) or not exit_status:
            raise ValueError(f"Missing terminal agent status for {iid}")
        filename = f"agent_{index:03d}.json"
        write(output / filename, {"instance_id": iid, "exit_status": exit_status, "turns": turns})
        agents.append({"instance_id": iid, "file": filename, "sha256": digest(output / filename),
                       "turns": len(turns), "exit_status": exit_status})
    manifest = {"schema": SCHEMA, "timing_boundary": "client_http_send_to_full_response",
                "timing_quality": "measured",
                "source_run_config": config, "source_run_manifest": read(run_dir / "run_manifest.json"),
                "agents": agents, "instance_count": len(ids),
                "request_count": sum(agent["turns"] for agent in agents)}
    # The manifest is written last; an interrupted export is never a valid corpus.
    write(output / "manifest.json", manifest)
    load_trace(output)
    return manifest


def export_legacy(run_dir, request_dirs, output, expected_instances=100):
    """Join by response ID, never prompt similarity or wall-clock proximity.

    Legacy file names timestamp the end of logging, not response delivery. The
    reconstructed gap includes network/tool/retry waits and logging error; its
    error is not bounded by the filename's nanosecond resolution.
    """
    run_dir, output = Path(run_dir), Path(output)
    all_ids = (run_dir / "instances.txt").read_text().splitlines()
    config = read(run_dir / "run_config.json")
    if config.get("instance_ids") != all_ids or len(set(all_ids)) != len(all_ids):
        raise ValueError("Selected IDs disagree with frozen config or contain duplicates")
    ids = all_ids[:expected_instances]
    if expected_instances <= 0 or len(ids) != expected_instances:
        raise ValueError("Not enough instances for the requested fixed prefix")
    trajectories, wanted = {}, set()
    for iid in ids:
        candidates = list((run_dir / "batches").glob(f"batch_*/{iid}/{iid}.traj.json"))
        if len(candidates) != 1:
            raise ValueError(f"Missing or duplicate trajectory for {iid}")
        trajectory = read(candidates[0])
        responses = [message["extra"]["response"] for message in trajectory["messages"]
                     if isinstance(message.get("extra", {}).get("response"), dict)]
        rids = [response["id"] for response in responses]
        if not rids or len(rids) != len(set(rids)) or wanted.intersection(rids):
            raise ValueError(f"Empty/duplicate model responses for {iid}")
        if trajectory["info"]["model_stats"]["api_calls"] != len(rids):
            raise ValueError(f"Trajectory API call accounting mismatch for {iid}")
        trajectories[iid] = (candidates[0], rids, trajectory["info"]["exit_status"])
        wanted.update(rids)
    index = {}
    for directory in request_dirs:
        for path in Path(directory).glob("*.json"):
            record = read(path)
            rid = record.get("request_id")
            if rid not in wanted:
                continue
            if rid in index:
                raise ValueError(f"Ambiguous duplicate server request ID: {rid}")
            if record.get("status") != "success" or record.get("response", {}).get("id") != rid:
                raise ValueError(f"Failed or mismatched server request: {path}")
            index[rid] = path
    if set(index) != wanted:
        raise ValueError(f"Missing {len(wanted - set(index))} server request records")
    output.mkdir(parents=True, exist_ok=False)
    agents = []
    for number, iid in enumerate(ids):
        trajectory_path, rids, exit_status = trajectories[iid]
        turns, previous_end, previous_directory = [], None, None
        for turn, rid in enumerate(rids):
            path = index[rid]
            record = read(path)
            stamp = path.name.split("_")[0]
            if not stamp.isdigit() or len(stamp) not in (13, 19):
                raise ValueError(f"Unsupported legacy log timestamp: {path.name}")
            end = int(stamp) / (1e9 if len(stamp) == 19 else 1e3)
            latency = record["elapsed_s"]
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"Invalid legacy request duration: {path}")
            if previous_directory is not None and previous_directory != path.parent:
                raise ValueError(f"One trajectory spans different server runs/clocks: {iid}")
            previous_directory = path.parent
            gap = None if previous_end is None else end - latency - previous_end
            if gap is not None and gap < 0:
                raise ValueError(f"Negative estimated gap in {iid}; no silent clipping is allowed")
            previous_end = end
            response = record["response"]
            turns.append({"turn": turn, "request": record["request"], "response": response,
                          "think_time_s": gap,
                          "completion_tokens": positive_count(response["usage"]["completion_tokens"]),
                          "source_request_id": rid, "source_file": str(path), "source_sha256": digest(path)})
        filename = f"agent_{number:03d}.json"
        write(output / filename, {"instance_id": iid, "exit_status": exit_status, "turns": turns})
        agents.append({"instance_id": iid, "file": filename, "sha256": digest(output / filename),
                       "turns": len(turns), "exit_status": exit_status,
                       "source_trajectory": str(trajectory_path), "source_sha256": digest(trajectory_path)})
    manifest = {"schema": SCHEMA, "timing_boundary": "server_log_filename_minus_elapsed",
                "timing_quality": "estimated", "timing_error_bound_s": None,
                "timing_caveat": "Logging/serialization overhead is unknown; gaps include tool, network and retry delays. Not exact client think time.",
                "selection": "first_N_in_source_instances_txt_no_outcome_filter",
                "source_run_config": config, "source_run_manifest": read(run_dir / "run_manifest.json"),
                "agents": agents, "instance_count": len(ids), "request_count": len(wanted)}
    write(output / "manifest.json", manifest)
    load_trace(output)
    return manifest


def load_trace(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    if manifest.get("schema") != SCHEMA or not manifest.get("agents"):
        raise ValueError("Unsupported or empty agent trace")
    ids = [agent["instance_id"] for agent in manifest["agents"]]
    if len(ids) != manifest["instance_count"] or len(ids) != len(set(ids)):
        raise ValueError("Duplicate or missing trace instances")
    for agent in manifest["agents"]:
        path = root / agent["file"]
        if Path(agent["file"]).name != agent["file"] or digest(path) != agent["sha256"]:
            raise ValueError(f"Trace content hash/path mismatch: {agent['instance_id']}")
        data = read(path)
        if data["instance_id"] != agent["instance_id"] or len(data["turns"]) != agent["turns"]:
            raise ValueError("Trace trajectory identity/count mismatch")
        for number, turn in enumerate(data["turns"]):
            gap, body = turn["think_time_s"], turn["request"]
            if turn["turn"] != number or (number == 0 and gap is not None) or (
                    number > 0 and (not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap < 0)):
                raise ValueError("Invalid turn order or recorded delay")
            if (not body.get("messages") or body.get("stream") or body.get("chain_id")
                    or body.get("chain_append_start") is not None or body.get("n", 1) != 1):
                raise ValueError("Replay requires non-streaming single-choice, full-history chat requests")
            if positive_count(turn["completion_tokens"]) != positive_count(turn["response"]["usage"]["completion_tokens"]):
                raise ValueError("Recorded completion count disagrees with original response")
    if sum(agent["turns"] for agent in manifest["agents"]) != manifest["request_count"]:
        raise ValueError("Trace total request count mismatch")
    return manifest


def replay_body(turn, model):
    body = copy.deepcopy(turn["request"])
    # Fix decode work, not generated text. Stop strings/EOS must not shorten it.
    for key in ("max_completion_tokens", "stop", "stop_token_ids"):
        body.pop(key, None)
    body.update(model=model, stream=False, max_tokens=positive_count(turn["completion_tokens"]),
                ignore_eos=True)
    return body


def replay_agent(agent, send, *, sleep=time.sleep, clock=time.perf_counter,
                 think_time_scale=1.0, after_turn=None):
    rows = []
    failed = False
    for turn in agent["turns"]:
        row = {"instance_id": agent["instance_id"], "turn": turn["turn"],
               "expected_completion_tokens": turn["completion_tokens"]}
        if failed:
            rows.append({**row, "status": "skipped_by_policy", "reason": "previous_turn_failed"})
            continue
        sleep((turn["think_time_s"] or 0) * think_time_scale)
        start = clock()
        try:
            response = send(turn)
            elapsed = clock() - start
            row["response"] = response
            tokens = positive_count(response["usage"]["completion_tokens"])
            if not response.get("choices") or tokens != turn["completion_tokens"]:
                raise ValueError(f"Response/token work mismatch: expected {turn['completion_tokens']}, got {tokens}")
            row.update(status="success", latency_s=elapsed, completion_tokens=tokens, response=response)
            if after_turn is not None:
                prune_start = clock()
                after_turn(turn)
                row["prefix_prune_elapsed_s"] = clock() - prune_start
        except Exception as exc:
            error_response = getattr(exc, "response", None)
            if error_response is not None:
                row["http_status"] = error_response.status_code
                row["error_response_body"] = error_response.text
            row.update(status="model_failed", latency_s=clock() - start,
                       error=f"{type(exc).__name__}: {exc}")
            failed = True
        rows.append(row)
    return rows


def grade(current, baseline, max_slowdown=1.10):
    if not math.isfinite(max_slowdown) or max_slowdown < 1:
        raise ValueError("Maximum slowdown must be finite and >= 1")
    for report in (current, baseline):
        if report.get("status") != "success":
            raise ValueError("Cannot grade incomplete/failed replay results")
        if any(not math.isfinite(report[key]) or report[key] <= 0 for key in ("latency_s_p95", "elapsed_s")):
            raise ValueError("Invalid baseline/replay timing metrics")
    if current["comparison_contract"] != baseline["comparison_contract"]:
        raise ValueError("Baseline mismatch: trace, model, concurrency or replay protocol changed")
    ratios = {"latency_p95_ratio": current["latency_s_p95"] / baseline["latency_s_p95"],
              "elapsed_ratio": current["elapsed_s"] / baseline["elapsed_s"]}
    return {"status": "passed" if max(ratios.values()) <= max_slowdown else "failed",
            "maximum_slowdown": max_slowdown, **ratios}


def run_replay(args):
    import httpx

    if not args.agent_trace or not args.agent_api_base or not args.agent_server_manifest:
        raise ValueError("agent_trace requires --agent_trace, --agent_api_base and --agent_server_manifest")
    if args.agent_concurrency <= 0 or not math.isfinite(args.agent_request_timeout) or args.agent_request_timeout <= 0:
        raise ValueError("Concurrency and request timeout must be positive")
    think_scale = getattr(args, "agent_think_time_scale", 1.0)
    keep_ratio = getattr(args, "agent_prefix_prune_keep_ratio", None)
    tokenizer_path = getattr(args, "agent_prefix_prune_tokenizer", None)
    trigger_tokens = getattr(args, "agent_prefix_prune_trigger_tokens", 8192)
    if trigger_tokens <= 0:
        raise ValueError("agent_prefix_prune_trigger_tokens must be positive")
    if not math.isfinite(think_scale) or think_scale < 0:
        raise ValueError("Think-time scale must be finite and nonnegative")
    if keep_ratio is not None and (not math.isfinite(keep_ratio) or not 0 <= keep_ratio < 1 or not tokenizer_path):
        raise ValueError("Tool pruning requires keep ratio in [0,1) and a tokenizer path")
    root = Path(args.agent_trace)
    manifest = load_trace(root)
    if manifest.get("timing_quality") != "measured" and not args.agent_allow_estimated_timing:
        raise ValueError("Legacy timing is estimated; explicitly pass --agent_allow_estimated_timing to replay it")
    server = read(args.agent_server_manifest)
    for field in ("model_path", "served_model_name", "engine_kwargs", "hardware"):
        if not server.get(field):
            raise ValueError(f"Server manifest must record {field}")
    gpu_ids = sorted(set(re.findall(r"GPU-[0-9a-fA-F-]+", json.dumps(server["hardware"]))))
    if not gpu_ids:
        raise ValueError("Server manifest hardware must identify actual GPU UUIDs")
    out = Path(args.output_root or os.getenv("SPARSEENGINE_OUTPUT_DIR", "outputs")) / "sparseengine_regression" / (
        args.run_id or time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    contract = {"trace_sha256": digest(root / "manifest.json"), "model_path": server["model_path"],
                "model_config": server.get("model_config"), "concurrency": args.agent_concurrency,
                "gpu_uuids": gpu_ids, "engine_kwargs": server["engine_kwargs"], "backend": server.get("backend"),
                "protocol": "closed_loop_recorded_inputs_fixed_decode_count_v1",
                "timing_boundary": "client_http_nonstreaming", "request_timeout_s": args.agent_request_timeout}
    if think_scale != 1.0 or keep_ratio is not None:
        contract.update(protocol="closed_loop_recorded_inputs_fixed_decode_count_v2",
                        think_time_scale=think_scale,
                        prefix_prune=({"policy": "kvzip_global", "target": "tool_results",
                                       "schedule": "accumulated_tool_tokens", "trigger_tokens": trigger_tokens,
                                       "keep_ratio": keep_ratio}
                                      if keep_ratio is not None else None),
                        elapsed_boundary="all_requests_plus_pruning_and_scaled_think_time")
    write(out / "resolved_manifest.json", {"trace": manifest, "server": server, "comparison_contract": contract})
    if args.dry_run:
        write(out / "grade_summary.json", {"status": "skipped_by_policy", "reason": "dry_run", "layer": "agent_trace"})
        return 0
    headers = {}
    key = os.getenv(args.agent_api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    all_rows = []
    if keep_ratio is not None:
        # Resolve Transformers' lazy modules once on the main thread. Concurrent
        # first imports can expose a partially initialized AutoTokenizer module.
        from transformers import AutoTokenizer
        from sparseengine.entrypoints.openai.protocol.chat import ChatCompletionRequest
        from sparseengine.entrypoints.openai.reasoning import detect_reasoning_capabilities
        from sparseengine.entrypoints.openai.render import _chat_request_prompt

    def worker(entry):
        agent = read(root / entry["file"])
        prune = None
        if keep_ratio is not None:
            from benchmark.swe_bench_lite.prefix_prune_client import PrefixPruneClient
            prune = PrefixPruneClient(api_base=args.agent_api_base, tokenizer_path=tokenizer_path,
                                      keep_ratio=keep_ratio, trigger_tokens=trigger_tokens,
                                      events_path=out / (entry["file"] + ".prune.jsonl"))
        with httpx.Client(timeout=args.agent_request_timeout, trust_env=False, headers=headers) as client:
            def send(turn):
                response = client.post(args.agent_api_base.rstrip("/") + "/chat/completions",
                                       json=replay_body(turn, server["served_model_name"]))
                response.raise_for_status()
                return response.json()
            def after_turn(turn):
                # Prune the recorded input that was actually prefetched; never the
                # generated response, which is not substituted into later inputs.
                prune._maybe_prune(replay_body(turn, server["served_model_name"]))
            rows = replay_agent(agent, send, think_time_scale=think_scale,
                                after_turn=after_turn if prune is not None else None)
        # One writer per agent; no concurrent appends and no lost results on another worker's failure.
        write(out / entry["file"], {"instance_id": agent["instance_id"], "requests": rows})
        return rows

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.agent_concurrency) as pool:
            for rows in pool.map(worker, manifest["agents"]):
                all_rows.extend(rows)
    except Exception as exc:
        write(out / "grade_summary.json", {"layer": "agent_trace", "status": "failed",
                                          "error": f"{type(exc).__name__}: {exc}"})
        raise
    elapsed = time.perf_counter() - started
    summary = {**http_trace_summary(all_rows, elapsed), "comparison_contract": contract,
               "layer": "agent_trace", "server_manifest": server,
               "source_timing_quality": manifest["timing_quality"]}
    write(out / "agent_trace.json", summary)
    result = {"status": "failed" if summary["status"] != "success" else "baseline_created",
              "layer": "agent_trace"}
    if args.agent_baseline:
        try:
            result = {"layer": "agent_trace", **grade(summary, read(args.agent_baseline), args.agent_max_slowdown)}
        except ValueError as exc:
            result.update(status="failed", error=str(exc))
    write(out / "grade_summary.json", result)
    print(json.dumps({"output": str(out), **result}))
    return 1 if result["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description="Export a fully recorded MiniSWE run for agent_trace regression")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-instances", type=int, default=100)
    parser.add_argument("--legacy-server-requests", type=Path, action="append",
                        help="Explicit legacy request-log directories; enables estimated timing import")
    args = parser.parse_args()
    manifest = (export_legacy(args.run_dir, args.legacy_server_requests, args.output, args.expected_instances)
                if args.legacy_server_requests else export_trace(args.run_dir, args.output, args.expected_instances))
    print(json.dumps({key: manifest[key] for key in ("instance_count", "request_count")}))


if __name__ == "__main__":
    main()
