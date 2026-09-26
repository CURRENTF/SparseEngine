"""Export MiniSWE HTTP traces and replay closed-loop agent workloads.

One worker owns a trajectory until it ends. Recorded prompts are immutable.
The forced-replay protocol makes each generated continuation match the next
recorded prompt at the model token boundary.
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


def select_legacy_ids(run_dir, all_ids, count, selection, excluded_ids=()):
    excluded_ids = set(excluded_ids)
    if excluded_ids - set(all_ids):
        raise ValueError(f"Excluded instances absent from source order: {sorted(excluded_ids - set(all_ids))}")
    candidates = [(index, iid) for index, iid in enumerate(all_ids) if iid not in excluded_ids]
    if count <= 0 or len(candidates) < count:
        raise ValueError("Not enough instances for the requested selection")
    if selection == "first":
        return [iid for _, iid in candidates[:count]]
    scores = []
    for index, iid in candidates:
        candidates = list((run_dir / "batches").glob(f"batch_*/{iid}/{iid}.traj.json"))
        if len(candidates) != 1:
            raise ValueError(f"Missing or duplicate trajectory for {iid}")
        trajectory = read(candidates[0])
        responses = [message["extra"]["response"] for message in trajectory["messages"]
                     if isinstance(message.get("extra", {}).get("response"), dict)]
        if trajectory["info"]["model_stats"]["api_calls"] != len(responses) or not responses:
            raise ValueError(f"Trajectory API call accounting mismatch for {iid}")
        if selection == "longest_turns":
            score = len(responses)
        else:
            prompt_tokens = [positive_count(response["usage"]["prompt_tokens"])
                             for response in responses]
            score = sum(prompt_tokens) if selection == "total_prompt_tokens" else max(prompt_tokens)
        scores.append((-score, index, iid))
    return [iid for _, _, iid in sorted(scores)[:count]]


def export_legacy(run_dir, request_dirs, output, expected_instances=100, selection="first",
                  excluded_instance_ids=()):
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
    if selection not in ("first", "longest_turns", "total_prompt_tokens", "max_prompt_tokens"):
        raise ValueError(f"Unsupported legacy trajectory selection: {selection}")
    ids = select_legacy_ids(run_dir, all_ids, expected_instances, selection, excluded_instance_ids)
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
                "selection": ("first_N_in_source_instances_txt_no_outcome_filter" if selection == "first"
                              else f"top_N_{selection}_descending_source_order_tiebreak_no_outcome_filter"),
                "source_run_config": config, "source_run_manifest": read(run_dir / "run_manifest.json"),
                "agents": agents, "instance_count": len(ids), "request_count": len(wanted)}
    if excluded_instance_ids:
        manifest["excluded_instance_ids"] = sorted(set(excluded_instance_ids))
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
    if "benchmark_forced_token_ids" in turn:
        body["benchmark_forced_token_ids"] = turn["benchmark_forced_token_ids"]
    return body


def post_with_chain_recovery(client, url, body):
    """Retry one evicted chain as a full prompt, matching the MiniSWE client."""
    response = client.post(url, json=body)
    recovered = False
    if response.status_code == 410 and body.get("chain_id"):
        detail = response.json().get("detail")
        if isinstance(detail, dict) and detail.get("code") == "chain_gone":
            retry_body = {key: value for key, value in body.items()
                          if key not in ("chain_id", "chain_append_start")}
            response = client.post(url, json=retry_body)
            recovered = True
    response.raise_for_status()
    return response.json(), recovered


def prepare_forced_agent(agent, tokenizer, model):
    """Derive exact target-model output tokens from adjacent recorded prompts."""
    from sparseengine.engine.input_processor import tokenize_text_prompt
    from sparseengine.entrypoints.openai.protocol.chat import ChatCompletionRequest
    from sparseengine.entrypoints.openai.render import (
        _chat_prompt, _chat_request_append_prompt, _chat_request_prompt,
        resolve_chat_template_kwargs, resolve_chat_tools,
    )

    turns = agent["turns"]
    prepared = []
    for current, following in zip(turns, turns[1:]):
        request = ChatCompletionRequest.model_validate(replay_body(current, model))
        next_request = ChatCompletionRequest.model_validate(replay_body(following, model))
        prefix_count = len(request.messages)
        if (next_request.messages[:prefix_count] != request.messages
                or len(next_request.messages) <= prefix_count
                or next_request.messages[prefix_count].role != "assistant"):
            raise ValueError(f"Nonappendable agent transcript: {agent['instance_id']} turn {current['turn']}")
        prompt = tokenize_text_prompt(tokenizer, _chat_request_prompt(tokenizer, request))
        through_answer = tokenize_text_prompt(tokenizer, _chat_prompt(
            tokenizer, next_request.messages[:prefix_count + 1],
            resolve_chat_template_kwargs(request), resolve_chat_tools(request),
            add_generation_prompt=False,
        ))
        if through_answer[:len(prompt)] != prompt or len(through_answer) == len(prompt):
            raise ValueError(f"Recorded answer is not a token-prefix extension: {agent['instance_id']} turn {current['turn']}")
        next_prompt = tokenize_text_prompt(tokenizer, _chat_request_prompt(tokenizer, next_request))
        next_request.chain_id = "validation"
        next_request.chain_append_start = prefix_count + 1
        append = tokenizer.encode(_chat_request_append_prompt(tokenizer, next_request),
                                  add_special_tokens=False)
        if through_answer + append != next_prompt:
            raise ValueError(f"Chain append does not reconstruct the next prompt: {agent['instance_id']} turn {current['turn']}")
        forced = through_answer[len(prompt):]
        prepared.append({"token_ids": forced, "chain_append_start": prefix_count + 1})
    prepared.append({"token_ids": None, "chain_append_start": None})
    return prepared


def synthetic_think_time(instance_id, turn, seed, maximum_s):
    """Return a stable per-turn pause shared by all replay variants."""
    if turn == 0:
        return 0.0
    key = f"{seed}\0{instance_id}\0{turn}".encode()
    fraction = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return fraction * maximum_s


def replay_agent(agent, send, *, sleep=time.sleep, clock=time.perf_counter,
                 think_time_scale=1.0, after_turn=None, delay_fn=None):
    rows = []
    failed = False
    for turn in agent["turns"]:
        row = {"instance_id": agent["instance_id"], "turn": turn["turn"],
               "expected_completion_tokens": turn["completion_tokens"]}
        if failed:
            rows.append({**row, "status": "skipped_by_policy", "reason": "previous_turn_failed"})
            continue
        delay = (delay_fn(agent["instance_id"], turn["turn"])
                 if delay_fn is not None else (turn["think_time_s"] or 0) * think_time_scale)
        row["replay_wait_s"] = delay
        sleep(delay)
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
    synthetic_wait_max = getattr(args, "agent_synthetic_think_time_max_s", None)
    synthetic_wait_seed = getattr(args, "agent_synthetic_think_time_seed", 42)
    force_recorded = getattr(args, "agent_force_recorded_responses", False)
    if getattr(args, "agent_forced_workload", None) and not force_recorded:
        raise ValueError("--agent_forced_workload requires --agent_force_recorded_responses")
    keep_ratio = getattr(args, "agent_prefix_prune_keep_ratio", None)
    tokenizer_path = getattr(args, "agent_prefix_prune_tokenizer", None)
    trigger_tokens = getattr(args, "agent_prefix_prune_trigger_tokens", 8192)
    if trigger_tokens <= 0:
        raise ValueError("agent_prefix_prune_trigger_tokens must be positive")
    if not math.isfinite(think_scale) or think_scale < 0:
        raise ValueError("Think-time scale must be finite and nonnegative")
    if synthetic_wait_max is not None and (
            not math.isfinite(synthetic_wait_max) or synthetic_wait_max < 0
            or think_scale != 1.0):
        raise ValueError("Synthetic think time requires a finite nonnegative maximum and unscaled recorded timing")
    if keep_ratio is not None and (not math.isfinite(keep_ratio) or not 0 <= keep_ratio < 1 or not tokenizer_path):
        raise ValueError("Tool pruning requires keep ratio in [0,1) and a tokenizer path")
    root = Path(args.agent_trace)
    manifest = load_trace(root)
    if (synthetic_wait_max is None and manifest.get("timing_quality") != "measured"
            and not args.agent_allow_estimated_timing):
        raise ValueError("Legacy timing is estimated; explicitly pass --agent_allow_estimated_timing to replay it")
    server = read(args.agent_server_manifest)
    for field in ("model_path", "served_model_name", "engine_kwargs", "hardware"):
        if not server.get(field):
            raise ValueError(f"Server manifest must record {field}")
    gpu_ids = sorted(set(re.findall(r"GPU-[0-9a-fA-F-]+", json.dumps(server["hardware"]))))
    if not gpu_ids:
        raise ValueError("Server manifest hardware must identify actual GPU UUIDs")
    cache_mode = server["engine_kwargs"].get("prefix_cache_mode")
    if cache_mode == "chain" and not force_recorded:
        raise ValueError("Chain Cache replay requires server-side forced recorded responses")
    prepared_agents = {}
    forced_digest = hashlib.sha256()
    expected_tokens = 0
    if force_recorded:
        forced_file = getattr(args, "agent_forced_workload", None)
        cached = read(forced_file) if forced_file else None
        if cached is not None and (
                cached.get("schema") != "agent_forced_workload_v1"
                or cached.get("trace_sha256") != digest(root / "manifest.json")
                or cached.get("model_path") != server["model_path"]):
            raise ValueError("Prepared forced workload does not match trace and model")
        if cached is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(server["model_path"], use_fast=True)
        vocabulary = int(server["model_config"]["vocab_size"])
        for entry in manifest["agents"]:
            agent = read(root / entry["file"])
            prepared = (cached["agents"][entry["file"]] if cached is not None
                        else prepare_forced_agent(agent, tokenizer, server["served_model_name"]))
            if len(prepared) != len(agent["turns"]):
                raise ValueError(f"Prepared forced workload turn count mismatch: {entry['instance_id']}")
            for index, (turn, spec) in enumerate(zip(agent["turns"], prepared)):
                ids = spec["token_ids"]
                if (index < len(prepared) - 1 and (
                        ids is None or spec["chain_append_start"] != len(turn["request"]["messages"]) + 1)):
                    raise ValueError(f"Missing forced continuation: {entry['instance_id']} turn {turn['turn']}")
                if ids is not None and (not isinstance(ids, list) or not ids
                                        or any(type(token_id) is not int or token_id < 0 or token_id >= vocabulary
                                               for token_id in ids)
                                        or len(ids) > int(server["engine_kwargs"]["max_model_len"])):
                    raise ValueError(f"Forced answer exceeds model contract: {entry['instance_id']} turn {turn['turn']}")
                count = len(ids) if ids is not None else turn["completion_tokens"]
                expected_tokens += count
                forced_digest.update(json.dumps([entry["instance_id"], turn["turn"], ids, count],
                                                separators=(",", ":")).encode())
            prepared_agents[entry["file"]] = prepared
        if cached is not None and cached.get("forced_workload_sha256") != forced_digest.hexdigest():
            raise ValueError("Prepared forced workload content digest mismatch")
    out = Path(args.output_root or os.getenv("SPARSEENGINE_OUTPUT_DIR", "outputs")) / "sparseengine_regression" / (
        args.run_id or time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    protocol = ("closed_loop_forced_recorded_answers" if force_recorded
                else "closed_loop_recorded_inputs_fixed_decode_count")
    if synthetic_wait_max is not None:
        protocol += "_synthetic_wait"
    protocol += "_v2" if think_scale != 1.0 or keep_ratio is not None else "_v1"
    contract = {"trace_sha256": digest(root / "manifest.json"), "model_path": server["model_path"],
                "model_config": server.get("model_config"), "concurrency": args.agent_concurrency,
                "gpu_uuids": gpu_ids, "engine_kwargs": server["engine_kwargs"], "backend": server.get("backend"),
                "protocol": protocol,
                "timing_boundary": "client_http_nonstreaming", "request_timeout_s": args.agent_request_timeout}
    if force_recorded:
        contract.update(forced_workload_sha256=forced_digest.hexdigest(),
                        expected_completion_tokens=expected_tokens,
                        forced_response_scope="all_nonterminal_agent_turns",
                        chain_gone_policy="one_full_prompt_retry_on_410")
    if synthetic_wait_max is not None:
        contract.update(think_time_policy="deterministic_uniform",
                        synthetic_think_time_min_s=0.0,
                        synthetic_think_time_max_s=synthetic_wait_max,
                        synthetic_think_time_seed=synthetic_wait_seed,
                        elapsed_boundary="all_requests_plus_synthetic_wait")
    if think_scale != 1.0 or keep_ratio is not None:
        contract.update(think_time_scale=think_scale,
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
        if force_recorded:
            prepared = prepared_agents[entry["file"]]
            for index, (turn, spec) in enumerate(zip(agent["turns"], prepared)):
                if spec["token_ids"] is not None:
                    turn["benchmark_forced_token_ids"] = spec["token_ids"]
                    turn["completion_tokens"] = len(spec["token_ids"])
                turn["chain_append_start"] = prepared[index - 1]["chain_append_start"] if index else None
        prune = None
        if keep_ratio is not None:
            from benchmark.swe_bench_lite.prefix_prune_client import PrefixPruneClient
            prune = PrefixPruneClient(api_base=args.agent_api_base, tokenizer_path=tokenizer_path,
                                      keep_ratio=keep_ratio, trigger_tokens=trigger_tokens,
                                      events_path=out / (entry["file"] + ".prune.jsonl"))
        with httpx.Client(timeout=args.agent_request_timeout, trust_env=False, headers=headers) as client:
            chain_id = None
            def send(turn):
                nonlocal chain_id
                body = replay_body(turn, server["served_model_name"])
                if cache_mode == "chain":
                    if chain_id is not None:
                        body["chain_id"] = chain_id
                        body["chain_append_start"] = turn["chain_append_start"]
                url = args.agent_api_base.rstrip("/") + "/chat/completions"
                if cache_mode == "chain":
                    result, recovered = post_with_chain_recovery(client, url, body)
                else:
                    response = client.post(url, json=body)
                    response.raise_for_status()
                    result, recovered = response.json(), False
                if cache_mode == "chain":
                    status = result.get("chain_status")
                    if status != ("created" if chain_id is None or recovered else "resumed"):
                        raise ValueError(f"Chain did not match recorded continuation: status={status!r}")
                    chain_id = result.get("chain_id")
                    if not chain_id:
                        raise ValueError("Chain response omitted chain_id")
                    if recovered:
                        result["benchmark_chain_recovered_from_410"] = True
                return result
            def after_turn(turn):
                # Prune the recorded input that was actually prefetched; never the
                # generated response, which is not substituted into later inputs.
                prune._maybe_prune(replay_body(turn, server["served_model_name"]))
            delay_fn = (lambda iid, turn: synthetic_think_time(
                iid, turn, synthetic_wait_seed, synthetic_wait_max)) if synthetic_wait_max is not None else None
            rows = replay_agent(agent, send, think_time_scale=think_scale, delay_fn=delay_fn,
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
    cached_counts = [int((row.get("response", {}).get("usage", {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens", row.get("response", {}).get("usage", {}).get("reused_tokens", 0)))
        for row in all_rows if row["status"] == "success"]
    if any(count < 0 for count in cached_counts):
        raise ValueError("Negative cached-token count in replay response")
    summary["cache_reuse"] = {
        "successful_requests_with_cached_tokens": sum(count > 0 for count in cached_counts),
        "cached_tokens": sum(cached_counts),
        "max_cached_tokens": max(cached_counts, default=0),
        "chain_resumed_requests": sum(row.get("response", {}).get("chain_status") == "resumed"
                                      for row in all_rows if row["status"] == "success"),
        "chain_recovery_requests": sum(bool(row.get("response", {}).get("benchmark_chain_recovered_from_410"))
                                       for row in all_rows if row["status"] == "success"),
    }
    if getattr(args, "agent_require_cache_hit", False) and not summary["cache_reuse"]["cached_tokens"]:
        summary["status"] = "failed"
        summary["cache_reuse"]["error"] = "No cached tokens observed in successful responses"
    if force_recorded and summary["status"] == "success" and summary.get("completion_tokens") != expected_tokens:
        summary["status"] = "failed"
        summary["forced_workload_error"] = "Measured completion count differs from the target-model forced workload"
    if cache_mode == "chain" and summary["status"] == "success" and not summary["cache_reuse"]["chain_resumed_requests"]:
        summary["status"] = "failed"
        summary["cache_reuse"]["error"] = "No resumed Chain Cache request observed"
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
    parser.add_argument("--selection", choices=("first", "longest_turns", "total_prompt_tokens",
                                                 "max_prompt_tokens"), default="first",
                        help="Legacy trajectory ranking; ties retain source instances.txt order")
    parser.add_argument("--exclude-instance-id", action="append", default=[],
                        help="Exclude a named source trajectory before ranking and refill from the next candidate")
    args = parser.parse_args()
    if args.selection != "first" and not args.legacy_server_requests:
        parser.error("--selection requires --legacy-server-requests")
    if args.exclude_instance_id and not args.legacy_server_requests:
        parser.error("--exclude-instance-id requires --legacy-server-requests")
    manifest = (export_legacy(args.run_dir, args.legacy_server_requests, args.output,
                              args.expected_instances, args.selection, args.exclude_instance_id)
                if args.legacy_server_requests else export_trace(args.run_dir, args.output, args.expected_instances))
    print(json.dumps({key: manifest[key] for key in ("instance_count", "request_count")}))


if __name__ == "__main__":
    main()
