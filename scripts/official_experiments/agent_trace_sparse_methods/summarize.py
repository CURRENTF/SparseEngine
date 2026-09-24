#!/usr/bin/env python3
"""Validate completed matched agent-trace cases and write compact results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


CASES = (
    ("vanilla-prefix", 40), ("omnikv-prefix", 40), ("quest-prefix", 40),
    ("snapkv-chain", 40), ("h2o-chain", 40),
)
SHARED_ENGINE_KEYS = (
    "tensor_parallel_size", "expert_parallel_size", "data_parallel_size",
    "gpu_memory_utilization", "max_model_len", "max_num_seqs_in_batch",
    "max_decoding_seqs", "max_num_seqs_in_gpu", "max_num_batched_tokens",
    "engine_prefill_chunk_size", "decode_graph", "decode_graph_capture_sizes",
)


def read(path):
    return json.loads(path.read_text())


def preemption_evidence(case, method, concurrency, complete_log_cases, expected_requests):
    snapshot_path = case / "worker_load_after_replay.json"
    if snapshot_path.exists():
        snapshot = read(snapshot_path)
        if snapshot["total_preemptions"] or snapshot["total_recompute_replays"]:
            raise ValueError(f"Active requests were preempted: {case}")
        return {"source": "final_worker_load", "total_preemptions": 0,
                "total_recompute_replays": 0}
    if (method, concurrency) not in complete_log_cases:
        raise ValueError(f"Missing final worker-load snapshot: {case}")
    log_path = case / f"run/{method}/full/server.log"
    counts = {"request_finish": 0, "preemption": 0, "recompute": 0}
    with log_path.open(errors="replace") as server_log:
        for line in server_log:
            counts["request_finish"] += "request_finish id=" in line
            counts["preemption"] += "驱逐请求 id =" in line
            counts["recompute"] += "recompute_replay_start" in line
    if counts != {"request_finish": expected_requests, "preemption": 0, "recompute": 0}:
        raise ValueError(f"Incomplete or preempted server log: {log_path}: {counts}")
    return {"source": "complete_server_log", "request_finish_count": expected_requests,
            "total_preemptions": 0, "total_recompute_replays": 0}


def case_result(root, model, method, concurrency, expected_requests, expected_trace_hash,
                complete_log_cases=frozenset()):
    case = root / f"{model}_{method}_c{concurrency}"
    summary = read(case / "sparseengine_regression/replay/agent_trace.json")
    server = summary["server_manifest"]
    contract = summary["comparison_contract"]
    engine = server["engine_kwargs"]
    if (summary["status"] != "success" or summary["request_count"] != expected_requests
            or summary["unsuccessful_request_count"] != 0
            or summary["completion_tokens"] != contract["expected_completion_tokens"]
            or (expected_trace_hash is not None and contract["trace_sha256"] != expected_trace_hash)
            or summary["cache_reuse"]["cached_tokens"] <= 0
            or engine["max_num_seqs_in_batch"] != concurrency
            or engine["max_decoding_seqs"] != concurrency
            or engine["max_num_seqs_in_gpu"] * 2 != concurrency * 3):
        raise ValueError(f"Incomplete or protocol-invalid case: {case}")
    if method.endswith("-chain") and summary["cache_reuse"]["chain_resumed_requests"] <= 0:
        raise ValueError(f"Chain Cache did not resume: {case}")
    if contract.get("protocol") != "closed_loop_forced_recorded_answers_synthetic_wait_v1":
        raise ValueError(f"Wrong replay protocol: {case}")
    return {
        "model": model,
        "method": method, "concurrency": concurrency,
        "device": {"gpu_indices": server["cuda_visible_devices"], "gpu_uuids": contract["gpu_uuids"]},
        "launch_args": {"model_path": server["model_path"], "served_model_name": server["served_model_name"],
                        "engine_kwargs": engine, "agent_concurrency": concurrency,
                        "synthetic_think_time_max_s": contract["synthetic_think_time_max_s"],
                        "synthetic_think_time_seed": contract["synthetic_think_time_seed"],
                        "trace_sha256": contract["trace_sha256"],
                        "forced_workload_sha256": contract["forced_workload_sha256"]},
        "final_result": {key: summary[key] for key in (
            "request_count", "completion_tokens", "elapsed_s", "output_token_throughput_tps",
            "latency_s_p50", "latency_s_p95", "latency_s_p99", "cache_reuse")},
        "preemption_evidence": preemption_evidence(
            case, method, concurrency, complete_log_cases, expected_requests),
        "git_commit": server["git_commit"],
    }


def preempted_capacity_case(root, model, method, concurrency, snapshot_path):
    case = root / f"{model}_{method}_c{concurrency}"
    if snapshot_path.resolve().parent != case.resolve():
        raise ValueError("Preemption snapshot must belong to the failed case")
    snapshot = read(snapshot_path)
    server = read(case / f"run/{method}/full/server_manifest.json")
    engine = server["engine_kwargs"]
    if (snapshot["total_preemptions"] <= 0
            or snapshot["max_num_seqs_in_batch"] != concurrency
            or snapshot["max_decoding_seqs"] != concurrency
            or snapshot["max_num_seqs_in_gpu"] * 2 != concurrency * 3
            or engine["max_num_seqs_in_batch"] != concurrency
            or engine["max_decoding_seqs"] != concurrency
            or engine["max_num_seqs_in_gpu"] * 2 != concurrency * 3):
        raise ValueError(f"Invalid preemption evidence: {snapshot_path}")
    return {
        "model": model, "method": method, "concurrency": concurrency,
        "status": "preempted_before_complete_replay",
        "device": {"gpu_indices": server["cuda_visible_devices"]},
        "model_path": server["model_path"],
        "engine_kwargs": engine,
        "total_preemptions": snapshot["total_preemptions"],
        "total_recompute_replays": snapshot["total_recompute_replays"],
        "waiting_requests_at_snapshot": snapshot["waiting_requests"],
        "decoding_requests_at_snapshot": snapshot["decoding_requests"],
        "git_commit": server["git_commit"],
    }


def failed_replay_case(root, model, method, concurrency, expected_requests):
    case = root / f"{model}_{method}_c{concurrency}"
    summary = read(case / "sparseengine_regression/replay/agent_trace.json")
    server = summary["server_manifest"]
    engine = server["engine_kwargs"]
    if (summary["status"] != "failed" or summary["request_count"] != expected_requests
            or summary["unsuccessful_request_count"] <= 0
            or engine["max_num_seqs_in_batch"] != concurrency
            or engine["max_decoding_seqs"] != concurrency
            or engine["max_num_seqs_in_gpu"] * 2 != concurrency * 3):
        raise ValueError(f"Invalid failed replay: {case}")
    counts = Counter()
    errors = Counter()
    agent_files = list((case / "sparseengine_regression/replay").glob("agent_[0-9][0-9][0-9].json"))
    if len(agent_files) != 100:
        raise ValueError(f"Incomplete per-agent failure evidence: {case}")
    for path in agent_files:
        for request in read(path)["requests"]:
            status = request["status"]
            counts[status] += 1
            if status != "model_failed":
                continue
            body = request.get("error_response_body", "")
            if request.get("http_status") == 503 and '"code":"chain_capacity_unavailable"' in body:
                errors["chain_capacity_unavailable_503"] += 1
            elif request.get("http_status") == 500 and "KV cache is too small for the sole remaining decode request" in body:
                errors["decode_no_forward_progress_500"] += 1
            elif request.get("http_status") is None and "Connection refused" in request.get("error", ""):
                errors["connection_refused_after_server_failure"] += 1
            else:
                raise ValueError(f"Unexpected failed request in {path}: {request}")
    if (sum(counts.values()) != expected_requests
            or counts["model_failed"] + counts["skipped_by_policy"] != summary["unsuccessful_request_count"]
            or counts["model_failed"] != sum(errors.values())
            or errors["chain_capacity_unavailable_503"] <= 0):
        raise ValueError(f"Failed replay evidence does not match aggregate: {case}")
    return {
        "model": model, "method": method, "concurrency": concurrency,
        "status": "failed_replay",
        "device": {"gpu_indices": server["cuda_visible_devices"]},
        "model_path": server["model_path"], "engine_kwargs": engine,
        "request_status_counts": dict(counts), "error_counts": dict(errors),
        "elapsed_s": summary["elapsed_s"], "git_commit": server["git_commit"],
    }


def validate_pair(base, sparse):
    for field in ("model_path", "trace_sha256", "forced_workload_sha256",
                  "agent_concurrency", "synthetic_think_time_max_s", "synthetic_think_time_seed"):
        if base["launch_args"][field] != sparse["launch_args"][field]:
            raise ValueError(f"Unmatched pair: {field}")
    if base["device"] != sparse["device"]:
        raise ValueError("Unmatched pair: device")
    for key in SHARED_ENGINE_KEYS:
        if base["launch_args"]["engine_kwargs"][key] != sparse["launch_args"]["engine_kwargs"][key]:
            raise ValueError(f"Unmatched pair: {key}")
    b, s = base["final_result"], sparse["final_result"]
    if b["completion_tokens"] != s["completion_tokens"]:
        raise ValueError("Unmatched pair: completion token work")
    return {
        "model_path": base["launch_args"]["model_path"],
        "method": sparse["method"], "concurrency": sparse["concurrency"],
        "replay_elapsed_speedup_vs_vanilla": b["elapsed_s"] / s["elapsed_s"],
        "output_token_throughput_speedup_vs_vanilla":
            s["output_token_throughput_tps"] / b["output_token_throughput_tps"],
        "request_p95_latency_speedup_vs_vanilla": b["latency_s_p95"] / s["latency_s_p95"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path,
                        help="Validate the selected trace hash and request count; defaults to the historical 6,490-request trace.")
    parser.add_argument("--case", action="append", default=[], metavar="METHOD:CONCURRENCY",
                        help="Completed cases to package; defaults to the historical C40 five-method set.")
    parser.add_argument("--vanilla-c100-preemption-snapshot", type=Path)
    parser.add_argument("--failed-c100-replay", action="append", default=[],
                        choices=("snapkv-chain", "h2o-chain"))
    parser.add_argument("--complete-log-preemption-evidence", action="append", default=[],
                        metavar="METHOD:CONCURRENCY")
    args = parser.parse_args()
    if args.trace_manifest:
        trace_bytes = args.trace_manifest.read_bytes()
        trace = json.loads(trace_bytes)
        if trace.get("instance_count") != 100 or not isinstance(trace.get("request_count"), int) or trace["request_count"] <= 0:
            raise ValueError("Invalid 100-agent trace manifest")
        expected_requests = trace["request_count"]
        expected_trace_hash = hashlib.sha256(trace_bytes).hexdigest()
    else:
        expected_requests = 6490
        expected_trace_hash = None
    cases = []
    for value in args.case:
        method, separator, concurrency = value.rpartition(":")
        if (not separator or method not in {name for name, _ in CASES}
                or not concurrency.isdigit() or int(concurrency) <= 0):
            raise ValueError(f"Invalid case: {value}")
        cases.append((method, int(concurrency)))
    cases = tuple(cases) if cases else CASES
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate case")
    complete_log_cases = set()
    for value in args.complete_log_preemption_evidence:
        method, separator, concurrency = value.rpartition(":")
        if not separator or not concurrency.isdigit() or (method, int(concurrency)) not in cases:
            raise ValueError(f"Unknown complete-log preemption evidence case: {value}")
        complete_log_cases.add((method, int(concurrency)))
    results = []
    comparisons = []
    capacity_failures = []
    for model in ("glm47",):
        by_case = {}
        for method, concurrency in cases:
            row = case_result(args.run_root, model, method, concurrency,
                              expected_requests, expected_trace_hash, complete_log_cases)
            results.append(row)
            by_case[method, concurrency] = row
        if args.vanilla_c100_preemption_snapshot:
            capacity_failures.append(preempted_capacity_case(
                args.run_root, model, "vanilla-prefix", 100,
                args.vanilla_c100_preemption_snapshot,
            ))
        for method in args.failed_c100_replay:
            capacity_failures.append(failed_replay_case(
                args.run_root, model, method, 100, expected_requests))
        for method, concurrency in cases:
            if method != "vanilla-prefix" and ("vanilla-prefix", concurrency) in by_case:
                comparisons.append(validate_pair(by_case["vanilla-prefix", concurrency],
                                                 by_case[method, concurrency]))
    commits = {row["git_commit"] for row in results + capacity_failures}
    if len(commits) != 1:
        raise ValueError("Experiment cases came from different Git commits")
    markdown = args.output.with_name("RESULTS.md") if args.output.name == "results.json" else None
    if args.output.exists() or (markdown is not None and markdown.exists()):
        raise FileExistsError(args.output if args.output.exists() else markdown)
    args.output.write_text(json.dumps({
        "results": results, "comparisons": comparisons,
        "capacity_failures": capacity_failures,
    }, indent=2) + "\n")
    if markdown is not None:
        speedups = {(row["model_path"].rsplit("/", 1)[-1], row["method"], row["concurrency"]):
                    row["replay_elapsed_speedup_vs_vanilla"] for row in comparisons}
        lines = ["# Agent trajectory serving results", "",
                 f"The 100-agent, {expected_requests:,}-request trajectory was generated by GLM-4.7-Flash.",
                 "Only GLM-4.7-Flash is evaluated in this campaign.",
                 "Tool waits are deterministic synthetic pauses in [0, 2) seconds.",
                 "A preempted or incomplete run has no end-to-end throughput result or matched speedup.",
                 "See [README.md](README.md) for the cache and timing protocol and [results.json](results.json) for devices and launch arguments.",
                 "", f"Git commit: `{next(iter(commits))}`", "",
                 "| Model | Method | Concurrency | Replay hours | Output tokens/s | Speedup vs vanilla | Cached tokens | Chain resumes |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in results:
            result = row["final_result"]
            model_name = row["launch_args"]["model_path"].rsplit("/", 1)[-1]
            speedup = (1.0 if row["method"] == "vanilla-prefix" else
                       speedups.get((model_name, row["method"], row["concurrency"])))
            speedup_label = f"{speedup:.3f}×" if speedup is not None else "n/a"
            lines.append(
                f"| {model_name} | {row['method']} | {row['concurrency']} | "
                f"{result['elapsed_s'] / 3600:.2f} | {result['output_token_throughput_tps']:.1f} | "
                f"{speedup_label} | "
                f"{result['cache_reuse']['cached_tokens']} | "
                f"{result['cache_reuse']['chain_resumed_requests']} |"
            )
        if capacity_failures:
            lines.extend(["", "## Capacity failures", ""])
            for failure in capacity_failures:
                if failure["status"] == "preempted_before_complete_replay":
                    lines.append(
                        f"- {failure['method']} C{failure['concurrency']}: "
                        f"{failure['total_preemptions']} active-request preemptions and "
                        f"{failure['total_recompute_replays']} recompute replays observed "
                        "before stopping the incomplete replay."
                    )
                else:
                    counts = failure["error_counts"]
                    lines.append(
                        f"- {failure['method']} C{failure['concurrency']}: "
                        f"{failure['request_status_counts']['model_failed']} failed requests "
                        f"({counts.get('chain_capacity_unavailable_503', 0)} chain-capacity 503, "
                        f"{counts.get('decode_no_forward_progress_500', 0)} decode-capacity 500); "
                        "the complete replay has no valid throughput."
                    )
        markdown.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
