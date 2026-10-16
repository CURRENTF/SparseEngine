"""Validate matched probe records and summarize exported Nsight NVTX kernel sums."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def family(name):
    if "prefill_score" in name or "prefill_logit_score" in name:
        return "reconstruction_or_prefill_score"
    if "flash::" in name or "flash_attn" in name:
        return "attention"
    if "nvjet" in name or "gemm" in name.lower() or "xmma" in name:
        return "gemm"
    return "other"


def profile_summary(case):
    with (case / "stats_nvtx_kern_sum.csv").open() as f:
        rows = list(csv.DictReader(f))
    scopes = {}
    for row in rows:
        name = row["NVTX Range"].lstrip(":")
        if name not in scopes:
            scopes[name] = dict(kernel_ms=0., kernel_instances=0,
                                families_ms=collections.Counter(), kernels_ms=collections.Counter())
        s = scopes[name]
        ms = int(row["Total Time (ns)"]) / 1e6
        s["kernel_ms"] += ms
        s["kernel_instances"] += int(row["Kern Inst"])
        s["families_ms"][family(row["Kernel Name"])] += ms
        s["kernels_ms"][row["Kernel Name"]] += ms
    with (case / "stats_nvtx_gpu_proj_sum.csv").open() as f:
        for row in csv.DictReader(f):
            name = row["Range"].lstrip(":")
            if name in scopes:
                scopes[name].update(projected_ms=int(row["Total Proj Time (ns)"]) / 1e6,
                                    host_range_ms=int(row["Total Range Time (ns)"]) / 1e6,
                                    range_instances=int(row["Range Instances"]))
    return scopes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = dict(measurement_scope="unprofiled pooled request means; Nsight summed GPU kernel time by inclusive NVTX launch scope",
                   ncu_status="unavailable_ERR_NVGPUCTRPERM", baselines={}, profiles={})
    baselines = {}
    for bs in (1, 4):
        for method in ("snapkv", "snapkv-logits", "kvzip"):
            key = f"{method}-baseline-c{bs}"
            records = load_rows(args.run_root / key / "raw_samples.jsonl")
            if len(records) != 3 or any(r["status"] != "success" for r in records):
                raise ValueError(f"Incomplete baseline: {key}")
            requests = [q for r in records for q in r["request_results"]]
            if len(requests) != bs * 3 or any(q["status"] != "success" or q["generated_tokens"] != 128 for q in requests):
                raise ValueError(f"Incomplete requests: {key}")
            for r in records:
                graph = r["decode_cuda_graph_counter_delta"]
                if graph["replay_count"] != 127 or any(graph[k] for k in ("capture_count", "recapture_count", "eager_static_count", "force_eager_count")):
                    raise ValueError(f"Unexpected Graph execution: {key}")
                if method == "kvzip":
                    evidence = r["context_compaction"]
                    if not evidence or evidence["executed_compactions"] != bs or evidence["expected_contexts"] != bs:
                        raise ValueError(f"Missing context compaction: {key}: {evidence}")
            baselines[method, bs] = records
            summary["baselines"][key] = dict(
                requests=len(requests), iterations=3,
                ttft_ms_mean=statistics.mean(q["ttft_ms"] for q in requests),
                tpot_ms_mean=statistics.mean(q["tpot_ms"] for q in requests),
                workload_elapsed_s_mean=statistics.mean(r["elapsed_s"] for r in records),
                output_tokens_per_s=sum(q["generated_tokens"] for q in requests) / sum(r["elapsed_s"] for r in records),
                per_iteration_elapsed_s=[r["elapsed_s"] for r in records],
                graph_replays_per_iteration=[r["decode_cuda_graph_counter_delta"]["replay_count"] for r in records])
            key = f"{method}-profile-c{bs}"
            prof = load_rows(args.run_root / key / "raw_samples.jsonl")
            if len(prof) != 1 or prof[0]["status"] != "success" or prof[0]["trace"] != records[0]["trace"]:
                raise ValueError(f"Unmatched profile: {key}")
            graph = prof[0]["decode_cuda_graph_counter_delta"]
            if graph["replay_count"] != 127 or any(graph[k] for k in ("capture_count", "recapture_count", "eager_static_count", "force_eager_count")):
                raise ValueError(f"Unexpected profiled Graph execution: {key}")
            if len(prof[0]["request_results"]) != bs or any(q["generated_tokens"] != 128 for q in prof[0]["request_results"]):
                raise ValueError(f"Incomplete profile: {key}")
            summary["profiles"][key] = profile_summary(args.run_root / key)
        for method in ("snapkv", "snapkv-logits"):
            for a, b in zip(baselines[method, bs], baselines["kvzip", bs]):
                if a["trace"] != b["trace"]:
                    raise ValueError(f"Methods have different input traces: {method} c{bs}")
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["baselines"], indent=2))


if __name__ == "__main__":
    main()
