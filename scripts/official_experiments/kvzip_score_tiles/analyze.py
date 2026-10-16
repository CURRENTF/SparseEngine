"""Validate matched workloads and separate request timings from Nsight diagnostics."""

import argparse
import json
from pathlib import Path
import statistics

from scripts.official_experiments.snapkv_kvzip_nsight.analyze import load_rows, profile_summary


def validate(rows, batch, iterations):
    if len(rows) != iterations or any(row["status"] != "success" for row in rows):
        raise ValueError("Missing or failed measured workload")
    for row in rows:
        requests = row["request_results"]
        if len(requests) != batch or any(q["status"] != "success" or q["generated_tokens"] != 128 for q in requests):
            raise ValueError("Missing or failed request")
        graph = row["decode_cuda_graph_counter_delta"]
        if graph["replay_count"] != 127 or any(graph[k] for k in ("capture_count", "recapture_count", "eager_static_count", "force_eager_count")):
            raise ValueError("Unexpected graph execution")
        if row["context_compaction"] != {"expected_contexts": batch, "executed_compactions": batch}:
            raise ValueError("Unexpected compaction count")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--original-profile-root", type=Path, required=True)
    parser.add_argument("--optimized-name", default="optimized-final")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"measurement_scope": "unprofiled pooled request means; throughput = output tokens / total workload seconds",
              "ncu_status": "unavailable_ERR_NVGPUCTRPERM", "serving": {}, "profiles": {}}
    for batch in (1, 4):
        records = {}
        for variant in ("baseline", "optimized"):
            variant_dir = args.optimized_name if variant == "optimized" else variant
            case = args.run_root / variant_dir / f"kvzip-baseline-c{batch}"
            rows = load_rows(case / "raw_samples.jsonl")
            validate(rows, batch, 3)
            records[variant] = rows
            requests = [q for r in rows for q in r["request_results"]]
            result["serving"][f"{variant}_c{batch}"] = {
                "requests": len(requests), "iterations": len(rows),
                "ttft_ms_mean": statistics.mean(q["ttft_ms"] for q in requests),
                "tpot_ms_mean": statistics.mean(q["tpot_ms"] for q in requests),
                "elapsed_s_mean": statistics.mean(r["elapsed_s"] for r in rows),
                "output_tokens_per_s": sum(q["generated_tokens"] for q in requests) / sum(r["elapsed_s"] for r in rows),
                "elapsed_s_samples": [r["elapsed_s"] for r in rows],
                "ttft_ms_samples": [r["ttft_ms_mean"] for r in rows],
                "tpot_ms_samples": [r["tpot_ms_mean"] for r in rows],
                "graph_replays": [r["decode_cuda_graph_counter_delta"]["replay_count"] for r in rows],
            }
        for before, after in zip(records["baseline"], records["optimized"]):
            if before["trace"] != after["trace"] or before["protocol"] != after["protocol"]:
                raise ValueError("Mismatched trace or protocol")
        for variant, root in (("baseline", args.original_profile_root),
                              ("optimized", args.run_root / args.optimized_name)):
            case = root / f"kvzip-profile-c{batch}"
            rows = load_rows(case / "raw_samples.jsonl")
            validate(rows, batch, 1)
            if rows[0]["trace"] != records[variant][0]["trace"] or rows[0]["protocol"] != records[variant][0]["protocol"]:
                raise ValueError("Mismatched profiler workload")
            scopes = profile_summary(case)
            result["profiles"][f"{variant}_c{batch}"] = {
                name: {"kernel_ms": scope["kernel_ms"], "families_ms": scope["families_ms"]}
                for name, scope in scopes.items()
                if name in ("model_run_prefill", "model_run_decode", "kvzip.score",
                            "kvzip.reconstruction_forward", "kvzip.global_selection", "kvzip.compact")
            }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["serving"], indent=2))


if __name__ == "__main__":
    main()
