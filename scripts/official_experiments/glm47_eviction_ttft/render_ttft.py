"""Validate request logs and plot matched BS1 TTFT by prompt length.

Run from the repository root with a Python environment containing matplotlib.
The input is the completed GLM-4.7-Flash BS1 campaign root; no GPU is needed.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmark.efficiency.metrics import request_summary  # noqa: E402


LENGTHS = (16384, 32768, 58114, 115924, 173734)
METHODS = ("svllm-dense", "h2o", "snapkv")
LABELS = {"svllm-dense": "Dense", "h2o": "H2O", "snapkv": "SnapKV"}
COLORS = {"svllm-dense": "#333b48", "h2o": "#007f73", "snapkv": "#ce6b34"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"Empty artifact: {path}")
    return rows


def collect(source_root):
    result = []
    traces = {}
    commits = set()
    for length in LENGTHS:
        for method in METHODS:
            matches = list(source_root.glob(f"*/full/{method}/request/p{length}"))
            if len(matches) != 1:
                raise ValueError(f"Expected one full run for {method}/{length}: {matches}")
            path = matches[0]
            status = read_json(path / "run_status.json")
            manifest = read_json(path / "run_manifest.json")
            summary_file = read_json(path / "summary.json")
            config = read_json(path.parents[3] / f"{method}.json")
            raw = read_jsonl(path / "raw_samples.jsonl")
            sample_path = path / "request_samples.jsonl"
            samples = read_jsonl(sample_path)
            if any(item.get("status") != "success" for item in (status, summary_file)):
                raise ValueError(f"Non-success run: {path}")
            if len(raw) != 3 or len(samples) != 3 or any(x.get("status") != "success" for x in raw + samples):
                raise ValueError(f"Incomplete measured iterations: {path}")
            args = manifest["args"]
            expected_method = "vanilla" if method == "svllm-dense" else method
            expected = {
                "engine": "sparsevllm", "sparse_method": expected_method,
                "prompt_lens": [length], "output_lens": [2048], "batch_sizes": [1],
                "scenario": "fixed", "seed": 42, "prompt_length_jitter": 0,
                "output_length_jitter": 0, "tensor_parallel_size": 1,
                "expert_parallel_size": 1, "max_num_batched_tokens": 32768,
                "gpu_memory_utilization": 0.9, "num_warmups": 1, "num_iters": 3,
            }
            if any(args.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Protocol mismatch: {path}")
            if manifest["workload"]["prefix_caching_enabled"]:
                raise ValueError(f"Prefix caching enabled: {path}")
            if manifest["status"] != "success" or "GLM-4.7-Flash" not in args["model_path"]:
                raise ValueError(f"Manifest mismatch: {path}")
            if any(config.get(key) != value for key, value in {
                "engine_prefill_chunk_size": 32768,
                "max_num_batched_tokens": 32768,
                "decode_graph": True,
                "enable_prefix_caching": False,
            }.items()):
                raise ValueError(f"Engine configuration mismatch: {path}")
            if method == "h2o" and any(config.get(key) != value for key, value in {
                "h2o_prefill_budget": 8192, "h2o_decode_budget": 4096,
                "h2o_decode_eviction_interval": 128,
                "sparse_prefill_score_mode": "logits",
            }.items()):
                raise ValueError(f"H2O budget mismatch: {path}")
            if method == "snapkv" and any(config.get(key) != value for key, value in {
                "sink_keep_tokens": 64, "recent_keep_tokens": 512,
                "decode_keep_tokens": 7616, "snapkv_window_size": 32,
                "sparse_prefill_score_mode": "probability",
            }.items()):
                raise ValueError(f"SnapKV budget mismatch: {path}")
            if "H100" not in manifest["runtime_environment"]["physical_cuda_devices"][0]["name"]:
                raise ValueError(f"Device mismatch: {path}")
            commits.add(manifest["git"]["git_commit"])
            for x in samples:
                if (x["prompt_len"], x["output_len"], x["concurrency"], x["generated_tokens"]) != (length, 2048, 1, 2048):
                    raise ValueError(f"Incomplete or mismatched request: {path}")
                if x["timing_source"] != "sparsevllm_step_token_publication_no_extra_sync_v1":
                    raise ValueError(f"Timing boundary mismatch: {path}")
            for x in raw:
                if x["prefill_token_count"] != length or x["decode_token_count"] != 2047:
                    raise ValueError(f"Incomplete token work: {path}")
            summary = summary_file["summary"]
            if len(summary) != 1 or summary[0]["status"] != "success":
                raise ValueError(f"Invalid summary: {path}")
            aggregated = request_summary(samples)
            for key in ("ttft_ms_mean", "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99"):
                if not math.isclose(aggregated[key], summary[0][key], rel_tol=1e-9):
                    raise ValueError(f"Request aggregate disagrees with summary: {path}, {key}")
            trace = {(x["iteration"], x["trace_seed"], x["prompt_digest"]) for x in samples}
            if length in traces and traces[length] != trace:
                raise ValueError(f"Trace mismatch across methods at prompt {length}")
            traces[length] = trace
            result.append({
                "prompt_tokens": length, "method": method, "measured_requests": len(samples),
                "ttft_s_mean": aggregated["ttft_ms_mean"] / 1000,
                "ttft_s_p50": aggregated["ttft_ms_p50"] / 1000,
                "ttft_s_p95": aggregated["ttft_ms_p95"] / 1000,
                "ttft_s_p99": aggregated["ttft_ms_p99"] / 1000,
                "source_relative": str(path.relative_to(source_root)),
            })
    if len(commits) != 1:
        raise ValueError(f"Mixed Git commits: {commits}")
    return result, commits.pop()


def plot(rows, output):
    by_key = {(x["prompt_tokens"], x["method"]): x for x in rows}
    fig, (ax, ratio_ax) = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True,
                                       gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.10})
    x = [v / 1000 for v in LENGTHS]
    for method in METHODS:
        y = [by_key[(v, method)]["ttft_s_mean"] for v in LENGTHS]
        ax.plot(x, y, marker="o", linewidth=2.3, markersize=6, color=COLORS[method], label=LABELS[method])
        if method != "svllm-dense":
            ratio = [by_key[(v, "svllm-dense")]["ttft_s_mean"] / by_key[(v, method)]["ttft_s_mean"] for v in LENGTHS]
            ratio_ax.plot(x, ratio, marker="o", linewidth=2.1, markersize=5, color=COLORS[method], label=LABELS[method])
    ratio_ax.axhline(1, color="#777", linewidth=1, linestyle="--")
    ax.set_ylabel("Mean request TTFT (s)")
    ratio_ax.set_ylabel("Dense / method\nTTFT ratio")
    ratio_ax.set_xlabel("Prompt tokens (thousands)")
    ratio_ax.set_xticks(x, [f"{v/1000:g}" for v in LENGTHS])
    ax.grid(alpha=0.22)
    ratio_ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("GLM-4.7-Flash BF16 · H100 · BS1 · no request queue")
    fig.text(0.5, 0.005, "2048 output tokens · 1 discarded + 3 measured requests per point · matched traces",
             ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(bottom=0.12, top=0.90, left=0.11, right=0.98)
    fig.savefig(output / "ttft_vs_prompt.png", dpi=180)
    fig.savefig(output / "ttft_vs_prompt.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows, commit = collect(args.source_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ttft_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output_dir)
    print(f"Validated {len(rows)} matched points, commit {commit}")


if __name__ == "__main__":
    main()
