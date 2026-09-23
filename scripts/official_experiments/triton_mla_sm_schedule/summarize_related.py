"""Revalidate the fixed matched pre/post native H2O full-model windows."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmark.efficiency.metrics import stage_throughput


parser = argparse.ArgumentParser()
parser.add_argument("--baseline-root", type=Path, required=True)
parser.add_argument("--new-root", type=Path, required=True)
parser.add_argument("--output", type=Path)
args = parser.parse_args()

cases = []
for source, engine in (
    (args.baseline_root, "Legacy Triton"),
    (args.new_root, "SM Triton"),
):
    for name, concurrency in (("04_glm_common8", 8), ("06_glm_wave64", 64)):
        cases.append({
            "model": "GLM-4.7-Flash",
            "method": "H2O",
            "engine": engine,
            "directory": str((source / "queue" / name).resolve()),
            "concurrency": concurrency,
            "phase": "measure",
        })

summary, samples, traces = [], [], {}
for case in cases:
    root = Path(case["directory"])
    batch = case["concurrency"]
    if json.loads((root / "exit.json").read_text())["exit_code"] != 0:
        raise RuntimeError(f"Failed case: {root}")
    if json.loads((root / "run_status.json").read_text())["status"] != "success":
        raise RuntimeError(f"Failed probe: {root}")
    records = [
        json.loads(line)
        for line in (root / "raw_samples.jsonl").read_text().splitlines()
    ]
    requests = [
        json.loads(line)
        for line in (root / "request_samples.jsonl").read_text().splitlines()
    ]
    if len(records) != 3 or len(requests) != batch * 3 or any(
        row["status"] != "success" or row["generated_tokens"] != 512
        for row in requests
    ):
        raise RuntimeError(f"Incomplete measured requests: {root}")
    trace = [
        [
            (request["prompt_digest"], request["prompt_len"], request["output_len"])
            for request in record["trace"]["requests"]
        ]
        for record in records
    ]
    trace_key = (case["model"], case["method"], batch)
    if trace_key in traces and traces[trace_key] != trace:
        raise RuntimeError(f"Mismatched comparison trace: {trace_key}")
    traces[trace_key] = trace

    source = root / "raw_samples.jsonl"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    windows = [record["decode_only_window"] for record in records]
    rates = []
    for iteration, (window, record) in enumerate(zip(windows, records)):
        expected_graph = {
            "capture_count": 0,
            "replay_count": 256,
            "eager_decode_count": 0,
        }
        if (
            window["status"] != "success"
            or window["concurrency"] != batch
            or window["decode_steps"] != 256
            or window["discarded_full_decode_steps"] != 8
            or window["decode_stage_tokens"] != batch * 256
            or window["prefill_steps"] != 0
            or window["graph_counter_delta"] != expected_graph
            or record["peak_scheduler_decoding_requests"] != batch
        ):
            raise RuntimeError(f"Invalid decode window: {root}, {iteration}")
        rate = stage_throughput(
            window["decode_stage_tokens"], window["decode_stage_elapsed_s"]
        )
        if not math.isclose(
            rate, window["decode_stage_throughput_tps"], rel_tol=1e-12
        ):
            raise RuntimeError(f"Inconsistent decode arithmetic: {root}, {iteration}")
        rates.append(rate)
        samples.append({
            **window,
            "model": case["model"],
            "method": case["method"],
            "engine": case["engine"],
            "iteration": iteration,
            "source_path": str(source),
            "source_sha256": digest,
        })
    summary.append({
        "model": case["model"],
        "method": case["method"],
        "engine": case["engine"],
        "concurrency": batch,
        "mean_tps": statistics.fmean(rates),
        "sd_tps": statistics.stdev(rates),
        "pooled_tps": stage_throughput(
            sum(window["decode_stage_tokens"] for window in windows),
            sum(window["decode_stage_elapsed_s"] for window in windows),
        ),
        "source_path": str(source),
        "status": "success",
    })

result = {
    "status": "success",
    "metric": "decode_stage_throughput_tps",
    "scope": (
        "full-residency continuous decode-only engine window; "
        "8 discard + 256 measured; boundary sync only"
    ),
    "caveats": [
        "Same historical H2O implementation and workload; only three MLA schedule/provider files replaced",
        "32K/512 TP1 boundary-synchronized diagnostic, not the 128K/2K per-step-synchronized figure",
        "Three repetitions on H100 only; not a cross-GPU performance guarantee",
        "Wave2 admission excluded at B64; request metrics are boundary-perturbed",
    ],
    "cases": summary,
    "samples": samples,
}
output = args.output or args.new_root / "comparison.json"
with output.open("x") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
for row in summary:
    print(
        row["model"],
        row["method"],
        row["engine"],
        row["concurrency"],
        f"{row['mean_tps']:.2f} ± {row['sd_tps']:.2f}",
    )
