"""Two fixed launch profiles of the same score-producing MLA decode kernel."""
import argparse
from dataclasses import asdict, replace
import gc
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

import torch

from sparseengine.kernels.tilelang.mla.runtime import (
    TileMlaDecodeKernel, TileMlaLaunchConfig,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_cases(config):
    cases = []
    for h, b, length in itertools.product(config["heads"], config["batches"], config["contexts"]):
        cases.append(dict(heads=h, batch=b, context=length, distribution="uniform"))
    for h, b, length in itertools.product(config["heads"], config["ragged_batches"], config["ragged_contexts"]):
        cases.append(dict(heads=h, batch=b, context=length, distribution="ragged"))
    for h, length in itertools.product(config["heads"], config["large_batch_contexts"]):
        cases.append(dict(heads=h, batch=config["large_batch"], context=length, distribution="uniform"))
    for case in cases:
        case["id"] = f'h{case["heads"]}-b{case["batch"]}-l{case["context"]}-{case["distribution"]}'
    random.Random(config["seed"]).shuffle(cases)
    return cases


def profiles(case, config, sm_count):
    # The historical baseline must not track changes to production selection.
    legacy = config["legacy_profile"]
    bucket = next((i for i, b in enumerate(legacy["batch_buckets"]) if case["batch"] <= b),
                  len(legacy["batch_buckets"]) - 1)
    old = TileMlaLaunchConfig(num_split=legacy["splits"][str(case["heads"])][bucket],
                             block_h=32 if case["heads"] == 20 and case["batch"] > 1 else 16,
                             score_mode="per_head")
    padded = 32 if case["heads"] == 20 else 16
    head_tiles = padded // old.block_h
    target = sm_count * max(math.log2(case["context"] / 64), 1)
    raw_splits = target / (case["batch"] * head_tiles)
    split = next((s for s in config["legal_splits"] if s >= math.ceil(raw_splits)),
                 config["legal_splits"][-1])
    new = replace(old, num_split=split)
    return {"legacy": old, "formula": new}, {
        "formula_id": config["formula"], "sm_count": sm_count,
        "target_ctas": target, "base_ctas": case["batch"] * head_tiles,
        "raw_splits": raw_splits, "head_tiles": head_tiles,
        "rounding": "ceil to next legal split; cap at 32; no minimum-work cap",
        "configs": {"legacy": asdict(old), "formula": asdict(new)},
        "identical_config": old == new,
    }


def timed(graph, replays, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (replays * calls)


@torch.inference_mode()
def run_case(case, config, sm_count, raw_file):
    started = time.monotonic()
    b, h, length = case["batch"], case["heads"], case["context"]
    torch.manual_seed(config["seed"] + b * 100 + h + length)
    torch.cuda.reset_peak_memory_stats()
    # Serving-like strided views of a packed query; distinct KV storage per request.
    packed = torch.randn(b, h, 576, device="cuda", dtype=torch.bfloat16)
    q, qr = packed[..., :512], packed[..., 512:]
    kv = torch.randn(b * length, 1, 512, device="cuda", dtype=torch.bfloat16)
    kr = torch.randn(b * length, 1, 64, device="cuda", dtype=torch.bfloat16)
    assert length % 64 == 0
    page_slots = []
    for row in range(b):
        pages = torch.randperm(length // 64, device="cuda")
        page_slots.append((pages[:, None] * 64 + torch.arange(64, device="cuda")
                           + row * length).flatten().to(torch.int32))
    slots = torch.stack(page_slots)
    # Permuted request indices exercise the slot-table indirection independently.
    req = torch.randperm(b, device="cuda").to(torch.int32)
    host_lengths = [length] * b
    if case["distribution"] == "ragged":
        pattern = [0, 17, length // 8 + 1, length // 2 - 1, length]
        host_lengths = [pattern[i % len(pattern)] for i in range(b)]
        host_lengths[-1] = length
    lengths = torch.tensor(host_lengths, dtype=torch.int32, device="cuda")
    configs, plan = profiles(case, config, sm_count)
    outputs = {arm: torch.empty_like(q) for arm in configs}
    scores = {arm: torch.full((b, h, length), -1e20, device="cuda", dtype=torch.float32)
              for arm in configs}
    runners = {arm: TileMlaDecodeKernel(device="cuda:0", softmax_scale=256**-0.5,
                                        valid_heads=h, fixed_config=cfg)
               for arm, cfg in configs.items()}

    def call(arm):
        runners[arm](q, qr, kv, kr, slots, req, lengths, outputs[arm],
                     attn_score=scores[arm], max_context_len=length)

    oracle_rows = sorted({0, b // 2, b - 1})
    references = {}
    for row in oracle_rows:
        n = host_lengths[row]
        if n:
            ids = slots[int(req[row]), :n].long()
            keys, rope = kv[ids, 0].float(), kr[ids, 0].float()
            raw = q[row].float() @ keys.T + qr[row].float() @ rope.T
            expected = (raw * (256**-0.5)).softmax(-1) @ keys
            references[row] = (raw, expected.to(torch.bfloat16))
    if references:
        del keys, rope, raw, expected, ids

    def check(arm):
        if not torch.isfinite(outputs[arm]).all():
            raise AssertionError(f"Non-finite output: {arm}")
        for row in oracle_rows:
            n = host_lengths[row]
            if n:
                raw_ref, expected_ref = references[row]
                torch.testing.assert_close(outputs[arm][row], expected_ref,
                                           rtol=config["rtol"], atol=config["atol"])
                torch.testing.assert_close(scores[arm][row, :, :n], raw_ref,
                                           rtol=config["rtol"], atol=config["atol"])
            else:
                torch.testing.assert_close(outputs[arm][row], torch.zeros_like(outputs[arm][row]))
            if not torch.all(scores[arm][row, :, n:] == -1e20):
                raise AssertionError("Kernel overwrote masked score tail")

    cold_ms, graphs = {}, {}
    first_order = ["legacy", "formula"] if b % 2 else ["formula", "legacy"]
    for arm in first_order:
        cold_start = time.monotonic()
        call(arm)
        torch.cuda.synchronize()
        cold_ms[arm] = (time.monotonic() - cold_start) * 1000
        check(arm)
        for _ in range(config["warmup_calls"]):
            call(arm)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(config["calls_per_graph"]):
                call(arm)
        graph.replay()
        torch.cuda.synchronize()
        check(arm)
        graphs[arm] = graph
    pilots = {arm: timed(graphs[arm], 5, config["calls_per_graph"]) for arm in configs}
    replays = max(1, min(config["max_graph_replays"], math.ceil(
        config["sample_target_ms"] * 1000 / (max(pilots.values()) * config["calls_per_graph"]))))
    samples = {arm: [] for arm in configs}
    for round_index in range(config["abba_rounds"]):
        order = first_order + first_order[::-1]
        if round_index % 2:
            order = order[::-1]
            # ABBA is palindromic: explicitly swap the two labels for BAAB.
            order = ["formula" if arm == "legacy" else "legacy" for arm in order]
        for position, arm in enumerate(order):
            for _ in range(3):
                graphs[arm].replay()
            us = timed(graphs[arm], replays, config["calls_per_graph"])
            samples[arm].append(us)
            raw_file.write(json.dumps({"case_id": case["id"], "arm": arm,
                "round": round_index, "position": position, "latency_us": us,
                "graph_replays": replays, "calls_per_graph": config["calls_per_graph"],
                "status": "success"}) + "\n")
            raw_file.flush()
    medians = {arm: statistics.median(values) for arm, values in samples.items()}
    result = {**case, **plan, "status": "success", "median_us": medians,
              "speedup": medians["legacy"] / medians["formula"],
              "samples_us": samples, "cold_bind_ms": cold_ms,
              "oracle": {"status": "success", "rows": oracle_rows,
                         "checks": ["eager", "graph_replay"], "rtol": config["rtol"],
                         "atol": config["atol"], "reference": "torch FP32 raw QK and softmax @ V"},
              "q_strides": list(q.stride()), "actual_lengths": host_lengths,
              "kv_bytes": (kv.numel() + kr.numel()) * kv.element_size(),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "wall_seconds": time.monotonic() - started}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--case-ids", type=Path, help="Explicit subset for an independent repeat")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    if config["formula"] != "sm_log_context_v1" or config["legal_splits"] != [1, 2, 4, 8, 16, 32]:
        raise ValueError("Unknown formula or legal splits")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    cases = make_cases(config)
    if args.smoke:
        cases = [{"heads": 10, "batch": 5, "context": 2048, "distribution": "ragged",
                  "id": "h10-b5-l2048-ragged"}]
    if args.case_ids:
        wanted = json.loads(args.case_ids.read_text())
        cases = [case for case in cases if case["id"] in wanted]
        if len(cases) != len(wanted):
            raise ValueError("Subset IDs are missing or duplicated")
    repo = Path(__file__).resolve().parents[3]
    versions = {name: importlib.metadata.version(name) for name in ["torch", "triton", "tilelang", "apache-tvm-ffi"]}
    manifest = {"command": sys.argv, "interpreter": sys.executable, "config": config,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
                "versions": versions, "cuda": torch.version.cuda,
                "gpu": {"name": props.name, "sm_count": props.multi_processor_count,
                        "capability": list(torch.cuda.get_device_capability()),
                        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]},
                "gpu_state": subprocess.check_output(["nvidia-smi", "-q", "-i", os.environ["CUDA_VISIBLE_DEVICES"]], text=True),
                "protocol": "kernel wrapper CUDA Graph; split + combine; warm resident KV; no L2 flush; distinct KV per request; shuffled 64-token pages; no engine/MoE/EP",
                "cases": cases, "status": "running"}
    write_json(args.output / "run_manifest.json", manifest)
    write_json(args.output / "generated_profiles.json", [dict(case=case, **profiles(case, config, props.multi_processor_count)[1]) for case in cases])
    with (args.output / "raw_samples.jsonl").open("x") as raw, (args.output / "cases.jsonl").open("x") as out:
        for index, case in enumerate(cases):
            print(f'START {index + 1}/{len(cases)} {case["id"]}', flush=True)
            try:
                result = run_case(case, config, props.multi_processor_count, raw)
            except Exception as exc:
                out.write(json.dumps({**case, "status": "model_failed", "error": repr(exc), "traceback": traceback.format_exc()}) + "\n")
                out.flush()
                manifest["status"] = "failed"
                write_json(args.output / "run_manifest.json", manifest)
                raise
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(f'DONE {index + 1}/{len(cases)} {case["id"]} legacy={result["median_us"]["legacy"]:.2f}us formula={result["median_us"]["formula"]:.2f}us speedup={result["speedup"]:.3f}', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    manifest["status"] = "completed"
    write_json(args.output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
