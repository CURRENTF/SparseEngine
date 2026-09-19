"""Matched persistent-MLA schedule ablation, including reset and merge."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
import triton

from sparseengine.kernels.triton.mla import (
    MlaDecodeLaunchConfig, allocate_mla_decode_workspace,
    run_mla_decode, select_glm_mla_decode_config,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def legacy_config(batch, heads):
    # Frozen pre-change H100 table: the benchmark baseline, not production policy.
    medium = MlaDecodeLaunchConfig(program_count=264, blocks_per_program=2,
        block_n=32, block_q_heads=8, stage1_num_warps=8,
        stage1_pipeline_stages=4, stage2_pipeline_stages=1)
    small = replace(medium, program_count=256, blocks_per_program=4,
                    block_q_heads=16, stage1_pipeline_stages=6)
    wide = replace(medium, program_count=128, blocks_per_program=8)
    large = replace(wide, program_count=256)
    if heads == 20:
        return medium if batch == 1 else small if batch <= 16 else wide
    if heads == 10:
        return medium if batch == 1 else small if batch <= 4 else wide if batch <= 16 else large
    return medium if batch <= 4 else large


@torch.inference_mode()
def measure(case, out, raw, repeats):
    batch, heads, length = case["batch"], case["heads"], case["length"]
    torch.manual_seed(42 + batch)
    capacity = max(length, 33408)
    packed = torch.randn(batch, heads, 576, device="cuda", dtype=torch.bfloat16)
    q, qr = packed[..., :512], packed[..., 512:]
    kv = torch.randn(batch * length, 1, 512, device="cuda", dtype=torch.bfloat16)
    kr = torch.randn(batch * length, 1, 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.full((batch, capacity), -1, device="cuda", dtype=torch.int32)
    for row in range(batch):
        pages = torch.randperm(length // 64, device="cuda")
        slots[row, :length] = (pages[:, None] * 64 + torch.arange(64, device="cuda") + row * length).flatten().int()
    req = torch.randperm(batch, device="cuda").int()
    lens = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    if case.get("ragged"):
        lens[::3] = length // 4
        if batch > 1:
            lens[-1] = 0
            req[-1] = -1
    oracle = {}
    for row in sorted({0, batch // 2, batch - 1}):
        n = int(lens[row])
        if not n:
            oracle[row] = (torch.zeros(heads, 512, device="cuda"), None)
            continue
        ids = slots[int(req[row]), :n].long()
        k, r = kv[ids, 0].float(), kr[ids, 0].float()
        logits = q[row].float() @ k.T + qr[row].float() @ r.T
        oracle[row] = ((logits / 16).softmax(-1) @ k, logits.amax(0))
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    configs = {"legacy": legacy_config(batch, heads),
               "sm_rule": select_glm_mla_decode_config(batch_size=batch, local_q_heads=heads, sm_count=sm)}
    if batch == 64 and heads == 20 and length == 4160:
        configs["measured_512x8"] = replace(configs["legacy"], program_count=512)
    if case.get("tune"):
        for head_tile in (8, 16):
            for waves in (1, 2, 4):
                configs[f"h{head_tile}_waves{waves}"] = replace(configs["sm_rule"],
                    block_q_heads=head_tile,
                    target_splits_per_request=math.ceil(waves * sm / math.ceil(heads / head_tile)))
    arms = {}
    for name, cfg in configs.items():
        output = torch.empty_like(q)
        score = torch.empty(batch, capacity, device="cuda", dtype=torch.float32)
        ws = allocate_mla_decode_workspace(batch_size=batch, head_count=heads, device="cuda", config=cfg)
        def call(cfg=cfg, output=output, score=score, ws=ws):
            run_mla_decode(q, qr, kv, kr, slots, req, lens, output, ws,
                softmax_scale=1/16, attn_score=score, max_context_len=capacity,
                config=cfg, validate_metadata=False)
        def validate(output=output, score=score):
            for row, (expected, expected_score) in oracle.items():
                torch.testing.assert_close(output[row].float(), expected, rtol=.02, atol=.02)
                n = int(lens[row])
                if n:
                    torch.testing.assert_close(score[row, :n], expected_score, rtol=.002, atol=.02)
                assert bool(torch.all(score[row, n:] == -1e20))
        start = time.perf_counter()
        call(); torch.cuda.synchronize(); validate()
        cold_ms = 1000 * (time.perf_counter() - start)
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay(); torch.cuda.synchronize(); validate()
        block = int(ws.block_size.item())
        counts = [(int(n) + block - 1) // block for n in lens.tolist()]
        assert sum(counts) <= ws.mid_output.shape[1]
        arms[name] = dict(graph=graph, keepalive=(output, score, ws), samples=[], metadata={
            **case, "name": name, "config": asdict(cfg), "sm_count": sm,
            "block_size": block, "splits": counts, "cold_ms": cold_ms,
            "workspace_bytes": sum(t.numel() * t.element_size() for t in (ws.block_size, ws.batch_start_indices, ws.mid_output, ws.mid_logsumexp)),
            "correctness": "FP32 Torch rows first/middle/last, eager and Graph"})
    rng = random.Random(42)
    for iteration in range(repeats):
        names = list(arms); rng.shuffle(names)
        for name in names:
            arm = arms[name]
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); begin.record()
            for _ in range(5):
                arm["graph"].replay()
            end.record(); end.synchronize()
            ms = begin.elapsed_time(end) / 5
            arm["samples"].append(ms)
            raw.write(json.dumps({**case, "name": name, "iteration": iteration, "ms": ms}) + "\n")
            raw.flush()
    result = [{**a["metadata"], "median_ms": statistics.median(a["samples"]),
               "min_ms": min(a["samples"]), "max_ms": max(a["samples"])} for a in arms.values()]
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    write(args.output_dir / "manifest.json", {"config": cfg, "torch": torch.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "boundary": "schedule + score reset + stage1 + stage2; GPU CUDA event Graph replay", "warmups": 5, "replays_per_sample": 5})
    results = []
    try:
        with (args.output_dir / "raw_samples.jsonl").open("x") as raw:
            for case in cfg["cases"]:
                results.extend(measure(case, args.output_dir, raw, cfg["repeats"]))
                write(args.output_dir / "partial.json", results)
        write(args.output_dir / "summary.json", {"status": "success", "cases": results})
    except BaseException as exc:
        write(args.output_dir / "failure.json", {"status": "failed", "error": repr(exc), "completed_cases": len(results)})
        raise


if __name__ == "__main__":
    main()
