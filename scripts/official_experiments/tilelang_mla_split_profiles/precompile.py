"""Optionally fill TileLang's atomic disk cache on CPU, without GPU execution."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time


def compile_one(item):
    import torch
    from sparseengine.kernels.tilelang.mla.decode import build_glm_mla_decode_kernel

    if torch.cuda.is_available():
        raise RuntimeError("Precompilation must run with CUDA_VISIBLE_DEVICES empty")
    case, cfg = item
    b, h, length = case["batch"], case["heads"], case["context"]
    start = time.monotonic()
    build_glm_mla_decode_kernel(batch=b, h_q=32 if h == 20 else 16, h_kv=1,
        valid_output_heads=h, cache_slots=b * length, slot_rows=b,
        active_slot_width=length, max_seqlen_pad=length, dv=512, dpe=64,
        block_N=cfg["block_n"], block_H=cfg["block_h"], num_split=cfg["num_split"],
        block_size=cfg["block_n"], softmax_scale=256**-0.5, need_score=True,
        score_mode=cfg["score_mode"], q_latent_strides=(h * 576, 576, 1),
        q_rope_strides=(h * 576, 576, 1))
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA context in CPU precompiler")
    return {"case_id": case["id"], "config": cfg, "status": "success",
            "wall_seconds": time.monotonic() - start, "pid": os.getpid()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Explicitly hide all GPUs before importing TileLang")
    manifest = json.loads((args.run / "run_manifest.json").read_text())
    major, minor = manifest["gpu"]["capability"]
    arch = f"sm_{major}{minor}" + ("a" if major >= 9 else "")
    os.environ["TILELANG_TARGET"] = f"cuda -arch={arch}"
    plans = json.loads((args.run / "generated_profiles.json").read_text())
    unique = {}
    for plan in plans:
        for cfg in plan["configs"].values():
            case = plan["case"]
            key = (case["heads"], case["batch"], case["context"], json.dumps(cfg, sort_keys=True))
            unique[key] = (case, cfg)
    items = list(unique.values())
    if args.limit:
        items = items[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {"command": sys.argv, "target": os.environ["TILELANG_TARGET"],
                "cache": os.environ["TILELANG_CACHE_DIR"], "count": len(items),
                "status": "running", "gpu_execution": False}
    (args.output / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    with (args.output / "results.jsonl").open("x") as f:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for index, result in enumerate(pool.map(compile_one, items)):
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(f"COMPILED {index + 1}/{len(items)} {result['case_id']}", flush=True)
    identity["status"] = "completed"
    (args.output / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
