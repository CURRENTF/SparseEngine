"""Run one aligned AIME batch with Vortex QuEST and save MathBench records."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import vortex_torch  # noqa: F401: registers the SGLang attention backend
from vortex_torch.engine.sgl import get_engine


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compilation-cache", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=41984)
    parser.add_argument("--max-new-tokens", type=int, default=40960)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(args.inputs.read_text(encoding="utf-8"))
    if len(rows) != 60 or [row["id"] for row in rows] != [str(i) for i in range(60)]:
        raise ValueError("Expected 60 aligned AIME requests with IDs 0..59")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.model.joinpath("config.json").is_file() or not args.flow.is_file():
        raise FileNotFoundError("Model config or QuEST flow is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("Vortex QuEST requires a visible CUDA device")

    args.output.mkdir(parents=True)
    args.compilation_cache.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "engine": "vortex_sglang",
        "model": str(args.model.resolve()),
        "flow": str(args.flow.resolve()),
        "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "requests": len(rows),
        "seed": args.seed,
        "sampling": {"max_new_tokens": args.max_new_tokens, "temperature": 0.6,
                     "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "engine_args": {
            "context_length": args.max_model_len,
            "vortex_max_seq_lens": args.max_model_len,
            "max_running_requests": args.concurrency,
            "mem_fraction_static": args.mem_fraction_static,
            "vortex_block_size": 16,
            "vortex_topk_val": 128,
            "vortex_topk_ratio": 0.0,
            "vortex_block_reserved_bos": 1,
            "vortex_block_reserved_eos": 4,
            "vortex_layers_skip": [0, 1],
            "vortex_module_name": "gqa_quest_sparse_attention",
            "vortex_workload_chunk_size": 32,
            "disable_radix_cache": True,
            "cuda_graph_bs": [
                size for size in (1, 2, 4, 8, 16, 24)
                if size < args.concurrency
            ] + [args.concurrency],
            "chunked_prefill_size": 4096,
            "max_prefill_tokens": 65536,
            "vortex_compilation_cache_dir": str(args.compilation_cache.resolve()),
            "log_level": "info",
        },
    }
    write_json(args.output / "manifest.json", manifest)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    llm = None
    try:
        llm = get_engine(
            model_path=str(args.model.resolve()),
            vortex_module_path=str(args.flow.resolve()),
            random_seed=args.seed,
            **manifest["engine_args"],
        )
        sampling = manifest["sampling"]
        start = time.perf_counter()
        outputs = llm.generate(
            input_ids=[row["input_ids"] for row in rows],
            sampling_params=sampling,
            rid=[row["id"] for row in rows],
        )
        elapsed = time.perf_counter() - start
        if not isinstance(outputs, list) or len(outputs) != len(rows):
            raise RuntimeError(f"Expected 60 Vortex outputs, got {type(outputs).__name__} / {len(outputs)}")
        with (args.output / "raw_outputs.jsonl").open("w", encoding="utf-8") as raw_out, (
            args.output / "aime2024.jsonl"
        ).open("w", encoding="utf-8") as pred_out:
            for row, result in zip(rows, outputs):
                if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                    raise RuntimeError(f"Invalid output for request {row['id']}")
                meta = result.get("meta_info") or {}
                if str(meta.get("id", row["id"])) != row["id"]:
                    raise RuntimeError(f"Request order mismatch for {row['id']}: {meta.get('id')}")
                raw = {"id": row["id"], "result": result}
                pred = {"id": row["id"], "status": "success", "pred": result["text"], "gold": row["gold"]}
                raw_out.write(json.dumps(raw, ensure_ascii=False) + "\n")
                pred_out.write(json.dumps(pred, ensure_ascii=False) + "\n")
        completion_tokens = sum(int(result["meta_info"]["completion_tokens"]) for result in outputs)
        manifest.update(status="generated", generation_elapsed_s=elapsed,
                        completion_tokens=completion_tokens,
                        completion_tokens_per_s=completion_tokens / elapsed)
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps({"status": "generated", "generation_elapsed_s": elapsed,
                          "completion_tokens": completion_tokens}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output / "manifest.json", manifest)
        raise
    finally:
        if llm is not None:
            llm.shutdown()


if __name__ == "__main__":
    main()
