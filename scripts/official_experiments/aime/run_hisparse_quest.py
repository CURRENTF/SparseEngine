"""Run the aligned AIME requests with the pinned HiSparse QuEST runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=41984)
    parser.add_argument("--max-new-tokens", type=int, default=40960)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inputs = args.inputs.resolve(strict=True)
    model = args.model.resolve(strict=True)
    source = args.sglang_source.resolve(strict=True)
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.concurrency != 20 or not 0 < args.mem_fraction_static <= 1:
        raise ValueError("This matched run requires C20 and a valid GPU memory fraction")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one idle CUDA device")
    rows = json.loads(inputs.read_text(encoding="utf-8"))
    if len(rows) != 60 or [str(row["id"]) for row in rows] != [str(i) for i in range(60)]:
        raise ValueError("Expected the aligned 60 AIME requests with IDs 0..59")
    if any(
        not isinstance(row.get("input_ids"), list)
        or not row["input_ids"]
        or not isinstance(row.get("gold"), dict)
        or len(row["input_ids"]) + args.max_new_tokens > args.max_model_len
        for row in rows
    ):
        raise ValueError("Invalid prompt IDs, gold row, or context length")

    import sglang

    if not Path(sglang.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"Unexpected SGLang import: {sglang.__file__}")
    from sglang.srt.entrypoints.engine import Engine

    sampling = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    }
    hisparse = {
        "algorithm": "quest",
        "backend": "fa3",
        "page_size": 16,
        "fixed_selected_pages": 128,
        "num_recent_pages": 5,
        "min_sparse_prompt_len": 0,
        "enable_cuda_graph_retrieval": True,
    }
    engine_args = {
        "model_path": str(model),
        "tp_size": 1,
        "dtype": "bfloat16",
        "context_length": args.max_model_len,
        "max_running_requests": args.concurrency,
        "mem_fraction_static": args.mem_fraction_static,
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 65536,
        "disable_radix_cache": True,
        "attention_backend": "fa3",
        "page_size": 16,
        "enable_hisparse": True,
        "hisparse_config": json.dumps(hisparse, separators=(",", ":")),
        "cuda_graph_config": {
            "prefill": {"backend": "disabled"},
            "decode": {"backend": "breakable", "bs": [1, 2, 4, 8, 16, 20]},
        },
        "random_seed": args.seed,
        "log_level": "info",
    }
    manifest = {
        "status": "running",
        "engine": "hisparse_quest_sglang",
        "model": str(model),
        "sglang_source": str(source),
        "sglang_import": str(Path(sglang.__file__).resolve()),
        "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "requests": len(rows),
        "sampling": sampling,
        "hisparse_config": hisparse,
        "engine_args": engine_args,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "manifest.json", manifest)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    engine = None
    try:
        engine = Engine(**engine_args)
        start = time.perf_counter()
        outputs = engine.generate(
            input_ids=[row["input_ids"] for row in rows],
            sampling_params=sampling,
            rid=[row["id"] for row in rows],
        )
        elapsed = time.perf_counter() - start
        if not isinstance(outputs, list) or len(outputs) != len(rows):
            raise RuntimeError(f"Expected 60 outputs, got {type(outputs).__name__}")
        completion_tokens = 0
        with (args.output / "raw_outputs.jsonl").open("w", encoding="utf-8") as raw_out, (
            args.output / "aime2024.jsonl"
        ).open("w", encoding="utf-8") as pred_out:
            for row, result in zip(rows, outputs):
                if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                    raise RuntimeError(f"Invalid output for request {row['id']}")
                meta = result.get("meta_info")
                if not isinstance(meta, dict) or str(meta.get("id")) != row["id"]:
                    raise RuntimeError(f"Request ID mismatch for {row['id']}: {meta}")
                completion_tokens += int(meta["completion_tokens"])
                raw_out.write(json.dumps({"id": row["id"], "result": result}, ensure_ascii=False) + "\n")
                pred_out.write(json.dumps({
                    "id": row["id"], "status": "success", "pred": result["text"],
                    "gold": row["gold"],
                }, ensure_ascii=False) + "\n")
        manifest.update(
            status="generated",
            generation_elapsed_s=elapsed,
            completion_tokens=completion_tokens,
            completion_tokens_per_s=completion_tokens / elapsed,
        )
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps({"status": "generated", "generation_elapsed_s": elapsed,
                          "completion_tokens": completion_tokens}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output / "manifest.json", manifest)
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
