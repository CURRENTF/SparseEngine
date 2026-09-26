#!/usr/bin/env python3
"""Evaluate frozen LongBench token prompts with the author DeltaKV HF model."""

import argparse
from contextlib import nullcontext
import json
import os
import sys
import time
import traceback
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("v1", "v2"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sdpa-kernel", choices=("default", "efficient", "cudnn"), default="default")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in (args.model / "config.json", args.checkpoint / "model.safetensors", args.input, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(args.config.read_text())
    samples = read_rows(args.input)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        samples = samples[: args.limit]
    if not samples or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("empty input or duplicate sample IDs")
    if any(not s["prompt_token_ids"] or len(s["prompt_token_ids"]) + s["max_new_tokens"] > 131072 for s in samples):
        raise ValueError("invalid prompt length")

    raw_path = args.output_dir / "raw_outputs.jsonl"
    if args.resume:
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        prior = read_rows(raw_path)
        if len(prior) > len(samples) or any(
            row["sample_id"] != sample["sample_id"] or row["status"] != "success"
            for row, sample in zip(prior, samples)
        ):
            raise ValueError("resume requires a successful output prefix matching input order")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        prior = []

    import torch
    import transformers
    from transformers import AutoTokenizer
    from deltakv.get_chat_api import get_generate_api

    manifest = {
        "status": "running",
        "command": sys.argv,
        "source_commit": args.source_commit,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attention_implementation": os.environ.get("DELTAKV_HF_ATTN_IMPLEMENTATION", "flash_attention_2"),
        "sdpa_kernel": args.sdpa_kernel,
        "benchmark": args.benchmark,
        "samples": len(samples),
        "resumed_from": len(prior),
        "config": config,
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    try:
        generate, model = get_generate_api(
            model_path=str(args.model),
            infer_config=config,
            deltakv_checkpoint_path=str(args.checkpoint),
            sparse_method=config["sparse_method"],
            backend="hf",
            cuda_device=0,
            return_model=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        eos = model.config.eos_token_id
        if isinstance(eos, int):
            eos = [eos]
        with raw_path.open("a" if args.resume else "w", encoding="utf-8") as output:
            for index in range(len(prior), len(samples)):
                sample = samples[index]
                seed = 42 + sample.get("eval_index", index) if args.benchmark == "v1" else 20260901
                start = time.monotonic()
                failure_status = "invalid_input"
                try:
                    ids = sample["prompt_token_ids"]
                    prompt = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    if tokenizer.encode(prompt, add_special_tokens=False) != ids:
                        raise ValueError(f"prompt token roundtrip mismatch: {sample['sample_id']}")
                    failure_status = "model_failed"
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    if args.sdpa_kernel != "default":
                        from torch.nn.attention import SDPBackend, sdpa_kernel

                        backend = {
                            "efficient": SDPBackend.EFFICIENT_ATTENTION,
                            "cudnn": SDPBackend.CUDNN_ATTENTION,
                        }[args.sdpa_kernel]
                        attention_context = sdpa_kernel(backends=[backend])
                    else:
                        attention_context = nullcontext()
                    with attention_context:
                        raw_pred = generate(
                            prompt,
                            max_new_tokens=sample["max_new_tokens"],
                            do_sample=False,
                            temperature=0,
                            top_p=1,
                            top_k=1,
                            eos_token_id=eos,
                        )
                    if not isinstance(raw_pred, str):
                        raise TypeError(f"expected string prediction, got {type(raw_pred).__name__}")
                    record = {
                        "sample_id": sample["sample_id"],
                        "dataset": sample.get("dataset"),
                        "status": "success",
                        "raw_pred": raw_pred,
                        "prompt_tokens": len(ids),
                        "elapsed_seconds": time.monotonic() - start,
                        "seed": seed,
                    }
                except Exception:
                    record = {
                        "sample_id": sample["sample_id"],
                        "status": failure_status,
                        "error": traceback.format_exc(),
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    raise
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{index + 1}/{len(samples)} {sample['sample_id']} {record['elapsed_seconds']:.2f}s", flush=True)
        manifest["status"] = "completed"
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
