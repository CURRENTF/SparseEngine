"""Generate aligned AIME answers with the upstream Hugging Face R-KV patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("rkv", "fullkv"), required=True)
    parser.add_argument("--kv-budget", type=int, default=4176)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--mix-lambda", type=float, default=0.1)
    parser.add_argument("--compression-interval", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=41984)
    parser.add_argument("--max-new-tokens", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one idle CUDA device")
    if args.kv_budget <= args.window_size or args.compression_interval <= 0:
        raise ValueError("Invalid R-KV budget, query window, or compression interval")
    if args.limit < 1 or args.limit > 60:
        raise ValueError("Limit must be between 1 and 60")

    upstream = args.upstream.resolve(strict=True)
    if git_commit(upstream) != args.upstream_commit:
        raise ValueError("Upstream R-KV commit differs from the requested revision")
    hf_source = upstream / "HuggingFace"
    if not (hf_source / "rkv" / "monkeypatch.py").is_file():
        raise FileNotFoundError("Upstream HuggingFace/rkv source is missing")
    model_path = args.model.resolve(strict=True)
    rows = json.loads(args.inputs.read_text(encoding="utf-8"))
    if len(rows) != 60 or [str(row["id"]) for row in rows] != [str(i) for i in range(60)]:
        raise ValueError("Expected the aligned 60 AIME requests with IDs 0..59")
    rows = rows[: args.limit]
    if any(
        not isinstance(row.get("input_ids"), list)
        or not row["input_ids"]
        or not isinstance(row.get("gold"), dict)
        or len(row["input_ids"]) + args.max_new_tokens > args.max_model_len
        for row in rows
    ):
        raise ValueError("Invalid input token IDs, gold row, or model length")

    config = {
        "method": args.method,
        "model": str(model_path),
        "upstream": str(upstream),
        "upstream_commit": args.upstream_commit,
        "inputs": str(args.inputs.resolve(strict=True)),
        "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "requests": len(rows),
        "kv_budget": args.kv_budget,
        "window_size": args.window_size,
        "kernel_size": args.kernel_size,
        "mix_lambda": args.mix_lambda,
        "compression_interval": args.compression_interval,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": 0,
        "seed_rule": "seed + request ID, reset before each request",
        "attention": "flash_attention_2",
        "compression_content": "all",
        "request_state_reset": "delete model.length; set model.config.compression=None",
    }
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("Resume config differs from the original run")
    else:
        output.mkdir(parents=True)
        (output / "samples").mkdir()
        manifest = {"status": "running", "config": config, "completed": 0}
        write_json(manifest_path, manifest)

    sys.path.insert(0, str(hf_source))
    if args.method == "rkv":
        from rkv.monkeypatch import replace_qwen3

        replace_qwen3({
            "method": "rkv",
            "method_config": {
                "budget": args.kv_budget,
                "window_size": args.window_size,
                "kernel_size": args.kernel_size,
                "mix_lambda": args.mix_lambda,
                "retain_ratio": 0.2,
                "retain_direction": "last",
            },
            "compression": None,
            "update_kv": True,
        })

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            attn_implementation="flash_attention_2",
        ).eval()
        devices = {parameter.device.type for parameter in model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"Model parameters are not entirely on CUDA: {devices}")
        if args.method == "rkv":
            model.config.update({
                "divide_method": "step_length",
                "divide_length": args.compression_interval,
                "compression_content": "all",
            })
            model.newline_token_ids = [tokenizer.encode(text)[-1] for text in (
                "\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"
            )]
            model.after_think_token_ids = [tokenizer.encode("</think>")[-1]]

        for row in rows:
            sample_path = output / "samples" / f"{int(row['id']):02d}.json"
            if sample_path.exists():
                sample = json.loads(sample_path.read_text(encoding="utf-8"))
                if sample.get("id") != str(row["id"]) or sample.get("status") != "success":
                    raise ValueError(f"Invalid saved sample {sample_path}")
                continue
            request_seed = args.seed + int(row["id"])
            random.seed(request_seed)
            np.random.seed(request_seed)
            torch.manual_seed(request_seed)
            torch.cuda.manual_seed_all(request_seed)
            if args.method == "rkv":
                # The upstream forward stores these fields on the model between
                # generate() calls; each AIME request needs a fresh decode count.
                if hasattr(model, "length"):
                    del model.length
                model.config.compression = None
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
            attention_mask = torch.ones_like(ids)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    num_beams=1,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            output_ids = generated.sequences[0, ids.shape[1]:].tolist()
            cache = generated.past_key_values
            cache_lengths = [key.shape[-2] for key in cache.key_cache]
            if len(cache_lengths) != model.config.num_hidden_layers or len(set(cache_lengths)) != 1:
                raise ValueError(f"Unexpected per-layer cache lengths: {cache_lengths}")
            if (args.method == "rkv" and ids.shape[1] + len(output_ids) >
                    args.kv_budget + args.compression_interval and
                    cache_lengths[0] >= ids.shape[1] + len(output_ids) - 1):
                raise ValueError("R-KV generated past its budget without evicting KV")
            sample = {
                "id": str(row["id"]),
                "status": "success",
                "seed": request_seed,
                "prompt_tokens": ids.shape[1],
                "completion_tokens": len(output_ids),
                "final_kv_tokens": cache_lengths[0],
                "completion_ids": output_ids,
                "text": tokenizer.decode(output_ids, skip_special_tokens=True),
                "generation_elapsed_s": elapsed,
            }
            write_json(sample_path, sample)
            del generated, cache
            manifest["completed"] = len(list((output / "samples").glob("*.json")))
            write_json(manifest_path, manifest)
            print(json.dumps({key: sample[key] for key in (
                "id", "completion_tokens", "generation_elapsed_s"
            )}), flush=True)

        samples = [json.loads((output / "samples" / f"{i:02d}.json").read_text())
                   for i in range(len(rows))]
        with (output / "raw_outputs.jsonl").open("w", encoding="utf-8") as raw, (
            output / "aime2024.jsonl"
        ).open("w", encoding="utf-8") as predictions:
            for row, sample in zip(rows, samples):
                raw.write(json.dumps(sample, ensure_ascii=False) + "\n")
                predictions.write(json.dumps({
                    "id": row["id"], "status": "success", "pred": sample["text"],
                    "gold": row["gold"],
                }, ensure_ascii=False) + "\n")
        manifest.update(
            status="generated",
            completed=len(samples),
            completion_tokens=sum(sample["completion_tokens"] for sample in samples),
            generation_elapsed_s=sum(sample["generation_elapsed_s"] for sample in samples),
        )
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
