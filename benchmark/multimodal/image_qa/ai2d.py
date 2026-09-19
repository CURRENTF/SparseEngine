#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import random
import re
import shlex
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from PIL import Image
from transformers import LlavaOnevisionProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_ROOT = Path(os.getenv("SPARSEENGINE_MODEL_ROOT", PROJECT_ROOT / "models"))
DEFAULT_DATA_ROOT = Path(os.getenv("SPARSEENGINE_DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("SPARSEENGINE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
DEFAULT_HF_CACHE = Path(os.getenv("HF_HOME", PROJECT_ROOT / ".hf_cache"))


PAPER_AI2D_TARGETS = {
    "llava-onevision-qwen2-0.5b-ov-hf": 57.1,
    "llava-onevision-qwen2-7b-ov-hf": 81.4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HF LLaVA-OneVision on AI2D with the LMMs-Eval default "
            "prompt format used by the LLaVA-OneVision paper."
        )
    )
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT / "llava-onevision-qwen2-0.5b-ov-hf"))
    parser.add_argument("--dataset_path", default="lmms-lab/ai2d")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATA_ROOT / "lmms-lab_ai2d"))
    parser.add_argument("--dataset_cache_dir", default=str(DEFAULT_HF_CACHE / "datasets"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT / "multimodal" / "ai2d"))
    parser.add_argument("--methods", default="vanilla", help="Comma-separated model-adapter methods.")
    parser.add_argument("--num_samples", type=int, default=32, help="Use -1 for the full AI2D test split.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--image_processor_use_fast", action="store_true")
    parser.add_argument("--visual_keep_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--print_records", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < -1:
        raise ValueError("--num_samples must be -1 for all rows or a non-negative count.")
    if args.num_samples == 0:
        raise ValueError("--num_samples=0 does not evaluate any rows.")
    if args.sample_start < 0:
        raise ValueError("--sample_start must be non-negative.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1.")
    if args.log_every < 1:
        raise ValueError("--log_every must be >= 1.")
    if args.visual_keep_ratio <= 0.0 or args.visual_keep_ratio > 1.0:
        raise ValueError("--visual_keep_ratio must be in (0, 1].")


def init_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to read git commit: {completed.stderr.strip()}")
    return completed.stdout.strip()


def resolve_local_parquet(dataset_dir: Path) -> list[str]:
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        return []
    return sorted(str(path) for path in data_dir.glob("test-*.parquet"))


def validate_ai2d_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    required = ("question", "options", "answer", "image")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"AI2D row from {source} is missing fields: {missing}")
    if not str(row["question"]).strip():
        raise ValueError(f"AI2D row from {source} has an empty question.")
    choices = [str(choice) for choice in row["options"]]
    if not choices:
        raise ValueError(f"AI2D row from {source} has no options.")
    answer_idx = int(row["answer"])
    if answer_idx < 0 or answer_idx >= len(choices):
        raise ValueError(f"AI2D row from {source} has answer out of range: {answer_idx} for {len(choices)} options.")
    return {
        "question": str(row["question"]),
        "options": choices,
        "answer": answer_idx,
        "image": row["image"],
    }


def load_ai2d_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_dir = Path(args.dataset_dir)
    parquet_files = resolve_local_parquet(dataset_dir)
    source = str(dataset_dir)
    if parquet_files:
        dataset = load_dataset("parquet", data_files={"test": parquet_files}, split="test")
    else:
        source = args.dataset_path
        dataset = load_dataset(args.dataset_path, split="test", cache_dir=args.dataset_cache_dir)

    start = args.sample_start
    stop = len(dataset) if args.num_samples < 0 else min(len(dataset), start + args.num_samples)
    if start >= len(dataset) or stop <= start:
        raise RuntimeError(
            f"No AI2D rows selected: total={len(dataset)} sample_start={args.sample_start} num_samples={args.num_samples}"
        )
    return [validate_ai2d_row(dataset[idx], source=source) for idx in range(start, stop)]


def normalize_for_match(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(str(text).lower().translate(table).split())


def extract_ai2d_choice(prediction: str, choices: list[str]) -> str:
    text = prediction.strip()
    direct = re.match(r"^\s*([A-Z])(?:[\.\):\s]|$)", text, flags=re.IGNORECASE)
    if direct:
        return direct.group(1).upper()

    normalized = normalize_for_match(text)
    for idx, choice in enumerate(choices):
        if normalized == normalize_for_match(choice):
            return chr(ord("A") + idx)

    for idx, choice in enumerate(choices):
        choice_norm = normalize_for_match(choice)
        if choice_norm and choice_norm in normalized:
            return chr(ord("A") + idx)
    return text[:32]


def to_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    raise TypeError(f"Unsupported AI2D image value: {type(image)!r}")


def build_ai2d_prompt(processor: LlavaOnevisionProcessor, row: dict[str, Any]) -> str:
    choices = list(row["options"])
    choices_str = "\n".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices))
    text = (
        f"{row['question']}\n"
        f"{choices_str}\n"
        "Answer with the option's letter from the given choices directly."
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def batch_to_device(inputs: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
    return inputs


def ensure_left_padding(processor: LlavaOnevisionProcessor) -> None:
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def iter_batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    for start in range(0, len(rows), batch_size):
        yield start, rows[start : start + batch_size]


def iter_requested_methods(methods: str) -> Iterable[tuple[str, str]]:
    from benchmark.multimodal.model_adapters.llava_onevision import iter_requested_methods as iter_methods

    yield from iter_methods(methods, allow_fastvid=False)


def load_method_model(method_kind: str, args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    from benchmark.multimodal.model_adapters.llava_onevision import load_model_for_method

    return load_model_for_method(method_kind, args, dtype, device)


def infer_paper_target(model_path: str) -> float | None:
    model_name = Path(model_path.rstrip("/")).name.lower()
    return PAPER_AI2D_TARGETS.get(model_name)


def build_run_info(args: argparse.Namespace, output_dir: Path, row_count: int) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "git_commit": get_git_commit(),
        "benchmark": "ai2d",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "dataset_dir": args.dataset_dir,
        "dataset_cache_dir": args.dataset_cache_dir,
        "output_dir": str(output_dir),
        "methods": args.methods,
        "num_samples_arg": args.num_samples,
        "sample_start": args.sample_start,
        "evaluated_sample_count": row_count,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "decoding": {
            "do_sample": False,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
        },
        "runtime_params": {"visual_keep_ratio": args.visual_keep_ratio},
    }


def open_artifacts(output_dir: Path, method: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_outputs": output_dir / f"{method}_raw_outputs.jsonl",
        "parsed_outputs": output_dir / f"{method}_parsed_outputs.jsonl",
        "per_sample_results": output_dir / f"{method}_per_sample_results.jsonl",
    }
    handles = {key: path.open("w", encoding="utf-8") for key, path in paths.items()}
    return paths, handles


def write_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.inference_mode()
def run_method(
    requested_method: str,
    method: str,
    model,
    policy: dict[str, Any] | None,
    processor: LlavaOnevisionProcessor,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats(device)
    effective_batch_size = max(1, int(args.batch_size)) if method == "vanilla" else 1
    ensure_left_padding(processor)
    artifact_paths, handles = open_artifacts(output_dir, method)

    records = []
    total_new_tokens = 0
    total_seconds = 0.0
    total_batches = 0
    status_counts: dict[str, int] = {}

    try:
        for batch_start, batch_rows in iter_batches(rows, effective_batch_size):
            images = [to_rgb_image(row["image"]) for row in batch_rows]
            prompts = [build_ai2d_prompt(processor, row) for row in batch_rows]
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            input_len = int(inputs["input_ids"].shape[1])
            input_token_counts = inputs["attention_mask"].sum(dim=1).tolist()
            image_token_id = getattr(model.config, "image_token_id", None)
            visual_token_counts = (
                (inputs["input_ids"] == image_token_id).sum(dim=1).tolist()
                if image_token_id is not None
                else [0] * len(batch_rows)
            )
            inputs = batch_to_device(inputs, device, dtype)

            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            generated_ids = output_ids[:, input_len:]
            decoded_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)
            new_tokens = int(generated_ids.shape[1])
            batch_new_tokens = new_tokens * len(batch_rows)
            total_new_tokens += batch_new_tokens
            total_seconds += elapsed
            total_batches += 1
            batch_tok_s = batch_new_tokens / elapsed if elapsed > 0 else 0.0

            for offset, (row, decoded) in enumerate(zip(batch_rows, decoded_batch)):
                sample_idx = batch_start + offset + 1
                choices = list(row["options"])
                prediction = decoded.strip()
                pred_choice = extract_ai2d_choice(prediction, choices)
                target_choice = chr(ord("A") + int(row["answer"]))
                correct = pred_choice == target_choice
                status = "success"
                status_counts[status] = status_counts.get(status, 0) + 1
                record = {
                    "status": status,
                    "sample_idx": args.sample_start + sample_idx - 1,
                    "question": row["question"],
                    "options": choices,
                    "answer_index": int(row["answer"]),
                    "target_choice": target_choice,
                    "raw_prediction": decoded,
                    "prediction": prediction,
                    "pred_choice": pred_choice,
                    "correct": correct,
                    "input_tokens": int(input_token_counts[offset]),
                    "padded_input_tokens": input_len,
                    "visual_tokens": int(visual_token_counts[offset]),
                    "new_tokens": new_tokens,
                    "seconds": elapsed / len(batch_rows),
                    "batch_seconds": elapsed,
                    "batch_new_tokens_per_s": batch_tok_s,
                }
                records.append(record)
                write_jsonl(
                    handles["raw_outputs"],
                    {
                        "sample_idx": record["sample_idx"],
                        "raw_prediction": decoded,
                    },
                )
                write_jsonl(
                    handles["parsed_outputs"],
                    {
                        "sample_idx": record["sample_idx"],
                        "answer": prediction,
                        "pred_choice": pred_choice,
                        "target_choice": target_choice,
                        "correct": correct,
                        "status": status,
                    },
                )
                write_jsonl(handles["per_sample_results"], record)
                if sample_idx <= 5 or sample_idx == len(rows) or sample_idx % args.log_every == 0:
                    print(
                        f"[ai2d:{method}] {sample_idx}/{len(rows)} batch={len(batch_rows)} "
                        f"input={input_token_counts[offset]} padded={input_len} visual={visual_token_counts[offset]} "
                        f"pred={pred_choice!r} target={target_choice!r} correct={correct} "
                        f"batch_time={elapsed:.3f}s batch_tok/s={batch_tok_s:.2f} raw={prediction[:80]!r}",
                        flush=True,
                    )
    finally:
        for handle in handles.values():
            handle.close()

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    accuracy = sum(record["correct"] for record in records) / max(len(records), 1)
    paper_target = infer_paper_target(args.model_path)
    metrics = {
        "benchmark": "ai2d",
        "method": method,
        "requested_method": requested_method,
        "model_path": args.model_path,
        "num_samples": len(records),
        "status_counts": status_counts,
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "total_batches": total_batches,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "paper_ai2d_target_percent": paper_target,
        "delta_vs_paper_points": (accuracy * 100.0 - paper_target) if paper_target is not None else None,
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_seconds,
        "new_tokens_per_s": total_new_tokens / total_seconds if total_seconds > 0 else 0.0,
        "examples_per_s": len(records) / total_seconds if total_seconds > 0 else 0.0,
        "mean_batch_seconds": total_seconds / max(total_batches, 1),
        "peak_memory_gb": peak_gb,
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
    }
    if policy is not None:
        metrics["cache_policy"] = policy
    aggregate_path = output_dir / f"{method}_aggregate_metrics.json"
    aggregate_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics["artifact_paths"]["aggregate_metrics"] = str(aggregate_path)
    return metrics


def main() -> None:
    args = parse_args()
    validate_args(args)
    init_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    rows = load_ai2d_rows(args)
    print(f"[dataset] benchmark=ai2d rows={len(rows)} dataset_dir={args.dataset_dir} output_dir={output_dir}", flush=True)

    processor = LlavaOnevisionProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=bool(args.image_processor_use_fast),
    )
    ensure_left_padding(processor)

    run_info = build_run_info(args, output_dir, len(rows))
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    results = []
    for requested_method, method_kind in iter_requested_methods(args.methods):
        model, policy, method_label = load_method_model(method_kind, args, dtype, device)
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        result = run_method(
            requested_method,
            method_label,
            model,
            policy,
            processor,
            rows,
            args,
            dtype,
            device,
            output_dir,
        )
        results.append(result)
        if method_label == "vanilla":
            legacy = dict(result)
            legacy["records"] = f"{result['num_samples']} records saved to {result['artifact_paths']['per_sample_results']}"
            (output_dir / "last_ai2d_vanilla_result.json").write_text(
                json.dumps(legacy, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    (output_dir / "last_ai2d_result.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    printable = results if args.print_records else [{k: v for k, v in item.items() if k != "records"} for item in results]
    print("[summary]")
    print(json.dumps(printable, indent=2, ensure_ascii=False), flush=True)
    print(f"[saved] {output_dir / 'last_ai2d_result.json'}", flush=True)


if __name__ == "__main__":
    main()
