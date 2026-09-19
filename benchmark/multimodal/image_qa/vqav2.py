#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from PIL import Image
from transformers import LlavaOnevisionProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_ROOT = Path(os.getenv("SPARSEENGINE_MODEL_ROOT", PROJECT_ROOT / "models"))
DEFAULT_DATA_ROOT = Path(os.getenv("SPARSEENGINE_DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("SPARSEENGINE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))


VQA_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "lets": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "shouldnt": "shouldn't",
    "shouldve": "should've",
    "thats": "that's",
    "thered": "there'd",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whats": "what's",
    "whenll": "when'll",
    "whens": "when's",
    "whered": "where'd",
    "wherell": "where'll",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "whyd": "why'd",
    "whyll": "why'll",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

VQA_DIGIT_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

VQA_ARTICLES = {"a", "an", "the"}
VQA_PUNCT = set(string.punctuation)
VQA_PERIOD_STRIP = re.compile(r"(?<!\d)\.(?!\d)")
VQA_COMMA_STRIP = re.compile(r"(?<=\d)(,)(?=\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standard-style VQAv2 evaluation for LLaVA-OneVision. Validation "
            "uses VQA soft accuracy; test/testdev write submission JSON without "
            "fabricating metrics."
        )
    )
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT / "llava-onevision-qwen2-7b-ov-hf"))
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATA_ROOT / "VQAv2"))
    parser.add_argument("--split", default="validation", choices=["train", "validation", "testdev", "test"])
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT / "multimodal" / "vqav2"))
    parser.add_argument("--methods", default="vanilla")
    parser.add_argument("--num_samples", type=int, default=32, help="Use -1 for the full split.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--prompt_suffix", default="Answer the question using a single word or phrase.")
    parser.add_argument("--prediction_parse_mode", default="raw_strip", choices=["raw_strip", "first_line"])
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--image_processor_use_fast", action="store_true")
    parser.add_argument("--visual_keep_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
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
    if not args.prompt_suffix.strip():
        raise ValueError("--prompt_suffix must not be empty.")


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


def normalize_vqa_answer(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\t", " ").strip().lower()
    text = VQA_COMMA_STRIP.sub("", text)
    text = VQA_PERIOD_STRIP.sub("", text)
    chars = []
    for char in text:
        if char in VQA_PUNCT and char not in {"'", ":"}:
            chars.append(" ")
        else:
            chars.append(char)
    words = []
    for word in " ".join("".join(chars).split()).split():
        mapped = VQA_DIGIT_MAP.get(word, word)
        if mapped not in VQA_ARTICLES:
            words.append(VQA_CONTRACTIONS.get(mapped, mapped))
    return " ".join(words)


def vqa_score(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    pred = normalize_vqa_answer(prediction)
    normalized_answers = [normalize_vqa_answer(answer) for answer in answers]
    if len(normalized_answers) == 1:
        return float(pred == normalized_answers[0])
    scores = []
    for idx, _answer in enumerate(normalized_answers):
        other_answers = normalized_answers[:idx] + normalized_answers[idx + 1 :]
        matching = sum(pred == other_answer for other_answer in other_answers)
        scores.append(min(1.0, matching / 3.0))
    return sum(scores) / len(scores)


def parse_prediction(text: str, mode: str) -> str:
    parsed = str(text).strip()
    if mode == "first_line":
        parsed = parsed.splitlines()[0].strip() if parsed else ""
    elif mode != "raw_strip":
        raise ValueError(f"Unsupported prediction_parse_mode: {mode}")
    return parsed


def answer_list(raw_answers: Any) -> list[str]:
    if raw_answers is None:
        return []
    values = []
    for item in raw_answers:
        if isinstance(item, dict):
            answer = item.get("answer")
        else:
            answer = item
        if answer is not None and str(answer).strip():
            values.append(str(answer))
    return values


def validate_vqav2_row(row: dict[str, Any], *, split: str, source: str) -> dict[str, Any]:
    required = ("question_id", "image_id", "question", "image")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"VQAv2 row from {source} is missing fields: {missing}")
    if not str(row["question"]).strip():
        raise ValueError(f"VQAv2 row from {source} has an empty question: question_id={row['question_id']!r}")
    image = row["image"]
    if not isinstance(image, dict) or image.get("bytes") is None:
        raise ValueError(f"VQAv2 row from {source} has no image bytes: question_id={row['question_id']!r}")
    answers = answer_list(row.get("answers"))
    has_ground_truth = bool(answers)
    if split in {"train", "validation"} and not has_ground_truth:
        raise ValueError(f"VQAv2 {split} row has no annotator answers: question_id={row['question_id']!r}")
    return {
        "question_id": int(row["question_id"]),
        "image_id": int(row["image_id"]),
        "question": str(row["question"]),
        "question_type": row.get("question_type"),
        "answer_type": row.get("answer_type"),
        "multiple_choice_answer": row.get("multiple_choice_answer"),
        "answers": answers,
        "has_ground_truth": has_ground_truth,
        "image_bytes": image["bytes"],
    }


def split_parquet_files(dataset_dir: Path, split: str) -> list[Path]:
    files = sorted((dataset_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No VQAv2 {split} parquet files found under {dataset_dir / 'data'}")
    return files


def selected_row_count(dataset_dir: Path, split: str, sample_start: int, num_samples: int) -> int:
    import pyarrow.parquet as pq

    total = sum(pq.ParquetFile(path).metadata.num_rows for path in split_parquet_files(dataset_dir, split))
    available = max(0, total - sample_start)
    return available if num_samples < 0 else min(num_samples, available)


def iter_vqav2_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    emitted = 0
    skipped = 0
    target = None if args.num_samples < 0 else args.num_samples
    columns = [
        "question_type",
        "multiple_choice_answer",
        "answers",
        "image_id",
        "answer_type",
        "question_id",
        "question",
        "image",
    ]
    for parquet_file in split_parquet_files(Path(args.dataset_dir), args.split):
        if target is not None and emitted >= target:
            break
        table = pq.read_table(parquet_file, columns=columns)
        for raw in table.to_pylist():
            if skipped < args.sample_start:
                skipped += 1
                continue
            if target is not None and emitted >= target:
                break
            emitted += 1
            yield validate_vqav2_row(raw, split=args.split, source=str(parquet_file))


def iter_batches(iterator: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch = []
    for row in iterator:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def to_rgb_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Cannot decode empty VQAv2 image bytes.")
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def build_prompt(processor: LlavaOnevisionProcessor, question: str, prompt_suffix: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question.strip()} {prompt_suffix.strip()}"},
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


def iter_requested_methods(methods: str) -> Iterable[tuple[str, str]]:
    from benchmark.multimodal.model_adapters.llava_onevision import iter_requested_methods as iter_methods

    yield from iter_methods(methods, allow_fastvid=False)


def load_method_model(method_kind: str, args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    from benchmark.multimodal.model_adapters.llava_onevision import load_model_for_method

    return load_model_for_method(method_kind, args, dtype, device)


def build_run_info(args: argparse.Namespace, output_dir: Path, row_count: int) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "git_commit": get_git_commit(),
        "benchmark": "vqav2",
        "model_path": args.model_path,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "output_dir": str(output_dir),
        "methods": args.methods,
        "num_samples_arg": args.num_samples,
        "sample_start": args.sample_start,
        "evaluated_sample_count": row_count,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_suffix": args.prompt_suffix,
        "prediction_parse_mode": args.prediction_parse_mode,
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
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    output_dir: Path,
    row_count: int,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats(device)
    supports_batch = method == "vanilla" if policy is None else bool(policy.get("supports_batch_generation", False))
    effective_batch_size = max(1, int(args.batch_size)) if supports_batch else 1
    ensure_left_padding(processor)

    artifact_paths, handles = open_artifacts(output_dir, method)
    submission = []
    status_counts: dict[str, int] = {}
    total_score = 0.0
    scored = 0
    total_new_tokens = 0
    total_seconds = 0.0
    total_batches = 0
    first_records = []

    try:
        rows = iter_vqav2_rows(args)
        for batch_start, batch_rows in enumerate(iter_batches(rows, effective_batch_size)):
            images = [to_rgb_image(row["image_bytes"]) for row in batch_rows]
            prompts = [build_prompt(processor, row["question"], args.prompt_suffix) for row in batch_rows]
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            input_len = int(inputs["input_ids"].shape[1])
            attention_mask = inputs.get("attention_mask")
            input_token_counts = attention_mask.sum(dim=1).tolist() if attention_mask is not None else [input_len] * len(batch_rows)
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
            total_new_tokens += new_tokens * len(batch_rows)
            total_seconds += elapsed
            total_batches += 1

            for offset, (row, raw_decoded) in enumerate(zip(batch_rows, decoded_batch)):
                sample_idx = batch_start * effective_batch_size + offset + 1
                prediction = parse_prediction(raw_decoded, args.prediction_parse_mode)
                score = vqa_score(prediction, row["answers"]) if row["has_ground_truth"] else None
                if score is not None:
                    total_score += score
                    scored += 1
                status = "success"
                status_counts[status] = status_counts.get(status, 0) + 1
                normalized_prediction = normalize_vqa_answer(prediction)
                normalized_answers = [normalize_vqa_answer(answer) for answer in row["answers"]]
                record = {
                    "status": status,
                    "question_id": row["question_id"],
                    "image_id": row["image_id"],
                    "question": row["question"],
                    "question_type": row["question_type"],
                    "answer_type": row["answer_type"],
                    "multiple_choice_answer": row["multiple_choice_answer"],
                    "answers": row["answers"],
                    "raw_prediction": raw_decoded,
                    "prediction": prediction,
                    "normalized_prediction": normalized_prediction,
                    "normalized_answers": normalized_answers,
                    "vqa_score": score,
                    "input_tokens": int(input_token_counts[offset]),
                    "padded_input_tokens": input_len,
                    "visual_tokens": int(visual_token_counts[offset]),
                    "new_tokens": new_tokens,
                    "seconds": elapsed / len(batch_rows),
                    "batch_seconds": elapsed,
                }
                if len(first_records) < 5:
                    first_records.append(record)
                write_jsonl(
                    handles["raw_outputs"],
                    {
                        "question_id": row["question_id"],
                        "image_id": row["image_id"],
                        "raw_prediction": raw_decoded,
                    },
                )
                write_jsonl(
                    handles["parsed_outputs"],
                    {
                        "question_id": row["question_id"],
                        "answer": prediction,
                        "normalized_answer": normalized_prediction,
                        "vqa_score": score,
                        "status": status,
                    },
                )
                write_jsonl(handles["per_sample_results"], record)
                submission.append({"question_id": row["question_id"], "answer": prediction})

                if sample_idx <= 5 or sample_idx == row_count or sample_idx % args.log_every == 0:
                    score_text = "NA" if score is None else f"{score:.3f}"
                    print(
                        f"[vqav2:{method}] {sample_idx}/{row_count} qid={row['question_id']} "
                        f"batch={len(batch_rows)} input={input_token_counts[offset]} padded={input_len} "
                        f"visual={visual_token_counts[offset]} score={score_text} pred={prediction[:80]!r}",
                        flush=True,
                    )
    finally:
        for handle in handles.values():
            handle.close()

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    mean_vqa_score = total_score / scored if scored else None
    metrics = {
        "benchmark": "vqav2",
        "split": args.split,
        "method": method,
        "requested_method": requested_method,
        "num_samples": sum(status_counts.values()),
        "scored_samples": scored,
        "status_counts": status_counts,
        "batch_size": effective_batch_size,
        "total_batches": total_batches,
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_seconds,
        "new_tokens_per_s": total_new_tokens / total_seconds if total_seconds > 0 else 0.0,
        "examples_per_s": sum(status_counts.values()) / total_seconds if total_seconds > 0 else 0.0,
        "mean_batch_seconds": total_seconds / max(total_batches, 1),
        "peak_memory_gb": peak_gb,
        "mean_vqa_score": mean_vqa_score,
        "accuracy_percent": mean_vqa_score * 100.0 if mean_vqa_score is not None else None,
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
    }
    if policy is not None:
        metrics["cache_policy"] = policy
    (output_dir / f"{method}_aggregate_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{method}_vqav2_submission.json").write_text(
        json.dumps(submission, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics["artifact_paths"]["aggregate_metrics"] = str(output_dir / f"{method}_aggregate_metrics.json")
    metrics["artifact_paths"]["submission"] = str(output_dir / f"{method}_vqav2_submission.json")
    if args.print_records:
        metrics["first_records"] = first_records
    return metrics


def main() -> None:
    args = parse_args()
    validate_args(args)
    init_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_count = selected_row_count(Path(args.dataset_dir), args.split, args.sample_start, args.num_samples)
    if row_count <= 0:
        raise RuntimeError(
            f"No VQAv2 rows selected: split={args.split} sample_start={args.sample_start} num_samples={args.num_samples}"
        )
    print(f"[dataset] benchmark=vqav2 split={args.split} rows={row_count} dataset_dir={args.dataset_dir}", flush=True)

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    processor = LlavaOnevisionProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=bool(args.image_processor_use_fast),
    )
    ensure_left_padding(processor)

    run_info = build_run_info(args, output_dir, row_count)
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    results = []
    for requested_method, method_kind in iter_requested_methods(args.methods):
        model, policy, method_label = load_method_model(method_kind, args, dtype, device)
        result = run_method(
            requested_method,
            method_label,
            model,
            policy,
            processor,
            args,
            dtype,
            device,
            output_dir,
            row_count,
        )
        results.append(result)
        del model
        torch.cuda.empty_cache()

    (output_dir / "last_vqav2_result.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    printable = results if args.print_records else [{k: v for k, v in item.items() if k != "first_records"} for item in results]
    print("[summary]")
    print(json.dumps(printable, indent=2, ensure_ascii=False), flush=True)
    print(f"[saved] {output_dir / 'last_vqav2_result.json'}", flush=True)


if __name__ == "__main__":
    main()
