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

import pyarrow.parquet as pq
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_MODEL_ROOT = Path(os.getenv("SPARSEENGINE_MODEL_ROOT", PROJECT_ROOT / "models"))
DEFAULT_DATA_ROOT = Path(os.getenv("SPARSEENGINE_DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("SPARSEENGINE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

BENCHMARKS = {
    "scienceqa_img": {
        "repo_id": "lmms-lab/ScienceQA-IMG",
        "default_dir": str(DEFAULT_DATA_ROOT / "ScienceQA-IMG_hf"),
        "default_split": "validation",
        "splits": {"train", "validation", "test"},
        "metric": "choice_accuracy",
    },
    "pope": {
        "repo_id": "lmms-lab/POPE",
        "default_dir": str(DEFAULT_DATA_ROOT / "POPE_hf"),
        "default_split": "test",
        "splits": {"test"},
        "metric": "yes_no_accuracy",
    },
    "mmbench_en": {
        "repo_id": "lmms-lab/MMBench_EN",
        "default_dir": str(DEFAULT_DATA_ROOT / "MMBench_EN_hf"),
        "default_split": "dev",
        "splits": {"dev", "test"},
        "metric": "choice_accuracy",
    },
    "mme": {
        "repo_id": "lmms-lab/MME",
        "default_dir": str(DEFAULT_DATA_ROOT / "MME_hf"),
        "default_split": "test",
        "splits": {"test"},
        "metric": "yes_no_accuracy",
    },
    "mmmu": {
        "repo_id": "lmms-lab/MMMU",
        "default_dir": str(DEFAULT_DATA_ROOT / "MMMU_hf"),
        "default_split": "validation",
        "splits": {"dev", "validation"},
        "metric": "mixed_accuracy",
        "force_batch_size": 1,
    },
}

CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small single-image benchmarks for LLaVA-OneVision: ScienceQA-IMG and POPE."
    )
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT / "llava-onevision-qwen2-7b-ov-hf"))
    parser.add_argument(
        "--pact_official_repo_dir",
        default=str(PROJECT_ROOT / "baselines/PACT"),
        help="PACT official source tree used by method=pact_official_repo.",
    )
    parser.add_argument(
        "--pact_official_pretrained",
        default="lmms-lab/llava-onevision-qwen2-7b-ov",
        help="Pretrained checkpoint passed to the PACT official LLaVA-OneVision loader.",
    )
    parser.add_argument(
        "--pact_official_config_path",
        default=str(PROJECT_ROOT / "baselines/PACT/configs/pact.json"),
        help="PACT visual-token reduction config JSON.",
    )
    parser.add_argument("--pact_official_conv_template", default="qwen_1_5")
    parser.add_argument("--pact_official_model_name", default="llava_qwen")
    parser.add_argument("--pact_official_attn_implementation", default="sdpa")
    parser.add_argument("--pact_official_cutoff", type=float, default=0.21)
    parser.add_argument("--pact_official_pruning_tokeep_percentage_value", type=float, default=0.55)
    parser.add_argument(
        "--model_family",
        default="llava_onevision",
        choices=["llava_onevision", "qwen3_vl"],
        help="Model adapter family. qwen3_vl currently supports vanilla only and runs batch_size=1.",
    )
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--methods", default="vanilla", help="Comma-separated model-adapter methods.")
    parser.add_argument("--num_samples", type=int, default=32, help="Use -1 for the full split.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--image_processor_use_fast", action="store_true")
    parser.add_argument("--visual_keep_ratio", type=float, default=1.0)
    parser.add_argument(
        "--pope_categories",
        default="all",
        help="POPE only: comma-separated adversarial,popular,random, or all.",
    )
    parser.add_argument("--dry_run_metadata", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--print_records", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    spec = BENCHMARKS[args.benchmark]
    if not args.split:
        args.split = spec["default_split"]
    if not args.dataset_dir:
        args.dataset_dir = spec["default_dir"]
    if not args.output_dir:
        args.output_dir = str(DEFAULT_OUTPUT_ROOT / "multimodal" / args.benchmark)
    if args.split not in spec["splits"]:
        raise ValueError(f"{args.benchmark} split must be one of {sorted(spec['splits'])}, got {args.split!r}.")
    if args.num_samples < -1:
        raise ValueError("--num_samples must be -1 for all rows or a non-negative count.")
    if args.num_samples == 0 and not args.dry_run_metadata:
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


def parquet_files(dataset_dir: Path, split: str) -> list[Path]:
    files = sorted((dataset_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} parquet files found under {dataset_dir / 'data'}")
    return files


def selected_pope_categories(raw: str) -> set[str] | None:
    values = {item.strip().lower() for item in str(raw).split(",") if item.strip()}
    if not values or values == {"all"}:
        return None
    valid = {"adversarial", "popular", "random"}
    unknown = values - valid
    if unknown:
        raise ValueError(f"Unknown POPE categories: {sorted(unknown)}; valid={sorted(valid)}")
    return values


def normalize_text(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(str(text).lower().translate(table).split())


def image_from_value(value: Any, *, source: str) -> Image.Image:
    if not isinstance(value, dict):
        raise ValueError(f"Image field from {source} must be a dict, got {type(value).__name__}.")
    if value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    path = value.get("path")
    if path:
        return Image.open(path).convert("RGB")
    raise ValueError(f"Image field from {source} has neither bytes nor path.")


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.lower() == "nan"


def literal_list(value: Any, *, source: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        import ast

        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Cannot parse list field from {source}: {value!r}") from exc
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Expected list-like field from {source}, got {type(value).__name__}: {value!r}")


def validate_scienceqa_row(raw: dict[str, Any], *, source: str, row_idx: int) -> dict[str, Any]:
    required = ("image", "question", "choices", "answer")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"ScienceQA-IMG row from {source} is missing fields: {missing}")
    choices = [str(choice).strip() for choice in raw["choices"]]
    if not choices:
        raise ValueError(f"ScienceQA-IMG row {row_idx} has no choices.")
    answer_idx = int(raw["answer"])
    if answer_idx < 0 or answer_idx >= len(choices):
        raise ValueError(f"ScienceQA-IMG row {row_idx} answer={answer_idx} out of range for {len(choices)} choices.")
    question = str(raw["question"]).strip()
    if not question:
        raise ValueError(f"ScienceQA-IMG row {row_idx} has empty question.")
    return {
        "benchmark": "scienceqa_img",
        "question_id": f"{source}:{row_idx}",
        "question": question,
        "choices": choices,
        "answer": CHOICE_LETTERS[answer_idx],
        "answer_text": choices[answer_idx],
        "hint": str(raw.get("hint") or "").strip(),
        "task": raw.get("task"),
        "grade": raw.get("grade"),
        "subject": raw.get("subject"),
        "topic": raw.get("topic"),
        "category": raw.get("category"),
        "image": raw["image"],
    }


def validate_pope_row(raw: dict[str, Any], *, source: str, row_idx: int) -> dict[str, Any]:
    required = ("image", "question", "answer")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"POPE row from {source} is missing fields: {missing}")
    answer = str(raw["answer"]).strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError(f"POPE row {row_idx} has invalid answer={raw['answer']!r}.")
    question = str(raw["question"]).strip()
    if not question:
        raise ValueError(f"POPE row {row_idx} has empty question.")
    return {
        "benchmark": "pope",
        "question_id": str(raw.get("id") or raw.get("question_id") or f"{source}:{row_idx}"),
        "image_id": raw.get("image_source"),
        "question": question,
        "answer": answer,
        "category": str(raw.get("category") or ""),
        "image": raw["image"],
    }


def validate_mmbench_row(raw: dict[str, Any], *, source: str, row_idx: int) -> dict[str, Any]:
    required = ("image", "question", "answer")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"MMBench_EN row from {source} is missing fields: {missing}")
    choices = []
    for letter in CHOICE_LETTERS[:8]:
        if letter in raw and not is_missing_value(raw[letter]):
            choices.append(str(raw[letter]).strip())
    if not choices:
        raise ValueError(f"MMBench_EN row {row_idx} has no choices.")
    answer = str(raw["answer"]).strip().upper()[:1]
    if answer not in CHOICE_LETTERS[: len(choices)]:
        raise ValueError(f"MMBench_EN row {row_idx} answer={raw['answer']!r} out of range for {len(choices)} choices.")
    question = str(raw["question"]).strip()
    if not question:
        raise ValueError(f"MMBench_EN row {row_idx} has empty question.")
    return {
        "benchmark": "mmbench_en",
        "question_id": str(raw.get("index") or f"{source}:{row_idx}"),
        "question": question,
        "choices": choices,
        "answer": answer,
        "answer_text": choices[CHOICE_LETTERS.index(answer)],
        "hint": "" if is_missing_value(raw.get("hint")) else str(raw.get("hint")).strip(),
        "category": raw.get("category"),
        "sub_category": raw.get("l2-category"),
        "source": raw.get("source"),
        "image": raw["image"],
    }


def validate_mme_row(raw: dict[str, Any], *, source: str, row_idx: int) -> dict[str, Any]:
    required = ("image", "question", "answer")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"MME row from {source} is missing fields: {missing}")
    answer = str(raw["answer"]).strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError(f"MME row {row_idx} has invalid answer={raw['answer']!r}.")
    question = str(raw["question"]).strip()
    if not question:
        raise ValueError(f"MME row {row_idx} has empty question.")
    return {
        "benchmark": "mme",
        "question_id": str(raw.get("question_id") or f"{source}:{row_idx}"),
        "question": question,
        "answer": answer,
        "category": raw.get("category"),
        "image": raw["image"],
    }


def validate_mmmu_row(raw: dict[str, Any], *, source: str, row_idx: int) -> dict[str, Any]:
    required = ("id", "question", "answer", "question_type")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"MMMU row from {source} is missing fields: {missing}")
    images = []
    for idx in range(1, 8):
        image = raw.get(f"image_{idx}")
        if isinstance(image, dict) and (image.get("bytes") or image.get("path")):
            images.append(image)
    if not images:
        raise ValueError(f"MMMU row {row_idx} has no images.")
    question = str(raw["question"]).strip()
    if not question:
        raise ValueError(f"MMMU row {row_idx} has empty question.")
    question_type = str(raw["question_type"]).strip().lower()
    answer = str(raw["answer"]).strip()
    if not answer:
        raise ValueError(f"MMMU row {row_idx} has empty answer.")
    choices = []
    if question_type == "multiple-choice":
        choices = [str(item).strip() for item in literal_list(raw.get("options"), source=source)]
        if not choices:
            raise ValueError(f"MMMU row {row_idx} has multiple-choice type but no options.")
        answer = answer.upper()[:1]
        if answer not in CHOICE_LETTERS[: len(choices)]:
            raise ValueError(f"MMMU row {row_idx} answer={raw['answer']!r} out of range for {len(choices)} choices.")
    return {
        "benchmark": "mmmu",
        "question_id": str(raw["id"]),
        "question": question,
        "choices": choices,
        "answer": answer,
        "answer_text": choices[CHOICE_LETTERS.index(answer)] if choices else answer,
        "question_type": question_type,
        "subfield": raw.get("subfield"),
        "topic_difficulty": raw.get("topic_difficulty"),
        "image": images[0],
        "images": images,
    }


def iter_raw_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    categories = selected_pope_categories(args.pope_categories) if args.benchmark == "pope" else None
    seen_after_filter = 0
    emitted = 0
    target = None if args.num_samples < 0 else args.num_samples
    for path in parquet_files(Path(args.dataset_dir), args.split):
        if target is not None and emitted >= target:
            break
        rows = pq.read_table(path).to_pylist()
        for row_idx, raw in enumerate(rows):
            if args.benchmark == "pope" and categories is not None and str(raw.get("category", "")).lower() not in categories:
                continue
            if seen_after_filter < args.sample_start:
                seen_after_filter += 1
                continue
            if target is not None and emitted >= target:
                break
            source = str(path)
            if args.benchmark == "scienceqa_img":
                row = validate_scienceqa_row(raw, source=source, row_idx=row_idx)
            elif args.benchmark == "pope":
                row = validate_pope_row(raw, source=source, row_idx=row_idx)
            elif args.benchmark == "mmbench_en":
                row = validate_mmbench_row(raw, source=source, row_idx=row_idx)
            elif args.benchmark == "mme":
                row = validate_mme_row(raw, source=source, row_idx=row_idx)
            elif args.benchmark == "mmmu":
                row = validate_mmmu_row(raw, source=source, row_idx=row_idx)
            else:
                raise AssertionError(args.benchmark)
            emitted += 1
            seen_after_filter += 1
            yield row


def selected_row_count(args: argparse.Namespace) -> int:
    categories = selected_pope_categories(args.pope_categories) if args.benchmark == "pope" else None
    total = 0
    for path in parquet_files(Path(args.dataset_dir), args.split):
        if args.benchmark != "pope" or categories is None:
            total += pq.ParquetFile(path).metadata.num_rows
            continue
        table = pq.read_table(path, columns=["category"])
        total += sum(str(item["category"]).lower() in categories for item in table.to_pylist())
    available = max(0, total - args.sample_start)
    return available if args.num_samples < 0 else min(args.num_samples, available)


def build_prompt_text(row: dict[str, Any]) -> str:
    if row["benchmark"] == "scienceqa_img":
        choices = "\n".join(f"{CHOICE_LETTERS[idx]}. {choice}" for idx, choice in enumerate(row["choices"]))
        context = f"Context: {row['hint']}\n" if row.get("hint") else ""
        text = (
            f"{context}Question: {row['question']}\n"
            f"Choices:\n{choices}\n"
            "Answer with only the option letter."
        )
    elif row["benchmark"] == "mmbench_en":
        choices = "\n".join(f"{CHOICE_LETTERS[idx]}. {choice}" for idx, choice in enumerate(row["choices"]))
        context = f"Context: {row['hint']}\n" if row.get("hint") else ""
        text = (
            f"{context}Question: {row['question']}\n"
            f"Choices:\n{choices}\n"
            "Answer with only the option letter."
        )
    elif row["benchmark"] == "mmmu":
        image_refs = " ".join(f"Image {idx + 1}." for idx in range(len(row.get("images", []))))
        question = re.sub(r"<image\s*(\d+)>", r"Image \1", row["question"], flags=re.IGNORECASE)
        if row.get("choices"):
            choices = "\n".join(f"{CHOICE_LETTERS[idx]}. {choice}" for idx, choice in enumerate(row["choices"]))
            text = (
                f"{image_refs}\nQuestion: {question}\n"
                f"Choices:\n{choices}\n"
                "Answer with only the option letter."
            )
        else:
            text = f"{image_refs}\nQuestion: {question}\nAnswer with a short phrase."
    elif row["benchmark"] == "pope":
        text = f"{row['question']}\nAnswer with only yes or no."
    elif row["benchmark"] == "mme":
        text = f"{row['question']}\nAnswer with only yes or no."
    else:
        raise AssertionError(row["benchmark"])
    return text


def build_prompt(processor: LlavaOnevisionProcessor, row: dict[str, Any]) -> str:
    text = build_prompt_text(row)
    image_count = len(row.get("images", [row["image"]]))
    content = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": text})
    conversation = [{"role": "user", "content": content}]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def parse_scienceqa_prediction(text: str, choices: list[str]) -> str:
    valid = CHOICE_LETTERS[: len(choices)]
    match = re.match(r"^\s*([A-Z])(?:[\.\):\s]|$)", str(text), flags=re.IGNORECASE)
    if match and match.group(1).upper() in set(valid):
        return match.group(1).upper()
    normalized = normalize_text(text)
    for idx, choice in enumerate(choices):
        if normalized == normalize_text(choice):
            return CHOICE_LETTERS[idx]
    for idx, choice in enumerate(choices):
        choice_norm = normalize_text(choice)
        if choice_norm and choice_norm in normalized:
            return CHOICE_LETTERS[idx]
    return ""


def parse_pope_prediction(text: str) -> str:
    stripped = str(text).strip().lower()
    if stripped.startswith("yes"):
        return "yes"
    if stripped.startswith("no"):
        return "no"
    match = YES_NO_RE.search(stripped)
    return match.group(1).lower() if match else ""


def parse_prediction(row: dict[str, Any], raw_text: str) -> str:
    if row["benchmark"] in {"scienceqa_img", "mmbench_en"}:
        return parse_scienceqa_prediction(raw_text, row["choices"])
    if row["benchmark"] == "mmmu":
        if row.get("choices"):
            return parse_scienceqa_prediction(raw_text, row["choices"])
        return normalize_text(str(raw_text).strip())
    if row["benchmark"] in {"pope", "mme"}:
        return parse_pope_prediction(raw_text)
    raise AssertionError(row["benchmark"])


def status_for_prediction(prediction: str) -> str:
    return "success" if prediction else "parse_failed"


def iter_batches(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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


def iter_requested_methods(methods: str, model_family: str = "llava_onevision") -> Iterable[tuple[str, str]]:
    if model_family == "qwen3_vl":
        from benchmark.multimodal.model_adapters.qwen3_vl import iter_requested_methods as iter_qwen3_methods

        yield from iter_qwen3_methods(methods)
        return
    from benchmark.multimodal.model_adapters.llava_onevision import iter_requested_methods as iter_llava_methods

    yield from iter_llava_methods(methods, allow_fastvid=False)


def load_method_model(method_kind: str, args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    if args.model_family == "qwen3_vl":
        from benchmark.multimodal.model_adapters.qwen3_vl import load_model_for_method

        return load_model_for_method(method_kind, args, dtype, device)

    from benchmark.multimodal.model_adapters.llava_onevision import load_model_for_method

    return load_model_for_method(method_kind, args, dtype, device)


def build_run_info(args: argparse.Namespace, output_dir: Path, row_count: int) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "git_commit": get_git_commit(),
        "benchmark": args.benchmark,
        "repo_id": BENCHMARKS[args.benchmark]["repo_id"],
        "model_path": args.model_path,
        "pact_official_repo": {
            "repo_dir": args.pact_official_repo_dir,
            "pretrained": args.pact_official_pretrained,
            "config_path": args.pact_official_config_path,
            "conv_template": args.pact_official_conv_template,
            "model_name": args.pact_official_model_name,
            "attn_implementation": args.pact_official_attn_implementation,
            "cutoff": args.pact_official_cutoff,
            "pruning_tokeep_percentage_value": args.pact_official_pruning_tokeep_percentage_value,
        },
        "model_family": args.model_family,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "output_dir": str(output_dir),
        "methods": args.methods,
        "num_samples_arg": args.num_samples,
        "sample_start": args.sample_start,
        "evaluated_sample_count": row_count,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "pope_categories": args.pope_categories,
        "seed": args.seed,
        "decoding": {
            "do_sample": False,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
        },
        "runtime_params": {
            "visual_keep_ratio": args.visual_keep_ratio,
        },
    }


def open_artifacts(output_dir: Path, method: str):
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
    processor,
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    output_dir: Path,
    row_count: int,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats(device)
    is_qwen3_vl = args.model_family == "qwen3_vl"
    official_repo_image = hasattr(model, "generate_image_qa")
    supports_batch = (method == "vanilla" and not is_qwen3_vl) if policy is None else bool(policy.get("supports_batch_generation", False))
    effective_batch_size = max(1, int(args.batch_size)) if supports_batch else 1
    if BENCHMARKS[args.benchmark].get("force_batch_size"):
        effective_batch_size = int(BENCHMARKS[args.benchmark]["force_batch_size"])
    if official_repo_image:
        if int(args.batch_size) != 1:
            raise RuntimeError(f"{method} uses an official repository runtime and requires batch_size=1.")
    else:
        ensure_left_padding(processor)

    artifact_paths, handles = open_artifacts(output_dir, method)
    status_counts: dict[str, int] = {}
    correct = 0
    total = 0
    total_new_tokens = 0
    total_seconds = 0.0
    total_batches = 0
    first_records = []

    try:
        for batch_start, batch_rows in enumerate(iter_batches(iter_raw_rows(args), effective_batch_size)):
            if args.benchmark == "mmmu":
                if len(batch_rows) != 1:
                    raise RuntimeError("MMMU uses batch_size=1 because samples may contain different image counts.")
                images = [
                    image_from_value(image, source=str(batch_rows[0]["question_id"]))
                    for image in batch_rows[0].get("images", [batch_rows[0]["image"]])
                ]
            else:
                images = [image_from_value(row["image"], source=str(row["question_id"])) for row in batch_rows]
            if official_repo_image:
                if len(batch_rows) != 1:
                    raise RuntimeError(f"{method} official repo image runtime requires single-sample batches.")
                torch.cuda.synchronize(device)
                call_start = time.perf_counter()
                output = model.generate_image_qa(
                    text=build_prompt_text(batch_rows[0]),
                    images=images,
                    max_new_tokens=int(args.max_new_tokens),
                )
                torch.cuda.synchronize(device)
                elapsed = float(output["generation_seconds"])
                decoded_batch = [str(output["text"])]
                input_len = int(output["input_tokens"])
                input_token_counts = [input_len]
                visual_token_counts = [int(output["visual_tokens"] or 0)]
                new_tokens = int(output["new_tokens"])
                call_elapsed = time.perf_counter() - call_start
            elif is_qwen3_vl:
                from benchmark.multimodal.model_adapters.qwen3_vl import decode_generated, prepare_inputs

                if len(batch_rows) != 1:
                    raise RuntimeError("Qwen3-VL image evaluator currently requires effective batch_size=1.")
                inputs, input_len, visual_tokens = prepare_inputs(
                    processor,
                    text=build_prompt_text(batch_rows[0]),
                    media_kind="image",
                    media=images,
                    device=device,
                    dtype=dtype,
                )
                input_token_counts = [int(inputs["input_ids"].shape[1])]
                visual_token_counts = [visual_tokens]
            else:
                prompts = [build_prompt(processor, row) for row in batch_rows]
                inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
                input_len = int(inputs["input_ids"].shape[1])
                attention_mask = inputs.get("attention_mask")
                input_token_counts = (
                    attention_mask.sum(dim=1).tolist() if attention_mask is not None else [input_len] * len(batch_rows)
                )
                image_token_id = getattr(model.config, "image_token_id", None)
                visual_token_counts = (
                    (inputs["input_ids"] == image_token_id).sum(dim=1).tolist()
                    if image_token_id is not None
                    else [0] * len(batch_rows)
                )
                inputs = batch_to_device(inputs, device, dtype)

            if not official_repo_image:
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

            if official_repo_image:
                pass
            elif is_qwen3_vl:
                decoded_batch = decode_generated(processor, output_ids, inputs["input_ids"])
                new_tokens = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
            else:
                generated_ids = output_ids[:, input_len:]
                decoded_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)
                new_tokens = int(generated_ids.shape[1])
            total_new_tokens += new_tokens * len(batch_rows)
            total_seconds += elapsed
            total_batches += 1

            for offset, (row, raw_decoded) in enumerate(zip(batch_rows, decoded_batch)):
                sample_idx = batch_start * effective_batch_size + offset + 1
                prediction = parse_prediction(row, raw_decoded)
                status = status_for_prediction(prediction)
                gold = row["answer"] if row.get("choices") or row["benchmark"] in {"pope", "mme"} else normalize_text(row["answer"])
                is_correct = status == "success" and prediction == gold
                correct += int(is_correct)
                total += 1
                status_counts[status] = status_counts.get(status, 0) + 1
                record = {
                    "status": status,
                    "benchmark": row["benchmark"],
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "normalized_answer": gold,
                    "answer_text": row.get("answer_text"),
                    "prediction": prediction,
                    "raw_prediction": raw_decoded,
                    "correct": bool(is_correct),
                    "category": row.get("category"),
                    "subject": row.get("subject"),
                    "topic": row.get("topic"),
                    "image_id": row.get("image_id"),
                    "input_tokens": int(input_token_counts[offset]),
                    "padded_input_tokens": input_len,
                    "visual_tokens": int(visual_token_counts[offset]),
                    "new_tokens": new_tokens,
                    "seconds": elapsed / len(batch_rows),
                    "batch_seconds": elapsed,
                }
                if official_repo_image:
                    record["end_to_end_sample_seconds"] = call_elapsed
                if len(first_records) < 5:
                    first_records.append(record)
                write_jsonl(
                    handles["raw_outputs"],
                    {"question_id": row["question_id"], "raw_prediction": raw_decoded},
                )
                write_jsonl(
                    handles["parsed_outputs"],
                    {
                        "question_id": row["question_id"],
                        "answer": prediction,
                        "status": status,
                        "correct": bool(is_correct),
                    },
                )
                write_jsonl(handles["per_sample_results"], record)
                if sample_idx <= 5 or sample_idx == row_count or sample_idx % args.log_every == 0:
                    print(
                        f"[{args.benchmark}:{method}] {sample_idx}/{row_count} "
                        f"batch={len(batch_rows)} input={input_token_counts[offset]} padded={input_len} "
                        f"visual={visual_token_counts[offset]} ans={row['answer']} pred={prediction or '?'} "
                        f"ok={bool(is_correct)}",
                        flush=True,
                    )
    finally:
        for handle in handles.values():
            handle.close()

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    accuracy = correct / total if total else 0.0
    metrics = {
        "benchmark": args.benchmark,
        "split": args.split,
        "method": method,
        "requested_method": requested_method,
        "num_samples": total,
        "correct": correct,
        "status_counts": status_counts,
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "batch_size": effective_batch_size,
        "total_batches": total_batches,
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_seconds,
        "new_tokens_per_s": total_new_tokens / total_seconds if total_seconds > 0 else 0.0,
        "examples_per_s": total / total_seconds if total_seconds > 0 else 0.0,
        "mean_batch_seconds": total_seconds / max(total_batches, 1),
        "peak_memory_gb": peak_gb,
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
    }
    if policy is not None:
        metrics["cache_policy"] = policy
    aggregate_path = output_dir / f"{method}_aggregate_metrics.json"
    aggregate_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics["artifact_paths"]["aggregate_metrics"] = str(aggregate_path)
    if args.print_records:
        metrics["first_records"] = first_records
    return metrics


def dry_run(args: argparse.Namespace, output_dir: Path, row_count: int) -> None:
    examples = []
    for idx, row in enumerate(iter_raw_rows(args)):
        if idx >= 3:
            break
        examples.append({key: value for key, value in row.items() if key not in {"image", "images"}})
    payload = {
        "benchmark": args.benchmark,
        "repo_id": BENCHMARKS[args.benchmark]["repo_id"],
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "rows": row_count,
        "sample_start": args.sample_start,
        "num_samples_arg": args.num_samples,
        "pope_categories": args.pope_categories if args.benchmark == "pope" else None,
        "examples": examples,
    }
    out_path = output_dir / f"{args.benchmark}_metadata_dry_run.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "path": str(out_path)}, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    init_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_count = selected_row_count(args)
    if row_count <= 0:
        raise RuntimeError(
            f"No rows selected: benchmark={args.benchmark} split={args.split} "
            f"sample_start={args.sample_start} num_samples={args.num_samples}"
        )
    print(f"[dataset] benchmark={args.benchmark} split={args.split} rows={row_count} dir={args.dataset_dir}", flush=True)
    if args.dry_run_metadata:
        dry_run(args, output_dir, row_count)
        return

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    method_pairs = list(iter_requested_methods(args.methods, args.model_family))
    has_pact_official = any(method_kind == "pact_official_repo" for _, method_kind in method_pairs)
    if has_pact_official and len(method_pairs) != 1:
        raise RuntimeError("pact_official_repo must run alone in a fresh evaluator process; do not mix it with HF methods.")
    needs_processor = not has_pact_official
    if args.model_family == "qwen3_vl":
        from benchmark.multimodal.model_adapters.qwen3_vl import load_processor

        processor = load_processor(args.model_path)
    elif needs_processor:
        from transformers import LlavaOnevisionProcessor

        processor = LlavaOnevisionProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=bool(args.image_processor_use_fast),
        )
        ensure_left_padding(processor)
    else:
        processor = None

    run_info = build_run_info(args, output_dir, row_count)
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    results = []
    for requested_method, method_kind in method_pairs:
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
        gc.collect()
        torch.cuda.empty_cache()

    summary_path = output_dir / f"last_{args.benchmark}_result.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    printable = results if args.print_records else [{k: v for k, v in item.items() if k != "first_records"} for item in results]
    print("[summary]")
    print(json.dumps(printable, indent=2, ensure_ascii=False), flush=True)
    print(f"[saved] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
