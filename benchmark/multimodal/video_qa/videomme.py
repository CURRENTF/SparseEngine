#!/usr/bin/env python3
import argparse
import gc
import json
import os
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_ROOT = Path(os.getenv("SPARSEENGINE_MODEL_ROOT", PROJECT_ROOT / "models"))
DEFAULT_DATA_ROOT = Path(os.getenv("SPARSEENGINE_DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("SPARSEENGINE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

from benchmark.multimodal.video_qa import streamingbench as streaming


VIDEOMME_PROMPT_TEMPLATE = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
{options}
The best answer is:"""

VIDEOMME_SUBTITLE_PROMPT_TEMPLATE = """This video's subtitles are listed below:
{subtitles}
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
{options}
The best answer is:"""

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
CHOICES = {"A", "B", "C", "D"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LLaVA-OneVision on Video-MME."
    )
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT / "llava-onevision-qwen2-0.5b-ov-hf"))
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATA_ROOT / "Video-MME_hf"))
    parser.add_argument("--annotation_path", default="")
    parser.add_argument("--video_dir", default="")
    parser.add_argument("--subtitle_dir", default="")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT / "multimodal" / "videomme"))
    parser.add_argument(
        "--durations",
        default="all",
        help="Comma-separated duration filters: short, medium, long, or all.",
    )
    parser.add_argument("--domains", default="", help="Optional comma-separated Video-MME domain filters.")
    parser.add_argument("--methods", default="vanilla")
    parser.add_argument("--num_samples", type=int, default=16, help="Rows to evaluate after filtering. Use -1 for all rows.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_video_frames", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--choice_parse_mode",
        default="official_first_char",
        choices=["official_first_char", "robust"],
    )
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--image_processor_use_fast", action="store_true")
    parser.add_argument("--frame_cache_dir", default="")
    parser.add_argument("--reuse_frame_cache", action="store_true")
    parser.add_argument("--frame_load_workers", type=int, default=1)
    parser.add_argument(
        "--preprocess_prefetch_batches",
        type=int,
        default=0,
        help=(
            "Number of future batches to preprocess concurrently while generation runs. "
            "0 disables prefetch. Values >1 use multiple CPU preprocessor workers and preserve output order."
        ),
    )
    parser.add_argument("--frame_sampling_backend", default="decord", choices=["decord"])
    parser.add_argument("--use_subtitles", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_missing_videos", action="store_true")
    parser.add_argument("--dry_run_metadata", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--print_records", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_samples < -1:
        raise ValueError("--num_samples must be -1 for all rows or a non-negative count.")
    if args.num_samples == 0 and not args.dry_run_metadata:
        raise ValueError("--num_samples=0 does not evaluate any rows.")
    if args.sample_start < 0:
        raise ValueError("--sample_start must be non-negative.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.num_video_frames < 1:
        raise ValueError("--num_video_frames must be >= 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1.")
    if args.log_every < 1:
        raise ValueError("--log_every must be >= 1.")
    if args.frame_load_workers < 1:
        raise ValueError("--frame_load_workers must be >= 1.")


def selected_values(raw: str, valid: set[str], *, field: str) -> set[str] | None:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    if not values or values == {"all"}:
        return None
    unknown = values - valid
    if unknown:
        raise ValueError(f"Unknown {field} filters: {sorted(unknown)}; valid={sorted(valid)}")
    return values


def build_video_index(video_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not video_dir.exists():
        return index
    for path in video_dir.rglob("*"):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        index.setdefault(path.stem, path)
    return index


def normalize_options(options) -> list[str]:
    values = [str(option).strip() for option in list(options)]
    if len(values) != 4:
        raise ValueError(f"Video-MME options must contain exactly 4 choices, got {len(values)}: {values!r}")
    return streaming._normalize_labeled_options(values)


def count_by_key(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def load_subtitle_text(subtitle_dir: Path, video_id: str) -> str:
    path = subtitle_dir / f"{video_id}.srt"
    if not path.exists():
        return ""
    lines = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        lines.append(line)
    return " ".join(lines)


def load_videomme_rows(args, *, resolve_videos: bool = True):
    dataset_dir = Path(args.dataset_dir)
    annotation_path = Path(args.annotation_path) if args.annotation_path else dataset_dir / "videomme" / "test-00000-of-00001.parquet"
    video_dir = Path(args.video_dir) if args.video_dir else dataset_dir / "videos"
    subtitle_dir = Path(args.subtitle_dir) if args.subtitle_dir else dataset_dir / "subtitle"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing Video-MME annotation parquet: {annotation_path}")
    if resolve_videos and not video_dir.exists():
        raise FileNotFoundError(f"Video-MME video directory does not exist: {video_dir}")

    df = pd.read_parquet(annotation_path)
    required_cols = {
        "video_id",
        "duration",
        "domain",
        "sub_category",
        "videoID",
        "question_id",
        "task_type",
        "question",
        "options",
        "answer",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Video-MME annotation is missing required columns: {missing_cols}")

    durations = selected_values(args.durations, {"short", "medium", "long"}, field="duration")
    domains = selected_values(args.domains, set(map(str, df["domain"].unique())), field="domain") if args.domains else None
    video_index = build_video_index(video_dir) if resolve_videos else {}
    selected_rows = []
    missing_videos = []

    for raw in df.to_dict("records"):
        duration = str(raw["duration"]).strip()
        domain = str(raw["domain"]).strip()
        if durations is not None and duration not in durations:
            continue
        if domains is not None and domain not in domains:
            continue

        answer = str(raw["answer"]).strip().upper()[:1]
        if answer not in CHOICES:
            raise ValueError(f"Video-MME row has invalid answer={raw['answer']!r} for question_id={raw['question_id']!r}.")

        video_id = str(raw["videoID"]).strip()
        video_path = video_index.get(video_id)
        if resolve_videos and video_path is None:
            missing_videos.append({"question_id": raw["question_id"], "videoID": video_id, "duration": duration})
            continue
        subtitles = load_subtitle_text(subtitle_dir, video_id) if args.use_subtitles else ""

        selected_rows.append(
            {
                "task": duration,
                "task_type": str(raw["task_type"]).strip(),
                "question_id": str(raw["question_id"]).strip(),
                "sample_id": int(str(raw["video_id"]).strip()),
                "video_id": str(raw["video_id"]).strip(),
                "videoID": video_id,
                "duration": duration,
                "domain": domain,
                "sub_category": str(raw["sub_category"]).strip(),
                "question": str(raw["question"]).strip(),
                "time_stamp": "full",
                "timestamp_seconds": 1.0e9,
                "answer": answer,
                "options": normalize_options(raw["options"]),
                "context": subtitles,
                "video_path": str(video_path) if video_path is not None else "",
            }
        )

    if missing_videos and not args.allow_missing_videos:
        raise FileNotFoundError(
            f"Missing videos for {len(missing_videos)} Video-MME rows; first missing: {missing_videos[:5]}. "
            "Download the video shards or pass --allow_missing_videos for an intentional partial run."
        )

    start = int(args.sample_start)
    rows = selected_rows[start:]
    if args.num_samples >= 0:
        rows = rows[: args.num_samples]
    return rows, {
        "annotation_path": str(annotation_path),
        "video_dir": str(video_dir),
        "subtitle_dir": str(subtitle_dir),
        "indexed_video_count": len(video_index),
        "missing_video_rows": len(missing_videos),
        "missing_video_examples": missing_videos[:10],
        "selected_rows_before_slice": len(selected_rows),
        "evaluated_duration_counts": count_by_key(rows, "duration"),
        "evaluated_domain_counts": count_by_key(rows, "domain"),
        "evaluated_sub_category_counts": count_by_key(rows, "sub_category"),
        "evaluated_task_type_counts": count_by_key(rows, "task_type"),
        "use_subtitles": bool(args.use_subtitles),
    }


def add_accuracy(stats: dict, correct: bool, status: str):
    stats["total"] += 1
    stats["correct"] += int(correct)
    stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1


def finalize_stats(stats: dict):
    for item in stats.values():
        total = max(item["total"], 1)
        item["accuracy"] = item["correct"] / total
        item["accuracy_pct"] = 100.0 * item["accuracy"]


def add_videomme_fields_and_stats(result: dict, rows: list[dict]) -> None:
    by_qid = {row["question_id"]: row for row in rows}
    duration_stats: dict[str, dict] = {}
    domain_stats: dict[str, dict] = {}
    sub_category_stats: dict[str, dict] = {}
    task_type_stats: dict[str, dict] = {}
    for record in result["records"]:
        row = by_qid[record["question_id"]]
        for key in ("video_id", "videoID", "duration", "domain", "sub_category"):
            record[key] = row[key]
        for stats, key in (
            (duration_stats, "duration"),
            (domain_stats, "domain"),
            (sub_category_stats, "sub_category"),
            (task_type_stats, "task_type"),
        ):
            bucket = stats.setdefault(record[key], {"total": 0, "correct": 0, "status_counts": {}})
            add_accuracy(bucket, bool(record["correct"]), record["status"])
    for stats in (duration_stats, domain_stats, sub_category_stats, task_type_stats):
        finalize_stats(stats)
    result["videomme_stats"] = {
        "duration": duration_stats,
        "domain": domain_stats,
        "sub_category": sub_category_stats,
        "task_type": task_type_stats,
    }


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


def build_run_info(args, dataset_info: dict, row_count: int) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "git_commit": get_git_commit(),
        "benchmark": "Video-MME",
        "model_path": args.model_path,
        "methods": args.methods,
        "dataset_dir": args.dataset_dir,
        "annotation_path": dataset_info["annotation_path"],
        "video_dir": dataset_info["video_dir"],
        "durations": args.durations,
        "domains": args.domains,
        "num_video_frames": args.num_video_frames,
        "frame_sampling_backend": args.frame_sampling_backend,
        "choice_parse_mode": args.choice_parse_mode,
        "prompt_template": VIDEOMME_SUBTITLE_PROMPT_TEMPLATE if args.use_subtitles else VIDEOMME_PROMPT_TEMPLATE,
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
        },
        "seed": args.seed,
        "sample_start": args.sample_start,
        "num_samples_arg": args.num_samples,
        "evaluated_sample_count": row_count,
        "dataset_info": dataset_info,
        "runtime_params": {
            "frame_load_workers": args.frame_load_workers,
            "preprocess_prefetch_batches": args.preprocess_prefetch_batches,
        },
    }


def init_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_videomme_prompt(processor, row: dict):
    options = "\n".join(row["options"])
    if row.get("context"):
        text = VIDEOMME_SUBTITLE_PROMPT_TEMPLATE.format(
            subtitles=row["context"],
            question=row["question"],
            options=options,
        )
    else:
        text = VIDEOMME_PROMPT_TEMPLATE.format(question=row["question"], options=options)
    conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def install_videomme_prompt() -> None:
    streaming.PROMPT_TEMPLATE = VIDEOMME_PROMPT_TEMPLATE
    streaming.SQA_PROMPT_TEMPLATE = VIDEOMME_SUBTITLE_PROMPT_TEMPLATE
    streaming.build_prompt = build_videomme_prompt


def main():
    args = parse_args()
    args.streamingbench_profile = "videomme"
    args.context_seconds = -1.0
    args.tasks = args.durations
    validate_args(args)
    init_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    install_videomme_prompt()

    rows, dataset_info = load_videomme_rows(args, resolve_videos=not args.dry_run_metadata)
    if args.dry_run_metadata:
        out_path = Path(args.output_dir) / "videomme_metadata_dry_run.json"
        out_path.write_text(json.dumps({"rows": len(rows), "dataset_info": dataset_info}, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"rows": len(rows), "dataset_info": dataset_info, "path": str(out_path)}, indent=2, ensure_ascii=False))
        return
    if not rows:
        raise RuntimeError(f"No Video-MME rows with available videos were found. Dataset info: {dataset_info}")

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    print(
        "[dataset] "
        f"rows={len(rows)} annotation={dataset_info['annotation_path']} "
        f"video_dir={dataset_info['video_dir']} indexed_videos={dataset_info['indexed_video_count']} "
        f"missing_video_rows={dataset_info['missing_video_rows']}",
        flush=True,
    )
    run_info = build_run_info(args, dataset_info, len(rows))

    from transformers import LlavaOnevisionProcessor

    processor = LlavaOnevisionProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=args.image_processor_use_fast,
    )
    results = []
    for requested_method, method_kind in streaming.iter_methods(args.methods):
        from benchmark.multimodal.model_adapters.llava_onevision import load_vanilla_model

        model = load_vanilla_model(args, dtype, device)
        method_label = "vanilla"
        policy = None

        result = streaming.run_method(method_label, model, processor, rows, args, dtype, device, policy=policy)
        result["requested_method"] = requested_method
        result["dataset_info"] = dataset_info
        add_videomme_fields_and_stats(result, rows)
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    streaming.add_vanilla_comparison(results)
    output_dir = Path(args.output_dir)
    for result in results:
        result["artifact_paths"] = streaming.save_method_artifacts(output_dir, result, run_info)

    out_path = output_dir / "last_videomme_result.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print("[summary]")
    if args.print_records:
        for result in results:
            for record in result["records"]:
                print(json.dumps(record, ensure_ascii=False))
    for result in results:
        print(
            f"{result['method']}: n={result['num_samples']} acc={result['accuracy_pct']:.2f}% "
            f"new_tok/s={result['new_tokens_per_s']:.2f} e2e_ex/s={result['end_to_end_examples_per_s']:.4f} "
            f"mem={result['peak_memory_gb']:.2f}GB artifacts={result['artifact_paths']['aggregate_metrics']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
