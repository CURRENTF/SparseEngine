#!/usr/bin/env python3
from __future__ import annotations

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

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.multimodal.video_qa import streamingbench as streaming
from benchmark.multimodal.video_qa.datasets import load_video_qa_rows

DEFAULT_MODEL_ROOT = Path(os.getenv("SPARSEENGINE_MODEL_ROOT", PROJECT_ROOT / "models"))
DEFAULT_DATA_ROOT = Path(os.getenv("SPARSEENGINE_DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("SPARSEENGINE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

DEFAULT_DATASET_DIRS = {
    "mvbench": str(DEFAULT_DATA_ROOT / "MVBench_hf"),
    "longvideobench": str(DEFAULT_DATA_ROOT / "LongVideoBench_hf"),
    "mlvu": str(DEFAULT_DATA_ROOT / "MLVU_hf"),
    "videomme": str(DEFAULT_DATA_ROOT / "Video-MME_modelscope"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified LLaVA-OneVision video QA evaluator.")
    parser.add_argument("--benchmark", required=True, choices=sorted(DEFAULT_DATASET_DIRS))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT / "llava-onevision-qwen2-7b-ov-hf"))
    parser.add_argument(
        "--fastvid_official_repo_dir",
        default=str(PROJECT_ROOT / "baselines/FastVID/fastvid_llavaonevision"),
        help="FastVID official LLaVA-OneVision source tree used by method=fastvid_official_repo.",
    )
    parser.add_argument(
        "--fastvid_official_pretrained",
        default="lmms-lab/llava-onevision-qwen2-7b-ov",
        help="Pretrained checkpoint passed to FastVID official LLaVA-OneVision loader.",
    )
    parser.add_argument("--fastvid_official_conv_template", default="qwen_1_5")
    parser.add_argument("--fastvid_official_model_name", default="llava_qwen")
    parser.add_argument("--fastvid_official_dyseg_c", type=int, default=8)
    parser.add_argument("--fastvid_official_dyseg_tau", type=float, default=0.9)
    parser.add_argument("--fastvid_official_stprune_d", type=float, default=0.4)
    parser.add_argument("--fastvid_official_dtm_p", type=int, default=4)
    parser.add_argument("--fastvid_official_dtm_beta", type=float, default=0.6)
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
    parser.add_argument("--annotation_dir", default="")
    parser.add_argument("--annotation_path", default="")
    parser.add_argument("--video_dir", default="")
    parser.add_argument("--subtitle_dir", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--methods", default="vanilla", help="Comma-separated model-adapter methods.")
    parser.add_argument("--num_samples", type=int, default=16, help="Use -1 for all rows.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_video_frames", type=int, default=32)
    parser.add_argument("--context_seconds", type=float, default=-1.0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--choice_parse_mode", default="official_first_char", choices=["official_first_char", "robust"])
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--image_processor_use_fast", action="store_true")
    parser.add_argument("--visual_keep_ratio", type=float, default=1.0)
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
    parser.add_argument("--frame_sampling_backend", default="decord", choices=["decord", "ffmpeg", "official_clip"])
    parser.add_argument("--durations", default="all", help="VideoMME only: short,medium,long,all.")
    parser.add_argument("--domains", default="", help="VideoMME only: optional comma-separated domain filter.")
    parser.add_argument("--use_subtitles", action="store_true", help="VideoMME only.")
    parser.add_argument("--allow_missing_videos", action="store_true")
    parser.add_argument("--dry_run_metadata", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--print_records", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < -1:
        raise ValueError("--num_samples must be -1 or a non-negative count.")
    if args.num_samples == 0 and not args.dry_run_metadata:
        raise ValueError("--num_samples=0 does not evaluate any rows.")
    if args.sample_start < 0:
        raise ValueError("--sample_start must be non-negative.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.num_video_frames < 1:
        raise ValueError("--num_video_frames must be >= 1.")
    if args.context_seconds == 0:
        raise ValueError("--context_seconds=0 produces an empty clip; use -1 for full video.")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1.")
    if args.frame_load_workers < 1:
        raise ValueError("--frame_load_workers must be >= 1.")
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
    completed = subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to read git commit: {completed.stderr.strip()}")
    return completed.stdout.strip()


def iter_methods(methods: str, model_family: str = "llava_onevision"):
    if model_family == "qwen3_vl":
        from benchmark.multimodal.model_adapters.qwen3_vl import iter_requested_methods as iter_qwen3_methods

        yield from iter_qwen3_methods(methods)
        return
    from benchmark.multimodal.model_adapters.llava_onevision import iter_requested_methods as iter_llava_methods

    yield from iter_llava_methods(methods, allow_fastvid=True)


def load_model_for_method(method_kind: str, args, dtype, device):
    if args.model_family == "qwen3_vl":
        from benchmark.multimodal.model_adapters.qwen3_vl import load_model_for_method as load_qwen3_model

        return load_qwen3_model(method_kind, args, dtype, device)

    from benchmark.multimodal.model_adapters.llava_onevision import load_model_for_method

    return load_model_for_method(method_kind, args, dtype, device)


def build_run_info(args, dataset_info: dict, row_count: int) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "git_commit": get_git_commit(),
        "benchmark": args.benchmark,
        "model_path": args.model_path,
        "fastvid_official_repo": {
            "repo_dir": args.fastvid_official_repo_dir,
            "pretrained": args.fastvid_official_pretrained,
            "conv_template": args.fastvid_official_conv_template,
            "model_name": args.fastvid_official_model_name,
            "dyseg_c": args.fastvid_official_dyseg_c,
            "dyseg_tau": args.fastvid_official_dyseg_tau,
            "stprune_d": args.fastvid_official_stprune_d,
            "dtm_p": args.fastvid_official_dtm_p,
            "dtm_beta": args.fastvid_official_dtm_beta,
        },
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
        "methods": args.methods,
        "dataset_dir": args.dataset_dir,
        "annotation_dir": args.annotation_dir,
        "annotation_path": args.annotation_path,
        "video_dir": args.video_dir,
        "num_video_frames": args.num_video_frames,
        "context_seconds": args.context_seconds,
        "frame_sampling_backend": args.frame_sampling_backend,
        "choice_parse_mode": args.choice_parse_mode,
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
            "visual_keep_ratio": args.visual_keep_ratio,
            "frame_load_workers": args.frame_load_workers,
            "preprocess_prefetch_batches": args.preprocess_prefetch_batches,
        },
    }


def add_benchmark_fields(result: dict, rows: list[dict]) -> None:
    by_qid = {row["question_id"]: row for row in rows}
    for record in result["records"]:
        row = by_qid.get(record["question_id"], {})
        record["benchmark"] = row.get("benchmark")
    result["benchmark"] = rows[0].get("benchmark") if rows else None


def validate_dataset_completeness(args: argparse.Namespace, dataset_info: dict) -> None:
    expected = dataset_info.get("expected_row_count")
    if expected is None:
        return
    if args.num_samples != -1:
        return
    if args.allow_missing_videos:
        return
    actual = int(dataset_info.get("selected_rows_before_slice", 0))
    if actual == int(expected) and int(dataset_info.get("skipped_rows", 0)) == 0:
        return
    examples = dataset_info.get("skipped_examples", [])
    reasons = dataset_info.get("skipped_by_reason", {})
    raise RuntimeError(
        f"{args.benchmark} full evaluation requires {expected} resolved rows, got {actual}. "
        f"skipped_rows={dataset_info.get('skipped_rows', 0)} skipped_by_reason={reasons}. "
        "This usually means the dataset media is incomplete. For MVBench, the Hugging Face README states "
        "that NTU RGB+D videos must be downloaded manually from ROSE Lab due to license restrictions; "
        "place those files under the MVBench video directory or pass --video_dir to a directory containing them. "
        "Use --allow_missing_videos only for partial/shard smoke runs. "
        f"First skipped examples: {examples[:5]}"
    )


def main() -> None:
    args = parse_args()
    benchmark = str(args.benchmark).lower()
    if not args.dataset_dir:
        args.dataset_dir = DEFAULT_DATASET_DIRS[benchmark]
    if not args.output_dir:
        args.output_dir = str(DEFAULT_OUTPUT_ROOT / "multimodal" / f"{benchmark}_unified_eval")
    args.streamingbench_profile = f"unified_{benchmark}"
    args.tasks = "all"
    validate_args(args)
    init_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, dataset_info = load_video_qa_rows(args)
    if args.dry_run_metadata:
        out_path = output_dir / f"{benchmark}_metadata_dry_run.json"
        out_path.write_text(json.dumps({"rows": len(rows), "dataset_info": dataset_info}, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"rows": len(rows), "dataset_info": dataset_info, "path": str(out_path)}, indent=2, ensure_ascii=False))
        return
    validate_dataset_completeness(args, dataset_info)
    if not rows:
        raise RuntimeError(f"No {benchmark} rows with resolved videos were found. Dataset info: {dataset_info}")

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    print(
        f"[dataset] benchmark={benchmark} rows={len(rows)} dataset_dir={args.dataset_dir} "
        f"video_dir={dataset_info.get('video_dir')} skipped={dataset_info.get('skipped_rows', 0)}",
        flush=True,
    )
    run_info = build_run_info(args, dataset_info, len(rows))

    method_pairs = list(iter_methods(args.methods, args.model_family))
    official_repo_methods = {"fastvid_official_repo", "pact_official_repo"}
    has_pact_official = any(method_kind == "pact_official_repo" for _, method_kind in method_pairs)
    if has_pact_official and len(method_pairs) != 1:
        raise RuntimeError("pact_official_repo must run alone in a fresh evaluator process; do not mix it with HF methods.")
    needs_processor = any(method_kind not in official_repo_methods for _, method_kind in method_pairs)
    if args.model_family == "qwen3_vl":
        from benchmark.multimodal.model_adapters.qwen3_vl import load_processor

        processor = load_processor(args.model_path)
    elif needs_processor:
        from transformers import LlavaOnevisionProcessor

        processor = LlavaOnevisionProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=args.image_processor_use_fast,
        )
    else:
        processor = None
    results = []
    for requested_method, method_kind in method_pairs:
        model, policy, method_label = load_model_for_method(method_kind, args, dtype, device)
        result = streaming.run_method(method_label, model, processor, rows, args, dtype, device, policy=policy)
        result["requested_method"] = requested_method
        result["dataset_info"] = dataset_info
        add_benchmark_fields(result, rows)
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    streaming.add_vanilla_comparison(results)
    for result in results:
        result["artifact_paths"] = streaming.save_method_artifacts(output_dir, result, run_info)

    out_path = output_dir / f"last_{benchmark}_result.json"
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
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
