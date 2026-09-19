#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaOnevisionProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_ROOT = Path(os.getenv("SPARSEENGINE_MODEL_ROOT", PROJECT_ROOT / "models"))
DEFAULT_DATA_ROOT = Path(os.getenv("SPARSEENGINE_DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("SPARSEENGINE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

from benchmark.multimodal.model_adapters.llava_onevision import (
    batch_to_device,
    ensure_left_padding,
    load_vanilla_model,
)


CHOICE_LETTERS = "ABCDEFGH"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLaVA-OneVision on QAEGO4D-test-mc using the ReKV paper "
            "multiple-choice protocol: 0.5 FPS video stream, 64 QA context frames, "
            "ReKV-style prompt, and accuracy."
        )
    )
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_ROOT / "llava-onevision-qwen2-7b-ov-hf"))
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATA_ROOT / "rekv_qaego4d"))
    parser.add_argument("--anno_path", default="")
    parser.add_argument("--video_dir", default="")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT / "multimodal" / "rekv_qaego4d"))
    parser.add_argument("--methods", default="vanilla")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of QA pairs to evaluate. Use -1 for all 500.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--max_context_frames", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=7)
    parser.add_argument("--torch_dtype", default="float16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--frame_cache_dir", default="")
    parser.add_argument("--reuse_frame_cache", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--print_records", action="store_true")
    return parser.parse_args()


def decord_context_frame_indices(video_path: Path, sample_fps: float, max_context_frames: int):
    try:
        from decord import VideoReader, cpu
    except ImportError as e:
        raise RuntimeError(
            "QAEGO4D ReKV-style frame sampling requires decord. "
            "Install decord before running this benchmark."
        ) from e
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        if len(vr) <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        avg_fps = float(vr.get_avg_fps())
        rounded_fps = max(round(avg_fps), 1)
        step_frames = max(int(rounded_fps / sample_fps), 1)
        frame_indices = list(range(0, len(vr), step_frames))
        if len(frame_indices) > max_context_frames:
            keep_indices = torch.linspace(0, len(frame_indices) - 1, steps=max_context_frames).round().long().tolist()
            frame_indices = [frame_indices[idx] for idx in keep_indices]
        frame_indices = [min(max(0, idx), max(len(vr) - 1, 0)) for idx in frame_indices]
        duration = len(vr) / max(avg_fps, 1e-6)
        return duration, avg_fps, frame_indices
    except Exception as e:
        raise RuntimeError(f"Failed to read video with decord for ReKV sampling: {video_path}") from e


def decord_extract_frames_by_index(video_path: Path, frame_indices: list[int], output_paths: list[Path]):
    try:
        from decord import VideoReader, cpu
    except ImportError as e:
        raise RuntimeError(
            "QAEGO4D ReKV-style frame extraction requires decord. "
            "Install decord before running this benchmark."
        ) from e
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        if len(vr) <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        safe_indices = [min(max(0, idx), max(len(vr) - 1, 0)) for idx in frame_indices]
        batch = vr.get_batch(safe_indices).asnumpy()
        if len(batch) == 0:
            raise RuntimeError(f"Decord returned no frames for video={video_path}")
        for frame, output_path in zip(batch, output_paths):
            Image.fromarray(frame).save(output_path, quality=95)
        missing = [str(path) for path in output_paths if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"Decord extraction produced missing/empty frame files: {missing[:5]}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract QAEGO4D frames with decord: {video_path}") from e


def frame_cache_key(video_path: Path, sample_fps: float, max_context_frames: int) -> str:
    raw = f"{video_path.resolve()}:{sample_fps:.6f}:{max_context_frames}:decord:v2"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def load_video_frames(video_path: Path, args) -> list[Image.Image]:
    cache_root = Path(args.frame_cache_dir) if args.frame_cache_dir else Path(args.output_dir) / "frame_cache"
    cache_dir = cache_root / frame_cache_key(video_path, args.sample_fps, args.max_context_frames)
    if cache_dir.exists() and not args.reuse_frame_cache:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _duration, _fps, frame_indices = decord_context_frame_indices(video_path, args.sample_fps, args.max_context_frames)
    frame_paths = [cache_dir / f"frame_{idx:03d}.jpg" for idx in range(len(frame_indices))]
    if not (args.reuse_frame_cache and all(path.exists() and path.stat().st_size > 0 for path in frame_paths)):
        for frame_path in frame_paths:
            if frame_path.exists():
                frame_path.unlink()
        decord_extract_frames_by_index(video_path, frame_indices, frame_paths)

    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB").copy())
    return frames


def resolve_video_path(video_path: str, dataset_dir: Path, video_dir: Path) -> Path:
    raw = Path(video_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(dataset_dir / raw)
    candidates.append(video_dir / raw.name)
    candidates.append(video_dir / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve video path {video_path!r}; tried {[str(c) for c in candidates]}")


def load_qaego4d_rows(args):
    dataset_dir = Path(args.dataset_dir)
    anno_path = Path(args.anno_path) if args.anno_path else dataset_dir / "test_mc.json"
    video_dir = Path(args.video_dir) if args.video_dir else dataset_dir / "videos"
    data = json.loads(anno_path.read_text())
    rows = []
    for video_sample in data:
        video_path = resolve_video_path(video_sample["video_path"], dataset_dir, video_dir)
        for conv_idx, sample in enumerate(video_sample["conversations"]):
            choices = list(sample["choices"])
            if not choices or len(choices) > len(CHOICE_LETTERS):
                raise ValueError(
                    f"QAEGO4D sample has invalid choices length={len(choices)}. "
                    f"video_id={video_sample.get('video_id')!r} conv_idx={conv_idx}"
                )
            answer = sample["answer"]
            if answer is None:
                raise ValueError(
                    f"QAEGO4D sample has answer=None; refusing to score with a default. "
                    f"video_id={video_sample.get('video_id')!r} conv_idx={conv_idx}"
                )
            if answer not in choices:
                raise ValueError(
                    f"QAEGO4D answer is not in choices. video_id={video_sample.get('video_id')!r} "
                    f"conv_idx={conv_idx} answer={answer!r} choices={choices!r}"
                )
            correct_choice = CHOICE_LETTERS[choices.index(answer)]
            rows.append(
                {
                    "video_id": video_sample["video_id"],
                    "conv_idx": conv_idx,
                    "video_path": str(video_path),
                    "duration": float(video_sample.get("duration", 0.0) or 0.0),
                    "question": sample["question"],
                    "choices": choices,
                    "answer": answer,
                    "correct_choice": correct_choice,
                    "temporal_windows": sample.get("temporal_windows", []),
                }
            )

    start = max(0, int(args.sample_start))
    rows = rows[start:]
    if args.num_samples >= 0:
        rows = rows[: args.num_samples]
    return rows, {"anno_path": str(anno_path), "video_dir": str(video_dir), "num_videos": len(data)}


def build_rekv_prompt(processor, question: str, choices: list[str]) -> str:
    formatted_choices = "\n".join(
        f"({CHOICE_LETTERS[idx]}) {candidate}" for idx, candidate in enumerate(choices)
    )
    formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": formatted_question}]},
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True) + "Best option: ("


def extract_rekv_choice(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if ")" in text:
        index = text.index(")")
        if index > 0:
            return text[index - 1 : index].upper()
    return text[0].upper()


def iter_batches(rows, batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield start, rows[start : start + batch_size]


@torch.inference_mode()
def run_method(method: str, model, processor, rows: list[dict], args, dtype, device, policy=None):
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    total_new_tokens = 0
    total_time = 0.0
    total_batches = 0
    effective_batch_size = max(1, int(args.batch_size))
    if effective_batch_size > 1:
        ensure_left_padding(processor)
    elif getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token_id is None:
        ensure_left_padding(processor)

    log_every = max(1, int(args.log_every))
    for batch_start, batch_rows in iter_batches(rows, effective_batch_size):
        videos = [load_video_frames(Path(row["video_path"]), args) for row in batch_rows]
        prompts = [build_rekv_prompt(processor, row["question"], row["choices"]) for row in batch_rows]
        processor_kwargs = {"text": prompts, "videos": videos, "return_tensors": "pt"}
        if len(batch_rows) > 1:
            processor_kwargs["padding"] = True
        inputs = processor(**processor_kwargs)
        input_len = int(inputs["input_ids"].shape[1])
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            input_token_counts = attention_mask.sum(dim=1).tolist()
        else:
            input_token_counts = [input_len for _ in batch_rows]
        video_token_counts = (inputs["input_ids"] == model.config.video_token_id).sum(dim=1).tolist()
        inputs = batch_to_device(inputs, device, dtype)

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        generated_ids = output_ids[:, input_len:]
        decoded_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)
        new_tokens = int(generated_ids.shape[1])
        batch_new_tokens = new_tokens * len(batch_rows)
        total_new_tokens += batch_new_tokens
        total_time += elapsed
        total_batches += 1
        batch_tok_s = batch_new_tokens / elapsed if elapsed > 0 else 0.0

        for offset, (row, decoded) in enumerate(zip(batch_rows, decoded_batch)):
            sample_idx = batch_start + offset + 1
            decoded = decoded.strip()
            prediction = extract_rekv_choice(decoded)
            correct = prediction == row["correct_choice"]
            record = {
                "video_id": row["video_id"],
                "conv_idx": row["conv_idx"],
                "video_path": row["video_path"],
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "correct_choice": row["correct_choice"],
                "pred_answer": decoded,
                "pred_choice": prediction,
                "qa_acc": float(correct) * 100.0,
                "input_tokens": int(input_token_counts[offset]),
                "padded_input_tokens": input_len,
                "video_tokens": int(video_token_counts[offset]),
                "new_tokens": new_tokens,
                "seconds": elapsed / len(batch_rows),
                "batch_seconds": elapsed,
                "new_tokens_per_s": new_tokens / (elapsed / len(batch_rows)) if elapsed > 0 else 0.0,
                "batch_new_tokens_per_s": batch_tok_s,
            }
            records.append(record)
            if sample_idx <= 5 or sample_idx == len(rows) or sample_idx % log_every == 0:
                print(
                    f"[{method}] {sample_idx}/{len(rows)} video={row['video_id']} "
                    f"batch={len(batch_rows)} input={input_token_counts[offset]} padded={input_len} "
                    f"video_tokens={video_token_counts[offset]} new={new_tokens} "
                    f"batch_time={elapsed:.3f}s batch_tok/s={batch_tok_s:.2f} "
                    f"ans={row['correct_choice']} pred={prediction or '?'} ok={correct} raw={decoded[:80]!r}",
                    flush=True,
                )

    total = max(len(records), 1)
    result = {
        "method": method,
        "num_samples": len(records),
        "sample_fps": args.sample_fps,
        "max_context_frames": args.max_context_frames,
        "batch_size": effective_batch_size,
        "total_batches": total_batches,
        "total_new_tokens": total_new_tokens,
        "total_seconds": total_time,
        "new_tokens_per_s": total_new_tokens / total_time if total_time > 0 else 0.0,
        "examples_per_s": len(records) / total_time if total_time > 0 else 0.0,
        "mean_batch_seconds": total_time / max(total_batches, 1),
        "mean_seconds": total_time / total,
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "qa_acc": sum(record["qa_acc"] for record in records) / total,
        "records": records,
    }
    if policy is not None:
        result["visual_cache_policy"] = policy
    return result


def iter_methods(methods: str):
    for raw_method in [part.strip() for part in methods.split(",") if part.strip()]:
        method = raw_method.lower()
        if method == "vanilla":
            yield "vanilla", "vanilla"
        else:
            raise ValueError("QAEGO4D ReKV protocol script supports only method: vanilla.")


def write_official_style_csv(result: dict, output_dir: Path):
    csv_path = output_dir / f"{result['method']}_results.csv"
    fieldnames = [
        "video_id",
        "question",
        "choices",
        "answer",
        "correct_choice",
        "pred_answer",
        "pred_choice",
        "qa_acc",
        "retrieve_size",
        "chunk_size",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in result["records"]:
            writer.writerow(
                {
                    "video_id": record["video_id"],
                    "question": record["question"],
                    "choices": record["choices"],
                    "answer": record["answer"],
                    "correct_choice": record["correct_choice"],
                    "pred_answer": record["pred_answer"],
                    "pred_choice": record["pred_choice"],
                    "qa_acc": record["qa_acc"],
                    "retrieve_size": result["max_context_frames"],
                    "chunk_size": 1,
                }
            )
    return csv_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    rows, dataset_info = load_qaego4d_rows(args)
    if not rows:
        raise RuntimeError("No QAEGO4D rows selected.")
    print(
        "[dataset] "
        f"rows={len(rows)} anno_path={dataset_info['anno_path']} video_dir={dataset_info['video_dir']} "
        f"sample_fps={args.sample_fps} max_context_frames={args.max_context_frames}",
        flush=True,
    )

    processor = LlavaOnevisionProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    results = []
    for requested_method, method_kind in iter_methods(args.methods):
        model = load_vanilla_model(args, dtype, device)
        method_label = "vanilla"
        policy = None

        result = run_method(method_label, model, processor, rows, args, dtype, device, policy=policy)
        result["requested_method"] = requested_method
        result["dataset_info"] = dataset_info
        result["official_rekv_protocol"] = {
            "benchmark": "QAEGO4Dtest-mc",
            "metric": "accuracy",
            "sample_fps": args.sample_fps,
            "n_local": 15000,
            "retrieve_size_or_context_frames": args.max_context_frames,
            "prompt_style": "ReKV official MC prompt ending with 'Best option: ('",
        }
        result["csv_path"] = str(write_official_style_csv(result, output_dir))
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if len(results) == 2:
        base = next((item for item in results if item["method"] == "vanilla"), None)
        candidate = next((item for item in results if item["method"] != "vanilla"), None)
        if base and candidate:
            candidate["qa_acc_delta_vs_vanilla"] = candidate["qa_acc"] - base["qa_acc"]
            candidate["speedup_vs_vanilla"] = candidate["new_tokens_per_s"] / base["new_tokens_per_s"]
            candidate["memory_delta_gb_vs_vanilla"] = candidate["peak_memory_gb"] - base["peak_memory_gb"]

    out_path = output_dir / "last_rekv_qaego4d_result.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print("[summary]")
    if args.print_records:
        printable_results = results
    else:
        printable_results = []
        for result in results:
            item = dict(result)
            item["records"] = f"{len(result.get('records', []))} records saved to {out_path}"
            printable_results.append(item)
    print(json.dumps(printable_results, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
