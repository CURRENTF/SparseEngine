#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_ROOT = Path(os.getenv("SPARSEENGINE_DATA_DIR", PROJECT_ROOT / "data"))

from benchmark.multimodal.video_qa import streamingbench


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare StreamingBench frame cache without loading the VLM.")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATA_ROOT / "StreamingBench_hf"))
    parser.add_argument("--csv_dir", default="")
    parser.add_argument("--video_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frame_cache_dir", default="")
    parser.add_argument("--tasks", default="livevlm_table4")
    parser.add_argument(
        "--streamingbench_profile",
        default="livevlm_table4",
        choices=["custom", "official_60s", "official_all_context", "livevlm_table4"],
    )
    parser.add_argument("--num_video_frames", type=int, default=32)
    parser.add_argument("--context_seconds", type=float, default=-1.0)
    parser.add_argument("--frame_sampling_backend", default="decord", choices=["decord", "ffmpeg"])
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--allow_missing_videos", action="store_true")
    parser.add_argument("--overwrite_frame_cache", action="store_true")
    parser.add_argument(
        "--ffmpeg_fallback_on_decord_error",
        action="store_true",
        help="Opt in to ffmpeg extraction when decord fails; fallback use is recorded per row.",
    )
    parser.add_argument(
        "--official_clip_fallback_on_decord_error",
        action="store_true",
        help=(
            "Opt in to StreamingBench-style ffmpeg clip re-encoding when decord fails on the "
            "source video; fallback use is recorded per row."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--manifest_path", default="")
    return parser.parse_args()


def validate_args(args):
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    if args.log_every < 1:
        raise ValueError("--log_every must be >= 1.")
    if args.num_video_frames < 1:
        raise ValueError("--num_video_frames must be >= 1.")
    if args.context_seconds < 0 and args.context_seconds != -1:
        raise ValueError("--context_seconds must be -1 for all context or a non-negative window in seconds.")


def ffmpeg_prepare_into_current_cache(row: dict, worker_args) -> list[Path]:
    cache_dir, frame_paths, start_seconds, end_seconds = streamingbench.frame_cache_paths_for_row(row, worker_args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frame_paths:
        if frame_path.exists():
            frame_path.unlink()
    if worker_args.num_video_frames <= 1:
        timestamps = [(start_seconds + end_seconds) / 2.0]
    else:
        step = (end_seconds - start_seconds) / (worker_args.num_video_frames - 1)
        timestamps = [start_seconds + step * idx for idx in range(worker_args.num_video_frames)]
    for timestamp, frame_path in zip(timestamps, frame_paths):
        streamingbench.extract_required_frame(Path(row["video_path"]), timestamp, frame_path)
    return frame_paths


def official_clip_prepare_into_current_cache(row: dict, worker_args) -> tuple[list[Path], str]:
    cache_dir, frame_paths, start_seconds, end_seconds = streamingbench.frame_cache_paths_for_row(row, worker_args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frame_paths:
        if frame_path.exists():
            frame_path.unlink()
    stderr = streamingbench.official_clip_extract_context_frames(
        Path(row["video_path"]),
        start_seconds,
        end_seconds,
        len(frame_paths),
        frame_paths,
    )
    metadata = {
        "fallback": "official_clip_after_decord_error",
        "video_path": row["video_path"],
        "question_id": row["question_id"],
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "num_video_frames": len(frame_paths),
        "ffmpeg_stderr_tail": stderr[-2000:],
    }
    write_json(cache_dir / "fallback_metadata.json", metadata)
    return frame_paths, stderr


def worker_prepare(row: dict, args_dict: dict) -> dict:
    worker_args = SimpleNamespace(**args_dict)
    start = time.time()
    fallback = ""
    try:
        frame_paths = streamingbench.ensure_frame_cache(row, worker_args)
        missing = [str(path) for path in frame_paths if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"Missing/empty frame cache files after extraction: {missing[:5]}")
        status = "success"
        error = ""
    except Exception as exc:
        if worker_args.official_clip_fallback_on_decord_error and worker_args.frame_sampling_backend == "decord":
            try:
                frame_paths, _stderr = official_clip_prepare_into_current_cache(row, worker_args)
                missing = [str(path) for path in frame_paths if not path.exists() or path.stat().st_size == 0]
                if missing:
                    raise RuntimeError(f"Missing/empty official clip fallback frame files: {missing[:5]}")
                status = "success"
                fallback = "official_clip_after_decord_error"
                error = f"decord_error={repr(exc)}"
            except Exception as fallback_exc:
                frame_paths = []
                status = "frame_extract_failed"
                error = f"decord_error={repr(exc)}; official_clip_fallback_error={repr(fallback_exc)}"
        elif worker_args.ffmpeg_fallback_on_decord_error and worker_args.frame_sampling_backend == "decord":
            try:
                frame_paths = ffmpeg_prepare_into_current_cache(row, worker_args)
                missing = [str(path) for path in frame_paths if not path.exists() or path.stat().st_size == 0]
                if missing:
                    raise RuntimeError(f"Missing/empty ffmpeg fallback frame files: {missing[:5]}")
                status = "success"
                fallback = "ffmpeg_after_decord_error"
                error = f"decord_error={repr(exc)}"
            except Exception as fallback_exc:
                frame_paths = []
                status = "frame_extract_failed"
                error = f"decord_error={repr(exc)}; ffmpeg_fallback_error={repr(fallback_exc)}"
        else:
            frame_paths = []
            status = "frame_extract_failed"
            error = repr(exc)
    return {
        "status": status,
        "error": error,
        "fallback": fallback,
        "task": row["task"],
        "task_type": row["task_type"],
        "question_id": row["question_id"],
        "sample_id": row["sample_id"],
        "video_path": row["video_path"],
        "frame_count": len(frame_paths),
        "elapsed_sec": time.time() - start,
    }


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    validate_args(args)
    args = streamingbench.apply_streamingbench_profile(args)
    args.reuse_frame_cache = not args.overwrite_frame_cache

    rows, dataset_info = streamingbench.load_streamingbench_rows(args)
    streamingbench.validate_livevlm_table4_rows(args, rows)
    if not rows:
        raise RuntimeError("No StreamingBench rows selected for frame-cache preparation.")

    output_dir = Path(args.output_dir)
    frame_cache_dir = Path(args.frame_cache_dir) if args.frame_cache_dir else output_dir / "frame_cache"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    args.frame_cache_dir = str(frame_cache_dir)

    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / "frame_cache_manifest.json"
    records_path = manifest_path.with_suffix(".jsonl")
    if records_path.exists():
        records_path.unlink()

    args_dict = {
        "output_dir": str(output_dir),
        "frame_cache_dir": str(frame_cache_dir),
        "reuse_frame_cache": args.reuse_frame_cache,
        "num_video_frames": args.num_video_frames,
        "context_seconds": args.context_seconds,
        "frame_sampling_backend": args.frame_sampling_backend,
        "ffmpeg_fallback_on_decord_error": args.ffmpeg_fallback_on_decord_error,
        "official_clip_fallback_on_decord_error": args.official_clip_fallback_on_decord_error,
    }
    run_info = {
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_info": dataset_info,
        "config": {
            "dataset_dir": args.dataset_dir,
            "csv_dir": args.csv_dir,
            "video_dir": args.video_dir,
            "output_dir": str(output_dir),
            "frame_cache_dir": str(frame_cache_dir),
            "tasks": args.tasks,
            "streamingbench_profile": args.streamingbench_profile,
            "num_video_frames": args.num_video_frames,
            "context_seconds": args.context_seconds,
            "frame_sampling_backend": args.frame_sampling_backend,
            "ffmpeg_fallback_on_decord_error": args.ffmpeg_fallback_on_decord_error,
            "official_clip_fallback_on_decord_error": args.official_clip_fallback_on_decord_error,
            "num_samples": args.num_samples,
            "sample_start": args.sample_start,
            "allow_missing_videos": args.allow_missing_videos,
            "reuse_frame_cache": args.reuse_frame_cache,
            "workers": args.workers,
        },
    }

    counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    failures = []
    started = time.time()
    print(
        f"[prepare] rows={len(rows)} workers={args.workers} frame_cache_dir={frame_cache_dir} "
        f"reuse_frame_cache={args.reuse_frame_cache}",
        flush=True,
    )
    with records_path.open("a", encoding="utf-8") as records_handle:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker_prepare, row, args_dict) for row in rows]
            for done_count, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                counts[record["status"]] = counts.get(record["status"], 0) + 1
                if record.get("fallback"):
                    fallback_counts[record["fallback"]] = fallback_counts.get(record["fallback"], 0) + 1
                if record["status"] != "success":
                    failures.append(record)
                records_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if done_count <= 5 or done_count == len(rows) or done_count % args.log_every == 0:
                    rate = done_count / max(time.time() - started, 1e-6)
                    print(
                        f"[prepare] {done_count}/{len(rows)} status={record['status']} "
                        f"qid={record['question_id']} elapsed={record['elapsed_sec']:.2f}s "
                        f"rate={rate:.2f} rows/s counts={counts}",
                        flush=True,
                    )

    manifest = {
        **run_info,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
        "total_rows": len(rows),
        "status_counts": counts,
        "fallback_counts": fallback_counts,
        "failure_count": len(failures),
        "failure_examples": failures[:10],
        "records_path": str(records_path),
    }
    write_json(manifest_path, manifest)
    print(f"[prepare] wrote manifest={manifest_path}", flush=True)
    if failures:
        raise RuntimeError(f"Frame-cache preparation failed for {len(failures)} rows; first={failures[:3]}")


if __name__ == "__main__":
    main()
