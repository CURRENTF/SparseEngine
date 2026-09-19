#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / "data")
DEFAULT_HF_CACHE = os.getenv("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

DATASETS = {
    "mvbench": {
        "repo_id": "OpenGVLab/MVBench",
        "local_dir": "MVBench_hf",
        "metadata": ["README.md", "json/**"],
        "full": ["README.md", "json/**", "video/**"],
    },
    "longvideobench": {
        "repo_id": "longvideobench/LongVideoBench",
        "local_dir": "LongVideoBench_hf",
        "metadata": ["README.md", "*.json", "*.parquet", "subtitles.tar"],
        "full": ["README.md", "*.json", "*.parquet", "subtitles.tar", "videos.tar.part.*"],
        "gated": True,
    },
    "mlvu": {
        "repo_id": "sy1998/MLVU",
        "local_dir": "MLVU_hf",
        "metadata": ["README.md", "MC/MC.parquet", "MLVU/json/**"],
        "full": ["README.md", "MC/MC.parquet", "MLVU/json/**", "MLVU/video/**"],
    },
    "videomme": {
        "repo_id": "lmms-lab/Video-MME",
        "local_dir": "Video-MME_hf",
        "metadata": ["README.md", "videomme/test-00000-of-00001.parquet", "subtitle.zip"],
        "full": ["README.md", "videomme/test-00000-of-00001.parquet", "subtitle.zip", "videos_chunked_*.zip"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download video QA benchmark metadata or full media.")
    parser.add_argument("--benchmark", default="all", choices=["all", *sorted(DATASETS)])
    parser.add_argument("--scope", default="metadata", choices=["metadata", "full"])
    parser.add_argument("--data_root", default=os.getenv("SPARSEENGINE_DATA_DIR", DEFAULT_DATA_ROOT))
    parser.add_argument("--cache_dir", default=DEFAULT_HF_CACHE)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--local_dir", default="", help="Override local dir for single-benchmark downloads.")
    parser.add_argument("--token", default=None)
    parser.add_argument("--list_only", action="store_true", help="List matching repo files without downloading.")
    return parser.parse_args()


def selected_benchmarks(name: str) -> list[str]:
    return sorted(DATASETS) if name == "all" else [name]


def local_dir_for(args: argparse.Namespace, name: str) -> Path:
    if args.local_dir:
        if args.benchmark == "all":
            raise ValueError("--local_dir can only be used with a single --benchmark.")
        return Path(args.local_dir)
    default = Path(DATASETS[name]["local_dir"])
    return default if default.is_absolute() else Path(args.data_root) / default


def main() -> None:
    args = parse_args()
    api = HfApi(token=args.token)
    summary = {}
    for name in selected_benchmarks(args.benchmark):
        spec = DATASETS[name]
        patterns = spec[args.scope]
        local_dir = local_dir_for(args, name)
        print(
            f"[download] benchmark={name} repo={spec['repo_id']} scope={args.scope} "
            f"local_dir={local_dir} patterns={patterns}",
            flush=True,
        )
        try:
            files = api.list_repo_files(spec["repo_id"], repo_type="dataset")
            matching = [path for path in files if _matches_any(path, patterns)]
            print(f"[download] benchmark={name} matching_files={len(matching)} total_repo_files={len(files)}", flush=True)
            if args.list_only:
                summary[name] = {"status": "listed", "matching_files": matching[:200], "matching_file_count": len(matching)}
                continue
            path = snapshot_download(
                repo_id=spec["repo_id"],
                repo_type="dataset",
                local_dir=str(local_dir),
                cache_dir=args.cache_dir,
                allow_patterns=patterns,
                max_workers=args.max_workers,
                token=args.token,
            )
            postprocess = postprocess_download(name=name, scope=args.scope, local_dir=local_dir)
            summary[name] = {
                "status": "downloaded",
                "path": path,
                "matching_file_count": len(matching),
                "postprocess": postprocess,
            }
        except Exception as exc:
            summary[name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "gated": bool(spec.get("gated", False)),
            }
            print(f"[download:error] benchmark={name} {type(exc).__name__}: {exc}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if any(item.get("status") == "failed" for item in summary.values()):
        sys.exit(1)


def _matches_any(path: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(path, pattern) for pattern in patterns)


def postprocess_download(*, name: str, scope: str, local_dir: Path) -> dict:
    if name == "mvbench" and scope == "full":
        return unzip_archives(local_dir / "video", "*.zip")
    if name == "videomme" and (local_dir / "subtitle.zip").exists():
        return unzip_archives(local_dir, "subtitle.zip")
    return {"status": "skipped"}


def unzip_archives(root: Path, pattern: str) -> dict:
    if not root.exists():
        return {"status": "skipped", "reason": f"missing root {root}"}
    archives = sorted(root.glob(pattern))
    results = []
    for archive in archives:
        dest = archive.parent / archive.stem
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[postprocess] unzip {archive} -> {dest}", flush=True)
        completed = subprocess.run(
            ["unzip", "-n", str(archive), "-d", str(dest)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        results.append(
            {
                "archive": str(archive),
                "dest": str(dest),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Failed to unzip {archive}: {completed.stderr[-2000:]}")
    return {"status": "done", "archive_count": len(archives), "archives": results}


if __name__ == "__main__":
    main()
