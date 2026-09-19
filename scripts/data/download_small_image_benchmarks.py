#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / "data")
DEFAULT_HF_CACHE = os.getenv("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

DATASETS = {
    "scienceqa_img": {
        "repo_id": "lmms-lab/ScienceQA-IMG",
        "local_dir": "ScienceQA-IMG_hf",
        "patterns": ["README.md", "data/validation-*.parquet", "data/test-*.parquet"],
    },
    "pope": {
        "repo_id": "lmms-lab/POPE",
        "local_dir": "POPE_hf",
        "patterns": ["README.md", "data/test-*.parquet"],
    },
    "mmbench_en": {
        "repo_id": "lmms-lab/MMBench_EN",
        "local_dir": "MMBench_EN_hf",
        "patterns": ["README.md", "data/dev-*.parquet"],
    },
    "mme": {
        "repo_id": "lmms-lab/MME",
        "local_dir": "MME_hf",
        "patterns": ["README.md", "data/test-*.parquet"],
    },
    "mmmu": {
        "repo_id": "lmms-lab/MMMU",
        "local_dir": "MMMU_hf",
        "patterns": ["README.md", "data/dev-*.parquet", "data/validation-*.parquet"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download small image QA benchmark parquet datasets.")
    parser.add_argument("--benchmark", default="all", choices=["all", *sorted(DATASETS)])
    parser.add_argument("--data_root", default=os.getenv("SPARSEENGINE_DATA_DIR", DEFAULT_DATA_ROOT))
    parser.add_argument("--cache_dir", default=DEFAULT_HF_CACHE)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--local_dir", default="", help="Override local dir for a single benchmark.")
    parser.add_argument("--token", default=None)
    parser.add_argument("--list_only", action="store_true")
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


def matches(path: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(path, pattern) for pattern in patterns)


def main() -> None:
    args = parse_args()
    api = HfApi(token=args.token)
    summary = {}
    for name in selected_benchmarks(args.benchmark):
        spec = DATASETS[name]
        local_dir = local_dir_for(args, name)
        patterns = spec["patterns"]
        print(
            f"[download] benchmark={name} repo={spec['repo_id']} local_dir={local_dir} patterns={patterns}",
            flush=True,
        )
        try:
            repo_files = api.list_repo_files(spec["repo_id"], repo_type="dataset")
            matching = [path for path in repo_files if matches(path, patterns)]
            print(f"[download] benchmark={name} matching_files={len(matching)} total_repo_files={len(repo_files)}")
            if args.list_only:
                summary[name] = {"status": "listed", "matching_file_count": len(matching), "matching_files": matching}
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
            summary[name] = {"status": "downloaded", "path": path, "matching_file_count": len(matching)}
        except Exception as exc:
            summary[name] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            print(f"[download:error] benchmark={name} {type(exc).__name__}: {exc}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if any(item.get("status") == "failed" for item in summary.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
