#!/usr/bin/env python3
"""Compatibility wrapper for the SparseEngine microbenchmark.

The benchmark implementation lives under `benchmark/` with the rest of the
repo benchmark entrypoints. Keep this script so existing runbooks and harnesses
that call `scripts/benchmarks/bench_sparse_engine.py` continue to work.
"""

from pathlib import Path
import multiprocessing as mp
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
src_path = str(REPO_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from benchmark.microbench import main


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
