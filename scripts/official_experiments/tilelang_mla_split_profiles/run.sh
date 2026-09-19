#!/usr/bin/env bash
set -euo pipefail
: "${RUN_ROOT:?Set a fresh durable output directory}"
: "${CUDA_VISIBLE_DEVICES:?Select one idle GPU}"
: "${CONDA_EXE:?Set the conda executable}"
: "${SPARSE_ENGINE_ENV:?Set the conda environment}"
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_CACHE_DIR="$RUN_ROOT/cache/tilelang"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export CUDA_CACHE_PATH="$RUN_ROOT/cache/cuda"
export TMPDIR="${RUN_ROOT}.compiler-work"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
"$CONDA_EXE" run --no-capture-output -p "$SPARSE_ENGINE_ENV" python -u \
    "$REPO_ROOT/scripts/official_experiments/tilelang_mla_split_profiles/run.py" \
    --output "$RUN_ROOT" "$@"
