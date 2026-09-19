#!/usr/bin/env bash
# Run inside the project's activated CUDA environment (or conda run).
set -euo pipefail
: "${RUN_ROOT:?Set a fresh durable validation directory}"
: "${CUDA_VISIBLE_DEVICES:?Select one idle GPU}"
REPO_ROOT=$(git rev-parse --show-toplevel)
mkdir -p "$RUN_ROOT"
test ! -e "$RUN_ROOT/guard-ready.json"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$RUN_ROOT/compiler-work"
export TILELANG_CACHE_DIR="$RUN_ROOT/cache/tilelang"
export CUDA_CACHE_PATH="$RUN_ROOT/cache/cuda"
export OMP_NUM_THREADS=8
mkdir -p "$TMPDIR"
guard_pid=""
worker_pid=""
cleanup() {
    if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then kill "$worker_pid"; fi
    if [[ -n "$guard_pid" ]] && kill -0 "$guard_pid" 2>/dev/null; then kill "$guard_pid"; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
python "$REPO_ROOT/scripts/official_experiments/sparse_decode_efficiency/decode_capacity_guard.py" \
    --parent "$$" --ready "$RUN_ROOT/guard-ready.json" --max-seconds 3600 > "$RUN_ROOT/guard.log" 2>&1 &
guard_pid=$!
for ((attempt=0; attempt<90; attempt++)); do
    kill -0 "$guard_pid"
    if [[ -f "$RUN_ROOT/guard-ready.json" ]]; then break; fi
    sleep 1
done
test -f "$RUN_ROOT/guard-ready.json"
git rev-parse HEAD > "$RUN_ROOT/git-head.txt"
git diff -- src/sparseengine/kernels/tilelang/mla/runtime.py src/sparseengine/operators/mla_attention.py \
    tests/test_tilelang_mla_kernel.py tests/test_tilelang_mla_operator.py > "$RUN_ROOT/change.patch"
printf 'gpu-tests\trunning\n' >> "$RUN_ROOT/status.tsv"
python -m pytest -q tests/test_tilelang_mla_kernel.py --junitxml="$RUN_ROOT/gpu-tests.xml" \
    > "$RUN_ROOT/gpu-tests.log" 2>&1 &
worker_pid=$!
if wait "$worker_pid"; then
    kill -0 "$guard_pid"
    printf 'gpu-tests\tcompleted\n' >> "$RUN_ROOT/status.tsv"
else
    printf 'gpu-tests\tfailed\n' >> "$RUN_ROOT/status.tsv"
    exit 1
fi
