#!/usr/bin/env bash
# Run from an activated SparseEngine environment. All measured work uses the canonical probe.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-4B-Instruct-2507 checkpoint}"
: "${RUN_ROOT:?Set RUN_ROOT to a new persistent output directory}"
GPU="${GPU:-0}"
METHOD="${1:?Usage: run.sh snapkv|kvzip baseline|profile 1|4}"
MODE="${2:?Set baseline or profile}"
BS="${3:?Set concurrency}"
case "$METHOD" in snapkv|kvzip) ;; *) exit 2 ;; esac
case "$MODE" in baseline|profile) ;; *) exit 2 ;; esac
case "$BS" in 1|4) ;; *) exit 2 ;; esac
ACTIVE=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$ACTIVE" ]]; then printf 'GPU %s is busy: %s\n' "$GPU" "$ACTIVE" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false RUN_ROOT
SCORE_MODE="${SCORE_MODE:-probability}"
case "$SCORE_MODE" in probability|logits) ;; *) exit 2 ;; esac
LABEL="$METHOD"
if [[ "$METHOD" == snapkv && "$SCORE_MODE" == logits ]]; then LABEL=snapkv-logits; fi
OUT="$RUN_ROOT/$LABEL-$MODE-c$BS"
if [[ -e "$OUT" ]]; then echo "Refusing to overwrite $OUT" >&2; exit 2; fi
mkdir -p "$OUT" "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"
HP='{"max_model_len":40832,"kv_cache_layout":"token","sink_keep_tokens":64,"recent_keep_tokens":64,"decode_keep_tokens":3968,"kvzip_token_budget":4096,"kvzip_score_chunk_size":2000,"engine_prefill_chunk_size":8192,"decode_graph":true,"decode_graph_capture_sizes":[1,2,4]}'
ITERS=3
if [[ "$MODE" == profile ]]; then ITERS=1; fi
ARGS=(--engine sparseengine --model-path "$MODEL_PATH" --sparse-method "$METHOD"
  --tensor-parallel-size 1 --scenario fixed --prompt-lens 32768 --output-lens 128
  --batch-sizes "$BS" --prompt-length-jitter 0 --output-length-jitter 0
  --question-suffix-tokens 128 --context-prefix-length 64 --num-warmups 1
  --num-iters "$ITERS" --gpu-memory-utilization 0.80 --monitor-gpus "$GPU"
  --hyper-params "$HP" --output-dir "$OUT")
if [[ "$METHOD" == snapkv ]]; then ARGS+=(--sparse-prefill-score-mode "$SCORE_MODE"); fi
CMD=(python -u benchmark/efficiency/bench_probe.py)
if [[ "$MODE" == profile ]]; then
  export SPARSEENGINE_NVTX=1
  CMD=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
    --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop
    --force-overwrite=false --output="$OUT/timeline"
    python -u scripts/official_experiments/snapkv_kvzip_nsight/capture.py)
fi
printf '%q ' "${CMD[@]}" "${ARGS[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
trap 'code=$?; printf "%s\t%s\t%s\n" "$LABEL-$MODE-c$BS" "$code" "$(date -Is)" >> "$RUN_ROOT/status.tsv"' EXIT
timeout 1200 "${CMD[@]}" "${ARGS[@]}" > "$OUT/stdout.log" 2>&1
python - "$OUT" "$ITERS" "$BS" <<'CHECK'
import json
from pathlib import Path
import sys
out, iterations, batch = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(x) for x in (out / "raw_samples.jsonl").read_text().splitlines()]
if len(rows) != iterations or any(r["status"] != "success" for r in rows):
    raise RuntimeError("Missing or failed measured workload")
for row in rows:
    requests = row["request_results"]
    if len(requests) != batch or any(r["status"] != "success" or r["generated_tokens"] != 128 for r in requests):
        raise RuntimeError("Incomplete request output")
CHECK
if [[ "$MODE" == profile ]]; then
  nsys stats --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_api_sum \
    --format csv --output "$OUT/stats" "$OUT/timeline.nsys-rep" > "$OUT/export.log" 2>&1
fi
