#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# LongBench regression quality evaluation script for H2O Sparse Attention on SparseEngine
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-smoke}" # smoke | full
MODEL_NAME="${2:-qwen3_30b}" # qwen3_30b | qwen3_8b | qwen25_7b
GPUS="${3:-2,3}"
SPARSE_PREFILL_SCORE_MODE="${SPARSE_PREFILL_SCORE_MODE:-logits}"
case "${SPARSE_PREFILL_SCORE_MODE}" in
  probability|logits) ;;
  *)
    echo "ERROR: SPARSE_PREFILL_SCORE_MODE must be probability or logits." >&2
    exit 2
    ;;
esac
H2O_PREFILL_SCORE_WINDOW="${H2O_PREFILL_SCORE_WINDOW:-128}"

# 1. Environment & Python Setup
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${SPARSEENGINE_LONGBENCH_DATA_DIR:-${SPARSEENGINE_DATA_DIR:-data/LongBench}}"
OUTPUT_ROOT="${SPARSEENGINE_OUTPUT_DIR:-outputs}/h2o_longbench_quality_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_ROOT}"

# 2. Model & TP Configuration
case "${MODEL_NAME}" in
  qwen3_30b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
    TP_SIZE=2
    ;;
  qwen3_8b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
    TP_SIZE=1
    ;;
  qwen25_7b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct-1M}"
    TP_SIZE=1
    ;;
  *)
    MODEL_PATH="${MODEL_NAME}"
    TP_SIZE=1
    ;;
esac

# 3. Tasks & Sample Sizing
TASKS="qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc"
if [ "${MODE}" = "smoke" ]; then
  SAMPLES_PER_TASK=2
  BATCH_SIZE=2
  echo "=== Running H2O LongBench Smoke Test (${SAMPLES_PER_TASK} samples/task) ==="
else
  SAMPLES_PER_TASK=50
  BATCH_SIZE=16
  echo "=== Running H2O LongBench Full Regression Test (${SAMPLES_PER_TASK} samples/task) ==="
fi

# 4. Idle GPU Preflight Check
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
for GPU_ID in "${GPU_ARR[@]}"; do
  ACTIVE_APPS=$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  if [ -n "${ACTIVE_APPS}" ]; then
    echo "ERROR: GPU ${GPU_ID} is not idle! Active PIDs: ${ACTIVE_APPS}" >&2
    exit 2
  fi
done

# 5. Export Environment Variables
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TMPDIR="${TMPDIR:-/tmp}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export SPARSEENGINE_DATA_DIR="${DATA_ROOT}"
export SPARSEENGINE_LONGBENCH_DATA_DIR="${DATA_ROOT}"
export TOKENIZERS_PARALLELISM="false"
export LOG_LEVEL="INFO"

echo "Using Model: ${MODEL_PATH} (TP=${TP_SIZE})"
echo "Using GPUs: ${GPUS}"
echo "H2O prefill score mode/window: ${SPARSE_PREFILL_SCORE_MODE}/${H2O_PREFILL_SCORE_WINDOW}"
echo "Output Directory: ${OUTPUT_ROOT}"

# 6. Run Quality Benchmark for Vanilla & H2O
for METHOD in vanilla h2o; do
  METHOD_OUT="${OUTPUT_ROOT}/${METHOD}"
  mkdir -p "${METHOD_OUT}"

  echo "------------------------------------------------------------"
  echo "Starting LongBench Quality Run for [${METHOD}]..."
  echo "------------------------------------------------------------"

  if [ "${METHOD}" = "h2o" ]; then
    HPARAMS="{\"tensor_parallel_size\": ${TP_SIZE}, \"gpu_memory_utilization\": 0.85, \"h2o_decode_budget\": 2048, \"h2o_decode_eviction_interval\": 128, \"h2o_prefill_score_window\": ${H2O_PREFILL_SCORE_WINDOW}, \"sparse_prefill_score_mode\": \"${SPARSE_PREFILL_SCORE_MODE}\", \"h2o_recent_ratio\": 0.2}"
  else
    HPARAMS="{\"tensor_parallel_size\": ${TP_SIZE}, \"gpu_memory_utilization\": 0.85}"
  fi

  "${PYTHON_BIN}" -u benchmark/long_bench/pred.py \
    --model_path "${MODEL_PATH}" \
    --task "${TASKS}" \
    --output_root "${METHOD_OUT}" \
    --sparse_method "${METHOD}" \
    --hyper_param "${HPARAMS}" \
    --max_model_len 32768 \
    --num_samples "${SAMPLES_PER_TASK}" \
    --temperature 0.0 \
    --top_p 1.0 \
    --top_k 1 \
    --batch_size "${BATCH_SIZE}"

  echo "[${METHOD}] Completed. Results written to ${METHOD_OUT}"
done

# 7. Summary & Comparison Report
echo "============================================================"
echo "LongBench Quality Evaluation Complete!"
echo "============================================================"
"${PYTHON_BIN}" - <<PY
import json
import os

root = "${OUTPUT_ROOT}"
print(f"\nSummary Report for: {root}\n")
print(f"{'Task':<25} | {'Vanilla Score':<15} | {'H2O Score':<15} | {'Quality Ratio':<15}")
print("-" * 75)

vanilla_res = {}
h2o_res = {}

v_file = os.path.join(root, "vanilla", "result.json")
h_file = os.path.join(root, "h2o", "result.json")

if os.path.exists(v_file):
    with open(v_file) as f:
        vanilla_res = json.load(f)
if os.path.exists(h_file):
    with open(h_file) as f:
        h2o_res = json.load(f)

tasks = ["qasper", "hotpotqa", "multi_news", "trec", "passage_retrieval_en", "lcc"]
for task in tasks:
    v_s = float(vanilla_res.get(task, 0.0) or 0.0)
    h_s = float(h2o_res.get(task, 0.0) or 0.0)
    ratio = (h_s / v_s * 100) if v_s > 0 else 0.0
    print(f"{task:<25} | {v_s:<15.2f} | {h_s:<15.2f} | {ratio:<14.1f}%")

v_avg = float(vanilla_res.get("overall_category_avg", 0.0) or 0.0)
h_avg = float(h2o_res.get("overall_category_avg", 0.0) or 0.0)
ratio_avg = (h_avg / v_avg * 100) if v_avg > 0 else 0.0
print("-" * 75)
print(f"{'OVERALL AVERAGE':<25} | {v_avg:<15.2f} | {h_avg:<15.2f} | {ratio_avg:<14.1f}%")
print("=" * 75 + "\n")
PY
