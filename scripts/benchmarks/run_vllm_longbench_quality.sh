#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# LongBench regression quality & full-lifecycle GPU monitoring script for upstream vLLM
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-full}" # smoke | full
MODEL_NAME="${2:-qwen3_30b}" # qwen3_30b | qwen3_8b | qwen25_7b
GPUS="${3:-0,1}"

# 1. Environment & Python Setup
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${SPARSEENGINE_LONGBENCH_DATA_DIR:-${SPARSEENGINE_DATA_DIR:-data/LongBench}}"
OUTPUT_ROOT="${SPARSEENGINE_OUTPUT_DIR:-outputs}/vllm_longbench_quality_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_ROOT}/vllm"

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
  SAMPLE_ARGS=(--num_samples "${SAMPLES_PER_TASK}")
  BATCH_SIZE=2
  echo "=== Running vLLM LongBench Smoke Test (${SAMPLES_PER_TASK} samples/task) ==="
else
  SAMPLES_PER_TASK=-1
  SAMPLE_ARGS=()
  BATCH_SIZE=16
  echo "=== Running vLLM LongBench Full Regression Test (Full 1500 samples) ==="
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
export VLLM_LOGGING_LEVEL="INFO"

echo "Using Model: ${MODEL_PATH} (TP=${TP_SIZE})"
echo "Using GPUs: ${GPUS}"
echo "Output Directory: ${OUTPUT_ROOT}"

# 6. Start Full-Lifecycle GPU Monitor in background
MONITOR_LOG="${OUTPUT_ROOT}/vllm/gpu_timeline.json"
"${PYTHON_BIN}" "${REPO_ROOT}/benchmark/efficiency/hardware_monitor.py" \
  --gpus "${GPUS}" \
  --interval_ms 200 \
  --output_file "${MONITOR_LOG}" &
MONITOR_PID=$!
cleanup_monitor() {
  local rc=0
  if [ -n "${MONITOR_PID:-}" ]; then
    if kill -0 "${MONITOR_PID}" 2>/dev/null; then
      kill -INT "${MONITOR_PID}" 2>/dev/null || true
    fi
    wait "${MONITOR_PID}" 2>/dev/null || rc=$?
  fi
  MONITOR_PID=""
  return "${rc}"
}
trap 'cleanup_monitor || true' EXIT
echo "[Monitor] GPU Lifecycle Logger started (PID: ${MONITOR_PID}). Sampling every 200ms..."

# 7. Run vLLM Benchmark on LongBench
"${PYTHON_BIN}" -u benchmark/long_bench/pred_vllm.py \
  --model_path "${MODEL_PATH}" \
  --task "${TASKS}" \
  --output_root "${OUTPUT_ROOT}/vllm" \
  --tensor_parallel_size "${TP_SIZE}" \
  --gpu_memory_utilization 0.85 \
  --max_model_len 32768 \
  "${SAMPLE_ARGS[@]}" \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 1 \
  --batch_size "${BATCH_SIZE}"

# 8. Stop GPU Monitor & Flush Stats
cleanup_monitor

echo "============================================================"
echo "vLLM LongBench Quality & GPU Lifecycle Run Completed!"
echo "============================================================"
