#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Scenario B: LongBench Dynamic Sequence Switching & Hardware/CacheManager Profiler Runner
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SYSTEMS="${1:-sengine-vanilla,sengine-snapkv,vllm-vanilla}" # comma-separated
MODEL_NAME="${2:-qwen3_30b}"
GPUS="${3:-6,7}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-8}"
TASKS="${TASKS:-qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc}"
SPARSE_PREFILL_SCORE_MODE="${SPARSE_PREFILL_SCORE_MODE:-probability}"
case "${SPARSE_PREFILL_SCORE_MODE}" in
  probability|logits) ;;
  *)
    echo "ERROR: SPARSE_PREFILL_SCORE_MODE must be probability or logits." >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${SPARSEENGINE_LONGBENCH_DATA_DIR:-${SPARSEENGINE_DATA_DIR:-data/LongBench}}"
BASE_OUT="${SPARSEENGINE_OUTPUT_DIR:-outputs}/longbench_churn_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BASE_OUT}"

case "${MODEL_NAME}" in
  qwen3_30b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
    TP_SIZE=2
    ;;
  *)
    MODEL_PATH="${MODEL_NAME}"
    TP_SIZE=2
    ;;
esac

# Idle GPU Preflight Check
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
for GPU_ID in "${GPU_ARR[@]}"; do
  if ! ACTIVE_APPS=$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits); then
    echo "ERROR: failed to query GPU ${GPU_ID}." >&2
    exit 2
  fi
  ACTIVE_APPS=$(printf '%s\n' "${ACTIVE_APPS}" | sed '/^[[:space:]]*$/d')
  if [ -n "${ACTIVE_APPS}" ]; then
    echo "ERROR: GPU ${GPU_ID} is currently busy! Active PIDs: ${ACTIVE_APPS}" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export SPARSEENGINE_DATA_DIR="${DATA_ROOT}"
export SPARSEENGINE_LONGBENCH_DATA_DIR="${DATA_ROOT}"
export TOKENIZERS_PARALLELISM="false"
export PROFILER_SENGINE="1"

MON_PID=""
cleanup_monitor() {
  local rc=0
  if [ -n "${MON_PID}" ]; then
    if kill -0 "${MON_PID}" 2>/dev/null; then
      kill -INT "${MON_PID}" 2>/dev/null || true
    fi
    wait "${MON_PID}" 2>/dev/null || rc=$?
  fi
  MON_PID=""
  return "${rc}"
}
trap 'cleanup_monitor || true' EXIT

echo "================================================================================="
echo "Starting Scenario B: LongBench Dynamic Sequence Switching Benchmark"
echo "================================================================================="
echo "Model:            ${MODEL_PATH} (TP=${TP_SIZE})"
echo "GPUs:             ${GPUS}"
echo "Systems:          ${SYSTEMS}"
echo "Tasks:            ${TASKS}"
echo "Samples Per Task: ${SAMPLES_PER_TASK}"
echo "Output Directory: ${BASE_OUT}"
echo "================================================================================="

IFS=',' read -ra SYS_ARR <<< "${SYSTEMS}"
for SYS in "${SYS_ARR[@]}"; do
  SYS_TRIM=$(echo "${SYS}" | xargs)
  SYS_OUT="${BASE_OUT}/${SYS_TRIM}"
  mkdir -p "${SYS_OUT}"

  echo ""
  echo "---------------------------------------------------------------------------------"
  echo "Running System: [${SYS_TRIM}]"
  echo "---------------------------------------------------------------------------------"

  # 1. Start Hardware Monitor
  HW_LOG="${SYS_OUT}/gpu_timeline.json"
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmark/efficiency/hardware_monitor.py" \
    --gpus "${GPUS}" \
    --interval_ms 200 \
    --output_file "${HW_LOG}" &
  MON_PID=$!
  echo "[Monitor] GPU Lifecycle Logger started (PID: ${MON_PID})."

  # 2. Run Inference
  if [ "${SYS_TRIM}" = "vllm-vanilla" ] || [ "${SYS_TRIM}" = "vllm" ]; then
    "${PYTHON_BIN}" -u "${REPO_ROOT}/benchmark/long_bench/pred_vllm.py" \
      --model_path "${MODEL_PATH}" \
      --task "${TASKS}" \
      --output_root "${SYS_OUT}" \
      --tensor_parallel_size "${TP_SIZE}" \
      --gpu_memory_utilization 0.85 \
      --max_model_len 32768 \
      --num_samples "${SAMPLES_PER_TASK}" \
      --min_required_samples "${SAMPLES_PER_TASK}" \
      --temperature 0.0 \
      --top_p 1.0 \
      --top_k 1 \
      --batch_size 8
  else
    case "${SYS_TRIM}" in
      sengine-vanilla)
        SPARSE_METHOD="vanilla"
        HPARAMS="{\"tensor_parallel_size\": ${TP_SIZE}, \"gpu_memory_utilization\": 0.85, \"decode_graph\": true}"
        ;;
      sengine-snapkv)
        SPARSE_METHOD="snapkv"
        HPARAMS="{\"tensor_parallel_size\": ${TP_SIZE}, \"gpu_memory_utilization\": 0.85, \"snapkv_window_size\": 64, \"sparse_prefill_score_mode\": \"${SPARSE_PREFILL_SCORE_MODE}\", \"sink_keep_tokens\": 64, \"decode_keep_tokens\": 2048, \"recent_keep_tokens\": 64, \"pool_kernel_size\": 7, \"decode_graph\": true}"
        ;;
      *)
        SPARSE_METHOD="${SYS_TRIM}"
        HPARAMS="{\"tensor_parallel_size\": ${TP_SIZE}, \"gpu_memory_utilization\": 0.85, \"decode_graph\": true}"
        ;;
    esac

    "${PYTHON_BIN}" -u "${REPO_ROOT}/benchmark/long_bench/pred.py" \
      --model_path "${MODEL_PATH}" \
      --task "${TASKS}" \
      --output_root "${SYS_OUT}" \
      --sparse_method "${SPARSE_METHOD}" \
      --hyper_param "${HPARAMS}" \
      --max_model_len 32768 \
      --num_samples "${SAMPLES_PER_TASK}" \
      --temperature 0.0 \
      --top_p 1.0 \
      --top_k 1 \
      --batch_size 8

    # Run Eval for Sparse-Engine
    "${PYTHON_BIN}" "${REPO_ROOT}/benchmark/long_bench/eval.py" --path "${SYS_OUT}"
  fi

  # 3. Stop Hardware Monitor
  cleanup_monitor
  echo "[${SYS_TRIM}] Run completed."
done

"${PYTHON_BIN}" "${REPO_ROOT}/benchmark/efficiency/validate_unified_suite.py" \
  --root "${BASE_OUT}" --systems "${SYSTEMS}" --tasks "${TASKS}" \
  --expected-count "${SAMPLES_PER_TASK}" --longbench-only

echo ""
echo "================================================================================="
echo "Scenario B Complete. Artifacts saved in: ${BASE_OUT}"
echo "================================================================================="
