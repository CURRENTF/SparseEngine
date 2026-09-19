#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Unified synthetic-efficiency and matched LongBench lifecycle benchmark suite.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${1:-6,7}"
SYSTEMS="${2:-sengine-vanilla,sengine-snapkv,vllm-vanilla}"
MODEL_NAME="${3:-qwen3_30b}"
PROMPT_LENS="${PROMPT_LENS:-8192,16384,32768}"
OUTPUT_LENS="${OUTPUT_LENS:-512}"
BATCH_SIZES="${BATCH_SIZES:-1,4,8}"
LONGBENCH_SAMPLES="${LONGBENCH_SAMPLES:-10}"
BENCH_SCENARIO="${BENCH_SCENARIO:-all}"
BENCH_SEED="${BENCH_SEED:-42}"
PROMPT_LENGTH_JITTER="${PROMPT_LENGTH_JITTER:-0.10}"
OUTPUT_LENGTH_JITTER="${OUTPUT_LENGTH_JITTER:-0.25}"
CHURN_REQUEST_MULTIPLIER="${CHURN_REQUEST_MULTIPLIER:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
SPARSE_PREFILL_SCORE_MODE="${SPARSE_PREFILL_SCORE_MODE:-probability}"
DELTAKV_COMPRESSOR_PATH="${DELTAKV_COMPRESSOR_PATH:-}"
OMNIKV_FULL_ATTENTION_LAYERS="${OMNIKV_FULL_ATTENTION_LAYERS:-}"
MANIFEST_MODEL_ID="${BENCH_MANIFEST_MODEL_ID:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
DATA_ROOT="${SPARSEENGINE_LONGBENCH_DATA_DIR:-${SPARSEENGINE_DATA_DIR:-data/LongBench}}"
BASE_OUT="${SPARSEENGINE_OUTPUT_DIR:-outputs}/unified_efficiency_$(date +%Y%m%d_%H%M%S)"
MANIFEST_PATH="${SPARSEENGINE_REGRESSION_MANIFEST:-${REPO_ROOT}/benchmark/sparseengine_regression/manifest.json}"
TASKS="qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

case "${SPARSE_PREFILL_SCORE_MODE}" in
  probability|logits) ;;
  *)
    echo "ERROR: SPARSE_PREFILL_SCORE_MODE must be probability or logits." >&2
    exit 2
    ;;
esac

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
    if [ -z "${MANIFEST_MODEL_ID}" ]; then
      MANIFEST_MODEL_ID=qwen25_7b
    fi
    ;;
  *)
    MODEL_PATH="${MODEL_NAME}"
    TP_SIZE="${TP_SIZE:-1}"
    ;;
esac

resolve_system() {
  local system="$1"
  SYSTEM_ENGINE=""
  SYSTEM_METHOD=""
  SYSTEM_LONG_BENCH_ARGS=()
  case "${system}" in
    sengine-vanilla)
      SYSTEM_ENGINE=sparseengine
      SYSTEM_METHOD=vanilla
      ;;
    sengine-snapkv)
      SYSTEM_ENGINE=sparseengine
      SYSTEM_METHOD=snapkv
      ;;
    sengine-h2o)
      SYSTEM_ENGINE=sparseengine
      SYSTEM_METHOD=h2o
      ;;
    sengine-omnikv)
      SYSTEM_ENGINE=sparseengine
      SYSTEM_METHOD=omnikv
      ;;
    sengine-deltakv)
      SYSTEM_ENGINE=sparseengine
      SYSTEM_METHOD=deltakv
      if [ -z "${DELTAKV_COMPRESSOR_PATH}" ]; then
        echo "ERROR: sengine-deltakv requires DELTAKV_COMPRESSOR_PATH." >&2
        return 2
      fi
      if [ ! -e "${DELTAKV_COMPRESSOR_PATH}" ]; then
        echo "ERROR: DELTAKV_COMPRESSOR_PATH does not exist: ${DELTAKV_COMPRESSOR_PATH}" >&2
        return 2
      fi
      SYSTEM_LONG_BENCH_ARGS=(--deltakv_checkpoint_path "${DELTAKV_COMPRESSOR_PATH}")
      ;;
    vllm-vanilla|vllm)
      SYSTEM_ENGINE=vllm
      SYSTEM_METHOD=vanilla
      ;;
    *)
      echo "ERROR: unknown benchmark system '${system}'." >&2
      return 2
      ;;
  esac

  local overrides_json
  overrides_json=$("${PYTHON_BIN}" -c '
import json, sys
tp_size, method, score_mode, compressor, omnikv_layers = sys.argv[1:]
params = {"tensor_parallel_size": int(tp_size), "gpu_memory_utilization": 0.85}
if method in {"snapkv", "h2o"}:
    params["sparse_prefill_score_mode"] = score_mode
if method == "deltakv":
    params["deltakv_checkpoint_path"] = compressor
if method == "omnikv" and omnikv_layers:
    params["full_attention_layers"] = omnikv_layers
print(json.dumps(params, separators=(",", ":")))
' "${TP_SIZE}" "${SYSTEM_METHOD}" "${SPARSE_PREFILL_SCORE_MODE}" "${DELTAKV_COMPRESSOR_PATH}" "${OMNIKV_FULL_ATTENTION_LAYERS}")

  local -a config_args
  config_args=(
    -m benchmark.sparseengine_regression.manifest
    --manifest "${MANIFEST_PATH}"
    --method "${SYSTEM_METHOD}"
    --overrides-json "${overrides_json}"
  )
  if [ -n "${MANIFEST_MODEL_ID}" ]; then
    config_args+=(--model-id "${MANIFEST_MODEL_ID}")
  fi
  if [ "${SYSTEM_METHOD}" = omnikv ] && [ -z "${OMNIKV_FULL_ATTENTION_LAYERS}" ]; then
    config_args+=(--require-model-config)
  fi
  SYSTEM_HPARAMS=$("${PYTHON_BIN}" "${config_args[@]}")
}

IFS=',' read -ra SYS_ARR <<< "${SYSTEMS}"
if [ "${#SYS_ARR[@]}" -eq 0 ]; then
  echo "ERROR: SYSTEMS must contain at least one system." >&2
  exit 2
fi
for SYS in "${SYS_ARR[@]}"; do
  SYS_TRIM=$(printf '%s' "${SYS}" | xargs)
  if [ -z "${SYS_TRIM}" ]; then
    echo "ERROR: SYSTEMS contains an empty system name." >&2
    exit 2
  fi
  resolve_system "${SYS_TRIM}" || exit $?
done

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
for GPU_ID in "${GPU_ARR[@]}"; do
  if ! ACTIVE_APPS=$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits); then
    echo "ERROR: failed to query GPU ${GPU_ID}." >&2
    exit 2
  fi
  ACTIVE_APPS=$(printf '%s\n' "${ACTIVE_APPS}" | sed '/^[[:space:]]*$/d')
  if [ -n "${ACTIVE_APPS}" ]; then
    echo "ERROR: GPU ${GPU_ID} is busy; active PIDs: ${ACTIVE_APPS}" >&2
    exit 2
  fi
done

mkdir -p "${BASE_OUT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export SPARSEENGINE_DATA_DIR="${DATA_ROOT}"
export SPARSEENGINE_LONGBENCH_DATA_DIR="${DATA_ROOT}"

MON_PID=""
stop_monitor() {
  local rc=0
  if [ -n "${MON_PID}" ]; then
    if kill -0 "${MON_PID}" 2>/dev/null; then
      kill -INT "${MON_PID}" 2>/dev/null || true
    fi
    wait "${MON_PID}" || rc=$?
  fi
  MON_PID=""
  return "${rc}"
}
cleanup_monitor() {
  stop_monitor >/dev/null 2>&1 || true
}
trap cleanup_monitor EXIT

record_stage_status() {
  local output_dir="$1"
  local stage="$2"
  local system="$3"
  local task_rc="$4"
  local monitor_rc="$5"
  "${PYTHON_BIN}" -c \
    'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); task=int(sys.argv[4]); mon=int(sys.argv[5]); p.write_text(json.dumps({"stage":sys.argv[2],"system":sys.argv[3],"status":"success" if task == 0 and mon == 0 else "failed","task_exit_code":task,"monitor_exit_code":mon}, indent=2)+"\n")' \
    "${output_dir}/stage_status.json" "${stage}" "${system}" "${task_rc}" "${monitor_rc}"
}

SCENARIO_A_DIR="${BASE_OUT}/scenario_a_synthetic"
SCENARIO_B_DIR="${BASE_OUT}/scenario_b_longbench"
mkdir -p "${SCENARIO_A_DIR}" "${SCENARIO_B_DIR}"
OVERALL_FAILED=0

for SYS in "${SYS_ARR[@]}"; do
  SYS_TRIM=$(printf '%s' "${SYS}" | xargs)
  SYS_OUT="${SCENARIO_A_DIR}/${SYS_TRIM}"
  mkdir -p "${SYS_OUT}"
  resolve_system "${SYS_TRIM}" || exit $?

  PROBE_ARGS=(
    --engine "${SYSTEM_ENGINE}" --model-path "${MODEL_PATH}" --sparse-method "${SYSTEM_METHOD}"
    --prompt-lens "${PROMPT_LENS}" --output-lens "${OUTPUT_LENS}" --batch-sizes "${BATCH_SIZES}"
    --scenario "${BENCH_SCENARIO}" --seed "${BENCH_SEED}"
    --prompt-length-jitter "${PROMPT_LENGTH_JITTER}"
    --output-length-jitter "${OUTPUT_LENGTH_JITTER}"
    --churn-request-multiplier "${CHURN_REQUEST_MULTIPLIER}"
    --tensor-parallel-size "${TP_SIZE}" --gpu-memory-utilization 0.85
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --num-warmups 1 --num-iters 3 --output-dir "${SYS_OUT}"
    --monitor-gpus "${GPUS}"
    --hyper-params "${SYSTEM_HPARAMS}"
  )
  if [ "${SYSTEM_METHOD}" = snapkv ] || [ "${SYSTEM_METHOD}" = h2o ]; then
    PROBE_ARGS+=(--sparse-prefill-score-mode "${SPARSE_PREFILL_SCORE_MODE}")
  fi
  TASK_RC=0
  "${PYTHON_BIN}" -u "${REPO_ROOT}/benchmark/efficiency/bench_probe.py" "${PROBE_ARGS[@]}" || TASK_RC=$?
  MONITOR_RC=0  # Synthetic hardware metrics are collected inside each measured case.
  record_stage_status "${SYS_OUT}" scenario_a "${SYS_TRIM}" "${TASK_RC}" "${MONITOR_RC}"
  if [ "${TASK_RC}" -ne 0 ] || [ "${MONITOR_RC}" -ne 0 ]; then
    OVERALL_FAILED=1
  fi
done

for SYS in "${SYS_ARR[@]}"; do
  SYS_TRIM=$(printf '%s' "${SYS}" | xargs)
  SYS_OUT="${SCENARIO_B_DIR}/${SYS_TRIM}"
  mkdir -p "${SYS_OUT}"
  resolve_system "${SYS_TRIM}" || exit $?
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmark/efficiency/hardware_monitor.py" \
    --gpus "${GPUS}" --interval_ms 200 --output_file "${SYS_OUT}/gpu_timeline.json" &
  MON_PID=$!

  TASK_RC=0
  if [ "${SYSTEM_ENGINE}" = vllm ]; then
    "${PYTHON_BIN}" -u "${REPO_ROOT}/benchmark/long_bench/pred_vllm.py" \
      --model_path "${MODEL_PATH}" --task "${TASKS}" --output_root "${SYS_OUT}" \
      --tensor_parallel_size "${TP_SIZE}" --gpu_memory_utilization 0.85 \
      --max_model_len 32768 --num_samples "${LONGBENCH_SAMPLES}" \
      --min_required_samples "${LONGBENCH_SAMPLES}" --temperature 0.0 \
      --top_p 1.0 --top_k 1 --batch_size 16 --seed "${BENCH_SEED}" || TASK_RC=$?
  else
    "${PYTHON_BIN}" -u "${REPO_ROOT}/benchmark/long_bench/pred.py" \
      --model_path "${MODEL_PATH}" --task "${TASKS}" --output_root "${SYS_OUT}" \
      --sparse_method "${SYSTEM_METHOD}" --hyper_param "${SYSTEM_HPARAMS}" --max_model_len 32768 \
      --num_samples "${LONGBENCH_SAMPLES}" --temperature 0.0 --top_p 1.0 \
      --top_k 1 --batch_size 16 --seed "${BENCH_SEED}" \
      "${SYSTEM_LONG_BENCH_ARGS[@]}" || TASK_RC=$?
  fi
  MONITOR_RC=0
  stop_monitor || MONITOR_RC=$?
  record_stage_status "${SYS_OUT}" scenario_b "${SYS_TRIM}" "${TASK_RC}" "${MONITOR_RC}"
  if [ "${TASK_RC}" -ne 0 ] || [ "${MONITOR_RC}" -ne 0 ]; then
    OVERALL_FAILED=1
  fi
done

AGGREGATE_RC=0
"${PYTHON_BIN}" "${REPO_ROOT}/benchmark/efficiency/validate_unified_suite.py" \
  --root "${BASE_OUT}" --systems "${SYSTEMS}" --tasks "${TASKS}" \
  --expected-count "${LONGBENCH_SAMPLES}" || AGGREGATE_RC=$?

if [ "${AGGREGATE_RC}" -ne 0 ]; then
  OVERALL_FAILED=1
fi
echo "Unified suite artifacts: ${BASE_OUT}"
exit "${OVERALL_FAILED}"
