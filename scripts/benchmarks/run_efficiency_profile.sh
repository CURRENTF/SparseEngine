#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-case Nsight Systems diagnostic after the standard efficiency suite finds a regression.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SYSTEM="${1:-sengine-vanilla}"
MODEL_PATH="${2:?usage: run_efficiency_profile.sh SYSTEM MODEL_PATH GPUS}"
GPUS="${3:?usage: run_efficiency_profile.sh SYSTEM MODEL_PATH GPUS}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROMPT_LEN="${PROMPT_LEN:-16384}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
CONCURRENCY="${CONCURRENCY:-8}"
CHURN_REQUEST_MULTIPLIER="${CHURN_REQUEST_MULTIPLIER:-2}"
SPARSE_PREFILL_SCORE_MODE="${SPARSE_PREFILL_SCORE_MODE:-probability}"
NSYS_GPU_METRICS_FREQUENCY="${NSYS_GPU_METRICS_FREQUENCY:-1000}"
BASE_OUT="${SPARSEENGINE_OUTPUT_DIR:-outputs}/efficiency_profile_$(date +%Y%m%d_%H%M%S)"

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: Nsight Systems (nsys) is required for diagnostic profiling." >&2
  exit 2
fi

GPU_HELP=$(nsys profile --gpu-metrics-devices=help true 2>&1 || true)
if printf '%s\n' "${GPU_HELP}" | grep -q "Insufficient privilege"; then
  echo "ERROR: Nsight GPU hardware counters are unavailable due to insufficient privilege." >&2
  echo "Enable NVIDIA performance counters before running this diagnostic; no estimated fallback is used." >&2
  exit 2
fi

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

case "${SYSTEM}" in
  sengine-vanilla)
    ENGINE=sparseengine
    METHOD=vanilla
    ;;
  sengine-snapkv)
    ENGINE=sparseengine
    METHOD=snapkv
    ;;
  vllm-vanilla|vllm)
    ENGINE=vllm
    METHOD=vanilla
    ;;
  *)
    echo "ERROR: unsupported profiling system '${SYSTEM}'." >&2
    exit 2
    ;;
esac

mkdir -p "${BASE_OUT}/probe"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PROFILER_SENGINE=1

PROBE_ARGS=(
  --engine "${ENGINE}"
  --model-path "${MODEL_PATH}"
  --sparse-method "${METHOD}"
  --scenario churn
  --prompt-lens "${PROMPT_LEN}"
  --output-lens "${OUTPUT_LEN}"
  --batch-sizes "${CONCURRENCY}"
  --churn-request-multiplier "${CHURN_REQUEST_MULTIPLIER}"
  --num-warmups 1
  --num-iters 1
  --monitor-gpus "${GPUS}"
  --output-dir "${BASE_OUT}/probe"
)
if [ "${METHOD}" = snapkv ]; then
  PROBE_ARGS+=(--sparse-prefill-score-mode "${SPARSE_PREFILL_SCORE_MODE}")
fi

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --gpu-metrics-devices="${GPUS}" \
  --gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY}" \
  --force-overwrite=false \
  --output="${BASE_OUT}/timeline" \
  "${PYTHON_BIN}" -u "${REPO_ROOT}/benchmark/efficiency/bench_probe.py" "${PROBE_ARGS[@]}"

printf 'Nsight report: %s\n' "${BASE_OUT}/timeline.nsys-rep"
printf 'Probe artifacts: %s\n' "${BASE_OUT}/probe"
