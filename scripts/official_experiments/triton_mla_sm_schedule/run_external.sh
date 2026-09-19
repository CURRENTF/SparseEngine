#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: run_external.sh FROZEN_REPO LANE GPU' >&2
  exit 2
fi
export EXTERNAL_OUTPUT_ROOT EXTERNAL_SCRATCH_ROOT CONDA_EXE SPARSE_ENGINE_ENV VLLM_ENV
export QWEN3_MODEL TANGRAM_ENV HISPARSE_ENV
python3 "$1/scripts/official_experiments/sparse_decode_efficiency/sweep_decode_capacity.py" \
  --repo "$1" --config "$1/scripts/official_experiments/triton_mla_sm_schedule/external_capacity.json" \
  --model qwen3-30b-fp8 --lanes "$2" --gpus "$3" --attempt initial
