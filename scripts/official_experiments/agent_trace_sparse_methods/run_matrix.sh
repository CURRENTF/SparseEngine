#!/usr/bin/env bash
set -euo pipefail

output_root=${1:?persistent output root}
trace_dir=${2:?pinned trace directory}
gpus=${3:?two idle GPU indices}
port=${4:?unused local port}
model_path=${5:?model checkpoint directory}
forced_workload=${6:?prepared forced workload}
model_label=${AGENT_TRACE_MODEL_LABEL:-glm47}
snapkv_decode_eviction=${AGENT_TRACE_SNAPKV_DECODE_EVICTION_ON_SNAPKV:-0}
if [[ ! "$model_label" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "Invalid AGENT_TRACE_MODEL_LABEL: $model_label" >&2
  exit 2
fi
if [[ "$snapkv_decode_eviction" != 0 && "$snapkv_decode_eviction" != 1 ]]; then
  echo "AGENT_TRACE_SNAPKV_DECODE_EVICTION_ON_SNAPKV must be 0 or 1" >&2
  exit 2
fi
case_specs=("${@:7}")
if (( ${#case_specs[@]} == 0 )); then
  case_specs=(vanilla-prefix:40 omnikv-prefix:40 quest-prefix:40
              vanilla-prefix:100 snapkv-chain:100 h2o-chain:100)
fi
script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$script_dir/../../.."
mkdir -p "$output_root"
if [[ -e "$output_root/matrix_status.tsv" ]]; then
  echo "Matrix already started; refusing to overwrite $output_root" >&2
  exit 1
fi

record() {
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" >>"$output_root/matrix_status.tsv"
}

active_case_pid=
stop_active_case() {
  if [[ -n "$active_case_pid" ]]; then
    kill -TERM "$active_case_pid" 2>/dev/null || true
    wait "$active_case_pid" || true
    active_case_pid=
  fi
}
trap 'stop_active_case; exit 143' INT TERM
trap stop_active_case EXIT

failed=0
for case_spec in "${case_specs[@]}"; do
  method=${case_spec%:*}
  concurrency=${case_spec#*:}
  idle=0
  for ((attempt=0; attempt<300; attempt++)); do
    if python3 - "$gpus" <<'PY' 2>/dev/null
import sys
from scripts.official_experiments.chain_cache_miniswe.run import idle_pair
idle_pair(sys.argv[1])
PY
    then
      idle=1
      break
    fi
    sleep 2
  done
  if (( ! idle )); then
    record "$model_label" "$case_spec" gpu_conflict
    exit 2
  fi
  record "$model_label" "$case_spec" running
  case_snapkv_eviction=0
  if [[ "$method" == snapkv-chain ]]; then
    case_snapkv_eviction=$snapkv_decode_eviction
  fi
  AGENT_TRACE_SNAPKV_DECODE_EVICTION="$case_snapkv_eviction" \
    bash "$script_dir/run_case.sh" "$model_label" "$model_path" "$method" "$concurrency" \
      "$output_root" "$trace_dir" "$gpus" "$port" full "$forced_workload" &
  active_case_pid=$!
  if wait "$active_case_pid"; then
    record "$model_label" "$case_spec" completed
  else
    record "$model_label" "$case_spec" failed
    failed=1
  fi
  active_case_pid=
  if [[ ${AGENT_TRACE_ARCHIVE_REQUEST_LOGS:-0} == 1 ]]; then
    if ! bash "$script_dir/archive_case_requests.sh" \
        "$output_root/${model_label}_${method}_c${concurrency}" "$method"; then
      record "$model_label" "$case_spec" archive_failed
      exit 1
    fi
    record "$model_label" "$case_spec" request_logs_archived
  fi
done
exit "$failed"
