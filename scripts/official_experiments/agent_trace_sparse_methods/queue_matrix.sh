#!/usr/bin/env bash
set -euo pipefail

output_root=${1:?persistent output root}
trace_dir=${2:?pinned trace directory}
port=${3:?unused local port}
glm_model=${4:?GLM checkpoint directory}
glm_forced=${5:?prepared GLM forced workload}
script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$script_dir/../../.."

for path in "$trace_dir/manifest.json" "$glm_model/config.json" "$glm_forced"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done
mkdir -p "$output_root"
if [[ -e "$output_root/queue_status.tsv" || -e "$output_root/matrix_status.tsv" ]]; then
  echo "Queue or matrix already started: $output_root" >&2
  exit 1
fi
exec >>"$output_root/queue.log" 2>&1

record() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >>"$output_root/queue_status.tsv"
}

record waiting_for_two_idle_gpus
pair=
for ((attempt=0; attempt<720; attempt++)); do
  if pair=$(python3 - <<'PY'
import itertools
import subprocess
import sys
from scripts.official_experiments.chain_cache_miniswe.run import idle_pair

raw = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits"], text=True, timeout=30)
idle = []
for line in raw.splitlines():
    index, memory, activity = [item.strip() for item in line.split(",")]
    if int(memory) <= 128 and int(activity) == 0:
        idle.append(index)
for first, second in itertools.combinations(idle, 2):
    pair = f"{first},{second}"
    try:
        idle_pair(pair)
    except RuntimeError:
        continue
    print(pair)
    sys.exit(0)
sys.exit(42)
PY
  ); then
    break
  else
    probe_status=$?
    if (( probe_status != 42 )); then
      record gpu_probe_failed
      exit "$probe_status"
    fi
  fi
  pair=
  sleep 30
done
if [[ -z "$pair" ]]; then
  record gpu_wait_timed_out_after_six_hours
  exit 2
fi
printf '%s\n' "$pair" >"$output_root/gpu_pair.txt"
record "selected_gpus=$pair"

record matrix_running
bash "$script_dir/run_matrix.sh" "$output_root" "$trace_dir" "$pair" "$port" \
  "$glm_model" "$glm_forced" "${@:6}"
record matrix_finished
