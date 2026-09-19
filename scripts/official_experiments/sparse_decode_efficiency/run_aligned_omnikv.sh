#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 BENCHMARK_REPO RESOLVED_CONFIG OUTPUT_ROOT" >&2
    exit 2
fi
benchmark_repo=$(realpath "$1")
config=$(realpath "$2")
run_root=$(realpath "$3")
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test ! -e "$run_root/run.log"
exec > >(tee "$run_root/run.log") 2>&1
trap 'code=$?; printf "%s\tqueue\texit\t%s\n" "$(date -Is)" "$code" >> "$run_root/status.tsv"' EXIT
hostname
date -Is
git -C "$benchmark_repo" rev-parse HEAD
git -C "$benchmark_repo" status --short
printf '%s\tqueue\tstarted\n' "$(date -Is)" >> "$run_root/status.tsv"

# Acquire the two-GPU topology first; each canonical sweep runs a fresh smoke,
# keeps its reservation through the boundary search, and validates raw outputs.
for model in glm4.7-flash qwen3-30b-fp8; do
    gpus=auto:1
    if [[ "$model" == glm4.7-flash ]]; then gpus=auto:2; fi
    command=(python3 "$package/sweep_decode_capacity.py" --config "$config"
             --repo "$benchmark_repo" --model "$model" --gpus "$gpus"
             --lanes sengine-omnikv --attempt total2048)
    printf 'COMMAND:'; printf ' %q' "${command[@]}"; printf '\n'
    "${command[@]}" 2>&1 | tee "$run_root/$model.run.log"
    printf '%s\t%s\tcompleted\n' "$(date -Is)" "$model" >> "$run_root/status.tsv"
done
