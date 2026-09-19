#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 BENCHMARK_REPO PREPARED_RUN_ROOT VALIDATION_ROOT GPUS" >&2
    exit 2
fi
benchmark_repo=$(realpath "$1")
run_root=$(realpath "$2")
validation_root=$(realpath "$3")
gpus=$4
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CONDA_EXE:?Set conda executable}"
: "${SPARSE_ENGINE_ENV:?Set native CUDA conda environment}"
test -f "$run_root/manifest.json"
test ! -e "$run_root/run.log"
exec > >(tee "$run_root/run.log") 2>&1
trap 'code=$?; printf "%s\tqueue\texit\t%s\n" "$(date -Is)" "$code" >> "$run_root/status.tsv"' EXIT
printf '%s\tvalidation\twaiting\n' "$(date -Is)" >> "$run_root/status.tsv"
for ((attempt=0; attempt<240; attempt++)); do
    if rg -q '^gpu-tests\s+failed' "$validation_root/status.tsv"; then exit 1; fi
    if rg -q '^gpu-tests\s+completed' "$validation_root/status.tsv"; then break; fi
    sleep 5
done
rg -q '^gpu-tests\s+completed' "$validation_root/status.tsv"
printf '%s\tvalidation\tcompleted\n' "$(date -Is)" >> "$run_root/status.tsv"
export TILELANG_CACHE_DIR="$run_root/cache/tilelang"
python3 "$package/sweep_decode_capacity.py" --config "$run_root/config.json" \
    --repo "$benchmark_repo" --model glm4.7-flash --gpus "$gpus" \
    --lanes sengine-omnikv --attempt sm-parallel
"$CONDA_EXE" run --no-capture-output -p "$SPARSE_ENGINE_ENV" python \
    "$package/update_mla_profile_results.py" export --run-root "$run_root"
printf '%s\texport\tcompleted\n' "$(date -Is)" >> "$run_root/status.tsv"
