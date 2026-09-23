#!/usr/bin/env bash
# Keep two GPUs occupied with independent Qwen TP1 lanes, then use the pair for
# serial GLM TP2/EP2 lanes. Child failures never suppress later queued lanes.
set -uo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 RUN_ROOT GPU_A GPU_B [BENCHMARK_REPO]" >&2
    exit 2
fi

run_root=$(realpath "$1")
gpu_a=$2
gpu_b=$3
if [[ ! "$gpu_a" =~ ^[0-9]+$ || ! "$gpu_b" =~ ^[0-9]+$ || "$gpu_a" == "$gpu_b" ]]; then
    echo "GPU_A and GPU_B must be two distinct single-device indices" >&2
    exit 2
fi
glm_gpus="$gpu_a,$gpu_b"
benchmark_repo=$(realpath "${4:-.}")
package="$benchmark_repo/scripts/official_experiments/sparse_decode_efficiency"
runner="$package/run_lanes.sh"
config="$run_root/config.json"
launch="$run_root/launch.json"
status_file="$run_root/native-128k.queue.status.tsv"
lanes="sengine-vanilla,sengine-snapkv,sengine-quest,sengine-omnikv"

test -x "$runner"
test -f "$config"
test -f "$launch"
test ! -e "$status_file"

python3 - "$config" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
if config.get("measurement_protocol") != "boundary_sync_v2":
    raise SystemExit("Expected boundary_sync_v2")
if (config.get("input_len"), config.get("output_len")) != (131072, 2048):
    raise SystemExit("Expected 128K input / 2K output")
expected = {"qwen3-30b-fp8": (1, 1), "glm4.7-flash": (2, 2)}
observed = {name: (value["tp"], value["ep"]) for name, value in config["models"].items()}
if observed != expected:
    raise SystemExit(f"Unexpected model topology: {observed}")
PY

printf 'time\tmodel\tstatus\texit_code\tgpus\tattempt\n' > "$status_file"
failed=0
declare -A active_lane=()
declare -A active_gpu=()
declare -A active_attempt=()

on_stop() {
    printf '%s\tqueue\tuser_stopped\t130\t-\t-\n' "$(date -Is)" >> "$status_file"
    for pid in "${!active_lane[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for pid in "${!active_lane[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit 130
}
trap on_stop INT TERM HUP

launch_lane() {
    local model=$1
    local lane=$2
    local gpus=$3
    local attempt=$4
    printf '%s\t%s/%s\tstarted\t-\t%s\t%s\n' "$(date -Is)" "$model" "$lane" "$gpus" "$attempt" >> "$status_file"
    setsid bash "$runner" "$run_root" "$model" "$gpus" "$lane" "$attempt" \
        "$benchmark_repo" "" -- --continue-after-contention --isolated-cache-root --no-binary-search &
    local pid=$!
    active_lane[$pid]="$model/$lane"
    active_gpu[$pid]="$gpus"
    active_attempt[$pid]="$attempt"
}

finish_lane() {
    local pid=$1
    local code=$2
    local name=${active_lane[$pid]}
    local gpus=${active_gpu[$pid]}
    local attempt=${active_attempt[$pid]}
    if [[ $code -eq 0 ]]; then
        state=completed
    else
        state=failed_or_terminated_continuing
        failed=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$name" "$state" "$code" "$gpus" "$attempt" >> "$status_file"
    unset 'active_lane[$pid]' 'active_gpu[$pid]' 'active_attempt[$pid]'
}

wait_for_finished_lane() {
    local pid running_pid is_running
    local -a running_pids
    while :; do
        # wait -n skips children that exited before the call. Their statuses
        # remain available through wait PID, so collect one before blocking.
        mapfile -t running_pids < <(jobs -pr)
        for pid in "${!active_lane[@]}"; do
            is_running=0
            for running_pid in "${running_pids[@]}"; do
                if [[ "$pid" == "$running_pid" ]]; then
                    is_running=1
                    break
                fi
            done
            if (( !is_running )); then
                finished_pid=$pid
                if wait "$pid"; then code=0; else code=$?; fi
                return
            fi
        done
        unset finished_pid
        if wait -n -p finished_pid "${!active_lane[@]}"; then code=0; else code=$?; fi
        [[ -n ${finished_pid-} ]] && return
    done
}

qwen_lanes=(sengine-vanilla sengine-snapkv sengine-quest sengine-omnikv)
next_qwen=0
for gpu in "$gpu_a" "$gpu_b"; do
    lane=${qwen_lanes[$next_qwen]}
    launch_lane qwen3-30b-fp8 "$lane" "$gpu" "native-128k-qwen-${lane#sengine-}"
    next_qwen=$((next_qwen + 1))
done

while ((${#active_lane[@]})); do
    wait_for_finished_lane
    freed_gpu=${active_gpu[$finished_pid]}
    finish_lane "$finished_pid" "$code"
    if ((next_qwen < ${#qwen_lanes[@]})); then
        lane=${qwen_lanes[$next_qwen]}
        launch_lane qwen3-30b-fp8 "$lane" "$freed_gpu" "native-128k-qwen-${lane#sengine-}"
        next_qwen=$((next_qwen + 1))
    fi
done

for lane in sengine-vanilla sengine-snapkv sengine-quest sengine-omnikv; do
    launch_lane glm4.7-flash "$lane" "$glm_gpus" "native-128k-glm-${lane#sengine-}"
    pid=${!active_lane[@]}
    if wait "$pid"; then code=0; else code=$?; fi
    finish_lane "$pid" "$code"
done

printf '%s\tqueue\t%s\t%s\t-\t-\n' "$(date -Is)" "$([[ $failed -eq 0 ]] && echo completed || echo completed_with_failures)" "$failed" >> "$status_file"
exit "$failed"
