#!/usr/bin/env bash
set -euo pipefail

model_label=${1:?model label}
model_path=${2:?model checkpoint path}
method=${3:?method from chain_cache_miniswe/setting.json}
concurrency=${4:?agent and engine concurrency}
data_root=${5:?persistent output root}
trace_dir=${6:?trace directory}
gpus=${7:?two idle GPU indices, comma separated}
port=${8:?unused local port}
mode=${9:-full}
forced_workload=${10:?prepared forced workload JSON}
case_root="$data_root/${model_label}_${method}_c${concurrency}"
scratch_root="${AGENT_TRACE_SCRATCH_BASE:-$data_root}/${model_label}_${method}_c${concurrency}"
run_root="$case_root/run"
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)
recipe=scripts/official_experiments/chain_cache_miniswe/run.py
snapkv_decode_eviction=${AGENT_TRACE_SNAPKV_DECODE_EVICTION:-0}
snapkv_total_budget=${AGENT_TRACE_SNAPKV_TOTAL_BUDGET:-0}
h2o_swelite_setting=${AGENT_TRACE_H2O_SWELITE_SETTING:-0}

cd "$repo_root"
if [[ -n ${VENV_ACTIVATE:-} ]]; then
  source "$VENV_ACTIVATE"
else
  source "${CONDA_BASE:?set CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH:?set CONDA_ENV_PATH}"
fi
export PYTHONPATH="$PWD/src:$PWD"
export CUDA_VISIBLE_DEVICES="$gpus"
export SPARSEENGINE_GRAPH_CAPTURE_DIAGNOSTICS=1
export SPARSEVLLM_MASTER_PORT="${SPARSEVLLM_MASTER_PORT:?set SPARSEVLLM_MASTER_PORT}"
export TMPDIR="$scratch_root/tmp"
export XDG_CACHE_HOME="$scratch_root/cache"
mkdir -p "$case_root" "$TMPDIR" "$scratch_root/compiler" "$XDG_CACHE_HOME"
exec >>"$case_root/run.log" 2>&1

status() {
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$model_label" "$method" "$1" >>"$case_root/status.tsv"
}

server_pid=
phase=
stop_server() {
  if [[ -n "$server_pid" ]]; then
    if [[ -e "$run_root/$method/$phase/server_process.json" ]]; then
      python "$recipe" stop --root "$run_root" --method "$method" --phase "$phase" --reason "${1:-coordinator_failure}" || true
    fi
    wait "$server_pid" || true
    server_pid=
  fi
}
trap 'stop_server coordinator_failure' EXIT INT TERM

wait_gpu_release() {
  local attempt
  for ((attempt=0; attempt<150; attempt++)); do
    if python - "$gpus" <<'PY' >/dev/null 2>&1
import sys
from scripts.official_experiments.chain_cache_miniswe.run import idle_pair
idle_pair(sys.argv[1])
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "GPU pair did not become idle after stopping the smoke server" >&2
  return 1
}

wait_ready() {
  local i
  # A 30-minute startup indicates an abnormal capture or collective stall.
  for ((i=0; i<900; i++)); do
    if curl --silent --show-error --noproxy '*' --max-time 2 "http://127.0.0.1:$port/readyz" > "$case_root/readyz.$phase.json" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before readiness: $phase" >&2
      return 1
    fi
    sleep 2
  done
  echo "Server readiness timed out after 30 minutes: $phase" >&2
  return 1
}

wait_port_release() {
  local i
  for ((i=0; i<60; i++)); do
    if python - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.socket() as probe:
    probe.bind(("127.0.0.1", int(sys.argv[1])))
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "Server port remains occupied after two minutes: $port" >&2
  return 1
}

start_server() {
  phase=$1
  wait_port_release
  status "${phase}_server_starting"
  python "$recipe" serve --root "$run_root" --method "$method" --phase "$phase" \
    --python "$(command -v python)" --model "$model_path" \
    --served-model-name "$model_label-$method" \
    --gpus "$gpus" --port "$port" --timeout 86400 \
    --compile-cache-root "$scratch_root/compiler" \
    >"$case_root/server_driver.$phase.log" 2>&1 &
  server_pid=$!
  wait_ready
  status "${phase}_server_ready"
}

if [[ ! -d "$model_path" || ! -f "$trace_dir/manifest.json" || ! -f "$forced_workload" ]]; then
  echo "Missing model, trace, or prepared forced workload" >&2
  exit 1
fi
case "$method" in
  vanilla-prefix|omnikv-prefix|quest-prefix|snapkv-chain|h2o-chain) ;;
  *) echo "Method has no frozen completed MiniSWE concurrency: $method" >&2; exit 2 ;;
esac
if [[ "$snapkv_decode_eviction" != 0 && "$snapkv_decode_eviction" != 1 ]]; then
  echo "AGENT_TRACE_SNAPKV_DECODE_EVICTION must be 0 or 1" >&2
  exit 2
fi
if [[ "$snapkv_decode_eviction" == 1 && "$method" != snapkv-chain ]]; then
  echo "SnapKV decode eviction requires snapkv-chain" >&2
  exit 2
fi
if [[ ! "$snapkv_total_budget" =~ ^[0-9]+$ ]]; then
  echo "AGENT_TRACE_SNAPKV_TOTAL_BUDGET must be a nonnegative integer" >&2
  exit 2
fi
if (( snapkv_total_budget > 0 )) && [[ "$method" != snapkv-chain || "$snapkv_decode_eviction" != 1 ]]; then
  echo "A SnapKV total budget requires snapkv-chain with decode eviction enabled" >&2
  exit 2
fi
if [[ "$h2o_swelite_setting" != 0 && "$h2o_swelite_setting" != 1 ]]; then
  echo "AGENT_TRACE_H2O_SWELITE_SETTING must be 0 or 1" >&2
  exit 2
fi
if [[ "$h2o_swelite_setting" == 1 && "$method" != h2o-chain ]]; then
  echo "The SWE-lite H2O setting requires h2o-chain" >&2
  exit 2
fi
if [[ "$mode" != smoke-only && "$mode" != full ]]; then
  echo "Mode must be smoke-only or full" >&2
  exit 2
fi
if [[ -e "$case_root/status.tsv" ]]; then
  echo "Case already started; refusing to overwrite: $case_root" >&2
  exit 1
fi
resident=$((concurrency * 3 / 2))
if (( resident * 2 != concurrency * 3 )); then
  echo "1.5 times concurrency is not an integer" >&2
  exit 1
fi
setting_args=()
if [[ "$snapkv_decode_eviction" == 1 || "$h2o_swelite_setting" == 1 ]]; then
  python - "$case_root/effective_setting.json" "$snapkv_decode_eviction" "$h2o_swelite_setting" "$snapkv_total_budget" <<'PY'
import json
import sys
from pathlib import Path

from sparseengine.configs.groups import SparseMethodConfig

setting = json.loads(Path("scripts/official_experiments/chain_cache_miniswe/setting.json").read_text())
if int(sys.argv[2]):
    method = setting["methods"]["snapkv-chain"]
    method["snapkv_decode_eviction"] = True
    method["decode_eviction_interval"] = SparseMethodConfig().decode_eviction_interval
    total_budget = int(sys.argv[4])
    if total_budget:
        selected = total_budget - int(method["sink_keep_tokens"]) - int(method["recent_keep_tokens"])
        if selected <= 0:
            raise ValueError("SnapKV total budget must exceed sink plus recent tokens")
        method["decode_keep_tokens"] = selected
if int(sys.argv[3]):
    engine = setting["engine"]
    engine["engine_prefill_chunk_size"] = 16384
    engine["favor_min_decoding_seqs"] = 48
    engine["sparse_attn_score_dtype"] = "float32"
    engine["decode_reservation_tokens"] = 1024
    method = setting["methods"]["h2o-chain"]
    method["sparse_prefill_score_mode"] = "logits"
    method["h2o_decode_eviction"] = False
Path(sys.argv[1]).write_text(json.dumps(setting, indent=2) + "\n")
PY
  setting_args=(--setting "$case_root/effective_setting.json")
fi
python "$recipe" prepare "${setting_args[@]}" --root "$run_root" --concurrency "$concurrency" \
  --engine-concurrency "$concurrency" --engine-resident-concurrency "$resident"
python - "$run_root/$method/engine.json" "$concurrency" "$resident" "$snapkv_decode_eviction" "$h2o_swelite_setting" "$snapkv_total_budget" <<'PY'
import json,sys
config=json.load(open(sys.argv[1]))
c,r=map(int,sys.argv[2:4])
assert config['max_num_seqs_in_batch']==config['max_decoding_seqs']==c
assert config['max_num_seqs_in_gpu']==r
if int(sys.argv[4]):
    assert config['snapkv_decode_eviction'] is True
    assert config['decode_eviction_interval'] == 1024
    if int(sys.argv[6]):
        assert (config['sink_keep_tokens'] + config['recent_keep_tokens']
                + config['decode_keep_tokens']) == int(sys.argv[6])
if int(sys.argv[5]):
    assert config['sparse_prefill_score_mode'] == 'logits'
    assert config['sparse_attn_score_dtype'] == 'float32'
    assert config['h2o_decode_eviction'] is False
    assert config['h2o_prefill_budget'] == 16384
    assert config['h2o_decode_budget'] == 8192
    assert config['h2o_prefill_score_window'] == 128
    assert config['h2o_recent_ratio'] == 0.5
    assert config['decode_reservation_tokens'] == 1024
    assert config['engine_prefill_chunk_size'] == 16384
    assert config['favor_min_decoding_seqs'] == 48
PY
status "prepared"

start_server smoke
python - "$trace_dir" "$model_label-$method" "$model_path" "$method" "$port" "$case_root/smoke.json" <<'PY'
import json,sys,time
from pathlib import Path
import httpx
from transformers import AutoTokenizer
from benchmark.sparseengine_regression.agent_trace import replay_body,prepare_forced_agent
trace, model, model_path, method, port, output=sys.argv[1:]
manifest=json.load(open(Path(trace)/'manifest.json'))
agent=json.load(open(Path(trace)/manifest['agents'][0]['file']))
agent['turns']=agent['turns'][:3]
prepared=prepare_forced_agent(agent,AutoTokenizer.from_pretrained(model_path,use_fast=True),model)
chain_mode=method in ('snapkv-chain','h2o-chain')
results=[]
chain_id=None
with httpx.Client(timeout=900,trust_env=False) as client:
    for index,(turn,spec) in enumerate(zip(agent['turns'][:2],prepared[:2])):
        turn['benchmark_forced_token_ids']=spec['token_ids']
        turn['completion_tokens']=len(spec['token_ids'])
        payload=replay_body(turn,model)
        if chain_mode and chain_id is not None:
            payload['chain_id']=chain_id
            payload['chain_append_start']=prepared[index-1]['chain_append_start']
        start=time.perf_counter()
        response=client.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=payload)
        response.raise_for_status()
        body=response.json()
        count=body['usage']['completion_tokens']
        if count != turn['completion_tokens']:
            raise ValueError(f'Smoke completion count mismatch: {count} != {turn["completion_tokens"]}')
        cached=int((body['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0))
        if chain_mode:
            expected='created' if chain_id is None else 'resumed'
            if body.get('chain_status')!=expected:
                raise ValueError(f'Chain smoke status {body.get("chain_status")!r} != {expected!r}')
            chain_id=body.get('chain_id')
        results.append({'completion_tokens':count,'latency_s':time.perf_counter()-start,
                        'cached_tokens':cached,'chain_status':body.get('chain_status')})
if results[1]['cached_tokens']<=0:
    raise ValueError('Second smoke request did not reuse cached prompt tokens')
Path(output).write_text(json.dumps({'status':'success','requests':results})+'\n')
PY
status "smoke_passed"
stop_server requested_by_operator
if [[ "$mode" == smoke-only ]]; then
  exit 0
fi

wait_gpu_release
start_server full
status "replay_running"
python benchmark/sparseengine_regression/run_suite.py --layer agent_trace \
  --agent_trace "$trace_dir" --agent_api_base "http://127.0.0.1:$port/v1" \
  --agent_server_manifest "$run_root/$method/full/server_manifest.json" \
  --agent_concurrency "$concurrency" --agent_request_timeout 900 \
  --agent_synthetic_think_time_max_s 2 --agent_synthetic_think_time_seed 42 \
  --agent_force_recorded_responses \
  --agent_forced_workload "$forced_workload" \
  --agent_require_cache_hit \
  --output_root "$case_root" --run_id replay
curl --fail --silent --show-error --noproxy '*' --max-time 10 \
  "http://127.0.0.1:$port/v1/worker/load" > "$case_root/worker_load_after_replay.json"
python - "$case_root/worker_load_after_replay.json" <<'PY'
import json,sys
load=json.load(open(sys.argv[1]))
if load["total_preemptions"] or load["total_recompute_replays"]:
    raise ValueError(
        "Replay did not sustain its configured concurrency without active-request "
        f"preemption: preemptions={load['total_preemptions']} "
        f"recompute_replays={load['total_recompute_replays']}"
    )
PY
python - "$case_root/sparseengine_regression/replay/agent_trace.json" "$case_root" "$trace_dir/manifest.json" <<'PY'
import json,sys
from pathlib import Path
summary=json.load(open(sys.argv[1]))
expected=json.load(open(sys.argv[3]))['request_count']
if summary['status']!='success' or summary['request_count']!=expected:
    raise ValueError(f'Incomplete replay: {summary["status"]}, {summary["request_count"]} of {expected} requests')
Path(sys.argv[2],'summary.json').write_text(json.dumps({k:summary.get(k) for k in ('status','request_count','elapsed_s','latency_s_p50','latency_s_p95','latency_s_p99','output_token_throughput_tps','cache_reuse')},indent=2)+'\n')
PY
status "replay_passed"
stop_server requested_by_operator
