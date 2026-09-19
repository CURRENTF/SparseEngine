#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_ROOT="${DELTAKV_MODEL_ROOT:-${REPO_ROOT}/models}"
OUTPUT_BASE="${SPARSEENGINE_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
CACHE_ROOT="${DELTAKV_CACHE_DIR:-${REPO_ROOT}/.cache}"
CONDA_ENVS_ROOT="${CONDA_ENVS_ROOT:-${HOME}/.conda/envs}"
RUN_NAME="${RUN_NAME:-claw_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_BASE}/Sparse-Engine/claw-eval}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
TRACE_DIR="${RUN_DIR}/traces"
REQUEST_LOG_DIR="${RUN_DIR}/sparseengine_requests"
ENGINE_KWARGS_FILE="${ENGINE_KWARGS_FILE:-${RUN_DIR}/engine_kwargs.json}"
RUN_MANIFEST="${RUN_MANIFEST:-${RUN_DIR}/run_manifest.json}"
mkdir -p "${LOG_DIR}" "${TRACE_DIR}" "${REQUEST_LOG_DIR}"

MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-7B-Instruct-1M}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-sparseengine-claw}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-18000}"
SERVER_HEALTH_URL="${SERVER_HEALTH_URL:-http://${SERVER_HOST}:${SERVER_PORT}/health}"
START_SPARSEENGINE_SERVER="${START_SPARSEENGINE_SERVER:-1}"
SPARSEENGINE_OPENAI_BASE_URL="${SPARSEENGINE_OPENAI_BASE_URL:-http://${SERVER_HOST}:${SERVER_PORT}/v1}"
SPARSEENGINE_OPENAI_API_KEY="${SPARSEENGINE_OPENAI_API_KEY:-local-sparseengine}"
SPARSEENGINE_CLAW_MODEL_ID="${SPARSEENGINE_CLAW_MODEL_ID:-${SERVED_MODEL_NAME}}"
SPARSEENGINE_CONTEXT_WINDOW="${SPARSEENGINE_CONTEXT_WINDOW:-131072}"
CLAW_EVAL_JUDGE_MODEL="${CLAW_EVAL_JUDGE_MODEL:-google/gemini-3-flash-preview}"
CLAW_EVAL_JUDGE_BASE_URL="${CLAW_EVAL_JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"

CLAW_EVAL_DIR="${CLAW_EVAL_DIR:-${REPO_ROOT}/../claw-eval}"
CLAW_EVAL_REF="${CLAW_EVAL_REF:-main}"
CLAW_EVAL_CONFIG_TEMPLATE="${CLAW_EVAL_CONFIG_TEMPLATE:-${REPO_ROOT}/benchmark/claw_eval/sparseengine_config.yaml}"
CLAW_EVAL_RESULT_VALIDATOR="${CLAW_EVAL_RESULT_VALIDATOR:-${REPO_ROOT}/benchmark/claw_eval/validate_results.py}"
CLAW_EVAL_TASK_SELECTOR="${CLAW_EVAL_TASK_SELECTOR:-${REPO_ROOT}/benchmark/claw_eval/select_tasks.py}"
CLAW_EVAL_CONFIG="${RUN_DIR}/sparseengine_config.yaml"
CLAW_EVAL_ARGS="${CLAW_EVAL_ARGS:-batch --config ${CLAW_EVAL_CONFIG} --sandbox --trials 3 --parallel 1}"
CLAW_EVAL_TEXT_ONLY="${CLAW_EVAL_TEXT_ONLY:-0}"
CLAW_EVAL_RESUME_TRACE_DIR="${CLAW_EVAL_RESUME_TRACE_DIR:-}"
CLAW_EVAL_SELECTION_SUMMARY=""
CLAW_EVAL_SKIPPED_RESULTS=""
CLAW_EVAL_UPDATE_REPO="${CLAW_EVAL_UPDATE_REPO:-1}"
CLAW_EVAL_SANDBOX_IMAGE="${CLAW_EVAL_SANDBOX_IMAGE:-claw-eval-agent:latest}"
CLAW_EVAL_BUILD_SANDBOX_IMAGE="${CLAW_EVAL_BUILD_SANDBOX_IMAGE:-0}"
CLAW_EVAL_DOCKER_BUILD_ARGS="${CLAW_EVAL_DOCKER_BUILD_ARGS:-}"
CLAW_EVAL_COMMIT=""
CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE=""
CLAW_EVAL_SANDBOX_IMAGE_ID=""
CLAW_EVAL_SANDBOX_IMAGE_SIZE_BYTES=""
CLAW_EVAL_PRE_RUN_SNAPSHOT="${RUN_DIR}/claw_eval_pre_run_snapshot.json"
CLAW_EVAL_PER_SAMPLE_RESULTS="${RUN_DIR}/per_sample_results.jsonl"
CLAW_EVAL_FINAL_SUMMARY="${RUN_DIR}/final_summary.json"
SETUP_ONLY="${SETUP_ONLY:-0}"

SPARSEENGINE_CONDA_ENV="${SPARSEENGINE_CONDA_ENV:-${CONDA_ENVS_ROOT}/sparse-engine-tf530}"
CLAW_EVAL_CONDA_ENV="${CLAW_EVAL_CONDA_ENV:-${CONDA_ENVS_ROOT}/claw-eval-py311}"
SPARSEENGINE_MASTER_PORT="${SPARSEENGINE_MASTER_PORT:-2333}"
SPARSEENGINE_DATA_PARALLEL_SIZE="${SPARSEENGINE_DATA_PARALLEL_SIZE:-1}"
if [[ -z "${ENGINE_KWARGS:-}" ]]; then
  ENGINE_KWARGS="{\"tensor_parallel_size\":1,\"gpu_memory_utilization\":0.88,\"max_model_len\":${SPARSEENGINE_CONTEXT_WINDOW},\"engine_prefill_chunk_size\":4096,\"sparse_method\":\"vanilla\"}"
fi
SPARSEENGINE_PYTHON_BIN="${SPARSEENGINE_PYTHON_BIN:-python}"

DOWNLOAD_HTTP_PROXY="${DOWNLOAD_HTTP_PROXY:-http://127.0.0.1:7898}"
DOWNLOAD_HTTPS_PROXY="${DOWNLOAD_HTTPS_PROXY:-${DOWNLOAD_HTTP_PROXY}}"
CLASH_DIR="${CLASH_DIR:-${HOME}/clash}"
CLASH_DOWNLOAD_CONFIG="${CLASH_DOWNLOAD_CONFIG:-${CLASH_DIR}/config.download.yaml}"
CLASH_DOWNLOAD_LOG="${LOG_DIR}/clash_download.log"

export RUN_NAME
export REPO_ROOT
export RUN_DIR
export TRACE_DIR
export REQUEST_LOG_DIR
export MODEL_PATH
export SERVED_MODEL_NAME
export CUDA_VISIBLE_DEVICES
export SERVER_HEALTH_URL
export START_SPARSEENGINE_SERVER
export SPARSEENGINE_OPENAI_BASE_URL
export SPARSEENGINE_CLAW_MODEL_ID
export SPARSEENGINE_CONTEXT_WINDOW
export CLAW_EVAL_JUDGE_BASE_URL
export CLAW_EVAL_JUDGE_MODEL
export CLAW_EVAL_DIR
export CLAW_EVAL_CONFIG
export CLAW_EVAL_ARGS
export CLAW_EVAL_UPDATE_REPO
export CLAW_EVAL_SANDBOX_IMAGE
export CLAW_EVAL_BUILD_SANDBOX_IMAGE
export CLAW_EVAL_COMMIT
export CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE
export CLAW_EVAL_SANDBOX_IMAGE_ID
export CLAW_EVAL_SANDBOX_IMAGE_SIZE_BYTES
export CLAW_EVAL_RESULT_VALIDATOR
export CLAW_EVAL_TEXT_ONLY
export CLAW_EVAL_RESUME_TRACE_DIR
export CLAW_EVAL_SELECTION_SUMMARY
export CLAW_EVAL_SKIPPED_RESULTS
export SPARSEENGINE_CONDA_ENV
export CLAW_EVAL_CONDA_ENV
export SPARSEENGINE_MASTER_PORT
export SPARSEENGINE_DATA_PARALLEL_SIZE
export SETUP_ONLY
export DOWNLOAD_HTTP_PROXY
export ENGINE_KWARGS_FILE

require_file() {
  local path="$1"
  local what="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing ${what}: ${path}" >&2
    exit 2
  fi
}

require_bool() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[ERROR] ${name} must be 0 or 1, got: ${value}" >&2
    exit 2
  fi
}

start_download_clash_if_needed() {
  local proxy_address="${DOWNLOAD_HTTP_PROXY#*://}"
  proxy_address="${proxy_address%%/*}"
  local proxy_port="${proxy_address##*:}"
  if [[ ! "${proxy_port}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] DOWNLOAD_HTTP_PROXY must include a numeric port: ${DOWNLOAD_HTTP_PROXY}" >&2
    exit 3
  fi
  if ss -ltn 2>/dev/null | grep -qE ":${proxy_port}[[:space:]]"; then
    return
  fi
  require_file "${CLASH_DIR}/clash" "Clash binary"
  require_file "${CLASH_DOWNLOAD_CONFIG}" "download-only Clash config"
  echo "[INFO] Starting download-only Clash for ${DOWNLOAD_HTTP_PROXY}"
  (
    cd "${CLASH_DIR}"
    nohup ./clash -d "${CLASH_DIR}" -f "${CLASH_DOWNLOAD_CONFIG}" >"${CLASH_DOWNLOAD_LOG}" 2>&1 &
  )
  for _ in $(seq 1 30); do
    if ss -ltn 2>/dev/null | grep -qE ":${proxy_port}[[:space:]]"; then
      return
    fi
    sleep 1
  done
  echo "[ERROR] download-only Clash did not open port ${proxy_port}. See ${CLASH_DOWNLOAD_LOG}" >&2
  exit 3
}

download_env() {
  http_proxy="${DOWNLOAD_HTTP_PROXY}" \
  https_proxy="${DOWNLOAD_HTTPS_PROXY}" \
  HTTP_PROXY="${DOWNLOAD_HTTP_PROXY}" \
  HTTPS_PROXY="${DOWNLOAD_HTTPS_PROXY}" \
  NO_PROXY="127.0.0.1,localhost,::1" \
  no_proxy="127.0.0.1,localhost,::1" \
  "$@"
}

activate_conda_env() {
  set +u
  conda activate "$1"
  set -u
}

deactivate_conda_env() {
  set +u
  conda deactivate
  set -u
}

require_clean_claw_eval_checkout() {
  local status
  status="$(git -C "${CLAW_EVAL_DIR}" status --porcelain --untracked-files=all)"
  if [[ -n "${status}" ]]; then
    echo "[ERROR] Claw-Eval checkout must be clean for a reproducible run:" >&2
    printf '%s\n' "${status}" | sed -n '1,20p' >&2
    exit 3
  fi
}

prepare_claw_eval_repo() {
  mkdir -p "$(dirname "${CLAW_EVAL_DIR}")"
  if [[ "${CLAW_EVAL_UPDATE_REPO}" == "0" ]]; then
    require_file "${CLAW_EVAL_DIR}/.git" "existing Claw-Eval checkout"
    CLAW_EVAL_COMMIT="$(git -C "${CLAW_EVAL_DIR}" rev-parse HEAD)"
    export CLAW_EVAL_COMMIT
    require_clean_claw_eval_checkout
    echo "[INFO] Reusing Claw-Eval checkout at ${CLAW_EVAL_COMMIT}"
    return
  fi

  start_download_clash_if_needed
  if [[ -d "${CLAW_EVAL_DIR}/.git" ]]; then
    echo "[INFO] Updating Claw-Eval repo at ${CLAW_EVAL_DIR}"
    download_env git -C "${CLAW_EVAL_DIR}" fetch origin "${CLAW_EVAL_REF}"
    download_env git -C "${CLAW_EVAL_DIR}" checkout "${CLAW_EVAL_REF}"
    download_env git -C "${CLAW_EVAL_DIR}" pull --ff-only
  else
    echo "[INFO] Cloning Claw-Eval repo to ${CLAW_EVAL_DIR}"
    download_env git clone https://github.com/claw-eval/claw-eval "${CLAW_EVAL_DIR}"
    download_env git -C "${CLAW_EVAL_DIR}" checkout "${CLAW_EVAL_REF}"
  fi
  CLAW_EVAL_COMMIT="$(git -C "${CLAW_EVAL_DIR}" rev-parse HEAD)"
  export CLAW_EVAL_COMMIT
  require_clean_claw_eval_checkout
}

prepare_claw_eval_env() {
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
  mkdir -p "${PIP_CACHE_DIR}"
  if [[ ! -x "${CLAW_EVAL_CONDA_ENV}/bin/python" ]]; then
    start_download_clash_if_needed
    echo "[INFO] Creating Claw-Eval Python 3.11 env at ${CLAW_EVAL_CONDA_ENV}"
    download_env conda create -y -p "${CLAW_EVAL_CONDA_ENV}" python=3.11 pip
  fi

  activate_conda_env "${CLAW_EVAL_CONDA_ENV}"
  if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("claw_eval") else 1)
PY
  then
    start_download_clash_if_needed
    echo "[INFO] Installing Claw-Eval into ${CLAW_EVAL_CONDA_ENV}"
    download_env python -m pip install -U pip
    download_env python -m pip install -e "${CLAW_EVAL_DIR}[mock,sandbox]"
  fi
  deactivate_conda_env
}

render_config() {
  require_file "${CLAW_EVAL_CONFIG_TEMPLATE}" "Claw-Eval config template"
  export SPARSEENGINE_OPENAI_API_KEY
  export SPARSEENGINE_OPENAI_BASE_URL
  export SPARSEENGINE_CLAW_MODEL_ID
  export SPARSEENGINE_CONTEXT_WINDOW
  export CLAW_EVAL_TRACE_DIR="${TRACE_DIR}"
  export CLAW_EVAL_JUDGE_MODEL
  export CLAW_EVAL_JUDGE_BASE_URL
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
  "${CLAW_EVAL_CONDA_ENV}/bin/python" - "${CLAW_EVAL_CONFIG_TEMPLATE}" "${CLAW_EVAL_CONFIG}" <<'PY'
import os
import sys
from string import Template

src, dst = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as f:
    text = f.read()
rendered = Template(text).substitute(os.environ)
with open(dst, "w", encoding="utf-8") as f:
    f.write(rendered)
os.chmod(dst, 0o600)
PY
}

write_engine_kwargs_file() {
  ENGINE_KWARGS_RAW="${ENGINE_KWARGS}" \
    CLAW_EVAL_ARGS_RAW="${CLAW_EVAL_ARGS}" \
    SPARSEENGINE_DATA_PARALLEL_SIZE="${SPARSEENGINE_DATA_PARALLEL_SIZE}" \
    "${CLAW_EVAL_CONDA_ENV}/bin/python" - "${ENGINE_KWARGS_FILE}" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

raw = os.environ["ENGINE_KWARGS_RAW"].strip()
if raw.startswith("{"):
    data = json.loads(raw)
else:
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"ENGINE_KWARGS is neither JSON nor an existing path: {raw}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
if not isinstance(data, dict):
    raise ValueError("ENGINE_KWARGS must resolve to a JSON object")
args = shlex.split(os.environ["CLAW_EVAL_ARGS_RAW"])
parallel_values = []
for index, arg in enumerate(args):
    if arg == "--parallel":
        if index + 1 >= len(args):
            raise ValueError("--parallel requires a value")
        parallel_values.append(args[index + 1])
    elif arg.startswith("--parallel="):
        parallel_values.append(arg.split("=", 1)[1])
if len(parallel_values) != 1:
    raise ValueError(
        "CLAW_EVAL_ARGS must set --parallel exactly once so server capacity is reproducible"
    )
parallel = int(parallel_values[0])
dp_size = int(os.environ["SPARSEENGINE_DATA_PARALLEL_SIZE"])
if parallel <= 0 or dp_size <= 0:
    raise ValueError(f"parallel and data-parallel size must be positive: {parallel=}, {dp_size=}")
if parallel % dp_size:
    raise ValueError(
        "benchmark parallelism must be divisible by data-parallel size: "
        f"{parallel} % {dp_size} != 0"
    )
data["max_decoding_seqs"] = parallel // dp_size
dst = Path(sys.argv[1])
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")
print(
    f"server_max_decoding_seqs={data['max_decoding_seqs']} "
    f"benchmark_parallel={parallel} data_parallel_size={dp_size}"
)
PY
}

resolve_claw_eval_args() {
  CLAW_EVAL_ARGS_TEMPLATE="${CLAW_EVAL_ARGS}" \
    CLAW_EVAL_CONFIG="${CLAW_EVAL_CONFIG}" \
    TRACE_DIR="${TRACE_DIR}" \
    "${CLAW_EVAL_CONDA_ENV}/bin/python" - <<'PY'
import os
from string import Template

print(Template(os.environ["CLAW_EVAL_ARGS_TEMPLATE"]).safe_substitute(os.environ))
PY
}

prepare_resume_args() {
  if [[ -z "${CLAW_EVAL_RESUME_TRACE_DIR}" ]]; then
    return
  fi
  require_file "${CLAW_EVAL_RESUME_TRACE_DIR}" "Claw-Eval resume trace directory"
  CLAW_EVAL_ARGS_RAW="${CLAW_EVAL_ARGS}" \
    CLAW_EVAL_RESUME_TRACE_DIR="${CLAW_EVAL_RESUME_TRACE_DIR}" \
    TRACE_DIR="${TRACE_DIR}" \
    "${CLAW_EVAL_CONDA_ENV}/bin/python" - <<'PY'
import os
import shlex
from pathlib import Path

args = shlex.split(os.environ["CLAW_EVAL_ARGS_RAW"])
if "--continue" in args or any(arg.startswith("--continue=") for arg in args):
    raise SystemExit("CLAW_EVAL_ARGS already contains --continue")
resume = Path(os.environ["CLAW_EVAL_RESUME_TRACE_DIR"]).resolve()
trace_dir = Path(os.environ["TRACE_DIR"]).resolve()
if not resume.is_relative_to(trace_dir):
    raise SystemExit(
        f"resume trace must be inside this run's TRACE_DIR: {resume} is not under {trace_dir}"
    )
PY
  printf -v resume_trace_quoted '%q' "${CLAW_EVAL_RESUME_TRACE_DIR}"
  CLAW_EVAL_ARGS+=" --continue ${resume_trace_quoted}"
}

prepare_task_selection() {
  if [[ "${CLAW_EVAL_TEXT_ONLY}" != "1" ]]; then
    return
  fi
  require_file "${CLAW_EVAL_TASK_SELECTOR}" "Claw-Eval task selector"
  local source_tasks_dir="${CLAW_EVAL_DIR}/tasks"
  local selection_root="${RUN_DIR}/text_only_selection"
  local selected_tasks_dir="${selection_root}/tasks"
  CLAW_EVAL_SELECTION_SUMMARY="${RUN_DIR}/task_selection.json"
  CLAW_EVAL_SKIPPED_RESULTS="${RUN_DIR}/skipped_by_policy.jsonl"
  export CLAW_EVAL_SELECTION_SUMMARY
  export CLAW_EVAL_SKIPPED_RESULTS

  local tag
  tag="$({
    CLAW_EVAL_ARGS_RAW="${CLAW_EVAL_ARGS}" "${CLAW_EVAL_CONDA_ENV}/bin/python" - <<'PY'
import os
import shlex

args = shlex.split(os.environ["CLAW_EVAL_ARGS_RAW"])
if "--tasks-dir" in args or any(arg.startswith("--tasks-dir=") for arg in args):
    raise SystemExit("CLAW_EVAL_TEXT_ONLY=1 owns --tasks-dir; remove it from CLAW_EVAL_ARGS")
values = []
for index, arg in enumerate(args):
    if arg == "--tag":
        if index + 1 >= len(args):
            raise SystemExit("--tag requires a value")
        values.append(args[index + 1])
    elif arg.startswith("--tag="):
        values.append(arg.split("=", 1)[1])
if len(values) > 1:
    raise SystemExit("CLAW_EVAL_ARGS must not set --tag more than once")
print(values[0] if values else "")
PY
  })"

  local -a selector_args=(
    --source-tasks-dir "${source_tasks_dir}"
    --output-root "${selection_root}"
    --summary "${CLAW_EVAL_SELECTION_SUMMARY}"
    --skipped-results "${CLAW_EVAL_SKIPPED_RESULTS}"
  )
  if [[ -n "${tag}" ]]; then
    selector_args+=(--tag "${tag}")
  fi
  "${CLAW_EVAL_CONDA_ENV}/bin/python" "${CLAW_EVAL_TASK_SELECTOR}" "${selector_args[@]}"
  printf -v selected_tasks_quoted '%q' "${selected_tasks_dir}"
  CLAW_EVAL_ARGS+=" --tasks-dir ${selected_tasks_quoted}"
}

resolve_effective_sandbox_image() {
  CLAW_EVAL_ARGS_RAW="${CLAW_EVAL_ARGS}" \
    CLAW_EVAL_SANDBOX_IMAGE="${CLAW_EVAL_SANDBOX_IMAGE}" \
    "${CLAW_EVAL_CONDA_ENV}/bin/python" - <<'PY'
import os
import shlex

args = shlex.split(os.environ["CLAW_EVAL_ARGS_RAW"])
if "--sandbox" not in args:
    print("")
    raise SystemExit(0)

image = os.environ["CLAW_EVAL_SANDBOX_IMAGE"]
for index, arg in enumerate(args):
    if arg == "--sandbox-image":
        if index + 1 >= len(args):
            raise SystemExit("--sandbox-image requires a value")
        image = args[index + 1]
    elif arg.startswith("--sandbox-image="):
        image = arg.split("=", 1)[1]
if not image:
    raise SystemExit("sandbox image must not be empty")
print(image)
PY
}

prepare_sandbox_image() {
  CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE="$(resolve_effective_sandbox_image)"
  export CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE
  if [[ -z "${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}" ]]; then
    echo "[INFO] Claw-Eval sandbox is disabled for this run"
    return
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] --sandbox requires the docker command" >&2
    exit 6
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] --sandbox requires a running Docker daemon" >&2
    exit 6
  fi

  if ! docker image inspect "${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}" >/dev/null 2>&1; then
    if [[ "${CLAW_EVAL_BUILD_SANDBOX_IMAGE}" != "1" ]]; then
      echo "[ERROR] Missing sandbox image ${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}." >&2
      echo "[ERROR] Set CLAW_EVAL_BUILD_SANDBOX_IMAGE=1 to build it explicitly." >&2
      exit 6
    fi
    require_file "${CLAW_EVAL_DIR}/Dockerfile.agent" "Claw-Eval Dockerfile.agent"
    local -a build_args=()
    if [[ -n "${CLAW_EVAL_DOCKER_BUILD_ARGS}" ]]; then
      mapfile -t build_args < <(
        BUILD_ARGS="${CLAW_EVAL_DOCKER_BUILD_ARGS}" \
          "${CLAW_EVAL_CONDA_ENV}/bin/python" - <<'PY'
import os
import shlex

for arg in shlex.split(os.environ["BUILD_ARGS"]):
    print(arg)
PY
      )
    fi
    echo "[INFO] Building sandbox image ${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}"
    docker build \
      --file "${CLAW_EVAL_DIR}/Dockerfile.agent" \
      --tag "${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}" \
      "${build_args[@]}" \
      "${CLAW_EVAL_DIR}" 2>&1 | tee "${LOG_DIR}/sandbox_image_build.log"
  else
    echo "[INFO] Using existing sandbox image ${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}"
  fi

  docker image inspect "${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}" \
    > "${RUN_DIR}/sandbox_image.json"
  CLAW_EVAL_SANDBOX_IMAGE_ID="$(
    docker image inspect "${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}" --format '{{.Id}}'
  )"
  CLAW_EVAL_SANDBOX_IMAGE_SIZE_BYTES="$(
    docker image inspect "${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}" --format '{{.Size}}'
  )"
  export CLAW_EVAL_SANDBOX_IMAGE_ID
  export CLAW_EVAL_SANDBOX_IMAGE_SIZE_BYTES

  echo "[INFO] Starting sandbox preflight container"
  CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE="${CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE}" \
    "${CLAW_EVAL_CONDA_ENV}/bin/python" - <<'PY' \
    2>&1 | tee "${LOG_DIR}/sandbox_preflight.log"
import os

import docker
import httpx

from claw_eval.config import SandboxConfig
from claw_eval.runner.sandbox_runner import SandboxRunner

image = os.environ["CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE"]
runner = SandboxRunner(SandboxConfig(enabled=True, image=image), image=image)
handle = None
run_id = f"preflight-{os.getpid()}"
try:
    handle = runner.start_container(run_id=run_id)
    health = httpx.get(f"{handle.sandbox_url}/health", timeout=5.0)
    health.raise_for_status()
    assert health.json() == {"status": "ok"}, health.text
    result = httpx.post(
        f"{handle.sandbox_url}/exec",
        json={
            "command": "python --version && test ! -e /workspace/grader.py && printf sandbox-ok",
            "timeout_seconds": 10,
        },
        timeout=15.0,
    )
    result.raise_for_status()
    payload = result.json()
    assert payload["exit_code"] == 0, payload
    assert "sandbox-ok" in payload["stdout"], payload
    print(f"sandbox_preflight=passed image={image}")
finally:
    if handle is not None:
        runner.stop_container(handle)
    else:
        try:
            docker.from_env().containers.get(f"claw-agent-{run_id}").remove(force=True)
        except docker.errors.NotFound:
            pass
PY
}

wait_for_server() {
  for _ in $(seq 1 180); do
    if curl -fsS "${SERVER_HEALTH_URL}" >/dev/null 2>&1; then
      return
    fi
    if [[ "${START_SPARSEENGINE_SERVER}" == "1" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[ERROR] Sparse-Engine OpenAI server exited early. See ${LOG_DIR}/sparseengine_openai_server.log" >&2
      exit 4
    fi
    sleep 2
  done
  echo "[ERROR] OpenAI server did not become healthy at ${SERVER_HEALTH_URL}" >&2
  exit 5
}

write_run_manifest() {
  "${CLAW_EVAL_CONDA_ENV}/bin/python" - "${RUN_MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

engine_kwargs_file = Path(os.environ["ENGINE_KWARGS_FILE"])
with engine_kwargs_file.open("r", encoding="utf-8") as f:
    engine_kwargs = json.load(f)
manifest = {
    "run_name": os.environ["RUN_NAME"],
    "repo_root": os.environ["REPO_ROOT"],
    "claw_eval_dir": os.environ["CLAW_EVAL_DIR"],
    "claw_eval_commit": os.environ["CLAW_EVAL_COMMIT"],
    "claw_eval_checkout_clean": True,
    "sparseengine_conda_env": os.environ["SPARSEENGINE_CONDA_ENV"],
    "claw_eval_conda_env": os.environ["CLAW_EVAL_CONDA_ENV"],
    "model_path": os.environ["MODEL_PATH"],
    "served_model_name": os.environ["SERVED_MODEL_NAME"],
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "engine_kwargs_file": str(engine_kwargs_file),
    "engine_kwargs": engine_kwargs,
    "server_base_url": os.environ["SPARSEENGINE_OPENAI_BASE_URL"],
    "server_health_url": os.environ["SERVER_HEALTH_URL"],
    "start_sparseengine_server": os.environ["START_SPARSEENGINE_SERVER"] == "1",
    "judge_base_url": os.environ["CLAW_EVAL_JUDGE_BASE_URL"],
    "judge_model": os.environ["CLAW_EVAL_JUDGE_MODEL"],
    "sparseengine_master_port": os.environ["SPARSEENGINE_MASTER_PORT"],
    "sparseengine_data_parallel_size": int(os.environ["SPARSEENGINE_DATA_PARALLEL_SIZE"]),
    "claw_eval_args": os.environ["CLAW_EVAL_ARGS"],
    "claw_eval_text_only": os.environ["CLAW_EVAL_TEXT_ONLY"] == "1",
    "claw_eval_resume_trace_dir": os.environ["CLAW_EVAL_RESUME_TRACE_DIR"] or None,
    "task_selection_summary": os.environ["CLAW_EVAL_SELECTION_SUMMARY"] or None,
    "skipped_results": os.environ["CLAW_EVAL_SKIPPED_RESULTS"] or None,
    "sandbox_enabled": bool(os.environ["CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE"]),
    "sandbox_image": os.environ["CLAW_EVAL_EFFECTIVE_SANDBOX_IMAGE"] or None,
    "sandbox_image_id": os.environ["CLAW_EVAL_SANDBOX_IMAGE_ID"] or None,
    "sandbox_image_size_bytes": (
        int(os.environ["CLAW_EVAL_SANDBOX_IMAGE_SIZE_BYTES"])
        if os.environ["CLAW_EVAL_SANDBOX_IMAGE_SIZE_BYTES"]
        else None
    ),
    "build_sandbox_image": os.environ["CLAW_EVAL_BUILD_SANDBOX_IMAGE"] == "1",
    "setup_only": os.environ["SETUP_ONLY"],
    "download_http_proxy": os.environ["DOWNLOAD_HTTP_PROXY"],
    "trace_dir": os.environ["TRACE_DIR"],
    "request_log_dir": os.environ["REQUEST_LOG_DIR"],
}
dst = Path(sys.argv[1])
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")
PY
}

claw_eval_judge_enabled() {
  [[ " ${CLAW_EVAL_ARGS} " != *" --no-judge "* ]]
}

main() {
  require_bool "SETUP_ONLY" "${SETUP_ONLY}"
  require_bool "START_SPARSEENGINE_SERVER" "${START_SPARSEENGINE_SERVER}"
  require_bool "CLAW_EVAL_UPDATE_REPO" "${CLAW_EVAL_UPDATE_REPO}"
  require_bool "CLAW_EVAL_BUILD_SANDBOX_IMAGE" "${CLAW_EVAL_BUILD_SANDBOX_IMAGE}"
  require_bool "CLAW_EVAL_TEXT_ONLY" "${CLAW_EVAL_TEXT_ONLY}"
  require_file "${CLAW_EVAL_RESULT_VALIDATOR}" "Claw-Eval result validator"

  source "${HOME}/miniconda3/etc/profile.d/conda.sh"

  prepare_claw_eval_repo
  prepare_claw_eval_env
  render_config
  CLAW_EVAL_ARGS="$(resolve_claw_eval_args)"
  prepare_resume_args
  prepare_task_selection
  export CLAW_EVAL_ARGS
  write_engine_kwargs_file
  prepare_sandbox_image
  write_run_manifest

  if [[ "${SETUP_ONLY}" == "1" ]]; then
    echo "[INFO] SETUP_ONLY=1 finished. Run directory: ${RUN_DIR}"
    exit 0
  fi

  if claw_eval_judge_enabled && [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[ERROR] OPENROUTER_API_KEY is required unless CLAW_EVAL_ARGS includes --no-judge." >&2
    exit 2
  fi

  export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
  export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
  export http_proxy="${DOWNLOAD_HTTP_PROXY}"
  export https_proxy="${DOWNLOAD_HTTPS_PROXY}"
  export HTTP_PROXY="${DOWNLOAD_HTTP_PROXY}"
  export HTTPS_PROXY="${DOWNLOAD_HTTPS_PROXY}"
  export NO_PROXY="127.0.0.1,localhost,::1"
  export no_proxy="127.0.0.1,localhost,::1"

  if [[ "${START_SPARSEENGINE_SERVER}" == "1" ]]; then
    require_file "${MODEL_PATH}" "model path"
    activate_conda_env "${SPARSEENGINE_CONDA_ENV}"
    export CUDA_VISIBLE_DEVICES
    export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
    echo "[INFO] Starting Sparse-Engine OpenAI server on GPUs ${CUDA_VISIBLE_DEVICES}"
    "${SPARSEENGINE_PYTHON_BIN}" -u -m sparseengine.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${SERVER_HOST}" \
      --port "${SERVER_PORT}" \
      --engine-kwargs "${ENGINE_KWARGS_FILE}" \
      --request-log-dir "${REQUEST_LOG_DIR}" \
      >"${LOG_DIR}/sparseengine_openai_server.log" 2>&1 &
    SERVER_PID="$!"
    echo "${SERVER_PID}" >"${RUN_DIR}/sparseengine_openai_server.pid"
    trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT
  else
    echo "[INFO] Using external OpenAI server at ${SPARSEENGINE_OPENAI_BASE_URL}"
  fi
  wait_for_server

  "${CLAW_EVAL_CONDA_ENV}/bin/python" "${CLAW_EVAL_RESULT_VALIDATOR}" snapshot \
    --trace-dir "${TRACE_DIR}" \
    --output "${CLAW_EVAL_PRE_RUN_SNAPSHOT}"

  echo "[INFO] Running claw-eval ${CLAW_EVAL_ARGS}"
  set +e
  (
    activate_conda_env "${CLAW_EVAL_CONDA_ENV}"
    cd "${CLAW_EVAL_DIR}"
    # shellcheck disable=SC2086
    "${CLAW_EVAL_CONDA_ENV}/bin/claw-eval" ${CLAW_EVAL_ARGS}
  ) 2>&1 | tee "${LOG_DIR}/claw_eval.log"
  claw_eval_exit="${PIPESTATUS[0]}"
  set -e

  set +e
  validator_args=(
    validate
    --trace-dir "${TRACE_DIR}" \
    --snapshot "${CLAW_EVAL_PRE_RUN_SNAPSHOT}" \
    --per-sample "${CLAW_EVAL_PER_SAMPLE_RESULTS}" \
    --final-summary "${CLAW_EVAL_FINAL_SUMMARY}"
  )
  if [[ -n "${CLAW_EVAL_SKIPPED_RESULTS}" ]]; then
    validator_args+=(--skipped-results "${CLAW_EVAL_SKIPPED_RESULTS}")
  fi
  "${CLAW_EVAL_CONDA_ENV}/bin/python" "${CLAW_EVAL_RESULT_VALIDATOR}" "${validator_args[@]}" \
    2>&1 | tee "${LOG_DIR}/claw_eval_result_validation.log"
  validation_exit="${PIPESTATUS[0]}"
  set -e

  if [[ "${claw_eval_exit}" != "0" ]]; then
    echo "[ERROR] claw-eval exited with code ${claw_eval_exit}" >&2
    return "${claw_eval_exit}"
  fi
  if [[ "${validation_exit}" != "0" ]]; then
    echo "[ERROR] Claw-Eval result validation failed with code ${validation_exit}" >&2
    return "${validation_exit}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
