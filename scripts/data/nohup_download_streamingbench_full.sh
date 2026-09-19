#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${SPARSEENGINE_DATA_DIR:-${REPO_ROOT}/data}"
LOG_DIR="${STREAMINGBENCH_LOG_DIR:-${DATA_ROOT}/logs}"

LOG_PATH="${STREAMINGBENCH_DOWNLOAD_LOG:-${LOG_DIR}/streamingbench_full_download.nohup.log}"
PID_PATH="${STREAMINGBENCH_DOWNLOAD_PID:-${LOG_DIR}/streamingbench_full_download.pid}"

mkdir -p "$(dirname "${LOG_PATH}")"

setsid nohup bash "${SCRIPT_DIR}/download_streamingbench_full.sh" > "${LOG_PATH}" 2>&1 < /dev/null &
pid="$!"
echo "${pid}" > "${PID_PATH}"

echo "${pid}"
echo "[info] log=${LOG_PATH}"
echo "[info] pid_file=${PID_PATH}"
