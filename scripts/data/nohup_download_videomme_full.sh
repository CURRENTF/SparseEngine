#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${SPARSEENGINE_DATA_DIR:-${REPO_ROOT}/data}"

ROOT="${VIDEOMME_ROOT:-${DATA_ROOT}/Video-MME_hf}"
LOG_DIR="${VIDEOMME_LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="${LOG_DIR}/download_videomme_full_${timestamp}.log"

nohup bash "${SCRIPT_DIR}/download_videomme_full.sh" >"${log_path}" 2>&1 &
pid="$!"

echo "${pid}" >"${LOG_DIR}/download_videomme_full.pid"
echo "[info] pid=${pid}"
echo "[info] log=${log_path}"
