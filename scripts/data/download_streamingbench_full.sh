#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${SPARSEENGINE_DATA_DIR:-${REPO_ROOT}/data}"

ROOT="${STREAMINGBENCH_ROOT:-${DATA_ROOT}/StreamingBench_hf}"
CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-${REPO_ROOT}/.hf_cache}}"
HF_BIN="${HF_BIN:-hf}"
REPO_ID="${STREAMINGBENCH_REPO_ID:-mjuicem/StreamingBench}"
PROXY_URL="${PROXY_URL:-http://localhost:7890}"
MAX_WORKERS="${HF_MAX_WORKERS:-1}"
DOWNLOAD_SCOPE="${STREAMINGBENCH_DOWNLOAD_SCOPE:-full}"

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  echo "[error] hf command not found: ${HF_BIN}" >&2
  exit 1
fi

mkdir -p "${ROOT}" "${CACHE_DIR}" "${ROOT}/videos"

export HF_HOME="${CACHE_DIR}"
export HF_HUB_CACHE="${CACHE_DIR}/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"

echo "[info] start_time=$(date -Is)"
echo "[info] repo=${REPO_ID}"
echo "[info] root=${ROOT}"
echo "[info] cache=${CACHE_DIR}"
echo "[info] proxy=${PROXY_URL}"
echo "[info] max_workers=${MAX_WORKERS}"
echo "[info] download_scope=${DOWNLOAD_SCOPE}"

declare -a DOWNLOAD_FILES=()
if [[ "${DOWNLOAD_SCOPE}" == "table4" ]]; then
  DOWNLOAD_FILES=(
    "StreamingBench/Real_Time_Visual_Understanding.csv"
    "StreamingBench/Omni_Source_Understanding.csv"
    "StreamingBench/Contextual_Understanding.csv"
    "Real-Time Visual Understanding_1-50.zip"
    "Real-Time Visual Understanding_51-100.zip"
    "Real-Time Visual Understanding_101-150.zip"
    "Real-Time Visual Understanding_151-200.zip"
    "Real-Time Visual Understanding_201-250.zip"
    "Real-Time Visual Understanding_251-300.zip"
    "Real-Time Visual Understanding_301-350.zip"
    "Real-Time Visual Understanding_351-400.zip"
    "Real-Time Visual Understanding_401-450.zip"
    "Real-Time Visual Understanding_451-500.zip"
    "Emotion Recognition.zip"
    "Multimodal Alignment.zip"
    "Scene Understanding_1-25.zip"
    "Scene Understanding_26-50.zip"
    "Source Discrimination.zip"
    "Anomaly Context Understanding.zip"
    "Misleading Context Understanding.zip"
  )
elif [[ "${DOWNLOAD_SCOPE}" != "full" ]]; then
  echo "[error] unsupported STREAMINGBENCH_DOWNLOAD_SCOPE=${DOWNLOAD_SCOPE}; expected full or table4" >&2
  exit 1
fi

set +e
"${HF_BIN}" download "${REPO_ID}" \
  "${DOWNLOAD_FILES[@]}" \
  --repo-type dataset \
  --local-dir "${ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --max-workers "${MAX_WORKERS}"
download_rc="$?"
set -e
echo "[info] hf_download_rc=${download_rc}"
if [[ "${download_rc}" -ne 0 ]]; then
  exit "${download_rc}"
fi

echo "[info] download_done=$(date -Is)"
echo "[info] unzip downloaded shards"

while IFS= read -r -d '' zip_path; do
  base="$(basename "${zip_path}" .zip)"
  dest="${ROOT}/videos/${base}"
  mkdir -p "${dest}"
  echo "[info] unzip ${zip_path} -> ${dest}"
  unzip -n "${zip_path}" -d "${dest}"
done < <(find "${ROOT}" -type f -name '*.zip' ! -path "${ROOT}/videos/*" -print0 | sort -z)

echo "[info] unzip_done=$(date -Is)"
echo "[info] zip_count=$(find "${ROOT}" -type f -name '*.zip' ! -path "${ROOT}/videos/*" | wc -l)"
echo "[info] video_count=$(find "${ROOT}/videos" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.avi' -o -iname '*.mov' \) | wc -l)"
echo "[info] done=$(date -Is)"
