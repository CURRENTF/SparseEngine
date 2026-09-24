#!/usr/bin/env bash
set -euo pipefail

case_root=${1:?completed case root}
method=${2:?method}
if [[ ! -f "$case_root/status.tsv" ]]; then
  echo "Missing case status: $case_root" >&2
  exit 1
fi

for phase in smoke full; do
  phase_root="$case_root/run/$method/$phase"
  directory="$phase_root/server_requests"
  if [[ ! -d "$directory" ]]; then
    continue
  fi
  archive="$phase_root/server_requests.tar"
  temporary="$archive.tmp"
  if [[ -e "$archive" || -e "$temporary" ]]; then
    echo "Request archive already exists: $archive" >&2
    exit 1
  fi
  count=$(find "$directory" -maxdepth 1 -type f -name '*.json' | wc -l)
  if (( count == 0 )); then
    echo "Empty request log directory: $directory" >&2
    exit 1
  fi
  tar -C "$phase_root" -cf "$temporary" server_requests
  members=$(tar -tf "$temporary" | awk -F/ 'NF == 2 && $2 ~ /\.json$/ { count++ } END { print count+0 }')
  if (( members != count )); then
    echo "Request archive count mismatch: $members != $count" >&2
    exit 1
  fi
  tar -df "$temporary" -C "$phase_root"
  mv "$temporary" "$archive"
  rm -r -- "$directory"
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" \
    "$phase" "$count" "$(sha256sum "$archive" | cut -d' ' -f1)" \
    >> "$case_root/request_log_archives.tsv"
done
