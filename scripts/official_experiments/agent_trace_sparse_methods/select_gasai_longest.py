#!/usr/bin/env python3
"""Freeze the Gasai trajectories with the most assistant request turns."""

import argparse
import hashlib
import json
from pathlib import Path


ASSISTANT = "<|assistant|>"
TOOL_RESULT = "<|tool_result|>"


def select(source: Path, download_manifest: Path, output: Path, count: int) -> None:
    download = json.loads(download_manifest.read_text(encoding="utf-8"))
    if count <= 0:
        raise ValueError("Selection count must be positive")

    candidates = []
    source_digest = hashlib.sha256()
    with source.open("rb") as stream:
        for row, raw in enumerate(stream):
            source_digest.update(raw)
            record = json.loads(raw)
            serialized = record["gasai"]
            turns = serialized.count(ASSISTANT)
            tool_results = serialized.count(TOOL_RESULT)
            if (not serialized.startswith("<|bos|><|system|>")
                    or not serialized.endswith("<|eos|>")
                    or turns < 1 or tool_results != turns - 1):
                raise ValueError(f"Invalid Gasai trajectory structure at source row {row}")
            candidates.append({"source_row_zero_based": row,
                               "trajectory_id": record["trajectory_id"],
                               "assistant_turns": turns,
                               "tool_results": tool_results,
                               "chars": len(serialized)})

    if not candidates or count > len(candidates):
        raise ValueError("Selection count exceeds the number of source rows")
    if len({item["trajectory_id"] for item in candidates}) != len(candidates):
        raise ValueError("Source contains duplicate trajectory IDs")
    if source_digest.hexdigest() != download["train_sha256"]:
        raise ValueError("Source JSONL hash differs from download manifest")
    if source.stat().st_size != download["files"][source.name]:
        raise ValueError("Source JSONL byte size differs from download manifest")

    candidates.sort(key=lambda item: (-item["assistant_turns"], item["source_row_zero_based"]))
    selected = candidates[:count]
    for draw_index, item in enumerate(selected):
        item["draw_index"] = draw_index
    result = {"source_repo": download["repo"],
              "source_revision": download["revision"],
              "source_file": source.name,
              "source_sha256": source_digest.hexdigest(),
              "sampling": "longest_assistant_turns_source_order_tiebreak",
              "seed": None,
              "source_rows": len(candidates),
              "selected_trajectories": count,
              "assistant_requests": sum(item["assistant_turns"] for item in selected),
              "items": selected}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "source_rows": len(candidates),
                      "selected_trajectories": count,
                      "assistant_requests": result["assistant_requests"],
                      "minimum_assistant_turns": selected[-1]["assistant_turns"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    select(args.source_jsonl, args.download_manifest, args.output, args.count)


if __name__ == "__main__":
    main()
