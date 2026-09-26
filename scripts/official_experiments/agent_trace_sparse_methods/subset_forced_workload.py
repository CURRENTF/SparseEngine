#!/usr/bin/env python3
"""Reuse validated forced tokens for an ordered prefix of an agent trace."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmark.sparseengine_regression.agent_trace import load_trace


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-trace", type=Path, required=True)
    parser.add_argument("--subset-trace", type=Path, required=True)
    parser.add_argument("--parent-workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    parent = load_trace(args.parent_trace)
    subset = load_trace(args.subset_trace)
    parent_hash = sha256(args.parent_trace / "manifest.json")
    subset_hash = sha256(args.subset_trace / "manifest.json")
    workload = json.loads(args.parent_workload.read_text())
    if subset.get("subset_parent_manifest_sha256") != parent_hash:
        raise ValueError("Subset does not identify this parent manifest")
    if subset["agents"] != parent["agents"][: subset["instance_count"]]:
        raise ValueError("Subset is not an ordered prefix of complete parent agents")
    if (workload["trace_sha256"] != parent_hash
            or workload["instance_count"] != parent["instance_count"]
            or workload["request_count"] != parent["request_count"]
            or len(workload["agents"]) != parent["instance_count"]):
        raise ValueError("Parent forced workload does not match the parent trace")

    selected = {}
    digest = hashlib.sha256()
    parent_digest = hashlib.sha256()
    expected_tokens = 0
    for index, entry in enumerate(parent["agents"]):
        prepared = workload["agents"][entry["file"]]
        turns = json.loads((args.parent_trace / entry["file"]).read_text())["turns"]
        if len(prepared) != len(turns):
            raise ValueError(f"Prepared turn count mismatch: {entry['file']}")
        for turn, spec in zip(turns, prepared):
            ids = spec["token_ids"]
            count = len(ids) if ids is not None else turn["completion_tokens"]
            item = json.dumps(
                [entry["instance_id"], turn["turn"], ids, count],
                separators=(",", ":"),
            ).encode()
            parent_digest.update(item)
            if index < subset["instance_count"]:
                expected_tokens += count
                digest.update(item)
        if index < subset["instance_count"]:
            selected[entry["file"]] = prepared
    if parent_digest.hexdigest() != workload["forced_workload_sha256"]:
        raise ValueError("Parent forced-workload contents fail their recorded digest")

    result = {
        "schema": "agent_forced_workload_v1",
        "trace_sha256": subset_hash,
        "model_path": workload["model_path"],
        "forced_workload_sha256": digest.hexdigest(),
        "instance_count": subset["instance_count"],
        "request_count": subset["request_count"],
        "expected_completion_tokens": expected_tokens,
        "agents": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "trace_sha256", "forced_workload_sha256", "instance_count",
        "request_count", "expected_completion_tokens",
    )}))


if __name__ == "__main__":
    main()
