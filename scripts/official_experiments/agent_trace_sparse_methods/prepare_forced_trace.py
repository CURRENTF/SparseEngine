#!/usr/bin/env python3
"""Validate and prepare exact model-token continuations for the agent trace."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from benchmark.sparseengine_regression.agent_trace import (
    digest, load_trace, prepare_forced_agent, read, write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_trace(args.trace)
    model = args.model.resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    vocabulary = int(read(model / "config.json")["vocab_size"])
    agents = {}
    workload_digest = hashlib.sha256()
    expected_tokens = 0
    for entry in manifest["agents"]:
        agent = read(args.trace / entry["file"])
        prepared = prepare_forced_agent(agent, tokenizer, "forced-trace")
        for turn, spec in zip(agent["turns"], prepared):
            ids = spec["token_ids"]
            if ids is not None and any(token < 0 or token >= vocabulary for token in ids):
                raise ValueError(f"Forced token exceeds model vocabulary: {entry['instance_id']} turn {turn['turn']}")
            count = len(ids) if ids is not None else turn["completion_tokens"]
            expected_tokens += count
            workload_digest.update(json.dumps([entry["instance_id"], turn["turn"], ids, count],
                                              separators=(",", ":")).encode())
        agents[entry["file"]] = prepared
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, {
        "schema": "agent_forced_workload_v1",
        "trace_sha256": digest(args.trace / "manifest.json"),
        "model_path": str(model),
        "forced_workload_sha256": workload_digest.hexdigest(),
        "instance_count": manifest["instance_count"],
        "request_count": manifest["request_count"],
        "expected_completion_tokens": expected_tokens,
        "agents": agents,
    })
    print(json.dumps({"output": str(args.output), "instance_count": manifest["instance_count"],
                      "request_count": manifest["request_count"],
                      "expected_completion_tokens": expected_tokens}))


if __name__ == "__main__":
    main()
