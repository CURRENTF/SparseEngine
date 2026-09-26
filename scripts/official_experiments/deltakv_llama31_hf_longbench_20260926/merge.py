#!/usr/bin/env python3
"""Restore frozen input order and verify each shard emitted one success row."""

import argparse
import json
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("v1", "v2"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()

    indexed = {}
    for gpu in range(args.workers):
        inputs = rows(args.input_root / f"{args.benchmark}-shard-{gpu}.jsonl")
        outputs = rows(args.run_root / args.benchmark / f"shard-{gpu}" / "raw_outputs.jsonl")
        if len(inputs) != len(outputs):
            raise ValueError(f"shard {gpu} output count {len(outputs)} != input {len(inputs)}")
        for sample, output in zip(inputs, outputs, strict=True):
            index = sample["eval_index"]
            if output["sample_id"] != sample["sample_id"] or output["status"] != "success":
                raise ValueError(f"shard {gpu} sample {index} failed or mismatched")
            if index in indexed:
                raise ValueError(f"duplicate eval_index {index}")
            indexed[index] = (sample, output)
    expected = 3750 if args.benchmark == "v1" else 503
    if set(indexed) != set(range(expected)):
        raise ValueError(f"incomplete {args.benchmark} coverage: {len(indexed)}/{expected}")
    input_path = args.run_root / args.benchmark / "merged_inputs.jsonl"
    output_path = args.run_root / args.benchmark / "raw_outputs.jsonl"
    with input_path.open("x", encoding="utf-8") as input_file, output_path.open("x", encoding="utf-8") as output_file:
        for index in range(expected):
            sample, output = indexed[index]
            input_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            output_file.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(json.dumps({"benchmark": args.benchmark, "samples": expected,
                      "merged_inputs": str(input_path), "raw_outputs": str(output_path)}))


if __name__ == "__main__":
    main()
