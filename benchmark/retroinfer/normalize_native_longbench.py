"""Align native LongBench raw outputs with the author's frozen-token score input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("v1", "v2"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared = rows(args.input)
    native = rows(args.native_raw)
    if len(prepared) != len(native):
        raise ValueError(f"Coverage mismatch: prepared={len(prepared)} native={len(native)}")
    by_id = {}
    for row in native:
        if args.benchmark == "v1":
            sample_id = f"{row['dataset']}:{row['source_idx']}"
            raw = row["raw_pred"]
        else:
            sample_id = row["_id"]
            raw = row["raw_response"]
        if sample_id in by_id:
            raise ValueError(f"Duplicate native sample {sample_id}")
        if not isinstance(raw, str):
            raise ValueError(f"Non-string output for {sample_id}")
        status = row["status"]
        if args.benchmark == "v2" and status == "parse_failed" and raw:
            # Let the shared scorer parse a successfully generated, malformed
            # answer in exactly the same way it parses the author output.
            status = "success"
        by_id[sample_id] = {"sample_id": sample_id, "status": status,
                            "raw_pred": raw, "native_status": row["status"],
                            "prompt_tokens": row["prompt_tokens"]}
    if {sample["sample_id"] for sample in prepared} != set(by_id):
        raise ValueError("Prepared and native sample IDs differ")
    for sample in prepared:
        row = by_id[sample["sample_id"]]
        if row["prompt_tokens"] != len(sample["prompt_token_ids"]):
            raise ValueError(f"Prompt token count differs for {sample['sample_id']}")
    with args.output.open("x", encoding="utf-8") as handle:
        for sample in prepared:
            row = by_id[sample["sample_id"]]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Aligned {len(prepared)} native outputs")


if __name__ == "__main__":
    main()
