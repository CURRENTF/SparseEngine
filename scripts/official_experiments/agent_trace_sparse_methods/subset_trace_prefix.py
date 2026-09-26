#!/usr/bin/env python3
"""Select the first N complete agent trajectories from a validated trace."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    source_manifest = args.source / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    entries = manifest["agents"]
    if not 0 < args.count <= len(entries):
        parser.error(f"--count must be in [1, {len(entries)}]")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if len(entries) != manifest["instance_count"] or sum(a["turns"] for a in entries) != manifest["request_count"]:
        raise ValueError("Source manifest counts disagree with its agent entries")

    selected = entries[: args.count]
    if len({a["instance_id"] for a in selected}) != len(selected):
        raise ValueError("Selected agent IDs are not unique")
    for entry in selected:
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError(f"Unsafe agent filename: {name}")
        source = args.source / name
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"Agent hash mismatch: {name}")

    args.output.mkdir(parents=True)
    for entry in selected:
        name = entry["file"]
        source = args.source / name
        os.link(source, args.output / name)

    subset = dict(manifest)
    subset["agents"] = selected
    subset["instance_count"] = args.count
    subset["request_count"] = sum(entry["turns"] for entry in selected)
    subset["subset_rule"] = f"first_{args.count}_agents_in_parent_manifest_order_all_turns"
    subset["subset_parent_manifest_sha256"] = sha256(source_manifest)
    (args.output / "manifest.json").write_text(json.dumps(subset, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(args.output / "manifest.json"),
        "manifest_sha256": sha256(args.output / "manifest.json"),
        "instance_count": subset["instance_count"],
        "request_count": subset["request_count"],
        "parent_manifest_sha256": subset["subset_parent_manifest_sha256"],
    }))


if __name__ == "__main__":
    main()
