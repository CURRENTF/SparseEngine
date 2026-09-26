"""Validate and summarize paired Llama 3.1 LongBench v1/v2 quality runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    result = {"model": "Llama-3.1-8B-Instruct", "author_commit": None,
              "benchmarks": {}}
    lines = ["# RetroInfer GPU-only Llama 3.1 LongBench comparison", "",
             "Matched frozen token inputs and pinned external scorers; greedy BF16.", ""]
    for benchmark, expected, metric in (("v1", 3750, "overall_category_avg"),
                                        ("v2", 503, "accuracy")):
        prepared = root / f"{benchmark}_prepared" / "paired_samples.jsonl"
        native = read(root / f"native_{benchmark}_full" / "paired_scored" / "aggregate_metrics.json")
        author = read(root / f"author_{benchmark}_full" / "scored" / "aggregate_metrics.json")
        author_run = read(root / f"author_{benchmark}_full" / "resolved_config.json")
        author_status = read(root / f"author_{benchmark}_full" / "run_status.json")
        native_status = read(root / f"native_{benchmark}_full" / "run_status.json")
        if native["status"] != "success" or author["status"] != "success":
            raise ValueError(f"{benchmark} scorer did not finish successfully")
        if native["samples"] != expected or author["samples"] != expected:
            raise ValueError(f"{benchmark} sample coverage mismatch")
        if native["failed_samples"] or author["failed_samples"]:
            raise ValueError(f"{benchmark} has execution failures")
        if native["scorer_script_sha256"] != author["scorer_script_sha256"]:
            raise ValueError(f"{benchmark} used different scorer scripts")
        if author_run["input_sha256"] != sha256(prepared):
            raise ValueError(f"{benchmark} author input identity changed")
        if author_status["status"] != "completed" or native_status["status"] not in {"completed", "success"}:
            raise ValueError(f"{benchmark} run status incomplete")
        if result["author_commit"] is None:
            result["author_commit"] = author_run["author_commit"]
        elif result["author_commit"] != author_run["author_commit"]:
            raise ValueError("Author commit differs between v1 and v2")
        native_score = native[metric]
        author_score = author[metric]
        result["benchmarks"][benchmark] = {
            "samples": expected,
            "metric": metric,
            "native": native,
            "author": author,
            "native_score": native_score,
            "author_score": author_score,
            "native_minus_author": round(native_score - author_score, 4),
            "scorer_script_sha256": native["scorer_script_sha256"],
            "input_sha256": author_run["input_sha256"],
            "author_batch_sizes": author_run["batch_sizes"],
        }
        lines.extend([
            f"## LongBench {benchmark}", "",
            f"{expected} samples; metric: {metric}.", "",
            "| Native GPU-only | Author GPU-only | Native − author |",
            "| ---: | ---: | ---: |",
            f"| {native_score:.2f} | {author_score:.2f} | {native_score - author_score:+.2f} |",
            "",
        ])
        if benchmark == "v1":
            lines.extend(["| Category | Native | Author |", "| --- | ---: | ---: |"])
            for category in native["category_scores"]:
                lines.append(f"| {category} | {native['category_scores'][category]:.2f} | "
                             f"{author['category_scores'][category]:.2f} |")
            lines.append("")
        else:
            lines.extend(["| Length | Native | Author |", "| --- | ---: | ---: |"])
            for length in native["by_official_length"]:
                left = native["by_official_length"][length]["accuracy"]
                right = author["by_official_length"][length]["accuracy"]
                lines.append(f"| {length} | {left:.2f} | {right:.2f} |")
            lines.append("")
    (root / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(root / "RESULTS.md")


if __name__ == "__main__":
    main()
