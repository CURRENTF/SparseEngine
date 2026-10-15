#!/usr/bin/env python3
"""Compare complete recorded three-turn traces, greedy tokens and all logits."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = []
    for root in (args.left, args.right):
        summary = json.loads((root / "summary.json").read_text())
        if summary["status"] != "success" or "profile" in summary:
            raise ValueError("Require completed model runs with captured logits.")
        data.append((summary, json.loads((root / "trace.json").read_text()),
                     torch.load(root / "logits.pt", weights_only=True)))
    tolerance = data[0][0]["logits_tolerance"]
    assert tolerance == data[1][0]["logits_tolerance"]
    assert len(data[0][1]) == len(data[0][2]) == len(data[1][1]) == len(data[1][2])
    samples, errors = [], []
    for i, (left, right, a, b) in enumerate(zip(data[0][1], data[1][1], data[0][2], data[1][2])):
        if left["round"] >= 3:
            continue
        assert {k: v for k, v in left.items() if k != "sampled"} == {
            k: v for k, v in right.items() if k != "sampled"}, f"Trace mismatch at step {i}"
        error = (a.float() - b.float()).abs()
        errors.append(error.flatten())
        passed = bool(torch.isfinite(error).all() and torch.allclose(a, b, **tolerance))
        mismatches = sum(x != y for x, y in zip(left["sampled"], right["sampled"]))
        samples.append(dict(step=i, round=left["round"], status="success" if passed else "metric_failed",
                            max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                            logit_values=a.numel(), greedy_mismatches=mismatches))
    assert {r["round"] for r in samples} == {0, 1, 2}, "Require all three chain rounds"
    status = "success" if all(r["status"] == "success" and r["greedy_mismatches"] == 0
                              for r in samples) else "metric_failed"
    pooled_error = torch.cat(errors)
    quantiles = torch.quantile(pooled_error, torch.tensor([0.95, 0.99])).tolist()
    result = dict(left=str(args.left), right=str(args.right), status=status, tolerance=tolerance,
                  max_abs_error=max(r["max_abs_error"] for r in samples),
                  p95_abs_error=quantiles[0], p99_abs_error=quantiles[1],
                  logit_values=sum(r["logit_values"] for r in samples),
                  greedy_mismatches=sum(r["greedy_mismatches"] for r in samples), samples=samples)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}))
    if status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
