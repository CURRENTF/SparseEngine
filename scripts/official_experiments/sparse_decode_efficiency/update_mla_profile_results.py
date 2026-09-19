"""Prepare an MLA profile rerun, then replace its one affected curve."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from plot_decode_capacity import validate_measurement, without_source_fingerprints

CHANGED = {"src/sparseengine/kernels/tilelang/mla/runtime.py",
           "src/sparseengine/operators/mla_attention.py"}
TARGET = ("glm4.7-flash", "sengine-omnikv")
RULE = "sm_parallel_nearest_v1"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_bindings(value):
    if isinstance(value, dict):
        if "selected_provider" in value:
            yield value
        for item in value.values():
            yield from selected_bindings(item)
    elif isinstance(value, list):
        for item in value:
            yield from selected_bindings(item)


def prepare(args):
    base = read(args.base_plot)
    if RULE not in (args.repo / "src/sparseengine/kernels/tilelang/mla/runtime.py").read_text():
        raise ValueError("Selected source does not contain the requested rule")
    if not args.scratch_root.is_absolute() or len(str(args.scratch_root)) > 65:
        raise ValueError("Use an absolute short scratch path (<=65 characters) for multiprocessing Unix sockets")
    args.run_root.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(base["config"])
    config["output_root"] = str(args.run_root.resolve())
    config["scratch_root"] = str(args.scratch_root)
    write(args.run_root / "config.json", config)
    write(args.run_root / "base_plot.json", base)
    write(args.run_root / "manifest.json", {
        "status": "prepared", "rule": RULE,
        "formula": "nearest by absolute distance to SM_count / (batch * head_tiles)",
        "alpha": 1, "candidates": [4, 8, 16, 32], "tie_break": "lower",
        "context_in_split_selection": False, "target": list(TARGET),
        "repo": str(args.repo.resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True).strip()),
        "profile_files": sorted(CHANGED),
        "base_plot_path": str(args.base_plot.resolve()), "base_plot_sha256": sha(args.base_plot),
        "config_sha256": sha(args.run_root / "config.json"),
        "quality_scope": "synthetic throughput; not a generation-quality evaluation",
    })
    print(f"Prepared {args.run_root}; source equality is not checked", flush=True)


def export(args):
    root = args.run_root
    manifest = read(root / "manifest.json")
    if sha(root / "base_plot.json") != manifest["base_plot_sha256"]:
        raise ValueError("Baseline plot data changed after preparation")
    if sha(root / "config.json") != manifest["config_sha256"]:
        raise ValueError("Rerun configuration changed after preparation")
    base, config = without_source_fingerprints(read(root / "base_plot.json")), read(root / "config.json")
    curves = copy.deepcopy(base["curves"])
    curve = next(c for c in curves if (c["model"], c["lane"]) == TARGET)
    paths = list((root / TARGET[0]).glob(f"*/{TARGET[1]}/capacity.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one capacity sweep, found {paths}")
    boundary = read(paths[0])
    if boundary["status"] != "completed":
        raise ValueError("Capacity sweep is incomplete")
    reference = Path(curve["points"][0]["artifact"]).parent
    old_identity, old_hp = read(reference / "identity.json"), read(reference / "hyper_params.json")
    plans = []
    for attempt in boundary["attempts"]:
        case = Path(attempt["artifact"]).parent
        identity = read(case / "identity.json")
        if identity["model_config_sha256"] != old_identity["model_config_sha256"]:
            raise ValueError(f"Model configuration changed: {case}")
        expected = {**old_hp, "decode_graph_capture_sizes": [attempt["concurrency"]]}
        if read(case / "hyper_params.json") != expected:
            raise ValueError(f"Unexpected hyperparameter change: {case}")
        if attempt["status"] == "success":
            rows = [json.loads(line) for line in (case / "performance.jsonl").read_text().splitlines()]
            if len(rows) != 1 or not rows[0]["decode_graph_active"] or rows[0]["decode_graph_force_eager_count"]:
                raise ValueError(f"Missing graph-only decode evidence: {case}")
            bindings = [b for b in selected_bindings(read(case / "aggregate_metrics.json"))
                        if b["selected_provider"] == "tilelang_score"]
            tp = config["models"][TARGET[0]]["tp"]
            if len(bindings) != tp or {b["device_caps"]["device_index"] for b in bindings} != set(range(tp)):
                raise ValueError(f"Missing per-rank TileLang binding evidence: {case}")
            for binding in bindings:
                plan = binding["provider_metadata"]["tilelang_launch_plan"]
                if plan["split_rule"] != RULE:
                    raise ValueError(f"Wrong bound split rule: {case}")
                for item in plan["batch_configs"]:
                    padded_heads = 32 if plan["local_q_heads"] == 20 else 16
                    parallelism = item["batch_size"] * (padded_heads // item["block_h"])
                    expected_split = min(manifest["candidates"],
                                         key=lambda s: abs(s * parallelism - plan["sm_count"]))
                    if item["num_split"] != expected_split:
                        raise ValueError(f"Bound profile does not follow the recorded rule: {case}")
                plans.append({"concurrency": attempt["concurrency"],
                              "device_index": binding["device_caps"]["device_index"], "plan": plan})
    curve.update({k: boundary[k] for k in ("max_concurrency", "first_failed_concurrency", "attempts")})
    curve["capacity_artifact"] = str(paths[0].resolve())
    curve["points"] = [validate_measurement(Path(a["artifact"]), a["concurrency"], config)
                       for a in sorted(boundary["attempts"], key=lambda a: a["concurrency"])
                       if a["status"] == "success"]
    for other in curves:
        if (other["model"], other["lane"]) == TARGET:
            continue
        for point in other["points"]:
            case = Path(point["artifact"]).parent
            if any(b["selected_provider"] == "tilelang_score"
                   for b in selected_bindings(read(case / "aggregate_metrics.json"))):
                raise ValueError(f"Another curve uses the changed provider and must be rerun: {case}")
            if validate_measurement(Path(point["artifact"]), point["concurrency"], config) != point:
                raise ValueError(f"Baseline raw data changed: {case}")
    output = root / "export"
    output.mkdir(exist_ok=False)
    write(output / "plot_data.input.json", {**base, "config": config, "curves": curves})
    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_decode_capacity.py")),
                    "--plot-data", str(output / "plot_data.input.json"), "--output-dir", str(output)], check=True)
    write(output / "generated_profiles.json", plans)
    portable = read(output / "plot_data.json")
    portable["config"] = read(Path(__file__).with_name("config.omnikv-total2048.json"))
    old_roots = {Path(base["config"]["output_root"]).resolve(): "omnikv-total2048"}
    # Both historical and current exports use root/model/attempt/lane/capacity.json.
    # The historical runner saved config.json at the root, not per-launch campaign.json.
    for other in base["curves"]:
        capacity_path = Path(other["capacity_artifact"]).resolve()
        if capacity_path.parent.name != other["lane"] or capacity_path.parents[2].name != other["model"]:
            raise ValueError(f"Unexpected capacity artifact layout: {capacity_path}")
        launch_config = read(capacity_path.parents[3] / "config.json")
        campaign_root = Path(launch_config["output_root"]).resolve()
        if campaign_root != capacity_path.parents[3]:
            raise ValueError(f"Campaign root does not match its recorded config: {capacity_path}")
        if campaign_root not in old_roots:
            old_roots[campaign_root] = "original"
    roots = {**old_roots, root.resolve(): "sm-parallel"}
    if len(set(roots.values())) != len(roots):
        raise ValueError("Ambiguous original campaign roots")

    def artifact_id(value):
        path = Path(value).resolve()
        for campaign_root, prefix in roots.items():
            if path.is_relative_to(campaign_root):
                return str(Path(prefix) / path.relative_to(campaign_root))
        raise ValueError(f"Artifact is outside declared campaigns: {path}")

    for item in portable["curves"]:
        item["capacity_artifact"] = artifact_id(item["capacity_artifact"])
        for point in item["attempts"] + item["points"]:
            point["artifact"] = artifact_id(point["artifact"])
    portable_dir = output / "portable"
    portable_dir.mkdir()
    write(portable_dir / "plot_data.json", portable)
    write(portable_dir / "generated_profiles.json", plans)
    write(portable_dir / "provenance.json", {
        "rule": manifest["rule"], "formula": manifest["formula"],
        "alpha": manifest["alpha"], "candidates": manifest["candidates"],
        "tie_break": manifest["tie_break"], "context_in_split_selection": False,
        "base_git_commit": manifest["git_commit"],
        "git_dirty": manifest.get("git_dirty"),
        "profile_files": sorted(CHANGED),
        "reused_curves": 9, "rerun_target": list(TARGET),
        "artifact_prefixes": sorted(roots.values()),
        "metric": portable["metric"], "quality_scope": manifest["quality_scope"],
    })
    with (portable_dir / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "lane", "concurrency",
            "decode_throughput_tps", "decode_tokens", "decode_elapsed_s", "measured_steps", "artifact"),
            lineterminator="\n")
        writer.writeheader()
        for item in portable["curves"]:
            for point in item["points"]:
                writer.writerow({"model": item["model"], "lane": item["lane"], **point})
    manifest.update(status="completed", accepted_points=sum(len(c["points"]) for c in curves),
                    reused_curves=len(curves)-1, rerun_points=len(curve["points"]),
                    output=str(output))
    write(root / "manifest.json", manifest)
    print(f"Validated all {manifest['accepted_points']} points; exported {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "export"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-plot", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.repo is None or args.base_plot is None or args.scratch_root is None:
            parser.error("prepare requires --repo, --base-plot and --scratch-root")
        prepare(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
