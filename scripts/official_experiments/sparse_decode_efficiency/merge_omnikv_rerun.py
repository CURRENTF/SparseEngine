"""Replace only OmniKV with a matched total-2048 rerun and revalidate raw data."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

from plot_decode_capacity import LANES, validate_measurement, without_source_fingerprints


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--rerun-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    original = without_source_fingerprints(read(args.original_root / "plots" / "plot_data.json"))
    config = read(args.rerun_root / "config.json")
    expected_config = copy.deepcopy(original["config"])
    expected_config["methods"]["omnikv"]["decode_keep_tokens"] = 1472
    for key in ("output_root", "scratch_root"):
        expected_config[key] = config[key]
    if config != expected_config:
        raise ValueError("Rerun config changes more than OmniKV budget and artifact roots")
    if sum(config["methods"]["omnikv"][key] for key in
           ("sink_keep_tokens", "recent_keep_tokens", "decode_keep_tokens")) != 2048:
        raise ValueError("OmniKV total budget must be 2048")
    curves = copy.deepcopy(original["curves"])
    if {(c["model"], c["lane"]) for c in curves} != {
            (model, lane) for model in config["models"] for lane in LANES}:
        raise ValueError("Original campaign is missing curves")
    for curve in curves:
        if curve["lane"] == "sengine-omnikv":
            candidates = list((args.rerun_root / curve["model"]).glob(
                "*/sengine-omnikv/capacity.json"))
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous or missing rerun for {curve['model']}")
            boundary = read(candidates[0])
            if boundary["status"] != "completed":
                raise ValueError(f"Rerun is not complete: {candidates[0]}")
            old_case = Path(curve["points"][0]["artifact"]).parent
            old_identity = read(old_case / "identity.json")
            expected_hp = read(old_case / "hyper_params.json")
            expected_hp["decode_keep_tokens"] = 1472
            for attempt in boundary["attempts"]:
                case = Path(attempt["artifact"]).parent
                expected_hp["decode_graph_capture_sizes"] = [attempt["concurrency"]]
                if read(case / "hyper_params.json") != expected_hp:
                    raise ValueError(f"Unexpected hyperparameter changes: {case}")
                identity = read(case / "identity.json")
                if identity["model_config_sha256"] != old_identity["model_config_sha256"]:
                    raise ValueError(f"Model config changed: {case}")
            curve.update({key: boundary[key] for key in
                          ("max_concurrency", "first_failed_concurrency", "attempts")})
            curve["capacity_artifact"] = str(candidates[0].resolve())
            curve["points"] = [validate_measurement(
                Path(attempt["artifact"]), attempt["concurrency"], config)
                for attempt in sorted(boundary["attempts"], key=lambda item: item["concurrency"])
                if attempt["status"] == "success"]
        else:
            for point in curve["points"]:
                checked = validate_measurement(Path(point["artifact"]), point["concurrency"], config)
                if checked != point:
                    raise ValueError(f"Original raw data changed: {point['artifact']}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original["config"], original["curves"] = config, curves
    merged = args.output_dir / "plot_data.input.json"
    merged.write_text(json.dumps(original, indent=2) + "\n")
    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_decode_capacity.py")),
                    "--plot-data", str(merged), "--output-dir", str(args.output_dir)], check=True)
    # Preserve a path-independent export for the official package and Vault.
    portable = read(args.output_dir / "plot_data.json")
    portable["config"] = read(Path(__file__).with_name("config.omnikv-total2048.json"))
    roots = [(args.original_root.resolve(), "original"),
             (args.rerun_root.resolve(), "omnikv-total2048")]

    def artifact_id(value):
        path = Path(value).resolve()
        for root, prefix in roots:
            if path.is_relative_to(root):
                return str(Path(prefix) / path.relative_to(root))
        raise ValueError(f"Artifact is outside the two declared campaigns: {path}")

    for curve in portable["curves"]:
        curve["capacity_artifact"] = artifact_id(curve["capacity_artifact"])
        for item in curve["attempts"] + curve["points"]:
            item["artifact"] = artifact_id(item["artifact"])
    target = args.output_dir / "portable"
    target.mkdir()
    (target / "plot_data.json").write_text(json.dumps(portable, indent=2) + "\n")
    with (target / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "lane", "concurrency",
            "decode_throughput_tps", "decode_tokens", "decode_elapsed_s", "measured_steps", "artifact"))
        writer.writeheader()
        for curve in portable["curves"]:
            for point in curve["points"]:
                writer.writerow({"model": curve["model"], "lane": curve["lane"], **point})


if __name__ == "__main__":
    main()
