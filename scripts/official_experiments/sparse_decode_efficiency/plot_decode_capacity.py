"""Validate completed capacity sweeps and plot synchronized pure decode rates.

Validate raw runs: python plot_decode_capacity.py --config campaign.json
Replot exported data: python plot_decode_capacity.py --plot-data plot_data.json
Override method colors: add --palette path/to/palette.json
Only observed results are accepted. Partial/omitted curves require explicit policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_PALETTE = Path(__file__).resolve().parent / "palettes" / "framework_families.json"
from benchmark.efficiency.paper import without_source_fingerprints

LANES = {
    "vllm-vanilla": ("vLLM (Vanilla)", "o"),
    "sengine-vanilla": ("Ours (Vanilla)", "s"),
    "sengine-snapkv": ("Ours (SnapKV)", "D"),
    "sengine-h2o": ("Ours (H2O)", "h"),
    "sengine-quest": ("Ours (QuEST)", "^"),
    "sengine-omnikv": ("Ours (OmniKV)", "P"),
}
EXTERNAL_LANES = {"tangram-snapkv": ("Tangram (SnapKV)", "v"),
                  "hisparse-quest": ("HiSparse (QuEST)", "X"),
                  "vortex-quest": ("Vortex (QuEST)", "*")}


class GridNotReadyError(ValueError):
    """Formal measurements are pending; this is not an artifact validation error."""


def configured_lanes(config):
    extras = config.get("external_lanes", {})
    if set(extras) - EXTERNAL_LANES.keys():
        raise ValueError("Unknown external curve identity")
    return {**LANES, **{key: EXTERNAL_LANES[key] for key in extras}}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_measurement(path, concurrency, config):
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one measurement: {path}")
    row = rows[0]
    if config.get("measurement_protocol") == "boundary_sync_v2":
        return validate_window_measurement(path, row, concurrency, config)
    expected = {"status": "success", "stage_metrics_status": "success",
                "measurement_scope": "full_batch_pure_decode_steps",
                "actual_decode_peak": concurrency, "completed_requests": concurrency,
                "length": config["input_len"], "output_len": config["output_len"],
                "scheduler_preemptions": 0, "synchronize_step_timing": True}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"{path}: {key}={row.get(key)!r}, expected {value!r}")
    case = path.parent / f"{row['method']}-{row['length']}-{concurrency}"
    outputs = read_rows(case / "raw_outputs.jsonl")
    if len(outputs) != concurrency or len({item["request_id"] for item in outputs}) != concurrency:
        raise ValueError(f"Missing or duplicate requests: {case}")
    if any(item["status"] != "success" or len(item["token_ids"]) != config["output_len"] for item in outputs):
        raise ValueError(f"Incomplete generation: {case}")
    steps = [item for item in read_rows(case / "steps.jsonl") if item["measured"]]
    if not steps or any(abs(item["tokens"]) != concurrency or not math.isfinite(item["elapsed_s"]) or item["elapsed_s"] <= 0 for item in steps):
        raise ValueError(f"Measured step is not a full batch: {case}")
    if row["engine"] == "sparseengine" and any(item["tokens"] >= 0 for item in steps):
        raise ValueError(f"Prefill included in native decode rate: {case}")
    if row["engine"] in ("vllm", "hisparse") and any(not item["pure_decode"] for item in steps):
        raise ValueError(f"Mixed or prefill step included in vLLM decode rate: {case}")
    tokens = sum(abs(item["tokens"]) for item in steps)
    elapsed = sum(item["elapsed_s"] for item in steps)
    if tokens != row["decode_stage_tokens"] or not math.isclose(elapsed, row["decode_stage_elapsed_s"], rel_tol=1e-10):
        raise ValueError(f"Stage accumulation disagrees with raw steps: {case}")
    if not math.isclose(tokens / elapsed, row["decode_stage_throughput_tps"], rel_tol=1e-10):
        raise ValueError(f"Throughput disagrees with raw steps: {case}")
    return {"concurrency": concurrency, "decode_throughput_tps": row["decode_stage_throughput_tps"],
            "decode_tokens": tokens, "decode_elapsed_s": elapsed,
            "artifact": str(path), "measured_steps": len(steps)}


def validate_window_measurement(path, aggregate, concurrency, config):
    """Validate completed work, not submitted batches or host step durations."""
    from benchmark.efficiency.metrics import aggregate_decode_windows

    repetitions = aggregate["repetitions"]
    if len(repetitions) != config["num_iters"]:
        raise ValueError(f"Missing repetitions: {path}")
    expected = dict(status="success", stage_metrics_status="success",
        measurement_scope="full_batch_decode_window", actual_decode_peak=concurrency,
        completed_requests=concurrency, length=config["input_len"], output_len=config["output_len"],
        scheduler_preemptions=0, synchronize_step_timing=False,
        stage_timing_scope="full_residency_contiguous_decode_only_boundary_sync_v2")
    portable = []
    for index, embedded in enumerate(repetitions):
        artifact = Path(embedded["artifact"])
        rows = read_rows(artifact)
        if len(rows) != 1 or embedded != {**rows[0], "artifact": str(artifact), "repetition": index}:
            raise ValueError(f"Repetition disagrees with original: {artifact}")
        row = rows[0]
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"{artifact}: invalid {key}")
        case = artifact.parent / f"{row['method']}-{row['length']}-{concurrency}"
        outputs = read_rows(case / "raw_outputs.jsonl")
        if (len(outputs) != concurrency or len({r["request_id"] for r in outputs}) != concurrency
                or any(r["status"] != "success" or len(r["token_ids"]) != config["output_len"] for r in outputs)):
            raise ValueError(f"Incomplete generation: {case}")
        window = json.loads((case / "window.json").read_text())
        if window != row["decode_window"]:
            raise ValueError(f"Window disagrees with row: {case}")
        if set(window["request_ids"]) != {r["request_id"] for r in outputs}:
            raise ValueError(f"Measured and completed request identities differ: {case}")
        expected_window = dict(status="success", scope=expected["stage_timing_scope"],
            concurrency=concurrency, discarded_full_decode_steps=config["decode_warmup_steps"],
            decode_steps=config["decode_window_steps"], completed_steps=config["decode_window_steps"],
            prefill_steps=0, cuda_synchronizations=2)
        if any(window.get(k) != v for k, v in expected_window.items()):
            raise ValueError(f"Window protocol mismatch: {case}")
        all_steps = read_rows(case / "window_steps.jsonl")
        steps = [r for r in all_steps if r["measured"]]
        n = config["decode_window_steps"]
        if len(steps) != n or any(not r["completed"] or not r["pure_decode"]
                or r.get("completed_decode_tokens") != concurrency
                or r["tokens"] != concurrency or r["request_ids"] != window["request_ids"] for r in steps):
            raise ValueError(f"Incomplete or mixed completed work: {case}")
        start = all_steps.index(steps[0])
        warmup = all_steps[start-config["decode_warmup_steps"]:start]
        if (len(warmup) != config["decode_warmup_steps"]
                or any(r["measured"] or not r["completed"] or not r["pure_decode"]
                       or r["tokens"] != concurrency or r["request_ids"] != window["request_ids"] for r in warmup)
                or [r["ticket"] for r in steps] != list(range(steps[0]["ticket"], steps[0]["ticket"] + n))):
            raise ValueError(f"Non-contiguous full-batch warmup/window: {case}")
        contexts = window["context_lengths_start"]
        if (len(contexts) != concurrency
                or any(r["context_lengths"] != [c + i for c in contexts] for i, r in enumerate(steps))
                or window["context_lengths_end"] != [c + n for c in contexts]):
            raise ValueError(f"Context progression mismatch: {case}")
        topology = row.get("resolved_parallel_topology", row.get("engine_hyper_params", {}))
        rank_count = topology["tensor_parallel_size"]
        deltas = window["graph_counter_delta"]
        if len(deltas) != rank_count or any(d != dict(capture_count=0, replay_count=n, eager_decode_count=0) for d in deltas):
            raise ValueError(f"Missing per-rank Graph completion: {case}")
        if row["engine"] == "vllm" and row.get("actual_async_scheduling") is not True:
            raise ValueError(f"vLLM async execution was disabled: {case}")
        if row["engine"] in ("hisparse", "vortex") and row.get("actual_overlap_scheduling") is not True:
            raise ValueError(f"HiSparse overlap was disabled: {case}")
        tokens, elapsed = concurrency * n, window["finished_monotonic_s"] - window["started_monotonic_s"]
        if (not math.isfinite(elapsed) or elapsed <= 0
                or row["decode_stage_tokens"] != tokens or window["decode_stage_tokens"] != tokens
                or any(not math.isclose(v, elapsed, rel_tol=1e-10) for v in
                       (row["decode_stage_elapsed_s"], window["decode_stage_elapsed_s"]))
                or any(not math.isclose(v, tokens / elapsed, rel_tol=1e-10) for v in
                       (row["decode_stage_throughput_tps"], window["decode_stage_throughput_tps"]))):
            raise ValueError(f"Window accounting mismatch: {case}")
        files = [artifact, case / "window.json", case / "window_steps.jsonl", case / "raw_outputs.jsonl",
                 artifact.parent / "run_info.json"]
        if row["engine"] == "vortex":
            for rank in range(1, rank_count):
                peer_file = case / f"rank{rank}" / "window.json"
                peer = json.loads(peer_file.read_text())
                for key in ("request_ids", "decode_steps", "decode_stage_tokens", "context_lengths_start",
                            "context_lengths_end", "graph_counter_delta"):
                    if peer[key] != window[key]:
                        raise ValueError(f"Vortex rank {rank} disagrees on {key}: {case}")
                files.extend([peer_file, peer_file.with_name("window_steps.jsonl")])
        portable.append(dict(repetition=index, decode_tokens=tokens, decode_elapsed_s=elapsed,
            decode_throughput_tps=tokens / elapsed, window=window,
            package_identity=without_source_fingerprints(row.get("package_identity")),
            vortex_identity=without_source_fingerprints(row.get("vortex_identity")),
            source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    rebuilt = aggregate_decode_windows(repetitions)
    comparison = dict(aggregate)
    key = "repetition_throughput_stdev_tps"
    actual, expected_stdev = comparison.get(key), rebuilt[key]
    # statistics.stdev can differ by an ULP across Python versions. Preserve
    # exact identity/count checks and tolerate only this derived float's roundoff.
    if (isinstance(actual, (int, float)) and expected_stdev is not None
            and math.isfinite(actual)
            and math.isclose(actual, expected_stdev, rel_tol=1e-12, abs_tol=1e-15)):
        comparison[key] = expected_stdev
    if comparison != rebuilt:
        raise ValueError(f"Aggregate disagrees with repetitions: {path}")
    return dict(concurrency=concurrency, decode_throughput_tps=rebuilt["decode_stage_throughput_tps"],
        decode_tokens=rebuilt["decode_stage_tokens"], decode_elapsed_s=rebuilt["decode_stage_elapsed_s"],
        artifact=str(path), measured_steps=rebuilt["measured_decode_steps_after_full"], repetitions=portable)


def load_campaign(config):
    root = Path(config["output_root"])
    curves = []
    for model in config["models"]:
        for lane in configured_lanes(config):
            omitted = config.get("omitted_curves", {}).get(model, {}).get(lane)
            if omitted:
                if not omitted.get("reason"):
                    raise ValueError("Omitted curve requires an explicit reason")
                curves.append(dict(model=model, lane=lane, status="omitted",
                                   points=[], attempts=[], **omitted))
                continue
            reason = config.get("unsupported", {}).get(model, {}).get(lane)
            if reason:
                curves.append({"model": model, "lane": lane, "status": "unsupported",
                               "reason": reason, "points": [], "attempts": []})
                continue
            partial = config.get("partial_curves", {}).get(model, {}).get(lane)
            if partial:
                if not partial.get("reason"):
                    raise ValueError("Partial curve requires an explicit stop reason")
                path = root / partial["capacity_artifact"]
                boundary = json.loads(path.read_text())
                attempts = boundary["attempts"]
                protocol = config.get("curve_protocols", {}).get(model, {}).get(lane, {})
                points = [validate_measurement(Path(a["artifact"]), a["concurrency"], {**config, **protocol})
                          for a in attempts if a["status"] == "success"]
                if not points:
                    raise ValueError("Partial curve has no validated measurements")
                curves.append(dict(model=model, lane=lane, status="partial", reason=partial["reason"],
                    max_concurrency=None, first_failed_concurrency=None,
                    max_measured_concurrency=max(p["concurrency"] for p in points),
                    theoretical_max_concurrency=partial.get("theoretical_max_concurrency"),
                    attempts=attempts, points=sorted(points, key=lambda p: p["concurrency"]),
                    capacity_artifact=str(path)))
                continue
            selected = config.get("capacity_artifacts", {}).get(model, {}).get(lane)
            matches = ([root / selected] if selected else
                       [path for path in (root / model).glob(f"*/{lane}/capacity.json")
                        if json.loads(path.read_text()).get("status") == "completed"])
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one completed {model}/{lane} sweep, found {len(matches)}")
            boundary = json.loads(matches[0].read_text())
            if boundary["status"] != "completed":
                raise ValueError(f"Incomplete sweep: {matches[0]}")
            maximum = boundary["max_concurrency"]
            if boundary["first_failed_concurrency"] != maximum + 1:
                raise ValueError(f"Integer maximum is unbounded: {matches[0]}")
            attempts = boundary["attempts"]
            if not any(item["concurrency"] == maximum + 1 and item["status"] == "capacity_exceeded" for item in attempts):
                raise ValueError(f"Missing capacity failure evidence: {matches[0]}")
            needed = {maximum}
            power = 1
            while power <= maximum:
                needed.add(power)
                power *= 2
            points = {}
            for attempt in attempts:
                if attempt["status"] == "success":
                    batch = attempt["concurrency"]
                    protocol = config.get("curve_protocols", {}).get(model, {}).get(lane, {})
                    points[batch] = validate_measurement(Path(attempt["artifact"]), batch, {**config, **protocol})
            if not needed <= points.keys():
                raise ValueError(f"Missing required concurrency points: {matches[0]}")
            curves.append({"model": model, "lane": lane, "max_concurrency": maximum,
                           "first_failed_concurrency": boundary["first_failed_concurrency"],
                           "attempts": attempts,
                           "points": [points[batch] for batch in sorted(points)],
                           "capacity_artifact": str(matches[0])})
    return curves


def load_grid(path):
    """Compose independently validated campaigns without mixing their lengths."""
    specification = json.loads(path.read_text())
    shape = specification["shape"]
    if (len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape)
            or len(specification["panels"]) != math.prod(shape)):
        raise ValueError("Grid must have exactly one campaign per panel")
    config = dict(models={}, panel_protocols={}, grid_shape=shape,
                  external_lanes={}, measurement_protocol="boundary_sync_v2",
                  figure_name="decode_capacity_128k32k_2k")
    curves = []
    for index, panel in enumerate(specification["panels"]):
        source = path.parent / os.path.expandvars(panel["config"])
        local = json.loads(source.read_text())
        model = panel["model"]
        local["models"] = {model: local["models"][model]}
        if local.get("measurement_protocol") != config["measurement_protocol"]:
            raise ValueError("Grid cannot mix boundary-sync and diagnostic protocols")
        # Explicit source selection avoids picking a duplicate/retry by glob order.
        for lane, policy in panel["curves"].items():
            if lane not in configured_lanes(local):
                raise ValueError(f"Unknown curve in grid: {lane}")
            capacity = Path(local["output_root"]) / policy["capacity_artifact"]
            if not policy.get("reason"):
                raise ValueError("Grid source selection requires its provenance/omission reason")
            local.setdefault("capacity_artifacts", {}).setdefault(model, {})[lane] = str(capacity)
            local.setdefault("partial_curves", {}).setdefault(model, {}).pop(lane, None)
            if capacity.exists():
                boundary = json.loads(capacity.read_text())
                status = boundary["status"]
                successes = [a for a in boundary["attempts"] if a["status"] == "success"]
            else:
                status, successes = "not_measured", []
            if status == "completed":
                continue
            if not policy.get("allow_partial"):
                raise ValueError(f"Grid requires a completed curve: {capacity}")
            evidence = dict(reason=policy["reason"], source_status=status,
                            capacity_artifact=str(capacity))
            if successes:
                local.setdefault("partial_curves", {}).setdefault(model, {})[lane] = evidence
            else:
                local.setdefault("omitted_curves", {}).setdefault(model, {})[lane] = evidence
        local_curves = load_campaign(local)
        for curve in local_curves:
            for point in curve["points"]:
                validate_grid_identity(point, model, local)
        # A populated comparison is required, not four empty axes or smoke points.
        if (sum(len(c["points"]) >= 2 for c in local_curves) < 2
                or not any(c["lane"] in ("sengine-vanilla", "vllm-vanilla")
                           and len(c["points"]) >= 2 for c in local_curves)):
            raise GridNotReadyError(f"Panel {model}/{local['input_len']} needs two measured curves with >=2 points, including a Vanilla baseline")
        panel_id = f"panel-{index}"
        config["models"][panel_id] = dict(local["models"][model], source_model=model)
        config["panel_protocols"][panel_id] = dict(local, source_model=model)
        config["external_lanes"].update(local.get("external_lanes", {}))
        curves.extend(dict(c, model=panel_id) for c in local_curves)
    return config, curves


def validate_grid_identity(point, model, config):
    """A valid token window from the wrong model/topology is not a panel point."""
    source = Path(point["artifact"]).parent / "identity.json"
    identity = json.loads(source.read_text())
    command = identity["command"]
    expected = {"--model-path": config["models"][model]["path"],
                "--tensor-parallel-size": str(config["models"][model]["tp"]),
                "--expert-parallel-size": str(config["models"][model]["ep"])}
    for flag, value in expected.items():
        if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
            raise ValueError(f"Grid source model/topology mismatch: {source}: {flag}")
    point["model_identity"] = dict(model=model, **config["models"][model],
        model_config_sha256=identity["model_config_sha256"],
        source_identity_sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def curve_config(config, model):
    return config.get("panel_protocols", {}).get(model, config)


def validate_portable_window_point(point, config):
    """Offline checks require saved repetitions; never open raw artifact paths."""
    repetitions = point.get("repetitions", [])
    n, batch = config["decode_window_steps"], point["concurrency"]
    if len(repetitions) != config["num_iters"]:
        raise ValueError("Portable window export is missing repetitions")
    for index, row in enumerate(repetitions):
        window = row["window"]
        if (row["repetition"] != index or row["decode_tokens"] != batch * n
                or window["scope"] != "full_residency_contiguous_decode_only_boundary_sync_v2"
                or window["decode_steps"] != n or window["concurrency"] != batch
                or window["discarded_full_decode_steps"] != config["decode_warmup_steps"]
                or row["decode_tokens"] != window["decode_stage_tokens"]
                or row["decode_elapsed_s"] != window["decode_stage_elapsed_s"]
                or not math.isclose(row["decode_throughput_tps"], row["decode_tokens"] / row["decode_elapsed_s"], rel_tol=1e-10)):
            raise ValueError("Portable window export has inconsistent protocol/accounting")
    if (sum(r["decode_tokens"] for r in repetitions) != point["decode_tokens"]
            or point["measured_steps"] != len(repetitions) * n
            or not math.isclose(sum(r["decode_elapsed_s"] for r in repetitions), point["decode_elapsed_s"], rel_tol=1e-10)):
        raise ValueError("Portable repetitions disagree with aggregate")


def apply_presentation(config, curves, presentation):
    """Select panels for display without dropping their high-concurrency evidence."""
    allowed = {"input_len", "line_max_concurrency", "panel_labels", "figure_name",
               "max_batch_bars", "line_ymin_padding"}
    if set(presentation) - allowed:
        raise ValueError("Unknown presentation setting")
    models = {key: value for key, value in config["models"].items()
              if curve_config(config, key)["input_len"] == presentation["input_len"]}
    if not models:
        raise ValueError("Presentation selected no model panels")
    for key in models:
        source = curve_config(config, key).get("source_model", key)
        limit = presentation["line_max_concurrency"][source]
        if type(limit) is not int or limit < 1:
            raise ValueError("Line concurrency limits must be positive integers")
    config = {**config, "models": models, "grid_shape": [1, len(models)],
              "batch_bands": False, "presentation": presentation,
              "figure_name": presentation["figure_name"]}
    if "panel_protocols" in config:
        config["panel_protocols"] = {key: config["panel_protocols"][key] for key in models}
    return config, [curve for curve in curves if curve["model"] in models]


def max_batch_points(curves):
    """Select the largest measured batch, not the throughput argmax."""
    rows = []
    for curve in curves:
        if not curve["points"]:
            continue
        point = max(curve["points"], key=lambda item: item["concurrency"])
        confirmed = (curve.get("status") != "partial"
                     and curve.get("max_concurrency") == point["concurrency"])
        rows.append(dict(panel=curve["model"], lane=curve["lane"],
                         capacity_confirmed=confirmed, **point))
    return rows


def relative_vllm_points(curves, config):
    """Derive percentages only for exact measured, same-protocol pairs."""
    rows = []
    for model in config['models']:
        local = curve_config(config, model)
        source = local.get('source_model', model)
        overrides = local.get('curve_protocols', {}).get(source, {})
        baseline_curves = [c for c in curves if c['model'] == model and c['lane'] == 'vllm-vanilla']
        if len(baseline_curves) != 1:
            raise ValueError(f'Expected one vLLM Vanilla curve for {model}')
        baseline = {p['concurrency']: p for p in baseline_curves[0]['points']}
        if len(baseline) != len(baseline_curves[0]['points']):
            raise ValueError('Duplicate baseline concurrency')
        reference = {**local, **overrides.get('vllm-vanilla', {})}
        for curve in (c for c in curves if c['model'] == model):
            effective = {**local, **overrides.get(curve['lane'], {})}
            protocol_keys = ('input_len', 'output_len', 'measurement_protocol',
                             'decode_window_steps', 'decode_warmup_steps', 'num_iters')
            compatible = all(effective.get(k) == reference.get(k) for k in protocol_keys)
            for point in curve['points']:
                rate = point['decode_throughput_tps']
                if not math.isfinite(rate) or rate <= 0:
                    raise ValueError('Relative throughput requires finite positive source rates')
                base = baseline.get(point['concurrency'])
                row = dict(panel=model, lane=curve['lane'], concurrency=point['concurrency'],
                           throughput_tps=rate, artifact=point.get('artifact'),
                           baseline_throughput_tps=None, baseline_artifact=None, delta_percent=None)
                if not compatible:
                    row.update(status='skipped_by_policy', reason='protocol_mismatch')
                elif base is None:
                    row.update(status='skipped_by_policy', reason='no_measured_baseline_at_same_concurrency')
                else:
                    denominator = base['decode_throughput_tps']
                    if not math.isfinite(denominator) or denominator <= 0:
                        raise ValueError('Relative throughput requires a finite positive baseline')
                    row.update(status='success', reason='', baseline_throughput_tps=denominator,
                               baseline_artifact=base.get('artifact'),
                               delta_percent=100 * (rate / denominator - 1))
                rows.append(row)
    return rows


def render_relative_vllm(curves, config, output, colors, lanes):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    derived = relative_vllm_points(curves, config)
    if not derived:
        raise ValueError('No observed points for relative throughput')
    (output / 'delta_points.json').write_text(json.dumps(dict(
        baseline='vllm-vanilla', formula='100 * (throughput / baseline_throughput - 1)',
        pairing='exact model, input/output protocol and measured concurrency; no interpolation',
        points=derived), indent=2) + '\n')
    with (output / 'delta_points.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(derived[0]))
        writer.writeheader()
        writer.writerows(derived)
    models = list(config['models'])
    rows, columns = config.get('grid_shape', [1, len(models)])
    fig, axes = plt.subplots(rows, columns, figsize=(5.1 * columns, 3.8 * rows),
                             layout='constrained', squeeze=False, sharey='row')
    visible_values = [[0.0] for _ in range(rows)]
    for index, (ax, model) in enumerate(zip(axes.flat, models)):
        local = curve_config(config, model)
        source = local.get('source_model', model)
        presentation = config.get('presentation', {})
        limit = presentation.get('line_max_concurrency', {}).get(source, math.inf)
        xs = []
        ax.axhline(0, color=colors['vllm-vanilla'], linewidth=1.5, linestyle='--',
                   label='vLLM Vanilla (0%)', zorder=1)
        for lane, (label, marker) in lanes.items():
            if lane == 'vllm-vanilla':
                continue
            points = [p for p in derived if p['panel'] == model and p['lane'] == lane
                      and p['concurrency'] <= limit]
            if not any(p['status'] == 'success' for p in points):
                continue
            # NaNs break lines at explicitly unmatched observations.
            ax.plot([p['concurrency'] for p in points],
                    [p['delta_percent'] if p['status'] == 'success' else math.nan for p in points],
                    color=colors[lane], marker=marker, markersize=7, linewidth=2, label=label,
                    linestyle='--' if lane == 'sengine-vanilla' else '-')
            matched = [p for p in points if p['status'] == 'success']
            xs.extend(p['concurrency'] for p in matched)
            visible_values[index // columns].extend(p['delta_percent'] for p in matched)
        if not xs:
            raise ValueError(f'No matched non-baseline points for {model}')
        identity = config['models'][model]
        name = presentation.get('panel_labels', {}).get(source, identity['display_name'])
        name = name.replace('-Instruct-2507', '').replace('-FP8', ' (FP8)')
        ax.set_xlabel(f"Concurrency\n{name} · TP={identity['tp']}, EP={identity['ep']}")
        maximum = max(xs)
        ax.set_xlim(.8, maximum + .2)
        ax.set_xticks(sorted(set(xs)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    for row, values in enumerate(visible_values):
        padding = max(5, (max(values) - min(values)) * .1)
        axes[row, 0].set_ylim(min(values) - padding, max(values) + padding)
    legend = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    fig.legend(list(legend.values()), list(legend), loc='outside upper center', ncols=4,
               frameon=False, columnspacing=.8, handlelength=1.5, handletextpad=.4)
    fig.supylabel('Δ throughput vs vLLM (%)', fontsize=14)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f"{config['figure_name']}_delta_vllm.{extension}", dpi=220)
    plt.close(fig)


def render(curves, config, output, palette_path=DEFAULT_PALETTE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import is_color_like
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, StrMethodFormatter
    import seaborn as sns

    palette = json.loads(Path(palette_path).read_text())
    colors = palette["colors"]
    lanes = configured_lanes(config)
    for lane in lanes:
        if lane not in colors or not is_color_like(colors[lane]):
            raise ValueError(f"Palette {palette_path} has a missing or invalid color for {lane}")

    sns.set_theme(context="notebook", style="whitegrid", font="DejaVu Sans",
                  rc={"axes.spines.top": False, "axes.spines.right": False,
                      "grid.alpha": 0.25, "axes.titleweight": "bold"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "palette.json").write_text(json.dumps(palette, indent=2) + "\n")

    if config.get("line_metric") == "relative-vllm":
        render_relative_vllm(curves, config, output, colors, lanes)
        return
    if config.get("batch_bands"):
        print("DEPRECATED: broken-axis figures; use continuous or relative-vllm line plots.", file=sys.stderr)
        (output / "DEPRECATED.md").write_text(
            "# Deprecated figure layout\n\nBroken-axis line figures are retained for historical reproduction only. "
            "Use the continuous panels or relative-vllm delta figures for new comparisons.\n")
    presentation = config.get("presentation")

    def caption(model):
        local = curve_config(config, model)
        source = local.get("source_model", model)
        identity = config["models"][model]
        name = presentation["panel_labels"].get(source, identity["display_name"])
        return f"{name} · TP={identity['tp']}, EP={identity['ep']}"

    def panel(ax, model, log_y, band=None):
        local_curves = []
        band_values = []
        cut = config.get("batch_band_cuts", {}).get(model, 4.5)
        for curve in curves:
            if curve["model"] != model:
                continue
            label, marker = lanes[curve["lane"]]
            local = curve_config(config, model)
            source_model = local.get("source_model", model)
            protocol = local.get("curve_protocols", {}).get(source_model, {}).get(curve["lane"], {})
            label += protocol.get("label_suffix", "")
            if curve.get("status") in ("unsupported", "omitted"):
                continue
            points = curve["points"]
            if presentation:
                limit = presentation["line_max_concurrency"][source_model]
                points = [p for p in points if p["concurrency"] <= limit]
            if band is not None:
                points = [p for p in points if (p["concurrency"] < cut) == (band == "low")]
            if not points:
                continue
            local_curves.append(curve)
            color = colors[curve["lane"]]
            xs = [point["concurrency"] for point in points]
            ys = [point["decode_throughput_tps"] for point in points]
            band_values.extend(ys)
            sns.lineplot(x=xs, y=ys, ax=ax, label=label, color=color, marker=marker,
                         markersize=6.25 if curve["lane"] == "sengine-vanilla" else 7.5,
                         linestyle="--" if curve["lane"] == "sengine-vanilla" else "-",
                         linewidth=2, estimator=None, errorbar=None)
        ax.set(xlabel="Concurrency", ylabel="Throughput (tok/s)")
        ax.set_xscale("log", base=2)
        # Integer-boundary probes can be adjacent on the log axis; keep native
        # major ticks at powers of two instead of overlapping endpoint labels.
        ax.xaxis.set_major_locator(LogLocator(base=2))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        elif band is None:
            ax.set_ylim(bottom=0)
        if band is not None and band_values:
            bottom = min(band_values) - config.get("batch_band_ymin_padding", 20)
            if log_y and bottom <= 0:
                raise ValueError("Log-y band minimum minus padding must be positive")
            ax.set_ylim(bottom=bottom)
        if ax.legend_ is not None:
            ax.legend_.remove()
        if presentation:
            local = curve_config(config, model)
            limit = presentation["line_max_concurrency"][local.get("source_model", model)]
            ax.set_xscale("linear")
            ax.set_xlim(.8, limit + .2)
            ax.set_xticks(range(1, limit + 1))
            if not band_values:
                raise ValueError(f"No observed values inside line range for {model}")
            bottom = min(band_values) - presentation.get("line_ymin_padding", 20)
            if log_y and bottom <= 0:
                raise ValueError("Log-y minimum minus padding must be positive")
            ax.set_ylim(bottom=bottom)
            ax.set_xlabel("Concurrency\n" + caption(model))
            return
        if band is not None:
            ax.set(xlabel="", ylabel="")
            if band == "low":
                ax.set_xscale("linear")
                ax.set_xlim(.8, cut)
                ax.set_xticks(range(1, math.ceil(cut)))
                ax.spines["right"].set_visible(False)
            else:
                maximum = max([64] + [p["concurrency"] for c in local_curves for p in c["points"]])
                ax.set_xlim(cut, maximum * 1.08)
                ticks = [5] + [2**n for n in range(3, math.floor(math.log2(maximum)) + 1)]
                ax.set_xticks([tick for tick in ticks if tick >= cut])
                ax.yaxis.tick_right()
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(True)
            if not log_y:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            # The adjacent axes have independent y scales. Mark the seam but
            # never connect curve endpoints across these different transforms.
            edge = 1 if band == "low" else 0
            for y in (0, 1):
                ax.plot([edge - .025, edge + .025], [y - .014, y + .014],
                        transform=ax.transAxes, color="#89969E", linewidth=1,
                        clip_on=False)
            return

        # Use observed low-batch points only; neither interpolate new samples nor
        # include the next point above six when determining the zoom's y range.
        low_values = [point["decode_throughput_tps"] for curve in local_curves
                      for point in curve["points"] if point["concurrency"] <= 6]
        if not low_values:
            return
        bounds = [.57, .12, .40, .44] if log_y else [.14, .50, .40, .46]
        zoom = ax.inset_axes(bounds)
        for curve in local_curves:
            points = [point for point in curve["points"] if point["concurrency"] <= 6]
            zoom.plot([p["concurrency"] for p in points],
                      [p["decode_throughput_tps"] for p in points],
                      color=colors[curve["lane"]], marker=lanes[curve["lane"]][1],
                      markersize=4.375, linewidth=1.2,
                      linestyle="--" if curve["lane"] == "sengine-vanilla" else "-")
        padding = max((max(low_values) - min(low_values)) * .10, max(low_values) * .03)
        zoom.set(xlim=(.85, 6.15), ylim=(max(0, min(low_values) - padding), max(low_values) + padding))
        zoom.set_xticks([1, 2, 4, 6])
        zoom.yaxis.set_major_locator(MaxNLocator(nbins=3))
        zoom.tick_params(axis="both", labelsize=8, length=2, pad=1)
        for spine in zoom.spines.values():
            spine.set_visible(True)
            spine.set_color("#9AA6AE")
            spine.set_linewidth(.8)
        # The linear inset can extend below the log overview's visible range.
        # Mark only the visible intersection to keep connectors inside the axes.
        x0, x1 = zoom.get_xlim()
        x0, x1 = max(x0, ax.get_xlim()[0]), min(x1, ax.get_xlim()[1])
        y0, y1 = zoom.get_ylim()
        y0, y1 = max(y0, ax.get_ylim()[0]), min(y1, ax.get_ylim()[1])
        indicator = ax.indicate_inset(
            (x0, y0, x1 - x0, y1 - y0), inset_ax=zoom,
            edgecolor="#9AA6AE", linewidth=.8, alpha=.6,
        )
        indicator.set_zorder(1)

    models = list(config["models"])
    rows, columns = config.get("grid_shape", [1, len(models)])
    if rows * columns != len(models):
        raise ValueError("Grid dimensions disagree with panel count")
    for log_y in (False, True):
        suffix = "_logy" if log_y else ""
        if config.get("batch_bands"):
            fig = plt.figure(figsize=(4.375 * columns, 3.5 * rows), layout="constrained")
            groups = fig.subfigures(rows, columns, squeeze=False, wspace=.12, hspace=.05)
            legend_axes = []
            for group, model in zip(groups.flat, models):
                axes = group.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.6], "wspace": .015})
                for ax, band in zip(axes, ("low", "high")):
                    panel(ax, model, log_y, band=band)
                group.supxlabel("Concurrency", fontsize=plt.rcParams["axes.labelsize"])
                legend_axes.extend(axes)
        else:
            fig, axes = plt.subplots(rows, columns, figsize=(4.375 * columns, 3.5 * rows),
                                     layout="constrained", squeeze=False)
            for ax, model in zip(axes.flat, models):
                panel(ax, model, log_y)
                ax.set_ylabel("")
            legend_axes = list(axes.flat)
        if config.get("batch_bands"):
            fig.supxlabel("DEPRECATED · broken-axis layout", fontsize=10, color="#64748B")
        fig.supylabel("Throughput (tok/s)", fontsize=plt.rcParams["axes.labelsize"])
        legend = {}
        for ax in legend_axes:
            handles, labels = ax.get_legend_handles_labels()
            legend.update(zip(labels, handles))
        labels, handles = list(legend), list(legend.values())
        fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False,
                   columnspacing=.9, handlelength=1.5, handletextpad=.4)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{config.get('figure_name', 'decode_capacity_128k2k')}{suffix}.{extension}", dpi=220)
        plt.close(fig)
        if "grid_shape" in config or config.get("batch_bands"):
            continue
        for model in models:
            fig, ax = plt.subplots(figsize=(5, 3.625), layout="constrained")
            panel(ax, model, log_y)
            ax.set_ylabel("")
            fig.supylabel("Throughput (tok/s)", fontsize=plt.rcParams["axes.labelsize"])
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
            for extension in ("png", "pdf", "svg"):
                fig.savefig(output / f"{model}{suffix}.{extension}", dpi=220)
            plt.close(fig)

    if presentation and presentation.get("max_batch_bars"):
        selected = max_batch_points(curves)
        (output / "max_batch_points.json").write_text(json.dumps(selected, indent=2) + "\n")
        with (output / "max_batch_points.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["panel", "lane", "concurrency",
                "decode_throughput_tps", "capacity_confirmed", "artifact"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
        # Match sparseengine_vs_vortex/plot.py's visual theme, not its mean/SD
        # aggregation: these bars retain the recorded pooled token/time rates.
        sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans", font_scale=1.15,
                      rc={"axes.edgecolor": "#D6DDE5", "grid.color": "#E9EEF3", "grid.linewidth": .7,
                          "grid.alpha": 1,
                          "text.color": "#364152", "axes.labelcolor": "#364152",
                          "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.bbox": None})
        width = 5.5 * len(models)
        fig, axes = plt.subplots(1, len(models), figsize=(width, width / 2.5),
                                 layout="constrained", squeeze=False, sharey=True)
        for ax, model in zip(axes.flat, models):
            points = [p for p in selected if p["panel"] == model]
            bars = ax.bar(range(len(points)), [p["decode_throughput_tps"] for p in points],
                          color=[colors[p["lane"]] for p in points], width=.704, edgecolor="none")
            labels = []
            for point in points:
                method_label = lanes[point["lane"]][0].replace(" (", "\n(")
                labels.append(f"{method_label}\n{point['decode_throughput_tps']:.1f}\nB={point['concurrency']}")
            ax.bar_label(bars, labels=labels, padding=4, fontsize=9, color="#364152")
            ax.set_xticks([])
            ax.set_xlabel(caption(model))
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax)
        axes[0, 0].set_ylim(0, max(p["decode_throughput_tps"] for p in selected) * 1.32)
        fig.supylabel("Throughput (tok/s)", fontsize=plt.rcParams["axes.labelsize"])
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{config['figure_name']}_max_batch.{extension}", dpi=220)
        plt.close(fig)
        (output / "max_batch_style.json").write_text(json.dumps(dict(
            reference="scripts/official_experiments/sparseengine_vs_vortex/plot.py",
            theme="paper/whitegrid", font_scale=1.15, figsize_inches=[width, width / 2.5],
            shared_y_axis=True, bar_labels="framework / method / pooled throughput (one decimal) / B=batch", legend=False,
            aggregation="unchanged pooled completed tokens / elapsed time; no mean/SD substitution",
            palette="unchanged method colors", capacity_marker="not displayed; verification status retained in data"), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--grid-config", type=Path, help="Row-major panel configs and explicit capacity sources")
    source.add_argument("--plot-data", type=Path, help="Replot a previously raw-validated portable export")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--export-data-dir", type=Path,
                        help="New external data directory for portable JSON/CSV and provenance; never overwrites")
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE,
                        help="Method-color JSON; defaults to palettes/framework_families.json beside this script")
    parser.add_argument("--line-metric", choices=("absolute", "relative-vllm"),
                        help="Absolute throughput or percentage change versus exact measured vLLM Vanilla points")
    parser.add_argument("--batch-bands", action="store_true",
                        help="DEPRECATED: historical adjacent axes with independent y scales")
    parser.add_argument("--presentation-config", type=Path,
                        help="Select an input length, low-BS line limits, captions and max-batch bars")
    parser.add_argument("--wait-seconds", type=int, default=0,
                        help="Bounded grid-only wait for formal points; validation errors still fail immediately")
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 172800 or (args.wait_seconds and not args.grid_config):
        raise ValueError("--wait-seconds requires --grid-config and must be between0 and172800")
    if args.plot_data:
        exported = without_source_fingerprints(json.loads(args.plot_data.read_text()))
        if exported["schema_version"] != 1:
            raise ValueError("Unsupported plot-data schema")
        config, curves = exported["config"], exported["curves"]
        expected = {(model, lane) for model in config["models"]
                    for lane in configured_lanes(curve_config(config, model))}
        if len(curves) != len(expected) or {(c["model"], c["lane"]) for c in curves} != expected:
            raise ValueError("Portable export is missing curves or contains duplicates")
        for curve in curves:
            local = curve_config(config, curve["model"])
            model = local.get("source_model", curve["model"])
            if curve.get("status") == "omitted":
                evidence = local.get("omitted_curves", {}).get(model, {}).get(curve["lane"])
                if (not evidence or any(curve.get(k) != v for k, v in evidence.items())
                        or curve["points"] or curve["attempts"]):
                    raise ValueError("Omitted curve requires explicit evidence and no fabricated points")
                continue
            if curve.get("status") == "unsupported":
                reason = local.get("unsupported", {}).get(model, {}).get(curve["lane"])
                if not reason or reason != curve.get("reason") or curve["points"] or curve["attempts"]:
                    raise ValueError("Unsupported curve requires explicit evidence and no fabricated points")
                continue
            points = curve["points"]
            protocol = local.get("curve_protocols", {}).get(model, {}).get(curve["lane"])
            if protocol:
                if (not protocol.get("reason") or not protocol.get("label_suffix")
                        or protocol["input_len"] + protocol["output_len"] != local["input_len"] + local["output_len"]
                        or any(p["measured_steps"] < protocol.get("minimum_measured_steps", local.get("decode_window_steps", 1)) for p in points)):
                    raise ValueError("Adjusted protocol requires explicit labelling, constant span and sufficient measured steps")
            batches = [p["concurrency"] for p in points]
            partial = local.get("partial_curves", {}).get(model, {}).get(curve["lane"])
            if curve.get("status") == "partial":
                maximum = curve["max_measured_concurrency"]
                if (not partial or not partial.get("reason") or curve.get("reason") != partial["reason"]
                        or curve["max_concurrency"] is not None or curve["first_failed_concurrency"] is not None
                        or curve.get("theoretical_max_concurrency") != partial.get("theoretical_max_concurrency")
                        or type(maximum) is not int or maximum < 1):
                    raise ValueError("Partial curve must distinguish measured extent from unverified capacity")
            else:
                maximum = curve["max_concurrency"]
            if curve.get("status") != "partial" and (type(maximum) is not int or maximum < 1
                    or curve["first_failed_concurrency"] != maximum + 1
                    or not any(item["concurrency"] == maximum + 1
                               and item["status"] == "capacity_exceeded"
                               for item in curve["attempts"])):
                raise ValueError("Portable export is missing an integer capacity boundary")
            successful = {item["concurrency"] for item in curve["attempts"]
                          if item["status"] == "success"}
            if set(batches) != successful:
                raise ValueError("Portable export points disagree with successful attempts")
            needed = {maximum}
            power = 1
            while power <= maximum and curve.get("status") != "partial":
                needed.add(power)
                power *= 2
            if (batches != sorted(set(batches)) or not needed <= set(batches)
                    or max(batches) != maximum):
                raise ValueError("Portable export has an incomplete concurrency curve")
            for point in points:
                if "grid_shape" in config:
                    identity = point["model_identity"]
                    if (identity["model"] != model or any(identity.get(k) != v
                            for k, v in local["models"][model].items())):
                        raise ValueError("Portable grid point disagrees with panel model/topology")
                tokens, elapsed, rate = (point[key] for key in
                                        ("decode_tokens", "decode_elapsed_s", "decode_throughput_tps"))
                if (not all(math.isfinite(v) and v > 0 for v in (tokens, elapsed, rate))
                        or not math.isclose(tokens / elapsed, rate, rel_tol=1e-10)):
                    raise ValueError("Portable export has inconsistent stage throughput")
                if config.get("measurement_protocol") == "boundary_sync_v2":
                    validate_portable_window_point(point, {**local, **(protocol or {})})
        output = args.output_dir or args.plot_data.parent / "plots"
    elif args.grid_config:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            try:
                config, curves = load_grid(args.grid_config)
                break
            except GridNotReadyError as error:
                if time.monotonic() >= deadline:
                    raise
                print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} waiting: {error}", flush=True)
                time.sleep(min(30, max(0, deadline - time.monotonic())))
        output = args.output_dir or args.grid_config.parent / "plots"
    else:
        config = json.loads(args.config.read_text())
        curves = load_campaign(config)
        output = args.output_dir or Path(config["output_root"]) / "plots"
    if args.presentation_config:
        if args.batch_bands:
            raise ValueError("Presentation panels cannot be combined with --batch-bands")
        config, curves = apply_presentation(config, curves, json.loads(args.presentation_config.read_text()))
    protocol = config.get("measurement_protocol", "step_sync_v1")
    if args.batch_bands:
        config["batch_bands"] = True
    if config.get("batch_bands"):
        config.setdefault("batch_band_ymin_padding", 20)
        name = config.get("figure_name", "decode_capacity_128k2k")
        if not name.endswith("_bands"):
            config["figure_name"] = name + "_bands"
    if args.line_metric:
        config["line_metric"] = args.line_metric
    relative = config.get("line_metric") == "relative-vllm"
    if relative:
        if args.batch_bands:
            raise ValueError("Relative delta plots require continuous axes")
        config["batch_bands"] = False
        config["figure_name"] = config.get("figure_name", "decode_capacity").removesuffix("_bands")
        if config.get("presentation"):
            config["presentation"]["max_batch_bars"] = False
    if protocol not in ("boundary_sync_v2", "step_sync_v1"):
        raise ValueError("Unknown measurement protocol")
    if args.export_data_dir:
        args.export_data_dir.mkdir(parents=True, exist_ok=False)
    render(curves, config, output, palette_path=args.palette)
    (output / "validated_curves.json").write_text(json.dumps(curves, indent=2) + "\n")
    (output / "plot_data.json").write_text(json.dumps({
        "schema_version": 1,
        "metric": ("actual_decode_tokens / sum_boundary_synchronized_decode_window_seconds"
                   if protocol == "boundary_sync_v2" else "actual_decode_tokens / sum_synchronized_full_batch_decode_step_seconds"),
        "validation": "raw steps and full outputs verified at export; replot checks exported sums only",
        "config": config, "curves": curves,
    }, indent=2) + "\n")
    fields = ["panel", "model", "input_len", "output_len", "lane", "concurrency", "decode_throughput_tps", "decode_tokens",
              "decode_elapsed_s", "measured_steps", "artifact"]
    with (output / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for curve in curves:
            for point in curve["points"]:
                local = curve_config(config, curve["model"])
                model = local.get("source_model", curve["model"])
                local = {**local, **local.get("curve_protocols", {}).get(model, {}).get(curve["lane"], {})}
                writer.writerow({"panel": curve["model"], "model": model, "input_len": local["input_len"],
                                 "output_len": local["output_len"], "lane": curve["lane"], **point})
    if args.export_data_dir:
        for name in ("plot_data.json", "points.csv", "palette.json"):
            (args.export_data_dir / name).write_bytes((output / name).read_bytes())
        provenance = {"measurement_protocol": protocol, "source_identities": {}}
        if args.plot_data:
            provenance["plot_data_source"] = str(args.plot_data.resolve())
            provenance["plot_data_sha256"] = hashlib.sha256(args.plot_data.read_bytes()).hexdigest()
        if config.get("presentation"):
            provenance["presentation"] = config["presentation"]
        if args.grid_config:
            provenance["grid_config"] = json.loads(args.grid_config.read_text())
            provenance["grid_config_sha256"] = hashlib.sha256(args.grid_config.read_bytes()).hexdigest()
        if args.config or args.grid_config:
            for curve in curves:
                for point in curve["points"]:
                    parent = Path(point["artifact"]).parent
                    for name in ("identity.json", "run_manifest.json"):
                        source = parent / name
                        if source.is_file():
                            provenance["source_identities"][str(source)] = without_source_fingerprints(json.loads(source.read_text()))
        (args.export_data_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if config.get("presentation", {}).get("max_batch_bars"):
            names = ["max_batch_points.json", "max_batch_points.csv", "max_batch_style.json"]
            names.extend(f"{config['figure_name']}_max_batch.{ext}" for ext in ("png", "pdf", "svg"))
            for name in names:
                (args.export_data_dir / name).write_bytes((output / name).read_bytes())
        if relative:
            for name in ["delta_points.json", "delta_points.csv", *[
                    f"{config['figure_name']}_delta_vllm.{ext}" for ext in ("png", "pdf", "svg")]]:
                (args.export_data_dir / name).write_bytes((output / name).read_bytes())
        if config.get("batch_bands"):
            (args.export_data_dir / "DEPRECATED.md").write_bytes((output / "DEPRECATED.md").read_bytes())
        if "grid_shape" in config and not relative:
            for suffix in ("", "_logy"):
                for extension in ("png", "pdf", "svg"):
                    name = f"{config['figure_name']}{suffix}.{extension}"
                    (args.export_data_dir / name).write_bytes((output / name).read_bytes())


if __name__ == "__main__":
    main()
