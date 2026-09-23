"""Reject misleading capacity curves even when a runner reports success."""
import json
import math

import pytest

from scripts.official_experiments.sparse_decode_efficiency.plot_decode_capacity import validate_measurement


def make_window_case(root):
    from benchmark.efficiency.metrics import PipelinedDecodeWindow, decode_window_fields, aggregate_decode_windows
    config = dict(measurement_protocol="boundary_sync_v2", input_len=10, output_len=6,
                  decode_window_steps=3, decode_warmup_steps=1, num_iters=2)
    repetitions = []
    for index, step_time in enumerate((1.0, 2.0)):
        now, graph = [0.0], dict(capture_count=0, replay_count=0, eager_decode_count=0)
        window = PipelinedDecodeWindow(2, 3, 1, synchronize=lambda: None,
            clock=lambda: now[0], graph_stats=lambda: [dict(graph)])
        for context in range(11, 15):
            window.boundary()
            ticket = window.submit(is_decode=True, request_ids=[0, 1], tokens=2,
                admission_complete=True, context_lengths=[context, context])
            now[0] += step_time
            graph["replay_count"] += 1
            window.complete(ticket, decode_tokens=2)
        window.boundary()
        artifact = root / f"repeat{index}" / "performance.jsonl"
        case = artifact.parent / "vanilla-10-2"
        case.mkdir(parents=True)
        row = dict(engine="sparseengine", method="vanilla", length=10, output_len=6, batch_size=2,
            status="success", actual_decode_peak=2, completed_requests=2, scheduler_preemptions=0,
            decode_warmup_steps_after_full=1, resolved_parallel_topology={"tensor_parallel_size": 1},
            **decode_window_fields(window.require_result()))
        artifact.write_text(json.dumps(row) + "\n")
        (artifact.parent / "run_info.json").write_text("{}")
        (case / "window.json").write_text(json.dumps(window.result))
        (case / "window_steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in window.records))
        (case / "raw_outputs.jsonl").write_text("".join(json.dumps(dict(request_id=i, token_ids=[7]*6, status="success")) + "\n" for i in (0, 1)))
        repetitions.append({**row, "artifact": str(artifact), "repetition": index})
    artifact = root / "performance.jsonl"
    artifact.write_text(json.dumps(aggregate_decode_windows(repetitions)) + "\n")
    return artifact, config


def test_window_export_weights_total_work_and_replots_without_raw_paths(tmp_path):
    from scripts.official_experiments.sparse_decode_efficiency.plot_decode_capacity import validate_portable_window_point
    artifact, config = make_window_case(tmp_path)
    point = validate_measurement(artifact, 2, config)
    assert point["decode_throughput_tps"] == 12 / 9  # Not mean(2, 1).
    point["artifact"] = "unavailable-machine/raw.jsonl"
    validate_portable_window_point(point, config)
    point["repetitions"].pop()
    with pytest.raises(ValueError, match="missing repetitions"):
        validate_portable_window_point(point, config)


@pytest.mark.parametrize("change", ["one_ulp", "wrong_stdev", "nan_stdev"])
def test_window_export_checks_stdev_without_requiring_cross_python_bit_identity(tmp_path, change):
    """Python 3.10/3.12 stdev can differ by one ULP for identical raw times."""
    artifact, config = make_window_case(tmp_path)
    row = json.loads(artifact.read_text())
    key = "repetition_throughput_stdev_tps"
    row[key] = (math.nextafter(row[key], math.inf) if change == "one_ulp"
                else row[key] * 1.01 if change == "wrong_stdev" else math.nan)
    artifact.write_text(json.dumps(row))
    if change == "one_ulp":
        assert validate_measurement(artifact, 2, config)["decode_throughput_tps"] == 12 / 9
    else:
        with pytest.raises(ValueError, match="Aggregate disagrees"):
            validate_measurement(artifact, 2, config)


@pytest.mark.parametrize("corruption", ["uncompleted", "wrong_context", "short_output", "wrong_sum", "old_protocol"])
def test_window_export_rejects_falsely_successful_artifacts(tmp_path, corruption):
    artifact, config = make_window_case(tmp_path)
    case = tmp_path / "repeat0" / "vanilla-10-2"
    if corruption in ("uncompleted", "wrong_context"):
        path = case / "window_steps.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if corruption == "uncompleted":
            rows[-1]["completed"] = False
        else:
            rows[-1]["context_lengths"][0] += 10
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif corruption == "short_output":
        (case / "raw_outputs.jsonl").write_text(json.dumps(dict(request_id=0, token_ids=[7], status="success")))
    else:
        row = json.loads(artifact.read_text())
        if corruption == "wrong_sum":
            row["decode_stage_elapsed_s"] = 2
        else:
            row["measurement_scope"] = "full_batch_pure_decode_steps"
        artifact.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        validate_measurement(artifact, 2, config)


def make_case(tmp_path, engine="sparseengine"):
    config = {"input_len": 128, "output_len": 4}
    row = dict(engine=engine, method="vanilla", length=128, output_len=4,
               status="success", stage_metrics_status="success",
               measurement_scope="full_batch_pure_decode_steps", actual_decode_peak=2,
               completed_requests=2, scheduler_preemptions=0, synchronize_step_timing=True,
               decode_stage_tokens=4, decode_stage_elapsed_s=0.5, decode_stage_throughput_tps=8.0)
    steps = [{"tokens": -2 if engine == "sparseengine" else 2, "elapsed_s": dt,
              "measured": True, "pure_decode": True} for dt in (0.2, 0.3)]
    case = tmp_path / "vanilla-128-2"
    case.mkdir()
    outputs = [{"request_id": i, "token_ids": [1, 2, 3, 4], "status": "success"} for i in range(2)]
    return config, row, steps, outputs, case


def persist(tmp_path, row, steps, outputs, case):
    artifact = tmp_path / "performance.jsonl"
    artifact.write_text(json.dumps(row) + "\n")
    (case / "steps.jsonl").write_text("".join(json.dumps(item) + "\n" for item in steps))
    (case / "raw_outputs.jsonl").write_text("".join(json.dumps(item) + "\n" for item in outputs))
    return artifact


@pytest.mark.parametrize("engine", ["sparseengine", "vllm", "hisparse"])
def test_reconstruct_stage_rate_from_independent_token_work(tmp_path, engine):
    config, row, steps, outputs, case = make_case(tmp_path, engine)
    artifact = persist(tmp_path, row, steps, outputs, case)
    assert validate_measurement(artifact, 2, config)["decode_throughput_tps"] == 8.0


@pytest.mark.parametrize("corruption", ["queued", "short_output", "duplicate", "prefill", "wrong_rate", "nan_time"])
def test_reject_false_success_before_plotting(tmp_path, corruption):
    config, row, steps, outputs, case = make_case(tmp_path)
    if corruption == "queued":
        row["actual_decode_peak"] = 1
    elif corruption == "short_output":
        outputs[1]["token_ids"].pop()
    elif corruption == "duplicate":
        outputs[1]["request_id"] = 0
    elif corruption == "prefill":
        steps[0]["tokens"] = 2
    elif corruption == "wrong_rate":
        row["decode_stage_throughput_tps"] = 10.0
    elif corruption == "nan_time":
        steps[0]["elapsed_s"] = float("nan")
    artifact = persist(tmp_path, row, steps, outputs, case)
    with pytest.raises(ValueError):
        validate_measurement(artifact, 2, config)


def test_reject_mixed_vllm_step_even_with_matching_aggregate(tmp_path):
    config, row, steps, outputs, case = make_case(tmp_path, "vllm")
    steps[0]["pure_decode"] = False
    artifact = persist(tmp_path, row, steps, outputs, case)
    with pytest.raises(ValueError, match="Mixed"):
        validate_measurement(artifact, 2, config)
def test_missing_graph_requires_explicit_capacity_evidence(tmp_path):
    """Do not misclassify graph wiring regressions as a concurrency limit."""
    from scripts.official_experiments.sparse_decode_efficiency.sweep_decode_capacity import capacity_failure

    log = tmp_path / "run.log"
    error = "decode CUDA Graph has no startup-captured graph for batch_size=48, path='long'."
    log.write_text("graphs=2 short=1 long=1 skipped_for_kv_capacity=0")
    assert not capacity_failure(error, log)
    log.write_text("graphs=1 short=1 long=0 skipped_for_kv_capacity=1")
    assert not capacity_failure(error, log)
    log.write_text("Startup CUDA Graph family exceeds KV capacity during prefill or decode preparation: batch=48 long=True.\n"
                   "Startup CUDA Graph capture complete: cached=1 capture_count=1 replay_count=0 skipped_for_kv_capacity=1.")
    assert capacity_failure(error, log)
    assert not capacity_failure(error.replace("48", "32"), log)
    assert not capacity_failure(error.replace("long", "short"), log)
    error = error.replace("long", "unified")
    assert not capacity_failure(error, log)
    log.write_text("Startup CUDA Graph family exceeds KV capacity during prefill or decode preparation: batch=48 path='unified'.\n")
    assert capacity_failure(error, log)
    assert not capacity_failure(error.replace("48", "32"), log)
    assert not capacity_failure(error.replace("unified", "dense"), log)
    assert not capacity_failure("kernel launch failed", log)
    log.write_text("Full decode batch capacity exceeded: scheduler preemption")
    assert capacity_failure("benchmark child exited with code -9", log)
    log.write_text("illegal memory access")
    assert not capacity_failure("benchmark child exited with code -9", log)


def test_finished_request_during_staged_admission_is_not_kv_capacity(tmp_path):
    """A short output window can expire before all waves enter decode without exhausting KV."""
    from scripts.official_experiments.sparse_decode_efficiency.sweep_decode_capacity import capacity_failure

    log = tmp_path / "run.log"
    log.write_text("")
    assert not capacity_failure(
        "Full decode batch capacity exceeded: request finished before all admission waves completed",
        log,
    )
    assert not capacity_failure(
        "Full decode batch capacity exceeded: requests finished before full decode admission",
        log,
    )
    assert capacity_failure(
        "Full decode batch capacity exceeded: not all requests entered decode together",
        log,
    )


@pytest.mark.parametrize("change", [None, "protocol", "command", "gpu", "model", "hyperparams", "gpu_equivalent", "gpu_mismatch"])
def test_boundary_resume_rejects_changed_workload_identity(tmp_path, monkeypatch, change):
    """Hash-free run records must still reject changed workloads and devices."""
    import hashlib
    from scripts.official_experiments.sparse_decode_efficiency.sweep_decode_capacity import validate_boundary_reuse

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    previous = tmp_path / "old" / "sengine-snapkv" / "bs2"
    previous.mkdir(parents=True)
    config = {"measurement_protocol": "boundary_sync_v2"}
    (previous.parent.parent / "campaign.json").write_text(json.dumps(config))
    hp = {"decode_graph": True}
    (previous / "hyper_params.json").write_text(json.dumps(hp))
    command = ["python", "probe.py", "--model-path", str(model), "--hyper-params", "@new.json", "--output-dir", "new"]
    old_command = command.copy()
    old_command[-1] = "old"
    old_command[-3] = "@old.json"
    identity = dict(env={"CUDA_VISIBLE_DEVICES": "5"}, command=old_command,
                    model_config_sha256=digest(model / "config.json"))
    (previous / "identity.json").write_text(json.dumps(identity))
    if change == "protocol":
        config["measurement_protocol"] = "step_sync_v1"
    elif change == "command":
        command += ["--synchronize-step-timing"]
    elif change == "model":
        (model / "config.json").write_text('{"changed": true}')
    elif change == "hyperparams":
        hp["decode_graph"] = False
    if change in ("gpu_equivalent", "gpu_mismatch"):
        from scripts.official_experiments.sparse_decode_efficiency import sweep_decode_capacity as sweep
        records = iter(["identical hardware", "identical hardware" if change == "gpu_equivalent" else "different hardware"])
        monkeypatch.setattr(sweep.subprocess, "check_output", lambda *a, **kw: next(records))
        if change == "gpu_equivalent":
            result = validate_boundary_reuse(previous, config, hp, command, "4", allow_equivalent_gpus=True)
            assert result["reuse_gpu_comparison"]["source_gpus"] == "5"
            assert result["reuse_gpu_comparison"]["target_gpus"] == "4"
        else:
            with pytest.raises(ValueError, match="hardware/configuration mismatch"):
                validate_boundary_reuse(previous, config, hp, command, "4", allow_equivalent_gpus=True)
    elif change is None:
        assert validate_boundary_reuse(previous, config, hp, command, "5") == identity
    else:
        with pytest.raises(ValueError):
            validate_boundary_reuse(previous, config, hp, command, "4" if change == "gpu" else "5")


@pytest.mark.parametrize("corruption", ["missing_failure", "wrong_boundary", "unobserved_point"])
def test_replot_rejects_unproven_capacity_before_rendering(tmp_path, monkeypatch, corruption):
    """A valid throughput alone must not turn an unbounded sweep into a maximum."""
    from scripts.official_experiments.sparse_decode_efficiency import plot_decode_capacity as plot

    curves = []
    for lane in plot.LANES:
        curves.append({
            "model": "fixture", "lane": lane, "max_concurrency": 1,
            "first_failed_concurrency": 2,
            "attempts": [{"concurrency": 1, "status": "success"},
                         {"concurrency": 2, "status": "capacity_exceeded"}],
            "points": [{"concurrency": 1, "decode_tokens": 2,
                        "decode_elapsed_s": 0.5, "decode_throughput_tps": 4.0}],
        })
    if corruption == "missing_failure":
        curves[0]["attempts"].pop()
    elif corruption == "wrong_boundary":
        curves[0]["first_failed_concurrency"] = 3
    else:
        curves[0]["attempts"].pop(0)
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"schema_version": 1, "config": {"models": {"fixture": {}}},
                               "curves": curves}))
    monkeypatch.setattr("sys.argv", ["plot", "--plot-data", str(path)])
    def unexpected_render(*args, **kwargs):
        pytest.fail("Invalid capacity export reached rendering")
    monkeypatch.setattr(plot, "render", unexpected_render)
    with pytest.raises(ValueError, match="capacity boundary|successful attempts"):
        plot.main()


@pytest.mark.parametrize("corruption", ["missing_evidence", "fabricated_zero"])
def test_unsupported_export_cannot_hide_missing_evidence_or_become_zero(tmp_path, monkeypatch, corruption):
    """A non-runnable combination must not silently become a measured data point."""
    from scripts.official_experiments.sparse_decode_efficiency import plot_decode_capacity as plot

    reason = "fixture rejects this attention contract"
    unsupported = {lane: reason for lane in plot.LANES}
    curves = [{"model": "fixture", "lane": lane, "status": "unsupported",
               "reason": reason, "points": [], "attempts": []} for lane in plot.LANES]
    if corruption == "missing_evidence":
        unsupported.pop(curves[0]["lane"])
    else:
        curves[0]["points"] = [{"concurrency": 1, "decode_throughput_tps": 0}]
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"schema_version": 1,
        "config": {"models": {"fixture": {}}, "unsupported": {"fixture": unsupported}},
        "curves": curves}))
    monkeypatch.setattr("sys.argv", ["plot", "--plot-data", str(path)])
    monkeypatch.setattr(plot, "render", lambda *a, **kw: pytest.fail("Invalid N/A reached rendering"))
    with pytest.raises(ValueError, match="Unsupported curve"):
        plot.main()


@pytest.mark.parametrize("corruption", [None, "claims_maximum", "unapproved_partial", "unmeasured_point"])
def test_partial_replot_does_not_turn_theoretical_capacity_into_measurement(tmp_path, monkeypatch, corruption):
    """Stopping a sweep may export real points, never invent an endpoint or max+1."""
    from scripts.official_experiments.sparse_decode_efficiency import plot_decode_capacity as plot
    config = {"models": {"fixture": {}}, "partial_curves": {"fixture": {}},
              "input_len": 128, "output_len": 4}
    curves = []
    for lane in plot.LANES:
        config["partial_curves"]["fixture"][lane] = dict(reason="user stopped", theoretical_max_concurrency=2)
        curves.append(dict(model="fixture", lane=lane, status="partial", reason="user stopped",
            max_concurrency=None, first_failed_concurrency=None, max_measured_concurrency=1,
            theoretical_max_concurrency=2, attempts=[dict(concurrency=1, status="success")],
            points=[dict(concurrency=1, decode_tokens=2, decode_elapsed_s=.5,
                         decode_throughput_tps=4., measured_steps=2)]))
    if corruption == "claims_maximum":
        curves[0]["max_concurrency"] = 2
    elif corruption == "unapproved_partial":
        config.pop("partial_curves")
    elif corruption == "unmeasured_point":
        curves[0]["points"].append({**curves[0]["points"][0], "concurrency": 2})
    source = tmp_path / "input.json"
    source.write_text(json.dumps(dict(schema_version=1, config=config, curves=curves)))
    output = tmp_path / "plots"
    monkeypatch.setattr("sys.argv", ["plot", "--plot-data", str(source), "--output-dir", str(output)])
    calls = []
    def render(*args, **kwargs):
        calls.append(args)
        output.mkdir()
    monkeypatch.setattr(plot, "render", render)
    if corruption:
        with pytest.raises(ValueError, match="Partial curve|successful attempts"):
            plot.main()
        assert not calls
    else:
        plot.main()
        assert calls
        assert json.loads((output / "plot_data.json").read_text())["curves"][0]["max_concurrency"] is None


@pytest.mark.parametrize("marker", ["capacity", "validity_file"])
def test_partial_curve_rejects_gpu_contention_before_loading_points(tmp_path, monkeypatch, marker):
    """An explicit partial-curve policy must not publish marked-invalid measurements."""
    from scripts.official_experiments.sparse_decode_efficiency import plot_decode_capacity as plot

    lane = "sengine-snapkv"
    capacity = tmp_path / "fixture" / "attempt" / lane / "capacity.json"
    capacity.parent.mkdir(parents=True)
    evidence = {"status": "failed", "attempts": [{"concurrency": 1,
                "status": "success", "artifact": str(tmp_path / "raw.jsonl")}]}
    if marker == "capacity":
        evidence["validity_status"] = "potentially_invalid_external_contention"
    else:
        (capacity.parent / "validity.json").write_text(json.dumps({
            "status": "potentially_invalid_external_contention"}))
    capacity.write_text(json.dumps(evidence))
    config = {"output_root": str(tmp_path), "models": {"fixture": {}},
              "unsupported": {"fixture": {name: "not measured" for name in plot.LANES if name != lane}},
              "partial_curves": {"fixture": {lane: {
                  "reason": "queue stopped", "capacity_artifact": str(capacity.relative_to(tmp_path))}}}}
    monkeypatch.setattr(plot, "validate_measurement",
                        lambda *_args, **_kwargs: pytest.fail("Invalid measurement reached validation"))

    with pytest.raises(ValueError, match="Capacity evidence is marked invalid"):
        plot.load_campaign(config)


def test_grid_preserves_per_panel_protocol_and_explicit_missing_curve(tmp_path, monkeypatch):
    """Combining context lengths must not validate both panels against one length."""
    from scripts.official_experiments.sparse_decode_efficiency import plot_decode_capacity as plot
    panels, seen = [], []
    for length in (128, 32):
        source = tmp_path / f"config-{length}.json"
        source.write_text(json.dumps(dict(models={"fixture": {}}, input_len=length,
            output_len=4, output_root=str(tmp_path), measurement_protocol="boundary_sync_v2")))
        panels.append(dict(config=source.name, model="fixture", curves={
            "sengine-snapkv": dict(capacity_artifact="missing.json", allow_partial=True,
                                  reason="not measured yet; do not invent a result")}))
    def collect(config):
        seen.append(config)
        return [dict(model="fixture", lane=lane, points=[dict(concurrency=1), dict(concurrency=2)])
                for lane in ("sengine-vanilla", "vllm-vanilla")]
    monkeypatch.setattr(plot, "load_campaign", collect)
    monkeypatch.setattr(plot, "validate_grid_identity", lambda *args: None)
    manifest = tmp_path / "grid.json"
    manifest.write_text(json.dumps(dict(shape=[2, 1], panels=panels)))
    config, curves = plot.load_grid(manifest)
    assert [c["input_len"] for c in seen] == [128, 32]
    assert {c["model"] for c in curves} == set(config["panel_protocols"])
    assert seen[0]["omitted_curves"]["fixture"]["sengine-snapkv"]["source_status"] == "not_measured"
    panels[0]["curves"]["sengine-snapkv"]["allow_partial"] = False
    manifest.write_text(json.dumps(dict(shape=[2, 1], panels=panels)))
    with pytest.raises(ValueError, match="requires a completed"):
        plot.load_grid(manifest)
    panels[0]["curves"]["sengine-snapkv"]["allow_partial"] = True
    manifest.write_text(json.dumps(dict(shape=[2, 1], panels=panels)))
    (tmp_path / "missing.json").write_text("not valid json")
    with pytest.raises(json.JSONDecodeError):
        plot.load_grid(manifest)


def test_grid_rejects_empty_comparison_panels(tmp_path, monkeypatch):
    """Smoke-only or missing formal data cannot satisfy figure acceptance."""
    from scripts.official_experiments.sparse_decode_efficiency import plot_decode_capacity as plot
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(models={"fixture": {}}, input_len=128,
        output_root=str(tmp_path), measurement_protocol="boundary_sync_v2")))
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps(dict(shape=[1, 1], panels=[
        dict(config=source.name, model="fixture", curves={})])))
    monkeypatch.setattr(plot, "load_campaign", lambda c: [])
    with pytest.raises(ValueError, match="two measured curves"):
        plot.load_grid(grid)


@pytest.mark.parametrize("wrong_flag", [None, "--model-path", "--tensor-parallel-size"])
def test_grid_rejects_valid_measurement_from_wrong_model_or_topology(tmp_path, wrong_flag):
    from scripts.official_experiments.sparse_decode_efficiency.plot_decode_capacity import validate_grid_identity
    command = ["python", "probe.py", "--model-path", "model-a", "--tensor-parallel-size", "2",
               "--expert-parallel-size", "2"]
    if wrong_flag:
        command[command.index(wrong_flag) + 1] = "wrong"
    (tmp_path / "identity.json").write_text(json.dumps(dict(command=command, model_config_sha256="fixture")))
    config = dict(models={"a": dict(path="model-a", tp=2, ep=2)})
    point = dict(artifact=str(tmp_path / "performance.jsonl"))
    if wrong_flag:
        with pytest.raises(ValueError, match="model/topology mismatch"):
            validate_grid_identity(point, "a", config)
    else:
        validate_grid_identity(point, "a", config)
        assert point["model_identity"]["model"] == "a"


def test_relative_delta_uses_baseline_denominator_and_exact_batches():
    """Catch reversed ratios and invented baseline interpolation between samples."""
    from scripts.official_experiments.sparse_decode_efficiency.plot_decode_capacity import relative_vllm_points
    config = {'models': {'m': {}}, 'input_len': 100, 'output_len': 10}
    curves = [dict(model='m', lane='vllm-vanilla', points=[
        dict(concurrency=1, decode_throughput_tps=100),
        dict(concurrency=4, decode_throughput_tps=200)]),
        dict(model='m', lane='sengine-snapkv', points=[
            dict(concurrency=1, decode_throughput_tps=150),
            dict(concurrency=2, decode_throughput_tps=180),
            dict(concurrency=4, decode_throughput_tps=150)])]
    rows = [r for r in relative_vllm_points(curves, config) if r['lane'] == 'sengine-snapkv']
    assert rows[0]['delta_percent'] == pytest.approx(50)
    assert rows[1]['status'] == 'skipped_by_policy'
    assert rows[1]['delta_percent'] is None
    assert rows[2]['delta_percent'] == pytest.approx(-25)


def test_relative_delta_excludes_changed_input_output_protocol():
    """Same batch and total span do not make adjusted contexts a matched pair."""
    from scripts.official_experiments.sparse_decode_efficiency.plot_decode_capacity import relative_vllm_points
    config = dict(models={'m': {}}, input_len=100, output_len=10,
                  curve_protocols={'m': {'sengine-snapkv': dict(input_len=90, output_len=20)}})
    curves = [dict(model='m', lane=lane, points=[dict(concurrency=1, decode_throughput_tps=100)])
              for lane in ('vllm-vanilla', 'sengine-snapkv')]
    row = relative_vllm_points(curves, config)[1]
    assert row['reason'] == 'protocol_mismatch' and row['delta_percent'] is None


def test_relative_delta_rejects_nonpositive_baseline():
    """Reject invalid denominators rather than exporting infinite plotted gains."""
    from scripts.official_experiments.sparse_decode_efficiency.plot_decode_capacity import relative_vllm_points
    curves = [dict(model='m', lane='vllm-vanilla', points=[dict(concurrency=1, decode_throughput_tps=0)])]
    with pytest.raises(ValueError, match='finite positive'):
        relative_vllm_points(curves, {'models': {'m': {}}})
