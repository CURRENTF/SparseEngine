"""Protect protocol forwarding and reject failed repetitions without GPU mocks."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from benchmark.efficiency.paper import run_paper_decode


def test_length_jobs_cannot_collide_in_per_batch_exports(tmp_path):
    """A completed128K BS1 previously collided with32K BS1 and stopped its GPU queue."""
    from scripts.official_experiments.sparse_decode_efficiency.prepare_vortex import validate_export_namespaces
    def job(name, export):
        return dict(name=name, command=["runner", "--export-measurements-dir", str(export),
                                       "--model", "same-model", "--attempt", "same-attempt"])
    first = job("short", tmp_path / "shared")
    with pytest.raises(ValueError, match="Export namespace collision"):
        validate_export_namespaces([first, job("long", tmp_path / "shared")])
    validate_export_namespaces([first, job("long", tmp_path / "long")])


def test_missing_boundary_probes_cannot_turn_generic_failure_into_maximum():
    """Partial-curve assembly needs measured max+1, not merely a larger failure."""
    from scripts.official_experiments.sparse_decode_efficiency.sweep_decode_capacity import LaneFailure, probe_capacity_boundary
    attempts = [dict(concurrency=b, status="success") for b in (1, 2, 4, 5)]
    with pytest.raises(LaneFailure):
        probe_capacity_boundary(attempts + [dict(concurrency=8, status="capacity_exceeded")])
    with pytest.raises(LaneFailure):
        probe_capacity_boundary(attempts + [dict(concurrency=6, status="failed")])
    assert probe_capacity_boundary(attempts + [dict(concurrency=6, status="capacity_exceeded")]) == (5, 6)
    with pytest.raises(LaneFailure):
        probe_capacity_boundary(attempts + [attempts[0], dict(concurrency=6, status="capacity_exceeded")])


def test_kv_hint_uses_limiting_rank_and_is_not_a_capacity_result(tmp_path):
    """A larger rank's KV pool must not overestimate the shared full-residency batch."""
    from scripts.official_experiments.sparse_decode_efficiency.sweep_decode_capacity import full_kv_capacity_hint
    (tmp_path / "rank.log").write_text("rank=0 kv_slots=901\nrank=1 kv_slots=799\n")
    assert full_kv_capacity_hint(tmp_path, 100) == 7
    assert full_kv_capacity_hint(tmp_path / "missing", 100) is None


def test_relocated_sweep_preflights_statistics_from_explicit_checkout(tmp_path):
    """The live continuation previously reached GPU smoke then failed importing metrics."""
    import os
    import shutil
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[1]
    package = repo / "scripts/official_experiments/sparse_decode_efficiency"
    relocated = tmp_path / "standalone"
    relocated.mkdir()
    for name in ("sweep_decode_capacity.py", "plot_decode_capacity.py"):
        shutil.copyfile(package / name, relocated / name)
    (tmp_path / "config.json").write_text("{}")
    config = dict(conda=sys.executable, native_env=str(tmp_path), vllm_env=str(tmp_path),
        output_root=str(tmp_path / "output"), scratch_root=str(tmp_path),
        models={"fixture": {"path": str(tmp_path)}}, measurement_protocol="boundary_sync_v2")
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, str(relocated / "sweep_decode_capacity.py"),
        "--repo", str(repo), "--config", str(campaign), "--model", "fixture", "--gpus", "none",
        "--lanes", "sengine-vanilla", "--check-only"], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("phase", ["smoke", "sweep"])
def test_failed_method_does_not_block_other_methods(phase):
    """A failed GLM lane used to prevent every later method from running."""
    from scripts.official_experiments.sparse_decode_efficiency.sweep_decode_capacity import LaneFailure, run_lane_stages
    calls, recorded = [], {}

    def execute(stage, lane):
        calls.append((stage, lane))
        if lane == "broken" and stage == phase:
            raise LaneFailure("invalid window; not capacity evidence")

    failures = run_lane_stages(["first", "broken", "last"],
        lambda lane: execute("smoke", lane), lambda lane: execute("sweep", lane),
        lambda lane, value: recorded.update({lane: value}), lambda: None)
    assert failures == recorded == {"broken": {"phase": phase, "error": "invalid window; not capacity evidence"}}
    assert calls.count(("sweep", "first")) == calls.count(("sweep", "last")) == 1
    assert (("sweep", "broken") in calls) == (phase == "sweep")


@pytest.mark.parametrize("fatal", [RuntimeError("GPU contention"), OSError("disk full"), InterruptedError("user stop")])
def test_infrastructure_and_user_stop_do_not_continue_queue(fatal):
    """Only explicit lane failures are recoverable; resource errors must escape."""
    from scripts.official_experiments.sparse_decode_efficiency.sweep_decode_capacity import run_lane_stages
    calls = []

    def fail(lane):
        calls.append(lane)
        raise fatal

    with pytest.raises(type(fatal), match=str(fatal)):
        run_lane_stages(["first", "must_not_run"], fail, fail, lambda *a: None, lambda: None)
    assert calls == ["first"]


def test_resource_loss_after_lane_failure_stops_before_continuation():
    """A crashed model must not hide a concurrently lost GPU reservation."""
    from scripts.official_experiments.sparse_decode_efficiency.sweep_decode_capacity import LaneFailure, run_lane_stages
    checks, recorded = [], []

    def guard():
        checks.append(True)
        if len(checks) > 1:
            raise RuntimeError("reservation exited")

    def fail(lane):
        raise LaneFailure("model crash")

    with pytest.raises(RuntimeError, match="reservation exited"):
        run_lane_stages(["first", "last"], fail, fail, lambda *a: recorded.append(a), guard)
    assert not recorded


def test_legacy_continuation_never_retries_started_or_failed_methods(tmp_path):
    """Deploy a new queue without rerunning an expensive completed/partial curve."""
    from scripts.official_experiments.sparse_decode_efficiency.continue_failed_queue import remaining_lanes
    (tmp_path / "complete" / "bs1").mkdir(parents=True)
    (tmp_path / "broken" / "bs2").mkdir(parents=True)
    rows = [dict(stage="broken/bs2", status="failed"),
            dict(stage="queue", status="failed", error="RuntimeError('Benchmark failed, inspect case/run.log')")]
    assert remaining_lanes(tmp_path, ["complete", "broken", "untouched"], rows) == ["untouched"]
    rows[-1]["error"] = "InterruptedError('Queue interrupted by signal 15')"
    with pytest.raises(RuntimeError, match="not an explicitly classified method error"):
        remaining_lanes(tmp_path, ["complete", "broken", "untouched"], rows)
    rows[-1].update(status="completed")
    assert remaining_lanes(tmp_path, ["complete", "untouched"], rows) == []
    assert remaining_lanes(tmp_path, ["untouched"], rows, explicit_followup=True) == ["untouched"]
    with pytest.raises(RuntimeError, match="must not repeat"):
        remaining_lanes(tmp_path, ["complete"], rows, explicit_followup=True)
    rows[-1]["status"] = "failed"
    with pytest.raises(RuntimeError, match="successfully completed"):
        remaining_lanes(tmp_path, ["untouched"], rows, explicit_followup=True)


def test_followup_accepts_only_persisted_lane_failures_not_infrastructure(tmp_path):
    """An optional baseline crash must not prevent unrelated planned GPU work."""
    from scripts.official_experiments.sparse_decode_efficiency.continue_failed_queue import remaining_lanes
    failure = dict(phase="sweep", error="benchmark failed")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "lane_failure.json").write_text(json.dumps(failure))
    (tmp_path / "queue_summary.json").write_text(json.dumps(dict(
        status="failed", lanes=["broken"], failures={"broken": failure})))
    rows = [dict(stage="queue", status="failed",
                 error='RuntimeError("Lane failures retained after continuing other methods: [broken]")')]
    kwargs = dict(explicit_followup=True, allow_failed_methods=True)
    assert remaining_lanes(tmp_path, ["untouched"], rows, **kwargs) == ["untouched"]
    rows[0]["error"] = "RuntimeError('GPU reservation exited; inspect guard.log')"
    with pytest.raises(RuntimeError, match="successfully completed"):
        remaining_lanes(tmp_path, ["untouched"], rows, **kwargs)


def test_status_reader_ignores_only_in_progress_tail(tmp_path):
    """Polling an append-only status file must not parse an unfinished write."""
    from scripts.official_experiments.sparse_decode_efficiency.continue_failed_queue import read_status
    path = tmp_path / "status.tsv"
    path.write_text('now\tqueue\tcompleted\t2,5\t{}\npartial')
    assert read_status(path) == [dict(time="now", stage="queue", status="completed", gpus="2,5")]
    path.write_text('now\tqueue\tcompleted\t2,5\tbad-json\n')
    with pytest.raises(json.JSONDecodeError):
        read_status(path)


def test_reservation_handoff_pins_only_verified_guard_and_rejects_other_gpu_pids(tmp_path):
    """Wrong parent/foreign GPU PIDs must be rejected before sending any stop."""
    import os
    import shutil
    import subprocess
    script = tmp_path / "decode_capacity_guard.py"
    script.write_text("import time; print('ready', flush=True); time.sleep(30)\n")
    # The system interpreter owns orchestration; some conda builds omit pidfd_open.
    python = shutil.which("python3", path=os.defpath)
    subprocess.run([python, "-c", """
import os, subprocess, sys
from scripts.official_experiments.sparse_decode_efficiency import continue_failed_queue as c
p = subprocess.Popen([sys.executable, sys.argv[1], '--parent', str(os.getpid())], stdout=subprocess.PIPE, text=True)
try:
    assert p.stdout.readline().strip() == 'ready'
    c.subprocess.check_output = lambda *a, **kw: str(p.pid)
    fd = c.pin_guard(p.pid, os.getpid(), 'fixture')
    os.close(fd)
    try:
        c.pin_guard(p.pid, os.getpid() + 1, 'fixture')
    except RuntimeError as e:
        assert 'identity mismatch' in str(e)
    else:
        raise AssertionError('accepted wrong owner')
    c.subprocess.check_output = lambda *a, **kw: f'{p.pid}\\n1'
    try:
        c.pin_guard(p.pid, os.getpid(), 'fixture')
    except RuntimeError as e:
        assert 'no active model or foreign' in str(e)
    else:
        raise AssertionError('accepted foreign GPU process')
    assert p.poll() is None
finally:
    p.terminate()
    p.wait(timeout=5)
""", str(script)], check=True, timeout=10)


def test_external_provenance_creates_case_before_model_start(tmp_path):
    from benchmark.efficiency.paper import record_package_source
    source = tmp_path / "package.py"
    source.write_text("# fixture package\n")
    case = tmp_path / "not-created" / "case"
    identity = record_package_source(NS(__file__=str(source)), case)
    assert json.loads((case / "external_source.json").read_text()) == identity


def test_legacy_source_cleanup_preserves_measurements_and_artifact_checksums():
    """Old exports must not reintroduce source inventories or lose raw-data identity."""
    from benchmark.efficiency.paper import without_source_fingerprints
    repetition = dict(repetition=0, window={"decode_stage_tokens": 6},
        source_sha256={"raw_outputs.jsonl": "raw-digest"}, decode_elapsed_s=2.5,
        package_identity={"git_head": "old-commit", "package_file_sha256": "code-digest"})
    legacy = dict(run_manifest={"git_head": "old-commit", "source_sha256": {"runner.py": "code"},
        "runtime_source_sha256": {"src/kernel.py": "kernel"}, "prompt_token_ids_sha256": {"12": "trace"}},
        repetitions=[repetition])
    clean = without_source_fingerprints(legacy)
    assert clean["run_manifest"] == {"git_head": "old-commit", "prompt_token_ids_sha256": {"12": "trace"}}
    assert clean["repetitions"] == [{**repetition, "package_identity": {"git_head": "old-commit"}}]
    assert without_source_fingerprints(clean) == clean
    assert "source_sha256" in legacy["run_manifest"]


def arguments(tmp_path):
    return NS(scenario="fixed", prompt_length_jitter=0, output_length_jitter=0,
        decode_only_steps=3, decode_only_warmup_steps=1, num_iters=2, num_warmups=1,
        seed=42, prompt_lens=[12], output_lens=[9], batch_sizes=[2],
        tensor_parallel_size=2, expert_parallel_size=2, gpu_memory_utilization=.9,
        max_num_batched_tokens=8192, prefill_wave_size=1, wave_decode_gap_steps=1,
        hyper_params="{}", engine_kwargs="{}", sparse_prefill_score_mode=None,
        output_dir=str(tmp_path / "run"), engine="sparseengine", model_path="unused",
        sparse_method="snapkv", backend_label=None)


@pytest.mark.parametrize("failed", [False, True])
def test_reuse_adapter_preserves_tp_wave_window_and_failure(monkeypatch, tmp_path, failed):
    import benchmark.efficiency.paper as paper
    monkeypatch.setattr(paper.subprocess, "check_output", lambda *a, **kw: "fixture" if kw.get("text") else b"fixture")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        value = lambda key: command[command.index(key) + 1]
        assert value("--decode_window_steps") == "3"
        assert value("--admission_wave_size") == "1"
        assert value("--wave_decode_gap_steps") == "1"
        assert "--synchronize_step_timing" not in command
        hp = json.loads(value("--hyper_params"))
        assert hp["tensor_parallel_size"] == hp["expert_parallel_size"] == 2
        output = Path(value("--output_dir"))
        output.mkdir()
        dt = (99, 2, 4)[len(calls)-1]
        window = dict(decode_stage_tokens=6, decode_stage_elapsed_s=dt)
        row = dict(engine="sparseengine", method="snapkv", length=12, output_len=9, batch_size=2,
            stage_timing_scope="fixture", decode_warmup_steps_after_full=1,
            measured_decode_steps_after_full=3, status="success", synchronize_step_timing=False,
            measurement_scope="full_batch_decode_window", decode_window=window,
            decode_stage_tokens=6, decode_stage_elapsed_s=dt, decode_stage_throughput_tps=6/dt)
        if failed:
            row.update(status="model_failed", error="allocator out of memory")
        (output / "performance.jsonl").write_text(json.dumps(row) + "\n")
        return NS(returncode=int(failed))

    monkeypatch.setattr(paper.subprocess, "run", run)
    args = arguments(tmp_path)
    if failed:
        with pytest.raises(RuntimeError, match="allocator out of memory"):
            run_paper_decode(args)
        result = json.loads((Path(args.output_dir) / "performance.jsonl").read_text())
        assert result["status"] == "model_failed" and len(calls) == 1
    else:
        run_paper_decode(args)
        result = json.loads((Path(args.output_dir) / "performance.jsonl").read_text())
        assert len(calls) == 3 and len(result["repetitions"]) == 2
        assert result["decode_stage_throughput_tps"] == 2
        with pytest.raises(FileExistsError):
            run_paper_decode(args)
