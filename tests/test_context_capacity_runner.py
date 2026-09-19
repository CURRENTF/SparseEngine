"""Campaign control-flow regressions; subprocess fixtures never launch GPUs."""
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from scripts.official_experiments.sparse_decode_efficiency import run_context_capacity as runner


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / "output"
    config = dict(
        output_root=str(root), scratch_root=str(tmp_path / "scratch"),
        model_id="model", models={"model": dict(path=str(tmp_path / "model"), tp=1, ep=1)},
        lanes=["sengine-vanilla", "vllm-vanilla"], input_lens=[128, 256],
        conda=str(tmp_path / "conda"), native_env=str(tmp_path / "native"),
        vllm_env=str(tmp_path / "vllm"), output_len=4,
        gpu_memory_utilization=.9, num_warmups=1, num_iters=1, case_timeout_s=11,
    )
    state = SimpleNamespace(config=config, root=root, processes=[], requests=[], sweeps=[],
                            signals=[], clock=0., fault=None, timed_out=False, guard=None)
    # Resolved per-case settings may differ from an omitted/global campaign
    # setting; request probes must reuse the measured case's actual settings.
    state.hyper_params = dict(max_num_batched_tokens=37, engine_prefill_chunk_size=29)

    class Process:
        def __init__(self, command, *, env, **kwargs):
            self.command, self.env = command, dict(env)
            self.pid = 10000 + len(state.processes)
            self.code, self.polls = None, 0
            state.processes.append(self)
            if any(str(arg).endswith("/sweep_decode_capacity.py") for arg in command):
                self.kind = "sweep"
                cfg = json.loads(Path(command[command.index("--config") + 1]).read_text())
                state.sweeps.append(cfg)
                sweep = Path(cfg["output_root"]) / cfg["model_id"] / ("_".join(cfg["lanes"]) + "-capacity")
                runner.write(sweep / "queue_summary.json", dict(status="completed", failures={}))
                for lane in cfg["lanes"]:
                    runner.write(sweep / lane / "capacity.json", dict(
                        status="completed", max_concurrency=1, first_failed_concurrency=2))
                    case = sweep / lane / "bs1"
                    runner.write(case / "hyper_params.json", state.hyper_params)
                    runner.write(case / "performance.jsonl", {})
                    external = cfg.get("external_lanes", {}).get(lane)
                    if external:
                        runner.write(case / "engine_kwargs.json", external["engine_kwargs"])
            elif any(str(arg).endswith("/decode_capacity_guard.py") for arg in command):
                self.kind = "guard"
                self.ready = Path(command[command.index("--ready") + 1])
                runner.write(self.ready, dict(pid=self.pid))
                state.guard = self
            else:
                self.kind = "request"
                self.timeout = state.fault == "timeout" and not state.timed_out
                state.timed_out |= self.timeout
                state.requests.append(self)

        def poll(self):
            self.polls += 1
            if self.kind == "guard" and state.fault == "guard_before_launch" and self.polls > 1:
                self.code = 1
            return self.code

        def wait(self, timeout):
            if self.code is not None:
                return self.code
            if self.kind == "request":
                if self.timeout or state.fault in ("guard_during_wait", "contention"):
                    state.clock += timeout
                    if state.fault == "guard_during_wait":
                        state.guard.code = 1
                    if state.fault == "contention":
                        runner.write(state.guard.ready.with_suffix(".contention.json"), dict(foreign_pids=[1]))
                    raise subprocess.TimeoutExpired(self.command, timeout)
                dest = Path(self.command[self.command.index("--output-dir") + 1])
                runner.write(dest / "run_status.json", dict(status="success"))
                runner.write(dest / "summary.json", dict(status="success"))
                (dest / "request_samples.jsonl").write_text(json.dumps(dict(
                    status="success", generated_tokens=config["output_len"])) + "\n")
                if state.fault == "guard_at_completion":
                    state.guard.code = 1
            state.clock += .01
            self.code = 0
            return self.code

    def killpg(pid, signum):
        state.signals.append((pid, signum))
        next(process for process in state.processes if process.pid == pid).code = -signum

    def query(command, **kwargs):
        return "" if command[0] == "nvidia-smi" else "fixture-git"

    config_path = tmp_path / "config.json"

    def run():
        config_path.write_text(json.dumps(config))
        return runner.main()

    state.run = run
    monkeypatch.setattr(runner.sys, "argv", [
        "runner", "--config", str(config_path), "--repo", str(tmp_path), "--gpus", "0"])
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner.subprocess, "check_output", query)
    monkeypatch.setattr(runner.os, "killpg", killpg)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runner.time, "monotonic", lambda: state.clock)
    return state


def test_template_variables_and_resolved_batch_budget_reach_request_probe(campaign, monkeypatch):
    """Template variables and an omitted batch-token override used to abort after the sweep."""
    monkeypatch.setenv("CONTEXT_REVIEW_OUTPUT", str(campaign.root))
    monkeypatch.setenv("CONTEXT_REVIEW_CONDA", campaign.config["conda"])
    campaign.config.update(output_root="${CONTEXT_REVIEW_OUTPUT}", conda="${CONTEXT_REVIEW_CONDA}")
    assert campaign.run() == 0
    saved = json.loads((campaign.root / "campaign.json").read_text())
    assert saved["output_root"] == str(campaign.root)
    assert "${" not in json.dumps(saved)
    for request in campaign.requests:
        command = request.command
        assert command[0] == saved["conda"]
        assert int(command[command.index("--max-num-batched-tokens") + 1]) == campaign.hyper_params["max_num_batched_tokens"]
    assert len((campaign.root / "results.jsonl").read_text().splitlines()) == 4


def test_unresolved_template_fails_before_artifact_creation(campaign, monkeypatch):
    """A missing variable must not create a directory named literally ${...}."""
    monkeypatch.delenv("CONTEXT_REVIEW_MISSING", raising=False)
    campaign.config["output_root"] = "${CONTEXT_REVIEW_MISSING}"
    with pytest.raises(ValueError, match="CONTEXT_REVIEW_MISSING"):
        campaign.run()
    assert not campaign.processes
    assert not campaign.root.exists()


@pytest.mark.parametrize("fault", ["guard_before_launch", "guard_during_wait", "guard_at_completion", "contention"])
def test_lost_gpu_reservation_aborts_and_rejects_request_results(campaign, fault):
    """A dead guard or contention must invalidate even an otherwise successful probe."""
    campaign.fault = fault
    with pytest.raises(RuntimeError, match="reservation exited|contention"):
        campaign.run()
    assert json.loads((campaign.root / "queue_status.json").read_text())["status"] == "aborted"
    assert len(campaign.requests) <= 1
    assert not (campaign.root / "results.jsonl").exists()
    if fault in ("guard_during_wait", "contention"):
        assert (campaign.requests[0].pid, signal.SIGTERM) in campaign.signals
    assert all(process.poll() is not None for process in campaign.processes)


def test_request_timeout_records_failure_cleans_up_and_continues(campaign):
    """One hung method must not drop subsequent lanes/lengths or silently retry the point."""
    campaign.fault = "timeout"
    assert campaign.run() == 1
    status = json.loads((campaign.root / "queue_status.json").read_text())
    assert status["status"] == "failed"
    assert len(status["failures"]) == 1
    assert status["failures"][0]["error"] == "request timeout"
    assert (campaign.requests[0].pid, signal.SIGTERM) in campaign.signals
    assert len(campaign.requests) == 4
    results = [json.loads(row) for row in (campaign.root / "results.jsonl").read_text().splitlines()]
    assert {(r["input_tokens"], r["lane"]) for r in results} == {
        (128, "vllm-vanilla"), (256, "sengine-vanilla"), (256, "vllm-vanilla")}
    assert all(process.poll() is not None for process in campaign.processes)


@pytest.mark.parametrize("environment_kind", ["venv", "conda"])
def test_external_request_uses_measured_engine_environment_and_kwargs(campaign, tmp_path, environment_kind):
    """A successful external capacity run used to be followed by a native request probe."""
    prefix = tmp_path / "external"
    prefix.mkdir()
    (prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    external = dict(engine="vllm", method="snapkv", env=str(prefix), environment_kind=environment_kind,
                    backend_label="tangram-snapkv", engine_kwargs={"compression_scorer": "snapkv"},
                    environment={"REVIEW_EXTERNAL_SETTING": "enabled"}, pythonpath=[str(tmp_path / "vendor")])
    campaign.config.update(lanes=["tangram-snapkv", "sengine-vanilla"], input_lens=[128],
                           external_lanes={"tangram-snapkv": external})
    assert campaign.run() == 0
    request, native = campaign.requests
    command = request.command
    assert command[command.index("--engine") + 1] == external["engine"]
    assert command[command.index("--sparse-method") + 1] == external["method"]
    assert command[command.index("--backend-label") + 1] == external["backend_label"]
    kwargs = Path(command[command.index("--engine-kwargs") + 1].removeprefix("@"))
    assert json.loads(kwargs.read_text()) == external["engine_kwargs"]
    assert request.env["REVIEW_EXTERNAL_SETTING"] == "enabled"
    assert external["pythonpath"][0] in request.env["PYTHONPATH"].split(os.pathsep)
    assert "REVIEW_EXTERNAL_SETTING" not in native.env
    assert external["pythonpath"][0] not in native.env["PYTHONPATH"].split(os.pathsep)
    if environment_kind == "venv":
        assert command[0] == str(prefix / "bin/python")
        assert request.env["VIRTUAL_ENV"] == str(prefix)
        assert request.env["PATH"].split(os.pathsep)[0] == str(prefix / "bin")
    else:
        assert command[command.index("-p") + 1] == str(prefix)


@pytest.mark.parametrize("engine", ["hisparse", "vortex"])
def test_decode_only_external_engine_rejected_before_campaign(campaign, engine):
    """Do not spend a capacity sweep on a backend with no request-metric adapter."""
    lane = engine + "-quest"
    campaign.config.update(lanes=[lane], external_lanes={lane: dict(engine=engine, method="quest")})
    with pytest.raises(ValueError, match="request probes do not support"):
        campaign.run()
    assert not campaign.processes
    assert not campaign.root.exists()
