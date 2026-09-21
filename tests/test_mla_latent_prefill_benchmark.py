import json
import os
from types import SimpleNamespace

import pytest

from benchmark import mla_latent_prefill as bench


def test_physical_gpu_selector_respects_visible_device_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,GPU-test-uuid")
    assert bench._physical_gpu_selector(0) == "6"
    assert bench._physical_gpu_selector(1) == "GPU-test-uuid"
    with pytest.raises(ValueError, match="unavailable"):
        bench._physical_gpu_selector(2)


def test_gpu_guard_scopes_nvidia_smi_to_selected_device(monkeypatch):
    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(stdout=f"{os.getpid()}\n")

    monkeypatch.setattr(bench.subprocess, "run", run)
    bench.gpu_guard("GPU-test-uuid")

    command, kwargs = observed[0]
    assert command[:3] == ["nvidia-smi", "--id", "GPU-test-uuid"]
    assert kwargs == {"check": True, "capture_output": True, "text": True}


def test_cold_measurement_synchronizes_each_call(monkeypatch):
    events = []
    times = iter((1.0, 1.25, 2.0, 2.5))
    monkeypatch.setattr(bench.torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))

    outputs, elapsed = bench._measure_cold_calls({
        "first": lambda: events.append("first") or 1,
        "second": lambda: events.append("second") or 2,
    })

    assert events == ["sync", "first", "sync", "sync", "second", "sync"]
    assert outputs == {"first": 1, "second": 2}
    assert elapsed == {"first": 0.25, "second": 0.5}


def test_failed_run_records_git_identity_case_and_terminal_status(tmp_path, monkeypatch):
    args = SimpleNamespace(
        output=tmp_path / "run", device=0, prefixes=[8], queries=[4], heads=5,
        warmup=1, repetitions=1,
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        bench, "_git_metadata",
        lambda: {"git_commit": "abc", "git_branch": "main", "git_dirty": True},
    )

    def fail(_args, _manifest, progress):
        progress["active_case"] = {"prefix": 8, "q": 4, "k": 12}
        raise AssertionError("bad result")

    monkeypatch.setattr(bench, "_run_benchmark", fail)
    with pytest.raises(AssertionError, match="bad result"):
        bench._execute(args)

    manifest = json.loads((args.output / "run_manifest.json").read_text())
    status = json.loads((args.output / "status.json").read_text())
    raw = [json.loads(line) for line in (args.output / "raw_samples.jsonl").read_text().splitlines()]
    assert manifest["git"] == {"git_commit": "abc", "git_branch": "main", "git_dirty": True}
    assert manifest["status"] == "metric_failed"
    assert status["status"] == "metric_failed"
    assert status["active_case"] == {"prefix": 8, "q": 4, "k": 12}
    assert "AssertionError" in status["traceback"]
    assert raw == [{"prefix": 8, "q": 4, "k": 12,
                    "status": "metric_failed", "error": "AssertionError('bad result')"}]


def test_invalid_visible_device_mapping_records_terminal_status(tmp_path, monkeypatch):
    args = SimpleNamespace(
        output=tmp_path / "run", device=0, prefixes=[8], queries=[4], heads=5,
        warmup=1, repetitions=1,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(
        bench, "_git_metadata",
        lambda: {"git_commit": "abc", "git_branch": "main", "git_dirty": False},
    )

    with pytest.raises(ValueError, match="unavailable"):
        bench._execute(args)

    manifest = json.loads((args.output / "run_manifest.json").read_text())
    status = json.loads((args.output / "status.json").read_text())
    assert manifest["status"] == "invalid_input"
    assert manifest["device"]["nvidia_smi_selector"] is None
    assert status["status"] == "invalid_input"
    assert status["completed_cases"] == 0
