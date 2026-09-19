"""CPU timing must preserve control flow and distinguish CPU from elapsed time."""
import importlib
import json

import pytest

module = importlib.import_module("sparseengine.utils.profiler")


def test_disabled_timing_does_not_wrap_or_read_clocks(monkeypatch):
    def forbidden():
        raise AssertionError("disabled timing read a clock")
    monkeypatch.setattr(module.time, "perf_counter_ns", forbidden)
    timer = module.CpuTiming(0)
    function = lambda value: value
    assert timer.timed(function) is function


def test_nested_timing_reports_inclusive_cpu_and_wall_without_early_flush(monkeypatch):
    clock = {"wall": 0, "cpu": 0}
    reports = []
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: clock["wall"])
    monkeypatch.setattr(module.time, "thread_time_ns", lambda: clock["cpu"])
    monkeypatch.setattr(module.logger, "info", lambda _, report: reports.append(json.loads(report)))
    timer = module.CpuTiming(1)

    @timer.timed
    def inner():
        clock.update(wall=2_000_000_000, cpu=100_000_000)

    @timer.timed
    def outer():
        inner()
        assert not reports
        clock.update(wall=3_000_000_000, cpu=150_000_000)
        return 17

    assert outer() == 17
    assert len(reports) == 1
    rows = reports[0]["stages"]
    assert rows[inner.__qualname__]["cpu_ms"] == 100
    assert rows[inner.__qualname__]["wall_ms"] == 2000
    assert rows[outer.__qualname__]["cpu_ms"] == 150
    assert rows[outer.__qualname__]["wall_ms"] == 3000
    assert reports[0]["inclusive"]
    assert not timer.local.state["stats"]


def test_timing_preserves_failure_and_restores_nesting(monkeypatch):
    clock = {"wall": 0, "cpu": 0}
    reports = []
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: clock["wall"])
    monkeypatch.setattr(module.time, "thread_time_ns", lambda: clock["cpu"])
    monkeypatch.setattr(module.logger, "info", lambda _, report: reports.append(json.loads(report)))
    timer = module.CpuTiming(1)
    error = RuntimeError("original failure")

    @timer.timed
    def fail():
        clock.update(wall=2_000_000_000, cpu=20_000_000)
        raise error

    with pytest.raises(RuntimeError) as caught:
        fail()
    assert caught.value is error
    assert timer.local.state["depth"] == 0
    assert reports[0]["stages"][fail.__qualname__]["errors"] == 1
