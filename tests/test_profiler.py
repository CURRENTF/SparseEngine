from sparseengine.utils.profiler import Profiler


def test_profiler_snapshot_is_serializable_and_reports_average():
    instance = Profiler()
    instance.times["moe_router"] = 0.25
    instance.counts["moe_router"] = 2

    assert instance.snapshot() == {
        "moe_router": {
            "calls": 2,
            "total_s": 0.25,
            "avg_ms": 125.0,
        }
    }


def test_trace_is_opt_in_and_balances_exceptions(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import pytest
    from sparseengine.utils import profiler as module
    events = []

    @contextmanager
    def trace_range(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    monkeypatch.setattr(module, "platforms", SimpleNamespace(
        current_platform=SimpleNamespace(trace_range=trace_range),
    ))
    instance = Profiler()
    instance.nvtx_enabled = False
    with instance.trace("disabled"):
        pass
    assert events == []
    instance.nvtx_enabled = True
    with pytest.raises(ValueError):
        with instance.trace("outer"):
            with instance.trace("inner"):
                raise ValueError("test")
    assert events == [("enter", "outer"), ("enter", "inner"),
                      ("exit", "inner"), ("exit", "outer")]
    assert instance.snapshot() == {}
