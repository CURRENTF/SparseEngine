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
