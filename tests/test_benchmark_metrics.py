"""Independent metric oracles: request weighting, waiting, and stage boundaries."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.efficiency.metrics import (
    event_window_metrics,
    request_metrics,
    request_summary,
    stage_throughput,
)


def test_pool_requests_instead_of_batch_maxima_or_means():
    # Unequal iteration populations catch averaging summaries; existing probe
    # tests cover one batch only. Linear quantiles have an independent oracle.
    batches = [
        [request_metrics(0.1, 0.2, 3), request_metrics(0.3, 0.6, 4)],
        [request_metrics(0.8, 0, 1)],
    ]
    result = request_summary([row for batch in batches for row in batch])
    assert result["ttft_ms_mean"] == pytest.approx(400)
    assert result["ttft_ms_p50"] == pytest.approx(300)
    assert result["ttft_ms_p95"] == pytest.approx(750)
    assert result["ttft_ms_p99"] == pytest.approx(790)
    assert result["tpot_ms_mean"] == pytest.approx(150)
    assert result["tpot_ms_p50"] == pytest.approx(150)
    assert result["tpot_ms_p95"] == pytest.approx(195)
    assert result["tpot_ms_p99"] == pytest.approx(199)
    assert result["measured_request_count"] == 3
    assert result["tpot_request_count"] == 2


def test_prefill_interference_remains_in_request_tpot():
    # Two decode steps (0.1 s each) separated by another request's 0.8 s
    # prefill: request TPOT is 0.5 s, while decode stage rate is 10 token/s.
    row = request_metrics(0.2, 1.0, 3)
    assert row["tpot_ms"] == pytest.approx(500)
    assert stage_throughput(2, 0.1 + 0.1) == pytest.approx(10)
    windows = event_window_metrics(
        total_input_tokens=10, total_output_tokens=3, request_count=1,
        prefill_elapsed_s=0.2, decode_elapsed_s=1.0,
    )
    assert windows["batch_decode_token_throughput_tps"] == pytest.approx(2)
    assert windows["stage_metrics_status"] == "not_measured"


@pytest.mark.parametrize("change", [
    {"status": "model_failed"}, {"ttft_ms": float("nan")},
    {"tpot_ms": 3}, {"generated_tokens": 0},
])
def test_bad_artifact_cannot_silently_produce_valid_statistics(change):
    with pytest.raises(ValueError):
        request_summary([{**request_metrics(0.1, 0.2, 3), **change}])


def test_single_token_and_unmeasured_stage_have_no_decode_rate():
    result = request_summary([request_metrics(0.1, 0, 1)])
    assert result["tpot_ms_mean"] is None
    assert result["tpot_ms_p99"] is None
    assert stage_throughput(0, 0) is None
    with pytest.raises(ValueError):
        stage_throughput(2, 0)


def test_engine_summary_rejects_mixed_finish_events():
    with pytest.raises(ValueError, match="timing_source"):
        request_summary([
            {**request_metrics(0.1, 0.2, 3), "timing_source": source}
            for source in ("finished_time", "last_token_ts")
        ])


def test_suite_rejects_old_batch_maximum_contract():
    from benchmark.efficiency.validate_unified_suite import _validate_synthetic_rows

    errors = []
    _validate_synthetic_rows(
        [{"scenario": "fixed_batch", "status": "success"}],
        system="sengine-vanilla", errors=errors,
    )
    assert any("request metric contract mismatch" in error for error in errors)


def test_cli_preserves_workload_and_timing_source_boundaries(tmp_path):
    path = tmp_path / "request_samples.jsonl"
    common = dict(engine="vllm", sparse_method="vanilla", scenario="fixed_batch",
                  nominal_prompt_len=16, nominal_output_len=3, concurrency=2,
                  status="success")
    rows = [
        {**common, "timing_source": source, **request_metrics(ttft, 0.2, 3)}
        for source, ttft in [("legacy", 0.1), ("legacy", 0.3), ("v1", 0.9)]
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    command = [sys.executable, "benchmark/efficiency/metrics.py", str(path)]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                               check=True, capture_output=True, text=True)
    records = json.loads(completed.stdout)["records"]
    assert len(records) == 2
    assert records[0]["ttft_ms_mean"] == pytest.approx(200)
    assert records[1]["ttft_ms_mean"] == pytest.approx(900)
