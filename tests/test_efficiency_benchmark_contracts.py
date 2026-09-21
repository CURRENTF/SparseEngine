import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.efficiency import bench_probe
from benchmark.efficiency.bench_probe import (
    HardwareMetricError,
    _actual_hardware_metrics,
    _attach_churn_comparisons,
    _physical_gpu_metadata,
    _decode_graph_counter_delta,
    _record_batch_first_tokens,
    _phase_throughput_metrics,
    _request_phase_metrics_from_timestamps,
    _resolve_sparse_probe_protocol,
    _tpot_concurrency_proxy_tps,
    _vllm_batch_phase_seconds,
    _vllm_phase_metrics,
    _vllm_request_phase_seconds,
)
from benchmark.efficiency.hardware_monitor import GPUHardwareMonitor
from benchmark.efficiency.metrics_calculator import (
    ModelArchitectureSpecs,
    detect_gpu_hardware,
)
from benchmark.efficiency.validate_unified_suite import validate_suite
from benchmark.efficiency.workload import build_request_trace, derive_trace_seed
from benchmark.long_bench.pred_vllm import _effective_num_samples, build_chat_prompt
from benchmark.long_bench.pred import _merge_worker_outputs
from benchmark.long_bench.prompt_budget import encode_prompt_with_generation_budget


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_zero_jitter_keeps_every_request_at_the_requested_length():
    """Multi-request capacity runs previously shortened prompts despite jitter=0."""
    traces = build_request_trace(seed=42, request_count=7,
        nominal_prompt_len=101, nominal_output_len=17, vocab_size=1000,
        prompt_jitter_fraction=0, output_jitter_fraction=0,
        vary_output_lengths=True)
    assert all(trace.prompt_len == 101 and trace.output_len == 17 for trace in traces)


def test_shared_prompt_reuses_only_within_workload(monkeypatch, tmp_path):
    """Prevent a cache benchmark's warmup or prior repeat from priming its measured prefix."""
    monkeypatch.setattr(sys, "argv", ["probe", "--model-path", "model", "--output-dir", str(tmp_path),
                                     "--shared-prompt", "--prompt-length-jitter", "0"])
    args = bench_probe.parse_args()
    specs = SimpleNamespace(vocab_size=1000)
    def trace(phase, iteration):
        return bench_probe._trace_for_iteration(args, specs, scenario="fixed_batch", phase=phase,
            prompt_len=101, output_len=17, concurrency=4, iteration=iteration,
            request_count=4, vary_output_lengths=False)
    measured = trace("measure", 0)
    assert len({r.prompt_digest for r in measured}) == 1
    assert len({r.request_index for r in measured}) == 4
    assert measured == trace("measure", 0)
    assert measured[0].prompt_digest != trace("warmup", 0)[0].prompt_digest
    assert measured[0].prompt_digest != trace("measure", 1)[0].prompt_digest
    from benchmark.efficiency.workload import trace_metadata
    with pytest.raises(RuntimeError, match="duplicate"):
        trace_metadata(measured)
    assert trace_metadata(measured, allow_duplicate_prompts=True)["request_count"] == 4


def test_fork_constructor_options_cannot_change_matched_workload(monkeypatch, tmp_path):
    """Catch a fork config silently enabling prefix hits or changing concurrency."""
    monkeypatch.setattr(sys, "argv", ["probe", "--model-path", "model",
                                    "--output-dir", str(tmp_path), "--engine", "vllm"])
    args = bench_probe.parse_args()
    args.sparse_method = "snapkv"
    args.engine_kwargs = json.dumps({"compression_scorer": "snapkv",
                                    "compression_budget_tokens": 1024})
    resolved = bench_probe._vllm_engine_kwargs(args)
    assert resolved["compression_scorer"] == "snapkv"
    assert resolved["compression_budget_tokens"] == 1024
    for key, value in [("enable_prefix_caching", True), ("max_num_seqs", 999),
                       ("model", "different-model"), ("disable_log_stats", True)]:
        args.engine_kwargs = json.dumps({key: value})
        with pytest.raises(ValueError, match="cannot override matched"):
            bench_probe._vllm_engine_kwargs(args)


@pytest.mark.parametrize("method,options", [
    ("snapkv", {}),
    ("vanilla", {"compression_scorer": "snapkv", "compression_budget_tokens": 1024}),
    ("snapkv", {"compression_scorer": "snapkv", "compression_budget_tokens": 0}),
    ("snapkv", {"compression_scorer": "snapkv", "compression_budget_tokens": "1024"}),
    ("snapkv", {"compression_scorer": "snapkv", "compression_budget_tokens": True}),
    ("h2o", {}),
])
def test_vllm_probe_rejects_method_labels_inconsistent_with_launch_options(
    monkeypatch, tmp_path, method, options,
):
    """Prevent dense runs being reported as sparse, or compression as vanilla."""
    monkeypatch.setattr(sys, "argv", ["probe", "--model-path", "model",
                                    "--output-dir", str(tmp_path), "--engine", "vllm",
                                    "--sparse-method", method])
    args = bench_probe.parse_args()
    args.engine_kwargs = json.dumps(options)
    with pytest.raises(ValueError, match="method label"):
        bench_probe._vllm_engine_kwargs(args)


def test_vllm_probe_accepts_uncompressed_constructor_options(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["probe", "--model-path", "model",
                                    "--output-dir", str(tmp_path), "--engine", "vllm"])
    args = bench_probe.parse_args()
    args.engine_kwargs = json.dumps({"dtype": "bfloat16", "compression_scorer": "none"})
    resolved = bench_probe._vllm_engine_kwargs(args)
    assert resolved["dtype"] == "bfloat16"
    assert resolved["compression_scorer"] == "none"


def test_probe_cli_parser_builds_with_new_workload_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_probe.py",
            "--model-path",
            "model",
            "--output-dir",
            "output",
            "--scenario",
            "all",
            "--batch-sizes",
            "1,4",
            "--expert-parallel-size",
            "4",
        ],
    )

    args = bench_probe.parse_args()

    assert args.scenario == "all"
    assert args.batch_sizes == [1, 4]
    assert args.expert_parallel_size == 4
    assert args.churn_request_multiplier == 4


@pytest.mark.parametrize(
    "hyper_params,expected_batch,expected_resident",
    [
        ("{}", 16, 16),
        ('{"data_parallel_size":2,"max_num_seqs_in_gpu":64}', 8, 64),
    ],
)
def test_fixed_probe_sets_engine_capacity_from_largest_batch_size(
    monkeypatch,
    tmp_path,
    hyper_params,
    expected_batch,
    expected_resident,
):
    import sparseengine

    captured = {}

    class StopAfterEngineInit(Exception):
        pass

    def capture_engine_kwargs(model_path, **kwargs):
        captured.update(kwargs)
        raise StopAfterEngineInit(model_path)

    monkeypatch.setattr(sparseengine, "LLM", capture_engine_kwargs)
    args = SimpleNamespace(
        scenario="fixed",
        output_dir=str(tmp_path),
        hyper_params=hyper_params,
        tensor_parallel_size=1,
        expert_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=8192,
        model_path="model",
        sparse_method="vanilla",
        sparse_prefill_score_mode=None,
        allow_single_omnikv_full_layer=False,
        prompt_lens=[1024],
        output_lens=[32],
        batch_sizes=[4, 16, 8],
    )

    with pytest.raises(StopAfterEngineInit):
        bench_probe.run_sparseengine_probe(args, SimpleNamespace())

    assert captured["max_num_seqs_in_batch"] == expected_batch
    assert captured["max_decoding_seqs"] == expected_batch
    assert captured["max_num_seqs_in_gpu"] == expected_resident
    assert captured["expert_parallel_size"] == 1


def test_physical_gpu_metadata_uses_nvidia_smi_without_cuda_init(monkeypatch):
    monkeypatch.setattr(
        bench_probe.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="1, NVIDIA H100 80GB HBM3, 9.0, 81559, 590.44\n"
        ),
    )

    assert _physical_gpu_metadata([1]) == [
        {
            "physical_device_index": 1,
            "name": "NVIDIA H100 80GB HBM3",
            "compute_capability": [9, 0],
            "total_memory_mib": 81559,
            "driver_version": "590.44",
        }
    ]


def test_decode_graph_counter_delta_reports_runtime_capture_churn():
    delta = _decode_graph_counter_delta(
        {"capture_count": 28, "replay_count": 10, "eviction_count": 0},
        {
            "capture_count": 30,
            "replay_count": 110,
            "eviction_count": 2,
            "recapture_count": 1,
        },
    )

    assert delta["capture_count"] == 2
    assert delta["replay_count"] == 100
    assert delta["eviction_count"] == 2
    assert delta["recapture_count"] == 1


def test_unknown_hardware_does_not_fall_back_to_h100():
    with pytest.raises(ValueError, match="Unknown GPU hardware"):
        detect_gpu_hardware("Mystery Accelerator")


def test_ambiguous_h100_requires_explicit_profile():
    with pytest.raises(ValueError, match="Ambiguous H100"):
        detect_gpu_hardware("NVIDIA H100 80GB HBM3")
    assert detect_gpu_hardware("h100_sxm").peak_bandwidth_tbs == 3.35


def test_model_specs_require_real_architecture_fields():
    with pytest.raises(ValueError, match="missing required architecture fields"):
        ModelArchitectureSpecs.from_config_dict({"hidden_size": 2048})


def test_model_specs_accept_nested_explicit_non_factorized_head_dim():
    specs = ModelArchitectureSpecs.from_config_dict(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "vocab_size": 248320,
                "intermediate_size": 17408,
            },
        }
    )

    assert specs.hidden_size == 5120
    assert specs.num_attention_heads == 24
    assert specs.num_key_value_heads == 4
    assert specs.head_dim == 256


def test_model_specs_require_factorized_head_dim_when_not_explicit():
    with pytest.raises(ValueError, match="must define head_dim"):
        ModelArchitectureSpecs.from_config_dict(
            {
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "vocab_size": 248320,
                "intermediate_size": 17408,
            }
        )


def test_model_specs_resolve_mla_qk_head_dim():
    specs = ModelArchitectureSpecs.from_config_dict(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 47,
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "vocab_size": 154880,
            "n_routed_experts": 64,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1536,
            "intermediate_size": 10240,
        }
    )

    assert specs.head_dim == 256


def test_probe_writes_metric_failed_when_model_discovery_fails(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"hidden_size": 2048}))
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_probe.py",
            "--model-path",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--hardware",
            "h100_sxm",
        ],
    )

    with pytest.raises(ValueError, match="missing required architecture fields"):
        bench_probe.main()

    assert json.loads((output_dir / "run_status.json").read_text())["status"] == "metric_failed"
    assert json.loads((output_dir / "summary.json").read_text())["status"] == "metric_failed"


def test_probe_refuses_to_reuse_artifact_directory(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "raw_samples.jsonl").write_text("old\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_probe.py",
            "--model-path",
            "unused",
            "--output-dir",
            str(output_dir),
            "--hardware",
            "h100_sxm",
        ],
    )

    with pytest.raises(FileExistsError, match="Refusing to mix benchmark runs"):
        bench_probe.main()
    assert (output_dir / "raw_samples.jsonl").read_text() == "old\n"


def test_hardware_monitor_zero_samples_is_metric_failed(tmp_path):
    output = tmp_path / "timeline.json"
    monitor = GPUHardwareMonitor([0], output_file=output)
    monitor.start_time = 1.0
    monitor.end_time = 2.0

    summary = monitor.analyze_and_save()

    assert summary["status"] == "metric_failed"
    assert summary["total_samples"] == 0
    assert "host_launch_bubble" not in json.dumps(summary)
    assert json.loads(output.read_text())["summary"]["status"] == "metric_failed"


def test_hardware_monitor_names_sampled_idle_without_host_attribution():
    monitor = GPUHardwareMonitor([0])
    monitor.start_time = 1.0
    monitor.end_time = 2.0
    monitor.samples = [
        {
            "time_s": 0.1,
            "gpu0_util": 0.0,
            "gpu0_mem_util": 0.0,
            "gpu0_mem_mb": 1.0,
            "gpu0_power_w": 10.0,
            "gpu0_temp_c": 30.0,
        }
    ]

    summary = monitor.analyze_and_save()

    assert summary["status"] == "success"
    assert summary["gpus"]["gpu_0"]["coarse_gpu_idle_duty_pct"] == 100.0
    assert summary["gpus"]["gpu_0"]["avg_memory_io_activity_pct"] == 0.0
    assert summary["aggregate"]["mean_memory_io_activity_pct"] == 0.0
    assert "host_launch_bubble_pct" not in summary["gpus"]["gpu_0"]


def test_vllm_phase_metrics_use_one_request_timeline():
    output = SimpleNamespace(
        metrics=SimpleNamespace(arrival_time=10.0, first_token_time=10.2, finished_time=10.5),
        outputs=[SimpleNamespace(token_ids=[1, 2, 3, 4])],
    )

    ttft_ms, tpot_ms = _vllm_phase_metrics([output], 4)

    assert ttft_ms == pytest.approx(200.0)
    assert tpot_ms == pytest.approx(100.0)


def test_vllm_phase_metrics_average_per_request_tpot():
    outputs = [
        SimpleNamespace(
            metrics=SimpleNamespace(
                arrival_time=10.0,
                first_token_time=10.2,
                finished_time=10.5,
            ),
            outputs=[SimpleNamespace(token_ids=[1, 2, 3, 4])],
        ),
        SimpleNamespace(
            metrics=SimpleNamespace(
                arrival_time=10.0,
                first_token_time=10.3,
                finished_time=10.75,
            ),
            outputs=[SimpleNamespace(token_ids=[1, 2, 3, 4])],
        ),
    ]

    ttft_ms, tpot_ms = _vllm_phase_metrics(outputs, 4)

    assert ttft_ms == pytest.approx(250.0)
    assert tpot_ms == pytest.approx(125.0)


def test_vllm_single_token_request_has_no_tpot():
    output = SimpleNamespace(
        metrics=SimpleNamespace(arrival_time=10.0, first_token_time=10.2, finished_time=10.2),
        outputs=[SimpleNamespace(token_ids=[1])],
    )

    _ttft_ms, tpot_ms = _vllm_phase_metrics([output], 1)

    assert tpot_ms is None


def test_request_tpot_and_batch_decode_window_use_explicit_matched_scopes():
    request_metrics = _request_phase_metrics_from_timestamps(
        arrival_times={11: 0.0, 12: 0.05},
        first_token_times={11: 0.2, 12: 0.35},
        finished_times={11: 0.5, 12: 0.8},
        generated_counts={11: 4, 12: 4},
    )

    assert request_metrics["ttft_ms"] == pytest.approx(250.0)
    assert request_metrics["tpot_ms"] == pytest.approx(125.0)
    assert request_metrics["prefill_elapsed_s"] == pytest.approx(0.3)
    assert request_metrics["decode_elapsed_s"] == pytest.approx(0.6)

    batch_metrics = _phase_throughput_metrics(
        total_input_tokens=200,
        total_output_tokens=8,
        request_count=2,
        prefill_elapsed_s=request_metrics["prefill_elapsed_s"],
        decode_elapsed_s=request_metrics["decode_elapsed_s"],
    )
    assert batch_metrics["batch_decode_token_throughput_tps"] == pytest.approx(10.0)
    assert batch_metrics["decode_token_throughput_tps"] == pytest.approx(10.0)
    sparse_proxy = _tpot_concurrency_proxy_tps(
        concurrency=2,
        tpot_ms=request_metrics["tpot_ms"],
    )
    baseline_proxy = _tpot_concurrency_proxy_tps(concurrency=2, tpot_ms=250.0)
    assert sparse_proxy == pytest.approx(16.0)
    assert sparse_proxy / baseline_proxy == pytest.approx(
        250.0 / request_metrics["tpot_ms"]
    )


def test_request_timing_rejects_incomplete_event_coverage():
    with pytest.raises(RuntimeError, match="completion coverage mismatch"):
        _request_phase_metrics_from_timestamps(
            arrival_times={11: 0.0, 12: 0.0},
            first_token_times={11: 0.1, 12: 0.2},
            finished_times={11: 0.4},
            generated_counts={11: 4, 12: 4},
        )


def test_phase_throughput_uses_separate_prefill_and_decode_windows():
    metrics = _phase_throughput_metrics(
        total_input_tokens=200,
        total_output_tokens=10,
        request_count=2,
        prefill_elapsed_s=0.5,
        decode_elapsed_s=0.2,
    )

    assert metrics["prefill_token_count"] == 200
    assert metrics["decode_token_count"] == 8
    assert metrics["prefill_token_throughput_tps"] == pytest.approx(400.0)
    assert metrics["decode_token_throughput_tps"] == pytest.approx(40.0)
    assert metrics["batch_decode_token_throughput_tps"] == pytest.approx(40.0)


def test_phase_throughput_allows_ttft_only_workload():
    metrics = _phase_throughput_metrics(
        total_input_tokens=200,
        total_output_tokens=2,
        request_count=2,
        prefill_elapsed_s=0.5,
        decode_elapsed_s=0.0,
    )

    assert metrics["prefill_token_throughput_tps"] == pytest.approx(400.0)
    assert metrics["decode_token_count"] == 0
    assert metrics["decode_elapsed_s"] is None
    assert metrics["decode_token_throughput_tps"] is None


def test_markdown_report_shows_tpot_without_saturation_metrics():
    report = bench_probe._format_markdown_report(
        [
            {
                "engine": "sparseengine",
                "sparse_method": "quest",
                "protocol_label": "sparseengine-quest",
                "scenario": "fixed_batch",
                "prompt_len_min": 100,
                "prompt_len_max": 120,
                "output_len_min": 16,
                "output_len_max": 16,
                "concurrency": 4,
                "request_throughput_rps": 2.0,
                "prefill_token_throughput_tps": 200.0,
                "decode_token_throughput_tps": 30.0,
                "ttft_ms_p50": 10.0,
                "ttft_ms_p99": 12.0,
                "tpot_ms_mean": 2.5,
                "tpot_concurrency_proxy_tps": 1600.0,
                "gpu_compute_activity_pct_mean": 90.0,
                "gpu_memory_io_activity_pct_mean": 40.0,
                "peak_vram_gb_max": 70.0,
                "status": "success",
            }
        ],
        model_name="model",
        tp_size=4,
        ep_size=4,
    )

    assert "Request TPOT mean (ms)" in report
    assert "TPOT-equivalent concurrent tok/s" in report
    assert "| 2.50 |" in report
    assert "scaling" not in report.lower()
    assert "observed decode peak" not in report.lower()


def test_vllm_batch_phase_windows_span_request_events():
    outputs = [
        SimpleNamespace(
            metrics=SimpleNamespace(
                arrival_time=10.0,
                first_token_time=10.2,
                finished_time=10.5,
            )
        ),
        SimpleNamespace(
            metrics=SimpleNamespace(
                arrival_time=10.0,
                first_token_time=10.3,
                finished_time=10.6,
            )
        ),
    ]

    prefill_s, decode_s = _vllm_batch_phase_seconds(outputs)

    assert prefill_s == pytest.approx(0.3)
    assert decode_s == pytest.approx(0.4)


def test_vllm_v1_phase_metrics_use_latency_and_monotonic_timestamps():
    ttft_s, decode_s, source = _vllm_request_phase_seconds(
        SimpleNamespace(
            arrival_time=1_700_000_000.0,
            first_token_latency=0.25,
            first_token_ts=10.0,
            last_token_ts=10.6,
        )
    )

    assert ttft_s == pytest.approx(0.25)
    assert decode_s == pytest.approx(0.6)
    assert source == "vllm_v1_first_token_latency_and_monotonic_decode"


def test_sparse_batch_ttft_waits_for_every_request_first_token():
    expected = {11, 12}
    observed: set[int] = set()

    assert not _record_batch_first_tokens(expected, observed, [(11, [1])], [])
    assert _record_batch_first_tokens(
        expected,
        observed,
        [],
        [(12, [2], None, None)],
    )
    assert observed == expected


def test_final_prompt_budget_includes_special_tokens_and_generation():
    class Tokenizer:
        bos_token = "<bos>"

        def __init__(self):
            self.add_special_tokens = None

        def encode(self, prompt, *, add_special_tokens):
            self.add_special_tokens = add_special_tokens
            token_ids = list(range(10, 20))
            return [1, *token_ids] if add_special_tokens else token_ids

    tokenizer = Tokenizer()
    token_ids = encode_prompt_with_generation_budget(
        tokenizer,
        "rendered chat prompt",
        max_model_len=8,
        max_gen=3,
    )

    assert tokenizer.add_special_tokens is True
    assert token_ids == [1, 10, 17, 18, 19]
    assert len(token_ids) + 3 == 8


def test_final_prompt_budget_rejects_generation_without_prompt_space():
    tokenizer = SimpleNamespace(
        bos_token=None,
        encode=lambda prompt, add_special_tokens: [1],
    )
    with pytest.raises(ValueError, match="leaves no prompt budget"):
        encode_prompt_with_generation_budget(
            tokenizer,
            "prompt",
            max_model_len=4,
            max_gen=4,
        )


def test_vllm_runner_requires_consistent_sample_limit():
    args = SimpleNamespace(num_samples=10, samples_per_task=8)
    with pytest.raises(ValueError, match="disagree"):
        _effective_num_samples(args)


def test_vllm_chat_template_failure_is_not_silently_hidden():
    tokenizer = SimpleNamespace(
        chat_template="template",
        apply_chat_template=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad template")),
    )
    with pytest.raises(RuntimeError, match="bad template"):
        build_chat_prompt(tokenizer, "prompt")


def test_snapkv_probe_requires_and_records_explicit_score_mode():
    args = SimpleNamespace(
        hyper_params="{}",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=8192,
        sparse_method="snapkv",
        sparse_prefill_score_mode=None,
    )
    with pytest.raises(ValueError, match="explicit --sparse-prefill-score-mode"):
        _resolve_sparse_probe_protocol(args)

    args.sparse_prefill_score_mode = "probability"
    _hyper, budget, protocol, label = _resolve_sparse_probe_protocol(args)
    assert budget == 2176
    assert protocol == {
        "score_mode": "probability",
        "score_window": 64,
        "sparse_budget": 2176,
        "max_num_batched_tokens": 8192,
    }
    assert "probability-budget2176-window64" in label


def test_h2o_probe_records_explicit_budget_protocol():
    args = SimpleNamespace(
        hyper_params='{"h2o_prefill_score_window": 0}',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=8192,
        sparse_method="h2o",
        sparse_prefill_score_mode="probability",
    )

    hyper, budget, protocol, label = _resolve_sparse_probe_protocol(args)

    assert budget is None
    assert hyper["h2o_decode_budget"] == 4096
    assert protocol["decode_budget"] == 4096
    assert protocol["prefill_budget"] == 8192
    assert protocol["score_mode"] == "probability"
    assert protocol["max_num_batched_tokens"] == 8192
    assert "h2o-probability-decode4096-prefill8192-window0" in label


def test_omnikv_probe_requires_explicit_calibrated_layers():
    args = SimpleNamespace(
        hyper_params="{}",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=8192,
        sparse_method="omnikv",
        sparse_prefill_score_mode=None,
        allow_single_omnikv_full_layer=False,
    )

    with pytest.raises(ValueError, match="explicit calibrated method config"):
        _resolve_sparse_probe_protocol(args)

    args.hyper_params = json.dumps(
        {
            "engine_prefill_chunk_size": 16384,
            "full_attention_layers": "0,2,4,11,16,22",
            "sink_keep_tokens": 0,
            "recent_keep_tokens": 32,
            "decode_keep_tokens": 2048,
            "pool_kernel_size": 1,
        }
    )
    hyper, budget, _protocol, label = _resolve_sparse_probe_protocol(args)

    assert hyper["full_attention_layers"] == "0,2,4,11,16,22"
    assert budget is None
    assert label == "sparseengine-omnikv"


def test_random_trace_is_matched_reproducible_and_variable_length():
    seed = derive_trace_seed(
        42,
        scenario="fixed_batch",
        phase="measure",
        nominal_prompt_len=100,
        nominal_output_len=16,
        concurrency=4,
        iteration=0,
    )
    first = build_request_trace(
        seed=seed,
        request_count=4,
        nominal_prompt_len=100,
        nominal_output_len=16,
        vocab_size=1000,
        prompt_jitter_fraction=0.10,
        output_jitter_fraction=0.25,
        vary_output_lengths=False,
    )
    second = build_request_trace(
        seed=seed,
        request_count=4,
        nominal_prompt_len=100,
        nominal_output_len=16,
        vocab_size=1000,
        prompt_jitter_fraction=0.10,
        output_jitter_fraction=0.25,
        vary_output_lengths=False,
    )

    assert [trace.prompt_digest for trace in first] == [
        trace.prompt_digest for trace in second
    ]
    assert len({trace.prompt_digest for trace in first}) == 4
    assert len({trace.prompt_len for trace in first}) > 1
    assert all(90 <= trace.prompt_len <= 100 for trace in first)
    assert {trace.output_len for trace in first} == {16}


def test_random_trace_changes_every_measurement_iteration():
    seeds = [
        derive_trace_seed(
            42,
            scenario="fixed_batch",
            phase="measure",
            nominal_prompt_len=64,
            nominal_output_len=8,
            concurrency=2,
            iteration=iteration,
        )
        for iteration in range(3)
    ]
    digest_sets = []
    for seed in seeds:
        trace = build_request_trace(
            seed=seed,
            request_count=2,
            nominal_prompt_len=64,
            nominal_output_len=8,
            vocab_size=1000,
            prompt_jitter_fraction=0.10,
            output_jitter_fraction=0.25,
            vary_output_lengths=False,
        )
        digest_sets.append({request.prompt_digest for request in trace})

    assert len(set(seeds)) == 3
    assert not (digest_sets[0] & digest_sets[1] | digest_sets[0] & digest_sets[2] | digest_sets[1] & digest_sets[2])


def test_churn_trace_varies_prompt_and_output_lengths():
    trace = build_request_trace(
        seed=123,
        request_count=16,
        nominal_prompt_len=128,
        nominal_output_len=32,
        vocab_size=1000,
        prompt_jitter_fraction=0.10,
        output_jitter_fraction=0.25,
        vary_output_lengths=True,
    )

    assert len({request.prompt_len for request in trace}) > 1
    assert len({request.output_len for request in trace}) > 1


def test_actual_hardware_metrics_use_sampled_values_and_cross_gpu_peak():
    metrics = _actual_hardware_metrics(
        {
            "status": "success",
            "total_samples": 5,
            "sampling_interval_ms": 100,
            "aggregate": {
                "mean_compute_util_pct": 60.0,
                "mean_memory_io_activity_pct": 30.0,
                "mean_coarse_gpu_active_duty_pct": 80.0,
                "avg_total_power_w": 500.0,
            },
            "gpus": {
                "gpu_0": {"peak_vram_gb": 70.0},
                "gpu_1": {"peak_vram_gb": 71.5},
            },
        }
    )

    assert metrics["metric_source"] == "nvidia-smi sampled activity"
    assert metrics["gpu_compute_activity_pct_mean"] == 60.0
    assert metrics["gpu_memory_io_activity_pct_mean"] == 30.0
    assert metrics["peak_vram_gb_max"] == 71.5


def test_actual_hardware_metrics_do_not_fall_back_to_estimates():
    with pytest.raises(HardwareMetricError, match="collection failed"):
        _actual_hardware_metrics({"status": "metric_failed", "error": "no samples"})


def test_churn_summary_is_compared_to_matched_fixed_batch():
    rows = [
        {
            "engine": "sparseengine",
            "sparse_method": "vanilla",
            "protocol_label": "sparseengine-vanilla",
            "scenario": "fixed_batch",
            "prompt_len": 100,
            "output_len": 16,
            "concurrency": 4,
            "decode_token_throughput_tps": 100.0,
            "request_throughput_rps": 10.0,
            "ttft_ms_p99": 20.0,
        },
        {
            "engine": "sparseengine",
            "sparse_method": "vanilla",
            "protocol_label": "sparseengine-vanilla",
            "scenario": "oversubscribed_churn",
            "prompt_len": 100,
            "output_len": 16,
            "concurrency": 4,
            "decode_token_throughput_tps": 80.0,
            "request_throughput_rps": 8.0,
            "ttft_ms_p99": 50.0,
        },
    ]

    _attach_churn_comparisons(rows)

    assert rows[1]["fixed_batch_comparison_status"] == "success"
    assert rows[1]["churn_decode_tps_ratio_vs_fixed_batch"] == pytest.approx(0.8)
    assert rows[1]["churn_ttft_p99_delta_ms_vs_fixed_batch"] == pytest.approx(30.0)


def test_decode_comparisons_skip_ttft_only_workload():
    rows = [
        {
            "engine": "sparseengine",
            "sparse_method": "vanilla",
            "protocol_label": "sparseengine-vanilla",
            "scenario": scenario,
            "prompt_len": 100,
            "output_len": 1,
            "concurrency": 4,
            "decode_token_throughput_tps": None,
            "request_throughput_rps": request_rate,
            "ttft_ms_p99": ttft,
        }
        for scenario, request_rate, ttft in (
            ("fixed_batch", 10.0, 20.0),
            ("oversubscribed_churn", 8.0, 50.0),
        )
    ]

    _attach_churn_comparisons(rows)

    assert rows[1]["fixed_batch_comparison_status"] == "success"
    assert rows[1]["churn_decode_tps_comparison_status"] == "skipped_by_policy"
    assert rows[1]["churn_decode_tps_ratio_vs_fixed_batch"] is None
    assert rows[1]["churn_request_rps_ratio_vs_fixed_batch"] == pytest.approx(0.8)


def test_longbench_scorer_rejects_missing_status(tmp_path):
    prediction = {
        "pred": "label",
        "answers": ["label"],
        "all_classes": ["label"],
        "length": 1,
    }
    (tmp_path / "trec.jsonl").write_text(json.dumps(prediction) + "\n")

    result = subprocess.run(
        [sys.executable, "benchmark/long_bench/eval.py", "--path", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["status"] == "failed"
    assert metrics["task_statuses"]["trec"]["invalid_statuses"] == {"missing": 1}


def _write_valid_suite_fixture(root: Path, systems: list[str]) -> None:
    protocols = {
        "sengine-vanilla": ("sparseengine", "vanilla"),
        "sengine-snapkv": ("sparseengine", "snapkv"),
        "sengine-h2o": ("sparseengine", "h2o"),
        "sengine-omnikv": ("sparseengine", "omnikv"),
        "sengine-deltakv": ("sparseengine", "deltakv"),
        "vllm-vanilla": ("vllm", "vanilla"),
        "vllm": ("vllm", "vanilla"),
    }
    for scenario in ("scenario_a_synthetic", "scenario_b_longbench"):
        for system in systems:
            engine, method = protocols[system]
            system_dir = root / scenario / system
            system_dir.mkdir(parents=True)
            (system_dir / "stage_status.json").write_text(
                json.dumps({"status": "success"}) + "\n"
            )
            (system_dir / "gpu_timeline_summary.json").write_text(
                json.dumps({"status": "success", "aggregate": {}}) + "\n"
            )
            if scenario == "scenario_a_synthetic":
                hardware = {
                    "metric_source": "nvidia-smi sampled activity",
                    "sample_count": 2,
                }
                fixed = {
                    "request_metric_contract": bench_probe.REQUEST_METRIC_CONTRACT,
                    "status": "success",
                    "scenario": "fixed_batch",
                    "prompt_len": 100,
                    "output_len": 16,
                    "prompt_len_min": 90,
                    "prompt_len_max": 100,
                    "output_len_min": 16,
                    "output_len_max": 16,
                    "concurrency": 2,
                    "request_count": 2,
                    "request_throughput_rps": 2.0,
                    "prefill_token_throughput_tps": 200.0,
                    "decode_token_throughput_tps": 30.0,
                    "batch_decode_token_throughput_tps": 30.0,
                    "tpot_ms_mean": 2.0,
                    "tpot_timing_scope": "mean_per_request_first_token_to_finish_v2",
                    "tpot_concurrency_proxy_tps": 1000.0,
                    "decode_metric_status": "success",
                    "actual_hardware_metrics": hardware,
                }
                churn = {
                    **fixed,
                    "scenario": "oversubscribed_churn",
                    "output_len_min": 12,
                    "request_count": 8,
                    "fixed_batch_comparison_status": "success",
                    "churn_decode_tps_comparison_status": "success",
                }
                (system_dir / "summary.json").write_text(
                    json.dumps({"status": "success", "summary": [fixed, churn]})
                    + "\n"
                )
                raw_rows = []
                for iteration, scenario_name in enumerate(
                    ("fixed_batch", "oversubscribed_churn")
                ):
                    raw_rows.append(
                        {
                            "status": "success",
                            "scenario": scenario_name,
                            "prompt_len": 100,
                            "output_len": 16,
                            "concurrency": 2,
                            "iteration": iteration,
                            "trace": {
                                "prompt_lengths": [90, 100],
                                "output_lengths": [16, 16],
                                "prompt_digests": [
                                    f"digest-{scenario_name}-a",
                                    f"digest-{scenario_name}-b",
                                ],
                            },
                        }
                    )
                (system_dir / "raw_samples.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in raw_rows)
                )
                (system_dir / "request_samples.jsonl").write_text(
                    json.dumps({"status": "success", "request_index": 0}) + "\n"
                )
                (system_dir / "run_manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "success",
                            "args": {"engine": engine, "sparse_method": method},
                            "workload": {
                                "prefix_caching_enabled": False,
                                "iteration_prompt_reuse_allowed": False,
                            },
                        }
                    )
                    + "\n"
                )
                if engine == "sparseengine":
                    (system_dir / "operator_runtime_stats.json").write_text(
                        json.dumps(
                            {
                                "status": "success",
                                "world_ranks": [
                                    {
                                        "world_rank": 0,
                                        "bindings": [
                                            {
                                                "operator_type": "test",
                                                "selected_provider": "test_provider",
                                            }
                                        ],
                                        "operators": {},
                                    }
                                ],
                            }
                        )
                        + "\n"
                    )
            else:
                (system_dir / "result.json").write_text(
                    json.dumps({"status": "success"}) + "\n"
                )
                sample = {
                    "dataset": "task",
                    "sample_idx": 0,
                    "source_idx": 0,
                    "status": "success",
                }
                task_sample = {"status": "success", "source_idx": 0}
                (system_dir / "task.jsonl").write_text(
                    json.dumps(task_sample) + "\n"
                )
                (system_dir / "run_status.json").write_text(
                    json.dumps({"status": "success"}) + "\n"
                )
                for artifact in (
                    "raw_outputs.jsonl",
                    "parsed_outputs.jsonl",
                    "sample_results.jsonl",
                ):
                    (system_dir / artifact).write_text(json.dumps(sample) + "\n")
                resolved = {
                    "backend": engine,
                    "args": {
                        "seed": 42,
                        "enable_prefix_caching": False,
                    },
                }
                if engine == "sparseengine":
                    resolved["sparse_method"] = method
                    resolved["effective_runtime"] = {
                        "prefix_cache_enabled": False,
                    }
                if method == "omnikv":
                    resolved["requested_runtime"] = {
                        "config": {
                            "full_attention_layers": "0,2,4,11,16,22",
                        },
                    }
                    resolved["effective_runtime"]["benchmark_config"] = {
                        "full_attention_layers": [0, 2, 4, 11, 16, 22],
                        "obs_layer_ids": [0, 2, 4, 11, 16, 22],
                    }
                (system_dir / "resolved_config.json").write_text(
                    json.dumps(resolved) + "\n"
                )
                if engine == "sparseengine":
                    (system_dir / "operator_runtime_stats.json").write_text(
                        json.dumps(
                            {
                                "status": "success",
                                "world_ranks": [
                                    {
                                        "world_rank": 0,
                                        "bindings": [
                                            {
                                                "operator_type": "test",
                                                "selected_provider": "test_provider",
                                            }
                                        ],
                                        "operators": {},
                                    }
                                ],
                            }
                        )
                        + "\n"
                    )


def test_unified_suite_validator_rejects_any_failed_system(tmp_path):
    systems = ["sengine-vanilla", "vllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    failed = tmp_path / "scenario_a_synthetic/vllm-vanilla/stage_status.json"
    failed.write_text(json.dumps({"status": "failed", "task_exit_code": 3}) + "\n")

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("failed stage scenario_a_synthetic/vllm-vanilla" in error for error in report["errors"])


def test_unified_suite_validator_accepts_ttft_only_probe(tmp_path):
    systems = ["sengine-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    system_dir = tmp_path / "scenario_a_synthetic/sengine-vanilla"
    summary_path = system_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    for row in summary["summary"]:
        row["output_len"] = 1
        row["output_len_min"] = 1
        row["output_len_max"] = 1
        row["decode_token_throughput_tps"] = None
        row["batch_decode_token_throughput_tps"] = None
        row["tpot_ms_mean"] = None
        row["tpot_concurrency_proxy_tps"] = None
        row["decode_metric_status"] = "skipped_by_policy"
        if row["scenario"] == "oversubscribed_churn":
            row["churn_decode_tps_comparison_status"] = "skipped_by_policy"
    summary_path.write_text(json.dumps(summary) + "\n")

    raw_path = system_dir / "raw_samples.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    for row in raw_rows:
        row["output_len"] = 1
        row["trace"]["output_lengths"] = [1, 1]
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "success", report["errors"]


def test_longbench_worker_outputs_merge_without_shared_append(tmp_path):
    for rank, source_idx in ((0, 2), (1, 1)):
        worker_dir = tmp_path / f".worker_rank{rank}"
        worker_dir.mkdir()
        task_row = {"status": "success", "source_idx": source_idx}
        structured = {
            "dataset": "task",
            "sample_idx": source_idx,
            "source_idx": source_idx,
            "status": "success",
        }
        (worker_dir / "task.jsonl").write_text(json.dumps(task_row) + "\n")
        for artifact in (
            "raw_outputs.jsonl",
            "parsed_outputs.jsonl",
            "sample_results.jsonl",
        ):
            (worker_dir / artifact).write_text(json.dumps(structured) + "\n")

    _merge_worker_outputs(
        str(tmp_path),
        datasets=["task"],
        world_size=2,
    )

    for artifact in (
        "task.jsonl",
        "raw_outputs.jsonl",
        "parsed_outputs.jsonl",
        "sample_results.jsonl",
    ):
        rows = [json.loads(line) for line in (tmp_path / artifact).read_text().splitlines()]
        assert [row["source_idx"] for row in rows] == [1, 2]


def test_unified_suite_validator_requires_longbench_structured_artifacts(tmp_path):
    systems = ["sengine-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    missing = (
        tmp_path
        / "scenario_b_longbench/sengine-vanilla/raw_outputs.jsonl"
    )
    missing.unlink()

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("raw_outputs.jsonl" in error for error in report["errors"])


def test_unified_suite_validator_requires_matched_source_ids(tmp_path):
    systems = ["sengine-vanilla", "vllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    mismatched = tmp_path / "scenario_b_longbench/vllm-vanilla/task.jsonl"
    mismatched.write_text(json.dumps({"status": "success", "source_idx": 7}) + "\n")

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("source IDs differ" in error for error in report["errors"])


def test_unified_suite_validator_rejects_method_label_mismatch(tmp_path):
    systems = ["sengine-omnikv"]
    _write_valid_suite_fixture(tmp_path, systems)
    resolved = tmp_path / "scenario_b_longbench/sengine-omnikv/resolved_config.json"
    resolved.write_text(
        json.dumps({"backend": "sparseengine", "sparse_method": "vanilla"}) + "\n"
    )

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("protocol mismatch" in error for error in report["errors"])


def test_unified_suite_validator_rejects_single_layer_omnikv_runtime(tmp_path):
    systems = ["sengine-omnikv"]
    _write_valid_suite_fixture(tmp_path, systems)
    resolved_path = tmp_path / "scenario_b_longbench/sengine-omnikv/resolved_config.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["effective_runtime"]["benchmark_config"]["full_attention_layers"] = [0]
    resolved_path.write_text(json.dumps(resolved) + "\n")

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("invalid OmniKV full_attention_layers" in error for error in report["errors"])


def test_unified_suite_validator_rejects_omnikv_requested_effective_mismatch(
    tmp_path,
):
    systems = ["sengine-omnikv"]
    _write_valid_suite_fixture(tmp_path, systems)
    resolved_path = tmp_path / "scenario_b_longbench/sengine-omnikv/resolved_config.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["effective_runtime"]["benchmark_config"]["full_attention_layers"] = [
        0,
        3,
        9,
    ]
    resolved_path.write_text(json.dumps(resolved) + "\n")

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("layer mismatch" in error for error in report["errors"])


def test_unified_suite_validator_requires_sparse_operator_bindings(tmp_path):
    systems = ["sengine-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    stats_path = (
        tmp_path
        / "scenario_a_synthetic/sengine-vanilla/operator_runtime_stats.json"
    )
    stats_path.write_text(
        json.dumps({"status": "success", "world_ranks": []}) + "\n"
    )

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("empty operator runtime stats" in error for error in report["errors"])


def test_unified_suite_validator_requires_identical_random_traces(tmp_path):
    systems = ["sengine-vanilla", "vllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    path = tmp_path / "scenario_a_synthetic/vllm-vanilla/raw_samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["trace"]["prompt_digests"][0] = "different-digest"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("synthetic random traces differ" in error for error in report["errors"])


def test_unified_suite_validator_rejects_repeated_prompts_across_iterations(tmp_path):
    systems = ["sengine-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    path = tmp_path / "scenario_a_synthetic/sengine-vanilla/raw_samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["trace"]["prompt_digests"] = rows[0]["trace"]["prompt_digests"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("prompts repeat across synthetic iterations" in error for error in report["errors"])


def test_unified_suite_validator_requires_identical_output_length_trace(tmp_path):
    systems = ["sengine-vanilla", "vllm-vanilla"]
    _write_valid_suite_fixture(tmp_path, systems)
    path = tmp_path / "scenario_a_synthetic/vllm-vanilla/raw_samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["trace"]["output_lengths"][0] = 15
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = validate_suite(tmp_path, systems, ["task"], 1)

    assert report["status"] == "failed"
    assert any("synthetic random traces differ" in error for error in report["errors"])


def test_probe_dp_concurrency_covers_global_trace_without_multiplying_each_replica():
    from benchmark.efficiency.bench_probe import _replica_concurrency

    for requests, replicas in ((1, 2), (5, 2), (16, 4)):
        local = _replica_concurrency(requests, {"data_parallel_size": replicas})
        assert local * replicas >= requests
        assert (local - 1) * replicas < requests


def test_native_probe_retains_explicit_graph_capacity_and_rejects_truncated_workload():
    # Short-request measurements must not silently shrink a 128K graph contract.
    from types import SimpleNamespace
    from benchmark.efficiency.bench_probe import _sparse_probe_max_model_len
    args = SimpleNamespace(prompt_lens=[2048, 8192], output_lens=[64])
    assert _sparse_probe_max_model_len(args, {'max_model_len': 131072}) == 131072
    with pytest.raises(ValueError, match='cannot cover'):
        _sparse_probe_max_model_len(args, {'max_model_len': 8192})
