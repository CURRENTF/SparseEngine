import json
import random
import types

import pytest

from sparseengine.config import Config
from scripts.benchmarks import bench_prefix_cache as bench


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


@pytest.mark.parametrize("prune_status", ["completed", "failed"])
def test_shared_prefix_benchmark_requires_successful_prune(tmp_path, monkeypatch, prune_status):
    """A failed prune must not produce a misleading unpruned speed result."""
    import torch

    batches = []
    requests = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def batch(**kwargs):
        batches.append(kwargs["specs"])
        return []

    def start(tokens, **kwargs):
        requests.append((tokens, kwargs))
        return {"prune_id": "job"}

    monkeypatch.setattr(bench, "_run_request_batch", batch)
    engine = types.SimpleNamespace(
        prefix_cache_prune_start=start,
        run_pending_prefix_prune=lambda: True,
        prefix_cache_prune_status=lambda _: {"status": prune_status},
    )
    args = types.SimpleNamespace(
        shared_prefix_len=16, output_len=2, max_steps_per_round=20,
        shared_prefix_keep_ratio=0.25, shared_prefix_prune_ranges="[[1,5],[8,12]]",
        shared_prompts=1, shared_suffix_len=2, shared_suffix_min_len=None,
    )
    kwargs = dict(llm=engine, tokenizer=FakeTokenizer(), vocab_ids=[5, 6, 7],
                  args=args, rng=random.Random(12), block_size=1,
                  per_turn_path=tmp_path/"per_turn_results.jsonl",
                  raw_output_path=tmp_path/"raw_outputs.jsonl")
    if prune_status == "failed":
        with pytest.raises(RuntimeError, match="Prefix prune failed"):
            bench._run_shared_prefix_workload(**kwargs)
        assert len(batches) == 1
    else:
        bench._run_shared_prefix_workload(**kwargs)
        assert len(batches) == 2
        assert batches[1][0].prompt_token_ids[:16] == requests[0][0]
    assert requests[0][1]["ranges"] == [[1, 5], [8, 12]]
    assert requests[0][1]["keep_tokens"] == 2
    assert json.loads((tmp_path/"prefix_prune.json").read_text())["job"]["status"] == prune_status


def _summary_args():
    return types.SimpleNamespace(
        system_prompt_len=1,
        session_prefix_min_len=1,
        session_prefix_len=1,
        user_min_len=1,
        user_len=1,
        output_len=2,
        turns=2,
        shared_prefix_len=1,
        shared_suffix_min_len=1,
        shared_suffix_len=1,
        history_update="generated",
        require_omnikv_prefill_path=False,
        sink_keep_tokens=0,
        decode_keep_tokens=4,
        recent_keep_tokens=0,
        engine_prefill_chunk_size=4,
        min_performance_prompt_len=0,
        min_cacheable_prefix_len=0,
    )


@pytest.mark.parametrize("keep_ratio", [None, 0.5])
def test_case_timing_uses_only_current_pruning_work(tmp_path, monkeypatch, keep_ratio):
    """Reusing a case directory must not charge a baseline for an earlier prune."""
    import sparseengine
    import torch
    from transformers import AutoTokenizer

    monkeypatch.setattr(bench.sys, "argv", ["bench_prefix_cache", "--model_path", str(tmp_path)])
    args = bench.parse_args()
    args.workloads = "shared_prefix"
    args.shared_prefix_len = 16
    args.shared_suffix_len = 2
    args.shared_prompts = 2
    args.output_len = 2
    args.prefix_cache_block_size = 1
    args.shared_prefix_keep_ratio = keep_ratio
    args.shared_prefix_prune_ranges = "[[0, 16]]"
    prune_path = tmp_path / "prefix_prune.json"
    prune_path.write_text(json.dumps({"elapsed_s": 99.0, "job": {"status": "completed"}}))
    clock = [0.0]
    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: FakeTokenizer())
    monkeypatch.setattr(bench, "_token_vocab", lambda _: [5, 6, 7])
    monkeypatch.setattr(bench, "_cache_stats", lambda _: {})

    def run_prune():
        clock[0] += 3.0
        return True

    engine = types.SimpleNamespace(
        debug_sparse_state_summaries=lambda: [],
        prefix_cache_prune_start=lambda *a, **kw: {"prune_id": "current"},
        run_pending_prefix_prune=run_prune,
        prefix_cache_prune_status=lambda _: {"prune_id": "current", "status": "completed"},
        exit=lambda: None,
    )
    monkeypatch.setattr(sparseengine, "LLM", lambda *a, **kw: engine)

    def batch(**kwargs):
        specs = kwargs["specs"]
        duration = 10.0 if specs[0].phase == "warmup" else 2.0
        clock[0] += duration
        return [dict(
            workload=spec.workload, turn=spec.turn, phase=spec.phase,
            status="success", batch_wall_time_s=duration, ttft_s=0.1,
            latency_s=duration, prompt_tokens=len(spec.prompt_token_ids),
            generated_tokens=spec.output_len, cached_tokens=0,
            eligible_cache_tokens=spec.eligible_cache_tokens,
        ) for spec in specs]

    monkeypatch.setattr(bench, "_run_request_batch", batch)
    bench._run_case_worker("prefix_full", vars(args), str(tmp_path))
    summary = json.loads((tmp_path / "aggregate_metrics.json").read_text())
    expected_elapsed = 2.0 if keep_ratio is None else 5.0
    assert summary["status"] == "success"
    assert summary["elapsed_s"] == expected_elapsed
    assert summary["output_token_throughput"] == pytest.approx(4 / expected_elapsed)


def test_prefix_cache_bench_flags_impossible_cache_hit(tmp_path):
    spec = bench.RequestSpec(
        request_key="req",
        workload="shared_prefix",
        phase="bench",
        session_id=0,
        turn=0,
        prompt_token_ids=[1, 2, 3, 4],
        output_len=2,
        eligible_cache_tokens=4,
        expected_reuse_tokens=4,
    )
    state = bench.RequestState(
        spec=spec,
        seq_id=7,
        add_s=1.0,
        first_token_s=1.5,
        finish_s=2.0,
        generated_token_ids=[9, 10],
        prefix_cache_hit_len=8,
        prefix_cache_hit_blocks=2,
    )

    records = bench._write_request_records(
        states={7: state},
        tokenizer=FakeTokenizer(),
        per_turn_path=tmp_path / "per_turn_results.jsonl",
        raw_output_path=tmp_path / "raw_outputs.jsonl",
        batch_start_s=1.0,
        block_size=4,
    )

    assert records[0]["status"] == "metric_failed"
    assert records[0]["eligible_cache_tokens"] == 4
    assert "exceeds planned_eligible_cache_tokens" in records[0]["error_message"]


def test_prefix_cache_bench_flags_incorrect_chain_reuse(tmp_path):
    spec = bench.RequestSpec(
        request_key="chain",
        workload="multiturn",
        phase="turn",
        session_id=0,
        turn=1,
        prompt_token_ids=[1, 2, 3, 4, 5],
        output_len=2,
        eligible_cache_tokens=3,
        expected_reuse_tokens=3,
    )
    state = bench.RequestState(
        spec=spec,
        seq_id=7,
        add_s=1.0,
        first_token_s=1.5,
        finish_s=2.0,
        generated_token_ids=[9, 10],
        chain_expected=True,
        chain_id="chain-1",
        chain_status="resumed",
        reused_tokens=2,
        prefilled_tokens=3,
        physical_residency=[4, 4],
    )

    records = bench._write_request_records(
        states={7: state},
        tokenizer=FakeTokenizer(),
        per_turn_path=tmp_path / "per_turn_results.jsonl",
        raw_output_path=tmp_path / "raw_outputs.jsonl",
        batch_start_s=1.0,
        block_size=4,
    )

    assert records[0]["status"] == "metric_failed"
    assert "does not match expected_reuse_tokens=3" in records[0]["error_message"]


def test_prefix_cache_bench_does_not_silently_skip_missing_chain_metrics(
    tmp_path,
):
    spec = bench.RequestSpec(
        request_key="chain",
        workload="multiturn",
        phase="turn",
        session_id=0,
        turn=0,
        prompt_token_ids=[1, 2],
        output_len=1,
        eligible_cache_tokens=0,
        expected_reuse_tokens=0,
    )
    state = bench.RequestState(
        spec=spec,
        seq_id=7,
        add_s=1.0,
        first_token_s=1.5,
        finish_s=2.0,
        generated_token_ids=[9],
        chain_expected=True,
        chain_status="disabled",
    )

    records = bench._write_request_records(
        states={7: state},
        tokenizer=FakeTokenizer(),
        per_turn_path=tmp_path / "per_turn_results.jsonl",
        raw_output_path=tmp_path / "raw_outputs.jsonl",
        batch_start_s=1.0,
        block_size=4,
    )

    assert records[0]["status"] == "metric_failed"
    assert "without a chain_id" in records[0]["error_message"]


def test_prefix_cache_bench_summary_preserves_metric_failure_status():
    summary = bench._summarize_records(
        case_name="chain_snapkv",
        case_config=bench.CASE_PRESETS["chain_snapkv"],
        records=[
            {
                "status": "metric_failed",
                "phase": "turn",
                "error_message": "wrong chain boundary",
            }
        ],
        args=_summary_args(),
        engine_kwargs={},
        cache_stats_before={},
        cache_stats_after={},
        peak_memory_gb=0.0,
        elapsed_s=1.0,
    )

    assert summary["status"] == "metric_failed"
    assert summary["failure_status_counts"] == {"metric_failed": 1}


def test_prefix_cache_bench_labels_chain_reuse_as_logical_tokens():
    summary = bench._summarize_records(
        case_name="chain_snapkv",
        case_config=bench.CASE_PRESETS["chain_snapkv"],
        records=[
            {
                "status": "success",
                "phase": "turn",
                "workload": "multiturn",
                "turn": 1,
                "ttft_s": 0.1,
                "latency_s": 0.2,
                "prompt_tokens": 100,
                "generated_tokens": 2,
                "cached_tokens": 80,
                "eligible_cache_tokens": 80,
                "physical_residency_by_layer": [12, 14],
            }
        ],
        args=_summary_args(),
        engine_kwargs={},
        cache_stats_before={},
        cache_stats_after={},
        peak_memory_gb=0.0,
        elapsed_s=1.0,
    )

    assert summary["logical_token_reuse_rate"] == 0.8
    assert "physical_kv_reuse_rate" not in summary


def test_prefix_cache_bench_token_plan_uses_max_bounds():
    args = types.SimpleNamespace(
        system_prompt_len=100,
        session_prefix_min_len=10,
        session_prefix_len=20,
        user_min_len=3,
        user_len=7,
        output_len=5,
        turns=3,
        shared_prefix_len=50,
        shared_suffix_min_len=4,
        shared_suffix_len=9,
    )

    plan = bench._token_count_plan(args)

    assert plan["session_prefix_min"] == 10
    assert plan["session_prefix_max"] == 20
    assert plan["user_min"] == 3
    assert plan["user_max"] == 7
    assert plan["shared_suffix_min"] == 4
    assert plan["shared_suffix_max"] == 9
    assert plan["multiturn_first_prompt"] == 127
    assert plan["multiturn_max_prompt"] == 100 + 20 + 3 * (7 + 5)
    assert plan["shared_prefix_max_prompt"] == 59


def test_prefix_cache_bench_engine_kwargs_are_sparseengine_config_fields():
    args = types.SimpleNamespace(
        gpu_memory_utilization=0.65,
        tensor_parallel_size=1,
        expert_parallel_size=2,
        decode_graph=True,
        max_active_requests=4,
        max_num_batched_tokens=8192,
        engine_prefill_chunk_size=4096,
        sink_keep_tokens=8,
        recent_keep_tokens=256,
        decode_keep_tokens=2048,
        require_omnikv_prefill_path=True,
        full_attention_layers="0,1,2,4,7,14",
        quest_chunk_size=16,
        prefix_cache_block_size=16,
        prefix_cache_max_blocks=None,
        prefix_cache_salt="prefix-cache-bench-test",
        output_len=128,
        max_model_len_margin=64,
        hyper_params="{}",
    )

    kwargs = bench._case_engine_kwargs(args, "baseline_full", max_prompt_len=4096)

    config_fields = set(Config.__dataclass_fields__)
    unknown = sorted(set(kwargs) - config_fields)
    assert unknown == []
    assert kwargs["expert_parallel_size"] == 2
    assert kwargs["decode_graph"] is True


def test_prefix_cache_bench_requires_graph_capture_and_replay_on_every_rank():
    args = _summary_args()
    args.decode_graph = True
    summary = bench._summarize_records(
        case_name="prefix_full",
        case_config=bench.CASE_PRESETS["prefix_full"],
        records=[],
        args=args,
        engine_kwargs={},
        cache_stats_before={},
        cache_stats_after={},
        peak_memory_gb=0.0,
        elapsed_s=1.0,
        decode_graph_before=[
            {
                "world_rank": rank,
                "capture_count": 0,
                "replay_count": 0,
                "eager_static_count": 0,
                "force_eager_count": 0,
            }
            for rank in (0, 1)
        ],
        decode_graph_after=[
            {
                "world_rank": rank,
                "capture_count": 1,
                "replay_count": 3,
                "eager_static_count": 0,
                "force_eager_count": 0,
            }
            for rank in (0, 1)
        ],
    )

    assert summary["decode_graph_failures"] == []
    assert summary["decode_graph_delta"] == [
        {
            "world_rank": rank,
            "capture_count": 1,
            "replay_count": 3,
            "eager_static_count": 0,
            "force_eager_count": 0,
        }
        for rank in (0, 1)
    ]


def test_prefix_cache_bench_accepts_warmup_capture_and_business_replay():
    args = _summary_args()
    args.decode_graph = True
    counters = {
        "capture_count": 1,
        "eager_static_count": 0,
        "force_eager_count": 0,
    }

    summary = bench._summarize_records(
        case_name="prefix_full",
        case_config=bench.CASE_PRESETS["prefix_full"],
        records=[],
        args=args,
        engine_kwargs={},
        cache_stats_before={},
        cache_stats_after={},
        peak_memory_gb=0.0,
        elapsed_s=1.0,
        decode_graph_before=[
            {"world_rank": 0, "replay_count": 1, **counters}
        ],
        decode_graph_after=[
            {"world_rank": 0, "replay_count": 5, **counters}
        ],
    )

    assert summary["status"] == "success"
    assert summary["decode_graph_failures"] == []
    assert summary["decode_graph_delta"][0]["capture_count"] == 0
    assert summary["decode_graph_delta"][0]["replay_count"] == 4


def test_prefix_cache_bench_trace_uses_resolved_sparse_budgets():
    args = types.SimpleNamespace(
        system_prompt_len=100,
        session_prefix_min_len=10,
        session_prefix_len=20,
        user_min_len=3,
        user_len=7,
        output_len=5,
        turns=3,
        shared_prefix_len=50,
        shared_suffix_min_len=4,
        shared_suffix_len=9,
        history_update="generated",
        require_omnikv_prefill_path=True,
        sink_keep_tokens=8,
        decode_keep_tokens=2048,
        recent_keep_tokens=256,
        engine_prefill_chunk_size=16384,
        min_performance_prompt_len=0,
        min_cacheable_prefix_len=0,
    )
    engine_kwargs = {
        "sink_keep_tokens": 4,
        "decode_keep_tokens": 512,
        "recent_keep_tokens": 128,
        "engine_prefill_chunk_size": 4096,
    }

    summary = bench._trace_sparse_path_summary(args, engine_kwargs)

    assert summary["quest_sparse_decode_threshold"] == 4 + 512 + 128
    assert summary["omnikv_prefill_long_text_threshold"] == 4 + 512 + 128 + 4096


def test_prefix_cache_bench_sample_length_validates_bounds():
    rng = random.Random(1)

    for _ in range(20):
        assert 4 <= bench._sample_length(rng, 9, 4) <= 9
    with pytest.raises(ValueError, match="min must be <= max"):
        bench._sample_length(rng, 3, 4)


def test_prefix_cache_bench_multiturn_allows_shared_system_reuse_on_first_turn():
    assert (
        bench._multiturn_reusable_prefix_len(
            turn=0,
            session_id=0,
            shared_system_len=8192,
            history_len=12288,
        )
        == 0
    )
    assert (
        bench._multiturn_reusable_prefix_len(
            turn=0,
            session_id=2,
            shared_system_len=8192,
            history_len=12288,
        )
        == 8192
    )
    assert (
        bench._multiturn_reusable_prefix_len(
            turn=1,
            session_id=0,
            shared_system_len=8192,
            history_len=12352,
        )
        == 12352
    )


def test_prefix_cache_bench_writes_failed_samples_on_batch_error(tmp_path):
    class StuckLLM:
        def __init__(self):
            self.scheduler = type("SchedulerView", (), {"waiting": [], "decoding": []})()
            self.last_step_token_outputs = []
            self.next_seq_id = 1

        def add_request(self, prompt_token_ids, sampling_params):
            del prompt_token_ids, sampling_params
            seq_id = self.next_seq_id
            self.next_seq_id += 1
            return seq_id

        def step(self):
            return [], 0

    specs = [
        bench.RequestSpec(
            request_key=f"req_{idx}",
            workload="shared_prefix",
            phase="bench",
            session_id=idx,
            turn=0,
            prompt_token_ids=[idx, idx + 1],
            output_len=2,
            eligible_cache_tokens=0,
            expected_reuse_tokens=0,
        )
        for idx in range(2)
    ]
    per_turn_path = tmp_path / "per_turn_results.jsonl"
    raw_output_path = tmp_path / "raw_outputs.jsonl"

    with pytest.raises(RuntimeError, match="Exceeded max_steps"):
        bench._run_request_batch(
            llm=StuckLLM(),
            specs=specs,
            tokenizer=FakeTokenizer(),
            per_turn_path=per_turn_path,
            raw_output_path=raw_output_path,
            block_size=4,
            max_steps=1,
        )

    rows = [json.loads(line) for line in per_turn_path.read_text(encoding="utf-8").splitlines()]
    raw_rows = [json.loads(line) for line in raw_output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == ["model_failed", "model_failed"]
    assert [row["request_key"] for row in rows] == ["req_0", "req_1"]
    assert len(raw_rows) == 2
