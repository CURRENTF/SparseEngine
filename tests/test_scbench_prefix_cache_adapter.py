from __future__ import annotations

import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCBENCH_DIR = REPO_ROOT / "benchmark" / "scbench"
if str(SCBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCBENCH_DIR))

sys.modules.setdefault(
    "compute_scores",
    types.SimpleNamespace(compute_scores=lambda *args, **kwargs: {"stub": True}),
)
import run_scbench  # noqa: E402
from scripts.benchmarks import compare_scbench_prefix_cache as compare  # noqa: E402


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in str(text)]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class _RecordingSparseVLLMSearch(run_scbench.SparseVLLMSCBenchSearch):
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.calls = []

    def _run_one_request(
        self,
        prompt_token_ids,
        *,
        max_tokens,
        mode,
        turn_idx,
        reusable_prefix_tokens,
    ):
        self.calls.append(
            {
                "prompt_token_ids": list(prompt_token_ids),
                "max_tokens": int(max_tokens),
                "mode": mode,
                "turn_idx": int(turn_idx),
                "reusable_prefix_tokens": int(reusable_prefix_tokens),
            }
        )
        return f"{mode}-{turn_idx}"


def test_sparseengine_scdq_keeps_shared_context_as_reusable_prefix():
    search = _RecordingSparseVLLMSearch()
    example = {
        "prompts": [[10, 11, 12, 13], "Q1", "Q2"],
        "ground_truth": ["gt1", "gt2"],
    }

    output = search.test_scdq(example, max_length=7)

    assert output["answers"] == ["scdq-0", "scdq-1"]
    assert len(search.calls) == 2
    assert search.calls[0]["mode"] == "scdq"
    assert search.calls[0]["turn_idx"] == 0
    assert search.calls[0]["prompt_token_ids"] == [10, 11, 12, 13] + [ord("Q"), ord("1")]
    assert search.calls[0]["reusable_prefix_tokens"] == 4
    assert search.calls[1]["prompt_token_ids"] == [10, 11, 12, 13] + [ord("Q"), ord("2")]
    assert search.calls[1]["reusable_prefix_tokens"] == 4


def test_sparseengine_multiturn_accumulates_history_prefix():
    search = _RecordingSparseVLLMSearch()
    example = {
        "prompts": [[1, 2], "A", "B"],
        "ground_truth": ["gt0", "gt1", "gt2"],
    }

    output = search.test(example, max_length=3, disable_golden_context=False)

    assert output["answers"] == ["multi_turn-0", "multi_turn-1", "multi_turn-2"]
    assert len(search.calls) == 3
    assert search.calls[0]["prompt_token_ids"] == [1, 2]
    assert search.calls[0]["reusable_prefix_tokens"] == 0
    assert search.calls[1]["prompt_token_ids"] == [1, 2, ord("A")]
    assert search.calls[1]["reusable_prefix_tokens"] == 2
    assert search.calls[2]["prompt_token_ids"] == [1, 2, ord("A"), ord("B")]
    assert search.calls[2]["reusable_prefix_tokens"] == 3


def test_sparseengine_adapter_accepts_hit_beyond_planned_session_prefix():
    accounting = run_scbench._cache_reuse_accounting(
        cached_tokens=16,
        planned_session_eligible_tokens=0,
    )
    assert accounting["cached_tokens_beyond_planned_session_prefix"] == 16
    assert run_scbench.SparseVLLMSCBenchSearch._trace_metric_error(
        cached_tokens=16,
        cached_blocks=1,
        max_usable_cache_tokens=16,
        block_size=16,
    ) == ""
    assert run_scbench.SparseVLLMSCBenchSearch._trace_metric_error(
        cached_tokens=32,
        cached_blocks=2,
        max_usable_cache_tokens=16,
        block_size=16,
    ) == "cached_tokens=32 exceeds max_usable_cache_tokens=16."


def test_sparseengine_prefix_summary_reports_reuse(tmp_path):
    trace_path = tmp_path / "prefix_cache_trace_scbench_kv_scdq.jsonl"
    summary_path = tmp_path / "prefix_cache_summary_scbench_kv_scdq.json"
    records = [
        {
            "status": "success",
            "prompt_tokens": 128,
            "generated_tokens": 8,
            "planned_session_eligible_cache_tokens": 0,
            "cached_tokens": 16,
            "cached_blocks": 1,
            "planned_session_prefix_covered_tokens": 999,
            "cached_tokens_beyond_planned_session_prefix": 0,
            "latency_s": 2.0,
            "prefix_cache_stats_before": {"prefix_cache_hit_tokens": 0},
            "prefix_cache_stats_after": {"prefix_cache_hit_tokens": 16},
        },
        {
            "status": "success",
            "prompt_tokens": 128,
            "generated_tokens": 8,
            "planned_session_eligible_cache_tokens": 112,
            "cached_tokens": 112,
            "cached_blocks": 7,
            "latency_s": 1.0,
            "prefix_cache_stats_before": {"prefix_cache_hit_tokens": 16},
            "prefix_cache_stats_after": {"prefix_cache_hit_tokens": 128},
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = run_scbench._write_sparseengine_prefix_summary(trace_path, summary_path)

    assert summary["status"] == "success"
    assert summary["request_count"] == 2
    assert summary["hit_requests"] == 2
    assert summary["total_cached_tokens"] == 128
    assert summary["total_cached_blocks"] == 8
    assert summary["total_planned_session_eligible_cache_tokens"] == 112
    assert summary["total_planned_session_prefix_covered_tokens"] == 112
    assert summary["total_cached_tokens_beyond_planned_session_prefix"] == 16
    assert summary["beyond_planned_session_prefix_hit_requests"] == 1
    assert summary["eligible_cache_hit_rate"] == 1.0
    assert summary["planned_session_prefix_coverage_rate"] == 1.0
    assert summary["prefix_cache_stats_delta"] == {"prefix_cache_hit_tokens": 128}
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_compare_script_builds_isolated_on_off_commands(tmp_path):
    base_hyper = {"gpu_memory_utilization": 0.55}
    off_hyper = compare.build_hyper_param(
        base_hyper,
        enable_prefix_caching=False,
        prefix_cache_block_size=16,
        prefix_cache_salt="unit",
    )
    on_hyper = compare.build_hyper_param(
        base_hyper,
        enable_prefix_caching=True,
        prefix_cache_block_size=16,
        prefix_cache_salt="unit",
    )

    assert base_hyper == {"gpu_memory_utilization": 0.55}
    assert off_hyper["enable_prefix_caching"] is False
    assert on_hyper["enable_prefix_caching"] is True
    assert on_hyper["sparse_method"] == "vanilla"

    cmd = compare.build_scbench_command(
        python="python",
        task="scbench_kv",
        model_name_or_path="/models/tiny",
        output_dir=tmp_path / "on",
        mode="scdq",
        num_eval_examples=1,
        max_turns=5,
        max_seq_length=4096,
        tensor_parallel_size=1,
        hyper_param=on_hyper,
        trust_remote_code=True,
        use_chat_template=False,
        ws=1,
        extra_scbench_args=[],
    )

    assert "benchmark/scbench/run_scbench.py" in cmd[2]
    assert "--same_context_different_query" in cmd
    assert cmd[cmd.index("--attn_type") + 1] == "sparseengine"
    encoded_hyper = json.loads(cmd[cmd.index("--hyper_param") + 1])
    assert encoded_hyper["enable_prefix_caching"] is True
    assert encoded_hyper["prefix_cache_block_size"] == 16

    multi_cmd = compare.build_scbench_command(
        python="python",
        task="scbench_kv",
        model_name_or_path="/models/tiny",
        output_dir=tmp_path / "off",
        mode="multi_turn",
        num_eval_examples=1,
        max_turns=5,
        max_seq_length=4096,
        tensor_parallel_size=1,
        hyper_param=off_hyper,
        trust_remote_code=False,
        use_chat_template=False,
        ws=1,
        extra_scbench_args=[],
    )
    assert "--same_context_different_query" not in multi_cmd
