from __future__ import annotations

import json

from benchmark.sparseengine_regression.manifest import load_manifest
from scripts.benchmarks import run_scbench_sparseengine_methods as runner


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return [ord(ch) for ch in str(text)]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)


def test_cache_trace_accepts_hit_beyond_planned_session_prefix():
    accounting = runner.cache_reuse_accounting(
        cached_tokens=16,
        planned_session_eligible_tokens=0,
    )

    assert accounting == {
        "planned_session_prefix_covered_tokens": 0,
        "cached_tokens_beyond_planned_session_prefix": 16,
        "cache_hit_scope": "beyond_planned_session_prefix",
    }
    assert (
        runner.BatchedSparseVLLMGenerator.trace_metric_error(
            cached_tokens=16,
            cached_blocks=1,
            max_usable_tokens=16,
            block_size=16,
        )
        == ""
    )


def test_cache_trace_rejects_hit_beyond_prompt_usable_prefix():
    assert runner.BatchedSparseVLLMGenerator.trace_metric_error(
        cached_tokens=32,
        cached_blocks=2,
        max_usable_tokens=16,
        block_size=16,
    ) == "cached_tokens=32 exceeds max_usable_cache_tokens=16."


def test_method_runtime_config_enables_prefix_cache_and_aligns_quest_blocks():
    manifest = load_manifest()

    vanilla_cfg = runner.method_runtime_config(
        manifest,
        model_id="qwen3_4b",
        method_id="vanilla",
        batch_size=4,
        max_seq_length=4096,
        tensor_parallel_size=1,
        prefix_cache_block_size=16,
        gpu_memory_utilization=0.5,
        scbench_max_steps=123,
        prefix_cache_salt="unit",
    )
    assert vanilla_cfg["enable_prefix_caching"] is True
    assert vanilla_cfg["decode_graph"] is False
    assert vanilla_cfg["max_num_seqs_in_batch"] == 4
    assert vanilla_cfg["gpu_memory_utilization"] == 0.5
    assert vanilla_cfg["prefix_cache_block_size"] == 16

    quest_cfg = runner.method_runtime_config(
        manifest,
        model_id="qwen3_4b",
        method_id="quest",
        batch_size=4,
        max_seq_length=4096,
        tensor_parallel_size=1,
        prefix_cache_block_size=32,
        gpu_memory_utilization=None,
        scbench_max_steps=123,
        prefix_cache_salt="unit",
    )
    assert quest_cfg["quest_chunk_size"] == 16
    assert quest_cfg["prefix_cache_block_size"] == 16

    mixed_quest_cfg = runner.method_runtime_config(
        manifest,
        model_id="qwen36",
        method_id="quest",
        batch_size=4,
        max_seq_length=32768,
        tensor_parallel_size=1,
        prefix_cache_block_size=4096,
        gpu_memory_utilization=0.8,
        scbench_max_steps=123,
        prefix_cache_salt="unit",
    )
    assert mixed_quest_cfg["quest_chunk_size"] == 16
    assert mixed_quest_cfg["prefix_cache_block_size"] == 4096

    graph_cfg = runner.method_runtime_config(
        manifest,
        model_id="qwen3_4b",
        method_id="quest",
        batch_size=4,
        max_seq_length=4096,
        tensor_parallel_size=1,
        prefix_cache_block_size=16,
        gpu_memory_utilization=0.5,
        scbench_max_steps=123,
        prefix_cache_salt="unit",
        decode_graph=True,
    )
    assert graph_cfg["enable_prefix_caching"] is True
    assert graph_cfg["decode_graph"] is True
    assert graph_cfg["decode_graph_capture_sampling"] is False


def test_build_turn_specs_reuses_multiturn_history_prefix():
    tokenizer = FakeTokenizer()
    state = runner.ExampleState(
        source_idx=7,
        example={"multi_turns": [{"answer": "gt0"}, {"answer": "gt1"}]},
        encoded={"prompts": [[1, 2], "A"], "ground_truth": ["gt0", "gt1"]},
    )

    first_specs = runner.build_turn_specs(
        [state],
        data_name="scbench_kv",
        tokenizer=tokenizer,
        turn_idx=0,
        max_new_tokens=5,
        disable_golden_context=False,
    )
    assert first_specs[0].prompt_token_ids == (1, 2)
    assert first_specs[0].reusable_prefix_tokens == 0

    state.input_ids = list(first_specs[0].prompt_token_ids)
    state.answers.append("pred0")
    second_specs = runner.build_turn_specs(
        [state],
        data_name="scbench_kv",
        tokenizer=tokenizer,
        turn_idx=1,
        max_new_tokens=5,
        disable_golden_context=False,
    )
    assert second_specs[0].prompt_token_ids == (1, 2, ord("A"))
    assert second_specs[0].reusable_prefix_tokens == 2


def test_build_turn_specs_counts_cached_decode_continuation_prefix():
    tokenizer = FakeTokenizer()
    state = runner.ExampleState(
        source_idx=7,
        example={"multi_turns": [{"answer": "gt0"}, {"answer": "gt1"}]},
        encoded={"prompts": [[1, 2], "predA"], "ground_truth": ["gt0", "gt1"]},
        input_ids=[1, 2],
        cache_prefix_token_ids=[1, 2, ord("p"), ord("r"), ord("e"), ord("d")],
    )

    specs = runner.build_turn_specs(
        [state],
        data_name="scbench_kv",
        tokenizer=tokenizer,
        turn_idx=1,
        max_new_tokens=5,
        disable_golden_context=False,
    )

    assert specs[0].prompt_token_ids == (
        1,
        2,
        ord("p"),
        ord("r"),
        ord("e"),
        ord("d"),
        ord("A"),
    )
    assert specs[0].reusable_prefix_tokens == 6


def test_prepare_states_supports_preprocessed_scdq_prompts():
    tokenizer = FakeTokenizer()
    states = runner.prepare_states(
        data_name="scbench_kv",
        examples=[
            (
                3,
                {
                    "prompts": ["abcdef", " question one", " question two"],
                    "ground_truth": ["gt0", "gt1"],
                },
            )
        ],
        tokenizer=tokenizer,
        use_chat_template=False,
        disable_golden_context=False,
        max_input_length=4,
        max_turns=2,
    )

    state = states[0]
    assert state.encoded["prompt_format"] == "preprocessed_scdq"
    assert state.encoded["context_token_ids"] == [ord("a"), ord("b"), ord("e"), ord("f")]
    assert state.encoded["prompts"] == [" question one", " question two"]
    assert state.encoded["ground_truth"] == ["gt0", "gt1"]

    first_specs = runner.build_turn_specs(
        states,
        data_name="scbench_kv",
        tokenizer=tokenizer,
        turn_idx=0,
        max_new_tokens=5,
        disable_golden_context=False,
    )
    assert first_specs[0].prompt_token_ids == tuple(
        [ord("a"), ord("b"), ord("e"), ord("f")] + [ord(ch) for ch in " question one"]
    )
    assert first_specs[0].reusable_prefix_tokens == 0

    state.cache_prefix_token_ids = list(first_specs[0].prompt_token_ids) + [111, 222]
    second_specs = runner.build_turn_specs(
        states,
        data_name="scbench_kv",
        tokenizer=tokenizer,
        turn_idx=1,
        max_new_tokens=5,
        disable_golden_context=False,
    )
    assert second_specs[0].prompt_token_ids == tuple(
        [ord("a"), ord("b"), ord("e"), ord("f")] + [ord(ch) for ch in " question two"]
    )
    assert second_specs[0].reusable_prefix_tokens == 14


def test_prefix_summary_reports_cache_reuse(tmp_path):
    trace_path = tmp_path / "prefix_cache_trace_scbench_kv_multi_turn.jsonl"
    summary_path = tmp_path / "prefix_cache_summary_scbench_kv_multi_turn.json"
    rows = [
        {
            "status": "success",
            "prompt_tokens": 64,
            "generated_tokens": 4,
            "eligible_cache_tokens": 0,
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
            "prompt_tokens": 80,
            "generated_tokens": 4,
            "eligible_cache_tokens": 64,
            "cached_tokens": 64,
            "cached_blocks": 4,
            "latency_s": 1.0,
            "prefix_cache_stats_before": {"prefix_cache_hit_tokens": 16},
            "prefix_cache_stats_after": {"prefix_cache_hit_tokens": 80},
        },
    ]
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = runner.prefix_summary(trace_path, summary_path)

    assert summary["status"] == "success"
    assert summary["request_count"] == 2
    assert summary["hit_requests"] == 2
    assert summary["total_cached_tokens"] == 80
    assert summary["total_cached_blocks"] == 5
    assert summary["total_planned_session_eligible_cache_tokens"] == 64
    assert summary["total_planned_session_prefix_covered_tokens"] == 64
    assert summary["total_cached_tokens_beyond_planned_session_prefix"] == 16
    assert summary["beyond_planned_session_prefix_hit_requests"] == 1
    assert summary["eligible_cache_hit_rate"] == 1.0
    assert summary["planned_session_prefix_coverage_rate"] == 1.0
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
