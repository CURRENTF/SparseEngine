import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.ruler_vt import pred as ruler_pred
from benchmark.sparseengine_regression import run_suite
from benchmark.sparseengine_regression.grading import grade_ruler_quality


def _ruler_args(**overrides):
    values = {
        "hyper_param": None,
        "hyper_param_json": None,
        "allow_prefix_caching": False,
        "prefix_cache_replay": False,
        "require_prefix_cache_hit": False,
        "temperature": 0.0,
        "max_model_len": None,
        "minimum_context_utilization": 0.9,
        "tokens_to_generate": 2,
        "ws": 1,
        "worker_rank": -1,
        "batch_size": 1,
        "max_new_tokens": 2,
        "samples_per_length": 1,
        "no_answer_prefix": False,
        "tokenizer_path": "/tokenizer",
        "model_path": "/model",
        "deltakv_checkpoint_path": None,
        "sparse_method": "vanilla",
    }
    values.update(overrides)
    return Namespace(**values)


def test_ruler_runtime_requires_explicit_prefix_cache_replay_opt_in():
    config_json = json.dumps({"enable_prefix_caching": True})
    args = _ruler_args(hyper_param_json=config_json)
    with pytest.raises(ValueError, match="--allow-prefix-caching"):
        ruler_pred.build_infer_config(args, [1024])

    args.allow_prefix_caching = True
    args.prefix_cache_replay = True
    args.require_prefix_cache_hit = True
    config = ruler_pred.build_infer_config(args, [1024])
    assert config["enable_prefix_caching"] is True
    assert config["max_model_len"] == 2050


def test_ruler_rejects_data_parallel_workers_with_tensor_parallelism():
    args = _ruler_args(
        ws=2,
        hyper_param_json=json.dumps({"tensor_parallel_size": 2}),
    )

    with pytest.raises(ValueError, match=r"--ws > 1.*tensor_parallel_size > 1"):
        ruler_pred.build_infer_config(args, [1024])


def test_ruler_parsed_artifact_preserves_model_failure(tmp_path: Path):
    sample = ruler_pred.VTSample(
        index=0,
        context_length=1024,
        input="prompt",
        outputs=["answer"],
        length=1000,
        answer_prefix=" Answer:",
        query="needle",
    )
    raw_path = tmp_path / "raw_outputs.jsonl"
    parsed_path = tmp_path / "parsed_outputs.jsonl"
    result_path = tmp_path / "per_sample_results.jsonl"

    ruler_pred._append_batch_records(
        batch=[sample],
        prompts=["prompt Answer:"],
        predictions=[""],
        status="model_failed",
        error="RuntimeError('CUDA out of memory')",
        raw_path=raw_path,
        parsed_path=parsed_path,
        result_path=result_path,
    )

    parsed = ruler_pred.read_jsonl(parsed_path)
    assert parsed[0]["status"] == "model_failed"
    assert parsed[0]["error"] == "RuntimeError('CUDA out of memory')"


def test_ruler_prefix_cache_replay_records_hits_and_output_equivalence(
    tmp_path: Path,
    monkeypatch,
):
    class RuntimeState:
        hit_requests = 0
        hit_tokens = 0

        def free_slot_stats(self):
            return {
                "prefix_cache_hit_requests": self.hit_requests,
                "prefix_cache_hit_tokens": self.hit_tokens,
            }

    runtime = RuntimeState()
    seen: set[str] = set()

    def generate(prompt, **_kwargs):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        for item in prompts:
            if item in seen:
                runtime.hit_requests += 1
                runtime.hit_tokens += 16
            seen.add(item)
        outputs = ["AAAAA BBBBB" for _ in prompts]
        return outputs[0] if isinstance(prompt, str) else outputs

    generate._sparseengine_llm = SimpleNamespace(
        model_runner=SimpleNamespace(runtime_state=runtime)
    )
    monkeypatch.setattr(
        ruler_pred,
        "get_sparseengine_generate_api",
        lambda **_kwargs: generate,
    )
    args = _ruler_args(
        hyper_param_json=json.dumps({"enable_prefix_caching": True}),
        allow_prefix_caching=True,
        prefix_cache_replay=True,
        require_prefix_cache_hit=True,
    )
    samples = [
        ruler_pred.VTSample(
            index=0,
            context_length=32,
            input="prompt ",
            outputs=["AAAAA", "BBBBB"],
            length=30,
            answer_prefix="answer",
            query="12345",
        )
    ]
    infer_config = ruler_pred.build_infer_config(args, [32])
    ruler_pred.evaluate_samples(args, samples, tmp_path, infer_config)
    summary = ruler_pred.write_prefix_cache_summary(tmp_path, 1)
    ruler_pred.validate_run(args, tmp_path, [32])

    assert summary["replay_hit_requests"] == 1
    assert summary["replay_hit_tokens"] == 16
    assert summary["output_mismatch_indices"] == []
    primary = ruler_pred.read_jsonl(tmp_path / "per_sample_results.jsonl")
    replay = ruler_pred.read_jsonl(
        tmp_path / "per_sample_results_prefix_cache_replay.jsonl"
    )
    assert primary[0]["prediction"] == replay[0]["prediction"]
    assert primary[0]["score"] == replay[0]["score"] == 1.0


def test_ruler_regression_command_enables_exact_cache_replay():
    command = run_suite._ruler_command(
        task="vt",
        task_config={
            "tokens_to_generate": 30,
            "max_new_tokens": 30,
            "num_chains": 1,
            "num_hops": 4,
        },
        model_id="qwen3_4b",
        method_id="vanilla",
        model={"model_path": "/model", "tokenizer_path": "/tokenizer"},
        method={
            "sparse_method": "vanilla",
            "requires_compressor": False,
            "config": {"engine_prefill_chunk_size": 4096},
        },
        ruler={
            "tasks": ["vt"],
            "task_configs": {
                "vt": {
                    "tokens_to_generate": 30,
                    "max_new_tokens": 30,
                    "num_chains": 1,
                    "num_hops": 4,
                }
            },
            "context_lengths": [16384, 32768],
            "samples_per_length": 2,
            "batch_size": 1,
            "worker_world_size": 1,
            "minimum_context_utilization": 0.9,
            "temperature": 0.0,
            "enable_prefix_caching": True,
            "prefix_cache_block_size": 16,
        },
        performance={"decode_graph": False},
        output_root=Path("/output"),
    )

    assert "--allow-prefix-caching" in command
    assert "--prefix-cache-replay" in command
    assert "--require-prefix-cache-hit" in command
    assert command[command.index("--task") + 1] == "vt"
    config = json.loads(command[command.index("--hyper-param-json") + 1])
    assert config["enable_prefix_caching"] is True
    assert config["prefix_cache_block_size"] == 16


def test_ruler_regression_command_rejects_data_parallel_with_tp():
    with pytest.raises(
        ValueError,
        match=r"worker_world_size > 1.*tensor_parallel_size > 1",
    ):
        run_suite._ruler_command(
            task="vt",
            task_config={
                "tokens_to_generate": 30,
                "max_new_tokens": 30,
                "num_chains": 1,
                "num_hops": 4,
            },
            model_id="qwen3_4b",
            method_id="vanilla",
            model={"model_path": "/model", "tokenizer_path": "/tokenizer"},
            method={
                "sparse_method": "vanilla",
                "requires_compressor": False,
                "config": {},
            },
            ruler={
                "context_lengths": [16384],
                "samples_per_length": 1,
                "batch_size": 1,
                "worker_world_size": 2,
                "minimum_context_utilization": 0.9,
                "temperature": 0.0,
            },
            performance={"tensor_parallel_size": 2, "decode_graph": False},
            output_root=Path("/output"),
        )


def test_ruler_quality_grades_each_context_length_with_configured_loss_limit():
    assert grade_ruler_quality(
        100.0,
        100.0,
        minimum_vanilla_score=80.0,
        maximum_score_loss=5.0,
    ).grade == "A"
    assert grade_ruler_quality(
        100.0,
        97.5,
        minimum_vanilla_score=80.0,
        maximum_score_loss=5.0,
    ).grade == "B"
    assert grade_ruler_quality(
        100.0,
        95.0,
        minimum_vanilla_score=80.0,
        maximum_score_loss=5.0,
    ).grade == "C"
    assert grade_ruler_quality(
        100.0,
        90.0,
        minimum_vanilla_score=80.0,
        maximum_score_loss=5.0,
    ).grade == "D"


def test_ruler_pair_grading_does_not_hide_one_long_context_regression(
    tmp_path: Path,
):
    vanilla_root = tmp_path / "vanilla"
    sparse_root = tmp_path / "sparse"
    vanilla_root.mkdir()
    sparse_root.mkdir()
    dataset = [
        {
            "index": index,
            "context_length": context_length,
            "input": f"prompt-{index}",
            "outputs": [f"answer-{index}"],
            "length": context_length - 1,
            "answer_prefix": "answer: ",
            "others": {"query": str(index), "task": "vt"},
        }
        for index, context_length in enumerate((16384, 32768))
    ]
    for root in (vanilla_root, sparse_root):
        (root / "dataset.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in dataset),
            encoding="utf-8",
        )
    vanilla_rows = [
        {
            "index": index,
            "context_length": context_length,
            "length": context_length - 1,
            "score": 1.0,
            "status": "success",
        }
        for index, context_length in enumerate((16384, 32768))
    ]
    sparse_rows = [dict(vanilla_rows[0]), {**vanilla_rows[1], "score": 0.9}]
    (vanilla_root / "per_sample_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in vanilla_rows),
        encoding="utf-8",
    )
    (sparse_root / "per_sample_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sparse_rows),
        encoding="utf-8",
    )

    grades = run_suite._grade_ruler_pair(
        vanilla_root,
        sparse_root,
        ruler={
            "context_lengths": [16384, 32768],
            "samples_per_length": 1,
            "minimum_vanilla_score": 80.0,
            "maximum_score_loss": 5.0,
            "minimum_context_utilization": 0.9,
        },
        task="vt",
    )

    assert [(length, grade.grade) for length, grade in grades] == [
        (16384, "A"),
        (32768, "D"),
    ]

    underfilled = [{**vanilla_rows[0], "length": 1000}, vanilla_rows[1]]
    (vanilla_root / "per_sample_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in underfilled),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not exercise context_length=16384"):
        run_suite._grade_ruler_pair(
            vanilla_root,
            sparse_root,
            ruler={
                "context_lengths": [16384, 32768],
                "samples_per_length": 1,
                "minimum_vanilla_score": 80.0,
                "maximum_score_loss": 5.0,
                "minimum_context_utilization": 0.9,
            },
            task="vt",
        )
