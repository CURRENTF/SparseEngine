import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from benchmark.sparseengine_regression.longbench_mini import select_longbench_mini_samples
from benchmark.long_bench.pred import build_chat, strip_thinking_content
from benchmark.sparseengine_regression.run_suite import (
    _require_successful_perf_matrix,
    _require_synchronized_step_timing,
    main as run_suite_main,
)
from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager.base import CacheManager
from sparseengine.distributed import ParallelContext, ParallelGroup


def _single_process_parallel_context() -> ParallelContext:
    group = ParallelGroup(process_group=None, ranks=(0,), rank=0, size=1)
    return ParallelContext(world=group, moe_tp=group, attn_tp=group, moe_ep=group, attn_dp=group)


class FakeTokenizer:
    bos_token = None
    chat_template = None

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(range(len(str(text).split())))


class FakeCacheManager(CacheManager):
    def __init__(self):
        hf_config = types.SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=4,
            hidden_size=4,
            num_attention_heads=1,
            dtype=torch.float16,
        )
        config = types.SimpleNamespace(
            hf_config=hf_config,
            runtime_layout=RuntimeLayout.dense(2),
            max_model_len=10,
            max_num_seqs_in_gpu=2,
            max_num_seqs_in_batch=2,
            num_kvcache_slots=16,
        )
        super().__init__(config, _single_process_parallel_context())
        self.kv_cache = torch.empty((2, 2, 16, 1, 4), dtype=torch.float16)
        self.buffer_req_to_token_slots = torch.empty((2, 10), dtype=torch.int32)
        self.latent_scales = torch.empty((2, 16, 1), dtype=torch.float16)
        self.row_seq_lens = np.array([3, 2], dtype=np.int32)
        self._num_free_slots = 8

    @property
    def num_free_slots(self):
        return self._num_free_slots

    def allocate_kv_cache(self):
        raise NotImplementedError

    def get_layer_batch_states(self, layer_idx):
        raise NotImplementedError

    def get_layer_kv_cache(self, layer_idx):
        raise NotImplementedError

    def get_layer_store_view(self, layer_idx):
        raise NotImplementedError

    def get_layer_compute_tensors(self, layer_idx, selection=None):
        del selection
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx):
        raise NotImplementedError

    def free_seq(self, seq_id):
        raise NotImplementedError

    def free_part_slots(self, layer_idx, seq, keep_indices):
        raise NotImplementedError

    def _prepare_prefill(self, seqs):
        raise NotImplementedError

    def _prepare_decode(self, seqs):
        raise NotImplementedError


class SparseVLLMRegressionGradingTest(unittest.TestCase):
    def test_regression_rejects_unsynchronized_success_throughput(self):
        with self.assertRaisesRegex(RuntimeError, "requires synchronized"):
            _require_synchronized_step_timing(
                [
                    {
                        "status": "SUCCESS",
                        "method": "h2o",
                        "synchronize_step_timing": False,
                    }
                ],
                artifact=Path("/tmp/perf.jsonl"),
            )

    def test_regression_requires_complete_successful_perf_matrix(self):
        rows = [
            {
                "status": "SUCCESS",
                "method": method,
                "length": length,
                "batch_size": batch_size,
            }
            for method in ("vanilla", "h2o")
            for length in (32000, 64000)
            for batch_size in (4, 8)
        ]

        _require_successful_perf_matrix(
            rows,
            methods=("vanilla", "h2o"),
            lengths=(32000, 64000),
            batch_sizes=(4, 8),
            artifact=Path("/tmp/perf.jsonl"),
        )

        with self.assertRaisesRegex(RuntimeError, "missing"):
            _require_successful_perf_matrix(
                rows[:-1],
                methods=("vanilla", "h2o"),
                lengths=(32000, 64000),
                batch_sizes=(4, 8),
                artifact=Path("/tmp/perf.jsonl"),
            )

        failed_rows = copy.deepcopy(rows)
        failed_rows[-1]["status"] = "FAILED"
        with self.assertRaisesRegex(RuntimeError, "non-success"):
            _require_successful_perf_matrix(
                failed_rows,
                methods=("vanilla", "h2o"),
                lengths=(32000, 64000),
                batch_sizes=(4, 8),
                artifact=Path("/tmp/perf.jsonl"),
            )

    def test_regression_rejects_duplicate_or_unexpected_perf_cases(self):
        row = {
            "status": "SUCCESS",
            "method": "vanilla",
            "length": 32000,
            "batch_size": 4,
        }
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            _require_successful_perf_matrix(
                [row, dict(row)],
                methods=("vanilla",),
                lengths=(32000,),
                batch_sizes=(4,),
                artifact=Path("/tmp/perf.jsonl"),
            )

        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            _require_successful_perf_matrix(
                [row, {**row, "method": "h2o"}],
                methods=("vanilla",),
                lengths=(32000,),
                batch_sizes=(4,),
                artifact=Path("/tmp/perf.jsonl"),
            )

    def test_perf_layer_applies_h2o_prefill_non_regression_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "qwen25-model"
            model_path.mkdir()

            def fake_run_and_record(summary, cmd, **kwargs):
                del kwargs
                output_path = Path(cmd[cmd.index("--output_jsonl") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                rows = [
                    {
                        "status": "SUCCESS",
                        "method": "vanilla",
                        "length": length,
                        "batch_size": batch_size,
                        "prefill_tp": 100.0,
                        "decode_tp": 100.0,
                        "decode_graph_expected": True,
                        "decode_graph_active": True,
                        "synchronize_step_timing": True,
                    }
                    for length in (32000, 64000)
                    for batch_size in (4, 8)
                ]
                rows.extend(
                    {
                        "status": "SUCCESS",
                        "method": "h2o",
                        "length": length,
                        "batch_size": batch_size,
                        "prefill_tp": 99.0,
                        "decode_tp": 110.0,
                        "decode_graph_expected": True,
                        "decode_graph_active": True,
                        "synchronize_step_timing": True,
                        "memory_accounting": {"observed_savings": 0.3},
                    }
                    for length in (32000, 64000)
                    for batch_size in (4, 8)
                )
                output_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                summary["commands"].append({"status": "success", "cmd": cmd})

            argv = [
                "run_suite.py",
                "--layer",
                "perf",
                "--models",
                "qwen25_7b",
                "--methods",
                "h2o",
                "--run_id",
                "h2o-prefill-gate",
                "--output_root",
                str(root),
            ]
            with patch.object(sys, "argv", argv), patch.dict(
                os.environ,
                {
                    "DELTAKV_MODEL_QWEN25_7B": str(model_path),
                    "DELTAKV_TOKENIZER_QWEN25_7B": str(model_path),
                },
                clear=False,
            ), patch(
                "benchmark.sparseengine_regression.run_suite._run_and_record",
                side_effect=fake_run_and_record,
            ):
                with self.assertRaisesRegex(RuntimeError, "Required regression gates failed"):
                    run_suite_main()

            summary = json.loads(
                (
                    root
                    / "sparseengine_regression"
                    / "h2o-prefill-gate"
                    / "grade_summary.json"
                ).read_text(encoding="utf-8")
            )
            perf_grades = [
                grade
                for grade in summary["grades"]
                if grade["name"] == "performance" and grade["method"] == "h2o"
            ]
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["worst_required_grade"], "D")
            self.assertEqual(len(perf_grades), 4)
            self.assertTrue(all(grade["grade"] == "D" for grade in perf_grades))
            self.assertTrue(
                all(grade["metrics"]["prefill_speedup"] == 0.99 for grade in perf_grades)
            )

    def test_legacy_raw_prompt_tasks_skip_chat_template(self):
        class QwenTokenizer:
            chat_template = "template"

            def apply_chat_template(self, messages, **kwargs):
                raise AssertionError("legacy raw-prompt tasks must not apply a chat template")

        tokenizer = QwenTokenizer()

        prompt = build_chat(tokenizer, "classify this", "trec", thinking_mode="off")

        self.assertEqual(prompt, "classify this")
        self.assertFalse(hasattr(tokenizer, "kwargs"))
        self.assertEqual(
            build_chat(tokenizer, "classify this", "trec", no_chat_template=True, thinking_mode="off"),
            "classify this",
        )

    def test_strip_thinking_content_requires_complete_reasoning(self):
        self.assertEqual(
            strip_thinking_content("reasoning\n</think>\nfinal answer"),
            "final answer",
        )
        with self.assertRaisesRegex(ValueError, "ended before </think>"):
            strip_thinking_content("truncated reasoning")

    def test_longbench_mini_selects_fixed_long_samples(self):
        data = [
            {"context": "short"},
            {"context": "one two three four five"},
            {"context": "one two three four five six"},
            {"context": "tiny"},
        ]
        selected, meta = select_longbench_mini_samples(
            data=data,
            tokenizer=FakeTokenizer(),
            dataset="lcc",
            prompt_format="{context}",
            min_prompt_tokens=5,
            samples_per_task=2,
            min_required_samples=2,
            no_chat_template=True,
        )
        self.assertEqual(meta["status"], "success")
        self.assertEqual([item.source_idx for item in selected], [1, 2])

        selected, meta = select_longbench_mini_samples(
            data=data,
            tokenizer=FakeTokenizer(),
            dataset="lcc",
            prompt_format="{context}",
            min_prompt_tokens=10,
            samples_per_task=2,
            min_required_samples=1,
            no_chat_template=True,
        )
        self.assertEqual(selected, [])
        self.assertEqual(meta["status"], "skipped_by_policy")

    def test_cache_manager_memory_accounting_fake_tensors(self):
        manager = FakeCacheManager()
        accounting = manager.memory_accounting()
        self.assertEqual(accounting["status"], "success")
        self.assertEqual(accounting["dense_baseline_bytes"], 512)
        self.assertEqual(accounting["kv_or_latent_tensor_bytes"], 512)
        self.assertEqual(accounting["slot_map_bytes"], 80)
        self.assertEqual(accounting["scale_min_metadata_bytes"], 64)
        self.assertEqual(accounting["logical_live_kv_bytes"], 160)
        self.assertEqual(accounting["allocated_tensor_bytes"], 656)
        self.assertLess(accounting["observed_savings"], 0)


if __name__ == "__main__":
    unittest.main()
