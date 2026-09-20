import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmark.multimodal.video_qa import audit_livevlm_table4 as livevlm_audit
from benchmark.multimodal.video_qa import streamingbench
from benchmark.multimodal.image_qa import vqav2
from sparseengine.config import Config


class ResearchFailFastTest(unittest.TestCase):
    def test_streamingbench_missing_videos_fail_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_dir = root / "StreamingBench"
            csv_dir.mkdir()
            video_dir = root / "videos"
            video_dir.mkdir()
            csv_path = csv_dir / "Real_Time_Visual_Understanding.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "question_id",
                        "task_type",
                        "question",
                        "time_stamp",
                        "answer",
                        "options",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "question_id": "Real-Time Visual Understanding_sample_1_1",
                        "task_type": "Real-Time Visual Understanding",
                        "question": "What is shown?",
                        "time_stamp": "00:00:05",
                        "answer": "A",
                        "options": "['one', 'two', 'three', 'four']",
                    }
                )

            args = SimpleNamespace(
                dataset_dir=str(root),
                csv_dir="",
                video_dir=str(video_dir),
                tasks="real",
                allow_missing_videos=False,
                sample_start=0,
                num_samples=-1,
            )
            with self.assertRaisesRegex(FileNotFoundError, "Missing videos"):
                streamingbench.load_streamingbench_rows(args)

            args.allow_missing_videos = True
            rows, info = streamingbench.load_streamingbench_rows(args)
            self.assertEqual(rows, [])
            self.assertEqual(info["missing_video_rows"], 1)

    def test_streamingbench_rejects_silent_row_defaults(self):
        with self.assertRaisesRegex(ValueError, "empty timestamp"):
            streamingbench.parse_timestamp("")
        with self.assertRaisesRegex(ValueError, "exactly 4 choices"):
            streamingbench.parse_options("['one', 'two']")

    def test_streamingbench_repairs_labeled_option_fragments_explicitly(self):
        options, reason = streamingbench.parse_options_with_repair(
            [
                "A. first choice",
                "B. second",
                "choice",
                "C. third choice",
                "",
                "D. fourth choice",
            ]
        )
        self.assertEqual(reason, "reconstructed_repeated_or_fragmented_labeled_options")
        self.assertEqual(options, ["A. first choice", "B. second choice", "C. third choice", "D. fourth choice"])

        options, reason = streamingbench.parse_options_with_repair(
            [
                "A. truncated",
                "B. truncated",
                "C. third",
                "D. fourth.A. complete first",
                "B. complete second",
                "C. complete third",
                "D. complete fourth",
            ]
        )
        self.assertEqual(reason, "reconstructed_repeated_or_fragmented_labeled_options")
        self.assertEqual(
            options,
            ["A. complete first", "B. complete second", "C. complete third", "D. complete fourth"],
        )

    def test_streamingbench_validates_runtime_args(self):
        args = SimpleNamespace(
            num_samples=1,
            sample_start=-1,
            batch_size=1,
            num_video_frames=32,
            max_new_tokens=8,
            log_every=1,
            frame_load_workers=1,
            preprocess_prefetch_batches=0,
            context_seconds=60.0,
        )
        with self.assertRaisesRegex(ValueError, "sample_start"):
            streamingbench.validate_args(args)
        args.sample_start = 0
        args.context_seconds = -2.0
        with self.assertRaisesRegex(ValueError, "context_seconds"):
            streamingbench.validate_args(args)
        args.context_seconds = 60.0
        args.preprocess_prefetch_batches = 2
        with self.assertRaisesRegex(ValueError, "preprocess_prefetch_batches"):
            streamingbench.validate_args(args)

    def test_streamingbench_livevlm_table4_scope_and_stats(self):
        args = SimpleNamespace(
            streamingbench_profile="livevlm_table4",
            tasks="real",
            num_video_frames=8,
            context_seconds=60.0,
            frame_sampling_backend="ffmpeg",
        )
        streamingbench.apply_streamingbench_profile(args)
        self.assertEqual(args.tasks, "livevlm_table4")
        self.assertEqual(streamingbench.list_tasks(args.tasks), ["real", "omni", "contextual"])
        self.assertEqual(args.num_video_frames, 32)
        self.assertEqual(args.context_seconds, -1.0)
        self.assertEqual(args.frame_sampling_backend, "decord")

        records = [
            {
                "task_type": "Object Perception",
                "status": "success",
                "correct": True,
            },
            {
                "task_type": "Causal Reasoning",
                "status": "parse_failed",
                "correct": False,
            },
            {
                "task_type": "Anomaly Context Understanding",
                "status": "success",
                "correct": False,
            },
        ]
        stats = streamingbench.compute_livevlm_table4_stats(records)
        self.assertEqual(stats["overall"]["total"], 3)
        self.assertEqual(stats["overall"]["correct"], 1)
        self.assertEqual(stats["overall"]["status_counts"], {"success": 2, "parse_failed": 1})
        self.assertEqual(stats["expected_llava_onevision_7b_overall_pct"], 58.85)
        self.assertEqual(stats["expected_overall_row_count"], 4000)
        self.assertEqual(stats["subtasks"][0]["abbr"], "OP")
        self.assertEqual(stats["subtasks"][0]["expected_rows"], 369)
        self.assertFalse(stats["subtasks"][0]["matches_expected_rows"])
        self.assertEqual(stats["subtasks"][0]["expected_llava_onevision_7b_pct"], 80.38)
        self.assertEqual(stats["subtasks"][1]["status_counts"], {"parse_failed": 1})
        self.assertEqual(stats["overall_extra_subtasks"][0]["abbr"], "ACU")
        self.assertEqual(stats["overall_extra_subtasks"][0]["expected_rows"], 250)
        self.assertTrue(stats["overall_extra_subtasks"][0]["used_for_paper_overall"])
        self.assertAlmostEqual(stats["expected_display_weighted_accuracy_pct"], 77.30)
        self.assertAlmostEqual(stats["implied_expected_extra_subtasks_accuracy_pct"], 21.95)

    def test_streamingbench_choice_parse_modes(self):
        self.assertEqual(streamingbench.extract_choice(" A", "official_first_char"), "A")
        self.assertEqual(streamingbench.extract_choice("The answer is A", "official_first_char"), "T")
        self.assertEqual(streamingbench.extract_choice("The answer is A", "robust"), "A")

    def test_streamingbench_video_resolution_uses_task_type_hints(self):
        video_index = {
            1: [
                Path("/tmp/videos/Emotion Recognition/sample_1.mp4"),
                Path("/tmp/videos/Source Discrimination/sample_1.mp4"),
                Path("/tmp/videos/Anomaly Context Understanding/sample_1.mp4"),
                Path("/tmp/videos/Misleading Context Understanding/sample_1.mp4"),
            ]
        }
        self.assertEqual(
            streamingbench.resolve_video_path(video_index, "omni", "Source Discrimination", 1),
            Path("/tmp/videos/Source Discrimination/sample_1.mp4"),
        )
        self.assertEqual(
            streamingbench.resolve_video_path(video_index, "contextual", "Misleading Context Recognition", 1),
            Path("/tmp/videos/Misleading Context Understanding/sample_1.mp4"),
        )

    def test_streamingbench_livevlm_table4_requires_full_row_scope(self):
        args = SimpleNamespace(
            streamingbench_profile="livevlm_table4",
            num_samples=-1,
            sample_start=0,
            allow_missing_videos=False,
        )
        rows = [{"task_type": "Object Perception"}]
        with self.assertRaisesRegex(RuntimeError, "4000-row StreamingBench scope"):
            streamingbench.validate_livevlm_table4_rows(args, rows)

        rows = []
        for task_type, count in streamingbench.LIVEVLM_TABLE4_EXPECTED_TASK_TYPE_COUNTS.items():
            rows.extend({"task_type": task_type} for _ in range(count))
        streamingbench.validate_livevlm_table4_rows(args, rows)

        rows[-1] = {"task_type": "Object Perception"}
        with self.assertRaisesRegex(RuntimeError, "mismatched_task_type_counts"):
            streamingbench.validate_livevlm_table4_rows(args, rows)

        args.num_samples = 4
        streamingbench.validate_livevlm_table4_rows(args, rows)

    def test_livevlm_table4_audit_requires_complete_metrics(self):
        subtasks = []
        for abbr, (task_type, expected_rows) in livevlm_audit.EXPECTED_VISIBLE_SUBTASKS.items():
            subtasks.append(
                {
                    "abbr": abbr,
                    "task_type": task_type,
                    "total": expected_rows,
                    "expected_rows": expected_rows,
                    "matches_expected_rows": True,
                    "correct": expected_rows // 2,
                    "accuracy_pct": 50.0,
                    "expected_llava_onevision_7b_pct": 50.0,
                    "delta_vs_expected_pct": 0.0,
                }
            )
        extra_subtasks = []
        for abbr, (task_type, expected_rows) in livevlm_audit.EXPECTED_EXTRA_SUBTASKS.items():
            extra_subtasks.append(
                {
                    "abbr": abbr,
                    "task_type": task_type,
                    "total": expected_rows,
                    "expected_rows": expected_rows,
                    "matches_expected_rows": True,
                    "correct": expected_rows // 2,
                    "accuracy_pct": 50.0,
                }
            )
        metrics = {
            "method": "vanilla",
            "num_samples": 4000,
            "status_counts": {"success": 4000},
            "livevlm_table4_stats": {
                "expected_llava_onevision_7b_overall_pct": 58.85,
                "expected_overall_row_count": 4000,
                "expected_display_weighted_accuracy_pct": 50.0,
                "implied_expected_extra_subtasks_accuracy_pct": 120.8,
                "overall": {
                    "total": 4000,
                    "correct": 2000,
                    "accuracy_pct": 50.0,
                    "matches_expected_row_count": True,
                    "status_counts": {"success": 4000},
                },
                "subtasks": subtasks,
                "overall_extra_subtasks": extra_subtasks,
            },
        }
        summary = livevlm_audit.audit_metrics(metrics)
        self.assertEqual(summary["overall_total"], 4000)
        self.assertEqual(summary["visible_subtask_count"], 14)
        self.assertEqual(summary["overall_extra_subtask_count"], 2)
        self.assertAlmostEqual(summary["observed_extra_subtasks_accuracy_pct"], 50.0)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            task_counts = livevlm_audit.expected_task_type_counts()
            rows = []
            for task_type, count in task_counts.items():
                for _ in range(count):
                    question_id = f"{task_type.replace(' ', '_')}_{len(rows)}"
                    rows.append(
                        {
                            "question_id": question_id,
                            "task_type": task_type,
                            "answer": "A",
                            "prediction": "A",
                            "status": "success",
                            "correct": True,
                            "raw_prediction": "A",
                            "parsed_text": "A",
                        }
                    )

            (output_dir / "vanilla_raw_outputs.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "question_id": row["question_id"],
                            "sample_id": idx,
                            "task": "real",
                            "task_type": row["task_type"],
                            "raw_prediction": row["raw_prediction"],
                        }
                    )
                    for idx, row in enumerate(rows)
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "vanilla_parsed_outputs.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "question_id": row["question_id"],
                            "prediction": row["prediction"],
                            "answer": row["answer"],
                            "status": row["status"],
                            "correct": row["correct"],
                        }
                    )
                    for row in rows
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "vanilla_per_sample_results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            (output_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "model_path": str(livevlm_audit.DEFAULT_EXPECTED_MODEL_PATH),
                        "methods": "vanilla",
                        "streamingbench_profile": "livevlm_table4",
                        "tasks": "livevlm_table4",
                        "num_video_frames": 32,
                        "context_seconds": -1.0,
                        "frame_sampling_backend": "decord",
                        "choice_parse_mode": "official_first_char",
                        "sample_start": 0,
                        "num_samples_arg": -1,
                        "evaluated_sample_count": 4000,
                        "seed": 0,
                        "decoding": {
                            "max_new_tokens": 8,
                            "do_sample": False,
                            "torch_dtype": "float16",
                            "attn_implementation": "flash_attention_2",
                        },
                        "dataset_info": {"evaluated_task_type_counts": task_counts},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "last_streamingbench_result.json").write_text(
                json.dumps([{"method": "vanilla", "num_samples": 4000}]) + "\n",
                encoding="utf-8",
            )
            artifact_summary = livevlm_audit.audit_output_artifacts(
                output_dir,
                metrics,
                str(livevlm_audit.DEFAULT_EXPECTED_MODEL_PATH),
                "flash_attention_2",
                "livevlm_table4",
                -1.0,
            )
            self.assertEqual(artifact_summary["per_sample_rows"], 4000)

        metrics["livevlm_table4_stats"]["overall"]["total"] = 3999
        with self.assertRaisesRegex(RuntimeError, "Overall row count mismatch"):
            livevlm_audit.audit_metrics(metrics)

        metrics["livevlm_table4_stats"]["overall"]["total"] = 4000
        metrics["livevlm_table4_stats"]["subtasks"][0]["total"] = 1
        with self.assertRaisesRegex(RuntimeError, "visible subtask row-count mismatch"):
            livevlm_audit.audit_metrics(metrics)

    def test_vqav2_soft_score_and_validation(self):
        answers = ["cat", "cat", "cat", "dog", "dog", "dog", "dog", "dog", "dog", "dog"]
        self.assertAlmostEqual(vqav2.vqa_score("cat", answers), 0.9)
        self.assertEqual(vqav2.normalize_vqa_answer("The TWO, cats."), "2 cats")

        row = {
            "question_id": 1,
            "image_id": 2,
            "question": "What is shown?",
            "image": {"bytes": b"not-empty", "path": None},
            "answers": [{"answer": "cat", "answer_confidence": "yes", "answer_id": 1}],
            "multiple_choice_answer": "cat",
            "question_type": "what",
            "answer_type": "other",
        }
        parsed = vqav2.validate_vqav2_row(row, split="validation", source="unit")
        self.assertEqual(parsed["answers"], ["cat"])

        no_gt = dict(row)
        no_gt["answers"] = None
        no_gt["multiple_choice_answer"] = None
        self.assertFalse(vqav2.validate_vqav2_row(no_gt, split="testdev", source="unit")["has_ground_truth"])
        with self.assertRaisesRegex(ValueError, "no annotator answers"):
            vqav2.validate_vqav2_row(no_gt, split="validation", source="unit")

    def test_sparseengine_raw_config_fallback_is_not_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                '{"model_type": "qwen2", "dtype": "float16", "max_position_embeddings": 32768}\n',
                encoding="utf-8",
            )
            with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "Refusing to silently fall back"):
                    Config(model=str(model_dir))

    def test_sparseengine_rejects_unsupported_model_type(self):
        hf_config = SimpleNamespace(
            model_type="deepseek_v2",
            dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=hf_config):
                with self.assertRaisesRegex(NotImplementedError, "Unsupported SparseEngine model_type"):
                    Config(model=tmp)

    def test_sparseengine_deltakv_requires_checkpoint_path(self):
        hf_config = SimpleNamespace(
            model_type="qwen2",
            dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=hf_config):
                with self.assertRaisesRegex(ValueError, "requires deltakv_checkpoint_path"):
                    Config(model=tmp, sparse_method="deltakv")
                with self.assertRaisesRegex(ValueError, "requires deltakv_checkpoint_path"):
                    Config(model=tmp, sparse_method="deltakv", deltakv_checkpoint_path="none")
                with self.assertRaisesRegex(ValueError, "requires deltakv_checkpoint_path"):
                    Config(model=tmp, sparse_method="deltakv-less-memory")
                cfg = Config(
                    model=tmp,
                    sparse_method="deltakv-less-memory",
                    full_attention_layers=[0],
                    allow_missing_deltakv_path=True,
                    deltakv_latent_quant_bits=0,
                )
                self.assertEqual(cfg.sparse_method, "deltakv")

    def test_sparseengine_missing_model_dir_has_clear_error(self):
        missing = "/tmp/sparseengine-definitely-missing-model-dir"
        with self.assertRaisesRegex(FileNotFoundError, "Model directory does not exist"):
            Config(model=missing)


if __name__ == "__main__":
    unittest.main()
