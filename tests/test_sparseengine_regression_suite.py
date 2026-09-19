import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmark.sparseengine_regression import run_suite
from benchmark.sparseengine_regression.manifest import (
    REQUIRED_ARTIFACTS,
    deltakv_checkpoint_path_for,
    load_manifest,
    resolve_manifest_paths,
)


class SparseVLLMRegressionSuiteTest(unittest.TestCase):
    def test_quality_command_explicitly_opts_into_prefix_caching(self):
        command = run_suite._quality_command(
            model_id="qwen3_4b",
            method_id="vanilla",
            model={"model_path": "/model", "tokenizer_path": "/tokenizer"},
            method={
                "sparse_method": "vanilla",
                "requires_compressor": False,
                "config": {"sparse_method": "vanilla"},
            },
            quality={
                "tasks": ["qasper"],
                "worker_world_size": 1,
                "batch_size": 1,
                "min_prompt_tokens": 0,
                "samples_per_task": 1,
                "min_required_samples": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "enable_prefix_caching": True,
                "prefix_cache_block_size": 16,
            },
            output_root=Path("/output"),
        )

        self.assertIn("--allow_prefix_caching", command)
        hyper_params = json.loads(command[command.index("--hyper_param") + 1])
        self.assertIs(hyper_params["enable_prefix_caching"], True)

    def test_deltakv_checkpoint_uses_canonical_manifest_field(self):
        with patch.dict(
            "os.environ",
            {"DELTAKV_COMPRESSOR_QWEN25_7B": "/checkpoints/qwen25-7b"},
            clear=False,
        ):
            resolved = resolve_manifest_paths(load_manifest())

        model = resolved["models"]["qwen25_7b"]
        method = resolved["methods"]["deltakv"]
        self.assertEqual(
            deltakv_checkpoint_path_for(model, method),
            "/checkpoints/qwen25-7b",
        )
        self.assertNotIn("compressor_path", model)
        self.assertNotIn("compressor_path", method)

    def test_validate_layer_runs_as_a_standard_test_and_writes_required_artifacts(self):
        # Keep the default tests on the cheap validate layer only. Full
        # quality/performance/stress regression runs require model paths,
        # checkpoints, datasets, and GPUs, so they must be launched explicitly
        # through benchmark/sparseengine_regression/run_suite.py.
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "run_suite.py",
                "--layer",
                "validate",
                "--models",
                "qwen25_7b",
                "--methods",
                "vanilla",
                "--run_id",
                "unit_validate",
                "--output_root",
                tmp,
            ]

            stdout = io.StringIO()
            with patch("sys.argv", argv), redirect_stdout(stdout):
                self.assertEqual(run_suite.main(), 0)
            self.assertIn("[validate] manifest ok:", stdout.getvalue())

            run_root = Path(tmp) / "sparseengine_regression" / "unit_validate"
            missing_artifacts = [name for name in REQUIRED_ARTIFACTS if not (run_root / name).exists()]
            self.assertEqual(missing_artifacts, [])

            with (run_root / "grade_summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["layer"], "validate")
            self.assertEqual(summary["models"], ["qwen25_7b"])
            self.assertEqual(summary["methods"], ["vanilla"])
            self.assertEqual(summary["commands"], [])
            self.assertEqual(summary["skipped"], [])

            for jsonl_name in ("raw_outputs.jsonl", "parsed_outputs.jsonl", "sample_results.jsonl", "perf.jsonl"):
                self.assertEqual((run_root / jsonl_name).read_text(encoding="utf-8"), "")

    def test_run_command_timeout_records_and_terminates(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "timeout.log"
            cmd = [sys.executable, "-c", "import time; time.sleep(10)"]

            with self.assertRaises(run_suite.CommandExecutionError) as raised:
                run_suite._run_command(
                    cmd,
                    cwd=Path.cwd(),
                    dry_run=False,
                    log_path=log_path,
                    timeout_s=0.1,
                )

            self.assertEqual(raised.exception.record["status"], "timeout")
            self.assertEqual(raised.exception.record["timeout_s"], 0.1)
            self.assertIn("command exceeded timeout_s=0.1", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
