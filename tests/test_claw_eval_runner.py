import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from string import Template

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "benchmark" / "claw_eval" / "run_sparseengine_claw_eval.sh"
CONFIG = REPO_ROOT / "benchmark" / "claw_eval" / "sparseengine_config.yaml"


class ClawEvalRunnerTest(unittest.TestCase):
    def test_runner_has_valid_bash_syntax(self):
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    def test_sandbox_image_override_is_resolved_from_final_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "python-env"
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "python").symlink_to(sys.executable)
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_ROOT": str(root / "outputs"),
                    "RUN_NAME": "unit-test",
                    "CLAW_EVAL_CONDA_ENV": str(env_dir),
                    "CLAW_EVAL_ARGS": (
                        "batch --config config.yaml --sandbox "
                        "--sandbox-image custom-agent:test --no-judge"
                    ),
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; resolve_effective_sandbox_image',
                    "bash",
                    str(RUNNER),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.stdout.strip(), "custom-agent:test")

    def test_config_routes_model_and_sandbox_without_secrets_in_runner(self):
        rendered = Template(CONFIG.read_text(encoding="utf-8")).substitute(
            {
                "SPARSEENGINE_OPENAI_API_KEY": "local-test-key",
                "SPARSEENGINE_OPENAI_BASE_URL": "http://127.0.0.1:18000/v1",
                "SPARSEENGINE_CLAW_MODEL_ID": "test-model",
                "SPARSEENGINE_CONTEXT_WINDOW": "32768",
                "OPENROUTER_API_KEY": "judge-test-key",
                "CLAW_EVAL_JUDGE_BASE_URL": "https://openrouter.ai/api/v1",
                "CLAW_EVAL_JUDGE_MODEL": "judge-model",
                "CLAW_EVAL_TRACE_DIR": "/tmp/traces",
                "CLAW_EVAL_SANDBOX_IMAGE": "claw-eval-agent:test",
            }
        )
        config = yaml.safe_load(rendered)

        self.assertEqual(config["model"]["base_url"], "http://127.0.0.1:18000/v1")
        self.assertEqual(config["model"]["model_id"], "test-model")
        self.assertFalse(config["sandbox"]["enabled"])
        self.assertEqual(config["sandbox"]["image"], "claw-eval-agent:test")
        self.assertNotIn("sk-", RUNNER.read_text(encoding="utf-8").lower())

    def test_manifest_records_external_server_and_docker_identity(self):
        script = RUNNER.read_text(encoding="utf-8")

        for field in (
            '"claw_eval_commit"',
            '"start_sparseengine_server"',
            '"server_health_url"',
            '"sandbox_image_id"',
            '"sandbox_image_size_bytes"',
        ):
            self.assertIn(field, script)
        self.assertIn('START_SPARSEENGINE_SERVER="${START_SPARSEENGINE_SERVER:-1}"', script)
        self.assertIn("Starting sandbox preflight container", script)
        self.assertIn("Set CLAW_EVAL_BUILD_SANDBOX_IMAGE=1", script)
        self.assertIn("require_clean_claw_eval_checkout", script)
        self.assertIn("claw_eval_result_validation.log", script)
        self.assertIn("validate_results.py", script)

    def test_dirty_claw_eval_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkout = root / "claw-eval"
            checkout.mkdir()
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            (checkout / "local_change.py").write_text("changed = True\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_ROOT": str(root / "outputs"),
                    "RUN_NAME": "dirty-checkout-test",
                    "CLAW_EVAL_DIR": str(checkout),
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; require_clean_claw_eval_checkout',
                    "bash",
                    str(RUNNER),
                ],
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.returncode, 3)
        self.assertIn("checkout must be clean", result.stderr)

    def test_server_decoding_capacity_matches_parallelism_per_dp_replica(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "python-env"
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "python").symlink_to(sys.executable)
            engine_path = root / "engine.json"
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_ROOT": str(root / "outputs"),
                    "RUN_NAME": "capacity-test",
                    "CLAW_EVAL_CONDA_ENV": str(env_dir),
                    "ENGINE_KWARGS_FILE": str(engine_path),
                    "ENGINE_KWARGS": json.dumps(
                        {"max_num_seqs_in_batch": 4, "max_decoding_seqs": 99}
                    ),
                    "CLAW_EVAL_ARGS": "batch --parallel 24",
                    "SPARSEENGINE_DATA_PARALLEL_SIZE": "2",
                }
            )

            subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; write_engine_kwargs_file',
                    "bash",
                    str(RUNNER),
                ],
                check=True,
                env=env,
            )

            engine = json.loads(engine_path.read_text(encoding="utf-8"))
            self.assertEqual(engine["max_decoding_seqs"], 12)
            self.assertEqual(engine["max_num_seqs_in_batch"], 4)

    def test_server_capacity_rejects_non_divisible_parallelism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "python-env"
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "python").symlink_to(sys.executable)
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_ROOT": str(root / "outputs"),
                    "RUN_NAME": "capacity-test",
                    "CLAW_EVAL_CONDA_ENV": str(env_dir),
                    "ENGINE_KWARGS": "{}",
                    "CLAW_EVAL_ARGS": "batch --parallel 7",
                    "SPARSEENGINE_DATA_PARALLEL_SIZE": "2",
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; write_engine_kwargs_file',
                    "bash",
                    str(RUNNER),
                ],
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be divisible", result.stderr)

    def test_resume_trace_is_added_for_the_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "python-env"
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "python").symlink_to(sys.executable)
            resume_trace = root / "outputs" / "resume-test" / "traces" / "model-run"
            resume_trace.mkdir(parents=True)
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_ROOT": str(root / "outputs"),
                    "RUN_NAME": "resume-test",
                    "CLAW_EVAL_CONDA_ENV": str(env_dir),
                    "CLAW_EVAL_ARGS": "batch --parallel 8",
                    "CLAW_EVAL_RESUME_TRACE_DIR": str(resume_trace),
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; prepare_resume_args; printf "%s" "$CLAW_EVAL_ARGS"',
                    "bash",
                    str(RUNNER),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertIn(f"--continue {resume_trace}", result.stdout)


if __name__ == "__main__":
    unittest.main()
