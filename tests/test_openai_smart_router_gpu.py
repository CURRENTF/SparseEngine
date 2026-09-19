import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.validation import run_openai_router_smoke as router_smoke


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


class OpenAISmartRouterGpuSmokeTest(unittest.TestCase):
    def test_summary_validation_requires_real_route_headers_and_prefix_match(self):
        args = SimpleNamespace(
            require_prefix_route=True,
            require_overload_reroute=True,
            busy_requests=2,
            balance_requests=3,
            random_requests=1,
        )
        summary = {
            "warmup": {
                "headers": {"worker": "http://127.0.0.1:18181", "reason": "target_worker"},
                "choice_count": 1,
            },
            "prefix_match_after_warmup": {
                "status": 200,
                "payload": {"supported": True, "enabled": True, "matched_tokens": 128},
            },
            "balanced_prefix_route": {
                "headers": {"worker": "http://127.0.0.1:18181", "reason": "best_prefix_match"},
                "choice_count": 1,
            },
            "overload_probe_route": {
                "headers": {
                    "worker": "http://127.0.0.1:18182",
                    "reason": "prefix_match_overloaded_lowest_load",
                },
                "choice_count": 1,
            },
            "busy_results": [{}, {}],
            "balance_burst": [{}, {}, {}],
            "requests": [{"choice_count": 1}],
            "delete_subtree": {"status": 200},
        }

        router_smoke.validate_summary(summary, args)

        summary["balanced_prefix_route"]["headers"]["reason"] = "lowest_load_no_prefix_match"
        with self.assertRaisesRegex(RuntimeError, "prefix match"):
            router_smoke.validate_summary(summary, args)

    def test_failed_smoke_summary_preserves_partial_route_records(self):
        args = SimpleNamespace(
            router_port=19180,
            model="/models/local",
            served_model_name="router-smoke-model",
        )
        partial = {
            "warmup": {
                "headers": {"worker": "http://127.0.0.1:18181", "reason": "target_worker"},
                "choice_count": 1,
            },
            "balanced_prefix_route": {
                "headers": {"worker": "http://127.0.0.1:18181", "reason": "lowest_load_no_prefix_match"},
                "choice_count": 1,
            },
        }

        summary = router_smoke.build_failure_summary(
            partial,
            args,
            methods=["omnikv", "snapkv"],
            gpus=["2", "3"],
            ports=[18181, 18182],
            exc=RuntimeError("balanced request should choose prefix match"),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["warmup"], partial["warmup"])
        self.assertEqual(summary["balanced_prefix_route"], partial["balanced_prefix_route"])
        self.assertIn("prefix match", summary["error"])
        self.assertEqual(summary["served_model_name"], "router-smoke-model")

    @unittest.skipUnless(_env_flag("SPARSEENGINE_ROUTER_GPU_TEST"), "set SPARSEENGINE_ROUTER_GPU_TEST=1 to run")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the real router GPU smoke")
    def test_real_two_worker_router_smoke_on_gpus(self):
        model = os.getenv("SPARSEENGINE_ROUTER_GPU_MODEL")
        if not model:
            self.fail("SPARSEENGINE_ROUTER_GPU_MODEL must point to a local model path.")
        if not Path(model).exists():
            self.fail(f"SPARSEENGINE_ROUTER_GPU_MODEL does not exist: {model}")

        gpus = _split_csv("SPARSEENGINE_ROUTER_GPU_GPUS")
        if len(gpus) != 2:
            self.fail("SPARSEENGINE_ROUTER_GPU_GPUS must contain exactly two GPU ids, for example '2,3'.")

        methods = os.getenv("SPARSEENGINE_ROUTER_GPU_METHODS", "omnikv,snapkv")
        ports = os.getenv("SPARSEENGINE_ROUTER_GPU_PORTS", "19181,19182")
        router_port = os.getenv("SPARSEENGINE_ROUTER_GPU_ROUTER_PORT", "19180")
        output_root = Path(os.getenv("SPARSEENGINE_ROUTER_GPU_OUTPUT_ROOT", tempfile.gettempdir()))
        output_dir = output_root / "sparseengine_router_gpu_smoke"
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "scripts/validation/run_openai_router_smoke.py",
            "--model",
            model,
            "--methods",
            methods,
            "--gpus",
            ",".join(gpus),
            "--worker-ports",
            ports,
            "--router-port",
            router_port,
            "--output-dir",
            str(output_dir),
            "--max-model-len",
            os.getenv("SPARSEENGINE_ROUTER_GPU_MAX_MODEL_LEN", "4096"),
            "--prefix-words",
            os.getenv("SPARSEENGINE_ROUTER_GPU_PREFIX_WORDS", "512"),
            "--busy-requests",
            os.getenv("SPARSEENGINE_ROUTER_GPU_BUSY_REQUESTS", "2"),
            "--busy-max-tokens",
            os.getenv("SPARSEENGINE_ROUTER_GPU_BUSY_MAX_TOKENS", "64"),
            "--balance-requests",
            os.getenv("SPARSEENGINE_ROUTER_GPU_BALANCE_REQUESTS", "3"),
            "--random-requests",
            os.getenv("SPARSEENGINE_ROUTER_GPU_RANDOM_REQUESTS", "4"),
            "--require-prefix-route",
            "--require-overload-reroute",
        ]
        run_info = {
            "command": command,
            "cwd": str(Path.cwd()),
            "env": {
                "SPARSEENGINE_ROUTER_GPU_MODEL": model,
                "SPARSEENGINE_ROUTER_GPU_GPUS": ",".join(gpus),
                "SPARSEENGINE_ROUTER_GPU_METHODS": methods,
                "SPARSEENGINE_ROUTER_GPU_PORTS": ports,
                "SPARSEENGINE_ROUTER_GPU_ROUTER_PORT": router_port,
            },
        }
        with (output_dir / "pytest_invocation.json").open("w", encoding="utf-8") as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), str(Path.cwd() / "src"), env.get("PYTHONPATH", "")])
        subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            check=True,
            timeout=float(os.getenv("SPARSEENGINE_ROUTER_GPU_TIMEOUT_S", "900")),
        )

        with (output_dir / "router_smoke_summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["status"], "success")
