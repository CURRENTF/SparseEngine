"""CPU contracts: preserve failures, distinguish cost skips, reject lost samples."""
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import os
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recipe = load("run")
timed = load("timed_mini")


def fake_instance(instance, output_dir, fail=False):
    if fail:
        raise ValueError("original failure")
    return {"unchanged": instance}


class RecipeContracts(unittest.TestCase):
    def test_stop_server_terminates_only_recorded_process_group(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "snapkv-chain/smoke"
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
            reaper = threading.Thread(target=process.wait)
            reaper.start()
            recipe.write(directory / "server_process.json", {
                "pid": process.pid,
                "process_group": os.getpgid(process.pid),
                "start_ticks": recipe.process_start_ticks(process.pid),
                "hostname": socket.gethostname(),
                "boot_id": recipe.boot_id(),
                "command": "test sleep",
            })
            try:
                with patch("builtins.print"):
                    recipe.stop_server(SimpleNamespace(
                        root=root, method="snapkv-chain", phase="smoke", reason="coordinator_failure"
                    ))
            finally:
                if process.poll() is None:
                    process.kill()
                reaper.join(timeout=5)
            self.assertEqual(process.returncode, -15)
            self.assertEqual(recipe.read(directory / "server_stop.json")["reason"], "coordinator_failure")
            self.assertEqual(
                recipe.read(directory / "server_stop.json")["status"],
                "stop_requested",
            )

    def test_server_manifest_serializes_command_for_driver_validation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            recipe.write(root / "setting.json", {"backend": "sparseengine"})
            recipe.write(root / "snapkv-chain/engine.json", {
                "sparse_method": "snapkv", "max_model_len": 1024})
            recipe.write(root / "model/config.json", {"dtype": "bfloat16"})
            args = SimpleNamespace(
                root=root, method="snapkv-chain", phase="smoke",
                python=sys.executable, model=root / "model", gpus="4,5", port=0,
                timeout=1, compile_cache_root=root / "node-local")
            with patch.object(recipe, "environment", return_value=(sys.executable, {})), \
                    patch.object(recipe, "idle_pair", return_value={"gpus": "", "processes": ""}), \
                    patch.object(recipe, "capture", return_value="test-value"), \
                    patch.object(recipe, "run_logged") as launch:
                recipe.serve(args)
            manifest = recipe.read(root / "snapkv-chain/smoke/server_manifest.json")
            self.assertIsInstance(manifest["command"], str)
            self.assertIn("sparseengine.entrypoints.openai.api_server", manifest["command"])
            for key, value in manifest["compiler_environment"].items():
                self.assertEqual(launch.call_args.args[1][key], value)
                self.assertTrue(Path(value).is_dir())
                self.assertTrue(Path(value).is_relative_to(args.compile_cache_root))

    def test_timing_preserves_return_and_exception(self):
        with tempfile.TemporaryDirectory() as root:
            wrapped = timed.instrument(fake_instance)
            with patch.object(timed.time, "perf_counter", side_effect=[10., 13.]):
                result = wrapped({"instance_id": "a"}, root)
            self.assertEqual(result, {"unchanged": {"instance_id": "a"}})
            self.assertEqual(recipe.read(Path(root) / "a/a.wall_time.json")["elapsed_s"], 3)
            with self.assertRaisesRegex(ValueError, "original failure"):
                wrapped({"instance_id": "b"}, root, fail=True)
            row = recipe.read(Path(root) / "b/b.wall_time.json")
            self.assertEqual(row["uncaught_exception"], "ValueError")
            with self.assertRaises(FileExistsError):
                wrapped({"instance_id": "a"}, root)

    def test_timeout_record_is_distinct_from_crash(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with self.assertRaises(RuntimeError):
                recipe.run_logged([sys.executable, "-c", "import time; time.sleep(10)"],
                                  recipe.environment(sys.executable)[1], root, "generate", .05)
            self.assertEqual(recipe.read(root / "generate.result.json")["status"], "timeout")

    def test_slow_gate_does_not_classify_crash_as_slow(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            recipe.write(root / "setting.json", {"slow_baseline_policy": {
                "pilot_instances": 64, "full_generation_limit_seconds": 43200, "slowdown_ratio": 4}})
            recipe.write(root / "snapkv-chain/pilot/generate.result.json",
                         {"status": "success", "elapsed_seconds": 100})
            path = root / "snapkv-no-chain/pilot/generate.result.json"
            recipe.write(path, {"status": "failed", "elapsed_seconds": 900})
            with self.assertRaisesRegex(ValueError, "implementation failure"):
                recipe.gate(root)
            path.write_text(json.dumps({"status": "timeout", "elapsed_seconds": 900}))
            self.assertEqual(recipe.gate(root)["status"], "skipped_by_policy")

    def test_collector_rejects_missing_or_duplicate_samples(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "snapkv-chain/smoke/benchmark"
            recipe.write(root / "setting.json", {"slow_baseline_policy": {"pilot_instances": 64}})
            recipe.write(directory / "final_summary.json", {"total_instances": 1})
            (directory / "instances.txt").write_text("expected\n")
            (directory / "generation_results.jsonl").write_text('{"instance_id":"wrong"}\n')
            with self.assertRaisesRegex(ValueError, "unexpected samples"):
                recipe.collect(SimpleNamespace(root=root, method="snapkv-chain", phase="smoke"))

    def test_collection_preserves_e2e_boundaries_and_cache_accounting(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "snapkv-chain/smoke"
            recipe.write(root / "setting.json", {"slow_baseline_policy": {"pilot_instances": 64}})
            recipe.write(directory / "benchmark/final_summary.json", {"total_instances": 1, "resolved_instances": 1})
            (directory / "benchmark/instances.txt").write_text("a\n")
            for name in ("generation_results", "per_sample_results"):
                (directory / f"benchmark/{name}.jsonl").write_text(json.dumps(
                    {"instance_id": "a", "status": "success", "resolved": True}) + "\n")
            recipe.write(directory / "benchmark/batches/batch_000/a/a.wall_time.json",
                         {"instance_id": "a", "elapsed_s": 12})
            for stage, duration in (("generate", 20), ("evaluate", 40)):
                recipe.write(directory / f"{stage}.result.json", {"status": "success", "elapsed_seconds": duration})
            recipe.write(directory / "server_requests/a.json", {
                "status": "success", "request_id": "a", "elapsed_s": 3,
                "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                       "prompt_tokens_details": {"cached_tokens": 60}}}})
            with patch("builtins.print"):
                recipe.collect(SimpleNamespace(root=root, method="snapkv-chain", phase="smoke"))
            report = recipe.read(directory / "report.json")
            self.assertEqual(report["requests"]["uncached_prompt_tokens"], 40)
            self.assertEqual(report["task_completion"]["resolved"]["p50_s"], 12)
            self.assertEqual(report["generation_stage"]["elapsed_seconds"], 20)
            self.assertEqual(report["official_evaluation_stage"]["elapsed_seconds"], 40)
            row = json.loads((directory / "request_samples.jsonl").read_text())
            self.assertIsNone(row["ttft_ms"])


class RemoteLifecycleContracts(unittest.TestCase):
    def args(self, root, **overrides):
        values = dict(root=Path(root), method="snapkv-chain", phase="full", stage="generate",
                      ssh_command="ssh worker", worker_command="bash '/worker path/job.sh' generate --detach",
                      timeout=100, recovery_timeout=10, reconnect_attempts=30, poll_interval=.01,
                      python=sys.executable, swe_bench_dir=Path(root), api_base="http://unused/v1",
                      server_manifest=None)
        return SimpleNamespace(**(values | overrides))

    def test_control_disconnect_does_not_end_wait_or_duplicate_worker_command(self):
        # The server-owning shell must not reach its EXIT trap on a transient 255.
        states = [subprocess.CompletedProcess([], 255, "", "connection lost"),
                  subprocess.CompletedProcess([], 0, json.dumps({"status": "running", "api_ready": False}), ""),
                  subprocess.CompletedProcess([], 0, json.dumps({"status": "running", "api_ready": True}), ""),
                  subprocess.CompletedProcess([], 0, json.dumps({"status": "success", "exit_code": 0}), "")]
        with tempfile.TemporaryDirectory() as root, patch.object(recipe.subprocess, "run", side_effect=states) as run:
            recipe.wait_remote(self.args(root))
            self.assertEqual(run.call_count, 4)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertTrue(all(c == commands[0] for c in commands))
            self.assertEqual(recipe.read(Path(root) / "snapkv-chain/full/generate.remote.result.json")["status"], "success")

    def test_failed_http_probes_cannot_abort_a_running_worker(self):
        # Regression: HTTP readiness failed for longer than the reconnect budget
        # while real generation still completed requests. Keep waiting for its result.
        states = [subprocess.CompletedProcess([], 0, json.dumps(
            {"status": "running", "api_ready": False}), "")] * 40
        states.append(subprocess.CompletedProcess([], 0, json.dumps({"status": "success"}), ""))
        with tempfile.TemporaryDirectory() as root, patch.object(recipe.subprocess, "run", side_effect=states) as run:
            recipe.wait_remote(self.args(root, recovery_timeout=.02))
            self.assertEqual(run.call_count, len(states))
            result = recipe.read(Path(root) / "snapkv-chain/full/generate.remote.result.json")
            self.assertEqual(result["status"], "success")

    def test_unstarted_worker_retains_bounded_readiness_wait(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recipe.subprocess, "run", return_value=
                subprocess.CompletedProcess([], 0, json.dumps({"status": "waiting_for_server", "api_ready": False}), "")):
            with self.assertRaisesRegex(TimeoutError, "recovery deadline"):
                recipe.wait_remote(self.args(root, recovery_timeout=.02))
            result = recipe.read(Path(root) / "snapkv-chain/full/generate.remote.result.json")
            self.assertEqual(result["status"], "recovery_timeout")

    def test_running_worker_still_obeys_stage_wall_time_limit(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recipe.subprocess, "run", return_value=
                subprocess.CompletedProcess([], 0, json.dumps({"status": "running"}), "")):
            with self.assertRaisesRegex(TimeoutError, "stage wall-time"):
                recipe.wait_remote(self.args(root, timeout=.03))
            result = recipe.read(Path(root) / "snapkv-chain/full/generate.remote.result.json")
            self.assertEqual(result["status"], "timeout")

    def test_remote_stage_failure_is_not_retried_as_a_transport_failure(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recipe.subprocess, "run", return_value=
                subprocess.CompletedProcess([], 0, json.dumps({"status": "failed", "error": "bad dataset"}), "")) as run:
            with self.assertRaisesRegex(RuntimeError, "bad dataset"):
                recipe.wait_remote(self.args(root))
            self.assertEqual(run.call_count, 1)

    def test_each_recovered_outage_gets_a_fresh_retry_budget(self):
        disconnected = subprocess.CompletedProcess([], 255, "", "lost")
        running = subprocess.CompletedProcess([], 0, json.dumps({"status": "running", "api_ready": False}), "")
        finished = subprocess.CompletedProcess([], 0, json.dumps({"status": "success"}), "")
        states = [disconnected, disconnected, running, disconnected, disconnected, finished]
        with tempfile.TemporaryDirectory() as root, patch.object(recipe.subprocess, "run", side_effect=states) as run:
            recipe.wait_remote(self.args(root, reconnect_attempts=2))
            self.assertEqual(run.call_count, 6)

    def test_persistent_disconnect_exhausts_exact_retry_budget(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recipe.subprocess, "run", return_value=
                subprocess.CompletedProcess([], 255, "", "lost")) as run:
            with self.assertRaisesRegex(ConnectionError, "attempts exhausted"):
                recipe.wait_remote(self.args(root, reconnect_attempts=3))
            self.assertEqual(run.call_count, 4)  # initial loss plus three retries
            result = recipe.read(Path(root) / "snapkv-chain/full/generate.remote.result.json")
            self.assertEqual(result["status"], "reconnect_exhausted")

    def test_lost_launch_reply_attaches_to_existing_tmux_stage(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recipe, "get", return_value={"data": [{"id": "glm47-snapkv-chain"}]}), \
                patch.object(recipe.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            args = self.args(root)
            self.assertEqual(recipe.detached_benchmark(args)["status"], "running")
            self.assertEqual(recipe.detached_benchmark(args)["status"], "running")
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(sum("new-session" in c for c in commands), 1)
            self.assertEqual(sum("has-session" in c for c in commands), 1)

    def test_existing_worker_poll_never_depends_on_http_readiness(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recipe, "get", return_value={"data": [{"id": "glm47-snapkv-chain"}]}) as get, \
                patch.object(recipe.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            args = self.args(root)
            recipe.detached_benchmark(args)
            get.side_effect = OSError("HTTP unavailable while worker is making progress")
            self.assertEqual(recipe.detached_benchmark(args)["status"], "running")
            self.assertEqual(get.call_count, 1)  # Only the initial admission probes HTTP.

    def test_startup_probe_failure_retains_error_without_launching_worker(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recipe, "get", side_effect=TimeoutError("probe deadline")), \
                patch.object(recipe.subprocess, "run") as run:
            state = recipe.detached_benchmark(self.args(root))
            self.assertEqual(state["status"], "waiting_for_server")
            self.assertIn("TimeoutError: probe deadline", state["error"])
            run.assert_not_called()
            self.assertFalse((Path(root) / "snapkv-chain/full/generate.detached.json").exists())

    def test_vanished_worker_without_result_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recipe, "get", return_value={"data": [{"id": "glm47-snapkv-chain"}]}), \
                patch.object(recipe.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            args = self.args(root)
            recipe.detached_benchmark(args)
            with patch.object(recipe.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)) as run:
                with self.assertRaisesRegex(RuntimeError, "vanished"):
                    recipe.detached_benchmark(args)
                self.assertEqual(run.call_count, 1)

    def test_preflight_exception_is_published_as_terminal_failure(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recipe, "benchmark", side_effect=ValueError("manifest mismatch")):
            args = self.args(root)
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                recipe.detached_child(args)
            result = recipe.detached_benchmark(args)
            self.assertEqual(result["status"], "failed")
            self.assertIn("manifest mismatch", result["error"])

    def test_policy_skip_remains_distinct_through_detached_and_remote_status(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            recipe.write(root / "setting.json", {"slow_baseline_policy": {
                "pilot_instances": 64, "full_generation_limit_seconds": 10,
                "slowdown_ratio": 2}})
            recipe.write(root / "snapkv-chain/pilot/generate.result.json",
                         {"status": "success", "elapsed_seconds": 1})
            recipe.write(root / "snapkv-no-chain/pilot/generate.result.json",
                         {"status": "timeout", "elapsed_seconds": 20})
            args = self.args(root, method="snapkv-no-chain")

            with patch("builtins.print"):
                recipe.detached_child(args)
            detached = recipe.read(
                root / "snapkv-no-chain/full/generate.detached.result.json")
            self.assertEqual(detached["status"], "skipped_by_policy")
            self.assertEqual(detached["exit_code"], 0)

            response = subprocess.CompletedProcess([], 0, json.dumps(detached), "")
            with patch.object(recipe.subprocess, "run", return_value=response) as run:
                recipe.wait_remote(args)
            self.assertEqual(run.call_count, 1)
            remote = recipe.read(
                root / "snapkv-no-chain/full/generate.remote.result.json")
            self.assertEqual(remote["status"], "skipped_by_policy")
            self.assertEqual(remote["exit_code"], 0)

    @unittest.skipUnless(shutil.which("tmux"), "tmux required for detached lifecycle integration")
    def test_real_tmux_worker_survives_control_process_exit_and_starts_once(self):
        # A bounded fake workload exercises real process/session ownership and
        # shell quoting. HTTP health fails after launch while the worker progresses.
        probes = []

        class Readiness(BaseHTTPRequestHandler):
            def do_GET(self):
                probes.append(self.path)
                self.send_response(200 if len(probes) == 1 else 503)
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"glm47-snapkv-chain"}]}')

            def log_message(self, *args):
                pass

        http = ThreadingHTTPServer(("127.0.0.1", 0), Readiness)
        http_thread = threading.Thread(target=http.serve_forever, daemon=True)
        http_thread.start()
        self.addCleanup(http.server_close)
        self.addCleanup(http_thread.join, 5)
        self.addCleanup(http.shutdown)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            worker = root / "worker python"
            worker.write_text(f"#!{sys.executable}\n" + '''import json, pathlib, time
root = pathlib.Path(__file__).parent
with (root / "launches").open("a") as f:
    f.write("started\\n")
time.sleep(1)
result = root / "snapkv-chain/full/generate.detached.result.json"
temp = result.with_suffix(".tmp")
temp.write_text(json.dumps({"status": "success", "exit_code": 0}))
temp.replace(result)
''')
            worker.chmod(0o755)
            command = [sys.executable, str(Path(recipe.__file__).resolve()), "bench",
                       "--root", str(root), "--method", "snapkv-chain", "--phase", "full",
                       "--stage", "generate", "--python", str(worker),
                       "--api-base", f"http://127.0.0.1:{http.server_port}/v1",
                       "--swe-bench-dir", str(root), "--detach"]
            session = None
            try:
                first = json.loads(subprocess.check_output(command, text=True, timeout=10))
                session = first["session"]
                self.assertEqual(first["status"], "running")
                state = json.loads(subprocess.check_output(command, text=True, timeout=10))
                self.assertIn(state["status"], {"running", "success"})
                deadline = time.monotonic() + 10
                result = root / "snapkv-chain/full/generate.detached.result.json"
                while not result.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertEqual(recipe.read(result)["status"], "success")
                self.assertEqual((root / "launches").read_text().splitlines(), ["started"])
                self.assertEqual(probes, ["/v1/models"])
            finally:
                if session:
                    subprocess.run(["tmux", "kill-session", "-t", "=" + session], capture_output=True)


if __name__ == "__main__":
    unittest.main()
