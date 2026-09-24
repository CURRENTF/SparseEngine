"""Protect dataset and score completeness at the official recipe boundary."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("aime_recipe", Path(__file__).with_name("run.py"))
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class ArtifactContracts(unittest.TestCase):
    def test_method_boundary_waits_for_transient_gpu_teardown(self):
        ready = {"devices": "idle", "compute_processes": ""}
        with patch.object(recipe, "idle_gpus", side_effect=[RuntimeError("GPU 3 is busy"), ready]) as check:
            with patch.object(recipe.time, "sleep") as sleep:
                self.assertEqual(recipe.wait_idle_gpus("3", 1, timeout=1), ready)
        self.assertEqual(check.call_count, 2)
        sleep.assert_called_once()

    def test_duplicate_questions_cannot_count_as_complete_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            recipe.write(path, [{"Problem": "same", "Answer": 1}] * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                recipe.validate_data(path, 2)

    def artifacts(self, folder, ids):
        rows = [{"id": sample, "status": "parse_failed", "correct": False} for sample in ids]
        for name in ("aime2024.jsonl", "aime2024_parsed_outputs.jsonl", "aime2024_per_sample_results.jsonl"):
            (folder / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        recipe.write(folder / "result.json", {"aime2024": {"total": 2, "correct": 0, "pass@1": 0.0}})

    def test_duplicate_prediction_ids_reject_apparently_complete_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.artifacts(folder, ["0", "0"])
            with self.assertRaisesRegex(ValueError, "coverage"):
                recipe.validate_results(folder, ["0", "1"])

    def test_parse_failures_remain_in_denominator_and_stale_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.artifacts(folder, ["0", "1"])
            self.assertEqual(recipe.validate_results(folder, ["0", "1"])["total"], 2)
            recipe.write(folder / "result.json", {"aime2024": {"total": 2, "correct": 1, "pass@1": 50.0}})
            with self.assertRaisesRegex(ValueError, "counts disagree"):
                recipe.validate_results(folder, ["0", "1"])


@unittest.skipUnless(sys.platform == "linux", "Uses Linux process groups and /proc")
class ProcessCleanupContracts(unittest.TestCase):
    def test_exited_leader_does_not_leave_a_term_resistant_worker(self):
        self.check_worker_cleanup(timeout=False)

    def test_timeout_cleans_workers_after_the_leader_terminates(self):
        self.check_worker_cleanup(timeout=True)

    def check_worker_cleanup(self, *, timeout):
        """The old leader-only wait leaked workers in both failure paths."""
        code = """
import os, signal, sys, time
from pathlib import Path
marker = Path(sys.argv[1])
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    marker.write_text(str(os.getpid()))
    time.sleep(30)
    os._exit(0)
deadline = time.monotonic() + 5
while not marker.exists() and time.monotonic() < deadline:
    time.sleep(.01)
if sys.argv[2] == 'timeout':
    time.sleep(30)
os._exit(3)
"""
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "worker.pid"
            command = [sys.executable, "-c", code, str(marker),
                       "timeout" if timeout else "exit"]
            error_type = subprocess.TimeoutExpired if timeout else subprocess.CalledProcessError
            try:
                with open(os.devnull, "w") as log:
                    with self.assertRaises(error_type):
                        recipe.run_command(command, dict(os.environ), log, 2 if timeout else 5)
                pid = int(marker.read_text())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
                    except FileNotFoundError:
                        return
                    if state == "Z":  # A killed orphan may await reaping by PID 1.
                        return
                    time.sleep(.01)
                self.fail("Owned worker survived process-group cleanup")
            finally:
                if marker.exists():
                    try:
                        os.kill(int(marker.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
