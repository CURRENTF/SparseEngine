import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from benchmark.swe_bench_lite.docker_memory_guard import (
    DockerMemoryGuard, DockerMemoryLimitExceeded, constrain_sdk_kwargs,
)
from benchmark.swe_bench_lite.docker_writable_guard import (
    DockerWritableLayerLimitExceeded, GuardState,
)


class DockerMemoryGuardTests(unittest.TestCase):
    def test_constructor_validates_parent_ancestry_instead_of_a_single_component(self):
        """Nested parent paths must work without accepting lookalike siblings."""
        cases = [
            ("/batch/run", "/batch/run/container", True),
            ("/batch/./run/", "/batch/run/container", True),
            ("batch/run", "/delegated/batch/run/container", True),
            ("work-job.slice", "/work.slice/work-job.slice/container", True),
            ("/batch/run", "/batch/runner/container", False),
            ("/batch/run", "/elsewhere/batch/run/container", False),
            ("batch/run", "/batch/other/run/container", False),
        ]
        for parent, cgroup, accepted in cases:
            with self.subTest(parent=parent, cgroup=cgroup):
                record = {
                    "HostConfig": {"Memory": 64, "MemorySwap": 64,
                                   "CgroupParent": parent, "AutoRemove": False},
                    "State": {"Pid": 123},
                }
                with patch.dict(os.environ, {
                    "SPARSEENGINE_DOCKER_MEMORY_LIMIT_BYTES": "64",
                    "SPARSEENGINE_DOCKER_MEMORY_PARENT": parent,
                    "SPARSEENGINE_DOCKER_MEMORY_EVENTS": "oom.jsonl",
                }), patch.object(DockerMemoryGuard, "inspect", return_value=record), \
                        patch.object(Path, "read_text", side_effect=[
                            f"0::{cgroup}\n", "oom 0\noom_kill 0\n",
                        ]):
                    if accepted:
                        guard = DockerMemoryGuard("docker", "container", "image")
                        self.assertEqual(guard.events, Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.events")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "outside the bounded"):
                            DockerMemoryGuard("docker", "container", "image")

    def guard(self, directory):
        guard = DockerMemoryGuard.__new__(DockerMemoryGuard)
        guard.container_id = 'owned-container'
        guard.image = 'test-image'
        guard.failed = False
        guard.events = Path(directory) / 'memory.events'
        guard.log_path = Path(directory) / 'oom.jsonl'
        return guard

    def test_child_process_oom_fails_task_even_when_container_init_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = self.guard(directory)
            guard.events.write_text('oom 0\noom_kill 0\n')
            guard.check()
            self.assertFalse(guard.log_path.exists())
            guard.events.write_text('oom 1\noom_kill 1\n')
            for _ in range(2):
                with self.assertRaises(DockerMemoryLimitExceeded):
                    guard.check()
            rows = [json.loads(s) for s in guard.log_path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['container_id'], guard.container_id)

    def test_ancestor_limit_victim_is_detected_without_local_oom_event(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = self.guard(directory)
            guard.events.write_text('oom 0\noom_kill 1\n')
            with self.assertRaises(DockerMemoryLimitExceeded):
                guard.check()

    def test_removed_cgroup_uses_persisted_docker_oom_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = self.guard(directory)
            with patch.object(guard, 'inspect', return_value={'State': {'OOMKilled': True}}):
                with self.assertRaises(DockerMemoryLimitExceeded):
                    guard.check()

    def test_unexplained_container_loss_is_not_falsely_reported_as_oom(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = self.guard(directory)
            with patch.object(guard, 'inspect', return_value={'State': {'OOMKilled': False}}):
                with self.assertRaisesRegex(RuntimeError, 'accounting disappeared'):
                    guard.check()
            self.assertFalse(guard.log_path.exists())

    def test_disk_failure_survives_missing_cgroup_even_during_memory_check(self):
        """Disk-triggered termination must retain its cause across monitor races."""
        for fail_during_check in (False, True):
            with self.subTest(fail_during_check=fail_during_check), tempfile.TemporaryDirectory() as directory:
                env = self.environment()
                env._memory_guard = self.guard(directory)
                state = env._writable_guard_state
                failure = DockerWritableLayerLimitExceeded("disk limit exceeded")
                if not fail_during_check:
                    state.fail(failure)

                def inspect_stopped_container():
                    state.fail(failure)
                    return {"State": {"OOMKilled": False}}

                with patch.object(env._memory_guard, "inspect", side_effect=inspect_stopped_container):
                    with self.assertRaises(DockerWritableLayerLimitExceeded) as caught:
                        env._raise_if_guard_failed()
                self.assertIs(caught.exception, failure)

    def test_healthy_disk_guard_preserves_memory_failure(self):
        """Combining guards must still propagate OOM and unexplained accounting loss."""
        for oom in (False, True):
            with self.subTest(oom=oom), tempfile.TemporaryDirectory() as directory:
                env = self.environment()
                env._memory_guard = self.guard(directory)
                expected = DockerMemoryLimitExceeded if oom else RuntimeError
                with patch.object(env._memory_guard, "inspect", return_value={"State": {"OOMKilled": oom}}):
                    with self.assertRaises(expected) as caught:
                        env._raise_if_guard_failed()
                self.assertIs(type(caught.exception), expected)

    def environment(self):
        # The upstream base is unused: exercise our guard composition without Docker.
        module = types.ModuleType("minisweagent.environments.docker")
        module.DockerEnvironment = object
        path = Path(__file__).resolve().parents[1] / "benchmark/swe_bench_lite/guarded_docker_environment.py"
        with patch.dict(sys.modules, {module.__name__: module}):
            cls = runpy.run_path(str(path))["GuardedDockerEnvironment"]
        env = cls.__new__(cls)
        env._writable_guard_state = GuardState("owned-container", "test-image")
        return env

    @patch.dict(os.environ, {'SPARSEENGINE_DOCKER_MEMORY_LIMIT_BYTES': '67108864',
                            'SPARSEENGINE_DOCKER_MEMORY_PARENT': 'test.slice'})
    def test_scorer_cannot_override_limits_and_input_is_not_mutated(self):
        original = {'name': 'task-container'}
        result = constrain_sdk_kwargs(original)
        self.assertEqual(original, {'name': 'task-container'})
        self.assertEqual(result['memswap_limit'], result['mem_limit'])
        for key in ('mem_limit', 'memswap_limit', 'cgroup_parent'):
            with self.assertRaises(ValueError):
                constrain_sdk_kwargs({key: 'conflicting-value'})


if __name__ == '__main__':
    unittest.main()
