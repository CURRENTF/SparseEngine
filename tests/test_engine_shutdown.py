import gc
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import weakref

from sparseengine.engine.llm_engine import LLMEngine


def test_explicit_exit_unregisters_hook_and_releases_runtime_references():
    cleanup_calls = []

    class Resource:
        def __init__(self):
            self.cycle = self

    class Runner:
        def __init__(self, resource):
            self.platform = SimpleNamespace(
                empty_cache=lambda: cleanup_calls.append("empty_cache"),
            )
            self.model = resource
            self.cache_manager = resource

        def call(self, method_name):
            cleanup_calls.append(method_name)

    resource = Resource()
    resource_ref = weakref.ref(resource)
    runner = Runner(resource)
    runner_ref = weakref.ref(runner)
    engine = object.__new__(LLMEngine)
    engine._exited = False
    engine.model_runner = runner
    engine.scheduler = SimpleNamespace(memory_oracle=resource)
    engine.ps = []
    engine._atexit_callback = engine.exit
    callback = engine._atexit_callback
    del resource, runner

    with (
        patch("sparseengine.engine.llm_engine.atexit.unregister") as unregister,
        patch("sparseengine.engine.llm_engine.gc.collect", wraps=gc.collect) as collect,
    ):
        engine.exit()

    unregister.assert_called_once_with(callback)
    collect.assert_called_once_with()
    assert cleanup_calls == ["exit", "empty_cache"]
    assert not hasattr(engine, "scheduler")
    assert not hasattr(engine, "model_runner")
    assert runner_ref() is None
    assert resource_ref() is None


def test_engine_exit_timeout_still_terminates_workers():
    class SharedMemory:
        def __init__(self):
            self.closed = False
            self.unlinked = False

        def close(self):
            self.closed = True

        def unlink(self):
            self.unlinked = True

    class BlockingRunner:
        def __init__(self):
            self.call_started = threading.Event()
            self.shm = SharedMemory()

        def call(self, method_name):
            assert method_name == "exit"
            self.call_started.set()
            threading.Event().wait()

    class Worker:
        pid = 12345

        def __init__(self):
            self.alive = True
            self.terminated = False
            self.killed = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.killed = True
            self.alive = False

        def join(self, timeout=None):
            del timeout

    runner = BlockingRunner()
    worker = Worker()
    engine = object.__new__(LLMEngine)
    engine._exited = False
    engine.model_runner = runner
    engine.ps = [worker]

    with patch.dict(
        os.environ,
        {
            "SPARSEENGINE_ENGINE_EXIT_TIMEOUT_S": "0.05",
            "SPARSEENGINE_WORKER_JOIN_TIMEOUT_S": "0.05",
        },
    ):
        started = time.perf_counter()
        engine.exit()
        elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert runner.call_started.is_set()
    assert runner.shm.closed
    assert runner.shm.unlinked
    assert worker.terminated
    assert not worker.killed


def test_engine_exit_joins_graceful_worker_before_terminate():
    class Runner:
        def call(self, method_name):
            assert method_name == "exit"

    class Worker:
        pid = 12346

        def __init__(self):
            self.alive = True
            self.terminated = False
            self.closed = False

        def join(self, timeout=None):
            assert timeout == 0.05
            self.alive = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True

        def close(self):
            self.closed = True

    worker = Worker()
    engine = object.__new__(LLMEngine)
    engine._exited = False
    engine.model_runner = Runner()
    engine.ps = [worker]
    engine.events = [object()]

    with patch.dict(
        os.environ,
        {
            "SPARSEENGINE_ENGINE_EXIT_TIMEOUT_S": "0.05",
            "SPARSEENGINE_WORKER_JOIN_TIMEOUT_S": "0.05",
        },
    ):
        engine.exit()

    assert not worker.terminated
    assert worker.closed
    assert engine.events == []
