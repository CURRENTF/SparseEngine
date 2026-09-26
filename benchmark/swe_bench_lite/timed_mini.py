#!/usr/bin/env python3
"""Delegate mini-extra unchanged, measuring each process_instance invocation."""
import functools
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time


def instrument(original):
    signature = inspect.signature(original)
    required = {"instance", "output_dir"}
    if not required <= set(signature.parameters):
        raise RuntimeError(f"Unsupported MiniSWE process_instance signature: {signature}")
    source = Path(inspect.getsourcefile(original))
    identity = {"path": str(source)}

    @functools.wraps(original)
    def measured(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        iid = bound.arguments["instance"]["instance_id"]
        directory = Path(bound.arguments["output_dir"]) / iid
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{iid}.wall_time.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite task timing: {path}")
        started = time.perf_counter()
        wall_started = time.time()
        error = None
        trace = None
        token = None
        if os.getenv("MINISWE_RECORD_AGENT_TRACE") == "1":
            from benchmark.swe_bench_lite.agent_trace import AgentTrace, CURRENT_TRACE
            trace = AgentTrace(directory / "agent_trace.jsonl", iid)
            token = CURRENT_TRACE.set(trace)
        try:
            return original(*args, **kwargs)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            if trace is not None:
                CURRENT_TRACE.reset(token)
                trace.finish(error)
            duration = time.perf_counter() - started
            # Upstream may handle errors internally; final status comes from its
            # trajectory and official per-sample results, never from return alone.
            with path.open("x") as stream:
                json.dump({"instance_id": iid, "started_unix_s": wall_started,
                           "elapsed_s": duration, "uncaught_exception": error,
                           "scope": "worker_start_to_instance_return_includes_environment_and_tools_excludes_executor_queue_and_official_evaluation",
                           "upstream_source": identity}, stream)
    return measured


def main():
    if os.getenv("MINISWE_RECORD_AGENT_TRACE") == "1":
        from benchmark.swe_bench_lite.agent_trace import install_http_recorder
        install_http_recorder()
    # mini-SWE moved this module between releases. Select the installed location,
    # fail on ambiguity or unsupported signatures; do not replace its algorithm.
    modules = []
    for name in ("minisweagent.run.extra.swebench", "minisweagent.run.benchmarks.swebench"):
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError as exc:
            if not name.startswith(str(exc.name)):
                raise
            spec = None
        if spec is not None:
            modules.append(importlib.import_module(name))
    if len(modules) != 1:
        raise RuntimeError(f"Expected exactly one installed MiniSWE SWE-bench module, found {modules}")
    module = modules[0]
    module.process_instance = instrument(module.process_instance)
    entrypoints = list(importlib.metadata.entry_points(group="console_scripts", name="mini-extra"))
    if len(entrypoints) != 1:
        raise RuntimeError("Expected one installed mini-extra console entrypoint")
    entrypoints[0].load()()


if __name__ == "__main__":
    main()
