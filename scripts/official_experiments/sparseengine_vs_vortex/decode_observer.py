"""Benchmark-only scheduler observation; no vFlow or method modifications."""
import importlib.util
import json
import os
from pathlib import Path
import time


def install():
    import vortex_torch  # Install the existing Vortex integration first.
    import torch
    from sglang.srt.managers.scheduler import Scheduler

    spec = importlib.util.spec_from_file_location("decode_window_metrics", os.environ["PAPER_DECODE_WINDOW_METRICS"])
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    output = Path(os.environ["PAPER_DECODE_WINDOW_OUTPUT"])
    expected = int(os.environ["PAPER_DECODE_WINDOW_BATCH"])
    steps = int(os.environ.get("PAPER_DECODE_WINDOW_STEPS", "256"))
    warmup = int(os.environ.get("PAPER_DECODE_WINDOW_WARMUP", "8"))
    original_recv, original_run = Scheduler.recv_requests, Scheduler.run_batch

    def graph_stats(scheduler):
        runner = scheduler.tp_worker.model_runner
        return {"capture_count": getattr(runner.graph_runner, "paper_capture_count", 0),
                "replay_count": getattr(runner.graph_runner, "paper_replay_count", 0),
                "eager_decode_count": getattr(runner, "paper_eager_decode_count", 0)}

    def recv(scheduler, *args, **kwargs):
        window = getattr(scheduler, "_paper_decode_window", None)
        if window is not None and window.result is None:
            window.boundary()
            if window.result is not None:
                with output.open("a") as handle:
                    handle.write(json.dumps({**window.result, "engine": "vortex",
                                             "overlap_enabled": scheduler.enable_overlap}) + "\n")
        return original_recv(scheduler, *args, **kwargs)

    def run(scheduler, batch, *args, **kwargs):
        ids = tuple(sorted(req.rid for req in batch.reqs))
        window = getattr(scheduler, "_paper_decode_window", None)
        if len(ids) == expected and (window is None or
                (window.result is not None and ids != window.ids)):
            window = metrics.DecodeOnlyWindow(
                expected, steps, warmup, synchronize=torch.cuda.synchronize,
                clock=time.perf_counter, graph_stats=lambda: graph_stats(scheduler))
            scheduler._paper_decode_window = window
        if window is not None:
            window.observe(is_decode=batch.forward_mode.is_decode(), request_ids=ids,
                           tokens=len(ids), admission_complete=(not scheduler.waiting_queue
                                                               and scheduler.chunked_req is None))
        return original_run(scheduler, batch, *args, **kwargs)

    Scheduler.recv_requests, Scheduler.run_batch = recv, run
