import time
import os
import json
import math
import threading
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import wraps

import torch
import sparseengine.platforms as platforms
from sparseengine.utils.log import logger

class Profiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.enabled = False
        self.nvtx_enabled = os.environ.get("SPARSEENGINE_NVTX", "0") == "1"
        self._no_trace = nullcontext()
        self.rank = 0
        # 通过环境变量开启设备同步，以准确测量设备耗时；保留旧 CUDA 名称兼容。
        self.device_sync = (
            os.environ.get("SPARSEENGINE_SYNC_DEVICE", "0") == "1"
            or os.environ.get("CUDA_SYNC_SENGINE", "0") == "1"
        )

    def trace(self, name: str):
        """Annotate launches without timing, device queries, or synchronization.

        Nsight attributes GPU work by launch correlation inside this CPU range;
        the duration of the range itself is not a GPU duration.
        """
        if self.nvtx_enabled:
            return platforms.current_platform.trace_range(name)
        return self._no_trace

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def set_rank(self, rank: int):
        self.rank = rank

    @contextmanager
    def record(self, name):
        if not self.enabled:
            yield
            return
        
        platform = platforms.current_platform
        capturing = platform.is_stream_capturing()
        if self.device_sync and not capturing:
            platform.synchronize()
        t1 = time.perf_counter()
        yield
        capturing = platform.is_stream_capturing()
        if self.device_sync and not capturing:
            platform.synchronize()
        t2 = time.perf_counter()
        
        self.times[name] += (t2 - t1)
        self.counts[name] += 1

    def reset(self):
        self.times.clear()
        self.counts.clear()

    def snapshot(self):
        return {
            name: {
                "calls": int(self.counts[name]),
                "total_s": float(total_s),
                "avg_ms": (
                    float(total_s) * 1000.0 / int(self.counts[name])
                    if self.counts[name]
                    else 0.0
                ),
            }
            for name, total_s in sorted(self.times.items())
        }

    def print_stats(self):
        if not self.enabled or not self.times:
            return

        logger.info(f"\n=== SparseEngine Profiler Report (Rank {self.rank}) ===")
        # 按照总耗时降序排列
        sorted_keys = sorted(self.times.keys(), key=lambda x: self.times[x], reverse=True)
        
        # 尝试找出总耗时作为基准 (通常是 step)
        total_time = self.times.get("step", sum(self.times.values()))
        if total_time == 0: total_time = 1e-9

        print(f"{'Category':<30} {'Calls':<10} {'Avg (ms)':<15} {'Total (s)':<15} {'Percentage':<10}")
        print("-" * 80)
        for key in sorted_keys:
            t = self.times[key]
            c = self.counts[key]
            avg = (t / c) * 1000 if c > 0 else 0
            pct = (t / total_time) * 100
            print(f"{key:<30} {c:<10} {avg:<15.4f} {t:<15.4f} {pct:<10.2f}%")
        print("-" * 80)

# 全局单例
profiler = Profiler()


class CpuTiming:
    """Opt-in inclusive host timings; never query or synchronize the device.

    Thread CPU excludes sleep but can include driver polling. Wall time includes
    waits; neither is GPU kernel time. Nested categories must not be summed.
    """

    def __init__(self, interval_s: float):
        if not math.isfinite(interval_s) or interval_s < 0:
            raise ValueError("CPU timing interval must be finite and nonnegative")
        self.interval_ns = int(interval_s * 1e9)
        self.local = threading.local()

    def timed(self, function):
        if not self.interval_ns:
            return function
        name = function.__qualname__

        @wraps(function)
        def wrapped(*args, **kwargs):
            state = getattr(self.local, "state", None)
            if state is None:
                state = {"start": time.perf_counter_ns(), "depth": 0, "stats": {}}
                self.local.state = state
            state["depth"] += 1
            wall_start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            failed = False
            try:
                return function(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                cpu_ns = time.thread_time_ns() - cpu_start
                end = time.perf_counter_ns()
                wall_ns = end - wall_start
                stats = state["stats"].setdefault(name, [0, 0, 0, 0, 0])
                stats[0] += 1
                stats[1] += cpu_ns
                stats[2] += wall_ns
                stats[3] = max(stats[3], wall_ns)
                stats[4] += int(failed)
                state["depth"] -= 1
                if state["depth"] == 0 and end - state["start"] >= self.interval_ns:
                    stages = {
                        key: dict(calls=row[0], cpu_ms=row[1] / 1e6,
                                  wall_ms=row[2] / 1e6, max_wall_ms=row[3] / 1e6,
                                  errors=row[4])
                        for key, row in sorted(state["stats"].items())
                    }
                    report = dict(pid=os.getpid(), rank=profiler.rank,
                                  thread=threading.get_native_id(),
                                  window_s=(end - state["start"]) / 1e9,
                                  inclusive=True, stages=stages)
                    state["stats"].clear()
                    state["start"] = end
                    logger.info("cpu_timing {}", json.dumps(report, separators=(",", ":")))

        return wrapped


cpu_timing = CpuTiming(float(os.environ.get("SPARSEENGINE_CPU_TIMING_INTERVAL_S", "0")))
