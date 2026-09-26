"""Run a Vortex request probe using the shared zero-jitter v2 trace contract."""
import os
import runpy
import sys
from pathlib import Path

vortex_repo = Path(os.environ["VORTEX_REPO"])
sys.path.insert(0, str(vortex_repo))
from benchmark.efficiency import workload

workload.TRACE_GENERATOR_VERSION = "random-varlen-v2"
original_lengths = workload._jittered_lengths


def zero_jitter_lengths(rng, *, target_len, count, jitter_fraction, vary):
    if jitter_fraction == 0:
        return [target_len] * count
    return original_lengths(
        rng,
        target_len=target_len,
        count=count,
        jitter_fraction=jitter_fraction,
        vary=vary,
    )


workload._jittered_lengths = zero_jitter_lengths
probe = vortex_repo / "benchmark/efficiency/bench_probe.py"
sys.argv[0] = str(probe)
runpy.run_path(str(probe), run_name="__main__")
