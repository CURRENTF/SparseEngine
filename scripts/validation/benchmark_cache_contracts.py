"""CPU allocation or prefix attach plus cleanup; not end-to-end serving.

Copy this script and tests/cache_contracts into both worktrees to compare the
same fixture/driver against old and new production code. No oracle work is timed.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--max-regression', type=float, default=0.05)
    parser.add_argument('--attach-blocks', type=int, nargs='*', default=[])
    parser.add_argument('--attach-iterations', type=int, default=20)
    args = parser.parse_args()
    if args.iterations <= 0 or args.samples < 3 or args.max_regression < 0:
        parser.error('positive iterations, >=3 samples, and non-negative regression limit required')
    if args.attach_iterations <= 0 or any(n <= 0 for n in args.attach_blocks):
        parser.error('attach block counts and iterations must be positive')
    import torch
    from cache_contracts.cases import (
        CHAIN_CASES, RADIX_CASES, make_chain, make_radix, allocate_chain, make_inflight_radix,
    )
    from cache_contracts.ownership import ChainHarness, RadixHarness
    torch.set_num_threads(1)
    result = {'python': platform.python_version(), 'torch': str(torch.__version__),
              'platform': platform.platform(), 'threads': 1,
              'iterations': args.iterations, 'samples': args.samples,
              'attach_blocks': args.attach_blocks, 'attach_iterations': args.attach_iterations,
              'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
              'scope': 'CPU allocation or prefix attach plus cleanup; no model, DMA, graph or TP',
              'provenance': {
                  'root': str(ROOT), 'command': sys.argv, 'interpreter': sys.executable,
                  'device': 'cpu', 'slot_dtype': 'int32', 'warmup': 100,
                  'timer': 'perf_counter_ns, GC disabled during samples',
              },
              'cases': {}}
    jobs = []
    for case in CHAIN_CASES:
        m = make_chain(case)
        jobs.append(('chain:' + case.id, m, ChainHarness(m, shared_slots=case.shared),
                     lambda m=m, c=case: allocate_chain(m, c, 71, (3, 3)), args.iterations))
    for method in RADIX_CASES:
        m = make_radix(method)
        jobs.append(('radix:' + (method or 'vanilla'), m, RadixHarness(m, paged=method == 'quest'),
                     lambda m=m: m._allocate(71, 3), args.iterations))
    for method in ('', 'quest'):
        for blocks in args.attach_blocks:
            for grouping, group_size in (('shared', blocks), ('fragmented', 1)):
                m, seq = make_inflight_radix(method, blocks, group_size)
                jobs.append((f'attach:{method or "standard"}:{blocks}:{grouping}', m,
                             RadixHarness(m, paged=method == 'quest'),
                             lambda m=m, seq=seq: m._attach_prefix_cache_if_needed(seq),
                             args.attach_iterations))
    for name, manager, oracle, allocate, iterations in jobs:
        def cleanup():
            manager.free_seq(71)
            if name.startswith('attach:'):
                manager._prefix_offload_step_h2d_operations.clear()
        oracle.observe()
        for _ in range(100):
            allocate()
            cleanup()
        samples = []
        enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(args.samples):
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    allocate()
                    cleanup()
                samples.append((time.perf_counter_ns() - start) / iterations)
        finally:
            if enabled:
                gc.enable()
        oracle.observe()
        result['cases'][name] = {'median_ns': statistics.median(samples), 'samples_ns': samples}
    failures = []
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        for field in ('python', 'torch', 'platform', 'threads', 'iterations', 'samples',
                      'cpu_affinity', 'scope', 'attach_blocks', 'attach_iterations'):
            if baseline.get(field) != result[field]:
                parser.error(f'baseline environment/driver mismatch: {field}')
        if baseline['cases'].keys() != result['cases'].keys():
            parser.error('baseline configuration matrix differs')
        for name, value in result['cases'].items():
            ratio = value['median_ns'] / baseline['cases'][name]['median_ns']
            value['baseline_ratio'] = ratio
            if ratio > 1 + args.max_regression:
                failures.append(name)
    result['regressions'] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
