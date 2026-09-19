"""Matched raw-QK -> OmniKV token score / token top-k microbenchmark, not model decode."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import triton

from sparseengine.engine.sparse_methods.base import SparseMethodRuntime
from sparseengine.operators.omnikv_score import OmniKVScoreSpec, prepare_omnikv_score_provider


def baseline(raw, lengths, sink, recent, scale, dtype):
    runtime = SimpleNamespace(attn_softmax_scale=scale, config=SimpleNamespace(hf_config=SimpleNamespace(dtype=dtype)))
    return SparseMethodRuntime._decode_softmax_token_scores(
        runtime, raw, candidate_start=sink,
        candidate_lens=(lengths - recent).clamp_min(sink) - sink,
    )


def topk(scores, lengths, sink, recent, keep):
    search = scores[:, sink:]
    candidate_lens = (lengths - recent).clamp_min(sink) - sink
    search.masked_fill_(torch.arange(search.shape[1], device=search.device) >= candidate_lens[:, None], -1e10)
    return search.topk(min(keep, search.shape[1]), dim=1, sorted=False).indices.to(torch.int32) + sink


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def measure(fn, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--heads', type=int, default=10)
    p.add_argument('--capacity', type=int, default=202752)
    p.add_argument('--lengths', nargs='+', type=int, default=[16384, 65536, 202752])
    p.add_argument('--block', type=int, default=2048)
    p.add_argument('--output-block', type=int, default=256)
    p.add_argument('--samples', type=int, default=15)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--seed', type=int, default=20260916)
    p.add_argument('--eager', action='store_true')
    args = p.parse_args()
    if min(args.batch, args.heads, args.samples, args.iterations) < 1 or args.capacity <= 576:
        p.error('positive batch/heads/samples/iterations and capacity > sink + recent (576) required')
    if any(length <= 576 or length > args.capacity for length in args.lengths):
        p.error('lengths must exceed 576 and not exceed capacity')
    if any(value <= 0 or value & (value - 1) for value in (args.block, args.output_block)):
        p.error('kernel tile sizes must be positive powers of two')
    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, obj):
        with (args.output / name).open('x') as f:
            json.dump(obj, f, indent=2)
    manifest = {'args':vars(args) | {'output':str(args.output)}, 'torch':torch.__version__, 'triton':triton.__version__,
                'cuda':torch.version.cuda, 'device':torch.cuda.get_device_name(), 'started':time.time(),
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'dirty':subprocess.check_output(['git','status','--short'],text=True),
                'scope':'FP32 raw QK to BF16 token scores, optionally including existing masked torch.topk; no QK generation or model execution'}
    write('run_manifest.json', manifest)
    torch.manual_seed(args.seed)
    raw = torch.randn(args.batch, args.heads, args.capacity, device='cuda', dtype=torch.float32) * 16
    sink, recent, keep, scale = 64, 512, 1472, 0.0625
    provider = prepare_omnikv_score_provider(OmniKVScoreSpec(sink, recent, scale, torch.bfloat16), device=raw.device)
    provider.block, provider.output_block = args.block, args.output_block
    provider.prepare(raw, slot=0)
    results = []
    try:
        for length in args.lengths:
            lengths = torch.full((args.batch,), length, device='cuda', dtype=torch.int32)
            def reference():
                return baseline(raw, lengths, sink, recent, scale, torch.bfloat16)
            def candidate():
                return provider.run(raw, lengths, slot=0)
            expected, actual = reference(), candidate()
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-8)
            valid = (torch.arange(args.capacity,device='cuda')[None,:] >= sink) & (torch.arange(args.capacity,device='cuda')[None,:] < lengths[:,None]-recent)
            error = {'max_abs':(actual.float()[valid]-expected.float()[valid]).abs().max().item(),
                     'exact_fraction':(actual[valid]==expected[valid]).float().mean().item()}
            expected_ids = topk(expected, lengths, sink, recent, keep).long()
            actual_ids = topk(actual, lengths, sink, recent, keep).long()
            # Independent selection oracle: no selected value may be below the
            # reference kth score except by the declared normalization tolerance.
            threshold = expected.gather(1, expected_ids).amin(1)
            selected = expected.gather(1, actual_ids).amin(1)
            assert bool(torch.all(selected.float() >= threshold.float() * (1 - .008))), (selected, threshold)
            intersection = torch.zeros_like(expected, dtype=torch.bool).scatter_(1, expected_ids, True).gather(1, actual_ids).float().mean().item()
            error['topk_set_overlap'] = intersection
            for scope in ('normalize', 'normalize_and_topk'):
                functions = [reference, candidate] if scope == 'normalize' else [lambda: topk(reference(),lengths,sink,recent,keep), lambda: topk(candidate(),lengths,sink,recent,keep)]
                graphs = [capture(fn) for fn in functions] if not args.eager else []
                calls = [g.replay for g,_ in graphs] if graphs else functions
                for _ in range(5):
                    for fn in calls:
                        fn()
                samples = [[], []]
                for sample in range(args.samples):
                    for i in ([0,1] if sample % 2 == 0 else [1,0]):
                        samples[i].append(measure(calls[i], args.iterations))
                row = {'length':length, 'scope':scope, 'baseline_ms':samples[0], 'candidate_ms':samples[1],
                       'baseline_median_ms':statistics.median(samples[0]), 'candidate_median_ms':statistics.median(samples[1]),
                       'speedup':statistics.median(samples[0])/statistics.median(samples[1]), 'correctness':error}
                results.append(row)
                with (args.output/'raw_samples.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
                del graphs
        write('summary.json', {'status':'completed','cases':results})
    except BaseException as error:
        write('failure.json', {'status':'failed','type':type(error).__name__,'error':str(error)})
        raise


if __name__ == '__main__':
    main()
