"""Compare reconstruction scoring against a saved baseline source and FP64 oracle."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch

from sparseengine.kernels.triton import prefill_score as optimized


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("baseline_prefill_score", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def oracle(q, k, slots, context, sink, start, end):
    replay, heads, dim = q.shape
    kv_heads = k.shape[1]
    groups = heads // kv_heads
    positions = torch.cat((torch.arange(sink), torch.arange(start, end),
                           torch.arange(context, context + replay))).to(q.device)
    query_positions = context + torch.arange(replay, device=q.device)
    future = positions[None, :] > query_positions[:, None]
    output = torch.zeros((1, kv_heads, context), device=q.device)
    for head in range(kv_heads):
        queries = q[:, head * groups:(head + 1) * groups].double()
        keys = k[slots[0, positions].long(), head].double()
        logits = torch.einsum("qhd,kd->hqk", queries, keys) * dim**-0.5
        logits.masked_fill_(future[None], -torch.inf)
        output[0, head, start:end] = logits.softmax(-1)[
            :, :, sink:sink + end - start
        ].amax((0, 1)).float()
    return output


def timed(fn, iterations):
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=32640)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--sink", type=int, default=64)
    parser.add_argument("--replay-overhead", type=int, default=16)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.sink < args.context or min(args.chunk, args.rounds, args.iterations) <= 0:
        raise ValueError("Require positive chunks/rounds/iterations and sink < context")
    baseline = load_baseline(args.baseline_source)
    torch.manual_seed(47)
    starts = sorted({args.sink,
                     args.sink + ((args.context - args.sink - 1) // args.chunk // 2) * args.chunk,
                     args.sink + ((args.context - args.sink - 1) // args.chunk) * args.chunk})
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "baseline_sha256": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cases": []}
    for start in starts:
        end = min(start + args.chunk, args.context)
        replay = end - start + args.replay_overhead
        width = args.context + replay
        q = torch.randn(replay, args.heads, args.dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(width, args.kv_heads, args.dim, device="cuda", dtype=q.dtype)
        slots = torch.randperm(width, device="cuda").int()[None]
        zero = torch.zeros(1, device="cuda", dtype=torch.int32)
        cached = torch.full_like(zero, args.context)
        lengths = cached + replay
        calls, outputs, workspaces = {}, {}, {}
        for name, module in (("baseline", baseline), ("optimized", optimized)):
            score = torch.empty((1, width), device="cuda")
            heads = torch.zeros((1, args.kv_heads, args.context), device="cuda")
            workspace = module.PrefillScoreWorkspace()
            def run(module=module, score=score, heads=heads, workspace=workspace):
                module.prefill_score_fwd(
                    q, k, score, zero, zero, lengths, cached, replay, slots, cached, lengths,
                    reconstruction_range=(args.sink, start, end),
                    reconstruction_head_scores=heads, workspace=workspace,
                )
            calls[name], outputs[name], workspaces[name] = run, heads, workspace
            for _ in range(5):
                run()
        expected = oracle(q, k, slots, args.context, args.sink, start, end)
        for output in outputs.values():
            torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-3)
        torch.testing.assert_close(outputs["optimized"], outputs["baseline"], atol=2e-6, rtol=2e-3)
        samples = {name: [] for name in calls}
        for round_index in range(args.rounds):
            # Alternate order to limit systematic clock/temperature bias.
            for name in (list(calls) if round_index % 2 == 0 else list(reversed(calls))):
                samples[name].append(timed(calls[name], args.iterations))
        medians = {name: statistics.median(values) for name, values in samples.items()}
        row = {"start": start, "end": end, "replay": replay, "samples_ms": samples,
               "median_ms": medians, "speedup": medians["baseline"] / medians["optimized"],
               "max_abs_diff": (outputs["optimized"] - outputs["baseline"]).abs().max().item(),
               "max_abs_oracle_error": (outputs["optimized"] - expected).abs().max().item(),
               "partial_workspace_bytes": {name: w._partial_lse.numel() * 4
                                           for name, w in workspaces.items()}}
        result["cases"].append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
