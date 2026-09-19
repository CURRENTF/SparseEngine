"""Compare MLA partial-attention wrappers against Git or vLLM baselines."""

import argparse
import hashlib
from functools import partial
import importlib.metadata
import importlib.util
import json
import itertools
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def check_sampled_oracle(q, k, v, cu_q, cu_k, output, lse, causal):
    # Bound oracle memory even for long contexts; cover first/middle/last Q.
    for qa, qb, ka, kb in zip(cu_q, cu_q[1:], cu_k, cu_k[1:]):
        qn, kn = qb - qa, kb - ka
        rows = torch.tensor(sorted({0, qn // 2, qn - 1}), device=q.device)
        z = torch.einsum("qhd,khd->hqk", q[qa + rows].float(), k[ka:kb].float()) * 0.0625
        if causal:
            z.masked_fill_(torch.arange(kn, device=q.device)[None, None] > (rows + kn - qn)[None, :, None], -torch.inf)
        expected = torch.einsum("hqk,khd->qhd", z.softmax(-1), v[ka:kb].float())
        torch.testing.assert_close(output[qa + rows].float(), expected, atol=0.006, rtol=0.03)
        if lse is not None:
            torch.testing.assert_close(lse[:, qa + rows], z.logsumexp(-1), atol=0.003, rtol=0.001)


def comparison_cases(suite):
    """Fixed-work prefill batches plus short-query history replacement probes."""
    if suite == "original":
        for qn, kn, causal in [
            (4058, 4058, True), (8169, 8169, True),
            (8192, 16233, False), (8192, 32938, False), (8192, 64581, False),
        ]:
            yield dict(group="original", layout="uniform", queries=[qn], keys=[kn], causal=causal)
        return
    if suite == "multibatch":
        for batch in (1, 2, 4, 8, 16):
            for layout in (("uniform",) if batch == 1 else ("uniform", "ragged")):
                weights = [1 if layout == "uniform" else (1, 3, 7, 13)[i % 4] for i in range(batch)]
                queries = [8192 * w // sum(weights) for w in weights]
                queries[-1] += 8192 - sum(queries)
                yield dict(group="fixed_total_q", layout=layout, queries=queries, keys=queries, causal=True)
                for history in (16384, 32768, 65536):
                    keys = [history if layout == "uniform" else max(1, history * (4 - i % 4) // 4 - i % 3) for i in range(batch)]
                    yield dict(group="fixed_total_q", layout=layout, queries=queries, keys=keys, causal=False)
    elif suite == "longcontext":
        for qn, history in [(8192, 131072), (8192, 262144), (8192, 524288), (32, 524288), (128, 524288)]:
            yield dict(group="long_context", layout="uniform", queries=[qn], keys=[history], causal=False)
    elif suite == "shortquery":
        for batch in (1, 8):
            for qn in (1, 32, 128, 256):
                for history in (16384, 65536):
                    yield dict(group="short_query", layout="uniform", queries=[qn] * batch, keys=[history] * batch, causal=False)


def check_comparison_oracle(q, k, v, queries, keys, out, lse, causal):
    # Every sequence/head; tile-boundary rows plus first/middle/last. Check all
    # rows for short queries. Head-wise conversion bounds memory at large B*K.
    qa = ka = 0
    maxout = maxlse = max_relative_l2 = 0.0
    for qn, kn in zip(queries, keys):
        rows = range(qn) if qn <= 256 else sorted({i for i in (0, 1, 31, 32, 127, 128, 255, 256, qn // 2, qn - 1) if i < qn})
        qi = torch.tensor(list(rows), device=q.device)
        for h in range(q.shape[1]):
            z = (q[qa + qi, h].float() @ k[ka:ka + kn, h].float().T) * 0.0625
            if causal:
                z.masked_fill_(torch.arange(kn, device=q.device)[None] > qi[:, None] + kn - qn, -float("inf"))
            ref = z.softmax(-1) @ v[ka:ka + kn, h].float()
            refl = z.logsumexp(-1)
            torch.testing.assert_close(out[qa + qi, h].float(), ref, atol=0.002, rtol=0.02)
            torch.testing.assert_close(lse[h, qa + qi], refl, atol=1e-4, rtol=1e-4)
            error = out[qa + qi, h].float() - ref
            relative_l2 = (error.norm() / ref.norm().clamp_min(torch.finfo(torch.float32).tiny)).item()
            if not relative_l2 <= 0.02:
                raise AssertionError(f"Output relative L2 error {relative_l2} exceeds 0.02 at sequence Q offset {qa}, head {h}")
            max_relative_l2 = max(max_relative_l2, relative_l2)
            maxout = max(maxout, error.abs().max().item())
            maxlse = max(maxlse, (lse[h, qa + qi] - refl).abs().max().item())
        qa += qn
        ka += kn
    return dict(output_max_abs=maxout, lse_max_abs=maxlse, output_max_relative_l2=max_relative_l2)


def compare_vllm(args):
    """Matched GLM kernel comparison, including allocation and natural-log LSE."""
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )

    SEED, SCALE = 20260915, 0.0625
    fa_version = get_flash_attn_version(head_size=256, head_size_v=256)
    if fa_version is None:
        raise RuntimeError("CUDA vLLM FlashAttention version resolution failed; check vLLM dependency imports.")

    def vllm_prefill(q, k, v, cq, ck, qn, kn, causal):
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            max_seqlen_q=qn,
            max_seqlen_k=kn,
            softmax_scale=SCALE,
            causal=causal,
            return_softmax_lse=True,
            fa_version=fa_version,
        )

    def make_prefill_inputs(queries, keys):
        qn, kn = sum(queries), sum(keys)
        torch.manual_seed(SEED + qn + kn)
        q = torch.randn(qn, 20, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(kn, 20, 256, device="cuda", dtype=torch.bfloat16)
        projected = torch.randn(kn, 20, 448 if args.strided else 256, device="cuda", dtype=torch.bfloat16)
        q.mul_(0.2)
        k.mul_(0.2)
        projected.mul_(0.2)
        v = projected[..., 192:] if args.strided else projected
        cq = torch.tensor([0, *itertools.accumulate(queries)], device="cuda", dtype=torch.int32)
        ck = torch.tensor([0, *itertools.accumulate(keys)], device="cuda", dtype=torch.int32)
        return q, k, v, cq, ck

    args.output_dir.mkdir(parents=True, exist_ok=False)
    from sparseengine.kernels.triton.mla.prefill_pipelined import attention_partial

    candidate = attention_partial
    if args.candidate_module:
        module, function = args.candidate_module.split(":", 1)
        candidate = getattr(importlib.import_module(module), function)
    elif args.candidate_kernel == "pipelined-split":
        candidate = partial(attention_partial, split_kv=True)
    elif args.candidate_kernel == "hopper":
        from sparseengine.kernels.triton.mla.prefill_hopper import attention_partial

        candidate = partial(attention_partial, split_kv=True)
    elif args.candidate_kernel == "provider":
        from sparseengine.operators.mla_attention import (
            MlaAttentionOpSpec,
            MlaTritonProvider,
        )

        spec = MlaAttentionOpSpec(
            num_q_heads=20,
            kv_lora_rank=512,
            rope_dim=64,
            qk_head_dim=256,
            value_head_dim=256,
            activation_dtype=torch.bfloat16,
            cache_dtype=torch.bfloat16,
            tp_size=1,
            cuda_graph=False,
        )
        provider = MlaTritonProvider(op_spec=spec, device="cuda:0", max_batch_size=16)

        def candidate(q, k, v, cq, ck, qn, kn, *, scale, causal):
            return provider.run_prefill_chunk(q, k, v, cq, ck, qn, kn, causal=causal)

    cases = list(comparison_cases(args.suite))
    if args.case_indices is not None:
        cases = [cases[i] for i in args.case_indices]
    manifest = dict(
        command=sys.argv,
        cwd=os.getcwd(),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        torch=torch.__version__,
        triton=importlib.metadata.version("triton"),
        vllm=importlib.metadata.version("vllm"),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        compute_capability=list(torch.cuda.get_device_capability(0)),
        graph=False,
        topology="TP1 EP1; BF16; 20 local heads; QK/V dimension 256",
        seed=SEED,
        suite=args.suite,
        cases=cases,
        include_old=args.include_old,
        oracle="FP32 Torch, every sequence/head, all Q rows up to 256 otherwise ten boundary rows; relative L2 <= 0.02 also required",
        validate_only=args.validate_only,
        status="running",
        timing="not timed" if args.validate_only else "allocating wrapper, eager CUDA events, explicit warmups, palindromic interleaved, warm caches",
        fa_version=fa_version,
        rounds=args.rounds,
        warmup=args.warmup,
    )
    mp = args.output_dir / "run_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2))
    results = []
    try:
        with (args.output_dir / "raw_samples.jsonl").open("x") as raw:
            for case_id, case in enumerate(cases):
                queries, keys, causal = case["queries"], case["keys"], case["causal"]
                qn, kn = max(queries), max(keys)
                identity = dict(case_id=case_id, **case, batch=len(queries), total_q=sum(queries), total_k=sum(keys), qn=qn, kn=kn)
                print(json.dumps(dict(status="validating", **identity)), flush=True)
                q, k, v, cq, ck = make_prefill_inputs(queries, keys)
                calls = {
                    "candidate": partial(candidate, q, k, v, cq, ck, qn, kn, scale=SCALE, causal=causal),
                    "vllm": partial(vllm_prefill, q, k, v, cq, ck, qn, kn, causal),
                }
                if args.include_old:
                    from sparseengine.kernels.triton.mla.prefill import attention_partial as old_partial
                    calls["old"] = partial(old_partial, q, k, v, cq, ck, qn, kn, scale=SCALE, causal=causal)
                errors = {}
                cold_ms = {}
                for name, call in calls.items():
                    print(json.dumps(dict(status="checking_implementation", implementation=name, **identity)), flush=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    out, lse = call()[:2]
                    torch.cuda.synchronize()
                    cold_ms[name] = (time.perf_counter() - start) * 1000
                    errors[name] = check_comparison_oracle(q, k, v, queries, keys, out, lse, causal)
                    del out, lse
                    if not args.validate_only:
                        for _ in range(args.warmup):
                            call()
                torch.cuda.synchronize()
                if args.validate_only:
                    row = dict(**identity, correctness="passed", errors=errors)
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2))
                    del calls, call, q, k, v, cq, ck
                    torch.cuda.empty_cache()
                    continue
                samples = {n: [] for n in calls}
                for r in range(args.rounds):
                    for name in [*calls, *reversed(calls)]:
                        a, b = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        a.record()
                        calls[name]()
                        b.record()
                        b.synchronize()
                        ms = a.elapsed_time(b)
                        samples[name].append(ms)
                        raw.write(
                            json.dumps(
                                dict(
                                    **identity,
                                    implementation=name,
                                    round=r,
                                    ms=ms,
                                )
                            )
                            + "\n"
                        )
                        raw.flush()
                times = {n: statistics.median(vs) for n, vs in samples.items()}
                row = dict(
                    **identity,
                    strides=dict(q=q.stride(), k=k.stride(), v=v.stride()),
                    provider_paths=provider.runtime_kernel_stats() if args.candidate_kernel == "provider" and not args.candidate_module else None,
                    correctness="passed",
                    errors=errors,
                    cold_first_call_ms=cold_ms,
                    median_ms=times,
                    throughput_ratio=times["vllm"] / times["candidate"],
                    speedup_over_old=times["old"] / times["candidate"] if args.include_old else None,
                    samples={n: len(vs) for n, vs in samples.items()},
                )
                results.append(row)
                print(json.dumps(row), flush=True)
                (args.output_dir / "summary.json").write_text(
                    json.dumps(results, indent=2)
                )
                del calls, call, q, k, v, cq, ck
                torch.cuda.empty_cache()
        manifest.update(
            status="passed"
            if args.validate_only or all(r["throughput_ratio"] >= 0.95 for r in results)
            else "target_not_met",
            completed_cases=len(results),
        )
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        mp.write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline_group = parser.add_mutually_exclusive_group(required=True)
    baseline_group.add_argument("--baseline-ref")
    baseline_group.add_argument("--vllm", action="store_true")
    parser.add_argument("--git-kernel", choices=("original", "pipelined"), default="original",
                        help="Kernel family for baseline-ref/worktree comparison; pipelined enables split-KV")
    parser.add_argument("--candidate-module", help="Explicit experimental module:callable; overrides candidate-kernel")
    parser.add_argument(
        "--candidate-kernel",
        choices=("pipelined", "pipelined-split", "hopper", "provider"),
        default="provider",
    )
    parser.add_argument("--suite", choices=("original", "multibatch", "shortquery", "longcontext"), default="original")
    parser.add_argument("--include-old", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Run independent numerical oracles without warmup or timing")
    parser.add_argument("--case-indices", type=int, nargs="+", help="Run selected suite cases, for smoke or a bounded repeat")
    parser.add_argument("--strided-v", dest="strided", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--heads", type=int, choices=(5, 10, 20), default=20)
    args = parser.parse_args()
    if not args.vllm and (args.suite != "original" or args.include_old or args.validate_only or args.case_indices is not None):
        parser.error("suite, include-old, validate-only and case-indices require --vllm")
    if args.case_indices is not None:
        count = len(list(comparison_cases(args.suite)))
        if any(i < 0 or i >= count for i in args.case_indices):
            parser.error(f"case-indices must be in [0, {count})")
    if args.warmup < 1 or args.rounds < 1:
        parser.error("warmup and rounds must be positive")
    if args.vllm:
        if args.git_kernel != "original":
            parser.error("git-kernel requires --baseline-ref")
        if args.heads != 20:
            parser.error("the matched GLM vLLM comparison requires 20 heads")
        return compare_vllm(args)
    module_name = "prefill_pipelined" if args.git_kernel == "pipelined" else "prefill"
    prefill = importlib.import_module(f"sparseengine.kernels.triton.mla.{module_name}")
    from sparseengine.kernels.triton.context_flashattention_nopad import context_attention_fwd

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run_manifest.json").exists() or (root / "raw_samples.jsonl").exists():
        raise FileExistsError("Use a new output directory to preserve previous results")
    path = f"src/sparseengine/kernels/triton/mla/{module_name}.py"
    source = git("show", f"{args.baseline_ref}:{path}") + "\n"
    baseline_path = root / "baseline_prefill.py"
    baseline_path.write_text(source)
    spec = importlib.util.spec_from_file_location("sparseengine.kernels.triton.mla.baseline_prefill", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    props = torch.cuda.get_device_properties(0)
    manifest = {
        "status": "running", "command": [sys.executable, *sys.argv],
        "repo": git("rev-parse", "--show-toplevel"), "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "git_status": git("status", "--short"),
        "baseline": git("rev-parse", args.baseline_ref),
        "git_kernel": args.git_kernel,
        "gpu": props.name, "capability": [props.major, props.minor],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "gpu_state": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm,power.limit", "--format=csv"], text=True),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "triton", "sglang-kernel")},
        "cuda": torch.version.cuda, "seed": 42, "dtype": "bfloat16", "heads": args.heads,
        "head_dim": 256, "graph": False, "warmup": args.warmup, "rounds": args.rounds,
        "timing": "CUDA events around wrapper including output/LSE allocation; warm caches; ABBA pairs; no clock control",
        "candidate_sha256": hashlib.sha256(Path(prefill.__file__).read_bytes()).hexdigest(),
    }
    (root / "candidate_prefill.py").write_text(Path(prefill.__file__).read_text())
    (root / "benchmark_script.py").write_text(Path(__file__).read_text())
    (root / "changes.patch").write_text(git("diff", "HEAD"))
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    results = []
    try:
        with (root / "raw_samples.jsonl").open("x") as raw:
            for queries, keys, causal in (
                ((4096,), (4096,), True), ((8192,), (8192,), True),
                ((4096, 7), (4096, 7), True),
                ((32,), (16384,), False), ((128,), (16384,), False),
                ((129,), (16384,), False), ((512,), (16384,), False),
                ((8192,), (16384,), False),
            ):
                torch.manual_seed(42)
                q = torch.randn(sum(queries), args.heads, 256, device="cuda", dtype=torch.bfloat16) * 0.2
                k = torch.randn(sum(keys), args.heads, 256, device="cuda", dtype=torch.bfloat16) * 0.2
                # Match the V view returned by the serving joint KV projection.
                projected = torch.randn(sum(keys), args.heads, 448, device="cuda", dtype=torch.bfloat16) * 0.2
                v = projected[..., 192:]
                cq = [0, *torch.tensor(queries).cumsum(0).tolist()]
                ck = [0, *torch.tensor(keys).cumsum(0).tolist()]
                cu_q, cu_k = [torch.tensor(x, device="cuda", dtype=torch.int32) for x in (cq, ck)]

                def call(module):
                    kwargs = {"split_kv": True} if args.git_kernel == "pipelined" else {}
                    return module.attention_partial(q, k, v, cu_q, cu_k, max(queries), max(keys), scale=0.0625, causal=causal, **kwargs)

                calls = {"pr_original": lambda: call(baseline), "candidate": lambda: call(prefill)}
                if causal and args.git_kernel == "original":
                    slots = torch.full((len(keys), max(keys)), -1, device="cuda", dtype=torch.int32)
                    for i, (a, b) in enumerate(zip(ck, ck[1:])):
                        slots[i, :b - a] = torch.arange(a, b, device="cuda", dtype=torch.int32)
                    rows = torch.arange(len(keys), device="cuda", dtype=torch.int32)
                    lengths = cu_k[1:] - cu_k[:-1]
                    cached = lengths - (cu_q[1:] - cu_q[:-1])
                    old_output = torch.empty_like(q)

                    def legacy():
                        context_attention_fwd(q, k, v, old_output, rows, cu_q[:-1], lengths, cached, max(queries), slots)
                        return old_output, None

                    calls["legacy_causal"] = legacy
                cold_ms = {}
                for name, fn in calls.items():
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output, lse = fn()
                    torch.cuda.synchronize()
                    cold_ms[name] = (time.perf_counter() - start) * 1000
                    check_sampled_oracle(q, k, v, cq, ck, output, lse, causal)
                    for _ in range(args.warmup):
                        fn()
                torch.cuda.synchronize()
                samples = {name: [] for name in calls}
                pairs = [("pr_original", "candidate")]
                if causal and args.git_kernel == "original":
                    pairs.append(("legacy_causal", "candidate"))
                for pair in pairs:
                    for iteration in range(args.rounds):
                        for name in (*pair, *reversed(pair)):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            calls[name]()
                            end.record()
                            end.synchronize()
                            elapsed = start.elapsed_time(end)
                            samples[name].append(elapsed)
                            raw.write(json.dumps({"queries": queries, "keys": keys, "causal": causal, "pair": pair, "iteration": iteration, "implementation": name, "ms": elapsed}) + "\n")
                row = {"queries": queries, "keys": keys, "causal": causal, "status": "success", "cold_first_call_ms": cold_ms,
                       "strides": {"q": q.stride(), "k": k.stride(), "v": v.stride()},
                       "timings": {name: {"n": len(values), "median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)} for name, values in samples.items()}}
                results.append(row)
                print(json.dumps(row), flush=True)
                (root / "summary.json").write_text(json.dumps(results, indent=2))
        manifest["status"] = "success"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
