"""Matched MLA prefill comparison including absorption, KV expansion and V reconstruction.

Run with PYTHONPATH=src:. python benchmark/mla_latent_prefill.py --output <RUN_DIR>.
This measures one synthetic attention layer, not model/serving latency.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import torch
import triton

from sparseengine import platforms
from sparseengine.engine.cache_manager.base import AttentionViewMeta, MlaLatentPayload, PrefillComputeView
from sparseengine.operators.mla_attention import MlaAttentionOpSpec, project_mla_values
from sparseengine.operators.mla_compressed_prefill import SglLatentPrefill, TritonLatentPrefill
from sparseengine.operators.mla_prefill import ChunkedMlaPrefill
from sparseengine.operators.mla_prefill_attention import resolve_mla_prefill


def _physical_gpu_selector(logical_device: int) -> str:
    logical_device = int(logical_device)
    if logical_device < 0:
        raise ValueError(f"CUDA device index must be nonnegative, got {logical_device}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(logical_device)
    devices = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if not devices or devices == ["-1"] or logical_device >= len(devices):
        raise ValueError(
            f"cuda:{logical_device} is unavailable under CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return devices[logical_device]


def gpu_guard(gpu_selector: str):
    result = subprocess.run([
        "nvidia-smi", "--id", str(gpu_selector), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    pids = []
    for row in result.stdout.splitlines():
        row = row.strip()
        if not row:
            continue
        try:
            pids.append(int(row))
        except ValueError as error:
            raise RuntimeError(f"GPU guard returned an invalid PID row: {row!r}") from error
    foreign = [pid for pid in pids if pid != os.getpid()]
    if foreign:
        raise RuntimeError(
            f"GPU guard for {gpu_selector}: foreign compute processes {foreign}"
        )


def _git_metadata():
    def run_git(*arguments):
        try:
            result = subprocess.run(
                ["git", *arguments], check=True, capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[1],
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run_git("status", "--porcelain")
    return {
        "git_commit": run_git("rev-parse", "HEAD"),
        "git_branch": run_git("branch", "--show-current"),
        "git_dirty": None if status is None else bool(status),
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _measure_cold_calls(calls):
    cold, outputs = {}, {}
    for name, call in calls.items():
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs[name] = call()
        torch.cuda.synchronize()
        cold[name] = time.perf_counter() - started
    return outputs, cold


def _run_benchmark(args, manifest, progress):
    gpu_selector = manifest["device"]["nvidia_smi_selector"]
    gpu_guard(gpu_selector)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    caps = platforms.current_platform.get_device_caps(args.device)
    manifest["device"].update(
        name=caps.device_name,
        capability=caps.compute_capability,
    )
    _write_json(args.output / "run_manifest.json", manifest)

    spec = MlaAttentionOpSpec(args.heads, 512, 64, 256, 256, torch.bfloat16, torch.bfloat16, 1, False)
    latent_provider = TritonLatentPrefill(spec, caps)
    expanded = resolve_mla_prefill(spec, caps)
    support = SglLatentPrefill.supports(spec, caps)
    sgl = SglLatentPrefill(spec, caps, 1) if support.supported else None
    summary = []
    with (args.output / "raw_samples.jsonl").open("w") as raw:
        for prefix in args.prefixes:
            for qn in args.queries:
                progress["active_case"] = {"prefix": prefix, "q": qn, "k": prefix + qn}
                gpu_guard(gpu_selector)
                torch.manual_seed(20260921)
                kn, heads = prefix + qn, args.heads
                q = torch.randn(qn, heads, 256, device=device, dtype=torch.bfloat16) * .3
                c = torch.randn(kn, 1, 512, device=device, dtype=torch.bfloat16)
                r = torch.randn(kn, 1, 64, device=device, dtype=torch.bfloat16) * .2
                weight = torch.randn(heads, 448, 512, device=device, dtype=torch.bfloat16) * .03
                slots = torch.randperm(kn, device=device).int()[None]
                view = PrefillComputeView(
                    AttentionViewMeta(slots, torch.zeros(1, device=device, dtype=torch.int32),
                                      torch.tensor([kn], device=device, dtype=torch.int32)),
                    MlaLatentPayload(c, r), host_request_layout=((0, qn, 0, kn),),
                )
                cu = torch.tensor([0, qn], device=device, dtype=torch.int32)
                scope = object()

                def project(x):
                    return torch.nn.functional.linear(x, weight.flatten(0, 1))

                def absorb(x):
                    return torch.bmm(x.transpose(0, 1), weight[:, :192]).transpose(0, 1)

                def partial(q, k, v, cq, ck, mq, mk, *, causal):
                    return expanded(q, k, v, cq, ck, mq, mk, scale=spec.softmax_scale, causal=causal)

                runner = ChunkedMlaPrefill(spec, SimpleNamespace(run_prefill_chunk=partial), 16384)
                plan = runner.prepare(view, cu, scope)

                def compressed(provider):
                    out, _ = provider.run(absorb(q[..., :192]), q[..., 192:], view, plan)
                    return project_mla_values(out, weight[:, 192:])

                calls = {
                    "triton_latent": lambda: compressed(latent_provider),
                    f"expanded_{expanded.name}": lambda: runner.run(q, view, cu, scope, project, absorb)[0],
                }
                if sgl is not None:
                    calls["sgl_fa3_latent"] = lambda: compressed(sgl)
                outputs, cold = _measure_cold_calls(calls)
                reference = outputs[f"expanded_{expanded.name}"]
                errors = {}
                for name, output in outputs.items():
                    torch.testing.assert_close(output, reference, atol=.015, rtol=.04)
                    errors[name] = float((output.float() - reference.float()).abs().max())
                del outputs, reference
                for _ in range(args.warmup):
                    for call in calls.values():
                        call()
                torch.cuda.synchronize()
                samples = {name: [] for name in calls}
                names = list(calls)
                for repetition in range(args.repetitions):
                    for name in names[repetition % len(names):] + names[:repetition % len(names)]:
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        calls[name]()
                        stop.record()
                        stop.synchronize()
                        ms = start.elapsed_time(stop)
                        samples[name].append(ms)
                        raw.write(json.dumps(dict(prefix=prefix, q=qn, k=kn, provider=name, repetition=repetition, ms=ms, status="success")) + "\n")
                    raw.flush()
                gpu_guard(gpu_selector)
                result = dict(prefix=prefix, q=qn, k=kn, status="success", cold_seconds=cold,
                              splits=latent_provider.splits(plan), scratch_bytes=latent_provider.workspace_bytes(plan),
                              max_abs_error=errors, median_ms={k: statistics.median(v) for k, v in samples.items()})
                summary.append(result)
                _write_json(args.output / "summary.json", summary)
                print(json.dumps(result), flush=True)
                del calls, runner, plan, view, q, c, r, weight, output
                progress["completed_cases"] = len(summary)
                progress["active_case"] = None
    return summary


@torch.inference_mode()
def _execute(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        command=[sys.executable, *sys.argv], cwd=os.getcwd(), git=_git_metadata(),
        python=sys.executable, torch=torch.__version__, triton=triton.__version__,
        cuda=torch.version.cuda, seed=20260921, dtype="bfloat16", heads=args.heads,
        warmup=args.warmup, repetitions=args.repetitions, graph=False,
        timing="interleaved CUDA events per complete attention wrapper; warm cache",
        boundary="one synthetic MLA layer including Q absorption or KV expansion and value reconstruction; no output projection, scores or serving",
        prefixes=args.prefixes, queries=args.queries,
        device={"logical_index": args.device, "nvidia_smi_selector": None},
        status="running",
    )
    progress = {"completed_cases": 0, "active_case": None}
    _write_json(args.output / "run_manifest.json", manifest)
    _write_json(args.output / "summary.json", [])
    _write_json(args.output / "status.json", {"status": "running", **progress})
    try:
        gpu_selector = _physical_gpu_selector(args.device)
        manifest["device"]["nvidia_smi_selector"] = gpu_selector
        _write_json(args.output / "run_manifest.json", manifest)
        summary = _run_benchmark(args, manifest, progress)
    except Exception as error:
        if isinstance(error, AssertionError):
            failure_status = "metric_failed"
        elif progress["active_case"] is None and isinstance(error, ValueError):
            failure_status = "invalid_input"
        else:
            failure_status = "model_failed"
        failure = {
            "status": failure_status,
            **progress,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        if progress["active_case"] is not None:
            with (args.output / "raw_samples.jsonl").open("a", encoding="utf-8") as raw:
                raw.write(json.dumps({
                    **progress["active_case"], "status": failure_status,
                    "error": repr(error),
                }) + "\n")
        manifest["status"] = failure_status
        _write_json(args.output / "run_manifest.json", manifest)
        _write_json(args.output / "status.json", failure)
        raise
    manifest["status"] = "success"
    _write_json(args.output / "run_manifest.json", manifest)
    _write_json(args.output / "status.json", {"status": "success", "cases": len(summary)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA device index.")
    parser.add_argument("--prefixes", type=int, nargs="+", default=[0, 32768, 122880])
    parser.add_argument("--queries", type=int, nargs="+", default=[16, 128, 384, 1024, 4096])
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if min(args.queries) <= 0 or min(args.prefixes) < 0 or min(args.warmup, args.repetitions, args.heads) <= 0:
        parser.error("require nonnegative prefixes and positive queries, heads and iteration counts")
    if args.device < 0:
        parser.error("--device must be nonnegative")
    _execute(args)


if __name__ == "__main__":
    main()
