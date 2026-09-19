import argparse
import gc
import math
from collections.abc import Callable

import torch

from sparseengine.kernels.triton.prefill_score import (
    PrefillScoreWorkspace,
    prefill_score_fwd,
)


def _make_case(
    *,
    length: int,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    window: int,
    candidate_start: int,
    recent_keep_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
):
    prompt_cache_len = length - window
    if prompt_cache_len < 0:
        raise ValueError(f"length must be >= window, got length={length} window={window}")
    q = torch.randn((batch * window, num_heads, head_dim), dtype=dtype, device=device)
    k_cache = torch.randn(
        (batch * length, num_kv_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    req_to_tokens = torch.arange(
        0,
        batch * length,
        dtype=torch.int32,
        device=device,
    ).view(batch, length)
    b_req_idx = torch.arange(batch, dtype=torch.int32, device=device)
    b_start_loc = torch.arange(batch, dtype=torch.int32, device=device) * window
    context_lens = torch.full((batch,), length, dtype=torch.int32, device=device)
    prompt_cache_lens = torch.full(
        (batch,),
        prompt_cache_len,
        dtype=torch.int32,
        device=device,
    )
    score_starts = prompt_cache_lens.clone()
    score_ends = context_lens.clone()
    return {
        "q": q,
        "k_cache": k_cache,
        "req_to_tokens": req_to_tokens,
        "b_req_idx": b_req_idx,
        "b_start_loc": b_start_loc,
        "context_lens": context_lens,
        "prompt_cache_lens": prompt_cache_lens,
        "score_starts": score_starts,
        "score_ends": score_ends,
        "window": window,
        "length": length,
        "candidate_start": candidate_start,
        "recent_keep_tokens": recent_keep_tokens,
        "workspace": PrefillScoreWorkspace(),
        "probability_out": torch.empty(
            (batch, length), dtype=torch.float32, device=device
        ),
        "logits_out": torch.empty(
            (batch, length), dtype=torch.float32, device=device
        ),
    }


@torch.inference_mode()
def _optimized_triton(
    case: dict[str, torch.Tensor | int | PrefillScoreWorkspace],
    *,
    score_mode: str,
) -> torch.Tensor:
    out = case[f"{score_mode}_out"]
    prefill_score_fwd(
        case["q"],
        case["k_cache"],
        out,
        case["b_req_idx"],
        case["b_start_loc"],
        case["context_lens"],
        case["prompt_cache_lens"],
        int(case["window"]),
        case["req_to_tokens"],
        case["score_starts"],
        case["score_ends"],
        candidate_start=int(case["candidate_start"]),
        recent_keep_tokens=int(case["recent_keep_tokens"]),
        score_mode=score_mode,
        workspace=case["workspace"],
    )
    return out


@torch.inference_mode()
def _torch_expanded(case: dict[str, torch.Tensor | int]) -> torch.Tensor:
    q = case["q"]
    k_cache = case["k_cache"]
    length = int(case["length"])
    window = int(case["window"])
    candidate_start = int(case["candidate_start"])
    candidate_end = max(candidate_start, length - int(case["recent_keep_tokens"]))
    batch = int(case["context_lens"].numel())
    out = torch.zeros((batch, length), dtype=torch.float32, device=q.device)
    if candidate_end <= candidate_start:
        return out

    positions = torch.arange(
        candidate_start, candidate_end, dtype=torch.int64, device=q.device
    )
    q_positions = torch.arange(
        length - window, length, dtype=torch.int64, device=q.device
    )
    kv_group = q.shape[1] // k_cache.shape[1]
    head_to_kv = torch.arange(q.shape[1], device=q.device) // kv_group
    causal = q_positions[None, :, None] >= positions[None, None, :]
    for batch_idx in range(batch):
        q_batch = q[batch_idx * window : (batch_idx + 1) * window]
        slots = case["req_to_tokens"][batch_idx, candidate_start:candidate_end].long()
        k_candidates = k_cache.index_select(0, slots)
        k_expanded = k_candidates.index_select(1, head_to_kv)
        logits = torch.einsum("mhd,chd->hmc", q_batch.float(), k_expanded.float())
        logits = logits * (q.shape[-1] ** -0.5)
        logits = logits.masked_fill(~causal, -torch.inf)
        probs = torch.softmax(logits, dim=-1)
        scores = probs.sum(dim=1) / float(window)
        out[batch_idx, candidate_start:candidate_end] = scores.max(dim=0).values
    return out


@torch.inference_mode()
def _torch_grouped(case: dict[str, torch.Tensor | int]) -> torch.Tensor:
    q = case["q"]
    k_cache = case["k_cache"]
    length = int(case["length"])
    window = int(case["window"])
    candidate_start = int(case["candidate_start"])
    candidate_end = max(candidate_start, length - int(case["recent_keep_tokens"]))
    batch = int(case["context_lens"].numel())
    out = torch.zeros((batch, length), dtype=torch.float32, device=q.device)
    if candidate_end <= candidate_start:
        return out

    positions = torch.arange(candidate_start, candidate_end, dtype=torch.int64, device=q.device)
    q_positions = torch.arange(length - window, length, dtype=torch.int64, device=q.device)
    causal = q_positions[None, :, None] >= positions[None, None, :]
    kv_group = q.shape[1] // k_cache.shape[1]
    scale = q.shape[-1] ** -0.5
    for batch_idx in range(batch):
        q_batch = q[batch_idx * window : (batch_idx + 1) * window]
        slots = case["req_to_tokens"][batch_idx, candidate_start:candidate_end].long()
        k_candidates = k_cache.index_select(0, slots)
        best = torch.zeros(
            (candidate_end - candidate_start,), dtype=torch.float32, device=q.device
        )
        for kv_h in range(k_cache.shape[1]):
            q_group = q_batch[:, kv_h * kv_group : (kv_h + 1) * kv_group, :]
            k_group = k_candidates[:, kv_h, :]
            logits = (
                torch.einsum("mhd,cd->hmc", q_group.float(), k_group.float())
                * scale
            )
            logits = logits.masked_fill(~causal, -torch.inf)
            probs = torch.softmax(logits, dim=-1)
            scores = probs.sum(dim=1) / float(window)
            best = torch.maximum(best, scores.max(dim=0).values)
        out[batch_idx, candidate_start:candidate_end] = best
    return out


@torch.inference_mode()
def _torch_logits(case: dict[str, torch.Tensor | int]) -> torch.Tensor:
    q = case["q"]
    k_cache = case["k_cache"]
    length = int(case["length"])
    window = int(case["window"])
    candidate_start = int(case["candidate_start"])
    candidate_end = max(candidate_start, length - int(case["recent_keep_tokens"]))
    batch = int(case["context_lens"].numel())
    out = torch.full(
        (batch, length), -torch.inf, dtype=torch.float32, device=q.device
    )
    if candidate_end <= candidate_start:
        return out

    positions = torch.arange(
        candidate_start, candidate_end, dtype=torch.int64, device=q.device
    )
    q_positions = torch.arange(
        length - window, length, dtype=torch.int64, device=q.device
    )
    causal = q_positions[:, None] >= positions[None, :]
    kv_group = q.shape[1] // k_cache.shape[1]
    for batch_idx in range(batch):
        q_batch = q[batch_idx * window : (batch_idx + 1) * window]
        slots = case["req_to_tokens"][batch_idx, candidate_start:candidate_end].long()
        k_candidates = k_cache.index_select(0, slots)
        best = torch.full(
            (candidate_end - candidate_start,),
            -torch.inf,
            dtype=torch.float32,
            device=q.device,
        )
        for kv_h in range(k_cache.shape[1]):
            q_group = q_batch[:, kv_h * kv_group : (kv_h + 1) * kv_group, :]
            k_group = k_candidates[:, kv_h, :]
            logits = torch.einsum("mhd,cd->hmc", q_group.float(), k_group.float())
            logits = logits.masked_fill(~causal[None, :, :], -torch.inf)
            best = torch.maximum(best, logits.amax(dim=(0, 1)))
        out[batch_idx, candidate_start:candidate_end] = best
    return out


def _measure_latency(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


def _measure_peak_and_output(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    return out, peak / (1024.0 * 1024.0)


def _iters_for_length(length: int) -> int:
    if length >= 65536:
        return 20
    if length >= 32768:
        return 30
    return 50


def main():
    parser = argparse.ArgumentParser(description="Microbench Sparse-Engine prefill score implementations.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--candidate-start", type=int, default=4)
    parser.add_argument("--num-recent-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this microbench.")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(20260625)

    print(
        "shape: "
        f"B={args.batch} Hq={args.num_heads} Hkv={args.num_kv_heads} "
        f"D={args.head_dim} W={args.window} dtype=bf16 "
        f"candidate_start={args.candidate_start} recent_keep_tokens={args.recent_keep_tokens}"
    )
    print(f"gpu: {torch.cuda.get_device_name(device)}")
    print("| L | method | latency_ms | speedup_vs_grouped | peak_extra_mib | max_abs_diff_vs_grouped |")
    print("|---:|---|---:|---:|---:|---:|")
    for length in args.lengths:
        case = _make_case(
            length=length,
            batch=args.batch,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            window=args.window,
            candidate_start=args.candidate_start,
            recent_keep_tokens=args.recent_keep_tokens,
            dtype=dtype,
            device=device,
        )
        methods = [
            (
                "probability_triton",
                lambda case=case: _optimized_triton(
                    case, score_mode="probability"
                ),
            ),
            ("probability_torch_expanded", lambda case=case: _torch_expanded(case)),
            ("probability_torch_grouped", lambda case=case: _torch_grouped(case)),
            (
                "logits_triton",
                lambda case=case: _optimized_triton(case, score_mode="logits"),
            ),
            ("logits_torch", lambda case=case: _torch_logits(case)),
        ]
        compiled_once = [
            _optimized_triton(case, score_mode="probability"),
            _optimized_triton(case, score_mode="logits"),
        ]
        torch.cuda.synchronize()
        del compiled_once
        torch.cuda.empty_cache()

        outputs: dict[str, torch.Tensor] = {}
        peaks: dict[str, float] = {}
        latencies: dict[str, float] = {}
        iters = args.iters if args.iters > 0 else _iters_for_length(length)
        for name, fn in methods:
            out, peak_mib = _measure_peak_and_output(fn)
            outputs[name] = out
            peaks[name] = peak_mib
            latencies[name] = _measure_latency(fn, warmup=args.warmup, iters=iters)
        refs = {
            "probability": outputs["probability_torch_grouped"],
            "logits": outputs["logits_torch"],
        }
        torch_latencies = {
            "probability": latencies["probability_torch_grouped"],
            "logits": latencies["logits_torch"],
        }
        for name, _fn in methods:
            mode = "logits" if name.startswith("logits") else "probability"
            max_abs_diff = (outputs[name] - refs[mode]).abs().nan_to_num().max().item()
            speedup = (
                torch_latencies[mode] / latencies[name]
                if latencies[name] > 0
                else math.inf
            )
            print(
                f"| {length} | {name} | {latencies[name]:.4f} | {speedup:.3f} | "
                f"{peaks[name]:.1f} | {max_abs_diff:.6g} |"
            )
        del case, outputs, peaks, latencies, refs, torch_latencies
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
