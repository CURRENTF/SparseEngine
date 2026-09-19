import argparse
import sys
import time
import torch

from sparseengine.kernels.triton.omnikv_fused import build_omnikv_keep_and_slots


def reference_build(topk_indices, hist_lens, recent_chunk_lens, buffer_req_to_token_slots, req_indices, num_sink):
    device = topk_indices.device
    batch_size, k = topk_indices.shape
    new_context_lens = num_sink + k + recent_chunk_lens
    max_s = int(new_context_lens.max().item())

    keep_indices = torch.zeros((batch_size, max_s), dtype=torch.int32, device=device)
    if num_sink > 0:
        keep_indices[:, :num_sink] = torch.arange(num_sink, device=device, dtype=torch.int32)
    keep_indices[:, num_sink:num_sink + k] = topk_indices

    max_rc_len = max_s - (num_sink + k)
    if max_rc_len > 0:
        rc_range = torch.arange(max_rc_len, device=device, dtype=torch.int32).unsqueeze(0)
        rc_indices = hist_lens.unsqueeze(1) + rc_range
        rc_mask = rc_range < recent_chunk_lens.unsqueeze(1)
        keep_indices[:, num_sink + k:] = torch.where(rc_mask, rc_indices, torch.zeros(1, dtype=torch.int32, device=device))

    buffer_req = buffer_req_to_token_slots[req_indices]
    active_slots = torch.gather(buffer_req, 1, keep_indices.to(torch.int64))
    return keep_indices, active_slots, new_context_lens


def run_once(device, max_model_len, batch_size, num_sink, k, recent_max, rows):
    hist_min = num_sink + k
    if hist_min >= max_model_len:
        return True

    hist_lens = torch.randint(
        low=hist_min,
        high=max_model_len - 1,
        size=(batch_size,),
        dtype=torch.int32,
        device=device,
    )
    recent_chunk_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
    for i in range(batch_size):
        max_recent_i = min(recent_max, max_model_len - int(hist_lens[i].item()))
        if max_recent_i <= 0:
            recent_chunk_lens[i] = 0
        else:
            recent_chunk_lens[i] = torch.randint(0, max_recent_i + 1, (1,), device=device, dtype=torch.int32)[0]

    topk_indices = torch.empty((batch_size, k), dtype=torch.int32, device=device)
    for i in range(batch_size):
        topk_indices[i] = torch.randint(num_sink, int(hist_lens[i].item()), (k,), device=device, dtype=torch.int32)

    req_indices = torch.randint(0, rows, (batch_size,), device=device, dtype=torch.int32)
    buffer_req_to_token_slots = torch.randint(
        0,
        max_model_len,
        (rows, max_model_len),
        device=device,
        dtype=torch.int32,
    )

    ref_keep, ref_slots, ref_new_lens = reference_build(
        topk_indices,
        hist_lens,
        recent_chunk_lens,
        buffer_req_to_token_slots,
        req_indices,
        num_sink,
    )
    ker_keep, ker_slots, ker_new_lens = build_omnikv_keep_and_slots(
        topk_indices,
        hist_lens,
        recent_chunk_lens,
        buffer_req_to_token_slots,
        req_indices,
        num_sink,
    )

    if not torch.equal(ref_keep, ker_keep):
        print("keep_indices mismatch")
        return False
    if not torch.equal(ref_slots, ker_slots):
        print("active_slots mismatch")
        return False
    if not torch.equal(ref_new_lens, ker_new_lens):
        print("new_context_lens mismatch")
        return False
    return True


def benchmark(device, max_model_len, batch_size, num_sink, k, recent_max, rows, iters, warmup):
    hist_min = num_sink + k
    if hist_min >= max_model_len:
        return None

    hist_lens = torch.randint(
        low=hist_min,
        high=max_model_len - 1,
        size=(batch_size,),
        dtype=torch.int32,
        device=device,
    )
    recent_chunk_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
    for i in range(batch_size):
        max_recent_i = min(recent_max, max_model_len - int(hist_lens[i].item()))
        if max_recent_i <= 0:
            recent_chunk_lens[i] = 0
        else:
            recent_chunk_lens[i] = torch.randint(0, max_recent_i + 1, (1,), device=device, dtype=torch.int32)[0]

    topk_indices = torch.empty((batch_size, k), dtype=torch.int32, device=device)
    for i in range(batch_size):
        topk_indices[i] = torch.randint(num_sink, int(hist_lens[i].item()), (k,), device=device, dtype=torch.int32)

    req_indices = torch.randint(0, rows, (batch_size,), device=device, dtype=torch.int32)
    buffer_req_to_token_slots = torch.randint(
        0,
        max_model_len,
        (rows, max_model_len),
        device=device,
        dtype=torch.int32,
    )

    for _ in range(warmup):
        reference_build(topk_indices, hist_lens, recent_chunk_lens, buffer_req_to_token_slots, req_indices, num_sink)
        build_omnikv_keep_and_slots(topk_indices, hist_lens, recent_chunk_lens, buffer_req_to_token_slots, req_indices, num_sink)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        reference_build(topk_indices, hist_lens, recent_chunk_lens, buffer_req_to_token_slots, req_indices, num_sink)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) * 1000 / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        build_omnikv_keep_and_slots(topk_indices, hist_lens, recent_chunk_lens, buffer_req_to_token_slots, req_indices, num_sink)
    torch.cuda.synchronize()
    ker_ms = (time.perf_counter() - t0) * 1000 / iters

    return ref_ms, ker_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--batch-min", type=int, default=1)
    parser.add_argument("--batch-max", type=int, default=32)
    parser.add_argument("--num-sink-max", type=int, default=16)
    parser.add_argument("--num-top-max", type=int, default=2048)
    parser.add_argument("--recent-max", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--bench-warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return 0

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    for i in range(args.iters):
        batch_size = torch.randint(args.batch_min, args.batch_max + 1, (1,)).item()
        num_sink = torch.randint(0, args.num_sink_max + 1, (1,)).item()
        k = torch.randint(1, args.num_top_max + 1, (1,)).item()
        ok = run_once(
            device,
            args.max_model_len,
            batch_size,
            num_sink,
            k,
            args.recent_max,
            args.rows,
        )
        if not ok:
            print(f"Failed at iter {i}")
            return 1
    print("All checks passed")

    bench = benchmark(
        device,
        args.max_model_len,
        args.batch_max,
        args.num_sink_max,
        args.num_top_max,
        args.recent_max,
        args.rows,
        args.bench_iters,
        args.bench_warmup,
    )
    if bench is not None:
        ref_ms, ker_ms = bench
        print(f"Reference avg: {ref_ms:.3f} ms, Fused kernel avg: {ker_ms:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
