#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from statistics import median

import torch
from flash_attn import flash_attn_func

from sparseengine.engine.cache_manager.raw_kv_offload import RawKVOffloadBuffer


def parse_int_list(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def cuda_elapsed_ms(fn, *, repeats: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wall_ms: list[float] = []
    event_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        event_ms.append(float(start.elapsed_time(end)))
    return median(wall_ms), median(event_ms)


def wall_elapsed_ms(fn, *, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return median(times)


def build_chunked_buffer(
    *,
    total_len: int,
    storage_chunk_size: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    pin_memory: bool,
) -> RawKVOffloadBuffer:
    buffer = RawKVOffloadBuffer(pin_memory=pin_memory, mode="chunked")
    buffer.ensure_entry(
        layer_idx=0,
        row_idx=0,
        kind="sparse_pre_rope",
        total_len=total_len,
        k_shape_tail=(kv_heads, head_dim),
        v_shape_tail=(kv_heads, head_dim),
        dtype=dtype,
    )
    for start in range(0, total_len, storage_chunk_size):
        this_len = min(storage_chunk_size, total_len - start)
        shape = (this_len, kv_heads, head_dim)
        k = torch.empty(shape, dtype=dtype, device="cuda")
        v = torch.empty(shape, dtype=dtype, device="cuda")
        buffer.put_range(layer_idx=0, row_idx=0, kind="sparse_pre_rope", start=start, k=k, v=v)
        del k, v
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return buffer


def legacy_get_prefix_cpu(
    buffer: RawKVOffloadBuffer,
    *,
    layer_idx: int,
    row_idx: int,
    kind: str,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(layer_idx), int(row_idx), str(kind))
    entry = buffer._entries.get(key)
    if entry is None:
        raise RuntimeError(f"RawKVOffloadBuffer entry is missing for key={key}.")
    end = int(end)
    if end < 0 or end > int(entry.filled_until):
        raise RuntimeError(
            "legacy get_prefix_cpu reads an unwritten range: "
            f"key={key} end={end} filled_until={int(entry.filled_until)}."
        )
    if entry.chunks is None:
        if entry.k is None or entry.v is None:
            raise RuntimeError(f"RawKVOffloadBuffer contiguous entry is missing tensors for key={key}.")
        return entry.k[:end], entry.v[:end]

    k_out = torch.empty((end, *entry.k_shape_tail), dtype=entry.dtype, device="cpu", pin_memory=buffer.pin_memory)
    v_out = torch.empty((end, *entry.v_shape_tail), dtype=entry.dtype, device="cpu", pin_memory=buffer.pin_memory)
    cursor = 0
    for chunk_start in sorted(entry.chunks):
        k_chunk, v_chunk = entry.chunks[chunk_start]
        chunk_start = int(chunk_start)
        chunk_end = chunk_start + int(k_chunk.shape[0])
        if chunk_start > cursor:
            raise RuntimeError(
                "legacy get_prefix_cpu found a gap: "
                f"key={key} cursor={cursor} next_start={chunk_start}."
            )
        copy_end = min(end, chunk_end)
        if cursor < copy_end:
            src_start = cursor - chunk_start
            src_end = copy_end - chunk_start
            k_out[cursor:copy_end].copy_(k_chunk[src_start:src_end], non_blocking=True)
            v_out[cursor:copy_end].copy_(v_chunk[src_start:src_end], non_blocking=True)
            cursor = copy_end
        if cursor >= end:
            break
    if cursor != end:
        raise RuntimeError(f"legacy get_prefix_cpu did not fill requested prefix: key={key} cursor={cursor} end={end}.")
    return k_out, v_out


def measure_current_prefetch_schedule(
    *,
    buffer: RawKVOffloadBuffer,
    end: int,
    dtype: torch.dtype,
    device: torch.device,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    stream = torch.cuda.Stream(device=device)

    def once() -> None:
        k_cpu, v_cpu = legacy_get_prefix_cpu(buffer, layer_idx=0, row_idx=0, kind="sparse_pre_rope", end=end)
        with torch.cuda.stream(stream):
            k_gpu = torch.empty(tuple(k_cpu.shape), device=device, dtype=dtype)
            v_gpu = torch.empty(tuple(v_cpu.shape), device=device, dtype=dtype)
            k_gpu.copy_(k_cpu, non_blocking=True)
            v_gpu.copy_(v_cpu, non_blocking=True)
        torch.cuda.current_stream(device).wait_stream(stream)
        del k_cpu, v_cpu, k_gpu, v_gpu

    return cuda_elapsed_ms(once, repeats=repeats, warmup=warmup)


def measure_direct_stage_prefetch(
    *,
    buffer: RawKVOffloadBuffer,
    end: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    stream = torch.cuda.Stream(device=device)
    k_stage = torch.empty((end, kv_heads, head_dim), device=device, dtype=dtype)
    v_stage = torch.empty((end, kv_heads, head_dim), device=device, dtype=dtype)

    def once() -> None:
        with torch.cuda.stream(stream):
            buffer.copy_prefix_to(
                layer_idx=0,
                row_idx=0,
                kind="sparse_pre_rope",
                end=end,
                k_out=k_stage,
                v_out=v_stage,
            )
        torch.cuda.current_stream(device).wait_stream(stream)

    out = cuda_elapsed_ms(once, repeats=repeats, warmup=warmup)
    del k_stage, v_stage
    torch.cuda.empty_cache()
    return out


def measure_cpu_reassembly(
    *,
    buffer: RawKVOffloadBuffer,
    end: int,
    repeats: int,
    warmup: int,
) -> float:
    def once() -> None:
        k_cpu, v_cpu = legacy_get_prefix_cpu(buffer, layer_idx=0, row_idx=0, kind="sparse_pre_rope", end=end)
        del k_cpu, v_cpu

    return wall_elapsed_ms(once, repeats=repeats, warmup=warmup)


def measure_sync_direct_stage_miss(
    *,
    buffer: RawKVOffloadBuffer,
    end: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    k_stage = torch.empty((end, kv_heads, head_dim), device=device, dtype=dtype)
    v_stage = torch.empty((end, kv_heads, head_dim), device=device, dtype=dtype)

    def once() -> None:
        buffer.copy_prefix_to(
            layer_idx=0,
            row_idx=0,
            kind="sparse_pre_rope",
            end=end,
            k_out=k_stage,
            v_out=v_stage,
        )

    out = cuda_elapsed_ms(once, repeats=repeats, warmup=warmup)
    del k_stage, v_stage
    torch.cuda.empty_cache()
    return out


def measure_ideal_contiguous_h2d(
    *,
    end: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    shape = (end, kv_heads, head_dim)
    k_cpu = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    v_cpu = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    stream = torch.cuda.Stream(device=device)

    def once() -> None:
        with torch.cuda.stream(stream):
            k_gpu = torch.empty(shape, device=device, dtype=dtype)
            v_gpu = torch.empty(shape, device=device, dtype=dtype)
            k_gpu.copy_(k_cpu, non_blocking=True)
            v_gpu.copy_(v_cpu, non_blocking=True)
        torch.cuda.current_stream(device).wait_stream(stream)
        del k_gpu, v_gpu

    out = cuda_elapsed_ms(once, repeats=repeats, warmup=warmup)
    del k_cpu, v_cpu
    torch.cuda.empty_cache()
    return out


def measure_attention(
    *,
    history_n: int,
    prefill_chunk_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    kv_len = history_n + prefill_chunk_size
    q = torch.empty((1, prefill_chunk_size, q_heads, head_dim), dtype=dtype, device="cuda")
    k = torch.empty((1, kv_len, kv_heads, head_dim), dtype=dtype, device="cuda")
    v = torch.empty((1, kv_len, kv_heads, head_dim), dtype=dtype, device="cuda")

    def once() -> None:
        out = flash_attn_func(q, k, v, causal=True)
        del out

    out = cuda_elapsed_ms(once, repeats=repeats, warmup=warmup)
    del q, k, v
    torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure current RawKV chunked prefetch path costs.")
    parser.add_argument("--history-n", default="65536,294912,589824,900000")
    parser.add_argument("--prefill-chunk-sizes", default="32768,65536")
    parser.add_argument("--storage-chunk-size", type=int, default=65536)
    parser.add_argument("--q-heads", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    history_values = parse_int_list(args.history_n)
    prefill_chunks = parse_int_list(args.prefill_chunk_sizes)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for history_n in history_values:
        torch.cuda.empty_cache()
        buffer = build_chunked_buffer(
            total_len=history_n,
            storage_chunk_size=args.storage_chunk_size,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            pin_memory=True,
        )
        cpu_reassembly_ms = measure_cpu_reassembly(
            buffer=buffer,
            end=history_n,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        current_wall_ms, current_event_ms = measure_current_prefetch_schedule(
            buffer=buffer,
            end=history_n,
            dtype=dtype,
            device=device,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        sync_miss_wall_ms, sync_miss_event_ms = measure_sync_direct_stage_miss(
            buffer=buffer,
            end=history_n,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device=device,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        direct_stage_wall_ms, direct_stage_event_ms = measure_direct_stage_prefetch(
            buffer=buffer,
            end=history_n,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device=device,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        ideal_wall_ms, ideal_event_ms = measure_ideal_contiguous_h2d(
            end=history_n,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device=device,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        kv_gib = 2 * history_n * args.kv_heads * args.head_dim * torch.tensor([], dtype=dtype).element_size() / (1024**3)
        for prefill_chunk_size in prefill_chunks:
            attn_wall_ms, attn_event_ms = measure_attention(
                history_n=history_n,
                prefill_chunk_size=prefill_chunk_size,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            row = {
                "device": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "history_n": history_n,
                "prefill_chunk_size": prefill_chunk_size,
                "storage_chunk_size": args.storage_chunk_size,
                "kv_gib": kv_gib,
                "cpu_reassembly_ms": cpu_reassembly_ms,
                "current_prefetch_wall_ms": current_wall_ms,
                "current_prefetch_event_ms": current_event_ms,
                "sync_miss_direct_stage_wall_ms": sync_miss_wall_ms,
                "sync_miss_direct_stage_event_ms": sync_miss_event_ms,
                "direct_stage_prefetch_wall_ms": direct_stage_wall_ms,
                "direct_stage_prefetch_event_ms": direct_stage_event_ms,
                "ideal_contiguous_h2d_wall_ms": ideal_wall_ms,
                "ideal_contiguous_h2d_event_ms": ideal_event_ms,
                "attention_wall_ms": attn_wall_ms,
                "attention_event_ms": attn_event_ms,
                "current_prefetch_over_ideal": current_wall_ms / ideal_wall_ms if ideal_wall_ms > 0 else math.inf,
                "direct_stage_over_ideal": direct_stage_wall_ms / ideal_wall_ms if ideal_wall_ms > 0 else math.inf,
                "attention_over_current_prefetch": attn_wall_ms / current_wall_ms if current_wall_ms > 0 else math.inf,
                "attention_over_direct_stage": attn_wall_ms / direct_stage_wall_ms if direct_stage_wall_ms > 0 else math.inf,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        buffer.release_row(0)
        torch.cuda.empty_cache()

    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
