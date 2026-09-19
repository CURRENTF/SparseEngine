from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch

from sparseengine.platforms import device_runtime

from .offload.host_pool import HostTensorPool


@dataclass
class _RawKVEntry:
    k: torch.Tensor | None
    v: torch.Tensor | None
    filled_until: int = 0
    capacity: int = 0
    k_shape_tail: tuple[int, ...] = ()
    v_shape_tail: tuple[int, ...] = ()
    dtype: torch.dtype | None = None
    chunks: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    producer_events: dict[int, tuple[int, Any]] | None = None


class RawKVOffloadBuffer:
    """CPU backing store for chunked prefill raw KV.

    The buffer is intentionally shape-explicit: callers allocate one row/layer/kind
    with a known total prompt length, then write and restore contiguous ranges.
    """

    def __init__(self, *, pin_memory: bool = True, mode: str | None = None):
        self.pin_memory = bool(pin_memory)
        raw_mode = mode if mode is not None else os.getenv("SPARSEENGINE_RAWKV_BUFFER_MODE", "chunked")
        self.mode = str(raw_mode).strip().lower().replace("-", "_")
        if self.mode not in {"contiguous", "chunked"}:
            raise ValueError(
                "SPARSEENGINE_RAWKV_BUFFER_MODE must be 'contiguous' or 'chunked', "
                f"got {raw_mode!r}."
            )
        self._entries: dict[tuple[int, int, str], _RawKVEntry] = {}

    def ensure_entry(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        kind: str,
        total_len: int,
        k_shape_tail: tuple[int, ...],
        v_shape_tail: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        key = (int(layer_idx), int(row_idx), str(kind))
        total_len = int(total_len)
        if total_len < 0:
            raise ValueError(f"RawKVOffloadBuffer total_len must be non-negative, got {total_len}.")
        entry = self._entries.get(key)
        if entry is not None:
            if int(entry.capacity) < total_len:
                raise RuntimeError(
                    "RawKVOffloadBuffer cannot grow an existing entry: "
                    f"key={key} existing={int(entry.capacity)} requested={total_len}."
                )
            if tuple(entry.k_shape_tail) != tuple(k_shape_tail) or tuple(entry.v_shape_tail) != tuple(v_shape_tail):
                raise RuntimeError(
                    "RawKVOffloadBuffer existing entry shape tail mismatch: "
                    f"key={key} existing_k_tail={entry.k_shape_tail} requested_k_tail={k_shape_tail} "
                    f"existing_v_tail={entry.v_shape_tail} requested_v_tail={v_shape_tail}."
                )
            return
        if self.mode == "chunked":
            self._entries[key] = _RawKVEntry(
                k=None,
                v=None,
                capacity=total_len,
                k_shape_tail=tuple(k_shape_tail),
                v_shape_tail=tuple(v_shape_tail),
                dtype=dtype,
                chunks={},
                producer_events={},
            )
        else:
            k, v = HostTensorPool(
                ((total_len, *k_shape_tail), (total_len, *v_shape_tail)),
                dtype=dtype, pin_memory=self.pin_memory,
            ).tensors
            self._entries[key] = _RawKVEntry(
                k=k,
                v=v,
                capacity=total_len,
                k_shape_tail=tuple(k_shape_tail),
                v_shape_tail=tuple(v_shape_tail),
                dtype=dtype,
                producer_events={},
            )

    @staticmethod
    def _record_producer_event(
        entry: _RawKVEntry,
        *,
        start: int,
        end: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if entry.producer_events is None or (not k.is_cuda and not v.is_cuda):
            return
        device = k.device if k.is_cuda else v.device
        event = device_runtime.new_event(device=device)
        if event is None:
            raise RuntimeError(
                "RawKVOffloadBuffer needs a device event for asynchronous CUDA offload, "
                f"but the active platform does not support streams for device={device}."
            )
        device_runtime.record_event(event, device=device)
        entry.producer_events[int(start)] = (int(end), event)

    @staticmethod
    def _wait_for_producer_event(event: Any, dst: torch.Tensor) -> None:
        if dst.is_cuda:
            device_runtime.wait_event(event, device=dst.device)
        else:
            device_runtime.synchronize_event(event)

    @classmethod
    def _wait_for_range_producers(
        cls,
        entry: _RawKVEntry,
        *,
        start: int,
        end: int,
        dst: torch.Tensor,
    ) -> None:
        if entry.producer_events is None:
            return
        start = int(start)
        end = int(end)
        for event_start, (event_end, event) in sorted(entry.producer_events.items()):
            event_start = int(event_start)
            event_end = int(event_end)
            if event_start >= end:
                break
            if event_end <= start:
                continue
            cls._wait_for_producer_event(event, dst)

    @torch.no_grad()
    def put_range(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        kind: str,
        start: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        key = (int(layer_idx), int(row_idx), str(kind))
        entry = self._entries.get(key)
        if entry is None:
            raise RuntimeError(f"RawKVOffloadBuffer entry is missing for key={key}.")
        start = int(start)
        end = start + int(k.shape[0])
        if start < 0 or end > int(entry.capacity) or int(v.shape[0]) != int(k.shape[0]):
            raise RuntimeError(
                "RawKVOffloadBuffer put_range shape mismatch: "
                f"key={key} start={start} end={end} capacity={int(entry.capacity)} "
                f"k={tuple(k.shape)} v={tuple(v.shape)}."
            )
        if tuple(k.shape[1:]) != tuple(entry.k_shape_tail) or tuple(v.shape[1:]) != tuple(entry.v_shape_tail):
            raise RuntimeError(
                "RawKVOffloadBuffer put_range shape tail mismatch: "
                f"key={key} k={tuple(k.shape)} expected_tail={entry.k_shape_tail} "
                f"v={tuple(v.shape)} expected_tail={entry.v_shape_tail}."
            )
        if entry.chunks is not None:
            if start > int(entry.filled_until):
                raise RuntimeError(
                    "RawKVOffloadBuffer chunked put_range cannot leave a gap: "
                    f"key={key} start={start} filled_until={int(entry.filled_until)}."
                )
            k_cpu, v_cpu = HostTensorPool(
                (k.shape, v.shape), dtype=entry.dtype, pin_memory=self.pin_memory,
            ).tensors
            k_cpu.copy_(k.detach().to(dtype=entry.dtype), non_blocking=True)
            v_cpu.copy_(v.detach().to(dtype=entry.dtype), non_blocking=True)
            entry.chunks[start] = (k_cpu, v_cpu)
            self._record_producer_event(entry, start=start, end=end, k=k, v=v)
        else:
            if entry.k is None or entry.v is None:
                raise RuntimeError(f"RawKVOffloadBuffer contiguous entry is missing tensors for key={key}.")
            entry.k[start:end].copy_(k.detach().to(dtype=entry.k.dtype), non_blocking=True)
            entry.v[start:end].copy_(v.detach().to(dtype=entry.v.dtype), non_blocking=True)
            self._record_producer_event(entry, start=start, end=end, k=k, v=v)
        entry.filled_until = max(int(entry.filled_until), end)

    @torch.no_grad()
    def copy_prefix_to(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        kind: str,
        end: int,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
    ) -> None:
        key = (int(layer_idx), int(row_idx), str(kind))
        entry = self._entries.get(key)
        if entry is None:
            raise RuntimeError(f"RawKVOffloadBuffer entry is missing for key={key}.")
        end = int(end)
        if end < 0 or end > int(entry.filled_until):
            raise RuntimeError(
                "RawKVOffloadBuffer copy_prefix_to reads an unwritten range: "
                f"key={key} end={end} filled_until={int(entry.filled_until)}."
            )
        if int(k_out.shape[0]) < end or int(v_out.shape[0]) < end:
            raise RuntimeError(
                "RawKVOffloadBuffer copy_prefix_to destination is too short: "
                f"key={key} end={end} k_out={tuple(k_out.shape)} v_out={tuple(v_out.shape)}."
            )
        if tuple(k_out.shape[1:]) != tuple(entry.k_shape_tail) or tuple(v_out.shape[1:]) != tuple(entry.v_shape_tail):
            raise RuntimeError(
                "RawKVOffloadBuffer copy_prefix_to destination tail mismatch: "
                f"key={key} k_out={tuple(k_out.shape)} expected_k_tail={entry.k_shape_tail} "
                f"v_out={tuple(v_out.shape)} expected_v_tail={entry.v_shape_tail}."
            )
        if entry.chunks is not None:
            cursor = 0
            for chunk_start in sorted(entry.chunks):
                k_chunk, v_chunk = entry.chunks[chunk_start]
                chunk_start = int(chunk_start)
                chunk_end = chunk_start + int(k_chunk.shape[0])
                if chunk_start > cursor:
                    raise RuntimeError(
                        "RawKVOffloadBuffer chunked copy_prefix_to found a gap: "
                        f"key={key} cursor={cursor} next_start={chunk_start}."
                    )
                if chunk_end <= cursor:
                    continue
                copy_start = max(cursor, chunk_start)
                copy_end = min(end, chunk_end)
                if copy_start < copy_end:
                    src_start = copy_start - chunk_start
                    src_end = copy_end - chunk_start
                    self._wait_for_range_producers(
                        entry,
                        start=copy_start,
                        end=copy_end,
                        dst=k_out,
                    )
                    k_out[copy_start:copy_end].copy_(k_chunk[src_start:src_end], non_blocking=True)
                    v_out[copy_start:copy_end].copy_(v_chunk[src_start:src_end], non_blocking=True)
                    cursor = copy_end
                if cursor >= end:
                    break
            if cursor != end:
                raise RuntimeError(
                    "RawKVOffloadBuffer chunked copy_prefix_to did not fill requested prefix: "
                    f"key={key} cursor={cursor} end={end}."
                )
            return
        if entry.k is None or entry.v is None:
            raise RuntimeError(f"RawKVOffloadBuffer contiguous entry is missing tensors for key={key}.")
        self._wait_for_range_producers(entry, start=0, end=end, dst=k_out)
        k_out[:end].copy_(entry.k[:end], non_blocking=True)
        v_out[:end].copy_(entry.v[:end], non_blocking=True)

    def release_layer(self, *, layer_idx: int, row_idx: int, kind: str | None = None) -> None:
        prefix = (int(layer_idx), int(row_idx))
        for key in list(self._entries):
            if key[:2] == prefix and (kind is None or key[2] == str(kind)):
                del self._entries[key]

    def release_row(self, row_idx: int) -> None:
        row_idx = int(row_idx)
        for key in list(self._entries):
            if key[1] == row_idx:
                del self._entries[key]
