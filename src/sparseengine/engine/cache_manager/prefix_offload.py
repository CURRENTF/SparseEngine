from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sparseengine.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex
from sparseengine.platforms import device_runtime


def _load_kvcache_transfer_ops():
    try:
        from sgl_kernel.kvcacheio import (
            transfer_kv_all_layer,
            transfer_kv_per_layer,
        )
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Prefix cache offload requires the installed sgl_kernel.kvcacheio "
            "transfer_kv_all_layer and transfer_kv_per_layer kernels."
        ) from exc
    if not callable(transfer_kv_all_layer) or not callable(transfer_kv_per_layer):
        raise RuntimeError("sgl_kernel.kvcacheio does not expose the required transfer kernels.")
    return transfer_kv_all_layer, transfer_kv_per_layer


def _payload_device_slots(block: PrefixCacheBlock, block_size: int) -> torch.Tensor:
    payload = getattr(block.payload, "kv_payload", block.payload)
    slots = getattr(payload, "token_slots", None)
    retained_offsets = getattr(payload, "retained_offsets", None)
    expected = int(block_size) if retained_offsets is None else len(retained_offsets)
    if not isinstance(slots, torch.Tensor) or int(slots.numel()) != expected:
        raise RuntimeError(
            "Prefix offload block has inconsistent device-slot payload: "
            f"block={block.stable_block_id.hex()[:16]} expected_slots={expected}."
        )
    return slots.reshape(-1)


def _payload_retained_offsets(block: PrefixCacheBlock, block_size: int) -> tuple[int, ...] | None:
    payload = getattr(block.payload, "kv_payload", block.payload)
    offsets = getattr(payload, "retained_offsets", None)
    if offsets is None:
        return None
    normalized = tuple(int(offset) for offset in offsets)
    if any(offset < 0 or offset >= int(block_size) for offset in normalized):
        raise RuntimeError("Prefix block contains invalid retained token offsets.")
    return normalized


def _payload_host_index(block: PrefixCacheBlock) -> int:
    index = getattr(block.payload, "host_block_index", None)
    if index is None:
        raise RuntimeError(
            "Prefix offload block is missing its host block index: "
            f"block={block.stable_block_id.hex()[:16]}."
        )
    return int(index)


def _payload_device_page(block: PrefixCacheBlock) -> int:
    payload = getattr(block.payload, "kv_payload", block.payload)
    page = getattr(payload, "block_slot", None)
    if page is None:
        raise RuntimeError(
            "QuEST prefix offload block is missing its device page slot: "
            f"block={block.stable_block_id.hex()[:16]}."
        )
    return int(page)


class PinnedPrefixBlockPool:
    """Host block indices shared by concrete physical host pools."""

    def __init__(self, *, capacity_blocks, num_layers, block_size):
        if capacity_blocks <= 0 or num_layers <= 0 or block_size <= 0:
            raise ValueError("Prefix pool dimensions must be positive.")
        self.capacity_blocks = int(capacity_blocks)
        self.num_layers = int(num_layers)
        self.block_size = int(block_size)
        self._free_indices = list(range(self.capacity_blocks - 1, -1, -1))
        self._allocated: set[int] = set()

    @property
    def free_blocks(self) -> int:
        return len(self._free_indices)

    @property
    def used_blocks(self) -> int:
        return len(self._allocated)

    def allocate(self, count: int) -> list[int]:
        count = int(count)
        if count <= 0:
            raise ValueError(f"Host prefix allocation count must be positive, got {count}.")
        if count > self.free_blocks:
            raise RuntimeError(
                "Prefix host pool is out of blocks: "
                f"need={count} free={self.free_blocks} capacity={self.capacity_blocks}."
            )
        indices = [self._free_indices.pop() for _ in range(count)]
        self._allocated.update(indices)
        return indices

    def free(self, indices: list[int]) -> None:
        normalized = [int(index) for index in indices]
        if len(set(normalized)) != len(normalized):
            raise RuntimeError("Prefix host free contains duplicate block indices.")
        missing = [index for index in normalized if index not in self._allocated]
        if missing:
            raise RuntimeError(f"Prefix host free contains unallocated indices: {missing[:8]}.")
        for index in normalized:
            self._allocated.remove(index)
            self._free_indices.append(index)

    def token_indices(self, block_indices: list[int], device: torch.device) -> torch.Tensor:
        # Transfer one ID per block; expand full-page offsets on the device.
        bases = torch.tensor(block_indices, dtype=torch.long, device=device) * self.block_size
        offsets = torch.arange(self.block_size, dtype=torch.long, device=device)
        return (bases[:, None] + offsets[None, :]).reshape(-1)

    def retained_token_indices(
        self,
        block_indices: list[int],
        retained_offsets: list[tuple[int, ...] | None],
        device: torch.device,
    ) -> torch.Tensor:
        if len(block_indices) != len(retained_offsets):
            raise ValueError("Host block indices and retained offsets must have equal length.")
        if all(offsets is None for offsets in retained_offsets):
            return self.token_indices(block_indices, device)
        # Ragged compacted blocks retain their explicit offset order. Expand
        # full blocks in native CPU loops and transfer the packed vector once.
        full_offsets = np.arange(self.block_size, dtype=np.int64)
        parts = [
            int(block_index) * self.block_size
            + (full_offsets if offsets is None else np.asarray(offsets, dtype=np.int64))
            for block_index, offsets in zip(block_indices, retained_offsets)
        ]
        return torch.from_numpy(np.concatenate(parts)).to(device=device)

    def reset(self) -> None:
        self._allocated.clear()
        self._free_indices = list(range(self.capacity_blocks - 1, -1, -1))


class PinnedPrefixKVPool(PinnedPrefixBlockPool):
    """Fixed-capacity pinned host storage, indexed in logical prefix blocks."""

    def __init__(
        self,
        *,
        capacity_blocks: int,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        page_size: int | None = None,
    ) -> None:
        if int(capacity_blocks) <= 0:
            raise ValueError(
                f"Pinned prefix host capacity must be positive, got {capacity_blocks}."
            )
        super().__init__(
            capacity_blocks=capacity_blocks, num_layers=num_layers, block_size=block_size,
        )
        if not device_runtime.supports_pin_memory():
            raise RuntimeError(
                "Prefix cache offload requires pinned host memory, but the active platform "
                "does not support it."
            )
        self.cache = torch.empty(
            (
                2,
                self.num_layers,
                self.capacity_blocks,
                self.block_size,
                int(num_kv_heads),
                int(head_dim),
            ),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        if page_size is not None:
            if self.block_size % page_size != 0:
                raise ValueError("QuEST prefix block size must contain whole pages.")
            self.pages_per_block = self.block_size // page_size
            self.metadata_cache = torch.empty(
                (2, self.num_layers, self.capacity_blocks * self.pages_per_block, num_kv_heads, head_dim),
                dtype=dtype, device="cpu", pin_memory=True,
            )


@dataclass
class PrefixD2HOperation:
    blocks: list[PrefixCacheBlock]
    host_indices: list[int]
    device_token_indices: torch.Tensor
    host_token_indices: torch.Tensor
    producer_event: Any
    completion_event: Any
    byte_count: int
    auxiliary_tensors: tuple[torch.Tensor, ...] = ()


@dataclass
class PrefixH2DOperation:
    blocks: list[PrefixCacheBlock]
    host_token_indices: torch.Tensor
    device_token_indices: torch.Tensor
    producer_event: Any
    layer_events: list[Any]
    completion_event: Any
    byte_count: int
    auxiliary_tensors: tuple[torch.Tensor, ...] = ()
    auxiliary_event: Any | None = None
    auxiliary_layer_events: dict[int, Any] | None = None


class PrefixOffloadController:
    """Prefix transfer lifecycle, independent of physical payload and copy backend."""

    def __init__(self, *, prefix_cache, host_pool, block_size, device):
        self.prefix_cache = prefix_cache
        self.host_pool = host_pool
        self.block_size = int(block_size)
        self.device = device
        self._init_transfer_runtime()

    def _init_transfer_runtime(self) -> None:
        device = self.device
        self.d2h_stream = device_runtime.new_stream(device=device)
        self.h2d_stream = device_runtime.new_stream(device=device)
        if self.d2h_stream is None or self.h2d_stream is None:
            raise RuntimeError(
                f"Prefix cache offload could not create transfer streams for device={device}."
            )
        self.d2h_operations: deque[PrefixD2HOperation] = deque()
        self.h2d_operations: deque[PrefixH2DOperation] = deque()
        self._h2d_by_block_id: dict[bytes, PrefixH2DOperation] = {}
        self.d2h_bytes = 0
        self.h2d_bytes = 0
        self.d2h_submitted_operations = 0
        self.d2h_completed_operations = 0
        self.h2d_submitted_operations = 0
        self.h2d_completed_operations = 0
        self.d2h_merged_blocks = 0
        self.h2d_merged_blocks = 0
        self.layer_waits = 0

    @staticmethod
    def _new_event(device: torch.device, purpose: str) -> Any:
        event = device_runtime.new_event(device=device)
        if event is None:
            raise RuntimeError(
                f"Prefix cache offload could not create {purpose} event for device={device}."
            )
        return event

    def _host_token_indices(
        self,
        blocks: list[PrefixCacheBlock],
        host_indices: list[int],
    ) -> torch.Tensor:
        offsets = [
            _payload_retained_offsets(block, self.block_size) for block in blocks
        ]
        return self.host_pool.retained_token_indices(
            host_indices,
            offsets,
            self.device,
        )

    @torch.no_grad()
    def submit_d2h(self, blocks: list[PrefixCacheBlock]) -> None:
        if not blocks:
            return
        if device_runtime.is_stream_capturing():
            raise RuntimeError("Prefix D2H submission is forbidden during graph capture.")
        transfer_positions = []
        slot_parts = []
        for position, block in enumerate(blocks):
            slots = _payload_device_slots(block, self.block_size)
            if slots.numel():
                transfer_positions.append(position)
                slot_parts.append(slots)
        host_indices = self.host_pool.allocate(len(blocks))
        begun: list[PrefixCacheBlock] = []
        if not slot_parts:
            try:
                for block in blocks:
                    self.prefix_cache.begin_d2h(block)
                    begun.append(block)
            except Exception:
                for block in reversed(begun):
                    self.prefix_cache.abort_d2h(block)
                self.host_pool.free(host_indices)
                raise
            for block, host_index in zip(blocks, host_indices):
                setattr(block.payload, "host_block_index", int(host_index))
                self.prefix_cache.finish_d2h(block)
            return
        try:
            device_slots = torch.cat(slot_parts, dim=0).to(device=self.device, dtype=torch.long)
            transfer_blocks = [blocks[i] for i in transfer_positions]
            transfer_host_indices = [host_indices[i] for i in transfer_positions]
            host_token_indices = self._host_token_indices(transfer_blocks, transfer_host_indices)
            auxiliary_tensors = self._prepare_d2h_auxiliary(transfer_blocks, transfer_host_indices)
            events = self._d2h_events()
            for block in blocks:
                self.prefix_cache.begin_d2h(block)
                begun.append(block)
            self._copy_d2h(device_slots, host_token_indices, auxiliary_tensors, events)
        except Exception:
            device_runtime.synchronize_stream(self.d2h_stream)
            for block in reversed(begun):
                self.prefix_cache.abort_d2h(block)
            self.host_pool.free(host_indices)
            raise

        self._record_d2h_operation(
            blocks, host_indices, device_slots, host_token_indices,
            auxiliary_tensors, events,
            self._transfer_token_byte_count(int(device_slots.numel())),
            len(transfer_positions),
        )

    def _d2h_events(self) -> tuple[Any, Any]:
        return (
            self._new_event(self.device, "D2H producer"),
            self._new_event(self.device, "D2H completion"),
        )

    def _copy_d2h(self, device_slots, host_token_indices, auxiliary_tensors, events) -> None:
        producer_event, completion_event = events
        device_runtime.record_event(producer_event, device=self.device)
        with device_runtime.stream_context(self.d2h_stream):
            device_runtime.stream_wait_event(self.d2h_stream, producer_event)
            self._submit_d2h_payload(device_slots, host_token_indices, auxiliary_tensors)
            device_runtime.record_event(completion_event, device=self.device)

    def _record_d2h_operation(
        self, blocks, host_indices, device_slots, host_token_indices,
        auxiliary_tensors, events, byte_count, transferred_blocks,
    ) -> None:
        producer_event, completion_event = events
        self.d2h_operations.append(
            PrefixD2HOperation(
                blocks=list(blocks),
                host_indices=host_indices,
                device_token_indices=device_slots,
                host_token_indices=host_token_indices,
                producer_event=producer_event,
                completion_event=completion_event,
                byte_count=byte_count,
                auxiliary_tensors=auxiliary_tensors,
            )
        )
        self.d2h_bytes += byte_count
        self.d2h_submitted_operations += 1
        self.d2h_merged_blocks += transferred_blocks

    def _finish_d2h(self, operation: PrefixD2HOperation) -> None:
        for block, host_index in zip(operation.blocks, operation.host_indices):
            setattr(block.payload, "host_block_index", int(host_index))
            self.prefix_cache.finish_d2h(block)
        self.d2h_completed_operations += 1

    def poll_d2h(self) -> int:
        completed = 0
        while self.d2h_operations:
            operation = self.d2h_operations[0]
            if not device_runtime.is_event_complete(operation.completion_event):
                break
            self.d2h_operations.popleft()
            self._finish_d2h(operation)
            completed += 1
        return completed

    def wait_oldest_d2h(self) -> bool:
        if not self.d2h_operations:
            return False
        device_runtime.synchronize_event(self.d2h_operations[0].completion_event)
        self.poll_d2h()
        return True

    @torch.no_grad()
    def submit_h2d(self, blocks: list[PrefixCacheBlock]) -> PrefixH2DOperation | None:
        if not blocks:
            raise ValueError("Prefix H2D submission requires at least one block.")
        if device_runtime.is_stream_capturing():
            raise RuntimeError("Prefix H2D submission is forbidden during graph capture.")
        promotion_blocks = blocks
        empty_blocks = []
        blocks = []
        slot_parts = []
        for block in promotion_blocks:
            slots = _payload_device_slots(block, self.block_size)
            if slots.numel() == 0:
                empty_blocks.append(block)
            else:
                blocks.append(block)
                slot_parts.append(slots)
        begun: list[PrefixCacheBlock] = []
        if not blocks:
            try:
                for block in promotion_blocks:
                    self.prefix_cache.begin_h2d(block)
                    begun.append(block)
            except Exception:
                for block in reversed(begun):
                    self.prefix_cache.abort_h2d(block)
                raise
            for block in empty_blocks:
                self.prefix_cache.finish_h2d(block)
            return None
        # Zero-token blocks still participate in the root-to-leaf residency
        # transaction; only their physical payload is omitted from the transfer.
        host_indices = [_payload_host_index(block) for block in blocks]
        host_token_indices = self._host_token_indices(blocks, host_indices)
        device_slots = torch.cat(slot_parts, dim=0).to(device=self.device, dtype=torch.long)
        auxiliary_tensors = self._prepare_h2d_auxiliary(blocks, host_indices)
        try:
            layer_events = [
                self._new_event(self.device, f"H2D layer {layer_index}")
                for layer_index in range(self.host_pool.num_layers)
            ]
            producer_event = self._new_event(self.device, "H2D producer")
            completion_event = self._new_event(self.device, "H2D completion")
            for block in promotion_blocks:
                self.prefix_cache.begin_h2d(block)
                begun.append(block)
            device_runtime.record_event(producer_event, device=self.device)
            with device_runtime.stream_context(self.h2d_stream):
                device_runtime.stream_wait_event(self.h2d_stream, producer_event)
                auxiliary_layer_events = {}
                for kind, layer_index in self._h2d_transfer_schedule():
                    if kind == "kv":
                        self._submit_h2d_layer(
                            layer_index,
                            host_token_indices,
                            device_slots,
                            auxiliary_tensors,
                        )
                        device_runtime.record_event(
                            layer_events[layer_index], device=self.device
                        )
                    elif kind == "auxiliary":
                        self._submit_h2d_auxiliary_layer(layer_index, auxiliary_tensors)
                        event = self._new_event(
                            self.device, f"H2D auxiliary layer {layer_index}"
                        )
                        device_runtime.record_event(event, device=self.device)
                        auxiliary_layer_events[int(layer_index)] = event
                    else:
                        raise RuntimeError(f"Unknown prefix H2D transfer kind={kind!r}.")
                auxiliary_event = None
                device_runtime.record_event(completion_event, device=self.device)
        except Exception:
            device_runtime.synchronize_stream(self.h2d_stream)
            for block in reversed(begun):
                self.prefix_cache.abort_h2d(block)
            raise

        byte_count = self._h2d_token_byte_count(int(device_slots.numel()))
        operation = PrefixH2DOperation(
            blocks=list(blocks),
            host_token_indices=host_token_indices,
            device_token_indices=device_slots,
            producer_event=producer_event,
            layer_events=layer_events,
            completion_event=completion_event,
            byte_count=byte_count,
            auxiliary_tensors=auxiliary_tensors,
            auxiliary_event=auxiliary_event,
            auxiliary_layer_events=auxiliary_layer_events,
        )
        self.h2d_operations.append(operation)
        for block in blocks:
            self._h2d_by_block_id[block.stable_block_id] = operation
        self.h2d_bytes += byte_count
        self.h2d_submitted_operations += 1
        self.h2d_merged_blocks += len(blocks)
        for block in empty_blocks:
            self.prefix_cache.finish_h2d(block)
        return operation

    def _prepare_d2h_auxiliary(
        self,
        blocks: list[PrefixCacheBlock],
        host_indices: list[int],
    ) -> tuple[torch.Tensor, ...]:
        del blocks, host_indices
        return ()

    def _prepare_h2d_auxiliary(
        self,
        blocks: list[PrefixCacheBlock],
        host_indices: list[int],
    ) -> tuple[torch.Tensor, ...]:
        del blocks, host_indices
        return ()

    def _submit_d2h_payload(
        self,
        device_slots: torch.Tensor,
        host_token_indices: torch.Tensor,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        raise NotImplementedError

    def _submit_h2d_layer(
        self,
        layer_index: int,
        host_token_indices: torch.Tensor,
        device_slots: torch.Tensor,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        raise NotImplementedError

    def _h2d_transfer_schedule(self) -> list[tuple[str, int]]:
        return [("kv", layer_idx) for layer_idx in range(self.host_pool.num_layers)]

    def _submit_h2d_auxiliary_layer(
        self,
        layer_index: int,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        del layer_index, auxiliary_tensors
        raise RuntimeError("This prefix offload controller has no auxiliary layers.")

    def _h2d_token_byte_count(self, token_count: int) -> int:
        return self._transfer_token_byte_count(token_count)

    def _transfer_byte_count(self, block_count: int) -> int:
        return self._transfer_token_byte_count(int(block_count) * self.block_size)

    def _transfer_token_byte_count(self, token_count: int) -> int:
        raise NotImplementedError

    def h2d_operation_for_block(self, block: PrefixCacheBlock) -> PrefixH2DOperation | None:
        return self._h2d_by_block_id.get(block.stable_block_id)

    def poll_h2d(self) -> int:
        completed = 0
        # Every completion event is recorded on the same H2D stream. A pending
        # head also bounds the readiness of every later operation.
        while self.h2d_operations:
            operation = self.h2d_operations[0]
            if not device_runtime.is_event_complete(operation.completion_event):
                break
            self.h2d_operations.popleft()
            for block in operation.blocks:
                self.prefix_cache.finish_h2d(block)
                self._h2d_by_block_id.pop(block.stable_block_id, None)
            self.h2d_completed_operations += 1
            completed += 1
        return completed

    def poll(self) -> tuple[int, int]:
        return self.poll_d2h(), self.poll_h2d()

    def wait_for_layer(self, operation: PrefixH2DOperation, layer_index: int) -> None:
        layer_index = int(layer_index)
        if layer_index < 0 or layer_index >= len(operation.layer_events):
            raise RuntimeError(
                "Prefix H2D layer event is missing: "
                f"layer={layer_index} events={len(operation.layer_events)}."
            )
        device_runtime.wait_event(operation.layer_events[layer_index], device=self.device)
        self.layer_waits += 1

    def wait_for_auxiliary(self, operation: PrefixH2DOperation) -> None:
        events = operation.auxiliary_layer_events or {}
        if not events:
            raise RuntimeError("Prefix H2D operation has no auxiliary readiness events.")
        for event in events.values():
            device_runtime.wait_event(event, device=self.device)

    def free_host_payloads(self, blocks: list[PrefixCacheBlock]) -> None:
        indices: list[int] = []
        for block in blocks:
            if not block.residency.host_present:
                continue
            indices.append(_payload_host_index(block))
            setattr(block.payload, "host_block_index", None)
        if indices:
            self.host_pool.free(indices)

    def synchronize_all(self) -> None:
        for operation in self.d2h_operations:
            device_runtime.synchronize_event(operation.completion_event)
        self.poll_d2h()
        for operation in self.h2d_operations:
            device_runtime.synchronize_event(operation.completion_event)
        self.poll_h2d()

    def reset(self) -> None:
        self.synchronize_all()
        self.d2h_operations.clear()
        self.h2d_operations.clear()
        self._h2d_by_block_id.clear()
        self.host_pool.reset()
        self.d2h_bytes = 0
        self.h2d_bytes = 0
        self.d2h_submitted_operations = 0
        self.d2h_completed_operations = 0
        self.h2d_submitted_operations = 0
        self.h2d_completed_operations = 0
        self.d2h_merged_blocks = 0
        self.h2d_merged_blocks = 0
        self.layer_waits = 0

    def stats(self) -> dict[str, int]:
        return {
            "prefix_cache_host_capacity_blocks": int(self.host_pool.capacity_blocks),
            "prefix_cache_host_used_blocks": int(self.host_pool.used_blocks),
            "prefix_cache_host_free_blocks": int(self.host_pool.free_blocks),
            "prefix_cache_d2h_bytes": int(self.d2h_bytes),
            "prefix_cache_h2d_bytes": int(self.h2d_bytes),
            "prefix_cache_d2h_submitted_operations": int(self.d2h_submitted_operations),
            "prefix_cache_d2h_completed_operations": int(self.d2h_completed_operations),
            "prefix_cache_h2d_submitted_operations": int(self.h2d_submitted_operations),
            "prefix_cache_h2d_completed_operations": int(self.h2d_completed_operations),
            "prefix_cache_d2h_merged_blocks": int(self.d2h_merged_blocks),
            "prefix_cache_h2d_merged_blocks": int(self.h2d_merged_blocks),
            "prefix_cache_h2d_layer_waits": int(self.layer_waits),
            "prefix_cache_d2h_inflight_operations": int(len(self.d2h_operations)),
            "prefix_cache_h2d_inflight_operations": int(len(self.h2d_operations)),
        }


class StandardPrefixOffloadController(PrefixOffloadController):
    """Asynchronous write-through transfers for Standard/OmniKV prefix blocks."""

    def __init__(
        self,
        *,
        prefix_cache: RadixPrefixIndex,
        kv_cache: torch.Tensor,
        host_pool: PinnedPrefixKVPool,
        block_size: int,
        device: torch.device,
        device_metadata_cache: torch.Tensor | None = None,
    ) -> None:
        if not device_runtime.supports_streams(device):
            raise RuntimeError(
                "Prefix cache offload requires asynchronous device streams; "
                f"device={device}."
            )
        self.prefix_cache = prefix_cache
        self.kv_cache = kv_cache
        self.host_pool = host_pool
        self.block_size = int(block_size)
        self.device = device
        self.item_size = int(
            self.kv_cache.shape[-2]
            * self.kv_cache.shape[-1]
            * self.kv_cache.element_size()
        )
        if self.item_size <= 0 or self.item_size % 8 != 0:
            raise RuntimeError(
                "sgl_kernel prefix transfer item size must be positive and divisible by 8: "
                f"item_size={self.item_size}."
            )
        if not self.kv_cache.is_contiguous() or not self.host_pool.cache.is_contiguous():
            raise RuntimeError("Prefix transfer requires contiguous layer-first KV pools.")
        if not self.host_pool.cache.is_pinned():
            raise RuntimeError("Prefix transfer host KV pool must be pinned.")
        if (
            tuple(self.kv_cache.shape[:2]) != (2, self.host_pool.num_layers)
            or tuple(self.kv_cache.shape[-2:]) != tuple(self.host_pool.cache.shape[-2:])
        ):
            raise RuntimeError(
                "Prefix device and host KV pool shapes are incompatible: "
                f"device={tuple(self.kv_cache.shape)} host={tuple(self.host_pool.cache.shape)}."
            )
        self._transfer_all_layers, self._transfer_per_layer = _load_kvcache_transfer_ops()
        self.device_k_ptrs = torch.tensor(
            [self.kv_cache[0, layer].data_ptr() for layer in range(self.host_pool.num_layers)],
            dtype=torch.uint64,
            device=self.device,
        )
        self.device_v_ptrs = torch.tensor(
            [self.kv_cache[1, layer].data_ptr() for layer in range(self.host_pool.num_layers)],
            dtype=torch.uint64,
            device=self.device,
        )
        self.host_k_ptrs = torch.tensor(
            [self.host_pool.cache[0, layer].data_ptr() for layer in range(self.host_pool.num_layers)],
            dtype=torch.uint64,
            device=self.device,
        )
        self.host_v_ptrs = torch.tensor(
            [self.host_pool.cache[1, layer].data_ptr() for layer in range(self.host_pool.num_layers)],
            dtype=torch.uint64,
            device=self.device,
        )
        self._init_transfer_runtime()
        self._quest_metadata = (
            QuestMetadataTransfer(self, device_metadata_cache)
            if device_metadata_cache is not None else None
        )

    def _prepare_d2h_auxiliary(self, blocks, host_indices):
        if self._quest_metadata is None:
            return ()
        return self._quest_metadata.page_indices(blocks, host_indices)

    def _prepare_h2d_auxiliary(self, blocks, host_indices):
        if self._quest_metadata is None:
            return ()
        return self._quest_metadata.page_indices(blocks, host_indices)


    def _submit_d2h_payload(
        self,
        device_slots: torch.Tensor,
        host_token_indices: torch.Tensor,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        self._transfer_all_layers(
            src_k_layers=self.device_k_ptrs,
            dst_k_layers=self.host_k_ptrs,
            src_v_layers=self.device_v_ptrs,
            dst_v_layers=self.host_v_ptrs,
            src_indices=device_slots,
            dst_indices=host_token_indices,
            item_size=self.item_size,
            num_layers=self.host_pool.num_layers,
        )
        if self._quest_metadata is not None:
            self._quest_metadata.copy_to_host(*auxiliary_tensors[:2])

    def _submit_h2d_layer(
        self,
        layer_index: int,
        host_token_indices: torch.Tensor,
        device_slots: torch.Tensor,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        self._transfer_per_layer(
            src_k=self.host_pool.cache[0, layer_index],
            dst_k=self.kv_cache[0, layer_index],
            src_v=self.host_pool.cache[1, layer_index],
            dst_v=self.kv_cache[1, layer_index],
            src_indices=host_token_indices,
            dst_indices=device_slots,
            item_size=self.item_size,
        )
        if self._quest_metadata is not None:
            self._quest_metadata.copy_to_device(layer_index, *auxiliary_tensors[:2])

    def _transfer_token_byte_count(self, token_count: int) -> int:
        metadata_pages = (
            token_count // self.block_size * self.host_pool.pages_per_block
            if self._quest_metadata is not None else 0
        )
        return (token_count + metadata_pages) * self.host_pool.num_layers * 2 * self.item_size


class QuestMetadataTransfer:
    """Copy min/max page summaries in the same transfer as their KV tokens."""

    def __init__(self, controller: StandardPrefixOffloadController, device_cache: torch.Tensor):
        self._transfer_all_layers = controller._transfer_all_layers
        self._transfer_per_layer = controller._transfer_per_layer
        self.item_size = controller.item_size
        self.device_cache = device_cache
        self.host_cache = controller.host_pool.metadata_cache
        self.pages_per_block = controller.host_pool.pages_per_block
        self.num_layers = controller.host_pool.num_layers
        self.device = controller.device
        if (
            not device_cache.is_contiguous()
            or not self.host_cache.is_contiguous()
            or not self.host_cache.is_pinned()
        ):
            raise RuntimeError("QuEST metadata transfer requires contiguous pinned host metadata.")
        if (
            tuple(device_cache.shape[:2]) != (2, self.num_layers)
            or tuple(device_cache.shape[-2:]) != tuple(self.host_cache.shape[-2:])
        ):
            raise RuntimeError(
                "QuEST device and host metadata shapes are incompatible: "
                f"device={tuple(device_cache.shape)} host={tuple(self.host_cache.shape)}."
            )
        item_size = device_cache.shape[-2] * device_cache.shape[-1] * device_cache.element_size()
        if item_size != controller.item_size:
            raise RuntimeError(
                "QuEST KV-token and metadata item sizes must match for shared transfer kernels: "
                f"kv={controller.item_size} metadata={item_size}."
            )
        self.device_max, self.device_min = self._pointers(self.device_cache)
        self.host_max, self.host_min = self._pointers(self.host_cache)

    def _pointers(self, cache: torch.Tensor):
        return tuple(
            torch.tensor(
                [layer.data_ptr() for layer in component],
                dtype=torch.uint64, device=self.device,
            )
            for component in cache
        )

    def page_indices(self, blocks, host_indices) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(blocks[0].payload, "kv_payload", None) is None:
            device_pages = torch.tensor(
                [_payload_device_page(block) for block in blocks],
                dtype=torch.long, device=self.device,
            )
        else:
            page_parts = []
            for block in blocks:
                slots = getattr(block.payload.kv_payload, "block_slots", None)
                if slots is None and self.pages_per_block == 1:
                    slots = torch.tensor([_payload_device_page(block)], device=self.device)
                if slots is None or slots.numel() != self.pages_per_block:
                    raise RuntimeError(
                        "Mixed QuEST device page count does not match prefix block geometry."
                    )
                page_parts.append(slots.to(device=self.device, dtype=torch.long).reshape(-1))
            device_pages = torch.cat(page_parts)
        host_pages = torch.tensor(
            [idx * self.pages_per_block + offset
             for idx in host_indices for offset in range(self.pages_per_block)],
            dtype=torch.long, device=self.device,
        )
        return device_pages, host_pages

    def copy_to_host(self, device_pages, host_pages) -> None:
        self._transfer_all_layers(
            src_k_layers=self.device_max, dst_k_layers=self.host_max,
            src_v_layers=self.device_min, dst_v_layers=self.host_min,
            src_indices=device_pages, dst_indices=host_pages,
            item_size=self.item_size, num_layers=self.num_layers,
        )

    def copy_to_device(self, layer_index, device_pages, host_pages) -> None:
        self._transfer_per_layer(
            src_k=self.host_cache[0, layer_index], dst_k=self.device_cache[0, layer_index],
            src_v=self.host_cache[1, layer_index], dst_v=self.device_cache[1, layer_index],
            src_indices=host_pages, dst_indices=device_pages,
            item_size=self.item_size,
        )
