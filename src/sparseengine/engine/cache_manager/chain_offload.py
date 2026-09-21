"""Whole-turn chain snapshots. Physical state remains owned by CacheManager."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

from sparseengine.platforms import device_runtime

from .offload.host_pool import HostTensorPool
from .prefix_offload import _load_kvcache_transfer_ops


def _load_single_transfer():
    try:
        from sgl_kernel.kvcacheio import transfer_kv_per_layer_mla
    except (ImportError, OSError) as exc:
        raise RuntimeError("Chain latent offload requires sgl_kernel.kvcacheio.transfer_kv_per_layer_mla.") from exc
    if not callable(transfer_kv_per_layer_mla):
        raise RuntimeError("The installed SGL kernel lacks the chain latent transfer API.")
    return transfer_kv_per_layer_mla


@dataclass
class ChainMethodState:
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Any = None


@dataclass
class ChainSnapshot:
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]]
    method: ChainMethodState
    nbytes: int
    valid: bool = False
    completion: Any = None
    keepalive: list[Any] = field(default_factory=list)


class ChainOffloadController:
    def __init__(self, manager, capacity_bytes: int):
        self.manager = manager
        self.device = manager.device
        self.capacity_bytes = int(capacity_bytes)
        if self.capacity_bytes <= 0:
            raise ValueError("Chain offload requires a positive host byte budget.")
        if not device_runtime.supports_pin_memory() or not device_runtime.supports_streams(self.device):
            raise RuntimeError("Chain offload requires pinned memory and asynchronous device streams.")
        if manager.num_kv_layers != manager.num_layers:
            raise ValueError("Chain offload does not yet snapshot recurrent/linear layer state.")
        for layer in manager.kv_transformer_layer_indices():
            for tensor in manager.chain_storage_tensors(layer):
                if tensor.ndim != 3 or not tensor.is_contiguous():
                    raise ValueError("Chain offload requires contiguous token storage.")
                if tensor[0].numel() * tensor.element_size() % 8:
                    raise ValueError("Chain offload requires each token size to be divisible by 8 bytes.")
        _, self.transfer = _load_kvcache_transfer_ops()
        self.single_transfer = None
        if any(a.shape != b.shape or a.dtype != b.dtype for a, b in
               (manager.chain_storage_tensors(layer) for layer in manager.kv_transformer_layer_indices())):
            self.single_transfer = _load_single_transfer()
        self.stream = device_runtime.new_stream(self.device)
        if self.stream is None:
            raise RuntimeError("Cannot create chain offload stream.")
        self.snapshots: dict[int, ChainSnapshot] = {}
        # Transfer queue only. Completed host snapshots stay in snapshots and
        # must not make admission polling proportional to retained history.
        self._pending: OrderedDict[int, ChainSnapshot] = OrderedDict()
        self.used_bytes = 0
        self.d2h_bytes = 0
        self.h2d_bytes = 0

    def _event(self):
        event = device_runtime.new_event(self.device)
        if event is None:
            raise RuntimeError("Cannot create chain offload event.")
        return event

    def _transfer(self, source, destination, source_slots, destination_slots):
        if source[0].shape[1:] == source[1].shape[1:] and source[0].dtype == source[1].dtype:
            self.transfer(source[0], destination[0], source[1], destination[1],
                          source_slots, destination_slots, source[0][0].numel() * source[0].element_size())
        else:
            for src, dst in zip(source, destination):
                self.single_transfer(src, dst, source_slots, destination_slots,
                                     src[0].numel() * src.element_size())

    def required_bytes(self, seq_id: int, state: ChainMethodState) -> int:
        lengths = self.manager.chain_physical_residency(seq_id)
        size = sum(t.numel() * t.element_size() for t in state.tensors.values())
        for layer, length in zip(self.manager.kv_transformer_layer_indices(), lengths):
            k, v = self.manager.chain_storage_tensors(layer)
            size += length * (k[0].numel() * k.element_size() + v[0].numel() * v.element_size())
        return int(size)

    def wait(self, seq_id: int) -> ChainSnapshot:
        snapshot = self.snapshots[seq_id]
        if snapshot.completion is not None:
            device_runtime.synchronize_event(snapshot.completion)
            self._pending.pop(seq_id)
            snapshot.completion = None
            snapshot.keepalive.clear()
            snapshot.valid = True
        return snapshot

    def poll(self) -> None:
        # All D2H events share one stream. Query only the oldest pending copy;
        # targeted waits can remove another entry without scanning this queue.
        while self._pending:
            seq_id, snapshot = next(iter(self._pending.items()))
            if not device_runtime.is_event_complete(snapshot.completion):
                break
            self._pending.popitem(last=False)
            snapshot.completion = None
            snapshot.keepalive.clear()
            snapshot.valid = True

    def invalidate(self, seq_id: int) -> None:
        if seq_id in self.snapshots:
            self.wait(seq_id).valid = False

    def drop(self, seq_id: int) -> None:
        if seq_id in self.snapshots:
            self.wait(seq_id)
            self.used_bytes -= self.snapshots.pop(seq_id).nbytes

    @torch.no_grad()
    def save(self, seq_id: int, state: ChainMethodState) -> None:
        if device_runtime.is_stream_capturing():
            raise RuntimeError("Chain offload is forbidden during graph capture.")
        self.poll()
        nbytes = self.required_bytes(seq_id, state)
        old = self.snapshots.get(seq_id)
        if self.used_bytes - (old.nbytes if old else 0) + nbytes > self.capacity_bytes:
            raise RuntimeError("Chain host capacity must be reserved before D2H submission.")
        lengths = dict(zip(self.manager.kv_transformer_layer_indices(), self.manager.chain_physical_residency(seq_id)))
        reuse = (old is not None
                 and all(len(old.kv[layer][0]) == length for layer, length in lengths.items())
                 and old.method.tensors.keys() == state.tensors.keys()
                 and all(old.method.tensors[name].shape == tensor.shape
                         and old.method.tensors[name].dtype == tensor.dtype
                         for name, tensor in state.tensors.items()))
        self.drop(seq_id)
        kv = old.kv if reuse else {}
        host_state = ChainMethodState(old.method.tensors if reuse else {}, state.metadata)
        old = None
        slots = {}
        indices = {}
        for layer, length in lengths.items():
            k, v = self.manager.chain_storage_tensors(layer)
            if not reuse:
                kv[layer] = tuple(HostTensorPool(((length, *tensor.shape[1:]),), dtype=tensor.dtype).tensors[0]
                                  for tensor in (k, v))
            slots[layer] = self.manager.chain_token_slots(layer, seq_id).to(dtype=torch.int64, copy=True)
            indices[layer] = torch.arange(length, dtype=torch.int64, device=self.device)
        for name, tensor in state.tensors.items():
            if not reuse:
                host_state.tensors[name] = HostTensorPool((tensor.shape,), dtype=tensor.dtype).tensors[0]
        # Method tensors may alias reusable score workspaces. Freeze them on the
        # producer stream before later requests can reuse those workspaces.
        sources = {name: tensor.clone() for name, tensor in state.tensors.items()}
        producer, completion = self._event(), self._event()
        snapshot = ChainSnapshot(kv, host_state, nbytes, completion=completion)
        snapshot.keepalive = [slots, indices, sources, producer]
        device_runtime.record_event(producer, self.device)
        try:
            with device_runtime.stream_context(self.stream):
                device_runtime.stream_wait_event(self.stream, producer)
                for layer, (host_k, host_v) in kv.items():
                    if slots[layer].numel():
                        self._transfer(self.manager.chain_storage_tensors(layer), (host_k, host_v), slots[layer], indices[layer])
                for name, tensor in sources.items():
                    host_state.tensors[name].copy_(tensor, non_blocking=True)
                device_runtime.record_event(completion, self.device)
        except Exception:
            device_runtime.synchronize_stream(self.stream)
            raise
        self.snapshots[seq_id] = snapshot
        self._pending[seq_id] = snapshot
        self.used_bytes += nbytes
        self.d2h_bytes += nbytes

    @torch.no_grad()
    def restore(self, seq_id: int) -> None:
        if device_runtime.is_stream_capturing():
            raise RuntimeError("Chain restore is forbidden during graph capture.")
        snapshot = self.wait(seq_id)
        if not snapshot.valid:
            raise RuntimeError("Cannot restore an invalid chain snapshot.")
        if self.manager.chain_has_residency(seq_id):
            raise RuntimeError("Cannot restore over a resident chain.")
        layers = tuple(self.manager.kv_transformer_layer_indices())
        if set(snapshot.kv) != set(layers):
            raise RuntimeError("Chain snapshot does not cover the runtime KV layers.")
        # The manager preflights and rolls back its own physical layout. Offload
        # must not manipulate row deques or layer allocator counters directly.
        allocated = self.manager.allocate_chain_restore(
            seq_id, tuple(len(snapshot.kv[layer][0]) for layer in layers),
        )
        try:
            slots = {layer: value.to(dtype=torch.int64, copy=True)
                     for layer, value in allocated.items()}
            indices = {layer: torch.arange(len(k), dtype=torch.int64, device=self.device)
                       for layer, (k, _) in snapshot.kv.items()}
            producer, completion = self._event(), self._event()
            device_runtime.record_event(producer, self.device)
            with device_runtime.stream_context(self.stream):
                device_runtime.stream_wait_event(self.stream, producer)
                for layer, (host_k, host_v) in snapshot.kv.items():
                    if slots[layer].numel():
                        self._transfer((host_k, host_v), self.manager.chain_storage_tensors(layer), indices[layer], slots[layer])
                self.manager.restore_chain_method_state(seq_id, snapshot.method)
                device_runtime.record_event(completion, self.device)
            # A single admission boundary; no synchronization in decode steps.
            device_runtime.synchronize_event(completion)
        except Exception:
            device_runtime.synchronize_stream(self.stream)
            self.manager.free_seq(seq_id)
            raise
        self.h2d_bytes += snapshot.nbytes

    def reset(self) -> None:
        for seq_id in list(self.snapshots):
            self.drop(seq_id)
        self.d2h_bytes = self.h2d_bytes = 0
