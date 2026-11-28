from __future__ import annotations

from collections import deque

import torch

from sparseengine.engine.cache_manager.prefix_offload import (
    PinnedPrefixKVPool,
    StandardPrefixOffloadController,
    _payload_device_slots,
)
from sparseengine.engine.prefix_cache import PrefixCacheBlock
from sparseengine.engine.recurrent_state_manager import (
    RecurrentPrefixPayload,
    RecurrentStateSpec,
)
from sparseengine.platforms import device_runtime


_DEFAULT_D2H_RECURRENT_STAGING_BYTE_BUDGET = 64 * 1024 * 1024


class PinnedMixedPrefixPool(PinnedPrefixKVPool):
    """Fixed mixed-prefix host pool for KV plus model-declared recurrent state."""

    def __init__(
        self,
        *,
        state_spec: RecurrentStateSpec,
        recurrent_layer_indices: tuple[int, ...],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(state_spec, RecurrentStateSpec):
            raise TypeError("Mixed prefix offload requires a RecurrentStateSpec.")
        layers = tuple(int(layer_idx) for layer_idx in recurrent_layer_indices)
        if not layers or len(set(layers)) != len(layers):
            raise ValueError(
                f"Mixed prefix offload requires unique recurrent layers, got {layers}."
            )
        self.state_spec = state_spec
        self.recurrent_layer_indices = layers
        self.recurrent_cache: dict[int, dict[str, torch.Tensor]] = {
            layer_idx: {
                tensor_spec.name: torch.empty(
                    (self.capacity_blocks, *tensor_spec.shape),
                    dtype=tensor_spec.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for tensor_spec in state_spec.tensor_specs
            }
            for layer_idx in layers
        }
        for buffers in self.recurrent_cache.values():
            for buffer in buffers.values():
                if not buffer.is_contiguous() or not buffer.is_pinned():
                    raise RuntimeError(
                        "Mixed prefix recurrent host tensors must be contiguous and pinned."
                    )

    @property
    def recurrent_bytes_per_block(self) -> int:
        return self.state_spec.bytes_for_layers(len(self.recurrent_layer_indices))


class MixedPrefixOffloadController(StandardPrefixOffloadController):
    """Coordinator-owned atomic transfer of mixed KV and recurrent snapshots."""

    host_pool: PinnedMixedPrefixPool

    def __init__(
        self,
        *,
        kv_transformer_layer_indices: tuple[int, ...],
        d2h_recurrent_staging_byte_budget: int = (
            _DEFAULT_D2H_RECURRENT_STAGING_BYTE_BUDGET
        ),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(self.host_pool, PinnedMixedPrefixPool):
            raise RuntimeError("Mixed prefix offload requires PinnedMixedPrefixPool.")
        self.kv_transformer_layer_indices = tuple(
            int(layer_idx) for layer_idx in kv_transformer_layer_indices
        )
        if len(self.kv_transformer_layer_indices) != int(self.kv_cache.shape[1]):
            raise RuntimeError(
                "Mixed prefix KV transformer-layer mapping does not match KV layers."
            )
        if set(self.kv_transformer_layer_indices) & set(
            self.host_pool.recurrent_layer_indices
        ):
            raise RuntimeError("Mixed prefix full and recurrent layer sets overlap.")
        self.d2h_recurrent_staging_byte_budget = int(
            d2h_recurrent_staging_byte_budget
        )
        if self.d2h_recurrent_staging_byte_budget <= 0:
            raise ValueError(
                "Mixed prefix D2H recurrent staging byte budget must be positive: "
                f"got={d2h_recurrent_staging_byte_budget}."
            )
        recurrent_bytes_per_block = int(self.host_pool.recurrent_bytes_per_block)
        if recurrent_bytes_per_block <= 0:
            raise RuntimeError(
                "Mixed prefix D2H requires positive recurrent bytes per block."
            )
        self.d2h_staging_blocks = max(
            1,
            self.d2h_recurrent_staging_byte_budget // recurrent_bytes_per_block,
        )
        self._pending_d2h_batches: deque[
            tuple[list[PrefixCacheBlock], list[int]]
        ] = deque()

    @property
    def pending_d2h_blocks(self) -> int:
        return sum(len(blocks) for blocks, _ in self._pending_d2h_batches)

    def _abort_pending_d2h(
        self,
        current_blocks: list[PrefixCacheBlock],
        current_host_indices: list[int],
    ) -> None:
        pending = tuple(self._pending_d2h_batches)
        self._pending_d2h_batches.clear()
        for block in current_blocks:
            self.prefix_cache.abort_d2h(block)
        self.host_pool.free(current_host_indices)
        for blocks, host_indices in pending:
            for block in blocks:
                self.prefix_cache.abort_d2h(block)
            self.host_pool.free(host_indices)

    @torch.no_grad()
    def _submit_reserved_d2h_batch(
        self,
        blocks: list[PrefixCacheBlock],
        host_indices: list[int],
    ) -> None:
        try:
            device_slots = torch.cat(
                [
                    _payload_device_slots(block, self.block_size)
                    for block in blocks
                ],
                dim=0,
            ).to(device=self.device, dtype=torch.long)
            host_token_indices = self.host_pool.token_indices(
                host_indices, self.device
            )
            auxiliary_tensors = self._prepare_d2h_auxiliary(
                blocks, host_indices
            )
            events = self._d2h_events()
            self._copy_d2h(device_slots, host_token_indices, auxiliary_tensors, events)
        except Exception:
            device_runtime.synchronize_stream(self.d2h_stream)
            raise

        self._record_d2h_operation(
            blocks, host_indices, device_slots, host_token_indices,
            auxiliary_tensors, events, self._transfer_byte_count(len(blocks)), len(blocks),
        )

    def _start_next_d2h_batch(self) -> None:
        if self.d2h_operations or not self._pending_d2h_batches:
            return
        blocks, host_indices = self._pending_d2h_batches.popleft()
        try:
            self._submit_reserved_d2h_batch(blocks, host_indices)
        except Exception:
            self._abort_pending_d2h(blocks, host_indices)
            raise

    @torch.no_grad()
    def submit_d2h(self, blocks: list[PrefixCacheBlock]) -> None:
        if not blocks:
            return
        if device_runtime.is_stream_capturing():
            raise RuntimeError("Prefix D2H submission is forbidden during graph capture.")
        host_indices = self.host_pool.allocate(len(blocks))
        begun: list[PrefixCacheBlock] = []
        try:
            # Reserve the whole chain before staging so queued children remain
            # protected by D2H state and cannot be selected or demoted twice.
            for block in blocks:
                self.prefix_cache.begin_d2h(block)
                begun.append(block)
        except Exception:
            for block in reversed(begun):
                self.prefix_cache.abort_d2h(block)
            self.host_pool.free(host_indices)
            raise

        for start in range(0, len(blocks), self.d2h_staging_blocks):
            end = min(start + self.d2h_staging_blocks, len(blocks))
            self._pending_d2h_batches.append(
                (list(blocks[start:end]), list(host_indices[start:end]))
            )
        self._start_next_d2h_batch()

    def poll_d2h(self) -> int:
        completed = super().poll_d2h()
        self._start_next_d2h_batch()
        return completed

    def wait_oldest_d2h(self) -> bool:
        self._start_next_d2h_batch()
        if not self.d2h_operations:
            return False
        device_runtime.synchronize_event(
            self.d2h_operations[0].completion_event
        )
        self.poll_d2h()
        return True

    def synchronize_all(self) -> None:
        while self.d2h_operations or self._pending_d2h_batches:
            self._start_next_d2h_batch()
            device_runtime.synchronize_event(
                self.d2h_operations[0].completion_event
            )
            self.poll_d2h()
        super().synchronize_all()

    def stats(self) -> dict[str, int]:
        stats = super().stats()
        stats.update(
            {
                "prefix_cache_d2h_staging_blocks": int(
                    self.d2h_staging_blocks
                ),
                "prefix_cache_d2h_pending_blocks": int(
                    self.pending_d2h_blocks
                ),
            }
        )
        return stats

    def _recurrent_payload(self, block: PrefixCacheBlock) -> RecurrentPrefixPayload:
        mixed_payload = block.payload
        payload = getattr(mixed_payload, "recurrent_payload", None)
        if not isinstance(payload, RecurrentPrefixPayload):
            raise RuntimeError("Mixed prefix block has no recurrent snapshot payload.")
        expected_layers = set(self.host_pool.recurrent_layer_indices)
        if set(int(layer_idx) for layer_idx in payload.layer_states) != expected_layers:
            raise RuntimeError(
                "Mixed recurrent snapshot layers do not match state_spec: "
                f"expected={sorted(expected_layers)} got={sorted(payload.layer_states)}."
            )
        expected_names = set(self.host_pool.state_spec.state_names)
        specs = {spec.name: spec for spec in self.host_pool.state_spec.tensor_specs}
        for layer_idx in self.host_pool.recurrent_layer_indices:
            states = payload.layer_states[layer_idx]
            if set(states) != expected_names:
                raise RuntimeError(
                    "Mixed recurrent snapshot names do not match state_spec: "
                    f"layer={layer_idx} expected={sorted(expected_names)} got={sorted(states)}."
                )
            for name, tensor in states.items():
                spec = specs[name]
                if (
                    not torch.is_tensor(tensor)
                    or tuple(tensor.shape) != spec.shape
                    or tensor.dtype != spec.dtype
                    or tensor.device != self.device
                ):
                    raise RuntimeError(
                        "Mixed recurrent snapshot tensor does not match state_spec: "
                        f"layer={layer_idx} name={name} expected={spec.shape}/{spec.dtype}/{self.device} "
                        f"got={getattr(tensor, 'shape', None)}/{getattr(tensor, 'dtype', None)}/"
                        f"{getattr(tensor, 'device', None)}."
                    )
        return payload

    def allocate_device_recurrent(self, blocks: list[PrefixCacheBlock]) -> None:
        for block in blocks:
            mixed_payload = block.payload
            recurrent = getattr(mixed_payload, "recurrent_payload", None)
            if not isinstance(recurrent, RecurrentPrefixPayload):
                raise RuntimeError("Mixed prefix block has no recurrent snapshot payload.")
            if recurrent.layer_states:
                raise RuntimeError("Host-only mixed prefix block still owns GPU recurrent tensors.")
            recurrent.layer_states = {
                layer_idx: {
                    spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
                    for spec in self.host_pool.state_spec.tensor_specs
                }
                for layer_idx in self.host_pool.recurrent_layer_indices
            }

    def free_device_recurrent(self, block: PrefixCacheBlock) -> None:
        recurrent = self._recurrent_payload(block)
        recurrent.layer_states = {}

    def _ordered_recurrent_tensors(
        self,
        block: PrefixCacheBlock,
    ) -> list[torch.Tensor]:
        payload = self._recurrent_payload(block)
        return [
            payload.layer_states[layer_idx][spec.name]
            for layer_idx in self.host_pool.recurrent_layer_indices
            for spec in self.host_pool.state_spec.tensor_specs
        ]

    def _prepare_d2h_auxiliary(
        self,
        blocks: list[PrefixCacheBlock],
        host_indices: list[int],
    ) -> tuple[torch.Tensor, ...]:
        host_index_tensor = torch.tensor(host_indices, dtype=torch.long, device="cpu")
        stacks: list[torch.Tensor] = []
        tensor_count = len(self._ordered_recurrent_tensors(blocks[0]))
        ordered = [self._ordered_recurrent_tensors(block) for block in blocks]
        for tensor_idx in range(tensor_count):
            stacks.append(torch.stack([tensors[tensor_idx] for tensors in ordered], dim=0))
        metadata = super()._prepare_d2h_auxiliary(blocks, host_indices)
        return (*metadata, host_index_tensor, *stacks)

    def _prepare_h2d_auxiliary(
        self,
        blocks: list[PrefixCacheBlock],
        host_indices: list[int],
    ) -> tuple[torch.Tensor, ...]:
        host_index_tensor = torch.tensor(host_indices, dtype=torch.long, device="cpu")
        destinations = [
            tensor
            for block in blocks
            for tensor in self._ordered_recurrent_tensors(block)
        ]
        metadata = super()._prepare_h2d_auxiliary(blocks, host_indices)
        return (*metadata, host_index_tensor, *destinations)

    def _host_recurrent_tensors(self) -> list[torch.Tensor]:
        return [
            self.host_pool.recurrent_cache[layer_idx][spec.name]
            for layer_idx in self.host_pool.recurrent_layer_indices
            for spec in self.host_pool.state_spec.tensor_specs
        ]

    def _submit_d2h_payload(
        self,
        device_slots: torch.Tensor,
        host_token_indices: torch.Tensor,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        super()._submit_d2h_payload(device_slots, host_token_indices, auxiliary_tensors)
        recurrent = auxiliary_tensors[2:] if self._quest_metadata is not None else auxiliary_tensors
        self._submit_recurrent_d2h(recurrent)

    def _submit_recurrent_d2h(
        self,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        host_indices = auxiliary_tensors[0].tolist()
        stacks = auxiliary_tensors[1:]
        host_tensors = self._host_recurrent_tensors()
        if len(stacks) != len(host_tensors):
            raise RuntimeError("Mixed recurrent D2H tensor count does not match state_spec.")
        runs: list[tuple[int, int, int]] = []
        source_start = 0
        while source_start < len(host_indices):
            source_end = source_start + 1
            while (
                source_end < len(host_indices)
                and int(host_indices[source_end])
                == int(host_indices[source_end - 1]) + 1
            ):
                source_end += 1
            runs.append((source_start, source_end, int(host_indices[source_start])))
            source_start = source_end
        for host_tensor, source_stack in zip(host_tensors, stacks):
            for source_start, source_end, host_start in runs:
                host_tensor[host_start : host_start + source_end - source_start].copy_(
                    source_stack[source_start:source_end],
                    non_blocking=True,
                )

    def _h2d_transfer_schedule(self) -> list[tuple[str, int]]:
        schedule = [
            (int(transformer_layer), "kv", kv_layer)
            for kv_layer, transformer_layer in enumerate(self.kv_transformer_layer_indices)
        ]
        schedule.extend(
            (int(layer_idx), "auxiliary", int(layer_idx))
            for layer_idx in self.host_pool.recurrent_layer_indices
        )
        schedule.sort(key=lambda item: item[0])
        return [(kind, index) for _, kind, index in schedule]

    def _submit_h2d_auxiliary_layer(
        self,
        layer_index: int,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        if self._quest_metadata is not None:
            auxiliary_tensors = auxiliary_tensors[2:]
        host_indices = auxiliary_tensors[0].tolist()
        destinations = auxiliary_tensors[1:]
        host_tensors = self._host_recurrent_tensors()
        tensors_per_block = len(host_tensors)
        if len(destinations) != len(host_indices) * tensors_per_block:
            raise RuntimeError("Mixed recurrent H2D tensor count does not match state_spec.")
        try:
            layer_position = self.host_pool.recurrent_layer_indices.index(int(layer_index))
        except ValueError as exc:
            raise RuntimeError(
                f"Mixed prefix H2D requested undeclared recurrent layer={layer_index}."
            ) from exc
        specs_per_layer = len(self.host_pool.state_spec.tensor_specs)
        tensor_start = layer_position * specs_per_layer
        for block_idx, host_idx in enumerate(host_indices):
            for spec_offset in range(specs_per_layer):
                tensor_idx = tensor_start + spec_offset
                host_tensor = host_tensors[tensor_idx]
                destination = destinations[block_idx * tensors_per_block + tensor_idx]
                destination.copy_(host_tensor[int(host_idx)], non_blocking=True)

    def _h2d_token_byte_count(self, token_count: int) -> int:
        return token_count * self.host_pool.num_layers * 2 * self.item_size

    def _transfer_byte_count(self, block_count: int) -> int:
        return super()._transfer_byte_count(block_count) + int(
            int(block_count) * self.host_pool.recurrent_bytes_per_block
        )
