from __future__ import annotations

from collections import deque

import torch

from sparseengine.engine.cache_manager.prefix_offload import (
    PinnedPrefixKVPool,
    PrefixD2HOperation,
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


class PinnedMixedQuestPrefixPool(PinnedMixedPrefixPool):
    def __init__(self, *, page_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.page_size = int(page_size)
        if self.block_size % self.page_size != 0:
            raise ValueError("Mixed QuEST prefix block size must contain whole pages.")
        self.pages_per_block = self.block_size // self.page_size
        self.metadata_cache = torch.empty(
            (
                2,
                self.num_layers,
                self.capacity_blocks * self.pages_per_block,
                self.cache.shape[-2],
                self.cache.shape[-1],
            ),
            dtype=self.cache.dtype,
            device="cpu",
            pin_memory=True,
        )


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
            producer_event = self._new_event(self.device, "D2H producer")
            completion_event = self._new_event(self.device, "D2H completion")
            device_runtime.record_event(producer_event, device=self.device)
            with device_runtime.stream_context(self.d2h_stream):
                device_runtime.stream_wait_event(self.d2h_stream, producer_event)
                self._submit_d2h_payload(
                    device_slots,
                    host_token_indices,
                    auxiliary_tensors,
                )
                device_runtime.record_event(
                    completion_event, device=self.device
                )
        except Exception:
            device_runtime.synchronize_stream(self.d2h_stream)
            raise

        byte_count = self._transfer_byte_count(len(blocks))
        self.d2h_operations.append(
            PrefixD2HOperation(
                blocks=list(blocks),
                host_indices=list(host_indices),
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
        self.d2h_merged_blocks += len(blocks)

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
        return (host_index_tensor, *stacks)

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
        return (host_index_tensor, *destinations)

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
        self._submit_recurrent_d2h(auxiliary_tensors)

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

    def _transfer_byte_count(self, block_count: int) -> int:
        return super()._transfer_byte_count(block_count) + int(
            int(block_count) * self.host_pool.recurrent_bytes_per_block
        )


class MixedQuestPrefixOffloadController(MixedPrefixOffloadController):
    host_pool: PinnedMixedQuestPrefixPool

    def __init__(self, *, device_metadata_cache: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(self.host_pool, PinnedMixedQuestPrefixPool):
            raise RuntimeError("Mixed QuEST offload requires PinnedMixedQuestPrefixPool.")
        self.device_metadata_cache = device_metadata_cache
        host_metadata = self.host_pool.metadata_cache
        if not self.device_metadata_cache.is_contiguous() or not host_metadata.is_pinned():
            raise RuntimeError("Mixed QuEST metadata pools must be contiguous and pinned.")
        if (
            tuple(self.device_metadata_cache.shape[:2]) != (2, self.host_pool.num_layers)
            or tuple(self.device_metadata_cache.shape[-2:]) != tuple(host_metadata.shape[-2:])
        ):
            raise RuntimeError("Mixed QuEST metadata pool shapes are incompatible.")
        metadata_item_size = int(
            self.device_metadata_cache.shape[-2]
            * self.device_metadata_cache.shape[-1]
            * self.device_metadata_cache.element_size()
        )
        if metadata_item_size != self.item_size:
            raise RuntimeError("Mixed QuEST KV and metadata item sizes must match.")
        self.device_metadata_max_ptrs = torch.tensor(
            [
                self.device_metadata_cache[0, layer].data_ptr()
                for layer in range(self.host_pool.num_layers)
            ],
            dtype=torch.uint64,
            device=self.device,
        )
        self.device_metadata_min_ptrs = torch.tensor(
            [
                self.device_metadata_cache[1, layer].data_ptr()
                for layer in range(self.host_pool.num_layers)
            ],
            dtype=torch.uint64,
            device=self.device,
        )
        self.host_metadata_max_ptrs = torch.tensor(
            [host_metadata[0, layer].data_ptr() for layer in range(self.host_pool.num_layers)],
            dtype=torch.uint64,
            device=self.device,
        )
        self.host_metadata_min_ptrs = torch.tensor(
            [host_metadata[1, layer].data_ptr() for layer in range(self.host_pool.num_layers)],
            dtype=torch.uint64,
            device=self.device,
        )

    def _device_pages(self, block: PrefixCacheBlock) -> torch.Tensor:
        kv_payload = getattr(block.payload, "kv_payload", None)
        pages = getattr(kv_payload, "block_slots", None)
        if pages is None:
            page = getattr(kv_payload, "block_slot", None)
            if page is None:
                raise RuntimeError("Mixed QuEST block has no device page payload.")
            pages = torch.tensor([int(page)], dtype=torch.int32, device=self.device)
        pages = pages.to(device=self.device, dtype=torch.long).reshape(-1)
        if int(pages.numel()) != self.host_pool.pages_per_block:
            raise RuntimeError(
                "Mixed QuEST device page count does not match prefix block geometry."
            )
        return pages

    def _host_pages(self, host_indices: list[int]) -> torch.Tensor:
        values = [
            int(host_idx) * self.host_pool.pages_per_block + page_offset
            for host_idx in host_indices
            for page_offset in range(self.host_pool.pages_per_block)
        ]
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def _prepare_d2h_auxiliary(
        self,
        blocks: list[PrefixCacheBlock],
        host_indices: list[int],
    ) -> tuple[torch.Tensor, ...]:
        recurrent = super()._prepare_d2h_auxiliary(blocks, host_indices)
        device_pages = torch.cat([self._device_pages(block) for block in blocks])
        return (device_pages, self._host_pages(host_indices), *recurrent)

    def _prepare_h2d_auxiliary(
        self,
        blocks: list[PrefixCacheBlock],
        host_indices: list[int],
    ) -> tuple[torch.Tensor, ...]:
        recurrent = super()._prepare_h2d_auxiliary(blocks, host_indices)
        device_pages = torch.cat([self._device_pages(block) for block in blocks])
        return (device_pages, self._host_pages(host_indices), *recurrent)

    def _submit_d2h_payload(
        self,
        device_slots: torch.Tensor,
        host_token_indices: torch.Tensor,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        StandardPrefixOffloadController._submit_d2h_payload(
            self, device_slots, host_token_indices, auxiliary_tensors
        )
        device_pages, host_pages = auxiliary_tensors[:2]
        self._transfer_all_layers(
            src_k_layers=self.device_metadata_max_ptrs,
            dst_k_layers=self.host_metadata_max_ptrs,
            src_v_layers=self.device_metadata_min_ptrs,
            dst_v_layers=self.host_metadata_min_ptrs,
            src_indices=device_pages,
            dst_indices=host_pages,
            item_size=self.item_size,
            num_layers=self.host_pool.num_layers,
        )
        self._submit_recurrent_d2h(auxiliary_tensors[2:])

    def _submit_h2d_layer(
        self,
        layer_index: int,
        host_token_indices: torch.Tensor,
        device_slots: torch.Tensor,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        StandardPrefixOffloadController._submit_h2d_layer(
            self,
            layer_index,
            host_token_indices,
            device_slots,
            auxiliary_tensors,
        )
        device_pages, host_pages = auxiliary_tensors[:2]
        host_metadata = self.host_pool.metadata_cache
        self._transfer_per_layer(
            src_k=host_metadata[0, layer_index],
            dst_k=self.device_metadata_cache[0, layer_index],
            src_v=host_metadata[1, layer_index],
            dst_v=self.device_metadata_cache[1, layer_index],
            src_indices=host_pages,
            dst_indices=device_pages,
            item_size=self.item_size,
        )

    def _submit_h2d_auxiliary_layer(
        self,
        layer_index: int,
        auxiliary_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        super()._submit_h2d_auxiliary_layer(layer_index, auxiliary_tensors[2:])

    def _transfer_byte_count(self, block_count: int) -> int:
        metadata_bytes = int(
            int(block_count)
            * self.host_pool.pages_per_block
            * self.host_pool.num_layers
            * 2
            * self.item_size
        )
        return super()._transfer_byte_count(block_count) + metadata_bytes
