from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from pathlib import Path

import torch

from sparseengine.operators.indexed_host_copy import (
    append_rows,
    gather_prefill_rows,
    gather_prefill_history,
    scatter_prefill_current,
    gather_rows,
)
from sparseengine.utils.context import get_context

from .capacity import fit_omnikv_host_slots, plan_omnikv_pools
from .lru import OmniKVLRU
from .storage import OmniKVStorage, payload_tensors
from ...standard import StandardCacheManager
from ...storage import HeterogeneousExplicitKVStorage


class OmniKVCacheManager(StandardCacheManager):
    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        self.offload_enabled = bool(config.enable_omnikv_offload)
        self._prefetched = set()
        self._pending_prefetch = deque()
        self._current_writes = {}
        self.lru = None
        self._prefill_next_layer = {}
        self._prefill_prefetched_layer = None
        super().__init__(
            config, parallel_context, allocation_budget_bytes=allocation_budget_bytes
        )

    def allocate_kv_cache(self):
        if not self.offload_enabled:
            return super().allocate_kv_cache()
        original = self.attention_cache_storage
        if self.device.type != "cuda" or isinstance(
            original, HeterogeneousExplicitKVStorage
        ):
            raise ValueError(
                "OmniKV offload requires CUDA uniform explicit KV or MLA latent storage."
            )
        if original.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("OmniKV offload supports FP16/BF16 cache storage.")
        available, per_layer = self._get_available_slots_info()
        plan = plan_omnikv_pools(
            self.config,
            [self.kv_layer_index(i) for i in self.config.full_attention_layers],
            self.num_kv_layers,
            self.max_buffer_rows,
            per_layer,
        )
        full = plan.full_layers
        sparse = self.num_kv_layers - len(full)
        self.selected_capacity = plan.selected_capacity
        cache_capacity = plan.cache_capacity
        layer_groups = plan.layer_groups
        staging_slots = self.max_buffer_rows * self.selected_capacity
        slots = (available - plan.fixed_bytes) // plan.slot_bytes
        # Host backing is bounded across worker processes, not once per GPU.
        meminfo = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        host_budget = (
            int(meminfo["MemAvailable"].split()[0]) * 1024 // (2 * self.world_size)
        )
        prefix_bytes = (
            int((self.config.prefix_cache_host_size_gb or 0) * 1024**3)
            if self.config.enable_prefix_cache_offload
            else 0
        )
        self.prefix_host_blocks = prefix_bytes // (
            self.num_kv_layers * per_layer * self.config.prefix_cache_block_size
        )
        slots = min(slots, self.max_model_len * self.max_buffer_rows)
        slots = fit_omnikv_host_slots(
            slots,
            host_budget,
            [
                heads * dim * original.dtype.itemsize
                for heads, dim in OmniKVStorage.payload_shapes(original)
            ],
            sparse,
            len(full),
            self.prefix_host_blocks * self.config.prefix_cache_block_size,
        )
        if self.world_size > 1:
            capacity = torch.tensor(slots, dtype=torch.int64, device=self.device)
            self.parallel_context.world.all_reduce(
                capacity, op=torch.distributed.ReduceOp.MIN
            )
            slots = int(capacity.item())
        if slots <= 0 or (
            getattr(self.config, "startup_cache_phase", "production") != "profiling"
            and slots < self.max_model_len
        ):
            raise MemoryError(
                f"OmniKV offload pools cannot fit: slots={slots}, max_model_len={self.max_model_len}."
            )
        self.config.num_kvcache_slots = int(slots)
        if self.config.prefix_cache_max_blocks is not None:
            self.config.prefix_cache_max_blocks = min(
                self.config.prefix_cache_max_blocks,
                slots // self.config.prefix_cache_block_size,
            )
        storage = OmniKVStorage(
            original,
            num_layers=self.num_kv_layers,
            num_slots=slots,
            full_layers=full,
            device=self.device,
            prefix_slots=self.prefix_host_blocks * self.config.prefix_cache_block_size,
        )
        self.attention_cache_storage = storage
        self.kv_cache = None

        def allocate(count):
            return tuple(
                torch.empty(count, *shape, dtype=storage.dtype, device=self.device)
                for shape in storage.shapes
            )

        self.prefill_staging = allocate(slots)
        self.selected_staging = {
            i: allocate(staging_slots)
            for i in range(self.num_kv_layers)
            if i not in storage.full_layers and not cache_capacity
        }
        # Preserve the input table's capacity for provider planning: FA3 uses
        # its width to choose splits, even when the effective context is short.
        self.selected_slots = torch.zeros(
            self.max_buffer_rows,
            self.max_model_len,
            dtype=torch.int32,
            device=self.device,
        )
        self.selected_slots[:, : self.selected_capacity].copy_(
            torch.arange(staging_slots, dtype=torch.int32, device=self.device).view(
                self.max_buffer_rows, self.selected_capacity
            )
        )
        if cache_capacity:
            self.lru = OmniKVLRU(
                storage,
                layer_groups,
                self.max_buffer_rows,
                cache_capacity,
                self.selected_capacity,
                self.device,
                view=self.selected_slots,
            )
            self.selected_staging = self.lru.parts
        self.selected_rows = torch.arange(
            self.max_buffer_rows, dtype=torch.int32, device=self.device
        )
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.selection_done = torch.cuda.Event()
        self.layer_ready = {i: torch.cuda.Event() for i in self.selected_staging}
        self._selection_pending = False
        if not self.config.prefill_sparse_method:
            next_layer = None
            for layer in reversed(self.kv_transformer_layer_indices()):
                self._prefill_next_layer[layer] = next_layer
                if self.kv_layer_index(layer) not in storage.full_layers:
                    next_layer = layer

    def _init_prefix_offload(self):
        if not self.offload_enabled:
            return super()._init_prefix_offload()
        from .prefix import OmniKVPrefixOffloadController, OmniKVPrefixPool

        if self.prefix_host_blocks <= 0:
            raise ValueError(
                "OmniKV prefix offload requires positive prefix_cache_host_size_gb."
            )
        required = self.config.num_kvcache_slots // self.prefix_cache_block_size
        if self.prefix_cache.max_blocks is not None:
            required = min(required, self.prefix_cache.max_blocks)
        if self.prefix_host_blocks < required:
            raise ValueError(
                f"Prefix host pool needs {required} blocks, has {self.prefix_host_blocks}."
            )
        pool = OmniKVPrefixPool(
            self.attention_cache_storage,
            self.prefix_host_blocks,
            self.prefix_cache_block_size,
            self.device,
        )
        self.prefix_offload_controller = OmniKVPrefixOffloadController(
            prefix_cache=self.prefix_cache,
            storage=self.attention_cache_storage,
            host_pool=pool,
            block_size=self.prefix_cache_block_size,
            device=self.device,
        )

    def _iter_accounting_tensors(self):
        yield from super()._iter_accounting_tensors()
        if self.lru is not None:
            for i, tensor in enumerate(self.lru.tensors()):
                yield f"omnikv_lru.{i}", tensor
        if self.offload_enabled and self.prefix_offload_controller is not None:
            pool = self.prefix_offload_controller.host_pool
            for layer, parts in enumerate(pool.layers):
                for component, tensor in enumerate(parts):
                    yield f"prefix_host_cache.{layer}.{component}", tensor

    def prefix_kv_payload_nbytes(self, payload):
        if not self.offload_enabled:
            return super().prefix_kv_payload_nbytes(payload)
        return (
            payload.token_slots.numel()
            * len(self.attention_cache_storage.full_layers)
            * self.attention_cache_storage.bytes_per_slot_per_layer()
        )

    def memory_accounting(self):
        result = super().memory_accounting()
        if self.offload_enabled:
            tensors = result["tensors"]
            gpu_bytes = sum(
                t["nbytes"] for t in tensors if t["device"].startswith("cuda")
            )
            host_bytes = sum(t["nbytes"] for t in tensors if t["device"] == "cpu")
            baseline = (
                self.config.num_kvcache_slots
                * self.num_kv_layers
                * self.attention_cache_storage.bytes_per_slot_per_layer()
            )
            result.update(
                allocated_device_tensor_bytes=gpu_bytes,
                allocated_host_tensor_bytes=host_bytes,
                omnikv_gpu_only_kv_bytes=baseline,
                observed_savings=1 - gpu_bytes / baseline,
            )
        return result

    def begin_selection_step(self):
        self._prefetched.clear()
        self._pending_prefetch.clear()
        self._current_writes.clear()
        self._selection_pending = False
        if self.lru is not None:
            self.lru.planned.clear()

    def free_seq(self, seq_id):
        if self.lru is not None:
            self.lru.invalidate(self.seq_id_to_row[seq_id])
        return super().free_seq(seq_id)

    def _prepare_prefill(self, seqs):
        self._prefill_prefetched_layer = None
        self.begin_selection_step()
        return super()._prepare_prefill(seqs)

    def _prepare_decode(self, seqs):
        self.begin_selection_step()
        return super()._prepare_decode(seqs)

    def _prepare_decode_graph_buffers(self, seqs, **kwargs):
        self.begin_selection_step()
        return super()._prepare_decode_graph_buffers(seqs, **kwargs)

    @contextmanager
    def selection_stream(self):
        self.prefetch_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.prefetch_stream):
            yield

    def prefetch_selections(self, selections):
        self.selection_done.record(self.prefetch_stream)
        self._selection_pending = True
        self._pending_prefetch.extend(selections)
        self._prefetch_next_layer()

    def _prefetch_next_layer(self):
        if not self._pending_prefetch:
            return
        layer_idx, selection = self._pending_prefetch.popleft()
        kv_idx = self.kv_layer_index(layer_idx)
        slots = self._default_active_slots_for_selection(layer_idx, selection)
        self._gather_decode(
            kv_idx, slots, selection.req_indices, selection.context_lens
        )
        self.layer_ready[kv_idx].record(self.prefetch_stream)
        self._prefetched.add(kv_idx)

    def _gather_decode(self, kv_idx, slots, rows, lengths):
        plan = None
        miss_tokens = miss_counts = None
        if self.lru is not None:
            plan = self.lru.prepare(
                kv_idx,
                slots,
                rows,
                self.layer_batch_state.req_indices,
                lengths,
                self.layer_batch_state.slot_mapping,
            )
            miss_tokens, miss_counts = self.lru.misses(kv_idx)
        for component, destination in enumerate(self.selected_staging[kv_idx]):
            gather_rows(
                self.attention_cache_storage.pointers[kv_idx],
                destination,
                slots,
                rows,
                lengths,
                capacity=self.selected_capacity,
                component=component,
                plan=plan,
                miss_tokens=miss_tokens,
                miss_counts=miss_counts,
                skip_last=self.config.recent_keep_tokens > 0,
                exclude_slots=self.layer_batch_state.slot_mapping,
                # Bound the whole batch footprint to leave SMs for model work.
                block_budget=32,
                slot_map=self.attention_cache_storage.host_slot_map,
            )

    def store_attention_payload(self, layer_idx, payload):
        slots = super().store_attention_payload(layer_idx, payload)
        if (
            self.offload_enabled
            and not get_context().is_prefill
            and self.kv_layer_index(layer_idx) in self.selected_staging
        ):
            self._current_writes[layer_idx] = payload
        return slots

    def get_layer_kv_cache(self, layer_idx):
        if not self.offload_enabled:
            return super().get_layer_kv_cache(layer_idx)
        return payload_tensors(
            self.attention_cache_storage.layer_payload(self.kv_layer_index(layer_idx))
        )

    def get_layer_compute_payload(
        self, layer_idx, active_slots, req_indices, context_lens, selection=None
    ):
        if not self.offload_enabled:
            return super().get_layer_compute_payload(
                layer_idx, active_slots, req_indices, context_lens, selection
            )
        kv_idx = self.kv_layer_index(layer_idx)
        storage = self.attention_cache_storage
        stream = torch.cuda.current_stream(self.device)
        if kv_idx in storage.full_layers:
            # The next observer reuses the shared raw score buffer.
            if self._selection_pending:
                stream.wait_event(self.selection_done)
            return (
                storage.layer_payload(kv_idx),
                active_slots,
                req_indices,
                context_lens,
            )
        if kv_idx in self._prefetched:
            stream.wait_event(self.layer_ready[kv_idx])
            # Advance only when the current consumer reaches its KV view. The
            # next layer's transfer overlaps this layer's attention and MLP.
            with self.selection_stream():
                self._prefetch_next_layer()
        else:
            self._gather_decode(kv_idx, active_slots, req_indices, context_lens)
        parts = self.selected_staging[kv_idx]
        for source, destination in zip(
            payload_tensors(self._current_writes.pop(layer_idx)), parts
        ):
            append_rows(
                source,
                destination,
                context_lens,
                self.layer_batch_state.slot_mapping,
                self.selected_capacity,
                plan=None if self.lru is None else self.lru.plan(kv_idx),
                table=active_slots if self.config.recent_keep_tokens == 0 else None,
                rows=req_indices if self.config.recent_keep_tokens == 0 else None,
            )
        batch = req_indices.numel()
        return (
            storage.make_payload(parts),
            self.selected_slots[:batch, : active_slots.shape[1]],
            self.selected_rows[:batch],
            context_lens,
        )

    def get_prefill_compute_payload(
        self,
        layer_idx,
        k_current,
        v_current,
        selection,
        active_slots,
        req_indices,
        context_lens,
    ):
        if not self.offload_enabled:
            return super().get_prefill_compute_payload(
                layer_idx,
                k_current,
                v_current,
                selection,
                active_slots,
                req_indices,
                context_lens,
            )
        kv_idx = self.kv_layer_index(layer_idx)
        storage = self.attention_cache_storage
        if kv_idx in storage.full_layers:
            return (
                storage.layer_payload(kv_idx),
                active_slots,
                req_indices,
                context_lens,
            )
        prefetched = self._prefill_prefetched_layer == layer_idx
        if prefetched:
            torch.cuda.current_stream(self.device).wait_event(self.layer_ready[kv_idx])
            self._prefill_prefetched_layer = None
        for component, (current, destination) in enumerate(
            zip((k_current, v_current), self.prefill_staging)
        ):
            if prefetched:
                scatter_prefill_current(
                    current, destination, self.layer_batch_state.slot_mapping
                )
                continue
            gather_prefill_rows(
                storage.pointers[kv_idx],
                current,
                destination,
                active_slots,
                req_indices,
                context_lens,
                get_context().cu_seqlens_q,
                storage.host_slot_map,
                capacity=int(selection.max_context_len),
                component=component,
            )
        return (
            storage.make_payload(self.prefill_staging),
            active_slots,
            req_indices,
            context_lens,
        )

    def on_layer_attention_end(self, layer_idx):
        super().on_layer_attention_end(layer_idx)
        if not self.offload_enabled or not get_context().is_prefill:
            return
        next_layer = self._prefill_next_layer.get(layer_idx)
        if next_layer is None or self._prefill_prefetched_layer is not None:
            return
        # Prefix restoration may remap host slots. Order it before the early
        # read, and reuse the single staging pool only after attention consumes it.
        self.before_prefill_layer_attention(next_layer, None)
        storage = self.attention_cache_storage
        state = self.layer_batch_state
        kv_idx = self.kv_layer_index(next_layer)
        with self.selection_stream():
            for component, destination in enumerate(self.prefill_staging):
                gather_prefill_history(
                    storage.pointers[kv_idx],
                    destination,
                    self.buffer_req_to_token_slots,
                    state.req_indices,
                    state.context_lens,
                    get_context().cu_seqlens_q,
                    storage.host_slot_map,
                    component=component,
                )
            self.layer_ready[kv_idx].record(self.prefetch_stream)
        self._prefill_prefetched_layer = next_layer

    def decode_graph_keepalive_tensors(self):
        result = super().decode_graph_keepalive_tensors()
        if self.offload_enabled:
            result.extend(self.attention_cache_storage.accounting_tensors())
            result.extend(self.prefill_staging)
            result.extend(x for parts in self.selected_staging.values() for x in parts)
            result.extend((self.selected_slots, self.selected_rows))
            if self.lru is not None:
                result.extend(self.lru.tensors())
        return result

    def on_forward_end(self, seqs, is_prefill):
        if self.offload_enabled:
            # Join every captured fork and protect staging before the next step.
            torch.cuda.current_stream(self.device).wait_stream(self.prefetch_stream)
        return super().on_forward_end(seqs, is_prefill)
