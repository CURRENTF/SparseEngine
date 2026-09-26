from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import torch

from sparseengine.engine.cache_manager.base import (
    CacheManager, CompressionComputeView, LayerBatchStates, PackedSharedKVPayload,
    SharedKVPayload,
)
from sparseengine.engine.cache_manager.storage.compressed_family import CompressedKVFamily
from sparseengine.engine.cache_manager.storage.packed_shared_kv import PackedSharedKVPool
from sparseengine.engine.cache_manager.storage.shared_kv_state import SharedKVStateRows
from sparseengine.engine.prefix_cache import (
    PrefixCacheBlock, RadixPrefixIndex, build_prefix_cache_fingerprint, usable_prefix_cache_tokens,
)


@dataclass
class NativeSharedKVRequest:
    row: int
    length: int = 0
    leases: dict = field(default_factory=dict)
    prefix_blocks: list = field(default_factory=list)


@dataclass(eq=False)
class NativeSharedKVPrefix:
    row: int
    length: int
    leases: dict


@dataclass(eq=False)
class NativePrefixRecord:
    payload: NativeSharedKVPrefix
    block_id: bytes
    parent_id: bytes | None
    index: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class SharedKVCandidateView:
    """Cache-owned physical candidates; logical visibility belongs to runtime."""
    query_positions: torch.Tensor
    query_rows: torch.Tensor
    window_slots: torch.Tensor
    compressed_slots: torch.Tensor
    index_pages: torch.Tensor | None
    index_slots: torch.Tensor
    window_logical_positions: torch.Tensor
    payload: PackedSharedKVPayload | SharedKVPayload


class DeepSeekV4CacheManager(CacheManager):
    """Native shared-KV storage, carry, physical views and prefix ownership."""

    page_size = 64
    window_size = 128

    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        super().__init__(config, parallel_context, allocation_budget_bytes=allocation_budget_bytes)
        self.ratios = tuple(int(r) for r in self.hf_config.compress_ratios[:self.num_layers])
        if len(self.ratios) != self.num_layers or any(r not in (0, 4, 128) for r in self.ratios):
            raise ValueError("Native shared KV requires one compression ratio 0/4/128 per layer")
        if self.tp_size != 1 or self.head_dim != 512:
            raise ValueError("Native shared KV requires attention TP=1 and D512")
        self.enable_prefix_caching = bool(config.enable_prefix_caching) and getattr(config, "startup_cache_phase", "") != "profiling"
        if self.enable_prefix_caching and config.resolved_prefix_cache_mode != "radix":
            raise ValueError("Native shared KV prefix caching requires radix mode")
        if getattr(config, "enable_prefix_cache_offload", False):
            raise ValueError("Native shared KV prefix offload is not implemented")
        self.prefix_cache_block_size = int(config.prefix_cache_block_size)
        self.requests = {}
        self.pending_prefix = {}
        self.private_prefix_records = {}
        self.layer_batch_state = LayerBatchStates()
        self.compression_planners = {}
        self.compression_views = {}
        self.decode_selection_workspaces = {}
        self.step_is_prefill = False
        self.allocate_kv_cache()
        self.prefix_cache = (RadixPrefixIndex(
            block_size=self.prefix_cache_block_size,
            fingerprint=build_prefix_cache_fingerprint(config, self.prefix_cache_block_size),
            max_blocks=self.prefix_rows,
        ) if self.enable_prefix_caching else None)

    @classmethod
    def profiling_budget_bytes(cls, config, token_slots):
        ratios = tuple(config.hf_config.compress_ratios[:config.hf_config.num_hidden_layers])
        rows = 2*int(config.max_num_seqs_in_gpu)
        fixed = rows*(SharedKVStateRows.bytes_per_row(ratios)+ratios.count(4)*cls.window_size*68)
        page_bytes = ((584*cls.page_size+575)//576)*576
        requests = max(int(config.max_num_seqs_in_batch), int(config.max_decoding_seqs))
        units = ceil(token_slots/(128*cls.page_size))+requests
        pages = sum(units*ceil(128/r)
                    *ratios.count(r)*(page_bytes+(cls.page_size*68 if r == 4 else 0))
                    for r in (4, 128) if r in ratios)
        tables = rows*sum(ceil(config.max_model_len/r)*4 for r in set(ratios) if r)
        return fixed + pages + tables + ratios.count(0)*page_bytes

    def allocate_kv_cache(self):
        budget = self.allocation_budget_bytes
        if budget is None:
            free, total = self.platform.get_available_memory(self.device.index or 0)
            budget = max(0, int(total*self.config.gpu_memory_utilization) - (total-free))
        row_bytes = SharedKVStateRows.bytes_per_row(self.ratios)
        # Index reserved pages supply safe per-row scratch destinations too.
        row_bytes += self.ratios.count(4)*(self.window_size//self.page_size)*self.page_size*68
        self.scratch_rows = self.max_buffer_rows
        live_rows = self.max_buffer_rows
        self.prefix_rows = 0
        if self.enable_prefix_caching and getattr(self.config, "startup_cache_phase", "") != "profiling":
            requested = self.config.prefix_cache_max_blocks or max(16, live_rows)
            self.prefix_rows = min(int(requested), max(0, (budget//4)//row_bytes))
            if self.prefix_rows < 1:
                raise MemoryError("Native prefix cache cannot afford one window/carry snapshot")
        self.num_rows = self.scratch_rows + live_rows + self.prefix_rows
        reserved_pages = self.num_rows*(self.window_size//self.page_size)
        page_bytes = ((584*self.page_size+575)//576)*576
        table_bytes = self.num_rows*sum(ceil(self.max_model_len/r)*4 for r in set(self.ratios) if r)
        remaining = budget-self.num_rows*row_bytes-table_bytes-self.ratios.count(0)*page_bytes
        if remaining <= 0:
            raise MemoryError("Native KV budget cannot afford window/carry rows and address tables")
        unit_bytes = sum(self.ratios.count(r)*ceil(128/r)
                         *(page_bytes+(self.page_size*68 if r == 4 else 0))
                         for r in (4, 128) if r in self.ratios)
        units = remaining//unit_bytes if unit_bytes else 0
        self.families = {}
        self.pools = {}
        windows = {}
        self.family_slots = {}
        for ratio in (4, 128):
            layers = [i for i, r in enumerate(self.ratios) if r == ratio]
            if not layers:
                continue
            pages = max(1, units*ceil(128/ratio))
            if pages*sum(page_bytes+(self.page_size*68 if ratio == 4 else 0) for _ in layers) > remaining:
                raise MemoryError("Native KV budget cannot afford one compressed page per family")
            family = CompressedKVFamily(layer_ids=layers, with_index=ratio == 4,
                                        num_pages=reserved_pages+pages, reserved_pages=reserved_pages,
                                        page_size=self.page_size, device=self.device)
            self.families[ratio] = family
            self.pools.update(family.attention)
            self.family_slots[ratio] = torch.zeros(self.num_rows, ceil(self.max_model_len/ratio),
                                                   dtype=torch.int32, device=self.device)
            for layer, pool in family.attention.items():
                windows[layer] = pool.byte_storage[:reserved_pages].view(self.num_rows, -1)
        for layer, ratio in enumerate(self.ratios):
            if ratio:
                continue
            pool = PackedSharedKVPool(num_pages=reserved_pages+1, page_size=self.page_size,
                                      reserved_pages=reserved_pages, device=self.device)
            self.pools[layer] = pool
            windows[layer] = pool.byte_storage[:reserved_pages].view(self.num_rows, -1)
        self.state_rows = SharedKVStateRows(num_rows=self.num_rows, reserved_rows=self.scratch_rows,
                                           compress_ratios=self.ratios, device=self.device,
                                           window_storage=windows)
        self.config.num_kvcache_slots = max(1, self.num_free_slots)

    def set_model_layers(self, layers):
        # Bind physical planning hooks once. Cache owns every resulting plan and buffer.
        for layer in layers:
            for compressor in (getattr(layer.attn, "compressor", None),
                               getattr(getattr(layer.attn, "indexer", None), "compressor", None)):
                if compressor is not None:
                    self.compression_planners[compressor.provider.spec.ratio] = compressor.provider.kernels.prefill_plan

    @property
    def num_free_slots(self):
        return min((f.allocator.num_free_pages*self.page_size*r for r, f in self.families.items()),
                   default=self.max_model_len*self.max_buffer_rows)

    def get_layer_batch_states(self, layer_idx):
        return self.layer_batch_state

    def get_layer_kv_cache(self, layer_idx):
        raise TypeError("Native shared KV exposes typed packed payloads")

    def get_layer_store_view(self, layer_idx):
        raise TypeError("Native shared KV uses shared-KV store views")

    def get_layer_compute_tensors(self, layer_idx, selection=None):
        raise TypeError("Native shared KV exposes typed candidate views")

    def get_layer_buffer_req_to_token_slots(self, layer_idx):
        ratio = self.ratios[layer_idx]
        if not ratio:
            raise TypeError("Window-only shared KV has no dense token table")
        return self.family_slots[ratio]

    def _free_prefix_payload(self, payload):
        for ratio, lease in payload.leases.items():
            self.families[ratio].release(lease)
        self.state_rows.release(payload.row)

    def _evict_one_prefix(self):
        if self.prefix_cache is None:
            return False
        before = len(self.prefix_cache)
        blocks = self.prefix_cache.evict_until_freeable(1)
        for block in blocks:
            self._free_prefix_payload(block.payload)
        return len(self.prefix_cache) < before

    def _new_request(self, seq_id):
        if seq_id not in self.requests:
            while not self.state_rows.num_free_rows and self._evict_one_prefix():
                pass
            row = self.state_rows.allocate()
            self.requests[seq_id] = NativeSharedKVRequest(row, leases={r: f.new_lease() for r, f in self.families.items()})
        return self.requests[seq_id]

    def _reserve(self, request, length):
        if length > self.max_model_len:
            raise ValueError("Native shared KV exceeds max_model_len")
        for ratio, family in self.families.items():
            lease = request.leases[ratio]
            target = length//ratio
            while family.reservation_pages(lease, target) > family.allocator.num_free_pages:
                if not self._evict_one_prefix():
                    raise MemoryError("Native compressed KV family exhausted")
            family.reserve(lease, target)
            slots = family.physical_slots(lease, 0, target)
            if slots:
                self.family_slots[ratio][request.row, :target].copy_(
                    torch.tensor(slots, device=self.device, dtype=torch.int32))

    def _device(self, values, dtype=torch.int32):
        host = torch.tensor(values, dtype=dtype)
        if self.device.type == "cuda":
            host = host.pin_memory()
        return host.to(self.device, non_blocking=self.device.type == "cuda")

    def _prepare_prefill(self, seqs):
        self.step_is_prefill = True
        self.compression_views = {}
        input_ids, positions, rows, cu, histories, persistent, layouts = [], [], [], [0], [], [], []
        for seq in seqs:
            self._attach_prefix_cache_if_needed(seq)
            request = self._new_request(seq.seq_id)
            start, size = int(seq.num_prefilled_tokens), int(seq.current_chunk_size)
            if request.length != start:
                raise ValueError("Native KV prefill length differs from scheduler position")
            end, offset = start+size, len(input_ids)
            self._reserve(request, end)
            self._record_prefix_materialization(seq, [], None)
            history_len = min(start, self.window_size-1)
            histories.extend(request.row*self.window_size+(p%self.window_size) for p in range(start-history_len, start))
            layouts.append((offset, size, request.row, start, end, history_len))
            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            rows.extend([request.row]*size)
            persistent.extend(range(offset+max(0, size-self.window_size), offset+size))
            cu.append(len(input_ids))
            request.length = end
        self.prefill_layouts = tuple(layouts)
        self.query_positions = self._device(positions)
        self.query_rows = self._device(rows)
        self.history_slots = self._device(histories)
        self.persist_indices = self._device(persistent, torch.int64)
        self.window_write_slots = self.query_rows*self.window_size + self.query_positions.remainder(self.window_size)
        tokens = len(input_ids)
        self.prefill_pool = PackedSharedKVPool(num_pages=max(2, 1+ceil(tokens/self.page_size)),
                                               reserved_pages=1, page_size=self.page_size, device=self.device)
        self.prefill_write_slots = torch.arange(tokens, device=self.device, dtype=torch.int32)+self.page_size
        self.current_kv = torch.empty(tokens, 1, 512, device=self.device, dtype=torch.bfloat16)
        self.history_kv = torch.empty(len(histories), 1, 512, device=self.device, dtype=torch.bfloat16)
        self.compressed_prefill_slots = {}
        self.compressed_prefill_offsets = {}
        self.compression_plans = {}
        self.compression_store_slots = {}
        for ratio in self.families:
            all_slots, offsets, output_slots, offset = [], [], [], 0
            for _, _, row, start, end, _ in layouts:
                lease = self.requests[next(s.seq_id for s in seqs if self.requests[s.seq_id].row == row)].leases[ratio]
                count = end//ratio
                offsets.append(offset)
                all_slots.extend(self.families[ratio].physical_slots(lease, 0, count))
                output_slots.extend(self.families[ratio].physical_slots(lease, start//ratio, count))
                offset += count
            self.compressed_prefill_slots[ratio] = self._device(all_slots)
            self.compressed_prefill_offsets[ratio] = tuple(offsets)
            self.compression_store_slots[ratio] = self._device(output_slots)
            lengths = torch.tensor([v[4] for v in layouts], dtype=torch.int64)
            extends = torch.tensor([v[1] for v in layouts], dtype=torch.int64)
            self.compression_plans[ratio] = self.compression_planners[ratio](
                lengths, extends, num_tokens=tokens, device=self.device)
        self.step_seq_lens = self._device([v[4] for v in layouts])
        self.step_request_rows = self._device([v[2] for v in layouts])
        self.layer_batch_state = LayerBatchStates(slot_mapping=self.window_write_slots,
                                                  context_lens=self.step_seq_lens,
                                                  req_indices=self.step_request_rows,
                                                  max_context_len=max((v[4] for v in layouts), default=0))
        return self._device(input_ids, torch.int64), self.query_positions.to(torch.int64), self._device(cu)

    def _prepare_decode(self, seqs):
        capacity = len(seqs)
        inputs = [torch.empty(capacity, dtype=dtype, device=self.device)
                  for dtype in (torch.int64, torch.int64, torch.int32, torch.int32, torch.int32)]
        self.prepare_decode_static(seqs, *inputs)
        return inputs[0], inputs[1], None

    def prepare_decode_static(self, seqs, input_ids, positions, slot_mapping, context_lens, req_indices):
        self.step_is_prefill = False
        self.compression_views = {}
        capacity = input_ids.numel()
        ids, pos, lengths, rows = [0]*capacity, [0]*capacity, [1]*capacity, list(range(capacity))
        for i, seq in enumerate(seqs):
            request = self._new_request(seq.seq_id)
            position = int(seq.decode_input_position)
            self._reserve(request, position+1)
            ids[i], pos[i], lengths[i], rows[i] = int(seq.decode_input_token), position, position+1, request.row
            request.length = position+1
        for out, values in ((input_ids, ids), (positions, pos), (context_lens, lengths), (req_indices, rows)):
            out.copy_(self._device(values, out.dtype))
        slot_mapping.copy_(req_indices*self.window_size+positions.to(torch.int32).remainder(self.window_size))
        self.query_positions = positions
        self.query_rows = self.step_request_rows = req_indices
        self.step_seq_lens = context_lens
        self.window_write_slots = slot_mapping
        self.layer_batch_state = LayerBatchStates(slot_mapping=slot_mapping, context_lens=context_lens,
                                                  req_indices=req_indices, max_context_len=self.max_model_len)
        return input_ids, positions, None

    def compression_compute_view(self, layer_idx, *, index=False):
        key = (layer_idx, index)
        if key not in self.compression_views:
            ratio = self.ratios[layer_idx]
            pool = self.state_rows.index_carry[layer_idx] if index else self.state_rows.carry[layer_idx]
            plan = self.compression_plans[ratio] if self.step_is_prefill else None
            # Upstream prefill writes at ragged query-token indices. The
            # provider compacts completed groups only after normalization.
            rows = len(self.query_rows)
            self.compression_views[key] = CompressionComputeView(
                carry=pool.state, rows=self.step_request_rows, seq_lens=self.step_seq_lens,
                output=torch.empty(rows, pool.head_dim, dtype=torch.float32, device=self.device),
                prefill_plan=plan,
            )
        return self.compression_views[key]

    def compressed_write_slots(self, layer_idx):
        ratio = self.ratios[layer_idx]
        if self.step_is_prefill:
            return self.compression_store_slots[ratio]
        ordinal = (self.step_seq_lens//ratio-1).clamp_min(0).to(torch.int64)
        slots = self.family_slots[ratio][self.query_rows.long(), ordinal]
        # Incomplete groups and padding publish only into their row's reserved scratch page.
        return torch.where(self.step_seq_lens.remainder(ratio) == 0, slots,
                           (torch.arange(len(self.query_rows), device=self.device)*self.window_size+self.window_size-1)).to(torch.int32)

    def index_cache(self, layer_idx):
        return self.families[4].index[layer_idx].byte_storage

    def on_forward_end(self, seqs, is_prefill):
        for seq in seqs:
            request = self.requests[seq.seq_id]
            for ratio, lease in request.leases.items():
                self.families[ratio].mark_materialized(lease, request.length//ratio)
        if is_prefill and self.enable_prefix_caching:
            for seq in seqs:
                self._freeze_prefix_snapshot(seq)
            if getattr(self, "_async_prefix_records", None) is None:
                self.publish_pending_prefix_blocks(seqs)
        super().on_forward_end(seqs, is_prefill)

    def free_seq(self, seq_id):
        request = self.requests.pop(seq_id, None)
        if request is None:
            return
        for ratio, lease in request.leases.items():
            self.families[ratio].release(lease)
        self.state_rows.release(request.row)
        for block in request.prefix_blocks:
            self.prefix_cache.release_block_ref(block)
        self.pending_prefix.pop(seq_id, None)
        for record in self.private_prefix_records.pop(seq_id, set()):
            self._free_prefix_payload(record.payload)

    def free_part_slots(self, layer_idx, seq, keep_indices):
        raise TypeError("Native shared KV does not prune its physical compression history")

    def refresh_prefix_cache_hit(self, seq):
        self.clear_prefix_cache_hit(seq)
        seq.prefix_cache_enabled = self.enable_prefix_caching
        seq.prefix_cache_block_size = self.prefix_cache_block_size
        seq.prefix_cache_method = "deepseek_v4"
        if self.prefix_cache is None or seq.seq_id in self.requests:
            return
        usable = usable_prefix_cache_tokens(seq.num_prompt_tokens, self.prefix_cache_block_size)
        hit, last, blocks = self.prefix_cache.match_longest_prefix(seq.token_ids, max_usable_tokens=usable)
        seq.prefix_cache_hit_len, seq.prefix_cache_hit_last_block_id, seq.prefix_cache_hit_block_count = hit, last, blocks

    def _attach_prefix_cache_if_needed(self, seq):
        if seq.seq_id in self.requests or not seq.prefix_cache_hit_len or self.prefix_cache is None:
            return
        self.refresh_prefix_cache_hit(seq)
        if not seq.prefix_cache_hit_len:
            return
        chain = self.prefix_cache.get_chain(seq.prefix_cache_hit_last_block_id, seq.prefix_cache_hit_block_count)
        for block in chain:
            self.prefix_cache.acquire_block_ref(block)
        payload = chain[-1].payload
        row, leases = None, {}
        try:
            row = self.state_rows.copy(payload.row)
            for ratio, lease in payload.leases.items():
                leases[ratio] = self.families[ratio].snapshot(lease)
            request = NativeSharedKVRequest(row, payload.length, leases, chain)
            self.requests[seq.seq_id] = request
            self._reserve(request, payload.length)
            seq.num_prefilled_tokens = payload.length
        except BaseException:
            self.requests.pop(seq.seq_id, None)
            for ratio, lease in leases.items():
                self.families[ratio].release(lease)
            if row is not None:
                self.state_rows.release(row)
            for block in chain:
                self.prefix_cache.release_block_ref(block)
            raise

    def _record_prefix_materialization(self, seq, token_ids, slots):
        # Window/carry state is frozen only after every layer has completed.
        return

    def _freeze_prefix_snapshot(self, seq):
        request = self.requests[seq.seq_id]
        size = self.prefix_cache_block_size
        length = request.length
        if length % size or length > seq.num_prompt_tokens:
            return
        index = length//size-1
        private = self.private_prefix_records.setdefault(seq.seq_id, set())
        if len(request.prefix_blocks)+len(private) != index:
            return
        # Preserve row headroom for admitted requests. Snapshots are optional.
        if self.state_rows.num_free_rows <= self.max_buffer_rows-len(self.requests):
            return
        block_ids = self.prefix_cache.block_ids_for_tokens(seq.token_ids, max_tokens=length)
        parent = block_ids[-2] if len(block_ids) > 1 else None
        tokens = tuple(seq.token_ids[index*size:length])
        row, leases = self.state_rows.copy(request.row), {}
        try:
            for ratio, lease in request.leases.items():
                leases[ratio] = self.families[ratio].snapshot(lease)
            record = NativePrefixRecord(NativeSharedKVPrefix(row, length, leases),
                                         block_ids[-1], parent, index, tokens)
        except BaseException:
            for ratio, lease in leases.items():
                self.families[ratio].release(lease)
            self.state_rows.release(row)
            raise
        private.add(record)
        asynchronous = getattr(self, "_async_prefix_records", None)
        if asynchronous is not None:
            asynchronous.append((seq, list(tokens), record))
        else:
            self._record_frozen_prefix_materialization(seq, list(tokens), record)

    def _record_frozen_prefix_materialization(self, seq, token_ids, record):
        if not isinstance(record, NativePrefixRecord):
            raise TypeError("Native prefix publication requires a frozen window/carry snapshot")
        if tuple(token_ids) != record.tokens:
            raise ValueError("Retired prefix tokens differ from the frozen snapshot")
        if record not in self.private_prefix_records.get(seq.seq_id, set()):
            raise ValueError("Prefix snapshot is not owned by this request")
        self.pending_prefix.setdefault(seq.seq_id, []).append(record)

    def publish_pending_prefix_blocks(self, seqs):
        if self.prefix_cache is None:
            return
        for seq in seqs:
            request = self.requests.get(seq.seq_id)
            for record in self.pending_prefix.pop(seq.seq_id, []):
                private = self.private_prefix_records[seq.seq_id]
                private.remove(record)
                if request is None or len(request.prefix_blocks) != record.index:
                    self._free_prefix_payload(record.payload)
                    continue
                existing = self.prefix_cache.get_block(record.block_id)
                if existing is not None:
                    self.prefix_cache.acquire_block_ref(existing)
                    request.prefix_blocks.append(existing)
                    self._free_prefix_payload(record.payload)
                    continue
                while len(self.prefix_cache) >= self.prefix_rows and self._evict_one_prefix():
                    pass
                if len(self.prefix_cache) >= self.prefix_rows:
                    self._free_prefix_payload(record.payload)
                    continue
                block = PrefixCacheBlock(record.block_id, record.parent_id, self.prefix_cache_block_size,
                                         record.index, record.payload, record.tokens, ref_count=1)
                try:
                    inserted = self.prefix_cache.insert_block(block)
                    request.prefix_blocks.append(inserted)
                except BaseException:
                    self._free_prefix_payload(record.payload)
                    raise

    def reset_prefix_cache(self):
        if self.requests:
            raise RuntimeError("Native prefix reset requires drained requests")
        if self.prefix_cache is not None:
            for block in self.prefix_cache.blocks.values():
                if block.ref_count:
                    raise RuntimeError("Native prefix reset requires unreferenced blocks")
                self._free_prefix_payload(block.payload)
            self.prefix_cache = RadixPrefixIndex(
                block_size=self.prefix_cache_block_size,
                fingerprint=build_prefix_cache_fingerprint(self.config, self.prefix_cache_block_size),
                max_blocks=self.prefix_rows,
            )

    def prefill_step_free_slots_for(self, seq):
        start = self._request_length(seq)
        remaining = min(seq.num_prompt_tokens-start,
                        self.num_free_slots+self.prefill_private_slots_for(seq))
        if self.enable_prefix_caching:
            return min(remaining, self.prefix_cache_block_size-start%self.prefix_cache_block_size)
        return min(remaining, self.config.max_num_batched_tokens)

    def _request_length(self, seq):
        request = self.requests.get(seq.seq_id)
        return (request.length if request is not None else
                max(seq.num_prefilled_tokens, seq.prefix_cache_hit_len))

    def _append_page_costs(self, seq, tokens):
        request = self.requests.get(seq.seq_id)
        end = self._request_length(seq)+int(tokens)
        costs = {}
        for ratio, family in self.families.items():
            if request is None:
                pages = ceil((end//ratio)/self.page_size)
            else:
                pages = family.reservation_pages(request.leases[ratio], end//ratio)
            costs[f"ratio_{ratio}"] = pages
        return costs

    def _page_cost_to_slots(self, costs):
        return max((int(costs.get(f"ratio_{ratio}", 0))*ratio*self.page_size
                    for ratio in self.families), default=0)

    def prefill_step_reservation_cost(self, seq, scheduled_tokens):
        return self._page_cost_to_slots(self._append_page_costs(seq, scheduled_tokens))

    def decode_step_reservation_cost(self, seq):
        return self.prefill_step_reservation_cost(seq, 1)

    def decode_window_costs(self, seq, tokens):
        return self._append_page_costs(seq, tokens)

    def decode_window_budgets(self):
        return {f"ratio_{ratio}": family.allocator.num_free_pages
                for ratio, family in self.families.items()}

    def prefill_capacity_after_decode_reservations(self, free_slots, reserved, *, admission):
        capacity = min(((family.allocator.num_free_pages-int(reserved.get(f"ratio_{ratio}", 0)))
                        *ratio*self.page_size for ratio, family in self.families.items()),
                       default=free_slots)
        return min(int(free_slots), capacity)

    def prefill_private_slots_for(self, seq):
        request = self.requests.get(seq.seq_id)
        if request is None:
            return 0
        capacities = []
        for ratio, family in self.families.items():
            lease = request.leases[ratio]
            capacity = len(lease.pages)*self.page_size*ratio-request.length
            tail = lease.materialized_tokens%self.page_size
            if tail and family.allocator.reference_count(
                    lease.pages[lease.materialized_tokens//self.page_size]) > 1:
                capacity = min(capacity, ratio-1-request.length%ratio)
            capacities.append(max(0, capacity))
        return min(capacities, default=self.max_model_len-request.length)

    def decode_step_free_slots_for(self, seq):
        return self.num_free_slots+self.prefill_private_slots_for(seq)

    def prompt_admission_budgets(self, waiting_seqs, engine_prefill_chunk_size):
        budgets = self.decode_window_budgets()
        for seq in waiting_seqs:
            if seq.seq_id in self.requests:
                remaining = max(0, seq.num_prompt_tokens-self._request_length(seq))
                for name, cost in self._append_page_costs(seq, remaining).items():
                    budgets[name] = max(0, budgets[name]-cost)
        budgets["rows"] = min(self.max_buffer_rows-len(self.requests), self.state_rows.num_free_rows)
        return budgets

    def prompt_admission_costs(self, seq):
        length = seq.num_prompt_tokens
        # Conservatively account for the final page and prefix tail COW.
        return {**{f"ratio_{r}": ceil((length//r)/self.page_size) for r in self.families},
                "rows": int(seq.seq_id not in self.requests)}

    def free_slot_stats(self):
        return {"free_slots": self.num_free_slots, "free_rows": self.state_rows.num_free_rows,
                **{f"free_pages_ratio_{r}": f.allocator.num_free_pages for r, f in self.families.items()}}

    def debug_live_seq_slots(self):
        return {seq_id: request.length for seq_id, request in self.requests.items()}


    def init_decode_graph_state(self, contract, inputs):
        state = super().init_decode_graph_state(contract, inputs)
        state.compression_views = {}
        for layer, ratio in enumerate(self.ratios):
            for index in ((False, True) if ratio == 4 else (False,) if ratio else ()):
                pool = self.state_rows.index_carry[layer] if index else self.state_rows.carry[layer]
                state.compression_views[layer, index] = CompressionComputeView(
                    carry=pool.state, rows=inputs.request_indices, seq_lens=inputs.context_lens,
                    output=torch.empty(contract.batch_capacity, pool.head_dim,
                                       dtype=torch.float32, device=self.device),
                )
        return state

    def prepare_decode_graph_step(self, seqs, state):
        result = super().prepare_decode_graph_step(seqs, state)
        self.compression_views = state.compression_views
        return result

    def decode_graph_state_keepalive_tensors(self, state):
        return [*state.inputs.keepalive_tensors(),
                *(view.output for view in state.compression_views.values()),
                *self.state_rows.accounting_tensors(), *self.family_slots.values(),
                *(pool.byte_storage for pool in self.pools.values()),
                *(pool.byte_storage for family in self.families.values() for pool in family.index.values())]

    def update_window(self, layer_idx, values, norm_weight, freqs, transforms):
        arena = self.pools[layer_idx].byte_storage
        if not self.step_is_prefill:
            transforms.normalize_rotate_store(values, norm_weight, freqs, self.query_positions,
                                                self.window_write_slots, arena)
            return
        # Capture the previous window before this chunk can overwrite its ring.
        transforms.gather(arena, self.history_slots, out=self.history_kv)
        transforms.normalize_rotate_store(values, norm_weight, freqs, self.query_positions,
                                            self.prefill_write_slots, self.prefill_pool.byte_storage)
        transforms.gather(self.prefill_pool.byte_storage, self.prefill_write_slots, out=self.current_kv)
        keep = self.persist_indices
        transforms.normalize_rotate_store(values[keep], norm_weight, freqs, self.query_positions[keep],
                                            self.window_write_slots[keep], arena)

    def candidate_view(self, layer_idx, transforms):
        ratio = self.ratios[layer_idx]
        positions, rows = self.query_positions, self.query_rows
        columns = torch.arange(self.window_size, device=self.device, dtype=torch.int32)
        logical_window = (positions[:, None].to(torch.int32)-self.window_size+1).clamp_min(0)+columns
        index_pages = self.index_cache(layer_idx) if ratio == 4 else None
        if not self.step_is_prefill:
            window = rows[:, None]*self.window_size + logical_window.remainder(self.window_size)
            compressed = (self.family_slots[ratio][rows.long()] if ratio else
                          torch.empty(len(rows), 0, device=self.device, dtype=torch.int32))
            return SharedKVCandidateView(positions, rows, window.to(torch.int32), compressed,
                                         index_pages, compressed, logical_window,
                                         self.pools[layer_idx].layer_payload())
        history_offset = 0
        windows = []
        width = max((v[4]//ratio for v in self.prefill_layouts), default=0) if ratio else 0
        compressed = torch.zeros(len(rows), width, device=self.device, dtype=torch.int32)
        total_history, total_tokens = len(self.history_slots), len(rows)
        for i, (offset, size, row, start, end, history_len) in enumerate(self.prefill_layouts):
            logical = logical_window[offset:offset+size]
            windows.append(torch.where(logical < start, history_offset+logical-start+history_len,
                                       total_history+offset+logical-start))
            if ratio:
                count = end//ratio
                compressed[offset:offset+size, :count] = (total_history+total_tokens
                    +self.compressed_prefill_offsets[ratio][i]
                    +torch.arange(count, device=self.device, dtype=torch.int32))
            history_offset += history_len
        window = torch.cat(windows) if windows else torch.empty(0, self.window_size, device=self.device, dtype=torch.int32)
        index_slots = (self.family_slots[ratio][rows.long(), :width].contiguous() if ratio else compressed)
        parts = [self.history_kv, self.current_kv]
        if ratio:
            parts.append(transforms.gather(self.pools[layer_idx].byte_storage, self.compressed_prefill_slots[ratio]))
        return SharedKVCandidateView(positions, rows, window.to(torch.int32), compressed.contiguous(),
                                     index_pages, index_slots, logical_window,
                                     SharedKVPayload(torch.cat(parts)))


    def index_selection_workspace(self, rows, width):
        key = (rows, width)
        if self.step_is_prefill:
            return (torch.empty(rows, width, dtype=torch.float32, device=self.device),
                    torch.empty(rows, 512, dtype=torch.int32, device=self.device))
        if key not in self.decode_selection_workspaces:
            self.decode_selection_workspaces[key] = (
                torch.empty(rows, width, dtype=torch.float32, device=self.device),
                torch.empty(rows, 512, dtype=torch.int32, device=self.device))
        return self.decode_selection_workspaces[key]


    def _iter_accounting_tensors(self):
        for layer, pool in self.pools.items():
            yield f"native_kv_cache.layer_{layer}", pool.byte_storage
        for layer, pool in self.state_rows.carry.items():
            yield f"native_carry.layer_{layer}", pool.state
        for layer, pool in self.state_rows.index_carry.items():
            yield f"native_index_carry.layer_{layer}", pool.state
        for family in self.families.values():
            for layer, pool in family.index.items():
                yield f"native_index_cache.layer_{layer}", pool.byte_storage
        for ratio, slots in self.family_slots.items():
            yield f"native_address_tables.ratio_{ratio}", slots
