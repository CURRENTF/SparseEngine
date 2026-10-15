"""Exclusive chain cache with independently allocated KV-head slots."""

from dataclasses import dataclass

import numpy as np
import torch

from sparseengine.engine.decode_graph_contract import CacheDecodeGraphState
from sparseengine.kernels.triton.head_slots import publish_head_slots, record_head_queries
from sparseengine.kernels.triton.prefill_score import PrefillScoreWorkspace, prefill_score_fwd
from sparseengine.utils.profiler import profiler

from .base import CacheManager, DecodeComputeView, HeadSlotViewMeta, LayerBatchStates, PrefillComputeView
from .head_slot_allocator import HeadSlotAllocator
from .storage import create_attention_cache_storage


def head_slot_auxiliary_bytes(config) -> int:
    """Conservative live metadata, scoring and selection envelope, excluding KV."""
    l, h = config.runtime_layout.num_kv_layers, config.hf_config.num_key_value_heads
    if config.sparse_method not in {"snapkv", "adakv"}:
        from sparseengine.models.layout import resolve_attention_qk_head_dim
        d = resolve_attention_qk_head_dim(config.hf_config)
        r, t = config.max_num_seqs_in_gpu + 1, config.max_model_len
        return (l * r * h * (t * 8 + 4)
                + l * config.max_num_batched_tokens * h * 8
                + 4 * config.max_num_batched_tokens * config.hf_config.num_attention_heads * d)
    q = config.hf_config.num_attention_heads
    from sparseengine.models.layout import resolve_attention_qk_head_dim
    d = (config.runtime_layout.kv_head_dims[0] if config.runtime_layout.kv_head_dims
         else resolve_attention_qk_head_dim(config.hf_config))
    r, t, w = config.max_num_seqs_in_gpu + 1, config.max_model_len, config.observation_window_size
    window_tile = max(16, 1 << (w - 1).bit_length())
    metadata = l * r * h * t * 8 + l * r * h * 4
    queries = l * r * w * q * d * 2
    # LSE partials round GQA head groups up; selections across all layers remain
    # live until every layer has passed validation, then mappings are published.
    score = 2 * q * window_tile * ((t + 63) // 64 + 1) * 4
    score += h * t * 4  # Native per-KV-head FP32 score output.
    selection = l * h * t * 16 + h * t * 8 + w * q * d * 4
    if config.sparse_method == "adakv":
        # Live score/masks, per-head and global int64 orderings, sorting scratch.
        selection += 64 * h * t
    append = l * config.max_num_batched_tokens * h * 8
    # Graph-private buffers and captured attention/model outputs are measured by
    # startup's CUDA Graph phase and deducted from the global KV budget once.
    return metadata + queries + score + selection + append


@dataclass
class HeadSlotGraphState(CacheDecodeGraphState):
    writes: torch.Tensor
    lengths: torch.Tensor
    query_keep: torch.Tensor


class HeadSlotCacheManager(CacheManager):
    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        super().__init__(config, parallel_context, allocation_budget_bytes=allocation_budget_bytes)
        self.attention_cache_storage = create_attention_cache_storage(
            config, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim)
        self.allocate_kv_cache()
        l, r, h, t = self.num_kv_layers, self.max_buffer_rows + 1, self.num_kv_heads, self.max_model_len
        self.allocator = HeadSlotAllocator((self.config.num_kvcache_slots,) * l,
                                           h, t, self.max_buffer_rows)
        self.slot_table = torch.zeros((l, r, h, t), dtype=torch.int32, device=self.device)
        self.original_positions = torch.full_like(self.slot_table, -1)
        self.valid_lengths = torch.zeros((l, r, h), dtype=torch.int32, device=self.device)
        self.compression_enabled = (config.sparse_method == "adakv"
                                    or (config.sparse_method == "snapkv"
                                        and config.head_slot_budgets is not None))
        self.query_ring = (torch.zeros((l, r, config.observation_window_size,
                                       self.hf_config.num_attention_heads, self.head_dim),
                                      dtype=self.hf_config.dtype, device=self.device)
                           if self.compression_enabled else None)
        self.query_counts: dict[int, int] = {}
        self.last_positions: dict[int, int] = {}
        self.layer_batch_states = [LayerBatchStates() for _ in range(l)]
        self._head_lengths = None
        self._token_rows = self._positions = self._query_keep = None
        self._chain_coordinator = None
        self.chain_selector = None
        self.score_workspace = PrefillScoreWorkspace()
        self.last_compaction: dict[int, dict] = {}
        self.compaction_stats = {"prefill": 0, "chain_turn": 0, "released_head_slots": 0,
                                 "nonuniform_layers": 0}


    def allocate_kv_cache(self):
        available, slot_bytes = self._get_available_slots_info()
        self.auxiliary_reserved_bytes = head_slot_auxiliary_bytes(self.config)
        capacity = (available - self.auxiliary_reserved_bytes) // (self.num_kv_layers * slot_bytes)
        capacity = min(capacity, self.max_buffer_rows * self.num_kv_heads * self.max_model_len)
        if capacity < self.num_kv_heads:
            raise RuntimeError("Insufficient memory after head-slot metadata/workspace reservation.")
        self.config.num_kvcache_slots = int(capacity)
        self.attention_cache_storage.allocate(num_layers=self.num_kv_layers, num_slots=int(capacity),
                                               device=self.device)
        self.kv_cache = self.attention_cache_storage.cache

    def attention_cache_bytes_per_slot_per_layer(self):
        return self.attention_cache_storage.bytes_per_slot_per_layer()

    def bind_chain_cache(self, coordinator):
        self._chain_coordinator = coordinator

    def _outstanding(self, exclude_seq_ids=()):
        if self._chain_coordinator is None:
            return (0,) * self.num_kv_layers
        slots, _ = self._chain_coordinator._outstanding_active_reservations(
            exclude_seq_ids=frozenset(exclude_seq_ids))
        return slots or (0,) * self.num_kv_layers

    @property
    def scratch_slot_reserve(self):
        return 0

    @property
    def num_free_slots(self):
        # The scheduler unit is common logical append tokens, never head slots.
        return max(0, min((int(f) - r) // self.num_kv_heads
                          for f, r in zip(self.allocator.free_counts, self._outstanding()))
                      - self.scratch_slot_reserve)

    def prefill_step_free_slots(self):
        # Executing a step may realize already admitted chain reservations.
        return max(0, int(self.allocator.free_counts.min()) // self.num_kv_heads
                   - self.scratch_slot_reserve)

    decode_step_free_slots = prefill_step_free_slots

    def prefill_step_free_slots_for(self, seq):
        # Replay scratch belongs to the shared pools, not every resident row.
        # Subtract it before append_capacity applies the independent row limit.
        reserved = tuple(r + self.scratch_slot_reserve * self.num_kv_heads
                         for r in self._outstanding((seq.seq_id,)))
        return self.allocator.append_capacity(seq.seq_id, reserved)

    decode_step_free_slots_for = prefill_step_free_slots_for

    def decode_window_budgets(self):
        # RuntimeState charges pending chain promises once, alongside decode
        # reservations. Expose raw pools here, in physical head-slot units.
        return {f"layer_{layer}": max(0, int(free) - self.scratch_slot_reserve * self.num_kv_heads)
                for layer, free in zip(self.kv_transformer_layer_indices(), self.allocator.free_counts)}

    def decode_window_costs(self, seq, tokens):
        return {f"layer_{layer}": int(tokens) * self.num_kv_heads
                for layer in self.kv_transformer_layer_indices()}

    def prefill_capacity_after_decode_reservations(self, free_slots, reserved, *, admission):
        # Decode windows own physical head slots; scheduler capacities count
        # logical tokens, each of which appends every KV head in every layer.
        head_slots = max(reserved.values(), default=0)
        return int(free_slots) - (head_slots + self.num_kv_heads - 1) // self.num_kv_heads

    def prefill_reservation_owned_by_chain(self, seq):
        return seq.chain_id is not None

    def reserved_prefill_slots(self, waiting_seqs, engine_prefill_chunk_size):
        return sum(self.remaining_prefill_tokens(seq) for seq in waiting_seqs
                   if not self.prefill_reservation_owned_by_chain(seq)
                   and 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens)

    def prompt_admission_budgets(self, waiting_seqs, engine_prefill_chunk_size):
        reserved = self.reserved_prefill_slots(waiting_seqs, engine_prefill_chunk_size) * self.num_kv_heads
        return {name: max(0, free - reserved) for name, free in self.decode_window_budgets().items()}

    def prompt_admission_costs(self, seq):
        cost = self.prompt_admission_cost(seq) * self.num_kv_heads
        return {name: cost for name in self.decode_window_budgets()}

    def prompt_admission_cost(self, seq):
        if seq.chain_id is not None:
            return 0  # The chain coordinator already reserved the prompt suffix.
        return self.remaining_prefill_tokens(seq)

    def prompt_admission_failure_action(self):
        return "raise"

    def chain_capacity_deficits(self, *, suffix_tokens, generation_tokens=0,
                               existing_slots_by_layer=(), outstanding_reserved_slots_by_layer=(),
                               outstanding_reserved_rows=0, needs_resident_row):
        need = (int(suffix_tokens) + int(generation_tokens)) * self.num_kv_heads
        reserved = outstanding_reserved_slots_by_layer or (0,) * self.num_kv_layers
        deficits = tuple(max(0, need + reserved[l] - int(self.allocator.free_counts[l]))
                         for l in range(self.num_kv_layers))
        rows = int(needs_resident_row)
        return ((need,) * self.num_kv_layers, rows, deficits,
                max(0, rows + outstanding_reserved_rows - len(self.allocator.free_rows)))

    def chain_has_residency(self, seq_id):
        return seq_id in self.allocator.seq_rows

    def chain_physical_residency(self, seq_id):
        row = self.allocator.seq_rows.get(seq_id)
        if row is None:
            return (0,) * self.num_kv_layers
        return tuple(map(int, self.allocator.lengths[:, row].sum(axis=1)))

    def chain_physical_kv_len(self, layer_idx, seq_id):
        row = self.allocator.seq_rows[seq_id]
        return int(self.allocator.lengths[layer_idx, row].max())

    def get_layer_batch_states(self, layer_idx):
        return self.layer_batch_states[layer_idx]

    def get_layer_kv_cache(self, layer_idx):
        p = self.attention_cache_storage.layer_payload(layer_idx)
        return p.k_cache, p.v_cache

    def get_layer_store_view(self, layer_idx):
        return (*self.get_layer_kv_cache(layer_idx), self.layer_batch_states[layer_idx].slot_mapping)

    def get_layer_compute_tensors(self, layer_idx, selection=None):
        return self.get_layer_kv_cache(layer_idx)

    def get_layer_buffer_req_to_token_slots(self, layer_idx):
        return self.slot_table[layer_idx]

    def store_attention_payload(self, layer_idx, payload):
        slots = self.layer_batch_states[layer_idx].slot_mapping
        self.attention_cache_storage.store(layer_idx, slots, payload)
        return slots

    def _view(self, layer_idx, view_type):
        state = self.layer_batch_states[layer_idx]
        return view_type(HeadSlotViewMeta(
            active_slots=self.slot_table[layer_idx], req_indices=state.req_indices,
            context_lens=self._head_lengths[layer_idx], positions=self.original_positions[layer_idx],
            max_context_len=self.max_model_len), self.attention_cache_storage.layer_payload(layer_idx))

    def build_prefill_compute_view(self, layer_idx, k_current, v_current, selection):
        return self._view(layer_idx, PrefillComputeView)

    def build_decode_compute_view(self, layer_idx, q, selection, **kwargs):
        return self._view(layer_idx, DecodeComputeView)

    def _append(self, seqs, is_prefill):
        requests, tokens, all_positions, rows, keep = [], [], [], [], []
        cu = [0]
        for seq in seqs:
            if is_prefill:
                start, n = int(seq.num_prefilled_tokens), int(seq.current_chunk_size)
                token_ids = (seq.token_ids[start:start + n]
                             if len(seq.token_ids) > n else seq.token_ids)
                if len(token_ids) != n:
                    raise ValueError("Prefill token slice does not match the scheduled chunk.")
            else:
                start, n, token_ids = seq.decode_input_position, 1, [seq.decode_input_token]
            if self.prefill_step_free_slots_for(seq) < n:
                raise RuntimeError("Head-slot batch would consume another chain's reservation.")
            requests.append((seq.seq_id, list(range(start, start + n))))
            tokens.extend(token_ids)
            all_positions.extend(range(start, start + n))
            if self.compression_enabled:
                keep.extend(i >= n - self.config.observation_window_size for i in range(n))
            cu.append(cu[-1] + n)
        if len(tokens) > self.config.max_num_batched_tokens:
            raise RuntimeError("Head-slot batch exceeds the reserved append workspace envelope.")
        other_reserved = self._outstanding(seq.seq_id for seq in seqs)
        need = (len(tokens) + self.scratch_slot_reserve) * self.num_kv_heads
        if any(need + reserved > free for reserved, free
               in zip(other_reserved, self.allocator.free_counts)):
            raise RuntimeError("Whole batch exceeds unreserved plus its own reserved head slots.")
        plan = self.allocator.plan_append(requests)
        writes, columns = self.allocator.apply_append(plan)
        for seq, row, pos in zip(seqs, plan.rows, plan.positions):
            rows.extend([row] * len(pos))
            if self.compression_enabled:
                count = self.query_counts.get(seq.seq_id, 0)
                self.query_counts[seq.seq_id] = min(self.config.observation_window_size, count + len(pos))
                self.last_positions[seq.seq_id] = pos[-1]
        to_device = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=self.device)
        writes_gpu = to_device(writes, torch.int32)
        positions_gpu = to_device(all_positions, torch.int64)
        token_rows = to_device(rows, torch.int32)
        with profiler.record("head_slot_metadata"):
            publish_head_slots(self.slot_table, self.original_positions, writes_gpu,
                               to_device(columns, torch.int32), token_rows, positions_gpu)
            lengths = to_device(self.allocator.lengths[:, list(plan.rows)], torch.int32)
            self.valid_lengths[:, list(plan.rows)] = lengths
        return (to_device(tokens, torch.int64), positions_gpu,
                to_device(cu, torch.int32) if is_prefill else None,
                writes_gpu, lengths, to_device(plan.rows, torch.int32), token_rows,
                to_device(keep, torch.bool))

    def _bind(self, writes, lengths, reqs, token_rows, positions, query_keep):
        self._head_lengths = lengths
        self._token_rows, self._positions, self._query_keep = token_rows, positions, query_keep
        logical_lengths = lengths.max(dim=2).values
        for l, state in enumerate(self.layer_batch_states):
            state.slot_mapping = writes[l]
            state.context_lens = logical_lengths[l]
            state.max_context_len = self.max_model_len
            state.req_indices = reqs

    def _prepare_prefill(self, seqs):
        ids, pos, cu, writes, lens, req, rows, keep = self._append(seqs, True)
        self._bind(writes, lens, req, rows, pos, keep)
        return ids, pos, cu

    def _prepare_decode(self, seqs):
        ids, pos, cu, writes, lens, req, rows, keep = self._append(seqs, False)
        self._bind(writes, lens, req, rows, pos, keep)
        return ids, pos, cu

    def record_prefill_query(self, layer_idx, q, view, **kwargs):
        if self.compression_enabled:
            record_head_queries(q, self.query_ring[layer_idx], self._token_rows,
                                self._positions, self._query_keep)

    def record_decode_query(self, layer_idx, q):
        self.record_prefill_query(layer_idx, q, None)

    def score_chain_layer(self, seq_id, layer_idx):
        row, count = self.allocator.seq_rows[seq_id], self.query_counts[seq_id]
        if count <= 0:
            raise RuntimeError("Cannot score a chain without observed queries.")
        h = self.num_kv_heads
        end = self.last_positions[seq_id] + 1
        qpos = torch.arange(end - count, end, device=self.device) % self.config.observation_window_size
        queries = self.query_ring[layer_idx, row, qpos]
        lengths = self.valid_lengths[layer_idx, row:row + 1]
        width = int(self.allocator.lengths[layer_idx, row].max())
        score = torch.empty((1, h, width), dtype=torch.float32, device=self.device)
        req_idx = torch.full((1,), row, dtype=torch.int32, device=self.device)
        query_start = torch.zeros(1, dtype=torch.int32, device=self.device)
        cached_lengths = lengths - count
        with profiler.record("head_slot_scoring"):
            prefill_score_fwd(queries, self.get_layer_kv_cache(layer_idx)[0].unsqueeze(1), score,
                              req_idx, query_start, lengths, cached_lengths, count,
                              self.slot_table[layer_idx], cached_lengths,
                              lengths, score_mode=self.config.sparse_prefill_score_mode,
                              workspace=self.score_workspace)
        return score[0]

    def on_chain_turn_finished(self, seq_id, processed_token_count):
        if not self.compression_enabled:
            return
        if self.last_positions.get(seq_id) != processed_token_count - 1:
            raise RuntimeError("Chain finish boundary does not match the last physical append.")
        if self.query_counts.get(seq_id, 0):
            self.compact_head_slots(seq_id, reason="chain_turn")

    def compact_head_slots(self, seq_id, *, reason):
        before = self.chain_physical_residency(seq_id)
        released = (0,) * self.num_kv_layers
        if self.compression_enabled:
            if self.chain_selector is None:
                raise RuntimeError("Head-slot chain selector was not bound.")
            selections = self.chain_selector(seq_id)
            cpu = [[idx.cpu().numpy() for idx in sel.indices] for sel in selections]
            self.allocator.validate_selection(seq_id, cpu)
            with profiler.record("head_slot_compaction"):
                released = self.allocator.compact(seq_id, cpu)
                row = self.allocator.seq_rows[seq_id]
                self.slot_table[:, row].copy_(torch.from_numpy(self.allocator.slots[:, row]))
                self.original_positions[:, row].copy_(torch.from_numpy(self.allocator.positions[:, row]))
                self.valid_lengths[:, row].copy_(torch.from_numpy(self.allocator.lengths[:, row]))
        self.last_compaction[seq_id] = dict(reason=reason, before=before, released=released,
                                            after=self.chain_physical_residency(seq_id))
        self.compaction_stats[reason] += 1
        self.compaction_stats["released_head_slots"] += sum(released)
        row = self.allocator.seq_rows[seq_id]
        lengths = self.allocator.lengths[:, row]
        self.compaction_stats["nonuniform_layers"] += int((lengths.max(axis=1) != lengths.min(axis=1)).sum())
        self.query_counts[seq_id] = 0

    def publish_seq_metadata(self, seq_id):
        row = self.allocator.seq_rows[seq_id]
        self.slot_table[:, row].copy_(torch.from_numpy(self.allocator.slots[:, row]))
        self.original_positions[:, row].copy_(torch.from_numpy(self.allocator.positions[:, row]))
        self.valid_lengths[:, row].copy_(torch.from_numpy(self.allocator.lengths[:, row]))

    def free_seq(self, seq_id):
        row = self.allocator.seq_rows.get(seq_id)
        if row is None:
            return
        self.allocator.free_seq(seq_id)
        self.slot_table[:, row].zero_()
        self.original_positions[:, row].fill_(-1)
        self.valid_lengths[:, row].zero_()
        if self.query_ring is not None:
            self.query_ring[:, row].zero_()
        self.query_counts.pop(seq_id, None)
        self.last_positions.pop(seq_id, None)
        self.last_compaction.pop(seq_id, None)

    def reset_after_warmup(self):
        for seq_id in list(self.allocator.seq_rows):
            self.free_seq(seq_id)
        for key in self.compaction_stats:
            self.compaction_stats[key] = 0

    def free_part_slots(self, layer_idx, seq, keep_indices):
        raise TypeError("Independent heads require typed per-head selection at chain turn end.")

    def init_decode_graph_state(self, contract, inputs):
        inputs.validate(contract)
        shape = (self.num_kv_layers, contract.batch_capacity, self.num_kv_heads)
        return HeadSlotGraphState(contract, inputs,
                                   torch.full(shape, -1, dtype=torch.int32, device=self.device),
                                   torch.zeros(shape, dtype=torch.int32, device=self.device),
                                   torch.zeros(contract.batch_capacity, dtype=torch.bool, device=self.device))

    def prepare_decode_graph_step(self, seqs, state):
        ids, pos, _, writes, lens, req, _, _ = self._append(seqs, False)
        inputs, n = state.inputs, len(seqs)
        inputs.input_ids.zero_()
        inputs.positions.zero_()
        inputs.request_indices.fill_(-1)
        inputs.context_lens.zero_()
        inputs.write_slot_mapping.fill_(-1)
        inputs.active_mask.zero_()
        inputs.input_ids[:n].copy_(ids)
        inputs.positions[:n].copy_(pos)
        inputs.request_indices[:n].copy_(req)
        inputs.context_lens[:n].copy_(lens.max(dim=2).values.max(dim=0).values)
        inputs.active_mask[:n] = True
        state.writes.fill_(-1)
        state.lengths.zero_()
        state.query_keep.zero_()
        state.writes[:, :n].copy_(writes)
        state.lengths[:, :n].copy_(lens)
        state.query_keep[:n] = True
        self._bind(state.writes, state.lengths, inputs.request_indices,
                   inputs.request_indices, inputs.positions, state.query_keep)
        return inputs.input_ids, inputs.positions, None

    def decode_graph_state_keepalive_tensors(self, state):
        return [state.writes, state.lengths, state.query_keep, *self.decode_graph_keepalive_tensors()]

    def decode_graph_keepalive_tensors(self):
        tensors = [self.slot_table, self.original_positions, self.valid_lengths, self.kv_cache]
        if self.query_ring is not None:
            tensors.append(self.query_ring)
        return tensors

    def free_slot_stats(self):
        return {"free_slots": self.num_free_slots, "head_slots_free": int(self.allocator.free_counts.min()),
                "head_slots_reserved": max(self._outstanding()),
                "head_slots_allocated": int(self.allocator.lengths.sum()),
                "resident_seqs": len(self.allocator.seq_rows)}

    def debug_state_summary(self):
        return dict(cache_manager=type(self).__name__, free_slot_stats=self.free_slot_stats(),
                    compaction_stats=dict(self.compaction_stats),
                    head_slots_free_by_layer=self.allocator.free_counts.tolist(),
                    head_slots_reserved_by_layer=list(self._outstanding()),
                    head_slots_allocated_by_layer=self.allocator.lengths.sum(axis=(1, 2)).tolist(),
                    live_rows={sid: dict(row=row, head_lengths=self.allocator.lengths[:, row].tolist())
                               for sid, row in self.allocator.seq_rows.items()})

    def debug_live_seq_slots(self):
        return {sid: sum(self.chain_physical_residency(sid)) for sid in self.allocator.seq_rows}

    def _logical_live_kv_bytes(self):
        return int(self.allocator.lengths.sum()) * self.attention_cache_bytes_per_slot_per_layer()

    def _dense_baseline_slots(self):
        # The generic baseline counts full tokens; this pool counts single heads.
        return int(self.config.num_kvcache_slots) // self.num_kv_heads

    def create_prefill_history_manager(self, config, parallel_context):
        config.sparse_method, config.prefill_sparse_method = "", None
        config.head_slot_budgets = None
        return HeadSlotPrefillHistoryCacheManager(config, parallel_context)


class HeadSlotPrefillHistoryCacheManager(HeadSlotCacheManager):
    def allocate_kv_cache(self):
        slots = int(self.config.num_kvcache_slots) * self.num_kv_heads
        self.config.num_kvcache_slots = slots
        storage = self.attention_cache_storage
        storage.allocate(num_layers=1, num_slots=slots, device=self.device)
        storage.cache.zero_()
        storage.kv_cache = storage.cache.expand(-1, self.num_kv_layers, -1, -1)
        self.kv_cache = storage.cache

    def seed_history(self, seq):
        self.allocator.apply_append(self.allocator.plan_append([
            (seq.seq_id, list(range(seq.num_prefilled_tokens)))]))
        self.publish_seq_metadata(seq.seq_id)
