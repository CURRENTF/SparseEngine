"""Token-preserving compressed pages with shared eager/graph decode updates."""

import math

import numpy as np
import torch

from sparseengine.kernels.triton.quantized_kv import encode_pages, quantized_decode_append, materialize_sequence
from .base import ExplicitKVPayload, ExplicitKVWrite
from .quantized_pages import QuantizedPagePool
from .standard import StandardCacheManager
from .storage.quantized_kv import QuantizedKVStorage, quantized_kv_reserved_bytes


class QuantizedCacheManager(StandardCacheManager):
    def allocate_kv_cache(self):
        config = self.config
        self.page_size = int(config.kv_quant_page_size)
        method = config.sparse_method
        bits = config.kivi_bits if method == "kivi" else config.turboquant_bits if method == "turboquant" else 8
        storage = QuantizedKVStorage(
            format=method, bits=bits, page_size=self.page_size,
            num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            dtype=self.hf_config.dtype, seed=config.turboquant_seed,
        )
        self.attention_cache_storage = storage
        available, _ = self._get_available_slots_info()
        h, d, g = self.num_kv_heads, self.head_dim, self.page_size
        self.prefill_capacity = int(config.max_num_seqs_in_batch) * self.max_model_len
        fixed = quantized_kv_reserved_bytes(config, num_layers=self.num_kv_layers, num_heads=h, head_dim=d)
        page_bytes = self.num_kv_layers * storage.bytes_per_page_per_layer() + g * 4
        num_pages = (available - fixed) // page_bytes
        minimum_pages = 1 if getattr(config, "startup_cache_phase", "") == "profiling" else math.ceil(self.max_model_len / g)
        if num_pages < minimum_pages:
            raise RuntimeError(
                "Insufficient memory for quantized KV pages and bounded prefill/tail workspaces: "
                f"available={available}, reserved={fixed}, page_bytes={page_bytes}. "
                "Reduce max_model_len or max_num_seqs_in_batch."
            )
        config.num_kvcache_slots = int(num_pages * g)
        self.page_pool = QuantizedPagePool(int(num_pages), g, self.max_model_len)
        storage.allocate(num_layers=self.num_kv_layers, num_slots=config.num_kvcache_slots,
                         num_rows=self.max_buffer_rows, device=self.device)
        self.kv_cache = None
        self.prefill_kv = torch.empty(2, self.prefill_capacity, h, d,
                                      dtype=self.hf_config.dtype, device=self.device)
        self.prefill_rotated = (torch.empty(2, self.prefill_capacity, h, d,
                                            dtype=torch.float32, device=self.device)
                                if method == "turboquant" else None)
        self.prefill_slot_map = torch.arange(self.prefill_capacity, dtype=torch.int32, device=self.device).reshape(
            int(config.max_num_seqs_in_batch), self.max_model_len)
        self._write_plan = []
        self._prefill_active = False

    @property
    def num_free_slots(self):
        return len(self.page_pool.free) * self.page_size

    def _allocate(self, seq_id, size):
        if seq_id not in self.seq_id_to_row and not self.free_rows:
            raise RuntimeError("No free quantized cache rows.")
        plan = self.page_pool.append(seq_id, int(size))
        row = self._get_free_row(seq_id)
        first_page = plan.start // self.page_size
        pages = torch.tensor(plan.pages, dtype=torch.int32, device=self.device)
        logical = torch.arange(plan.start, plan.end, dtype=torch.int32, device=self.device)
        slots = pages[(logical // self.page_size - first_page).long()] * self.page_size + logical % self.page_size
        self.buffer_req_to_token_slots[row, plan.start:plan.end] = slots
        self.row_seq_lens[row] = plan.end
        self.row_logical_lens[row] = plan.end
        return slots

    def _allocate_batch(self, seq_ids, size):
        if size != 1 or len(seq_ids) != len(set(seq_ids)):
            raise ValueError("Quantized decode requires one token per distinct sequence.")
        needed = sum(self.page_pool.append_cost(sid, 1) for sid in seq_ids)
        new_rows = sum(sid not in self.seq_id_to_row for sid in seq_ids)
        if needed > len(self.page_pool.free) or new_rows > len(self.free_rows):
            raise RuntimeError("Insufficient quantized KV pages/rows for decode batch.")
        return torch.cat([self._allocate(sid, 1) for sid in seq_ids])

    def _prepare_prefill(self, seqs):
        needed = sum(self.page_pool.append_cost(seq.seq_id, seq.current_chunk_size) for seq in seqs)
        new_rows = sum(seq.seq_id not in self.seq_id_to_row for seq in seqs)
        if needed > len(self.page_pool.free) or new_rows > len(self.free_rows):
            raise RuntimeError("Insufficient quantized KV pages/rows for prefill batch.")
        result = super()._prepare_prefill(seqs)
        self._prepare_prefill_write_plan(seqs)
        return result

    def _prepare_decode(self, seqs):
        result = super()._prepare_decode(seqs)
        self._prefill_active = False
        self._write_plan = []
        return result

    def _allocate_decode_batch_static(self, seq_ids, *, row_indices, pending_rows):
        slots = self._allocate_batch(seq_ids, 1)
        rows = np.asarray([self.seq_id_to_row[sid] for sid in seq_ids], dtype=np.int32)
        return slots, self.row_seq_lens[rows].copy(), rows

    def _prepare_decode_graph_buffers(self, seqs, **kwargs):
        result = super()._prepare_decode_graph_buffers(seqs, **kwargs)
        self._prefill_active = False
        self._write_plan = []
        return result

    def _prepare_prefill_write_plan(self, seqs):
        self._prefill_active = True
        self._write_plan = []
        offset = 0
        for seq in seqs:
            count = seq.current_chunk_size
            end = self.page_pool.lengths[seq.seq_id]
            row = self.seq_id_to_row[seq.seq_id]
            pages = self.page_pool.pages[seq.seq_id]
            start = end - count
            full_start, full_end = start // self.page_size, end // self.page_size
            page_ids = torch.tensor(pages[full_start:full_end], dtype=torch.int32, device=self.device)
            self._write_plan.append((row, start, end, offset, page_ids))
            offset += count

    def store_attention_payload(self, layer_idx, payload):
        if not isinstance(payload, ExplicitKVWrite):
            raise TypeError("Quantized cache requires ExplicitKVWrite.")
        storage = self.attention_cache_storage
        layer = storage.layer_payload(self.kv_layer_index(layer_idx))
        k, v = payload.key, payload.value
        input_count = int(self.layer_batch_state.slot_mapping.numel())
        count = sum(end - start for _, start, end, _, _ in self._write_plan) if self._prefill_active else input_count
        expected = (count, self.num_kv_heads, self.head_dim)
        if k.shape != v.shape or tuple(k.shape) != (input_count, *expected[1:]):
            raise ValueError(f"Quantized KV write expects {expected}, got {tuple(k.shape)}, {tuple(v.shape)}.")
        if k.dtype != self.hf_config.dtype or v.dtype != k.dtype or k.device != self.device or v.device != self.device:
            raise ValueError("Quantized KV write dtype/device differs from the model/cache.")
        if not self._prefill_active:
            # Device metadata controls page completion and padded writes, so the
            # same state transition runs eagerly and inside a captured graph.
            state = self.layer_batch_state
            if layer.rotation is not None:
                k = (k.float() @ layer.rotation).to(self.hf_config.dtype)
                v = (v.float() @ layer.rotation).to(self.hf_config.dtype)
            quantized_decode_append(k, v, layer, self.buffer_req_to_token_slots,
                                    state.req_indices, state.context_lens, state.slot_mapping)
            return state.slot_mapping
        k, v = k[:expected[0]], v[:expected[0]]
        # Standard prefill consumes exact current K/V, including pages just encoded.
        for index, (row, start, end, offset, _) in enumerate(self._write_plan):
            base = index * self.max_model_len
            self.prefill_kv[0, base + start:base + end].copy_(k[offset:offset + end - start])
            self.prefill_kv[1, base + start:base + end].copy_(v[offset:offset + end - start])
            if start:
                outputs = self.prefill_rotated if layer.rotation is not None else self.prefill_kv
                # Read history before overwriting its incomplete raw page.
                materialize_sequence(layer, self.buffer_req_to_token_slots, row, start,
                                     outputs[0, base:base + start], outputs[1, base:base + start])
                if layer.rotation is not None:
                    for kv in range(2):
                        restored = outputs[kv, base:base + start] @ layer.rotation.T
                        self.prefill_kv[kv, base:base + start].copy_(restored)
        if layer.rotation is not None:
            k = (k.float() @ layer.rotation).to(self.hf_config.dtype)
            v = (v.float() @ layer.rotation).to(self.hf_config.dtype)
        for row, start, end, offset, page_ids in self._write_plan:
            current_k, current_v = k[offset:offset + end - start], v[offset:offset + end - start]
            previous = start % self.page_size
            if previous:
                current_k = torch.cat((layer.raw_key[row, :previous], current_k))
                current_v = torch.cat((layer.raw_value[row, :previous], current_v))
            full_tokens = (current_k.shape[0] // self.page_size) * self.page_size
            if full_tokens:
                encode_pages(current_k[:full_tokens], current_v[:full_tokens], page_ids, layer)
            residual = current_k.shape[0] - full_tokens
            if residual:
                layer.raw_key[row, :residual].copy_(current_k[full_tokens:])
                layer.raw_value[row, :residual].copy_(current_v[full_tokens:])
        return self.layer_batch_state.slot_mapping

    def get_prefill_compute_payload(self, layer_idx, k_current, v_current, selection,
                                    active_slots, req_indices, context_lens):
        batch = len(self._write_plan)
        rows = torch.arange(batch, dtype=torch.int32, device=self.device)
        return (ExplicitKVPayload(k_cache=self.prefill_kv[0], v_cache=self.prefill_kv[1]),
                self.prefill_slot_map[:batch], rows, context_lens)

    def get_layer_kv_cache(self, layer_idx):
        raise RuntimeError("Quantized cache requires a typed compressed compute payload.")

    def prefill_step_reservation_cost(self, seq, scheduled_tokens):
        return self.page_pool.append_cost(seq.seq_id, int(scheduled_tokens)) * self.page_size

    def decode_window_costs(self, seq, tokens):
        return {"slots": self.page_pool.append_cost(seq.seq_id, tokens) * self.page_size}

    def decode_step_reservation_cost(self, seq):
        return self.page_pool.append_cost(seq.seq_id, 1) * self.page_size

    def prefill_step_free_slots_for(self, seq):
        length = self.page_pool.lengths.get(seq.seq_id, 0)
        return self.num_free_slots + (-length % self.page_size)

    def prefill_private_slots_for(self, seq):
        return -self.page_pool.lengths.get(seq.seq_id, 0) % self.page_size

    def decode_step_free_slots_for(self, seq):
        return self.prefill_step_free_slots_for(seq)

    def prompt_admission_cost(self, seq):
        return math.ceil(seq.num_prompt_tokens / self.page_size) * self.page_size

    def reserved_prefill_slots(self, waiting_seqs, engine_prefill_chunk_size):
        return sum(self.prefill_step_reservation_cost(seq, seq.num_prompt_tokens - seq.num_prefilled_tokens)
                   for seq in waiting_seqs if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens)

    def free_seq(self, seq_id):
        if seq_id not in self.seq_id_to_row:
            raise ValueError(f"Unknown quantized cache sequence {seq_id}.")
        self.page_pool.release(seq_id)
        row = self.seq_id_to_row.pop(seq_id)
        self.row_seq_lens[row] = self.row_logical_lens[row] = 0
        self.buffer_req_to_token_slots[row].zero_()
        self.free_rows.append(row)

    def reset_after_warmup(self):
        if self.seq_id_to_row:
            raise RuntimeError("Cannot reset quantized KV cache with active sequences.")
        self._write_plan = []
        self._prefill_active = False

    def _logical_live_kv_bytes(self):
        full_pages = sum(length // self.page_size for length in self.page_pool.lengths.values())
        tail_tokens = sum(length % self.page_size for length in self.page_pool.lengths.values())
        storage = self.attention_cache_storage
        return self.num_kv_layers * (
            full_pages * storage.bytes_per_page_per_layer()
            + tail_tokens * 2 * self.num_kv_heads * self.head_dim * storage.raw.element_size()
        )
