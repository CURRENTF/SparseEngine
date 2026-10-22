"""FP8 token storage using the Standard allocator and prefix/pruning lifecycle."""

from sparseengine.kernels.triton.fp8_kv_store import fp8_store_tokens

from .base import ExplicitKVWrite
from .quantized import QuantizedCacheManager, allocate_quantized_storage, materialize_prefill_history
from .standard import StandardCacheManager


class FP8CacheManager(StandardCacheManager):
    def allocate_kv_cache(self):
        allocate_quantized_storage(self)

    get_prefill_compute_payload = QuantizedCacheManager.get_prefill_compute_payload

    def _prepare_prefill(self, seqs):
        result = super()._prepare_prefill(seqs)
        self._prefill_active = True
        self._write_plan = []
        offset = 0
        for seq in seqs:
            row = self.seq_id_to_row[seq.seq_id]
            end = int(self.row_seq_lens[row])
            start = end - seq.current_chunk_size
            slots = self.buffer_req_to_token_slots[row, start:end]
            self._write_plan.append((row, start, end, offset, slots))
            offset += end - start
        return result

    def _attach_prefill_prefixes(self, seqs):
        super()._attach_prefill_prefixes(seqs)
        self._ensure_step_slots(seqs, prefill=True)

    def _prepare_decode(self, seqs):
        self._reserve_decode_prefix_promotions(seqs)
        result = super()._prepare_decode(seqs)
        self._prefill_active = False
        self._write_plan = []
        return result

    def _prepare_decode_graph_buffers(self, seqs, **kwargs):
        self._reserve_decode_prefix_promotions(seqs)
        result = super()._prepare_decode_graph_buffers(seqs, **kwargs)
        self._prefill_active = False
        self._write_plan = []
        return result

    def store_attention_payload(self, layer_idx, payload):
        if not isinstance(payload, ExplicitKVWrite):
            raise TypeError("FP8 cache requires ExplicitKVWrite.")
        key, value = payload.key, payload.value
        count = self.layer_batch_state.slot_mapping.numel()
        if key.shape != value.shape or tuple(key.shape) != (count, self.num_kv_heads, self.head_dim):
            raise ValueError("FP8 KV write shape differs from the prepared batch.")
        if key.dtype != self.hf_config.dtype or value.dtype != key.dtype or key.device != self.device or value.device != self.device:
            raise ValueError("FP8 KV write dtype/device differs from the model/cache.")
        layer = self.attention_cache_storage.layer_payload(self.kv_layer_index(layer_idx))
        if not self._prefill_active:
            state = self.layer_batch_state
            fp8_store_tokens(key, value, layer, state.slot_mapping,
                             rows=state.req_indices, lengths=state.context_lens)
        else:
            # Model KV storage precedes Attention.forward's prefill hook. The
            # history materializer already reads restored prefix bytes/scales.
            self.before_prefill_layer_attention(layer_idx, None)
            materialize_prefill_history(self, layer, key, value)
            for row, start, end, offset, slots in self._write_plan:
                fp8_store_tokens(key[offset:offset + end - start], value[offset:offset + end - start],
                                 layer, slots, row=row, start=start, end=end)
        return self.layer_batch_state.slot_mapping

    def get_layer_kv_cache(self, layer_idx):
        raise RuntimeError("FP8 cache requires a typed compressed compute payload.")

    def _get_free_row(self, seq_id):
        new = seq_id not in self.seq_id_to_row
        row = super()._get_free_row(seq_id)
        if new:
            self.attention_cache_storage.encoded_prefix_lens[row] = 0
        return row

    def _attach_prefix_cache_if_needed(self, seq):
        attached = seq.seq_id in self.seq_id_to_prefix_blocks
        super()._attach_prefix_cache_if_needed(seq)
        if not attached and seq.seq_id in self.seq_id_to_prefix_blocks:
            row = self.seq_id_to_row[seq.seq_id]
            # A pruned prefix can end anywhere within the raw-tail interval.
            # Keep these tokens encoded; do not dequantize/requantize them.
            self.attention_cache_storage.encoded_prefix_lens[row] = int(self.row_seq_lens[row])

    def attach_prefix_kv_payloads(self, seq, payloads):
        super().attach_prefix_kv_payloads(seq, payloads)
        if payloads:
            row = self.seq_id_to_row[seq.seq_id]
            self.attention_cache_storage.encoded_prefix_lens[row] = int(self.row_seq_lens[row])

    def _reset_prefix_cache_allocator_after_clear(self):
        super()._reset_prefix_cache_allocator_after_clear()
        self.attention_cache_storage.encoded_prefix_lens.zero_()
        self._write_plan = []
        self._prefill_active = False

    def free_seq(self, seq_id):
        row = self.seq_id_to_row.get(seq_id)
        if row is not None:
            self.attention_cache_storage.encoded_prefix_lens[row] = 0
        super().free_seq(seq_id)

    def prefix_kv_payload_nbytes(self, payload):
        slots = payload.token_slots
        if slots is None:
            raise RuntimeError("FP8 prefix payload has no device slots.")
        return slots.numel() * self.num_kv_layers * self.attention_cache_storage.bytes_per_slot_per_layer()

    def _logical_live_kv_bytes(self):
        return ((self.config.num_kvcache_slots - self.num_free_slots) * self.num_kv_layers
                * self.attention_cache_storage.bytes_per_slot_per_layer())

    def prefill_step_reservation_cost(self, seq, scheduled_tokens):
        return int(scheduled_tokens) + self._prefix_recompute_promotion_cost(seq, int(scheduled_tokens), prefill=True)

    def step_allocation_budgets(self):
        if not self._prefix_offload_enabled():
            return None
        return {"slots": self.prefill_step_free_slots()}

    def step_allocation_costs(self, seq, tokens):
        prefill = self.remaining_prefill_tokens(seq) > 0
        return {"slots": int(tokens) + self._prefix_recompute_promotion_cost(seq, int(tokens), prefill=prefill)}

    def prompt_admission_cost(self, seq):
        remaining = self.remaining_prefill_tokens(seq)
        return super().prompt_admission_cost(seq) + self._prefix_recompute_promotion_cost(seq, remaining, prefill=True)

    def reserved_prefill_slots(self, waiting_seqs, engine_prefill_chunk_size):
        return sum(self.prefill_step_reservation_cost(seq, self.remaining_prefill_tokens(seq))
                   for seq in waiting_seqs if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens)

    def decode_step_reservation_cost(self, seq):
        return 1 + self._prefix_recompute_promotion_cost(seq, 1, prefill=False)

    def decode_window_costs(self, seq, tokens):
        cost = int(tokens)
        if self._prefix_offload_enabled():
            row = self.seq_id_to_row.get(seq.seq_id)
            start = 0 if row is None else int(self.row_logical_lens[row])
            size = self.prefix_cache_block_size
            cost += ((start + tokens) // size - start // size) * size
        return {"slots": cost}

    def _prefix_recompute_promotion_cost(self, seq, count, *, prefill):
        if not self._prefix_offload_enabled() or self._prefix_prune_scoring is not None:
            return 0
        state = self.prefix_runtime_states.get(seq.seq_id)
        pending = [] if state is None else state.pending_tokens
        size = self.prefix_cache_block_size
        if len(pending) + count < size:
            return 0
        parent = getattr(seq, "prefix_cache_hit_last_block_id", None) if state is None else state.parent_block_id
        if prefill:
            tokens = seq.token_ids
            if len(tokens) > count:
                start = max(seq.num_prefilled_tokens, seq.prefix_cache_hit_len)
                tokens = tokens[start:start + count]
        else:
            tokens = [seq.decode_input_token]
        tokens = pending + list(tokens)
        cost = 0
        reclaimable = self.prefix_cache.device_reclaimable_block_ids()
        for start in range(0, len(tokens) - size + 1, size):
            parent = self.prefix_cache.stable_block_id(tokens[start:start + size], parent)
            block = self.prefix_cache.get_block(parent)
            # Scheduling can reclaim this continuation before publication. Its
            # restore must fit even if it is still device-resident at admission.
            if block is not None and (not block.residency.device_present or parent in reclaimable):
                cost += self._standard_payload(block).resident_tokens(size)
        return cost

    def _reserve_decode_prefix_promotions(self, seqs):
        if self._prefix_offload_enabled():
            self._ensure_step_slots(seqs, prefill=False)

    def _ensure_step_slots(self, seqs, *, prefill):
        reserved = (getattr(self, "_prefill_slot_reservations", None) or {}) if prefill else {}

        def required():
            return sum((self.prefill_step_reservation_cost(seq, seq.current_chunk_size)
                        if prefill else self.decode_step_reservation_cost(seq))
                       - (reserved[seq.seq_id].numel() if seq.seq_id in reserved else 0) for seq in seqs)

        needed = required()
        # Demotion can turn a recomputed continuation into a host-only block,
        # increasing its promotion cost. Recheck before consuming query slots.
        blocks = 0 if self.prefix_cache is None else len(self.prefix_cache.blocks)
        for _ in range(blocks + 1):
            self._evict_prefix_cache_until_free(needed)
            updated = required()
            if self.num_free_slots >= updated:
                return
            if updated <= needed:
                break
            needed = updated
        raise RuntimeError("Insufficient FP8 slots for query and prefix continuation promotion.")

    def _hold_materialized_prefix_block_ref(self, seq, block):
        if not block.residency.device_present:
            # Preserve the index's immutable host copy and the request's private
            # recomputation when the generated continuation duplicates a block.
            payload = self._standard_payload(block)
            slots = self._take_prefix_device_slots(payload.resident_tokens(self.prefix_cache_block_size))
            payload.token_slots = slots
            try:
                self.prefix_offload_controller.submit_h2d([block])
            except BaseException:
                payload.token_slots = None
                self._return_prefix_device_slots(slots)
                raise
        super()._hold_materialized_prefix_block_ref(seq, block)
