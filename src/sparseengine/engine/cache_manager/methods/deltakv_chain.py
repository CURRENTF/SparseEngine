"""Exclusive continuation over DeltaKV's existing physical pools and staging."""
from __future__ import annotations

import torch

from sparseengine.engine.prefill import PREFILL_EXECUTION_CHUNKED
from sparseengine.utils.context import get_context
from sparseengine.kernels.triton.quant import unpack_quantized_to_16bit
from ..base import AttentionViewMeta, ExplicitKVPayload, PrefillComputeView


class DeltaKVChainMixin:
    def chain_capacity_keys(self):
        return ("full_layers", "deltakv_raw", "deltakv_latent", "full_layer_kivi_blocks")

    def chain_has_residency(self, seq_id):
        return int(seq_id) in self.seq_id_to_row

    def chain_capacity_residency(self, seq_id):
        row = self.seq_id_to_row[int(seq_id)]
        full_len, sparse_len = self._deltakv_allocation_extents(row)
        full = int((self.full_layer_slots_map[row, :full_len] >= 0).sum().item())
        raw = int(self._active_deltakv_raw_slots_for_free(row, sparse_len).numel())
        latent = int((self.sparse_layer_latent_slots_map[row, :sparse_len] >= 0).sum().item())
        blocks = 0
        if self._full_layer_kivi_enabled():
            slots = self.full_layer_kivi_block_slots_map[row, :full_len]
            blocks = int(torch.unique(slots[slots >= 0]).numel())
        return full, raw, latent, blocks

    def chain_physical_residency(self, seq_id):
        full, raw, latent, blocks = self.chain_capacity_residency(seq_id)
        # Packed KIVI blocks retain group_size token entries. Sparse raw centers
        # and latent payloads can overlap in logical position but own separate slots.
        return tuple(full + blocks * self._full_layer_kivi_group_size()
                     if layer in self.full_layer_to_idx else raw + latent
                     for layer in self.kv_transformer_layer_indices())

    def chain_physical_kv_len(self, layer_idx, seq_id):
        return self.chain_physical_residency(seq_id)[self.kv_layer_index(layer_idx)]

    def chain_capacity_deficits(self, *, suffix_tokens, generation_tokens=0,
                               existing_slots_by_layer=(), outstanding_reserved_slots_by_layer=(),
                               outstanding_reserved_rows=0, needs_resident_row):
        existing = existing_slots_by_layer or (0, 0, 0, 0)
        full, raw, latent, blocks = existing
        old_len = full + blocks * self._full_layer_kivi_group_size()
        added = max(0, int(suffix_tokens)) + max(0, int(generation_tokens) - 1)
        end = old_len + added
        new_latent = max(0, self._estimate_deltakv_latent_slots_for_total_len(end) - latent)
        new_blocks = max(0, self._estimate_full_layer_kivi_blocks_for_total_len(end) - blocks) if self._full_layer_kivi_enabled() else 0
        # Cold full-prefill stages before committing; resumed chunks allocate
        # before eviction. Reserving the entire suffix is a conservative peak.
        full_need = added
        raw_need = added if old_len else self._estimate_deltakv_raw_slots_for_total_len(end)
        if self._full_layer_kivi_enabled():
            if old_len:
                peak = (int(self.config.sink_keep_tokens) + int(self.config.full_layer_kivi_residual_length)
                        + self._full_layer_kivi_group_size())
                full_need = min(added, max(1, peak - full))
            else:
                full_need = end - new_blocks * self._full_layer_kivi_group_size()
        required = (full_need, raw_need, new_latent, new_blocks)
        budgets = self.decode_window_budgets()
        free = tuple(budgets.get(k, 0) for k in self.chain_capacity_keys())
        outstanding = outstanding_reserved_slots_by_layer or (0, 0, 0, 0)
        deficits = tuple(max(0, need + reserved - available)
                         for need, reserved, available in zip(required, outstanding, free))
        rows = int(needs_resident_row)
        return required, rows, deficits, max(0, rows + outstanding_reserved_rows - len(self.free_rows))

    def prefill_reservation_owned_by_chain(self, seq):
        return bool(getattr(seq, "chain_id", None))

    def prompt_admission_cost(self, seq):
        return 0 if self.prefill_reservation_owned_by_chain(seq) else super().prompt_admission_cost(seq)

    def prompt_admission_costs(self, seq):
        return {} if self.prefill_reservation_owned_by_chain(seq) else super().prompt_admission_costs(seq)

    def on_prompt_admitted(self, seq, costs):
        if not self.prefill_reservation_owned_by_chain(seq):
            super().on_prompt_admitted(seq, costs)

    def prefill_execution_mode(self, seq):
        if self._has_attached_prefill_prefix(seq):
            return PREFILL_EXECUTION_CHUNKED
        return super().prefill_execution_mode(seq)

    def prefill_step_free_slots_for(self, seq):
        available = super().prefill_step_free_slots_for(seq)
        if self._has_attached_prefill_prefix(seq) and self._full_layer_kivi_enabled():
            row = self.seq_id_to_row[seq.seq_id]
            quantized = max(0, int(self.row_full_layer_kivi_quantized_lens[row])
                            - int(self.config.sink_keep_tokens))
            raw = int(self.row_seq_lens[row]) - quantized
            peak = (int(self.config.sink_keep_tokens) + int(self.config.full_layer_kivi_residual_length)
                    + self._full_layer_kivi_group_size())
            available = min(available, max(1, peak - raw))
        return available

    def prefill_batch_compatibility_key(self, seq):
        # Cold staging and resident append have different store destinations.
        return (super().prefill_batch_compatibility_key(seq), self._has_attached_prefill_prefix(seq))

    def step_allocation_costs(self, seq, tokens):
        costs = super().step_allocation_costs(seq, tokens)
        if self._has_attached_prefill_prefix(seq) and seq.num_prefilled_tokens < seq.num_prompt_tokens:
            costs["deltakv_prefill_staging"] = int(seq.num_prefilled_tokens) + int(tokens)
        return costs

    def free_seq(self, seq_id):
        self.reset_prefill_execution_state(seq_id)
        if not self.chain_has_residency(seq_id):
            self._release_prompt_admission_reservations(seq_id)
            return
        super().free_seq(seq_id)

    @torch.no_grad()
    def build_prefill_compute_view(self, layer_idx, k_current, v_current, selection):
        seqs = get_context().seqs
        resumed = any(self._has_attached_prefill_prefix(seq) for seq in seqs)
        sparse = layer_idx in self.deltakv_layer_to_idx
        if not resumed or (not sparse and not self._full_layer_kivi_enabled()):
            return super().build_prefill_compute_view(layer_idx, k_current, v_current, selection)
        # Reconstruct into the existing layer-shared prefill workspace. Never
        # replace persistent center/raw slot IDs or graph-captured decode storage.
        lengths = [int(self.row_seq_lens[self.seq_id_to_row[seq.seq_id]]) for seq in seqs]
        if sum(lengths) > self.deltakv_prefill_staging_num_slots:
            raise RuntimeError("DeltaKV resumed prefill exceeds the scheduled staging capacity.")
        stage = self.deltakv_prefill_staging_kv_cache
        active = torch.full((len(seqs), max(lengths)), -1, dtype=torch.int32, device=self.device)
        offset = 0
        for b, (seq, length) in enumerate(zip(seqs, lengths)):
            row = self.seq_id_to_row[seq.seq_id]
            active[b, :length] = torch.arange(offset, offset + length, dtype=torch.int32, device=self.device)
            if sparse:
                self._chain_materialize_sparse(layer_idx, row, length, stage[:, offset:offset + length])
            else:
                self._chain_materialize_kivi(layer_idx, row, length, stage[:, offset:offset + length])
            offset += length
        return PrefillComputeView(
            meta=AttentionViewMeta(active_slots=active,
                req_indices=torch.arange(len(seqs), dtype=torch.int32, device=self.device),
                context_lens=selection.context_lens, max_context_len=max(lengths),
                attn_score=selection.attn_score),
            payload=ExplicitKVPayload(k_cache=stage[0], v_cache=stage[1]))

    def _chain_materialize_sparse(self, layer_idx, row, length, out):
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        cache = self.deltakv_full_kv_cache[:, l_idx]
        dim = self.num_kv_heads * self.head_dim
        for start in range(0, length, self._deltakv_long_prefill_restore_block_tokens()):
            end = min(length, start + self._deltakv_long_prefill_restore_block_tokens())
            positions = torch.arange(start, end, device=self.device)
            raw = self.sparse_layer_raw_slots_map[row, start:end].long()
            latent_slots = self.sparse_layer_latent_slots_map[row, start:end].long()
            # Compressed positions, including centers with a retained raw copy,
            # use their residual payload for attention, just like decode.
            present = latent_slots < 0
            if (raw[present] < 0).any():
                raise RuntimeError("DeltaKV chain history has neither raw nor residual storage.")
            k = torch.empty((end - start, self.num_kv_heads, self.head_dim), dtype=cache.dtype, device=self.device)
            v = torch.empty_like(k)
            k[present], v[present] = cache[0, raw[present]], cache[1, raw[present]]
            missing = ~present
            latent = latent_slots[missing]
            if latent.numel():
                if (latent < 0).any():
                    raise RuntimeError("DeltaKV chain history has neither raw nor residual storage.")
                fathers = self.deltakv_latent_to_full_slots[l_idx, latent].long()
                if (fathers < 0).any():
                    raise RuntimeError("DeltaKV chain residual has missing reference centers.")
                residual = self._load_residual(l_idx, latent, 2 * dim)
                base = torch.cat((cache[0, fathers].flatten(2), cache[1, fathers].flatten(2)), dim=-1).float().mean(dim=1)
                restored = (base + residual.float()).to(cache.dtype)
                k[missing] = restored[:, :dim].reshape(-1, self.num_kv_heads, self.head_dim)
                v[missing] = restored[:, dim:].reshape(-1, self.num_kv_heads, self.head_dim)
            out[0, start:end] = self._apply_sparse_key_rope(l_idx, positions, self._apply_sparse_k_norm_if_needed(l_idx, k))
            out[1, start:end] = v

    def _chain_materialize_kivi(self, layer_idx, row, length, out):
        l_idx = self.full_layer_to_idx[layer_idx]
        raw = self.full_layer_slots_map[row, :length].long()
        present = raw >= 0
        out[:, present] = self.full_kv_cache[:, l_idx, raw[present]]
        group = self._full_layer_kivi_group_size()
        slots = self.full_layer_kivi_block_slots_map[row, :length]
        blocks = torch.unique(slots[slots >= 0]).long()
        # Bound dequantization temporaries by the same token chunk as sparse reconstruction.
        block_chunk = max(1, self._deltakv_long_prefill_restore_block_tokens() // group)
        for start in range(0, blocks.numel(), block_chunk):
            ids = blocks[start:start + block_chunk]
            k = unpack_quantized_to_16bit(self.full_layer_kivi_key_packed[l_idx, ids],
                self.full_layer_kivi_key_scales[l_idx, ids].unsqueeze(-1),
                self.full_layer_kivi_key_mins[l_idx, ids].unsqueeze(-1), group, 4).permute(0, 3, 1, 2)
            v = unpack_quantized_to_16bit(self.full_layer_kivi_value_packed[l_idx, ids],
                self.full_layer_kivi_value_scales[l_idx, ids], self.full_layer_kivi_value_mins[l_idx, ids], group, 4).permute(0, 2, 1, 3)
            positions = self.full_layer_kivi_block_start_pos[ids].long()[:, None] + torch.arange(group, device=self.device)
            out[0, positions.flatten()] = k.reshape(-1, self.num_kv_heads, self.head_dim).to(out.dtype)
            out[1, positions.flatten()] = v.reshape(-1, self.num_kv_heads, self.head_dim).to(out.dtype)
