from __future__ import annotations

import math
import os
from dataclasses import replace

import torch

from sparseengine.engine.sequence import Sequence
from sparseengine.kernels.triton.quant import (
    triton_dequantize_2d_int4_grouped,
    triton_quantize_and_pack_2d_int4_grouped,
    triton_quantize_and_pack_along_last_dim,
    unpack_quantized_to_16bit,
)
from sparseengine.kernels.triton.deltakv_kernels import deltakv_materialize_sparse_view
from sparseengine.layers.rotary_embedding import apply_rotary_emb
from sparseengine.platforms import device_runtime
from sparseengine.utils.compressor import create_compressor
from sparseengine.utils.context import get_context
from sparseengine.utils.log import logger
from sparseengine.utils.profiler import profiler

from ..base import (
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
    SparseSelection,
)
from .deltakv_base import DeltaKVCacheTritonManagerV4


class DeltaKVLessMemoryCacheManager(DeltaKVCacheTritonManagerV4):
    """DeltaKV compressor-latent runtime.

    Sparse layers store compressor latent residuals, optionally packed as int4.
    Full layers are either raw BF16/FP16 KV or KIVI int4. Legacy direct-residual,
    sparse int2, and full-layer non-KIVI residual-quant paths are intentionally
    not active in the slim runtime.
    """

    @classmethod
    def profiling_kv_budget_bytes(cls, config, num_slots: int, slot_bytes_per_layer: int) -> int:
        """Profile raw full-layer storage with its fixed reconstruction pools."""
        layers = int(config.runtime_layout.num_kv_layers)
        full_layers = len(config.full_attention_layers)
        sparse_layers = layers - full_layers
        rows = int(config.max_num_seqs_in_gpu)
        decode_rows = cls.decode_scratch_sequence_capacity(config)
        sink = int(config.sink_keep_tokens)
        recent = int(config.recent_keep_tokens)
        top = int(config.decode_keep_tokens)
        raw_slots = cls._resident_sparse_raw_overhead_slots(rows, sink, recent)
        raw_slots += cls._decode_reconstruct_scratch_slots(decode_rows, top, sink, recent)
        fixed_bytes = (raw_slots + int(num_slots)) * sparse_layers * slot_bytes_per_layer
        # Prefill stages one layer's K/V plus pre-RoPE K, even for short profiles.
        fixed_bytes += int(config.max_model_len) * slot_bytes_per_layer * 3 // 2
        fixed_bytes += cls.materialized_sparse_compute_slot_capacity(
            config, decode_rows, sink, recent, top,
        ) * slot_bytes_per_layer
        # Retain the conservative temporary payload allowance, but do not use
        # it to pay for method-owned fixed buffers needed at every capacity.
        return fixed_bytes + int(num_slots) * layers * slot_bytes_per_layer * 2

    def _extra_workspace_reserve_bytes(self) -> int:
        return 0

    def _full_layer_quant_enabled(self) -> bool:
        # The slim runtime removed the legacy full-layer DeltaKV residual-quant
        # mode. full_layer_kv_quant_bits=4 always means KIVI int4.
        return False

    def _full_layer_kivi_enabled(self) -> bool:
        config = getattr(self, "config", None)
        if config is None:
            return False
        return (
            bool(getattr(config, "enable_full_layer_kivi_quant", True))
            and int(getattr(config, "full_layer_kv_quant_bits", 0) or 0) == 4
        )

    def _materialize_deltakv_active_postrope_view(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        new_context_lens: torch.Tensor,
        already_postrope_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build an attention-facing view from raw sparse-layer K slots.

        Persistent DeltaKV sparse-layer slots keep K in the pre-RoPE space for
        compressor/reconstruction math. Decode attention needs post-RoPE K, so
        materialize the current sparse view into scratch slots and free it after
        the layer attention finishes.
        """
        if active_slots.numel() == 0:
            return active_slots, torch.empty((0,), device=active_slots.device, dtype=torch.int32)
        if not get_context().is_prefill or self._is_stream_capturing():
            return self._materialize_deltakv_active_postrope_view_static(
                layer_idx,
                active_slots,
                new_context_lens,
                already_postrope_slots,
            )

        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_cache = self.deltakv_full_kv_cache[0, l_idx]
        v_cache = self.deltakv_full_kv_cache[1, l_idx]
        debug_layers = os.getenv("SPARSEENGINE_DEBUG_RECONSTRUCT_LAYERS")
        debug_capture = False
        if debug_layers:
            debug_capture = int(layer_idx) in {int(part) for part in debug_layers.split(",") if part.strip()}
        debug_raw: list[torch.Tensor] = []
        debug_norm: list[torch.Tensor] = []
        debug_raw_rope: list[torch.Tensor] = []
        debug_norm_rope: list[torch.Tensor] = []
        debug_raw_positions: list[torch.Tensor] = []
        materialized = []
        for b in range(int(active_slots.shape[0])):
            view_len = int(new_context_lens[b].item())
            if view_len <= 0:
                continue
            src_slots = active_slots[b, :view_len].to(torch.long)
            if (src_slots < 0).any():
                raise RuntimeError(f"DeltaKV less-memory: active sparse view contains negative slot, layer={layer_idx}.")
            already = self._already_postrope_mask(layer_idx, src_slots, already_postrope_slots)
            raw_idx = torch.nonzero(~already, as_tuple=False).flatten()
            if raw_idx.numel() == 0:
                continue

            raw_slots = src_slots.index_select(0, raw_idx)
            pos = self.deltakv_slot_to_pos[raw_slots].to(torch.long)
            if (pos < 0).any():
                raise RuntimeError(f"DeltaKV less-memory: active sparse view slot has unknown position, layer={layer_idx}.")

            out_slots = self._allocate_temp_deltakv_full(int(raw_idx.numel())).to(torch.int32)
            cos_sin = self.cos_sin_cache[pos]
            cos, sin = cos_sin.chunk(2, dim=-1)
            k_raw = k_cache[raw_slots]
            k_raw_rope = apply_rotary_emb(k_raw, cos, sin)
            k_normed = self._apply_sparse_k_norm_if_needed(l_idx, k_raw)
            k_postrope = apply_rotary_emb(k_normed, cos, sin)
            if debug_capture:
                debug_raw.append(k_raw.detach().float().cpu())
                debug_norm.append(k_normed.detach().float().cpu())
                debug_raw_rope.append(k_raw_rope.detach().float().cpu())
                debug_norm_rope.append(k_postrope.detach().float().cpu())
                debug_raw_positions.append(pos.detach().cpu())
            out_i64 = out_slots.to(torch.long)
            k_cache[out_i64] = k_postrope.to(k_cache.dtype)
            v_cache[out_i64] = v_cache[raw_slots].to(v_cache.dtype)
            self.deltakv_slot_to_pos[out_i64] = pos.to(torch.int32)
            active_slots[b, raw_idx] = out_slots
            materialized.append(out_slots)

        if debug_capture and debug_raw_positions:
            debug = getattr(self, "debug_last_reconstruct_alternatives", {})
            debug[int(layer_idx)] = {
                "positions": torch.cat(debug_raw_positions, dim=0),
                "k_raw": torch.cat(debug_raw, dim=0).permute(1, 0, 2).unsqueeze(0),
                "k_norm": torch.cat(debug_norm, dim=0).permute(1, 0, 2).unsqueeze(0),
                "k_raw_rope": torch.cat(debug_raw_rope, dim=0).permute(1, 0, 2).unsqueeze(0),
                "k_norm_rope": torch.cat(debug_norm_rope, dim=0).permute(1, 0, 2).unsqueeze(0),
            }
            self.debug_last_reconstruct_alternatives = debug

        if not materialized:
            return active_slots, torch.empty((0,), device=active_slots.device, dtype=torch.int32)
        return active_slots, torch.cat(materialized, dim=0).to(torch.int32)

    def _materialize_deltakv_active_postrope_view_static(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        new_context_lens: torch.Tensor,
        already_postrope_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Graph-safe post-RoPE materialization for raw sparse K slots.

        The original less-memory implementation allocated temp slots on every
        call.  That is unsafe once the function is captured.  Here the output
        slots are pre-reserved per (active_slots.shape, device) through the base
        DeltaKV static-materialization cache, and invalid/padded entries preserve
        previous cache contents through masked writes.
        """
        # Runtime decode tracks post-RoPE status through the per-layer slot mask.
        # Tests and partial object construction may not initialize that mask, so
        # _already_postrope_mask falls back to the explicit slots in that case.
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_cache = self.deltakv_full_kv_cache[0, l_idx]
        v_cache = self.deltakv_full_kv_cache[1, l_idx]

        flat_active = active_slots.reshape(-1).to(torch.int32)
        out_slots = self._ensure_decode_static_materialized_slots(active_slots).to(torch.int32)
        if flat_active.numel() == 0:
            return active_slots, torch.empty((0,), device=active_slots.device, dtype=torch.int32)

        if active_slots.dim() == 2:
            cols = torch.arange(active_slots.shape[1], device=active_slots.device, dtype=new_context_lens.dtype)
            visible = (cols.unsqueeze(0) < new_context_lens.to(device=active_slots.device).unsqueeze(1)).reshape(-1)
        else:
            visible = torch.ones_like(flat_active, dtype=torch.bool)

        valid = visible & (flat_active >= 0)
        already = self._already_postrope_mask(layer_idx, flat_active, already_postrope_slots)
        raw_mask = valid & ~already
        safe_raw_slots = torch.where(raw_mask, flat_active, torch.zeros_like(flat_active)).to(torch.long)
        raw_pos = self.deltakv_slot_to_pos[safe_raw_slots].to(torch.long)

        is_capturing = self._is_stream_capturing()
        if not is_capturing and ((raw_pos < 0) & raw_mask).any():
            raise RuntimeError(f"DeltaKV less-memory static sparse raw slot has unknown position, layer={layer_idx}.")
        safe_pos = torch.where(raw_mask, raw_pos, torch.zeros_like(raw_pos))

        cos_sin = self.cos_sin_cache[safe_pos]
        cos, sin = cos_sin.chunk(2, dim=-1)
        k_raw = k_cache[safe_raw_slots]
        k_normed = self._apply_sparse_k_norm_if_needed(l_idx, k_raw)
        k_postrope = apply_rotary_emb(k_normed, cos, sin)

        out_i64 = out_slots.to(torch.long)
        prev_k = k_cache[out_i64]
        prev_v = v_cache[out_i64]
        write_mask = raw_mask[:, None, None]
        k_cache[out_i64] = torch.where(write_mask, k_postrope.to(k_cache.dtype), prev_k)
        v_cache[out_i64] = torch.where(write_mask, v_cache[safe_raw_slots].to(v_cache.dtype), prev_v)
        prev_pos = self.deltakv_slot_to_pos[out_i64]
        self.deltakv_slot_to_pos[out_i64] = torch.where(raw_mask, safe_pos.to(torch.int32), prev_pos)

        flat_out = torch.where(raw_mask, out_slots, flat_active).to(torch.int32)
        active_slots.copy_(flat_out.view_as(active_slots))
        return active_slots, torch.empty((0,), device=active_slots.device, dtype=torch.int32)

    def _full_layer_kivi_group_size(self) -> int:
        group_size = int(getattr(self.config, "full_layer_kivi_group_size", 32) or 32)
        if group_size <= 0:
            raise ValueError(f"full_layer_kivi_group_size must be > 0, got {group_size}.")
        if group_size % 8 != 0:
            raise ValueError(
                "Full-layer KIVI int4 packing requires group_size divisible by 8; "
                f"got full_layer_kivi_group_size={group_size}."
            )
        if self.head_dim % group_size != 0:
            raise ValueError(
                "Full-layer KIVI value quantization requires head_dim divisible by group_size; "
                f"head_dim={self.head_dim}, group_size={group_size}."
            )
        return group_size

    def _full_layer_kivi_block_bytes(self, dtype_size: int) -> int:
        group_size = self._full_layer_kivi_group_size()
        num_kv_heads = int(self.num_kv_heads)
        head_dim = int(self.head_dim)
        key_packed_width = group_size // 8
        value_packed_width = head_dim // 8
        value_groups = head_dim // group_size
        key_bytes = num_kv_heads * head_dim * key_packed_width * 4
        key_meta_bytes = num_kv_heads * head_dim * 2 * int(dtype_size)
        value_bytes = num_kv_heads * group_size * value_packed_width * 4
        value_meta_bytes = num_kv_heads * group_size * value_groups * 2 * int(dtype_size)
        return int(key_bytes + key_meta_bytes + value_bytes + value_meta_bytes)

    def _full_layer_kivi_blocks_for_tokens(self, total_len: int) -> int:
        group_size = self._full_layer_kivi_group_size()
        return (max(0, int(total_len)) + group_size - 1) // group_size

    def _full_layer_cluster_ratio(self) -> float:
        ratio = float(getattr(self.config, "full_layer_cluster_ratio", 0.0) or 0.0)
        return ratio if ratio > 0.0 else float(self.config.deltakv_center_ratio or 0.0)

    def _deltakv_reset_full_prefill_staging(self, *, clear_plans: bool = True):
        super()._deltakv_reset_full_prefill_staging(clear_plans=clear_plans)
        self._deltakv_clear_long_prefill_offload_prefetch()
        if hasattr(self, "_full_layer_kivi_full_prefill_plans"):
            if clear_plans:
                self._full_layer_kivi_full_prefill_plans = {}
            self._full_layer_kivi_full_prefill_materialized_layers = set()

    def _should_use_full_prefill_staging(self, seqs: list[Sequence]) -> bool:
        if super()._should_use_full_prefill_staging(seqs):
            return True
        if not self._full_layer_kivi_enabled():
            return False
        if getattr(self.config, "prefill_schedule_policy", None) != "long_bs1full_short_batch":
            return False
        if not self.deltakv_layer_ids or not seqs:
            return False
        for seq in seqs:
            if self.requires_long_prefill_offload(seq):
                return False
            remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
            if not (
                remaining > 0
                and int(seq.num_prefilled_tokens) == 0
                and int(seq.current_chunk_size) == remaining
            ):
                return False
        return True

    def should_schedule_full_prefill(self, seq: Sequence) -> bool:
        if not self._full_layer_kivi_enabled():
            return False
        if getattr(self.config, "prefill_schedule_policy", None) != "long_bs1full_short_batch":
            return False
        if not self.deltakv_layer_ids:
            return False
        if self.requires_long_prefill_offload(seq):
            return False
        if int(seq.num_prefilled_tokens) != 0:
            return False
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        if remaining <= 0:
            return False
        staging_slots = int(getattr(self, "deltakv_prefill_staging_num_slots", 0) or 0)
        if staging_slots > 0 and remaining > staging_slots:
            return False
        return True

    def has_prefill_staging_view(self, layer_idx: int) -> bool:
        if super().has_prefill_staging_view(layer_idx):
            return True
        return bool(
            self._full_layer_kivi_enabled()
            and self._deltakv_prefill_staging_active
            and layer_idx in self.full_layer_to_idx
        )

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        if self.should_schedule_full_prefill(seq):
            staging_slots = int(getattr(self, "deltakv_prefill_staging_num_slots", 0) or 0)
            return max(0, staging_slots - int(seq.num_prefilled_tokens))
        return super().prefill_step_free_slots_for(seq)

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        if self.should_schedule_full_prefill(seq):
            return 0
        return super().prefill_step_reservation_cost(seq, scheduled_tokens)

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        self._deltakv_less_memory_prepare_seqs = seqs
        self._deltakv_less_memory_full_prefill_staging_offset = 0
        self._deltakv_less_memory_prepare_full_prefill_staging = bool(
            is_prefill
            and (
                self._should_use_full_prefill_staging(seqs)
                or self._should_use_long_prefill_offload_staging(seqs)
            )
        )
        try:
            return super().prepare_step(seqs, is_prefill)
        finally:
            self._deltakv_less_memory_prepare_seqs = None
            self._deltakv_less_memory_full_prefill_staging_offset = 0
            self._deltakv_less_memory_prepare_full_prefill_staging = False

    @staticmethod
    def _packed_residual_bytes(payload_dim: int, quant_bits: int, dtype_size: int, group_size: int) -> int:
        if quant_bits:
            feat_per_int = 32 // int(quant_bits)
            if payload_dim % feat_per_int != 0:
                raise ValueError(
                    f"int{quant_bits} residual packing requires kv_dim divisible by "
                    f"{feat_per_int}, got {payload_dim}."
                )
            if group_size <= 0 or payload_dim % group_size != 0:
                raise ValueError(
                    "Quantized residual storage requires payload_dim divisible by group_size; "
                    f"payload_dim={payload_dim}, group_size={group_size}."
                )
            num_groups = payload_dim // group_size
            return (payload_dim // feat_per_int) * 4 + 2 * num_groups * dtype_size
        return payload_dim * dtype_size

    def _quant_group_size(self, payload_dim: int) -> int:
        group_size = int(getattr(self.config, "deltakv_latent_quant_group_size", 0) or 0)
        if group_size <= 0:
            group_size = int(payload_dim)
        if payload_dim % group_size != 0:
            raise ValueError(
                "DeltaKV slim runtime requires payload_dim divisible by deltakv_latent_quant_group_size; "
                f"payload_dim={payload_dim}, deltakv_latent_quant_group_size={group_size}."
            )
        return group_size

    def _sparse_payload_dim(self, kv_dim: int) -> int:
        del kv_dim
        return int(self.config.deltakv_latent_dim)

    @staticmethod
    def _resident_sparse_raw_overhead_slots(max_seqs: int, sink: int, recent: int) -> int:
        # Decode eviction runs after attention, so each sequence can temporarily
        # hold two recent windows plus the current token in raw sparse slots.
        return int(max_seqs) * (int(sink) + 2 * int(recent) + 1)

    @staticmethod
    def _decode_reconstruct_scratch_slots(max_seqs: int, top_decode: int, sink: int, recent: int) -> int:
        # Decode materializes the sparse attention view into post-RoPE temp slots:
        # selected top tokens plus a materialized raw/post-RoPE sparse view.
        # Use a conservative 2x top-k reserve until the materialized workspace
        # is backed by a reusable fixed buffer rather than temp full-cache slots.
        del sink, recent
        return int(max_seqs) * (2 * int(top_decode))

    def _resident_full_layer_raw_overhead_slots(self, max_seqs: int, sink: int, recent: int) -> int:
        if self._full_layer_kivi_enabled():
            group_size = self._full_layer_kivi_group_size()
            residual_length = int(
                getattr(self.config, "full_layer_kivi_residual_length", group_size)
                or group_size
            )
            return int(max_seqs) * (int(sink) + int(residual_length) + int(group_size) + 1)
        return int(max_seqs) * (int(sink) + 2 * int(recent) + 1)

    def _materialized_sparse_compute_slots(self, max_seqs: int, sink: int, recent: int, top_decode: int) -> int:
        return self.materialized_sparse_compute_slot_capacity(self.config, max_seqs, sink, recent, top_decode)

    @staticmethod
    def materialized_sparse_compute_slot_capacity(config, max_seqs: int, sink: int, recent: int, top_decode: int) -> int:
        max_decode_visible = int(sink) + int(top_decode) + max(2 * int(recent), int(recent) + 1)
        return max(1, int(max_seqs) * max_decode_visible, int(getattr(config, "max_num_batched_tokens", 0) or 0))

    def _already_postrope_mask(
        self,
        layer_idx: int,
        slots: torch.Tensor,
        already_postrope_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid = slots >= 0
        masks = getattr(self, "_deltakv_postrope_slot_mask", None)
        if masks is None:
            if already_postrope_slots is None or already_postrope_slots.numel() == 0:
                return torch.zeros_like(valid, dtype=torch.bool)
            already = already_postrope_slots.to(device=slots.device, dtype=slots.dtype).flatten()
            already = already[already >= 0]
            if already.numel() == 0:
                return torch.zeros_like(valid, dtype=torch.bool)
            membership = slots.reshape(-1, 1) == already.reshape(1, -1)
            return valid & membership.any(dim=1).view_as(slots)
        l_idx = self.deltakv_layer_to_idx[int(layer_idx)]
        layer_mask = masks[l_idx]
        safe_slots = slots.to(torch.long).clamp(0, int(layer_mask.shape[0]) - 1)
        return valid & layer_mask[safe_slots]

    def _set_postrope_slots(self, layer_idx: int, slots: torch.Tensor, *, validate: bool = True) -> None:
        masks = getattr(self, "_deltakv_postrope_slot_mask", None)
        if masks is None or slots is None:
            return
        l_idx = self.deltakv_layer_to_idx[int(layer_idx)]
        layer_mask = masks[l_idx]
        layer_mask.zero_()
        if slots.numel() == 0:
            return

        slots_i32 = slots.to(layer_mask.device).to(torch.int32).flatten()
        is_capturing = self._is_stream_capturing()
        if is_capturing:
            raise RuntimeError(
                "DeltaKV less-memory is not CUDA Graph capture-safe. "
                "Use sparse_method='deltakv' with decode_graph=True for graph runs."
            )
        else:
            if validate:
                valid = (slots_i32 >= 0) & (slots_i32 < int(layer_mask.shape[0]))
                layer_mask.index_fill_(0, slots_i32[valid].to(torch.long), True)
            else:
                layer_mask.index_fill_(0, slots_i32.to(torch.long), True)

    def _describe_deltakv_full_slots_for_debug(self, slots: torch.Tensor, row_idx: int, total_len: int) -> str:
        """Return expensive slot provenance details for invariant failures only."""
        if slots.numel() == 0:
            return "slot_debug=empty"
        slots_i32 = slots.to(device=self.device, dtype=torch.int32).flatten()
        slot_list = [int(x) for x in slots_i32.detach().cpu().tolist()]
        details: list[str] = []

        free_count = int(getattr(self, "_num_free_slots_deltakv_full", 0) or 0)
        free_stack = self.free_slots_stack_deltakv_full[:free_count]
        if free_stack.numel() > 0:
            free_hits = torch.isin(slots_i32, free_stack).detach().cpu().tolist()
        else:
            free_hits = [False for _ in slot_list]
        details.append(f"free_count={free_count}/{int(self.deltakv_full_num_slots)}")
        details.append(f"free_stack_contains={list(zip(slot_list, [bool(x) for x in free_hits]))}")

        temp_cache = getattr(self, "_deltakv_decode_static_temp_slots_by_shape", None) or {}
        temp_owners: list[tuple[int, str]] = []
        for slot in slot_list:
            owner = "none"
            for key, temp_slots in temp_cache.items():
                matches = (temp_slots == int(slot)).nonzero(as_tuple=False)
                if matches.numel() > 0:
                    owner = f"{key}@{matches[0].detach().cpu().tolist()}"
                    break
            temp_owners.append((slot, owner))
        details.append(f"static_temp_owners={temp_owners}")

        raw_map = self.sparse_layer_raw_slots_map[int(row_idx), : int(total_len)]
        raw_positions: list[tuple[int, list[int]]] = []
        for slot in slot_list:
            matches = (raw_map == int(slot)).nonzero(as_tuple=False).flatten()[:16]
            raw_positions.append((slot, [int(x) for x in matches.detach().cpu().tolist()]))
        details.append(f"raw_map_positions={raw_positions}")
        events = getattr(self, "_deltakv_debug_slot_events", None)
        if events:
            filtered_events = [
                event
                for event in events
                if any(f"slots={slot}" in event or f"slots=[{slot}" in event or f", {slot}" in event for slot in slot_list)
            ]
            details.append(f"debug_events={filtered_events[-32:]}")

        return " ".join(details)

    def _debug_track_deltakv_full_slots(self, slots: torch.Tensor | None, event: str, **fields):
        if os.getenv("DELTAKV_DEBUG_TRACK_FULL_SLOTS", "0") != "1":
            return
        if slots is None or slots.numel() == 0:
            return
        slots_list = [int(x) for x in slots.to(dtype=torch.int32).flatten().detach().cpu().tolist()]
        explicit = os.getenv("DELTAKV_DEBUG_TRACK_SLOT_IDS", "").strip()
        if explicit:
            tracked = {int(x) for x in explicit.replace(",", " ").split() if x.strip()}
        else:
            lo = os.getenv("DELTAKV_DEBUG_TRACK_SLOT_MIN", "").strip()
            hi = os.getenv("DELTAKV_DEBUG_TRACK_SLOT_MAX", "").strip()
            tracked = None
            if lo and hi:
                lo_i = int(lo)
                hi_i = int(hi)
                tracked = set(range(lo_i, hi_i + 1))
        if explicit or tracked is not None:
            hits = [slot for slot in slots_list if slot in tracked]
        else:
            hits = slots_list[:64]
        if not hits:
            return
        entries = getattr(self, "_deltakv_debug_slot_events", None)
        if entries is None:
            entries = []
            self._deltakv_debug_slot_events = entries
        field_text = ",".join(f"{key}={value}" for key, value in sorted(fields.items()))
        entries.append(f"{event}: slots={hits} {field_text}")
        del entries[:-512]

    def _allocate_deltakv_full(self, seq_id: int, size: int) -> torch.Tensor:
        slots = super()._allocate_deltakv_full(seq_id, size)
        row_idx = self.seq_id_to_row.get(int(seq_id), None)
        self._debug_track_deltakv_full_slots(slots, "alloc_persistent", seq_id=int(seq_id), row=row_idx, size=int(size))
        return slots

    def _allocate_deltakv_full_positions(self, seq_id: int, positions: torch.Tensor) -> torch.Tensor:
        slots = super()._allocate_deltakv_full_positions(seq_id, positions)
        row_idx = self.seq_id_to_row.get(int(seq_id), None)
        pos_sample = positions[:8].detach().cpu().tolist() if positions.numel() else []
        self._debug_track_deltakv_full_slots(
            slots,
            "alloc_positions",
            seq_id=int(seq_id),
            row=row_idx,
            positions=pos_sample,
            num_positions=int(positions.numel()),
        )
        return slots

    def _allocate_batch_deltakv_full(self, seq_ids: list[int], size: int) -> torch.Tensor:
        slots = super()._allocate_batch_deltakv_full(seq_ids, size)
        rows = [self.seq_id_to_row.get(int(seq_id), None) for seq_id in seq_ids]
        self._debug_track_deltakv_full_slots(slots, "alloc_batch", seq_ids=[int(x) for x in seq_ids], rows=rows, size=int(size))
        return slots

    def _allocate_temp_deltakv_full(self, size: int) -> torch.Tensor:
        slots = super()._allocate_temp_deltakv_full(size)
        self._debug_track_deltakv_full_slots(slots, "alloc_temp", size=int(size))
        return slots

    def free_temp_deltakv_full(self, slots: torch.Tensor | None):
        self._debug_track_deltakv_full_slots(slots, "free_temp")
        return super().free_temp_deltakv_full(slots)

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        refs = super().decode_graph_keepalive_tensors()
        if self._full_layer_quant_enabled():
            for attr_name in (
                "_full_layer_quant_k_cache",
                "_full_layer_quant_v_cache",
                "_full_layer_quant_active_slots",
                "_full_layer_quant_local_req",
                "_full_layer_quant_positions",
                "_full_layer_quant_out_slots",
            ):
                self._append_tensor_refs(refs, getattr(self, attr_name, None))
        self._append_tensor_refs(refs, getattr(self, "_deltakv_materialized_sparse_view_by_shape", None))
        return refs

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        dtype_size = torch.tensor([], dtype=self.hf_config.dtype).element_size()
        sink = int(config.sink_keep_tokens)
        recent = int(config.recent_keep_tokens)
        max_seqs = self._max_decode_scratch_seqs()
        max_admission_seqs = int(config.max_num_seqs_in_gpu)
        max_model_len = int(config.max_model_len)
        top_decode = int(config.decode_keep_tokens)
        self.deltakv_materialized_compute_num_slots = self._materialized_sparse_compute_slots(
            max_seqs,
            sink,
            recent,
            top_decode,
        )
        self.deltakv_prefill_staging_num_slots = self._deltakv_prefill_staging_capacity()
        prefill_staging_bytes = int(self.deltakv_prefill_staging_num_slots) * int(slot_bytes_per_layer)
        prefill_pre_rope_bytes = (
            int(self.deltakv_prefill_staging_num_slots)
            * int(self.num_kv_heads)
            * int(self.head_dim)
            * int(dtype_size)
        )
        materialized_compute_bytes = int(self.deltakv_materialized_compute_num_slots) * int(slot_bytes_per_layer)
        fixed_workspace_bytes = int(prefill_pre_rope_bytes) + int(materialized_compute_bytes)
        workspace_reserve_bytes = self._extra_workspace_reserve_bytes()
        persistent_memory = (
            int(available_memory)
            - int(prefill_staging_bytes)
            - fixed_workspace_bytes
            - workspace_reserve_bytes
        )
        if persistent_memory <= 0:
            raise RuntimeError(
                "Not enough GPU memory for DeltaKV less-memory workspaces. "
                f"staging_slots={self.deltakv_prefill_staging_num_slots} "
                f"materialized_slots={self.deltakv_materialized_compute_num_slots} "
                f"workspace_reserve_bytes={workspace_reserve_bytes}."
            )

        num_full_layers = len(self.full_layer_ids)
        num_deltakv_layers = len(self.deltakv_layer_ids)
        assert num_full_layers > 0, "DeltaKV less-memory requires at least one full-attention layer."
        assert num_deltakv_layers > 0, "DeltaKV less-memory requires at least one sparse layer."

        kv_dim = 2 * self.num_kv_heads * self.head_dim
        quant_bits = int(config.deltakv_latent_quant_bits or 0)
        if quant_bits not in (0, 4):
            raise ValueError(f"DeltaKV slim runtime supports deltakv_latent_quant_bits=0 or 4, got {quant_bits}.")

        sparse_payload_dim = self._sparse_payload_dim(kv_dim)
        sparse_group_size = self._quant_group_size(sparse_payload_dim) if quant_bits else sparse_payload_dim
        latent_bytes = self._packed_residual_bytes(sparse_payload_dim, quant_bits, dtype_size, sparse_group_size)

        deltakv_center_ratio = max(0.0, float(config.deltakv_center_ratio))
        full_quant_bits = int(getattr(config, "full_layer_kv_quant_bits", 0) or 0)
        if full_quant_bits not in (0, 4):
            raise ValueError(f"DeltaKV slim runtime supports full_layer_kv_quant_bits=0 or 4, got {full_quant_bits}.")
        full_quant_enabled = self._full_layer_quant_enabled()
        full_kivi_enabled = self._full_layer_kivi_enabled()
        sparse_ref_slot_bytes = 0
        full_cluster_ratio = max(0.0, self._full_layer_cluster_ratio()) if full_quant_enabled else 0.0
        full_group_size = self._quant_group_size(kv_dim) if full_quant_enabled else kv_dim
        full_latent_bytes = (
            self._packed_residual_bytes(kv_dim, full_quant_bits, dtype_size, full_group_size)
            if full_quant_enabled
            else 0
        )
        full_kivi_group_size = self._full_layer_kivi_group_size() if full_kivi_enabled else 0
        full_kivi_block_bytes = self._full_layer_kivi_block_bytes(dtype_size) if full_kivi_enabled else 0
        full_kivi_token_bytes = (
            float(full_kivi_block_bytes) / float(full_kivi_group_size)
            if full_kivi_enabled
            else 0.0
        )

        if full_quant_enabled:
            per_token_bytes = (
                num_full_layers * (full_cluster_ratio * slot_bytes_per_layer + full_latent_bytes)
                + num_deltakv_layers * (deltakv_center_ratio * slot_bytes_per_layer + latent_bytes)
            )
        elif full_kivi_enabled:
            per_token_bytes = (
                num_full_layers * full_kivi_token_bytes
                + num_deltakv_layers * (deltakv_center_ratio * slot_bytes_per_layer + latent_bytes)
            )
        else:
            per_token_bytes = (
                num_full_layers * slot_bytes_per_layer
                + num_deltakv_layers * (deltakv_center_ratio * slot_bytes_per_layer + latent_bytes)
            )
        if per_token_bytes <= 0:
            raise ValueError("Invalid DeltaKV less-memory allocation configuration.")

        total_top_slots = self._decode_reconstruct_scratch_slots(max_seqs, top_decode, sink, recent)
        deltakv_persistent_overhead_slots = self._resident_sparse_raw_overhead_slots(max_admission_seqs, sink, recent)
        deltakv_overhead_slots = deltakv_persistent_overhead_slots + total_top_slots
        if not full_kivi_enabled and getattr(config, "startup_cache_phase", None) == "profiling":
            # Reuse the startup workload definition instead of passing a
            # method-specific capacity field through ModelRunner/Config.
            from sparseengine.engine.startup.capacity import profiling_kv_slots

            deltakv_overhead_slots += profiling_kv_slots(config)
        full_overhead_slots = self._resident_full_layer_raw_overhead_slots(max_admission_seqs, sink, recent)

        capacity_memory = persistent_memory
        if not full_quant_enabled and not full_kivi_enabled:
            # Raw recent windows and decode reconstruction are fixed costs,
            # not part of the center-ratio cost of each resident token.
            capacity_memory -= deltakv_overhead_slots * (
                num_deltakv_layers * slot_bytes_per_layer + sparse_ref_slot_bytes
            )
            if capacity_memory <= 0:
                raise RuntimeError(
                    "Not enough GPU memory for DeltaKV fixed raw pools: "
                    f"persistent_memory={persistent_memory} raw_overhead_slots={deltakv_overhead_slots}."
                )
        memory_max_tokens = max(1, int(capacity_memory / per_token_bytes))
        reserve_ratio = float(config.deltakv_full_pool_reserve_ratio)
        if reserve_ratio > 0:
            reserve_ratio = max(0.0, min(0.5, reserve_ratio))
            memory_max_tokens = max(1, int(memory_max_tokens * (1.0 - reserve_ratio)))
        capacity_margin = float(getattr(config, "deltakv_cache_capacity_margin", 1.05) or 1.05)
        center_capacity_margin = float(getattr(config, "deltakv_center_capacity_margin", 1.5) or 1.5)
        configured_token_capacity = max(1, int(math.ceil(max_admission_seqs * max_model_len * capacity_margin)))
        max_tokens = min(memory_max_tokens, configured_token_capacity)
        estimated_deltakv_centers = int(self._estimate_deltakv_centers_for_total_len_exact(max_model_len))
        desired_deltakv_centers = max(
            1,
            int(math.ceil(max_admission_seqs * estimated_deltakv_centers * center_capacity_margin)),
        )
        self.deltakv_latent_num_slots = max_tokens
        self.full_layer_latent_num_slots = max_tokens if full_quant_enabled else 0
        self.full_layer_kivi_num_blocks = self._full_layer_kivi_blocks_for_tokens(max_tokens) if full_kivi_enabled else 0

        bytes_latent = self.deltakv_latent_num_slots * num_deltakv_layers * latent_bytes
        bytes_full_latent = self.full_layer_latent_num_slots * num_full_layers * full_latent_bytes
        bytes_full_kivi = self.full_layer_kivi_num_blocks * num_full_layers * full_kivi_block_bytes
        bytes_left = persistent_memory - bytes_latent - bytes_full_latent - bytes_full_kivi
        if bytes_left <= 0 and not full_kivi_enabled:
            raise RuntimeError("Not enough GPU memory left after allocating DeltaKV residual caches.")

        full_center_capacity = 0
        closed_loop_plan = None
        if full_quant_enabled:
            min_raw_bytes = (
                num_full_layers * full_overhead_slots * slot_bytes_per_layer
                + num_deltakv_layers * deltakv_overhead_slots * slot_bytes_per_layer
                + deltakv_overhead_slots * sparse_ref_slot_bytes
            )
            if bytes_left <= min_raw_bytes:
                raise RuntimeError(
                    "Not enough GPU memory for DeltaKV less-memory raw pools after residual allocation: "
                    f"bytes_left={bytes_left} required_min={min_raw_bytes}."
                )
            raw_extra_bytes = bytes_left - min_raw_bytes

            estimated_full_centers = int(self._estimate_full_layer_centers_for_total_len(max_model_len))
            desired_full_centers = max(
                1,
                int(math.ceil(max_admission_seqs * estimated_full_centers * center_capacity_margin)),
            )
            full_center_capacity = min(
                desired_full_centers,
                int(raw_extra_bytes // (num_full_layers * slot_bytes_per_layer)),
            )
            raw_extra_bytes -= full_center_capacity * num_full_layers * slot_bytes_per_layer
            self.full_num_slots = full_overhead_slots + full_center_capacity

            desired_centers = desired_deltakv_centers
            centers_capacity = min(
                desired_centers,
                int(raw_extra_bytes // (num_deltakv_layers * slot_bytes_per_layer + sparse_ref_slot_bytes)),
            )
            self.deltakv_full_num_slots = deltakv_overhead_slots + centers_capacity
        elif full_kivi_enabled:
            base_raw_bytes = (
                num_full_layers * full_overhead_slots * slot_bytes_per_layer
                + num_deltakv_layers * deltakv_overhead_slots * slot_bytes_per_layer
            )
            fixed_extra_bytes = 0
            if persistent_memory <= base_raw_bytes + fixed_extra_bytes:
                raise RuntimeError(
                    "Not enough GPU memory for DeltaKV raw overhead: "
                    f"persistent_memory={persistent_memory} "
                    f"base_raw_bytes={base_raw_bytes} "
                    f"fixed_extra_bytes={fixed_extra_bytes}."
                )

            latent_index_bytes_per_slot = (
                num_deltakv_layers
                * int(config.deltakv_neighbor_count)
                * 4
            )
            sparse_latent_bytes_per_slot = (
                num_deltakv_layers * latent_bytes
                + latent_index_bytes_per_slot
            )
            full_kivi_bytes_per_block = num_full_layers * full_kivi_block_bytes

            def plan_for_centers(center_cap: int) -> dict[str, int]:
                center_cap = int(center_cap)
                raw_slots = int(deltakv_overhead_slots + center_cap)
                latent_slots = self._max_resource_under_deltakv_center_budget(
                    center_cap,
                    self._estimate_deltakv_latent_slots_for_total_len,
                )
                latent_slots = max(1, int(math.ceil(latent_slots * capacity_margin)))
                kivi_blocks = self._max_resource_under_deltakv_center_budget(
                    center_cap,
                    self._estimate_full_layer_kivi_blocks_for_total_len,
                )
                kivi_blocks = max(1, int(math.ceil(kivi_blocks * capacity_margin)))
                raw_bytes = base_raw_bytes + (
                    center_cap * num_deltakv_layers * slot_bytes_per_layer
                )
                persistent_sparse_ref_bytes = (
                    raw_slots * sparse_ref_slot_bytes
                )
                packed_bytes = (
                    latent_slots * sparse_latent_bytes_per_slot
                    + kivi_blocks * full_kivi_bytes_per_block
                )
                total_bytes = raw_bytes + packed_bytes + fixed_extra_bytes + persistent_sparse_ref_bytes
                return {
                    "center_cap": center_cap,
                    "latent_slots": latent_slots,
                    "kivi_blocks": kivi_blocks,
                    "raw_bytes": int(raw_bytes),
                    "packed_bytes": int(packed_bytes),
                    "fixed_extra_bytes": int(fixed_extra_bytes),
                    "persistent_sparse_ref_bytes": int(persistent_sparse_ref_bytes),
                    "total_bytes": int(total_bytes),
                }

            lo, hi = 1, int(desired_deltakv_centers)
            best = None
            while lo <= hi:
                mid = (lo + hi) // 2
                plan = plan_for_centers(mid)
                if plan["total_bytes"] <= persistent_memory:
                    best = plan
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best is None:
                raise RuntimeError(
                    "Not enough GPU memory for any useful DeltaKV center capacity "
                    "after reserving raw overhead."
                )

            centers_capacity = int(best["center_cap"])
            self.full_num_slots = int(full_overhead_slots)
            self.deltakv_full_num_slots = int(deltakv_overhead_slots + centers_capacity)
            self.deltakv_latent_num_slots = int(best["latent_slots"])
            self.full_layer_latent_num_slots = 0
            self.full_layer_kivi_num_blocks = int(best["kivi_blocks"])
            closed_loop_plan = best
        else:
            self.full_num_slots = max_tokens
            bytes_full_layers = self.full_num_slots * num_full_layers * slot_bytes_per_layer
            bytes_left -= bytes_full_layers
            if bytes_left <= 0:
                raise RuntimeError(
                    "Not enough GPU memory left for DeltaKV less-memory full-KV pool after "
                    "allocating full layers + residual cache."
                )
            max_deltakv_full_slots = max(
                1,
                int(bytes_left // (num_deltakv_layers * slot_bytes_per_layer + sparse_ref_slot_bytes)),
            )
            if max_deltakv_full_slots <= deltakv_overhead_slots:
                raise RuntimeError(
                    f"DeltaKV less-memory full-KV pool too small: max={max_deltakv_full_slots}, "
                    f"required>={deltakv_overhead_slots + 1}."
                )
            desired_centers = desired_deltakv_centers
            centers_capacity = min(desired_centers, max_deltakv_full_slots - deltakv_overhead_slots)
            self.deltakv_full_num_slots = deltakv_overhead_slots + centers_capacity
            full_center_capacity = 0
        self._deltakv_centers_capacity = int(centers_capacity)
        self._deltakv_decode_reconstruct_full_reserve = min(self.deltakv_full_num_slots, int(total_top_slots))
        self._deltakv_temp_full_reserve = self._deltakv_decode_reconstruct_full_reserve
        persistent_sparse_ref_bytes_actual = int(self.deltakv_full_num_slots) * int(sparse_ref_slot_bytes)

        if closed_loop_plan is not None:
            logger.info(
                "DeltaKV less-memory closed-loop allocation: "
                f"full_layer_resident_raw_slots={self.full_num_slots}; "
                f"prefill_staging_slots={self.deltakv_prefill_staging_num_slots}; "
                f"sparse_raw_slots={self.deltakv_full_num_slots}; "
                f"sparse_raw_overhead={deltakv_overhead_slots}; "
                f"sparse_raw_persistent_overhead={deltakv_persistent_overhead_slots}; "
                f"sparse_raw_decode_reconstruct_reserve={self._deltakv_decode_reconstruct_full_reserve}; "
                f"center_capacity={centers_capacity}; "
                f"latent_slots={self.deltakv_latent_num_slots}; "
                f"full_kivi_blocks={self.full_layer_kivi_num_blocks}; "
                f"raw_bytes={closed_loop_plan['raw_bytes']}; "
                f"packed_bytes={closed_loop_plan['packed_bytes']}; "
                f"fixed_extra_bytes={closed_loop_plan['fixed_extra_bytes']}; "
                f"workspace_bytes={fixed_workspace_bytes}; "
                f"workspace_reserve_bytes={workspace_reserve_bytes}; "
                f"persistent_sparse_ref_bytes={closed_loop_plan.get('persistent_sparse_ref_bytes', 0)}; "
                f"total_planned_bytes={closed_loop_plan['total_bytes']}; "
                f"persistent_memory={persistent_memory}; "
                f"estimated_centers_per_seq={estimated_deltakv_centers}; "
                f"capacity_margin={capacity_margin}; "
                f"center_capacity_margin={center_capacity_margin}; "
                f"deltakv_full_pool_reserve_ratio={reserve_ratio:.3f} "
                "is not used to pre-shrink memory_max_tokens in this closed-loop path; "
                f"sparse_payload_dim={sparse_payload_dim}; "
                f"deltakv_latent_quant_bits={quant_bits}; "
                f"deltakv_latent_quant_group_size={sparse_group_size}; "
                f"use_compression={bool(getattr(config, 'use_compression', True))}; "
                f"full_layer_kivi_group_size={full_kivi_group_size}."
            )
        else:
            logger.info(
                f"DeltaKV less-memory allocation: full_layers_slots={self.full_num_slots}; "
                f"deltakv_full_slots={self.deltakv_full_num_slots} "
                f"(overhead={deltakv_overhead_slots}, "
                f"persistent_overhead={deltakv_persistent_overhead_slots}, "
                f"decode_reconstruct_reserve={self._deltakv_decode_reconstruct_full_reserve}, "
                f"centers={centers_capacity}); "
                f"residual_slots={self.deltakv_latent_num_slots}; kv_dim={kv_dim}; "
                f"memory_capacity_tokens={memory_max_tokens}; configured_capacity_tokens={configured_token_capacity}; "
                f"capacity_margin={capacity_margin}; center_capacity_margin={center_capacity_margin}; "
                f"estimated_deltakv_centers_per_seq={estimated_deltakv_centers}; "
                f"sparse_payload_dim={sparse_payload_dim}; deltakv_latent_quant_bits={quant_bits}; "
                f"deltakv_latent_quant_group_size={sparse_group_size}; use_compression={bool(getattr(config, 'use_compression', True))}; "
                f"full_layer_kv_quant_bits={full_quant_bits}; full_layer_centers={full_center_capacity}; "
                f"full_layer_kivi_blocks={self.full_layer_kivi_num_blocks}; "
                f"full_layer_kivi_group_size={full_kivi_group_size}; "
                f"workspace_bytes={fixed_workspace_bytes}; "
                f"workspace_reserve_bytes={workspace_reserve_bytes}; "
                f"persistent_sparse_ref_bytes={persistent_sparse_ref_bytes_actual}."
            )

        self.full_kv_cache = torch.empty(
            2,
            num_full_layers,
            self.full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )
        self.deltakv_full_kv_cache = torch.empty(
            2,
            num_deltakv_layers,
            self.deltakv_full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )
        self.deltakv_materialized_kv_cache = torch.empty(
            2,
            self.deltakv_materialized_compute_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )
        self._deltakv_postrope_slot_mask = torch.zeros(
            num_deltakv_layers,
            self.deltakv_full_num_slots,
            dtype=torch.bool,
            device=self.device,
        )
        self._deltakv_materialized_active_slots = None
        self._deltakv_materialized_local_req = None
        self._deltakv_materialized_flat_slots = None
        self.deltakv_prefill_staging_kv_cache = torch.empty(
            2,
            self.deltakv_prefill_staging_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )
        self.deltakv_prefill_staging_pre_rope_k_cache = torch.empty(
            self.deltakv_prefill_staging_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.dtype,
            device=self.device,
        )
        latent_width = sparse_payload_dim // (32 // quant_bits) if quant_bits else sparse_payload_dim
        latent_dtype = torch.int32 if quant_bits else self.hf_config.dtype
        self.deltakv_latent_cache = torch.empty(
            num_deltakv_layers,
            self.deltakv_latent_num_slots,
            latent_width,
            dtype=latent_dtype,
            device=self.device,
        )
        if quant_bits:
            sparse_num_groups = sparse_payload_dim // sparse_group_size
            self.deltakv_latent_scales = torch.empty(
                num_deltakv_layers,
                self.deltakv_latent_num_slots,
                sparse_num_groups,
                dtype=self.hf_config.dtype,
                device=self.device,
            )
            self.deltakv_latent_mins = torch.empty_like(self.deltakv_latent_scales)
        else:
            self.deltakv_latent_scales = None
            self.deltakv_latent_mins = None

        self.deltakv_latent_to_full_slots = torch.full(
            (num_deltakv_layers, self.deltakv_latent_num_slots, config.deltakv_neighbor_count),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.deltakv_slot_to_pos = torch.full(
            (self.deltakv_full_num_slots,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.full_layer_slot_to_pos = torch.full(
            (self.full_num_slots,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

        if full_quant_enabled:
            full_latent_width = kv_dim // (32 // full_quant_bits)
            full_num_groups = kv_dim // full_group_size
            self.full_layer_latent_cache = torch.empty(
                num_full_layers,
                self.full_layer_latent_num_slots,
                full_latent_width,
                dtype=torch.int32,
                device=self.device,
            )
            self.full_layer_latent_scales = torch.empty(
                num_full_layers,
                self.full_layer_latent_num_slots,
                full_num_groups,
                dtype=self.hf_config.dtype,
                device=self.device,
            )
            self.full_layer_latent_mins = torch.empty_like(self.full_layer_latent_scales)
            self.full_layer_latent_to_full_slots = torch.full(
                (num_full_layers, self.full_layer_latent_num_slots, config.deltakv_neighbor_count),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.free_slots_stack_full_layer_latent = torch.arange(
                self.full_layer_latent_num_slots,
                dtype=torch.int32,
                device=self.device,
            )
            self._num_free_slots_full_layer_latent = self.full_layer_latent_num_slots
            self.full_layer_latent_slots_map = torch.full(
                (self.max_buffer_rows, self.max_model_len),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.row_full_layer_compressed_lens = torch.zeros(
                (self.max_buffer_rows,), dtype=torch.int32, device="cpu"
            ).numpy()
            self._full_layer_quant_k_cache = None
            self._full_layer_quant_v_cache = None
            self._full_layer_quant_active_slots = None
            self._full_layer_quant_local_req = None
        else:
            self.full_layer_latent_cache = None
            self.full_layer_latent_scales = None
            self.full_layer_latent_mins = None
            self.full_layer_latent_to_full_slots = None
            self.free_slots_stack_full_layer_latent = None
            self._num_free_slots_full_layer_latent = 0
            self.full_layer_latent_slots_map = None
            self.row_full_layer_compressed_lens = None
            self._full_layer_quant_k_cache = None
            self._full_layer_quant_v_cache = None
            self._full_layer_quant_active_slots = None
            self._full_layer_quant_local_req = None

        if full_kivi_enabled:
            kivi_blocks = int(self.full_layer_kivi_num_blocks)
            key_packed_width = full_kivi_group_size // 8
            value_packed_width = self.head_dim // 8
            value_groups = self.head_dim // full_kivi_group_size
            self.full_layer_kivi_key_packed = torch.empty(
                num_full_layers,
                kivi_blocks,
                self.num_kv_heads,
                self.head_dim,
                key_packed_width,
                dtype=torch.int32,
                device=self.device,
            )
            self.full_layer_kivi_key_scales = torch.empty(
                num_full_layers,
                kivi_blocks,
                self.num_kv_heads,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self.full_layer_kivi_key_mins = torch.empty_like(self.full_layer_kivi_key_scales)
            self.full_layer_kivi_value_packed = torch.empty(
                num_full_layers,
                kivi_blocks,
                self.num_kv_heads,
                full_kivi_group_size,
                value_packed_width,
                dtype=torch.int32,
                device=self.device,
            )
            self.full_layer_kivi_value_scales = torch.empty(
                num_full_layers,
                kivi_blocks,
                self.num_kv_heads,
                full_kivi_group_size,
                value_groups,
                dtype=self.hf_config.dtype,
                device=self.device,
            )
            self.full_layer_kivi_value_mins = torch.empty_like(self.full_layer_kivi_value_scales)
            self.free_slots_stack_full_layer_kivi = torch.arange(kivi_blocks, dtype=torch.int32, device=self.device)
            self._num_free_slots_full_layer_kivi = kivi_blocks
            self.full_layer_kivi_block_slots_map = torch.full(
                (self.max_buffer_rows, self.max_model_len),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.full_layer_kivi_block_start_pos = torch.full(
                (kivi_blocks,),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self._full_layer_kivi_full_prefill_plans = {}
            self._full_layer_kivi_full_prefill_materialized_layers = set()
        else:
            self.full_layer_kivi_key_packed = None
            self.full_layer_kivi_key_scales = None
            self.full_layer_kivi_key_mins = None
            self.full_layer_kivi_value_packed = None
            self.full_layer_kivi_value_scales = None
            self.full_layer_kivi_value_mins = None
            self.free_slots_stack_full_layer_kivi = None
            self._num_free_slots_full_layer_kivi = 0
            self.full_layer_kivi_block_slots_map = None
            self.full_layer_kivi_block_start_pos = None
            self._full_layer_kivi_full_prefill_plans = {}
            self._full_layer_kivi_full_prefill_materialized_layers = set()
        self.row_full_layer_kivi_quantized_lens = (
            torch.zeros((self.max_buffer_rows,), dtype=torch.int32, device="cpu").numpy()
            if self._full_layer_kivi_enabled()
            else None
        )
        self.row_full_layer_kivi_quantized_lens_gpu = (
            torch.zeros((self.max_buffer_rows,), dtype=torch.int32, device=self.device)
            if self._full_layer_kivi_enabled()
            else None
        )

    def _init_compressor_modules(self, config, num_deltakv_layers: int):
        if not bool(getattr(config, "use_compression", True)):
            raise ValueError("DeltaKV slim runtime is compressor-only; set use_compression=True.")
        if not getattr(config, "deltakv_checkpoint_path", None):
            raise ValueError("DeltaKV compressor sparse layers require deltakv_checkpoint_path.")
        self.compress_down = []
        self.compress_up = []
        for _ in range(num_deltakv_layers):
            self.compress_down.append(create_compressor(is_down=True, config=config).to(device=self.device))
            self.compress_up.append(create_compressor(is_down=False, config=config).to(device=self.device))

    def _collect_k_norm_weights(self, layers, layer_ids: list[int]):
        weights = []
        eps = None
        for layer_idx in layer_ids:
            attn = getattr(layers[layer_idx], "self_attn", None)
            k_norm = getattr(attn, "k_norm", None)
            if k_norm is None:
                return None, None
            weights.append(
                k_norm.weight.detach()
                .to(device=self.device, dtype=self.hf_config.dtype)
                .clone()
            )
            eps = float(getattr(k_norm, "eps", 1e-6))
        return torch.stack(weights, dim=0) if weights else None, eps

    def set_model_layers(self, layers):
        self.deltakv_k_norm_weight, self.deltakv_k_norm_eps = self._collect_k_norm_weights(
            layers,
            self.deltakv_layer_ids,
        )

    def get_layer_store_tensors(
        self,
        layer_idx: int,
        *,
        k_post_rope: torch.Tensor,
        v: torch.Tensor,
        pre_rope_k: torch.Tensor | None = None,
        pre_rope_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.deltakv_layer_to_idx or self.has_prefill_staging_view(layer_idx):
            return k_post_rope, v
        source_k = pre_rope_k
        if source_k is None:
            raise RuntimeError("DeltaKV sparse raw storage requires pre-qk-norm/pre-RoPE key states.")
        source_v = pre_rope_v if pre_rope_v is not None else v
        if int(source_k.shape[0]) != int(k_post_rope.shape[0]) or int(source_v.shape[0]) != int(v.shape[0]):
            raise RuntimeError(
                "DeltaKV sparse raw storage shape mismatch: "
                f"k_raw={tuple(source_k.shape)} k_post_rope={tuple(k_post_rope.shape)} "
                f"v_raw={tuple(source_v.shape)} v={tuple(v.shape)}."
            )
        return source_k, source_v

    def _stores_sparse_raw_kv(self, layer_idx: int) -> bool:
        return layer_idx in self.deltakv_layer_to_idx and not self.has_prefill_staging_view(layer_idx)

    def save_raw_kv_if_needed(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        slot_mapping = None
        if self._stores_sparse_raw_kv(layer_idx):
            slot_mapping = self._store_layer_kv(layer_idx, k, v)
        elif self._prefill_pre_rope_stage_active() and layer_idx in self.deltakv_layer_to_idx:
            _, _, slot_mapping = self.get_layer_store_view(layer_idx)
        if slot_mapping is not None:
            self.on_pre_rope_kv_stored(layer_idx, k, v, slot_mapping)

    def on_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if not self._full_layer_kivi_enabled():
            return None
        if layer_idx not in self.full_layer_to_idx or not self.has_prefill_staging_view(layer_idx):
            return None
        return None

    def _ensure_materialized_sparse_view(self, batch_size: int, width: int, device: torch.device):
        batch_size = int(batch_size)
        width = int(width)
        total = batch_size * width
        if total > int(self.deltakv_materialized_compute_num_slots):
            raise RuntimeError(
                "DeltaKV materialized sparse workspace is too small: "
                f"need={total} capacity={int(self.deltakv_materialized_compute_num_slots)} "
                f"batch={batch_size} width={width}. Increase max_num_batched_tokens or reduce decode keep tokens."
            )
        cache = getattr(self, "_deltakv_materialized_sparse_view_by_shape", None)
        if cache is None:
            cache = {}
            self._deltakv_materialized_sparse_view_by_shape = cache
        key = (max(1, batch_size), max(1, width), str(device))
        buffers = cache.get(key)
        if buffers is None:
            active = torch.empty(
                max(1, batch_size),
                max(1, width),
                dtype=torch.int32,
                device=device,
            )
            flat = torch.arange(max(1, total), dtype=torch.int32, device=device)
            local_req = torch.arange(max(1, batch_size), dtype=torch.int32, device=device)
            buffers = (active, flat, local_req)
            cache[key] = buffers
        active, flat, local_req = buffers
        if total > 0:
            active[:batch_size, :width].copy_(flat[:total].view(batch_size, width))
        self._deltakv_materialized_active_slots = active
        self._deltakv_materialized_flat_slots = flat
        self._deltakv_materialized_local_req = local_req
        return active[:batch_size, :width], local_req[:batch_size]

    def get_layer_compute_view(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        selection: SparseSelection | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_idx not in self.deltakv_layer_to_idx or self.has_prefill_staging_view(layer_idx):
            return super().get_layer_compute_view(
                layer_idx,
                active_slots,
                req_indices,
                context_lens,
                selection,
            )
        if active_slots.dim() != 2:
            raise RuntimeError(
                "DeltaKV sparse materialization expects a 2D active slot table, "
                f"got shape={tuple(active_slots.shape)}."
            )
        batch_size, width = int(active_slots.shape[0]), int(active_slots.shape[1])
        total = batch_size * width
        local_active_slots, local_req = self._ensure_materialized_sparse_view(batch_size, width, active_slots.device)
        k_out = self.deltakv_materialized_kv_cache[0, :total]
        v_out = self.deltakv_materialized_kv_cache[1, :total]
        if total == 0:
            return k_out, v_out, local_active_slots, local_req, context_lens

        l_idx = self.deltakv_layer_to_idx[layer_idx]
        with profiler.record("deltakv_materialize_sparse_view"):
            if active_slots.is_cuda:
                k_norm_weight = getattr(self, "deltakv_k_norm_weight", None)
                if k_norm_weight is not None:
                    k_norm_weight = k_norm_weight[int(l_idx)]
                postrope_mask = getattr(self, "_deltakv_postrope_slot_mask", None)
                if postrope_mask is not None:
                    postrope_mask = postrope_mask[int(l_idx)]
                deltakv_materialize_sparse_view(
                    active_slots=active_slots,
                    context_lens=context_lens,
                    slot_to_pos=self.deltakv_slot_to_pos,
                    postrope_mask=postrope_mask,
                    k_cache=self.deltakv_full_kv_cache[0, l_idx],
                    v_cache=self.deltakv_full_kv_cache[1, l_idx],
                    out_k=k_out,
                    out_v=v_out,
                    cos_sin=self.cos_sin_cache,
                    k_norm_weight=k_norm_weight,
                    k_norm_eps=float(getattr(self, "deltakv_k_norm_eps", 1e-6) or 1e-6),
                    block_tokens=int(getattr(self.config, "deltakv_triton_materialize_block_tokens", 16) or 16),
                )
            else:
                flat_slots = active_slots.reshape(-1).to(torch.long).clamp_min(0)
                raw_k = self.deltakv_full_kv_cache[0, l_idx, flat_slots]
                raw_v = self.deltakv_full_kv_cache[1, l_idx, flat_slots]
                pos = self.deltakv_slot_to_pos[flat_slots].to(torch.long).clamp_min(0)
                k_normed = self._apply_sparse_k_norm_if_needed(l_idx, raw_k)
                cos_sin = self.cos_sin_cache[pos]
                cos, sin = cos_sin.chunk(2, dim=-1)
                k_postrope = apply_rotary_emb(k_normed, cos, sin)
                already_postrope = self._already_postrope_mask(layer_idx, flat_slots)
                k_postrope = torch.where(already_postrope.view(-1, 1, 1), raw_k, k_postrope)
                k_out.copy_(k_postrope.to(k_out.dtype))
                v_out.copy_(raw_v.to(v_out.dtype))
        return k_out, v_out, local_active_slots, local_req, context_lens

    def _full_layer_base_cluster_step(self) -> int:
        ratio = self._full_layer_cluster_ratio()
        if ratio <= 0.0:
            raise ValueError(f"full-layer residual quantization requires center ratio > 0, got {ratio}.")
        return max(1, int(1.0 / max(1e-6, ratio)))

    def _full_layer_center_rel_for_block(
        self,
        row_idx: int,
        *,
        start: int,
        end: int,
        update_state: bool,
    ) -> torch.Tensor:
        centers = self._deltakv_center_positions_cpu(
            start=int(start),
            end=int(end),
            base_step=self._full_layer_base_cluster_step(),
        )
        if not centers:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        rel = [pos - int(start) for pos in centers if int(start) <= pos < int(end)]
        if not rel:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        return torch.tensor(rel, dtype=torch.long, device=self.device)

    def _estimate_full_layer_centers_for_total_len(self, total_len: int) -> int:
        total_len = int(total_len)
        if not self._full_layer_quant_enabled():
            return total_len
        sink = int(self.config.sink_keep_tokens or 0)
        recent = int(self.config.recent_keep_tokens or 0)
        effective_end = max(sink, total_len - recent)
        if effective_end <= sink:
            return 0
        centers = self._deltakv_center_positions_cpu(
            start=sink,
            end=effective_end,
            base_step=self._full_layer_base_cluster_step(),
        )
        return len(centers)

    def decode_window_budgets(self) -> dict[str, int]:
        budgets = super().decode_window_budgets()
        if self._full_layer_kivi_enabled():
            budgets["full_layer_kivi_blocks"] = int(self._num_free_slots_full_layer_kivi)
        elif self._full_layer_quant_enabled():
            budgets["full_layer_latent"] = int(self._num_free_slots_full_layer_latent)
        return budgets

    def decode_window_costs(self, seq: Sequence, tokens: int) -> dict[str, int]:
        costs = super().decode_window_costs(seq, tokens)
        row = self.seq_id_to_row[seq.seq_id]
        end = int(self.row_seq_lens[row]) + int(tokens)
        if self._full_layer_kivi_enabled():
            group = self._full_layer_kivi_group_size()
            done = max(0, int(self.row_full_layer_kivi_quantized_lens[row])
                       - int(self.config.sink_keep_tokens)) // group
            residual = int(getattr(self.config, "full_layer_kivi_residual_length", group) or group)
            resident = int(self.row_seq_lens[row]) - done * group
            # Quantization frees whole groups after attention. Reserve the
            # append-before-quantize peak, including one slot if eviction is overdue.
            raw_peak = int(self.config.sink_keep_tokens) + residual + group
            costs["full_layers"] = min(int(tokens), max(1, raw_peak - resident))
            costs["full_layer_kivi_blocks"] = max(
                0, self._estimate_full_layer_kivi_blocks_for_total_len(end) - done)
        elif self._full_layer_quant_enabled():
            costs["full_layer_latent"] = max(
                0, end - int(self.config.sink_keep_tokens)
                - int(self.row_full_layer_compressed_lens[row]))
        return costs

    def prompt_admission_cost(self, seq: Sequence) -> int:
        if self._full_layer_kivi_enabled():
            total_len = int(seq.num_prompt_tokens)
            resident = int(self.config.sink_keep_tokens) + int(
                getattr(self.config, "full_layer_kivi_residual_length", self._full_layer_kivi_group_size())
                or self._full_layer_kivi_group_size()
            ) + 1
            return min(total_len, max(1, resident))
        if not self._full_layer_quant_enabled():
            return super().prompt_admission_cost(seq)
        total_len = int(seq.num_prompt_tokens)
        resident = min(total_len, int(self.config.sink_keep_tokens) + int(self.config.recent_keep_tokens))
        return resident + self._estimate_full_layer_centers_for_total_len(total_len)

    def prompt_admission_budgets(self, waiting_seqs, engine_prefill_chunk_size: int) -> dict[str, int]:
        budgets = super().prompt_admission_budgets(waiting_seqs, engine_prefill_chunk_size)
        latent_reserved = sum(
            max(0, reserved - (int(self.row_deltakv_compressed_lens[self.seq_id_to_row[seq_id]])
                              if seq_id in self.seq_id_to_row else 0))
            for seq_id, reserved in self._deltakv_latent_reserved_by_seq.items()
        )
        budgets["deltakv_latent"] = max(0, self._num_free_slots_deltakv_latent - latent_reserved)
        if self._full_layer_kivi_enabled():
            group = self._full_layer_kivi_group_size()
            kivi_reserved = sum(
                max(0, reserved - (max(0, int(self.row_full_layer_kivi_quantized_lens[self.seq_id_to_row[seq_id]])
                                           - int(self.config.sink_keep_tokens)) // group
                                   if seq_id in self.seq_id_to_row else 0))
                for seq_id, reserved in self._full_layer_kivi_reserved_by_seq.items()
            )
            budgets["full_layer_kivi_blocks"] = max(0, self._num_free_slots_full_layer_kivi - kivi_reserved)
        return budgets

    def _estimate_full_layer_kivi_blocks_for_total_len(self, total_len: int) -> int:
        total_len = int(total_len)
        group_size = self._full_layer_kivi_group_size()
        sink = min(int(self.config.sink_keep_tokens), total_len)
        residual_length = int(getattr(self.config, "full_layer_kivi_residual_length", group_size) or group_size)
        quant_tokens = max(0, total_len - sink - residual_length)
        return quant_tokens // group_size

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        costs = super().prompt_admission_costs(seq)
        total_len = int(seq.num_prompt_tokens)
        costs["deltakv_centers"] = self._estimate_deltakv_centers_for_total_len_exact(total_len)
        costs["deltakv_latent"] = self._estimate_deltakv_latent_slots_for_total_len(total_len)
        if self._full_layer_kivi_enabled():
            costs["full_layers"] = self.prompt_admission_cost(seq)
            costs["full_layer_kivi_blocks"] = self._estimate_full_layer_kivi_blocks_for_total_len(total_len)
        elif self._full_layer_quant_enabled():
            resident = min(total_len, int(self.config.sink_keep_tokens) + int(self.config.recent_keep_tokens))
            costs["full_layers"] = resident + self._estimate_full_layer_centers_for_total_len(total_len)
        return costs

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]):
        super().on_prompt_admitted(seq, costs)
        seq_id = int(seq.seq_id)
        if seq_id not in self._deltakv_latent_reserved_by_seq:
            latent = int(costs.get("deltakv_latent", 0) or 0)
            self._deltakv_latent_reserved_by_seq[seq_id] = latent
            self._deltakv_latent_reserved_total += latent
        if self._full_layer_kivi_enabled() and seq_id not in self._full_layer_kivi_reserved_by_seq:
            kivi = int(costs.get("full_layer_kivi_blocks", 0) or 0)
            self._full_layer_kivi_reserved_by_seq[seq_id] = kivi
            self._full_layer_kivi_reserved_total += kivi

    @torch.no_grad()
    def _allocate_full_layer_latent(self, size: int) -> torch.Tensor:
        if self._num_free_slots_full_layer_latent < size:
            raise RuntimeError(
                "Out of full-layer residual cache slots: "
                f"need={size}, free={self._num_free_slots_full_layer_latent}."
            )
        ptr = self._num_free_slots_full_layer_latent
        select_index = self.free_slots_stack_full_layer_latent[ptr - size: ptr]
        self._num_free_slots_full_layer_latent -= size
        return select_index

    def _should_stage_full_layer_kivi_prefill(self, seq: Sequence, size: int) -> bool:
        if not self._full_layer_kivi_enabled():
            return False
        # Only stage when prepare_step() has classified the entire homogeneous
        # prefill batch as a staging step. The per-step cursor in _allocate_full()
        # gives each short prompt a disjoint range in the shared staging buffer.
        if not bool(getattr(self, "_deltakv_less_memory_prepare_full_prefill_staging", False)):
            return False
        if int(size) != int(seq.current_chunk_size):
            return False
        return True

    @torch.no_grad()
    def _allocate_full(self, seq_id: int, size: int) -> torch.Tensor:
        seq = None
        for candidate in getattr(self, "_deltakv_less_memory_prepare_seqs", None) or []:
            if int(candidate.seq_id) == int(seq_id):
                seq = candidate
                break
        if seq is not None and self._should_stage_full_layer_kivi_prefill(seq, size):
            row_idx = self.seq_id_to_row.get(seq_id)
            cur_len = 0 if row_idx is None else int(self.row_seq_lens[row_idx])
            uses_offload_staging = self._should_use_long_prefill_offload_staging([seq])
            if cur_len != 0 and not uses_offload_staging:
                raise RuntimeError("Full-layer KIVI full-prefill staging only supports first-prefill prompts.")
            if uses_offload_staging:
                staging_start = cur_len
            else:
                staging_start = int(
                    getattr(self, "_deltakv_less_memory_full_prefill_staging_offset", 0)
                    or 0
                )
            staging_end = staging_start + int(size)
            if staging_end > int(self.deltakv_prefill_staging_num_slots):
                raise RuntimeError(
                    "Full-layer KIVI full-prefill staging capacity is too small: "
                    f"staging_end={staging_end} staging_slots={self.deltakv_prefill_staging_num_slots}."
                )
            if not uses_offload_staging and cur_len + int(size) > self.full_layer_slots_map.shape[1]:
                raise RuntimeError("KV row length exceeds max_model_len in DeltaKV full-layer KIVI staging.")
            staging_slots = torch.arange(staging_start, staging_end, dtype=torch.int32, device=self.device)
            row_idx = self._get_free_row(seq_id)
            if not uses_offload_staging:
                self._deltakv_less_memory_full_prefill_staging_offset = staging_end
                self.full_layer_slots_map[row_idx, cur_len: cur_len + int(size)] = staging_slots
            return staging_slots
        return super()._allocate_full(seq_id, size)

    def _prepare_full_prefill_staging_plan(
        self,
        seq: Sequence,
        row_idx: int,
        total_len: int,
        *,
        staging_start: int = 0,
    ):
        super()._prepare_full_prefill_staging_plan(
            seq,
            row_idx,
            total_len,
            staging_start=staging_start,
        )
        if self._full_layer_kivi_enabled():
            self._prepare_full_layer_kivi_full_prefill_plan(
                seq,
                row_idx,
                total_len,
                staging_start=staging_start,
            )

    def _prepare_full_layer_kivi_full_prefill_plan(
        self,
        seq: Sequence,
        row_idx: int,
        total_len: int,
        *,
        staging_start: int = 0,
    ):
        total_len = int(total_len)
        group_size = self._full_layer_kivi_group_size()
        sink = min(int(self.config.sink_keep_tokens), total_len)
        residual_length = int(getattr(self.config, "full_layer_kivi_residual_length", group_size) or group_size)
        if residual_length <= 0:
            raise ValueError(f"full_layer_kivi_residual_length must be > 0, got {residual_length}.")

        quant_rel_end = max(0, total_len - sink - residual_length)
        quant_rel_end = (quant_rel_end // group_size) * group_size
        quant_end = sink + quant_rel_end
        block_starts = tuple(range(sink, quant_end, group_size))

        keep_positions = tuple(range(0, sink)) + tuple(range(quant_end, total_len))
        keep_pos = self._tensor_from_positions(keep_positions)
        keep_slots = self._allocate_full_positions(seq.seq_id, keep_pos)

        block_slots = self._allocate_full_layer_kivi_blocks(len(block_starts))
        block_start_pos = self._tensor_from_positions(block_starts)

        self.full_layer_slots_map[row_idx, :total_len] = -1
        if keep_pos.numel() > 0:
            self.full_layer_slots_map[row_idx, keep_pos.to(torch.long)] = keep_slots
        block_pos = None
        if block_slots.numel() > 0:
            offsets = torch.arange(group_size, device=self.device, dtype=torch.long)
            block_pos = block_start_pos.to(torch.long)[:, None] + offsets[None, :]
            block_slot_values = block_slots.to(torch.int32)[:, None].expand(-1, group_size)
            self.full_layer_kivi_block_slots_map[row_idx, block_pos.reshape(-1)] = block_slot_values.reshape(-1)
            self.full_layer_kivi_block_start_pos[block_slots.to(torch.long)] = block_start_pos.to(torch.int32)

        if self.row_full_layer_kivi_quantized_lens is None:
            raise RuntimeError("Full-layer KIVI state was not initialized.")
        self.row_full_layer_kivi_quantized_lens[row_idx] = int(quant_end)
        self.row_full_layer_kivi_quantized_lens_gpu[row_idx] = int(quant_end)
        self._full_layer_kivi_full_prefill_plans[int(row_idx)] = {
            "row_idx": int(row_idx),
            "total_len": int(total_len),
            "staging_start": int(staging_start),
            "keep_pos": keep_pos,
            "keep_slots": keep_slots,
            "block_slots": block_slots,
            "block_pos": block_pos,
        }

    @torch.no_grad()
    def _full_layer_kivi_materialize_full_prefill_layer(self, layer_idx: int):
        with profiler.record("deltakv_full_prefill_kivi_materialize_total"):
            if layer_idx in self._full_layer_kivi_full_prefill_materialized_layers:
                raise RuntimeError(f"Full-layer KIVI full-prefill layer materialized twice: layer={layer_idx}.")
            if layer_idx not in self.full_layer_to_idx:
                return
            l_idx = self.full_layer_to_idx[layer_idx]
            k_stage = self.deltakv_prefill_staging_kv_cache[0]
            v_stage = self.deltakv_prefill_staging_kv_cache[1]
            block_chunk_size = self._full_layer_kivi_store_block_chunk_size()

            for plan in self._full_layer_kivi_full_prefill_plans.values():
                staging_start = int(plan.get("staging_start", 0) or 0)
                keep_pos = plan["keep_pos"].to(torch.long)
                keep_slots = plan["keep_slots"].to(torch.long)
                if keep_slots.numel() > 0:
                    with profiler.record("deltakv_full_prefill_kivi_copy_keep"):
                        stage_keep_pos = keep_pos + staging_start
                        self.full_kv_cache[0, l_idx, keep_slots] = k_stage[stage_keep_pos]
                        self.full_kv_cache[1, l_idx, keep_slots] = v_stage[stage_keep_pos]

                block_slots = plan["block_slots"].to(torch.long)
                if block_slots.numel() > 0:
                    if "block_pos" in plan:
                        block_pos_all = plan["block_pos"].to(torch.long)
                    else:
                        offsets = torch.arange(
                            self._full_layer_kivi_group_size(),
                            device=k_stage.device,
                            dtype=torch.long,
                        )
                        block_pos_all = plan["block_start_pos"].to(torch.long)[:, None] + offsets[None, :]
                    with profiler.record("deltakv_full_prefill_kivi_store_blocks"):
                        for start in range(0, int(block_slots.numel()), block_chunk_size):
                            end = min(int(block_slots.numel()), start + block_chunk_size)
                            block_pos = block_pos_all[start:end] + staging_start
                            self._store_full_layer_kivi_blocks(
                                l_idx=l_idx,
                                block_slots=block_slots[start:end],
                                key_post_rope=k_stage[block_pos],
                                value=v_stage[block_pos],
                            )

            self._full_layer_kivi_full_prefill_materialized_layers.add(layer_idx)

    def _full_layer_kivi_store_block_chunk_size(self) -> int:
        token_chunk_size = int(getattr(self.config, "mlp_chunk_size", 16384) or 16384)
        if token_chunk_size <= 0:
            raise RuntimeError(f"mlp_chunk_size must be > 0 for full-layer KIVI store, got {token_chunk_size}.")
        group_size = self._full_layer_kivi_group_size()
        return max(1, token_chunk_size // group_size)

    def _deltakv_finish_full_prefill_staging(self):
        super()._deltakv_finish_full_prefill_staging()
        self._deltakv_clear_long_prefill_offload_prefetch()
        self._full_layer_kivi_full_prefill_plans = {}
        self._full_layer_kivi_full_prefill_materialized_layers = set()

    @torch.no_grad()
    def _allocate_full_positions(self, seq_id: int, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.to(device=self.device, dtype=torch.int32).contiguous()
        size = int(positions.numel())
        if size == 0:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        if self._num_free_slots_full < size:
            raise RuntimeError(
                "Out of full-layer raw cache slots for KIVI keep tokens: "
                f"need={size}, free={self._num_free_slots_full}."
            )
        row_idx = self._get_free_row(seq_id)
        ptr = self._num_free_slots_full
        select_index = self.free_slots_stack_full[ptr - size: ptr].to(torch.int32)
        self._num_free_slots_full -= size
        self.full_layer_slots_map[row_idx, positions.to(torch.long)] = select_index
        if self.full_layer_slot_to_pos is not None:
            self.full_layer_slot_to_pos[select_index.to(torch.long)] = positions
        self._consume_full_layer_reservation(seq_id, size)
        return select_index

    @torch.no_grad()
    def _allocate_full_layer_kivi_blocks(self, size: int) -> torch.Tensor:
        size = int(size)
        if size == 0:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        if self._num_free_slots_full_layer_kivi < size:
            raise RuntimeError(
                "Out of full-layer KIVI packed blocks: "
                f"need={size}, free={self._num_free_slots_full_layer_kivi}."
            )
        ptr = self._num_free_slots_full_layer_kivi
        select_index = self.free_slots_stack_full_layer_kivi[ptr - size: ptr].to(torch.int32)
        self._num_free_slots_full_layer_kivi -= size
        return select_index

    def _store_full_layer_kivi_blocks(
        self,
        *,
        l_idx: int,
        block_slots: torch.Tensor,
        key_post_rope: torch.Tensor,
        value: torch.Tensor,
    ):
        group_size = self._full_layer_kivi_group_size()
        quant_bits = int(getattr(self.config, "full_layer_kv_quant_bits", 0) or 0)
        if quant_bits != 4:
            raise ValueError(f"Full-layer KIVI storage expects int4, got {quant_bits}.")
        block_slots = block_slots.to(device=key_post_rope.device, dtype=torch.long).contiguous()
        num_blocks = int(block_slots.numel())
        if num_blocks == 0:
            return
        expected_shape = (num_blocks, group_size, self.num_kv_heads, self.head_dim)
        if tuple(key_post_rope.shape) != expected_shape:
            raise ValueError(
                "Full-layer KIVI key blocks shape mismatch: "
                f"got={tuple(key_post_rope.shape)} expected={expected_shape}."
            )
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"Full-layer KIVI value blocks shape mismatch: got={tuple(value.shape)}.")

        key_states = key_post_rope.permute(0, 2, 3, 1).contiguous()
        packed_k, scale_k, mn_k = triton_quantize_and_pack_along_last_dim(key_states, group_size, quant_bits)
        self.full_layer_kivi_key_packed[l_idx, block_slots] = packed_k.to(torch.int32)
        self.full_layer_kivi_key_scales[l_idx, block_slots] = scale_k.squeeze(-1).to(
            self.full_layer_kivi_key_scales.dtype
        )
        self.full_layer_kivi_key_mins[l_idx, block_slots] = mn_k.squeeze(-1).to(
            self.full_layer_kivi_key_mins.dtype
        )

        value_states = value.permute(0, 2, 1, 3).contiguous()
        packed_v, scale_v, mn_v = triton_quantize_and_pack_along_last_dim(value_states, group_size, quant_bits)
        self.full_layer_kivi_value_packed[l_idx, block_slots] = packed_v.to(torch.int32)
        self.full_layer_kivi_value_scales[l_idx, block_slots] = scale_v.to(self.full_layer_kivi_value_scales.dtype)
        self.full_layer_kivi_value_mins[l_idx, block_slots] = mn_v.to(self.full_layer_kivi_value_mins.dtype)

    def _deltakv_long_prefill_offload_kind(self, layer_idx: int) -> str:
        if layer_idx in self.deltakv_layer_to_idx:
            return "sparse_pre_rope"
        if self._full_layer_kivi_enabled() and layer_idx in self.full_layer_to_idx:
            return "full_post_rope"
        raise RuntimeError(f"DeltaKV long-prefill offload does not own layer={layer_idx}.")

    def _deltakv_long_prefill_offload_layer_order(self) -> list[int]:
        layers = set(int(layer_idx) for layer_idx in self.deltakv_layer_to_idx)
        if self._full_layer_kivi_enabled():
            layers.update(int(layer_idx) for layer_idx in self.full_layer_to_idx)
        return sorted(layers)

    def _deltakv_long_prefill_offload_prefetch_enabled(self) -> bool:
        if os.getenv("SPARSEENGINE_RAWKV_PREFETCH", "1") == "0":
            return False
        return device_runtime.supports_streams(self.device)

    def _deltakv_clear_long_prefill_offload_prefetch(self):
        states = getattr(self, "_deltakv_long_prefill_offload_prefetch_states", None)
        if states is None:
            state = getattr(self, "_deltakv_long_prefill_offload_prefetch_state", None)
            states = {} if state is None else {self._deltakv_prefetch_key_from_state(state): state}
        for state in list(states.values()):
            event = state.get("event")
            if event is not None:
                device_runtime.wait_event(event, device=self.device)
        self._deltakv_long_prefill_offload_prefetch_states = {}
        self._deltakv_long_prefill_offload_prefetch_state = None

    @staticmethod
    def _deltakv_prefetch_key_from_state(state: dict) -> tuple[int, int, str, int]:
        return (
            int(state["layer_idx"]),
            int(state["row_idx"]),
            str(state["kind"]),
            int(state["end"]),
        )

    def _deltakv_drop_long_prefill_offload_prefetch(self, key: tuple[int, int, str, int]):
        states = getattr(self, "_deltakv_long_prefill_offload_prefetch_states", None) or {}
        state = states.pop(key, None)
        if state is not None:
            event = state.get("event")
            if event is not None:
                device_runtime.wait_event(event, device=self.device)
        self._deltakv_long_prefill_offload_prefetch_states = states

    def _deltakv_consume_long_prefill_offload_staged_prefetch(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        kind: str,
        end: int,
    ) -> bool:
        key = (int(layer_idx), int(row_idx), str(kind), int(end))
        states = getattr(self, "_deltakv_long_prefill_offload_prefetch_states", None)
        if states is None:
            old_state = getattr(self, "_deltakv_long_prefill_offload_prefetch_state", None)
            states = {} if old_state is None else {self._deltakv_prefetch_key_from_state(old_state): old_state}
            self._deltakv_long_prefill_offload_prefetch_states = states
            self._deltakv_long_prefill_offload_prefetch_state = None
        state = states.pop(key, None)
        if state is None:
            return False
        if not bool(state.get("direct_stage", False)):
            # Backward-compatible cleanup for any stale temporary-tensor state.
            event = state.get("event")
            if event is not None:
                device_runtime.wait_event(event, device=self.device)
            self._deltakv_long_prefill_offload_prefetch_states = states
            return False
        with profiler.record("deltakv_long_prefill_offload_prefetch_wait"):
            device_runtime.wait_event(state["event"], device=self.device)
        self._deltakv_long_prefill_offload_prefetch_states = states
        return True

    def _deltakv_copy_long_prefill_offload_prefix_to_staging(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        kind: str,
        end: int,
    ) -> None:
        if kind == "sparse_pre_rope":
            k_dst = self.deltakv_prefill_staging_pre_rope_k_cache[:end]
            v_dst = self.deltakv_prefill_staging_kv_cache[1, :end]
        else:
            k_dst = self.deltakv_prefill_staging_kv_cache[0, :end]
            v_dst = self.deltakv_prefill_staging_kv_cache[1, :end]
        with profiler.record("deltakv_long_prefill_offload_direct_stage_miss_copy"):
            self.raw_kv_offload_buffer.copy_prefix_to(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=kind,
                end=end,
                k_out=k_dst,
                v_out=v_dst,
            )

    def _deltakv_schedule_next_long_prefill_offload_prefetch(self, *, layer_idx: int, row_idx: int, end: int):
        if int(end) <= 0 or not self._deltakv_long_prefill_offload_prefetch_enabled():
            return
        future_layers = [
            candidate
            for candidate in self._deltakv_long_prefill_offload_layer_order()
            if int(candidate) > int(layer_idx)
        ][:1]
        if not future_layers:
            return
        states = getattr(self, "_deltakv_long_prefill_offload_prefetch_states", None) or {}
        keep_layers = set(int(layer) for layer in future_layers)
        for key in list(states):
            key_layer, key_row, _key_kind, key_end = key
            if key_layer <= int(layer_idx) or key_row != int(row_idx) or key_end != int(end) or key_layer not in keep_layers:
                self._deltakv_drop_long_prefill_offload_prefetch(key)
                states = getattr(self, "_deltakv_long_prefill_offload_prefetch_states", None) or {}
        stream = getattr(self, "_deltakv_long_prefill_offload_prefetch_stream", None)
        if stream is None:
            stream = device_runtime.new_stream(device=self.device)
            if stream is None:
                raise RuntimeError(
                    "DeltaKV long-prefill offload prefetch is enabled, but the active platform "
                    f"does not support streams for device={self.device}."
                )
            self._deltakv_long_prefill_offload_prefetch_stream = stream
        for next_layer in future_layers:
            kind = self._deltakv_long_prefill_offload_kind(next_layer)
            key = (int(next_layer), int(row_idx), kind, int(end))
            if key in states:
                continue
            with profiler.record("deltakv_long_prefill_offload_prefetch_schedule"):
                staging_available_event = device_runtime.new_event(device=self.device)
                if staging_available_event is None:
                    raise RuntimeError(
                        "DeltaKV long-prefill offload prefetch could not create a staging event "
                        f"for device={self.device}."
                    )
                device_runtime.record_event(staging_available_event, device=self.device)
                with device_runtime.stream_context(stream):
                    device_runtime.stream_wait_event(stream, staging_available_event)
                    if kind == "sparse_pre_rope":
                        k_dst = self.deltakv_prefill_staging_pre_rope_k_cache[:end]
                        v_dst = self.deltakv_prefill_staging_kv_cache[1, :end]
                    else:
                        k_dst = self.deltakv_prefill_staging_kv_cache[0, :end]
                        v_dst = self.deltakv_prefill_staging_kv_cache[1, :end]
                    self.raw_kv_offload_buffer.copy_prefix_to(
                        layer_idx=next_layer,
                        row_idx=row_idx,
                        kind=kind,
                        end=end,
                        k_out=k_dst,
                        v_out=v_dst,
                    )
                    event = device_runtime.new_event(device=self.device)
                    if event is None:
                        raise RuntimeError(
                            "DeltaKV long-prefill offload prefetch could not create a completion event "
                            f"for device={self.device}."
                        )
                    device_runtime.record_event(event, device=self.device)
            states[key] = {
                "layer_idx": int(next_layer),
                "row_idx": int(row_idx),
                "kind": kind,
                "end": int(end),
                "direct_stage": True,
                "staging_available_event": staging_available_event,
                "event": event,
            }
        self._deltakv_long_prefill_offload_prefetch_states = states

    def _deltakv_schedule_post_layer_long_prefill_offload_prefetch(self, layer_idx: int):
        if not bool(getattr(self, "_deltakv_long_prefill_offload_step_active", False)):
            return
        start = int(getattr(self, "_deltakv_long_prefill_offload_start", 0) or 0)
        if start <= 0:
            return
        row_idx = int(getattr(self, "_deltakv_long_prefill_offload_row_idx", -1))
        if row_idx < 0:
            raise RuntimeError("DeltaKV long-prefill offload prefetch has no active row.")
        with profiler.record("deltakv_long_prefill_offload_after_attention_prefetch"):
            self._deltakv_schedule_next_long_prefill_offload_prefetch(
                layer_idx=layer_idx,
                row_idx=row_idx,
                end=start,
            )

    def _deltakv_long_prefill_restore_block_tokens(self) -> int:
        config = getattr(self, "config", None)
        configured = int(getattr(config, "engine_prefill_chunk_size", 65536) or 65536)
        return max(1, min(configured, 65536))

    def _deltakv_restore_sparse_prefix_to_staging(self, layer_idx: int, start: int) -> None:
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        block_tokens = self._deltakv_long_prefill_restore_block_tokens()
        k_src = self.deltakv_prefill_staging_pre_rope_k_cache
        k_dst = self.deltakv_prefill_staging_kv_cache[0]
        for lo in range(0, int(start), int(block_tokens)):
            hi = min(int(start), lo + int(block_tokens))
            pos = torch.arange(lo, hi, dtype=torch.long, device=self.device)
            k_normed = self._apply_sparse_k_norm_if_needed(l_idx, k_src[lo:hi])
            k_postrope = self._apply_sparse_rope_to_key(pos, k_normed)
            k_dst[lo:hi] = k_postrope.to(k_dst.dtype)
            del pos, k_normed, k_postrope

    @torch.no_grad()
    def before_prefill_layer_attention(self, layer_idx: int, selection: SparseSelection):
        del selection
        if not bool(getattr(self, "_deltakv_long_prefill_offload_step_active", False)):
            return None
        if not self.has_prefill_staging_view(layer_idx):
            return None
        start = int(getattr(self, "_deltakv_long_prefill_offload_start", 0) or 0)
        if start <= 0:
            return None
        row_idx = int(getattr(self, "_deltakv_long_prefill_offload_row_idx", -1))
        if row_idx < 0:
            raise RuntimeError("DeltaKV long-prefill offload restore has no active row.")

        kind = self._deltakv_long_prefill_offload_kind(layer_idx)
        with profiler.record("deltakv_long_prefill_offload_before_attention_wait_or_restore"):
            staged = self._deltakv_consume_long_prefill_offload_staged_prefetch(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=kind,
                end=start,
            )
            if not staged:
                self._deltakv_copy_long_prefill_offload_prefix_to_staging(
                    layer_idx=layer_idx,
                    row_idx=row_idx,
                    kind=kind,
                    end=start,
                )
        if kind == "sparse_pre_rope":
            with profiler.record("deltakv_long_prefill_offload_restore_sparse_rerope"):
                self._deltakv_restore_sparse_prefix_to_staging(layer_idx, start)
        return None

    @torch.no_grad()
    def _offload_long_prefill_offload_layer(self, layer_idx: int):
        start = int(getattr(self, "_deltakv_long_prefill_offload_start", 0) or 0)
        end = int(getattr(self, "_deltakv_long_prefill_offload_end", 0) or 0)
        total_len = int(getattr(self, "_deltakv_long_prefill_offload_total_len", 0) or 0)
        row_idx = int(getattr(self, "_deltakv_long_prefill_offload_row_idx", -1))
        if row_idx < 0 or end <= start:
            raise RuntimeError(
                "DeltaKV long-prefill offload has invalid range: "
                f"row={row_idx} start={start} end={end}."
            )
        kind = self._deltakv_long_prefill_offload_kind(layer_idx)
        if kind == "sparse_pre_rope":
            k = self.deltakv_prefill_staging_pre_rope_k_cache[start:end]
            v = self.deltakv_prefill_staging_kv_cache[1, start:end]
        else:
            k = self.deltakv_prefill_staging_kv_cache[0, start:end]
            v = self.deltakv_prefill_staging_kv_cache[1, start:end]
        with profiler.record("deltakv_long_prefill_offload_ensure_entry"):
            self.raw_kv_offload_buffer.ensure_entry(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=kind,
                total_len=total_len,
                k_shape_tail=tuple(k.shape[1:]),
                v_shape_tail=tuple(v.shape[1:]),
                dtype=k.dtype,
            )
        with profiler.record("deltakv_long_prefill_offload_put_range"):
            self.raw_kv_offload_buffer.put_range(
                layer_idx=layer_idx,
                row_idx=row_idx,
                kind=kind,
                start=start,
                k=k,
                v=v,
            )

    def on_layer_attention_end(self, layer_idx: int):
        if not self.has_prefill_staging_view(layer_idx):
            return

        if bool(getattr(self, "_deltakv_long_prefill_offload_step_active", False)) and not bool(
            getattr(self, "_deltakv_long_prefill_offload_is_last_chunk", False)
        ):
            self._offload_long_prefill_offload_layer(layer_idx)
            self._deltakv_schedule_post_layer_long_prefill_offload_prefetch(layer_idx)
            return

        if self._full_layer_kivi_enabled() and layer_idx in self.full_layer_to_idx:
            self._full_layer_kivi_materialize_full_prefill_layer(layer_idx)
        elif layer_idx in self.deltakv_layer_to_idx:
            for plan in (getattr(self, "_deltakv_full_prefill_plans", None) or {}).values():
                self._debug_track_deltakv_full_slots(
                    plan.get("center_slots"),
                    "full_prefill_centers",
                    row=int(plan["row_idx"]),
                    layer=int(layer_idx),
                    evict_start=int(plan["evict_start"]),
                    num_centers=int(plan["center_slots"].numel()),
                )
            self._deltakv_compress_full_prefill_layer(layer_idx)
        else:
            return

        self._deltakv_schedule_post_layer_long_prefill_offload_prefetch(layer_idx)

        full_prefill_plans = getattr(self, "_full_layer_kivi_full_prefill_plans", {}) or {}
        materialized_layers = getattr(self, "_full_layer_kivi_full_prefill_materialized_layers", set())
        required_full_layers = (
            len(getattr(self, "full_layer_ids", []))
            if self._full_layer_kivi_enabled() and full_prefill_plans
            else 0
        )
        if (
            len(getattr(self, "_deltakv_full_prefill_compressed_layers", set())) == len(getattr(self, "deltakv_layer_ids", []))
            and len(materialized_layers) == required_full_layers
        ):
            self._deltakv_finish_full_prefill_staging()

    def _quantize_residual_bits(
        self,
        residual: torch.Tensor,
        quant_bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quant_bits = int(quant_bits or 0)
        if quant_bits != 4:
            raise ValueError(f"Quantized residual storage requires quant_bits=4, got {quant_bits}.")
        group_size = self._quant_group_size(int(residual.shape[-1]))
        if residual.dim() == 2:
            return triton_quantize_and_pack_2d_int4_grouped(residual, group_size)
        packed, scale, mn = triton_quantize_and_pack_along_last_dim(
            residual.unsqueeze(0).unsqueeze(0),
            group_size,
            quant_bits,
        )
        return packed.squeeze(0).squeeze(0), scale.squeeze(0).squeeze(0), mn.squeeze(0).squeeze(0)

    def _dequantize_residual_bits(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        mn: torch.Tensor,
        kv_dim: int,
        quant_bits: int,
    ) -> torch.Tensor:
        quant_bits = int(quant_bits or 0)
        if quant_bits != 4:
            raise ValueError(f"Quantized residual load requires quant_bits=4, got {quant_bits}.")
        group_size = self._quant_group_size(int(kv_dim))
        if packed.dim() == 2:
            return triton_dequantize_2d_int4_grouped(packed, scale, mn, group_size, int(packed.shape[-1]) * 8)
        return unpack_quantized_to_16bit(
            packed.unsqueeze(0).unsqueeze(0),
            scale.unsqueeze(0).unsqueeze(0),
            mn.unsqueeze(0).unsqueeze(0),
            group_size,
            quant_bits,
        ).squeeze(0).squeeze(0)

    def _quantize_residual(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._quantize_residual_bits(residual, int(self.config.deltakv_latent_quant_bits or 0))

    def _dequantize_residual(self, packed: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, kv_dim: int) -> torch.Tensor:
        return self._dequantize_residual_bits(packed, scale, mn, kv_dim, int(self.config.deltakv_latent_quant_bits or 0))

    def _store_residual(self, l_idx: int, latent_slots: torch.Tensor, residual: torch.Tensor):
        if latent_slots.numel() != residual.shape[0]:
            raise RuntimeError(
                "DeltaKV less-memory latent residual store shape mismatch: "
                f"slots={int(latent_slots.numel())}, residual_rows={int(residual.shape[0])}."
            )
        validate_slots = os.getenv("SPARSEENGINE_VALIDATE_DELTAKV_LATENT_SLOTS", "0") == "1"
        if validate_slots and not self._is_stream_capturing():
            latent_cap = int(self.deltakv_latent_cache.shape[1])
            bad_latent = (latent_slots < 0) | (latent_slots >= latent_cap)
            if bool(bad_latent.any()):
                bad = latent_slots[bad_latent][:16].detach().cpu().tolist()
                raise RuntimeError(
                    "DeltaKV less-memory latent residual store got slot outside cache: "
                    f"cap={latent_cap}, bad={bad}."
                )
        if int(self.config.deltakv_latent_quant_bits or 0) == 4:
            packed, scale, mn = self._quantize_residual(residual)
            self.deltakv_latent_cache[l_idx, latent_slots] = packed.to(self.deltakv_latent_cache.dtype)
            self.deltakv_latent_scales[l_idx, latent_slots] = scale.to(self.deltakv_latent_scales.dtype)
            self.deltakv_latent_mins[l_idx, latent_slots] = mn.to(self.deltakv_latent_mins.dtype)
        else:
            self.deltakv_latent_cache[l_idx, latent_slots] = residual.to(self.deltakv_latent_cache.dtype)

    def _deltakv_store_layer_latent(
        self,
        *,
        l_idx: int,
        latent_slots: torch.Tensor,
        kv_block: torch.Tensor,
        base_kv: torch.Tensor,
        to_compress_mask: torch.Tensor | None = None,
        store_indices: torch.Tensor | None = None,
        store_all: bool = False,
    ):
        if store_all:
            selected_count = int(kv_block.shape[1])
            if int(latent_slots.numel()) != selected_count:
                raise RuntimeError(
                    "DeltaKV less-memory latent store_all mismatch: "
                    f"kv_tokens={selected_count}, latent_slots={int(latent_slots.numel())}."
                )
            selected_indices = None
        elif store_indices is not None:
            store_indices = store_indices.to(device=kv_block.device, dtype=torch.long)
            selected_indices = store_indices
        else:
            if to_compress_mask is None:
                raise RuntimeError("DeltaKV latent store requires to_compress_mask unless store_all=True.")
            selected_indices = torch.nonzero(
                to_compress_mask.to(device=kv_block.device),
                as_tuple=False,
            ).flatten().to(torch.long)
        selected_count = int(kv_block.shape[1]) if store_all else int(selected_indices.numel())
        if selected_count != int(latent_slots.numel()):
            raise RuntimeError(
                "DeltaKV less-memory latent store selected-token mismatch: "
                f"selected={selected_count}, latent_slots={int(latent_slots.numel())}."
            )
        if selected_count == 0:
            return
        chunk_size = self._deltakv_latent_store_chunk_size()
        down = self.compress_down[l_idx]
        for start in range(0, selected_count, chunk_size):
            end = min(selected_count, start + chunk_size)
            chunk_latent_slots = latent_slots[start:end]
            if store_all:
                kv_to_store = kv_block[:, start:end]
                base_to_store = base_kv[:, start:end]
            else:
                chunk_indices = selected_indices[start:end]
                kv_to_store = kv_block.index_select(1, chunk_indices)
                base_to_store = base_kv.index_select(1, chunk_indices)
            pair_count = int(kv_to_store.shape[1])
            # Pairing improves small eviction chunks but raises peak memory and slows large full-prefill chunks.
            if pair_count <= 1024:
                encoded = down(torch.cat((kv_to_store, base_to_store), dim=1))
                residual = encoded[:, :pair_count]
                residual.sub_(encoded[:, pair_count:])
            else:
                residual = down(kv_to_store)
                residual.sub_(down(base_to_store))
            residual = residual.squeeze(0)
            self._store_residual(l_idx, chunk_latent_slots, residual)

    def _deltakv_latent_store_chunk_size(self) -> int:
        chunk_size = int(getattr(self.config, "mlp_chunk_size", 16384) or 16384)
        if chunk_size <= 0:
            raise RuntimeError(f"mlp_chunk_size must be > 0 for DeltaKV latent store, got {chunk_size}.")
        return chunk_size

    def _prefill_pre_rope_stage_active(self) -> bool:
        return bool(
            getattr(self, "_deltakv_prefill_staging_active", False)
            and hasattr(self, "deltakv_prefill_staging_pre_rope_k_cache")
        )

    @torch.no_grad()
    def on_pre_rope_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        slot_mapping: torch.Tensor,
    ):
        needs_pre_rope = (
            get_context().is_prefill
            and self._prefill_pre_rope_stage_active()
            and layer_idx in self.deltakv_layer_to_idx
        )
        if not needs_pre_rope:
            return
        if k is None:
            raise RuntimeError(
                "DeltaKV pre-RoPE key storage requires pre-RoPE key states."
            )
        if v is None:
            raise RuntimeError("DeltaKV sparse reference storage requires value states.")
        if slot_mapping is None or int(slot_mapping.numel()) == 0:
            return
        if int(slot_mapping.numel()) != int(k.shape[0]):
            raise RuntimeError(
                "DeltaKV pre-RoPE staging shape mismatch: "
                f"slots={int(slot_mapping.numel())}, k_tokens={int(k.shape[0])}."
            )
        valid = (slot_mapping >= 0) & (slot_mapping < int(self.deltakv_prefill_staging_num_slots))
        capturing = self._is_stream_capturing()
        if capturing:
            self.deltakv_prefill_staging_pre_rope_k_cache[slot_mapping.to(torch.long)] = k.to(
                self.deltakv_prefill_staging_pre_rope_k_cache.dtype
            )
            return
        if not bool(valid.any()):
            return
        slots = slot_mapping[valid].to(torch.long)
        self.deltakv_prefill_staging_pre_rope_k_cache[slots] = k[valid].to(
            self.deltakv_prefill_staging_pre_rope_k_cache.dtype
        )

    def _stage_pre_rope_kv_by_pos(self, pos: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
        pos_i64 = pos.to(torch.long)
        if validate and not self._is_stream_capturing() and (
            (pos_i64 < 0).any() or (pos_i64 >= int(self.deltakv_prefill_staging_num_slots)).any()
        ):
            raise RuntimeError("DeltaKV pre-RoPE staging position is outside staging cache.")
        k = self.deltakv_prefill_staging_pre_rope_k_cache[pos_i64]
        v = self.deltakv_prefill_staging_kv_cache[1, pos_i64]
        kv_dim_half = self.num_kv_heads * self.head_dim
        return torch.cat(
            [
                k.reshape(-1, kv_dim_half),
                v.reshape(-1, kv_dim_half),
            ],
            dim=-1,
        )

    def _deltakv_gather_raw_kv_from_cache(
        self,
        *,
        slots: torch.Tensor,
        pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self._prefill_pre_rope_stage_active()
            and k_cache.data_ptr() == self.deltakv_prefill_staging_kv_cache[0].data_ptr()
        ):
            return self._stage_pre_rope_kv_by_pos(pos)
        for l_idx in range(int(self.deltakv_full_kv_cache.shape[1])):
            if k_cache.data_ptr() == self.deltakv_full_kv_cache[0, l_idx].data_ptr():
                slots_i64 = slots.to(torch.long)
                k_raw = k_cache[slots_i64]
                v_raw = v_cache[slots_i64]
                kv_dim_half = self.num_kv_heads * self.head_dim
                return torch.cat(
                    [
                        k_raw.reshape(-1, kv_dim_half),
                        v_raw.reshape(-1, kv_dim_half),
                    ],
                    dim=-1,
                )
        return super()._deltakv_gather_raw_kv_from_cache(
            slots=slots,
            pos=pos,
            k_cache=k_cache,
            v_cache=v_cache,
        )

    def _deltakv_score_chunk_size(self, num_centers: int) -> int:
        configured = int(getattr(self.config, "deltakv_cluster_gather_chunk_size", 16384) or 16384)
        if configured <= 0:
            raise RuntimeError("deltakv_cluster_gather_chunk_size must be > 0 for chunked cluster scoring.")
        max_score_elements = int(os.getenv("SPARSEENGINE_DELTAKV_SCORE_MAX_ELEMS", str(128 * 1024 * 1024)))
        if max_score_elements <= 0:
            raise RuntimeError(
                "SPARSEENGINE_DELTAKV_SCORE_MAX_ELEMS must be > 0 when set, "
                f"got {max_score_elements}."
            )
        return max(1, min(configured, max_score_elements // max(1, int(num_centers))))

    def _cluster_compress_against_centers(
        self,
        *,
        kv_states: torch.Tensor,
        all_centers: torch.Tensor,
        existing_center_count: int,
        new_center_rel: torch.Tensor,
        row_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kv_states.dim() == 3 and kv_states.shape[0] == 1
        _, n, kv_dim = kv_states.shape
        num_centers = int(all_centers.shape[1])
        if num_centers == 0:
            raise RuntimeError("DeltaKV less-memory: no available reference centers.")

        k_eff = min(int(self.config.deltakv_neighbor_count), num_centers)
        if k_eff <= 0:
            raise RuntimeError("DeltaKV less-memory: no available centers to assign.")
        score_chunk_size = self._deltakv_score_chunk_size(num_centers)
        metric_type = self.config.cluster_metric
        m0 = int(existing_center_count)
        m_new = int(new_center_rel.numel())

        topk_chunks: list[torch.Tensor] = []
        base_chunks: list[torch.Tensor] = []
        for start in range(0, n, score_chunk_size):
            end = min(n, start + score_chunk_size)
            kv_chunk = kv_states[:, start:end, :]
            if metric_type == "l2":
                scores = self._metric_l2(kv_chunk, all_centers)
            elif metric_type == "dot":
                scores = self._metric_dot(kv_chunk, all_centers)
            elif metric_type == "cosine":
                scores = self._metric_cosine(kv_chunk, all_centers)
            elif metric_type == "fastdot":
                scores = torch.bmm(kv_chunk, all_centers.transpose(1, 2))
            else:
                raise ValueError(f"Unknown cluster_metric: {metric_type}")

            if m_new > 0:
                rows = torch.arange(row_start + start, row_start + end, device=kv_states.device).view(end - start, 1)
                cols = new_center_rel.view(1, m_new)
                mask_new = cols <= rows
                scores[:, :, m0:].masked_fill_(~mask_new.unsqueeze(0), float("-inf"))

            topk_indices_chunk = scores.topk(k=k_eff, dim=-1, sorted=False).indices
            gather_idx = topk_indices_chunk.reshape(1, -1)[:, :, None].expand(-1, -1, kv_dim)
            base_chunk = all_centers.gather(1, gather_idx).view(1, end - start, k_eff, kv_dim).mean(dim=2)
            topk_chunks.append(topk_indices_chunk)
            base_chunks.append(base_chunk)

        if len(topk_chunks) == 1:
            topk_indices = topk_chunks[0]
            base = base_chunks[0]
        else:
            topk_indices = torch.cat(topk_chunks, dim=1)
            base = torch.cat(base_chunks, dim=1)
        return topk_indices.squeeze(0).to(torch.int32), base

    def _deltakv_store_father_slots(
        self,
        *,
        l_idx: int,
        latent_slots: torch.Tensor,
        all_center_slots: torch.Tensor,
        topk_center_indices: torch.Tensor,
        store_indices: torch.Tensor | None = None,
    ):
        father_slots_full = all_center_slots[topk_center_indices.to(torch.long)]
        if store_indices is not None:
            father_slots = father_slots_full.index_select(
                0,
                store_indices.to(device=father_slots_full.device, dtype=torch.long),
            )
        else:
            father_slots = father_slots_full
        k_neighbors = self.deltakv_latent_to_full_slots.shape[-1]
        k_eff = father_slots.shape[1]
        if k_eff < k_neighbors:
            pad = father_slots[:, :1].expand(-1, k_neighbors - k_eff)
            father_slots = torch.cat([father_slots, pad], dim=1)
        elif k_eff > k_neighbors:
            father_slots = father_slots[:, :k_neighbors]
        self.deltakv_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)

    def _deltakv_compress_full_prefill_plan_layer(
        self,
        *,
        l_idx: int,
        layer_idx: int,
        plan: dict[str, torch.Tensor | int],
        sink_slots: torch.Tensor,
        center_pos: torch.Tensor,
        center_slots: torch.Tensor,
        latent_slots: torch.Tensor,
        evict_pos: torch.Tensor,
        latent_store_mask: torch.Tensor,
        latent_store_indices: torch.Tensor | None,
    ):
        contiguous_store_indices = bool(plan.get("latent_store_indices_contiguous", False))
        if contiguous_store_indices:
            if int(latent_slots.numel()) != int(evict_pos.numel()):
                raise RuntimeError(
                    "DeltaKV full-prefill contiguous latent store mismatch: "
                    f"evict_tokens={int(evict_pos.numel())}, latent_slots={int(latent_slots.numel())}."
                )
            store_indices_all = None
            store_count = int(latent_slots.numel())
        else:
            if latent_store_indices is None:
                store_indices_all = torch.nonzero(latent_store_mask, as_tuple=False).flatten().to(device=evict_pos.device)
            else:
                store_indices_all = latent_store_indices.to(device=evict_pos.device, dtype=torch.long)
            store_count = int(store_indices_all.numel())
            if store_count != int(latent_slots.numel()):
                raise RuntimeError(
                    "DeltaKV full-prefill latent store index mismatch: "
                    f"store_indices={store_count}, latent_slots={int(latent_slots.numel())}."
                )
        if store_count == 0:
            return
        if not contiguous_store_indices and not self._is_stream_capturing():
            if bool((store_indices_all < 0).any()) or bool((store_indices_all >= int(evict_pos.numel())).any()):
                raise RuntimeError("DeltaKV full-prefill store indices are outside the evict block.")
            if store_count > 1 and bool((store_indices_all[1:] < store_indices_all[:-1]).any()):
                raise RuntimeError("DeltaKV full-prefill store indices must be sorted by evict position.")

        with profiler.record("deltakv_full_prefill_build_centers"):
            kv_dim = 2 * self.num_kv_heads * self.head_dim
            staging_start = int(plan.get("staging_start", 0) or 0)
            sink_pos = plan["keep_pos"][: int(sink_slots.numel())].to(torch.int32)
            existing_centers = (
                self._stage_pre_rope_kv_by_pos(sink_pos + staging_start, validate=False).unsqueeze(0)
                if sink_slots.numel() > 0
                else evict_pos.new_zeros(
                    (1, 0, kv_dim), dtype=self.hf_config.dtype
                )
            )
            new_centers = (
                self._stage_pre_rope_kv_by_pos(
                    center_pos.to(torch.int32) + staging_start,
                    validate=False,
                ).unsqueeze(0)
                if center_pos.numel() > 0
                else existing_centers.new_zeros((1, 0, kv_dim))
            )
            all_centers = torch.cat([existing_centers, new_centers], dim=1)
            all_center_slots = torch.cat([sink_slots, center_slots], dim=0)
            if int(all_centers.shape[1]) == 0:
                raise RuntimeError("DeltaKV less-memory full-prefill: no available reference centers.")

        evict_start = int(plan["evict_start"])
        new_center_rel = (center_pos - evict_start).to(device=evict_pos.device, dtype=torch.long)
        store_cursor = 0
        store_chunk_size = self._deltakv_latent_store_chunk_size()
        for start in range(0, int(evict_pos.numel()), store_chunk_size):
            end = min(int(evict_pos.numel()), start + store_chunk_size)
            evict_chunk = evict_pos[start:end]
            with profiler.record("deltakv_full_prefill_gather_raw_chunk"):
                kv_chunk = self._stage_pre_rope_kv_by_pos(
                    evict_chunk + staging_start,
                    validate=False,
                ).unsqueeze(0)
            with profiler.record("deltakv_full_prefill_cluster_chunk"):
                topk_center_indices, base_kv = self._cluster_compress_against_centers(
                    kv_states=kv_chunk,
                    all_centers=all_centers,
                    existing_center_count=int(existing_centers.shape[1]),
                    new_center_rel=new_center_rel,
                    row_start=start,
                )

            if contiguous_store_indices:
                next_cursor = end
                selected_local = None
            else:
                next_cursor = int(
                    torch.searchsorted(
                        store_indices_all,
                        torch.tensor(end, device=store_indices_all.device, dtype=store_indices_all.dtype),
                        right=False,
                    ).item()
                )
                selected_global = store_indices_all[store_cursor:next_cursor]
                if not self._is_stream_capturing() and (
                    bool((selected_global < start).any()) or bool((selected_global >= end).any())
                ):
                    raise RuntimeError("DeltaKV full-prefill store index fell outside its chunk.")
                selected_local = selected_global - start
            if next_cursor == store_cursor:
                continue
            chunk_latent_slots = latent_slots[store_cursor:next_cursor]

            with profiler.record("deltakv_full_prefill_store_fathers_chunk"):
                self._deltakv_store_father_slots(
                    l_idx=l_idx,
                    latent_slots=chunk_latent_slots,
                    all_center_slots=all_center_slots,
                    topk_center_indices=topk_center_indices,
                    store_indices=selected_local,
                )
            with profiler.record("deltakv_full_prefill_store_latent_chunk"):
                self._deltakv_store_layer_latent(
                    l_idx=l_idx,
                    latent_slots=chunk_latent_slots,
                    kv_block=kv_chunk,
                    base_kv=base_kv,
                    to_compress_mask=None if contiguous_store_indices else latent_store_mask[start:end],
                    store_indices=selected_local,
                    store_all=bool(contiguous_store_indices),
                )
            store_cursor = next_cursor

        if store_cursor != store_count:
            raise RuntimeError(
                "DeltaKV full-prefill did not store all latent rows: "
                f"stored={store_cursor}, expected={store_count}."
            )

    @torch.no_grad()
    def _deltakv_compress_full_prefill_layer(self, layer_idx: int):
        with profiler.record("deltakv_full_prefill_compress_total"):
            if layer_idx in self._deltakv_full_prefill_compressed_layers:
                raise RuntimeError(f"DeltaKV full-prefill layer compressed twice: layer={layer_idx}.")
            if layer_idx not in self.deltakv_layer_to_idx:
                return
            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_persist = self.deltakv_full_kv_cache[0, l_idx]
            v_persist = self.deltakv_full_kv_cache[1, l_idx]

            for plan in self._deltakv_full_prefill_plans.values():
                staging_start = int(plan.get("staging_start", 0) or 0)
                keep_pos = plan["keep_pos"]
                keep_slots = plan["keep_slots"]
                if keep_slots.numel() > 0:
                    keep_pos_i64 = keep_pos.to(torch.long) + staging_start
                    keep_slots_i64 = keep_slots.to(torch.long)
                    with profiler.record("deltakv_full_prefill_copy_keep"):
                        k_persist[keep_slots_i64] = self.deltakv_prefill_staging_pre_rope_k_cache[keep_pos_i64].to(
                            k_persist.dtype
                        )
                        v_persist[keep_slots_i64] = self.deltakv_prefill_staging_kv_cache[1, keep_pos_i64].to(
                            v_persist.dtype
                        )

                sink_slots = plan["sink_slots"].to(torch.int32)
                center_pos = plan["center_pos"].to(torch.int32)
                center_slots = plan["center_slots"].to(torch.int32)
                self.row_deltakv_center_slots[int(plan["row_idx"])][layer_idx] = torch.cat(
                    [sink_slots, center_slots],
                    dim=0,
                )

                latent_slots = plan["latent_slots"].to(torch.int32)
                evict_pos = plan["evict_pos"].to(torch.int32)
                if latent_slots.numel() == 0:
                    continue

                to_compress_mask = plan["to_compress_mask"]
                latent_store_mask = plan.get("latent_store_mask", to_compress_mask)
                latent_store_indices = plan.get("latent_store_indices", None)
                self._deltakv_compress_full_prefill_plan_layer(
                    l_idx=l_idx,
                    layer_idx=layer_idx,
                    plan=plan,
                    sink_slots=sink_slots,
                    center_pos=center_pos,
                    center_slots=center_slots,
                    latent_slots=latent_slots,
                    evict_pos=evict_pos,
                    latent_store_mask=latent_store_mask,
                    latent_store_indices=latent_store_indices,
                )

            self._deltakv_full_prefill_compressed_layers.add(layer_idx)

    def _gather_sparse_ref_raw_kv_by_slots(
        self,
        l_idx: int,
        slots: torch.Tensor,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        if slots.numel() == 0:
            return torch.empty(
                (0, 2 * self.num_kv_heads * self.head_dim),
                dtype=self.hf_config.dtype,
                device=self.device,
            )
        slots_i32 = slots.to(torch.int32)
        capturing = self._is_stream_capturing()
        if self._prefill_pre_rope_stage_active():
            pos = self.deltakv_slot_to_pos[slots_i32.to(torch.long)].to(torch.long)
            if validate and not capturing and (pos < 0).any():
                raise RuntimeError("DeltaKV less-memory: reference slot has unknown position.")
            if capturing or (pos < int(self.deltakv_prefill_staging_num_slots)).all():
                return self._stage_pre_rope_kv_by_pos(pos, validate=validate)
        elif validate and not capturing:
            pos = self.deltakv_slot_to_pos[slots_i32.to(torch.long)]
            if (pos < 0).any():
                raise RuntimeError("DeltaKV less-memory: reference slot has unknown position.")
        k_raw = self.deltakv_full_kv_cache[0, l_idx, slots_i32.to(torch.long)]
        v = self.deltakv_full_kv_cache[1, l_idx, slots_i32.to(torch.long)]
        kv_dim_half = self.num_kv_heads * self.head_dim
        return torch.cat(
            [
                k_raw.reshape(-1, kv_dim_half),
                v.reshape(-1, kv_dim_half),
            ],
            dim=-1,
        )

    def _cluster_compress(
        self,
        layer_idx: int,
        kv_states: torch.Tensor,
        existing_center_slots: torch.Tensor,
        cluster_step: int,
        new_center_rel: torch.Tensor | None = None,
        validate_centers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kv_states.dim() == 3 and kv_states.shape[0] == 1
        _, n, kv_dim = kv_states.shape
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_neighbors = int(self.config.deltakv_neighbor_count)
        existing_centers = (
            self._gather_sparse_ref_raw_kv_by_slots(
                l_idx,
                existing_center_slots,
                validate=validate_centers,
            ).unsqueeze(0)
            if existing_center_slots.numel() > 0
            else kv_states.new_zeros((1, 0, kv_dim))
        )
        if new_center_rel is None:
            new_center_rel = torch.arange(0, n, max(1, int(cluster_step)), device=kv_states.device, dtype=torch.long)
        else:
            new_center_rel = new_center_rel.to(device=kv_states.device, dtype=torch.long)
        capturing = self._is_stream_capturing()
        if (
            validate_centers
            and
            not capturing
            and new_center_rel.numel() > 0
            and ((new_center_rel < 0).any() or (new_center_rel >= n).any())
        ):
            raise RuntimeError("DeltaKV less-memory: center positions are outside the current block.")
        new_centers = kv_states.index_select(1, new_center_rel) if new_center_rel.numel() else kv_states[:, :0, :]

        all_centers = torch.cat([existing_centers, new_centers], dim=1)
        m0 = int(existing_centers.shape[1])
        m_new = int(new_centers.shape[1])
        if all_centers.shape[1] == 0:
            raise RuntimeError("DeltaKV less-memory: no available reference centers.")

        k_eff = min(k_neighbors, all_centers.shape[1])
        if k_eff <= 0:
            raise RuntimeError("DeltaKV less-memory: no available centers to assign.")
        metric_type = self.config.cluster_metric
        score_chunk_size = self._deltakv_score_chunk_size(int(all_centers.shape[1]))

        topk_chunks: list[torch.Tensor] = []
        base_chunks: list[torch.Tensor] = []
        for start in range(0, n, score_chunk_size):
            end = min(n, start + score_chunk_size)
            kv_chunk = kv_states[:, start:end, :]
            if metric_type == "l2":
                scores = self._metric_l2(kv_chunk, all_centers)
            elif metric_type == "dot":
                scores = self._metric_dot(kv_chunk, all_centers)
            elif metric_type == "cosine":
                scores = self._metric_cosine(kv_chunk, all_centers)
            elif metric_type == "fastdot":
                scores = torch.bmm(kv_chunk, all_centers.transpose(1, 2))
            else:
                raise ValueError(f"Unknown cluster_metric: {metric_type}")

            if m_new > 0:
                rows = torch.arange(start, end, device=kv_states.device).view(end - start, 1)
                cols = new_center_rel.view(1, m_new)
                mask_new = cols <= rows
                scores[:, :, m0:].masked_fill_(~mask_new.unsqueeze(0), float("-inf"))

            topk_indices_chunk = scores.topk(k=k_eff, dim=-1, sorted=False).indices
            gather_idx = topk_indices_chunk.reshape(1, -1)[:, :, None].expand(-1, -1, kv_dim)
            base_chunk = all_centers.gather(1, gather_idx).view(1, end - start, k_eff, kv_dim).mean(dim=2)
            topk_chunks.append(topk_indices_chunk)
            base_chunks.append(base_chunk)

        if len(topk_chunks) == 1:
            topk_indices = topk_chunks[0]
            base = base_chunks[0]
        else:
            topk_indices = torch.cat(topk_chunks, dim=1)
            base = torch.cat(base_chunks, dim=1)
        return topk_indices.squeeze(0).to(torch.int32), base

    def _load_residual(self, l_idx: int, recon_latent: torch.Tensor, kv_dim: int) -> torch.Tensor:
        residual = self.deltakv_latent_cache[l_idx, recon_latent]
        if int(self.config.deltakv_latent_quant_bits or 0) == 4:
            scales = self.deltakv_latent_scales[l_idx, recon_latent]
            mins = self.deltakv_latent_mins[l_idx, recon_latent]
            payload_dim = self._sparse_payload_dim(kv_dim)
            residual = self._dequantize_residual(residual, scales, mins, payload_dim)
        return self.compress_up[l_idx](residual)

    def _deltakv_build_view_and_plan_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        req_indices: torch.Tensor,
    ):
        if not get_context().is_prefill or self._is_stream_capturing():
            req_ptr = int(req_indices.data_ptr()) if req_indices is not None and req_indices.numel() > 0 else 0
            req_n = int(req_indices.numel()) if req_indices is not None else 0
            if active_compressed_indices is None:
                act_ptr = 0
                act_b = req_n
                act_k = 0
            else:
                act_ptr = int(active_compressed_indices.data_ptr())
                act_b = int(active_compressed_indices.shape[0])
                act_k = int(active_compressed_indices.shape[1])

            key = (req_ptr, req_n, act_ptr, act_b, act_k)
            if self._deltakv_view_cache_key == key and self._deltakv_view_cache_value is not None:
                with profiler.record("deltakv_build_view_cache_hit"):
                    return self._deltakv_view_cache_value

            with profiler.record("deltakv_build_view_total"):
                out = self._deltakv_build_view_and_plan_reconstruct_static(
                    layer_idx,
                    active_compressed_indices,
                    req_indices,
                )
            self._deltakv_view_cache_key = key
            self._deltakv_view_cache_value = out
            return out
        return super()._deltakv_build_view_and_plan_reconstruct(layer_idx, active_compressed_indices, req_indices)

    def has_full_layer_quantized_view(self, layer_idx: int) -> bool:
        return (
            layer_idx in self.full_layer_to_idx
            and self._full_layer_quant_enabled()
        )

    def build_decode_compute_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        selection: SparseSelection,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> DecodeComputeView:
        if self._full_layer_kivi_enabled() and layer_idx in self.full_layer_to_idx:
            l_idx = self.full_layer_to_idx[layer_idx]
            if self.full_layer_kivi_key_packed is None or self.full_layer_kivi_value_packed is None:
                raise RuntimeError("Full-layer KIVI decode was requested before KIVI storage was initialized.")
            return DecodeComputeView(
                meta=AttentionViewMeta(
                    active_slots=self.full_layer_slots_map,
                    req_indices=selection.req_indices,
                    context_lens=selection.context_lens,
                    attn_score=selection.attn_score,
                    max_context_len=selection.max_context_len,
                ),
                payload=ExplicitKVPayload(
                    k_cache=self.full_kv_cache[0, l_idx],
                    v_cache=self.full_kv_cache[1, l_idx],
                    backend="full_layer_kivi",
                    metadata={
                        "kivi_block_slots_map": self.full_layer_kivi_block_slots_map,
                        "kivi_block_start_pos": self.full_layer_kivi_block_start_pos,
                        "key_packed": self.full_layer_kivi_key_packed[l_idx],
                        "key_scales": self.full_layer_kivi_key_scales[l_idx],
                        "key_mins": self.full_layer_kivi_key_mins[l_idx],
                        "value_packed": self.full_layer_kivi_value_packed[l_idx],
                        "value_scales": self.full_layer_kivi_value_scales[l_idx],
                        "value_mins": self.full_layer_kivi_value_mins[l_idx],
                        "group_size": self._full_layer_kivi_group_size(),
                        "block_n": int(getattr(self.config, "full_layer_kivi_decode_block_n", 16) or 16),
                        "num_warps": int(getattr(self.config, "full_layer_kivi_decode_num_warps", 2) or 2),
                        "num_stages": int(getattr(self.config, "full_layer_kivi_decode_num_stages", 3) or 3),
                    },
                ),
            )
        view = super().build_decode_compute_view(
            layer_idx,
            q,
            selection,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        if (
            layer_idx in self.deltakv_layer_to_idx
            and not self.has_prefill_staging_view(layer_idx)
            and getattr(self.config, "deltakv_sparse_decode_backend", "custom") == "fa2"
        ):
            if not isinstance(view.payload, ExplicitKVPayload):
                raise TypeError(
                    "DeltaKV FA2 decode requires ExplicitKVPayload, got "
                    f"{type(view.payload).__name__}."
                )
            view = replace(
                view,
                payload=replace(
                    view.payload,
                    backend="flash_attn_contiguous",
                ),
            )
        return view

    def get_layer_compute_tensors(self, layer_idx: int, selection: SparseSelection | None = None):
        del selection
        if self.has_prefill_staging_view(layer_idx):
            return self.deltakv_prefill_staging_kv_cache[0], self.deltakv_prefill_staging_kv_cache[1]
        if self.has_full_layer_quantized_view(layer_idx):
            if self._full_layer_quant_k_cache is None or self._full_layer_quant_v_cache is None:
                raise RuntimeError("Full-layer quantized view was requested before reconstruction.")
            return self._full_layer_quant_k_cache, self._full_layer_quant_v_cache
        raise NotImplementedError

    def _ensure_full_layer_quant_temp_cache(self, num_slots: int):
        num_slots = max(1, int(num_slots))
        needs_alloc = (
            self._full_layer_quant_k_cache is None
            or int(self._full_layer_quant_k_cache.shape[0]) < num_slots
        )
        if needs_alloc:
            self._full_layer_quant_k_cache = torch.empty(
                num_slots,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.hf_config.dtype,
                device=self.device,
            )
            self._full_layer_quant_v_cache = torch.empty_like(self._full_layer_quant_k_cache)

    def _ensure_full_layer_quant_decode_workspace(self, batch_size: int, max_len: int):
        batch_size = max(1, int(batch_size))
        max_len = max(1, int(max_len))
        self._ensure_full_layer_quant_temp_cache(batch_size * max_len)
        needs_active = (
            self._full_layer_quant_active_slots is None
            or int(self._full_layer_quant_active_slots.shape[0]) < batch_size
            or int(self._full_layer_quant_active_slots.shape[1]) < max_len
        )
        if needs_active:
            self._full_layer_quant_active_slots = torch.empty(
                batch_size,
                max_len,
                dtype=torch.int32,
                device=self.device,
            )
        needs_local_req = (
            self._full_layer_quant_local_req is None
            or int(self._full_layer_quant_local_req.numel()) < batch_size
        )
        if needs_local_req:
            self._full_layer_quant_local_req = torch.arange(batch_size, dtype=torch.int32, device=self.device)
        positions = getattr(self, "_full_layer_quant_positions", None)
        if positions is None or int(positions.numel()) < max_len:
            self._full_layer_quant_positions = torch.arange(max_len, dtype=torch.int32, device=self.device)
        out_slots = getattr(self, "_full_layer_quant_out_slots", None)
        total_slots = batch_size * max_len
        if out_slots is None or int(out_slots.numel()) < total_slots:
            self._full_layer_quant_out_slots = torch.arange(total_slots, dtype=torch.int32, device=self.device)

    def _gather_full_layer_raw_kv_by_slots(
        self,
        layer_idx: int,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        if slots.numel() == 0:
            return torch.empty(
                (0, 2 * self.num_kv_heads * self.head_dim),
                dtype=self.hf_config.dtype,
                device=self.device,
            )
        l_idx = self.full_layer_to_idx[layer_idx]
        slots_i32 = slots.to(torch.int32)
        pos = self.full_layer_slot_to_pos[slots_i32].to(torch.long)
        if (pos < 0).any():
            raise RuntimeError("Full-layer residual quantization: center slot has unknown position.")
        k_rope = self.full_kv_cache[0, l_idx, slots_i32.to(torch.long)]
        v = self.full_kv_cache[1, l_idx, slots_i32.to(torch.long)]
        cos_sin = self.cos_sin_cache[pos]
        cos, sin = cos_sin.chunk(2, dim=-1)
        raw_k = self._reverse_rotary_for_full_layer(k_rope, cos, sin)
        kv_dim_half = self.num_kv_heads * self.head_dim
        return torch.cat(
            [
                raw_k.reshape(-1, kv_dim_half),
                v.reshape(-1, kv_dim_half),
            ],
            dim=-1,
        )

    def _reverse_rotary_for_full_layer(self, k_rope: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        from sparseengine.layers.rotary_embedding import reverse_rotary_emb

        return reverse_rotary_emb(k_rope, cos, sin)

    def _apply_rotary_for_full_layer(self, raw_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        from sparseengine.layers.rotary_embedding import apply_rotary_emb

        return apply_rotary_emb(raw_k, cos, sin)

    def _full_layer_cluster_compress(
        self,
        layer_idx: int,
        kv_states: torch.Tensor,
        existing_center_slots: torch.Tensor,
        new_center_rel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kv_states.dim() == 3 and kv_states.shape[0] == 1
        _, n, kv_dim = kv_states.shape
        k_neighbors = int(self.config.deltakv_neighbor_count)
        existing_centers = (
            self._gather_full_layer_raw_kv_by_slots(layer_idx, existing_center_slots).unsqueeze(0)
            if existing_center_slots.numel() > 0
            else kv_states.new_zeros((1, 0, kv_dim))
        )
        new_center_rel = new_center_rel.to(device=kv_states.device, dtype=torch.long)
        if new_center_rel.numel() > 0 and ((new_center_rel < 0).any() or (new_center_rel >= n).any()):
            raise RuntimeError("Full-layer center positions are outside the current block.")
        new_centers = kv_states.index_select(1, new_center_rel) if new_center_rel.numel() else kv_states[:, :0, :]
        all_centers = torch.cat([existing_centers, new_centers], dim=1)
        m0 = existing_centers.shape[1]
        m_new = new_centers.shape[1]

        metric_type = self.config.cluster_metric
        if metric_type == "l2":
            scores = self._metric_l2(kv_states, all_centers)
        elif metric_type == "dot":
            scores = self._metric_dot(kv_states, all_centers)
        elif metric_type == "cosine":
            scores = self._metric_cosine(kv_states, all_centers)
        elif metric_type == "fastdot":
            scores = torch.bmm(kv_states, all_centers.transpose(1, 2))
        else:
            raise ValueError(f"Unknown cluster_metric: {metric_type}")

        if m_new > 0:
            rows = torch.arange(n, device=kv_states.device).view(n, 1)
            cols = new_center_rel.view(1, m_new)
            mask_new = cols <= rows
            mask_existing = torch.ones((n, m0), device=kv_states.device, dtype=torch.bool)
            scores = scores.masked_fill(~torch.cat([mask_existing, mask_new], dim=1).unsqueeze(0), float("-inf"))

        k_eff = min(k_neighbors, all_centers.shape[1])
        if k_eff <= 0:
            raise RuntimeError("Full-layer residual quantization: no available centers to assign.")
        topk_indices = scores.topk(k=k_eff, dim=-1, sorted=False).indices
        gather_idx = topk_indices.view(1, -1)[:, :, None].expand(-1, -1, kv_dim)
        base = all_centers.gather(1, gather_idx).view(1, n, k_eff, kv_dim).mean(dim=2)
        return topk_indices.squeeze(0).to(torch.int32), base

    def _store_full_layer_residual(
        self,
        l_idx: int,
        latent_slots: torch.Tensor,
        residual: torch.Tensor,
    ):
        quant_bits = int(self.config.full_layer_kv_quant_bits or 0)
        packed, scale, mn = self._quantize_residual_bits(residual, quant_bits)
        self.full_layer_latent_cache[l_idx, latent_slots] = packed.to(self.full_layer_latent_cache.dtype)
        self.full_layer_latent_scales[l_idx, latent_slots] = scale.to(self.full_layer_latent_scales.dtype)
        self.full_layer_latent_mins[l_idx, latent_slots] = mn.to(self.full_layer_latent_mins.dtype)

    def _load_full_layer_residual(self, l_idx: int, latent_slots: torch.Tensor, kv_dim: int) -> torch.Tensor:
        quant_bits = int(self.config.full_layer_kv_quant_bits or 0)
        return self._dequantize_residual_bits(
            self.full_layer_latent_cache[l_idx, latent_slots],
            self.full_layer_latent_scales[l_idx, latent_slots],
            self.full_layer_latent_mins[l_idx, latent_slots],
            kv_dim,
            quant_bits,
        )

    def _reconstruct_full_layer_tokens(
        self,
        *,
        l_idx: int,
        latent_slots: torch.Tensor,
        out_slots: torch.Tensor,
        out_pos: torch.Tensor,
    ):
        if latent_slots.numel() == 0:
            return
        k_out, v_out = self._reconstruct_full_layer_token_values(
            l_idx=l_idx,
            latent_slots=latent_slots,
            out_pos=out_pos,
        )
        out_i64 = out_slots.to(torch.long)
        self._full_layer_quant_k_cache[out_i64] = k_out.to(self._full_layer_quant_k_cache.dtype)
        self._full_layer_quant_v_cache[out_i64] = v_out.to(self._full_layer_quant_v_cache.dtype)

    def _reconstruct_full_layer_token_values(
        self,
        *,
        l_idx: int,
        latent_slots: torch.Tensor,
        out_pos: torch.Tensor,
        validate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_dim_half = self.num_kv_heads * self.head_dim
        kv_dim = 2 * kv_dim_half
        residual = self._load_full_layer_residual(l_idx, latent_slots, kv_dim)
        father_slots = self.full_layer_latent_to_full_slots[l_idx, latent_slots].to(torch.int32)
        if not self._is_stream_capturing():
            check_father_slots = father_slots if validate_mask is None else father_slots[validate_mask]
            if (check_father_slots < 0).any():
                raise RuntimeError("Full-layer residual quantization: missing father slots.")
        father_pos = self.full_layer_slot_to_pos[father_slots].to(torch.long)
        if not self._is_stream_capturing():
            check_father_pos = father_pos if validate_mask is None else father_pos[validate_mask]
            if (check_father_pos < 0).any():
                raise RuntimeError("Full-layer residual quantization: father slot has unknown position.")
        k_father_rope = self.full_kv_cache[0, l_idx, father_slots.to(torch.long)]
        v_father = self.full_kv_cache[1, l_idx, father_slots.to(torch.long)]
        cos_sin_f = self.cos_sin_cache[father_pos]
        cos_f, sin_f = cos_sin_f.chunk(2, dim=-1)
        raw_k_father = self._reverse_rotary_for_full_layer(k_father_rope, cos_f, sin_f)
        base = torch.cat(
            [
                raw_k_father.reshape(raw_k_father.shape[0], raw_k_father.shape[1], kv_dim_half),
                v_father.reshape(v_father.shape[0], v_father.shape[1], kv_dim_half),
            ],
            dim=-1,
        ).mean(dim=1)
        raw_kv = residual + base
        raw_k = raw_kv[:, :kv_dim_half].reshape(-1, self.num_kv_heads, self.head_dim)
        v_out = raw_kv[:, kv_dim_half:].reshape(-1, self.num_kv_heads, self.head_dim)
        cos_sin_t = self.cos_sin_cache[out_pos.to(torch.long)]
        cos_t, sin_t = cos_sin_t.chunk(2, dim=-1)
        k_out = self._apply_rotary_for_full_layer(raw_k, cos_t, sin_t)
        return k_out, v_out

    @torch.no_grad()
    def build_full_layer_quantized_view(
        self,
        layer_idx: int,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_full_layer_quantized_view(layer_idx):
            raise NotImplementedError
        if not get_context().is_prefill or self._is_stream_capturing():
            return self._build_full_layer_quantized_view_static(layer_idx, req_indices, context_lens)
        bsz = int(req_indices.numel())
        if bsz == 0:
            empty = torch.empty((0,), dtype=torch.int32, device=self.device)
            return torch.empty((0, 0), dtype=torch.int32, device=self.device), empty, empty
        max_len = int(context_lens.max().item())
        total_slots = bsz * max_len
        self._ensure_full_layer_quant_temp_cache(total_slots)
        active_slots = torch.full((bsz, max_len), -1, dtype=torch.int32, device=self.device)
        local_req = torch.arange(bsz, dtype=torch.int32, device=self.device)
        l_idx = self.full_layer_to_idx[layer_idx]

        req_cpu = req_indices.cpu().numpy()
        for b in range(bsz):
            row = int(req_cpu[b])
            total_len = int(context_lens[b].item())
            if total_len <= 0:
                continue
            out_start = b * max_len
            out_slots = torch.arange(out_start, out_start + total_len, dtype=torch.int32, device=self.device)
            active_slots[b, :total_len] = out_slots

            raw_slots = self.full_layer_slots_map[row, :total_len].to(torch.int32)
            raw_mask = raw_slots >= 0
            if raw_mask.any():
                src = raw_slots[raw_mask].to(torch.long)
                dst = out_slots[raw_mask].to(torch.long)
                self._full_layer_quant_k_cache[dst] = self.full_kv_cache[0, l_idx, src]
                self._full_layer_quant_v_cache[dst] = self.full_kv_cache[1, l_idx, src]

            need = ~raw_mask
            if need.any():
                pos = torch.arange(total_len, dtype=torch.int32, device=self.device)[need]
                latent_slots = self.full_layer_latent_slots_map[row, pos.to(torch.long)].to(torch.int32)
                if (latent_slots < 0).any():
                    raise RuntimeError("Full-layer quantized view found a token with neither raw nor residual slot.")
                self._reconstruct_full_layer_tokens(
                    l_idx=l_idx,
                    latent_slots=latent_slots,
                    out_slots=out_slots[need],
                    out_pos=pos,
                )

        return active_slots, local_req, context_lens

    def _build_full_layer_quantized_view_static(
        self,
        layer_idx: int,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = int(req_indices.numel())
        if bsz == 0:
            empty = torch.empty((0,), dtype=torch.int32, device=self.device)
            return torch.empty((0, 0), dtype=torch.int32, device=self.device), empty, empty
        max_len = getattr(self, "_decode_static_max_context_len", None)
        if max_len is None:
            max_len = self.full_layer_batch_states.max_context_len
        if max_len is None:
            raise RuntimeError("Full-layer quantized static decode requires a pinned max_context_len.")
        max_len = int(max_len)
        if max_len <= 0:
            raise RuntimeError(f"Full-layer quantized static decode got invalid max_context_len={max_len}.")
        if not self._is_stream_capturing():
            actual_max_len = int(context_lens.max().item())
            if actual_max_len > max_len:
                raise RuntimeError(
                    "Full-layer quantized static decode context exceeds graph max_context_len: "
                    f"context_lens_max={actual_max_len}, max_context_len={max_len}."
                )

        self._ensure_full_layer_quant_decode_workspace(bsz, max_len)
        active_slots = self._full_layer_quant_active_slots[:bsz, :max_len]
        local_req = self._full_layer_quant_local_req[:bsz]
        out_slots_2d = self._full_layer_quant_out_slots[: bsz * max_len].view(bsz, max_len)
        positions = self._full_layer_quant_positions[:max_len]
        valid = positions.unsqueeze(0) < context_lens[:bsz].to(torch.int32).unsqueeze(1)
        active_slots.copy_(out_slots_2d)
        active_slots.masked_fill_(~valid, -1)

        l_idx = self.full_layer_to_idx[layer_idx]
        rows = req_indices[:bsz].to(torch.long)
        pos_i64 = positions.to(torch.long)
        raw_slots = self.full_layer_slots_map[rows.unsqueeze(1), pos_i64.unsqueeze(0)].to(torch.int32)
        raw_mask = (raw_slots >= 0) & valid
        flat_raw_mask = raw_mask.reshape(-1)
        flat_raw_slots = raw_slots.reshape(-1)
        total_slots = bsz * max_len

        from sparseengine.kernels.triton.deltakv_kernels import full_layer_copy_raw_or_zero

        full_layer_copy_raw_or_zero(
            raw_k=self.full_kv_cache[0, l_idx],
            raw_v=self.full_kv_cache[1, l_idx],
            raw_slots=flat_raw_slots,
            raw_mask=flat_raw_mask,
            out_k=self._full_layer_quant_k_cache[:total_slots],
            out_v=self._full_layer_quant_v_cache[:total_slots],
        )

        flat_positions = positions.unsqueeze(0).expand(bsz, max_len).reshape(-1)
        need = valid & ~raw_mask
        flat_need = need.reshape(-1)
        latent_slots = self.full_layer_latent_slots_map[rows.unsqueeze(1), pos_i64.unsqueeze(0)].to(torch.int32)
        if not self._is_stream_capturing() and (latent_slots[need] < 0).any():
            raise RuntimeError("Full-layer quantized static view found a token with neither raw nor residual slot.")
        flat_latent_slots = latent_slots.reshape(-1)
        safe_latent_slots = torch.where(flat_need, flat_latent_slots, torch.zeros_like(flat_latent_slots)).to(torch.long)
        recon_k, recon_v = self._reconstruct_full_layer_token_values(
            l_idx=l_idx,
            latent_slots=safe_latent_slots,
            out_pos=flat_positions,
            validate_mask=flat_need,
        )
        choose_recon = flat_need.view(-1, 1, 1)
        self._full_layer_quant_k_cache[:total_slots].copy_(
            torch.where(
                choose_recon,
                recon_k.to(self._full_layer_quant_k_cache.dtype),
                self._full_layer_quant_k_cache[:total_slots],
            )
        )
        self._full_layer_quant_v_cache[:total_slots].copy_(
            torch.where(
                choose_recon,
                recon_v.to(self._full_layer_quant_v_cache.dtype),
                self._full_layer_quant_v_cache[:total_slots],
            )
        )
        return active_slots, local_req, context_lens

    @torch.no_grad()
    def _full_layer_kivi_evict(self, seqs: list[Sequence]):
        if not self._full_layer_kivi_enabled() or not self.full_layer_ids:
            return
        group_size = int(getattr(self.config, "full_layer_kivi_group_size", 32) or 32)
        residual_length = int(getattr(self.config, "full_layer_kivi_residual_length", group_size) or group_size)
        sink = int(self.config.sink_keep_tokens)
        if self.row_full_layer_kivi_quantized_lens is None:
            raise RuntimeError("Full-layer KIVI state was not initialized.")

        for seq in seqs:
            row_idx = self.seq_id_to_row.get(seq.seq_id, None)
            if row_idx is None:
                continue
            total_len = int(self.row_seq_lens[row_idx])
            buffer_len = max(0, total_len - sink)
            quant_rel_end = buffer_len - residual_length
            if quant_rel_end <= 0:
                continue
            quant_rel_end = (quant_rel_end // group_size) * group_size
            if quant_rel_end <= 0:
                continue
            quant_end = sink + quant_rel_end
            quant_start = max(sink, int(self.row_full_layer_kivi_quantized_lens[row_idx] or sink))
            quant_start = sink + (((quant_start - sink) + group_size - 1) // group_size) * group_size
            if quant_end <= quant_start:
                continue

            with profiler.record("deltakv_less_memory_full_layer_kivi_read_slots"):
                slots = self.full_layer_slots_map[row_idx, quant_start:quant_end].to(torch.int32)
            if (slots < 0).any():
                raise RuntimeError("Full-layer KIVI expects raw full-layer slots for the quantized block.")
            num_blocks = (quant_end - quant_start) // group_size
            with profiler.record("deltakv_less_memory_full_layer_kivi_alloc_blocks"):
                block_slots = self._allocate_full_layer_kivi_blocks(num_blocks).to(torch.int32)
            with profiler.record("deltakv_less_memory_full_layer_kivi_store_blocks"):
                block_starts = torch.arange(
                    quant_start,
                    quant_end,
                    group_size,
                    device=self.device,
                    dtype=torch.long,
                )
                block_raw_slots = slots.reshape(num_blocks, group_size)
                raw_i64 = block_raw_slots.to(torch.long)
                for layer_idx in self.full_layer_ids:
                    l_idx = self.full_layer_to_idx[layer_idx]
                    self._store_full_layer_kivi_blocks(
                        l_idx=l_idx,
                        block_slots=block_slots,
                        key_post_rope=self.full_kv_cache[0, l_idx, raw_i64],
                        value=self.full_kv_cache[1, l_idx, raw_i64],
                    )
                self.full_layer_kivi_block_slots_map[row_idx, quant_start:quant_end] = block_slots.repeat_interleave(group_size)
                self.full_layer_kivi_block_start_pos[block_slots.to(torch.long)] = block_starts.to(torch.int32)

            with profiler.record("deltakv_less_memory_full_layer_kivi_free_slots"):
                free_slots = slots
                ptr = self._num_free_slots_full
                self.free_slots_stack_full[ptr: ptr + free_slots.numel()] = free_slots
                self._num_free_slots_full += free_slots.numel()
                self.full_layer_slot_to_pos[free_slots.to(torch.long)] = -1
                self.full_layer_slots_map[row_idx, quant_start:quant_end] = -1
                self.row_full_layer_kivi_quantized_lens[row_idx] = int(quant_end)
                self.row_full_layer_kivi_quantized_lens_gpu[row_idx] = int(quant_end)

    def _validate_live_deltakv_center_slots(
        self,
        slots: torch.Tensor,
        *,
        row_idx: int,
        total_len: int,
        compressed_len: int,
        evict_start: int,
        evict_end: int,
        label: str,
        layer_idx: int | None = None,
    ):
        if slots.numel() == 0:
            return
        slot_to_pos_len = int(self.deltakv_slot_to_pos.numel())
        out_of_range = (slots < 0) | (slots >= slot_to_pos_len)
        layer_text = "" if layer_idx is None else f" layer={layer_idx}"
        # Clamp only the validation lookup, never the slots used by eviction.
        # Collect both error flags in one transfer before any unsafe gather.
        if slot_to_pos_len:
            safe_slots = slots.to(torch.long).clamp(0, slot_to_pos_len - 1)
            missing_pos = (self.deltakv_slot_to_pos[safe_slots] < 0) & ~out_of_range
        else:
            missing_pos = torch.zeros_like(out_of_range)
        range_error, position_error = torch.stack(
            (out_of_range.any(), missing_pos.any())
        ).cpu().tolist()
        if range_error:
            bad = slots[out_of_range][:16].detach().cpu().tolist()
            raise RuntimeError(
                f"DeltaKV less-memory eviction {label} center slots are out of range before slot_to_pos lookup: "
                f"row={row_idx}{layer_text} total_len={total_len} compressed_len={compressed_len} "
                f"evict=({evict_start},{evict_end}) slot_to_pos_len={slot_to_pos_len} bad_slots={bad}."
            )
        if position_error:
            bad_slots = slots[missing_pos][:16]
            bad = bad_slots.detach().cpu().tolist()
            debug = self._describe_deltakv_full_slots_for_debug(
                bad_slots,
                row_idx=row_idx,
                total_len=total_len,
            )
            raise RuntimeError(
                f"DeltaKV less-memory eviction found {label} center slots without live positions: "
                f"row={row_idx}{layer_text} total_len={total_len} compressed_len={compressed_len} "
                f"evict=({evict_start},{evict_end}) num_slots={int(slots.numel())} "
                f"num_invalid={int(missing_pos.sum().item())} bad_slots={bad}. {debug}"
            )

    @torch.no_grad()
    def deltakv_evict(self, seqs: list[Sequence]):
        with profiler.record("deltakv_less_memory_evict_total"):
            if not self.deltakv_layer_ids:
                return
            sink = int(self.config.sink_keep_tokens)
            recent = int(self.config.recent_keep_tokens)
            cluster_step = self._deltakv_base_cluster_step()
            cos_sin = self.cos_sin_cache[:, 0, :]

            for seq in seqs:
                row_idx = self.seq_id_to_row.get(seq.seq_id, None)
                if row_idx is None:
                    continue

                total_len = int(self.row_seq_lens[row_idx])
                compressed_len = int(self.row_deltakv_compressed_lens[row_idx])
                buffer_start = sink + compressed_len
                buffer_len = total_len - buffer_start
                if buffer_len <= recent:
                    continue

                evict_len = ((buffer_len - recent) // recent) * recent
                if evict_len <= 0:
                    continue

                evict_start = buffer_start
                evict_end = evict_start + evict_len
                with profiler.record("deltakv_less_memory_evict_read_slots"):
                    raw_slots_block = self.sparse_layer_raw_slots_map[row_idx, evict_start:evict_end].clone()
                invalid_raw = (raw_slots_block < 0) | (raw_slots_block >= int(self.deltakv_full_num_slots))
                if invalid_raw.any():
                    bad = raw_slots_block[invalid_raw][:16].detach().cpu().tolist()
                    raise RuntimeError(
                        "DeltaKV less-memory eviction expects in-range raw slots for the buffer block: "
                        f"row={row_idx} total_len={total_len} compressed_len={compressed_len} "
                        f"evict=({evict_start},{evict_end}) slot_to_pos_len={int(self.deltakv_full_num_slots)} "
                        f"bad_slots={bad}."
                    )

                with profiler.record("deltakv_less_memory_evict_select_centers"):
                    center_rel = self._deltakv_center_rel_for_block(
                        row_idx,
                        start=evict_start,
                        end=evict_end,
                        update_state=True,
                    )
                    new_center_slots = raw_slots_block[center_rel].to(torch.int32).clone()
                self._debug_track_deltakv_full_slots(
                    new_center_slots,
                    "evict_select_centers",
                    row=row_idx,
                    total_len=total_len,
                    compressed_len=compressed_len,
                    evict_start=evict_start,
                    evict_end=evict_end,
                )
                with profiler.record("deltakv_less_memory_evict_build_masks"):
                    sink_slots = self.sparse_layer_raw_slots_map[row_idx, :sink].to(torch.int32)
                    prev_center_slots_by_layer: dict[int, torch.Tensor] = {}
                    for layer_idx in self.deltakv_layer_ids:
                        existing = self.row_deltakv_center_slots[row_idx][layer_idx]
                        prev_center_slots_by_layer[layer_idx] = (
                            sink_slots if existing is None else existing.to(torch.int32)
                        )

                    is_center = torch.zeros((evict_len,), device=self.device, dtype=torch.bool)
                    is_center[center_rel] = True
                    to_free_mask = ~is_center

                with profiler.record("deltakv_less_memory_evict_alloc_latent"):
                    latent_slots = self._allocate_deltakv_latent(evict_len).to(torch.int32)
                    pos_all = torch.arange(evict_start, evict_end, device=self.device, dtype=torch.int32)
                    pos_to_free = pos_all[to_free_mask]
                    self.sparse_layer_latent_slots_map[row_idx, pos_all.to(torch.long)] = latent_slots
                    raw_slots_block_i32 = raw_slots_block.to(torch.int32)
                    raw_slots_block_i64 = raw_slots_block_i32.to(torch.long)

                with profiler.record("deltakv_less_memory_evict_validate_new_centers"):
                    self._validate_live_deltakv_center_slots(
                        new_center_slots,
                        row_idx=row_idx,
                        total_len=total_len,
                        compressed_len=compressed_len,
                        evict_start=evict_start,
                        evict_end=evict_end,
                        label="new",
                    )
                with profiler.record("deltakv_less_memory_evict_validate_centers"):
                    existing_center_values = [slots for slots in prev_center_slots_by_layer.values() if slots.numel() > 0]
                    if existing_center_values:
                        self._validate_live_deltakv_center_slots(
                            torch.cat(existing_center_values, dim=0),
                            row_idx=row_idx,
                            total_len=total_len,
                            compressed_len=compressed_len,
                            evict_start=evict_start,
                            evict_end=evict_end,
                            label="existing",
                        )

                shared_existing_center_slots = prev_center_slots_by_layer[self.deltakv_layer_ids[0]]
                shared_all_center_slots = torch.cat([shared_existing_center_slots, new_center_slots], dim=0)
                kv_dim_half = self.num_kv_heads * self.head_dim
                for layer_idx in self.deltakv_layer_ids:
                    l_idx = self.deltakv_layer_to_idx[layer_idx]
                    k_cache = self.deltakv_full_kv_cache[0, l_idx]
                    v_cache = self.deltakv_full_kv_cache[1, l_idx]
                    existing_center_slots = prev_center_slots_by_layer[layer_idx]
                    with profiler.record("deltakv_less_memory_evict_gather_raw"):
                        kv_block = torch.cat(
                            [
                                k_cache[raw_slots_block_i64].reshape(-1, kv_dim_half),
                                v_cache[raw_slots_block_i64].reshape(-1, kv_dim_half),
                            ],
                            dim=-1,
                        ).unsqueeze(0)

                    with profiler.record("deltakv_less_memory_evict_cluster"):
                        topk_center_indices, base_kv = self._cluster_compress(
                            layer_idx=layer_idx,
                            kv_states=kv_block,
                            existing_center_slots=existing_center_slots,
                            cluster_step=cluster_step,
                            new_center_rel=center_rel,
                            validate_centers=False,
                        )

                    with profiler.record("deltakv_less_memory_evict_store_fathers"):
                        father_slots_full = shared_all_center_slots[topk_center_indices.to(torch.long)]
                        father_slots = father_slots_full
                        K = self.deltakv_latent_to_full_slots.shape[-1]
                        k_eff = father_slots.shape[1]
                        if k_eff < K:
                            pad = father_slots[:, :1].expand(-1, K - k_eff)
                            father_slots = torch.cat([father_slots, pad], dim=1)
                        elif k_eff > K:
                            father_slots = father_slots[:, :K]
                        self.deltakv_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)

                    with profiler.record("deltakv_less_memory_evict_store_latent"):
                        self._deltakv_store_layer_latent(
                            l_idx=l_idx,
                            latent_slots=latent_slots,
                            kv_block=kv_block,
                            base_kv=base_kv,
                            store_all=True,
                        )

                for layer_idx in self.deltakv_layer_ids:
                    self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                        [prev_center_slots_by_layer[layer_idx], new_center_slots], dim=0
                    ).clone()

                with profiler.record("deltakv_less_memory_evict_free_full_slots"):
                    free_slots = raw_slots_block_i32[to_free_mask]
                    self._debug_track_deltakv_full_slots(
                        free_slots,
                        "evict_free_compressed",
                        row=row_idx,
                        total_len=total_len,
                        compressed_len=compressed_len,
                        evict_start=evict_start,
                        evict_end=evict_end,
                    )
                    ptr = self._num_free_slots_deltakv_full
                    self.free_slots_stack_deltakv_full[ptr: ptr + free_slots.numel()] = free_slots
                    self._num_free_slots_deltakv_full += free_slots.numel()
                    self.deltakv_slot_to_pos[free_slots] = -1

                    self.sparse_layer_raw_slots_map[row_idx, pos_to_free.to(torch.long)] = -1
                    self.row_deltakv_compressed_lens[row_idx] += evict_len
                    self.row_deltakv_compressed_lens_gpu[row_idx] += int(evict_len)

            with profiler.record("deltakv_less_memory_full_layer_quant_evict"):
                self._full_layer_quant_evict(seqs)
            with profiler.record("deltakv_less_memory_full_layer_kivi_evict"):
                self._full_layer_kivi_evict(seqs)

    @torch.no_grad()
    def _full_layer_quant_evict(self, seqs: list[Sequence]):
        if not self._full_layer_quant_enabled():
            return
        if not self.full_layer_ids:
            return
        sink = int(self.config.sink_keep_tokens)
        recent = int(self.config.recent_keep_tokens)
        cos_sin = self.cos_sin_cache[:, 0, :]

        for seq in seqs:
            row_idx = self.seq_id_to_row.get(seq.seq_id, None)
            if row_idx is None:
                continue

            total_len = int(self.row_seq_lens[row_idx])
            compressed_len = int(self.row_full_layer_compressed_lens[row_idx])
            buffer_start = sink + compressed_len
            buffer_len = total_len - buffer_start
            if buffer_len <= recent:
                continue

            evict_len = ((buffer_len - recent) // recent) * recent
            if evict_len <= 0:
                continue

            evict_start = buffer_start
            evict_end = evict_start + evict_len
            raw_slots_block = self.full_layer_slots_map[row_idx, evict_start:evict_end].clone().to(torch.int32)
            if (raw_slots_block < 0).any():
                raise RuntimeError("Full-layer residual quantization expects raw slots for the buffer block.")

            center_rel = self._full_layer_center_rel_for_block(
                row_idx,
                start=evict_start,
                end=evict_end,
                update_state=True,
            )
            new_center_slots = raw_slots_block[center_rel].to(torch.int32)
            sink_slots = self.full_layer_slots_map[row_idx, :sink].to(torch.int32)
            prev_center_slots_by_layer: dict[int, torch.Tensor] = {}
            for layer_idx in self.full_layer_ids:
                existing = self.row_deltakv_center_slots[row_idx][layer_idx]
                prev_center_slots_by_layer[layer_idx] = sink_slots if existing is None else existing.to(torch.int32)

            is_center = torch.zeros((evict_len,), device=self.device, dtype=torch.bool)
            if center_rel.numel() > 0:
                is_center[center_rel] = True
            to_compress_mask = ~is_center
            num_to_compress = int(evict_len) - int(center_rel.numel())
            if num_to_compress <= 0:
                for layer_idx in self.full_layer_ids:
                    self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                        [prev_center_slots_by_layer[layer_idx], new_center_slots],
                        dim=0,
                    )
                self.row_full_layer_compressed_lens[row_idx] += evict_len
                continue

            latent_slots = self._allocate_full_layer_latent(num_to_compress).to(torch.int32)
            pos_all = torch.arange(evict_start, evict_end, device=self.device, dtype=torch.int32)
            pos_to_compress = pos_all[to_compress_mask]
            self.full_layer_latent_slots_map[row_idx, pos_to_compress.to(torch.long)] = latent_slots

            for layer_idx in self.full_layer_ids:
                l_idx = self.full_layer_to_idx[layer_idx]
                k_cache = self.full_kv_cache[0, l_idx]
                v_cache = self.full_kv_cache[1, l_idx]
                kv_block = self._deltakv_gather_raw_kv(
                    slots=raw_slots_block,
                    pos=pos_all,
                    cos_sin=cos_sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                ).unsqueeze(0)

                existing_center_slots = prev_center_slots_by_layer[layer_idx]
                topk_center_indices, base_kv = self._full_layer_cluster_compress(
                    layer_idx=layer_idx,
                    kv_states=kv_block,
                    existing_center_slots=existing_center_slots,
                    new_center_rel=center_rel,
                )
                all_center_slots = torch.cat([existing_center_slots, new_center_slots], dim=0)
                father_slots_full = all_center_slots[topk_center_indices.to(torch.long)]
                father_slots = father_slots_full[to_compress_mask]
                k_neighbors = self.full_layer_latent_to_full_slots.shape[-1]
                k_eff = father_slots.shape[1]
                if k_eff < k_neighbors:
                    pad = father_slots[:, :1].expand(-1, k_neighbors - k_eff)
                    father_slots = torch.cat([father_slots, pad], dim=1)
                elif k_eff > k_neighbors:
                    father_slots = father_slots[:, :k_neighbors]
                self.full_layer_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)
                residual = (kv_block - base_kv).squeeze(0)[to_compress_mask]
                self._store_full_layer_residual(l_idx, latent_slots, residual)

            for layer_idx in self.full_layer_ids:
                self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                    [prev_center_slots_by_layer[layer_idx], new_center_slots],
                    dim=0,
                )

            free_slots = raw_slots_block[to_compress_mask]
            ptr = self._num_free_slots_full
            self.free_slots_stack_full[ptr: ptr + free_slots.numel()] = free_slots
            self._num_free_slots_full += free_slots.numel()
            self.full_layer_slot_to_pos[free_slots] = -1
            self.full_layer_slots_map[row_idx, pos_to_compress.to(torch.long)] = -1
            self.row_full_layer_compressed_lens[row_idx] += evict_len

    def free_seq(self, seq_id: int):
        if self._full_layer_kivi_enabled() and not self._full_layer_quant_enabled():
            with profiler.record("cache_free_seq"):
                self._release_prompt_admission_reservations(seq_id)
                row_idx = self.seq_id_to_row.pop(seq_id, None)
                if row_idx is None:
                    raise ValueError
                self.raw_kv_offload_buffer.release_row(int(row_idx))

                cur_len = int(self.row_seq_lens[row_idx])
                if cur_len <= 0:
                    raise ValueError

                full_slots = self.full_layer_slots_map[row_idx, :cur_len]
                full_mask = full_slots >= 0
                if full_mask.any():
                    slots = full_slots[full_mask].to(torch.int32)
                    ptr = self._num_free_slots_full
                    self.free_slots_stack_full[ptr: ptr + slots.numel()] = slots
                    self._num_free_slots_full += slots.numel()
                    self.full_layer_slot_to_pos[slots.to(torch.long)] = -1

                kivi_blocks = self.full_layer_kivi_block_slots_map[row_idx, :cur_len]
                kivi_mask = kivi_blocks >= 0
                if kivi_mask.any():
                    blocks = torch.unique(kivi_blocks[kivi_mask]).to(torch.int32)
                    ptr = self._num_free_slots_full_layer_kivi
                    self.free_slots_stack_full_layer_kivi[ptr: ptr + blocks.numel()] = blocks
                    self._num_free_slots_full_layer_kivi += blocks.numel()
                    self.full_layer_kivi_block_start_pos[blocks.to(torch.long)] = -1

                slots = self._active_deltakv_raw_slots_for_free(row_idx, cur_len)
                if slots.numel() > 0:
                    self._debug_track_deltakv_full_slots(slots, "free_seq_raw", seq_id=int(seq_id), row=row_idx)
                    ptr = self._num_free_slots_deltakv_full
                    self.free_slots_stack_deltakv_full[ptr: ptr + slots.numel()] = slots
                    self._num_free_slots_deltakv_full += slots.numel()
                    self.deltakv_slot_to_pos[slots.to(torch.long)] = -1

                latent_slots = self.sparse_layer_latent_slots_map[row_idx, :cur_len]
                mask_latent = latent_slots >= 0
                if mask_latent.any():
                    slots = latent_slots[mask_latent].to(torch.int32)
                    ptr = self._num_free_slots_deltakv_latent
                    self.free_slots_stack_deltakv_latent[ptr: ptr + slots.numel()] = slots
                    self._num_free_slots_deltakv_latent += slots.numel()

                self.full_layer_slots_map[row_idx, :] = 0
                self.full_layer_kivi_block_slots_map[row_idx, :] = -1
                self.sparse_layer_raw_slots_map[row_idx, :] = -1
                self.sparse_layer_latent_slots_map[row_idx, :] = -1
                self.row_seq_lens[row_idx] = 0
                self.row_deltakv_compressed_lens[row_idx] = 0
                self.row_deltakv_compressed_lens_gpu[row_idx] = 0
                if self.row_full_layer_kivi_quantized_lens is not None:
                    self.row_full_layer_kivi_quantized_lens[row_idx] = 0
                if self.row_full_layer_kivi_quantized_lens_gpu is not None:
                    self.row_full_layer_kivi_quantized_lens_gpu[row_idx] = 0
                self.row_deltakv_center_slots[row_idx] = [None for _ in range(self.num_layers)]
                self.free_rows.append(row_idx)
            return

        if not self._full_layer_quant_enabled():
            row_idx = self.seq_id_to_row.get(seq_id, None)
            if row_idx is not None and self.row_full_layer_kivi_quantized_lens is not None:
                cur_len = int(self.row_seq_lens[row_idx])
                if self._full_layer_kivi_enabled() and cur_len > 0:
                    kivi_blocks = self.full_layer_kivi_block_slots_map[row_idx, :cur_len]
                    mask = kivi_blocks >= 0
                    if mask.any():
                        blocks = torch.unique(kivi_blocks[mask]).to(torch.int32)
                        ptr = self._num_free_slots_full_layer_kivi
                        self.free_slots_stack_full_layer_kivi[ptr: ptr + blocks.numel()] = blocks
                        self._num_free_slots_full_layer_kivi += blocks.numel()
                        self.full_layer_kivi_block_start_pos[blocks.to(torch.long)] = -1
                        self.full_layer_kivi_block_slots_map[row_idx, :cur_len] = -1
                self.row_full_layer_kivi_quantized_lens[row_idx] = 0
                if self.row_full_layer_kivi_quantized_lens_gpu is not None:
                    self.row_full_layer_kivi_quantized_lens_gpu[row_idx] = 0
            return super().free_seq(seq_id)

        with profiler.record("cache_free_seq"):
            self._release_prompt_admission_reservations(seq_id)
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                raise ValueError
            self.raw_kv_offload_buffer.release_row(int(row_idx))

            cur_len = self.row_seq_lens[row_idx]
            if cur_len <= 0:
                raise ValueError

            full_slots = self.full_layer_slots_map[row_idx, :cur_len]
            full_mask = full_slots >= 0
            if full_mask.any():
                slots = full_slots[full_mask].to(torch.int32)
                ptr = self._num_free_slots_full
                self.free_slots_stack_full[ptr: ptr + slots.numel()] = slots
                self._num_free_slots_full += slots.numel()
                self.full_layer_slot_to_pos[slots] = -1

            full_latent_slots = self.full_layer_latent_slots_map[row_idx, :cur_len]
            full_latent_mask = full_latent_slots >= 0
            if full_latent_mask.any():
                slots = full_latent_slots[full_latent_mask].to(torch.int32)
                ptr = self._num_free_slots_full_layer_latent
                self.free_slots_stack_full_layer_latent[ptr: ptr + slots.numel()] = slots
                self._num_free_slots_full_layer_latent += slots.numel()

            slots = self._active_deltakv_raw_slots_for_free(row_idx, int(cur_len))
            if slots.numel() > 0:
                self._debug_track_deltakv_full_slots(slots, "free_seq_raw", seq_id=int(seq_id), row=row_idx)
                ptr = self._num_free_slots_deltakv_full
                self.free_slots_stack_deltakv_full[ptr: ptr + slots.numel()] = slots
                self._num_free_slots_deltakv_full += slots.numel()
                self.deltakv_slot_to_pos[slots] = -1

            latent_slots = self.sparse_layer_latent_slots_map[row_idx, :cur_len]
            mask_latent = latent_slots >= 0
            if mask_latent.any():
                slots = latent_slots[mask_latent].to(torch.int32)
                ptr = self._num_free_slots_deltakv_latent
                self.free_slots_stack_deltakv_latent[ptr: ptr + slots.numel()] = slots
                self._num_free_slots_deltakv_latent += slots.numel()

            self.full_layer_slots_map[row_idx, :] = 0
            self.full_layer_latent_slots_map[row_idx, :] = -1
            self.sparse_layer_raw_slots_map[row_idx, :] = -1
            self.sparse_layer_latent_slots_map[row_idx, :] = -1
            self.row_seq_lens[row_idx] = 0
            self.row_deltakv_compressed_lens[row_idx] = 0
            self.row_deltakv_compressed_lens_gpu[row_idx] = 0
            self.row_full_layer_compressed_lens[row_idx] = 0
            if self.row_full_layer_kivi_quantized_lens is not None:
                self.row_full_layer_kivi_quantized_lens[row_idx] = 0
            if self.row_full_layer_kivi_quantized_lens_gpu is not None:
                self.row_full_layer_kivi_quantized_lens_gpu[row_idx] = 0
            self.row_deltakv_center_slots[row_idx] = [None for _ in range(self.num_layers)]
            self.free_rows.append(row_idx)

    @torch.no_grad()
    def deltakv_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
        chunk_lens: torch.Tensor | None,
        return_reconstruct_temp_slots: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del context_lens, chunk_lens
        with profiler.record("deltakv_less_memory_reconstruct_total"):
            active_slots, local_req, new_context_lens, temp_slots, recon_pos, recon_latent, recon_out_slot = (
                self._deltakv_build_view_and_plan_reconstruct(layer_idx, active_compressed_indices, req_indices)
            )
            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_cache = self.deltakv_full_kv_cache[0, l_idx]
            v_cache = self.deltakv_full_kv_cache[1, l_idx]
            kv_dim = 2 * self.num_kv_heads * self.head_dim

            if recon_latent.numel() > 0:
                static_decode = not get_context().is_prefill
                safe_recon_latent = recon_latent.clamp_min(0) if static_decode else recon_latent
                father_slots = self.deltakv_latent_to_full_slots[l_idx, safe_recon_latent].to(torch.int32)
                if static_decode:
                    father_slots = father_slots.clamp_min(0)
                elif (father_slots < 0).any():
                    raise RuntimeError("DeltaKV less-memory: missing father slots for reconstruction.")

                cos_sin = self.cos_sin_cache[:, 0, :]
                with profiler.record("deltakv_less_memory_reconstruct_load_residual"):
                    kv_delta = self._load_residual(l_idx, safe_recon_latent, kv_dim)
                with profiler.record("deltakv_less_memory_reconstruct_writeback"):
                    self._deltakv_reconstruct_writeback(
                        kv_delta=kv_delta,
                        father_slots=father_slots,
                        slot_to_pos=self.deltakv_slot_to_pos,
                        out_slots=recon_out_slot,
                        out_pos=recon_pos,
                        cos_sin=cos_sin,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        l_idx=l_idx,
                    )

            static_decode = not get_context().is_prefill
            if static_decode:
                attn_active_slots = active_slots
                materialized_temp_slots = torch.empty((0,), device=active_slots.device, dtype=torch.int32)
            else:
                self._set_postrope_slots(layer_idx, recon_out_slot, validate=False)
                attn_active_slots = active_slots.clone()
                attn_active_slots, materialized_temp_slots = self._materialize_deltakv_active_postrope_view(
                    layer_idx,
                    attn_active_slots,
                    new_context_lens,
                    recon_out_slot,
                )
            if materialized_temp_slots.numel() > 0:
                postrope_slots = torch.cat((recon_out_slot, materialized_temp_slots.to(recon_out_slot.device)), dim=0)
            else:
                postrope_slots = recon_out_slot
            with profiler.record("deltakv_less_memory_reconstruct_mark_postrope"):
                self._set_postrope_slots(layer_idx, postrope_slots, validate=False)
            returned_temp_slots = []
            if materialized_temp_slots.numel() > 0:
                returned_temp_slots.append(materialized_temp_slots.to(temp_slots.device))
            if self._should_return_reconstruct_temp_slots(
                static_decode=static_decode,
                return_reconstruct_temp_slots=return_reconstruct_temp_slots,
                temp_slots=temp_slots,
            ):
                returned_temp_slots.append(temp_slots)
            if returned_temp_slots:
                temp_slots = torch.cat(returned_temp_slots, dim=0).to(torch.int32)
            else:
                temp_slots = torch.empty((0,), device=active_slots.device, dtype=torch.int32)

            debug_layers = os.getenv("SPARSEENGINE_DEBUG_RECONSTRUCT_LAYERS")
            if debug_layers:
                wanted = {int(part) for part in debug_layers.split(",") if part.strip()}
                if int(layer_idx) in wanted and active_slots.shape[0] > 0:
                    active_len = int(new_context_lens[0].item())
                    slots = attn_active_slots[0, :active_len].to(torch.long)
                    active_pos = getattr(self, "_deltakv_decode_static_active_pos", None)
                    if static_decode and isinstance(active_pos, torch.Tensor):
                        pos = active_pos[0, :active_len].to(torch.int32)
                    else:
                        pos = self.deltakv_slot_to_pos[slots].to(torch.int32)
                    if (pos < 0).any():
                        unknown = torch.nonzero(pos < 0, as_tuple=False).flatten()
                        preview = unknown[:16]
                        raise RuntimeError(
                            "DeltaKV less-memory reconstruct debug saw unknown position: "
                            f"layer={layer_idx}, active_len={active_len}, "
                            f"unknown_count={int(unknown.numel())}, "
                            f"unknown_indices={[int(x) for x in preview.detach().cpu().tolist()]}, "
                            f"unknown_slots={[int(x) for x in slots[preview].detach().cpu().tolist()]}."
                        )
                    debug = getattr(self, "debug_last_reconstruct", {})
                    k_max = int(active_compressed_indices.shape[1]) if active_compressed_indices is not None else 0
                    debug[int(layer_idx)] = {
                        "positions": pos.detach().cpu(),
                        "active_slots": slots.detach().cpu(),
                        "recon_pos": recon_pos[:k_max].detach().cpu(),
                        "recon_latent": recon_latent[:k_max].detach().cpu(),
                        "recon_out_slot": recon_out_slot[:k_max].detach().cpu(),
                        "k": k_cache[slots].permute(1, 0, 2).unsqueeze(0).detach().float().cpu(),
                        "v": v_cache[slots].permute(1, 0, 2).unsqueeze(0).detach().float().cpu(),
                    }
                    self.debug_last_reconstruct = debug

            return attn_active_slots, local_req, new_context_lens, temp_slots
