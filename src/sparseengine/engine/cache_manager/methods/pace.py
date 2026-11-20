"""PACE: H2O eviction over foresight-adjusted scores, with optional bounded positions.

See ``docs/development/pace-runtime.md``. When a prefill chunk's H2O score
entries are appended, each entry receives the request plan's bonus at its
logical position. At the first chunk of a resumed chain turn, the bonus
already folded into every resident score is replaced by the bonus under the
turn's plan. Selection, budgets, compaction and decode eviction are H2O's.

Row-aligned state per (layer, sequence) mirrors the H2O score vector:

* ``logical`` int64: session position (plans use these coordinates);
* ``bonus`` float32: bonus currently folded into the score;
* ``physical`` int64: RoPE position the stored key is rotated at;
* ``ref`` int64: reference-store entry, ``-1`` when none.

Prefill columns are recorded when their scores are appended. Decode columns
are recorded lazily, before the next eviction, renumber, turn end or
snapshot: until then the unrecorded columns of a row are exactly its newest
decode tokens, at consecutive logical positions after the recorded ones and
written under the sequence's current shift.

With ``pace_bounded_positions`` each sequence has ``physical = logical -
shift``. Once ``next_physical + headroom`` reaches the limit, every KV layer's
resident row is renumbered to ``0..R-1`` in logical order, its keys are
re-rotated, and the shift grows by ``next_physical - R``.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from sparseengine.engine.pace_plan import CompiledPACEBonus, PACEPlan, compile_bonus
from sparseengine.engine.sequence import Sequence
from sparseengine.method_registry import pace_position_headroom
from sparseengine.utils.log import log_once

from .h2o import H2OCacheManager
from .pace_positions import PACEReferenceStore, rotate_keys

_ROW_FIELDS = ("logical", "bonus", "physical", "ref")
_SNAPSHOT_VERSION = 1


def pace_reference_store_bytes(config) -> int:
    """Bytes reserved for bounded positions: the reference store and the float32
    re-rotation table (0 when bounded positions are off)."""
    if not (
        getattr(config, "sparse_method", "") == "pace"
        and bool(getattr(config, "pace_bounded_positions", False))
    ):
        return 0
    from sparseengine.models.layout import resolve_attention_qk_head_dim
    from sparseengine.models.rope import resolve_rope_max_position

    hf = config.hf_config
    head_dim = int(resolve_attention_qk_head_dim(hf))
    table = int(resolve_rope_max_position(hf, model_name="PACE")) * head_dim * 4
    capacity = int(getattr(config, "pace_reference_tokens", 0) or 0)
    if capacity <= 0:
        return table
    per_layer = PACEReferenceStore.nbytes(
        capacity=capacity,
        num_kv_heads=int(hf.num_key_value_heads),
        head_dim=head_dim,
        element_size=torch.empty((), dtype=hf.dtype).element_size(),
    )
    return per_layer * int(config.runtime_layout.num_kv_layers) + table


class PACECacheManager(H2OCacheManager):
    """H2O cache manager whose prefill scores carry a per-request foresight bonus."""

    def __init__(
        self,
        config,
        parallel_context,
        *,
        allocation_budget_bytes: int | None = None,
    ):
        super().__init__(
            config,
            parallel_context,
            allocation_budget_bytes=allocation_budget_bytes,
        )
        self._pace_init_state()

    def _pace_init_state(self) -> None:
        # Long-lived: allocated with the KV pool; per-sequence ownership resets.
        if not hasattr(self, "_pace_stores"):
            self._pace_stores: dict[int, PACEReferenceStore] = {}
        if not hasattr(self, "_pace_cos_sin"):
            self._pace_cos_sin: torch.Tensor | None = None
        # (layer_idx, seq_id) -> row-aligned state, same keys as _h2o_scores.
        self._pace_rows: dict[str, dict[tuple[int, int], torch.Tensor]] = {
            name: {} for name in _ROW_FIELDS
        }
        # (layer_idx, seq_id) -> logical end of the recorded columns.
        self._pace_end: dict[tuple[int, int], int] = {}
        # seq_id -> logical - physical for the sequence's current writes.
        self._pace_shift: dict[int, int] = {}
        # seq_id -> (plan, compiled bonus on self.device).
        self._pace_plans: dict[int, tuple[PACEPlan, CompiledPACEBonus]] = {}
        # seq_id -> (logical start, length, bonus or None) for the current step;
        # the bonus depends only on logical position, so layers share it.
        self._pace_chunk_bonus: dict[int, tuple[int, int, torch.Tensor | None]] = {}
        self._pace_counters = {
            "bonused_chunks": 0,
            "rebased_turns": 0,
            "renumbers": 0,
            "rerotated_keys": 0,
            "reference_overflow": 0,
        }
        for store in self._pace_stores.values():
            store.reset()

    # ------------------------------------------------------------------
    # Static configuration
    # ------------------------------------------------------------------
    @property
    def _pace_bounded(self) -> bool:
        return bool(getattr(self.config, "pace_bounded_positions", False))

    @property
    def _pace_limit(self) -> int:
        return int(self.config.pace_resolved_position_limit)

    @property
    def _pace_headroom(self) -> int:
        return pace_position_headroom(self.config.engine_prefill_chunk_size)

    def _pace_cos_sin_cache(self) -> torch.Tensor:
        """RoPE table for re-rotation: the models' parameters (Qwen2/Qwen3/Llama), built on
        the KV device. A CPU-built table differs by one float32 rounding step in some
        entries; the device-built one reproduces the reference implementation exactly."""
        if self._pace_cos_sin is None:
            from sparseengine.layers.rotary_embedding import build_rope_cos_sin_table
            from sparseengine.models.rope import (
                resolve_rope_max_position,
                resolve_rope_scaling,
                resolve_rope_theta,
            )

            hf = self.config.hf_config
            self._pace_cos_sin = build_rope_cos_sin_table(
                int(self.head_dim),
                resolve_rope_max_position(hf, model_name="PACE"),
                resolve_rope_theta(hf),
                resolve_rope_scaling(hf, model_name="PACE"),
                device=self.device,
            )
        return self._pace_cos_sin

    # ------------------------------------------------------------------
    # Memory: the reference store is part of the startup KV budget
    # ------------------------------------------------------------------
    def _get_available_slots_info(self) -> tuple[int, int]:
        available, slot_bytes = super()._get_available_slots_info()
        reserved = pace_reference_store_bytes(self.config)
        if reserved and int(available) <= reserved:
            raise RuntimeError(
                "Not enough memory for PACE bounded positions (reference store and RoPE "
                f"table): required={reserved} available={int(available)} bytes. Lower "
                "pace_reference_tokens if it is positive, or the RoPE limit."
            )
        return int(available) - reserved, slot_bytes

    def _pace_validate_rotary(self) -> None:
        """Check at startup, before serving, that re-rotation can be exact: the model must
        declare full-width rotary (the table is built from the head dimension, so only
        the checkpoint config can reveal partial rotary), and the table must match the
        stored keys. Otherwise an unsupported model would fail, or silently corrupt keys,
        at its first renumber thousands of tokens into a session.

        Layout: re-rotation assumes split-half (NeoX) RoPE. Configuration admits only
        the families in SPARSE_METHOD_MODEL_TYPES["pace"] (Qwen2, Qwen3, Llama), whose
        attention calls get_rope without ``interleaved`` (default False, split-half); interleaved
        (GPT-J style) models such as GLM are not admitted. The model type is checked
        again here so a registry change cannot bypass it. SparseEngine has no
        ``mistral`` model type: Mistral v0.3 (no sliding window, Llama weight layout)
        runs from a config relabelled to ``llama`` and passes as Llama."""
        from sparseengine.method_registry import SPARSE_METHOD_MODEL_TYPES
        from sparseengine.models.rope import resolve_declared_rotary_dim

        model_type = str(getattr(self.config.hf_config, "model_type", "") or "")
        if model_type not in SPARSE_METHOD_MODEL_TYPES["pace"]:
            raise NotImplementedError(
                f"pace_bounded_positions assumes split-half RoPE, validated for "
                f"{sorted(SPARSE_METHOD_MODEL_TYPES['pace'])}; got model_type={model_type!r}."
            )

        declared = resolve_declared_rotary_dim(self.config.hf_config, int(self.head_dim))
        if declared != int(self.head_dim):
            raise NotImplementedError(
                "pace_bounded_positions supports full-width split-half RoPE only; the model "
                f"declares rotary dimension {declared} for head dimension {int(self.head_dim)}."
            )
        table_dim = int(self._pace_cos_sin_cache().shape[-1])
        for layer_idx in self.kv_transformer_layer_indices():
            key_dim = int(self.get_layer_kv_cache(layer_idx)[0].shape[-1])
            if table_dim != key_dim:
                raise NotImplementedError(
                    "pace_bounded_positions supports full-width split-half RoPE only: "
                    f"rotary table dimension {table_dim} != key head dimension "
                    f"{key_dim} (layer {layer_idx})."
                )

    def allocate_kv_cache(self):
        result = super().allocate_kv_cache()
        self._pace_stores = {}
        if self._pace_bounded:
            # Rebuilt on every allocation: startup profiling replaces the KV pool, and
            # the table must live on the pool's device.
            self._pace_cos_sin = None
            self._pace_validate_rotary()
        capacity = int(getattr(self.config, "pace_reference_tokens", 0) or 0)
        if self._pace_bounded and capacity > 0:
            for layer_idx in self.kv_transformer_layer_indices():
                k_cache, _ = self.get_layer_kv_cache(layer_idx)
                self._pace_stores[self.kv_layer_index(layer_idx)] = PACEReferenceStore(
                    capacity=capacity,
                    num_kv_heads=int(k_cache.shape[-2]),
                    head_dim=int(k_cache.shape[-1]),
                    dtype=k_cache.dtype,
                    device=k_cache.device,
                )
        return result

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def _pace_compiled(self, seq: Sequence) -> CompiledPACEBonus | None:
        sid = int(seq.seq_id)
        plan = getattr(seq, "pace_plan", None)
        if plan is None:
            self._pace_plans.pop(sid, None)
            return None
        cached = self._pace_plans.get(sid)
        if cached is None or cached[0] != plan:
            compiled = compile_bonus(
                plan,
                scale=float(self.config.pace_bonus_scale),
                decay=float(self.config.pace_bonus_decay),
            ).to(self.device)
            cached = (plan, compiled)
            self._pace_plans[sid] = cached
        return None if cached[1].empty else cached[1]

    def _pace_chunk_bonus_for(
        self,
        seq: Sequence,
        start: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        sid = int(seq.seq_id)
        cached = self._pace_chunk_bonus.get(sid)
        if cached is not None and cached[0] == start and cached[1] == length:
            return cached[2]
        compiled = self._pace_compiled(seq)
        bonus = None
        if compiled is not None and length > 0:
            positions = torch.arange(start, start + length, dtype=torch.int64, device=device)
            bonus = compiled.to(device).lookup(positions)
            self._pace_counters["bonused_chunks"] += 1
        self._pace_chunk_bonus[sid] = (start, length, bonus)
        return bonus

    # ------------------------------------------------------------------
    # Row-aligned state
    # ------------------------------------------------------------------
    def _pace_row_len(self, layer_idx: int, seq_id: int) -> int | None:
        row = self.seq_id_to_row[layer_idx].get(int(seq_id))
        if row is None:
            return None
        return int(self.row_seq_lens[layer_idx][row])

    def _pace_recorded(self, key) -> int:
        logical = self._pace_rows["logical"].get(key)
        return 0 if logical is None else int(logical.numel())

    def _pace_append(self, key, logical, bonus, shift: int) -> None:
        new = {
            "logical": logical,
            "bonus": bonus,
            "physical": logical - int(shift),
            "ref": torch.full_like(logical, -1),
        }
        for name in _ROW_FIELDS:
            rows = self._pace_rows[name]
            rows[key] = new[name] if key not in rows else torch.cat((rows[key], new[name]))

    def _pace_sync(self, layer_idx: int, seq_id: int, length: int | None = None) -> None:
        """Record unrecorded newest columns (decode tokens, bonus 0) up to ``length``."""
        target = self._pace_row_len(layer_idx, seq_id) if length is None else int(length)
        if target is None:
            return
        key = self._score_key(layer_idx, seq_id)
        recorded = self._pace_recorded(key)
        if target < recorded:
            raise RuntimeError(
                "PACE row state is longer than the physical row: "
                f"layer={layer_idx} seq_id={seq_id} recorded={recorded} row={target}."
            )
        if target == recorded:
            return
        end = self._pace_end.get(key, 0)
        tail = target - recorded
        logical = torch.arange(end, end + tail, dtype=torch.int64, device=self.device)
        bonus = torch.zeros(tail, dtype=torch.float32, device=self.device)
        self._pace_append(key, logical, bonus, self._pace_shift.get(int(seq_id), 0))
        self._pace_end[key] = end + tail

    def _pace_sync_seq(self, seq_id: int) -> None:
        for layer_idx in self.kv_transformer_layer_indices():
            self._pace_sync(layer_idx, seq_id)

    def _pace_sync_all(self) -> None:
        for layer_idx, seq_id in list(self._pace_rows["logical"]):
            self._pace_sync(layer_idx, seq_id)

    def pace_row_state(self, layer_idx: int, seq_id: int) -> dict[str, torch.Tensor] | None:
        """Return the row-aligned state of a resident row, recording pending columns."""
        self._pace_sync(layer_idx, seq_id)
        key = self._score_key(layer_idx, seq_id)
        if key not in self._pace_rows["logical"]:
            return None
        return {name: self._pace_rows[name][key] for name in _ROW_FIELDS}

    def pace_shift(self, seq_id: int) -> int:
        return int(self._pace_shift.get(int(seq_id), 0))

    # ------------------------------------------------------------------
    # H2O hooks
    # ------------------------------------------------------------------
    def _on_prefill_score_appended(
        self,
        layer_idx: int,
        seq: Sequence,
        score: torch.Tensor,
        *,
        appended_from: int,
    ) -> None:
        sid = int(seq.seq_id)
        key = self._score_key(layer_idx, sid)
        self._pace_sync(layer_idx, sid, length=appended_from)
        start = int(seq.num_prefilled_tokens)
        recorded_end = self._pace_end.get(key, 0)
        if recorded_end != start:
            raise RuntimeError(
                "PACE prefill chunk does not continue the recorded row: "
                f"layer={layer_idx} seq_id={sid} chunk_start={start} "
                f"recorded_end={recorded_end}."
            )
        length = int(score.numel()) - int(appended_from)
        # The bonus relies on float32 scores. H2O creates every score vector in float32
        # (_expand_score, the decode workspaces) and derives the rest from those by
        # gather, pad or add; a snapshot restore keeps the stored dtype.
        if score.dtype != torch.float32:
            raise RuntimeError(f"PACE requires float32 H2O scores, got {score.dtype}.")
        bonus = self._pace_chunk_bonus_for(seq, start, length, score.device)
        if bonus is not None:
            score[appended_from:].add_(bonus)
            bonus = bonus.clone()
        else:
            bonus = torch.zeros(length, dtype=torch.float32, device=score.device)
        positions = torch.arange(start, start + length, dtype=torch.int64, device=score.device)
        self._pace_append(key, positions, bonus, self._pace_shift.get(sid, 0))
        self._pace_end[key] = start + length

    def _store_kept_score(
        self,
        layer_idx: int,
        seq_id: int,
        kept_score: torch.Tensor,
        keep_indices: torch.Tensor,
        *,
        previous_len: int,
    ) -> None:
        super()._store_kept_score(
            layer_idx, seq_id, kept_score, keep_indices, previous_len=previous_len,
        )
        key = self._score_key(layer_idx, seq_id)
        recorded = self._pace_recorded(key)
        if recorded != int(previous_len):
            raise RuntimeError(
                "PACE row state was not recorded before eviction: "
                f"layer={layer_idx} seq_id={seq_id} recorded={recorded} "
                f"row={int(previous_len)}."
            )
        logical = self._pace_rows["logical"][key]
        index = keep_indices.to(device=logical.device, dtype=torch.long).reshape(-1)
        store = self._pace_stores.get(self.kv_layer_index(layer_idx))
        if store is not None:
            dropped = torch.ones(recorded, dtype=torch.bool, device=logical.device)
            dropped[index] = False
            store.release(self._pace_rows["ref"][key][dropped])
        for name in _ROW_FIELDS:
            self._pace_rows[name][key] = self._pace_rows[name][key].index_select(0, index)

    # Every H2O eviction path enters through one of these; record pending decode
    # columns first, so keep indices address recorded state.
    def _evict(self, seqs: list[Sequence], *, is_prefill: bool):
        for seq in seqs:
            self._pace_sync_seq(int(seq.seq_id))
        return super()._evict(seqs, is_prefill=is_prefill)

    def _evict_decode_rows(self, seqs: list[Sequence]) -> None:
        self._pace_sync_all()
        return super()._evict_decode_rows(seqs)

    def free_part_slots(self, layer_idx, seq, keep_indices, *args, **kwargs):
        self._pace_sync(layer_idx, int(seq.seq_id))
        return super().free_part_slots(layer_idx, seq, keep_indices, *args, **kwargs)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def _pace_apply_offsets(self, seqs: list[Sequence], is_prefill: bool) -> None:
        """Publish each sequence's shift and check the physical bound for this step."""
        if not self._pace_bounded:
            return
        limit = self._pace_limit
        for seq in seqs:
            sid = int(seq.seq_id)
            if is_prefill and int(seq.num_prefilled_tokens) == 0:
                self._pace_shift.pop(sid, None)  # a new sequence starts unshifted
            shift = self._pace_shift.get(sid, 0)
            seq.position_offset = shift
            # Both branches compute the PHYSICAL range the forward will write.
            # Prefill: num_prefilled_tokens is the logical chunk start and the prefill
            # producers subtract position_offset themselves, so subtract it here too.
            # Decode: Sequence.decode_input_position already subtracts position_offset
            # (set on the line above); it is the physical position as given.
            if is_prefill:
                start = int(seq.num_prefilled_tokens) - shift
                end = start + int(seq.current_chunk_size)
            else:
                start = int(seq.decode_input_position)
                end = start + 1
            if start < 0 or end > limit:
                raise RuntimeError(
                    "PACE bounded positions violated: "
                    f"seq_id={sid} physical=[{start},{end}) limit={limit} shift={shift}."
                )

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        self._pace_chunk_bonus.clear()
        if is_prefill:
            for seq in seqs:
                reused = int(getattr(seq, "chain_reused_tokens", 0) or 0)
                if reused > 0 and int(seq.num_prefilled_tokens) == reused:
                    self._pace_rebase(seq)
        self._pace_apply_offsets(seqs, is_prefill)
        return super().prepare_step(seqs, is_prefill)

    def prepare_decode_graph_step(self, seqs: list[Sequence], state):
        # Not graph capture: eager decode (run_eager_static) also prepares through
        # this hook. Bounded positions reject decode_graph=True at configuration.
        self._pace_apply_offsets(seqs, False)
        return super().prepare_decode_graph_step(seqs, state)

    def validate_decode_graph_path_capacity(self, seqs: list[Sequence], *, capacity: int) -> None:
        if not self._pace_bounded:
            return super().validate_decode_graph_path_capacity(seqs, capacity=capacity)
        # Decode kernels see the physical row, not the logical session length.
        actual = 0
        for seq in seqs:
            for layer_idx in self.kv_transformer_layer_indices():
                length = self._pace_row_len(layer_idx, int(seq.seq_id))
                if length is not None:
                    actual = max(actual, length + 1)
        if int(capacity) < actual:
            raise RuntimeError(
                "decode path capacity does not cover the physical row: "
                f"capacity={capacity}, physical={actual}."
            )

    @torch.no_grad()
    def _pace_maybe_renumber(self, seq_ids) -> None:
        if not self._pace_bounded:
            return
        limit, headroom = self._pace_limit, self._pace_headroom
        for sid in dict.fromkeys(int(s) for s in seq_ids):
            layers = [
                layer for layer in self.kv_transformer_layer_indices()
                if self.seq_id_to_row[layer].get(sid) is not None
            ]
            if not layers:
                continue
            for layer in layers:
                self._pace_sync(layer, sid)
            keys = [self._score_key(layer, sid) for layer in layers]
            ends = {self._pace_end.get(key, 0) for key in keys}
            lengths = {self._pace_recorded(key) for key in keys}
            if len(ends) != 1 or len(lengths) != 1:
                raise RuntimeError(
                    f"PACE renumber needs aligned KV-layer rows: seq_id={sid} "
                    f"ends={sorted(ends)} lengths={sorted(lengths)}."
                )
            shift = self._pace_shift.get(sid, 0)
            next_physical = ends.pop() - shift
            if next_physical + headroom < limit:
                continue
            resident = lengths.pop()
            if resident + headroom >= limit:
                raise RuntimeError(
                    f"PACE renumber cannot bound seq_id={sid}: resident={resident} + "
                    f"headroom={headroom} >= limit={limit}."
                )
            cos_sin = self._pace_cos_sin_cache()
            for layer, key in zip(layers, keys):
                self._pace_renumber_row(layer, sid, key, resident, cos_sin)
            self._pace_shift[sid] = shift + next_physical - resident
            self._pace_counters["renumbers"] += 1

    def _pace_reference_priority(self, key) -> torch.Tensor:
        """Reference-entry order when the store runs short: tokens carrying a plan
        bonus first (the plan says they will be read again, so they survive the most
        renumbers), then the oldest. Exact int64: logical positions stay below 2**40."""
        bonus = self._pace_rows["bonus"][key]
        logical = self._pace_rows["logical"][key]
        return ((bonus > 0).to(torch.int64) << 40) - logical

    def _pace_renumber_row(self, layer_idx, seq_id, key, resident, cos_sin) -> None:
        # Row alignment: column i of the physical row (buffer_req_to_token_slots[row, i])
        # is entry i of every PACE row vector. H2O keeps its score vector aligned the
        # same way (_require_score_length); PACE vectors are appended with the row
        # (_on_prefill_score_appended, _pace_sync) and gathered with the same keep
        # indices whenever the row is compacted (_store_kept_score). Verify the lengths
        # before trusting the correspondence.
        row = self.seq_id_to_row[layer_idx][seq_id]
        physical_len = int(self.row_seq_lens[layer_idx][row])
        if physical_len != int(resident):
            raise RuntimeError(
                "PACE renumber: row state is not aligned with the physical row: "
                f"layer={layer_idx} seq_id={seq_id} recorded={resident} row={physical_len}."
            )
        logical = self._pace_rows["logical"][key]
        old = self._pace_rows["physical"][key]
        order = torch.argsort(logical, stable=True)
        new = torch.empty_like(old)
        new[order] = torch.arange(resident, dtype=old.dtype, device=old.device)
        slots = self.buffer_req_to_token_slots[layer_idx][row, :resident].long()
        k_cache, _ = self.get_layer_kv_cache(layer_idx)
        store = self._pace_stores.get(self.kv_layer_index(layer_idx))
        if store is not None:
            entries, overflow = store.renumber(
                k_cache, slots, self._pace_rows["ref"][key], old, new, cos_sin,
                priority=self._pace_reference_priority(key),
            )
            self._pace_rows["ref"][key] = entries
            self._pace_counters["reference_overflow"] += overflow
            if overflow:
                log_once(
                    "PACE reference store is full; some keys are re-rotated in place and "
                    "accumulate rounding error. The store is shared by all sequences on a "
                    "layer: raise pace_reference_tokens to concurrent bounded sessions x "
                    "resident tokens.",
                    level="WARNING",
                )
        else:
            log_once(
                "PACE bounded positions run without a reference store "
                "(pace_reference_tokens=0): keys are re-rotated in place and rounding error "
                "builds up with every renumber.",
                level="WARNING",
            )
            k_cache.index_copy_(
                0, slots, rotate_keys(k_cache.index_select(0, slots), cos_sin, old, new),
            )
        self._pace_rows["physical"][key] = new
        self._pace_counters["rerotated_keys"] += int(resident)

    def evict_after_intermediate_prefill(self, seqs: list[Sequence]) -> None:
        super().evict_after_intermediate_prefill(seqs)
        self._pace_maybe_renumber(s.seq_id for s in seqs if not s.is_last_chunk_prefill)

    def compact_final_prefill_for_decode(self, seqs: list[Sequence]) -> None:
        super().compact_final_prefill_for_decode(seqs)
        self._pace_maybe_renumber(s.seq_id for s in seqs if s.is_last_chunk_prefill)

    def evict_after_prefill(self, seqs: list[Sequence]) -> None:
        super().evict_after_prefill(seqs)
        self._pace_maybe_renumber(s.seq_id for s in seqs)

    def evict_after_decode(self, seqs: list[Sequence]):
        super().evict_after_decode(seqs)
        if seqs:
            self._pace_maybe_renumber(
                [s.seq_id for s in seqs] + sorted(self._h2o_active_decode_seq_ids)
            )

    # ------------------------------------------------------------------
    # Turn boundary: re-base resident bonuses on the new plan
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _pace_rebase(self, seq: Sequence) -> None:
        """``score += bonus(plan, logical) - bonus already folded in`` for resident tokens."""
        sid = int(seq.seq_id)
        compiled = self._pace_compiled(seq)
        for layer_idx in self.kv_transformer_layer_indices():
            if self.seq_id_to_row[layer_idx].get(sid) is None:
                continue
            self._pace_sync(layer_idx, sid)
            key = self._score_key(layer_idx, sid)
            score = self._h2o_scores.get(key)
            logical = self._pace_rows["logical"].get(key)
            if score is None or logical is None:
                raise RuntimeError(
                    f"PACE re-basing found no state for a resident row: "
                    f"layer={layer_idx} seq_id={sid}."
                )
            if int(score.numel()) != int(logical.numel()):
                raise RuntimeError(
                    "PACE re-basing: score and row state are misaligned: "
                    f"layer={layer_idx} seq_id={sid} scores={int(score.numel())} "
                    f"row={int(logical.numel())}."
                )
            old = self._pace_rows["bonus"][key]
            new = (
                torch.zeros_like(old)
                if compiled is None
                else compiled.to(logical.device).lookup(logical)
            )
            score.add_((new - old).to(device=score.device, dtype=score.dtype))
            self._pace_rows["bonus"][key] = new
        self._pace_counters["rebased_turns"] += 1

    # ------------------------------------------------------------------
    # Chain lifecycle
    # ------------------------------------------------------------------
    def on_chain_turn_finished(self, seq_id: int, processed_token_count: int) -> None:
        sid = int(seq_id)
        self._pace_sync_seq(sid)
        for layer_idx in self.kv_transformer_layer_indices():
            if self.seq_id_to_row[layer_idx].get(sid) is None:
                continue
            end = self._pace_end.get(self._score_key(layer_idx, sid), 0)
            if end != int(processed_token_count):
                raise RuntimeError(
                    "PACE chain turn bookkeeping: recorded logical end "
                    f"{end} != processed_token_count={int(processed_token_count)} "
                    f"(layer={layer_idx} seq_id={sid})."
                )
        super().on_chain_turn_finished(seq_id, processed_token_count)

    def snapshot_chain_method_state(self, seq_id: int):
        sid = int(seq_id)
        state = super().snapshot_chain_method_state(sid)
        ends = {}
        for layer_idx in self.kv_transformer_layer_indices():
            self._pace_sync(layer_idx, sid)
            key = self._score_key(layer_idx, sid)
            for name in ("logical", "bonus", "physical"):
                state.tensors[f"pace_{name}/{layer_idx}"] = self._pace_rows[name][key]
            ends[int(layer_idx)] = int(self._pace_end[key])
        # Reference copies are not snapshotted: a restored key is re-captured at
        # its next renumber, costing at most one extra rounding per offload. The
        # parent's metadata is nested, not replaced, and handed back on restore.
        # Chain snapshots live in the running engine's pinned host memory
        # (ChainOffloadController) and never outlive the process; nothing in this
        # method writes them elsewhere. The version guards any future persistence.
        state.metadata = {
            "pace": {"version": _SNAPSHOT_VERSION, "end": ends,
                     "shift": self._pace_shift.get(sid, 0)},
            "parent": state.metadata,
        }
        return state

    def restore_chain_method_state(self, seq_id: int, state) -> None:
        sid = int(seq_id)
        metadata = state.metadata
        if not isinstance(metadata, dict) or "pace" not in metadata:
            raise RuntimeError("PACE chain snapshot lacks row-state metadata.")
        version = metadata["pace"].get("version")
        if version != _SNAPSHOT_VERSION:
            raise RuntimeError(
                f"PACE chain snapshot format {version!r} is not {_SNAPSHOT_VERSION}; "
                "it was written by a different version of this method."
            )
        super().restore_chain_method_state(sid, replace(state, metadata=metadata["parent"]))
        ends = metadata["pace"]["end"]
        for layer_idx in self.kv_transformer_layer_indices():
            key = self._score_key(layer_idx, sid)
            store = self._pace_stores.get(self.kv_layer_index(layer_idx))
            previous = self._pace_rows["ref"].get(key)
            if store is not None and previous is not None:
                store.release(previous)
            for name in ("logical", "bonus", "physical"):
                self._pace_rows[name][key] = state.tensors[f"pace_{name}/{layer_idx}"].to(
                    self.device, non_blocking=True,
                )
            self._pace_rows["ref"][key] = torch.full_like(self._pace_rows["logical"][key], -1)
            self._pace_end[key] = int(ends[int(layer_idx)])
        self._pace_shift[sid] = int(metadata["pace"]["shift"])

    def free_seq(self, seq_id: int):
        sid = int(seq_id)
        for key in [k for k in self._pace_rows["ref"] if k[1] == sid]:
            store = self._pace_stores.get(self.kv_layer_index(key[0]))
            if store is not None:
                store.release(self._pace_rows["ref"][key])
        for rows in (*self._pace_rows.values(), self._pace_end):
            for key in [k for k in rows if k[1] == sid]:
                rows.pop(key, None)
        for per_seq in (self._pace_plans, self._pace_chunk_bonus, self._pace_shift):
            per_seq.pop(sid, None)
        super().free_seq(seq_id)

    def reset_after_warmup(self) -> None:
        self._pace_init_state()
        super().reset_after_warmup()

    def debug_state_summary(self) -> dict[str, object]:
        summary = super().debug_state_summary()
        summary["pace"] = {
            "counters": dict(self._pace_counters),
            "planned_seqs": sorted(self._pace_plans),
            "shifts": dict(sorted(self._pace_shift.items())),
        }
        return summary
