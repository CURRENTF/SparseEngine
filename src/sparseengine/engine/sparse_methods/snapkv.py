from __future__ import annotations

import torch
import torch.nn.functional as F

from sparseengine.engine.sequence import Sequence
from sparseengine.utils.log import log_level, logger
from sparseengine.utils.profiler import profiler

from .base import AttentionEndEvent, LayerBatchSparseState, SparseStepContext
from .passthrough import PassThroughRuntime


class ScoredCompactionRuntime(PassThroughRuntime):
    can_compact_decode_layers = False

    def __init__(self, config, cache_manager):
        super().__init__(config, cache_manager)
        self._snapkv_decode_reduced_attn_score_buffers: dict[
            int,
            torch.Tensor,
        ] = {}

    def clear_decode_attn_score_buffers(self) -> None:
        super().clear_decode_attn_score_buffers()
        self._snapkv_decode_reduced_attn_score_buffers.clear()

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        return list(self._snapkv_decode_reduced_attn_score_buffers.values())

    def _prepare_decode_attention_score(
        self,
        layer_idx: int,
        state: LayerBatchSparseState,
        batch_size: int,
        num_heads: int,
        max_len: int,
    ) -> None:
        del num_heads, max_len
        score_width = self._snapkv_decode_score_width(state)
        state.attn_score = self._get_snapkv_decode_score_buffer(
            layer_idx,
            batch_size,
            score_width,
            fill_value=-1e20,
        )

    def _get_snapkv_decode_score_buffer(
        self,
        layer_idx: int,
        batch_size: int,
        max_len: int,
        *,
        fill_value: float,
    ) -> torch.Tensor:
        if min(batch_size, max_len) <= 0:
            raise RuntimeError(
                "SnapKV decode score buffer requires positive dimensions: "
                f"layer={layer_idx} shape={(batch_size, max_len)}."
            )
        reduced = self._snapkv_decode_reduced_attn_score_buffers.get(
            int(layer_idx)
        )
        if (
            reduced is None
            or reduced.dtype != self.snapkv_decode_score_dtype
            or reduced.device != self.device
            or int(reduced.shape[0]) < batch_size
            or int(reduced.shape[1]) < max_len
        ):
            reduced = torch.empty(
                (batch_size, max_len),
                dtype=self.snapkv_decode_score_dtype,
                device=self.device,
            )
            self._snapkv_decode_reduced_attn_score_buffers[int(layer_idx)] = reduced
        view = reduced[:batch_size, :max_len]
        view.fill_(fill_value)
        return view

    def _snapkv_decode_score_width(self, state: LayerBatchSparseState) -> int:
        max_len = self._state_max_context_len(state)
        if not bool(getattr(self.config, "decode_graph", False)):
            return max_len
        graph_capacity = getattr(
            self.cache_manager,
            "_decode_static_max_context_len",
            None,
        )
        if graph_capacity is None or int(graph_capacity) < max_len:
            raise RuntimeError(
                "SnapKV decode CUDA graph requires a score capacity covering the "
                f"current context: graph_capacity={graph_capacity} current={max_len}."
            )
        return int(graph_capacity)

    def on_attention_end(self, event: AttentionEndEvent) -> None:
        layer_idx = event.layer_idx
        if not self._is_kv_layer(layer_idx) or event.forward_context.is_prefill:
            return
        attn_score = self.layer_batch_sparse_states[layer_idx].attn_score
        if attn_score is not None and attn_score.dim() != 2:
            raise RuntimeError(
                "SnapKV-family decode attention must write a fused head-reduced "
                f"[B, L] score tensor: layer={layer_idx} "
                f"shape={tuple(attn_score.shape)}."
            )

    def finish_step(self, step: SparseStepContext) -> None:
        if step.is_prefill and any(
            seq.is_last_chunk_prefill for seq in step.seqs
        ):
            self._snapkv_prefill_eviction(step.seqs)

    @torch.no_grad()
    def _snapkv_prefill_eviction(self, seqs: list[Sequence]):
        final_seqs = [seq for seq in seqs if bool(seq.is_last_chunk_prefill)]
        if not final_seqs:
            return

        free_batch = getattr(
            self.cache_manager,
            "free_part_slots_batch",
            None,
        )
        free_layers = getattr(
            self.cache_manager,
            "free_part_slots_batch_layers",
            None,
        )
        pool_kernel_size = int(
            getattr(self.config, "pool_kernel_size", 1) or 1
        )
        pending_layer_compactions: dict[
            tuple[tuple[int, ...], int, int, int, int],
            list[tuple[int, list[Sequence], torch.Tensor]],
        ] = {}

        for layer_idx in range(self.num_layers):
            if not self._is_kv_layer(layer_idx):
                continue
            budget = self._get_layer_budget(layer_idx, is_prefill=True)
            if budget is None:
                continue
            compatible_rows: dict[
                tuple[int, int, int],
                list[tuple[Sequence, torch.Tensor]],
            ] = {}
            for seq in final_seqs:
                physical_len = getattr(
                    self.cache_manager,
                    "chain_physical_kv_len",
                    None,
                )
                kv_len = (
                    int(physical_len(layer_idx, seq.seq_id))
                    if getattr(seq, "chain_status", "") == "resumed"
                    and not bool(getattr(seq, "is_recompute_replay", False))
                    and callable(physical_len)
                    else int(seq.num_prefilled_tokens) + int(seq.current_chunk_size)
                )
                if kv_len <= budget:
                    continue
                seq_scores = self.cache_manager.pop_prefill_attention_score(
                    layer_idx,
                    seq,
                )
                if seq_scores is None:
                    raise RuntimeError(
                        "SnapKV/PyramidKV prefill eviction requires prefill "
                        f"attention scores. method={self.sparse_method} "
                        f"layer={layer_idx} seq_id={seq.seq_id}"
                    )
                if seq_scores.dim() == 2:
                    seq_scores = seq_scores.max(dim=0).values
                if log_level == "DEBUG":
                    logger.debug(
                        "[SnapKV] prefill eviction: "
                        f"layer={layer_idx} seq_id={seq.seq_id} "
                        f"kv_len={kv_len} budget={budget}"
                    )
                compatibility_key = (
                    int(kv_len),
                    int(budget),
                    int(pool_kernel_size),
                )
                compatible_rows.setdefault(compatibility_key, []).append(
                    (seq, seq_scores[:kv_len])
                )

            for (kv_len, group_budget, group_pool_kernel_size), rows in (
                compatible_rows.items()
            ):
                group_seqs = [seq for seq, _scores in rows]
                if len(rows) == 1:
                    _seq, seq_scores = rows[0]
                    with profiler.record("snapkv_prefill_select"):
                        keep_indices = self._snapkv_select_indices(
                            seq_scores,
                            kv_len,
                            group_budget,
                            pool_kernel_size=group_pool_kernel_size,
                        )
                    keep_indices = keep_indices.unsqueeze(0)
                else:
                    with profiler.record("snapkv_prefill_select_batch"):
                        scores = torch.stack(
                            [seq_scores for _seq, seq_scores in rows],
                            dim=0,
                        )
                        keep_indices = self._snapkv_select_indices_batch(
                            scores,
                            kv_len,
                            group_budget,
                            pool_kernel_size=group_pool_kernel_size,
                        )

                if free_layers is None:
                    if len(group_seqs) > 1 and free_batch is not None:
                        with profiler.record("snapkv_prefill_compact_batch"):
                            free_batch(layer_idx, group_seqs, keep_indices)
                    else:
                        with profiler.record("snapkv_prefill_compact"):
                            for seq_idx, seq in enumerate(group_seqs):
                                self.cache_manager.free_part_slots(
                                    layer_idx,
                                    seq,
                                    keep_indices[seq_idx],
                                )
                    continue

                key = (
                    tuple(int(seq.seq_id) for seq in group_seqs),
                    int(kv_len),
                    int(group_budget),
                    int(group_pool_kernel_size),
                    int(keep_indices.shape[1]),
                )
                pending_layer_compactions.setdefault(key, []).append(
                    (int(layer_idx), group_seqs, keep_indices)
                )

        for entries in pending_layer_compactions.values():
            if len(entries) == 1:
                layer_idx, group_seqs, keep_indices = entries[0]
                if len(group_seqs) > 1 and free_batch is not None:
                    with profiler.record("snapkv_prefill_compact_batch"):
                        free_batch(layer_idx, group_seqs, keep_indices)
                else:
                    with profiler.record("snapkv_prefill_compact"):
                        for seq_idx, seq in enumerate(group_seqs):
                            self.cache_manager.free_part_slots(
                                layer_idx,
                                seq,
                                keep_indices[seq_idx],
                            )
                continue
            layer_indices = [entry[0] for entry in entries]
            group_seqs = entries[0][1]
            keep_indices = torch.stack(
                [entry[2] for entry in entries],
                dim=0,
            )
            with profiler.record("snapkv_prefill_compact_layers"):
                free_layers(layer_indices, group_seqs, keep_indices)

    @torch.no_grad()
    def _snapkv_decode_eviction(self, seqs: list[Sequence]):
        with profiler.record("snapkv_decode_eviction"):
            pending_compactions: dict[
                tuple[tuple[int, ...], tuple[int, ...]],
                list[tuple[int, list[Sequence], torch.Tensor]],
            ] = {}
            can_compact_layers = (
                self.can_compact_decode_layers
                and hasattr(
                    self.cache_manager,
                    "free_part_slots_batch_layers",
                )
            )

            for layer_idx in range(self.num_layers):
                if not self._is_kv_layer(layer_idx):
                    continue
                state = self.layer_batch_sparse_states[layer_idx]
                attn_scores = state.attn_score
                if attn_scores is None:
                    continue

                budget = self._get_layer_budget(layer_idx, is_prefill=False)
                if budget is None:
                    continue

                trigger_len = self._snapkv_decode_trigger_len(budget)
                max_context_len = state.max_context_len
                if max_context_len is not None and (
                    int(max_context_len) <= int(budget)
                    or int(max_context_len) < int(trigger_len)
                ):
                    continue

                kv_len_fn = getattr(
                    self.cache_manager,
                    "decode_kv_lens_for_layer",
                    None,
                )
                if kv_len_fn is not None:
                    kv_lens = kv_len_fn(layer_idx, seqs)
                else:
                    kv_lens = [
                        int(state.context_lens[batch_idx])
                        for batch_idx in range(len(seqs))
                    ]
                triggered: list[tuple[int, Sequence, int]] = []
                for batch_idx, (seq, kv_len) in enumerate(zip(seqs, kv_lens)):
                    if kv_len <= budget or kv_len < trigger_len:
                        continue
                    triggered.append((batch_idx, seq, kv_len))

                if not triggered:
                    continue
                if attn_scores.dim() != 2:
                    raise RuntimeError(
                        "SnapKV/PyramidKV post-forward eviction requires "
                        "head-reduced [B, L] scores: "
                        f"layer={layer_idx} shape={tuple(attn_scores.shape)}."
                    )
                if (
                    attn_scores.dtype != self.snapkv_decode_score_dtype
                    or attn_scores.device != self.device
                ):
                    raise RuntimeError(
                        "SnapKV/PyramidKV post-forward score dtype/device mismatch: "
                        f"layer={layer_idx} got={attn_scores.dtype}/"
                        f"{attn_scores.device} expected={self.snapkv_decode_score_dtype}/"
                        f"{self.device}."
                    )

                by_kv_len: dict[int, list[tuple[int, Sequence]]] = {}
                for batch_idx, seq, kv_len in triggered:
                    by_kv_len.setdefault(int(kv_len), []).append((batch_idx, seq))

                for kv_len, group in by_kv_len.items():
                    if log_level == "DEBUG":
                        for _batch_idx, seq in group:
                            logger.debug(
                                "[SnapKV] decode eviction: "
                                f"layer={layer_idx} seq_id={seq.seq_id} "
                                f"kv_len={kv_len} budget={budget} "
                                f"trigger_len={trigger_len}"
                            )
                    if len(group) == 1:
                        batch_idx, seq = group[0]
                        with profiler.record("snapkv_decode_select"):
                            keep_indices = self._snapkv_select_indices(
                                attn_scores[batch_idx, :kv_len],
                                kv_len,
                                budget,
                            )
                        with profiler.record("snapkv_decode_compact"):
                            self.cache_manager.free_part_slots(
                                layer_idx,
                                seq,
                                keep_indices,
                            )
                        continue

                    batch_indices = torch.tensor(
                        [batch_idx for batch_idx, _seq in group],
                        dtype=torch.long,
                        device=attn_scores.device,
                    )
                    with profiler.record("snapkv_decode_select"):
                        keep_indices = self._snapkv_select_indices_batch(
                            attn_scores.index_select(0, batch_indices)[:, :kv_len],
                            kv_len,
                            budget,
                        )
                    free_batch = getattr(
                        self.cache_manager,
                        "free_part_slots_batch",
                        None,
                    )
                    group_seqs = [seq for _batch_idx, seq in group]
                    if can_compact_layers:
                        key = (
                            tuple(int(seq.seq_id) for seq in group_seqs),
                            tuple(int(dim) for dim in keep_indices.shape),
                        )
                        pending_compactions.setdefault(key, []).append(
                            (layer_idx, group_seqs, keep_indices)
                        )
                    else:
                        with profiler.record("snapkv_decode_compact"):
                            if free_batch is None:
                                for row_idx, (_batch_idx, seq) in enumerate(group):
                                    self.cache_manager.free_part_slots(
                                        layer_idx,
                                        seq,
                                        keep_indices[row_idx],
                                    )
                            else:
                                free_batch(layer_idx, group_seqs, keep_indices)

            if pending_compactions:
                free_layers = getattr(
                    self.cache_manager,
                    "free_part_slots_batch_layers",
                )
                for entries in pending_compactions.values():
                    if len(entries) == 1:
                        layer_idx, group_seqs, keep_indices = entries[0]
                        with profiler.record("snapkv_decode_compact"):
                            self.cache_manager.free_part_slots_batch(
                                layer_idx,
                                group_seqs,
                                keep_indices,
                            )
                        continue
                    layer_indices = [entry[0] for entry in entries]
                    group_seqs = entries[0][1]
                    keep_indices = torch.stack(
                        [entry[2] for entry in entries],
                        dim=0,
                    )
                    with profiler.record("snapkv_decode_compact_layers"):
                        free_layers(layer_indices, group_seqs, keep_indices)

    def _snapkv_select_indices(
        self,
        scores: torch.Tensor,
        kv_len: int,
        budget: int,
        *,
        pool_kernel_size: int = 1,
    ) -> torch.Tensor:
        assert kv_len > budget
        sink_indices = torch.arange(self.num_sink, device=scores.device)
        recent_start = kv_len - self.num_recent
        recent_indices = torch.arange(
            recent_start,
            kv_len,
            device=scores.device,
        )
        num_topk = budget - self.num_sink - self.num_recent
        if num_topk > 0 and recent_start > self.num_sink:
            middle_scores = scores[self.num_sink:recent_start]
            pool_kernel_size = int(pool_kernel_size)
            if pool_kernel_size > 1:
                middle_scores = F.max_pool1d(
                    middle_scores[None, None, :],
                    kernel_size=pool_kernel_size,
                    padding=pool_kernel_size // 2,
                    stride=1,
                ).squeeze(0).squeeze(0)
            topk_indices_relative = middle_scores.topk(
                min(num_topk, middle_scores.shape[0]),
                dim=-1,
            ).indices
            topk_indices = topk_indices_relative + self.num_sink
            return torch.cat([sink_indices, topk_indices, recent_indices])
        return torch.cat([sink_indices, recent_indices])

    def _snapkv_select_indices_batch(
        self,
        scores: torch.Tensor,
        kv_len: int,
        budget: int,
        *,
        pool_kernel_size: int = 1,
    ) -> torch.Tensor:
        if scores.dim() != 2:
            raise ValueError(
                "Expected batched SnapKV scores with shape [B, L], got "
                f"{tuple(scores.shape)}."
            )
        assert kv_len > budget
        if int(scores.shape[1]) < int(kv_len):
            raise ValueError(
                "SnapKV batched scores are shorter than kv_len: "
                f"scores={tuple(scores.shape)} kv_len={kv_len}."
            )
        batch_size = int(scores.shape[0])
        sink_indices = torch.arange(
            self.num_sink,
            device=scores.device,
        ).expand(batch_size, -1)
        recent_start = kv_len - self.num_recent
        recent_indices = torch.arange(
            recent_start,
            kv_len,
            device=scores.device,
        ).expand(batch_size, -1)
        num_topk = budget - self.num_sink - self.num_recent
        if num_topk > 0 and recent_start > self.num_sink:
            middle_scores = scores[:, self.num_sink:recent_start]
            pool_kernel_size = int(pool_kernel_size)
            if pool_kernel_size > 1:
                middle_scores = F.max_pool1d(
                    middle_scores[:, None, :],
                    kernel_size=pool_kernel_size,
                    padding=pool_kernel_size // 2,
                    stride=1,
                ).squeeze(1)
            topk_indices_relative = middle_scores.topk(
                min(num_topk, middle_scores.shape[1]),
                dim=-1,
            ).indices
            topk_indices = topk_indices_relative + self.num_sink
            return torch.cat(
                [sink_indices, topk_indices, recent_indices],
                dim=1,
            )
        return torch.cat([sink_indices, recent_indices], dim=1)

    def _get_layer_budget(
        self,
        layer_idx: int,
        is_prefill: bool,
    ) -> int | None:
        del is_prefill
        kv_layer_idx = self._kv_layer_index(layer_idx)
        if kv_layer_idx < self.config.snapkv_num_full_layers:
            return None
        return self._sparse_layer_budget(kv_layer_idx)

    def _sparse_layer_budget(self, kv_layer_idx: int) -> int:
        del kv_layer_idx
        return self.num_sink + self.decode_keep_tokens + self.num_recent

    def _snapkv_decode_trigger_len(self, budget: int) -> int:
        top_budget = int(budget) - int(self.num_sink) - int(self.num_recent)
        return int(2.0 * top_budget)


class SnapKVRuntime(ScoredCompactionRuntime):
    can_compact_decode_layers = True

    def needs_attention_score(
        self,
        layer_idx: int,
        step: SparseStepContext,
    ) -> bool:
        del layer_idx, step
        return False


class PyramidKVRuntime(ScoredCompactionRuntime):
    def needs_attention_score(
        self,
        layer_idx: int,
        step: SparseStepContext,
    ) -> bool:
        if step.is_prefill:
            return False
        budget = self._get_layer_budget(layer_idx, is_prefill=False)
        if budget is None:
            return False
        trigger_len = self._snapkv_decode_trigger_len(budget)
        if bool(getattr(self.config, "decode_graph", False)):
            state = self.layer_batch_sparse_states[layer_idx]
            graph_capacity = self._snapkv_decode_score_width(state)
            return (
                int(graph_capacity) >= int(trigger_len)
                and int(graph_capacity) > int(budget)
            )

        kv_lens_fn = getattr(
            self.cache_manager,
            "decode_kv_lens_for_layer",
            None,
        )
        if kv_lens_fn is not None:
            kv_lens = kv_lens_fn(layer_idx, step.seqs)
            return any(
                int(kv_len) >= int(trigger_len)
                and int(kv_len) > int(budget)
                for kv_len in kv_lens
            )

        state = self.layer_batch_sparse_states[layer_idx]
        if state.context_lens is None:
            return False
        if state.max_context_len is not None:
            return (
                int(state.max_context_len) >= int(trigger_len)
                and int(state.max_context_len) > int(budget)
            )
        return bool(
            (
                (state.context_lens >= trigger_len)
                & (state.context_lens > budget)
            ).any()
        )

    def on_attention_end(self, event: AttentionEndEvent) -> None:
        super().on_attention_end(event)
        layer_idx = event.layer_idx
        context = event.forward_context
        if not self._is_kv_layer(layer_idx) or not context.is_prefill:
            return
        if not self.cache_manager.has_prefill_staging_view(layer_idx):
            return

        with profiler.record("pyramidkv_staging_materialize_layer"):
            budget = self._get_layer_budget(layer_idx, is_prefill=True)
            seqs = getattr(context, "seqs", None)
            if seqs is None:
                raise RuntimeError(
                    "PyramidKV full-prefill staging requires current seqs in context."
                )
            if any(not seq.is_last_chunk_prefill for seq in seqs):
                if any(
                    getattr(
                        self.cache_manager,
                        "requires_long_prefill_offload",
                        lambda _seq: False,
                    )(seq)
                    for seq in seqs
                ):
                    return
                raise RuntimeError(
                    "PyramidKV full-prefill staging should only run on the final "
                    "prefill chunk."
                )
            staging_context_lens = (
                self.cache_manager.prefill_staging_context_lens_cpu(layer_idx)
            )
            if (
                staging_context_lens is None
                or len(staging_context_lens) != len(seqs)
            ):
                raise RuntimeError(
                    "PyramidKV staging CPU context lengths do not match the "
                    f"current batch: layer={layer_idx} "
                    f"lengths={staging_context_lens} batch_size={len(seqs)}."
                )
            seq_keep_indices = []
            for batch_idx, seq in enumerate(seqs):
                kv_len = int(staging_context_lens[batch_idx])
                if budget is None or kv_len <= budget:
                    keep_indices = torch.arange(
                        kv_len,
                        device=self.device,
                        dtype=torch.long,
                    )
                else:
                    attn_scores = self.cache_manager.pop_prefill_attention_score(
                        layer_idx,
                        seq,
                    )
                    if attn_scores is None:
                        raise RuntimeError(
                            "PyramidKV full-prefill staging requires prefill "
                            f"attention scores. layer={layer_idx} seq_id={seq.seq_id}"
                        )
                    if attn_scores.dim() == 2:
                        attn_scores = attn_scores.max(dim=0).values
                    keep_indices = self._snapkv_select_indices(
                        attn_scores[:kv_len],
                        kv_len,
                        budget,
                        pool_kernel_size=int(
                            getattr(self.config, "pool_kernel_size", 1) or 1
                        ),
                    )
                seq_keep_indices.append((seq, keep_indices))
            self.cache_manager.materialize_prefill_staging_layer_batch(
                layer_idx,
                seq_keep_indices,
            )

    def finish_step(self, step: SparseStepContext) -> None:
        if step.is_prefill:
            if not any(seq.is_last_chunk_prefill for seq in step.seqs):
                return
            if getattr(
                self.cache_manager,
                "prefill_staging_was_active",
                lambda: False,
            )():
                return
            self._snapkv_prefill_eviction(step.seqs)
            return
        if not any(
            state.attn_score is not None
            for state in self.layer_batch_sparse_states.values()
        ):
            return
        self._snapkv_decode_eviction(step.seqs)

    def _sparse_layer_budget(self, kv_layer_idx: int) -> int:
        ratio = self.config.pyramid_layer_ratios[kv_layer_idx]
        base_ratio = self.config.pyramid_layer_ratios[0]
        scaled_top_tokens = int(self.decode_keep_tokens * ratio / base_ratio)
        return self.num_sink + scaled_top_tokens + self.num_recent

    def _snapkv_decode_trigger_len(self, budget: int) -> int:
        top_budget = int(budget) - int(self.num_sink) - int(self.num_recent)
        return int(budget) + int(top_budget)
