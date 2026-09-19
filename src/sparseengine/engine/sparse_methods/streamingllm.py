from __future__ import annotations

import torch

from sparseengine.engine.sequence import Sequence
from sparseengine.utils.log import log_level, logger
from sparseengine.utils.profiler import profiler

from .base import SparseStepContext
from .passthrough import PassThroughRuntime


class StreamingLLMRuntime(PassThroughRuntime):
    def finish_step(self, step: SparseStepContext) -> None:
        if step.is_prefill:
            if any(seq.is_last_chunk_prefill for seq in step.seqs):
                self._streamingllm_prefill_eviction(step.seqs)
            return
        self._streamingllm_decode_eviction(step.seqs)

    @torch.no_grad()
    def _streamingllm_prefill_eviction(self, seqs: list[Sequence]):
        budget = self._get_streamingllm_budget()
        if budget is None:
            return

        with profiler.record("streamingllm_prefill_eviction"):
            free_prefix_recent = getattr(
                self.cache_manager,
                "free_prefix_recent_slots_batch_layers",
                None,
            )
            free_layers = getattr(
                self.cache_manager,
                "free_part_slots_batch_layers",
                None,
            )
            free_batch = getattr(
                self.cache_manager,
                "free_part_slots_batch",
                None,
            )
            pending_prefix_recent: dict[
                tuple[tuple[int, ...], int],
                list[tuple[int, list[Sequence]]],
            ] = {}
            pending_compactions: dict[
                tuple[tuple[int, ...], int],
                list[tuple[int, list[Sequence], torch.Tensor]],
            ] = {}

            for layer_idx in range(self.num_layers):
                if not self._is_kv_layer(layer_idx):
                    continue
                state = self.layer_batch_sparse_states[layer_idx]
                triggered: list[tuple[int, Sequence, int]] = []
                for batch_idx, seq in enumerate(seqs):
                    if not seq.is_last_chunk_prefill:
                        continue
                    kv_len = int(state.context_lens[batch_idx])
                    if kv_len <= budget:
                        continue
                    triggered.append((batch_idx, seq, kv_len))

                if not triggered:
                    continue

                by_kv_len: dict[int, list[tuple[int, Sequence]]] = {}
                for batch_idx, seq, kv_len in triggered:
                    by_kv_len.setdefault(int(kv_len), []).append((batch_idx, seq))

                for kv_len, group in by_kv_len.items():
                    if log_level == "DEBUG":
                        for _batch_idx, seq in group:
                            logger.debug(
                                "[StreamingLLM] prefill eviction: "
                                f"layer={layer_idx} seq_id={seq.seq_id} "
                                f"kv_len={kv_len} budget={budget}"
                            )
                    group_seqs = [seq for _batch_idx, seq in group]
                    if free_prefix_recent is not None:
                        key = (
                            tuple(int(seq.seq_id) for seq in group_seqs),
                            int(kv_len),
                        )
                        pending_prefix_recent.setdefault(key, []).append(
                            (layer_idx, group_seqs)
                        )
                        continue
                    keep_indices = self._streamingllm_select_indices(kv_len).expand(
                        len(group),
                        -1,
                    )
                    if free_layers is not None:
                        key = (
                            tuple(int(seq.seq_id) for seq in group_seqs),
                            int(kv_len),
                        )
                        pending_compactions.setdefault(key, []).append(
                            (layer_idx, group_seqs, keep_indices)
                        )
                    elif free_batch is not None:
                        free_batch(layer_idx, group_seqs, keep_indices)
                    else:
                        for row_idx, (_batch_idx, seq) in enumerate(group):
                            self.cache_manager.free_part_slots(
                                layer_idx,
                                seq,
                                keep_indices[row_idx],
                            )

            if pending_prefix_recent:
                for (_seq_ids, kv_len), entries in pending_prefix_recent.items():
                    layer_indices = [
                        layer_idx for layer_idx, _group_seqs in entries
                    ]
                    group_seqs = entries[0][1]
                    free_prefix_recent(
                        layer_indices,
                        group_seqs,
                        kv_len=kv_len,
                        sink_keep_tokens=self.num_sink,
                        recent_keep_tokens=self.num_recent,
                    )
            if pending_compactions:
                for entries in pending_compactions.values():
                    if len(entries) == 1:
                        layer_idx, group_seqs, keep_indices = entries[0]
                        if free_batch is not None:
                            free_batch(layer_idx, group_seqs, keep_indices)
                        else:
                            for row_idx, seq in enumerate(group_seqs):
                                self.cache_manager.free_part_slots(
                                    layer_idx,
                                    seq,
                                    keep_indices[row_idx],
                                )
                        continue
                    layer_indices = [
                        layer_idx
                        for layer_idx, _group_seqs, _keep_indices in entries
                    ]
                    group_seqs = entries[0][1]
                    keep_indices = torch.stack(
                        [entry[2] for entry in entries],
                        dim=0,
                    )
                    free_layers(layer_indices, group_seqs, keep_indices)

    @torch.no_grad()
    def _streamingllm_decode_eviction(self, seqs: list[Sequence]):
        budget = self._get_streamingllm_budget()
        if budget is None:
            return
        trigger_len = int(2.0 * budget)

        with profiler.record("streamingllm_decode_eviction"):
            free_prefix_recent = getattr(
                self.cache_manager,
                "free_prefix_recent_slots_batch_layers",
                None,
            )
            free_layers = getattr(
                self.cache_manager,
                "free_part_slots_batch_layers",
                None,
            )
            free_batch = getattr(
                self.cache_manager,
                "free_part_slots_batch",
                None,
            )
            pending_prefix_recent: dict[
                tuple[tuple[int, ...], int],
                list[tuple[int, list[Sequence]]],
            ] = {}
            pending_compactions: dict[
                tuple[tuple[int, ...], int],
                list[tuple[int, list[Sequence], torch.Tensor]],
            ] = {}

            for layer_idx in range(self.num_layers):
                if not self._is_kv_layer(layer_idx):
                    continue
                state = self.layer_batch_sparse_states[layer_idx]
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
                    triggered.append((batch_idx, seq, int(kv_len)))

                if not triggered:
                    continue

                by_kv_len: dict[int, list[tuple[int, Sequence]]] = {}
                for batch_idx, seq, kv_len in triggered:
                    by_kv_len.setdefault(int(kv_len), []).append((batch_idx, seq))

                for kv_len, group in by_kv_len.items():
                    if log_level == "DEBUG":
                        for _batch_idx, seq in group:
                            logger.debug(
                                "[StreamingLLM] decode eviction: "
                                f"layer={layer_idx} seq_id={seq.seq_id} "
                                f"kv_len={kv_len} budget={budget} "
                                f"trigger_len={trigger_len}"
                            )
                    group_seqs = [seq for _batch_idx, seq in group]
                    if free_prefix_recent is not None:
                        key = (
                            tuple(int(seq.seq_id) for seq in group_seqs),
                            int(kv_len),
                        )
                        pending_prefix_recent.setdefault(key, []).append(
                            (layer_idx, group_seqs)
                        )
                        continue
                    keep_indices = self._streamingllm_select_indices(kv_len).expand(
                        len(group),
                        -1,
                    )
                    if free_layers is not None:
                        key = (
                            tuple(int(seq.seq_id) for seq in group_seqs),
                            int(kv_len),
                        )
                        pending_compactions.setdefault(key, []).append(
                            (layer_idx, group_seqs, keep_indices)
                        )
                    elif free_batch is not None:
                        free_batch(layer_idx, group_seqs, keep_indices)
                    else:
                        for row_idx, (_batch_idx, seq) in enumerate(group):
                            self.cache_manager.free_part_slots(
                                layer_idx,
                                seq,
                                keep_indices[row_idx],
                            )

            if pending_prefix_recent:
                for (_seq_ids, kv_len), entries in pending_prefix_recent.items():
                    layer_indices = [
                        layer_idx for layer_idx, _group_seqs in entries
                    ]
                    group_seqs = entries[0][1]
                    free_prefix_recent(
                        layer_indices,
                        group_seqs,
                        kv_len=kv_len,
                        sink_keep_tokens=self.num_sink,
                        recent_keep_tokens=self.num_recent,
                    )
            if pending_compactions:
                for entries in pending_compactions.values():
                    if len(entries) == 1:
                        layer_idx, group_seqs, keep_indices = entries[0]
                        if free_batch is not None:
                            free_batch(layer_idx, group_seqs, keep_indices)
                        else:
                            for row_idx, seq in enumerate(group_seqs):
                                self.cache_manager.free_part_slots(
                                    layer_idx,
                                    seq,
                                    keep_indices[row_idx],
                                )
                        continue
                    layer_indices = [
                        layer_idx
                        for layer_idx, _group_seqs, _keep_indices in entries
                    ]
                    group_seqs = entries[0][1]
                    keep_indices = torch.stack(
                        [entry[2] for entry in entries],
                        dim=0,
                    )
                    free_layers(layer_indices, group_seqs, keep_indices)

    def _get_streamingllm_budget(self) -> int | None:
        budget = self.num_sink + self.num_recent
        if budget <= 0:
            return None
        return budget

    def _streamingllm_select_indices(self, kv_len: int) -> torch.Tensor:
        assert kv_len > 0
        sink_end = min(self.num_sink, kv_len)
        recent_start = max(sink_end, kv_len - self.num_recent)
        sink_indices = torch.arange(
            sink_end,
            device=self.device,
            dtype=torch.long,
        )
        recent_indices = torch.arange(
            recent_start,
            kv_len,
            device=self.device,
            dtype=torch.long,
        )
        return torch.cat([sink_indices, recent_indices], dim=0)
