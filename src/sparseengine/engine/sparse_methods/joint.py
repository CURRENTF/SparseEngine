from __future__ import annotations

import torch

from sparseengine.engine.sequence import Sequence
from sparseengine.utils.log import log_level, logger
from sparseengine.utils.profiler import profiler

from .base import SparseStepContext
from .passthrough import PassThroughRuntime


class JointDecodeRuntime(PassThroughRuntime):
    profiler_name: str
    select_fn_name: str
    interval_config_name: str

    def finish_step(self, step: SparseStepContext) -> None:
        if step.is_prefill:
            return
        self._joint_decode_eviction(step.seqs)

    def _get_joint_decode_budget(self) -> int | None:
        budget = (
            int(self.num_sink)
            + int(self.decode_keep_tokens)
            + int(self.num_recent)
        )
        if budget <= 0:
            return None
        return budget

    @torch.no_grad()
    def _joint_decode_eviction(self, seqs: list[Sequence]):
        budget = self._get_joint_decode_budget()
        if budget is None:
            return
        interval = int(getattr(self.config, self.interval_config_name))
        trigger_len = int(budget) + interval
        select_fn = getattr(self.cache_manager, self.select_fn_name, None)
        if select_fn is None:
            raise RuntimeError(
                f"Cache manager {type(self.cache_manager).__name__} does not "
                f"implement {self.select_fn_name}."
            )
        select_batch_fn = getattr(
            self.cache_manager,
            f"{self.select_fn_name}_batch",
            None,
        )
        free_batch = getattr(
            self.cache_manager,
            "free_part_slots_batch",
            None,
        )

        with profiler.record(self.profiler_name):
            kv_len_fn = getattr(
                self.cache_manager,
                "decode_kv_lens_for_layer",
                None,
            )
            for layer_idx in range(self.num_layers):
                if not self._is_kv_layer(layer_idx):
                    continue
                state = self.layer_batch_sparse_states[layer_idx]
                attn_scores = state.attn_score
                if attn_scores is None:
                    continue

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

                if attn_scores is not None and attn_scores.dim() == 3:
                    attn_scores = self._decode_softmax_token_scores(
                        attn_scores,
                        candidate_start=self.num_sink,
                        candidate_lens=(
                            state.context_lens - self.num_sink
                        ).clamp_min(0),
                    )

                batch_keep_indices = None
                triggered_seqs = [seq for _, seq, _ in triggered]
                triggered_kv_lens = [kv_len for _, _, kv_len in triggered]
                if (
                    select_batch_fn is not None
                    and len(
                        set(int(kv_len) for kv_len in triggered_kv_lens)
                    )
                    == 1
                ):
                    if attn_scores is not None:
                        batch_indices = torch.tensor(
                            [
                                batch_idx
                                for batch_idx, _seq, _kv_len in triggered
                            ],
                            dtype=torch.long,
                            device=attn_scores.device,
                        )
                        select_importance_scores = attn_scores.index_select(
                            0,
                            batch_indices,
                        )
                    else:
                        select_importance_scores = None
                else:
                    select_importance_scores = None
                if select_importance_scores is not None:
                    batch_keep_indices = select_batch_fn(
                        layer_idx,
                        triggered_seqs,
                        select_importance_scores,
                        triggered_kv_lens,
                        budget,
                    )
                keep_batch: list[torch.Tensor] = []
                seq_batch: list[Sequence] = []
                for local_trigger_idx, (batch_idx, seq, kv_len) in enumerate(
                    triggered
                ):
                    if log_level == "DEBUG":
                        logger.debug(
                            "[{}] decode eviction: layer={} seq_id={} "
                            "kv_len={} budget={} trigger_len={}",
                            self.sparse_method,
                            layer_idx,
                            seq.seq_id,
                            kv_len,
                            budget,
                            trigger_len,
                        )
                    if batch_keep_indices is not None:
                        keep_indices = batch_keep_indices[local_trigger_idx]
                    else:
                        importance_scores = attn_scores[batch_idx, :kv_len]
                        keep_indices = select_fn(
                            layer_idx,
                            seq,
                            importance_scores,
                            kv_len,
                            budget,
                        )
                    keep_batch.append(keep_indices)
                    seq_batch.append(seq)

                if free_batch is not None and len(seq_batch) > 1:
                    keep_indices = torch.stack(keep_batch, dim=0)
                    free_batch(layer_idx, seq_batch, keep_indices)
                else:
                    for seq, keep_indices in zip(seq_batch, keep_batch):
                        self.cache_manager.free_part_slots(
                            layer_idx,
                            seq,
                            keep_indices,
                        )


class SkipKVRuntime(JointDecodeRuntime):
    profiler_name = "skipkv_decode_eviction"
    select_fn_name = "select_skipkv_indices"
    interval_config_name = "skipkv_compression_interval"

    def needs_attention_score(
        self,
        layer_idx: int,
        step: SparseStepContext,
    ) -> bool:
        if step.is_prefill:
            return False
        budget = self._get_joint_decode_budget()
        if budget is None:
            return False
        if bool(getattr(self.config, "decode_graph", False)):
            return True
        state = self.layer_batch_sparse_states[layer_idx]
        if state.context_lens is None:
            return False
        trigger_len = int(budget) + int(self.config.skipkv_compression_interval)
        return bool(
            (
                (state.context_lens >= trigger_len)
                & (state.context_lens > budget)
            ).any()
        )
