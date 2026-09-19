from __future__ import annotations

import torch

from sparseengine.utils.profiler import profiler

from .base import SparseStepContext
from .passthrough import PassThroughRuntime


class RKVRuntime(PassThroughRuntime):
    """Official vLLM R-KV global selection with per-layer physical slot maps."""

    @torch.no_grad()
    def finish_step(self, step: SparseStepContext) -> None:
        if step.is_prefill:
            return
        manager = self.cache_manager
        layers = tuple(manager.kv_transformer_layer_indices())
        if not layers:
            return
        budget = self.num_sink + self.decode_keep_tokens + self.num_recent
        interval = int(self.config.rkv_compression_interval)
        window = int(self.config.rkv_observation_tokens)
        lengths = manager.decode_kv_lens_for_layer(layers[0], step.seqs)
        for layer in layers[1:]:
            if manager.decode_kv_lens_for_layer(layer, step.seqs) != lengths:
                raise RuntimeError("R-KV global selection requires matching token domains in every KV layer.")
        groups = {}
        for seq, length in zip(step.seqs, lengths):
            if seq.is_recompute_replay:
                for layer in layers:
                    manager._clear_rkv_query_cache_row(layer, manager.seq_id_to_row[layer][seq.seq_id])
                continue
            decoded = seq.num_tokens - seq.num_prompt_tokens
            if decoded > 0 and decoded % interval == 0 and length >= budget + interval:
                groups.setdefault(length, []).append(seq)
        if not groups:
            return
        tp = manager.parallel_context.attn_tp
        pending = []
        with profiler.record("rkv_decode_eviction"):
            for length, seqs in groups.items():
                ready = torch.stack([manager.rkv_observation_ready(layer, seqs, length)
                                     for layer in layers]).all(dim=0).to(torch.int32)
                tp.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
                active = []
                for seq, has_window in zip(seqs, ready.tolist()):
                    if has_window:
                        active.append(seq)
                    elif length < seq.num_tokens:
                        raise RuntimeError("R-KV missing decode observations after KV eviction; refusing partial-window scoring.")
                if not active:
                    continue
                # Bound gathered K/query storage independently of request concurrency.
                scores_by_request = []
                for seq in active:
                    total = None
                    for layer in layers:
                        score = manager.rkv_joint_scores(layer, [seq], length)
                        total = score if total is None else total + score
                    scores_by_request.append(total)
                scores = torch.cat(scores_by_request, dim=0)
                tp.all_reduce(scores)
                if not bool(torch.isfinite(scores).all()):
                    raise RuntimeError("R-KV computed non-finite scores; refusing to evict.")
                history = scores.topk(budget - window, dim=-1).indices
                recent = torch.arange(length - window, length, device=scores.device).expand(len(active), -1)
                keep = torch.cat((history, recent), dim=-1).sort(dim=-1).values
                pending.append((active, keep))
            # No cache mutation until every request group's ranking is valid.
            for seqs, keep in pending:
                manager.free_part_slots_batch_layers(
                    list(layers), seqs, keep.unsqueeze(0).expand(len(layers), -1, -1),
                    keep_indices_sorted=True,
                )
                self.debug_dynamic_selection["rkv_compactions"] = (
                    int(self.debug_dynamic_selection.get("rkv_compactions", 0)) + len(seqs)
                )
