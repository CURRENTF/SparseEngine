from __future__ import annotations

import torch
import torch.distributed as dist

from .base import AuxiliaryPrefillRequest, SparseStepContext
from .passthrough import PassThroughRuntime
from sparseengine.engine.cache_manager.base import HeadSlotSelection


def select_kvzip_layers(scores: torch.Tensor, budget: int, prefix_length: int, *,
                        parallel_group=None, share_heads: bool = True) -> list[torch.Tensor] | list[HeadSlotSelection]:
    """Threshold original layer/head/token scores, optionally sharing layer unions."""
    layers, heads, length = scores.shape
    context_scores = scores[:, :, prefix_length:]
    if length - prefix_length <= budget:
        all_indices = torch.arange(length, device=scores.device)
        return ([all_indices for _ in range(layers)] if share_heads else
                [HeadSlotSelection(tuple(all_indices for _ in range(heads))) for _ in range(layers)])
    # TP shards contain different KV heads: MAX-reducing them before thresholding
    # would change upstream's score population and its global budget allocation.
    population = context_scores.flatten()
    if parallel_group is not None and parallel_group.size > 1:
        shards = parallel_group.gather(population)
        threshold = torch.empty((), dtype=scores.dtype, device=scores.device)
        if parallel_group.rank == 0:
            population = torch.cat(shards)
            threshold.copy_(population.sort(descending=True).values[
                layers * heads * parallel_group.size * budget - 1])
        parallel_group.broadcast(threshold)
    else:
        threshold = population.sort(descending=True).values[layers * heads * budget - 1]
    protected = torch.arange(prefix_length, device=scores.device)
    if not share_heads:
        return [HeadSlotSelection(tuple(torch.cat((protected,
                torch.nonzero(head > threshold).flatten() + prefix_length)) for head in layer))
                for layer in context_scores]
    union = (context_scores.amax(dim=1) > threshold).to(torch.uint8)
    if parallel_group is not None:
        parallel_group.all_reduce(union, op=dist.ReduceOp.MAX)
    return [torch.cat((protected, torch.nonzero(row).flatten() + prefix_length))
            for row in union]


class KVzipRuntime(PassThroughRuntime):
    """Query-agnostic reconstruction with storage-owned layer/head selections."""

    def set_auxiliary_prefill_prompt(self, token_ids: dict[str, list[int]]) -> None:
        self.cache_manager.set_reconstruction_prompt(token_ids)

    @torch.no_grad()
    def finish_step(self, step: SparseStepContext) -> None:
        if not step.is_prefill:
            return
        manager = self.cache_manager
        budget = int(self.config.kvzip_token_budget)
        for seq in step.seqs:
            if not seq.is_context_boundary_prefill:
                continue
            length, prefix = int(seq.context_length), int(seq.context_prefix_length)
            if length - prefix <= budget:
                continue
            manager.prompt_admission_costs(seq)
            source = manager.reconstruction_source(seq)
            scores = torch.zeros((manager.num_kv_layers, manager.num_kv_heads, length),
                                 dtype=torch.float32, device=self.device)
            chunk = int(self.config.kvzip_score_chunk_size)
            previous = min(chunk, int(self.config.kvzip_prev_postfix_size))
            template = manager.reconstruction_prompt_ids
            for start in range(prefix, length, chunk):
                end = min(length, start + chunk)
                instruction = template["first"] if start == prefix else (
                    template["continuation"] + tuple(source[max(prefix, start - previous):start]))
                replay = instruction + template["suffix"] + tuple(source[start:end])
                with manager.reconstruction_chunk(seq, len(replay), scores,
                                                   chunk_start=start, chunk_end=end):
                    self.auxiliary_prefill(AuxiliaryPrefillRequest(seq, replay, length))
            keep = select_kvzip_layers(scores, budget, prefix,
                parallel_group=manager.parallel_context.attn_tp,
                share_heads=manager.reconstruction_heads_shared)
            manager.compact_reconstruction(seq, keep)
