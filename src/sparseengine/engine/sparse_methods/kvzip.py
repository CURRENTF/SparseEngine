from __future__ import annotations

import torch
import torch.distributed as dist

from .base import AuxiliaryPrefillRequest, SparseStepContext
from .passthrough import PassThroughRuntime


class KVzipRuntime(PassThroughRuntime):
    """The repository's kvzip_global scoring policy, applied once after prefill."""

    def set_auxiliary_prefill_prompt(self, token_ids: list[int]) -> None:
        self.cache_manager.set_reconstruction_prompt(token_ids)

    @torch.no_grad()
    def finish_step(self, step: SparseStepContext) -> None:
        if not step.is_prefill:
            return
        manager = self.cache_manager
        budget = int(self.config.kvzip_token_budget)
        for seq in step.seqs:
            length = int(seq.num_prompt_tokens)
            if not seq.is_last_chunk_prefill or length <= budget:
                continue
            # Also validates replay headroom for direct ModelRunner callers.
            manager.prompt_admission_costs(seq)
            source = manager.reconstruction_source(seq)
            scores = torch.zeros(length, dtype=torch.float32, device=self.device)
            chunk = int(self.config.kvzip_score_chunk_size)
            previous = int(self.config.kvzip_prev_postfix_size)
            for start in range(0, length, chunk):
                end = min(length, start + chunk)
                replay = (manager.reconstruction_prompt_ids
                          + tuple(source[max(0, start - previous):end]))
                with manager.reconstruction_chunk(seq, len(replay), scores):
                    self.auxiliary_prefill(AuxiliaryPrefillRequest(seq, replay, length))
            manager.parallel_context.attn_tp.all_reduce(scores, op=dist.ReduceOp.MAX)
            # Stable descending sort gives the maintenance API's original-token
            # tie break without copying scores to the host for a Python sort.
            keep = torch.argsort(scores, descending=True, stable=True)[:budget].sort().values
            manager.compact_reconstruction(seq, keep)
