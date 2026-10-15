"""SnapKV query-window scores selected independently at chain turn boundaries."""

import torch
import torch.nn.functional as F

from sparseengine.engine.cache_manager.base import HeadSlotSelection
from .passthrough import PassThroughRuntime


class HeadSlotRuntime(PassThroughRuntime):
    def __init__(self, config, cache_manager):
        super().__init__(config, cache_manager)
        cache_manager.chain_selector = self.select_chain

    def select_chain(self, seq_id):
        manager = self.cache_manager
        row = manager.allocator.seq_rows[seq_id]
        selections = []
        for layer_idx, budgets in enumerate(self.config.head_slot_budgets):
            scores = manager.score_chain_layer(seq_id, layer_idx)
            indices = []
            for h, budget in enumerate(budgets):
                n = int(manager.allocator.lengths[layer_idx, row, h])
                keep = min(n, budget)
                score = scores[h, :n]
                if self.config.pool_kernel_size > 1:
                    kernel = self.config.pool_kernel_size
                    score = F.max_pool1d(score[None, None], kernel, stride=1,
                                         padding=kernel // 2).flatten()[:n]
                # Protected sink/recent entries count toward the explicit total
                # head budget, including when a budget is smaller than their sum.
                sink = min(self.config.sink_keep_tokens, keep)
                recent = min(self.config.recent_keep_tokens, keep - sink, n - sink)
                middle = keep - sink - recent
                candidates = torch.argsort(score[sink:n - recent], descending=True, stable=True)[:middle] + sink
                selected = torch.cat((torch.arange(sink, device=scores.device), candidates,
                                      torch.arange(n - recent, n, device=scores.device))).sort().values
                indices.append(selected)
            selections.append(HeadSlotSelection(tuple(indices)))
        return selections
