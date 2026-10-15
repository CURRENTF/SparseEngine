"""AdaKV safeguarded allocation over physically independent KV heads.

Algorithm reference: NVIDIA/KVPress AdaKVPress at revision
71640b4f9061054a7630c5049bb9ee659a01523c. This implementation extends the
selection problem to ragged candidates, protected sink/recent entries and
chain-turn compaction; it does not copy upstream's mask-based storage.
"""

import math

import torch
import torch.nn.functional as F

from sparseengine.engine.cache_manager.base import HeadSlotSelection
from .head_slots import HeadSlotRuntime
from .base import SparseStepContext


def select_adakv_indices(scores, lengths, *, budget, alpha, sink_keep=0,
                         recent_keep=0, pool_kernel_size=1):
    """Maximize retained score subject to per-head safeguards and a layer budget.

    Lengths are the owner's CPU ledger, so padded score entries never participate.
    Sorting and masks stay on the score device. Ragged index extraction forms an
    explicit compaction synchronization boundary, before any physical mutation.
    """
    lengths = tuple(int(n) for n in lengths)
    if (scores.ndim != 2 or scores.shape[0] != len(lengths) or not lengths
            or min(lengths) <= 0 or max(lengths) > scores.shape[1]):
        raise ValueError("AdaKV requires nonempty candidate lengths within [heads, width].")
    if not scores.is_floating_point():
        raise TypeError("AdaKV requires floating-point importance scores.")
    if (type(budget) is not int or budget < 1 or isinstance(alpha, bool)
            or not math.isfinite(alpha) or not 0 < alpha <= 1):
        raise ValueError("AdaKV requires a positive budget and alpha in (0, 1].")
    if (sink_keep < 0 or recent_keep < 0 or sink_keep + recent_keep > budget
            or type(pool_kernel_size) is not int or pool_kernel_size < 1):
        raise ValueError("Invalid AdaKV protected-token or pooling configuration.")
    heads, width = scores.shape
    columns = torch.arange(width, device=scores.device)
    valid = columns[None] < torch.tensor(lengths, device=scores.device)[:, None]
    if not bool((torch.isfinite(scores) | ~valid).all()):
        raise ValueError("AdaKV candidate scores contain NaN or infinity.")
    target = min(sum(lengths), heads * budget)
    if target == sum(lengths):
        return tuple(torch.arange(n, device=scores.device) for n in lengths)
    scores = scores.masked_fill(~valid, -torch.inf)
    if pool_kernel_size > 1:
        scores = F.max_pool1d(scores[:, None], pool_kernel_size, stride=1,
                              padding=pool_kernel_size // 2)[:, 0, :width]
    floor = max(1, math.floor(budget * alpha))
    sink = [min(n, sink_keep) for n in lengths]
    recent = [min(n - s, recent_keep) for n, s in zip(lengths, sink)]
    minimum = [min(n, max(floor, s + r)) for n, s, r in zip(lengths, sink, recent)]
    extra = [m - s - r for m, s, r in zip(minimum, sink, recent)]
    selected = ((columns[None] < torch.tensor(sink, device=scores.device)[:, None])
                | (columns[None] >= torch.tensor([n - r for n, r in zip(lengths, recent)],
                                                  device=scores.device)[:, None])) & valid
    # Stable ordering gives deterministic original-position ties within a head.
    order = torch.argsort(scores.masked_fill(selected | ~valid, -torch.inf),
                          dim=-1, descending=True, stable=True)
    safe = torch.zeros_like(selected).scatter_(
        1, order, columns[None] < torch.tensor(extra, device=scores.device)[:, None])
    selected |= safe
    remaining = target - sum(minimum)
    if remaining:
        # Flattening uses head-major order, defining deterministic cross-head ties.
        global_order = torch.argsort(scores.masked_fill(selected | ~valid, -torch.inf).flatten(),
                                     descending=True, stable=True)
        selected.flatten()[global_order[:remaining]] = True
    indices = tuple(torch.nonzero(selected[h, :n], as_tuple=True)[0] for h, n in enumerate(lengths))
    if sum(idx.numel() for idx in indices) != target or any(
            idx.numel() < m for idx, m in zip(indices, minimum)):
        raise RuntimeError("AdaKV selection violated total budget or per-head safeguard.")
    return indices


class AdaKVRuntime(HeadSlotRuntime):
    def finish_step(self, step: SparseStepContext) -> None:
        if step.is_prefill:
            for seq in step.seqs:
                if seq.is_last_chunk_prefill:
                    self.cache_manager.compact_head_slots(seq.seq_id, reason="prefill")

    def select_chain(self, seq_id):
        manager, config = self.cache_manager, self.config
        row = manager.allocator.seq_rows[seq_id]
        budget = config.sink_keep_tokens + config.decode_keep_tokens + config.recent_keep_tokens
        selections = []
        for layer in range(manager.num_kv_layers):
            indices = select_adakv_indices(
                manager.score_chain_layer(seq_id, layer), manager.allocator.lengths[layer, row],
                budget=budget, alpha=config.adakv_alpha, sink_keep=config.sink_keep_tokens,
                recent_keep=config.recent_keep_tokens, pool_kernel_size=config.pool_kernel_size)
            selections.append(HeadSlotSelection(indices))
        return selections
