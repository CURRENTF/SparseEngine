from __future__ import annotations

import torch
import torch.distributed as dist

from sparseengine.engine.cache_manager import SparseSelection
from sparseengine.method_registry import omnikv_prefill_layer_indices

from .base import SparseMethodRuntime
from .dynamic import build_omnikv_keep_and_slots


class OmniKVPrefillRuntime(SparseMethodRuntime):
    """Chunk-wide raw-QK selection, propagated between observation layers.

    Selections only change reads. Every layer still writes its entire current
    chunk to the original KV storage for the independently selected decoder.
    """

    def __init__(self, config, cache_manager):
        super().__init__(config, cache_manager)
        self.full_attention_layers = set(config.omnikv_prefill_full_attention_layers)
        self._max_score = getattr(config, "attention_cache_layout", "explicit_kv") == "mla_latent"
        kv_layers = omnikv_prefill_layer_indices(config)
        self.targets: dict[int, list[int]] = {}
        anchor = None
        for layer in kv_layers:
            if layer in self.full_attention_layers:
                anchor = layer
            else:
                if anchor is None:
                    raise ValueError("OmniKV prefill requires the first KV layer to be full.")
                self.targets.setdefault(anchor, []).append(layer)
        self.obs_layer_ids = set(self.targets)
        self.num_sink = config.omnikv_prefill_sink_keep_tokens
        self.num_recent = config.omnikv_prefill_recent_keep_tokens
        self.keep_tokens = config.omnikv_prefill_keep_tokens
        self._score_buffer: torch.Tensor | None = None

    def _begin_prepare_step(self, step):
        if not step.is_prefill:
            raise RuntimeError("OmniKVPrefillRuntime only accepts prefill steps.")
        self._score_buffer = None
        self._chunk_lens = (
            step.forward_context.cu_seqlens_q[1:]
            - step.forward_context.cu_seqlens_q[:-1]
        )
        # One host read per step bounds the variable-length selection workspace.
        self._max_chunk_len = int(self._chunk_lens.max().item())

    def needs_attention_score(self, layer_idx, step):
        return step.is_prefill and layer_idx in self.obs_layer_ids

    def _prepare_prefill_attention_score(self, state, batch_size, num_heads, max_len):
        shape = (batch_size, max_len) if self._max_score else (batch_size, num_heads, max_len)
        if self._score_buffer is None:
            self._score_buffer = torch.full(
                shape, -torch.inf if self._max_score else 0.0,
                dtype=torch.float32, device=self.device,
            )
        elif self._score_buffer.shape != shape:
            raise RuntimeError("OmniKV prefill requires a shared full-KV scoring domain.")
        state.attn_score = self._score_buffer

    def build_prefill_selection(self, request):
        state = self.layer_batch_sparse_states[request.layer_idx]
        if state.active_slots is None:
            return self._full_selection(request.layer_idx)
        return SparseSelection(
            kind="slots",
            req_indices=state.req_indices,
            context_lens=state.context_lens,
            max_context_len=state.max_context_len,
            active_slots=state.active_slots,
            active_indices=state.active_indices,
            global_req_indices=state.global_req_indices,
        )

    def build_decode_selection(self, request):
        raise RuntimeError("OmniKV prefill does not own decode selection.")

    def on_layer_end(self, event):
        targets = self.targets.get(event.layer_idx)
        if not targets:
            return
        state = self.layer_batch_sparse_states[event.layer_idx]
        if state.attn_score is None:
            raise RuntimeError("OmniKV prefill observation scores were not prepared.")
        if self._max_score:
            # MLA's existing block scorer reduces raw QK over queries and heads.
            scores = state.attn_score
        else:
            # Explicit KV sums causal raw QK over queries for each head.
            scores = state.attn_score.amax(dim=1)
            scores.div_(self._chunk_lens[:, None])
        if self.config.tensor_parallel_size > 1:
            self.cache_manager.parallel_context.attn_tp.all_reduce(
                scores, dist.ReduceOp.MAX
            )
        batch_size, width = scores.shape
        hist_lens = (
            state.context_lens - self._chunk_lens - self.num_recent
        ).clamp_min(self.num_sink)
        candidate_lens = hist_lens - self.num_sink
        candidates = scores[:, self.num_sink:]
        positions = torch.arange(candidates.shape[1], device=self.device)
        candidates.masked_fill_(positions[None, :] >= candidate_lens[:, None], -torch.inf)
        k = min(self.keep_tokens, candidates.shape[1])
        # Sorted top-k keeps all finite candidates before padded -inf entries
        # when a ragged row has fewer candidates than the common k.
        indices = (
            candidates.topk(k, dim=1, sorted=True).indices.to(torch.int32)
            + self.num_sink
        )
        topk_lens = candidate_lens.clamp(max=k).to(torch.int32)
        max_s = min(width, self.num_sink + k + self.num_recent + self._max_chunk_len)
        keep, slots, lengths = build_omnikv_keep_and_slots(
            indices,
            topk_lens,
            hist_lens,
            (state.context_lens - hist_lens).clamp_min(0),
            self.cache_manager.get_layer_buffer_req_to_token_slots(targets[0]),
            state.req_indices,
            self.num_sink,
            max_s=max_s,
            context_lens=state.context_lens,
        )
        rows = torch.arange(batch_size, dtype=torch.int32, device=self.device)
        for layer in targets:
            target = self.layer_batch_sparse_states[layer]
            target.active_indices = keep
            target.active_slots = slots
            target.context_lens = lengths
            target.max_context_len = max_s
            target.req_indices = rows
        # The next observation layer reuses the same atomic-score workspace.
        self._score_buffer.fill_(-torch.inf if self._max_score else 0.0)

    def finish_step(self, step):
        for state in self.layer_batch_sparse_states.values():
            state.attn_score = None
            state.active_indices = None
            state.active_slots = None
        self._score_buffer = None
