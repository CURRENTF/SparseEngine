from __future__ import annotations

from .base import SparseMethodRuntime


class PrefillOverrideRuntime(SparseMethodRuntime):
    """Override prefill for a runtime with no prefill cache-initialization work.

    Factory compatibility checks exclude scored-compaction methods: those need
    both algorithms during prefill and cannot use exclusive phase delegation.
    Decode graph state always belongs to the original decode runtime.
    """

    def __init__(self, prefill: SparseMethodRuntime, decode: SparseMethodRuntime):
        self.prefill = prefill
        self.decode = decode
        self.active = decode
        self.config = decode.config
        self.cache_manager = decode.cache_manager
        self.sparse_method = decode.sparse_method

    @property
    def requires_committed_token_history(self):
        return (self.prefill.requires_committed_token_history
                or self.decode.requires_committed_token_history)

    @property
    def layer_batch_sparse_states(self):
        return self.active.layer_batch_sparse_states

    @property
    def sparse_config(self):
        return self.active.sparse_config

    def prepare_step(self, step):
        self.active = self.prefill if step.is_prefill else self.decode
        self.active.prepare_step(step)

    def needs_attention_score(self, layer_idx, step):
        runtime = self.prefill if step.is_prefill else self.decode
        return runtime.needs_attention_score(layer_idx, step)

    def build_prefill_selection(self, request):
        return self.prefill.build_prefill_selection(request)

    def build_decode_selection(self, request):
        return self.decode.build_decode_selection(request)

    def on_attention_end(self, event):
        self.active.on_attention_end(event)

    def on_layer_end(self, event):
        self.active.on_layer_end(event)

    def finish_step(self, step):
        self.active.finish_step(step)
        # Graph replay restores captured state without calling prepare_step.
        # Expose the decode owner's objects once prefill has finished.
        self.active = self.decode

    def get_layer_max_context_len(self, layer_idx):
        return self.active.get_layer_max_context_len(layer_idx)

    def clear_decode_attn_score_buffers(self):
        self.decode.clear_decode_attn_score_buffers()

    def decode_graph_keepalive_tensors(self):
        return self.decode.decode_graph_keepalive_tensors()

    def reset_decode_attn_scores_for_graph(self, refs):
        return self.decode.reset_decode_attn_scores_for_graph(refs)

    def debug_state_summary(self):
        return self.active.debug_state_summary()
