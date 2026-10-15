"""Slow, independent FP32 position-based oracles, exclusively for validation."""

import torch


def adakv_selection(scores, lengths, *, budget, alpha, sink=0, recent=0, pool=1):
    """Independent Python ordering oracle; inputs are recorded CPU scores."""
    import math

    selected, remaining = [], []
    for head, length in enumerate(lengths):
        values = list(map(float, scores[head, :length]))
        if pool > 1:
            values = [max(values[max(0, i - pool // 2):min(length, i - pool // 2 + pool)])
                      for i in range(length)]
        protected = set(range(min(length, sink)))
        protected.update(range(max(len(protected), length - recent), length))
        floor = min(length, max(1, math.floor(budget * alpha)))
        order = sorted(range(length), key=lambda token: (-values[token], token))
        for token in order:
            if len(protected) >= floor:
                break
            protected.add(token)
        selected.append(protected)
        remaining.extend((values[token], head, token) for token in range(length) if token not in protected)
    remaining.sort(key=lambda item: (-item[0], item[1], item[2]))
    target = min(sum(lengths), len(lengths) * budget)
    for _, head, token in remaining[:target - sum(map(len, selected))]:
        selected[head].add(token)
    return [sorted(head) for head in selected]


def attention(q, view, cu, query_positions):
    m, p = view.meta, view.payload
    h, g = m.active_slots.shape[1], q.shape[1] // m.active_slots.shape[1]
    out = torch.zeros_like(q)
    for b, row in enumerate(m.req_indices.tolist()):
        if row < 0:
            continue
        begin, end = int(cu[b]), int(cu[b + 1])
        for head in range(h):
            n = int(m.context_lens[b, head])
            slots = m.active_slots[row, head, :n].long()
            keys, values = p.k_cache[slots].float(), p.v_cache[slots].float()
            query = q[begin:end, head * g:(head + 1) * g].float().transpose(0, 1)
            scores = query @ keys.T / q.shape[-1]**0.5
            allowed = query_positions[begin:end, None] >= m.positions[row, head, None, :n]
            scores.masked_fill_(~allowed[None], -torch.inf)
            result = scores.softmax(-1) @ values
            out[begin:end, head * g:(head + 1) * g] = result.transpose(0, 1)
    return out


class MaterializedReference:
    """Independent CPU-ledger addressing with the ordinary BF16 attention op.

    This isolates address/layout correctness from cumulative mixed-precision
    error. The separate FP32 oracle tests the attention equations themselves.
    Only this validation reference gathers history or loops over KV heads.
    """

    def __init__(self, manager):
        from sparseengine.operators.prefill_attention import PrefillAttentionOpSpec, prepare_prefill_attention_op
        from sparseengine.operators.decode_attention import DecodeAttentionOpSpec, prepare_decode_attention_op

        self.manager = manager
        spec = dict(num_query_heads=manager.hf_config.num_attention_heads // manager.num_kv_heads,
                    num_kv_heads=1, head_dim=manager.head_dim, activation_dtype=torch.bfloat16,
                    softmax_scale=manager.head_dim**-0.5)
        # Each call materializes one ordinary table and gets a fresh planning
        # scope below. Claiming a shared layer-varying table here rejects the
        # SM120 FA2 provider and compares FA2 production against Triton instead.
        self.prefill = prepare_prefill_attention_op(PrefillAttentionOpSpec(**spec))
        self.decode = prepare_decode_attention_op(
            DecodeAttentionOpSpec(**spec, max_batch_size=manager.config.max_num_seqs_in_batch * manager.num_kv_heads,
                                  layer_varying_page_table=True, cuda_graph=False)
        )
        if self.decode.name == "flashinfer_paged_decode":
            # Match the head adapter's public tensor-core arithmetic while
            # keeping CPU-ledger addressing and materialized history independent.
            # CUDA-core versus tensor-core BF16 differences can amplify through
            # the model; the separate FP32 oracle checks the attention equations.
            self.decode.provider.close()
            self.decode.provider._use_tensor_cores = True
            self.decode.provider.prepare(self.decode.spec)

    def __call__(self, q, view, cu, query_positions, *, is_prefill):
        from sparseengine.engine.cache_manager.base import (
            AttentionViewMeta, ExplicitKVPayload, PrefillComputeView, DecodeComputeView,
        )
        from sparseengine.utils.context import get_context

        m, p = view.meta, view.payload
        a = self.manager.allocator
        layer = self.manager.slot_table.data_ptr()
        layer_stride = self.manager.slot_table.stride(0) * self.manager.slot_table.element_size()
        layer = (m.active_slots.data_ptr() - layer) // layer_stride
        groups = q.shape[1] // a.heads
        queries, keys, values, lengths, chunks = [], [], [], [], []
        for b, row in enumerate(m.req_indices.tolist()):
            begin, end = int(cu[b]), int(cu[b + 1])
            if row < 0:
                raise ValueError("Materialized eager validator requires active rows only")
            for h in range(a.heads):
                n = int(a.lengths[layer, row, h])
                slots = torch.as_tensor(a.slots[layer, row, h, :n], device=q.device).long()
                positions = torch.as_tensor(a.positions[layer, row, h, :n], device=q.device)
                allowed = query_positions[begin:end, None] >= positions[None]
                causal = torch.arange(n, device=q.device)[None] <= (
                    torch.arange(end - begin, device=q.device)[:, None] + n - (end - begin))
                assert torch.equal(allowed, causal), "Original-position mask differs from dense reference"
                queries.append(q[begin:end, h * groups:(h + 1) * groups])
                keys.append(p.k_cache[slots, None])
                values.append(p.v_cache[slots, None])
                lengths.append(n)
                chunks.append(end - begin)
        # Keep the virtual batch shape: per-head calls can choose different BF16
        # reduction schedules and amplify small errors through the full model.
        # The independent ledger, K/V gather, Torch packing and dense addresses
        # test layout correctness without reusing the production metadata adapter.
        payload = ExplicitKVPayload(torch.cat(keys), torch.cat(values))
        table = torch.zeros((len(lengths), m.max_context_len), dtype=torch.int32, device=q.device)
        offset = 0
        for row, n in enumerate(lengths):
            table[row, :n] = torch.arange(offset, offset + n, dtype=torch.int32, device=q.device)
            offset += n
        meta = AttentionViewMeta(table, torch.arange(len(lengths), dtype=torch.int32, device=q.device),
                                 torch.tensor(lengths, dtype=torch.int32, device=q.device),
                                 m.max_context_len)
        query = torch.cat(queries)
        context = get_context()
        saved_scope = context.attention_validation_scope
        context.attention_validation_scope = object()
        try:
            if is_prefill:
                chunk_lens = torch.tensor(chunks, dtype=torch.int32, device=q.device)
                indptr = torch.cat((torch.zeros(1, dtype=torch.int32, device=q.device),
                                    chunk_lens.cumsum(0, dtype=torch.int32)))
                result = self.prefill.run(query, PrefillComputeView(meta, payload),
                    qo_indptr=indptr, chunk_lens=chunk_lens,
                    max_context_len=m.max_context_len, layer_idx=layer)
            else:
                result = self.decode.run(query, DecodeComputeView(meta, payload))
        finally:
            context.attention_validation_scope = saved_scope
        out, offset = torch.empty_like(q), 0
        for b in range(len(m.req_indices)):
            begin, end = int(cu[b]), int(cu[b + 1])
            for h in range(a.heads):
                out[begin:end, h * groups:(h + 1) * groups] = result[offset:offset + end - begin]
                offset += end - begin
        return out

    def close(self):
        self.prefill.close()
        self.decode.close()


def scores(manager, seq_id, layer):
    a = manager.allocator
    row = a.seq_rows[seq_id]
    count, end = manager.query_counts[seq_id], manager.last_positions[seq_id] + 1
    query_pos = torch.arange(end - count, end, device=manager.device)
    queries = manager.query_ring[layer, row, query_pos % manager.config.observation_window_size].float()
    groups = queries.shape[1] // manager.num_kv_heads
    result = []
    keys = manager.get_layer_kv_cache(layer)[0]
    for head in range(manager.num_kv_heads):
        n = int(a.lengths[layer, row, head])
        slots = manager.slot_table[layer, row, head, :n].long()
        query = queries[:, head * groups:(head + 1) * groups].transpose(0, 1)
        logits = query @ keys[slots].float().T
        allowed = query_pos[:, None] >= manager.original_positions[layer, row, head, None, :n]
        logits.masked_fill_(~allowed[None], -torch.inf)
        if manager.config.sparse_prefill_score_mode == "probability":
            score = (logits / keys.shape[-1]**0.5).softmax(-1).mean(1).max(0).values
        else:
            score = logits.flatten(0, 1).max(0).values
        result.append(score)
    return result
