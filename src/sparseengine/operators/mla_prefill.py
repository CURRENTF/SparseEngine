"""MLA history chunking and sparse score execution over tagged cache views."""

from dataclasses import dataclass

import torch

from sparseengine.engine.cache_manager.base import AttentionViewMeta, PrefillScoreRequest
from sparseengine.kernels.triton.mla.prefill import attention_partial, merge_partial
from sparseengine.kernels.triton.mla.prefill_score import score_block
from sparseengine.utils.profiler import profiler


@dataclass
class PrefillPlan:
    scope: object
    meta: AttentionViewMeta
    cu_q: torch.Tensor
    contexts: tuple[int, ...]
    query_starts: tuple[int, ...]
    rows: tuple[int, ...]
    current_slots: torch.Tensor
    history_chunks: tuple[tuple[int, int, int, torch.Tensor], ...]
    request_cu_q: tuple[torch.Tensor, ...]


def estimate_mla_prefill_workspace_bytes(
    *, plan, spec, chunk_size, hidden_size, projection_chunk_size, score_request=None,
    kernel_workspace_bytes=0
):
    """Modeled live tensors; opaque provider workspaces remain startup-profiled.

    Count a conservative overlap of projection, partial output, FP32 merge,
    and sparse score state. Historical KV is bounded independently of context.
    """
    tokens = plan.query_starts[-1]
    historical = max((n for _, _, n, _ in plan.history_chunks), default=0)
    block = max(tokens, historical)
    element = spec.activation_dtype.itemsize
    cache_element = spec.cache_dtype.itemsize
    heads = spec.local_q_heads
    width = spec.qk_head_dim - spec.rope_dim + spec.value_head_dim
    gathered = block * (spec.kv_lora_rank + spec.rope_dim) * cache_element
    projected = block * heads * width * element
    projection_scratch = (
        min(block, projection_chunk_size) * heads * width * element
        if block > projection_chunk_size
        else 0
    )
    keys = block * heads * spec.qk_head_dim * element
    # Current output, one partial, final dtype conversion and FP32 accumulator.
    outputs = tokens * heads * spec.value_head_dim * (3 * element + 4)
    lse = 2 * tokens * heads * 4
    metadata = 2 * (tokens + historical) * 8 + len(plan.history_chunks) * 8
    score_bytes = 0
    if score_request is not None:
        observed = sum(end - start for start, end in score_request.query_ranges)
        max_observed = max(end - start for start, end in score_request.query_ranges)
        # Final token scores, absorbed observation Q, per-query LSE, and the
        # largest block's probability/statistics scratch (not all blocks).
        score_bytes = (
            len(plan.contexts) * max(plan.contexts) * 4
            + observed * heads * (spec.kv_lora_rank * element + 4)
            + heads * max(block, chunk_size) * 4
            + heads * ((max(block, chunk_size) + 63) // 64) * max_observed * 4
        )
    projection_output = min(tokens, projection_chunk_size) * hidden_size * element
    return (
        gathered
        + projected
        + projection_scratch
        + keys
        + outputs
        + lse
        + metadata
        + score_bytes
        + projection_output
        + kernel_workspace_bytes
    )


class ChunkedMlaPrefill:
    def __init__(self, spec, provider, chunk_size):
        if int(chunk_size) <= 0:
            raise ValueError("MLA history chunk size must be positive.")
        self.spec = spec
        self.chunk_size = int(chunk_size)
        self.plan = None
        self.partial = getattr(provider, "run_prefill_chunk", None)
        self._partial_workspace = getattr(provider, "prefill_workspace_bytes", None)

    def kernel_workspace_bytes(self, plan):
        if self._partial_workspace is None:
            return 0
        queries = tuple(b - a for a, b in zip(plan.query_starts, plan.query_starts[1:]))
        maximum = max(queries)
        required = self._partial_workspace(
            tokens=plan.query_starts[-1], batch=len(queries), max_q=maximum, max_k=maximum,
        )
        for request, _, length, _ in plan.history_chunks:
            qn = queries[request]
            required = max(required, self._partial_workspace(
                tokens=qn, batch=1, max_q=qn, max_k=length,
            ))
        return required

    def clear(self):
        self.plan = None

    def prepare(self, view, cu_q, scope):
        old = self.plan
        meta = view.meta
        if (
            old is not None
            and old.scope is scope
            and old.meta.active_slots is meta.active_slots
            and old.meta.req_indices is meta.req_indices
            and old.meta.context_lens is meta.context_lens
            and old.cu_q is cu_q
        ):
            # Packing is shared across layers; optional score outputs are not.
            old.meta = meta
            return old
        contexts = tuple(int(x) for x in meta.context_lens.tolist())
        starts = tuple(int(x) for x in cu_q.tolist())
        rows = tuple(int(x) for x in meta.req_indices.tolist())
        if (
            len(starts) != len(contexts) + 1
            or len(rows) != len(contexts)
            or starts[0] != 0
        ):
            raise ValueError("Invalid MLA prefill request packing.")
        current_slots, history, request_cu = [], [], []
        for i, (row, context) in enumerate(zip(rows, contexts)):
            qn = starts[i + 1] - starts[i]
            if qn <= 0 or context < qn or context > meta.active_slots.shape[1]:
                raise ValueError("Invalid MLA query/context length.")
            if not 0 <= row < meta.active_slots.shape[0]:
                raise ValueError("Invalid MLA request row.")
            cached = context - qn
            current_slots.append(meta.active_slots[row, cached:context])
            request_cu.append(
                torch.tensor([0, qn], dtype=torch.int32, device=cu_q.device)
            )
            for offset in range(0, cached, self.chunk_size):
                length = min(self.chunk_size, cached - offset)
                cu_k = torch.tensor([0, length], dtype=torch.int32, device=cu_q.device)
                history.append((i, offset, length, cu_k))
        self.plan = PrefillPlan(
            scope,
            meta,
            cu_q,
            contexts,
            starts,
            rows,
            torch.cat(current_slots),
            tuple(history),
            tuple(request_cu),
        )
        return self.plan

    @staticmethod
    def gather(payload, slots):
        index = slots.long()
        return (
            payload.latent_cache.index_select(0, index).squeeze(1),
            payload.rope_cache.index_select(0, index).squeeze(1),
        )

    def expand(self, latent, rope, project):
        nope = self.spec.qk_head_dim - self.spec.rope_dim
        expanded = project(latent).view(
            -1, self.spec.local_q_heads, nope + self.spec.value_head_dim
        )
        kn, v = expanded.split((nope, self.spec.value_head_dim), dim=-1)
        k = torch.empty(
            (*kn.shape[:2], self.spec.qk_head_dim), dtype=kn.dtype, device=kn.device
        )
        k[..., :nope].copy_(kn)
        k[..., nope:].copy_(rope[:, None, :])
        return k, v

    def attention(self, q, k, v, cu_q, cu_k, max_q, max_k, causal):
        if self.partial is not None:
            return self.partial(q, k, v, cu_q, cu_k, max_q, max_k, causal=causal)
        return attention_partial(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_q,
            max_k,
            scale=self.spec.softmax_scale,
            causal=causal,
        )

    def score_request(
        self, plan: PrefillPlan, cache_request: PrefillScoreRequest | None = None,
    ) -> PrefillScoreRequest | None:
        """Resolve full-query raw maxima requested through the attention view."""
        if plan.meta.attn_score is None:
            return cache_request
        if cache_request is not None:
            raise ValueError("MLA prefill cannot combine main-attention and cache score requests.")
        return PrefillScoreRequest(
            query_ranges=tuple(
                (context - (b - a), context)
                for context, a, b in zip(
                    plan.contexts, plan.query_starts, plan.query_starts[1:]
                )
            ),
            mode="logits",
        )

    def run(self, q, view, cu_q, scope, project, absorb, score_request=None):
        plan = self.prepare(view, cu_q, scope)
        if plan.query_starts[-1] != q.shape[0]:
            raise ValueError("MLA query count differs from packed query metadata.")
        scorer = (
            MlaPrefillScores(self, q, plan, score_request, absorb, output=view.meta.attn_score)
            if score_request
            else None
        )
        latent, rope = self.gather(view.payload, plan.current_slots)
        k, v = self.expand(latent, rope, project)
        max_q = max(b - a for a, b in zip(plan.query_starts, plan.query_starts[1:]))
        output, lse = self.attention(q, k, v, cu_q, cu_q, max_q, max_q, True)
        if scorer is not None and not scorer.is_probability:
            with profiler.record("prefill_token_score"):
                for i, context in enumerate(plan.contexts):
                    a, b = plan.query_starts[i : i + 2]
                    scorer.consume(i, context - (b - a), k[a:b], mode="logits")
        del latent, rope, k, v
        if plan.history_chunks:
            # FP32 accumulation avoids one BF16 rounding per history block.
            output = output.float()
        for i, offset, length, cu_k in plan.history_chunks:
            a, b = plan.query_starts[i : i + 2]
            slots = view.meta.active_slots[plan.rows[i], offset : offset + length]
            latent, rope = self.gather(view.payload, slots)
            k, v = self.expand(latent, rope, project)
            partial, partial_lse = self.attention(
                q[a:b], k, v, plan.request_cu_q[i], cu_k, b - a, length, False
            )
            merge_partial(output[a:b], lse[:, a:b], partial, partial_lse)
            if scorer is not None and not scorer.is_probability:
                with profiler.record("prefill_token_score"):
                    scorer.consume(i, offset, k, mode="logits")
            del latent, rope, k, v, partial, partial_lse
        if scorer is not None and scorer.is_probability:
            with profiler.record("prefill_token_score"):
                scorer.finish_probability(view, lse)
        return output.to(q.dtype), lse, None if scorer is None else scorer.output


class MlaPrefillScores:
    def __init__(self, owner, q, plan, request, absorb, *, output=None):
        if request.mode not in {"logits", "probability"}:
            raise ValueError("Unsupported MLA prefill score mode.")
        if len(request.query_ranges) != len(plan.contexts):
            raise ValueError("MLA score ranges must cover the prefill batch.")
        self.owner, self.plan, self.request = owner, plan, request
        self.is_probability = request.mode == "probability"
        self.full_normalizer = (
            request.candidate_ranges is None
            and request.candidate_start == 0 and request.recent_keep_tokens == 0
        )
        shape = (len(plan.contexts), max(plan.contexts))
        fill = -torch.inf if request.mode == "logits" else 0.0
        if output is None:
            output = torch.full(shape, fill, dtype=torch.float32, device=q.device)
        else:
            if (
                output.shape != shape or output.dtype != torch.float32
                or output.device != q.device or output.stride(-1) != 1
            ):
                raise ValueError(
                    "MLA prefill score output must be float32 [batch, max_context] "
                    "with contiguous rows on the query device."
                )
            output.fill_(fill)
        self.output = output
        self.queries, self.rope_queries, self.lse = [], [], []
        nope = owner.spec.qk_head_dim - owner.spec.rope_dim
        for i, (start, end) in enumerate(request.query_ranges):
            a, b = plan.query_starts[i : i + 2]
            cached = plan.contexts[i] - (b - a)
            if end < start:
                raise ValueError("MLA observation range is reversed.")
            if end > start and not cached <= start < end <= plan.contexts[i]:
                raise ValueError("MLA observation window is outside current queries.")
            observed = (
                q[a + start - cached : a + end - cached] if end > start else q[:0]
            )
            self.queries.append(
                absorb(observed[..., :nope])
                if self.is_probability and end > start
                else observed
            )
            self.rope_queries.append(
                observed[..., nope:] if self.is_probability else None
            )
            self.lse.append(
                torch.full(
                    (q.shape[1], end - start),
                    -torch.inf,
                    dtype=torch.float32,
                    device=q.device,
                )
            )

    def consume(self, i, offset, keys, *, mode, rope=None):
        start, end = self.request.query_ranges[i]
        if end <= start:
            return
        candidate_start, candidate_end = self.request.candidate_bounds(i, self.plan.contexts[i])
        if (
            offset >= candidate_end
            or offset + keys.shape[0] <= candidate_start
        ):
            return
        score_block(
            self.queries[i],
            keys,
            self.output[i, offset : offset + keys.shape[0]],
            self.lse[i],
            query_start=start,
            key_start=offset,
            candidate_start=candidate_start,
            candidate_end=candidate_end,
            scale=self.owner.spec.softmax_scale,
            mode=mode,
            rope_q=self.rope_queries[i],
            rope_k=rope,
        )

    def finish_probability(self, view, attention_lse):
        if self.full_normalizer:
            for i, (start, end) in enumerate(self.request.query_ranges):
                a, b = self.plan.query_starts[i : i + 2]
                cached = self.plan.contexts[i] - (b - a)
                if end > start:
                    self.lse[i] = attention_lse[
                        :, a + start - cached : a + end - cached
                    ].contiguous()
        # Candidate-only softmax needs its own denominator; full-key scoring
        # reuses the main attention LSE. Both scans stay in latent space.
        modes = ("probability",) if self.full_normalizer else ("stats", "probability")
        for mode in modes:
            for i, (start, end) in enumerate(self.request.query_ranges):
                if end <= start:
                    continue
                context = self.plan.contexts[i]
                candidate_start, candidate_end = self.request.candidate_bounds(i, context)
                for offset in range(
                    candidate_start, candidate_end, self.owner.chunk_size
                ):
                    stop = min(offset + self.owner.chunk_size, candidate_end)
                    slots = view.meta.active_slots[self.plan.rows[i], offset:stop]
                    latent, rope = self.owner.gather(view.payload, slots)
                    self.consume(i, offset, latent[:, None, :], mode=mode, rope=rope)
                    del latent, rope
