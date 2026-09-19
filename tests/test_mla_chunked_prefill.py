from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from sparseengine.engine.cache_manager.base import (
    AttentionViewMeta,
    MlaLatentPayload,
    PrefillComputeView,
    PrefillScoreRequest,
)
from sparseengine.kernels.triton.mla.prefill import attention_partial
from sparseengine.operators.mla_attention import MlaAttentionOpSpec, MlaSglFa3Provider
from sparseengine.operators.mla_prefill import ChunkedMlaPrefill


def partial_provider(backend, spec, device, batch_size):
    if backend == "triton":
        return SimpleNamespace()
    if backend == "prepared":
        if torch.cuda.get_device_capability(device)[0] < 8:
            pytest.skip("pipelined MLA requires Ampere or newer")
        from sparseengine import platforms
        from sparseengine.operators.mla_prefill_attention import resolve_mla_prefill

        prepared = resolve_mla_prefill(spec, platforms.current_platform.get_device_caps(device.index))

        def partial(q, k, v, cu_q, cu_k, max_q, max_k, *, causal):
            return prepared(
                q, k, v, cu_q, cu_k, max_q, max_k,
                scale=spec.softmax_scale, causal=causal,
            )

        return SimpleNamespace(
            run_prefill_chunk=partial, prefill_workspace_bytes=prepared.workspace_bytes,
        )
    from sparseengine.kernels.external.sgl.fa3 import sgl_fa3_device_support

    supported, reason = sgl_fa3_device_support(device.index)
    if not supported:
        pytest.skip(reason)
    return MlaSglFa3Provider(op_spec=spec, device=device, max_batch_size=batch_size)


def test_reused_mla_plan_tracks_current_layer_score_request():
    # Adjacent full layers share packing but only the final one may observe
    # scores for a sparse successor. Single-view numerical tests miss this.
    runner = ChunkedMlaPrefill(SimpleNamespace(), SimpleNamespace(), 4)
    meta = AttentionViewMeta(
        active_slots=torch.arange(16, dtype=torch.int32).reshape(2, 8),
        req_indices=torch.tensor([1, 0], dtype=torch.int32),
        context_lens=torch.tensor([8, 5], dtype=torch.int32),
    )
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    scope = object()
    plan = runner.prepare(SimpleNamespace(meta=meta), cu_q, scope)
    current_slots = plan.current_slots
    score = torch.full((2, 8), -torch.inf)
    cache_request = PrefillScoreRequest(((6, 8), (2, 5)), "logits")
    for output in (None, score, None, score.clone()):
        view = SimpleNamespace(meta=replace(meta, attn_score=output))
        plan = runner.prepare(view, cu_q, scope)
        assert plan.current_slots is current_slots
        assert plan.meta.attn_score is output
        request = runner.score_request(plan)
        if output is None:
            assert request is None
            assert runner.score_request(plan, cache_request) is cache_request
        else:
            assert request.query_ranges == ((6, 8), (2, 5))
            with pytest.raises(ValueError, match="cannot combine"):
                runner.score_request(plan, cache_request)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "queries,keys",
    [((1025, 7), (1025, 7)), ((65, 3), (193, 131)), ((1025, 7), (17, 0))],
)
def test_triton_partial_ragged_causal_bounds_and_lse(causal, queries, keys):
    # Loop clipping must preserve bottom-right causal alignment, partial tiles,
    # and fully masked rows. Existing chunked tests use only short query tiles
    # and cannot catch a wrong bound after increasing the attention tile size.
    torch.manual_seed(51)
    device = "cuda"
    q = (
        torch.randn(5, sum(queries), 256, device=device, dtype=torch.bfloat16) * 0.2
    ).transpose(0, 1)
    k = torch.randn(sum(keys), 5, 256, device=device, dtype=torch.bfloat16) * 0.2
    # V is a strided view of the joint KV projection in the serving path.
    v = torch.randn(sum(keys), 5, 448, device=device, dtype=torch.bfloat16)[..., 192:]
    cu_q = torch.tensor([0, queries[0], sum(queries)], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, keys[0], sum(keys)], device=device, dtype=torch.int32)
    output, lse = attention_partial(
        q, k, v, cu_q, cu_k, max(queries), max(keys), scale=0.0625, causal=causal
    )
    qa = ka = 0
    for qn, kn in zip(queries, keys):
        logits = torch.einsum(
            "qhd,khd->hqk", q[qa : qa + qn].float(), k[ka : ka + kn].float()
        ) * 0.0625
        if causal:
            mask = torch.arange(kn, device=device)[None] <= (
                torch.arange(qn, device=device)[:, None] + kn - qn
            )
            logits.masked_fill_(~mask[None], -torch.inf)
        expected_lse = logits.logsumexp(-1)
        probabilities = logits.softmax(-1).nan_to_num(0)
        expected = torch.einsum("hqk,khd->qhd", probabilities, v[ka : ka + kn].float())
        torch.testing.assert_close(
            output[qa : qa + qn].float(), expected, atol=0.006, rtol=0.03
        )
        torch.testing.assert_close(
            lse[:, qa : qa + qn], expected_lse, atol=0.003, rtol=0.001
        )
        qa += qn
        ka += kn


def make_case(contexts=(73, 29, 41), queries=(17, 29, 9), heads=5):
    torch.manual_seed(129)
    device, dtype = "cuda", torch.bfloat16
    spec = MlaAttentionOpSpec(
        num_q_heads=heads,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=dtype,
        cache_dtype=dtype,
        tp_size=1,
        cuda_graph=False,
    )
    capacity = sum(contexts)
    latent = torch.randn(capacity, 1, 512, device=device, dtype=dtype)
    rope = torch.randn(capacity, 1, 64, device=device, dtype=dtype) * 0.2
    weight = torch.randn(heads, 448, 512, device=device, dtype=dtype) * 0.03
    q = torch.randn(sum(queries), heads, 256, device=device, dtype=dtype) * 0.3
    slots = torch.full(
        (len(contexts), max(contexts)), -1, device=device, dtype=torch.int32
    )
    rows = tuple(reversed(range(len(contexts))))
    permutation = torch.randperm(capacity, device=device).int()
    start = 0
    for row, n in zip(rows, contexts):
        slots[row, :n] = permutation[start : start + n]
        start += n
    view = PrefillComputeView(
        AttentionViewMeta(
            slots,
            torch.tensor(rows, device=device, dtype=torch.int32),
            torch.tensor(contexts, device=device, dtype=torch.int32),
        ),
        MlaLatentPayload(latent, rope),
    )
    cu = torch.tensor(
        [0, *torch.tensor(queries).cumsum(0).tolist()], device=device, dtype=torch.int32
    )

    def project(x):
        return torch.nn.functional.linear(x, weight.flatten(0, 1))

    def absorb(x):
        return torch.bmm(x.transpose(0, 1), weight[:, :192]).transpose(0, 1)

    return spec, q, view, cu, project, absorb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_main_attention_max_scores_reuse_output_and_match_full_qk():
    # Main-attention scores must cover every current query, reuse the runtime's
    # output, reset atomic maxima, and preserve ragged padding. Cache-method
    # tests above/below instead allocate outputs for trailing query windows.
    spec, q, view, cu, project, absorb = make_case(queries=(65, 29, 9))
    buffer = torch.full((3, 73), 1e6, device=q.device, dtype=torch.float32)
    view = replace(view, meta=replace(view.meta, attn_score=buffer))
    runner = ChunkedMlaPrefill(spec, SimpleNamespace(), 19)
    scope = object()
    plan = runner.prepare(view, cu, scope)
    request = runner.score_request(plan)
    for query in (q, -q):
        actual, _, scores = runner.run(query, view, cu, scope, project, absorb, request)
        assert scores is buffer
        for i, n in enumerate(plan.contexts):
            a, b = plan.query_starts[i:i + 2]
            slots = view.meta.active_slots[plan.rows[i], :n].long()
            expanded = project(view.payload.latent_cache[slots, 0]).view(n, spec.local_q_heads, 448)
            key = torch.cat((expanded[..., :192], view.payload.rope_cache[slots, 0, None].expand(-1, spec.local_q_heads, -1)), -1)
            raw = torch.einsum("qhd,khd->hqk", query[a:b].float(), key.float())
            visible = torch.arange(n, device=q.device)[None] <= torch.arange(n - (b - a), n, device=q.device)[:, None]
            raw.masked_fill_(~visible[None], -torch.inf)
            torch.testing.assert_close(scores[i, :n], raw.amax((0, 1)), atol=0.001, rtol=0.015)
            assert torch.isneginf(scores[i, n:]).all()
            reference = torch.einsum("hqk,khd->qhd", (raw * spec.softmax_scale).softmax(-1), expanded[..., 192:].float())
            torch.testing.assert_close(actual[a:b].float(), reference, atol=0.006, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mode,full_normalizer,backend",
    [
        ("logits", False, "triton"),
        ("probability", False, "triton"),
        ("logits", False, "prepared"),
        ("probability", False, "prepared"),
        ("probability", True, "prepared"),
        ("logits", False, "fa3"),
        ("probability", False, "fa3"),
        ("probability", True, "triton"),
        ("probability", True, "fa3"),
    ],
)
def test_chunked_attention_and_scores_match_explicit_oracle(
    mode, full_normalizer, backend
):
    # Independent full softmax protects masks, global normalization, physical
    # slot indirection, and score reductions across uneven history/query blocks.
    spec, q, view, cu, project, absorb = make_case()
    provider = partial_provider(backend, spec, q.device, 3)
    runner = ChunkedMlaPrefill(spec, provider, 19)
    contexts, starts = view.meta.context_lens.tolist(), cu.tolist()
    ranges = tuple(
        (n - min(7, b - a), n) for n, a, b in zip(contexts, starts, starts[1:])
    )
    request = PrefillScoreRequest(
        ranges, mode, 0 if full_normalizer else 3, 0 if full_normalizer else 4
    )
    actual, lse, scores = runner.run(q, view, cu, object(), project, absorb, request)
    for i, n in enumerate(contexts):
        a, b = starts[i : i + 2]
        row = int(view.meta.req_indices[i])
        indices = view.meta.active_slots[row, :n].long()
        latent = view.payload.latent_cache[indices, 0]
        rope = view.payload.rope_cache[indices, 0]
        expanded = project(latent).view(n, spec.local_q_heads, 448)
        k = torch.cat(
            (expanded[..., :192], rope[:, None].expand(-1, spec.local_q_heads, -1)), -1
        )
        v = expanded[..., 192:]
        raw = torch.einsum("qhd,khd->hqk", q[a:b].float(), k.float())
        qi = torch.arange(n - (b - a), n, device=q.device)
        ki = torch.arange(n, device=q.device)
        mask = qi[:, None] >= ki[None, :]
        z = (raw * spec.softmax_scale).masked_fill(~mask[None], -torch.inf)
        expected = torch.einsum("hqk,khd->qhd", z.softmax(-1), v.float())
        torch.testing.assert_close(actual[a:b].float(), expected, atol=0.006, rtol=0.03)
        torch.testing.assert_close(lse[:, a:b], z.logsumexp(-1), atol=0.003, rtol=0.001)
        observed = raw[:, -(ranges[i][1] - ranges[i][0]) :]
        valid = (
            mask[-observed.shape[1] :]
            & (ki[None] >= request.candidate_start)
            & (ki[None] < n - request.recent_keep_tokens)
        )
        if mode == "logits":
            ref = observed.masked_fill(~valid[None], -torch.inf).amax((0, 1))
            tolerance = 0.001
        else:
            ref = (
                (observed * spec.softmax_scale)
                .masked_fill(~valid[None], -torch.inf)
                .softmax(-1)
                .mean(1)
                .amax(0)
            )
            tolerance = 0.0002
        torch.testing.assert_close(scores[i, :n], ref, atol=tolerance, rtol=0.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", [None, "logits", "probability"])
@pytest.mark.parametrize("backend", ["triton", "fa3", "prepared"])
def test_chunk_size_preserves_outputs_and_bounds_history_projection(mode, backend):
    spec, q, view, cu, project, absorb = make_case(contexts=(171, 53), queries=(35, 5))
    provider = partial_provider(backend, spec, q.device, 2)
    # Scoring must not re-project historical KV after the attention pass.
    request = PrefillScoreRequest(((166, 171), (48, 53)), mode, 3, 4) if mode else None
    outputs = []
    for size in (17, 64):
        projected = []

        def tracked_project(x, projected=projected):
            projected.append(x.shape[0])
            return project(x)

        runner = ChunkedMlaPrefill(spec, provider, size)
        outputs.append(
            runner.run(q, view, cu, object(), tracked_project, absorb, request)[0]
        )
        assert projected[0] == q.shape[0]
        assert max(projected[1:]) <= size
        assert sum(projected) == sum(view.meta.context_lens.tolist())
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.006, rtol=0.03)


def test_history_budget_is_bounded_and_plan_released():
    # Long contexts previously grew the full-history workspace and retained the
    # temporary startup cache's mapping after runtime retirement.
    import weakref

    from sparseengine.operators.mla_prefill import estimate_mla_prefill_workspace_bytes

    spec = MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=1,
        cuda_graph=False,
    )
    runner = ChunkedMlaPrefill(spec, SimpleNamespace(), 256)
    estimates = []
    for length in (1024, 8192):
        view = PrefillComputeView(
            AttentionViewMeta(
                torch.arange(length, dtype=torch.int32)[None],
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([length], dtype=torch.int32),
            ),
            MlaLatentPayload(torch.empty(0, 1, 512), torch.empty(0, 1, 64)),
        )
        cu = torch.tensor([0, 128], dtype=torch.int32)
        scope = object()
        plan = runner.prepare(view, cu, scope)
        assert runner.prepare(view, cu, scope) is plan
        assert all(n <= runner.chunk_size for _, _, n, _ in plan.history_chunks)
        estimates.append(
            estimate_mla_prefill_workspace_bytes(
                plan=plan,
                spec=spec,
                chunk_size=256,
                hidden_size=2048,
                projection_chunk_size=128,
            )
        )
    # Only tiny packing metadata grows with history; expanded KV does not.
    assert estimates[1] - estimates[0] < 4096
    mapping = weakref.ref(view.meta.active_slots)
    del view, plan
    runner.clear()
    assert mapping() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["triton", "fa3", "prepared"])
def test_multi_tile_observations_and_empty_candidates(backend):
    # H2O can observe more than one 32-query tile. Entire candidate blocks may
    # be causally masked; they must contribute zero probability, never NaN.
    spec, q, view, cu, project, absorb = make_case((239,), (137,), heads=20)
    provider = partial_provider(backend, spec, q.device, 1)
    runner = ChunkedMlaPrefill(spec, provider, 53)
    req = PrefillScoreRequest(((102, 239),), "probability", 130, 3)
    _, _, scores = runner.run(q, view, cu, object(), project, absorb, req)
    slots = view.meta.active_slots[0, :239].long()
    latent, rope = runner.gather(view.payload, slots)
    expanded = project(latent).view(239, 20, 448)
    keys = torch.cat((expanded[..., :192], rope[:, None].expand(-1, 20, -1)), -1)
    z = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) * spec.softmax_scale
    ki = torch.arange(239, device=q.device)
    valid = (
        (torch.arange(102, 239, device=q.device)[:, None] >= ki)
        & (ki >= 130)
        & (ki < 236)
    )
    p = z.masked_fill(~valid, -torch.inf).softmax(-1).nan_to_num(0)
    ref = p.mean(1).amax(0)
    torch.testing.assert_close(scores[0], ref, atol=2e-4, rtol=0.015)
