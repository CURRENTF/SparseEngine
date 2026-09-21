"""Independent raw-QK oracle; interpreter runs do not establish CUDA correctness."""
import os

import pytest
import torch

from sparseengine.kernels.triton.mla.prefill_score import score_block


_INTERPRET = os.getenv("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(
    not _INTERPRET and not torch.cuda.is_available(), reason="requires CUDA or Triton interpreter"
)


@pytest.mark.parametrize("query_count,key_count,candidate", [(1, 1, 0), (35, 71, 3), (33, 65, 60)])
def test_latent_logits_match_raw_qk_and_reset(query_count, key_count, candidate):
    device = "cpu" if _INTERPRET else "cuda"
    torch.manual_seed(903)
    heads, dim = 3, 512
    # Match absorption's head-major layout and strided RoPE query slices.
    q = (torch.randn(heads, query_count, dim, device=device, dtype=torch.float16) * .2).transpose(0, 1)
    qr = torch.randn(query_count, heads, 256, device=device, dtype=torch.float16)[..., 192:]
    k = torch.randn(key_count, 1, dim, device=device, dtype=torch.float16) * .2
    kr = torch.randn(key_count, 64, device=device, dtype=torch.float16) * .2
    output = torch.empty(key_count, device=device)
    query_start = max(0, key_count - query_count)
    # This pointer must never be read for raw logits.
    lse = torch.full((heads, query_count), torch.nan, device=device)
    for current_q in (q, -q):
        output.fill_(-torch.inf)
        score_block(current_q, k, output, lse, query_start=query_start,
                    key_start=0, candidate_start=candidate, candidate_end=key_count,
                    scale=.125, mode="logits", rope_q=qr, rope_k=kr, latent=True)
        raw = torch.einsum("qhd,kd->hqk", current_q.float(), k[:, 0].float())
        raw += torch.einsum("qhd,kd->hqk", qr.float(), kr.float())
        ki = torch.arange(key_count, device=device)
        valid = (query_start + torch.arange(query_count, device=device)[:, None] >= ki) & (ki >= candidate)
        expected = raw.masked_fill(~valid[None], -torch.inf).amax((0, 1))
        torch.testing.assert_close(output, expected, atol=.003, rtol=.003)


@pytest.mark.parametrize("mode,candidate,recent", [
    ("logits", 0, 0), ("logits", 17, 3),
    ("probability", 0, 0), ("probability", 17, 3),
])
def test_post_attention_scorer_uses_latent_keys_with_ragged_rows(mode, candidate, recent):
    from types import SimpleNamespace
    from sparseengine.engine.cache_manager.base import AttentionViewMeta, MlaLatentPayload, PrefillComputeView, PrefillScoreRequest
    from sparseengine.operators.mla_prefill import ChunkedMlaPrefill

    device = "cpu" if _INTERPRET else "cuda"
    torch.manual_seed(71)
    contexts, queries, rows = (71, 43, 13), (35, 7, 1), (2, 0, 1)
    heads = 3
    latent = torch.randn(sum(contexts), 1, 512, device=device, dtype=torch.float16) * .2
    rope = torch.randn(sum(contexts), 1, 64, device=device, dtype=torch.float16) * .2
    q = torch.randn(sum(queries), heads, 128, device=device, dtype=torch.float16) * .2
    weight = torch.randn(heads, 64, 512, device=device, dtype=torch.float16) * .1
    def absorb(x):
        return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)
    slots = torch.full((3, max(contexts)), -1, dtype=torch.int32, device=device)
    permutation = torch.randperm(sum(contexts), device=device).int()
    offset = 0
    for row, n in zip(rows, contexts):
        slots[row, :n] = permutation[offset:offset + n]
        offset += n
    # Third row has no observation; it must remain at the output identity.
    ranges = ((36, 71), (38, 43), (0, 0))
    view = PrefillComputeView(
        AttentionViewMeta(slots, torch.tensor(rows, device=device, dtype=torch.int32),
                          torch.tensor(contexts, device=device, dtype=torch.int32)),
        MlaLatentPayload(latent, rope))
    cu = torch.tensor([0, 35, 42, 43], device=device, dtype=torch.int32)
    runner = ChunkedMlaPrefill(SimpleNamespace(qk_head_dim=128, rope_dim=64, softmax_scale=.125), SimpleNamespace(), 19)
    plan = runner.prepare(view, cu, object(), prepare_history=False)
    request = PrefillScoreRequest(ranges, mode, candidate, recent)
    absorbed = absorb(q[..., :64])
    lse = torch.empty(heads, sum(queries), device=device)
    expected = torch.full((3, max(contexts)), -torch.inf if mode == "logits" else 0.0, device=device)
    for i, n in enumerate(contexts):
        a, b = plan.query_starts[i:i + 2]
        indices = slots[rows[i], :n].long()
        raw = torch.einsum("qhd,kd->hqk", absorbed[a:b].float(), latent[indices, 0].float())
        raw += torch.einsum("qhd,kd->hqk", q[a:b, :, 64:].float(), rope[indices, 0].float())
        ki = torch.arange(n, device=device)
        valid = torch.arange(n - queries[i], n, device=device)[:, None] >= ki
        lse[:, a:b] = (raw * .125).masked_fill(~valid[None], -torch.inf).logsumexp(-1)
        start, end = ranges[i]
        if start == end:
            continue
        first, last = start - (n - queries[i]), end - (n - queries[i])
        valid = valid[first:last] & (ki >= candidate) & (ki < n - recent)
        observed = raw[:, first:last]
        if mode == "logits":
            expected[i, :n] = observed.masked_fill(~valid[None], -torch.inf).amax((0, 1))
        else:
            expected[i, :n] = (observed * .125).masked_fill(~valid[None], -torch.inf).softmax(-1).nan_to_num(0).mean(1).amax(0)
    actual = runner.score_compressed(q, view, plan, absorb, request, lse)
    torch.testing.assert_close(actual, expected, atol=.003 if mode == "logits" else 2e-5, rtol=.003)
    assert not plan.history_prepared
