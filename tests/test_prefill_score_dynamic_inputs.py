"""Changing observation lengths and candidate bounds must preserve sparse scores."""

import pytest
import torch


def _build_score_case(name, i):
    torch.manual_seed(314)
    if name.startswith("score_"):
        from sparseengine.kernels.triton.prefill_score import (
            PrefillScoreWorkspace,
            prefill_score_from_lse_fwd,
            prefill_score_fwd,
        )

        qn = 32 if name == "score_bounds" else 129 + i * 32
        length = 1024
        q = torch.randn(qn, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.2
        k = torch.randn(length, 2, 64, device="cuda", dtype=q.dtype) * 0.2
        out = torch.empty(1, length, device="cuda")
        pos = torch.arange(length, device="cuda")
        qp = torch.arange(length - qn, length, device="cuda")
        z = (
            torch.einsum("qhd,khd->hqk", q.float(), k.float().repeat_interleave(4, 1))
            / 8
        )
        z.masked_fill_(pos[None, None] > qp[None, :, None], -torch.inf)
        lse = z.logsumexp(-1)
        start, recent = (33 + i * 32, 17 + i * 16) if name == "score_bounds" else (0, 0)
        if name == "score_bounds":
            z.masked_fill_(
                ((pos < start) | (pos >= length - recent))[None, None], -torch.inf
            )
        expected = (
            z.amax((0, 1)) * 8
            if name == "score_logits"
            else z.softmax(-1).mean(1).amax(0)
        )
        ints = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
        args = (
            out,
            ints([0]),
            ints([0]),
            ints([length]),
            ints([length - qn]),
            qn,
            pos.int()[None],
            ints([length - qn]),
            ints([length]),
        )
        workspace = PrefillScoreWorkspace()

        def call():
            if name == "score_reuse":
                prefill_score_from_lse_fwd(q, k, lse, *args, workspace=workspace)
            else:
                prefill_score_fwd(
                    q,
                    k,
                    *args,
                    workspace=workspace,
                    candidate_start=start,
                    recent_keep_tokens=recent,
                    score_mode="logits" if name == "score_logits" else "probability",
                )
            return out

        def check(result):
            torch.testing.assert_close(
                result[0],
                expected,
                atol=2e-6 if name != "score_logits" else 0.005,
                rtol=0.015,
            )

        return call, check
    if name == "mla_latent_stride":
        from sparseengine.kernels.triton.mla.prefill_score import score_block

        qn, kn, h = 32 + i * 16, 512, 2
        q = (
            torch.randn(h, qn, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        ).transpose(0, 1)
        k = torch.randn(kn, 1, 512, device="cuda", dtype=q.dtype) * 0.05
        qr = torch.randn(qn, h, 64, device="cuda", dtype=q.dtype) * 0.05
        kr = torch.randn(kn, 1, 64, device="cuda", dtype=q.dtype) * 0.05
        lse = torch.empty(h, qn, device="cuda")
        out = torch.empty(kn, device="cuda")
        z = (
            torch.einsum("qhd,kd->hqk", q.float(), k[:, 0].float())
            + torch.einsum("qhd,kd->hqk", qr.float(), kr[:, 0].float())
        ) / 24
        z.masked_fill_(
            torch.arange(kn, device="cuda")[None, None]
            > torch.arange(kn - qn, kn, device="cuda")[None, :, None],
            -torch.inf,
        )
        expected = z.softmax(-1).mean(1).amax(0)

        def call():
            lse.fill_(-torch.inf)
            for mode in ("stats", "probability"):
                score_block(
                    q,
                    k,
                    out,
                    lse,
                    query_start=kn - qn,
                    key_start=0,
                    candidate_start=0,
                    candidate_end=kn,
                    scale=1 / 24,
                    mode=mode,
                    rope_q=qr,
                    rope_k=kr,
                )
            return out

        return (
            call,
            lambda x: torch.testing.assert_close(x, expected, atol=2e-6, rtol=0.015),
        )
    raise ValueError(name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "name",
    ["score_own", "score_logits", "score_reuse", "score_bounds", "mla_latent_stride"],
)
def test_prefill_score_dynamic_inputs(name):
    # Existing fixed-observation tests miss changing query tile counts, candidate
    # bounds, and the transposed absorbed-Q head stride used by MLA probability.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for index in (0, 1, 5, 0):
            call, check = _build_score_case(name, index)
            check(call())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
