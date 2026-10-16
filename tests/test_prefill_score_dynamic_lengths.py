"""Exact context lengths must not alter scoring or reuse stale workspace rows."""

import pytest
import torch

from sparseengine.kernels.triton.prefill_score import (
    PrefillScoreWorkspace,
    prefill_score_from_lse_fwd,
    prefill_score_fwd,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reconstruction_compact_ranges_with_workspace_reuse_and_graph():
    """Changing disjoint/overlapping tiles must not duplicate LSE or lose tails.

    Unlike the manager tests, this uses ragged rows and observes only a suffix
    of replay queries. The normalization still includes all earlier replay keys.
    """
    torch.manual_seed(71)
    workspace = PrefillScoreWorkspace()
    heads, kv_heads, dim = 6, 2, 128
    for cached_lengths, replays, sink, start, end in (
        ((4095, 4109), (137, 65), 65, 257, 1009),
        ((4095, 4109), (137, 65), 65, 65, 129),
        ((129, 145), (71, 137), 65, 65, 129),
        ((4095, 4109), (137, 65), 0, 3969, 4095),
    ):
        lengths = tuple(c + r for c, r in zip(cached_lengths, replays))
        width = max(lengths)
        q = torch.randn(heads, sum(replays), dim, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
        k = torch.randn(sum(lengths), kv_heads, dim * 2, device="cuda", dtype=q.dtype)[..., :dim]
        slots = torch.full((2, width), -1, device="cuda", dtype=torch.int32)
        permutation = torch.randperm(sum(lengths), device="cuda").int()
        slots[1, :lengths[0]] = permutation[:lengths[0]]
        slots[0, :lengths[1]] = permutation[lengths[0]:]
        tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
        reqs, offsets = tensor((1, 0)), tensor((0, replays[0]))
        cached, contexts = tensor(cached_lengths), tensor(lengths)
        score_starts = contexts - 17
        output = torch.empty((2, width + 7), device="cuda")[:, :width]
        accum = torch.full((2, kv_heads, width + 7), -7.0, device="cuda")[..., :width]
        expected = accum.clone()
        for batch, (context, replay) in enumerate(zip(cached_lengths, replays)):
            keys = torch.arange(lengths[batch], device="cuda")
            eligible = (keys < sink) | ((keys >= start) & (keys < end)) | (keys >= context)
            positions = keys[eligible]
            selected_k = k[slots[1 - batch, positions].long()].double().repeat_interleave(heads // kv_heads, 1)
            offset = sum(replays[:batch])
            observed_q = q[offset + replay - 17:offset + replay].double()
            z = torch.einsum("qhd,khd->hqk", observed_q, selected_k) * dim**-0.5
            query_positions = torch.arange(lengths[batch] - 17, lengths[batch], device="cuda")
            z.masked_fill_(positions[None, None] > query_positions[None, :, None], -torch.inf)
            probabilities = z.softmax(-1)[..., (positions >= start) & (positions < end)]
            expected[batch, :, start:end] = probabilities.reshape(kv_heads, heads // kv_heads, 17, -1).amax((1, 2)).float()

        def run():
            prefill_score_fwd(q, k, output, reqs, offsets, contexts, cached, 17,
                             slots, score_starts, contexts, workspace=workspace,
                             reconstruction_range=(sink, start, end),
                             reconstruction_head_scores=accum)

        for _ in range(3):
            run()
        torch.testing.assert_close(accum, expected, atol=2e-6, rtol=2e-3)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(2):
            accum[:, :, start:end].fill_(99)
            graph.replay()
            torch.testing.assert_close(accum, expected, atol=2e-6, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "heads,kv_heads,dim", [(8, 2, 128), (4, 4, 64), (28, 4, 128), (32, 1, 128)]
)
@pytest.mark.parametrize("window", [32, 137])
@pytest.mark.parametrize("reuse_lse", [False, True])
def test_probability_scores_across_context_lengths(
    heads, kv_heads, dim, window, reuse_lse
):
    # Existing tests check individual shapes. Reusing one workspace while widths
    # grow/shrink catches wrong runtime strides/masks, including head reduction
    # for multi-tile observations and changes to the denominator reduction tile.
    torch.manual_seed(53)
    workspace = PrefillScoreWorkspace()
    for length in (257, 319, 513, 385):
        contexts, chunks = (length, 23), (141, 17)
        q = (
            torch.randn(heads, sum(chunks), dim, device="cuda", dtype=torch.bfloat16)
            * 0.2
        ).transpose(0, 1)
        k = torch.randn(sum(contexts), kv_heads, dim * 2, device="cuda", dtype=q.dtype)
        k.mul_(0.2)
        k = k[..., :dim]
        starts = torch.tensor([0, chunks[0]], device="cuda", dtype=torch.int32)
        cached = torch.tensor(
            [n - c for n, c in zip(contexts, chunks)], device="cuda", dtype=torch.int32
        )
        lens = torch.tensor(contexts, device="cuda", dtype=torch.int32)
        score_starts = torch.tensor(
            [n - min(window, c) for n, c in zip(contexts, chunks)],
            device="cuda",
            dtype=torch.int32,
        )
        req_rows = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
        mapping = torch.full((2, length), -1, device="cuda", dtype=torch.int32)
        permutation = torch.randperm(sum(contexts), device="cuda").int()
        mapping[1, :length] = permutation[:length]
        mapping[0, : contexts[1]] = permutation[length:]
        # A sliced output need not have a batch stride equal to its logical width.
        output = torch.empty((2, length + 7), device="cuda", dtype=torch.float32)[
            :, :length
        ]
        expected = torch.zeros_like(output)
        lse = torch.empty((heads, sum(chunks)), device="cuda", dtype=torch.float32)
        offset = 0
        for i, (context, chunk) in enumerate(zip(contexts, chunks)):
            logical_k = k[mapping[1 - i, :context].long()].repeat_interleave(
                heads // kv_heads, dim=1
            )
            # FP64 reference avoids TF32 settings affecting the independent oracle.
            z = (
                torch.einsum(
                    "qhd,khd->hqk",
                    q[offset : offset + chunk].double(),
                    logical_k.double(),
                )
                * dim**-0.5
            )
            qi = torch.arange(context - chunk, context, device="cuda")
            ki = torch.arange(context, device="cuda")
            z.masked_fill_(ki[None, None] > qi[None, :, None], -torch.inf)
            lse[:, offset : offset + chunk] = z.logsumexp(-1).float()
            observed = z[:, -min(window, chunk) :]
            if not reuse_lse:
                # The shorter request has no eligible keys; its scores must be zero.
                observed = observed.masked_fill(
                    ((ki < 31) | (ki >= context - 7))[None, None], -torch.inf
                )
            probabilities = observed.softmax(-1).nan_to_num(0)
            expected[i, :context] = probabilities.mean(1).amax(0).float()
            offset += chunk
        common = (
            output,
            req_rows,
            starts,
            lens,
            cached,
            min(window, max(chunks)),
            mapping,
            score_starts,
            lens,
        )
        if reuse_lse:
            prefill_score_from_lse_fwd(q, k, lse, *common, workspace=workspace)
        else:
            prefill_score_fwd(
                q,
                k,
                *common,
                candidate_start=31,
                recent_keep_tokens=7,
                workspace=workspace,
            )
        torch.testing.assert_close(output, expected, atol=2e-5, rtol=0.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("reuse_lse", [False, True])
def test_probability_scores_with_different_block_logit_scales(reuse_lse):
    # Block LSEs must combine correctly when their maxima differ substantially.
    # A padded GQA head group and multiple Q tiles also exercise head reduction.
    torch.manual_seed(71)
    length, chunk, window, heads, kv_heads, dim = 513, 141, 137, 28, 4, 128
    q = torch.randn((chunk, heads, dim), device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn((length, kv_heads, dim), device="cuda", dtype=q.dtype) * 0.1
    q[..., 0] = torch.linspace(-1, 1, chunk, device="cuda")[:, None]
    k[..., 0] = (
        torch.linspace(-80, 80, length, device="cuda")[:, None] * dim**0.5
    )
    expanded_k = k.double().repeat_interleave(heads // kv_heads, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.double(), expanded_k) * dim**-0.5
    keys = torch.arange(length, device="cuda")
    queries = torch.arange(length - chunk, length, device="cuda")
    logits.masked_fill_(keys[None, None] > queries[None, :, None], -torch.inf)
    lse = logits.logsumexp(-1).float()
    observed = logits[:, -window:]
    if not reuse_lse:
        observed = observed.masked_fill(
            ((keys < 31) | (keys >= length - 7))[None, None], -torch.inf
        )
    expected = observed.softmax(-1).mean(1).amax(0).float()[None]
    output = torch.empty_like(expected)
    index = torch.zeros(1, device="cuda", dtype=torch.int32)
    seq_len = torch.tensor([length], device="cuda", dtype=torch.int32)
    common = (
        output,
        index,
        index,
        seq_len,
        seq_len - chunk,
        window,
        keys.int()[None],
        seq_len - window,
        seq_len,
    )
    if reuse_lse:
        prefill_score_from_lse_fwd(q, k, lse, *common)
    else:
        prefill_score_fwd(q, k, *common, candidate_start=31, recent_keep_tokens=7)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=0.002)
