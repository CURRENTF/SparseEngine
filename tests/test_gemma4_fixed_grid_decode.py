from __future__ import annotations

import pytest
import torch

from sparseengine.kernels.triton.sglang_gemma4_decode_attention import (
    sglang_gemma4_decode,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference(q, k, v, slots, request_indices, lengths, window):
    output = torch.empty_like(q)
    group_size = q.shape[1] // k.shape[1]
    for batch_idx, length in enumerate(lengths.tolist()):
        start = max(0, length - int(window or length))
        active = slots[request_indices[batch_idx], start:length].long()
        keys = k[active].repeat_interleave(group_size, dim=1)
        values = v[active].repeat_interleave(group_size, dim=1)
        logits = torch.einsum("hd,lhd->hl", q[batch_idx].float(), keys.float())
        output[batch_idx] = torch.einsum(
            "hl,lhd->hd",
            torch.softmax(logits, dim=-1),
            values.float(),
        ).to(q.dtype)
    return output


def _case(*, dtype, head_dim, query_heads, kv_heads, capacity, lengths, window):
    batch = len(lengths)
    torch.manual_seed(20260825 + head_dim + query_heads + capacity)
    q = torch.randn(
        batch,
        query_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    ).mul_(0.25)
    slots = torch.arange(
        batch * capacity,
        dtype=torch.int32,
        device="cuda",
    ).view(batch, capacity)
    k = torch.randn(
        batch * capacity,
        kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    ).mul_(0.25)
    v = torch.randn_like(k)
    request_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    context_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    mid_output = torch.empty(
        batch,
        query_heads,
        8,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    )
    mid_lse = torch.empty(
        batch,
        query_heads,
        8,
        dtype=torch.float32,
        device="cuda",
    )
    num_kv_splits = torch.empty(batch, dtype=torch.int32, device="cuda")
    return (
        q,
        k,
        v,
        slots,
        request_indices,
        context_lens,
        mid_output,
        mid_lse,
        num_kv_splits,
        window,
    )


def _run(case, *, score=None):
    q, k, v, slots, request_indices, lengths, mid, lse, splits, window = case
    return sglang_gemma4_decode(
        q,
        k,
        v,
        slots,
        request_indices,
        lengths,
        mid,
        lse,
        splits,
        sliding_window=window,
        multi_processor_count=torch.cuda.get_device_properties(0).multi_processor_count,
        attn_score=score,
    )


@pytest.mark.parametrize(
    "case_kwargs",
    [
        dict(
            dtype=torch.bfloat16,
            head_dim=256,
            query_heads=2,
            kv_heads=2,
            capacity=21,
            lengths=[21, 13],
            window=None,
        ),
        dict(
            dtype=torch.float16,
            head_dim=256,
            query_heads=8,
            kv_heads=2,
            capacity=1301,
            lengths=[1301, 1177],
            window=1024,
        ),
        dict(
            dtype=torch.bfloat16,
            head_dim=512,
            query_heads=8,
            kv_heads=1,
            capacity=513,
            lengths=[513, 377],
            window=None,
        ),
        dict(
            dtype=torch.bfloat16,
            head_dim=512,
            query_heads=16,
            kv_heads=1,
            capacity=8193,
            lengths=[8193],
            window=None,
        ),
    ],
)
def test_gemma4_fixed_grid_decode_matches_independent_oracle(case_kwargs):
    case = _case(**case_kwargs)
    actual = _run(case)
    q, k, v, slots, request_indices, lengths, *_, window = case
    expected = _reference(q, k, v, slots, request_indices, lengths, window)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("score_rank", [2, 3])
def test_gemma4_fixed_grid_decode_produces_raw_qk_scores(score_rank):
    case = _case(
        dtype=torch.bfloat16,
        head_dim=256,
        query_heads=4,
        kv_heads=2,
        capacity=33,
        lengths=[33, 21],
        window=None,
    )
    q, k, _, slots, request_indices, lengths, *_ = case
    score = torch.full(
        (2, 4, 33) if score_rank == 3 else (2, 33),
        -1e20,
        dtype=torch.float32,
        device="cuda",
    )
    _run(case, score=score)
    expected = torch.full(
        (2, 4, 33),
        -1e20,
        dtype=torch.float32,
        device="cuda",
    )
    group_size = q.shape[1] // k.shape[1]
    for batch_idx, length in enumerate(lengths.tolist()):
        active = slots[request_indices[batch_idx], :length].long()
        keys = k[active].repeat_interleave(group_size, dim=1)
        expected[batch_idx, :, :length] = torch.einsum(
            "hd,lhd->hl",
            q[batch_idx].float(),
            keys.float(),
        )
    if score_rank == 2:
        expected = expected.max(dim=1).values
    torch.testing.assert_close(score, expected, rtol=2e-2, atol=1.0)


@pytest.mark.parametrize(
    ("head_dim", "window"),
    [(256, None), (256, 16), (512, None)],
)
def test_gemma4_fixed_grid_decode_replays_new_lengths_and_rows(head_dim, window):
    case = _case(
        dtype=torch.bfloat16,
        head_dim=head_dim,
        query_heads=4,
        kv_heads=2,
        capacity=33,
        lengths=[33, 21],
        window=window,
    )
    _run(case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = _run(case)

    q, k, v, slots, request_indices, lengths, *_, window = case
    q.copy_(torch.randn_like(q))
    request_indices.copy_(torch.tensor([1, 0], dtype=torch.int32, device="cuda"))
    lengths.copy_(torch.tensor([17, 29], dtype=torch.int32, device="cuda"))
    expected = _reference(q, k, v, slots, request_indices, lengths, window)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected, rtol=3e-2, atol=3e-2)
