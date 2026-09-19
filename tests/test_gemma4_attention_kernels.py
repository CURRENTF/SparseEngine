from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.cache_manager.base import ExplicitKVPayload
from sparseengine.kernels.triton.gemma4_context_attention import (
    gemma4_context_attention,
)
from sparseengine.operators.gemma4_attention import Gemma4FlashInferPrefill
from sparseengine.utils.context import reset_context, set_context


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    ("head_dim", "q_heads", "kv_heads", "sliding_window", "sparse_history"),
    [(256, 4, 2, 32, False), (512, 4, 1, None, False),
     (512, 4, 1, None, True)],
)
def test_gemma4_flashinfer_prefill_matches_torch(
    head_dim, q_heads, kv_heads, sliding_window, sparse_history
):
    pytest.importorskip("flashinfer")
    torch.manual_seed(20260813)
    prefix, chunk, length = 11, 54, 65
    original_prefix = 2 * prefix if sparse_history else prefix
    original_length = original_prefix + chunk
    # Global sparse prefill must preserve the causal current tail when history
    # is noncontiguous in original token positions, including shared-KV reads.
    positions = torch.cat((
        torch.arange(0, original_prefix, 2 if sparse_history else 1, device="cuda"),
        torch.arange(original_prefix, original_length, device="cuda"),
    ))
    slots = torch.randperm(original_length, device="cuda")[positions].to(torch.int32)
    key = torch.randn(original_length, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn(chunk, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    view = SimpleNamespace(
        payload=ExplicitKVPayload(key, value),
        meta=SimpleNamespace(
            active_slots=slots.view(1, -1),
            req_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
            context_lens=torch.tensor([length], device="cuda", dtype=torch.int32),
            attn_score=None,
        ),
    )
    reset_context()
    set_context(
        True,
        cu_seqlens_q=torch.tensor([0, chunk], device="cuda", dtype=torch.int32),
    )
    prefill = Gemma4FlashInferPrefill()
    try:
        prefill.prepare()
        output = prefill.run(
            query,
            view,
            q_start=torch.zeros(1, device="cuda", dtype=torch.int32),
            chunk_lens=torch.tensor([chunk], device="cuda", dtype=torch.int32),
            max_context_len=length,
            sliding_window=sliding_window,
        )
    finally:
        prefill.close()
        reset_context()

    kv_head_ids = torch.arange(q_heads, device="cuda") // (q_heads // kv_heads)
    logical_key, logical_value = key[slots.long()], value[slots.long()]
    logits = torch.einsum(
        "qhd,khd->hqk", query, logical_key[:, kv_head_ids]
    ).float()
    query_positions = original_prefix + torch.arange(chunk, device="cuda")
    key_positions = positions
    visible = key_positions[None] <= query_positions[:, None]
    if sliding_window is not None:
        visible &= key_positions[None] > query_positions[:, None] - sliding_window
    probabilities = logits.masked_fill(~visible[None], -torch.inf).softmax(-1)
    reference = torch.einsum(
        "hqk,khd->qhd",
        probabilities.to(value.dtype),
        logical_value[:, kv_head_ids],
    )
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )
    assert torch.isfinite(output).all()
    assert cosine > 0.999


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("sliding_window", [None, 4])
def test_gemma4_prefill_matches_torch(head_dim, sliding_window):
    torch.manual_seed(3)
    prefix = torch.tensor([4, 2], dtype=torch.int32, device="cuda")
    chunks = torch.tensor([5, 3], dtype=torch.int32, device="cuda")
    lengths = prefix + chunks
    starts = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    slots = torch.zeros((2, 9), dtype=torch.int32, device="cuda")
    slots[0, :9] = torch.arange(9, dtype=torch.int32, device="cuda")
    slots[1, :5] = torch.arange(9, 14, dtype=torch.int32, device="cuda")
    key = torch.randn(14, 2, head_dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    query = torch.randn(8, 4, head_dim, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)

    gemma4_context_attention(
        query,
        key,
        value,
        output,
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        starts,
        lengths,
        prefix,
        5,
        slots,
        sliding_window=sliding_window,
    )
    reference = torch.empty_like(output)
    for batch, (prefix_len, chunk_len, start) in enumerate(
        zip(prefix.tolist(), chunks.tolist(), starts.tolist())
    ):
        for offset in range(chunk_len):
            end = prefix_len + offset + 1
            begin = max(0, end - (sliding_window or end))
            indices = slots[batch, begin:end].long()
            for head in range(query.shape[1]):
                logits = query[start + offset, head] @ key[indices, head // 2].T
                reference[start + offset, head] = logits.softmax(-1) @ value[
                    indices, head // 2
                ]
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )
    assert torch.isfinite(output).all()
    assert cosine > 0.999


def test_gemma4_long_window_prefill_matches_torch():
    torch.manual_seed(5)
    prefix = torch.tensor([256], dtype=torch.int32, device="cuda")
    chunks = torch.tensor([1088], dtype=torch.int32, device="cuda")
    lengths = prefix + chunks
    slots = torch.arange(1344, dtype=torch.int32, device="cuda").view(1, -1)
    key = torch.randn(1344, 2, 256, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    query = torch.randn(1088, 4, 256, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)
    gemma4_context_attention(
        query,
        key,
        value,
        output,
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        lengths,
        prefix,
        1088,
        slots,
        sliding_window=1024,
    )
    kv_heads = torch.arange(4, device="cuda") // 2
    logits = torch.bmm(
        query.permute(1, 0, 2),
        key[:, kv_heads].permute(1, 2, 0),
    ).float()
    query_positions = 256 + torch.arange(1088, device="cuda")
    key_positions = torch.arange(1344, device="cuda")
    visible = (key_positions[None] <= query_positions[:, None]) & (
        key_positions[None] > query_positions[:, None] - 1024
    )
    reference = torch.bmm(
        logits.masked_fill(~visible, -torch.inf).softmax(-1).to(value.dtype),
        value[:, kv_heads].permute(1, 0, 2),
    ).permute(1, 0, 2)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )
    assert torch.isfinite(output).all()
    assert cosine > 0.999


@pytest.mark.parametrize("score_rank", [2, 3])
def test_gemma4_prefill_collects_raw_qk_scores(score_rank):
    torch.manual_seed(17)
    head_dim = 256
    lengths = torch.tensor([4], dtype=torch.int32, device="cuda")
    slots = torch.arange(4, dtype=torch.int32, device="cuda").unsqueeze(0)
    key = torch.randn(4, 2, head_dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    query = torch.randn(4, 4, head_dim, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)
    score = torch.zeros(
        (1, 4, 4) if score_rank == 3 else (1, 4),
        dtype=torch.float32,
        device="cuda",
    )
    gemma4_context_attention(
        query,
        key,
        value,
        output,
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        lengths,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        4,
        slots,
        sliding_window=None,
        attn_score=score,
    )
    expected = torch.zeros(1, 4, 4, dtype=torch.float32, device="cuda")
    for head in range(4):
        expected[0, head] = (
            query[:, head].float() @ key[:, head // 2].float().T
        ).tril().sum(0)
    if score_rank == 2:
        expected = (expected / 4).max(1).values.clamp_min_(0)
    torch.testing.assert_close(score, expected, rtol=2e-2, atol=1.0)
