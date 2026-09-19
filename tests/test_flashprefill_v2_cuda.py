import pytest
import torch

from sparseengine.kernels.external.flashprefill_v2.prefill import (
    make_flashprefill_v2,
)
from sparseengine.operators.prefill_attention import FlashPrefillV2Semantics


def _dense_varlen_reference(
    q,
    k_cache,
    v_cache,
    page_table,
    cache_lens,
    q_lens,
    scale,
):
    outputs = []
    q_offset = 0
    num_query_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    group_size = num_query_heads // num_kv_heads
    for row, (q_len, cache_len) in enumerate(zip(q_lens, cache_lens.tolist())):
        request_q = q[q_offset : q_offset + q_len].float()
        slots = page_table[row, :cache_len].to(torch.long)
        request_k = k_cache[slots, 0].float().repeat_interleave(group_size, dim=1)
        request_v = v_cache[slots, 0].float().repeat_interleave(group_size, dim=1)
        logits = torch.einsum("qhd,khd->hqk", request_q, request_k) * scale
        prefix_len = int(cache_len) - int(q_len)
        q_positions = prefix_len + torch.arange(q_len, device=q.device)
        k_positions = torch.arange(int(cache_len), device=q.device)
        logits = logits.masked_fill(
            k_positions[None, None, :] > q_positions[None, :, None],
            float("-inf"),
        )
        probs = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, request_v))
        q_offset += int(q_len)
    return torch.cat(outputs).to(q.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashprefill_v2_dense_index_matches_varlen_paged_oracle():
    pytest.importorskip("flashprefill")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("The validated FlashPrefill V2 build targets SM90.")

    torch.manual_seed(23)
    device = torch.device("cuda")
    q_lens = (192, 129)
    cache_lens = torch.tensor([257, 193], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, q_lens[0], sum(q_lens)],
        dtype=torch.int32,
        device=device,
    )
    q = torch.randn(sum(q_lens), 8, 128, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(768, 1, 2, 128, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn_like(k_cache)
    page_table = torch.stack(
        (
            torch.randperm(768, device=device)[:257],
            torch.randperm(768, device=device)[:257],
        )
    ).to(torch.int32)
    scale = 128**-0.5
    pipeline = make_flashprefill_v2(
        semantics=FlashPrefillV2Semantics(
            k_block_m=128,
            k_block_n=128,
            abs_threshold=0.1,
            attention_sink_blocks=2,
            window_blocks=4,
            last_query_blocks=16,
            use_mean_correction=False,
        ),
        softmax_scale=scale,
    )

    actual = pipeline(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        cu_seqlens_q,
        q_lens=q_lens,
        max_cache_seqlen=int(cache_lens.max().item()),
        softmax_scale=scale,
    )
    expected = _dense_varlen_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        q_lens,
        scale,
    )

    torch.testing.assert_close(actual, expected, rtol=3.0e-2, atol=3.0e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashprefill_v2_sparse_index_all_blocks_matches_prefix_oracle():
    pytest.importorskip("flashprefill")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("The validated FlashPrefill V2 build targets SM90.")

    torch.manual_seed(29)
    device = torch.device("cuda")
    q_lens = (4096,)
    cache_lens = torch.tensor([4224], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 4096], dtype=torch.int32, device=device)
    q = torch.randn(4096, 8, 128, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(4352, 1, 2, 128, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn_like(k_cache)
    page_table = (
        torch.randperm(4352, device=device)[:4224]
        .unsqueeze(0)
        .to(torch.int32)
    )
    scale = 128**-0.5
    pipeline = make_flashprefill_v2(
        semantics=FlashPrefillV2Semantics(
            k_block_m=128,
            k_block_n=128,
            abs_threshold=0.0,
            attention_sink_blocks=0,
            window_blocks=0,
            last_query_blocks=0,
            min_sparse_q_len=4096,
            use_mean_correction=False,
        ),
        softmax_scale=scale,
    )

    sparse_index = pipeline.index_select(
        q,
        k_cache,
        page_table,
        cache_lens,
        cu_seqlens_q,
        v_cache=v_cache,
        q_lens=q_lens,
        max_cache_seqlen=4224,
        softmax_scale=scale,
    )
    assert sparse_index.block_sparse_idx.numel() > 0

    actual = pipeline(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        cu_seqlens_q,
        q_lens=q_lens,
        max_cache_seqlen=4224,
        softmax_scale=scale,
    )
    expected = _dense_varlen_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        q_lens,
        scale,
    )

    torch.testing.assert_close(actual, expected, rtol=3.0e-2, atol=3.0e-2)
