from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.engine.cache_manager import (
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
)
from sparseengine.kernels.external.sgl.fa3 import sgl_fa3_support
from sparseengine.kernels.triton.flash_decoding_stage2 import flash_decode_stage2
from sparseengine.kernels.triton.gqa_flash_decoding_stage1 import flash_decode_stage1
from sparseengine.kernels.triton.store_kvcache import store_kvcache
from sparseengine.operators.decode_attention import (
    DecodeAttentionOpSpec,
    SglFa3PagedDecodeAttentionProvider,
    prepare_decode_attention_op,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_minimax_m2_gqa_decode_cuda_graph_replay():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(27)
    batch_size = 4
    num_q_heads = 48
    num_kv_heads = 8
    head_dim = 128
    block_seq = 256
    max_context_len = 1024
    num_blocks = max_context_len // block_seq

    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    k = torch.randn(
        160,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    v = torch.randn(
        160,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    req_to_tokens = torch.zeros(8, max_context_len, dtype=torch.int32, device=device)
    for row in range(8):
        req_to_tokens[row, :18] = torch.arange(
            row * 18,
            (row + 1) * 18,
            dtype=torch.int32,
            device=device,
        )
    req_indices = torch.arange(4, dtype=torch.int32, device=device)
    context_lens = torch.full((batch_size,), 17, dtype=torch.int32, device=device)
    slot_mapping = torch.tensor([16, 34, 52, 70], dtype=torch.int32, device=device)
    new_k = torch.randn(
        batch_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    new_v = torch.randn(
        new_k.shape,
        device=device,
        dtype=new_k.dtype,
        generator=generator,
    )
    mid_o = torch.empty(
        batch_size,
        num_q_heads,
        num_blocks,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    mid_lse = torch.empty(
        batch_size,
        num_q_heads,
        num_blocks,
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty_like(q)

    def run_decode():
        store_kvcache(new_k, new_v, k, v, slot_mapping)
        flash_decode_stage1(
            q,
            k,
            v,
            req_to_tokens,
            req_indices,
            context_lens,
            max_context_len,
            mid_o,
            mid_lse,
            block_seq,
        )
        flash_decode_stage2(mid_o, mid_lse, context_lens, output, block_seq)

    for _ in range(2):
        run_decode()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_decode()

    for context_len in (17, 18):
        req_indices.copy_(torch.arange(4, 8, dtype=torch.int32, device=device))
        q.copy_(torch.randn(q.shape, device=device, dtype=q.dtype, generator=generator))
        new_k.copy_(torch.randn(new_k.shape, device=device, dtype=new_k.dtype, generator=generator))
        new_v.copy_(torch.randn(new_v.shape, device=device, dtype=new_v.dtype, generator=generator))
        context_lens.fill_(context_len)
        slot_mapping.copy_(
            torch.tensor(
                [context_len - 1 + row * 18 for row in range(4, 8)],
                dtype=torch.int32,
                device=device,
            )
        )
        k.index_fill_(0, slot_mapping.to(torch.long), float("nan"))
        v.index_fill_(0, slot_mapping.to(torch.long), float("nan"))
        graph.replay()
        graph_output = output.clone()
        k.index_fill_(0, slot_mapping.to(torch.long), float("nan"))
        v.index_fill_(0, slot_mapping.to(torch.long), float("nan"))
        run_decode()
        torch.cuda.synchronize()
        assert torch.equal(graph_output, output)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_minimax_m2_production_provider_replays_across_32k_boundary():
    torch.manual_seed(20260825)
    device = torch.device("cuda")
    query_heads, kv_heads, head_dim = 12, 2, 128
    capacity = 32769
    spec = DecodeAttentionOpSpec(
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        activation_dtype=torch.bfloat16,
        softmax_scale=head_dim**-0.5,
        max_batch_size=1,
        context_capacity=capacity,
    )
    prepared = prepare_decode_attention_op(spec, device_index=device.index or 0)
    assert isinstance(prepared.provider, SglFa3PagedDecodeAttentionProvider)

    q = 0.25 * torch.randn(
        1,
        query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = 0.25 * torch.randn(
        capacity,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    active_slots = torch.arange(
        capacity,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    context_lens = torch.tensor([32767], dtype=torch.int32, device=device)
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=active_slots,
            req_indices=torch.zeros(1, dtype=torch.int32, device=device),
            context_lens=context_lens,
            max_context_len=capacity,
        ),
        payload=ExplicitKVPayload(k_cache=k_cache, v_cache=v_cache),
    )

    launch_profile = Mock(name="context_dependent_launch_profile")
    validation_scope = object()
    with patch(
        "sparseengine.operators.decode_attention.get_context",
        return_value=SimpleNamespace(attention_validation_scope=validation_scope),
    ):
        prepared.run(q, view, decode_launch_op=launch_profile)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = prepared.run(q, view, decode_launch_op=launch_profile)

    for context_len in (32767, 32768, 32769):
        context_lens.fill_(context_len)
        graph.replay()
        torch.cuda.synchronize()

        active = active_slots[0, :context_len].long()
        group_size = query_heads // kv_heads
        expanded_k = k_cache[active].repeat_interleave(group_size, dim=1)
        expanded_v = v_cache[active].repeat_interleave(group_size, dim=1)
        logits = torch.einsum(
            "hd,lhd->hl",
            q[0].float(),
            expanded_k.float(),
        )
        probabilities = torch.softmax(logits * spec.softmax_scale, dim=-1)
        expected = torch.einsum(
            "hl,lhd->hd",
            probabilities,
            expanded_v.float(),
        ).to(torch.bfloat16)
        torch.testing.assert_close(
            graph_output[0],
            expected,
            rtol=3e-2,
            atol=3e-2,
        )

    launch_profile.launch_config.assert_not_called()
    prepared.close()
