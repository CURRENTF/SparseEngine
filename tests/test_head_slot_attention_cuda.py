"""Independent position-masked oracle for virtual MQA addressing and replay."""

import os

import pytest
import torch

from sparseengine.engine.cache_manager.base import (
    DecodeComputeView, ExplicitKVPayload, HeadSlotViewMeta, PrefillComputeView,
)
from sparseengine.operators.decode_attention import DecodeAttentionOpSpec, prepare_decode_attention_op
from sparseengine.operators.prefill_attention import PrefillAttentionOpSpec, prepare_prefill_attention_op
from sparseengine.engine.decode_graph_contract import DecodeGraphContract, DecodeGraphInputs


pytestmark = pytest.mark.skipif(os.environ.get("SPARSEENGINE_TEST_CUDA") != "1",
                                reason="Set SPARSEENGINE_TEST_CUDA=1 on an idle reserved GPU")


def test_ragged_head_writes_and_metadata_do_not_recompile_per_token_count():
    """SWE requests previously exhausted the guard with hundreds of length variants."""
    from sparseengine.kernels.triton.head_slots import publish_head_slots, store_head_slots, record_head_queries
    from sparseengine.utils.compilation_guard import RuntimeCompilationGuard
    h,d,width=2,256,512
    kp,vp=[torch.full((width*h,d),-7.,dtype=torch.bfloat16,device='cuda') for _ in range(2)]
    table=torch.zeros((1,2,h,width),dtype=torch.int32,device='cuda')
    positions=torch.full_like(table,-1)
    ring=torch.zeros((2,64,h*8,d),dtype=torch.bfloat16,device='cuda')
    def run(n):
        slots=torch.arange(n*h,dtype=torch.int32,device='cuda').reshape(n,h)
        columns=torch.arange(n,dtype=torch.int32,device='cuda')[None,:,None].expand(1,n,h).contiguous()
        rows=torch.zeros(n,dtype=torch.int32,device='cuda')
        pos=torch.arange(n,dtype=torch.int64,device='cuda')
        k,v=[torch.randn(n,h,d,dtype=kp.dtype,device='cuda') for _ in range(2)]
        q=torch.randn(n,h*8,d,dtype=kp.dtype,device='cuda')
        keep=torch.arange(n,device='cuda')>=max(0,n-64)
        publish_head_slots(table,positions,slots[None],columns,rows,pos)
        store_head_slots(k,v,kp,vp,slots)
        record_head_queries(q,ring,rows,pos,keep)
        torch.testing.assert_close(kp[:n*h].reshape_as(k),k,atol=0,rtol=0)
        torch.testing.assert_close(vp[:n*h].reshape_as(v),v,atol=0,rtol=0)
        for head in range(h):
            torch.testing.assert_close(table[0,0,head,:n],slots[:,head],atol=0,rtol=0)
            torch.testing.assert_close(positions[0,0,head,:n],pos.int(),atol=0,rtol=0)
        for position in range(max(0,n-64),n):
            torch.testing.assert_close(ring[0,position%64],q[position],atol=0,rtol=0)
    run(32)
    guard=RuntimeCompilationGuard(limit=0,rank=0)
    guard.arm()
    try:
        for n in (1,3,17,65,129,257):
            run(n)
        assert guard.count==0
    finally:
        guard.close()


def test_head_slot_writes_preserve_kv_pairs_and_ignore_padding():
    """Strided projection outputs must reach the same independent K/V address."""
    from sparseengine.kernels.triton.head_slots import store_head_slots

    torch.manual_seed(31)
    k, v = [torch.randn((5, 6, 256), device="cuda", dtype=torch.bfloat16)[:, ::2, ::2]
            for _ in range(2)]
    slots = torch.tensor([[2, 11, 7], [3, 9, 1], [5, 14, 13],
                          [4, 8, 10], [-1, -1, -1]], dtype=torch.int32, device="cuda")
    kp, vp = [torch.full((17, 128), -7, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    expected = [p.clone() for p in (kp, vp)]
    for t in range(4):
        for h in range(3):
            expected[0][slots[t, h]] = k[t, h]
            expected[1][slots[t, h]] = v[t, h]
    store_head_slots(k, v, kp, vp, slots)
    for actual, reference in zip((kp, vp), expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def oracle(q, k, v, table, positions, lens, rows, cu, query_positions):
    result = torch.zeros_like(q)
    h = table.shape[1]
    g = q.shape[1] // h
    for b, row in enumerate(rows.tolist()):
        if row < 0:
            continue
        for head in range(h):
            n = int(lens[b, head])
            slots = table[row, head, :n].long()
            key, value = k[slots].float(), v[slots].float()
            for token in range(int(cu[b]), int(cu[b + 1])):
                scores = q[token, head * g:(head + 1) * g].float() @ key.T / q.shape[-1]**0.5
                scores[:, positions[row, head, :n] > query_positions[token]] = -torch.inf
                result[token, head * g:(head + 1) * g] = torch.softmax(scores, -1) @ value
    return result


@pytest.mark.parametrize("history,groups,chunks", [(1, 1, [1, 1]), (7, 4, [3, 2]),
                                                    (65, 7, [5, 1]), (2048, 4, [4, 3])])
def test_head_attention_matches_position_oracle(history, groups, chunks):
    capability = torch.cuda.get_device_capability()
    if capability not in {(9, 0), (12, 0)}:
        pytest.skip('Current head-slot prefill providers require SM90 or SM120')
    torch.manual_seed(17)
    device, h, d, b = torch.device("cuda"), 2, 128, 2
    width = history + max(chunks)
    table = torch.zeros((b + 1, h, width), dtype=torch.int32, device=device)
    positions = torch.full_like(table, -1)
    lens = torch.zeros((b, h), dtype=torch.int32, device=device)
    pool_count = b * h * width + 11
    k, v = [torch.randn((pool_count, d), dtype=torch.bfloat16, device=device) for _ in range(2)]
    free = torch.randperm(pool_count, device=device).to(torch.int32)
    offset, qpositions = 0, []
    for row, chunk in enumerate(chunks):
        for head in range(h):
            old = max(1, history - head * max(1, history // 3) - row)
            n = old + chunk
            table[row, head, :n] = free[offset:offset + n]
            offset += n
            positions[row, head, :old] = torch.arange(old, device=device) * 2
            positions[row, head, old:n] = torch.arange(history * 2, history * 2 + chunk, device=device)
            lens[row, head] = n
        qpositions.extend(range(history * 2, history * 2 + chunk))
    rows = torch.arange(b, dtype=torch.int32, device=device)
    cu = torch.tensor([0, chunks[0], sum(chunks)], dtype=torch.int32, device=device)
    # A strided Q projection catches reshapes that silently assume contiguous tokens/heads.
    q = torch.randn((sum(chunks), h * groups * 2, d), dtype=torch.bfloat16, device=device)[:, ::2]
    meta = HeadSlotViewMeta(table, rows, lens, positions, width)
    payload = ExplicitKVPayload(k, v, backend="head_slots")
    shared = dict(num_query_heads=h * groups, num_kv_heads=h, head_dim=d,
                  activation_dtype=torch.bfloat16, softmax_scale=d**-0.5,
                  kv_cache_layout="head_slots", layer_varying_page_table=True)
    prefill = prepare_prefill_attention_op(PrefillAttentionOpSpec(**shared))
    expected = oracle(q, k, v, table, positions, lens, rows, cu,
                      torch.tensor(qpositions, device=device))
    actual = prefill.run(q, PrefillComputeView(meta, payload), qo_indptr=cu,
                         chunk_lens=cu[1:] - cu[:-1], max_context_len=width, layer_idx=0)
    # BF16 attention rounds the final weighted sum; FP32 oracle permits 2 BF16
    # ULP near unit magnitude plus reduction error. Set before the first run.
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    decode = prepare_decode_attention_op(DecodeAttentionOpSpec(**shared, max_batch_size=b,
                                                             cuda_graph=capability == (9, 0)))
    query = q[cu[1:].long() - 1].clone()
    decode_cu = torch.arange(b + 1, device=device)
    decode_pos = torch.tensor([history * 2 + chunk - 1 for chunk in chunks], device=device)
    expected_decode = oracle(query, k, v, table, positions, lens, rows, decode_cu, decode_pos)
    actual_decode = decode.run(query, DecodeComputeView(meta, payload))
    torch.testing.assert_close(actual_decode, expected_decode, atol=0.02, rtol=0.02)
    # Reuse the same provider and scope with a different layer's addresses and
    # head lengths. FlashInfer must plan the new view instead of reusing layer 0.
    table = table.roll(1, dims=1).contiguous()
    positions = positions.roll(1, dims=1).contiguous()
    lens = lens.roll(1, dims=1).contiguous()
    meta = HeadSlotViewMeta(table, rows, lens, positions, width)
    expected = oracle(q, k, v, table, positions, lens, rows, cu,
                      torch.tensor(qpositions, device=device))
    actual = prefill.run(q, PrefillComputeView(meta, payload), qo_indptr=cu,
                        chunk_lens=cu[1:] - cu[:-1], max_context_len=width, layer_idx=1)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    expected_decode = oracle(query, k, v, table, positions, lens, rows, decode_cu, decode_pos)
    actual_decode = decode.run(query, DecodeComputeView(meta, payload))
    torch.testing.assert_close(actual_decode, expected_decode, atol=0.02, rtol=0.02)
    if capability != (9, 0):
        prefill.close()
        decode.close()
        return  # The dedicated lifecycle test below covers FlashInfer graphs.
    # Real capture/replay, with changed addresses, lengths, and inactive rows.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode.run(query, DecodeComputeView(meta, payload))
    graph.replay()
    torch.testing.assert_close(captured, expected_decode, atol=0.02, rtol=0.02)
    rows[1] = -1
    lens[1] = 0
    lens[0] -= 1
    graph.replay()
    expected_replay = oracle(query, k, v, table, positions, lens, rows, decode_cu, decode_pos)
    torch.testing.assert_close(captured, expected_replay, atol=0.02, rtol=0.02)
    prefill.close()
    decode.close()


@pytest.mark.parametrize("groups,dim,capacity", [(1, 128, 4096), (4, 128, 4096),
                                                (7, 128, 4096), (4, 64, 128),
                                                (4, 256, 4096)])
def test_head_graph_replays_layer_ragged_lengths_and_request_churn(groups, dim, capacity):
    """One graph must read new per-layer/head lengths, including empty padding.

    The eager tests cannot catch a captured planner ignoring dynamic indptr or
    retaining an old layer's length, nor a workspace replacement after capture.
    """
    torch.manual_seed(82)
    capability = torch.cuda.get_device_capability()
    if capability not in {(9, 0), (12, 0)} or (capability == (9, 0) and dim == 64):
        pytest.skip("Current head-slot providers require SM90 D128/256 or SM120 D64/128/256")
    device, batch, heads = torch.device("cuda"), 2, 2
    spec = DecodeAttentionOpSpec(
        num_query_heads=heads * groups, num_kv_heads=heads, head_dim=dim,
        activation_dtype=torch.bfloat16, softmax_scale=dim**-0.5,
        kv_cache_layout="head_slots", layer_varying_page_table=True,
        cuda_graph=True, max_batch_size=batch, context_capacity=capacity,
    )
    contract = DecodeGraphContract("snapkv", "unified", batch, capacity)
    inputs = DecodeGraphInputs.allocate(contract, device=device, pin_memory=True)
    decode = prepare_decode_attention_op(spec)
    state = decode.init_decode_graph_state(contract, inputs) if decode.decode_graph_lifecycle else None
    if state is not None:
        # A reused integer workspace may contain arbitrary bytes. This catches
        # FA2 no-split graph plans that launch padded CTAs without a valid mask.
        int_workspace = torch.full((8 * 1024 * 1024,), 255, dtype=torch.uint8, device=device)
        state.wrapper.reset_workspace_buffer(state.workspace, int_workspace)
    query = torch.randn((batch, heads * groups, dim), dtype=torch.bfloat16, device=device)
    count = batch * heads * capacity
    payloads, metas = [], []
    for layer in range(2):
        k, v = [torch.randn((count, dim), dtype=query.dtype, device=device) for _ in range(2)]
        table = torch.zeros((batch + 1, heads, capacity), dtype=torch.int32, device=device)
        table[:batch] = torch.randperm(count, device=device).to(torch.int32).reshape(batch, heads, capacity)
        positions = torch.arange(capacity, device=device, dtype=torch.int32).expand_as(table).contiguous()
        rows = torch.arange(batch, device=device, dtype=torch.int32)
        lens = torch.full((batch, heads), capacity, device=device, dtype=torch.int32)
        metas.append(HeadSlotViewMeta(table, rows, lens, positions, capacity))
        payloads.append(ExplicitKVPayload(k, v, backend="head_slots"))

    def prepare():
        if state is not None:
            decode.prepare_decode_graph_out(state)
            decode.prepare_decode_graph_in(state)

    def run():
        return [decode.run(query, DecodeComputeView(meta, payload))
                for meta, payload in zip(metas, payloads)]

    try:
        prepare()
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        owners = [] if state is None else decode.decode_graph_keepalive_tensors(state)
        pointers = [t.data_ptr() for t in owners + captured]
        wrapper = None if state is None else state.wrapper
        # Replanning after capture would invalidate the fixed-envelope guarantee.
        if wrapper is not None:
            from unittest.mock import patch
            plan_guard = patch.object(wrapper, "plan", side_effect=AssertionError("runtime plan"))
        else:
            from contextlib import nullcontext
            plan_guard = nullcontext()
        with plan_guard:
            lengths = sorted({1, 31, 65, 127, capacity, *[n for n in (512, 1023, 2048) if n < capacity]})
            for step, length in enumerate(lengths + [1]):
                query.add_(0.015625)
                for layer, meta in enumerate(metas):
                    meta.context_lens.copy_(torch.tensor(
                        [[length, max(1, length // (layer + 2))],
                         [max(1, length - 1), max(1, length // 3)]],
                        device=device, dtype=torch.int32))
                    # Independent physical addresses change without changing the
                    # capture-time table address, then a padded row is reactivated.
                    meta.active_slots[:batch].copy_(meta.active_slots[:batch].roll(1, dims=-1))
                    meta.req_indices[1] = -1 if step % 2 else 1
                    if step % 2:
                        meta.context_lens[1].zero_()
                prepare()
                graph.replay()
                torch.cuda.synchronize()
                for output, meta, payload in zip(captured, metas, payloads):
                    expected = oracle(query, payload.k_cache, payload.v_cache,
                                      meta.active_slots, meta.positions, meta.context_lens,
                                      meta.req_indices, torch.arange(batch + 1, device=device),
                                      torch.full((batch,), capacity, device=device))
                    torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
                assert pointers == [t.data_ptr() for t in owners + captured]
                if state is not None:
                    assert state.wrapper is wrapper
    finally:
        if state is not None:
            decode.close_decode_graph_state(state)
        decode.close()
