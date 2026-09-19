import pytest
import torch
from types import SimpleNamespace

from sparseengine.kernels.triton.h2o_score import (
    h2o_softmax_accumulate,
    h2o_headwise_softmax_accumulate,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads,kv_heads,capacity,width", [(7, 1, 257, 129), (32, 8, 33408, 4225)])
def test_shared_h2o_workspace_preserves_layers_and_graphs(heads, kv_heads, capacity, width):
    """Catch overwritten earlier-layer scores and shared scratch replay contamination."""
    from sparseengine import platforms
    from sparseengine.operators.decode_attention import (
        DecodeAttentionOpSpec, FixedGridTritonPagedDecodeAttentionProvider,
        PreparedDecodeAttentionOp,
    )

    torch.manual_seed(208)
    layers, batch, dim = 2, 3, 128
    spec = DecodeAttentionOpSpec(
        num_query_heads=heads, num_kv_heads=kv_heads, head_dim=dim,
        activation_dtype=torch.bfloat16, softmax_scale=dim**-0.5,
        max_batch_size=batch, context_capacity=capacity,
        may_require_attention_scores=True, h2o_headwise_logits=True,
    )
    caps = platforms.get_current_platform().get_device_caps(0)
    provider = FixedGridTritonPagedDecodeAttentionProvider.bind(spec, caps)
    provider.prepare(spec, device_index=0)
    op = PreparedDecodeAttentionOp(spec, provider)
    q = torch.randn(layers, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(layers, batch * width, kv_heads, dim, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    slots = torch.randperm(batch * width, device="cuda").int().view(batch, width)
    lengths = torch.tensor([width, width // 2, 0], device="cuda", dtype=torch.int32)
    requests = torch.tensor([2, 0, -1], device="cuda", dtype=torch.int32)
    scores = torch.empty(layers, batch + 1, capacity + 3, device="cuda")[:, :batch, :capacity]
    scratch = provider._headwise_logits
    pointer = scratch.data_ptr()

    def run(size):
        outputs = []
        for layer in range(layers):
            view = SimpleNamespace(
                payload=SimpleNamespace(backend="dense", k_cache=k[layer], v_cache=v[layer]),
                meta=SimpleNamespace(active_slots=slots, req_indices=requests[:size],
                    context_lens=lengths[:size], attn_score=scores[layer, :size]),
            )
            outputs.append(op.run(q[layer, :size], view))
        return outputs

    graphs = {}
    for size in (2, 3):
        run(size)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = run(size)
        graphs[size] = (graph, outputs)
    for size, lens, reqs in [(3, [width, 17, 0], [2, 0, -1]),
                             (2, [33, 18, 0], [0, 2, -1]),
                             (3, [34, width, 1], [0, 2, 1])]:
        q.copy_(torch.randn_like(q))
        lengths.copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        requests.copy_(torch.tensor(reqs, device="cuda", dtype=torch.int32))
        scratch.fill_(torch.nan)
        scores.fill_(torch.nan)
        graph, outputs = graphs[size]
        graph.replay()
        torch.cuda.synchronize()
        assert provider._headwise_logits.data_ptr() == pointer
        for layer in range(layers):
            for row, length in enumerate(lens[:size]):
                if length:
                    selected = slots[reqs[row], :length].long()
                    keys = k[layer, selected].repeat_interleave(heads // kv_heads, 1).float()
                    values = v[layer, selected].repeat_interleave(heads // kv_heads, 1).float()
                    logits = torch.einsum("hd,khd->hk", q[layer, row].float(), keys)
                    probability = (logits * dim**-0.5).softmax(-1)
                    torch.testing.assert_close(scores[layer, row, :length], probability.sum(0),
                                               atol=2e-5, rtol=2e-5)
                    expected = torch.einsum("hk,khd->hd", probability, values).to(q.dtype)
                    torch.testing.assert_close(outputs[layer][row], expected, atol=2e-2, rtol=2e-2)
                else:
                    assert torch.count_nonzero(outputs[layer][row]) == 0
                assert torch.count_nonzero(scores[layer, row, length:]) == 0
    op.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads,kv_heads,width", [(4, 4, 129), (7, 1, 257), (32, 8, 4225)])
def test_h2o_fused_headwise_paged_graph_matches_independent_torch(heads, kv_heads, width):
    """Catch head-before-softmax, stale tails, padded rows and new-token history reads."""
    torch.manual_seed(49)
    batch, dim = 3, 128
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * width, kv_heads, dim, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    slots = torch.randperm(batch * width, device="cuda").int().view(batch, width)
    requests = torch.tensor([2, 0, -1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([width, width // 2, 0], device="cuda", dtype=torch.int32)
    # Sliced batch and token strides exercise the runtime's shared backing store.
    raw = torch.full((2, batch + 1, heads, width + 3), torch.nan, device="cuda")[:1, :batch, :, :width]
    cumulative = torch.randn(1, batch, width + 7, device="cuda")
    mid_o = torch.empty(batch, heads, 16, dim, device="cuda")
    mid_lse = torch.empty(batch, heads, 16, device="cuda")

    def run():
        output = paged_flash_decode(
            q, k, v, slots, requests, lengths, mid_o, mid_lse,
            attn_score=raw[0], target_tokens_per_split=256,
        )
        h2o_headwise_softmax_accumulate(raw, cumulative, lengths.view(1, batch), softmax_scale=dim**-0.5)
        return output

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    pointers = (raw.data_ptr(), cumulative.data_ptr(), output.data_ptr())
    for lens, reqs in [([width, width // 2, 0], [2, 0, -1]), ([33, 17, 0], [0, 2, -1]), ([34, 18, 1], [0, 2, 1])]:
        lengths.copy_(torch.tensor(lens, dtype=torch.int32, device="cuda"))
        requests.copy_(torch.tensor(reqs, dtype=torch.int32, device="cuda"))
        slots.copy_(slots.roll(3, 1))
        q.copy_(torch.randn_like(q))
        raw.fill_(torch.nan)
        before = cumulative.clone()
        expected = before.clone()
        graph.replay()
        torch.cuda.synchronize()
        for row, length in enumerate(lens):
            if not length:
                assert torch.count_nonzero(output[row]) == 0
                continue
            selected = slots[reqs[row], :length].long()
            keys = k[selected].repeat_interleave(heads // kv_heads, dim=1).float()
            values = v[selected].repeat_interleave(heads // kv_heads, dim=1).float()
            logits = torch.einsum("hd,khd->hk", q[row].float(), keys)
            probability = (logits * dim**-0.5).softmax(-1)
            torch.testing.assert_close(raw[0, row, :, :length], logits, atol=3e-4, rtol=3e-4)
            expected[0, row, length - 1] = 0
            expected[0, row, :length] += probability.sum(0)
            reference_output = torch.einsum("hk,khd->hd", probability, values).to(q.dtype)
            torch.testing.assert_close(output[row], reference_output, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(cumulative, expected, atol=2e-5, rtol=2e-5)
        assert pointers == (raw.data_ptr(), cumulative.data_ptr(), output.data_ptr())
from sparseengine.kernels.triton.h2o_decode_score import h2o_probability_from_lse
from sparseengine.kernels.triton.paged_flash_decoding import paged_flash_decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("batch_stride", [False, True])
def test_h2o_softmax_accumulate_matches_torch(batch, batch_stride):
    torch.manual_seed(0)
    layers, capacity, width = 3, 192, 129
    previous_width = width - 1
    allocated_batch = batch + 3 if batch_stride else batch
    logits_storage = torch.randn(layers, allocated_batch, capacity, device="cuda")
    logits = logits_storage[:, :batch]
    initial = torch.randn(layers, batch, capacity, device="cuda")
    actual = initial.clone()
    expected = initial.clone()
    scale = 128**-0.5

    probabilities = torch.softmax(logits[:, :, :width] * scale, dim=-1)
    expected[:, :, :previous_width].add_(
        probabilities[:, :, :previous_width]
    )
    expected[:, :, previous_width:width].copy_(
        probabilities[:, :, previous_width:width]
    )
    h2o_softmax_accumulate(
        logits,
        actual,
        width=width,
        previous_width=previous_width,
        softmax_scale=scale,
    )

    assert torch.allclose(
        actual[:, :, :width], expected[:, :, :width], atol=1e-6, rtol=1e-5
    )
    assert torch.equal(actual[:, :, width:], initial[:, :, width:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires idle CUDA GPU")
def test_h2o_decode_probability_graph_replays_compacted_and_padded_rows():
    """Catch stale LSE/scores when eviction shrinks a row or a batch reuses it."""
    torch.manual_seed(19)
    batch, heads, kv_heads, dim, capacity = 2, 8, 2, 128, 129
    q = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch * capacity, kv_heads, dim, dtype=q.dtype, device="cuda")
    v = torch.randn_like(k)
    slots = torch.arange(batch * capacity, dtype=torch.int32, device="cuda").view(batch, capacity)
    requests = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([129, 65], dtype=torch.int32, device="cuda")
    score = torch.empty(batch, capacity, dtype=torch.float32, device="cuda")
    mid_o = torch.empty(batch, heads, 8, dim, dtype=torch.float32, device="cuda")
    mid_lse = torch.empty(batch, heads, 8, dtype=torch.float32, device="cuda")
    lse = torch.empty(heads, batch, dtype=torch.float32, device="cuda")

    def run():
        output, _ = paged_flash_decode(
            q, k, v, slots, requests, lengths, mid_o, mid_lse,
            target_tokens_per_split=64, return_softmax_lse=True, output_lse=lse,
        )
        h2o_probability_from_lse(q, k, lse, slots, requests, lengths, score, softmax_scale=dim**-0.5)
        return output

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    pointers = (output.data_ptr(), score.data_ptr(), lse.data_ptr())
    for new_lengths, new_requests in [([33, 0], [1, 0]), ([34, 65], [1, 0]), ([129, 17], [0, 1])]:
        lengths.copy_(torch.tensor(new_lengths, dtype=torch.int32, device="cuda"))
        requests.copy_(torch.tensor(new_requests, dtype=torch.int32, device="cuda"))
        slots.copy_(slots.roll(1, dims=1))
        q.copy_(torch.randn_like(q))
        score.fill_(123)
        graph.replay()
        torch.cuda.synchronize()
        assert (output.data_ptr(), score.data_ptr(), lse.data_ptr()) == pointers
        for row, length in enumerate(new_lengths):
            if length:
                selected = slots[new_requests[row], :length].long()
                keys = k[selected].repeat_interleave(heads // kv_heads, dim=1)
                values = v[selected].repeat_interleave(heads // kv_heads, dim=1)
                logits = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * dim**-0.5
                probability = logits.softmax(-1)
                expected = torch.einsum("hk,khd->hd", probability, values.float()).to(q.dtype)
                torch.testing.assert_close(output[row], expected, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(score[row, :length], probability.sum(0), atol=2e-3, rtol=2e-3)
            assert torch.count_nonzero(score[row, length:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_h2o_decode_probability_matches_paged_torch(batch):
    torch.manual_seed(11)
    query_heads, kv_heads, head_dim, width = 16, 2, 128, 129
    capacity = batch * width
    q = torch.randn(
        batch,
        query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        capacity,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    page_table = torch.randperm(capacity, device="cuda", dtype=torch.int64).to(
        torch.int32
    ).view(batch, width)
    request_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    context_lens = torch.arange(
        width - batch + 1,
        width + 1,
        device="cuda",
        dtype=torch.int32,
    )
    scale = head_dim**-0.5
    attention_lse = torch.empty(
        (query_heads, batch), dtype=torch.float32, device="cuda"
    )
    expected_rows = []
    group = query_heads // kv_heads
    for batch_idx in range(batch):
        length = int(context_lens[batch_idx].item())
        keys = k.index_select(0, page_table[batch_idx, :length].long())
        expanded_keys = keys.repeat_interleave(group, dim=1)
        logits = torch.einsum(
            "hd,khd->hk",
            q[batch_idx].float(),
            expanded_keys.float(),
        ) * scale
        attention_lse[:, batch_idx] = torch.logsumexp(logits, dim=-1)
        expected_rows.append(torch.softmax(logits, dim=-1).sum(dim=0))
    actual = torch.full((batch, width), torch.nan, device="cuda")

    h2o_probability_from_lse(
        q,
        k,
        attention_lse,
        page_table,
        request_indices,
        context_lens,
        actual,
        softmax_scale=scale,
    )

    for batch_idx in range(batch):
        length = int(context_lens[batch_idx].item())
        assert torch.allclose(
            actual[batch_idx, :length],
            expected_rows[batch_idx],
            atol=2e-3,
            rtol=2e-3,
        )
        assert torch.all(actual[batch_idx, length:] == 0)
