import pytest
import torch

from sparseengine.engine.cache_manager.base import PackedSharedKVPayload, SharedKVPayload

from sparseengine.operators.shared_kv_attention import (
    SharedKVAttentionOpSpec, resolve_shared_kv_attention_provider,
)


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA",
)]


def pack_reference(kv, page_size):
    """Independent FlashMLA layout: 576-byte values then 8-byte scale rows."""
    pages = (len(kv) + page_size - 1) // page_size
    page_bytes = ((584 * page_size + 575) // 576) * 576
    storage = torch.zeros(pages, page_bytes, device=kv.device, dtype=torch.uint8)
    quantized = kv[:, :448].float().view(-1, 7, 64)
    exponent = (quantized.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil()
    scale = torch.exp2(exponent)
    quantized = (quantized / scale[..., None]).to(torch.float8_e4m3fn)
    dequantized = torch.cat((
        (quantized.float() * scale[..., None]).flatten(1), kv[:, 448:].float(),
    ), dim=-1)
    for i in range(len(kv)):
        page, offset = divmod(i, page_size)
        storage[page, offset * 576:offset * 576 + 448] = quantized[i].flatten().view(torch.uint8)
        storage[page, offset * 576 + 448:(offset + 1) * 576] = kv[i, 448:].contiguous().view(torch.uint8)
        start = page_size * 576 + offset * 8
        storage[page, start:start + 7] = (exponent[i] + 127).to(torch.uint8)
    # FlashMLA uses a logically contiguous page view; its kernel interprets the
    # internal values/scales split independently of the tensor's token stride.
    payload = storage.as_strided((pages, page_size, 1, 584), (page_bytes, 584, 584, 1))
    return storage, payload, dequantized


def attention_reference(query, kv, indices, lengths, sink):
    result = []
    for row in range(len(query)):
        ids = indices[row, 0, :int(lengths[row])].long()
        ids = ids[ids >= 0]
        selected = kv[ids].double()
        logits = query[row].double() @ selected.T / (512 ** 0.5)
        probabilities = torch.cat((logits, sink.double()[:, None]), dim=-1).softmax(-1)
        result.append(probabilities[:, :-1] @ selected)
    return torch.stack(result).bfloat16()


def test_prefill_adapts_decode_capacity_and_masks_window_holes():
    torch.manual_seed(29)
    kv = torch.randn(160, 512, device="cuda", dtype=torch.bfloat16)
    query = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(64, device="cuda")
    indices = torch.full((2, 1, 192), -1, device="cuda", dtype=torch.int32)
    indices[:, 0, :20] = torch.arange(20, device="cuda", dtype=torch.int32)
    indices[:, 0, 128] = 150
    lengths = torch.full((2,), 129, device="cuda", dtype=torch.int32)
    spec = SharedKVAttentionOpSpec(64, 512, torch.bfloat16, 64, 192, 512**-.5, True)
    provider = resolve_shared_kv_attention_provider(spec, device_index=torch.cuda.current_device())
    result = provider.prefill(query, SharedKVPayload(kv[:, None]), indices, lengths, sink)
    torch.testing.assert_close(result, attention_reference(query, kv, indices, lengths, sink),
                               rtol=.02, atol=.008)


def unpack_reference(storage, slots, page_size):
    values = []
    for slot in slots.tolist():
        page, offset = divmod(slot, page_size)
        raw = storage[page, offset * 576:(offset + 1) * 576]
        scale_start = page_size * 576 + offset * 8
        scales = torch.exp2(storage[page, scale_start:scale_start + 7].float() - 127)
        nope = raw[:448].view(torch.float8_e4m3fn).float().view(7, 64) * scales[:, None]
        rope = raw[448:].view(torch.bfloat16).float()
        values.append(torch.cat((nope.flatten(), rope)))
    return torch.stack(values)


@pytest.mark.parametrize("rows", [1, 3])
def test_sparse_prefill_packed_decode_dynamic_graph(rows):
    # Protects packed page strides, sink normalization, changed selections and
    # padded rows on replay. The oracle reads independently dequantized KV.
    torch.manual_seed(12)
    device = torch.device("cuda")
    kv = torch.randn(192, 512, device=device, dtype=torch.bfloat16)
    query = torch.randn(rows, 64, 512, device=device, dtype=torch.bfloat16)
    indices = torch.full((rows, 1, 128), -1, device=device, dtype=torch.int32)
    lengths = torch.full((rows,), 75, device=device, dtype=torch.int32)
    sink = torch.randn(64, device=device)
    indices[:, 0, :75] = torch.arange(75, device=device, dtype=torch.int32)
    spec = SharedKVAttentionOpSpec(64, 512, torch.bfloat16, 64, 128, 512 ** -0.5, True)
    provider = resolve_shared_kv_attention_provider(spec, device_index=torch.cuda.current_device())
    prefill = provider.prefill(query, SharedKVPayload(kv.unsqueeze(1)), indices, lengths, sink)
    torch.testing.assert_close(prefill, attention_reference(query, kv, indices, lengths, sink),
                               rtol=0.02, atol=0.008)
    storage, packed, dequantized = pack_reference(kv, 64)
    state = provider.create_decode_state(rows)

    def run():
        return provider.decode(query, PackedSharedKVPayload(packed, 64), indices, lengths, sink, state)

    torch.testing.assert_close(run(), attention_reference(query, dequantized, indices, lengths, sink),
                               rtol=0.02, atol=0.008)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    provider.prepare_decode_state(state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for count in [33, 128]:
        query.copy_(torch.randn_like(query))
        indices.fill_(-1)
        indices[:, 0, :count] = torch.randperm(192, device=device)[:count].int()
        lengths.fill_(count)
        if rows > 1:
            lengths[-1] = 0
            indices[-1].fill_(-1)
        graph.replay()
        torch.testing.assert_close(captured, attention_reference(query, dequantized, indices, lengths, sink),
                                   rtol=0.02, atol=0.008)


@pytest.mark.parametrize("page_size", [64, 128])
def test_sgl_norm_rope_store_page_boundary_and_graph(page_size):
    # Confirms the AOT writer and FlashMLA reader really use the same page-tail
    # scale layout; graph replay must read new positions and destination slots.
    from sgl_kernel import dsv4_fused_k_norm_rope_flashmla

    torch.manual_seed(27)
    kv = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(256, 32, device="cuda")
    freqs = torch.stack((angles.cos(), angles.sin()), -1).flatten(1).contiguous()
    positions = torch.arange(7, device="cuda", dtype=torch.int32)
    slots = torch.arange(page_size - 3, page_size + 4, device="cuda", dtype=torch.int32)
    page_bytes = ((584 * page_size + 575) // 576) * 576
    storage = torch.zeros(3, page_bytes, device="cuda", dtype=torch.uint8)

    def run():
        dsv4_fused_k_norm_rope_flashmla(kv, weight, freqs, positions, slots,
                                     storage, eps=1e-6, page_size=page_size)

    def check():
        norm = kv.float() * (kv.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        norm *= weight.float()
        pairs = norm[:, 448:].view(-1, 32, 2)
        rotation = freqs[positions.long()].view(-1, 32, 2)
        real = pairs[..., 0] * rotation[..., 0] - pairs[..., 1] * rotation[..., 1]
        imag = pairs[..., 0] * rotation[..., 1] + pairs[..., 1] * rotation[..., 0]
        rotated = torch.stack((real, imag), -1).flatten(1).bfloat16().float()
        actual = unpack_reference(storage, slots, page_size)
        torch.testing.assert_close(actual[:, :448], norm[:, :448], rtol=0.07, atol=0.01)
        torch.testing.assert_close(actual[:, 448:], rotated, rtol=0.01, atol=0.015)

    run()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    positions.add_(11)
    slots.add_(page_size)
    kv.copy_(torch.randn_like(kv))
    graph.replay()
    check()
