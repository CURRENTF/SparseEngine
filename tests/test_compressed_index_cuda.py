import pytest
import torch

from sparseengine.operators.compressed_index import (
    CompressedIndexOpSpec, resolve_compressed_index_provider,
)

pytestmark = pytest.mark.cuda


def _hadamard_reference(x):
    y = x.double().clone()
    step = 1
    while step < y.shape[-1]:
        blocks = y.unflatten(-1, (-1, 2, step))
        a, b = blocks[..., 0, :].clone(), blocks[..., 1, :].clone()
        blocks[..., 0, :] = a + b
        blocks[..., 1, :] = a - b
        step *= 2
    return (y / y.shape[-1] ** 0.5).to(x.dtype)


def _fp4_reference(x):
    blocks = x.double().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-126)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6)))
    values = (blocks / scale).clamp(-6, 6)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device, dtype=torch.float64)
    distances = (values.abs()[..., None] - grid).abs()
    # E2M1 even encoded mantissas win exact halfway ties.
    preference = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=x.device)
    tie = distances == distances.amin(-1, keepdim=True)
    cost = torch.where(tie, preference, 2)
    index = cost.argmin(-1)
    return (grid[index] * values.sign() * scale).flatten(-2).to(x.dtype)


def _decode_pages(pages, slots, page_size):
    page, offset = slots.long() // page_size, slots.long() % page_size
    byte = torch.arange(64, device=pages.device)
    payload = pages[page[:, None], offset[:, None] * 64 + byte]
    codes = torch.stack((payload & 15, payload >> 4), -1).flatten(-2)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=pages.device)
    value = grid[(codes & 7).long()] * torch.where(codes < 8, 1., -1.)
    exponents = pages[page[:, None], page_size * 64 + offset[:, None] * 4 + torch.arange(4, device=pages.device)]
    scales = torch.exp2(exponents.float() - 127)
    return (value.unflatten(-1, (4, 32)) * scales[..., None]).flatten(-2).bfloat16()


@pytest.fixture
def provider():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("validated SM90 target required")
    pytest.importorskip("fast_hadamard_transform")
    return resolve_compressed_index_provider(
        CompressedIndexOpSpec(64, 128, 64, 512, 64, True), device_index=0,
    )


def test_indexer_fp4_rounding_and_key_store(provider):
    torch.manual_seed(801)
    keys = torch.randn(641, 128, device="cuda", dtype=torch.bfloat16)
    keys[-1].zero_()
    slots = torch.randperm(1024, device="cuda", dtype=torch.int64)[:641].int().contiguous()
    pages = torch.zeros(16, 64 * 68, device="cuda", dtype=torch.uint8)
    provider.store_keys(keys, pages, slots)
    expected = _fp4_reference(_hadamard_reference(keys))
    torch.testing.assert_close(_decode_pages(pages, slots, 64), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        provider.store_keys(keys, pages, slots)
    keys.normal_()
    # All graph writes use real slots, including cache-reserved padding slots.
    slots.copy_(torch.randperm(1024, device="cuda")[:641].int())
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(_decode_pages(pages, slots, 64),
                               _fp4_reference(_hadamard_reference(keys)), rtol=0, atol=0)

    x = torch.tensor([-.25, -.75, -1.25, -1.75, -2.5, -3.5, -5, 0,
                       .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6] * 8,
                     device="cuda", dtype=torch.bfloat16).reshape(1, 128)
    torch.testing.assert_close(provider._fake_quant(x), _fp4_reference(x), rtol=0, atol=0)


def test_indexer_score_topk_and_dynamic_graph(provider):
    torch.manual_seed(802)
    device = "cuda"
    rows, width = 3, 640
    query = torch.randn(rows, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(width, 128, device=device, dtype=torch.bfloat16)
    write_slots = torch.randperm(1024, device=device)[:width].int().contiguous()
    pages = torch.zeros(16, 64 * 68, device=device, dtype=torch.uint8)
    provider.store_keys(keys, pages, write_slots)
    slots = write_slots.expand(rows, -1).contiguous()
    lengths = torch.tensor([0, 33, 639], device=device, dtype=torch.int32)
    head_weights = torch.randn(rows, 64, device=device, dtype=torch.bfloat16) * (128 ** -.5 * 64 ** -.5)
    angles = torch.randn(1024, 32, device=device)
    freqs = torch.stack((angles.cos(), angles.sin()), -1).flatten(-2).contiguous()
    positions = torch.tensor([4, 128, 300], device=device, dtype=torch.int32)
    # Attention slots deliberately differ from index-cache slots.
    mapping = torch.randperm(2048, device=device)[:width].int().expand(rows, -1).contiguous()
    scores = torch.empty(rows, width, device=device)
    selected = torch.empty(rows, 512, device=device, dtype=torch.int32)

    def run():
        prepared = provider.prepare_query(query, freqs, positions)
        provider.score(prepared, head_weights, pages, slots, lengths, out=scores)
        provider.select(scores, mapping, lengths, out=selected)
        return prepared

    prepared = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_query = run()

    for new_lengths, new_positions in [([0, 33, 639], [4, 128, 300]),
                                        ([128, 600, 1], [9, 129, 501])]:
        lengths.copy_(torch.tensor(new_lengths, device=device, dtype=torch.int32))
        positions.copy_(torch.tensor(new_positions, device=device, dtype=torch.int32))
        query.normal_()
        graph.replay()
        torch.cuda.synchronize()
        q = query.clone()
        tail = q[..., -64:].float().unflatten(-1, (32, 2))
        f = freqs[positions.long()].unflatten(-1, (32, 2))[:, None]
        rotated = torch.stack((tail[..., 0] * f[..., 0] - tail[..., 1] * f[..., 1],
                               tail[..., 0] * f[..., 1] + tail[..., 1] * f[..., 0]), -1)
        q[..., -64:] = rotated.flatten(-2).bfloat16()
        expected_query = _fp4_reference(_hadamard_reference(q))
        torch.testing.assert_close(graph_query, expected_query, rtol=0, atol=0)
        kv = _decode_pages(pages, write_slots, 64)
        dot = torch.matmul(expected_query.float(), kv.float().T).bfloat16()
        expected_scores = (dot.relu() * head_weights[..., None]).sum(1).float()
        for row, length in enumerate(new_lengths):
            torch.testing.assert_close(scores[row, :length], expected_scores[row, :length],
                                       rtol=.02, atol=.025)
            assert torch.isneginf(scores[row, length:]).all()
            n = min(512, length)
            assert (selected[row, n:] == -1).all()
            if length <= 512:
                torch.testing.assert_close(selected[row, :n], mapping[row, :n])
            else:
                # Selection is exact on provider scores, including smaller-index ties.
                ranked = torch.argsort(scores[row, :length], descending=True, stable=True)[:512]
                assert set(selected[row].tolist()) == set(mapping[row, ranked].tolist())
