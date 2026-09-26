import pytest
import torch

from sparseengine.kernels.external.sgl.v4_compression import SGLCompressionKernels
from sparseengine.engine.cache_manager.storage.compression_state import CompressionStatePool


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA",
)]


def reference(projected, ape, ratio, end, dim):
    if ratio == 4:
        start = max(0, end - 8)
        tokens = torch.arange(start, end, device=projected.device)
        overlap = tokens < end - 4
        values = torch.where(overlap[:, None], projected[start:end, :dim],
                             projected[start:end, dim:2 * dim])
        scores = torch.where(overlap[:, None], projected[start:end, 2 * dim:3 * dim],
                             projected[start:end, 3 * dim:])
        bias = ape[(tokens - (end - 8)).long()]
    else:
        values = projected[end - ratio:end, :dim]
        scores = projected[end - ratio:end, dim:]
        bias = ape
    probabilities = (scores.double() + bias.double()).softmax(0)
    return (values.double() * probabilities).sum(0).float()


@pytest.mark.parametrize("ratio,dim", [(4,512),(4,128),(128,512)])
def test_chunked_compression_carry_graph_and_fork(ratio, dim):
    # Protects overlap lane order, gate+APE softmax domain, partial-chunk carry,
    # mutable prefix fork state and dynamic compression-boundary graph replay.
    torch.manual_seed(31)
    kernels = SGLCompressionKernels(ratio=ratio, head_dim=dim,
                                   architecture=torch.cuda.get_device_capability())
    width = dim * (4 if ratio == 4 else 2)
    window = 8 if ratio == 4 else 128
    total = 260
    projected = torch.randn(total, width, device="cuda")
    ape = torch.randn(window, dim, device="cuda") * 0.3
    pool = CompressionStatePool(num_rows=3, ratio=ratio, head_dim=dim, device="cuda")
    carry = pool.state
    rows = torch.tensor([0], device="cuda", dtype=torch.int32)
    start = 0
    # Non-aligned chunks cross both 4 and 128 publication boundaries.
    for size in [3, 5, 121, 1]:
        end = start + size
        plans = kernels.prefill_plan(torch.tensor([end]), torch.tensor([size]),
                                    num_tokens=size, device=projected.device)
        output = torch.full((size, dim), float("nan"), device="cuda")
        kernels.prefill(carry, projected[start:end].contiguous(), output, ape, rows, plans)
        for position in range(start, end):
            if (position + 1) % ratio == 0:
                torch.testing.assert_close(output[position - start],
                                           reference(projected, ape, ratio, position + 1, dim),
                                           rtol=0.002, atol=0.0002)
        start = end
    # Fork: mutable carry must be independently owned. Eager and graph start
    # from exactly the same partial prefix, including all overlap state.
    snapshot = pool.snapshot(0)
    pool.fork(0, 1)
    pool.reset_row(0)
    pool.restore(0, snapshot)
    source = torch.empty(2, width, device="cuda")
    graph_rows = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([start + 1, 1], device="cuda", dtype=torch.int32)
    output = torch.empty(2, dim, device="cuda")
    saved = carry.clone()
    source.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            kernels.decode(carry, source, output, ape, graph_rows, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernels.decode(carry, source, output, ape, graph_rows, lengths)
    carry.copy_(saved)
    for position in range(start, total):
        source[0].copy_(projected[position])
        source[1].zero_()
        lengths[0] = position + 1
        graph.replay()
        # The original prefix's carry never changes during fork replay.
        torch.testing.assert_close(carry[0], saved[0], rtol=0, atol=0)
        if (position + 1) % ratio == 0:
            torch.testing.assert_close(output[0], reference(projected, ape, ratio, position + 1, dim),
                                       rtol=0.002, atol=0.0002)


@pytest.mark.parametrize("ratio,dim", [(4,512),(4,128),(128,512)])
def test_prepared_provider_projection_pooling_normalization_and_rope(ratio, dim):
    from sparseengine.engine.cache_manager.base import CompressionComputeView
    from sparseengine.operators.kv_compression import (
        KVCompressionOpSpec, resolve_kv_compression_provider,
    )

    torch.manual_seed(15)
    spec = KVCompressionOpSpec(ratio, dim, 4096, torch.bfloat16, 1e-6, True)
    provider = resolve_kv_compression_provider(spec, device_index=torch.cuda.current_device())
    tokens = ratio + 3
    width = spec.projection_width // 2
    hidden = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(width, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
    gate_weight = torch.randn_like(kv_weight) * 0.01
    ape = torch.randn(ratio, width, device="cuda")
    norm_weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    provider.prepare_weights(kv_weight, gate_weight, ape, norm_weight)
    projected = provider.project(hidden)
    expected_projection = hidden.double() @ torch.cat((kv_weight, gate_weight)).double().T
    torch.testing.assert_close(projected, expected_projection.float(), rtol=0.003, atol=0.0001)
    pool = CompressionStatePool(num_rows=1, ratio=ratio, head_dim=dim, device="cuda")
    lengths = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    rows = torch.tensor([0], device="cuda", dtype=torch.int32)
    plans = provider.kernels.prefill_plan(torch.tensor([tokens]), torch.tensor([tokens]),
                                         num_tokens=tokens, device=hidden.device)
    output = torch.full((tokens, dim), float("nan"), device="cuda")
    angles = torch.randn(256, 32, device="cuda")
    freqs = torch.stack((angles.cos(), angles.sin()), -1).flatten(1).contiguous()
    view = CompressionComputeView(pool.state, rows, lengths, output, plans)
    provider.compute(projected, view, freqs)
    expected_ape = (torch.cat((ape[:, :dim], ape[:, dim:])) if ratio == 4 else ape)
    for end in range(ratio, tokens + 1, ratio):
        pooled = reference(expected_projection.float(), expected_ape, ratio, end, dim).double()
        pooled *= (pooled.square().mean() + spec.rms_eps).rsqrt() * norm_weight.double()
        pairs = pooled[-64:].reshape(32, 2)
        angle = angles[end - ratio].double()
        rotated = torch.stack((pairs[:, 0] * angle.cos() - pairs[:, 1] * angle.sin(),
                               pairs[:, 0] * angle.sin() + pairs[:, 1] * angle.cos()), -1).flatten()
        pooled[-64:] = rotated
        torch.testing.assert_close(output[end - 1], pooled.float(), rtol=0.003, atol=0.0005)
