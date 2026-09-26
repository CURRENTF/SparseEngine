import pytest
import torch

from sparseengine.operators.hyper_connection import (
    HyperConnectionOpSpec, resolve_hyper_connection_provider,
)


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA",
)]


def reference(residual, projection, scale, base, eps, iterations):
    """FP64 projection and alternating row/column stochastic normalization."""
    x = residual.double()
    logits = x.flatten(1) @ projection.double().T
    logits *= (x.square().mean((1, 2)) + eps).rsqrt()[:, None]
    pre = (logits[:, :4] * scale[0] + base[:4]).sigmoid() + eps
    post = 2 * (logits[:, 4:8] * scale[1] + base[4:8]).sigmoid()
    mix = (logits[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4).softmax(-1) + eps
    for i in range(iterations):
        if i:
            mix = mix / (mix.sum(-1, keepdim=True) + eps)
        mix = mix / (mix.sum(-2, keepdim=True) + eps)
    layer = (pre[..., None] * x).sum(1).to(residual.dtype)
    return post[..., None].float(), mix.float(), layer


@pytest.mark.parametrize("rows", [1, 7, 128])
def test_mhc_pre_post_and_graph_dynamic_residual(rows):
    # Catches wrong RMS domain, residual-mix transpose and stale graph inputs
    # using the real V4 width. No vLLM runtime or routing mock is involved.
    torch.manual_seed(42)
    device = torch.device("cuda")
    residual = torch.randn(rows, 4, 4096, device=device, dtype=torch.bfloat16)
    projection = torch.randn(24, 16384, device=device) * 0.005
    scale = torch.tensor([0.7, 0.6, 0.5], device=device)
    base = torch.randn(24, device=device) * 0.2
    output = torch.randn(rows, 4096, device=device, dtype=torch.bfloat16)
    spec = HyperConnectionOpSpec(4096, 4, torch.bfloat16, 1e-6, 1e-6, 20)
    provider = resolve_hyper_connection_provider(spec, device_index=torch.cuda.current_device())

    def run():
        post, mix, layer = provider.pre(residual, projection, scale, base)
        return post, mix, layer, provider.post(output, residual, post, mix)

    def check(actual):
        expected = reference(residual, projection, scale, base, 1e-6, 20)
        for a, b in zip(actual[:3], expected):
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.008)
        p, m, _ = expected
        combined = torch.einsum("bij,bih->bjh", m.double(), residual.double())
        combined += p.double() * output.double().unsqueeze(1)
        torch.testing.assert_close(actual[3], combined.bfloat16(), rtol=0.02, atol=0.032)

    check(run())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(2):
        residual.copy_(torch.randn_like(residual))
        output.copy_(torch.randn_like(output))
        graph.replay()
        check(captured)


def test_mhc_head_merge_and_graph():
    torch.manual_seed(122)
    residual = torch.randn(3, 4, 4096, device="cuda", dtype=torch.bfloat16)
    projection = torch.randn(4, 16384, device="cuda") * .005
    scale = torch.tensor([.7], device="cuda")
    base = torch.randn(4, device="cuda") * .2
    provider = resolve_hyper_connection_provider(
        HyperConnectionOpSpec(4096, 4, torch.bfloat16, 1e-6, 1e-6, 20),
        device_index=torch.cuda.current_device(),
    )

    def run():
        return provider.head(residual, projection, scale, base)

    def check(actual):
        x = residual.double()
        mixes = x.flatten(1) @ projection.double().T
        mixes *= (x.square().mean((1, 2)) + 1e-6).rsqrt()[:, None]
        pre = (mixes * scale.double() + base.double()).sigmoid() + 1e-6
        torch.testing.assert_close(actual, (pre[..., None]*x).sum(1).bfloat16(),
                                   rtol=.015, atol=.015)

    check(run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    residual.copy_(torch.randn_like(residual))
    graph.replay()
    check(captured)
