import pytest
import torch

from sparseengine.kernels.triton.mla.prefill import merge_partial


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("blocks", [1, 2, 5])
def test_fused_merge_endpoints_preserve_fp32_history_accumulation(dtype, blocks):
    torch.manual_seed(631)
    # Non-contiguous head/LSE views and a non-power-of-two width.
    current = torch.randn(17, 10, 13, device="cuda", dtype=dtype)[:, ::2]
    parts = [torch.randn_like(current) for _ in range(blocks)]
    all_lse = [torch.randn(10, 17, device="cuda")[::2] for _ in range(blocks + 1)]
    expected = current.float().clone()
    expected_lse = all_lse[0].clone()
    for part, weight in zip(parts, all_lse[1:]):
        total = torch.logaddexp(expected_lse, weight)
        a = (expected_lse - total).exp().t().unsqueeze(-1)
        b = (weight - total).exp().t().unsqueeze(-1)
        expected = expected * a + part.float() * b
        expected_lse = total
    output = current.clone()
    lse = all_lse[0].clone()
    scratch = torch.empty_like(current, dtype=torch.float32)
    for i, (part, weight) in enumerate(zip(parts, all_lse[1:])):
        merge_partial(output if i == 0 else scratch, lse, part, weight,
                      destination=output if i == blocks - 1 else scratch)
    atol = {torch.bfloat16: .016, torch.float16: .002, torch.float32: 1e-6}[dtype]
    rtol = .01 if dtype != torch.float32 else 1e-5
    torch.testing.assert_close(output.float(), expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, expected_lse, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_single_history_merge_graph_replay_updates_bf16_output():
    torch.manual_seed(889)
    output = torch.randn(19, 5, 256, device="cuda", dtype=torch.bfloat16)
    partial = torch.randn_like(output)
    lse = torch.randn(5, 19, device="cuda")
    partial_lse = torch.randn_like(lse)
    # Warm the actual endpoint specialization before capture.
    merge_partial(output, lse, partial, partial_lse, destination=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        merge_partial(output, lse, partial, partial_lse, destination=output)
    for _ in range(3):
        output.normal_(); lse.normal_(); partial.normal_(); partial_lse.normal_()
        total = torch.logaddexp(lse, partial_lse)
        expected = output.float() * (lse-total).exp().t()[...,None] + partial.float() * (partial_lse-total).exp().t()[...,None]
        graph.replay()
        torch.testing.assert_close(output.float(), expected, atol=.016, rtol=.01)
        torch.testing.assert_close(lse, total, atol=1e-6, rtol=1e-5)
