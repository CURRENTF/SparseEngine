"""Dynamic counts retain numerical correctness and reuse one warmed JIT variant."""
import pytest
import torch
import triton

from sparseengine.kernels.triton.moe import _prepare_naive_assignment_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_runtime_counts_match_oracle_without_new_variant(monkeypatch):
    events = []
    previous = triton.knobs.compilation.listener
    def listener(**kwargs):
        if kwargs['src'].name == '_prepare_naive_assignment_kernel':
            events.append(kwargs['src'].name)
        if previous:
            previous(**kwargs)
    monkeypatch.setattr(triton.knobs.compilation, 'listener', listener)
    for size in [17, 19, 31]:
        ids = torch.arange(size, device='cuda', dtype=torch.int32) % 8
        experts = torch.empty_like(ids)
        padded = torch.empty(1, device='cuda', dtype=torch.int32)
        def run():
            _prepare_naive_assignment_kernel[(1,)](ids, experts, padded, size, 2, 6, size * 16, 32)
        def check():
            expected = torch.where((ids >= 2) & (ids < 6), ids - 2, -1)
            torch.testing.assert_close(experts, expected, atol=0, rtol=0)
            assert padded.item() == size * 16
        run()
        check()
        if size == 17:
            events.clear()
        else:
            assert not events
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    check()
