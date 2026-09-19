"""Dynamic counts retain numerical correctness and reuse one warmed JIT variant."""
import pytest
import torch
import triton

from sparseengine.kernels.triton.deltakv_kernels import full_layer_copy_raw_or_zero
from sparseengine.kernels.triton.moe import _prepare_naive_assignment_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('operation', ['copy', 'assignment'])
def test_runtime_counts_match_oracle_without_new_variant(operation, monkeypatch):
    names = {'copy': '_full_layer_copy_raw_or_zero_kernel', 'assignment': '_prepare_naive_assignment_kernel'}
    events = []
    previous = triton.knobs.compilation.listener
    def listener(**kwargs):
        if kwargs['src'].name == names[operation]:
            events.append(kwargs['src'].name)
        if previous:
            previous(**kwargs)
    monkeypatch.setattr(triton.knobs.compilation, 'listener', listener)
    for size in [17, 19, 31]:
        if operation == 'copy':
            key = torch.randn(64, 2, 80, device='cuda', dtype=torch.float16)
            value = torch.randn_like(key)
            slots = torch.arange(size, device='cuda', dtype=torch.int32)
            valid = slots % 3 != 0
            slots[~valid] = -1
            out_k = torch.empty(size, 2, 80, device='cuda', dtype=key.dtype)
            out_v = torch.empty_like(out_k)
            def run():
                full_layer_copy_raw_or_zero(raw_k=key, raw_v=value, raw_slots=slots,
                    raw_mask=valid, out_k=out_k, out_v=out_v)
            def check():
                for actual, source in [(out_k, key), (out_v, value)]:
                    expected = torch.where(valid[:, None, None], source[slots.long().clamp_min(0)], 0.)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        else:
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
