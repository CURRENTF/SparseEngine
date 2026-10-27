"""Native head output must preserve scoring math and independent slot addressing."""
import os

import pytest
import torch

from sparseengine.kernels.triton.prefill_score import PrefillScoreWorkspace, prefill_score_fwd

pytestmark = pytest.mark.skipif(os.environ.get('SPARSEENGINE_TEST_CUDA') != '1',
                               reason='requires an idle reserved CUDA device')


def reference(q, k, table, rows, starts, lengths, cached, mode):
    """Direct dense attention per request/head; no production layout reshaping."""
    hkv = table.shape[1] if table.ndim == 3 else k.shape[1]
    groups = q.shape[1] // hkv
    width = table.shape[-1]
    out = torch.full((len(rows), q.shape[1], width),
                     0.0 if mode == 'probability' else -torch.inf, device=q.device)
    for b, row in enumerate(rows.tolist()):
        for head in range(q.shape[1]):
            kh = head // groups
            n = int(lengths[b, kh] if lengths.ndim == 2 else lengths[b])
            prefix = int(cached[b, kh] if cached.ndim == 2 else cached[b])
            slots = table[row, kh, :n] if table.ndim == 3 else table[row, :n]
            keys = k[slots.long(), 0 if table.ndim == 3 else kh].float()
            query = q[int(starts[b]):int(starts[b]) + n - prefix, head].float()
            logits = query @ keys.T
            logits.masked_fill_(torch.arange(n, device=q.device)[None] >
                                torch.arange(prefix, n, device=q.device)[:, None], -torch.inf)
            out[b, head, :n] = ((logits / q.shape[-1]**0.5).softmax(-1).mean(0)
                                if mode == 'probability' else logits.amax(0))
    return out


@pytest.mark.parametrize('mode,window', [('probability', 32), ('probability', 160), ('logits', 160)])
def test_token_layout_3d_and_legacy_2d_match_independent_oracle(mode, window):
    """Catch head loss before softmax/query accumulation and strided 3D writes."""
    torch.manual_seed(57)
    width, hkv, groups, d = window + 73, 2, 7, 128
    q = torch.randn((window * 2, hkv * groups * 2, d), device='cuda', dtype=torch.bfloat16)[:, ::2]
    k = torch.randn((2 * width, hkv, d), device='cuda', dtype=q.dtype)
    table = torch.randperm(2 * width, device='cuda').int().view(2, width)
    ints = lambda v: torch.tensor(v, device='cuda', dtype=torch.int32)
    rows, starts = ints([1, 0]), ints([0, window])
    lengths = ints([width, width - 13])
    cached = lengths - window
    output = torch.empty((2, hkv * groups, width * 2), device='cuda')[..., ::2]
    reduced = torch.empty((2, width), device='cuda')
    args = (rows, starts, lengths, cached, window, table, cached, lengths)
    prefill_score_fwd(q, k, output, *args, score_mode=mode)
    prefill_score_fwd(q, k, reduced, *args, score_mode=mode)
    expected = reference(q, k, table, rows, starts, lengths, cached, mode)
    torch.testing.assert_close(output, expected, atol=2e-5 if mode == 'probability' else 0.002, rtol=3e-4)
    torch.testing.assert_close(reduced, expected.amax(1), atol=2e-5 if mode == 'probability' else 0.002, rtol=3e-4)
    grouped = torch.empty((2, hkv, width), device='cuda')
    prefill_score_fwd(q, k, grouped, *args, score_mode=mode)
    torch.testing.assert_close(grouped, expected.view(2, hkv, groups, width).amax(2),
                               atol=2e-5 if mode == 'probability' else 0.002, rtol=3e-4)
    # Packed output must still index lengths by source row and heads by KV group.
    packed = torch.empty_like(output[:1])
    prefill_score_fwd(q, k, packed, rows, starts, lengths, cached, window, table,
                      cached[1:], lengths[1:], score_mode=mode, batch_indices=ints([1]))
    torch.testing.assert_close(packed, output[1:], atol=2e-5, rtol=3e-4)


@pytest.mark.parametrize('mode,window,width,d', [
    ('probability', 32, 99, 128), ('probability', 160, 227, 128), ('logits', 160, 227, 128),
    ('probability', 32, 4096, 128), ('probability', 32, 32768, 128),
    ('probability', 32, 99, 256), ('probability', 32, 32768, 256),
])
@pytest.mark.parametrize('head_output', ['query', 'kv'])
def test_ragged_head_slots_and_graph_replay_match_independent_oracle(mode, window, width, d, head_output):
    """Catch cross-head slot/length reads, stale replay metadata and accumulated output."""
    torch.manual_seed(61)
    hkv, groups = 2, 7
    ints = lambda v: torch.tensor(v, device='cuda', dtype=torch.int32)
    q = torch.randn((window * 2, hkv * groups, d), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((2 * hkv * width, 1, d), device='cuda', dtype=q.dtype)
    table = torch.randperm(k.shape[0], device='cuda').int().view(2, hkv, width)
    rows, starts = ints([1, 0]), ints([0, window])
    lengths = ints([[width, width - 29], [width - 13, width - 47]])
    cached = lengths - window
    out = torch.empty((2, hkv * groups if head_output == 'query' else hkv, width), device='cuda')
    workspace = PrefillScoreWorkspace()

    def call():
        prefill_score_fwd(q, k, out, rows, starts, lengths, cached, window, table,
                          cached, lengths, score_mode=mode, workspace=workspace)

    def check():
        expected = reference(q, k, table, rows, starts, lengths, cached, mode)
        if head_output == 'kv':
            expected = expected.view(2, hkv, groups, width).amax(2)
        torch.testing.assert_close(out, expected, atol=2e-5 if mode == 'probability' else 0.002, rtol=3e-4)

    call()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    table.copy_(table.flip(-1))
    lengths.sub_(1)
    cached.sub_(1)
    q.mul_(0.7)
    for _ in range(2):
        graph.replay()
        check()
