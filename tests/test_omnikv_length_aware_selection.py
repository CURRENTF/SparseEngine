"""Catch reading stale capacity tails or dropping history at the keep boundary."""
import os

import pytest
import torch

from sparseengine.operators.omnikv_selection import (
    OmniKVSelectionSpec, TorchCPUOmniKVSelection, TritonOmniKVSelection,
    prepare_omnikv_selection,
)


def reference(scores, lengths, sink, k):
    result = torch.full((scores.shape[0], k), -1, dtype=torch.int32, device=scores.device)
    for row, length in enumerate(lengths.tolist()):
        if length <= k:
            chosen = torch.arange(length, device=scores.device)
        else:
            # Python's stable sort is independent of the radix selection kernel.
            values = scores[row, sink:sink + length].tolist()
            chosen = torch.tensor(sorted(range(length), key=lambda i: (-values[i], i))[:k], device=scores.device)
        result[row, :len(chosen)] = chosen.int() + sink
    return result


def kernel_device():
    if os.environ.get('TRITON_INTERPRET') == '1':
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    pytest.skip('requires CUDA or explicit Triton interpreter')


@pytest.mark.parametrize('k,capacity,dtype', [(1, 37, torch.float32), (7, 129, torch.float16), (4096, 131072, torch.bfloat16)])
@pytest.mark.parametrize('implementation', ['cpu', 'triton'])
def test_history_selection_ignores_short_row_scores_and_poisoned_tails(k, capacity, dtype, implementation):
    device = 'cpu' if implementation == 'cpu' else kernel_device()
    sink = 3
    torch.manual_seed(431)
    lengths = torch.tensor([0, k - 1, k, capacity - sink], dtype=torch.int32, device=device)
    # Duplicate scores protect deterministic tie handling; striding protects indexing.
    scores = torch.randint(-10, 10, (4, capacity * 2), device=device).to(dtype)[:, ::2]
    scores[:3].fill_(float('nan'))  # Short rows must never consult scores.
    scores[3, sink + int(lengths[3]):] = float('nan')
    cls = TorchCPUOmniKVSelection if implementation == 'cpu' else TritonOmniKVSelection
    provider = cls(op_spec=OmniKVSelectionSpec(sink, k))
    torch.testing.assert_close(
        provider.select(scores, lengths, k).sort(1).values,
        reference(scores, lengths, sink, k).sort(1).values,
    )


def test_selection_score_pipeline_preserves_unread_storage():
    # Unwritten short rows/tails deliberately contain NaNs, including on reuse.
    device = kernel_device()
    from sparseengine.kernels.triton.omnikv_score import launch_omnikv_decode_scores
    sink, recent, k, capacity = 3, 2, 7, 137
    raw = torch.randn(4, 3, capacity, device=device)
    lengths = torch.tensor([0, 12, 13, capacity], dtype=torch.int32, device=device)
    partial = torch.full((4, 3, 3, 2), float('nan'), device=device)
    stats = torch.full((4, 3, 2), float('nan'), device=device)
    scores = torch.full((4, capacity), float('nan'), device=device)
    selector = TritonOmniKVSelection(op_spec=OmniKVSelectionSpec(sink, k))
    for values in ([0, 12, 13, capacity], [capacity, 13, 12, 1]):
        lengths.copy_(torch.tensor(values, dtype=torch.int32, device=device))
        scores.fill_(float('nan'))
        partial.fill_(float('nan'))
        raw.normal_()
        launch_omnikv_decode_scores(
            raw, lengths, partial, stats, scores, sink=sink, recent=recent, scale=.5,
            min_score=torch.finfo(torch.float32).min, block=64, output_block=32, selection_keep=k,
        )
        candidate_lengths = (lengths - sink - recent).clamp_min(0)
        expected_scores = torch.full_like(scores, float('nan'))
        for row, length in enumerate(candidate_lengths.tolist()):
            if length > k:
                expected = (raw[row, :, sink:sink+length].double() * .5).softmax(-1).amax(0).float()
                expected_scores[row, sink:sink+length] = expected
        torch.testing.assert_close(scores, expected_scores, equal_nan=True, atol=1e-7, rtol=2e-5)
        torch.testing.assert_close(selector.select(scores, candidate_lengths, k).sort(1).values,
                                   reference(expected_scores, candidate_lengths, sink, k).sort(1).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('k', [7, 4096])
def test_prepared_selection_graph_replays_ragged_lengths(k):
    # Exercise the resolved upstream/portable provider, score skips, then repeated
    # short/long transitions without changing input/output addresses or graph.
    from sparseengine.operators.omnikv_score import OmniKVScoreSpec, prepare_omnikv_score_provider
    sink, recent, capacity = 64, 512, 131072
    raw = torch.randn(4, 3, capacity, device='cuda')
    lengths = torch.full((4,), capacity, dtype=torch.int32, device='cuda')
    scorer = prepare_omnikv_score_provider(OmniKVScoreSpec(sink, recent, .5, torch.float32), device=raw.device)
    selector = prepare_omnikv_selection(OmniKVSelectionSpec(sink, k), device=raw.device)
    scorer.prepare(raw, slot=0)
    def run():
        scores = scorer.run(raw, lengths, slot=0, selection_keep=k)
        return selector.select(scores, (lengths - sink - recent).clamp_min(0), k)
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    ptr = output.data_ptr()
    for values in ([0, 1, sink+recent+k, sink+recent+k+1], [8192, capacity, 0, 32768]):
        lengths.copy_(torch.tensor(values, device='cuda', dtype=torch.int32))
        raw.normal_()
        graph.replay()
        expected_scores = torch.full((4, capacity), float('nan'), device='cuda')
        candidates = (lengths-sink-recent).clamp_min(0)
        for row, length in enumerate(candidates.tolist()):
            if length > k:
                expected_scores[row, sink:sink+length] = (raw[row, :, sink:sink+length] * .5).softmax(-1).amax(0)
        expected = reference(expected_scores, candidates, sink, k)
        # The upstream transform emits logical order; membership is the contract.
        torch.testing.assert_close(output.sort(1).values, expected.sort(1).values)
        assert output.data_ptr() == ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_prepared_selector_preserves_scores_and_tied_topk_membership():
    # A short row longer than 2048 and ties at the selection boundary protect
    # full-budget selection, input ownership, and repeatable upstream output.
    sink, k, capacity = 3, 2051, 8195
    scores = torch.randint(-10, 10, (4, capacity), device='cuda').bfloat16()
    lengths = torch.tensor([0, 2049, 4097, 8192], device='cuda', dtype=torch.int32)
    scores[:2].fill_(float('nan'))
    original = scores.clone()
    selector = prepare_omnikv_selection(OmniKVSelectionSpec(sink, k), device=scores.device)
    previous = None
    for _ in range(2):
        output = selector.select(scores, lengths, k)
        torch.testing.assert_close(scores, original, equal_nan=True, atol=0, rtol=0)
        if previous is not None:
            torch.testing.assert_close(output, previous)
        previous = output.clone()
        for row, length in enumerate(lengths.tolist()):
            count = min(k, length)
            selected = output[row, :count].long() - sink
            assert selected.unique().numel() == count
            assert bool(((selected >= 0) & (selected < length)).all())
            if length > k:
                expected = scores[row, sink:sink+length].topk(k).values.sort().values
                torch.testing.assert_close(scores[row, sink+selected].sort().values, expected)
            else:
                torch.testing.assert_close(selected, torch.arange(count, device=scores.device))
                assert bool((output[row, count:] == -1).all())
