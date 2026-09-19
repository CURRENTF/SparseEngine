"""Request ownership must survive batching independently of score arithmetic."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager


def manager_rows():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(h2o_decode_budget=32, h2o_decode_eviction_interval=8)
    manager.seq_id_to_row = [{i: i for i in range(3)} for _ in range(2)]
    manager.row_seq_lens = [[5, 8, 11], [6, 9, 12]]
    manager._h2o_scores = {(l, i): torch.arange(manager.row_seq_lens[l][i]-1).float()
                           for l in range(2) for i in range(3)}
    return manager


def test_score_rows_survive_batch_churn_and_only_replaced_rows_rebuild():
    """Catch batch-keyed relocation, stale history and damage to unscheduled rows."""
    manager = manager_rows()
    pointers = {}
    expected = {k: v.clone() for k, v in manager._h2o_scores.items()}
    with patch('sparseengine.kernels.triton.h2o_score.h2o_headwise_softmax_accumulate_rows'):
        for step, ids in enumerate(([0, 1, 2], [2, 0], [1], [1, 2, 0])):
            replaced = (1, 2) if step == 3 else None
            if replaced:
                expected[replaced] = expected[replaced][::2].clone()
                manager._h2o_scores[replaced] = expected[replaced].clone()
            before = {k: v for k, v in manager._h2o_scores.items()}
            for l in range(2):
                for i in ids:
                    manager.row_seq_lens[l][i] = expected[l, i].numel() + 1
            manager.accumulate_decode_headwise_logits(
                [0, 1], [SimpleNamespace(seq_id=i) for i in ids],
                torch.empty(2, len(ids), 1, 32), softmax_scale=.25,
            )
            for k, row in manager._h2o_scores.items():
                if k[1] not in ids:
                    assert row is before[k]
                    continue
                torch.testing.assert_close(row[:-1], expected[k])
                if k in pointers and k != replaced:
                    assert row.data_ptr() == pointers[k]
                row[-1] = 23  # Simulate the new-token result for the next lifecycle step.
                expected[k] = torch.cat((expected[k], torch.tensor([23.])))
                pointers[k] = row.data_ptr()


def test_score_row_error_does_not_publish_partial_batch():
    """A corrupt later row must not advance an earlier request's score view."""
    manager = manager_rows()
    manager._h2o_scores[1, 2] = torch.zeros(1)
    before = dict(manager._h2o_scores)
    with pytest.raises(RuntimeError, match='align before appending'):
        manager.accumulate_decode_headwise_logits(
            [0, 1], [SimpleNamespace(seq_id=i) for i in range(3)],
            torch.empty(2, 3, 1, 32), softmax_scale=.25,
        )
    assert all(manager._h2o_scores[k] is v for k, v in before.items())
    assert not getattr(manager, '_h2o_decode_score_rows', {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('heads', [1, 5])
@pytest.mark.parametrize('normalize', [True, False])
def test_indirect_score_kernel_matches_torch_and_graph_replay(heads, normalize):
    """Catch row-pointer/stride errors, inactive dereferences and masked padding."""
    from sparseengine.kernels.triton.h2o_score import h2o_headwise_softmax_accumulate_rows
    torch.manual_seed(412)
    logits = torch.randn(2, 4, heads, 258, device='cuda')[:, :3, :, ::2]
    rows = [torch.randn(140, device='cuda') for _ in range(6)]
    lens = [1, 17, 129, 3, 65, 0]
    metadata = torch.tensor([[r.data_ptr(), n] for r, n in zip(rows, lens)],
                            device='cuda', dtype=torch.int64).view(2, 3, 2)
    # An inactive row may carry a null pointer; it must not be dereferenced.
    metadata[-1, -1, 0] = 0
    def run():
        h2o_headwise_softmax_accumulate_rows(logits, metadata, softmax_scale=.25,
                                           normalize_logits=normalize)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        logits.normal_()
        expected = []
        for index, (row, n) in enumerate(zip(rows, lens)):
            row.normal_()
            ref = row.clone()
            if n:
                ref[n-1] = 0
                values = logits[index//3, index%3, :, :n] * .25
                ref[:n] += (values.softmax(-1) if normalize else values).sum(0)
            logits[index//3, index%3, :, n:] = torch.nan
            expected.append(ref)
        graph.replay()
        for row, ref in zip(rows, expected):
            torch.testing.assert_close(row, ref, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('pin_memory', [False, True])
def test_score_metadata_reuse_respects_pin_memory_capability(pin_memory):
    """Repeated metadata DMA must preserve row ownership with or without pinning."""
    manager = manager_rows()
    manager._h2o_scores = {key: value.cuda() for key, value in manager._h2o_scores.items()}
    for ids in ([0, 1, 2], [2, 0], [1, 2, 0]):
        logits = torch.randn(2, len(ids), 3, 32, device='cuda')
        expected = {}
        for layer in range(2):
            for index, seq_id in enumerate(ids):
                previous = manager._h2o_scores[layer, seq_id]
                length = previous.numel() + 1
                manager.row_seq_lens[layer][seq_id] = length
                expected[layer, seq_id] = torch.cat((previous, previous.new_zeros(1)))
                expected[layer, seq_id] += (logits[layer, index, :, :length] * .25).softmax(-1).sum(0)
        with patch('sparseengine.platforms.device_runtime.supports_pin_memory', return_value=pin_memory):
            manager.accumulate_decode_headwise_logits(
                [0, 1], [SimpleNamespace(seq_id=i) for i in ids], logits, softmax_scale=.25,
            )
        assert manager._h2o_decode_score_metadata[0].is_pinned() == pin_memory
        for key, value in expected.items():
            torch.testing.assert_close(manager._h2o_scores[key], value)
