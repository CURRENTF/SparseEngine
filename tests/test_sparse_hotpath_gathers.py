"""Protect row ownership and ragged selection during gather/materialization changes."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('padded', [False, True])
def test_decode_pop_preserves_mixed_layer_rows_and_padding(device, padded):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    m = object.__new__(SnapKVCacheManager)
    m.device = torch.device(device)
    m.runtime_layout = RuntimeLayout.from_config(SimpleNamespace(
        num_hidden_layers=4, layer_types=['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
    ))
    m.num_layers = 4
    m.max_model_len = 16
    m.seq_id_to_row = [{}, {10: 2, 11: 0, 12: 1}, {}, {10: 0, 11: 1, 12: 2}]
    m.row_seq_lens = [np.zeros(3, dtype=np.int32), np.array([2, 5, 7]), np.zeros(3, dtype=np.int32), np.array([4, 3, 6])]
    original_lens = [x.copy() for x in m.row_seq_lens]
    m._num_free_slots = [0, 29, 0, 23]
    m._decode_static_index_buffers = {}
    m.free_slots_stack_tensor = torch.arange(64, device=device, dtype=torch.int32).reshape(2, 32)
    m.buffer_req_to_token_slots_tensor = torch.full((2, 3, 16), -1, device=device, dtype=torch.int32)
    out = torch.full((4, 4), -7, device=device, dtype=torch.int32) if padded else None
    selected, _, rows, wrote = m._allocate_decode_batch_all_layers([12, 10], slot_output=out)
    assert wrote == padded
    for logical, physical, ptr in [(1, 0, 29), (3, 1, 23)]:
        for col, seq in enumerate([12, 10]):
            row = m.seq_id_to_row[logical][seq]
            slot = physical * 32 + ptr - 2 + col
            assert selected[logical, col].item() == slot
            assert rows[logical, col] == row
            assert m.buffer_req_to_token_slots_tensor[physical, row, original_lens[logical][row]].item() == slot
            assert m.row_seq_lens[logical][row] == original_lens[logical][row] + 1
        untouched = m.seq_id_to_row[logical][11]
        assert m.row_seq_lens[logical][untouched] == original_lens[logical][untouched]
        assert (m.buffer_req_to_token_slots_tensor[physical, untouched] == -1).all()
        assert m._num_free_slots[logical] == ptr - 2
    if padded:
        assert selected is out
        assert (selected[:, 2:] == -7).all()
        assert (selected[[0, 2]] == -7).all()
    else:
        assert (selected[[0, 2]] == -1).all()
