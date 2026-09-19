from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparseengine.engine.decode_graph_staging import DecodeGraphHostInputs


def test_decode_graph_host_inputs_share_one_typed_backing() -> None:
    host = DecodeGraphHostInputs.allocate(4, pin_memory=False)

    assert host.storage.dtype == torch.uint8
    assert host.storage.device.type == "cpu"
    assert tuple(tensor.dtype for tensor in host.tensors()) == (
        torch.int64,
        torch.int64,
        torch.int32,
        torch.int32,
        torch.bool,
    )
    backing_pointer = host.storage.untyped_storage().data_ptr()
    assert all(
        tensor.untyped_storage().data_ptr() == backing_pointer
        for tensor in (*host.tensors(), host.sequence_ids)
    )
    assert len({tensor.data_ptr() for tensor in (*host.tensors(), host.sequence_ids)}) == 6


def test_decode_graph_host_inputs_pack_active_and_padding_rows() -> None:
    host = DecodeGraphHostInputs.allocate(4, pin_memory=False)
    seqs = [
        SimpleNamespace(decode_input_token=11, decode_input_position=5, seq_id=101),
        SimpleNamespace(decode_input_token=22, decode_input_position=7, seq_id=202),
    ]

    sequence_ids = host.pack_requests(seqs)
    host.pack_cache_facts(
        context_lens=np.asarray([6, 8], dtype=np.int32),
        request_indices=np.asarray([3, 4], dtype=np.int64),
        real_batch_size=2,
        padding_active=False,
    )

    assert sequence_ids.tolist() == [101, 202]
    assert host.input_ids.tolist() == [11, 22, 11, 11]
    assert host.positions.tolist() == [5, 7, 5, 5]
    assert host.context_lens.tolist() == [6, 8, 6, 6]
    assert host.request_indices.tolist() == [3, 4, 3, 3]
    assert host.active_mask.tolist() == [True, True, False, False]

    pointers = tuple(tensor.data_ptr() for tensor in host.tensors())
    host.pack_requests(
        [SimpleNamespace(decode_input_token=33, decode_input_position=9, seq_id=303)]
    )
    host.pack_cache_facts(
        context_lens=np.asarray([10], dtype=np.int32),
        request_indices=np.asarray([5], dtype=np.int32),
        real_batch_size=1,
        padding_active=False,
    )

    assert tuple(tensor.data_ptr() for tensor in host.tensors()) == pointers
    assert host.input_ids.tolist() == [33, 33, 33, 33]
    assert host.positions.tolist() == [9, 9, 9, 9]
    assert host.context_lens.tolist() == [10, 10, 10, 10]
    assert host.request_indices.tolist() == [5, 5, 5, 5]
    assert host.active_mask.tolist() == [True, False, False, False]


def test_decode_graph_host_inputs_reject_invalid_batch_shapes() -> None:
    host = DecodeGraphHostInputs.allocate(2, pin_memory=False)
    seqs = [SimpleNamespace(decode_input_token=1, decode_input_position=0, seq_id=1)]
    host.pack_requests(seqs)

    with pytest.raises(ValueError, match="context lengths"):
        host.pack_cache_facts(
            context_lens=np.asarray([1, 2], dtype=np.int32),
            request_indices=np.asarray([0], dtype=np.int32),
            real_batch_size=1,
            padding_active=False,
        )
