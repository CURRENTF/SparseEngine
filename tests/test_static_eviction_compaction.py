from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from sparseengine.config import RuntimeLayout
from sparseengine.configs.groups import SparseMethodConfig
from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
from sparseengine.engine.cache_manager.methods.skipkv import (
    SkipKVCacheManager,
    SkipKVSentence,
    SkipKVSequenceState,
)
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import SparseController


def _sequences(lengths: list[int]) -> list[Sequence]:
    seqs = []
    for seq_id, length in enumerate(lengths, start=10):
        seq = Sequence(list(range(length)))
        seq.seq_id = seq_id
        seq.num_prefilled_tokens = length
        seq.current_chunk_size = 0
        seqs.append(seq)
    return seqs


def test_snapkv_graph_uses_static_semantic_path_capacities():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        **vars(
            SparseMethodConfig(
                sparse_method="snapkv",
                sink_keep_tokens=64,
                decode_keep_tokens=8064,
                recent_keep_tokens=64,
            )
        ),
        max_model_len=131072,
    )

    capacity = manager.decode_graph_path_capacity()
    assert capacity == manager.config.max_model_len
    manager.validate_decode_graph_path_capacity(_sequences([2, 8193, 131072]), capacity=capacity)


def _page_table_manager(
    lengths_by_layer: list[list[int]],
    *,
    manager_cls=SnapKVCacheManager,
) -> tuple[SnapKVCacheManager, list[Sequence]]:
    num_layers = len(lengths_by_layer)
    batch_size = len(lengths_by_layer[0])
    assert all(len(lengths) == batch_size for lengths in lengths_by_layer)
    max_len = max(max(lengths) for lengths in lengths_by_layer)
    seqs = _sequences(
        [
            max(lengths_by_layer[layer][row] for layer in range(num_layers))
            for row in range(batch_size)
        ]
    )

    manager = object.__new__(manager_cls)
    manager.device = torch.device("cpu")
    manager.num_layers = num_layers
    manager.num_kv_layers = num_layers
    manager.runtime_layout = RuntimeLayout.dense(num_layers)
    manager._uniform_decode_metadata = True
    manager.seq_id_to_row = [
        {seq.seq_id: row_idx for row_idx, seq in enumerate(seqs)}
        for _layer_idx in range(num_layers)
    ]
    manager.row_seq_lens = [
        np.asarray(lengths, dtype=np.int32) for lengths in lengths_by_layer
    ]
    manager.buffer_req_to_token_slots_tensor = torch.zeros(
        (num_layers, batch_size, max_len),
        dtype=torch.int32,
    )
    manager.buffer_req_to_token_slots = [
        manager.buffer_req_to_token_slots_tensor[layer_idx]
        for layer_idx in range(num_layers)
    ]
    for layer_idx, lengths in enumerate(lengths_by_layer):
        for row_idx, length in enumerate(lengths):
            start = 1000 * layer_idx + 100 * row_idx
            manager.buffer_req_to_token_slots[layer_idx][row_idx, :length] = (
                torch.arange(start, start + length, dtype=torch.int32)
            )

    manager.free_slots_stack_tensor = torch.full(
        (num_layers, 64),
        -1,
        dtype=torch.int32,
    )
    manager.free_slots_stack = [
        manager.free_slots_stack_tensor[layer_idx]
        for layer_idx in range(num_layers)
    ]
    manager._num_free_slots = [2 for _layer_idx in range(num_layers)]
    for layer_idx in range(num_layers):
        manager.free_slots_stack[layer_idx][:2] = torch.tensor(
            [-100 - layer_idx, -200 - layer_idx],
            dtype=torch.int32,
        )
    return manager, seqs


def _assert_same_page_table_state(
    actual: SnapKVCacheManager,
    expected: SnapKVCacheManager,
) -> None:
    assert torch.equal(
        actual.buffer_req_to_token_slots_tensor,
        expected.buffer_req_to_token_slots_tensor,
    )
    assert actual._num_free_slots == expected._num_free_slots
    for actual_lens, expected_lens in zip(actual.row_seq_lens, expected.row_seq_lens):
        assert actual_lens.tolist() == expected_lens.tolist()
    for layer_idx, free_count in enumerate(actual._num_free_slots):
        assert torch.equal(
            actual.free_slots_stack[layer_idx][:free_count],
            expected.free_slots_stack[layer_idx][:free_count],
        )


@pytest.mark.parametrize("already_sorted", [False, True])
def test_shared_layer_batch_compaction_matches_scalar_oracle(already_sorted: bool):
    batched, batched_seqs = _page_table_manager([[6, 6], [6, 6]])
    scalar, scalar_seqs = _page_table_manager([[6, 6], [6, 6]])
    keep = torch.tensor(
        [
            [[5, 0, 2], [3, 1, 5]],
            [[4, 0, 1], [2, 0, 5]],
        ],
        dtype=torch.long,
    )
    if already_sorted:
        keep = torch.sort(keep, dim=2).values

    batched.free_part_slots_batch_layers(
        [0, 1],
        batched_seqs,
        keep,
        keep_indices_sorted=already_sorted,
    )
    for layer_idx in range(2):
        for seq_idx, seq in enumerate(scalar_seqs):
            scalar.free_part_slots(
                layer_idx,
                seq,
                keep[layer_idx, seq_idx],
                keep_indices_sorted=already_sorted,
            )

    _assert_same_page_table_state(batched, scalar)


def test_shared_layer_batch_compaction_zeroes_full_row_tail_like_scalar():
    batched, batched_seqs = _page_table_manager([[6, 6], [8, 8]])
    scalar, scalar_seqs = _page_table_manager([[6, 6], [8, 8]])
    batched.buffer_req_to_token_slots[0][:, 6:] = 99
    scalar.buffer_req_to_token_slots[0][:, 6:] = 99
    keep = torch.tensor(
        [
            [[0, 2, 5], [1, 3, 5]],
            [[0, 4, 7], [2, 5, 6]],
        ],
        dtype=torch.long,
    )

    batched.free_part_slots_batch_layers([0, 1], batched_seqs, keep)
    for layer_idx in range(2):
        for seq_idx, seq in enumerate(scalar_seqs):
            scalar.free_part_slots(layer_idx, seq, keep[layer_idx, seq_idx])

    _assert_same_page_table_state(batched, scalar)


def test_shared_layer_batch_nonuniform_lengths_fall_back_to_scalar_oracle():
    lengths = [[6, 5], [7, 6]]
    batched, batched_seqs = _page_table_manager(lengths)
    scalar, scalar_seqs = _page_table_manager(lengths)
    keep = torch.tensor(
        [
            [[5, 0, 2], [4, 1, 3]],
            [[6, 1, 3], [5, 0, 2]],
        ],
        dtype=torch.long,
    )

    batched.free_part_slots_batch_layers([0, 1], batched_seqs, keep)
    for layer_idx in range(2):
        for seq_idx, seq in enumerate(scalar_seqs):
            scalar.free_part_slots(layer_idx, seq, keep[layer_idx, seq_idx])

    _assert_same_page_table_state(batched, scalar)


def test_shared_layer_batch_compaction_tiles_compatible_layers():
    lengths = [[6], [6], [6], [6]]
    batched, batched_seqs = _page_table_manager(lengths)
    scalar, scalar_seqs = _page_table_manager(lengths)
    batched._page_table_compaction_tile_elements_override = 12
    for manager in (batched, scalar):
        ragged_stacks = []
        for layer_idx in range(4):
            stack = torch.full((10 + layer_idx,), -1, dtype=torch.int32)
            stack[:2] = manager.free_slots_stack[layer_idx][:2]
            ragged_stacks.append(stack)
        manager.free_slots_stack_tensor = None
        manager.free_slots_stack = ragged_stacks
    keep = torch.tensor(
        [
            [[5, 0, 2]],
            [[4, 0, 1]],
            [[5, 1, 3]],
            [[4, 2, 0]],
        ],
        dtype=torch.long,
    )

    with (
        patch.object(
            batched,
            "_compact_uniform_layer_tile",
            wraps=batched._compact_uniform_layer_tile,
        ) as compact_layer_tile,
        patch.object(
            batched,
            "_compact_uniform_rows_bounded",
            wraps=batched._compact_uniform_rows_bounded,
        ) as compact_rows,
    ):
        batched.free_part_slots_batch_layers(
            list(range(4)),
            batched_seqs,
            keep,
        )

    assert compact_layer_tile.call_count == 2
    assert compact_layer_tile.call_count < len(lengths)
    compact_rows.assert_not_called()
    for layer_idx in range(4):
        scalar.free_part_slots(layer_idx, scalar_seqs[0], keep[layer_idx, 0])
    _assert_same_page_table_state(batched, scalar)


def test_shared_layer_batch_compaction_tiles_rows_with_bounded_temporaries():
    batched, batched_seqs = _page_table_manager([[6, 6], [6, 6]])
    scalar, scalar_seqs = _page_table_manager([[6, 6], [6, 6]])
    batched._page_table_compaction_tile_elements_override = 6
    keep = torch.tensor(
        [
            [[5, 0, 2], [3, 1, 5]],
            [[4, 0, 1], [2, 0, 5]],
        ],
        dtype=torch.long,
    )

    with patch.object(
        batched,
        "_compact_uniform_row_tile",
        wraps=batched._compact_uniform_row_tile,
    ) as compact_tile:
        batched.free_part_slots_batch_layers([0, 1], batched_seqs, keep)

    assert compact_tile.call_count == 4
    for layer_idx in range(2):
        for seq_idx, seq in enumerate(scalar_seqs):
            scalar.free_part_slots(layer_idx, seq, keep[layer_idx, seq_idx])
    _assert_same_page_table_state(batched, scalar)


def test_shared_layer_batch_compaction_tiles_single_long_rows():
    batched, batched_seqs = _page_table_manager([[8], [8]])
    scalar, scalar_seqs = _page_table_manager([[8], [8]])
    batched._page_table_compaction_tile_elements_override = 4
    keep = torch.tensor(
        [
            [[7, 0, 3]],
            [[6, 1, 4]],
        ],
        dtype=torch.long,
    )

    with (
        patch.object(
            batched,
            "_compact_uniform_layer_tile",
            wraps=batched._compact_uniform_layer_tile,
        ) as compact_layer_tile,
        patch.object(
            batched,
            "_compact_uniform_rows_bounded",
            wraps=batched._compact_uniform_rows_bounded,
        ) as compact_rows,
        patch.object(
            batched,
            "_compact_single_row_column_tiles",
            wraps=batched._compact_single_row_column_tiles,
        ) as compact_row,
    ):
        batched.free_part_slots_batch_layers([0, 1], batched_seqs, keep)

    compact_layer_tile.assert_not_called()
    assert compact_rows.call_count == 2
    assert compact_row.call_count == 2
    for layer_idx in range(2):
        scalar.free_part_slots(layer_idx, scalar_seqs[0], keep[layer_idx, 0])
    _assert_same_page_table_state(batched, scalar)


@pytest.mark.parametrize(
    ("keep", "message"),
    [
        (torch.tensor([[[0, 2, 6], [1, 3, 5]]]), "out of bounds"),
        (torch.tensor([[[0, 2, 2], [1, 3, 5]]]), "must be unique"),
    ],
)
def test_shared_compaction_rejects_invalid_indices_without_mutation(
    keep: torch.Tensor,
    message: str,
):
    manager, seqs = _page_table_manager([[6, 6]])
    old_slots = manager.buffer_req_to_token_slots_tensor.clone()
    old_stack = manager.free_slots_stack_tensor.clone()
    old_lens = manager.row_seq_lens[0].copy()

    with pytest.raises(RuntimeError, match=message):
        manager.free_part_slots_batch_layers([0], seqs, keep)

    assert torch.equal(manager.buffer_req_to_token_slots_tensor, old_slots)
    assert torch.equal(manager.free_slots_stack_tensor, old_stack)
    assert manager.row_seq_lens[0].tolist() == old_lens.tolist()
    assert manager._num_free_slots == [2]
    assert manager._uniform_decode_metadata


def test_h2o_duplicate_indices_fail_before_score_or_page_table_mutation():
    manager, seqs = _page_table_manager([[6]], manager_cls=H2OCacheManager)
    score = torch.arange(6, dtype=torch.float32)
    score_key = (0, seqs[0].seq_id)
    manager._h2o_scores = {score_key: score}
    manager._h2o_decode_score_signature = ("active",)
    manager._h2o_decode_score_length = 6
    old_slots = manager.buffer_req_to_token_slots_tensor.clone()
    old_stack = manager.free_slots_stack_tensor.clone()
    old_lens = manager.row_seq_lens[0].copy()

    with pytest.raises(RuntimeError, match="must be unique"):
        manager.free_part_slots(0, seqs[0], torch.tensor([0, 2, 2]))

    assert manager._h2o_scores[score_key] is score
    assert torch.equal(manager.buffer_req_to_token_slots_tensor, old_slots)
    assert torch.equal(manager.free_slots_stack_tensor, old_stack)
    assert manager.row_seq_lens[0].tolist() == old_lens.tolist()
    assert manager._num_free_slots == [2]
    assert manager._h2o_decode_score_signature == ("active",)
    assert manager._h2o_decode_score_length == 6
    assert manager._uniform_decode_metadata


def test_rkv_layer_batch_compaction_invalidates_query_cache_rows():
    manager, seqs = _page_table_manager(
        [[6, 5], [6, 5]],
        manager_cls=RKVCacheManager,
    )
    manager._rkv_query_cache_enabled = True
    manager._rkv_batch_clear_query_cache_rows = True
    manager._rkv_query_cache = [
        torch.zeros((2, 2, 1, 1), dtype=torch.float32) for _ in range(2)
    ]
    manager._rkv_query_positions = [
        torch.arange(4, dtype=torch.int32).view(2, 2) for _ in range(2)
    ]
    keep = torch.tensor(
        [
            [[0, 2, 5], [1, 3, 4]],
            [[1, 2, 4], [0, 3, 4]],
        ],
        dtype=torch.long,
    )

    manager.free_part_slots_batch_layers([0, 1], seqs, keep)

    assert all(
        torch.equal(positions, torch.full_like(positions, -1))
        for positions in manager._rkv_query_positions
    )
    assert manager._num_free_slots == [7, 7]


def test_skipkv_layer_batch_compaction_updates_only_cache_side_state():
    manager, seqs = _page_table_manager(
        [[6, 5], [6, 5]],
        manager_cls=SkipKVCacheManager,
    )
    manager._rkv_query_cache_enabled = False
    manager._skipkv_row_gen_indices = [
        {
            0: [10, 11, 12, 13, 14, 15],
            1: [20, 21, 22, 23, 24, 25],
        }
        for _ in range(2)
    ]
    state = SkipKVSequenceState(
        num_prompt_tokens=10,
        open_start_gen=16,
        open_embedding_sum=torch.tensor([3.0]),
        open_embedding_count=2,
        non_execution_count=1,
    )
    sentence = SkipKVSentence(
        start_gen=11,
        end_gen=15,
        embedding=torch.tensor([1.0]),
    )
    state.sentences.append(sentence)
    manager._skipkv_seq_states = {seqs[0].seq_id: state}
    keep = torch.tensor(
        [
            [[5, 0, 2], [1, 3, 4]],
            [[4, 0, 2], [4, 1, 3]],
        ],
        dtype=torch.long,
    )

    manager.free_part_slots_batch_layers([0, 1], seqs, keep)

    assert manager._skipkv_row_gen_indices[0][0] == [10, 12, 15]
    assert manager._skipkv_row_gen_indices[1][0] == [10, 12, 14]
    assert sentence.cache_ranges == {0: (1, 2), 1: (1, 3)}
    assert state.open_start_gen == 16
    assert state.open_embedding_count == 2
    assert state.non_execution_count == 1


class _RecordingPrefillManager:
    device = torch.device("cpu")

    def __init__(self, scores: dict[tuple[int, int], torch.Tensor]):
        self.scores = scores
        self.scalar_calls = []
        self.batch_calls = []
        self.layer_calls = []

    def pop_prefill_attention_score(self, layer_idx, seq):
        return self.scores.pop((int(layer_idx), int(seq.seq_id)), None)

    def free_part_slots(self, layer_idx, seq, keep_indices):
        self.scalar_calls.append((int(layer_idx), int(seq.seq_id), keep_indices.clone()))

    def free_part_slots_batch(self, layer_idx, seqs, keep_indices):
        self.batch_calls.append((int(layer_idx), list(seqs), keep_indices.clone()))

    def free_part_slots_batch_layers(self, layer_indices, seqs, keep_indices):
        self.layer_calls.append((list(layer_indices), list(seqs), keep_indices.clone()))


def _controller_config(method: str, num_layers: int, ratios=None):
    return SimpleNamespace(
        sparse_method=method,
        validate_runtime_invariants=False,
        obs_layer_ids=[0],
        full_attention_layers=[],
        hf_config=SimpleNamespace(
            num_hidden_layers=num_layers,
            hidden_size=8,
            num_attention_heads=1,
        ),
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float32",
        snapkv_num_full_layers=0,
        pyramid_layer_ratios=ratios,
        pool_kernel_size=1,
    )


def _final_prefill_seq(seq_id: int, kv_len: int, *, final: bool = True) -> Sequence:
    prompt_len = kv_len if final else kv_len + 4
    seq = Sequence(list(range(prompt_len)))
    seq.seq_id = seq_id
    seq.num_prefilled_tokens = kv_len - 2
    seq.current_chunk_size = 2
    return seq


def test_snapkv_final_prefill_compacts_unique_length_across_layers():
    seqs = [
        _final_prefill_seq(10, 8),
        _final_prefill_seq(11, 8),
        _final_prefill_seq(12, 9),
        _final_prefill_seq(13, 8, final=False),
    ]
    scores = {
        (layer_idx, seq.seq_id): torch.arange(
            int(seq.num_prefilled_tokens + seq.current_chunk_size),
            dtype=torch.float32,
        )
        + 0.01 * layer_idx
        for layer_idx in range(2)
        for seq in seqs[:3]
    }
    manager = _RecordingPrefillManager(scores)
    controller = SparseController(_controller_config("snapkv", 2), manager)

    controller.runtime._snapkv_prefill_eviction(seqs)

    assert len(manager.layer_calls) == 2
    layer_indices, grouped_seqs, keep_indices = manager.layer_calls[0]
    assert layer_indices == [0, 1]
    assert [seq.seq_id for seq in grouped_seqs] == [10, 11]
    assert keep_indices.shape == (2, 2, 6)
    layer_indices, grouped_seqs, keep_indices = manager.layer_calls[1]
    assert layer_indices == [0, 1]
    assert [seq.seq_id for seq in grouped_seqs] == [12]
    assert keep_indices.shape == (2, 1, 6)
    assert not manager.scalar_calls
    assert not manager.batch_calls
    assert not manager.scores


def test_snapkv_final_prefill_single_layer_singleton_uses_scalar_fallback():
    seq = _final_prefill_seq(14, 8)
    manager = _RecordingPrefillManager(
        {(0, seq.seq_id): torch.arange(8, dtype=torch.float32)}
    )
    controller = SparseController(_controller_config("snapkv", 1), manager)

    controller.runtime._snapkv_prefill_eviction([seq])

    assert [(layer, seq_id) for layer, seq_id, _keep in manager.scalar_calls] == [
        (0, 14)
    ]
    assert not manager.batch_calls
    assert not manager.layer_calls
    assert not manager.scores


def test_pyramidkv_final_prefill_groups_layers_by_effective_budget():
    seqs = [_final_prefill_seq(20, 8), _final_prefill_seq(21, 8)]
    scores = {
        (layer_idx, seq.seq_id): torch.arange(8, dtype=torch.float32)
        + 0.01 * layer_idx
        for layer_idx in range(3)
        for seq in seqs
    }
    manager = _RecordingPrefillManager(scores)
    controller = SparseController(
        _controller_config("pyramidkv", 3, ratios=[1.0, 0.5, 0.5]),
        manager,
    )

    controller.runtime._snapkv_prefill_eviction(seqs)

    assert len(manager.layer_calls) == 1
    layer_indices, grouped_seqs, keep_indices = manager.layer_calls[0]
    assert layer_indices == [1, 2]
    assert [seq.seq_id for seq in grouped_seqs] == [20, 21]
    assert keep_indices.shape == (2, 2, 4)
    assert len(manager.batch_calls) == 1
    assert manager.batch_calls[0][0] == 0
    assert manager.batch_calls[0][2].shape == (2, 6)
    assert not manager.scalar_calls


def test_h2o_scalar_final_prefill_keeps_dense_relocation_dispatch():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        h2o_decode_budget=4,
        h2o_prefill_budget=8,
        h2o_recent_ratio=0.5,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._preflight_final_prefill_dense_capacity = Mock()
    manager._try_batched_evict = Mock(return_value=False)
    manager._physical_row_len = Mock(return_value=6)
    score = torch.arange(6, dtype=torch.float32)
    manager._require_score_length = Mock(return_value=score)
    keep = torch.tensor([1, 3, 4, 5], dtype=torch.long)
    manager.select_h2o_indices = Mock(return_value=keep)
    manager._compact_final_prefill_dense_batch = Mock()
    manager.free_part_slots = Mock()
    manager._score_key = lambda layer_idx, seq_id: (int(layer_idx), int(seq_id))
    manager._h2o_scores = {}
    manager._h2o_counters = {
        "intermediate_prefill_evictions": 0,
        "final_prefill_evictions": 0,
        "decode_evictions": 0,
        "dropped_tokens": 0,
    }
    seq = _final_prefill_seq(30, 6)

    manager._evict([seq], is_prefill=True)

    manager._compact_final_prefill_dense_batch.assert_called_once()
    manager.free_part_slots.assert_not_called()
    assert torch.equal(manager._h2o_scores[(0, 30)], score[keep])
    assert manager._h2o_counters["final_prefill_evictions"] == 1
    assert manager._h2o_counters["dropped_tokens"] == 2
