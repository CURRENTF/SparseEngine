import hashlib
import math
import pickle
import random
import tempfile
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sparseengine.config import Config
from sparseengine.engine.cache_manager.methods.quest import (
    QuestCacheManager,
    QuestDecodeGraphState,
    QuestPrefixBlockPayload,
)
from sparseengine.configs.model import RuntimeLayout
from sparseengine.engine.cache_manager import MlaLatentPayload
from sparseengine.engine.cache_manager.methods.omnikv.manager import OmniKVCacheManager
from sparseengine.engine.cache_manager.standard import StandardCacheManager, StandardPrefixBlockPayload
from sparseengine.engine.cache_manager.prefix_cache_mixin import PrefixLookupCache
from sparseengine.engine.cache_manager.storage import MlaLatentStorage
from sparseengine.engine.cache_manager.prefix_offload import (
    QuestPrefixOffloadController,
    StandardPrefixOffloadController,
)
from sparseengine.engine.decode_graph_contract import (
    DecodeGraphContract,
    DecodeGraphInputs,
)
from sparseengine.engine.prefix_cache import (
    PrefixBlockResidency,
    PrefixCacheBlock,
    PrefixTransferKind,
    RadixPrefixIndex,
    RadixTreeBackend,
    build_prefix_cache_fingerprint,
    resolve_prefix_cache_block_size,
    usable_prefix_cache_tokens,
)
from sparseengine.engine.prefix_prune import select_global_keep_indices
from sparseengine.engine.sequence import Sequence
from sparseengine.platforms import device_runtime


@pytest.mark.parametrize("block_size", [1, 3, 16])
@pytest.mark.parametrize("limit", [None, -1, 0, 1, 17, 100])
def test_bulk_prefix_ids_preserve_signed_little_endian_hash_chain(block_size, limit):
    tokens = [-(2**63), 2**63 - 1, -1, 0, 1] * 7
    index = RadixPrefixIndex(block_size=block_size, fingerprint=b"wire-compatibility")
    usable = len(tokens) if limit is None else min(limit, len(tokens))
    expected = []
    parent = None
    for offset in range(0, max(0, usable - block_size + 1), block_size):
        payload = b"".join(int(x).to_bytes(8, "little", signed=True)
                           for x in tokens[offset:offset + block_size])
        prefix = b"\x00" if parent is None else b"\x01" + parent
        parent = hashlib.sha256(b"wire-compatibility" + prefix + payload).digest()
        expected.append(parent)
    assert index.block_ids_for_tokens(tokens, max_tokens=limit) == expected
    assert index.block_ids_for_tokens(tuple(tokens), max_tokens=limit) == expected


def _cfg(method="", salt="", block_size=4):
    return SimpleNamespace(
        model="/models/qwen",
        hf_config=SimpleNamespace(model_type="qwen2", dtype=torch.float16),
        tensor_parallel_size=1,
        expert_parallel_size=1,
        data_parallel_size=1,
        sparse_method=method,
        prefix_cache_salt=salt,
        prefix_cache_block_size=block_size,
        decode_keep_tokens=64,
        sink_keep_tokens=4,
        recent_keep_tokens=8,
        full_attention_layers=[0],
        obs_layer_ids=None,
        quest_chunk_size=4,
        quest_skip_layers=2,
    )


def _insert_tokens(index: RadixPrefixIndex, token_ids: list[int]) -> bytes:
    parent_block_id = None
    last_block_id = None
    for logical_idx, start in enumerate(range(0, len(token_ids), index.block_size)):
        block_tokens = token_ids[start: start + index.block_size]
        stable_block_id = index.stable_block_id(block_tokens, parent_block_id)
        block = PrefixCacheBlock(
            stable_block_id=stable_block_id,
            parent_block_id=parent_block_id,
            block_size=index.block_size,
            logical_block_idx=logical_idx,
            payload=SimpleNamespace(name="dummy"),
            token_ids=tuple(block_tokens),
        )
        index.insert_block(block)
        parent_block_id = stable_block_id
        last_block_id = stable_block_id
    assert last_block_id is not None
    return last_block_id


def _hf_config():
    return SimpleNamespace(
        model_type="qwen2",
        dtype=torch.float16,
        max_position_embeddings=32768,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


def _make_config(**kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
            return Config(model=str(model_dir), **kwargs)


def _make_standard_manager_for_prefix(block_size=2, method=""):
    cfg = _cfg(method=method, block_size=block_size)
    cfg.num_kvcache_slots = 90
    fingerprint = build_prefix_cache_fingerprint(cfg, block_size)
    manager_type = OmniKVCacheManager if method == "omnikv" else StandardCacheManager
    manager = object.__new__(manager_type)
    manager.parallel_context = SimpleNamespace(attn_tp_size=1)
    manager.config = cfg
    manager.device = torch.device("cpu")
    manager.enable_prefix_caching = True
    manager.prefix_cache_block_size = block_size
    manager.prefix_cache = RadixPrefixIndex(block_size=block_size, fingerprint=fingerprint)
    manager.layer_batch_state = SimpleNamespace()
    manager.buffer_req_to_token_slots = torch.zeros((2, 16), dtype=torch.int32)
    manager.free_slots_stack = torch.arange(100, dtype=torch.int32)
    manager._num_free_slots = 90
    manager.seq_id_to_row = {}
    manager.free_rows = deque([0, 1])
    manager.row_seq_lens = np.zeros((2,), dtype=np.int32)
    manager.seq_id_to_prefix_blocks = {}
    manager.seq_id_to_cached_ranges = {}
    manager._scheduler_capacity_snapshot_depth = 0
    manager._scheduler_freeable_block_ids = None
    manager._scheduler_reclaimable_slots = None
    manager.prefix_offload_controller = None
    manager._prefix_offload_step_h2d_operations = {}
    manager._init_prefix_cache_runtime()
    return manager


def test_prefix_prune_selects_one_stable_global_token_mask():
    scores = torch.tensor([0.5, 0.9, 0.9, 0.1])
    keep = select_global_keep_indices(
        scores,
        keep_tokens=3,
        protected_indices=torch.tensor([3]),
    )
    assert keep.tolist() == [1, 2, 3]


def test_standard_prefix_prune_compacts_payload_and_preserves_logical_positions():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.row_logical_lens = np.zeros((2,), dtype=np.int32)
    manager._seq_prefix_cached_cursor = {}
    parent = None
    blocks = []
    for logical_idx, (tokens, slots) in enumerate(
        [([1, 2], [10, 11]), ([3, 4], [12, 13])]
    ):
        block_id = manager.prefix_cache.stable_block_id(tokens, parent)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent,
            block_size=2,
            logical_block_idx=logical_idx,
            payload=StandardPrefixBlockPayload(
                token_slots=torch.tensor(slots, dtype=torch.int32)
            ),
            token_ids=tuple(tokens),
        )
        manager.prefix_cache.insert_block(block)
        blocks.append(block)
        parent = block_id
    _remove_free_slots(manager, [10, 11, 12, 13])
    free_before = manager.num_free_slots

    result = manager.prefix_cache_prune(
        [1, 2, 3, 4],
        range_start=0,
        range_end=4,
        keep_indices=torch.tensor([0, 3]),
        policy="snapkv_global",
        prune_id="prune-test",
    )

    assert result["freed_device_slots"] == 2
    assert manager.num_free_slots == free_before + 2
    assert blocks[0].payload.retained_offsets == (0,)
    assert blocks[1].payload.retained_offsets == (1,)
    assert blocks[0].payload.token_slots.tolist() == [10]
    assert blocks[1].payload.token_slots.tolist() == [13]
    inspected = manager.prefix_cache.inspect_prefix([1, 2, 3, 4])
    assert inspected["path_blocks"][-1]["prune"]["quality_degraded"] is True

    seq = Sequence([1, 2, 3, 4, 5])
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 4
    seq.prefix_cache_hit_block_count = 2
    seq.prefix_cache_hit_last_block_id = parent
    seq.prefix_cache_block_size = 2
    manager._attach_prefix_cache_if_needed(seq)
    row = manager.seq_id_to_row[seq.seq_id]
    assert manager.row_seq_lens[row] == 2
    assert manager.row_logical_lens[row] == 4
    assert manager.buffer_req_to_token_slots[row, :2].tolist() == [10, 13]

    seq.num_prefilled_tokens = 4
    seq.current_chunk_size = 1
    input_ids, positions, _ = manager._prepare_prefill([seq])
    assert input_ids.tolist() == [5]
    assert positions.tolist() == [4]
    assert manager.layer_batch_state.context_lens.tolist() == [3]
    assert manager.row_logical_lens[row] == 5

    # Regression: committing a newly completed block after a compacted prefix
    # must use physical row offsets, including a block spanning two steps.
    manager.on_forward_end([seq], is_prefill=True)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(6)
    manager._prepare_decode([seq])
    manager.on_forward_end([seq], is_prefill=False)
    child_id = manager.prefix_cache.stable_block_id([5, 6], parent)
    child = manager.prefix_cache.get_block(child_id)
    assert child.payload.token_slots.tolist() == manager.buffer_req_to_token_slots[row, 2:4].tolist()
    assert manager.seq_id_to_cached_ranges[seq.seq_id] == [(0, 2), (2, 4)]
    manager.free_seq(seq.seq_id)
    assert all(block.ref_count == 0 for block in manager.prefix_cache.blocks.values())


def test_standard_prefix_prune_rejects_referenced_blocks_without_mutation():
    manager = _make_standard_manager_for_prefix(block_size=2)
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(
            token_slots=torch.tensor([10, 11], dtype=torch.int32)
        ),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(block)
    manager.prefix_cache.acquire_block_ref(block)

    with pytest.raises(RuntimeError, match="idle"):
        manager.prefix_cache_prune(
            [1, 2],
            range_start=0,
            range_end=2,
            keep_indices=torch.tensor([0]),
            policy="kvzip_global",
            prune_id="blocked",
        )
    assert block.payload.retained_offsets is None
    assert block.payload.token_slots.tolist() == [10, 11]


def test_standard_prefix_prune_rejects_unimplemented_recompression_without_mutation():
    manager = _make_standard_manager_for_prefix(block_size=2)
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(
            token_slots=torch.tensor([10, 11], dtype=torch.int32)
        ),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(block)
    _remove_free_slots(manager, [10, 11])
    manager.prefix_cache_prune(
        [1, 2],
        range_start=0,
        range_end=2,
        keep_indices=torch.tensor([0]),
        policy="snapkv_global",
        prune_id="first-prune",
    )
    free_before = manager.num_free_slots

    with pytest.raises(RuntimeError, match="allow_recompress is not implemented"):
        manager.prefix_cache_prune(
            [1, 2],
            range_start=0,
            range_end=2,
            keep_indices=torch.tensor([], dtype=torch.long),
            policy="snapkv_global",
            prune_id="second-prune",
            allow_recompress=True,
        )

    assert manager.num_free_slots == free_before
    assert block.payload.retained_offsets == (0,)
    assert block.payload.token_slots.tolist() == [10]


def test_quest_prefix_prune_drops_whole_pages_and_preserves_logical_positions():
    manager = _make_quest_manager_for_prefix(page_size=2)
    parent = None
    blocks = []
    for logical_idx, (tokens, page) in enumerate((([1, 2], 3), ([3, 4], 4))):
        block_id = manager.prefix_cache.stable_block_id(tokens, parent)
        slots = torch.tensor([page * 2, page * 2 + 1], dtype=torch.int32)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent,
            block_size=2,
            logical_block_idx=logical_idx,
            payload=QuestPrefixBlockPayload(block_slot=page, token_slots=slots),
            token_ids=tuple(tokens),
        )
        manager.prefix_cache.insert_block(block)
        _remove_free_page(manager, page)
        blocks.append(block)
        parent = block_id
    free_before = manager.num_free_slots

    result = manager.prefix_cache_prune(
        [1, 2, 3, 4],
        range_start=0,
        range_end=4,
        keep_indices=torch.tensor([2, 3]),
        policy="kvzip_global",
        prune_id="quest-pages",
    )

    assert result["selection_granularity"] == "quest_page"
    assert result["freed_device_slots"] == 2
    assert manager.num_free_slots == free_before + 2
    assert blocks[0].payload.retained_offsets == ()
    assert blocks[0].payload.token_slots.numel() == 0
    assert blocks[1].payload.retained_offsets == (0, 1)
    assert manager._prefix_evictable_slots() == 2
    assert manager.prefix_cache_match([1, 2, 3, 4, 5])["resident_kv_tokens"] == 2

    with pytest.raises(RuntimeError, match="already pruned"):
        manager.validate_prefix_cache_prune_target(
            [1, 2, 3, 4], ranges=[(2, 4)]
        )

    seq = Sequence([1, 2, 3, 4, 5])
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 4
    seq.prefix_cache_hit_block_count = 2
    seq.prefix_cache_hit_last_block_id = parent
    seq.prefix_cache_block_size = 2
    manager._attach_prefix_cache_if_needed(seq)
    row = manager.seq_id_to_row[seq.seq_id]
    assert manager.row_seq_lens[row] == 2
    assert manager.row_logical_lens[row] == 4
    assert manager.buffer_req_to_token_slots[row, :2].tolist() == [8, 9]

    seq.num_prefilled_tokens = 4
    seq.current_chunk_size = 1
    _, positions, _ = manager._prepare_prefill([seq])
    assert positions.tolist() == [4]
    assert manager.layer_batch_state.context_lens.tolist() == [3]
    assert manager.row_logical_lens[row] == 5
    manager.begin_prefix_prune_scoring(
        seq_id=seq.seq_id,
        candidate_start=0,
        query_start=4,
        query_end=5,
    )
    assert manager._prefix_prune_physical_score_window() == (2, 3, 0)
    assert manager._prefix_prune_scoring["logical_positions"].tolist() == [2, 3, 4]
    manager.abort_prefix_prune_scoring()


def test_quest_prefix_prune_rejects_partial_page_mask_without_mutation():
    manager = _make_quest_manager_for_prefix(page_size=2)
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=3,
            token_slots=torch.tensor([6, 7], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(block)
    _remove_free_page(manager, 3)
    free_before = manager.num_free_slots

    with pytest.raises(ValueError, match="partial page"):
        manager.prefix_cache_prune(
            [1, 2],
            range_start=0,
            range_end=2,
            keep_indices=torch.tensor([0]),
            policy="snapkv_global",
            prune_id="partial",
        )
    assert block.payload.retained_offsets is None
    assert block.payload.token_slots.tolist() == [6, 7]
    assert manager.num_free_slots == free_before


def test_quest_prefix_prune_multi_range_keeps_gap_page_unchanged():
    manager = _make_quest_manager_for_prefix(page_size=2)
    parent = None
    blocks = []
    for logical_idx in range(3):
        tokens = [logical_idx * 2 + 1, logical_idx * 2 + 2]
        page = logical_idx + 2
        block_id = manager.prefix_cache.stable_block_id(tokens, parent)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent,
            block_size=2,
            logical_block_idx=logical_idx,
            payload=QuestPrefixBlockPayload(
                block_slot=page,
                token_slots=torch.tensor([page * 2, page * 2 + 1], dtype=torch.int32),
            ),
            token_ids=tuple(tokens),
        )
        manager.prefix_cache.insert_block(block)
        _remove_free_page(manager, page)
        blocks.append(block)
        parent = block_id

    manager.prefix_cache_prune(
        [1, 2, 3, 4, 5, 6],
        ranges=[(0, 2), (4, 6)],
        keep_indices=torch.tensor([2, 3]),
        policy="kvzip_global",
        prune_id="quest-ranges",
    )

    assert blocks[0].payload.retained_offsets == ()
    assert blocks[1].payload.retained_offsets is None
    assert blocks[1].payload.token_slots.tolist() == [6, 7]
    assert blocks[2].payload.retained_offsets == (0, 1)


def test_quest_capacity_cache_reuses_weights_and_invalidates_compaction():
    manager = _make_quest_manager_for_prefix(page_size=2)
    index = manager.prefix_cache
    leaf_id = _insert_tokens(index, [1, 2, 3, 4])
    chain = index.get_chain(leaf_id, 2)
    for page, block in enumerate(chain, start=3):
        block.payload = QuestPrefixBlockPayload(
            block_slot=page,
            token_slots=torch.tensor([page * 2, page * 2 + 1], dtype=torch.int32),
        )

    with patch.object(
        manager, "_quest_payload", wraps=manager._quest_payload
    ) as payload:
        for _ in range(3):
            with manager.scheduler_capacity_snapshot():
                assert manager.prompt_admission_free_slots() == 24
                assert manager.prefill_step_free_slots() == 24
                assert manager.decode_step_free_slots() == 24
        assert payload.call_count == len(chain)

        leaf = chain[-1]
        leaf.payload.retained_offsets = ()
        leaf.payload.block_slot = None
        leaf.payload.token_slots = torch.empty(0, dtype=torch.int32)
        index.mark_payload_compacted([leaf])
        assert manager.prompt_admission_free_slots() == 22
        assert payload.call_count == len(chain) * 2


def test_quest_prefix_hit_capacity_reuses_chain_and_invalidates_compaction():
    manager = _make_quest_manager_for_prefix(page_size=2)
    index = manager.prefix_cache
    leaf_id = _insert_tokens(index, [1, 2, 3, 4])
    chain = index.get_chain(leaf_id, 2)
    for page, block in enumerate(chain, start=3):
        block.payload = QuestPrefixBlockPayload(
            block_slot=page,
            token_slots=torch.tensor([page * 2, page * 2 + 1], dtype=torch.int32),
        )
    seq = Sequence([1, 2, 3, 4, 5])
    seq.prefix_cache_hit_len = 4
    seq.prefix_cache_hit_block_count = 2
    seq.prefix_cache_hit_last_block_id = leaf_id

    with patch.object(index, "get_chain", wraps=index.get_chain) as get_chain, patch.object(
        manager, "_quest_payload", wraps=manager._quest_payload
    ) as payload:
        for _ in range(3):
            assert manager.prompt_admission_cost(seq) == 6
        assert get_chain.call_count == 1
        assert payload.call_count == len(chain)

        leaf = chain[-1]
        leaf.payload.retained_offsets = ()
        leaf.payload.block_slot = None
        leaf.payload.token_slots = torch.empty(0, dtype=torch.int32)
        index.mark_payload_compacted([leaf])
        assert manager.prompt_admission_cost(seq) == 4
        assert get_chain.call_count == 1
        assert payload.call_count == len(chain) * 2


def test_quest_shared_prefix_admission_exposes_stable_page_costs():
    manager = _make_quest_manager_for_prefix(page_size=2)
    index = manager.prefix_cache
    leaf_id = _insert_tokens(index, [1, 2, 3, 4])
    chain = index.get_chain(leaf_id, 2)
    for page, block in enumerate(chain, start=3):
        block.payload = QuestPrefixBlockPayload(
            block_slot=page,
            token_slots=torch.tensor([page * 2, page * 2 + 1], dtype=torch.int32),
        )
    seq = Sequence([1, 2, 3, 4, 5])
    seq.prefix_cache_hit_len = 4
    seq.prefix_cache_hit_block_count = 2
    seq.prefix_cache_hit_last_block_id = leaf_id

    assert manager.prompt_admission_cost(seq) == 6
    assert manager.prompt_admission_shared_costs(seq) == {
        "slots": {block.stable_block_id: 2 for block in chain}
    }


def test_quest_prefill_page_reservation_returns_pages_and_rows():
    manager = _make_quest_manager_for_prefix(page_size=2)
    free_pages = manager._num_free_pages
    free_rows = set(manager.free_rows)

    with manager.reserve_prefill_slots([(17, 3)]) as admitted:
        assert admitted == 1
        slots = manager._allocate(17, 3)
        assert slots.numel() == 3
        assert manager.row_logical_lens[manager.seq_id_to_row[17]] == 3

    assert 17 not in manager.seq_id_to_row
    assert manager._num_free_pages == free_pages
    assert set(manager.free_rows) == free_rows


class _FakeHostPool:
    def __init__(self, free_blocks=100):
        self.free_blocks = int(free_blocks)
        self.capacity_blocks = int(free_blocks)


class _FakePrefixOffloadController:
    def __init__(self, prefix_cache, free_blocks=100):
        self.prefix_cache = prefix_cache
        self.host_pool = _FakeHostPool(free_blocks)
        self.d2h_operations = []
        self.submitted_d2h = []
        self.submitted_h2d = []
        self._h2d_by_block_id = {}
        self.waited_layers = []
        self.reset_count = 0
        self.synchronize_count = 0

    def poll(self):
        return (0, 0)

    def wait_oldest_d2h(self):
        return False

    def submit_d2h(self, blocks):
        for block in blocks:
            self.prefix_cache.begin_d2h(block)
        self.submitted_d2h.append(list(blocks))
        self.host_pool.free_blocks -= len(blocks)

    def submit_h2d(self, blocks):
        for block in blocks:
            self.prefix_cache.begin_h2d(block)
        operation = SimpleNamespace(blocks=list(blocks), layer_events=[object(), object()])
        self.submitted_h2d.append(operation)
        for block in blocks:
            self._h2d_by_block_id[block.stable_block_id] = operation
        return operation

    def h2d_operation_for_block(self, block):
        return self._h2d_by_block_id.get(block.stable_block_id)

    def wait_for_layer(self, operation, layer_index):
        self.waited_layers.append((operation, int(layer_index)))

    def free_host_payloads(self, blocks):
        for block in blocks:
            block.payload.host_block_index = None
        self.host_pool.free_blocks += len(blocks)

    def synchronize_all(self):
        self.synchronize_count += 1

    def reset(self):
        self.reset_count += 1

    def stats(self):
        return {}


class _CompletingD2HPrefixOffloadController(_FakePrefixOffloadController):
    def wait_oldest_d2h(self):
        blocks = [
            block
            for block in self.prefix_cache.blocks.values()
            if block.residency.transfer == PrefixTransferKind.D2H
        ]
        if not blocks:
            return False
        for host_index, block in enumerate(blocks):
            block.payload.host_block_index = host_index
            self.prefix_cache.finish_d2h(block)
        return True

    def synchronize_all(self):
        super().synchronize_all()
        while self.wait_oldest_d2h():
            pass


def _make_quest_manager_for_prefix(page_size=2):
    cfg = _cfg(method="quest", block_size=page_size)
    fingerprint = build_prefix_cache_fingerprint(cfg, page_size)
    manager = object.__new__(QuestCacheManager)
    manager.parallel_context = SimpleNamespace(attn_tp_size=1)
    manager.config = cfg
    manager.runtime_layout = SimpleNamespace(kv_layer_index=lambda layer_idx: int(layer_idx))
    manager.device = torch.device("cpu")
    manager.enable_prefix_caching = True
    manager.page_size = page_size
    manager.num_pages = 10
    manager.prefix_cache_block_size = page_size
    manager.prefix_cache = RadixPrefixIndex(block_size=page_size, fingerprint=fingerprint)
    manager.layer_batch_state = SimpleNamespace()
    manager.page_offsets_i32 = torch.arange(page_size, dtype=torch.int32)
    manager.buffer_req_to_token_slots = torch.zeros((2, 16), dtype=torch.int32)
    manager.buffer_req_to_page_slots = torch.full((2, 8), -1, dtype=torch.int32)
    manager.free_pages_stack = torch.arange(10, dtype=torch.int32)
    manager._num_free_pages = 10
    manager.seq_id_to_row = {}
    manager.free_rows = deque([0, 1])
    manager.row_seq_lens = np.zeros((2,), dtype=np.int32)
    manager.row_logical_lens = np.zeros((2,), dtype=np.int32)
    manager.seq_id_to_prefix_blocks = {}
    manager.seq_id_to_cached_pages = {}
    manager._scheduler_capacity_snapshot_depth = 0
    manager._scheduler_freeable_block_ids = None
    manager._scheduler_reclaimable_pages = None
    manager.prefix_offload_controller = None
    manager._prefix_offload_step_h2d_operations = {}
    manager._prefix_prune_scoring = None
    manager._prefill_slot_reservations = None
    manager._prefill_metadata_full_pages = False
    manager._init_prefix_cache_runtime()
    return manager


def _make_quest_manager_for_no_prefix_graph(page_size=4):
    manager = _make_quest_manager_for_prefix(page_size=page_size)
    manager.enable_prefix_caching = False
    manager.prefix_cache = None
    manager.max_model_len = int(manager.buffer_req_to_token_slots.shape[1])
    manager.max_pages_per_row = int(manager.buffer_req_to_page_slots.shape[1])
    manager.buffer_req_to_page_slots_cpu = np.full(
        tuple(manager.buffer_req_to_page_slots.shape),
        -1,
        dtype=np.int32,
    )
    manager.free_pages_cpu_stack = np.arange(manager.num_pages, dtype=np.int32)
    manager._decode_row_page_slots = None
    manager._decode_row_page_slots_req_indices = None
    manager._decode_num_pages = None
    manager._decode_previous_page_counts = None
    manager._decode_page_geometry_context_lens = None
    manager._decode_page_geometry_ready = False
    return manager


def _remove_free_page(manager, page_slot: int):
    pages = [
        int(page)
        for page in manager.free_pages_stack[: manager._num_free_pages].tolist()
        if int(page) != int(page_slot)
    ]
    manager.free_pages_stack[: len(pages)] = torch.tensor(pages, dtype=torch.int32)
    manager._num_free_pages = len(pages)


def _remove_free_slots(manager, slots: list[int]):
    remove = {int(slot) for slot in slots}
    free_slots = [
        int(slot)
        for slot in manager.free_slots_stack[: manager._num_free_slots].tolist()
        if int(slot) not in remove
    ]
    manager.free_slots_stack[: len(free_slots)] = torch.tensor(free_slots, dtype=torch.int32)
    manager._num_free_slots = len(free_slots)


def test_usable_prefix_cache_tokens_leaves_logits_work():
    assert usable_prefix_cache_tokens(128, 16) == 112
    assert usable_prefix_cache_tokens(129, 16) == 128
    assert usable_prefix_cache_tokens(15, 16) == 0
    assert usable_prefix_cache_tokens(1, 16) == 0


def test_radix_prefix_index_block_id_is_stable_and_parent_sensitive():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)

    first = index.stable_block_id([1, 2, 3, 4], None)
    assert first == index.stable_block_id([1, 2, 3, 4], None)
    assert first != index.stable_block_id([1, 2, 3, 5], None)
    assert index.stable_block_id([5, 6, 7, 8], first) != index.stable_block_id([5, 6, 7, 8], None)


def test_prefix_cache_fingerprint_isolates_salt_and_method():
    vanilla = build_prefix_cache_fingerprint(_cfg(method="", salt="a"), 4)
    salted = build_prefix_cache_fingerprint(_cfg(method="", salt="b"), 4)
    omnikv = build_prefix_cache_fingerprint(_cfg(method="omnikv", salt="a"), 4)
    quest = build_prefix_cache_fingerprint(_cfg(method="quest", salt="a"), 4)

    assert vanilla != salted
    assert vanilla != omnikv
    assert omnikv != quest


def test_prefix_cache_fingerprint_isolates_prefill_semantics():
    dense = _cfg()
    flash = _cfg()
    flash.prefill_sparse_method = "flashprefill_v2"
    flash.flashprefill_v2_abs_threshold = 0.1
    retuned = _cfg()
    retuned.prefill_sparse_method = "flashprefill_v2"
    retuned.flashprefill_v2_abs_threshold = 0.2

    dense_fingerprint = build_prefix_cache_fingerprint(dense, 4)
    flash_fingerprint = build_prefix_cache_fingerprint(flash, 4)
    retuned_fingerprint = build_prefix_cache_fingerprint(retuned, 4)

    assert dense_fingerprint != flash_fingerprint
    assert flash_fingerprint != retuned_fingerprint


def test_prefix_cache_fingerprint_ignores_world_and_ep_rank():
    rank0 = _cfg()
    rank0.world_rank = 0
    rank0.ep_rank = 0
    rank1 = _cfg()
    rank1.world_rank = 1
    rank1.ep_rank = 1

    fingerprint0 = build_prefix_cache_fingerprint(rank0, 4)
    fingerprint1 = build_prefix_cache_fingerprint(rank1, 4)
    assert fingerprint0 == fingerprint1

    index0 = RadixPrefixIndex(block_size=4, fingerprint=fingerprint0)
    index1 = RadixPrefixIndex(block_size=4, fingerprint=fingerprint1)
    assert index0.stable_block_id([1, 2, 3, 4], None) == index1.stable_block_id(
        [1, 2, 3, 4],
        None,
    )


def test_prefix_cache_debug_summary_includes_refs_slots_and_stable_ids():
    manager = _make_standard_manager_for_prefix(block_size=2)
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    payload = StandardPrefixBlockPayload(
        token_slots=torch.tensor([10, 11], dtype=torch.int32)
    )
    manager.prefix_cache.insert_block(
        PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=None,
            block_size=2,
            logical_block_idx=0,
            payload=payload,
            token_ids=(1, 2),
            ref_count=2,
            eviction_priority=7,
            residency=PrefixBlockResidency(
                device_present=True,
                host_present=False,
                transfer=PrefixTransferKind.D2H,
            ),
        )
    )
    manager.prefix_cache.blocks[block_id].last_access = 11

    summary = manager.debug_state_summary()

    assert summary["prefix_cache"]["fingerprint"] == manager.prefix_cache.fingerprint.hex()
    block = summary["prefix_cache"]["blocks"][0]
    assert block["stable_block_id"] == block_id.hex()
    assert block["ref_count"] == 2
    assert block["last_access"] == 11
    assert block["eviction_priority"] == 7
    assert block["device_present"] is True
    assert block["host_present"] is False
    assert block["transfer"] == "d2h"
    assert block["payload"]["token_slots"]["shape"] == [2]


def test_standard_prompt_admission_accounts_for_free_rows():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.free_rows = deque([0])
    seq = Sequence([1, 2, 3])

    budgets = manager.prompt_admission_budgets(deque(), engine_prefill_chunk_size=4)
    costs = manager.prompt_admission_costs(seq)

    assert budgets["rows"] == 1
    assert costs["rows"] == 1
    assert budgets["slots"] == manager.prompt_admission_free_slots()
    assert costs["slots"] == manager.prompt_admission_cost(seq)


def test_lookup_returns_longest_full_block_prefix():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(8)))

    hit_len, hit_last_block_id, hit_blocks = index.lookup_longest_prefix(
        list(range(12)),
        max_usable_tokens=usable_prefix_cache_tokens(12, 4),
    )

    assert hit_len == 8
    assert hit_last_block_id == last_block_id
    assert hit_blocks == 2
    chain = index.get_chain(hit_last_block_id, hit_blocks)
    assert [block.logical_block_idx for block in chain] == [0, 1]


def test_routing_snapshot_is_immutable_and_refreshes_after_insert():
    fp = build_prefix_cache_fingerprint(_cfg(method="omnikv"), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    _insert_tokens(index, list(range(8)))

    first_snapshot = index.routing_snapshot("omnikv")
    assert first_snapshot is index.routing_snapshot("omnikv")
    assert first_snapshot.match(list(range(13)))["matched_tokens"] == 8

    _insert_tokens(index, list(range(12)))
    second_snapshot = index.routing_snapshot("omnikv")

    assert second_snapshot is not first_snapshot
    assert first_snapshot.match(list(range(13)))["matched_tokens"] == 8
    assert second_snapshot.match(list(range(13)))["matched_tokens"] == 12
    assert second_snapshot.match(list(range(13)))["snapshot"] is True


def test_routing_snapshot_removal_does_not_leave_false_hit():
    fp = build_prefix_cache_fingerprint(_cfg(method="omnikv"), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(8)))

    first_snapshot = index.routing_snapshot("omnikv")
    removed = index._remove_block_from_index(last_block_id)
    second_snapshot = index.routing_snapshot("omnikv")

    assert removed.stable_block_id == last_block_id
    assert first_snapshot.match(list(range(9)))["matched_tokens"] == 8
    assert second_snapshot.match(list(range(9)))["matched_tokens"] == 4
    assert second_snapshot.match(list(range(9)))["live_blocks"] == 1


def test_routing_snapshot_membership_structurally_shares_avl_nodes():
    fp = build_prefix_cache_fingerprint(_cfg(method="omnikv"), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)

    def insert_root_block(value: int) -> bytes:
        token_ids = tuple(range(value * 4, value * 4 + 4))
        stable_block_id = index.stable_block_id(token_ids, None)
        index.insert_block(
            PrefixCacheBlock(
                stable_block_id=stable_block_id,
                parent_block_id=None,
                block_size=4,
                logical_block_idx=0,
                payload=SimpleNamespace(name="dummy"),
                token_ids=token_ids,
            )
        )
        return stable_block_id

    def validate_tree(root) -> tuple[int, set[int]]:
        if root is None:
            return 0, set()
        left_height, left_ids = validate_tree(root.left)
        right_height, right_ids = validate_tree(root.right)
        assert abs(left_height - right_height) <= 1
        assert root.height == 1 + max(left_height, right_height)
        return root.height, left_ids | right_ids | {id(root)}

    for value in range(256):
        insert_root_block(value)
    first_snapshot = index.routing_snapshot("omnikv")
    assert first_snapshot.routing_membership is not None
    first_height, first_nodes = validate_tree(
        first_snapshot.routing_membership.root
    )

    inserted_block_id = insert_root_block(256)
    second_snapshot = index.routing_snapshot("omnikv")
    assert second_snapshot.routing_membership is not None
    second_height, second_nodes = validate_tree(
        second_snapshot.routing_membership.root
    )

    assert len(first_nodes) == 256
    assert len(second_nodes) == 257
    assert len(first_nodes & second_nodes) >= 240
    assert first_snapshot.routing_membership.contains(inserted_block_id) is False
    assert second_snapshot.routing_membership.contains(inserted_block_id) is True
    assert first_height == first_snapshot.routing_membership.height
    assert second_height == second_snapshot.routing_membership.height
    assert second_height <= 2 * math.ceil(
        math.log2(second_snapshot.routing_membership.live_blocks + 1)
    )

    index._remove_block_from_index(inserted_block_id)
    third_snapshot = index.routing_snapshot("omnikv")
    assert third_snapshot.routing_membership is not None
    third_height, third_nodes = validate_tree(
        third_snapshot.routing_membership.root
    )

    assert len(third_nodes) == 256
    assert len(second_nodes & third_nodes) >= 235
    assert second_snapshot.routing_membership.contains(inserted_block_id) is True
    assert third_snapshot.routing_membership.contains(inserted_block_id) is False
    assert third_height <= 2 * math.ceil(
        math.log2(third_snapshot.routing_membership.live_blocks + 1)
    )


def test_lookup_never_returns_half_block_match():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    _insert_tokens(index, [1, 2, 3, 4])

    hit_len, hit_last_block_id, hit_blocks = index.lookup_longest_prefix(
        [1, 2, 3, 4, 5, 6],
        max_usable_tokens=6,
    )

    assert hit_len == 4
    assert hit_last_block_id is not None
    assert hit_blocks == 1


def test_radix_backend_splits_edges_only_between_block_ids():
    backend = RadixTreeBackend()
    backend.insert((b"a", b"b", b"c"))
    backend.insert((b"a", b"b", b"d"))

    assert backend.lookup((b"a", b"b", b"c"), max_blocks=3).hit_block_count == 3
    assert backend.lookup((b"a", b"b", b"x"), max_blocks=3).hit_block_count == 2
    assert backend.child_count(b"a") == 1
    assert backend.child_count(b"b") == 2
    assert set(backend.leaf_block_ids()) == {b"c", b"d"}


def test_radix_backend_removes_leaf_from_compressed_segment_and_preserves_siblings():
    backend = RadixTreeBackend()
    backend.insert((b"a", b"b", b"c", b"d"))
    backend.insert((b"a", b"b", b"x", b"y"))
    backend.insert((b"a", b"q"))

    assert backend.path_to_block(b"d") == (b"a", b"b", b"c", b"d")
    assert backend.path_to_block(b"y") == (b"a", b"b", b"x", b"y")
    assert backend.child_count(b"b") == 2
    assert backend.child_count(b"c") == 1
    assert set(backend.subtree_block_ids(b"b")) == {b"b", b"c", b"d", b"x", b"y"}

    backend.remove_block(b"d")

    assert backend.path_to_block(b"c") == (b"a", b"b", b"c")
    assert backend.path_to_block(b"y") == (b"a", b"b", b"x", b"y")
    assert set(backend.leaf_block_ids()) == {b"c", b"y", b"q"}


def test_radix_backend_maintains_locations_incrementally(monkeypatch):
    backend = RadixTreeBackend()

    def fail_rebuild():
        raise AssertionError("Radix insert/remove should maintain locations incrementally.")

    monkeypatch.setattr(backend, "_rebuild_locations", fail_rebuild)

    backend.insert((b"a", b"b"))
    assert backend.path_to_block(b"b") == (b"a", b"b")

    backend.insert((b"a",))
    assert backend.path_to_block(b"a") == (b"a",)
    assert backend.path_to_block(b"b") == (b"a", b"b")

    backend.insert((b"a", b"c"))
    assert backend.child_count(b"a") == 2
    assert set(backend.leaf_block_ids()) == {b"b", b"c"}

    backend.remove_block(b"c")
    assert backend.child_count(b"a") == 1
    assert backend.path_to_block(b"b") == (b"a", b"b")
    with pytest.raises(KeyError):
        backend.path_to_block(b"c")


def test_radix_backend_insert_child_splits_compressed_parent_segment():
    backend = RadixTreeBackend()
    backend.insert((b"a", b"b", b"c"))

    backend.insert_child(b"b", b"x")

    assert backend.path_to_block(b"c") == (b"a", b"b", b"c")
    assert backend.path_to_block(b"x") == (b"a", b"b", b"x")
    assert backend.child_count(b"b") == 2
    assert set(backend.leaf_block_ids()) == {b"c", b"x"}


def test_prefix_index_insert_block_appends_without_recovering_parent_path(monkeypatch):
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)

    def fail_path_to_block(_block_id):
        raise AssertionError("insert_block should append through parent locations directly.")

    def fail_insert(_block_ids):
        raise AssertionError("insert_block should not rebuild and reinsert a full path.")

    with monkeypatch.context() as scoped:
        scoped.setattr(index.backend, "path_to_block", fail_path_to_block)
        scoped.setattr(index.backend, "insert", fail_insert)
        last_block_id = _insert_tokens(index, list(range(12)))

    hit_len, hit_last_block_id, hit_blocks = index.lookup_longest_prefix(
        list(range(13)),
        max_usable_tokens=usable_prefix_cache_tokens(13, 4),
    )

    assert hit_len == 12
    assert hit_last_block_id == last_block_id
    assert hit_blocks == 3


def test_radix_backend_stats_handles_deep_prefix_chain_iteratively():
    backend = RadixTreeBackend()
    parent = None
    for i in range(2000):
        block_id = f"block-{i}".encode()
        backend.insert_child(parent, block_id)
        parent = block_id

    assert backend.stats() == {
        "prefix_cache_tree_nodes": 2001,
        "prefix_cache_tree_edges": 2000,
    }


def test_lookup_does_not_touch_lru_state():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, [1, 2, 3, 4])
    block = index.get_chain(last_block_id, 1)[0]
    last_access = block.last_access

    index.lookup_longest_prefix([1, 2, 3, 4, 5], max_usable_tokens=4)

    assert block.last_access == last_access


def test_leaf_only_eviction_preserves_parent_until_child_is_removed():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    _insert_tokens(index, list(range(8)))

    evicted = index.evict_until_freeable(1)
    assert [block.logical_block_idx for block in evicted] == [1]
    assert index.evictable_blocks() == 1

    evicted = index.evict_until_freeable(1)
    assert [block.logical_block_idx for block in evicted] == [0]
    assert len(index) == 0


def test_prefix_residency_rejects_missing_payload_and_invalid_transfers():
    with pytest.raises(RuntimeError, match="no resident payload"):
        PrefixBlockResidency(device_present=False, host_present=False).validate()
    with pytest.raises(RuntimeError, match="D2H"):
        PrefixBlockResidency(
            device_present=True,
            host_present=True,
            transfer=PrefixTransferKind.D2H,
        ).validate()
    with pytest.raises(RuntimeError, match="H2D"):
        PrefixBlockResidency(
            device_present=False,
            host_present=True,
            transfer=PrefixTransferKind.H2D,
        ).validate()


def test_prefix_insert_rejects_device_child_below_host_only_parent():
    index = RadixPrefixIndex(block_size=2, fingerprint=b"device-root-contiguous")
    parent_id = _insert_tokens(index, [1, 2])
    parent = index.get_block(parent_id)
    assert parent is not None
    index.begin_d2h(parent)
    index.finish_d2h(parent)
    assert index.demote_device_until_freeable(1) == [parent]

    child_tokens = [3, 4]
    child = PrefixCacheBlock(
        stable_block_id=index.stable_block_id(child_tokens, parent_id),
        parent_block_id=parent_id,
        block_size=2,
        logical_block_idx=1,
        payload=SimpleNamespace(name="device-child"),
        token_ids=tuple(child_tokens),
    )
    with pytest.raises(RuntimeError, match="host-only parent"):
        index.insert_block(child)


def test_write_through_residency_demotes_device_without_deleting_radix_blocks():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(12)))
    chain = index.get_chain(last_block_id, 3)

    for block in chain:
        index.begin_d2h(block)
    for block in chain:
        index.finish_d2h(block)

    assert index.device_evictable_blocks() == 1
    assert index.device_freeable_blocks() == 3
    demoted = index.demote_device_until_freeable(2)

    assert [block.logical_block_idx for block in demoted] == [2, 1]
    assert len(index) == 3
    assert [block.residency.device_present for block in chain] == [True, False, False]
    assert all(block.residency.host_present for block in chain)
    hit_len, hit_last_block_id, hit_blocks = index.lookup_longest_prefix(
        list(range(13)),
        max_usable_tokens=12,
    )
    assert (hit_len, hit_last_block_id, hit_blocks) == (12, last_block_id, 3)


def test_device_reclaimable_blocks_include_only_demotable_or_inflight_d2h_chains():
    index = RadixPrefixIndex(block_size=2, fingerprint=b"reclaimable")
    last_block_id = _insert_tokens(index, list(range(6)))
    chain = index.get_chain(last_block_id, 3)
    for block in chain:
        index.begin_d2h(block)

    assert index.device_freeable_blocks() == 0
    assert index.device_reclaimable_block_ids() == {
        block.stable_block_id for block in chain
    }

    index.acquire_block_ref(chain[-1])
    assert index.device_reclaimable_blocks() == 0
    index.release_block_ref(chain[-1])
    index.set_subtree_eviction_priority(list(range(6)), -1)
    assert index.device_reclaimable_blocks() == 0

    h2d_index = RadixPrefixIndex(block_size=2, fingerprint=b"h2d-not-reclaimable")
    h2d_id = _insert_tokens(h2d_index, [1, 2])
    h2d_block = h2d_index.get_block(h2d_id)
    assert h2d_block is not None
    h2d_index.begin_d2h(h2d_block)
    h2d_index.finish_d2h(h2d_block)
    h2d_index.demote_device_until_freeable(1)
    h2d_index.begin_h2d(h2d_block)
    assert h2d_index.device_reclaimable_blocks() == 0


def test_host_eviction_only_deletes_cpu_only_logical_leaves():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(12)))
    chain = index.get_chain(last_block_id, 3)
    for block in chain:
        index.begin_d2h(block)
    for block in chain:
        index.finish_d2h(block)
    index.demote_device_until_freeable(2)

    assert index.evict_host_until_freeable(2) == [chain[2], chain[1]]
    assert len(index) == 1
    assert index.get_block(chain[0].stable_block_id) is chain[0]
    assert index.stats()["prefix_cache_device_demoted_blocks"] == 2
    assert index.stats()["prefix_cache_host_evicted_blocks"] == 2


def test_transfering_prefix_block_is_not_deleted_or_demoted():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    block_id = _insert_tokens(index, [1, 2, 3, 4])
    block = index.get_block(block_id)
    assert block is not None

    index.begin_d2h(block)

    assert index.evict_until_freeable(1) == []
    assert index.demote_device_until_freeable(1) == []
    result = index.safe_delete_subtree([1, 2, 3, 4])
    assert result.deleted_blocks == []
    assert [item.reason for item in result.blocked_blocks] == ["transfer_inflight"]


def test_h2d_promotion_requires_root_to_leaf_order():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(8)))
    parent, child = index.get_chain(last_block_id, 2)
    for block in (parent, child):
        index.begin_d2h(block)
    for block in (parent, child):
        index.finish_d2h(block)
    index.demote_device_until_freeable(2)

    with pytest.raises(RuntimeError, match="radix root"):
        index.begin_h2d(child)
    index.begin_h2d(parent)
    index.begin_h2d(child)
    index.finish_h2d(parent)
    index.finish_h2d(child)

    assert parent.residency.device_present
    assert child.residency.device_present
    assert index.device_freeable_blocks() == 2


def test_freeable_blocks_counts_cascade_evictable_chain_without_mutation():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    _insert_tokens(index, list(range(16)))

    assert index.evictable_blocks() == 1
    assert index.freeable_blocks() == 4
    assert len(index) == 4

    evicted = index.evict_until_freeable(4)
    assert [block.logical_block_idx for block in evicted] == [3, 2, 1, 0]
    assert len(index) == 0


def test_freeable_blocks_excludes_ancestors_of_referenced_descendant():
    fp = build_prefix_cache_fingerprint(_cfg(), 2)
    index = RadixPrefixIndex(block_size=2, fingerprint=fp)
    root_id = _insert_tokens(index, [1, 2])
    referenced_child_id = _insert_tokens(index, [1, 2, 3, 4])
    free_child_id = _insert_tokens(index, [1, 2, 5, 6])
    referenced_child = index.get_block(referenced_child_id)
    assert referenced_child is not None
    referenced_child.ref_count = 1

    assert index.freeable_block_ids() == {free_child_id}
    assert index.freeable_blocks() == 1
    assert root_id in index.blocks


def test_freeable_block_ids_updates_snapshot_without_rescanning_index():
    fp = build_prefix_cache_fingerprint(_cfg(), 2)
    index = RadixPrefixIndex(block_size=2, fingerprint=fp)
    root_id = _insert_tokens(index, [1, 2])
    child_id = _insert_tokens(index, [1, 2, 3, 4])
    child = index.get_block(child_id)
    assert child is not None

    expected = frozenset({root_id, child_id})
    assert index.freeable_block_ids() == expected
    assert index.freeable_block_ids() is index.freeable_block_ids()
    assert index.freeable_scans == 1
    assert index.freeable_cache_hits == 2

    index.acquire_block_ref(child)
    assert index.freeable_block_ids() == frozenset()
    assert index.freeable_scans == 1

    index.release_block_ref(child)
    assert index.freeable_block_ids() == expected
    assert index.freeable_scans == 1


def test_freeable_block_ids_invalidates_for_priority_transfer_and_removal():
    fp = build_prefix_cache_fingerprint(_cfg(), 2)
    index = RadixPrefixIndex(block_size=2, fingerprint=fp)
    block_id = _insert_tokens(index, [1, 2])
    block = index.get_block(block_id)
    assert block is not None

    assert index.freeable_block_ids() == frozenset({block_id})
    index.set_subtree_eviction_priority([1, 2], -1)
    assert index.freeable_block_ids() == frozenset()

    index.set_subtree_eviction_priority([1, 2], 0)
    assert index.freeable_block_ids() == frozenset({block_id})

    index.begin_d2h(block)
    assert index.freeable_block_ids() == frozenset()
    index.abort_d2h(block)
    assert index.freeable_block_ids() == frozenset({block_id})

    assert index.evict_until_freeable(1) == [block]
    assert index.freeable_block_ids() == frozenset()
    assert index.freeable_scans == 1


def test_incremental_freeable_membership_matches_virtual_leaf_eviction():
    """Catch stale ancestor capacity after interleaved branch mutations.

    Existing one-chain tests cannot expose sibling blocker-count drift after
    insertion, subtree priority changes, transfers and removal.
    """
    rng = random.Random(31)
    index = RadixPrefixIndex(block_size=2, fingerprint=b"incremental-capacity")
    prompts = {}
    serial = 0

    def insert():
        nonlocal serial
        serial += 1
        candidates = [None, *index.blocks]
        parent = rng.choice(candidates)
        if parent is not None and index.blocks[parent].residency.host_present:
            parent = None
        tokens = ([] if parent is None else prompts[parent]) + [serial, -serial]
        block_id = _insert_tokens(index, tokens)
        prompts[block_id] = tokens

    def oracle():
        # Simulate actual leaf removals without any cached capacity metadata.
        remaining = dict(index.blocks)
        removed = set()
        while True:
            parents = {b.parent_block_id for b in remaining.values()}
            leaves = {
                key for key, b in remaining.items()
                if key not in parents and b.ref_count == 0
                and b.eviction_priority >= 0 and b.residency.transfer is None
            }
            if not leaves:
                return removed
            for key in leaves:
                remaining.pop(key)
            removed.update(leaves)

    for _ in range(40):
        insert()
    for _ in range(300):
        before = index.freeable_block_ids()
        assert before == oracle()
        block = rng.choice(list(index.blocks.values()))
        operation = rng.randrange(6)
        if operation == 0:
            index.set_block_ref_count(block, rng.randrange(3))
        elif operation == 1:
            index.set_subtree_eviction_priority(prompts[block.stable_block_id], rng.choice([-1, 0, 3]))
        elif operation == 2:
            insert()
        elif operation == 3:
            index.evict_until_freeable(rng.randrange(1, 4))
        elif block.residency.transfer == PrefixTransferKind.D2H:
            if operation == 4:
                index.abort_d2h(block)
            else:
                index.finish_d2h(block)
        elif block.ref_count == 0 and not block.residency.host_present:
            parent = index.get_block(block.parent_block_id)
            if parent is None or parent.residency.host_present:
                index.begin_d2h(block)
        assert index.freeable_block_ids() == oracle()
        assert len(index._blocked_freeable_children) == len(index.blocks)


def test_incremental_freeable_membership_tracks_h2d_and_rollback():
    """Warm capacity state must survive demote/promote/abort and leaf rollback."""
    index = RadixPrefixIndex(block_size=2, fingerprint=b"incremental-transfers")
    root = index.get_block(_insert_tokens(index, [1, 2]))
    assert index.freeable_block_ids() == {root.stable_block_id}
    index.begin_d2h(root)
    assert not index.freeable_block_ids()
    index.finish_d2h(root)
    assert index.freeable_block_ids() == {root.stable_block_id}
    index.demote_device_until_freeable(1)
    index.begin_h2d(root)
    assert not index.freeable_block_ids()
    index.abort_h2d(root)
    assert index.freeable_block_ids() == {root.stable_block_id}
    index.begin_h2d(root)
    index.finish_h2d(root)
    assert index.freeable_block_ids() == {root.stable_block_id}
    child = index.get_block(_insert_tokens(index, [1, 2, 3, 4]))
    index.acquire_block_ref(child)
    assert not index.freeable_block_ids()
    index.release_block_ref(child)
    snapshot = index.freeable_block_ids()
    index.rollback_inserted_leaf(child)
    assert index.freeable_block_ids() == {root.stable_block_id}
    assert snapshot == {root.stable_block_id, child.stable_block_id}


def test_incremental_freeable_deep_chain_reference_release():
    """A long prompt must update ancestors without recursion or stale capacity."""
    index = RadixPrefixIndex(block_size=2, fingerprint=b"deep-capacity")
    leaf = _insert_tokens(index, list(range(2600)))
    chain = index.get_chain(leaf, 1300)
    all_ids = frozenset(index.blocks)
    assert index.freeable_block_ids() == all_ids
    index.acquire_block_ref(chain[-1])
    assert not index.freeable_block_ids()
    index.release_block_ref(chain[-1])
    assert index.freeable_block_ids() == all_ids
    for block in chain:
        index.acquire_block_ref(block)
    assert not index.freeable_block_ids()
    # Normal request cleanup releases a whole prefix in root-to-leaf order.
    for block in chain[:-1]:
        index.release_block_ref(block)
    assert not index.freeable_block_ids()
    index.release_block_ref(chain[-1])
    assert index.freeable_block_ids() == all_ids
    assert index.freeable_scans == 1


@pytest.mark.parametrize("serialized", [False, True])
def test_waiting_prefix_lookup_reuses_hash_chain_and_cached_result(serialized):
    manager = _make_standard_manager_for_prefix(block_size=2)
    assert manager.prefix_cache is not None
    _insert_tokens(manager.prefix_cache, [1, 2])
    seq = Sequence([1, 2, 3, 4, 5])

    def refresh(seq):
        # TP workers receive a new object on every control RPC.
        if serialized:
            seq = pickle.loads(pickle.dumps(seq))
        manager.refresh_prefix_cache_hit(seq)
        return seq

    seq = refresh(seq)
    assert seq.prefix_cache_hit_len == 2
    assert manager.prefix_cache.block_id_generation_requests == 1
    assert manager.prefix_cache.lookup_requests == 1

    seq = refresh(seq)
    assert seq.prefix_cache_hit_len == 2
    assert manager.prefix_cache.block_id_generation_requests == 1
    assert manager.prefix_cache.lookup_requests == 1

    _insert_tokens(manager.prefix_cache, [11, 12])
    seq = refresh(seq)
    assert seq.prefix_cache_hit_len == 2
    assert manager.prefix_cache.block_id_generation_requests == 1
    assert manager.prefix_cache.lookup_requests == 1

    root = manager.prefix_cache.get_chain(seq.prefix_cache_hit_last_block_id, 1)[0]
    manager.prefix_cache.acquire_block_ref(root)
    seq = refresh(seq)
    manager.prefix_cache.release_block_ref(root)
    assert manager.prefix_cache.block_id_generation_requests == 1
    assert manager.prefix_cache.lookup_requests == 1

    _insert_tokens(manager.prefix_cache, [1, 2, 3, 4])
    seq = refresh(seq)
    assert seq.prefix_cache_hit_len == 4
    assert manager.prefix_cache.block_id_generation_requests == 1
    assert manager.prefix_cache.lookup_requests == 2

    _insert_tokens(manager.prefix_cache, [7, 8])
    seq = refresh(seq)
    assert seq.prefix_cache_hit_len == 4
    assert manager.prefix_cache.block_id_generation_requests == 1
    assert manager.prefix_cache.lookup_requests == 2

    changed_seq = Sequence([1, 2, 9, 4, 5])
    changed_seq = refresh(changed_seq)
    assert changed_seq.prefix_cache_hit_len == 2
    assert manager.prefix_cache.block_id_generation_requests == 2
    assert manager.prefix_cache.lookup_requests == 3


def test_serialized_prefix_lookup_invalidates_on_eviction_and_prompt_replacement():
    """Stable IDs must not reuse a removed hit or an earlier chain turn's prompt."""
    manager = _make_standard_manager_for_prefix(block_size=2)
    _insert_tokens(manager.prefix_cache, [1, 2, 3, 4])
    seq = Sequence([1, 2, 3, 4, 5])
    manager.refresh_prefix_cache_hit(pickle.loads(pickle.dumps(seq)))
    manager.prefix_cache.evict_until_freeable(1)
    restored = pickle.loads(pickle.dumps(seq))
    manager.refresh_prefix_cache_hit(restored)
    assert restored.prefix_cache_hit_len == 2
    assert manager.prefix_cache.block_id_generation_requests == 1
    assert manager.prefix_cache.lookup_requests == 2

    changed = Sequence([7, 8, 3, 4, 5])
    changed.seq_id = seq.seq_id
    manager.refresh_prefix_cache_hit(changed)
    assert changed.prefix_cache_hit_len == 0
    assert manager.prefix_cache.block_id_generation_requests == 2


@pytest.mark.parametrize("hit_tokens", [[], [1, 2], [1, 2, 3, 4]])
def test_serialized_lookup_survives_unrelated_deletion_and_detects_path_changes(hit_tokens):
    """Eviction in another branch must not invalidate every waiting TP request."""
    manager = _make_standard_manager_for_prefix(block_size=2)
    index = manager.prefix_cache
    if hit_tokens:
        _insert_tokens(index, hit_tokens)
    seq = Sequence([1, 2, 3, 4, 5])

    def refresh():
        restored = pickle.loads(pickle.dumps(seq))
        manager.refresh_prefix_cache_hit(restored)
        return restored.prefix_cache_hit_len

    assert refresh() == len(hit_tokens)
    lookups = index.lookup_requests
    for i in range(8):
        tokens = [100 + i, 200 + i]
        _insert_tokens(index, tokens)
        assert len(index.safe_delete_subtree(tokens).deleted_blocks) == 1
        assert refresh() == len(hit_tokens)
    assert index.lookup_requests == lookups

    # Same-ID replacement is safe only for the ID/count memo, not payload caches.
    if hit_tokens:
        index.safe_delete_subtree([1, 2])
        _insert_tokens(index, hit_tokens)
        assert refresh() == len(hit_tokens)
    _insert_tokens(index, [1, 2, 3, 4])
    assert refresh() == 4
    index.safe_delete_subtree([1, 2, 3, 4])
    assert refresh() == 2
    index.safe_delete_subtree([1, 2])
    assert refresh() == 0


def test_prefix_lookup_memo_is_bounded_and_evicted_requests_can_be_requeried():
    """Cancelled waiting requests have no KV free call; memo retention stays bounded."""
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.prefix_lookup_cache = PrefixLookupCache(max_entries=2)
    _insert_tokens(manager.prefix_cache, [1, 2])
    seqs = [Sequence([1, 2, i]) for i in range(3, 6)]
    for seq in seqs:
        manager.refresh_prefix_cache_hit(pickle.loads(pickle.dumps(seq)))
        assert len(manager.prefix_lookup_cache.entries) <= 2
    assert manager.prefix_lookup_cache.get(seqs[0]) is None
    restored = pickle.loads(pickle.dumps(seqs[0]))
    manager.refresh_prefix_cache_hit(restored)
    assert restored.prefix_cache_hit_len == 2
    assert manager.prefix_cache.block_id_generation_requests == 4
    manager.prefix_lookup_cache.discard(restored.seq_id)
    assert manager.prefix_lookup_cache.get(restored) is None


def test_prefix_hit_admission_cost_reuses_chain_until_capacity_mutates():
    manager = _make_standard_manager_for_prefix(block_size=2)
    assert manager.prefix_cache is not None
    last_block_id = _insert_tokens(manager.prefix_cache, [1, 2, 3, 4])
    seq = Sequence([1, 2, 3, 4, 5])
    seq.prefix_cache_hit_len = 4
    seq.prefix_cache_hit_block_count = 2
    seq.prefix_cache_hit_last_block_id = last_block_id

    get_chain_calls = 0
    original_get_chain = manager.prefix_cache.get_chain

    def counted_get_chain(block_id, block_count):
        nonlocal get_chain_calls
        get_chain_calls += 1
        return original_get_chain(block_id, block_count)

    manager.prefix_cache.get_chain = counted_get_chain
    assert manager.prompt_admission_cost(seq) == 5
    assert manager.prompt_admission_cost(seq) == 5
    assert get_chain_calls == 1
    assert manager.prefix_cache.freeable_scans == 1

    _insert_tokens(manager.prefix_cache, [7, 8])
    assert manager.prompt_admission_cost(seq) == 5
    assert get_chain_calls == 1
    assert manager.prefix_cache.freeable_scans == 1

    referenced = original_get_chain(last_block_id, 2)[-1]
    manager.prefix_cache.acquire_block_ref(referenced)
    assert manager.prompt_admission_cost(seq) == 1
    assert get_chain_calls == 1
    assert manager.prefix_cache.freeable_scans == 1


def test_evictable_and_device_reclaimable_counts_reuse_epoch_cache():
    index = RadixPrefixIndex(block_size=2, fingerprint=b"capacity-count-cache")
    block_id = _insert_tokens(index, [1, 2])
    block = index.get_block(block_id)
    assert block is not None

    assert index.evictable_blocks() == 1
    assert index.evictable_blocks() == 1
    assert index.evictable_scans == 1
    assert index.evictable_cache_hits == 1

    index.begin_d2h(block)
    assert index.device_reclaimable_blocks() == 1
    assert index.device_reclaimable_blocks() == 1
    assert index.device_reclaimable_scans == 1
    assert index.device_reclaimable_cache_hits == 1

    index.acquire_block_ref(block)
    assert index.evictable_blocks() == 0
    assert index.device_reclaimable_blocks() == 0
    assert index.evictable_scans == 2
    assert index.device_reclaimable_scans == 2


def test_referenced_chain_growth_does_not_invalidate_capacity_cache():
    index = RadixPrefixIndex(block_size=2, fingerprint=b"referenced-growth")
    root_id = index.stable_block_id([1, 2], None)
    root = PrefixCacheBlock(
        stable_block_id=root_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=SimpleNamespace(name="root"),
        token_ids=(1, 2),
        ref_count=1,
    )
    index.insert_block(root)
    assert index.freeable_block_ids() == frozenset()
    capacity_epoch = index.capacity_epoch

    index.acquire_block_ref(root)
    index.release_block_ref(root)
    child_id = index.stable_block_id([3, 4], root_id)
    child = PrefixCacheBlock(
        stable_block_id=child_id,
        parent_block_id=root_id,
        block_size=2,
        logical_block_idx=1,
        payload=SimpleNamespace(name="child"),
        token_ids=(3, 4),
        ref_count=1,
    )
    index.insert_block(child)

    assert index.capacity_epoch == capacity_epoch
    assert index.freeable_block_ids() == frozenset()
    assert index.freeable_scans == 1
    assert index.freeable_cache_hits == 1


def test_unrelated_insert_churn_does_not_refresh_waiting_prefix_lookups():
    manager = _make_standard_manager_for_prefix(block_size=2)
    assert manager.prefix_cache is not None
    _insert_tokens(manager.prefix_cache, [1, 2])
    waiting = [Sequence([1, 2, 100 + idx, 200 + idx, 9]) for idx in range(24)]

    for seq in waiting:
        manager.refresh_prefix_cache_hit(seq)
        assert seq.prefix_cache_hit_len == 2
    assert manager.prefix_cache.lookup_requests == 24
    assert manager.prefix_cache.block_id_generation_requests == 24

    for step in range(12):
        _insert_tokens(manager.prefix_cache, [1000 + step, 2000 + step])
        for seq in waiting:
            manager.refresh_prefix_cache_hit(seq)

    assert manager.prefix_cache.lookup_requests == 24
    assert manager.prefix_cache.block_id_generation_requests == 24


def test_referenced_insert_churn_keeps_waiting_hit_capacity_cached():
    manager = _make_standard_manager_for_prefix(block_size=2)
    assert manager.prefix_cache is not None
    last_block_id = _insert_tokens(manager.prefix_cache, [1, 2, 3, 4])
    chain = manager.prefix_cache.get_chain(last_block_id, 2)
    for block in chain:
        manager.prefix_cache.acquire_block_ref(block)
    waiting = [Sequence([1, 2, 3, 4, 100 + idx]) for idx in range(24)]
    for seq in waiting:
        seq.prefix_cache_hit_len = 4
        seq.prefix_cache_hit_block_count = 2
        seq.prefix_cache_hit_last_block_id = last_block_id

    get_chain_calls = 0
    original_get_chain = manager.prefix_cache.get_chain

    def counted_get_chain(block_id, block_count):
        nonlocal get_chain_calls
        get_chain_calls += 1
        return original_get_chain(block_id, block_count)

    manager.prefix_cache.get_chain = counted_get_chain
    assert all(manager.prompt_admission_cost(seq) == 1 for seq in waiting)
    assert get_chain_calls == 24
    assert manager.prefix_cache.freeable_scans == 1
    capacity_epoch = manager.prefix_cache.capacity_epoch

    for step in range(12):
        block_tokens = [3000 + step, 4000 + step]
        block_id = manager.prefix_cache.stable_block_id(block_tokens, None)
        manager.prefix_cache.insert_block(
            PrefixCacheBlock(
                stable_block_id=block_id,
                parent_block_id=None,
                block_size=2,
                logical_block_idx=0,
                payload=SimpleNamespace(name="referenced"),
                token_ids=tuple(block_tokens),
                ref_count=1,
            )
        )
        assert all(manager.prompt_admission_cost(seq) == 1 for seq in waiting)

    assert manager.prefix_cache.capacity_epoch == capacity_epoch
    assert get_chain_calls == 24
    assert manager.prefix_cache.freeable_scans == 1


def test_bulk_eviction_scans_initial_leaves_once(monkeypatch):
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    for i in range(32):
        block_id = index.stable_block_id([i, i, i, i], None)
        index.insert_block(
            PrefixCacheBlock(
                stable_block_id=block_id,
                parent_block_id=None,
                block_size=4,
                logical_block_idx=i,
                payload=SimpleNamespace(name="dummy"),
                token_ids=(i, i, i, i),
            )
        )

    leaf_calls = 0
    original_leaf_block_ids = index.backend.leaf_block_ids

    def counted_leaf_block_ids():
        nonlocal leaf_calls
        leaf_calls += 1
        return original_leaf_block_ids()

    monkeypatch.setattr(index.backend, "leaf_block_ids", counted_leaf_block_ids)

    evicted = index.evict_until_freeable(16)

    assert len(evicted) == 16
    assert leaf_calls == 1


def test_bulk_eviction_queues_new_parent_leaf_with_priority_ordering():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    chain_last = _insert_tokens(index, list(range(8)))
    sibling_id = index.stable_block_id([8, 9, 10, 11], None)
    index.insert_block(
        PrefixCacheBlock(
            stable_block_id=sibling_id,
            parent_block_id=None,
            block_size=4,
            logical_block_idx=0,
            payload=SimpleNamespace(name="sibling"),
            token_ids=(8, 9, 10, 11),
        )
    )
    parent, child = index.get_chain(chain_last, 2)
    sibling = index.get_block(sibling_id)
    assert sibling is not None
    parent.eviction_priority = 10
    child.eviction_priority = 0
    sibling.eviction_priority = 0

    evicted = index.evict_until_freeable(2)

    assert evicted == [child, parent]
    assert sibling_id in index.blocks


def test_referenced_blocks_are_not_evictable():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, [1, 2, 3, 4])
    block = index.get_chain(last_block_id, 1)[0]
    index.acquire_block_ref(block)

    assert index.evict_until_freeable(1) == []
    index.release_block_ref(block)
    assert index.evict_until_freeable(1) == [block]


def test_duplicate_commit_returns_existing_block_and_counts_duplicate():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    stable_block_id = index.stable_block_id([1, 2, 3, 4], None)
    first = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=SimpleNamespace(name="first"),
        token_ids=(1, 2, 3, 4),
    )
    second = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=SimpleNamespace(name="second"),
        token_ids=(1, 2, 3, 4),
    )

    assert index.insert_block(first) is first
    assert index.insert_block(second) is first
    assert index.stats()["prefix_cache_duplicate_commits"] == 1


def test_eviction_priority_prefers_larger_positive_priority():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    first_id = _insert_tokens(index, [1, 2, 3, 4])
    second_id = _insert_tokens(index, [5, 6, 7, 8])
    first = index.get_block(first_id)
    second = index.get_block(second_id)
    assert first is not None and second is not None
    first.eviction_priority = 1
    second.eviction_priority = 10

    assert index.evict_until_freeable(1) == [second]
    assert first_id in index.blocks


def test_negative_priority_blocks_eviction_and_safe_delete():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    block_id = _insert_tokens(index, [1, 2, 3, 4])
    block = index.get_block(block_id)
    assert block is not None
    block.eviction_priority = -1

    assert index.evict_until_freeable(1) == []
    result = index.safe_delete_subtree([1, 2, 3, 4])
    assert result.deleted_blocks == []
    assert [blocked.reason for blocked in result.blocked_blocks] == ["negative_priority"]


def test_subtree_delete_reports_referenced_child_and_preserves_parent():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(8)))
    parent, child = index.get_chain(last_block_id, 2)
    child.ref_count = 1

    result = index.safe_delete_subtree(list(range(4)))

    assert result.deleted_blocks == []
    assert [blocked.reason for blocked in result.blocked_blocks] == ["referenced", "has_children"]
    assert parent.stable_block_id in index.blocks
    assert child.stable_block_id in index.blocks


def test_subtree_delete_deletes_safe_child_and_blocks_protected_branch():
    fp = build_prefix_cache_fingerprint(_cfg(), 2)
    index = RadixPrefixIndex(block_size=2, fingerprint=fp)
    root_id = _insert_tokens(index, [1, 2])
    referenced_child_id = _insert_tokens(index, [1, 2, 3, 4])
    free_child_id = _insert_tokens(index, [1, 2, 5, 6])
    referenced_child = index.get_block(referenced_child_id)
    assert referenced_child is not None
    referenced_child.ref_count = 1

    result = index.safe_delete_subtree([1, 2])

    assert [block.stable_block_id for block in result.deleted_blocks] == [free_child_id]
    assert {blocked.reason for blocked in result.blocked_blocks} == {"referenced", "has_children"}
    assert root_id in index.blocks
    assert referenced_child_id in index.blocks
    assert free_child_id not in index.blocks


def test_subtree_delete_preview_does_not_mutate_index():
    index = RadixPrefixIndex(block_size=2, fingerprint=b"delete-preview")
    last_id = _insert_tokens(index, [1, 2, 3, 4])
    plan = index.preview_delete_subtree([1, 2])

    assert [block.logical_block_idx for block in plan.deleted_blocks] == [1, 0]
    assert len(index) == 2
    assert last_id in index.blocks

    result = index.safe_delete_subtree([1, 2])
    assert result.to_dict() == plan.to_dict()
    assert len(index) == 0


def test_prefix_delete_plan_rejects_tp_divergence_before_mutation():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.world_size = 2
    manager.parallel_context = SimpleNamespace(
        world=SimpleNamespace(process_group="world"),
        attn_tp=SimpleNamespace(process_group="replica"),
        attn_dp_size=2, attn_tp_size=2,
    )
    block_id = _insert_tokens(manager.prefix_cache, [1, 2])
    local_plan = manager.prefix_cache.preview_delete_subtree([1, 2]).to_dict()

    def gather(plans, plan, group=None):
        assert group == "replica"
        plans[:] = [
            plan,
            {
                "deleted_block_ids": [],
                "deleted_block_count": 0,
                "blocked_blocks": [{"block_id": block_id.hex(), "reason": "transfer_inflight"}],
            },
        ]

    with patch(
        "sparseengine.engine.cache_manager.base.dist.all_gather_object",
        side_effect=gather,
    ):
        with pytest.raises(RuntimeError, match="deletion plan diverged"):
            manager.prefix_cache_delete_subtree([1, 2])

    assert block_id in manager.prefix_cache.blocks


def test_max_blocks_requires_explicit_capacity_before_insert():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp, max_blocks=1)
    _insert_tokens(index, [1, 2, 3, 4])

    stable_block_id = index.stable_block_id([5, 6, 7, 8], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=SimpleNamespace(name="dummy"),
        token_ids=(5, 6, 7, 8),
    )
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        index.insert_block(block)

    evicted = index.ensure_insert_capacity(1)
    assert len(evicted) == 1
    assert index.insert_block(block) is block


def test_get_chain_fails_fast_on_incomplete_chain():
    fp = build_prefix_cache_fingerprint(_cfg(), 4)
    index = RadixPrefixIndex(block_size=4, fingerprint=fp)
    last_block_id = _insert_tokens(index, list(range(8)))
    parent = index.get_chain(last_block_id, 2)[0]
    del index.blocks[parent.stable_block_id]

    with pytest.raises(RuntimeError, match="incomplete"):
        index.get_chain(last_block_id, 2)


def test_resolve_prefix_cache_block_size_uses_quest_page_size():
    assert resolve_prefix_cache_block_size(_cfg(method="quest", block_size=None)) == 4
    with pytest.raises(ValueError, match="quest_chunk_size"):
        resolve_prefix_cache_block_size(_cfg(method="quest", block_size=8))
    with pytest.raises(ValueError, match="positive integer"):
        resolve_prefix_cache_block_size(_cfg(block_size=16.9))


def test_config_rejects_unvalidated_prefix_cache_options():
    with pytest.raises(ValueError, match="capture_sampling"):
        _make_config(
            enable_prefix_caching=True,
            decode_graph=True,
            decode_graph_capture_sampling=True,
        )
    with pytest.raises(ValueError, match="quest_chunk_size"):
        _make_config(
            sparse_method="quest",
            enable_prefix_caching=True,
            quest_chunk_size=8,
            prefix_cache_block_size=16,
        )
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        _make_config(enable_prefix_caching="maybe")
    with pytest.raises(ValueError, match="prefix_cache_block_size"):
        _make_config(prefix_cache_block_size=16.9)
    with pytest.raises(ValueError, match="prefix_cache_max_blocks"):
        _make_config(prefix_cache_max_blocks="16.9")


def test_config_restricts_prefix_cache_offload_to_explicit_tp1_tp2_modes():
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        _make_config(
            enable_prefix_cache_offload=True,
            prefix_cache_host_size_gb=1,
        )
    with pytest.raises(ValueError, match="explicit prefix_cache_host_size_gb"):
        _make_config(
            enable_prefix_caching=True,
            enable_prefix_cache_offload=True,
        )
    quest = _make_config(
        sparse_method="quest",
        enable_prefix_caching=True,
        enable_prefix_cache_offload=True,
        prefix_cache_host_size_gb=1,
        decode_graph=True,
    )
    assert quest.enable_prefix_cache_offload is True
    assert quest.decode_graph is True
    tp2 = _make_config(
        enable_prefix_caching=True,
        enable_prefix_cache_offload=True,
        prefix_cache_host_size_gb=1,
        tensor_parallel_size=2,
    )
    assert tp2.tensor_parallel_size == 2
    with pytest.raises(ValueError, match="tensor_parallel_size=1 or 2"):
        _make_config(
            enable_prefix_caching=True,
            enable_prefix_cache_offload=True,
            prefix_cache_host_size_gb=1,
            tensor_parallel_size=3,
        )
    graph = _make_config(
        enable_prefix_caching=True,
        enable_prefix_cache_offload=True,
        prefix_cache_host_size_gb=1,
        decode_graph=True,
    )
    assert graph.decode_graph is True

    cfg = _make_config(
        sparse_method="omnikv",
        full_attention_layers=[0],
        enable_prefix_caching=True,
        enable_prefix_cache_offload=True,
        prefix_cache_host_size_gb=1,
    )
    assert cfg.enable_prefix_cache_offload is True
    assert cfg.prefix_cache_host_size_gb == 1

def test_standard_attach_pins_prefix_slots_and_free_seq_keeps_cached_slots():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2, 3])
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    _remove_free_slots(manager, [10, 11])
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = stable_block_id
    seq.prefix_cache_block_size = 2
    seq.prefix_cache_method = ""

    manager.refresh_prefix_cache_hit(seq)
    assert manager.prefix_lookup_cache.get(seq) is not None
    manager._attach_prefix_cache_if_needed(seq)
    assert manager.row_seq_lens[0] == 2
    assert manager.buffer_req_to_token_slots[0, :2].tolist() == [10, 11]
    assert block.ref_count == 1

    manager._allocate(seq.seq_id, 1)
    assert manager.row_seq_lens[0] == 3
    assert manager._num_free_slots == 87

    manager.free_seq(seq.seq_id)
    assert manager._num_free_slots == 88
    assert block.ref_count == 0
    assert manager.seq_id_to_row == {}
    assert manager.prefix_lookup_cache.get(seq) is None


def test_standard_latent_prefix_restores_mla_payload_and_cleans_request_state():
    manager = _make_standard_manager_for_prefix(block_size=2)
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=1, num_slots=100, device=torch.device("cpu"))
    manager.attention_cache_storage = storage

    owner = Sequence([1, 2])
    owner_slots = manager._allocate(owner.seq_id, 2).clone()
    assert storage.latent_cache is not None
    assert storage.rope_cache is not None
    storage.latent_cache[0, owner_slots[0]].fill_(11)
    storage.latent_cache[0, owner_slots[1]].fill_(22)
    storage.rope_cache[0, owner_slots[0]].fill_(33)
    storage.rope_cache[0, owner_slots[1]].fill_(44)
    manager._record_prefix_materialization(owner, [1, 2], owner_slots)
    manager.on_forward_end([owner], is_prefill=True)
    manager.free_seq(owner.seq_id)

    replay = Sequence([1, 2, 9])
    manager.refresh_prefix_cache_hit(replay)
    assert replay.prefix_cache_hit_len == 2
    manager._attach_prefix_cache_if_needed(replay)
    replay_row = manager.seq_id_to_row[replay.seq_id]
    replay_slots = manager.buffer_req_to_token_slots[replay_row, :2].clone()
    assert replay_slots.tolist() == owner_slots.tolist()
    payload = storage.layer_payload(0)
    assert isinstance(payload, MlaLatentPayload)
    torch.testing.assert_close(
        payload.latent_cache[replay_slots, 0, 0],
        torch.tensor([11, 22], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[replay_slots, 0, 0],
        torch.tensor([33, 44], dtype=torch.bfloat16),
    )

    manager.free_seq(replay.seq_id)
    assert replay.seq_id not in manager.seq_id_to_row
    assert replay.seq_id not in manager.seq_id_to_prefix_blocks
    assert replay.seq_id not in manager.seq_id_to_cached_ranges
    assert replay.seq_id not in manager.prefix_runtime_states
    manager.reset_prefix_cache()
    assert len(manager.prefix_cache) == 0
    assert manager._num_free_slots == 90

    unrelated = Sequence([7, 8, 9])
    manager.refresh_prefix_cache_hit(unrelated)
    assert unrelated.prefix_cache_hit_len == 0
    assert unrelated.seq_id not in manager.seq_id_to_row


def _assert_standard_latent_prefix_full_lifecycle(method: str):
    manager = _make_standard_manager_for_prefix(block_size=2, method=method)
    manager.max_model_len = 16
    manager.num_layers = 1
    manager.num_kv_layers = 1
    manager.runtime_layout = RuntimeLayout.dense(1)
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=1, num_slots=100, device=torch.device("cpu"))
    manager.attention_cache_storage = storage

    owner = Sequence([1, 2, 9])
    owner.current_chunk_size = 3
    input_ids, positions, cu_seqlens_q = manager.prepare_step([owner], is_prefill=True)
    assert input_ids.tolist() == [1, 2, 9]
    assert positions.tolist() == [0, 1, 2]
    assert cu_seqlens_q.tolist() == [0, 3]
    owner_slots = manager.layer_batch_state.slot_mapping.clone()
    payload = storage.layer_payload(0)
    payload.latent_cache[owner_slots] = (
        torch.tensor([11, 22, 99], dtype=torch.bfloat16)
        .view(3, 1, 1)
        .expand(3, 1, 512)
    )
    payload.rope_cache[owner_slots] = (
        torch.tensor([33, 44, 88], dtype=torch.bfloat16)
        .view(3, 1, 1)
        .expand(3, 1, 64)
    )
    manager.on_forward_end([owner], is_prefill=True)
    manager.free_seq(owner.seq_id)
    assert len(manager.prefix_cache) == 1
    assert manager._num_free_slots == 88

    replay = Sequence([1, 2, 8])
    manager.refresh_prefix_cache_hit(replay)
    assert replay.prefix_cache_hit_len == 2
    replay.num_prefilled_tokens = replay.prefix_cache_hit_len
    replay.current_chunk_size = 1
    input_ids, positions, cu_seqlens_q = manager.prepare_step([replay], is_prefill=True)
    assert input_ids.tolist() == [8]
    assert positions.tolist() == [2]
    assert cu_seqlens_q.tolist() == [0, 1]
    replay_row = manager.seq_id_to_row[replay.seq_id]
    replay_slots = manager.buffer_req_to_token_slots[replay_row, :3].clone()
    assert replay_slots[:2].tolist() == owner_slots[:2].tolist()
    torch.testing.assert_close(
        payload.latent_cache[replay_slots[:2], 0, 0],
        torch.tensor([11, 22], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[replay_slots[:2], 0, 0],
        torch.tensor([33, 44], dtype=torch.bfloat16),
    )
    payload.latent_cache[replay_slots[2:]] = 77
    payload.rope_cache[replay_slots[2:]] = 66
    manager.on_forward_end([replay], is_prefill=True)
    manager.free_seq(replay.seq_id)
    assert replay.seq_id not in manager.seq_id_to_row
    assert replay.seq_id not in manager.seq_id_to_prefix_blocks
    assert replay.seq_id not in manager.seq_id_to_cached_ranges
    assert replay.seq_id not in manager.seq_id_to_materialized_blocks
    assert replay.seq_id not in manager.pending_prefix_blocks
    assert replay.seq_id not in manager.prefix_runtime_states
    assert manager._num_free_slots == 88

    deleted = manager.prefix_cache_delete_subtree([1, 2])
    assert deleted["deleted_block_count"] == 1
    assert len(manager.prefix_cache) == 0
    assert manager._num_free_slots == 90

    replacement = Sequence([7, 8])
    replacement.current_chunk_size = 2
    manager.prepare_step([replacement], is_prefill=True)
    replacement_slots = manager.layer_batch_state.slot_mapping.clone()
    assert replacement_slots.tolist() == owner_slots[:2].tolist()
    payload.latent_cache[replacement_slots] = 55
    payload.rope_cache[replacement_slots] = 44
    torch.testing.assert_close(
        payload.latent_cache[replacement_slots, 0, 0],
        torch.tensor([55, 55], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[replacement_slots, 0, 0],
        torch.tensor([44, 44], dtype=torch.bfloat16),
    )
    manager.on_forward_end([replacement], is_prefill=True)
    manager.free_seq(replacement.seq_id)
    manager.prefix_cache_delete_subtree([7, 8])
    assert manager._num_free_slots == 90
    assert manager.seq_id_to_row == {}
    assert manager.seq_id_to_prefix_blocks == {}
    assert manager.seq_id_to_cached_ranges == {}
    assert manager.seq_id_to_materialized_blocks == {}
    assert manager.pending_prefix_blocks == {}
    assert manager.prefix_runtime_states == {}


def test_standard_latent_prefix_full_lifecycle_restores_and_reuses_slots():
    _assert_standard_latent_prefix_full_lifecycle("")


def test_standard_gpu_pressure_batches_partial_block_eviction_in_one_tree_scan(
    monkeypatch,
):
    manager = _make_standard_manager_for_prefix(block_size=4)
    blocks = []
    for logical_idx, (slots, priority) in enumerate(
        [([10], 30), ([11, 12, 13, 14], 20), ([15, 16, 17, 18], 0)]
    ):
        tokens = [logical_idx * 4 + offset for offset in range(4)]
        block_id = manager.prefix_cache.stable_block_id(tokens, None)
        retained_offsets = None if len(slots) == 4 else tuple(range(len(slots)))
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=None,
            block_size=4,
            logical_block_idx=logical_idx,
            payload=StandardPrefixBlockPayload(
                token_slots=torch.tensor(slots, dtype=torch.int32),
                retained_offsets=retained_offsets,
            ),
            token_ids=tuple(tokens),
            eviction_priority=priority,
        )
        manager.prefix_cache.insert_block(block)
        _remove_free_slots(manager, slots)
        blocks.append(block)

    leaf_scans = 0
    original_leaf_block_ids = manager.prefix_cache.backend.leaf_block_ids

    def counted_leaf_block_ids():
        nonlocal leaf_scans
        leaf_scans += 1
        return original_leaf_block_ids()

    monkeypatch.setattr(
        manager.prefix_cache.backend,
        "leaf_block_ids",
        counted_leaf_block_ids,
    )
    free_before = manager.num_free_slots

    manager._evict_prefix_cache_until_free(free_before + 5)

    assert manager.num_free_slots == free_before + 5
    assert leaf_scans == 1
    assert blocks[0].stable_block_id not in manager.prefix_cache.blocks
    assert blocks[1].stable_block_id not in manager.prefix_cache.blocks
    assert blocks[2].stable_block_id in manager.prefix_cache.blocks


def test_standard_offload_gpu_pressure_only_demotes_dual_resident_blocks():
    manager = _make_standard_manager_for_prefix(block_size=2)
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    last_block_id = None
    parent_id = None
    blocks = []
    for logical_idx, (tokens, slots) in enumerate(
        [([1, 2], [10, 11]), ([3, 4], [12, 13])]
    ):
        block_id = manager.prefix_cache.stable_block_id(tokens, parent_id)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent_id,
            block_size=2,
            logical_block_idx=logical_idx,
            payload=StandardPrefixBlockPayload(
                token_slots=torch.tensor(slots, dtype=torch.int32),
                host_block_index=logical_idx,
            ),
            token_ids=tuple(tokens),
        )
        manager.prefix_cache.insert_block(block)
        manager.prefix_cache.begin_d2h(block)
        manager.prefix_cache.finish_d2h(block)
        _remove_free_slots(manager, slots)
        blocks.append(block)
        parent_id = block_id
        last_block_id = block_id

    live_blocks = len(manager.prefix_cache)
    manager._evict_prefix_cache_until_free(manager._num_free_slots + 2)

    assert len(manager.prefix_cache) == live_blocks
    assert last_block_id in manager.prefix_cache.blocks
    assert blocks[0].residency.device_present is True
    assert blocks[1].residency.device_present is False
    assert blocks[1].residency.host_present is True
    assert blocks[1].payload.token_slots is None
    assert manager._num_free_slots == 88


def test_standard_offload_gpu_pressure_batches_partial_block_demotions():
    manager = _make_standard_manager_for_prefix(block_size=4)
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    blocks = []
    parent_id = None
    for logical_idx, (slots, retained_offsets) in enumerate(
        [([10, 11, 12, 13], None), ([14], (0,))]
    ):
        tokens = [logical_idx * 4 + offset for offset in range(4)]
        block_id = manager.prefix_cache.stable_block_id(tokens, parent_id)
        block = PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=parent_id,
            block_size=4,
            logical_block_idx=logical_idx,
            payload=StandardPrefixBlockPayload(
                token_slots=torch.tensor(slots, dtype=torch.int32),
                host_block_index=logical_idx,
                retained_offsets=retained_offsets,
            ),
            token_ids=tuple(tokens),
        )
        manager.prefix_cache.insert_block(block)
        manager.prefix_cache.begin_d2h(block)
        manager.prefix_cache.finish_d2h(block)
        _remove_free_slots(manager, slots)
        blocks.append(block)
        parent_id = block_id

    free_before = manager.num_free_slots
    with patch.object(
        manager.prefix_cache,
        "demote_device_until_weight",
        wraps=manager.prefix_cache.demote_device_until_weight,
    ) as demote_device_until_weight:
        manager._evict_prefix_cache_until_free(free_before + 5)

    assert demote_device_until_weight.call_count == 1
    assert manager.num_free_slots == free_before + 5
    assert all(not block.residency.device_present for block in blocks)
    assert all(block.residency.host_present for block in blocks)
    assert all(block.payload.token_slots is None for block in blocks)
    assert len(manager.prefix_cache) == 2


def test_standard_write_through_batches_only_unreferenced_root_contiguous_blocks():
    manager = _make_standard_manager_for_prefix(block_size=2)
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    last_block_id = _insert_tokens(manager.prefix_cache, [1, 2, 3, 4, 5, 6])
    chain = manager.prefix_cache.get_chain(last_block_id, 3)
    for idx, block in enumerate(chain):
        block.payload = StandardPrefixBlockPayload(
            token_slots=torch.tensor([10 + idx * 2, 11 + idx * 2], dtype=torch.int32)
        )
    chain[0].ref_count = 1

    manager._schedule_write_through_prefix_blocks(chain[1:])

    assert controller.submitted_d2h == []
    assert set(manager._prefix_write_through_candidates) == {
        chain[1].stable_block_id,
        chain[2].stable_block_id,
    }
    chain[0].ref_count = 0
    manager._schedule_write_through_prefix_blocks([chain[0]])
    assert controller.submitted_d2h == [chain]
    assert all(block.residency.transfer == PrefixTransferKind.D2H for block in chain)
    assert manager._prefix_write_through_candidates == {}


def test_standard_free_seq_starts_write_through_after_last_reference_release():
    manager = _make_standard_manager_for_prefix(block_size=2)
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    seq = Sequence([1, 2])
    slots = manager._allocate(seq.seq_id, 2)
    manager._record_prefix_materialization(seq, [1, 2], slots)
    manager.on_forward_end([seq], is_prefill=True)
    block = next(iter(manager.prefix_cache.blocks.values()))
    assert block.ref_count == 1

    manager.free_seq(seq.seq_id)

    assert block.ref_count == 0
    assert block.residency.transfer == PrefixTransferKind.D2H
    assert controller.submitted_d2h == [[block]]
    assert manager._num_free_slots == 88


def test_standard_offload_logical_pressure_demotes_then_deletes_cpu_leaf():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.prefix_cache.max_blocks = 1
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
            host_block_index=0,
        ),
        token_ids=(1, 2),
    )
    _remove_free_slots(manager, [10, 11])
    manager.prefix_cache.insert_block(block)
    manager.prefix_cache.begin_d2h(block)
    manager.prefix_cache.finish_d2h(block)

    manager._evict_prefix_cache_for_insert(1)

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_slots == 90
    assert manager.prefix_cache.device_demoted_blocks == 1
    assert manager.prefix_cache.host_evicted_blocks == 1
    assert manager.prefix_cache.evicted_blocks == 0


def test_standard_offload_logical_pressure_waits_for_inflight_d2h_leaf():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.prefix_cache.max_blocks = 1
    controller = _CompletingD2HPrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    _remove_free_slots(manager, [10, 11])
    manager.prefix_cache.insert_block(block)
    manager.prefix_cache.begin_d2h(block)

    manager._evict_prefix_cache_for_insert(1)

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_slots == 90
    assert manager.prefix_cache.device_demoted_blocks == 1
    assert manager.prefix_cache.host_evicted_blocks == 1
    assert manager.prefix_cache.evicted_blocks == 0


def test_standard_cpu_prefix_hit_allocates_slots_and_tracks_layer_wait_operation():
    manager = _make_standard_manager_for_prefix(block_size=2)
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    seq = Sequence([1, 2, 3])
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
            host_block_index=0,
        ),
        token_ids=(1, 2),
    )
    _remove_free_slots(manager, [10, 11])
    manager.prefix_cache.insert_block(block)
    manager.prefix_cache.begin_d2h(block)
    manager.prefix_cache.finish_d2h(block)
    manager.prefix_cache.demote_device_until_freeable(1)
    manager._free_device_prefix_block(block)
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = block_id
    seq.prefix_cache_block_size = 2

    assert manager.prompt_admission_cost(seq) == 3
    manager._attach_prefix_cache_if_needed(seq)

    assert block.ref_count == 1
    assert block.residency.transfer == PrefixTransferKind.H2D
    assert isinstance(block.payload.token_slots, torch.Tensor)
    assert manager.buffer_req_to_token_slots[0, :2].tolist() == block.payload.token_slots.tolist()
    assert len(manager._prefix_offload_step_h2d_operations) == 1
    assert len(controller.submitted_h2d) == 1


def test_h2d_transfer_stream_waits_for_index_producer_event(monkeypatch):
    prefix_cache = RadixPrefixIndex(block_size=2, fingerprint=b"h2d-order")
    block_id = prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(
            token_slots=torch.tensor([2, 3], dtype=torch.int32),
            host_block_index=0,
        ),
        token_ids=(1, 2),
    )
    prefix_cache.insert_block(block)
    prefix_cache.begin_d2h(block)
    prefix_cache.finish_d2h(block)
    prefix_cache.demote_device_until_freeable(1)

    trace = []
    controller = object.__new__(StandardPrefixOffloadController)
    controller.prefix_cache = prefix_cache
    controller.block_size = 2
    controller.device = torch.device("cpu")
    controller.kv_cache = torch.zeros((2, 2, 8, 1, 4), dtype=torch.float16)
    controller.host_pool = SimpleNamespace(
        cache=torch.zeros((2, 2, 1, 2, 1, 4), dtype=torch.float16),
        num_layers=2,
        retained_token_indices=lambda _block_indices, _offsets, device: torch.tensor(
            [0, 1], dtype=torch.long, device=device
        ),
    )
    controller.h2d_stream = object()
    controller.item_size = 8
    controller._new_event = lambda device, purpose: purpose
    controller._transfer_per_layer = lambda **kwargs: trace.append(
        ("transfer", int(kwargs["dst_k"].data_ptr()))
    )
    controller.h2d_operations = deque()
    controller._h2d_by_block_id = {}
    controller.h2d_bytes = 0
    controller.h2d_submitted_operations = 0
    controller.h2d_merged_blocks = 0

    monkeypatch.setattr(
        device_runtime,
        "record_event",
        lambda event, device=None: trace.append(("record", event)),
    )
    monkeypatch.setattr(
        device_runtime,
        "stream_context",
        lambda stream: nullcontext(),
    )
    monkeypatch.setattr(
        device_runtime,
        "stream_wait_event",
        lambda stream, event: trace.append(("wait", event)),
    )

    operation = controller.submit_h2d([block])

    assert trace[0] == ("record", "H2D producer")
    assert trace[1] == ("wait", "H2D producer")
    assert trace[2][0] == "transfer"
    assert operation.producer_event == "H2D producer"


def test_offload_controller_reset_clears_transfer_stats():
    class FakeHostPool:
        capacity_blocks = 8
        used_blocks = 3
        free_blocks = 5

        def reset(self):
            self.used_blocks = 0
            self.free_blocks = self.capacity_blocks

    controller = object.__new__(StandardPrefixOffloadController)
    controller.synchronize_all = lambda: None
    controller.host_pool = FakeHostPool()
    controller.d2h_operations = [object()]
    controller.h2d_operations = [object()]
    controller._h2d_by_block_id = {b"block": object()}
    controller.d2h_bytes = 10
    controller.h2d_bytes = 20
    controller.d2h_submitted_operations = 2
    controller.d2h_completed_operations = 1
    controller.h2d_submitted_operations = 3
    controller.h2d_completed_operations = 2
    controller.d2h_merged_blocks = 4
    controller.h2d_merged_blocks = 5
    controller.layer_waits = 6

    controller.reset()

    stats = controller.stats()
    assert stats["prefix_cache_host_used_blocks"] == 0
    assert stats["prefix_cache_host_free_blocks"] == 8
    assert stats["prefix_cache_d2h_inflight_operations"] == 0
    assert stats["prefix_cache_h2d_inflight_operations"] == 0
    for name in (
        "prefix_cache_d2h_bytes",
        "prefix_cache_h2d_bytes",
        "prefix_cache_d2h_submitted_operations",
        "prefix_cache_d2h_completed_operations",
        "prefix_cache_h2d_submitted_operations",
        "prefix_cache_h2d_completed_operations",
        "prefix_cache_d2h_merged_blocks",
        "prefix_cache_h2d_merged_blocks",
        "prefix_cache_h2d_layer_waits",
    ):
        assert stats[name] == 0


@pytest.mark.parametrize(
    "controller_type",
    [StandardPrefixOffloadController, QuestPrefixOffloadController],
)
def test_prefix_offload_rejects_transfer_submission_during_capture(
    monkeypatch,
    controller_type,
):
    controller = object.__new__(controller_type)
    block = SimpleNamespace()
    monkeypatch.setattr(device_runtime, "is_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="forbidden during graph capture"):
        controller.submit_d2h([block])
    with pytest.raises(RuntimeError, match="forbidden during graph capture"):
        controller.submit_h2d([block])


def test_standard_materializes_blocks_only_after_forward_end():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2])
    slots = torch.tensor([20, 21], dtype=torch.int32)

    manager._record_prefix_materialization(seq, [1, 2], slots)
    assert len(manager.prefix_cache) == 0

    manager.on_forward_end([seq], is_prefill=True)
    assert len(manager.prefix_cache) == 1
    block = next(iter(manager.prefix_cache.blocks.values()))
    assert block.ref_count == 1
    assert isinstance(block.payload, StandardPrefixBlockPayload)
    assert block.payload.token_slots.tolist() == [20, 21]
    assert manager.seq_id_to_cached_ranges[seq.seq_id] == [(0, 2)]


def test_standard_safe_delete_releases_payload_slots():
    manager = _make_standard_manager_for_prefix(block_size=2)
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    _remove_free_slots(manager, [10, 11])
    manager.prefix_cache.insert_block(block)
    assert manager._num_free_slots == 88

    result = manager.prefix_cache_delete_subtree([1, 2])

    assert result["deleted_block_ids"] == [stable_block_id.hex()]
    assert manager._num_free_slots == 90
    assert stable_block_id not in manager.prefix_cache.blocks


def test_standard_prefix_payload_release_invalidates_owner():
    manager = _make_standard_manager_for_prefix(block_size=2)
    payload = StandardPrefixBlockPayload(
        token_slots=torch.tensor([10, 11], dtype=torch.int32)
    )
    _remove_free_slots(manager, [10, 11])
    before = manager._num_free_slots

    manager.free_prefix_kv_payload(payload)

    assert manager._num_free_slots == before + 2
    assert payload.token_slots is None
    with pytest.raises(RuntimeError, match="missing token slots|no device slots"):
        manager.free_prefix_kv_payload(payload)
    assert manager._num_free_slots == before + 2


def test_standard_shared_prefix_admission_exposes_stable_block_cost():
    manager = _make_standard_manager_for_prefix(block_size=2)
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(
            token_slots=torch.tensor([10, 11], dtype=torch.int32)
        ),
        token_ids=(1, 2),
    )
    _remove_free_slots(manager, [10, 11])
    manager.prefix_cache.insert_block(block)
    seq = Sequence([1, 2, 3])
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = block_id

    assert manager.prompt_admission_cost(seq) == 3
    assert manager.prompt_admission_shared_costs(seq) == {
        "slots": {block_id: 2}
    }


def test_standard_safe_delete_partial_subtree_releases_only_deleted_child_slots():
    manager = _make_standard_manager_for_prefix(block_size=2)
    root_id = manager.prefix_cache.stable_block_id([1, 2], None)
    referenced_child_id = manager.prefix_cache.stable_block_id([3, 4], root_id)
    free_child_id = manager.prefix_cache.stable_block_id([5, 6], root_id)
    root = PrefixCacheBlock(
        stable_block_id=root_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    referenced_child = PrefixCacheBlock(
        stable_block_id=referenced_child_id,
        parent_block_id=root_id,
        block_size=2,
        logical_block_idx=1,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([12, 13], dtype=torch.int32)),
        token_ids=(3, 4),
        ref_count=1,
    )
    free_child = PrefixCacheBlock(
        stable_block_id=free_child_id,
        parent_block_id=root_id,
        block_size=2,
        logical_block_idx=1,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([14, 15], dtype=torch.int32)),
        token_ids=(5, 6),
    )
    _remove_free_slots(manager, [10, 11, 12, 13, 14, 15])
    manager.prefix_cache.insert_block(root)
    manager.prefix_cache.insert_block(referenced_child)
    manager.prefix_cache.insert_block(free_child)
    assert manager._num_free_slots == 84

    result = manager.prefix_cache_delete_subtree([1, 2])

    assert result["deleted_block_ids"] == [free_child_id.hex()]
    assert {item["reason"] for item in result["blocked_blocks"]} == {"referenced", "has_children"}
    assert manager._num_free_slots == 86
    assert root_id in manager.prefix_cache.blocks
    assert referenced_child_id in manager.prefix_cache.blocks
    assert free_child_id not in manager.prefix_cache.blocks


def test_standard_pending_slots_do_not_alias_free_stack_storage():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2])
    first_slot_view = manager.free_slots_stack[89:90]

    manager._record_prefix_materialization(seq, [1], first_slot_view)
    manager.free_slots_stack[89] = 777
    manager._record_prefix_materialization(seq, [2], torch.tensor([88], dtype=torch.int32))

    pending = manager.pending_prefix_blocks[seq.seq_id][0]
    assert pending.slots.tolist() == [89, 88]


def test_standard_admission_reserves_evictable_hit_blocks():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager._num_free_slots = 1
    seq = Sequence([1, 2, 3])
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = stable_block_id

    assert manager.prompt_admission_free_slots() == 3
    assert manager.prompt_admission_cost(seq) == 3

    manager.prefix_cache.acquire_block_ref(block)
    assert manager.prompt_admission_cost(seq) == 1


def test_standard_scheduler_capacity_snapshot_reuses_freeable_tree_scan():
    manager = _make_standard_manager_for_prefix(block_size=2)
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    manager.prefix_cache.insert_block(
        PrefixCacheBlock(
            stable_block_id=stable_block_id,
            parent_block_id=None,
            block_size=2,
            logical_block_idx=0,
            payload=StandardPrefixBlockPayload(
                token_slots=torch.tensor([10, 11], dtype=torch.int32)
            ),
            token_ids=(1, 2),
        )
    )

    with patch.object(
        manager.prefix_cache,
        "freeable_block_ids",
        wraps=manager.prefix_cache.freeable_block_ids,
    ) as freeable_block_ids, patch.object(
        manager,
        "_prefix_resident_slots_for_ids",
        wraps=manager._prefix_resident_slots_for_ids,
    ) as resident_slots_for_ids:
        with manager.scheduler_capacity_snapshot():
            assert manager.prompt_admission_free_slots() == 92
            assert manager.prefill_step_free_slots() == 92
            assert manager.decode_step_free_slots() == 92
            assert manager.prompt_admission_budgets(deque(), 2)["slots"] == 92
        assert freeable_block_ids.call_count == 1
        assert resident_slots_for_ids.call_count == 2

        assert manager.decode_step_free_slots() == 92
        assert freeable_block_ids.call_count == 2


def test_standard_capacity_reuses_weights_across_passes_and_invalidates_compaction():
    # The index already cached IDs, but each new scheduler pass still walked
    # every payload. Also catch stale totals when IDs survive compaction.
    manager = _make_standard_manager_for_prefix(block_size=2)
    index = manager.prefix_cache
    leaf_id = _insert_tokens(index, [1, 2, 3, 4])
    chain = index.get_chain(leaf_id, 2)
    for block in chain:
        block.payload = StandardPrefixBlockPayload(
            token_slots=torch.tensor([10, 11], dtype=torch.int32)
        )
    with patch.object(manager, "_block_resident_tokens_or_full",
                      wraps=manager._block_resident_tokens_or_full) as weight:
        for _ in range(3):
            index.touch_chain(chain)
            with manager.scheduler_capacity_snapshot():
                assert manager.prompt_admission_free_slots() == 94
        assert weight.call_count == len(chain)

        leaf = chain[-1]
        leaf.payload.token_slots = leaf.payload.token_slots[:1]
        leaf.payload.retained_offsets = (1,)
        index.mark_payload_compacted([leaf])
        assert manager.prompt_admission_free_slots() == 93

        index.acquire_block_ref(leaf)
        assert manager.prompt_admission_free_slots() == 90
        index.release_block_ref(leaf)
        assert manager.prompt_admission_free_slots() == 93
        index.set_subtree_eviction_priority([1, 2, 3, 4], -1)
        assert manager.prompt_admission_free_slots() == 90
        index.set_subtree_eviction_priority([1, 2, 3, 4], 0)
        assert manager.prompt_admission_free_slots() == 93
        assert index.evict_until_freeable(1) == [leaf]
        assert manager.prompt_admission_free_slots() == 92
        _insert_tokens(index, [5, 6])
        assert manager.prompt_admission_free_slots() == 94


def test_standard_capacity_membership_changes_reuse_unchanged_payload_weights():
    manager = _make_standard_manager_for_prefix(block_size=2)
    index = manager.prefix_cache
    first_id = _insert_tokens(index, [1, 2])
    first = index.get_block(first_id)
    with patch.object(manager, "_block_resident_tokens_or_full",
                      wraps=manager._block_resident_tokens_or_full) as weight:
        assert manager.prompt_admission_free_slots() == 92
        for _ in range(3):
            index.acquire_block_ref(first)
            assert manager.prompt_admission_free_slots() == 90
            index.release_block_ref(first)
            assert manager.prompt_admission_free_slots() == 92
        assert weight.call_count == 1
        _insert_tokens(index, [3, 4])
        assert manager.prompt_admission_free_slots() == 94
        assert weight.call_count == 2


def test_standard_capacity_deletion_keeps_surviving_payload_weights():
    manager = _make_standard_manager_for_prefix(block_size=2)
    index = manager.prefix_cache
    _insert_tokens(index, [1, 2])
    _insert_tokens(index, [3, 4])
    with patch.object(manager, "_block_resident_tokens_or_full",
                      wraps=manager._block_resident_tokens_or_full) as weight:
        assert manager.prompt_admission_free_slots() == 94
        assert weight.call_count == 2

        evicted = index.evict_until_freeable(1)
        assert len(evicted) == 1
        assert manager.prompt_admission_free_slots() == 92
        assert weight.call_count == 2

        block_id = _insert_tokens(index, list(evicted[0].token_ids))
        assert block_id == evicted[0].stable_block_id
        index.get_block(block_id).payload = StandardPrefixBlockPayload(
            token_slots=torch.tensor([10], dtype=torch.int32), retained_offsets=(0,)
        )
        assert manager.prompt_admission_free_slots() == 93
        assert weight.call_count == 3


def test_standard_cached_weights_follow_residency_and_transfer_rollback():
    manager = _make_standard_manager_for_prefix(block_size=2)
    index = manager.prefix_cache
    block_id = _insert_tokens(index, [1, 2])
    block = index.get_block(block_id)
    ids = frozenset({block_id})
    assert manager._prefix_resident_slots_for_ids(ids) == 2
    index.begin_d2h(block)
    index.finish_d2h(block)
    assert index.demote_device_until_freeable(1) == [block]
    assert manager._prefix_resident_slots_for_ids(ids) == 0
    index.begin_h2d(block)
    assert manager._prefix_resident_slots_for_ids(ids) == 2
    index.abort_h2d(block)
    assert manager._prefix_resident_slots_for_ids(ids) == 0
    index.begin_h2d(block)
    index.finish_h2d(block)
    assert manager._prefix_resident_slots_for_ids(ids) == 2


def test_standard_cached_weights_refresh_reinserted_stable_id():
    manager = _make_standard_manager_for_prefix(block_size=2)
    index = manager.prefix_cache
    block_id = _insert_tokens(index, [1, 2])
    assert manager.prompt_admission_free_slots() == 92
    index.evict_until_freeable(1)
    assert _insert_tokens(index, [1, 2]) == block_id
    index.get_block(block_id).payload = StandardPrefixBlockPayload(
        token_slots=torch.tensor([10], dtype=torch.int32), retained_offsets=(0,)
    )
    # No intermediate capacity query between deletion and reinsertion.
    assert manager.prompt_admission_free_slots() == 91


def test_standard_capacity_cache_does_not_cross_index_replacement():
    # Equal epochs and IDs in a rebuilt index do not imply equal payload sizes.
    manager = _make_standard_manager_for_prefix(block_size=2)
    first = manager.prefix_cache
    block_id = _insert_tokens(first, [1, 2])
    assert manager.prompt_admission_free_slots() == 92
    replacement = RadixPrefixIndex(block_size=2, fingerprint=first.fingerprint)
    assert _insert_tokens(replacement, [1, 2]) == block_id
    replacement.get_block(block_id).payload = StandardPrefixBlockPayload(
        token_slots=torch.tensor([10], dtype=torch.int32), retained_offsets=(0,)
    )
    assert replacement.capacity_epoch == first.capacity_epoch
    manager.prefix_cache = replacement
    assert manager.prompt_admission_free_slots() == 91


def test_standard_cached_capacity_tracks_transfer_and_residency_changes():
    # Offload queries share IDs but have different transfer eligibility; cached
    # totals must not count a host-only block or a transfer as ready capacity.
    manager = _make_standard_manager_for_prefix(block_size=2)
    index = manager.prefix_cache
    manager.prefix_offload_controller = _FakePrefixOffloadController(index)
    block = index.get_block(_insert_tokens(index, [1, 2]))
    block.payload = StandardPrefixBlockPayload(
        token_slots=torch.tensor([10], dtype=torch.int32), retained_offsets=(0,)
    )

    def capacity():
        return manager._prefix_evictable_slots(), manager._prefix_step_reclaimable_slots()

    assert capacity() == (0, 0)
    index.begin_d2h(block)
    assert capacity() == (0, 1)
    index.abort_d2h(block)
    assert capacity() == (0, 0)
    index.begin_d2h(block)
    index.finish_d2h(block)
    assert capacity() == (1, 1)
    with patch.object(index, "get_block", side_effect=AssertionError("repeated payload walk")):
        for _ in range(3):
            assert capacity() == (1, 1)
    assert index.demote_device_until_freeable(1) == [block]
    assert capacity() == (0, 0)
    index.begin_h2d(block)
    assert capacity() == (0, 0)
    index.abort_h2d(block)
    assert capacity() == (0, 0)
    index.begin_h2d(block)
    index.finish_h2d(block)
    assert capacity() == (1, 1)


def test_standard_capacity_skips_prefix_scan_with_physical_step_headroom():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.config.max_num_batched_tokens = 16
    manager.config.max_num_seqs_in_batch = 4
    manager.config.max_decoding_seqs = 4

    with patch.object(
        manager.prefix_cache,
        "freeable_block_ids",
        wraps=manager.prefix_cache.freeable_block_ids,
    ) as freeable_block_ids:
        assert manager.prefill_step_free_slots() == 90
        assert manager.decode_step_free_slots() == 90
        assert freeable_block_ids.call_count == 0


def test_standard_decode_uses_evictable_leaf_headroom_without_tree_scan():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.config.max_num_seqs_in_batch = 4
    manager.config.max_decoding_seqs = 4
    manager._num_free_slots = 0
    for token_ids in ([1, 2], [3, 4]):
        stable_block_id = manager.prefix_cache.stable_block_id(
            token_ids,
            None,
        )
        manager.prefix_cache.insert_block(
            PrefixCacheBlock(
                stable_block_id=stable_block_id,
                parent_block_id=None,
                block_size=2,
                logical_block_idx=0,
                payload=StandardPrefixBlockPayload(
                    token_slots=torch.tensor(
                        token_ids,
                        dtype=torch.int32,
                    )
                ),
                token_ids=tuple(token_ids),
            )
        )

    with patch.object(
        manager.prefix_cache,
        "freeable_block_ids",
        side_effect=AssertionError("full radix scan should be skipped"),
    ):
        assert manager.decode_step_free_slots() == 4


def test_standard_admission_counts_inflight_d2h_before_pressure_prompt():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager._num_free_slots = 1
    manager.prefix_offload_controller = _FakePrefixOffloadController(manager.prefix_cache)
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(
            token_slots=torch.tensor([10, 11], dtype=torch.int32)
        ),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(block)
    manager.prefix_cache.begin_d2h(block)
    pressure_seq = Sequence([7, 8, 9])

    assert manager._prefix_evictable_slots() == 0
    assert manager.prefill_step_free_slots() == 3
    assert manager.decode_step_free_slots() == 3
    assert manager.prompt_admission_cost(pressure_seq) == 3
    assert manager.prompt_admission_free_slots() == 3
    assert manager.prompt_admission_budgets(deque(), 2)["slots"] == 3
    hit_seq = Sequence([1, 2, 3])
    hit_seq.prefix_cache_hit_len = 2
    hit_seq.prefix_cache_hit_block_count = 1
    hit_seq.prefix_cache_hit_last_block_id = block_id
    assert manager.prompt_admission_cost(hit_seq) == 3

    manager.prefix_cache.acquire_block_ref(block)
    assert manager.prompt_admission_free_slots() == 1


def test_standard_materializes_child_after_prefix_hit_with_parent_sensitive_id():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2, 3, 4])
    root_id = manager.prefix_cache.stable_block_id([1, 2], None)
    root = PrefixCacheBlock(
        stable_block_id=root_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=StandardPrefixBlockPayload(token_slots=torch.tensor([10, 11], dtype=torch.int32)),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(root)
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = root_id
    seq.prefix_cache_block_size = 2

    manager._attach_prefix_cache_if_needed(seq)
    manager._record_prefix_materialization(seq, [3, 4], torch.tensor([20, 21], dtype=torch.int32))
    manager.on_forward_end([seq], is_prefill=True)

    child_id = manager.prefix_cache.stable_block_id([3, 4], root_id)
    child = manager.prefix_cache.get_block(child_id)
    assert child is not None
    assert child.parent_block_id == root_id
    assert child.logical_block_idx == 1
    assert child.ref_count == 1
    assert isinstance(child.payload, StandardPrefixBlockPayload)
    assert child.payload.token_slots.tolist() == [20, 21]
    assert [block.stable_block_id for block in manager.prefix_cache.get_chain(child_id, 2)] == [root_id, child_id]


def test_standard_duplicate_materialization_holds_parent_until_sequence_free():
    manager = _make_standard_manager_for_prefix(block_size=2)
    owner = Sequence([1, 2])
    owner_slots = manager._allocate(owner.seq_id, 2)
    manager._record_prefix_materialization(owner, [1, 2], owner_slots)
    manager.on_forward_end([owner], is_prefill=True)

    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = manager.prefix_cache.get_block(block_id)
    assert block is not None
    manager.free_seq(owner.seq_id)
    assert block.ref_count == 0

    replay = Sequence([1, 2, 3, 4])
    replay_slots = manager._allocate(replay.seq_id, 4)
    manager._record_prefix_materialization(replay, [1, 2], replay_slots[:2])
    manager.on_forward_end([replay], is_prefill=True)

    assert block.ref_count == 1
    assert not manager.prefix_cache.can_evict(block)
    assert manager.prefix_cache.evict_until_freeable(1) == []
    assert list(manager.seq_id_to_materialized_blocks[replay.seq_id].values()) == [block]
    assert manager.seq_id_to_cached_ranges.get(replay.seq_id, []) == []

    manager._record_prefix_materialization(replay, [3, 4], replay_slots[2:])
    manager.on_forward_end([replay], is_prefill=True)
    child_id = manager.prefix_cache.stable_block_id([3, 4], block_id)
    assert manager.prefix_cache.get_block(child_id) is not None

    manager.free_seq(replay.seq_id)
    assert block.ref_count == 0


def test_standard_decode_token_completes_pending_prefix_block_by_default():
    manager = _make_standard_manager_for_prefix(block_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    assert len(manager.prefix_cache) == 0
    assert manager.pending_prefix_blocks[seq.seq_id] == []

    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)
    manager._prepare_decode([seq])
    manager.on_forward_end([seq], is_prefill=False)

    block_id = manager.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = manager.prefix_cache.get_block(block_id)
    assert block is not None
    assert block.token_ids == (1, 2, 3, 4)
    assert block.logical_block_idx == 0
    assert block.ref_count == 1
    assert isinstance(block.payload, StandardPrefixBlockPayload)
    assert block.payload.token_slots.tolist() == [87, 88, 89, 86]
    assert manager.seq_id_to_cached_ranges[seq.seq_id] == [(0, 4)]


def test_standard_static_decode_padding_does_not_materialize_padded_rows():
    manager = _make_standard_manager_for_prefix(block_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)

    input_ids = torch.empty((4,), dtype=torch.int64)
    positions = torch.empty((4,), dtype=torch.int64)
    slot_mapping = torch.empty((4,), dtype=torch.int32)
    context_lens = torch.empty((4,), dtype=torch.int32)
    req_indices = torch.empty((4,), dtype=torch.int32)

    manager.prepare_decode_static([seq], input_ids, positions, slot_mapping, context_lens, req_indices)
    manager.on_forward_end([seq], is_prefill=False)

    assert input_ids.tolist() == [4, 4, 4, 4]
    assert positions.tolist() == [3, 3, 3, 3]
    assert slot_mapping.tolist()[1:] == [-1, -1, -1]
    assert context_lens.tolist() == [4, 4, 4, 4]
    assert req_indices.tolist() == [0, 0, 0, 0]
    assert len(manager.prefix_cache) == 1
    block = next(iter(manager.prefix_cache.blocks.values()))
    assert block.token_ids == (1, 2, 3, 4)
    assert block.payload.token_slots.numel() == 4
    assert manager._num_free_slots == 86


def test_standard_decode_graph_state_updates_stable_typed_inputs():
    manager = _make_standard_manager_for_prefix(block_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)

    contract = DecodeGraphContract(
        method="",
        topology_path_id="dense",
        batch_capacity=4,
        context_capacity=16,
    )
    inputs = DecodeGraphInputs.allocate(
        contract,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    state = manager.init_decode_graph_state(contract, inputs)
    pointers = inputs.data_ptrs()
    assert contract.capability_level == "strict"

    manager.prepare_decode_graph_step([seq], state)

    assert inputs.data_ptrs() == pointers
    assert inputs.input_ids.tolist() == [4, 4, 4, 4]
    assert inputs.positions.tolist() == [3, 3, 3, 3]
    assert inputs.write_slot_mapping.tolist()[1:] == [-1, -1, -1]
    assert inputs.context_lens.tolist() == [4, 4, 4, 4]
    assert inputs.request_indices.tolist() == [0, 0, 0, 0]
    assert inputs.active_mask.tolist() == [True, False, False, False]
    assert all(tensor.data_ptr() for tensor in inputs.keepalive_tensors())


def test_standard_decode_graph_publishes_reserved_slot_in_graph_phase():
    manager = _make_standard_manager_for_prefix(block_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)
    contract = DecodeGraphContract(
        method="",
        topology_path_id="dense",
        batch_capacity=4,
        context_capacity=16,
    )
    inputs = DecodeGraphInputs.allocate(
        contract,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    state = manager.init_decode_graph_state(contract, inputs)
    row = manager.seq_id_to_row[seq.seq_id]

    manager.prepare_decode_graph_step([seq], state)

    reserved_slot = int(inputs.write_slot_mapping[0])
    assert manager.buffer_req_to_token_slots[row, 3].item() == 0
    manager.prepare_decode_graph_in(state)
    assert manager.buffer_req_to_token_slots[row, 3].item() == reserved_slot


def test_standard_decode_graph_rejects_capacity_before_cache_mutation():
    manager = _make_standard_manager_for_prefix(block_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)
    contract = DecodeGraphContract(
        method="",
        topology_path_id="dense",
        batch_capacity=1,
        context_capacity=3,
    )
    inputs = DecodeGraphInputs.allocate(
        contract,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    state = manager.init_decode_graph_state(contract, inputs)
    free_slots_before = manager._num_free_slots
    row_len_before = int(manager.row_seq_lens[manager.seq_id_to_row[seq.seq_id]])

    with pytest.raises(ValueError, match="exceeded the captured graph context"):
        manager.prepare_decode_graph_step([seq], state)

    assert manager._num_free_slots == free_slots_before
    assert int(manager.row_seq_lens[manager.seq_id_to_row[seq.seq_id]]) == row_len_before


def test_standard_decode_graph_capacity_failure_does_not_claim_new_rows():
    manager = _make_standard_manager_for_prefix(block_size=4)
    existing = Sequence([1, 2, 3])
    manager._allocate(existing.seq_id, 3)
    existing.num_prefilled_tokens = existing.num_prompt_tokens
    existing.append_token(4)
    newcomer = Sequence([11])
    newcomer.num_prefilled_tokens = newcomer.num_prompt_tokens
    newcomer.append_token(12)
    contract = DecodeGraphContract(
        method="",
        topology_path_id="dense",
        batch_capacity=2,
        context_capacity=3,
    )
    inputs = DecodeGraphInputs.allocate(
        contract,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    state = manager.init_decode_graph_state(contract, inputs)
    mappings_before = dict(manager.seq_id_to_row)
    free_rows_before = tuple(manager.free_rows)

    with pytest.raises(ValueError, match="exceeded the captured graph context"):
        manager.prepare_decode_graph_step([newcomer, existing], state)

    assert manager.seq_id_to_row == mappings_before
    assert tuple(manager.free_rows) == free_rows_before


def test_standard_decode_materialized_block_can_seed_later_prefix_hit():
    manager = _make_standard_manager_for_prefix(block_size=4)
    first = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(first.seq_id, 3)
    manager._record_prefix_materialization(first, [1, 2, 3], prompt_slots)
    first.num_prefilled_tokens = first.num_prompt_tokens
    first.append_token(4)
    manager._prepare_decode([first])
    manager.on_forward_end([first], is_prefill=False)

    second = Sequence([1, 2, 3, 4, 5])
    manager.refresh_prefix_cache_hit(second)

    assert second.prefix_cache_hit_len == 4
    assert second.prefix_cache_hit_block_count == 1
    manager._attach_prefix_cache_if_needed(second)
    row_idx = manager.seq_id_to_row[second.seq_id]
    assert manager.row_seq_lens[row_idx] == 4
    assert manager.buffer_req_to_token_slots[row_idx, :4].tolist() == [87, 88, 89, 86]
    block = manager.prefix_cache.get_block(second.prefix_cache_hit_last_block_id)
    assert block is not None
    assert block.ref_count == 2


def test_standard_reset_prefix_cache_clears_warmup_blocks_and_restores_allocator():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2, 3, 4])
    slots = manager._allocate(seq.seq_id, 4)
    manager._record_prefix_materialization(seq, [1, 2, 3, 4], slots)
    manager.on_forward_end([seq], is_prefill=True)
    manager.free_seq(seq.seq_id)
    assert len(manager.prefix_cache) == 2
    assert manager._num_free_slots == 86

    manager.reset_prefix_cache()

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_slots == 90
    assert manager.free_slots_stack[:90].tolist() == list(range(90))


def test_standard_reset_after_warmup_restores_allocator_without_prefix_cache():
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.enable_prefix_caching = False
    manager.prefix_cache = None
    manager.free_slots_stack[:90] = torch.tensor(list(range(86)) + [89, 88, 87, 86], dtype=torch.int32)

    manager.reset_after_warmup()

    assert manager._num_free_slots == 90
    assert manager.free_slots_stack[:90].tolist() == list(range(90))


def test_standard_reset_after_warmup_clears_prefix_cache_and_allocator():
    manager = _make_standard_manager_for_prefix(block_size=2)
    seq = Sequence([1, 2, 3, 4])
    slots = manager._allocate(seq.seq_id, 4)
    manager._record_prefix_materialization(seq, [1, 2, 3, 4], slots)
    manager.on_forward_end([seq], is_prefill=True)
    manager.free_seq(seq.seq_id)

    manager.reset_after_warmup()

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_slots == 90
    assert manager.free_slots_stack[:90].tolist() == list(range(90))


def test_quest_attach_pins_pages_and_free_seq_keeps_cached_page():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2, 3])
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    _remove_free_page(manager, 5)
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = stable_block_id
    seq.prefix_cache_block_size = 2
    seq.prefix_cache_method = "quest"

    manager.refresh_prefix_cache_hit(seq)
    assert manager.prefix_lookup_cache.get(seq) is not None
    manager._attach_prefix_cache_if_needed(seq)
    assert manager.row_seq_lens[0] == 2
    assert manager.buffer_req_to_page_slots[0, 0].item() == 5
    assert manager.buffer_req_to_token_slots[0, :2].tolist() == [10, 11]
    assert block.ref_count == 1

    manager._allocate(seq.seq_id, 1)
    assert manager.row_seq_lens[0] == 3
    assert manager._num_free_pages == 8

    manager.free_seq(seq.seq_id)
    assert manager._num_free_pages == 9
    assert block.ref_count == 0
    assert manager.prefix_lookup_cache.get(seq) is None

    evicted = manager.prefix_cache.evict_until_freeable(1)
    manager._free_prefix_cache_blocks(evicted)
    assert manager._num_free_pages == 10
    reused = manager._allocate(Sequence([4]).seq_id, 1)
    assert reused.tolist() == [10]


def test_quest_prefix_hit_preserves_mla_latent_page_payload():
    manager = _make_quest_manager_for_prefix(page_size=2)
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=1, num_slots=20, device=torch.device("cpu"))
    manager.attention_cache_storage = storage

    owner = Sequence([1, 2])
    owner_slots = manager._allocate(owner.seq_id, 2).clone()
    payload = storage.layer_payload(0)
    payload.latent_cache[owner_slots] = 11
    payload.rope_cache[owner_slots] = 22
    manager._record_prefix_materialization(owner, [1, 2], owner_slots)
    manager.on_forward_end([owner], is_prefill=True)
    manager.free_seq(owner.seq_id)

    replay = Sequence([1, 2, 3])
    manager.refresh_prefix_cache_hit(replay)
    manager._attach_prefix_cache_if_needed(replay)
    row = manager.seq_id_to_row[replay.seq_id]
    replay_slots = manager.buffer_req_to_token_slots[row, :2].clone()

    assert replay_slots.tolist() == owner_slots.tolist()
    torch.testing.assert_close(
        payload.latent_cache[replay_slots],
        torch.full_like(payload.latent_cache[replay_slots], 11),
    )
    torch.testing.assert_close(
        payload.rope_cache[replay_slots],
        torch.full_like(payload.rope_cache[replay_slots], 22),
    )


def test_quest_prefill_replaces_stale_decode_graph_context_capacity():
    """Catches post-capture prefill reusing decode-only context metadata."""

    manager = _make_quest_manager_for_prefix(page_size=2)
    manager.layer_batch_state.max_context_len = 4672
    seq = Sequence([1, 2, 3, 4])
    seq.current_chunk_size = 4

    manager.prepare_step([seq], is_prefill=True)

    assert manager.layer_batch_state.context_lens.tolist() == [4]
    assert manager.layer_batch_state.max_context_len == 4


def test_quest_free_seq_starts_atomic_write_through_after_last_release():
    manager = _make_quest_manager_for_prefix(page_size=2)
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    seq = Sequence([1, 2])
    slots = manager._allocate(seq.seq_id, 2)
    manager._record_prefix_materialization(seq, [1, 2], slots)
    manager.on_forward_end([seq], is_prefill=True)
    block = next(iter(manager.prefix_cache.blocks.values()))

    manager.free_seq(seq.seq_id)

    assert block.ref_count == 0
    assert block.residency.transfer == PrefixTransferKind.D2H
    assert controller.submitted_d2h == [[block]]
    assert manager._num_free_pages == 9


def test_quest_gpu_pressure_demotes_dual_resident_page_without_tree_delete():
    manager = _make_quest_manager_for_prefix(page_size=2)
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
            host_block_index=0,
        ),
        token_ids=(1, 2),
    )
    _remove_free_page(manager, 5)
    manager.prefix_cache.insert_block(block)
    manager.prefix_cache.begin_d2h(block)
    manager.prefix_cache.finish_d2h(block)

    manager._evict_prefix_cache_until_free(manager.num_free_slots + 2)

    assert block_id in manager.prefix_cache.blocks
    assert block.residency.device_present is False
    assert block.residency.host_present is True
    assert block.payload.block_slot is None
    assert block.payload.token_slots is None
    assert manager._num_free_pages == 10


def test_quest_logical_pressure_waits_for_inflight_d2h_then_deletes_host_leaf():
    manager = _make_quest_manager_for_prefix(page_size=2)
    manager.prefix_cache.max_blocks = 1
    controller = _CompletingD2HPrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    _remove_free_page(manager, 5)
    manager.prefix_cache.insert_block(block)
    manager.prefix_cache.begin_d2h(block)

    manager._evict_prefix_cache_for_insert(1)

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_pages == 10
    assert manager.prefix_cache.device_demoted_blocks == 1
    assert manager.prefix_cache.host_evicted_blocks == 1
    assert manager.prefix_cache.evicted_blocks == 0


def test_quest_reset_rebinds_offload_controller_to_new_radix_index():
    manager = _make_quest_manager_for_prefix(page_size=2)
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    old_prefix_cache = manager.prefix_cache

    manager.reset_prefix_cache()

    assert manager.prefix_cache is not old_prefix_cache
    assert controller.prefix_cache is manager.prefix_cache
    assert controller.reset_count == 1


def test_quest_cpu_prefix_hit_promotes_page_and_tracks_layer_wait():
    manager = _make_quest_manager_for_prefix(page_size=2)
    controller = _FakePrefixOffloadController(manager.prefix_cache)
    manager.prefix_offload_controller = controller
    seq = Sequence([1, 2, 3])
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
            host_block_index=0,
        ),
        token_ids=(1, 2),
    )
    _remove_free_page(manager, 5)
    manager.prefix_cache.insert_block(block)
    manager.prefix_cache.begin_d2h(block)
    manager.prefix_cache.finish_d2h(block)
    manager.prefix_cache.demote_device_until_freeable(1)
    manager._free_device_prefix_block(block)
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = block_id

    assert manager.prompt_admission_cost(seq) == 4
    manager._attach_prefix_cache_if_needed(seq)

    assert block.ref_count == 1
    assert block.residency.transfer == PrefixTransferKind.H2D
    assert block.payload.block_slot is not None
    assert isinstance(block.payload.token_slots, torch.Tensor)
    assert manager.buffer_req_to_page_slots[0, 0].item() == block.payload.block_slot
    assert manager.buffer_req_to_token_slots[0, :2].tolist() == block.payload.token_slots.tolist()
    assert len(manager._prefix_offload_step_h2d_operations) == 1
    manager.before_prefill_layer_attention(0, SimpleNamespace())
    assert controller.waited_layers == [(controller.submitted_h2d[0], 0)]


def test_quest_allocate_can_fill_partial_page_without_free_pages():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2])
    manager._num_free_pages = 1

    first = manager._allocate(seq.seq_id, 1)
    assert first.tolist() == [0]
    assert manager._num_free_pages == 0

    second = manager._allocate(seq.seq_id, 1)
    assert second.tolist() == [1]
    assert manager._num_free_pages == 0


def test_quest_prefill_step_capacity_counts_partial_pages():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2])
    manager._num_free_pages = 1
    manager._allocate(seq.seq_id, 1)
    manager._num_free_pages = 0

    assert manager.prefill_step_free_slots() == 1
    assert manager.prefill_step_free_slots_for(seq) == 1
    assert manager.prefill_step_reservation_cost(seq, 1) == 1
    assert manager.prefill_step_reservation_cost(Sequence([3]), 1) == 2
    assert manager.prefill_step_free_slots_for(Sequence([3])) == 0


def test_quest_decode_capacity_counts_requests_not_tokens():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2])
    manager._num_free_pages = 1

    assert manager.decode_step_free_slots() == 2
    assert manager.decode_step_free_slots_for(seq) == 2
    assert manager.decode_step_reservation_cost(seq) == 2

    manager._allocate(seq.seq_id, 1)
    manager._num_free_pages = 0
    assert manager.decode_step_free_slots() == 1
    assert manager.decode_step_free_slots_for(seq) == 1
    assert manager.decode_step_reservation_cost(seq) == 1
    assert manager.decode_step_free_slots_for(Sequence([3])) == 0
    assert manager.decode_step_reservation_cost(Sequence([3])) == 2


def test_quest_materializes_pages_only_after_forward_end():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2])
    slots = torch.tensor([4, 5], dtype=torch.int32)

    manager._record_prefix_materialization(seq, [1, 2], slots)
    assert len(manager.prefix_cache) == 0

    manager.on_forward_end([seq], is_prefill=True)
    assert len(manager.prefix_cache) == 1
    block = next(iter(manager.prefix_cache.blocks.values()))
    assert block.ref_count == 1
    assert not hasattr(block, "page_slot")
    assert not hasattr(block, "slots")
    assert isinstance(block.payload, QuestPrefixBlockPayload)
    assert block.payload.block_slot == 2
    assert block.payload.token_slots.tolist() == [4, 5]
    assert manager.seq_id_to_cached_pages[seq.seq_id] == {0}


def test_quest_decode_token_completes_pending_prefix_page_by_default():
    manager = _make_quest_manager_for_prefix(page_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    assert len(manager.prefix_cache) == 0

    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)
    manager._prepare_decode([seq])
    manager.on_forward_end([seq], is_prefill=False)

    block_id = manager.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = manager.prefix_cache.get_block(block_id)
    assert block is not None
    assert block.token_ids == (1, 2, 3, 4)
    assert block.logical_block_idx == 0
    assert block.ref_count == 1
    assert isinstance(block.payload, QuestPrefixBlockPayload)
    assert block.payload.block_slot == 9
    assert block.payload.token_slots.tolist() == [36, 37, 38, 39]
    assert manager.seq_id_to_cached_pages[seq.seq_id] == {0}


def test_quest_static_decode_padding_does_not_materialize_padded_rows():
    manager = _make_quest_manager_for_prefix(page_size=4)
    seq = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(seq.seq_id, 3)
    manager._record_prefix_materialization(seq, [1, 2, 3], prompt_slots)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(4)

    input_ids = torch.empty((4,), dtype=torch.int64)
    positions = torch.empty((4,), dtype=torch.int64)
    slot_mapping = torch.empty((4,), dtype=torch.int32)
    context_lens = torch.empty((4,), dtype=torch.int32)
    req_indices = torch.empty((4,), dtype=torch.int32)

    manager.prepare_decode_static([seq], input_ids, positions, slot_mapping, context_lens, req_indices)
    manager.on_forward_end([seq], is_prefill=False)

    assert input_ids.tolist() == [4, 4, 4, 4]
    assert positions.tolist() == [3, 3, 3, 3]
    assert slot_mapping.tolist()[1:] == [-1, -1, -1]
    assert context_lens.tolist() == [4, 4, 4, 4]
    assert req_indices.tolist() == [0, 0, 0, 0]
    assert len(manager.prefix_cache) == 1
    block = next(iter(manager.prefix_cache.blocks.values()))
    assert block.token_ids == (1, 2, 3, 4)
    assert isinstance(block.payload, QuestPrefixBlockPayload)
    assert block.payload.block_slot == 9
    assert block.payload.token_slots.tolist() == [36, 37, 38, 39]
    assert manager._num_free_pages == 9


def test_quest_no_prefix_graph_publishes_page_boundary_in_graph_phase():
    manager = _make_quest_manager_for_no_prefix_graph(page_size=4)
    seq = Sequence([1, 2, 3, 4])
    manager._allocate(seq.seq_id, 4)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(5)
    row = manager.seq_id_to_row[seq.seq_id]

    contract = DecodeGraphContract(
        method="quest",
        topology_path_id="unified",
        batch_capacity=2,
        context_capacity=16,
    )
    inputs = DecodeGraphInputs.allocate(
        contract,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    state = manager.init_decode_graph_state(contract, inputs)
    assert isinstance(state, QuestDecodeGraphState)
    pointers = inputs.data_ptrs()

    manager.prepare_decode_graph_step([seq], state)

    reserved_slot = int(inputs.write_slot_mapping[0])
    reserved_page = reserved_slot // manager.page_size
    assert inputs.data_ptrs() == pointers
    assert inputs.active_mask.tolist() == [True, False]
    assert manager.buffer_req_to_page_slots[row, 1].item() == -1
    assert manager.buffer_req_to_token_slots[row, 4].item() == 0

    manager.prepare_decode_graph_in(state)

    assert manager.buffer_req_to_page_slots[row, 1].item() == reserved_page
    assert manager.buffer_req_to_token_slots[row, 4].item() == reserved_slot
    assert state.row_page_slots[:, :2].tolist() == [
        manager.buffer_req_to_page_slots[row, :2].tolist(),
        manager.buffer_req_to_page_slots[row, :2].tolist(),
    ]
    assert state.num_pages.tolist() == [2, 2]
    assert state.previous_page_counts.tolist() == [1, 1]


def test_quest_no_prefix_graph_capacity_failure_preserves_allocator():
    manager = _make_quest_manager_for_no_prefix_graph(page_size=4)
    seq = Sequence([1, 2, 3, 4])
    manager._allocate(seq.seq_id, 4)
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    seq.append_token(5)
    row = manager.seq_id_to_row[seq.seq_id]
    contract = DecodeGraphContract(
        method="quest",
        topology_path_id="unified",
        batch_capacity=1,
        context_capacity=4,
    )
    inputs = DecodeGraphInputs.allocate(
        contract,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    state = manager.init_decode_graph_state(contract, inputs)
    free_pages_before = manager._num_free_pages
    row_len_before = int(manager.row_seq_lens[row])
    page_table_before = manager.buffer_req_to_page_slots_cpu.copy()

    with pytest.raises(ValueError, match="exceeded the captured graph context"):
        manager.prepare_decode_graph_step([seq], state)

    assert manager._num_free_pages == free_pages_before
    assert int(manager.row_seq_lens[row]) == row_len_before
    assert np.array_equal(manager.buffer_req_to_page_slots_cpu, page_table_before)


def test_quest_decode_materialized_page_can_seed_later_prefix_hit():
    manager = _make_quest_manager_for_prefix(page_size=4)
    first = Sequence([1, 2, 3])
    prompt_slots = manager._allocate(first.seq_id, 3)
    manager._record_prefix_materialization(first, [1, 2, 3], prompt_slots)
    first.num_prefilled_tokens = first.num_prompt_tokens
    first.append_token(4)
    manager._prepare_decode([first])
    manager.on_forward_end([first], is_prefill=False)

    second = Sequence([1, 2, 3, 4, 5])
    manager.refresh_prefix_cache_hit(second)

    assert second.prefix_cache_hit_len == 4
    assert second.prefix_cache_hit_block_count == 1
    manager._attach_prefix_cache_if_needed(second)
    row_idx = manager.seq_id_to_row[second.seq_id]
    assert manager.row_seq_lens[row_idx] == 4
    assert manager.buffer_req_to_page_slots[row_idx, 0].item() == 9
    assert manager.buffer_req_to_token_slots[row_idx, :4].tolist() == [36, 37, 38, 39]
    req_indices = torch.tensor([row_idx], dtype=torch.int32)
    manager._prepare_decode_row_page_slots(req_indices, max_context_len=4)
    assert manager._decode_row_page_slots.tolist() == [[9]]
    packed_ptr = manager._decode_row_page_slots.data_ptr()
    manager._prepare_decode_row_page_slots(req_indices, max_context_len=4)
    assert manager._decode_row_page_slots.data_ptr() == packed_ptr
    block = manager.prefix_cache.get_block(second.prefix_cache_hit_last_block_id)
    assert block is not None
    assert block.ref_count == 2


def test_quest_reset_prefix_cache_clears_warmup_pages_and_restores_allocator():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2, 3, 4])
    slots = manager._allocate(seq.seq_id, 4)
    manager._record_prefix_materialization(seq, [1, 2, 3, 4], slots)
    manager.on_forward_end([seq], is_prefill=True)
    manager.free_seq(seq.seq_id)
    assert len(manager.prefix_cache) == 2
    assert manager._num_free_pages == 8

    manager.reset_prefix_cache()

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_pages == 10
    assert manager.free_pages_stack[:10].tolist() == list(range(10))


def test_quest_reset_after_warmup_restores_allocator_without_prefix_cache():
    manager = _make_quest_manager_for_prefix(page_size=2)
    manager.enable_prefix_caching = False
    manager.prefix_cache = None
    manager.free_pages_stack[:10] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], dtype=torch.int32)

    manager.reset_after_warmup()

    assert manager._num_free_pages == 10
    assert manager.free_pages_stack[:10].tolist() == list(range(10))


def test_quest_reset_after_warmup_clears_prefix_cache_and_allocator():
    manager = _make_quest_manager_for_prefix(page_size=2)
    seq = Sequence([1, 2, 3, 4])
    slots = manager._allocate(seq.seq_id, 4)
    manager._record_prefix_materialization(seq, [1, 2, 3, 4], slots)
    manager.on_forward_end([seq], is_prefill=True)
    manager.free_seq(seq.seq_id)

    manager.reset_after_warmup()

    assert len(manager.prefix_cache) == 0
    assert manager._num_free_pages == 10
    assert manager.free_pages_stack[:10].tolist() == list(range(10))


def test_quest_safe_delete_releases_payload_block_slot():
    manager = _make_quest_manager_for_prefix(page_size=2)
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    _remove_free_page(manager, 5)
    manager.prefix_cache.insert_block(block)
    assert manager._num_free_pages == 9

    result = manager.prefix_cache_delete_subtree([1, 2])

    assert result["deleted_block_ids"] == [stable_block_id.hex()]
    assert manager._num_free_pages == 10
    assert stable_block_id not in manager.prefix_cache.blocks


def test_quest_admission_is_page_aligned_and_reserves_hit_pages():
    manager = _make_quest_manager_for_prefix(page_size=2)
    manager._num_free_pages = 1
    seq = Sequence([1, 2, 3])
    stable_block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=stable_block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    _remove_free_page(manager, 5)
    manager.prefix_cache.insert_block(block)
    seq.prefix_cache_hit_len = 2
    seq.prefix_cache_hit_block_count = 1
    seq.prefix_cache_hit_last_block_id = stable_block_id

    assert manager.prompt_admission_free_slots() == 4
    assert manager.prompt_admission_cost(seq) == 4

    manager.prefix_cache.acquire_block_ref(block)
    assert manager.prompt_admission_cost(seq) == 2


def test_quest_admission_counts_inflight_d2h_before_pressure_prompt():
    manager = _make_quest_manager_for_prefix(page_size=2)
    manager._num_free_pages = 1
    manager.prefix_offload_controller = _FakePrefixOffloadController(manager.prefix_cache)
    block_id = manager.prefix_cache.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=2,
        logical_block_idx=0,
        payload=QuestPrefixBlockPayload(
            block_slot=5,
            token_slots=torch.tensor([10, 11], dtype=torch.int32),
        ),
        token_ids=(1, 2),
    )
    manager.prefix_cache.insert_block(block)
    manager.prefix_cache.begin_d2h(block)
    pressure_seq = Sequence([7, 8, 9])

    assert manager._prefix_evictable_slots() == 0
    assert manager.prefill_step_free_slots() == 4
    assert manager.decode_step_free_slots() == 4
    assert manager.prompt_admission_cost(pressure_seq) == 4
    assert manager.prompt_admission_free_slots() == 4
    assert manager.prompt_admission_budgets(deque(), 2)["slots"] == 4
    hit_seq = Sequence([1, 2, 3])
    hit_seq.prefix_cache_hit_len = 2
    hit_seq.prefix_cache_hit_block_count = 1
    hit_seq.prefix_cache_hit_last_block_id = block_id
    assert manager.prompt_admission_cost(hit_seq) == 4

    manager.prefix_cache.acquire_block_ref(block)
    assert manager.prompt_admission_free_slots() == 2


def test_quest_admission_counts_cascade_freeable_prefix_pages():
    manager = _make_quest_manager_for_prefix(page_size=2)
    manager._num_free_pages = 0
    parent_block_id = None
    for logical_idx, start in enumerate(range(0, 6, 2)):
        token_ids = [start + 1, start + 2]
        stable_block_id = manager.prefix_cache.stable_block_id(token_ids, parent_block_id)
        manager.prefix_cache.insert_block(
            PrefixCacheBlock(
                stable_block_id=stable_block_id,
                parent_block_id=parent_block_id,
                block_size=2,
                logical_block_idx=logical_idx,
                payload=QuestPrefixBlockPayload(
                    block_slot=logical_idx,
                    token_slots=torch.tensor([logical_idx * 2, logical_idx * 2 + 1], dtype=torch.int32),
                ),
                token_ids=tuple(token_ids),
            )
        )
        parent_block_id = stable_block_id

    assert manager.prefix_cache.evictable_blocks() == 1
    assert manager.prompt_admission_free_slots() == 6


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.cuda),
])
def test_prefill_inputs_keep_values_across_inflight_chunks(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    manager = _make_standard_manager_for_prefix()
    manager.device = torch.device(device)
    manager.buffer_req_to_token_slots = manager.buffer_req_to_token_slots.to(device)
    manager.free_slots_stack = manager.free_slots_stack.to(device)
    seq = Sequence([11, 12, 13, 14, 15, 16])
    seq.current_chunk_size = 3
    first = manager._prepare_prefill([seq])
    first_context = manager.layer_batch_state.context_lens
    seq.num_prefilled_tokens = 3
    second = manager._prepare_prefill([seq])
    second_context = manager.layer_batch_state.context_lens
    # Independent expected prompt slices and logical positions, checked only
    # after both uploads are submitted to exercise staging-buffer ownership.
    for actual, expected in zip(first, [[11, 12, 13], [0, 1, 2], [0, 3]]):
        assert actual.cpu().tolist() == expected
    for actual, expected in zip(second, [[14, 15, 16], [3, 4, 5], [0, 3]]):
        assert actual.cpu().tolist() == expected
    assert first_context.cpu().tolist() == [3]
    assert second_context.cpu().tolist() == [6]


@pytest.mark.parametrize("block_size,method,device,keep", [
    (1, "", "cpu", [1, 4]), (2, "omnikv", "cpu", [1, 4]),
    (1, "omnikv", "cpu", []),
    pytest.param(
        1,
        "omnikv",
        "cuda",
        [1, 4],
        marks=(
            pytest.mark.cuda,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ),
])
def test_multi_range_prune_preserves_gaps_empty_blocks_and_logical_positions(block_size, method, device, keep):
    # A global budget can empty an entire selected block; gap KV must survive.
    manager = _make_standard_manager_for_prefix(block_size=block_size, method=method)
    manager.device = torch.device(device)
    manager.free_slots_stack = manager.free_slots_stack.to(device)
    manager.buffer_req_to_token_slots = manager.buffer_req_to_token_slots.to(device)
    tokens = list(range(10))
    blocks = []
    parent = None
    for start in range(0, 10, block_size):
        part = tokens[start:start + block_size]
        block_id = manager.prefix_cache.stable_block_id(part, parent)
        block = PrefixCacheBlock(
            stable_block_id=block_id, parent_block_id=parent, block_size=block_size,
            logical_block_idx=start // block_size, token_ids=tuple(part),
            payload=StandardPrefixBlockPayload(token_slots=torch.arange(start + 10, start + 10 + block_size, dtype=torch.int32, device=device)),
        )
        manager.prefix_cache.insert_block(block)
        blocks.append(block)
        parent = block_id
    _remove_free_slots(manager, list(range(10, 20)))
    before = manager.num_free_slots
    result = manager.prefix_cache_prune(
        tokens, ranges=[(8, 10), (0, 2), (4, 6)], keep_indices=torch.tensor(keep, dtype=torch.long, device=device),
        policy="kvzip_global", prune_id="union",
    )
    assert result["logical_tokens"] == 6
    assert result["freed_device_slots"] == 6 - len(keep)
    assert manager.num_free_slots == before + 6 - len(keep)
    assert len(manager.prefix_cache.blocks) == len(blocks)
    for block in blocks:
        start = block.logical_block_idx * block_size
        if start in (2, 3, 6, 7):
            assert block.payload.retained_offsets is None
    seq = Sequence(tokens + [10])
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 10
    seq.prefix_cache_hit_block_count = len(blocks)
    seq.prefix_cache_hit_last_block_id = parent
    seq.prefix_cache_block_size = block_size
    manager._attach_prefix_cache_if_needed(seq)
    row = manager.seq_id_to_row[seq.seq_id]
    assert manager.row_logical_lens[row] == 10
    expected_slots = [11, 12, 13, 16, 17, 18] if keep else [12, 13, 16, 17]
    assert manager.row_seq_lens[row] == len(expected_slots)
    assert manager.buffer_req_to_token_slots[row, :len(expected_slots)].tolist() == expected_slots


def test_multi_range_prune_late_invalid_payload_does_not_mutate_any_block():
    # Preparation must finish for all ranges before the first allocator mutation.
    manager = _make_standard_manager_for_prefix(block_size=1)
    parent = None
    blocks = []
    for token in range(5):
        block_id = manager.prefix_cache.stable_block_id([token], parent)
        block = PrefixCacheBlock(
            stable_block_id=block_id, parent_block_id=parent, block_size=1,
            logical_block_idx=token, token_ids=(token,),
            payload=StandardPrefixBlockPayload(token_slots=torch.tensor([token + 10], dtype=torch.int32)),
        )
        manager.prefix_cache.insert_block(block)
        blocks.append(block)
        parent = block_id
    blocks[-1].payload.token_slots = torch.empty(0, dtype=torch.int32)
    before = manager.num_free_slots
    with pytest.raises(RuntimeError, match="inconsistent Standard device payload"):
        manager.prefix_cache_prune(
            list(range(5)), ranges=[(0, 1), (2, 3), (4, 5)], keep_indices=torch.tensor([0]),
            policy="kvzip_global", prune_id="bad-last",
        )
    assert manager.num_free_slots == before
    assert all(block.payload.retained_offsets is None and block.prune_record is None for block in blocks)


@pytest.mark.parametrize('block_size', [1, 2])
def test_prune_new_interval_after_compacted_ancestor_and_score_mapping(block_size):
    # Repeated rounds preserve the first mask and map physical scores back to
    # original token positions even when an ancestor block has no resident KV.
    manager = _make_standard_manager_for_prefix(block_size=block_size)
    parent = None
    blocks = []
    tokens = list(range(10))
    for left in range(0, 10, block_size):
        block_id = manager.prefix_cache.stable_block_id(tokens[left:left+block_size], parent)
        block = PrefixCacheBlock(stable_block_id=block_id, parent_block_id=parent,
            block_size=block_size, logical_block_idx=left//block_size,
            token_ids=tuple(tokens[left:left+block_size]),
            payload=StandardPrefixBlockPayload(token_slots=torch.arange(left+10,left+10+block_size,dtype=torch.int32)))
        manager.prefix_cache.insert_block(block)
        blocks.append(block)
        parent = block_id
    _remove_free_slots(manager, list(range(10,20)))
    manager.prefix_cache_prune(tokens, ranges=[(0,4)], keep_indices=torch.tensor([1]),
                               policy='kvzip_global', prune_id='first')
    initial = [b.payload.token_slots.clone() for b in blocks[:4//block_size]]
    with pytest.raises(RuntimeError, match='already pruned'):
        manager.validate_prefix_cache_prune_target(tokens, ranges=[(2,6)])
    second = manager.prefix_cache_prune(tokens, ranges=[(6,10)], keep_indices=torch.tensor([0,3]),
                                       policy='kvzip_global', prune_id='second')
    assert second['freed_device_slots'] == 2
    for block, expected in zip(blocks, initial):
        torch.testing.assert_close(block.payload.token_slots, expected)
    seq = Sequence(tokens+[10,11])
    seq.prefix_cache_enabled = True
    seq.prefix_cache_hit_len = 10
    seq.prefix_cache_hit_block_count = len(blocks)
    seq.prefix_cache_hit_last_block_id = parent
    seq.prefix_cache_block_size = block_size
    manager._attach_prefix_cache_if_needed(seq)
    row = manager.seq_id_to_row[seq.seq_id]
    manager.row_seq_lens[row] += 2
    manager.row_logical_lens[row] += 2
    manager._prefix_prune_scoring = None
    manager.begin_prefix_prune_scoring(seq_id=seq.seq_id, candidate_start=6, query_start=10,query_end=12)
    request = manager.prefill_score_request(0,[seq])
    assert request.query_ranges == ((5,7),)
    assert request.candidate_start == 3
    physical_score = torch.arange(1,8,dtype=torch.float32)
    manager._prefix_prune_scoring['score'] = physical_score
    logical_score = manager.finish_prefix_prune_scoring()
    expected = torch.zeros(12)
    expected[[1,4,5,6,9,10,11]] = physical_score
    torch.testing.assert_close(logical_score,expected)
    manager.free_seq(seq.seq_id)


@pytest.mark.parametrize('consume', [False, True])
def test_prefill_reservation_evicts_idle_cache_preserves_pinned_prefix_and_cleans_failure(consume):
    # A full pool with reclaimable cache is admissible; the scoring target and
    # active requests must survive, including a failure before/after allocation.
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.free_slots_stack = torch.arange(6, dtype=torch.int32)
    manager._num_free_slots = 6
    blocks = []
    for tokens, slots in [([1,2], [0,1]), ([3,4], [2,3])]:
        block = PrefixCacheBlock(
            stable_block_id=manager.prefix_cache.stable_block_id(tokens, None),
            parent_block_id=None, block_size=2, logical_block_idx=0,
            token_ids=tuple(tokens),
            payload=StandardPrefixBlockPayload(token_slots=torch.tensor(slots, dtype=torch.int32)),
        )
        manager.prefix_cache.insert_block(block)
        blocks.append(block)
    _remove_free_slots(manager, [0,1,2,3])
    manager.prefix_cache.acquire_block_ref(blocks[0])
    live = manager._allocate(100, 2).clone()
    assert manager.num_free_slots == 0
    with pytest.raises(RuntimeError, match='injected failure'):
        with manager.reserve_prefill_slots([(-1,2),(-2,2)]) as count:
            assert count == 1  # Only one row remains; reserve a smaller batch.
            assert manager.num_free_slots == 0
            assert manager.prefix_cache.get_block(blocks[0].stable_block_id) is blocks[0]
            assert manager.prefix_cache.get_block(blocks[1].stable_block_id) is None
            if consume:
                allocated = manager._allocate(-1,2).clone()
                assert set(allocated.tolist()) == {2,3}
                assert not set(allocated.tolist()).intersection(live.tolist())
            raise RuntimeError('injected failure')
    assert manager.num_free_slots == 2
    assert set(manager.free_slots_stack[:2].tolist()) == {2,3}
    assert set(manager.seq_id_to_row) == {100}
    assert len(manager.free_rows) == 1
    assert blocks[0].ref_count == 1
    manager.prefix_cache.release_block_ref(blocks[0])
    manager.free_seq(100)
    assert manager.num_free_slots == 4


def test_prefill_reservation_true_exhaustion_does_not_claim_rows_or_slots():
    manager = _make_standard_manager_for_prefix()
    manager.free_slots_stack = torch.arange(2, dtype=torch.int32)
    manager._num_free_slots = 2
    manager._allocate(100,2)
    rows = list(manager.free_rows)
    with pytest.raises(RuntimeError, match='after prefix eviction'):
        with manager.reserve_prefill_slots([(-1,1)]):
            pytest.fail('exhausted reservation was admitted')
    assert list(manager.free_rows) == rows
    assert set(manager.seq_id_to_row) == {100}
    assert manager.num_free_slots == 0


def test_prefill_reservation_shrinks_after_exhausting_reclaimable_slots():
    manager = _make_standard_manager_for_prefix()
    manager.free_slots_stack = torch.arange(3, dtype=torch.int32)
    manager._num_free_slots = 3
    with manager.reserve_prefill_slots([(-1,2),(-2,2)]) as admitted:
        assert admitted == 1
        assert manager.num_free_slots == 1
        manager._allocate(-1,2)
    assert manager.num_free_slots == 3
    assert not manager.seq_id_to_row
    assert len(set(manager.free_slots_stack[:3].tolist())) == 3


@pytest.mark.parametrize('capacity,expected,failure', [
    (9, 0, False), (10, 1, False), (12, 2, False), (10, 1, True),
])
def test_prefill_reservation_accounts_for_shared_cpu_prefix_promotion(capacity, expected, failure):
    # Query-only admission used to consume promotion headroom. Shared prefixes
    # must cost once, and attach/forward failures must release temporary rows.
    manager = _make_standard_manager_for_prefix(block_size=2)
    manager.free_slots_stack = torch.arange(capacity, dtype=torch.int32)
    manager._num_free_slots = capacity
    manager.prefix_offload_controller = _FakePrefixOffloadController(manager.prefix_cache)
    cache = manager.prefix_cache
    blocks, parent = [], None
    for i in range(4):
        tokens = (2 * i, 2 * i + 1)
        block_id = cache.stable_block_id(list(tokens), parent)
        block = PrefixCacheBlock(
            stable_block_id=block_id, parent_block_id=parent, block_size=2,
            logical_block_idx=i, token_ids=tokens,
            payload=StandardPrefixBlockPayload(token_slots=None, host_block_index=i),
            residency=PrefixBlockResidency(device_present=False, host_present=True),
        )
        cache.insert_block(block)
        cache.acquire_block_ref(block)
        blocks.append(block)
        parent = block_id
    error = ('after prefix eviction' if expected == 0 else 'injected failure')
    guard = pytest.raises(RuntimeError, match=error) if expected == 0 or failure else nullcontext()
    try:
        with guard:
            with manager.reserve_prefill_slots(
                [(-1, 2), (-2, 2)], prefix_blocks={-1: blocks, -2: blocks},
            ) as admitted:
                assert admitted == expected
                for sid in [-1, -2][:admitted]:
                    seq = Sequence(list(range(10)))
                    seq.seq_id = sid
                    seq.prefix_cache_enabled = True
                    seq.prefix_cache_hit_len = 8
                    seq.prefix_cache_hit_last_block_id = parent
                    seq.prefix_cache_hit_block_count = 4
                    seq.prefix_cache_block_size = 2
                    manager._attach_prefix_cache_if_needed(seq)
                    # Simulate transport completion; allocation and ownership
                    # transitions above use the actual cache implementation.
                    for block in blocks:
                        if block.residency.transfer == PrefixTransferKind.H2D:
                            cache.finish_h2d(block)
                    if failure:
                        raise RuntimeError('injected failure')
                    manager._allocate(sid, 2)
        assert not manager.seq_id_to_row
        assert len(manager.free_rows) == 2
        assert all(block.ref_count == 1 for block in blocks)
        assert manager.num_free_slots == capacity - (8 if expected else 0)
        owned = manager.free_slots_stack[:manager.num_free_slots].tolist()
        owned += [slot for block in blocks if block.residency.device_present
                  for slot in block.payload.token_slots.tolist()]
        assert sorted(owned) == list(range(capacity))
    finally:
        for block in blocks:
            cache.release_block_ref(block)


@pytest.mark.parametrize('method', ['standard', 'quest', 'snapkv'])
def test_decode_row_exhaustion_does_not_partially_claim_a_batch(method):
    if method == 'quest':
        manager = _make_quest_manager_for_prefix()
    else:
        manager = _make_standard_manager_for_prefix()
    if method == 'snapkv':
        from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
        manager = object.__new__(SnapKVCacheManager)
        manager._num_free_slots = [10]
        manager.seq_id_to_row = [{}]
        manager.free_rows = [deque([0])]
        mapping, rows = manager.seq_id_to_row[0], manager.free_rows[0]
        allocate = lambda: manager._allocate_batch(0,[10,11],1)
    else:
        manager.free_rows = deque([0])
        mapping, rows = manager.seq_id_to_row, manager.free_rows
        allocate = lambda: manager._allocate_batch([10,11],1)
    with pytest.raises(RuntimeError, match='free rows'):
        allocate()
    assert not mapping
    assert list(rows) == [0]


class _FailingDecodeIndexCopy:
    def __getitem__(self, _index):
        return self

    def copy_(self, *_args, **_kwargs):
        raise RuntimeError("injected metadata copy failure")


class _FailingDecodeTableWrite:
    def __init__(self, tensor):
        self.tensor = tensor
        self.fail_next_write = True

    @property
    def shape(self):
        return self.tensor.shape

    def __getitem__(self, index):
        return self.tensor[index]

    def __setitem__(self, index, value):
        if self.fail_next_write:
            self.fail_next_write = False
            raise RuntimeError("injected metadata scatter failure")
        self.tensor[index] = value


@pytest.mark.parametrize("method", ["standard", "snapkv"])
@pytest.mark.parametrize("failure", ["copy", "scatter"])
@pytest.mark.parametrize("fence_fails", [False, True])
def test_decode_metadata_failure_restores_rows_and_slots(
    method, failure, fence_fails, monkeypatch,
):
    # Row-exhaustion tests fail before ownership changes; inject after the slot
    # and row claims to cover copy/scatter transaction rollback.
    if method == "standard":
        manager = _make_standard_manager_for_prefix()
        manager.row_logical_lens = manager.row_seq_lens.copy()
        capacity = manager.num_free_slots
        mapping = manager.seq_id_to_row
        rows = manager.free_rows
        lengths = manager.row_seq_lens
        table = manager.buffer_req_to_token_slots
        stack = manager.free_slots_stack
        free_count = lambda: manager.num_free_slots
        allocate = lambda: manager._allocate_batch([10], 1)
    else:
        from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager

        manager = object.__new__(SnapKVCacheManager)
        manager.device = torch.device("cpu")
        manager.max_model_len = 4
        manager.seq_id_to_row = [{}]
        manager.free_rows = [deque([0, 1])]
        manager.row_seq_lens = [np.zeros(2, dtype=np.int32)]
        manager.buffer_req_to_token_slots = [torch.zeros((2, 4), dtype=torch.int32)]
        manager.free_slots_stack = [torch.arange(8, dtype=torch.int32)]
        manager._num_free_slots = [8]
        capacity = 8
        mapping = manager.seq_id_to_row[0]
        rows = manager.free_rows[0]
        lengths = manager.row_seq_lens[0]
        table = manager.buffer_req_to_token_slots[0]
        stack = manager.free_slots_stack[0]
        free_count = lambda: manager._num_free_slots[0]
        allocate = lambda: manager._allocate_batch(0, [10], 1)

    manager._decode_buf_capacity = 64
    manager._static_rows_gpu = (
        _FailingDecodeIndexCopy()
        if failure == "copy"
        else torch.empty(64, dtype=torch.long)
    )
    manager._static_cols_gpu = torch.empty(64, dtype=torch.long)
    if failure == "scatter":
        failing_table = _FailingDecodeTableWrite(table)
        if method == "standard":
            manager.buffer_req_to_token_slots = failing_table
        else:
            manager.buffer_req_to_token_slots[0] = failing_table
    if fence_fails:
        from sparseengine.platforms import device_runtime

        monkeypatch.setattr(device_runtime, "supports_streams", lambda _device: True)
        monkeypatch.setattr(
            device_runtime,
            "synchronize",
            lambda: (_ for _ in ()).throw(RuntimeError("injected fence failure")),
        )
    rows_before = list(rows)

    with pytest.raises(RuntimeError, match=f"injected metadata {failure} failure"):
        allocate()

    if fence_fails:
        assert free_count() == capacity - 1
        assert len(manager._quarantined_decode_allocations) == 1
        quarantined_slots = manager._quarantined_decode_allocations[0][-1]
        assert quarantined_slots.tolist() == [capacity - 1]
        return

    assert not mapping
    assert list(rows) == rows_before
    assert free_count() == capacity
    assert np.all(lengths == 0)
    free_ids = stack[:free_count()].tolist()
    owned_ids = [
        int(slot)
        for row_idx in mapping.values()
        for slot in table[row_idx, :int(lengths[row_idx])].tolist()
    ]
    assert sorted(free_ids + owned_ids) == list(range(capacity))


def test_static_decode_stale_row_plan_returns_slots_without_partial_row_commit():
    manager = _make_standard_manager_for_prefix()
    rows = list(manager.free_rows)
    before = manager.num_free_slots
    with pytest.raises(RuntimeError, match='plan changed'):
        manager._allocate_decode_batch_static([10,11],
            row_indices=np.array([0,1],dtype=np.int64), pending_rows=((10,0),(11,999)))
    assert not manager.seq_id_to_row
    assert list(manager.free_rows) == rows
    assert manager.num_free_slots == before
    assert len(set(manager.free_slots_stack[:before].tolist())) == before


@pytest.mark.parametrize('method', ['snapkv', 'quest', 'deltakv_full', 'deltakv_raw'])
@pytest.mark.parametrize('existing', [False, True])
def test_single_allocation_overflow_preserves_rows_and_pool(method, existing):
    # Invalid prefill lengths previously claimed empty rows or consumed slots
    # before failing; appending to a live row must preserve its contents too.
    mapping = {10: 0} if existing else {}
    rows = deque([1] if existing else [0, 1])
    lengths = np.array([3 if existing else 0, 0], dtype=np.int32)
    slots = torch.full((2, 4), -1, dtype=torch.int32)
    free_stack = torch.arange(12, dtype=torch.int32)
    size = 2 if existing else 5
    if method == 'snapkv':
        from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
        manager = object.__new__(SnapKVCacheManager)
        manager.max_model_len = 4
        manager.seq_id_to_row = [mapping]
        manager.free_rows = [rows]
        manager.row_seq_lens = [lengths]
        manager.buffer_req_to_token_slots = [slots]
        manager._num_free_slots = [12]
        manager.free_slots_stack = [free_stack]
        allocate = lambda: manager._allocate(0, 10, size)
        free_count = lambda: manager._num_free_slots[0]
    elif method == 'quest':
        manager = _make_quest_manager_for_prefix()
        manager.max_model_len = 4
        manager.seq_id_to_row = mapping
        manager.free_rows = rows
        manager.row_seq_lens = lengths
        manager.buffer_req_to_token_slots = slots
        manager._evict_prefix_cache_until_free = lambda _: pytest.fail('invalid request attempted eviction')
        allocate = lambda: manager._allocate(10, size)
        free_count = lambda: manager._num_free_pages
    else:
        from sparseengine.engine.cache_manager.methods.deltakv_base import DeltaKVCacheManager
        manager = object.__new__(DeltaKVCacheManager)
        manager.device = torch.device('cpu')
        manager.seq_id_to_row = mapping
        manager.free_rows = rows
        manager.row_seq_lens = lengths
        manager.full_layer_slots_map = slots
        manager.sparse_layer_raw_slots_map = slots
        manager._num_free_slots_full = 12
        manager._num_free_slots_deltakv_full = 12
        manager.free_slots_stack_full = free_stack
        manager.free_slots_stack_deltakv_full = free_stack
        manager._deltakv_temp_full_reserve = 0
        manager._deltakv_static_temp_slots_reserved_total = 0
        if method == 'deltakv_full':
            allocate = lambda: manager._allocate_full(10, size)
            free_count = lambda: manager._num_free_slots_full
        else:
            allocate = lambda: manager._allocate_deltakv_full(10, size)
            free_count = lambda: manager._num_free_slots_deltakv_full
    before = (dict(mapping), list(rows), lengths.copy(), free_count(), slots.clone(), free_stack.clone())
    with pytest.raises(RuntimeError, match='max_model_len|row capacity'):
        allocate()
    assert mapping == before[0]
    assert list(rows) == before[1]
    np.testing.assert_array_equal(lengths, before[2])
    assert free_count() == before[3]
    torch.testing.assert_close(slots, before[4])
    torch.testing.assert_close(free_stack, before[5])


def test_quest_single_allocation_without_rows_does_not_evict():
    manager = _make_quest_manager_for_prefix()
    manager.free_rows.clear()
    manager.seq_id_to_row = {10: 0, 11: 1}
    manager._evict_prefix_cache_until_free = lambda _: pytest.fail('row exhaustion attempted eviction')
    before = manager._num_free_pages
    with pytest.raises(RuntimeError, match='free rows'):
        manager._allocate(12, 1)
    assert manager.seq_id_to_row == {10: 0, 11: 1}
    assert not manager.free_rows
    assert manager._num_free_pages == before


@pytest.mark.parametrize('raw', [False, True])
@pytest.mark.parametrize('exhaustion', ['rows', 'length'])
def test_deltakv_decode_rejection_preserves_earlier_rows(raw, exhaustion):
    from sparseengine.engine.cache_manager.methods.deltakv_base import DeltaKVCacheManager
    manager = object.__new__(DeltaKVCacheManager)
    manager.device = torch.device('cpu')
    manager.seq_id_to_row = {11: 1} if exhaustion == 'length' else {}
    manager.free_rows = deque([0])
    manager.row_seq_lens = np.array([0, 4], dtype=np.int32)
    manager.full_layer_slots_map = torch.full((2, 4), -1, dtype=torch.int32)
    manager.sparse_layer_raw_slots_map = manager.full_layer_slots_map.clone()
    manager._num_free_slots_full = 8
    manager._num_free_slots_deltakv_full = 8
    manager._deltakv_temp_full_reserve = 0
    manager._deltakv_static_temp_slots_reserved_total = 0
    manager.free_slots_stack_full = torch.arange(8, dtype=torch.int32)
    manager.free_slots_stack_deltakv_full = torch.arange(8, dtype=torch.int32)
    before = dict(manager.seq_id_to_row)
    allocate = manager._allocate_batch_deltakv_full if raw else manager._allocate_batch_full
    with pytest.raises(RuntimeError, match='free rows' if exhaustion == 'rows' else 'max_model_len'):
        allocate([10, 11], 1)
    assert manager.seq_id_to_row == before
    assert list(manager.free_rows) == [0]
    assert manager._num_free_slots_full == manager._num_free_slots_deltakv_full == 8
    assert torch.all(manager.full_layer_slots_map == -1)
    assert torch.all(manager.sparse_layer_raw_slots_map == -1)


@pytest.mark.parametrize('staging_capacity', [2, 8])
def test_deltakv_staging_overflow_does_not_claim_row_or_advance_cursor(staging_capacity):
    from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager
    manager = object.__new__(DeltaKVLessMemoryCacheManager)
    manager.device = torch.device('cpu')
    manager.seq_id_to_row = {}
    manager.free_rows = deque([0])
    manager.row_seq_lens = np.zeros(1, dtype=np.int32)
    manager.full_layer_slots_map = torch.full((1, 2), -1, dtype=torch.int32)
    manager.deltakv_prefill_staging_num_slots = staging_capacity
    manager._deltakv_less_memory_full_prefill_staging_offset = 0
    manager._deltakv_less_memory_prepare_seqs = [SimpleNamespace(seq_id=10)]
    manager._should_stage_full_layer_kivi_prefill = lambda seq, size: True
    manager._should_use_long_prefill_offload_staging = lambda seqs: False
    with pytest.raises(RuntimeError, match='capacity is too small|max_model_len'):
        manager._allocate_full(10, 3)
    assert manager.seq_id_to_row == {}
    assert list(manager.free_rows) == [0]
    assert manager._deltakv_less_memory_full_prefill_staging_offset == 0
    assert torch.all(manager.full_layer_slots_map == -1)


def test_standard_old_capacity_view_does_not_cache_absent_block_weight():
    manager = _make_standard_manager_for_prefix(block_size=2)
    index = manager.prefix_cache
    block_id = _insert_tokens(index, [1, 2])
    snapshot = index.freeable_block_ids()
    assert manager._prefix_resident_slots_for_ids(snapshot) == 2
    index.evict_until_freeable(1)
    assert manager._prefix_resident_slots_for_ids(snapshot) == 0
    assert _insert_tokens(index, [1, 2]) == block_id
    assert manager._prefix_resident_slots_for_ids(snapshot) == 2


def test_quest_capacity_reuses_weights_across_reference_changes():
    manager = _make_quest_manager_for_prefix(page_size=2)
    index = manager.prefix_cache
    block_id = index.stable_block_id([1, 2], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id, parent_block_id=None, block_size=2,
        logical_block_idx=0, token_ids=(1, 2),
        payload=QuestPrefixBlockPayload(
            block_slot=3, token_slots=torch.tensor([6, 7], dtype=torch.int32),
        ),
    )
    index.insert_block(block)
    with patch.object(manager, "_prefix_block_capacity_weight",
                      wraps=manager._prefix_block_capacity_weight) as weight:
        assert manager._prefix_evictable_slots() == 2
        for _ in range(3):
            index.acquire_block_ref(block)
            assert manager._prefix_evictable_slots() == 0
            index.release_block_ref(block)
            assert manager._prefix_evictable_slots() == 2
        assert weight.call_count == 1
        assert manager._prefix_step_reclaimable_pages() == 1
        assert weight.call_count == 1
    index.begin_d2h(block)
    index.finish_d2h(block)
    index.demote_device_until_freeable(1)
    assert manager._prefix_evictable_slots() == 0
    index.begin_h2d(block)
    index.finish_h2d(block)
    assert manager._prefix_evictable_slots() == 2
