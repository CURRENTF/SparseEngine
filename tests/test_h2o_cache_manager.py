from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import numpy as np
import pytest
import torch

from sparseengine.engine.cache_manager.base import (
    AttentionViewMeta,
    CacheManager,
    DecodeComputeView,
    ExplicitKVPayload,
    LayerBatchStates,
    MlaLatentPayload,
    PrefillComputeView,
)
from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.cache_manager.storage import MlaLatentStorage
from sparseengine.engine.decode_graph_contract import (
    DecodeGraphContract,
    DecodeGraphInputs,
)
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.scheduler import Scheduler
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import SparseController
from sparseengine.engine.sparse_methods import SparseStepContext
from sparseengine.engine.sparse_methods.h2o import H2ORuntime
from sparseengine.method_registry import (
    PREFILL_POLICY_ALL_CHUNKED,
)
from sparseengine.utils.context import reset_context, set_context


@pytest.fixture(autouse=True)
def _reset_runtime_context():
    reset_context()
    yield
    reset_context()


def _layout(layers: int = 1):
    return SimpleNamespace(
        kv_idx_to_layer_idx=tuple(range(layers)),
        kv_layer_index=lambda layer: int(layer),
        is_full_attention=lambda layer: 0 <= int(layer) < layers,
    )


def _manager_with_layer_rows(
    lengths_by_layer: list[list[int]],
    *,
    decode_budget=4,
    decode_eviction_interval=3,
    prefill_budget=8,
    engine_prefill_chunk_size=4,
    validate_runtime_invariants=False,
):
    if not lengths_by_layer:
        raise ValueError("lengths_by_layer must not be empty")
    batch_size = len(lengths_by_layer[0])
    if any(len(lengths) != batch_size for lengths in lengths_by_layer):
        raise ValueError("all layers must contain the same number of rows")

    manager = object.__new__(H2OCacheManager)
    manager.device = torch.device("cpu")
    manager.num_layers = len(lengths_by_layer)
    manager.num_kv_layers = len(lengths_by_layer)
    manager.runtime_layout = _layout(manager.num_layers)
    manager.config = SimpleNamespace(
        sparse_method="h2o",
        prefill_sparse_method="h2o_prefill",
        h2o_decode_budget=decode_budget,
        h2o_decode_eviction_interval=decode_eviction_interval,
        h2o_prefill_budget=prefill_budget,
        h2o_recent_ratio=0.5,
        h2o_prefill_score_window=4,
        obs_layer_ids=[0],
        sparse_prefill_score_mode="probability",
        max_model_len=64,
        snapkv_window_size=4,
        snapkv_num_full_layers=0,
        pyramid_layer_ratios=None,
        prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        engine_prefill_chunk_size=engine_prefill_chunk_size,
        validate_runtime_invariants=validate_runtime_invariants,
    )
    manager.validate_runtime_invariants = bool(validate_runtime_invariants)
    manager.max_model_len = 64
    manager.num_kv_heads = 2
    manager.head_dim = 2
    manager.hf_config = SimpleNamespace(dtype=torch.float32)
    manager._h2o_scores = {}
    manager._h2o_active_decode_seq_ids = set()
    manager._h2o_counters = {
        "intermediate_prefill_evictions": 0,
        "final_prefill_evictions": 0,
        "decode_eviction_bursts": 0,
        "decode_evictions": 0,
        "dropped_tokens": 0,
    }
    manager._h2o_final_prefill_workspace = None
    manager._uniform_decode_metadata = False
    manager.seq_id_to_row = [
        {idx: idx for idx in range(batch_size)}
        for _ in lengths_by_layer
    ]
    manager.free_rows = [deque() for _ in lengths_by_layer]
    manager.row_seq_lens = [
        np.asarray(lengths, dtype=np.int32) for lengths in lengths_by_layer
    ]
    manager.buffer_req_to_token_slots_tensor = torch.zeros(
        (manager.num_layers, batch_size, 64), dtype=torch.int32
    )
    manager.buffer_req_to_token_slots = [
        manager.buffer_req_to_token_slots_tensor[layer_idx]
        for layer_idx in range(manager.num_layers)
    ]
    for layer_idx, lengths in enumerate(lengths_by_layer):
        for row, length in enumerate(lengths):
            slot_start = row * 100
            manager.buffer_req_to_token_slots[layer_idx][row, :length] = torch.arange(
                slot_start, slot_start + length, dtype=torch.int32
            )
    manager.free_slots_stack_tensor = None
    manager.free_slots_stack = [
        torch.zeros((512,), dtype=torch.int32) for _ in lengths_by_layer
    ]
    manager._num_free_slots = [32 for _ in lengths_by_layer]
    num_slots = max(4096, batch_size * 100 + 64)
    manager.config.num_kvcache_slots = num_slots
    manager.kv_cache = torch.zeros(
        (
            2,
            manager.num_layers,
            num_slots,
            manager.num_kv_heads,
            manager.head_dim,
        ),
        dtype=manager.hf_config.dtype,
    )
    return manager


def _manager_with_rows(
    lengths: list[int],
    *,
    decode_budget=4,
    decode_eviction_interval=3,
    prefill_budget=8,
    engine_prefill_chunk_size=4,
):
    return _manager_with_layer_rows(
        [lengths],
        decode_budget=decode_budget,
        decode_eviction_interval=decode_eviction_interval,
        prefill_budget=prefill_budget,
        engine_prefill_chunk_size=engine_prefill_chunk_size,
    )


def _set_scores_from_slot_rows(manager: H2OCacheManager):
    for layer_idx in manager.kv_transformer_layer_indices():
        for seq_id, row_idx in manager.seq_id_to_row[layer_idx].items():
            row_len = int(manager.row_seq_lens[layer_idx][row_idx])
            manager._h2o_scores[(layer_idx, seq_id)] = (
                manager.buffer_req_to_token_slots[layer_idx][row_idx, :row_len]
                .float()
                .clone()
            )


def _set_layer_row_slots(
    manager: H2OCacheManager,
    layer_idx: int,
    rows: list[list[int]],
):
    for row_idx, slots in enumerate(rows):
        row_len = int(manager.row_seq_lens[layer_idx][row_idx])
        assert len(slots) == row_len
        manager.buffer_req_to_token_slots[layer_idx][row_idx, :row_len] = torch.tensor(
            slots,
            dtype=torch.int32,
        )


def _fill_kv_by_physical_slot(manager: H2OCacheManager):
    for layer_idx in manager.kv_transformer_layer_indices():
        k_cache, v_cache = manager.get_layer_kv_cache(layer_idx)
        slot_values = torch.arange(k_cache.shape[0], dtype=k_cache.dtype).view(-1, 1, 1)
        offsets = torch.arange(
            manager.num_kv_heads * manager.head_dim,
            dtype=k_cache.dtype,
        ).view(1, manager.num_kv_heads, manager.head_dim)
        k_cache.copy_(slot_values * 10 + offsets)
        v_cache.copy_(-slot_values * 10 - offsets - 1)


def _use_kv_layout(manager: H2OCacheManager, layout: str):
    if layout == "tensor":
        return
    if layout != "list":
        raise ValueError(f"unknown KV layout: {layout}")
    manager.kv_cache = [
        (
            manager.kv_cache[0, layer_idx].clone(),
            manager.kv_cache[1, layer_idx].clone(),
        )
        for layer_idx in manager.kv_transformer_layer_indices()
    ]


def _assert_scores_match_slot_rows(manager: H2OCacheManager):
    for layer_idx in manager.kv_transformer_layer_indices():
        for seq_id, row_idx in manager.seq_id_to_row[layer_idx].items():
            row_len = int(manager.row_seq_lens[layer_idx][row_idx])
            assert torch.equal(
                manager._h2o_scores[(layer_idx, seq_id)],
                manager.buffer_req_to_token_slots[layer_idx][row_idx, :row_len].float(),
            )


def _append_decode_token(
    manager: H2OCacheManager,
    layer_idx: int,
    seq_id: int,
    *,
    token_slot: int,
    token_score: float,
    append_score: bool = True,
):
    row_idx = manager.seq_id_to_row[layer_idx][seq_id]
    row_len = int(manager.row_seq_lens[layer_idx][row_idx])
    score = manager._h2o_scores[(layer_idx, seq_id)]
    assert int(score.numel()) == row_len
    manager.buffer_req_to_token_slots[layer_idx][row_idx, row_len] = int(token_slot)
    manager.row_seq_lens[layer_idx][row_idx] = row_len + 1
    if append_score:
        manager._h2o_scores[(layer_idx, seq_id)] = torch.cat(
            (score, torch.tensor([token_score], dtype=torch.float32))
        )


def _token_score_map(manager: H2OCacheManager, layer_idx: int, seq_id: int):
    row_idx = manager.seq_id_to_row[layer_idx][seq_id]
    row_len = int(manager.row_seq_lens[layer_idx][row_idx])
    slots = manager.buffer_req_to_token_slots[layer_idx][row_idx, :row_len].tolist()
    scores = manager._h2o_scores[(layer_idx, seq_id)].tolist()
    assert len(slots) == len(scores)
    assert len(slots) == len(set(slots))
    return {int(slot): float(score) for slot, score in zip(slots, scores)}


def _seq(seq_id: int, prompt_len: int, *, prefilled: int, chunk: int) -> Sequence:
    seq = Sequence(list(range(prompt_len)))
    seq.seq_id = int(seq_id)
    seq.num_prefilled_tokens = int(prefilled)
    seq.current_chunk_size = int(chunk)
    return seq


def test_h2o_decode_does_not_request_scores_or_run_eviction():
    runtime = object.__new__(H2ORuntime)
    runtime.config = SimpleNamespace(
        sparse_method="h2o",
        sparse_prefill_score_mode="logits",
        h2o_prefill_score_window=0,
    )
    runtime.cache_manager = Mock()
    runtime.layer_batch_sparse_states = {
        0: SimpleNamespace(attn_score=None),
    }
    decode = SparseStepContext(
        seqs=[],
        is_prefill=False,
        forward_context=SimpleNamespace(is_prefill=False),
    )
    prefill = SparseStepContext(
        seqs=[],
        is_prefill=True,
        forward_context=SimpleNamespace(is_prefill=True),
    )

    assert runtime.needs_attention_score(0, decode) is False
    assert runtime.needs_attention_score(0, prefill) is True
    runtime.finish_step(decode)

    runtime.cache_manager.evict_after_decode.assert_not_called()


def test_h2o_flashprefill_uses_posthoc_scoring_and_preserves_eviction():
    runtime = object.__new__(H2ORuntime)
    runtime.config = SimpleNamespace(
        sparse_method="h2o",
        prefill_sparse_method="flashprefill_v2",
        sparse_prefill_score_mode="logits",
        h2o_prefill_score_window=0,
    )
    runtime.cache_manager = Mock()
    prefill = SparseStepContext(
        seqs=[],
        is_prefill=True,
        forward_context=SimpleNamespace(is_prefill=True),
    )

    assert runtime.needs_attention_score(0, prefill) is False
    runtime.finish_step(prefill)

    runtime.cache_manager.evict_after_intermediate_prefill.assert_not_called()
    runtime.cache_manager.compact_final_prefill_for_decode.assert_called_once_with([])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fusion", [False, True])
def test_mla_reduced_logits_accumulate_only_live_tokens_before_eviction(fusion):
    manager = _manager_with_layer_rows([[4, 3], [4, 3]], decode_budget=16)
    manager.device = torch.device("cuda")
    config = manager.config
    config.h2o_decode_eviction = True
    config.h2o_decode_score_fusion = fusion
    config.attention_cache_layout = "mla_latent"
    config.hf_config = SimpleNamespace(
        num_hidden_layers=2, hidden_size=8, num_attention_heads=2,
        qk_nope_head_dim=2, qk_rope_head_dim=2, dtype=torch.float32,
    )
    config.runtime_layout = manager.runtime_layout
    config.tensor_parallel_size = 1
    config.full_attention_layers = []
    config.sink_keep_tokens = config.recent_keep_tokens = 0
    config.decode_keep_tokens = 16
    config.sparse_attn_score_dtype = "float32"
    config.decode_graph = False
    runtime = H2ORuntime(config, manager)
    seqs = [_seq(i, length, prefilled=length, chunk=1)
            for i, length in enumerate([4, 3])]
    manager.evict_after_decode = Mock(wraps=manager.evict_after_decode)
    expected = {}
    for layer in range(2):
        for seq, length in zip(seqs, [4, 3]):
            expected[layer, seq.seq_id] = torch.arange(length - 1).float()
            manager._h2o_scores[layer, seq.seq_id] = expected[layer, seq.seq_id].to(manager.device)
    for lengths in ([4, 3], [5, 4]):
        scores = runtime._get_h2o_decode_score_buffer(2, 2, 8)
        assert scores.ndim == 3  # MLA cannot use the explicit-KV headwise buffer.
        scores.fill_(10000.0)  # Stale Graph padding must never enter softmax.
        for layer in range(2):
            runtime.layer_batch_sparse_states[layer].attn_score = scores[layer]
            runtime.layer_batch_sparse_states[layer].context_lens = torch.tensor(
                lengths, device=manager.device
            )
            manager.row_seq_lens[layer][:] = lengths
            for row, (seq, length) in enumerate(zip(seqs, lengths)):
                heads = torch.stack((torch.arange(length).float(),
                                     torch.arange(length).float().flip(0) + layer))
                scores[layer, row, :length] = heads.amax(0)
                probability = torch.softmax(heads.amax(0) / 2, dim=-1)
                expected[layer, seq.seq_id] = torch.cat(
                    (expected[layer, seq.seq_id], torch.zeros(1))
                ) + probability
        runtime._h2o_decode_eviction(seqs)
        for key, reference in expected.items():
            torch.testing.assert_close(manager._h2o_scores[key].cpu(), reference)
    assert manager.evict_after_decode.call_count == 2


@pytest.mark.parametrize("eviction", [False, True])
@pytest.mark.parametrize("fusion", [False, True])
def test_h2o_decode_switch_drives_score_update_before_physical_eviction(eviction, fusion):
    manager = _manager_with_layer_rows([[7, 6], [7, 6]])
    config = manager.config
    config.h2o_decode_eviction = eviction
    config.h2o_decode_score_fusion = fusion
    config.hf_config = SimpleNamespace(
        num_hidden_layers=2, hidden_size=4, num_attention_heads=2,
        head_dim=2, dtype=torch.float32,
    )
    config.runtime_layout = manager.runtime_layout
    config.tensor_parallel_size = 1
    config.full_attention_layers = []
    config.sink_keep_tokens = config.recent_keep_tokens = 0
    config.decode_keep_tokens = 4
    config.sparse_attn_score_dtype = "float32"
    config.decode_graph = False
    seqs = [_seq(i, length, prefilled=length, chunk=1) for i, length in enumerate([7, 6])]
    for layer in range(2):
        for seq, length in zip(seqs, [7, 6]):
            manager._h2o_scores[(layer, seq.seq_id)] = torch.zeros(length - 1)
    runtime = H2ORuntime(config, manager)
    manager.get_layer_batch_states = lambda _layer: LayerBatchStates(
        context_lens=torch.tensor([7, 6], dtype=torch.int32),
        max_context_len=7,
        req_indices=torch.tensor([0, 1], dtype=torch.int32),
    )
    before_slots = manager.buffer_req_to_token_slots_tensor.clone()
    before_free = list(manager._num_free_slots)
    step = SparseStepContext(seqs=seqs, is_prefill=False, forward_context=SimpleNamespace(is_prefill=False))
    runtime.prepare_step(step)
    for layer in range(2):
        score = runtime.layer_batch_sparse_states[layer].attn_score
        if not eviction:
            assert score is None
            continue
        assert score.shape == (2, 7)
        score.zero_()
        # The second token wins only if this step's scores are accumulated
        # before top-k; the two most recent tokens must remain protected.
        score[:, 1] = 0.9
        score[:, 0] = 0.1
    runtime.finish_step(step)
    if eviction:
        assert [row.tolist() for row in manager.row_seq_lens] == [[4, 6], [4, 6]]
        for layer in range(2):
            assert torch.equal(manager.buffer_req_to_token_slots_tensor[layer, 0, :4], before_slots[layer, 0, [0, 1, 5, 6]])
            assert manager._h2o_scores[(layer, 0)].tolist() == pytest.approx([0.1, 0.9, 0, 0])
            assert manager._h2o_scores[(layer, 1)].tolist() == pytest.approx([0.1, 0.9, 0, 0, 0, 0])
            assert manager._num_free_slots[layer] == before_free[layer] + 3
    else:
        assert torch.equal(manager.buffer_req_to_token_slots_tensor, before_slots)
        assert manager._num_free_slots == before_free
        assert runtime.decode_graph_keepalive_tensors() == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h2o_headwise_accumulation_preserves_eviction_and_reordered_history():
    """Catch scores revived/misaligned after real slot eviction and batch reorder."""
    torch.manual_seed(61)
    manager = _manager_with_layer_rows([[7, 6], [7, 6]], validate_runtime_invariants=True)
    manager.device = torch.device("cuda")
    manager.kv_cache = manager.kv_cache.cuda()
    manager.buffer_req_to_token_slots_tensor = manager.buffer_req_to_token_slots_tensor.cuda()
    manager.buffer_req_to_token_slots = list(manager.buffer_req_to_token_slots_tensor.unbind(0))
    manager.free_slots_stack = [row.cuda() for row in manager.free_slots_stack]
    seqs = [_seq(i, length, prefilled=length, chunk=1) for i, length in enumerate([7, 6])]
    expected = {}
    for layer in range(2):
        for seq, length in zip(seqs, [7, 6]):
            initial = torch.rand(length - 1, device="cuda")
            manager._h2o_scores[(layer, seq.seq_id)] = initial.clone()
            expected[(layer, seq.seq_id)] = initial.clone()
    for step in range(3):
        logits = torch.randn(2, 2, 5, 12, device="cuda") * 5
        for layer in range(2):
            for batch, seq in enumerate(seqs):
                length = manager._physical_row_len(layer, seq)
                logits[layer, batch, :, length:] = torch.nan
                key = (layer, seq.seq_id)
                expected[key] = torch.nn.functional.pad(expected[key], (0, 1)) + (
                    logits[layer, batch, :, :length] * 0.25
                ).softmax(-1).sum(0)
        manager.accumulate_decode_headwise_logits([0, 1], seqs, logits, softmax_scale=0.25)
        for key, reference in expected.items():
            torch.testing.assert_close(manager._h2o_scores[key], reference, atol=2e-6, rtol=2e-6)
        if step == 0:
            before = manager.buffer_req_to_token_slots_tensor.clone()
            manager.evict_after_decode(seqs)
            for layer in range(2):
                retained = manager.buffer_req_to_token_slots_tensor[layer, 0, :4]
                positions = (retained[:, None] == before[layer, 0, :7][None, :]).nonzero()[:, 1]
                expected[(layer, 0)] = expected[(layer, 0)][positions]
                torch.testing.assert_close(manager._h2o_scores[(layer, 0)], expected[(layer, 0)])
            assert manager._h2o_decode_score_signature is None
        if step == 1:
            seqs = list(reversed(seqs))
        for layer in range(2):
            manager.row_seq_lens[layer] += 1


@pytest.mark.parametrize(
    ("sparse_method", "prefill_sparse_method", "intermediate", "final"),
    [
        ("", "h2o_prefill", True, False),
        ("h2o", "", False, True),
        ("h2o", "h2o_prefill", True, True),
        ("", "", False, False),
    ],
)
def test_h2o_runtime_triggers_prefill_and_decode_boundaries_independently(
    sparse_method,
    prefill_sparse_method,
    intermediate,
    final,
):
    runtime = object.__new__(H2ORuntime)
    runtime.config = SimpleNamespace(
        sparse_method=sparse_method,
        prefill_sparse_method=prefill_sparse_method,
    )
    runtime.cache_manager = Mock()
    step = SparseStepContext(
        seqs=[],
        is_prefill=True,
        forward_context=SimpleNamespace(is_prefill=True),
    )

    runtime.finish_step(step)

    assert (
        runtime.cache_manager.evict_after_intermediate_prefill.called
        is intermediate
    )
    assert runtime.cache_manager.compact_final_prefill_for_decode.called is final


def test_h2o_cache_manager_factory_routes_first_class_method():
    expected = object()
    config = SimpleNamespace(
        sparse_method="h2o",
        hf_config=SimpleNamespace(model_type="qwen2"),
    )
    with patch(
        "sparseengine.engine.cache_manager.methods.h2o.H2OCacheManager",
        return_value=expected,
    ) as constructor:
        actual = CacheManager.create(config, SimpleNamespace())
    assert actual is expected
    constructor.assert_called_once_with(config, ANY)


def test_h2o_cache_manager_factory_routes_prefill_only_method():
    expected = object()
    config = SimpleNamespace(
        sparse_method="",
        prefill_sparse_method="h2o_prefill",
        hf_config=SimpleNamespace(model_type="qwen2"),
    )
    with patch(
        "sparseengine.engine.cache_manager.methods.h2o.H2OCacheManager",
        return_value=expected,
    ) as constructor:
        actual = CacheManager.create(config, SimpleNamespace())
    assert actual is expected
    constructor.assert_called_once_with(config, ANY)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"budget": 0, "recent_ratio": 0.5}, "budget must be positive"),
        ({"budget": 4, "recent_ratio": 0.0}, "recent_ratio must be"),
        ({"budget": 4, "recent_ratio": 1.0}, "recent_ratio must be"),
    ],
)
def test_h2o_selection_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        H2OCacheManager.select_h2o_indices(torch.arange(8, dtype=torch.float32), **kwargs)


def test_h2o_selection_keeps_heavy_and_recent_in_logical_order():
    scores = torch.tensor([1.0, 9.0, 2.0, 8.0, 3.0, 0.0, 0.0, 0.0])
    keep = H2OCacheManager.select_h2o_indices(scores, budget=4, recent_ratio=0.5)
    assert keep.tolist() == [1, 3, 6, 7]


def test_h2o_general_batch_selection_matches_scalar_multi_drop():
    scores = torch.tensor(
        [
            [9.0, 1.0, 7.0, 3.0, 5.0, 11.0, 13.0],
            [2.0, 10.0, 4.0, 8.0, 6.0, 12.0, 14.0],
            [15.0, 19.0, 17.0, 21.0, 16.0, 23.0, 25.0],
            [30.0, 26.0, 28.0, 24.0, 22.0, 27.0, 29.0],
        ]
    )

    batched = H2OCacheManager.select_h2o_indices_batch(
        scores, budget=4, recent_ratio=0.5
    )
    scalar = torch.stack(
        [
            H2OCacheManager.select_h2o_indices(
                scores[batch_idx],
                budget=4,
                recent_ratio=0.5,
            )
            for batch_idx in range(scores.shape[0])
        ]
    )

    assert torch.equal(batched, scalar)


def test_h2o_score_vector_expands_adds_and_gathers_with_physical_row():
    manager = _manager_with_rows([6])
    seq = _seq(0, 20, prefilled=0, chunk=6)
    manager._h2o_scores[(0, 0)] = torch.arange(6, dtype=torch.float32)

    manager.free_part_slots(0, seq, torch.tensor([1, 3, 4, 5]))

    assert manager.row_seq_lens[0].tolist() == [4]
    assert manager._h2o_scores[(0, 0)].tolist() == [1.0, 3.0, 4.0, 5.0]
    accumulated = manager._accumulate_score(
        manager._h2o_scores[(0, 0)],
        torch.tensor([0.25, 0.25, 0.25, 0.25, 0.5, 1.0]),
        new_len=6,
        weight=2.0,
    )
    assert accumulated.tolist() == [1.5, 3.5, 4.5, 5.5, 1.0, 2.0]


def test_h2o_prefill_score_ranges_use_compressed_physical_coordinates():
    manager = _manager_with_rows([11])
    seq = _seq(0, 100, prefilled=64, chunk=3)

    ranges = manager.prefill_score_ranges(0, [seq])

    assert ranges[0][2:] == (8, 8, 11)


def test_h2o_logit_prefill_score_window_zero_covers_full_current_chunk():
    manager = _manager_with_rows([11])
    manager.config.sparse_prefill_score_mode = "logits"
    manager.config.h2o_prefill_score_window = 0
    seq = _seq(0, 100, prefilled=64, chunk=6)

    ranges = manager.prefill_score_ranges(0, [seq])

    assert ranges[0][2:] == (5, 5, 11)


def test_h2o_logit_prefill_score_is_normalized_before_weighted_accumulation():
    logits = torch.tensor([0.0, 1.0, 2.0])
    normalized = H2OCacheManager._normalize_logit_prefill_score(logits, new_len=3)
    cumulative = H2OCacheManager._accumulate_score(
        torch.tensor([1.0, 2.0]),
        normalized,
        new_len=3,
        weight=4.0,
    )

    assert normalized.sum().item() == pytest.approx(1.0)
    assert torch.equal(normalized, torch.softmax(logits, dim=0))
    assert torch.equal(
        cumulative,
        torch.tensor([1.0, 2.0, 0.0]) + 4.0 * torch.softmax(logits, dim=0),
    )


def test_h2o_logit_prefill_score_converts_unscored_minus_inf_to_zero_prob():
    logits = torch.tensor([0.0, -torch.inf, 2.0])
    normalized = H2OCacheManager._normalize_logit_prefill_score(logits, new_len=3)
    assert normalized[1].item() == 0.0
    assert torch.isfinite(normalized).all()
    assert normalized.sum().item() == pytest.approx(1.0)


def test_h2o_logit_prefill_score_rejects_nan_or_all_inf():
    with pytest.raises(RuntimeError, match="invalid non-finite values"):
        H2OCacheManager._normalize_logit_prefill_score(
            torch.tensor([float("nan"), 1.0]),
            new_len=2,
        )
    with pytest.raises(RuntimeError, match="invalid non-finite values"):
        H2OCacheManager._normalize_logit_prefill_score(
            torch.tensor([-torch.inf, -torch.inf]),
            new_len=2,
        )


@pytest.mark.parametrize("precomputed", [False, True])
def test_h2o_prefill_score_collection_accumulates_in_physical_coordinates(precomputed):
    manager = _manager_with_rows([6])
    seq = _seq(0, 20, prefilled=8, chunk=2)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    view = PrefillComputeView(
        meta=AttentionViewMeta(
            active_slots=manager.buffer_req_to_token_slots[0],
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([6], dtype=torch.int32),
            max_context_len=6,
        ),
        payload=ExplicitKVPayload(
            k_cache=torch.empty((16, 1, 1)),
            v_cache=torch.empty((16, 1, 1)),
        ),
    )
    if precomputed:
        # The latent operator hands off scores; the manager still owns weighted
        # accumulation in physical coordinates after previous KV compaction.
        view = PrefillComputeView(
            view.meta, MlaLatentPayload(torch.empty(16, 1, 512), torch.empty(16, 1, 64)),
            token_scores=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        )
    request = manager.prefill_score_request(0, [seq])
    assert request.query_ranges == ((4, 6),)
    assert request.candidate_start == request.recent_keep_tokens == 0
    set_context(is_prefill=True, cache_manager=manager, seqs=[seq])

    def fake_run_prefill_score(
        q,
        k_cache,
        attn_score,
        meta,
        b_start_loc,
        prompt_cache_lens,
        max_query_len,
        score_starts,
        score_ends,
        **kwargs,
    ):
        del q, k_cache, meta, b_start_loc, max_query_len
        assert prompt_cache_lens.tolist() == [4]
        assert score_starts.tolist() == [4]
        assert score_ends.tolist() == [6]
        assert kwargs == {"candidate_start": 0, "recent_keep_tokens": 0}
        attn_score[0, :6] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    with patch.object(
        manager,
        "_run_prefill_score",
        side_effect=fake_run_prefill_score,
    ) as scorer:
        manager.collect_prefill_attention_score(
            0,
            torch.empty((2, 1, 1)),
            view,
            b_start_loc=torch.tensor([0], dtype=torch.int32),
            chunk_lens=torch.tensor([2], dtype=torch.int32),
        )

    assert scorer.call_count == (0 if precomputed else 1)
    assert manager._h2o_scores[(0, 0)].tolist() == pytest.approx(
        [1.2, 2.4, 3.6, 4.8, 1.0, 1.2]
    )


def test_h2o_logit_prefill_score_collection_normalizes_logits():
    manager = _manager_with_rows([6])
    manager.config.sparse_prefill_score_mode = "logits"
    manager.config.h2o_prefill_score_window = 0
    seq = _seq(0, 20, prefilled=8, chunk=2)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    view = PrefillComputeView(
        meta=AttentionViewMeta(
            active_slots=manager.buffer_req_to_token_slots[0],
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([6], dtype=torch.int32),
            max_context_len=6,
        ),
        payload=ExplicitKVPayload(
            k_cache=torch.empty((16, 1, 1)),
            v_cache=torch.empty((16, 1, 1)),
        ),
    )
    set_context(is_prefill=True, cache_manager=manager, seqs=[seq])
    logits = torch.arange(6, dtype=torch.float32)

    def fake_run_prefill_score(*args, **kwargs):
        del kwargs
        args[2][0, :6].copy_(logits)

    with patch.object(
        manager,
        "_run_prefill_score",
        side_effect=fake_run_prefill_score,
    ):
        manager.collect_prefill_attention_score(
            0,
            torch.empty((2, 1, 1)),
            view,
            b_start_loc=torch.tensor([0], dtype=torch.int32),
            chunk_lens=torch.tensor([2], dtype=torch.int32),
        )

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
    expected.add_(torch.softmax(logits, dim=0), alpha=2.0)
    assert torch.equal(manager._h2o_scores[(0, 0)], expected)


def test_h2o_logit_prefill_score_collection_consumes_fused_main_score():
    manager = _manager_with_rows([6])
    manager.config.sparse_prefill_score_mode = "logits"
    manager.config.h2o_prefill_score_window = 0
    seq = _seq(0, 20, prefilled=8, chunk=2)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    logits = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    view = PrefillComputeView(
        meta=AttentionViewMeta(
            active_slots=manager.buffer_req_to_token_slots[0],
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([6], dtype=torch.int32),
            max_context_len=6,
            attn_score=logits,
        ),
        payload=ExplicitKVPayload(
            k_cache=torch.empty((16, 1, 1)),
            v_cache=torch.empty((16, 1, 1)),
        ),
    )
    set_context(is_prefill=True, cache_manager=manager, seqs=[seq])

    with patch.object(
        manager,
        "_run_prefill_score",
        side_effect=AssertionError("launched posthoc scorer"),
    ):
        manager.collect_prefill_attention_score(
            0,
            torch.empty((2, 1, 1)),
            view,
            b_start_loc=torch.tensor([0], dtype=torch.int32),
            chunk_lens=torch.tensor([2], dtype=torch.int32),
        )

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
    expected.add_(torch.softmax(logits[0], dim=0), alpha=2.0)
    assert torch.equal(manager._h2o_scores[(0, 0)], expected)


def test_h2o_prefill_score_collection_rejects_misaligned_physical_view():
    manager = _manager_with_rows([6])
    seq = _seq(0, 20, prefilled=8, chunk=2)
    manager._h2o_scores[(0, 0)] = torch.ones(4)
    view = PrefillComputeView(
        meta=AttentionViewMeta(
            active_slots=manager.buffer_req_to_token_slots[0],
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([7], dtype=torch.int32),
            max_context_len=7,
        ),
        payload=ExplicitKVPayload(
            k_cache=torch.empty((16, 1, 1)),
            v_cache=torch.empty((16, 1, 1)),
        ),
    )
    set_context(is_prefill=True, cache_manager=manager, seqs=[seq])

    with pytest.raises(RuntimeError, match="compressed physical coordinates"):
        manager.collect_prefill_attention_score(
            0,
            torch.empty((2, 1, 1)),
            view,
            b_start_loc=torch.tensor([0], dtype=torch.int32),
            chunk_lens=torch.tensor([2], dtype=torch.int32),
        )


def test_h2o_missing_score_for_existing_physical_prefix_fails_fast():
    manager = _manager_with_rows([8])
    seq = _seq(0, 100, prefilled=64, chunk=3)
    with pytest.raises(RuntimeError, match="score vector is missing"):
        manager._require_score_length(0, seq, 8)


def test_h2o_intermediate_and_final_prefill_use_distinct_budgets_and_counters():
    manager = _manager_with_rows([10], decode_budget=4, prefill_budget=8)
    seq = _seq(0, 20, prefilled=0, chunk=10)
    manager._h2o_scores[(0, 0)] = torch.arange(10, dtype=torch.float32)

    manager.evict_after_prefill([seq])
    assert manager.row_seq_lens[0][0] == 8
    assert manager._h2o_counters["intermediate_prefill_evictions"] == 1

    manager.buffer_req_to_token_slots[0][0, 8:10] = torch.tensor([20, 21])
    manager.row_seq_lens[0][0] = 10
    manager._h2o_scores[(0, 0)] = manager._expand_score(
        manager._h2o_scores[(0, 0)], 10, device=manager.device
    )
    seq.num_prefilled_tokens = 10
    seq.current_chunk_size = 10
    manager.evict_after_prefill([seq])

    assert manager.row_seq_lens[0][0] == 4
    assert manager._h2o_counters["final_prefill_evictions"] == 1
    assert manager._h2o_counters["dropped_tokens"] == 8


def test_h2o_decode_score_update_supports_batch_with_different_kv_lengths():
    manager = _manager_with_rows([4, 3])
    seq0 = _seq(0, 10, prefilled=10, chunk=1)
    seq1 = _seq(1, 10, prefilled=10, chunk=1)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0, 3.0])
    manager._h2o_scores[(0, 1)] = torch.tensor([4.0, 5.0])
    normalized = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, -100.0]], dtype=torch.float32
    )

    manager.update_decode_attention_scores(0, [seq0, seq1], normalized)

    assert manager._h2o_scores[(0, 0)].tolist() == pytest.approx([1.1, 2.2, 3.3, 0.4])
    assert manager._h2o_scores[(0, 1)].tolist() == pytest.approx([4.5, 5.6, 0.7])


def test_h2o_all_layer_score_does_not_alias_graph_scratch_between_bursts():
    manager = _manager_with_layer_rows(
        [[6], [6]], decode_budget=4, decode_eviction_interval=3
    )
    seq = _seq(0, 10, prefilled=10, chunk=1)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    manager._h2o_scores[(1, 0)] = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0])
    graph_scratch = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1e20]],
            [[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -1e20]],
        ],
        dtype=torch.float32,
    )

    used_fast_path = manager.update_decode_attention_scores_all_layers(
        [0, 1],
        [seq],
        graph_scratch,
    )
    expected = {
        (0, 0): torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5, 0.6]),
        (1, 0): torch.tensor([6.6, 7.5, 8.4, 9.3, 10.2, 0.1]),
    }
    assert used_fast_path
    graph_scratch.fill_(-1e20)

    for key, score in expected.items():
        assert torch.allclose(manager._h2o_scores[key], score)
        assert manager._h2o_scores[key].untyped_storage().data_ptr() != (
            graph_scratch.untyped_storage().data_ptr()
        )


def test_h2o_eager_and_replayed_score_scratch_match_across_burst_boundary():
    def run(*, reuse_scratch: bool):
        manager = _manager_with_rows(
            [4], decode_budget=4, decode_eviction_interval=3, prefill_budget=8
        )
        seq = _seq(0, 7, prefilled=7, chunk=1)
        manager._h2o_scores[(0, 0)] = torch.tensor([10.0, 1.0, 5.0, 0.5])
        scratch = torch.empty((1, 1, 64), dtype=torch.float32)
        for token_slot in (4, 5, 6):
            _append_decode_token(
                manager,
                0,
                0,
                token_slot=token_slot,
                token_score=0.0,
                append_score=False,
            )
            if not reuse_scratch:
                scratch = torch.empty((1, 1, 64), dtype=torch.float32)
            scratch.fill_(-1e20)
            kv_len = int(manager.row_seq_lens[0][0])
            scratch[0, 0, :kv_len] = torch.linspace(
                0.1, 0.1 * kv_len, kv_len
            )
            manager.update_decode_attention_scores_all_layers(
                [0], [seq], scratch
            )
            manager.evict_after_decode([seq])
            scratch.fill_(-1e20)
        return (
            manager.buffer_req_to_token_slots[0][0, :4].clone(),
            manager._h2o_scores[(0, 0)].clone(),
            dict(manager._h2o_counters),
        )

    eager = run(reuse_scratch=False)
    replayed = run(reuse_scratch=True)

    assert torch.equal(eager[0], replayed[0])
    assert torch.allclose(eager[1], replayed[1])
    assert eager[2] == replayed[2]


def test_h2o_decode_score_update_batches_uniform_lengths():
    manager = _manager_with_rows([4, 4])
    seq0 = _seq(0, 10, prefilled=10, chunk=1)
    seq1 = _seq(1, 10, prefilled=10, chunk=1)
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 2.0, 3.0])
    manager._h2o_scores[(0, 1)] = torch.tensor([4.0, 5.0, 6.0])
    normalized = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32
    )

    with patch.object(
        H2OCacheManager,
        "_accumulate_score",
        side_effect=AssertionError("uniform score update used scalar fallback"),
    ):
        manager.update_decode_attention_scores(0, [seq0, seq1], normalized)

    assert manager._h2o_scores[(0, 0)].tolist() == pytest.approx([1.1, 2.2, 3.3, 0.4])
    assert manager._h2o_scores[(0, 1)].tolist() == pytest.approx([4.5, 5.6, 6.7, 0.8])


def test_h2o_all_layer_score_update_batches_uniform_rows_once():
    manager = _manager_with_layer_rows([[4, 4], [4, 4]])
    seqs = [
        _seq(0, 10, prefilled=10, chunk=1),
        _seq(1, 10, prefilled=10, chunk=1),
    ]
    previous = {}
    for layer_idx in range(2):
        for seq_id in range(2):
            score = torch.tensor(
                [
                    10.0 * layer_idx + 3.0 * seq_id + 1.0,
                    10.0 * layer_idx + 3.0 * seq_id + 2.0,
                    10.0 * layer_idx + 3.0 * seq_id + 3.0,
                ]
            )
            manager._h2o_scores[(layer_idx, seq_id)] = score
            previous[(layer_idx, seq_id)] = score.clone()
    reduced = torch.arange(2 * 2 * 5, dtype=torch.float32).view(2, 2, 5) / 10.0
    expected = reduced[:, :, :4].clone()
    expected[:, :, :3] += torch.stack(
        [previous[(layer_idx, seq_id)] for layer_idx in range(2) for seq_id in range(2)]
    ).view(2, 2, 3)

    with patch.object(
        manager,
        "update_decode_attention_scores",
        side_effect=AssertionError("uniform all-layer update used per-layer fallback"),
    ):
        used_fast_path = manager.update_decode_attention_scores_all_layers(
            [0, 1], seqs, reduced
        )

    assert used_fast_path
    for layer_idx in range(2):
        for seq_id in range(2):
            assert torch.equal(
                manager._h2o_scores[(layer_idx, seq_id)],
                expected[layer_idx, seq_id],
            )
            assert torch.equal(
                previous[(layer_idx, seq_id)],
                torch.tensor(
                    [
                        10.0 * layer_idx + 3.0 * seq_id + 1.0,
                        10.0 * layer_idx + 3.0 * seq_id + 2.0,
                        10.0 * layer_idx + 3.0 * seq_id + 3.0,
                    ]
                ),
            )
    storage_ptrs = {
        score.untyped_storage().data_ptr() for score in manager._h2o_scores.values()
    }
    assert len(storage_ptrs) == 1


def test_h2o_all_layer_score_update_reuses_persistent_workspace():
    manager = _manager_with_layer_rows([[4, 4], [4, 4]])
    seqs = [
        _seq(0, 10, prefilled=10, chunk=1),
        _seq(1, 10, prefilled=10, chunk=1),
    ]
    for layer_idx in range(2):
        for seq_id in range(2):
            manager._h2o_scores[(layer_idx, seq_id)] = torch.ones(3)

    first = torch.full((2, 2, 4), 0.25)
    manager.update_decode_attention_scores_all_layers([0, 1], seqs, first)
    workspace = manager._h2o_decode_score_workspace

    for layer_idx in range(2):
        manager.row_seq_lens[layer_idx][:2] = 5
    second = torch.full((2, 2, 5), 0.5)
    with patch("torch.stack", side_effect=AssertionError("rebuilt stable score rows")):
        manager.update_decode_attention_scores_all_layers([0, 1], seqs, second)

    assert manager._h2o_decode_score_workspace is workspace
    expected = torch.tensor([1.75, 1.75, 1.75, 0.75, 0.5])
    for layer_idx in range(2):
        for seq_id in range(2):
            assert torch.equal(manager._h2o_scores[(layer_idx, seq_id)], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lengths", [[[1, 17], [9, 33]], [[17, 17], [9, 9]]])
@pytest.mark.parametrize("validate", [False, True])
def test_h2o_mla_ragged_logits_fuse_updates_and_reuse_history(lengths, validate):
    """Catch scalar fallback, padding in softmax, and stale history after batch reorder.

    Unlike the probability-only tests, this executes the MLA normalization
    boundary and its real CUDA kernel, including lengths that differ by layer.
    """
    torch.manual_seed(719)
    manager = _manager_with_layer_rows(
        lengths, decode_budget=48, validate_runtime_invariants=validate
    )
    seqs = [_seq(i, 40, prefilled=40, chunk=1) for i in range(2)]
    for layer in range(2):
        for seq in seqs:
            length = int(manager.row_seq_lens[layer][seq.seq_id])
            manager._h2o_scores[(layer, seq.seq_id)] = torch.rand(length - 1, device="cuda")

    # Graph score views can have spare batch capacity and non-contiguous strides.
    logits = torch.empty(2, 4, 258, device="cuda")[:, :2, ::2]
    scale = 128**-0.5
    row_pointers = None
    for step in range(3):
        if step:
            for row_lengths in manager.row_seq_lens:
                row_lengths += 1
        if step == 2:
            seqs.reverse()
        logits.normal_()
        expected = {}
        for layer in range(2):
            for batch, seq in enumerate(seqs):
                length = int(manager.row_seq_lens[layer][seq.seq_id])
                key = (layer, seq.seq_id)
                expected[key] = torch.nn.functional.pad(manager._h2o_scores[key], (0, 1))
                expected[key] += torch.softmax(logits[layer, batch, :length] * scale, -1)
                logits[layer, batch, length:] = torch.nan

        with (
            patch("torch.softmax", side_effect=AssertionError("per-row softmax fallback")),
            patch.object(manager, "update_decode_attention_scores",
                         side_effect=AssertionError("per-layer accumulation fallback")),
        ):
            assert manager.update_decode_attention_scores_all_layers(
                [0, 1], seqs, logits, normalize_logits=True, softmax_scale=scale
            )
        current_pointers = {key: row.data_ptr() for key, row in manager._h2o_scores.items()}
        if row_pointers is not None:
            assert current_pointers == row_pointers
        row_pointers = current_pointers
        logits.fill_(torch.nan)
        for key, reference in expected.items():
            torch.testing.assert_close(manager._h2o_scores[key], reference, atol=1e-6, rtol=1e-5)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h2o_mla_logits_preserve_scores_across_compacted_row_shapes():
    """Catch incompatible workspace reuse when eviction changes uniform/ragged rows."""
    torch.manual_seed(720)
    manager = _manager_with_layer_rows([[6, 4], [5, 3]], decode_budget=16)
    seqs = [_seq(i, 10, prefilled=10, chunk=1) for i in range(2)]
    expected = {}
    for layer in range(2):
        for seq in seqs:
            length = int(manager.row_seq_lens[layer][seq.seq_id])
            key = (layer, seq.seq_id)
            expected[key] = torch.rand(length - 1, device="cuda")
            manager._h2o_scores[key] = expected[key].clone()

    for lengths in ([[6, 4], [5, 3]], [[3, 3], [3, 3]], [[4, 2], [3, 4]]):
        manager._invalidate_h2o_decode_score_workspace()
        logits = torch.randn(2, 2, 16, device="cuda")
        for layer in range(2):
            manager.row_seq_lens[layer][:] = lengths[layer]
            for seq in seqs:
                key = (layer, seq.seq_id)
                length = lengths[layer][seq.seq_id]
                # Retained scores stay in physical KV order after compaction.
                retained = expected[key][:length - 1].flip(0).clone()
                manager._h2o_scores[key] = retained.clone()
                expected[key] = torch.nn.functional.pad(retained, (0, 1))
                expected[key] += torch.softmax(logits[layer, seq.seq_id, :length] * 0.25, -1)
                logits[layer, seq.seq_id, length:] = torch.nan
        assert manager.update_decode_attention_scores_all_layers(
            [0, 1], seqs, logits, normalize_logits=True, softmax_scale=0.25
        )
        for key, reference in expected.items():
            torch.testing.assert_close(manager._h2o_scores[key], reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("lengths", [[[4, 0], [4, 3]], [[4, 3], [4, 9]]])
def test_h2o_mla_ragged_logits_reject_invalid_rows_before_mutation(lengths):
    """Reject empty or uncovered later-layer rows before any score is changed."""
    manager = _manager_with_layer_rows(lengths)
    seqs = [_seq(i, 10, prefilled=10, chunk=1) for i in range(2)]
    for layer in range(2):
        for seq in seqs:
            manager._h2o_scores[(layer, seq.seq_id)] = torch.ones(max(0, lengths[layer][seq.seq_id] - 1))
    previous = {key: score.clone() for key, score in manager._h2o_scores.items()}
    with pytest.raises(ValueError, match="do not cover the physical rows"):
        manager.update_decode_attention_scores_all_layers(
            [0, 1], seqs, torch.zeros(2, 2, 8), normalize_logits=True, softmax_scale=0.25
        )
    for key, reference in previous.items():
        torch.testing.assert_close(manager._h2o_scores[key], reference)


def test_h2o_all_layer_probability_update_uses_per_layer_path_for_nonuniform_rows():
    manager = _manager_with_layer_rows([[4, 3], [4, 3]])
    seqs = [
        _seq(0, 10, prefilled=10, chunk=1),
        _seq(1, 10, prefilled=10, chunk=1),
    ]
    for layer_idx in range(2):
        manager._h2o_scores[(layer_idx, 0)] = torch.tensor([1.0, 2.0, 3.0])
        manager._h2o_scores[(layer_idx, 1)] = torch.tensor([4.0, 5.0])
    reduced = torch.ones((2, 2, 4), dtype=torch.float32)
    original = manager.update_decode_attention_scores

    with patch.object(
        manager,
        "update_decode_attention_scores",
        wraps=original,
    ) as per_layer:
        used_fast_path = manager.update_decode_attention_scores_all_layers(
            [0, 1], seqs, reduced
        )

    assert not used_fast_path
    assert per_layer.call_count == 2
    for layer_idx in range(2):
        assert manager._h2o_scores[(layer_idx, 0)].tolist() == [2.0, 3.0, 4.0, 1.0]
        assert manager._h2o_scores[(layer_idx, 1)].tolist() == [5.0, 6.0, 1.0]


def test_h2o_decode_burst_compacts_all_layers_and_batch_once():
    manager = _manager_with_layer_rows(
        [[7, 7] for _ in range(28)],
        decode_budget=4,
        decode_eviction_interval=3,
        prefill_budget=8,
    )
    seqs = [
        _seq(0, 7, prefilled=7, chunk=1),
        _seq(1, 7, prefilled=7, chunk=1),
    ]
    _set_scores_from_slot_rows(manager)
    calls = []
    original = SnapKVCacheManager.free_part_slots_batch_layers

    def tracked_compact(self, layer_indices, batch_seqs, keep_indices, **kwargs):
        calls.append((list(layer_indices), list(batch_seqs), keep_indices.clone()))
        return original(self, layer_indices, batch_seqs, keep_indices, **kwargs)

    with patch.object(SnapKVCacheManager, "free_part_slots_batch_layers", tracked_compact):
        manager.evict_after_decode(seqs)

    assert len(calls) == 1
    assert calls[0][0] == list(range(28))
    assert calls[0][1] == seqs
    assert calls[0][2].shape == (28, 2, 4)
    assert [lengths.tolist() for lengths in manager.row_seq_lens] == [
        [4, 4] for _ in range(28)
    ]
    _assert_scores_match_slot_rows(manager)
    assert manager._num_free_slots == [38 for _ in range(28)]
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 0,
        "final_prefill_evictions": 0,
        "decode_eviction_bursts": 2,
        "decode_evictions": 56,
        "dropped_tokens": 168,
    }


def test_h2o_intermediate_prefill_reuses_uniform_batch_fast_path():
    manager = _manager_with_layer_rows(
        [[6, 6], [7, 7]], decode_budget=3, prefill_budget=4
    )
    seqs = [
        _seq(0, 20, prefilled=0, chunk=6),
        _seq(1, 20, prefilled=0, chunk=6),
    ]
    _set_scores_from_slot_rows(manager)
    calls = []
    original = SnapKVCacheManager.free_part_slots_batch_layers

    def tracked_batch_free(self, layer_indices, batch_seqs, keep_indices, **kwargs):
        calls.append((list(layer_indices), keep_indices.clone()))
        return original(self, layer_indices, batch_seqs, keep_indices, **kwargs)

    with patch.object(
        SnapKVCacheManager,
        "free_part_slots_batch_layers",
        new=tracked_batch_free,
    ):
        assert manager._try_batched_evict(seqs, is_prefill=True)

    assert len(calls) == 1
    assert calls[0][0] == [0, 1]
    assert calls[0][1].shape == (2, 2, 4)
    assert [lengths.tolist() for lengths in manager.row_seq_lens] == [[4, 4], [4, 4]]
    _assert_scores_match_slot_rows(manager)
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 4,
        "final_prefill_evictions": 0,
        "decode_eviction_bursts": 0,
        "decode_evictions": 0,
        "dropped_tokens": 10,
    }


def test_h2o_mixed_prefill_budgets_fall_back_with_exact_counters():
    manager = _manager_with_rows([6, 6], decode_budget=3, prefill_budget=4)
    seqs = [
        _seq(0, 6, prefilled=0, chunk=6),
        _seq(1, 20, prefilled=0, chunk=6),
    ]
    _set_scores_from_slot_rows(manager)
    _fill_kv_by_physical_slot(manager)
    k_cache, v_cache = manager.get_layer_kv_cache(0)
    selected_slots = torch.tensor([3, 4, 5], dtype=torch.long)
    expected_k = k_cache.index_select(0, selected_slots).clone()
    expected_v = v_cache.index_select(0, selected_slots).clone()

    assert not manager._try_batched_evict(seqs, is_prefill=True)
    manager.evict_after_prefill(seqs)

    assert manager.row_seq_lens[0].tolist() == [3, 4]
    assert manager._h2o_scores[(0, 0)].tolist() == [3.0, 4.0, 5.0]
    assert manager._h2o_scores[(0, 1)].tolist() == [102.0, 103.0, 104.0, 105.0]
    final_slots = manager.buffer_req_to_token_slots[0][0, :3].long()
    assert final_slots.tolist() == [0, 1, 2]
    assert torch.equal(k_cache.index_select(0, final_slots), expected_k)
    assert torch.equal(v_cache.index_select(0, final_slots), expected_v)
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 1,
        "final_prefill_evictions": 1,
        "decode_eviction_bursts": 0,
        "decode_evictions": 0,
        "dropped_tokens": 5,
    }


@pytest.mark.parametrize("kv_layout", ["tensor", "list"])
def test_h2o_final_prefill_page_table_compaction_preserves_logical_kv_alignment(
    kv_layout: str,
):
    manager = _manager_with_layer_rows(
        [[6, 6], [6, 6]], decode_budget=4, prefill_budget=8
    )
    rows_by_layer = [
        [[9, 2, 7, 1, 6, 4], [29, 22, 27, 21, 26, 24]],
        [[109, 102, 107, 101, 106, 104], [129, 122, 127, 121, 126, 124]],
    ]
    for layer_idx, rows in enumerate(rows_by_layer):
        _set_layer_row_slots(manager, layer_idx, rows)
        for seq_id in range(2):
            manager._h2o_scores[(layer_idx, seq_id)] = torch.tensor(
                [1.0, 9.0, 2.0, 8.0, 0.0, 0.0]
            )
    _use_kv_layout(manager, kv_layout)
    _fill_kv_by_physical_slot(manager)
    seqs = [
        _seq(0, 6, prefilled=0, chunk=6),
        _seq(1, 6, prefilled=0, chunk=6),
    ]
    keep = torch.tensor([1, 3, 4, 5], dtype=torch.long)
    keep_set = set(keep.tolist())
    expected = {}
    for layer_idx in range(2):
        k_cache, v_cache = manager.get_layer_kv_cache(layer_idx)
        for seq_id, row_slots in enumerate(rows_by_layer[layer_idx]):
            selected_slots = torch.tensor(row_slots, dtype=torch.long)[keep]
            expected[(layer_idx, seq_id)] = (
                k_cache.index_select(0, selected_slots).clone(),
                v_cache.index_select(0, selected_slots).clone(),
            )

    manager.evict_after_prefill(seqs)

    for layer_idx, rows in enumerate(rows_by_layer):
        k_cache, v_cache = manager.get_layer_kv_cache(layer_idx)
        released = []
        active = []
        for seq_id, row_slots in enumerate(rows):
            destination = torch.tensor(row_slots, dtype=torch.long)[keep].tolist()
            released.extend(
                slot for idx, slot in enumerate(row_slots) if idx not in keep_set
            )
            active.extend(destination)
            actual_slots = manager.buffer_req_to_token_slots[layer_idx][
                seq_id, :4
            ].long()
            assert actual_slots.tolist() == destination
            expected_k, expected_v = expected[(layer_idx, seq_id)]
            assert torch.equal(k_cache.index_select(0, actual_slots), expected_k)
            assert torch.equal(v_cache.index_select(0, actual_slots), expected_v)
            assert manager._h2o_scores[(layer_idx, seq_id)].tolist() == [
                9.0,
                8.0,
                0.0,
                0.0,
            ]
        assert len(active) == len(set(active))
        assert manager.free_slots_stack[layer_idx][32:36].tolist() == released
        assert manager._num_free_slots[layer_idx] == 36

    assert manager._h2o_final_prefill_workspace is None


def test_h2o_final_prefill_compacts_mla_latent_and_rope_slots():
    manager = _manager_with_rows([6], decode_budget=4, prefill_budget=8)
    _set_layer_row_slots(manager, 0, [[9, 2, 7, 1, 6, 4]])
    manager._h2o_scores[(0, 0)] = torch.tensor(
        [1.0, 9.0, 2.0, 8.0, 0.0, 0.0]
    )
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=1, num_slots=64, device=torch.device("cpu"))
    manager.attention_cache_storage = storage
    manager.kv_cache = None
    assert storage.latent_cache is not None
    assert storage.rope_cache is not None
    for slot in [9, 2, 7, 1, 6, 4]:
        storage.latent_cache[0, slot].fill_(slot)
        storage.rope_cache[0, slot].fill_(slot + 100)
    seq = _seq(0, 6, prefilled=0, chunk=6)

    manager.evict_after_prefill([seq])

    destination_slots = manager.buffer_req_to_token_slots[0][0, :4].long()
    assert destination_slots.tolist() == [2, 1, 6, 4]
    payload = storage.layer_payload(0)
    expected_sources = torch.tensor([2, 1, 6, 4], dtype=torch.bfloat16)
    torch.testing.assert_close(
        payload.latent_cache[destination_slots, 0, 0],
        expected_sources,
    )
    torch.testing.assert_close(
        payload.rope_cache[destination_slots, 0, 0],
        expected_sources + 100,
    )
    assert manager._h2o_scores[(0, 0)].tolist() == [9.0, 8.0, 0.0, 0.0]
    assert manager._h2o_final_prefill_workspace is None


def test_h2o_intermediate_prefill_does_not_move_kv_payloads():
    manager = _manager_with_rows([6], decode_budget=3, prefill_budget=4)
    _set_layer_row_slots(manager, 0, [[9, 2, 7, 1, 6, 4]])
    manager._h2o_scores[(0, 0)] = torch.tensor([1.0, 9.0, 2.0, 8.0, 0.0, 0.0])
    _fill_kv_by_physical_slot(manager)
    k_cache, v_cache = manager.get_layer_kv_cache(0)
    old_k = k_cache.clone()
    old_v = v_cache.clone()
    seq = _seq(0, 20, prefilled=0, chunk=6)

    manager.evict_after_prefill([seq])

    assert manager.buffer_req_to_token_slots[0][0, :4].tolist() == [2, 1, 6, 4]
    assert torch.equal(k_cache, old_k)
    assert torch.equal(v_cache, old_v)
    assert manager._h2o_final_prefill_workspace is None


def test_h2o_final_prefill_capacity_preflight_prevents_partial_layer_updates():
    manager = _manager_with_layer_rows([[6], [6]], decode_budget=4, prefill_budget=8)
    _set_scores_from_slot_rows(manager)
    _fill_kv_by_physical_slot(manager)
    manager._num_free_slots[1] = 511
    layer0_slots = manager.buffer_req_to_token_slots[0].clone()
    layer0_k, layer0_v = manager.get_layer_kv_cache(0)
    expected_k = layer0_k.clone()
    expected_v = layer0_v.clone()
    seq = _seq(0, 6, prefilled=0, chunk=6)

    with pytest.raises(RuntimeError, match=r"overflow.*layer=1"):
        manager.evict_after_prefill([seq])

    assert torch.equal(manager.buffer_req_to_token_slots[0], layer0_slots)
    assert torch.equal(layer0_k, expected_k)
    assert torch.equal(layer0_v, expected_v)
    assert manager.row_seq_lens[0].tolist() == [6]
    assert manager._num_free_slots[0] == 32
    assert manager._h2o_final_prefill_workspace is None


def test_h2o_decode_waits_for_interval_then_drops_full_burst():
    manager = _manager_with_rows(
        [5], decode_budget=4, decode_eviction_interval=3, prefill_budget=8
    )
    seq = _seq(0, 7, prefilled=7, chunk=1)
    _set_scores_from_slot_rows(manager)

    manager.evict_after_decode([seq])
    assert manager.row_seq_lens[0].tolist() == [5]
    _append_decode_token(manager, 0, 0, token_slot=5, token_score=5.0)
    manager.evict_after_decode([seq])
    assert manager.row_seq_lens[0].tolist() == [6]
    assert manager._h2o_counters["decode_evictions"] == 0

    _append_decode_token(manager, 0, 0, token_slot=6, token_score=6.0)
    manager.evict_after_decode([seq])

    assert manager.row_seq_lens[0].tolist() == [4]
    assert manager.buffer_req_to_token_slots[0][0, :4].tolist() == [3, 4, 5, 6]
    assert manager._h2o_scores[(0, 0)].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert manager._h2o_counters["decode_eviction_bursts"] == 1
    assert manager._h2o_counters["decode_evictions"] == 1
    assert manager._h2o_counters["dropped_tokens"] == 3


def test_h2o_decode_pressure_reclaims_over_budget_row_before_interval():
    manager = _manager_with_rows(
        [5], decode_budget=4, decode_eviction_interval=3, prefill_budget=8
    )
    seq = _seq(0, 4, prefilled=4, chunk=1)
    seq.append_token(4)
    _set_scores_from_slot_rows(manager)
    manager._num_free_slots = [0]

    manager.evict_after_decode([seq])

    assert manager.row_seq_lens[0].tolist() == [4]
    assert manager.buffer_req_to_token_slots[0][0, :4].tolist() == [1, 2, 3, 4]
    assert manager._h2o_scores[(0, 0)].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert manager._num_free_slots == [1]
    assert manager.decode_step_free_slots() == 1
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 0,
        "final_prefill_evictions": 0,
        "decode_eviction_bursts": 1,
        "decode_evictions": 1,
        "dropped_tokens": 1,
    }

    scheduler_config = SimpleNamespace(
        max_num_seqs_in_batch=1,
        max_num_seqs_in_gpu=1,
        max_num_batched_tokens=1,
        max_decoding_seqs=1,
        engine_prefill_chunk_size=1,
        prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        eos=-1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=4,
        sparse_method="h2o",
    )
    scheduler = Scheduler(scheduler_config, RuntimeState(scheduler_config, manager))
    scheduler.decoding.append(seq)

    scheduled, is_prefill, preempted = scheduler.schedule()

    assert scheduled == [seq]
    assert not is_prefill
    assert preempted == []


@pytest.mark.parametrize("use_tensor_table", [True, False])
def test_h2o_final_prefill_pressure_reclaims_unscheduled_active_decode_row(
    use_tensor_table: bool,
):
    manager = _manager_with_layer_rows(
        [[5, 2], [5, 2]],
        decode_budget=4,
        decode_eviction_interval=3,
        prefill_budget=8,
    )
    active_decode = _seq(0, 5, prefilled=5, chunk=1)
    final_prefill = _seq(1, 2, prefilled=0, chunk=2)
    _set_scores_from_slot_rows(manager)
    manager.evict_after_decode([active_decode])
    # Post-forward state: this final prefill consumed the last physical slot.
    manager._num_free_slots = [0, 0]
    if not use_tensor_table:
        manager.buffer_req_to_token_slots_tensor = None

    manager.evict_after_prefill([final_prefill])

    assert [lengths.tolist() for lengths in manager.row_seq_lens] == [
        [4, 2],
        [4, 2],
    ]
    _assert_scores_match_slot_rows(manager)
    assert manager._num_free_slots == [1, 1]
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 0,
        "final_prefill_evictions": 0,
        "decode_eviction_bursts": 1,
        "decode_evictions": 2,
        "dropped_tokens": 2,
    }


def test_h2o_decode_pressure_reclaims_unscheduled_active_row():
    manager = _manager_with_rows(
        [5, 4], decode_budget=4, decode_eviction_interval=3, prefill_budget=8
    )
    unscheduled = _seq(0, 5, prefilled=5, chunk=1)
    scheduled = _seq(1, 4, prefilled=4, chunk=1)
    _set_scores_from_slot_rows(manager)
    manager.evict_after_decode([unscheduled])
    manager._num_free_slots = [0]

    manager.evict_after_decode([scheduled])

    assert manager.row_seq_lens[0].tolist() == [4, 4]
    assert manager.buffer_req_to_token_slots[0][0, :4].tolist() == [1, 2, 3, 4]
    assert manager.buffer_req_to_token_slots[0][1, :4].tolist() == [100, 101, 102, 103]
    assert manager._h2o_scores[(0, 0)].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert manager._h2o_scores[(0, 1)].tolist() == [100.0, 101.0, 102.0, 103.0]
    assert manager._num_free_slots == [1]
    assert manager._h2o_counters["decode_eviction_bursts"] == 1
    assert manager._h2o_counters["decode_evictions"] == 1
    assert manager._h2o_counters["dropped_tokens"] == 1


def test_h2o_decode_without_pressure_leaves_unscheduled_row_until_interval():
    manager = _manager_with_rows(
        [5, 4], decode_budget=4, decode_eviction_interval=3, prefill_budget=8
    )
    unscheduled = _seq(0, 5, prefilled=5, chunk=1)
    scheduled = _seq(1, 4, prefilled=4, chunk=1)
    _set_scores_from_slot_rows(manager)

    manager.evict_after_decode([unscheduled])
    assert manager._h2o_active_decode_seq_ids == {0}
    manager.evict_after_decode([scheduled])

    assert manager.row_seq_lens[0].tolist() == [5, 4]
    assert manager._h2o_active_decode_seq_ids == {0, 1}
    _assert_scores_match_slot_rows(manager)
    assert manager._num_free_slots == [32]
    assert manager._h2o_counters["decode_evictions"] == 0


def test_h2o_decode_pressure_reclaims_all_unscheduled_rows_fast_and_fallback():
    retained = []
    for use_tensor_table in (True, False):
        manager = _manager_with_layer_rows(
            [[5, 6, 4], [5, 6, 4]],
            decode_budget=4,
            decode_eviction_interval=3,
            prefill_budget=8,
        )
        seqs = [
            _seq(0, 5, prefilled=5, chunk=1),
            _seq(1, 6, prefilled=6, chunk=1),
            _seq(2, 4, prefilled=4, chunk=1),
        ]
        _set_scores_from_slot_rows(manager)
        manager.evict_after_decode(seqs[:2])
        manager._num_free_slots = [0, 1]
        if not use_tensor_table:
            manager.buffer_req_to_token_slots_tensor = None

        manager.evict_after_decode([seqs[2]])

        retained.append(
            [
                manager.buffer_req_to_token_slots[layer_idx][seq_id, :4].tolist()
                for layer_idx in range(2)
                for seq_id in range(3)
            ]
        )
        assert [lengths.tolist() for lengths in manager.row_seq_lens] == [
            [4, 4, 4],
            [4, 4, 4],
        ]
        _assert_scores_match_slot_rows(manager)
        assert manager._num_free_slots == [3, 4]
        assert manager._h2o_counters == {
            "intermediate_prefill_evictions": 0,
            "final_prefill_evictions": 0,
            "decode_eviction_bursts": 2,
            "decode_evictions": 4,
            "dropped_tokens": 6,
        }

    assert retained[0] == retained[1]


def test_h2o_decode_pressure_does_not_compact_idle_chain_row():
    manager = _manager_with_rows(
        [5, 4], decode_budget=4, decode_eviction_interval=3, prefill_budget=8
    )
    idle = _seq(0, 5, prefilled=5, chunk=1)
    scheduled = _seq(1, 4, prefilled=4, chunk=1)
    _set_scores_from_slot_rows(manager)
    manager.evict_after_decode([idle])
    assert manager._h2o_active_decode_seq_ids == {0}
    manager.on_chain_turn_finished(0, processed_token_count=5)
    assert manager._h2o_active_decode_seq_ids == set()
    manager._num_free_slots = [0]

    manager.evict_after_decode([scheduled])

    assert manager.row_seq_lens[0].tolist() == [5, 4]
    _assert_scores_match_slot_rows(manager)
    assert manager._num_free_slots == [0]
    assert manager._h2o_counters["decode_evictions"] == 0


def test_h2o_decode_fallback_matches_cross_layer_batch_fast_path_with_ties():
    retained = []
    for use_tensor_table in (True, False):
        manager = _manager_with_layer_rows(
            [[7, 7], [7, 7]],
            decode_budget=4,
            decode_eviction_interval=3,
            prefill_budget=8,
        )
        seqs = [
            _seq(0, 7, prefilled=7, chunk=1),
            _seq(1, 7, prefilled=7, chunk=1),
        ]
        for layer_idx in range(2):
            for seq_id in range(2):
                manager._h2o_scores[(layer_idx, seq_id)] = torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 9.0]
                )
        if not use_tensor_table:
            manager.buffer_req_to_token_slots_tensor = None

        manager.evict_after_decode(seqs)

        retained.append(
            [
                manager.buffer_req_to_token_slots[layer_idx][seq_id, :4].tolist()
                for layer_idx in range(2)
                for seq_id in range(2)
            ]
        )
        assert manager._h2o_counters["decode_eviction_bursts"] == 2
        assert manager._h2o_counters["decode_evictions"] == 4
        assert manager._h2o_counters["dropped_tokens"] == 12

    assert retained[0] == retained[1]


def test_h2o_decode_mixed_graph_batch_compacts_triggered_sequence():
    manager = _manager_with_layer_rows(
        [[7, 6], [7, 6]],
        decode_budget=4,
        decode_eviction_interval=3,
        prefill_budget=8,
    )
    seqs = [
        _seq(0, 7, prefilled=7, chunk=1),
        _seq(1, 6, prefilled=6, chunk=1),
    ]
    _set_scores_from_slot_rows(manager)

    manager.evict_after_decode(seqs)

    assert [lengths.tolist() for lengths in manager.row_seq_lens] == [[4, 6], [4, 6]]
    _assert_scores_match_slot_rows(manager)
    assert manager._num_free_slots == [35, 35]
    assert manager._h2o_counters["decode_eviction_bursts"] == 1
    assert manager._h2o_counters["decode_evictions"] == 2
    assert manager._h2o_counters["dropped_tokens"] == 6


def test_h2o_decode_sequence_reorder_and_temporary_absence_are_independent():
    manager = _manager_with_rows(
        [4, 4], decode_budget=4, decode_eviction_interval=3, prefill_budget=8
    )
    seq0 = _seq(0, 7, prefilled=7, chunk=1)
    seq1 = _seq(1, 7, prefilled=7, chunk=1)
    _set_scores_from_slot_rows(manager)

    for slot in (1_000, 1_001):
        _append_decode_token(manager, 0, 1, token_slot=slot, token_score=float(slot))
        manager.evict_after_decode([seq1])
    assert manager.row_seq_lens[0].tolist() == [4, 6]

    for slot in (2_000, 2_001, 2_002):
        _append_decode_token(manager, 0, 0, token_slot=slot, token_score=float(slot))
        manager.evict_after_decode([seq0])
    assert manager.row_seq_lens[0].tolist() == [4, 6]

    _append_decode_token(manager, 0, 1, token_slot=1_002, token_score=1_002.0)
    manager.evict_after_decode([seq1, seq0])

    assert manager.row_seq_lens[0].tolist() == [4, 4]
    _assert_scores_match_slot_rows(manager)
    assert manager._h2o_counters["decode_eviction_bursts"] == 2
    assert manager._num_free_slots[0] == 38


def test_h2o_controller_batches_raw_decode_logits_for_cache_manager():
    class RecordingManager:
        device = torch.device("cpu")

        def __init__(self):
            self.raw_logits = None
            self.update_kwargs = None
            self.layer_indices = None
            self.evicted = False

        def update_decode_attention_scores_all_layers(
            self,
            layer_indices,
            seqs,
            raw_logits,
            **kwargs,
        ):
            self.layer_indices = list(layer_indices)
            self.raw_logits = raw_logits.clone()
            self.update_kwargs = kwargs

        def evict_after_decode(self, seqs):
            self.evicted = True

    manager = RecordingManager()
    config = SimpleNamespace(
        sparse_method="h2o",
        obs_layer_ids=[],
        full_attention_layers=[],
        runtime_layout=_layout(),
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=2,
            num_attention_heads=2,
            head_dim=1,
            dtype=torch.float32,
        ),
        tensor_parallel_size=1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float32",
    )
    controller = SparseController(config, manager)
    reduced_scores = controller.runtime._get_h2o_decode_score_buffer(1, 1, 3)
    reduced_scores[0].copy_(torch.tensor([[0.2, 0.3, 0.5]]))
    controller.layer_batch_sparse_states[0].attn_score = reduced_scores[0]
    controller.layer_batch_sparse_states[0].context_lens = torch.tensor([3])

    controller.runtime._h2o_decode_eviction(
        [_seq(0, 3, prefilled=3, chunk=1)]
    )

    assert manager.layer_indices == [0]
    assert manager.raw_logits.shape == (1, 1, 3)
    assert torch.allclose(manager.raw_logits, reduced_scores)
    assert manager.update_kwargs == {"normalize_logits": False}
    assert manager.evicted


def test_h2o_prepare_uses_one_contiguous_snapkv_style_buffer_for_all_kv_layers():
    manager = SimpleNamespace(device=torch.device("cpu"))
    config = SimpleNamespace(
        sparse_method="h2o",
        obs_layer_ids=[],
        full_attention_layers=[],
        runtime_layout=_layout(2),
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=2,
            num_attention_heads=2,
            head_dim=1,
            dtype=torch.float32,
        ),
        tensor_parallel_size=1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float32",
        decode_graph=False,
    )
    controller = SparseController(config, manager)
    for layer_idx in range(2):
        state = controller.layer_batch_sparse_states[layer_idx]
        state.context_lens = torch.tensor([4, 3], dtype=torch.int32)
        state.max_context_len = 4
    seqs = [
        _seq(0, 4, prefilled=4, chunk=1),
        _seq(1, 3, prefilled=3, chunk=1),
    ]
    original = controller.runtime._get_h2o_decode_score_buffer

    with patch.object(
        controller.runtime,
        "_get_h2o_decode_score_buffer",
        wraps=original,
    ) as allocate:
        controller.runtime._prepare_h2o_decode_attn_score_buffer(seqs)

    allocate.assert_called_once()
    assert len(controller.runtime._h2o_decode_attn_score_buffers) == 1
    backing = next(
        iter(controller.runtime._h2o_decode_attn_score_buffers.values())
    )
    assert backing.shape == (2, 2, 4)
    assert torch.all(backing == -1e20)
    for layer_idx in range(2):
        assert (
            controller.layer_batch_sparse_states[layer_idx].attn_score.data_ptr()
            == backing[layer_idx].data_ptr()
        )
    resolved_layers, resolved = (
        controller.runtime._resolve_h2o_decode_attn_score_buffer(
            {
                layer_idx: controller.layer_batch_sparse_states[
                    layer_idx
                ].attn_score
                for layer_idx in range(2)
            }
        )
    )
    assert resolved_layers == [0, 1]
    assert resolved.data_ptr() == backing.data_ptr()


def test_h2o_layer_end_keeps_fused_2d_logits_for_batched_normalization():
    manager = SimpleNamespace(device=torch.device("cpu"))
    config = SimpleNamespace(
        sparse_method="h2o",
        obs_layer_ids=[],
        full_attention_layers=[],
        runtime_layout=_layout(),
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=2,
            num_attention_heads=2,
            head_dim=1,
            dtype=torch.float16,
        ),
        tensor_parallel_size=1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float16",
        decode_graph=False,
    )
    controller = SparseController(config, manager)
    score = controller.runtime._get_h2o_decode_score_buffer(1, 1, 3)[0]
    score.copy_(torch.tensor([[0.0, 1.0, 2.0]]))
    controller.layer_batch_sparse_states[0].attn_score = score

    original_ptr = score.data_ptr()
    controller.on_layer_attention_end(0)

    assert score.data_ptr() == original_ptr
    assert score.dtype == torch.float32
    assert torch.equal(score, torch.tensor([[0.0, 1.0, 2.0]]))


def test_h2o_static_decode_reuses_uniform_rows_without_generic_metadata_rebuild():
    manager = _manager_with_layer_rows([[4], [4]])
    manager.max_buffer_rows = 1
    manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
    manager._decode_static_buffers = {}
    manager._decode_static_index_buffers = {}
    manager._decode_static_state_binding_key = None
    manager._decode_static_max_context_len = 5
    manager.free_slots_stack_tensor = torch.stack(
        (
            torch.arange(100, 132, dtype=torch.int32),
            torch.arange(200, 232, dtype=torch.int32),
        )
    )
    manager.free_slots_stack = [
        manager.free_slots_stack_tensor[0],
        manager.free_slots_stack_tensor[1],
    ]
    manager._num_free_slots = [32, 32]
    seq = _seq(0, 5, prefilled=4, chunk=1)
    input_ids = torch.empty(1, dtype=torch.int64)
    positions = torch.empty(1, dtype=torch.int64)
    slot_mapping = torch.empty(1, dtype=torch.int32)
    context_lens = torch.empty(1, dtype=torch.int32)
    req_indices = torch.empty(1, dtype=torch.int32)

    with patch.object(
        SnapKVCacheManager,
        "_allocate_decode_batch_all_layers",
        side_effect=AssertionError("generic metadata rebuild must not run"),
    ):
        manager.prepare_decode_static(
            [seq],
            input_ids,
            positions,
            slot_mapping,
            context_lens,
            req_indices,
        )

    assert manager._num_free_slots == [31, 31]
    assert [int(lengths[0]) for lengths in manager.row_seq_lens] == [5, 5]
    assert manager.buffer_req_to_token_slots[0][0, 4].item() == 131
    assert manager.buffer_req_to_token_slots[1][0, 4].item() == 231
    assert slot_mapping.tolist() == [131]
    assert context_lens.tolist() == [5]
    assert req_indices.tolist() == [0]
    assert manager.layer_batch_states[0].slot_mapping.data_ptr() != slot_mapping.data_ptr()
    assert manager.layer_batch_states[0].max_context_len == 5
    assert manager.layer_batch_states[1].max_context_len == 5


@pytest.mark.parametrize("graph_publish", [False, True])
def test_snapkv_decode_graph_populates_host_mirrors_for_provider_planning(
    graph_publish,
):
    manager = _manager_with_layer_rows([[4], [4]])
    if graph_publish:
        manager.config.sparse_method = "snapkv"
        manager.config.sink_keep_tokens = 0
        manager.config.decode_keep_tokens = manager.max_model_len
        manager.config.recent_keep_tokens = 0
        manager._uniform_decode_metadata = True
    manager.max_buffer_rows = 1
    manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
    manager._decode_static_buffers = {}
    manager._decode_static_index_buffers = {}
    manager._decode_static_state_binding_key = None
    manager._decode_static_max_context_len = 8
    manager.free_slots_stack_tensor = torch.stack(
        (
            torch.arange(100, 132, dtype=torch.int32),
            torch.arange(200, 232, dtype=torch.int32),
        )
    )
    manager.free_slots_stack = [
        manager.free_slots_stack_tensor[0],
        manager.free_slots_stack_tensor[1],
    ]
    manager._num_free_slots = [32, 32]
    seq = _seq(0, 5, prefilled=4, chunk=1)
    contract = DecodeGraphContract(
        method=manager.config.sparse_method,
        topology_path_id="unified",
        batch_capacity=4,
        context_capacity=8,
    )
    inputs = DecodeGraphInputs.allocate(
        contract,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    inputs.host.context_lens.zero_()
    inputs.host.request_indices.fill_(-1)
    state = manager.init_decode_graph_state(contract, inputs)

    manager.prepare_decode_graph_step([seq], state)

    assert inputs.host.input_ids.tolist() == inputs.input_ids.tolist()
    assert inputs.host.positions.tolist() == inputs.positions.tolist()
    assert inputs.host.context_lens.tolist() == [5, 5, 5, 5]
    assert inputs.host.context_lens.tolist() == inputs.context_lens.tolist()
    assert inputs.host.request_indices.tolist() == [0, 0, 0, 0]
    assert inputs.host.request_indices.tolist() == inputs.request_indices.tolist()
    assert inputs.host.active_mask.tolist() == [True, False, False, False]
    assert inputs.host.active_mask.tolist() == inputs.active_mask.tolist()
    expected_before_graph = [0, 0] if graph_publish else [131, 231]
    assert (
        manager.buffer_req_to_token_slots_tensor[:, 0, 4].tolist()
        == expected_before_graph
    )

    manager.prepare_decode_graph_in(state)

    assert manager.buffer_req_to_token_slots_tensor[:, 0, 4].tolist() == [131, 231]
    assert inputs.write_slot_mapping.tolist() == [131, -1, -1, -1]
    assert manager.layer_batch_states[0].slot_mapping.tolist() == [131, -1, -1, -1]
    assert manager.layer_batch_states[1].slot_mapping.tolist() == [231, -1, -1, -1]

def test_h2o_static_decode_rejects_divergent_layer_row_lengths_before_allocation():
    manager = _manager_with_layer_rows(
        [[4], [3]],
        validate_runtime_invariants=True,
    )
    manager.max_buffer_rows = 1
    manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
    manager._decode_static_buffers = {}
    manager._decode_static_index_buffers = {}
    manager._decode_static_state_binding_key = None
    manager._decode_static_max_context_len = 5
    manager.free_slots_stack_tensor = torch.stack(
        (
            torch.arange(100, 132, dtype=torch.int32),
            torch.arange(200, 232, dtype=torch.int32),
        )
    )
    manager.free_slots_stack = [
        manager.free_slots_stack_tensor[0],
        manager.free_slots_stack_tensor[1],
    ]
    manager._num_free_slots = [32, 32]
    seq = _seq(0, 5, prefilled=4, chunk=1)
    buffers = (
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
    )

    with pytest.raises(RuntimeError, match="uniform row lengths"):
        manager.prepare_decode_static([seq], *buffers)

    assert manager._num_free_slots == [32, 32]
    assert [int(lengths[0]) for lengths in manager.row_seq_lens] == [4, 3]


def test_h2o_static_decode_rechecks_row_lengths_after_row_cache_reuse():
    manager = _manager_with_layer_rows(
        [[4], [4]],
        validate_runtime_invariants=True,
    )
    manager.max_buffer_rows = 1
    manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
    manager._decode_static_buffers = {}
    manager._decode_static_index_buffers = {}
    manager._decode_static_state_binding_key = None
    manager._decode_static_max_context_len = 6
    manager.free_slots_stack_tensor = torch.stack(
        (
            torch.arange(100, 132, dtype=torch.int32),
            torch.arange(200, 232, dtype=torch.int32),
        )
    )
    manager.free_slots_stack = [
        manager.free_slots_stack_tensor[0],
        manager.free_slots_stack_tensor[1],
    ]
    manager._num_free_slots = [32, 32]
    seq = _seq(0, 6, prefilled=4, chunk=1)
    buffers = (
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
    )

    manager.prepare_decode_static([seq], *buffers)
    assert getattr(manager, "_h2o_decode_static_rows", None) is not None
    manager.row_seq_lens[1][0] = 4
    manager.buffer_req_to_token_slots[1][0, 4] = 0

    with pytest.raises(RuntimeError, match="uniform row lengths"):
        manager.prepare_decode_static([seq], *buffers)

    assert manager._num_free_slots == [31, 31]
    assert [int(lengths[0]) for lengths in manager.row_seq_lens] == [5, 4]


def test_snapkv_uniform_decode_cross_layer_checks_are_debug_only():
    def make_manager(enabled: bool):
        manager = object.__new__(SnapKVCacheManager)
        manager.validate_runtime_invariants = enabled
        manager.device = torch.device("cpu")
        manager.runtime_layout = _layout(2)
        manager.max_model_len = 8
        manager.seq_id_to_row = [{0: 0}, {0: 1}]
        manager.row_seq_lens = [
            np.asarray([1, 0], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
        ]
        manager._num_free_slots = [4, 4]
        manager.free_slots_stack = [
            torch.arange(4, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
        ]
        manager.buffer_req_to_token_slots_tensor = torch.zeros(
            (2, 2, 8), dtype=torch.int32
        )
        manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
        manager._decode_static_state_binding_key = None
        manager.attention_cache_storage = None
        return manager

    seq = _seq(0, 2, prefilled=1, chunk=1)

    debug_manager = make_manager(True)
    with pytest.raises(RuntimeError, match="identical request rows"):
        debug_manager._prepare_decode_static_uniform(
            [seq],
            torch.empty(1, dtype=torch.int64),
            torch.empty(1, dtype=torch.int64),
            torch.empty(1, dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
        )
    assert debug_manager._num_free_slots == [4, 4]

    fast_manager = make_manager(False)
    result = fast_manager._prepare_decode_static_uniform(
        [seq],
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
    )
    assert result is not None
    assert fast_manager._num_free_slots == [3, 3]


def test_h2o_multilayer_reduced_decode_context_bounds_are_checked_together():
    class RecordingManager:
        device = torch.device("cpu")

        def __init__(self):
            self.updates = 0
            self.evictions = 0

        def update_decode_attention_scores_all_layers(
            self,
            layer_indices,
            seqs,
            raw_logits,
            **kwargs,
        ):
            assert list(layer_indices) == [0, 1]
            assert len(seqs) == 1
            assert raw_logits.shape == (2, 1, 3)
            assert kwargs == {"normalize_logits": False}
            self.updates += 1

        def evict_after_decode(self, seqs):
            self.evictions += 1

    manager = RecordingManager()
    config = SimpleNamespace(
        sparse_method="h2o",
        obs_layer_ids=[],
        full_attention_layers=[],
        runtime_layout=_layout(2),
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=1,
            num_attention_heads=1,
            head_dim=1,
            dtype=torch.float32,
        ),
        tensor_parallel_size=1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float32",
        decode_graph=False,
        validate_runtime_invariants=True,
    )
    controller = SparseController(config, manager)
    reduced_scores = controller.runtime._get_h2o_decode_score_buffer(2, 1, 3)
    reduced_scores.zero_()
    controller.layer_batch_sparse_states[0].attn_score = reduced_scores[0]
    controller.layer_batch_sparse_states[1].attn_score = reduced_scores[1]
    controller.layer_batch_sparse_states[0].context_lens = torch.tensor([3])
    controller.layer_batch_sparse_states[1].context_lens = torch.tensor([2])
    seqs = [_seq(0, 3, prefilled=3, chunk=1)]

    controller.runtime._h2o_decode_eviction(seqs)
    assert manager.updates == 1
    assert manager.evictions == 1

    reduced_scores.zero_()
    controller.layer_batch_sparse_states[1].context_lens = torch.tensor([4])
    with pytest.raises(RuntimeError, match="exceed the reduced score width"):
        controller.runtime._h2o_decode_eviction(seqs)
    assert manager.updates == 1
    assert manager.evictions == 1

    config.validate_runtime_invariants = False
    fast_controller = SparseController(config, manager)
    config.validate_runtime_invariants = True
    fast_scores = fast_controller.runtime._get_h2o_decode_score_buffer(2, 1, 3)
    fast_scores.zero_()
    fast_controller.layer_batch_sparse_states[0].attn_score = fast_scores[0]
    fast_controller.layer_batch_sparse_states[1].attn_score = fast_scores[1]
    fast_controller.layer_batch_sparse_states[0].context_lens = torch.tensor([3])
    fast_controller.layer_batch_sparse_states[1].context_lens = torch.tensor([4])

    fast_controller.runtime._h2o_decode_eviction(seqs)

    assert fast_controller.runtime.validate_runtime_invariants is False
    assert manager.updates == 2
    assert manager.evictions == 2


def test_h2o_contiguous_decode_buffer_handles_padded_graph_batch_and_low_dtype():
    class RecordingManager:
        device = torch.device("cpu")

        def __init__(self):
            self.raw_logits = None
            self.update_kwargs = None
            self.evicted = False

        def update_decode_attention_scores_all_layers(
            self,
            layer_indices,
            seqs,
            raw_logits,
            **kwargs,
        ):
            assert list(layer_indices) == [0, 1]
            assert len(seqs) == 3
            self.raw_logits = raw_logits.clone()
            self.update_kwargs = kwargs

        def evict_after_decode(self, seqs):
            self.evicted = True

    manager = RecordingManager()
    config = SimpleNamespace(
        sparse_method="h2o",
        obs_layer_ids=[],
        full_attention_layers=[],
        runtime_layout=_layout(2),
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=2,
            num_attention_heads=2,
            head_dim=1,
            dtype=torch.float16,
        ),
        tensor_parallel_size=1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float16",
        decode_graph=True,
    )
    controller = SparseController(config, manager)
    reduced_scores = controller.runtime._get_h2o_decode_score_buffer(2, 4, 4)
    original_ptr = reduced_scores.data_ptr()
    reduced_scores.fill_(7.0)
    same_scores = controller.runtime._get_h2o_decode_score_buffer(2, 4, 4)
    assert same_scores.data_ptr() == original_ptr
    assert same_scores.dtype == torch.float32
    assert torch.all(same_scores == 7.0)

    # Startup subsequently prepares another batch. Replay must use the restored
    # four-row graph slices below, not this most recently prepared one-row view.
    manager._decode_static_max_context_len = 4
    for state in controller.layer_batch_sparse_states.values():
        state.context_lens = torch.tensor([2], dtype=torch.int32)
        state.max_context_len = 2
    controller.runtime._prepare_h2o_decode_attn_score_buffer(
        [_seq(0, 2, prefilled=2, chunk=1)]
    )

    valid_lens = torch.tensor([4, 3, 2, 0], dtype=torch.int32)
    for layer_idx in range(2):
        for batch_idx, valid_len in enumerate(valid_lens.tolist()):
            if valid_len:
                same_scores[layer_idx, batch_idx, :valid_len] = torch.softmax(
                    torch.arange(valid_len, dtype=torch.float32)
                    + 10 * layer_idx
                    + batch_idx,
                    dim=-1,
                )
        state = controller.layer_batch_sparse_states[layer_idx]
        state.attn_score = same_scores[layer_idx]
        state.context_lens = valid_lens
    expected = same_scores[:, :3].clone()

    controller.runtime._h2o_decode_eviction(
        [_seq(seq_id, 4, prefilled=4, chunk=1) for seq_id in range(3)]
    )

    assert manager.raw_logits.shape == (2, 3, 4)
    assert manager.raw_logits.dtype == torch.float32
    assert torch.allclose(manager.raw_logits, expected)
    assert manager.update_kwargs == {"normalize_logits": False}
    assert torch.equal(same_scores[:, :3], expected)
    assert manager.evicted
    assert any(
        tensor.data_ptr() == original_ptr
        for tensor in controller.decode_graph_keepalive_tensors()
    )

    refs = {
        layer_idx: {"attn_score": same_scores[layer_idx]}
        for layer_idx in range(2)
    }
    same_scores.fill_(1.0)
    assert controller.reset_decode_attn_scores_for_graph(refs)
    assert torch.all(same_scores == -1e20)
    controller.clear_decode_attn_score_buffers()
    assert controller.runtime._h2o_decode_attn_score_buffers == {}


def test_h2o_free_seq_cleans_score_vectors():
    manager = _manager_with_rows([2])
    manager._h2o_scores[(0, 0)] = torch.ones(2)
    buffer = torch.ones(8)
    manager._h2o_decode_score_rows = {(0, 0): (buffer, buffer[:2]), (1, 0): (buffer, buffer[:2])}
    manager._h2o_active_decode_seq_ids.add(0)
    with patch.object(SnapKVCacheManager, "free_seq", autospec=True) as parent_free:
        manager.free_seq(0)
    assert manager._h2o_scores == {}
    assert manager._h2o_decode_score_rows == {}
    assert manager._h2o_active_decode_seq_ids == set()
    parent_free.assert_called_once_with(manager, 0)


def test_h2o_chain_turn_aligns_score_free_decode_growth_until_reclaimed():
    manager = _manager_with_rows([4])
    manager._h2o_scores[(0, 0)] = torch.arange(2, dtype=torch.float32)
    manager._h2o_active_decode_seq_ids.add(0)

    manager.on_chain_turn_finished(0, processed_token_count=100)

    assert manager._h2o_scores[(0, 0)].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert manager._h2o_active_decode_seq_ids == set()
    assert manager.chain_physical_residency(0) == (4,)


def test_h2o_reset_after_warmup_clears_scores_and_counters():
    manager = _manager_with_rows([2])
    manager._h2o_scores[(0, 0)] = torch.ones(2)
    buffer = torch.ones(8)
    manager._h2o_decode_score_rows = {(0, 0): (buffer, buffer[:2])}
    manager._h2o_active_decode_seq_ids.add(0)
    manager._h2o_counters.update(
        {
            "intermediate_prefill_evictions": 1,
            "final_prefill_evictions": 2,
            "decode_eviction_bursts": 2,
            "decode_evictions": 3,
            "dropped_tokens": 4,
        }
    )

    manager.reset_after_warmup()
    assert manager._h2o_scores == {}
    assert manager._h2o_decode_score_rows == {}
    assert manager._h2o_active_decode_seq_ids == set()
    assert manager._h2o_counters == {
        "intermediate_prefill_evictions": 0,
        "final_prefill_evictions": 0,
        "decode_eviction_bursts": 0,
        "decode_evictions": 0,
        "dropped_tokens": 0,
    }


def test_h2o_capacity_follows_independent_prefill_and_decode_axes():
    manager = _manager_with_rows([8], decode_budget=4, prefill_budget=8)
    manager.free_rows = [deque([0])]

    manager.config.prefill_sparse_method = ""
    assert manager._prefill_append_peak(8, 20, 4) == 28
    required, _, _, _ = manager.chain_capacity_deficits(
        suffix_tokens=20,
        generation_tokens=5,
        existing_slots_by_layer=(8,),
        needs_resident_row=False,
    )
    assert required == (20,)

    manager.config.sparse_method = ""
    manager.config.prefill_sparse_method = "h2o_prefill"
    assert manager._prefill_append_peak(8, 20, 4) == 12
    required, _, _, _ = manager.chain_capacity_deficits(
        suffix_tokens=20,
        generation_tokens=5,
        existing_slots_by_layer=(8,),
        needs_resident_row=False,
    )
    assert required == (8,)


@pytest.mark.parametrize("existing", [4, 9])
@pytest.mark.parametrize("generation", [1, 2, 12])
def test_h2o_chain_online_eviction_reserves_peak_including_next_append(existing, generation):
    manager = _manager_with_rows([existing], decode_budget=4, decode_eviction_interval=3)
    manager.config.h2o_decode_eviction = True
    manager.free_rows = [deque([0])]
    resident = peak = existing
    for _ in range(generation - 1):
        resident += 1
        peak = max(peak, resident)
        if resident >= 7:
            resident = 4
    required, _, _, _ = manager.chain_capacity_deficits(
        suffix_tokens=0, generation_tokens=generation,
        existing_slots_by_layer=(existing,), needs_resident_row=False,
    )
    assert required == (peak - existing,)


def test_h2o_capacity_hooks_reserve_prefill_peak_and_gate_chunk_with_real_free_slots():
    manager = _manager_with_rows([8], decode_budget=4, prefill_budget=8)
    manager._num_free_slots = [4]
    seq = _seq(0, 100, prefilled=10, chunk=1)

    assert manager.prompt_admission_cost(seq) == 12
    assert manager.prompt_logical_reservation_cost(seq) == 12
    assert manager.reserved_prefill_slots(deque([seq]), 4) == 4
    assert manager.prefill_step_free_slots_for(seq) == 4
    assert manager.prefill_step_reservation_cost(seq, 4) == 4
    assert manager.decode_step_free_slots_for(seq) == 4
    assert manager.decode_step_reservation_cost(seq) == 1

    scheduler_config = SimpleNamespace(
        max_num_seqs_in_batch=4,
        max_num_seqs_in_gpu=4,
        max_num_batched_tokens=16,
        max_decoding_seqs=4,
        engine_prefill_chunk_size=16,
        prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        eos=-1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=4,
        sparse_method="h2o",
    )
    scheduler = Scheduler(scheduler_config, RuntimeState(scheduler_config, manager))
    scheduler.waiting.append(seq)
    scheduled, is_prefill, _ = scheduler.schedule()
    assert is_prefill
    assert scheduled == [seq]
    assert seq.current_chunk_size == 4


def test_h2o_reserved_prefill_slots_keeps_over_budget_resume_headroom():
    manager = _manager_with_rows(
        [6],
        decode_budget=4,
        decode_eviction_interval=3,
        prefill_budget=4,
        engine_prefill_chunk_size=4,
    )
    seq = _seq(0, 110, prefilled=100, chunk=0)

    assert manager.reserved_prefill_slots(deque([seq]), 4) == 4


def test_h2o_scheduler_does_not_admit_partial_prefills_that_fill_all_slots():
    manager = _manager_with_rows(
        [0, 0, 0],
        decode_budget=4,
        prefill_budget=8,
        engine_prefill_chunk_size=4,
    )
    manager._num_free_slots = [12]
    seqs = [_seq(seq_id, 20, prefilled=0, chunk=0) for seq_id in range(3)]
    scheduler_config = SimpleNamespace(
        max_num_seqs_in_batch=3,
        max_num_seqs_in_gpu=3,
        max_num_batched_tokens=12,
        max_decoding_seqs=3,
        engine_prefill_chunk_size=4,
        prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        eos=-1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=4,
        sparse_method="h2o",
    )
    scheduler = Scheduler(scheduler_config, RuntimeState(scheduler_config, manager))
    scheduler.waiting.extend(seqs)

    scheduled, is_prefill, _ = scheduler.schedule()

    assert is_prefill
    assert scheduled == [seqs[0]]
    assert seqs[0].current_chunk_size == 4

    progress = []
    while not scheduler.decoding:
        for seq in scheduled:
            row_idx = manager.seq_id_to_row[0][seq.seq_id]
            manager.row_seq_lens[0][row_idx] += int(seq.current_chunk_size)
            manager._num_free_slots[0] -= int(seq.current_chunk_size)
            progress.append((seq.seq_id, int(seq.num_prefilled_tokens)))
            budget = (
                manager.h2o_decode_budget
                if seq.is_last_chunk_prefill
                else manager.h2o_prefill_budget
            )
            physical_len = int(manager.row_seq_lens[0][row_idx])
            if physical_len > budget:
                manager.row_seq_lens[0][row_idx] = budget
                manager._num_free_slots[0] += physical_len - budget
        scheduler.postprocess(scheduled, [0] * len(scheduled), is_prefill=True)
        if scheduler.decoding:
            break
        assert manager.reserved_prefill_slots(scheduler.waiting, 4) >= 4
        scheduled, is_prefill, _ = scheduler.schedule()
        assert is_prefill
        assert scheduled

    assert scheduler.decoding[0] is seqs[0]
    assert seqs[0].num_prefilled_tokens == seqs[0].num_prompt_tokens
    assert len(progress) == 5


def test_h2o_scheduler_reserves_until_first_prefill_eviction_peak():
    manager = _manager_with_rows(
        [4, 0],
        decode_budget=50,
        prefill_budget=100,
        engine_prefill_chunk_size=4,
    )
    manager._num_free_slots = [108]
    first = _seq(0, 300, prefilled=4, chunk=0)
    second = _seq(1, 300, prefilled=0, chunk=0)
    scheduler_config = SimpleNamespace(
        max_num_seqs_in_batch=2,
        max_num_seqs_in_gpu=2,
        max_num_batched_tokens=8,
        max_decoding_seqs=2,
        engine_prefill_chunk_size=4,
        prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        eos=-1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=50,
        sparse_method="h2o",
    )
    scheduler = Scheduler(scheduler_config, RuntimeState(scheduler_config, manager))
    scheduler.waiting.extend((first, second))

    assert manager.reserved_prefill_slots(scheduler.waiting, 4) == 100
    scheduled, is_prefill, _ = scheduler.schedule()

    assert is_prefill
    assert scheduled == [first]
    assert first.current_chunk_size == 4


def test_h2o_debug_summary_exposes_auditable_eviction_counters():
    manager = _manager_with_rows([2])
    manager._h2o_scores[(0, 0)] = torch.ones(2)
    with patch.object(SnapKVCacheManager, "debug_state_summary", return_value={"live_rows": {}}):
        summary = manager.debug_state_summary()
    assert summary["h2o"]["counters"]["dropped_tokens"] == 0
    assert summary["h2o"]["counters"]["decode_eviction_bursts"] == 0
    assert summary["h2o"]["score_lengths"] == {"0:0": 2}
