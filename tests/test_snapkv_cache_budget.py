from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sparseengine.platforms as platforms
from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager.base import CacheManager
from sparseengine.engine.cache_manager.chain_offload import ChainMethodState
from sparseengine.engine.decode_graph_contract import DecodeGraphContract, DecodeGraphInputs
from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
from sparseengine.engine.cache_manager.methods.snapkv import (
    SnapKVCacheManager,
    resolve_snapkv_cache_capacity,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.platforms.cpu import CpuPlatform


def _parallel_context():
    return SimpleNamespace(
        world_rank=0,
        world_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        attn_dp_rank=0,
        attn_dp_size=1,
    )


def _manager_config(*, method: str, compression_interval: int = 1):
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            num_attention_heads=2,
            hidden_size=10,
            head_dim=5,
            dtype=torch.float32,
        ),
        runtime_layout=RuntimeLayout.dense(2),
        attention_cache_layout="explicit_kv",
        max_model_len=5,
        max_num_batched_tokens=10,
        max_num_seqs_in_gpu=3,
        sparse_method=method,
        pyramid_layer_ratios=[1.0, 0.5] if method == "pyramidkv" else None,
        prefill_schedule_policy="long_bs1full_short_batch",
        num_kvcache_slots=None,
        sink_keep_tokens=1,
        decode_keep_tokens=2,
        recent_keep_tokens=1,
        decode_reservation_tokens=3,
        decode_eviction_interval=1,
        observation_window_size=2,
        snapkv_decode_eviction=False,
        sparse_prefill_score_mode="logits",
        rkv_compression_interval=compression_interval,
        rkv_observation_tokens=2,
    )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def test_uniform_manager_persistent_tensors_match_capacity_budget():
    config = _manager_config(method="snapkv")
    with (
        patch.object(platforms, "_current_platform", CpuPlatform()),
        patch.object(
            CacheManager,
            "_get_available_slots_info",
            return_value=(1_000, 40),
        ),
    ):
        manager = SnapKVCacheManager(config, _parallel_context())

    kv_bytes = _tensor_nbytes(manager.kv_cache)
    row_slot_map_bytes = _tensor_nbytes(manager.buffer_req_to_token_slots_tensor)
    free_stack_bytes = _tensor_nbytes(manager.free_slots_stack_tensor)
    assert config.num_kvcache_slots == 10
    assert kv_bytes == 800
    assert row_slot_map_bytes == 120
    assert free_stack_bytes == 80
    assert kv_bytes + row_slot_map_bytes + free_stack_bytes == 1_000
    assert not hasattr(manager, "_free_slots_layer_indices")

    # torch.arange(num_slots).expand(...).clone() temporarily adds one int32
    # row while initializing the persistent free-stack tensor.
    transient_arange_bytes = config.num_kvcache_slots * 4
    assert transient_arange_bytes == 40


def test_decode_query_window_tracks_recent_queries_and_resets_after_compaction():
    config = _manager_config(method="snapkv")
    config.snapkv_decode_eviction = True
    config.observation_window_size = 3
    config.max_model_len = 10
    config.decode_eviction_interval = 2
    with (
        patch.object(platforms, "_current_platform", CpuPlatform()),
        patch.object(CacheManager, "_get_available_slots_info", return_value=(5_000, 40)),
    ):
        manager = SnapKVCacheManager(config, _parallel_context())

    seq = Sequence([1])
    manager._allocate(0, seq.seq_id, 6)
    row = manager.seq_id_to_row[0][seq.seq_id]
    state = manager.get_layer_batch_states(0)
    state.req_indices = torch.tensor([row], dtype=torch.int32)
    seen = []

    def score_oracle(q, _k, output, req_indices, starts, lengths, cached,
                     max_query_len, _slots, score_starts, score_ends, **_kwargs):
        seen.append((q[:, 0, 0].tolist(), max_query_len,
                     req_indices.tolist(), starts.tolist(), lengths.tolist(),
                     cached.tolist(), score_starts.tolist(), score_ends.tolist()))
        output.fill_(0)
        output[0, 1] = q[:, 0, 0].sum()

    with patch("sparseengine.engine.cache_manager.methods.snapkv.prefill_score_fwd", score_oracle):
        for position in (3, 4, 5):
            state.context_lens = torch.tensor([position + 1], dtype=torch.int32)
            manager.record_decode_query(0, torch.full((1, 2, 5), float(position)))
        first = manager.decode_query_scores(0, seq, 6)
        assert first[1].item() == 12
        assert seen[-1] == ([3.0, 4.0, 5.0], 3, [row], [0], [6], [3], [3], [6])

        manager.free_part_slots(0, seq, torch.tensor([0, 2, 4, 5]))
        with pytest.raises(RuntimeError, match="missing query observations"):
            manager.decode_query_scores(0, seq, 4)

        manager._allocate(0, seq.seq_id, 2)
        for position in (4, 5):
            state.context_lens = torch.tensor([position + 1], dtype=torch.int32)
            manager.record_decode_query(0, torch.full((1, 2, 5), float(position + 10)))
        second = manager.decode_query_scores(0, seq, 6)
        assert second[1].item() == 29
        assert seen[-1] == ([14.0, 15.0], 2, [row], [0], [6], [4], [4], [6])


def test_graph_padding_writes_decode_queries_only_to_real_rows():
    config = _manager_config(method="snapkv")
    config.snapkv_decode_eviction = True
    config.max_model_len = 10
    with (
        patch.object(platforms, "_current_platform", CpuPlatform()),
        patch.object(CacheManager, "_get_available_slots_info", return_value=(5_000, 40)),
    ):
        manager = SnapKVCacheManager(config, _parallel_context())
    seq = Sequence([1])
    for layer in manager.kv_transformer_layer_indices():
        manager._allocate(layer, seq.seq_id, 4)
    contract = DecodeGraphContract(
        method="snapkv", topology_path_id="unified",
        batch_capacity=2, context_capacity=10,
    )
    inputs = DecodeGraphInputs.allocate(contract, device=manager.device, pin_memory=False)
    graph_state = manager.init_decode_graph_state(contract, inputs)
    manager.prepare_decode_graph_step([seq], graph_state)
    assert manager._decode_query_active_mask is inputs.active_mask
    assert inputs.active_mask.tolist() == [True, False]

    row = manager.seq_id_to_row[0][seq.seq_id]
    state = manager.get_layer_batch_states(0)
    assert state.req_indices.tolist() == [row, row]
    q = torch.stack((torch.ones(2, 5), torch.full((2, 5), 99.0)))
    manager.record_decode_query(0, q)
    cache, positions = manager._decode_query_cache_layer(0)
    column = int(state.context_lens[0] - 1) % config.observation_window_size
    assert cache[row, column, 0, 0].item() == 1.0
    assert cache[manager.max_buffer_rows, column, 0, 0].item() == 99.0
    assert positions[row, column].item() == 4


def test_chain_restore_preserves_decode_query_history_across_row_reuse():
    config = _manager_config(method="snapkv")
    config.snapkv_decode_eviction = True
    config.max_model_len = 10
    with (
        patch.object(platforms, "_current_platform", CpuPlatform()),
        patch.object(CacheManager, "_get_available_slots_info", return_value=(5_000, 40)),
    ):
        manager = SnapKVCacheManager(config, _parallel_context())
    seq = Sequence([1])
    for layer in manager.kv_transformer_layer_indices():
        manager._allocate(layer, seq.seq_id, 6)
        row = manager.seq_id_to_row[layer][seq.seq_id]
        state = manager.get_layer_batch_states(layer)
        state.req_indices = torch.tensor([row], dtype=torch.int32)
        for position in (4, 5):
            state.context_lens = torch.tensor([position + 1], dtype=torch.int32)
            manager.record_decode_query(
                layer, torch.full((1, 2, 5), float(position + layer)),
            )
    state = manager.snapshot_chain_method_state(seq.seq_id)
    snapshot = ChainMethodState(tensors={
        name: tensor.clone() for name, tensor in state.tensors.items()
    })
    old_row = manager.seq_id_to_row[0][seq.seq_id]
    manager.free_seq(seq.seq_id)
    for layer in manager.kv_transformer_layer_indices():
        manager._allocate(layer, seq.seq_id, 6)
    assert manager.seq_id_to_row[0][seq.seq_id] != old_row
    manager.restore_chain_method_state(seq.seq_id, snapshot)
    for layer in manager.kv_transformer_layer_indices():
        row = manager.seq_id_to_row[layer][seq.seq_id]
        cache, positions = manager._decode_query_cache_layer(layer)
        torch.testing.assert_close(cache[row, [0, 1], 0, 0],
                                   torch.tensor([4.0 + layer, 5.0 + layer]))
        assert positions[row, [0, 1]].tolist() == [4, 5]


def test_pyramid_capacity_includes_staging_fixed_and_per_slot_metadata():
    config = _manager_config(method="pyramidkv")
    with (
        patch.object(platforms, "_current_platform", CpuPlatform()),
        patch.object(
            CacheManager,
            "_get_available_slots_info",
            return_value=(2_000, 40),
        ),
    ):
        manager = SnapKVCacheManager(config, _parallel_context())

    # The query ring, staging KV, row table, and free stacks share one budget.
    layer_slots = config.num_kvcache_slots
    assert len(layer_slots) == 2 and all(slots > 0 for slots in layer_slots)
    assert manager.pyramidkv_prefill_staging_kv_cache.shape == (2, 10, 1, 5)
    kv_bytes = sum(
        _tensor_nbytes(k_cache) + _tensor_nbytes(v_cache)
        for k_cache, v_cache in manager.kv_cache
    )
    staging_bytes = _tensor_nbytes(manager.pyramidkv_prefill_staging_kv_cache)
    query_bytes = sum(
        _tensor_nbytes(tensor)
        for tensor in (*manager._decode_query_cache, *manager._decode_query_positions)
        if tensor is not None
    )
    row_slot_map_bytes = _tensor_nbytes(manager.buffer_req_to_token_slots_tensor)
    free_stack_bytes = sum(
        _tensor_nbytes(stack)
        for stack in manager.free_slots_stack
        if stack is not None
    )
    assert staging_bytes == 400
    assert row_slot_map_bytes == 120
    assert query_bytes == manager._decode_query_cache_bytes()
    used_bytes = kv_bytes + staging_bytes + row_slot_map_bytes + free_stack_bytes + query_bytes
    assert used_bytes <= 2_000
    assert 2_000 - used_bytes < 2 * (40 + 4)


def test_pyramid_capacity_uses_larger_reservation_or_eviction_interval():
    def allocate(reservation, interval):
        config = _manager_config(method="pyramidkv")
        config.max_model_len = 2048
        config.decode_reservation_tokens = reservation
        config.decode_eviction_interval = interval
        with (
            patch.object(platforms, "_current_platform", CpuPlatform()),
            patch.object(CacheManager, "_get_available_slots_info", return_value=(400_000, 40)),
        ):
            SnapKVCacheManager(config, _parallel_context())
        return config.num_kvcache_slots

    reservation_limited = allocate(1024, 128)
    eviction_limited = allocate(128, 1024)
    assert reservation_limited == eviction_limited
    # The previous ratio-only split made the last layer the admission limit.
    peak_by_layer = (1 + 2 + 1 + 1024, 1 + 1 + 1 + 1024)
    supported = [slots // peak for slots, peak in zip(reservation_limited, peak_by_layer)]
    assert min(supported) >= 2
    assert max(supported) - min(supported) <= 1


def test_pyramid_capacity_reserves_full_attention_and_unstaged_prefill_peaks():
    def allocate(*, full_layers, staged):
        config = _manager_config(method="pyramidkv")
        config.max_model_len = 2048
        config.snapkv_num_full_layers = full_layers
        config.prefill_schedule_policy = (
            "long_bs1full_short_batch" if staged else "unstaged"
        )
        with (
            patch.object(platforms, "_current_platform", CpuPlatform()),
            patch.object(CacheManager, "_get_available_slots_info", return_value=(400_000, 40)),
        ):
            SnapKVCacheManager(config, _parallel_context())
        return config.num_kvcache_slots

    full_layer_slots = allocate(full_layers=1, staged=True)
    assert full_layer_slots[0] // 2048 == full_layer_slots[1] // (1 + 1 + 1 + 3)
    unstaged_slots = allocate(full_layers=0, staged=False)
    assert unstaged_slots[0] == unstaged_slots[1]
    assert unstaged_slots[0] >= 2048


@pytest.mark.parametrize(
    ("compression_interval", "enabled", "expected_slots", "expected_query_bytes"),
    [
        (1, True, 4, 528),
        (2, False, 10, 0),
    ],
)
def test_rkv_manager_query_reserve_matches_allocated_tensors(
    compression_interval,
    enabled,
    expected_slots,
    expected_query_bytes,
):
    config = _manager_config(
        method="rkv",
        compression_interval=compression_interval,
    )
    config.rkv_score_chunk_mb = 1
    scoring_reserve = config.rkv_score_chunk_mb * 1024**2 if enabled else 0
    available_bytes = 1_000 + scoring_reserve
    with (
        patch.object(platforms, "_current_platform", CpuPlatform()),
        patch.object(
            CacheManager,
            "_get_available_slots_info",
            return_value=(available_bytes, 40),
        ),
    ):
        manager = RKVCacheManager(config, _parallel_context())

    assert manager._rkv_query_cache_enabled is enabled
    assert config.num_kvcache_slots == expected_slots
    if enabled:
        query_cache_bytes = sum(
            _tensor_nbytes(tensor)
            for tensor in manager._rkv_query_cache
            if tensor is not None
        )
        query_position_bytes = sum(
            _tensor_nbytes(tensor)
            for tensor in manager._rkv_query_positions
            if tensor is not None
        )
    else:
        assert manager._rkv_query_cache == []
        assert manager._rkv_query_positions == []
        query_cache_bytes = 0
        query_position_bytes = 0

    kv_bytes = _tensor_nbytes(manager.kv_cache)
    row_slot_map_bytes = _tensor_nbytes(manager.buffer_req_to_token_slots_tensor)
    free_stack_bytes = _tensor_nbytes(manager.free_slots_stack_tensor)
    assert manager._rkv_query_cache_bytes() == expected_query_bytes
    assert kv_bytes == 2 * expected_slots * 40
    assert row_slot_map_bytes == 120
    assert free_stack_bytes == 2 * expected_slots * 4
    assert query_cache_bytes + query_position_bytes == expected_query_bytes
    assert (
        kv_bytes
        + row_slot_map_bytes
        + free_stack_bytes
        + query_cache_bytes
        + query_position_bytes
        + scoring_reserve
        == available_bytes
    )


@pytest.mark.parametrize(
    ("available_bytes", "message"),
    [
        (120, "row-slot metadata"),
        (121, "KV slots and free-slot metadata"),
    ],
)
def test_snapkv_capacity_fails_fast_when_persistent_budget_is_too_small(
    available_bytes,
    message,
):
    with pytest.raises(RuntimeError, match=message):
        resolve_snapkv_cache_capacity(
            available_bytes=available_bytes,
            slot_bytes_per_layer=40,
            num_kv_layers=2,
            max_buffer_rows=3,
            max_model_len=5,
        )
