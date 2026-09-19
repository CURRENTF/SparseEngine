from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sparseengine.platforms as platforms
from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager.base import CacheManager
from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
from sparseengine.engine.cache_manager.methods.snapkv import (
    SnapKVCacheManager,
    resolve_snapkv_cache_capacity,
)
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

    # 2,000 - 10 staging slots * 40 - 2 layers * 3 rows * 5 tokens * 4
    # leaves 1,480 bytes for proportional KV slots plus their int32 stacks.
    assert config.num_kvcache_slots == [22, 11]
    assert manager.pyramidkv_prefill_staging_kv_cache.shape == (2, 10, 1, 5)
    kv_bytes = sum(
        _tensor_nbytes(k_cache) + _tensor_nbytes(v_cache)
        for k_cache, v_cache in manager.kv_cache
    )
    staging_bytes = _tensor_nbytes(manager.pyramidkv_prefill_staging_kv_cache)
    row_slot_map_bytes = _tensor_nbytes(manager.buffer_req_to_token_slots_tensor)
    free_stack_bytes = sum(
        _tensor_nbytes(stack)
        for stack in manager.free_slots_stack
        if stack is not None
    )
    assert kv_bytes == 1_320
    assert staging_bytes == 400
    assert row_slot_map_bytes == 120
    assert free_stack_bytes == 132
    assert kv_bytes + staging_bytes + row_slot_map_bytes + free_stack_bytes == 1_972


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
