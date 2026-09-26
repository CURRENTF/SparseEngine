from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager import (
    AttentionViewMeta,
    LayerBatchStates,
    PrefillComputeView,
)
from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.cache_manager.storage import MlaLatentStorage
from sparseengine.engine.chain_cache import ChainCacheCoordinator
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import SparseController
from sparseengine.engine.sparse_methods.snapkv import SnapKVRuntime


def _latent_chain_manager(manager_type, method: str):
    capacity = 32
    config = SimpleNamespace(
        sparse_method=method,
        model="/models/glm-chain-test",
        hf_config=SimpleNamespace(
            model_type="glm4_moe_lite",
            dtype=torch.bfloat16,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
        tensor_parallel_size=1,
        max_model_len=capacity,
        max_num_seqs_in_gpu=1,
        prefix_cache_salt="",
        chain_cache_max_tombstones=8,
        full_attention_layers=[0],
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=2,
        observation_window_size=2,
        snapkv_num_full_layers=0,
        sparse_attn_score_dtype="float32",
        pool_kernel_size=1,
        pyramid_layer_ratios=None,
        prefill_schedule_policy="chunked",
        engine_prefill_chunk_size=8,
        h2o_decode_budget=4,
        h2o_decode_eviction_interval=3,
        h2o_prefill_budget=8,
        h2o_recent_ratio=0.5,
        h2o_prefill_score_window=2,
        rkv_compression_interval=2,
        rkv_observation_tokens=2,
        rkv_alpha=0.5,
        rkv_similarity_threshold=0.0,
        rkv_recent_similar_keep=0,
        rkv_max_redundancy_tokens=16,
        rkv_redundancy_window=0,
    )
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=1, num_slots=capacity, device=torch.device("cpu"))

    manager = object.__new__(manager_type)
    manager.config = config
    manager.hf_config = config.hf_config
    manager.device = torch.device("cpu")
    manager.tp_size = 1
    manager.head_dim = 256
    manager.max_model_len = capacity
    manager.max_buffer_rows = 1
    manager.num_layers = 1
    manager.num_kv_layers = 1
    manager.runtime_layout = RuntimeLayout.dense(1)
    manager.attention_cache_storage = storage
    manager.kv_cache = None
    manager._attention_key_materializers = {}
    manager.layer_num_slots = [capacity]
    manager.free_slots_stack_tensor = torch.arange(
        capacity, dtype=torch.int32
    ).view(1, capacity)
    manager.free_slots_stack = [manager.free_slots_stack_tensor[0]]
    manager._num_free_slots = [capacity]
    manager.buffer_req_to_token_slots_tensor = torch.zeros(
        (1, 1, capacity), dtype=torch.int32
    )
    manager.buffer_req_to_token_slots = [
        manager.buffer_req_to_token_slots_tensor[0]
    ]
    manager.seq_id_to_row = [{}]
    manager.free_rows = [deque([0])]
    manager.row_seq_lens = [np.zeros((1,), dtype=np.int32)]
    manager.layer_batch_states = [LayerBatchStates()]
    manager._decode_static_state_binding_key = None
    manager._decode_static_buffers = {}
    manager._decode_static_index_buffers = {}
    manager._prefill_attn_score_accumulators = {}
    manager._uniform_decode_metadata = False
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
    manager._rkv_query_cache_enabled = True
    manager._rkv_observation_tokens = 2
    manager._rkv_vectorized_prefill_query_cache = True
    manager._rkv_batch_clear_query_cache_rows = True
    manager._rkv_query_score_static_buffers = {}
    manager._rkv_query_cache = [
        torch.zeros((1, 2, 2, 256), dtype=torch.bfloat16)
    ]
    manager._rkv_query_positions = [
        torch.full((1, 2), -1, dtype=torch.int32)
    ]
    manager._pyramidkv_prefill_staging_active = False
    manager._pyramidkv_prefill_staging_was_active = False
    manager.pyramidkv_prefill_staging_kv_cache = None
    manager.pyramidkv_prefill_staging_num_slots = 0
    manager._pyramidkv_long_prefill_offload_prefetch_states = {}
    manager.raw_kv_offload_buffer = SimpleNamespace(
        release_layer=lambda **_kwargs: None,
    )
    return manager, storage, config


def _fill_latent_slots(storage, slots: torch.Tensor, values: list[int]) -> None:
    payload = storage.layer_payload(0)
    values_tensor = torch.tensor(values, dtype=torch.bfloat16)
    payload.latent_cache[slots] = values_tensor.view(-1, 1, 1).expand(
        -1, 1, 512
    )
    payload.rope_cache[slots] = (values_tensor + 100).view(
        -1, 1, 1
    ).expand(-1, 1, 64)


def test_snapkv_chain_resume_preserves_latent_payload_and_resets_prefill_scores():
    manager, storage, config = _latent_chain_manager(
        SnapKVCacheManager,
        "snapkv",
    )
    coordinator = ChainCacheCoordinator(config, manager)
    owner_tokens = list(range(8))
    owner = Sequence(owner_tokens)
    owner.seq_id = 0
    owner.chain_id = "chain-snap-latent"
    owner.chain_status = "created"
    owner.current_chunk_size = len(owner_tokens)
    created = coordinator.plan_admission(
        chain_id=owner.chain_id,
        seq_id=owner.seq_id,
        token_ids=owner_tokens,
    )
    assert created.status == "created"
    coordinator.apply_admission(created)

    manager._prepare_prefill([owner])
    owner_slots = manager.layer_batch_states[0].slot_mapping.clone().long()
    _fill_latent_slots(storage, owner_slots, owner_tokens)
    manager._prefill_attn_score_accumulators[(0, owner.seq_id)] = torch.tensor(
        [0.0, 1.0, 2.0, 9.0, 3.0, 8.0, 4.0, 0.0]
    )
    runtime = object.__new__(SnapKVRuntime)
    runtime.cache_manager = manager
    runtime.device = torch.device("cpu")
    runtime.sparse_method = "snapkv"
    runtime.num_layers = 1
    runtime.num_sink = 1
    runtime.num_recent = 1
    runtime.decode_keep_tokens = 2
    runtime.config = config
    runtime._is_kv_layer = lambda layer_idx: int(layer_idx) == 0
    runtime._kv_layer_index = lambda layer_idx: int(layer_idx)
    runtime._snapkv_prefill_eviction([owner])
    assert manager._prefill_attn_score_accumulators == {}
    resident_slots = manager.buffer_req_to_token_slots[0][0, :4].clone().long()
    assert resident_slots.tolist() == owner_slots[[0, 3, 5, 7]].tolist()
    payload = storage.layer_payload(0)
    torch.testing.assert_close(
        payload.latent_cache[resident_slots, 0, 0],
        torch.tensor([0, 3, 5, 7], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[resident_slots, 0, 0],
        torch.tensor([100, 103, 105, 107], dtype=torch.bfloat16),
    )
    coordinator.index.finish(
        owner.chain_id,
        token_ids=owner_tokens,
        processed_token_count=len(owner_tokens),
        physical_slots_by_layer=manager.chain_physical_residency(owner.seq_id),
    )

    resumed_tokens = owner_tokens + [8, 9]
    resumed_plan = coordinator.plan_admission(
        chain_id=owner.chain_id,
        seq_id=owner.seq_id,
        token_ids=resumed_tokens,
    )
    assert resumed_plan.status == "resumed"
    assert resumed_plan.reused_tokens == len(owner_tokens)
    coordinator.apply_admission(resumed_plan)
    resumed = Sequence(resumed_tokens)
    resumed.seq_id = owner.seq_id
    resumed.chain_id = owner.chain_id
    resumed.chain_status = "resumed"
    resumed.chain_reused_tokens = len(owner_tokens)
    resumed.num_prefilled_tokens = len(owner_tokens)
    resumed.current_chunk_size = 2
    manager._prefill_attn_score_accumulators[(0, resumed.seq_id)] = torch.full(
        (4,), 999.0
    )

    input_ids, positions, _ = manager._prepare_prefill([resumed])
    assert input_ids.tolist() == [8, 9]
    assert positions.tolist() == [8, 9]
    assert manager._prefill_attn_score_accumulators == {}
    resumed_row = manager.seq_id_to_row[0][resumed.seq_id]
    resumed_slots = manager.buffer_req_to_token_slots[0][
        resumed_row, :6
    ].clone().long()
    assert resumed_slots[:4].tolist() == resident_slots.tolist()
    _fill_latent_slots(storage, resumed_slots[4:], [8, 9])
    torch.testing.assert_close(
        payload.latent_cache[resumed_slots, 0, 0],
        torch.tensor([0, 3, 5, 7, 8, 9], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[resumed_slots, 0, 0],
        torch.tensor([100, 103, 105, 107, 108, 109], dtype=torch.bfloat16),
    )

    coordinator.index.finish(
        owner.chain_id,
        token_ids=resumed_tokens,
        processed_token_count=len(resumed_tokens),
        physical_slots_by_layer=manager.chain_physical_residency(resumed.seq_id),
    )
    coordinator.invalidate(owner.chain_id)
    manager.free_seq(resumed.seq_id)
    assert manager._prefill_attn_score_accumulators == {}
    assert manager.seq_id_to_row == [{}]
    assert manager._num_free_slots == [32]


def test_h2o_chain_resume_preserves_aligned_scores_and_cleans_side_state():
    manager, storage, config = _latent_chain_manager(H2OCacheManager, "h2o")
    coordinator = ChainCacheCoordinator(config, manager)
    owner_tokens = list(range(6))
    owner = Sequence(owner_tokens)
    owner.seq_id = 0
    owner.chain_id = "chain-h2o-latent"
    owner.chain_status = "created"
    owner.current_chunk_size = len(owner_tokens)
    created = coordinator.plan_admission(
        chain_id=owner.chain_id,
        seq_id=owner.seq_id,
        token_ids=owner_tokens,
    )
    assert created.status == "created"
    coordinator.apply_admission(created)

    manager._prepare_prefill([owner])
    owner_slots = manager.layer_batch_states[0].slot_mapping.clone().long()
    _fill_latent_slots(storage, owner_slots, owner_tokens)
    manager._h2o_scores[(0, owner.seq_id)] = torch.tensor(
        [1.0, 9.0, 2.0, 8.0, 0.0, 0.0]
    )
    manager.evict_after_prefill([owner])
    assert manager.row_seq_lens[0].tolist() == [4]
    resident_slots = manager.buffer_req_to_token_slots[0][0, :4].clone().long()
    payload = storage.layer_payload(0)
    torch.testing.assert_close(
        payload.latent_cache[resident_slots, 0, 0],
        torch.tensor([1, 3, 4, 5], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[resident_slots, 0, 0],
        torch.tensor([101, 103, 104, 105], dtype=torch.bfloat16),
    )
    assert manager._h2o_scores[(0, owner.seq_id)].tolist() == [9.0, 8.0, 0.0, 0.0]
    assert manager._h2o_final_prefill_workspace is None
    coordinator.index.finish(
        owner.chain_id,
        token_ids=owner_tokens,
        processed_token_count=len(owner_tokens),
        physical_slots_by_layer=manager.chain_physical_residency(owner.seq_id),
    )

    resumed_tokens = owner_tokens + [6, 7]
    resumed_plan = coordinator.plan_admission(
        chain_id=owner.chain_id,
        seq_id=owner.seq_id,
        token_ids=resumed_tokens,
    )
    assert resumed_plan.status == "resumed"
    assert resumed_plan.reused_tokens == len(owner_tokens)
    coordinator.apply_admission(resumed_plan)
    resumed = Sequence(resumed_tokens)
    resumed.seq_id = owner.seq_id
    resumed.chain_id = owner.chain_id
    resumed.chain_status = "resumed"
    resumed.chain_reused_tokens = len(owner_tokens)
    resumed.num_prefilled_tokens = len(owner_tokens)
    resumed.current_chunk_size = 2

    input_ids, positions, _ = manager._prepare_prefill([resumed])
    assert input_ids.tolist() == [6, 7]
    assert positions.tolist() == [6, 7]
    assert manager._h2o_scores[(0, resumed.seq_id)].tolist() == [9.0, 8.0, 0.0, 0.0]
    resumed_row = manager.seq_id_to_row[0][resumed.seq_id]
    resumed_slots = manager.buffer_req_to_token_slots[0][
        resumed_row, :6
    ].clone().long()
    assert resumed_slots[:4].tolist() == resident_slots.tolist()
    _fill_latent_slots(storage, resumed_slots[4:], [6, 7])
    manager._h2o_scores[(0, resumed.seq_id)] = manager._accumulate_score(
        manager._h2o_scores[(0, resumed.seq_id)],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 7.0, 6.0]),
        new_len=6,
        weight=1.0,
    )
    manager.evict_after_prefill([resumed])
    final_slots = manager.buffer_req_to_token_slots[0][
        resumed_row, :4
    ].clone().long()
    torch.testing.assert_close(
        payload.latent_cache[final_slots, 0, 0],
        torch.tensor([1, 3, 6, 7], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[final_slots, 0, 0],
        torch.tensor([101, 103, 106, 107], dtype=torch.bfloat16),
    )
    assert manager._h2o_scores[(0, resumed.seq_id)].tolist() == [9.0, 8.0, 7.0, 6.0]
    assert manager._h2o_counters["final_prefill_evictions"] == 2
    assert manager._h2o_final_prefill_workspace is None

    coordinator.index.finish(
        owner.chain_id,
        token_ids=resumed_tokens,
        processed_token_count=len(resumed_tokens),
        physical_slots_by_layer=manager.chain_physical_residency(resumed.seq_id),
    )
    coordinator.invalidate(owner.chain_id)
    manager.free_seq(resumed.seq_id)
    assert manager._h2o_scores == {}
    assert manager._prefill_attn_score_accumulators == {}
    assert manager.seq_id_to_row == [{}]
    assert manager._num_free_slots == [32]


def _rkv_prefill_view(manager, storage) -> PrefillComputeView:
    state = manager.layer_batch_states[0]
    return PrefillComputeView(
        meta=AttentionViewMeta(
            active_slots=state.slot_mapping,
            req_indices=state.req_indices,
            context_lens=state.context_lens,
            max_context_len=state.max_context_len,
        ),
        payload=storage.layer_payload(0),
    )


def test_rkv_chain_resume_rebuilds_query_observations_without_cross_turn_leak():
    manager, storage, config = _latent_chain_manager(RKVCacheManager, "rkv")
    coordinator = ChainCacheCoordinator(config, manager)
    owner_tokens = list(range(6))
    owner = Sequence(owner_tokens)
    owner.seq_id = 0
    owner.chain_id = "chain-rkv-latent"
    owner.chain_status = "created"
    owner.current_chunk_size = len(owner_tokens)
    created = coordinator.plan_admission(
        chain_id=owner.chain_id,
        seq_id=owner.seq_id,
        token_ids=owner_tokens,
    )
    assert created.status == "created"
    coordinator.apply_admission(created)

    manager._prepare_prefill([owner])
    owner_slots = manager.layer_batch_states[0].slot_mapping.clone().long()
    _fill_latent_slots(storage, owner_slots, owner_tokens)
    owner_q = torch.arange(6, dtype=torch.bfloat16).view(6, 1, 1).expand(
        6, 2, 256
    )
    manager.record_prefill_query(
        0,
        owner_q,
        _rkv_prefill_view(manager, storage),
        b_start_loc=torch.tensor([0], dtype=torch.int32),
        chunk_lens=torch.tensor([6], dtype=torch.int32),
    )
    assert manager._rkv_query_positions[0][0].tolist() == [-1, -1]

    manager.free_part_slots(
        0,
        owner,
        torch.tensor([0, 2, 4, 5], dtype=torch.long),
        keep_indices_sorted=True,
    )
    assert manager._rkv_query_positions[0][0].tolist() == [-1, -1]
    resident_slots = manager.buffer_req_to_token_slots[0][0, :4].clone().long()
    payload = storage.layer_payload(0)
    torch.testing.assert_close(
        payload.latent_cache[resident_slots, 0, 0],
        torch.tensor([0, 2, 4, 5], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[resident_slots, 0, 0],
        torch.tensor([100, 102, 104, 105], dtype=torch.bfloat16),
    )
    coordinator.index.finish(
        owner.chain_id,
        token_ids=owner_tokens,
        processed_token_count=len(owner_tokens),
        physical_slots_by_layer=manager.chain_physical_residency(owner.seq_id),
    )

    resumed_tokens = owner_tokens + [6, 7]
    resumed_plan = coordinator.plan_admission(
        chain_id=owner.chain_id,
        seq_id=owner.seq_id,
        token_ids=resumed_tokens,
    )
    assert resumed_plan.status == "resumed"
    assert resumed_plan.reused_tokens == len(owner_tokens)
    coordinator.apply_admission(resumed_plan)
    resumed = Sequence(resumed_tokens)
    resumed.seq_id = owner.seq_id
    resumed.chain_id = owner.chain_id
    resumed.chain_status = "resumed"
    resumed.chain_reused_tokens = len(owner_tokens)
    resumed.num_prefilled_tokens = len(owner_tokens)
    resumed.current_chunk_size = 2

    input_ids, positions, _ = manager._prepare_prefill([resumed])
    assert input_ids.tolist() == [6, 7]
    assert positions.tolist() == [6, 7]
    assert manager._rkv_query_positions[0][0].tolist() == [-1, -1]
    resumed_row = manager.seq_id_to_row[0][resumed.seq_id]
    resumed_slots = manager.buffer_req_to_token_slots[0][
        resumed_row, :6
    ].clone().long()
    assert resumed_slots[:4].tolist() == resident_slots.tolist()
    _fill_latent_slots(storage, resumed_slots[4:], [6, 7])
    resumed_q = torch.tensor([106, 107], dtype=torch.bfloat16).view(
        2, 1, 1
    ).expand(2, 2, 256)
    manager.record_prefill_query(
        0,
        resumed_q,
        _rkv_prefill_view(manager, storage),
        b_start_loc=torch.tensor([0], dtype=torch.int32),
        chunk_lens=torch.tensor([2], dtype=torch.int32),
    )
    assert manager._rkv_query_positions[0][0].tolist() == [-1, -1]
    manager.free_part_slots(
        0,
        resumed,
        torch.tensor([0, 2, 4, 5], dtype=torch.long),
        keep_indices_sorted=True,
    )
    assert manager._rkv_query_positions[0][0].tolist() == [-1, -1]
    final_slots = manager.buffer_req_to_token_slots[0][
        resumed_row, :4
    ].clone().long()
    torch.testing.assert_close(
        payload.latent_cache[final_slots, 0, 0],
        torch.tensor([0, 4, 6, 7], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[final_slots, 0, 0],
        torch.tensor([100, 104, 106, 107], dtype=torch.bfloat16),
    )

    coordinator.index.finish(
        owner.chain_id,
        token_ids=resumed_tokens,
        processed_token_count=len(resumed_tokens),
        physical_slots_by_layer=manager.chain_physical_residency(resumed.seq_id),
    )
    coordinator.invalidate(owner.chain_id)
    manager.free_seq(resumed.seq_id)
    assert manager._rkv_query_positions[0][0].tolist() == [-1, -1]
    assert manager._prefill_attn_score_accumulators == {}
    assert manager.seq_id_to_row == [{}]
    assert manager._num_free_slots == [32]
