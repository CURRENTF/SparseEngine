from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import Glm4MoeLiteConfig

from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager import LayerBatchStates
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
from sparseengine.engine.cache_manager.standard import StandardCacheManager
from sparseengine.engine.cache_manager.storage import MlaLatentStorage
from sparseengine.engine.cache_manager.methods.streamingllm import (
    StreamingLLMCacheManager,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import (
    LayerBatchSparseState,
    SparseController,
)
from sparseengine.engine.sparse_methods.snapkv import SnapKVRuntime
from sparseengine.engine.sparse_methods.streamingllm import StreamingLLMRuntime
from sparseengine.utils.context import reset_context, set_context

from glm_test_helpers import _single_rank_parallel_context


def test_glm_sparse_controller_uses_full_mla_qk_softmax_scale():
    hf_config = Glm4MoeLiteConfig(
        hidden_size=64,
        num_attention_heads=20,
        num_key_value_heads=20,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
    )
    config = SimpleNamespace(
        sparse_method="omnikv",
        obs_layer_ids=[],
        full_attention_layers=[],
        runtime_layout=RuntimeLayout.dense(1),
        hf_config=hf_config,
        tensor_parallel_size=1,
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=1,
        sparse_attn_score_dtype="float32",
    )
    manager = SimpleNamespace(device=torch.device("cpu"))

    controller = SparseController(config, manager)

    assert hf_config.head_dim == 64
    assert controller.runtime.attn_softmax_scale == pytest.approx(256**-0.5)
    assert controller.runtime.attn_softmax_scale != pytest.approx(64**-0.5)


def test_glm_rkv_query_cache_allocates_and_records_full_qk_head_width():
    hf_config = Glm4MoeLiteConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=20,
        num_key_value_heads=20,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        dtype=torch.bfloat16,
    )
    config = SimpleNamespace(
        hf_config=hf_config,
        runtime_layout=RuntimeLayout.dense(1),
        sparse_method="rkv",
        max_model_len=8,
        max_num_seqs_in_gpu=1,
        num_kvcache_slots=8,
        pyramid_layer_ratios=None,
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=1,
        rkv_compression_interval=1,
        rkv_observation_tokens=1,
    )
    cpu_platform = SimpleNamespace(
        get_device=lambda _rank: torch.device("cpu"),
        supports_pin_memory=lambda: False,
    )
    with (
        patch(
            "sparseengine.engine.cache_manager.base.platforms.get_current_platform",
            return_value=cpu_platform,
        ),
        patch(
            "sparseengine.engine.cache_manager.methods.snapkv.create_attention_cache_storage",
            return_value=SimpleNamespace(),
        ),
        patch.object(SnapKVCacheManager, "allocate_kv_cache", autospec=True),
    ):
        manager = RKVCacheManager(config, _single_rank_parallel_context())

    assert hf_config.head_dim == 64
    assert manager.head_dim == 256
    assert manager._rkv_query_cache[0].shape == (1, 1, 20, 256)
    manager.layer_batch_states[0] = LayerBatchStates(
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([4], dtype=torch.int32),
    )
    q = torch.arange(20 * 256, dtype=torch.float32).reshape(1, 20, 256)
    q = q.to(torch.bfloat16)

    manager.record_decode_query(0, q)

    assert manager._rkv_query_positions[0].tolist() == [[3]]
    torch.testing.assert_close(manager._rkv_query_cache[0][0, 0], q[0])


def _latent_snap_family_manager(manager_type, *, row_len: int):
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=1, num_slots=16, device=torch.device("cpu"))
    manager = object.__new__(manager_type)
    manager.attention_cache_storage = storage
    manager.kv_cache = None
    manager.device = torch.device("cpu")
    manager.num_layers = 1
    manager.num_kv_layers = 1
    manager.runtime_layout = RuntimeLayout.dense(1)
    manager._uniform_decode_metadata = True
    manager.buffer_req_to_token_slots_tensor = torch.zeros(
        (1, 1, 16), dtype=torch.int32
    )
    manager.buffer_req_to_token_slots_tensor[0, 0, :row_len] = torch.arange(
        row_len, dtype=torch.int32
    )
    manager.buffer_req_to_token_slots = [
        manager.buffer_req_to_token_slots_tensor[0]
    ]
    manager.seq_id_to_row = [{0: 0}]
    manager.row_seq_lens = [np.asarray([row_len], dtype=np.int32)]
    manager.free_slots_stack_tensor = None
    manager.free_slots_stack = [torch.zeros((16,), dtype=torch.int32)]
    manager._num_free_slots = [0]

    payload = storage.layer_payload(0)
    for slot in range(row_len):
        payload.latent_cache[slot].fill_(slot)
        payload.rope_cache[slot].fill_(slot + 100)
    return manager, payload


def test_streamingllm_budget_trigger_preserves_mla_latent_slot_payloads():
    manager, payload = _latent_snap_family_manager(
        StreamingLLMCacheManager,
        row_len=8,
    )

    seq = Sequence(list(range(8)))
    seq.seq_id = 0
    seq.num_prefilled_tokens = 0
    seq.current_chunk_size = 8
    runtime = object.__new__(StreamingLLMRuntime)
    runtime.cache_manager = manager
    runtime.device = torch.device("cpu")
    runtime.num_layers = 1
    runtime.num_sink = 2
    runtime.num_recent = 3
    runtime.layer_batch_sparse_states = {
        0: SimpleNamespace(
            context_lens=torch.tensor([8], dtype=torch.int32),
            max_context_len=8,
        )
    }
    runtime._is_kv_layer = lambda layer_idx: int(layer_idx) == 0

    runtime._streamingllm_prefill_eviction([seq])

    active_slots = manager.buffer_req_to_token_slots[0][0, :5].long()
    assert active_slots.tolist() == [0, 1, 5, 6, 7]
    assert manager.row_seq_lens[0].tolist() == [5]
    assert manager.free_slots_stack[0][:3].tolist() == [2, 3, 4]
    assert manager._num_free_slots == [3]
    torch.testing.assert_close(
        payload.latent_cache[active_slots, 0, 0],
        torch.tensor([0, 1, 5, 6, 7], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[active_slots, 0, 0],
        torch.tensor([100, 101, 105, 106, 107], dtype=torch.bfloat16),
    )


def test_snapkv_score_budget_trigger_preserves_mla_latent_slot_payloads():
    manager, payload = _latent_snap_family_manager(
        SnapKVCacheManager,
        row_len=8,
    )
    manager._prefill_attn_score_accumulators = {
        (0, 0): torch.tensor([0.0, 1.0, 2.0, 9.0, 3.0, 8.0, 4.0, 0.0])
    }
    seq = Sequence(list(range(8)))
    seq.seq_id = 0
    seq.num_prefilled_tokens = 0
    seq.current_chunk_size = 8
    runtime = object.__new__(SnapKVRuntime)
    runtime.cache_manager = manager
    runtime.device = torch.device("cpu")
    runtime.sparse_method = "snapkv"
    runtime.num_layers = 1
    runtime.num_sink = 1
    runtime.num_recent = 1
    runtime.decode_keep_tokens = 2
    runtime.config = SimpleNamespace(
        snapkv_num_full_layers=0,
        pyramid_layer_ratios=None,
        pool_kernel_size=1,
    )
    runtime._is_kv_layer = lambda layer_idx: int(layer_idx) == 0
    runtime._kv_layer_index = lambda layer_idx: int(layer_idx)

    runtime._snapkv_prefill_eviction([seq])

    active_slots = manager.buffer_req_to_token_slots[0][0, :4].long()
    assert active_slots.tolist() == [0, 3, 5, 7]
    assert manager.row_seq_lens[0].tolist() == [4]
    assert sorted(manager.free_slots_stack[0][:4].tolist()) == [1, 2, 4, 6]
    assert manager._num_free_slots == [4]
    assert manager._prefill_attn_score_accumulators == {}
    torch.testing.assert_close(
        payload.latent_cache[active_slots, 0, 0],
        torch.tensor([0, 3, 5, 7], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[active_slots, 0, 0],
        torch.tensor([100, 103, 105, 107], dtype=torch.bfloat16),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_omnikv_observation_selects_mla_latent_active_slots():
    device = torch.device("cuda")
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=2, num_slots=16, device=device)
    manager = object.__new__(StandardCacheManager)
    manager.attention_cache_storage = storage
    manager.kv_cache = None
    manager.runtime_layout = RuntimeLayout.dense(2)
    manager.buffer_req_to_token_slots = torch.zeros(
        (1, 16), dtype=torch.int32, device=device
    )
    physical_slots = torch.tensor(
        [8, 3, 11, 1, 14, 7], dtype=torch.int32, device=device
    )
    manager.buffer_req_to_token_slots[0, :6] = physical_slots
    payload = storage.layer_payload(1)
    for slot in physical_slots.tolist():
        payload.latent_cache[slot].fill_(slot)
        payload.rope_cache[slot].fill_(slot + 100)

    hf_config = Glm4MoeLiteConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=20,
        num_key_value_heads=20,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        dtype=torch.bfloat16,
    )
    controller = SparseController(
        SimpleNamespace(
            sparse_method="omnikv",
            obs_layer_ids=[0],
            full_attention_layers=[0],
            runtime_layout=RuntimeLayout.dense(2),
            hf_config=hf_config,
            tensor_parallel_size=1,
            sink_keep_tokens=1,
            recent_keep_tokens=1,
            decode_keep_tokens=2,
            sparse_attn_score_dtype="float32",
        ),
        manager,
    )
    assert controller.runtime.attn_softmax_scale == pytest.approx(256**-0.5)
    controller.runtime.layer_batch_sparse_states = {
        0: LayerBatchSparseState(
            attn_score=torch.tensor(
                [
                    [
                        [0.0, 10.0, 1.0, 9.0, 0.0, -1.0],
                        [0.0, 0.0, 8.0, 1.0, 7.0, -1.0],
                    ]
                ],
                dtype=torch.float32,
                device=device,
            ),
            req_indices=torch.tensor([0], dtype=torch.int32, device=device),
            context_lens=torch.tensor([6], dtype=torch.int32, device=device),
            max_context_len=6,
        ),
        1: LayerBatchSparseState(),
    }
    controller.runtime._is_kv_layer = lambda layer_idx: 0 <= int(layer_idx) < 2
    observer = controller.runtime.layer_batch_sparse_states[0]
    controller.runtime._omnikv_score_provider.prepare(observer.attn_score, slot=id(observer))

    set_context(
        False,
        cache_manager=manager,

        seqs=[Sequence([1])],
    )
    try:
        controller.on_layer_end(0, SimpleNamespace(is_prefill=False))
    finally:
        reset_context()

    target = controller.layer_batch_sparse_states[1]
    assert target.active_indices is not None
    assert target.active_slots is not None
    assert target.context_lens is not None
    assert target.context_lens.tolist() == [4]
    logical_keep = target.active_indices[0, :4].tolist()
    assert logical_keep[0] == 0
    assert set(logical_keep[1:3]) == {1, 2}
    assert logical_keep[3] == 5
    selected_slots = target.active_slots[0, :4].long()
    expected_slots = manager.buffer_req_to_token_slots[0].index_select(
        0,
        target.active_indices[0, :4].long(),
    )
    torch.testing.assert_close(selected_slots, expected_slots.long())
    torch.testing.assert_close(
        payload.latent_cache[selected_slots, 0, 0],
        selected_slots.to(torch.bfloat16),
    )
    torch.testing.assert_close(
        payload.rope_cache[selected_slots, 0, 0],
        selected_slots.to(torch.bfloat16) + 100,
    )


def test_snapkv_latent_score_handoff_preserves_final_window_and_accumulation():
    # Precomputed MLA scores must enter the same final-chunk accumulator and
    # selection lifecycle as explicit KV, without attempting another QK kernel.
    from sparseengine.engine.cache_manager.base import (
        AttentionViewMeta,
        PrefillComputeView,
    )

    manager, payload = _latent_snap_family_manager(SnapKVCacheManager, row_len=8)
    manager.config = SimpleNamespace(
        sparse_method="snapkv",
        snapkv_num_full_layers=0,
        snapkv_window_size=2,
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=2,
        sparse_prefill_score_mode="logits",
        sparse_attn_score_dtype="float32",
    )
    manager._prefill_attn_score_accumulators = {}
    manager._prefill_context_lens_cpu_by_layer = {0: (8,)}
    seq = Sequence(list(range(8)))
    seq.seq_id = 0
    seq.current_chunk_size = 4
    seq.num_prefilled_tokens = 0
    assert manager.prefill_score_request(0, [seq]) is None
    seq.num_prefilled_tokens = 4
    request = manager.prefill_score_request(0, [seq])
    assert request.query_ranges == ((6, 8),)
    assert request.candidate_start == request.recent_keep_tokens == 1
    score = torch.tensor([[-torch.inf, 1.0, 2.0, 9.0, 3.0, 8.0, 4.0, -torch.inf]])
    view = PrefillComputeView(
        AttentionViewMeta(
            manager.buffer_req_to_token_slots[0],
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
        ),
        payload,
        score,
    )
    set_context(True, cache_manager=manager, seqs=[seq])
    try:
        with patch.object(
            manager,
            "_run_prefill_score",
            side_effect=AssertionError("unexpected explicit QK"),
        ):
            manager.collect_prefill_attention_score(
                0,
                torch.empty(4, 20, 256),
                view,
                b_start_loc=torch.tensor([0], dtype=torch.int32),
                chunk_lens=torch.tensor([4], dtype=torch.int32),
            )
        torch.testing.assert_close(
            manager._prefill_attn_score_accumulators[0, 0], score[0]
        )
    finally:
        reset_context()
