from __future__ import annotations

import json
import os
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from sparseengine.config import RuntimeLayout
from sparseengine.configs.cuda_graph import (
    _default_decode_cuda_graph_capture_sizes,
    build_decode_cuda_graph_startup_plan,
)
from sparseengine.models.layout import resolve_attention_qk_head_dim
from sparseengine.method_registry import sparse_decode_attention_score_kind
from sparseengine.distributed import ParallelContext
from sparseengine.engine.cache_manager import LayerBatchStates
from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
from sparseengine.engine.cache_manager.methods.omnikv.manager import OmniKVCacheManager
from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.cache_manager.standard import StandardCacheManager
from sparseengine.engine.cache_manager.storage import MlaLatentStorage
from sparseengine.engine.cache_manager.methods.streamingllm import (
    StreamingLLMCacheManager,
)
from sparseengine.engine.decode_cuda_graph import DecodeCudaGraphRunner
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import SparseController
from sparseengine.layers.mla_attention import MLAAttention
from sparseengine.layers.rotary_embedding import RotaryEmbedding
from sparseengine.models.glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteSparseMoeBlock,
)
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.mla_attention import MlaAttentionOpSpec
from sparseengine.utils.context import get_context

from glm_test_helpers import (
    _glm_hf_config,
    _single_rank_parallel_context,
    _tensor_sha256,
)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("pyramidkv", AttentionScoreKind.NONE),
        ("omnikv", AttentionScoreKind.RAW_QK_PER_HEAD),
        ("skipkv", AttentionScoreKind.RAW_QK_PER_HEAD),
        ("deltakv", AttentionScoreKind.RAW_QK_PER_HEAD),
        ("vanilla", AttentionScoreKind.NONE),
    ],
)
def test_glm_sparse_method_declares_exact_decode_score_contract(
    method,
    expected,
):
    assert sparse_decode_attention_score_kind(method) is expected


def test_dense_startup_graph_plan_has_one_graph_per_batch() -> None:
    config = SimpleNamespace(
        decode_graph_capture_sizes=[1, 2, 4, 8],
        decode_graph_startup_capture_limit=8,
        sparse_method="",
        max_model_len=262144,
        sink_keep_tokens=64,
        decode_keep_tokens=4096,
        recent_keep_tokens=512,
    )

    assert build_decode_cuda_graph_startup_plan(config) == [
        (8, 262144),
        (4, 262144),
        (2, 262144),
        (1, 262144),
    ]


def test_sparse_startup_graph_plan_has_one_graph_per_batch_and_path() -> None:
    batches = _default_decode_cuda_graph_capture_sizes(64)
    config = SimpleNamespace(
        decode_graph_capture_sizes=batches,
        decode_graph_startup_capture_limit=48,
        sparse_method="snapkv",
        sink_keep_tokens=64,
        decode_keep_tokens=4096,
        recent_keep_tokens=512,
        max_model_len=65536,
    )

    plan = build_decode_cuda_graph_startup_plan(config)

    assert len(plan) == len(batches)
    assert {batch for batch, _ in plan} == set(batches)
    assert {context for _, context in plan} == {config.max_model_len}



def _make_glm_graph_lane(
    *,
    device: torch.device,
    parallel_context: ParallelContext,
    attention_state: dict[str, torch.Tensor] | None,
    embedding_state: dict[str, torch.Tensor] | None,
    head_state: dict[str, torch.Tensor] | None,
    initial_latent: torch.Tensor,
    initial_rope: torch.Tensor,
):
    hf_config = _glm_hf_config(
        num_hidden_layers=1,
        max_position_embeddings=128,
    )
    hf_config.rms_norm_eps = 1e-6
    spec = MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=1,
        cuda_graph=True,
        context_capacity=128,
        batch_capacity=1,
    )
    mla_attention = MLAAttention.bind(
        spec=spec,
        device=device,
        max_batch_size=1,
        prefill_workspace_bytes=1024 * 1024,
        hidden_size=64,
        projection_chunk_size=8,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with (
            patch(
                "sparseengine.models.glm4_moe_lite.get_parallel_context",
                return_value=parallel_context,
            ),
            patch(
                "sparseengine.layers.linear.get_parallel_context",
                return_value=parallel_context,
            ),
            torch.device(device),
        ):
            attention = Glm4MoeLiteAttention(
                hf_config,
                mla_attention,
                projection_chunk_size=8,
            )
            embedding = nn.Embedding(128, 64)
            lm_head = nn.Linear(64, 128, bias=False)
            rotary = RotaryEmbedding(
                64,
                64,
                128,
                1_000_000.0,
                backend="torch",
                interleaved=True,
            )
    finally:
        torch.set_default_dtype(previous_dtype)

    if attention_state is None:
        generator = torch.Generator(device=device).manual_seed(941)
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=device,
                        generator=generator,
                    )
                    * 0.02
                )
            embedding.weight.copy_(
                torch.randn(
                    embedding.weight.shape,
                    dtype=embedding.weight.dtype,
                    device=device,
                    generator=generator,
                )
                * 0.02
            )
            lm_head.weight.copy_(
                torch.randn(
                    lm_head.weight.shape,
                    dtype=lm_head.weight.dtype,
                    device=device,
                    generator=generator,
                )
                * 0.02
            )
    else:
        assert embedding_state is not None and head_state is not None
        attention.load_state_dict(attention_state)
        embedding.load_state_dict(embedding_state)
        lm_head.load_state_dict(head_state)

    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=1, num_slots=16, device=device)
    assert storage.latent_cache is not None and storage.rope_cache is not None
    storage.latent_cache.zero_()
    storage.rope_cache.zero_()
    storage.latent_cache[0, :3].copy_(initial_latent)
    storage.rope_cache[0, :3].copy_(initial_rope)

    runtime_config = SimpleNamespace(
        sparse_method="",
        runtime_layout=RuntimeLayout.dense(1),
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=20,
            hidden_size=64,
            head_dim=256,
            dtype=torch.bfloat16,
        ),
        obs_layer_ids=[],
        full_attention_layers=[],
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=0,
        sparse_attn_score_dtype="float32",
        tensor_parallel_size=1,
        max_model_len=128,
        decode_graph=True,
    )
    manager = object.__new__(StandardCacheManager)
    manager.config = runtime_config
    manager.parallel_context = parallel_context
    manager.device = device
    manager.runtime_layout = runtime_config.runtime_layout
    manager.num_layers = 1
    manager.num_kv_layers = 1
    manager.max_model_len = 128
    manager.attention_cache_storage = storage
    manager.kv_cache = None
    manager._decode_static_max_context_len = None
    manager._attention_key_materializers = {}
    manager.free_slots_stack = torch.empty(16, dtype=torch.int32, device=device)
    manager.free_slots_stack[:13].copy_(
        torch.arange(3, 16, dtype=torch.int32, device=device)
    )
    manager._num_free_slots = 13
    manager.buffer_req_to_token_slots = torch.zeros(
        (1, 128),
        dtype=torch.int32,
        device=device,
    )
    manager.buffer_req_to_token_slots[0, :3].copy_(
        torch.arange(3, dtype=torch.int32, device=device)
    )
    sequence = Sequence([5, 7, 11, 13])
    sequence.num_prefilled_tokens = sequence.num_prompt_tokens
    sequence.temperature = 0.0
    sequence.max_tokens = 8
    manager.seq_id_to_row = {sequence.seq_id: 0}
    manager.free_rows = deque()
    manager.row_seq_lens = np.asarray([3], dtype=np.int32)
    manager.layer_batch_state = LayerBatchStates()
    manager._decode_static_index_buffers = {}
    manager.enable_prefix_caching = False
    manager.prefix_cache_block_size = 4
    manager.prefix_cache = None
    manager.seq_id_to_prefix_blocks = {}
    manager.seq_id_to_cached_ranges = {}
    manager._scheduler_capacity_snapshot_depth = 0
    manager._scheduler_freeable_block_ids = None
    manager.prefix_offload_controller = None
    manager._prefix_offload_step_h2d_operations = {}
    manager._prefix_write_through_candidates = {}
    manager._init_prefix_cache_runtime()

    sparse_controller = SparseController(runtime_config, manager)
    runtime_state = RuntimeState(runtime_config, manager)

    def run_model(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        assert not is_prefill
        get_context().now_layer_idx = 0
        hidden_states = embedding(input_ids)
        hidden_states = attention(positions, hidden_states, rotary)
        return lm_head(hidden_states)

    runner = DecodeCudaGraphRunner(
        runtime_state=runtime_state,
        cache_manager=manager,
        recurrent_state_manager=None,
        sparse_controller=sparse_controller,
        run_model=run_model,
        method="",
        capture_sizes=[1],
    )
    return SimpleNamespace(
        attention=attention,
        embedding=embedding,
        lm_head=lm_head,
        manager=manager,
        storage=storage,
        sequence=sequence,
        runner=runner,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_glm_vanilla_decode_cuda_graph_matches_static_eager():
    device = torch.device("cuda")
    parallel_context = _single_rank_parallel_context()
    generator = torch.Generator(device=device).manual_seed(937)
    initial_latent = torch.randn(
        (3, 1, 512),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    initial_rope = torch.randn(
        (3, 1, 64),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    eager = _make_glm_graph_lane(
        device=device,
        parallel_context=parallel_context,
        attention_state=None,
        embedding_state=None,
        head_state=None,
        initial_latent=initial_latent,
        initial_rope=initial_rope,
    )
    graph = _make_glm_graph_lane(
        device=device,
        parallel_context=parallel_context,
        attention_state=eager.attention.state_dict(),
        embedding_state=eager.embedding.state_dict(),
        head_state=eager.lm_head.state_dict(),
        initial_latent=initial_latent,
        initial_rope=initial_rope,
    )

    step_evidence = []
    captured_graph = None
    for step in range(2):
        eager_logits = eager.runner.run_eager_static([eager.sequence])
        graph_logits, graph_token_ids = graph.runner.run(
            [graph.sequence],
            capture_sampling=True,
        )
        torch.cuda.synchronize()
        assert eager_logits is not None
        assert graph_logits is not None
        assert graph_token_ids is not None
        torch.testing.assert_close(graph_logits, eager_logits, rtol=0, atol=0)
        eager_token_ids = eager_logits.argmax(dim=-1)
        torch.testing.assert_close(graph_token_ids, eager_token_ids, rtol=0, atol=0)

        graph_states = [
            state
            for state in graph.runner._graphs.values()
            if state.graph is not None
        ]
        assert len(graph_states) == 1
        if captured_graph is None:
            captured_graph = graph_states[0].graph
        else:
            assert graph_states[0].graph is captured_graph

        eager_row_len = int(eager.manager.row_seq_lens[0])
        graph_row_len = int(graph.manager.row_seq_lens[0])
        assert eager_row_len == graph_row_len
        eager_slots = eager.manager.buffer_req_to_token_slots[0, :eager_row_len].long()
        graph_slots = graph.manager.buffer_req_to_token_slots[0, :graph_row_len].long()
        eager_latent = eager.storage.latent_cache[0].index_select(0, eager_slots)
        graph_latent = graph.storage.latent_cache[0].index_select(0, graph_slots)
        eager_rope = eager.storage.rope_cache[0].index_select(0, eager_slots)
        graph_rope = graph.storage.rope_cache[0].index_select(0, graph_slots)
        torch.testing.assert_close(graph_latent, eager_latent, rtol=0, atol=0)
        torch.testing.assert_close(graph_rope, eager_rope, rtol=0, atol=0)
        step_evidence.append(
            {
                "step": step + 1,
                "token": int(graph_token_ids[0].item()),
                "logits_sha256": _tensor_sha256(graph_logits),
                "latent_sha256": _tensor_sha256(graph_latent),
                "rope_sha256": _tensor_sha256(graph_rope),
            }
        )
        if step == 0:
            next_token = int(graph_token_ids[0].item())
            eager.sequence.append_token(next_token)
            graph.sequence.append_token(next_token)

    evidence = {
        "harness_scope": "tiny_random_attention_component",
        "real_checkpoint": False,
        "graph_active": captured_graph is not None,
        "graph_count": sum(
            state.graph is not None for state in graph.runner._graphs.values()
        ),
        "capture_count": graph.runner.capture_count,
        "replay_count": graph.runner.replay_count,
        "eager_static_count": graph.runner.eager_static_count,
        "force_eager": graph.manager.decode_graph_force_eager(),
        "force_eager_count": graph.runner.force_eager_count,
        "fallback": graph.runner.force_eager_count > 0,
        "steps": step_evidence,
    }
    assert evidence["capture_count"] == 1
    assert evidence["replay_count"] == 2
    assert evidence["eager_static_count"] == 0
    assert evidence["force_eager_count"] == 0
    assert evidence["fallback"] is False
    print("GLM_VANILLA_CUDA_GRAPH_EVIDENCE=" + json.dumps(evidence, sort_keys=True))


def _make_glm_full_graph_lane(
    *,
    device: torch.device,
    parallel_context: ParallelContext,
    model_state: dict[str, torch.Tensor] | None,
    initial_latent: torch.Tensor,
    initial_rope: torch.Tensor,
):
    model_config = _glm_hf_config(
        num_hidden_layers=2,
        mlp_layer_types=["dense", "sparse"],
        max_position_embeddings=129,
        moe_intermediate_size=16,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=torch.bfloat16,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
        },
    )
    mla_spec = MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=1,
        cuda_graph=True,
        context_capacity=128,
        batch_capacity=1,
    )
    mla_attention = MLAAttention.bind(
        spec=mla_spec,
        device=device,
        max_batch_size=1,
        prefill_workspace_bytes=1024 * 1024,
        hidden_size=64,
        projection_chunk_size=8,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with (
            patch(
                "sparseengine.models.glm4_moe_lite.get_parallel_context",
                return_value=parallel_context,
            ),
            patch(
                "sparseengine.layers.linear.get_parallel_context",
                return_value=parallel_context,
            ),
            patch(
                "sparseengine.layers.embed_head.get_parallel_context",
                return_value=parallel_context,
            ),
            torch.device(device),
        ):
            model = Glm4MoeLiteForCausalLM(
                model_config,
                mla_attention=mla_attention,
                mlp_chunk_size=8,
                decode_graph=True,
            )
    finally:
        torch.set_default_dtype(previous_dtype)

    if model_state is None:
        generator = torch.Generator(device=device).manual_seed(953)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm.weight" in name or "layernorm.weight" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.copy_(
                        torch.randn(
                            parameter.shape,
                            dtype=parameter.dtype,
                            device=device,
                            generator=generator,
                        )
                        * 0.02
                    )
    else:
        model.load_state_dict(model_state)

    num_layers = 2
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=num_layers, num_slots=16, device=device)
    assert storage.latent_cache is not None and storage.rope_cache is not None
    storage.latent_cache.zero_()
    storage.rope_cache.zero_()
    storage.latent_cache[:, :3].copy_(initial_latent)
    storage.rope_cache[:, :3].copy_(initial_rope)

    runtime_layout = RuntimeLayout.dense(num_layers)
    runtime_config = SimpleNamespace(
        sparse_method="",
        max_model_len=128,
        runtime_layout=runtime_layout,
        hf_config=model_config,
        obs_layer_ids=[],
        full_attention_layers=[],
        sink_keep_tokens=0,
        recent_keep_tokens=0,
        decode_keep_tokens=0,
        sparse_attn_score_dtype="float32",
        tensor_parallel_size=1,
        decode_graph=True,
    )
    manager = object.__new__(StandardCacheManager)
    manager.config = runtime_config
    manager.parallel_context = parallel_context
    manager.device = device
    manager.runtime_layout = runtime_layout
    manager.num_layers = num_layers
    manager.num_kv_layers = num_layers
    manager.max_model_len = 128
    manager.attention_cache_storage = storage
    manager.kv_cache = None
    manager._decode_static_max_context_len = None
    manager._attention_key_materializers = {}
    manager.free_slots_stack = torch.empty(16, dtype=torch.int32, device=device)
    manager.free_slots_stack[:13].copy_(
        torch.arange(3, 16, dtype=torch.int32, device=device)
    )
    manager._num_free_slots = 13
    manager.buffer_req_to_token_slots = torch.zeros(
        (1, 128),
        dtype=torch.int32,
        device=device,
    )
    manager.buffer_req_to_token_slots[0, :3].copy_(
        torch.arange(3, dtype=torch.int32, device=device)
    )
    sequence = Sequence([17, 19, 23, 29])
    sequence.num_prefilled_tokens = sequence.num_prompt_tokens
    sequence.temperature = 0.0
    sequence.max_tokens = 8
    manager.seq_id_to_row = {sequence.seq_id: 0}
    manager.free_rows = deque()
    manager.row_seq_lens = np.asarray([3], dtype=np.int32)
    manager.layer_batch_state = LayerBatchStates()
    manager._decode_static_index_buffers = {}
    manager.enable_prefix_caching = False
    manager.prefix_cache_block_size = 4
    manager.prefix_cache = None
    manager.seq_id_to_prefix_blocks = {}
    manager.seq_id_to_cached_ranges = {}
    manager._scheduler_capacity_snapshot_depth = 0
    manager._scheduler_freeable_block_ids = None
    manager.prefix_offload_controller = None
    manager._prefix_offload_step_h2d_operations = {}
    manager._prefix_write_through_candidates = {}
    manager._init_prefix_cache_runtime()

    sparse_controller = SparseController(runtime_config, manager)
    model.model.sparse_controller = sparse_controller
    runtime_state = RuntimeState(runtime_config, manager)

    def run_model(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        assert not is_prefill
        return model.compute_logits(model(input_ids, positions))

    runner = DecodeCudaGraphRunner(
        runtime_state=runtime_state,
        cache_manager=manager,
        recurrent_state_manager=None,
        sparse_controller=sparse_controller,
        run_model=run_model,
        method="",
        capture_sizes=[1],
    )
    return SimpleNamespace(
        model=model,
        manager=manager,
        storage=storage,
        sequence=sequence,
        runner=runner,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_glm_full_decoder_moe_cuda_graph_matches_static_eager():
    device = torch.device("cuda")
    parallel_context = _single_rank_parallel_context()
    generator = torch.Generator(device=device).manual_seed(947)
    initial_latent = torch.randn(
        (2, 3, 1, 512),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    initial_rope = torch.randn(
        (2, 3, 1, 64),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    eager = _make_glm_full_graph_lane(
        device=device,
        parallel_context=parallel_context,
        model_state=None,
        initial_latent=initial_latent,
        initial_rope=initial_rope,
    )
    graph = _make_glm_full_graph_lane(
        device=device,
        parallel_context=parallel_context,
        model_state=eager.model.state_dict(),
        initial_latent=initial_latent,
        initial_rope=initial_rope,
    )
    eager_moe = eager.model.model.layers[1].mlp
    graph_moe = graph.model.model.layers[1].mlp
    assert isinstance(eager_moe, Glm4MoeLiteSparseMoeBlock)
    assert isinstance(graph_moe, Glm4MoeLiteSparseMoeBlock)

    steps = []
    captured_graph = None
    with patch.dict(os.environ, {"SPARSEENGINE_DEBUG_MOE": "1"}):
        for step in range(2):
            eager_logits = eager.runner.run_eager_static([eager.sequence])
            graph_logits, graph_token_ids = graph.runner.run(
                [graph.sequence],
                capture_sampling=True,
            )
            torch.cuda.synchronize()
            assert eager_logits is not None
            assert graph_logits is not None
            assert graph_token_ids is not None
            torch.testing.assert_close(graph_logits, eager_logits, rtol=0, atol=0)
            eager_token_ids = eager_logits.argmax(dim=-1)
            torch.testing.assert_close(
                graph_token_ids,
                eager_token_ids,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                graph_moe.debug_last_topk_ids,
                eager_moe.debug_last_topk_ids,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                graph_moe.debug_last_topk_weights,
                eager_moe.debug_last_topk_weights,
                rtol=0,
                atol=0,
            )
            local_hit_count = graph_moe.debug_last_local_hit_count
            if isinstance(local_hit_count, torch.Tensor):
                local_hit_count = int(local_hit_count.item())
            assert local_hit_count == 4

            graph_states = [
                state
                for state in graph.runner._graphs.values()
                if state.graph is not None
            ]
            assert len(graph_states) == 1
            if captured_graph is None:
                captured_graph = graph_states[0].graph
            else:
                assert graph_states[0].graph is captured_graph

            row_len = int(graph.manager.row_seq_lens[0])
            assert row_len == int(eager.manager.row_seq_lens[0])
            eager_slots = eager.manager.buffer_req_to_token_slots[0, :row_len].long()
            graph_slots = graph.manager.buffer_req_to_token_slots[0, :row_len].long()
            eager_latent = eager.storage.latent_cache.index_select(1, eager_slots)
            graph_latent = graph.storage.latent_cache.index_select(1, graph_slots)
            eager_rope = eager.storage.rope_cache.index_select(1, eager_slots)
            graph_rope = graph.storage.rope_cache.index_select(1, graph_slots)
            torch.testing.assert_close(graph_latent, eager_latent, rtol=0, atol=0)
            torch.testing.assert_close(graph_rope, eager_rope, rtol=0, atol=0)
            steps.append(
                {
                    "step": step + 1,
                    "token": int(graph_token_ids[0].item()),
                    "logits_sha256": _tensor_sha256(graph_logits),
                    "latent_sha256": _tensor_sha256(graph_latent),
                    "rope_sha256": _tensor_sha256(graph_rope),
                    "topk_ids": graph_moe.debug_last_topk_ids.tolist(),
                    "topk_sha256": _tensor_sha256(
                        graph_moe.debug_last_topk_ids
                    ),
                    "local_expert_hits": local_hit_count,
                }
            )
            if step == 0:
                next_token = int(graph_token_ids[0].item())
                eager.sequence.append_token(next_token)
                graph.sequence.append_token(next_token)

    evidence = {
        "harness_scope": "tiny_random_full_decoder_moe",
        "model_class": "Glm4MoeLiteForCausalLM",
        "real_checkpoint": False,
        "moe_provider": graph_moe.experts.provider.name,
        "graph_active": captured_graph is not None,
        "graph_count": sum(
            state.graph is not None for state in graph.runner._graphs.values()
        ),
        "capture_count": graph.runner.capture_count,
        "replay_count": graph.runner.replay_count,
        "eager_static_count": graph.runner.eager_static_count,
        "force_eager_count": graph.runner.force_eager_count,
        "fallback": graph.runner.force_eager_count > 0,
        "steps": steps,
    }
    assert evidence["graph_count"] == 1
    assert evidence["capture_count"] == 1
    assert evidence["replay_count"] == 2
    assert evidence["eager_static_count"] == 0
    assert evidence["force_eager_count"] == 0
    assert evidence["fallback"] is False
    print("GLM_FULL_CUDA_GRAPH_EVIDENCE=" + json.dumps(evidence, sort_keys=True))


_GLM_GRAPH_METHOD_MANAGERS = {
    "streamingllm": StreamingLLMCacheManager,
    "snapkv": SnapKVCacheManager,
    "h2o": H2OCacheManager,
    "omnikv": OmniKVCacheManager,
    "rkv": RKVCacheManager,
}


def _glm_method_runtime_config(method: str, *, num_layers: int):
    hf_config = _glm_hf_config(
        num_hidden_layers=num_layers,
        max_position_embeddings=128,
    )
    hf_config.rms_norm_eps = 1e-6
    return SimpleNamespace(
        sparse_method=method,
        runtime_layout=RuntimeLayout.dense(num_layers),
        hf_config=hf_config,
        obs_layer_ids=[0] if method == "omnikv" else [],
        full_attention_layers=[0] if method == "omnikv" else [],
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=1,
        sparse_attn_score_dtype="float32",
        tensor_parallel_size=1,
        decode_graph=True,
        max_model_len=16,
        max_num_seqs_in_batch=1,
        max_num_seqs_in_gpu=1,
        max_num_batched_tokens=16,
        engine_prefill_chunk_size=8,
        pyramid_layer_ratios=None,
        snapkv_num_full_layers=0,
        observation_window_size=2,
        pool_kernel_size=1,
        prefill_schedule_policy="chunked",
        h2o_decode_budget=3,
        h2o_decode_eviction_interval=1,
        h2o_prefill_budget=4,
        h2o_recent_ratio=0.5,
        h2o_prefill_score_window=2,
        rkv_compression_interval=1,
        rkv_observation_tokens=1,
        rkv_alpha=0.5,
        rkv_kernel_size=1,
        rkv_score_chunk_mb=16,
        rkv_similarity_threshold=0.0,
        rkv_recent_similar_keep=0,
        rkv_max_redundancy_tokens=16,
        rkv_redundancy_window=0,
        enable_prefix_caching=False,
        prefix_cache_block_size=4,
    )


def _initialize_glm_method_cache_manager(
    *,
    method: str,
    config,
    parallel_context: ParallelContext,
    device: torch.device,
    storage: MlaLatentStorage,
    sequence: Sequence,
    initial_len: int,
):
    manager_type = _GLM_GRAPH_METHOD_MANAGERS[method]
    manager = object.__new__(manager_type)
    manager.config = config
    manager.parallel_context = parallel_context
    manager.rank = 0
    manager.world_size = 1
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.ep_rank = 0
    manager.ep_size = 1
    manager.dp_rank = 0
    manager.dp_size = 1
    manager.device = device
    manager.hf_config = config.hf_config
    manager.head_dim = resolve_attention_qk_head_dim(config.hf_config)
    manager.max_model_len = int(config.max_model_len)
    manager.max_buffer_rows = 1
    manager.num_layers = int(config.hf_config.num_hidden_layers)
    manager.num_kv_layers = manager.num_layers
    manager.runtime_layout = config.runtime_layout
    manager.attention_cache_storage = storage
    manager.kv_cache = None
    manager._decode_static_max_context_len = None
    manager._attention_key_materializers = {}
    num_slots = 16
    free_count = num_slots - int(initial_len)

    if method == "omnikv":
        manager.offload_enabled = False
        manager._prefetched = set()
        manager._pending_prefetch = deque()
        manager._current_writes = {}
        manager.lru = None
        manager.free_slots_stack = torch.empty(
            num_slots,
            dtype=torch.int32,
            device=device,
        )
        manager.free_slots_stack[:free_count].copy_(
            torch.arange(initial_len, num_slots, dtype=torch.int32, device=device)
        )
        manager._num_free_slots = free_count
        manager.buffer_req_to_token_slots = torch.zeros(
            (1, manager.max_model_len),
            dtype=torch.int32,
            device=device,
        )
        manager.buffer_req_to_token_slots[0, :initial_len].copy_(
            torch.arange(initial_len, dtype=torch.int32, device=device)
        )
        manager.seq_id_to_row = {sequence.seq_id: 0}
        manager.free_rows = deque()
        manager.row_seq_lens = np.asarray([initial_len], dtype=np.int32)
        manager.layer_batch_state = LayerBatchStates()
        manager._decode_static_index_buffers = {}
        manager.enable_prefix_caching = False
        manager.prefix_cache_block_size = 4
        manager.prefix_cache = None
        manager.seq_id_to_prefix_blocks = {}
        manager.seq_id_to_cached_ranges = {}
        manager._scheduler_capacity_snapshot_depth = 0
        manager._scheduler_freeable_block_ids = None
        manager.prefix_offload_controller = None
        manager._prefix_offload_step_h2d_operations = {}
        manager._prefix_write_through_candidates = {}
        manager._init_prefix_cache_runtime()
        return manager

    manager.layer_num_slots = [num_slots] * manager.num_layers
    manager.free_slots_stack_tensor = torch.empty(
        (manager.num_layers, num_slots),
        dtype=torch.int32,
        device=device,
    )
    for layer_idx in range(manager.num_layers):
        manager.free_slots_stack_tensor[layer_idx, :free_count].copy_(
            torch.arange(initial_len, num_slots, dtype=torch.int32, device=device)
        )
    manager.free_slots_stack = [
        manager.free_slots_stack_tensor[layer_idx]
        for layer_idx in range(manager.num_layers)
    ]
    manager._num_free_slots = [free_count] * manager.num_layers
    manager.buffer_req_to_token_slots_tensor = torch.zeros(
        (manager.num_layers, 1, manager.max_model_len),
        dtype=torch.int32,
        device=device,
    )
    manager.buffer_req_to_token_slots_tensor[:, 0, :initial_len].copy_(
        torch.arange(initial_len, dtype=torch.int32, device=device).expand(
            manager.num_layers,
            -1,
        )
    )
    manager.buffer_req_to_token_slots = [
        manager.buffer_req_to_token_slots_tensor[layer_idx]
        for layer_idx in range(manager.num_layers)
    ]
    manager.seq_id_to_row = [
        {sequence.seq_id: 0} for _ in range(manager.num_layers)
    ]
    manager.free_rows = [deque() for _ in range(manager.num_layers)]
    manager.row_seq_lens = [
        np.asarray([initial_len], dtype=np.int32)
        for _ in range(manager.num_layers)
    ]
    manager.layer_batch_states = [
        LayerBatchStates() for _ in range(manager.num_layers)
    ]
    manager._decode_static_buffers = {}
    manager._decode_static_index_buffers = {}
    manager._decode_static_state_binding_key = None
    manager._prefill_attn_score_accumulators = {}
    manager._uniform_decode_metadata = method == "streamingllm"
    manager.pyramidkv_prefill_staging_num_slots = 0
    manager.pyramidkv_prefill_staging_kv_cache = None
    manager._pyramidkv_prefill_staging_active = False
    manager._pyramidkv_prefill_staging_was_active = False
    manager._pyramidkv_prefill_staging_slot_mapping = None
    manager._pyramidkv_prefill_staging_active_slots = None
    manager._pyramidkv_prefill_staging_req_indices = None
    manager._pyramidkv_prefill_staging_context_lens = None
    manager._pyramidkv_prefill_staging_seq_offsets = {}
    manager._pyramidkv_prefill_staging_materialized_layers = set()
    manager._pyramidkv_long_prefill_offload_step_active = False
    manager._pyramidkv_long_prefill_offload_seq_id = None
    manager._pyramidkv_long_prefill_offload_start = 0
    manager._pyramidkv_long_prefill_offload_end = 0
    manager._pyramidkv_long_prefill_offload_total_len = 0
    manager._pyramidkv_long_prefill_offload_is_last_chunk = False
    manager._pyramidkv_long_prefill_offload_prefetch_stream = None
    manager._pyramidkv_long_prefill_offload_prefetch_states = {}
    manager.raw_kv_offload_buffer = SimpleNamespace(
        release_layer=lambda **_kwargs: None,
    )

    if method == "h2o":
        manager._h2o_scores = {
            (layer_idx, sequence.seq_id): torch.zeros(
                initial_len,
                dtype=torch.float32,
                device=device,
            )
            for layer_idx in range(manager.num_layers)
        }
        manager._h2o_active_decode_seq_ids = set()
        manager._h2o_counters = {
            "intermediate_prefill_evictions": 0,
            "final_prefill_evictions": 0,
            "decode_eviction_bursts": 0,
            "decode_evictions": 0,
            "dropped_tokens": 0,
        }
        manager._h2o_final_prefill_workspace = None
        manager._h2o_decode_static_rows = None
        manager._h2o_decode_static_topology = None

    if method == "rkv":
        obs = int(config.rkv_observation_tokens)
        heads = int(config.hf_config.num_attention_heads)
        manager._rkv_query_cache_enabled = True
        manager._rkv_observation_tokens = obs
        manager._rkv_vectorized_prefill_query_cache = True
        manager._rkv_batch_clear_query_cache_rows = True
        manager._rkv_query_score_static_buffers = {}
        manager._rkv_query_cache = [
            torch.zeros(
                (1, obs, heads, 256),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(manager.num_layers)
        ]
        manager._rkv_query_positions = [
            torch.full(
                (1, obs),
                -1,
                dtype=torch.int32,
                device=device,
            )
            for _ in range(manager.num_layers)
        ]
    return manager


def _make_glm_method_graph_lane(
    *,
    method: str,
    device: torch.device,
    parallel_context: ParallelContext,
    attention_states: list[dict[str, torch.Tensor]] | None,
    embedding_state: dict[str, torch.Tensor] | None,
    head_state: dict[str, torch.Tensor] | None,
    initial_latent: torch.Tensor,
    initial_rope: torch.Tensor,
):
    num_layers = 2
    config = _glm_method_runtime_config(method, num_layers=num_layers)
    spec = MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=1,
        cuda_graph=True,
        score_output=sparse_decode_attention_score_kind(method),
        context_capacity=16,
        batch_capacity=1,
    )
    mla_attention = MLAAttention.bind(
        spec=spec,
        device=device,
        max_batch_size=1,
        prefill_workspace_bytes=1024 * 1024,
        hidden_size=64,
        projection_chunk_size=8,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with (
            patch(
                "sparseengine.models.glm4_moe_lite.get_parallel_context",
                return_value=parallel_context,
            ),
            patch(
                "sparseengine.layers.linear.get_parallel_context",
                return_value=parallel_context,
            ),
            torch.device(device),
        ):
            attentions = nn.ModuleList(
                [
                    Glm4MoeLiteAttention(
                        config.hf_config,
                        mla_attention,
                        projection_chunk_size=8,
                    )
                    for _ in range(num_layers)
                ]
            )
            embedding = nn.Embedding(128, 64)
            lm_head = nn.Linear(64, 128, bias=False)
            rotary = RotaryEmbedding(
                64,
                64,
                128,
                1_000_000.0,
                backend="torch",
                interleaved=True,
            )
    finally:
        torch.set_default_dtype(previous_dtype)

    if attention_states is None:
        generator = torch.Generator(device=device).manual_seed(967)
        with torch.no_grad():
            for parameter in attentions.parameters():
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=device,
                        generator=generator,
                    )
                    * 0.02
                )
            embedding.weight.copy_(
                torch.randn(
                    embedding.weight.shape,
                    dtype=embedding.weight.dtype,
                    device=device,
                    generator=generator,
                )
                * 0.02
            )
            lm_head.weight.copy_(
                torch.randn(
                    lm_head.weight.shape,
                    dtype=lm_head.weight.dtype,
                    device=device,
                    generator=generator,
                )
                * 0.02
            )
    else:
        assert embedding_state is not None and head_state is not None
        assert len(attention_states) == num_layers
        for attention, state in zip(attentions, attention_states):
            attention.load_state_dict(state)
        embedding.load_state_dict(embedding_state)
        lm_head.load_state_dict(head_state)

    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=num_layers, num_slots=16, device=device)
    assert storage.latent_cache is not None and storage.rope_cache is not None
    storage.latent_cache.zero_()
    storage.rope_cache.zero_()
    initial_len = int(initial_latent.shape[1])
    storage.latent_cache[:, :initial_len].copy_(initial_latent)
    storage.rope_cache[:, :initial_len].copy_(initial_rope)

    sequence = Sequence([5, 7, 11])
    sequence.append_token(13)
    sequence.num_prefilled_tokens = sequence.num_prompt_tokens
    sequence.temperature = 0.0
    sequence.max_tokens = 8
    manager = _initialize_glm_method_cache_manager(
        method=method,
        config=config,
        parallel_context=parallel_context,
        device=device,
        storage=storage,
        sequence=sequence,
        initial_len=initial_len,
    )
    controller = SparseController(config, manager)
    runtime_state = RuntimeState(config, manager)

    def run_model(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        assert not is_prefill
        context = get_context()
        hidden_states = embedding(input_ids)
        for layer_idx, attention in enumerate(attentions):
            context.now_layer_idx = layer_idx
            hidden_states = attention(positions, hidden_states, rotary)
            controller.on_layer_end(layer_idx, context)
        return lm_head(hidden_states)

    runner = DecodeCudaGraphRunner(
        runtime_state=runtime_state,
        cache_manager=manager,
        recurrent_state_manager=None,
        sparse_controller=controller,
        run_model=run_model,
        method=method,
        capture_sizes=[1],
    )
    return SimpleNamespace(
        attentions=attentions,
        embedding=embedding,
        lm_head=lm_head,
        manager=manager,
        storage=storage,
        sequence=sequence,
        controller=controller,
        runtime_state=runtime_state,
        runner=runner,
    )


def _glm_method_row_lens(lane, method: str) -> list[int]:
    if method == "omnikv":
        return [int(lane.manager.row_seq_lens[0])] * len(lane.attentions)
    return [
        int(lane.manager.row_seq_lens[layer_idx][0])
        for layer_idx in range(len(lane.attentions))
    ]


def _glm_method_slot_rows(lane, method: str) -> list[list[int]]:
    row_lens = _glm_method_row_lens(lane, method)
    if method == "omnikv":
        row = lane.manager.buffer_req_to_token_slots[0, : row_lens[0]]
        return [row.tolist() for _ in lane.attentions]
    return [
        lane.manager.buffer_req_to_token_slots[layer_idx][
            0, : row_lens[layer_idx]
        ].tolist()
        for layer_idx in range(len(lane.attentions))
    ]


def _glm_method_trigger_state(lane, method: str) -> dict[str, object]:
    state: dict[str, object] = {
        "row_lens": _glm_method_row_lens(lane, method),
    }
    if method in {"snapkv", "h2o"}:
        assert lane.controller.layer_batch_sparse_states[0].attn_score is None
        mapping = lane.manager.layer_batch_states[0].slot_mapping
        assert mapping is not None
        state.update(
            attention_score_requested=False,
            mapping_ptr=int(mapping.data_ptr()),
        )
    elif method == "omnikv":
        target = lane.controller.layer_batch_sparse_states[1]
        assert target.active_indices is not None
        assert target.active_slots is not None
        assert target.context_lens is not None
        state.update(
            selection_ptr=int(target.active_slots.data_ptr()),
            active_indices=target.active_indices.tolist(),
            active_slots=target.active_slots.tolist(),
            active_context_lens=target.context_lens.tolist(),
        )
    elif method == "rkv":
        positions = lane.manager._rkv_query_positions[0]
        state.update(
            query_cache_ptr=int(lane.manager._rkv_query_cache[0].data_ptr()),
            query_positions=positions.tolist(),
            materializer_bound=lane.manager.has_attention_key_materializer(0),
        )
    else:
        mapping = lane.manager.layer_batch_states[0].slot_mapping
        assert mapping is not None
        state["mapping_ptr"] = int(mapping.data_ptr())
    return state


@pytest.mark.parametrize(
    "method",
    ["streamingllm", "snapkv", "h2o", "omnikv", "rkv"],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_glm_sparse_method_decode_cuda_graph_triggers_runtime_path(method: str):
    device = torch.device("cuda", torch.cuda.current_device())
    parallel_context = _single_rank_parallel_context()
    num_layers = 2
    generator = torch.Generator(device=device).manual_seed(971)
    initial_latent = torch.randn(
        (num_layers, 3, 1, 512),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    initial_rope = torch.randn(
        (num_layers, 3, 1, 64),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    eager = _make_glm_method_graph_lane(
        method=method,
        device=device,
        parallel_context=parallel_context,
        attention_states=None,
        embedding_state=None,
        head_state=None,
        initial_latent=initial_latent,
        initial_rope=initial_rope,
    )
    graph = _make_glm_method_graph_lane(
        method=method,
        device=device,
        parallel_context=parallel_context,
        attention_states=[
            attention.state_dict() for attention in eager.attentions
        ],
        embedding_state=eager.embedding.state_dict(),
        head_state=eager.lm_head.state_dict(),
        initial_latent=initial_latent,
        initial_rope=initial_rope,
    )

    steps = []
    stable_ptr = None
    trigger_count = 0
    for step in range(2):
        eager_logits = eager.runner.run_eager_static([eager.sequence])
        eager_before = _glm_method_trigger_state(eager, method)
        eager.controller.post_forward([eager.sequence], is_prefill=False)
        eager.runtime_state.on_forward_end([eager.sequence], is_prefill=False)
        eager_after_lens = _glm_method_row_lens(eager, method)

        graph_logits, graph_token_ids = graph.runner.run(
            [graph.sequence],
            capture_sampling=True,
        )
        torch.cuda.synchronize()
        graph_before = _glm_method_trigger_state(graph, method)
        graph.controller.post_forward([graph.sequence], is_prefill=False)
        graph.runtime_state.on_forward_end([graph.sequence], is_prefill=False)
        graph_after_lens = _glm_method_row_lens(graph, method)

        assert eager_logits is not None
        assert graph_logits is not None
        assert graph_token_ids is not None
        torch.testing.assert_close(graph_logits, eager_logits, rtol=0, atol=0)
        eager_token_ids = eager_logits.argmax(dim=-1)
        torch.testing.assert_close(graph_token_ids, eager_token_ids, rtol=0, atol=0)
        assert graph_before["row_lens"] == eager_before["row_lens"]
        assert graph_after_lens == eager_after_lens
        assert _glm_method_slot_rows(graph, method) == _glm_method_slot_rows(
            eager,
            method,
        )
        torch.testing.assert_close(
            graph.storage.latent_cache,
            eager.storage.latent_cache,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            graph.storage.rope_cache,
            eager.storage.rope_cache,
            rtol=0,
            atol=0,
        )

        if method == "omnikv":
            assert graph_before["active_indices"] == eager_before["active_indices"]
            assert graph_before["active_slots"] == eager_before["active_slots"]
            assert graph_before["active_context_lens"] == [3]
            pointer = int(graph_before["selection_ptr"])
        elif method == "rkv":
            assert graph_before["query_positions"] == eager_before["query_positions"]
            assert graph_before["materializer_bound"] is True
            assert graph_before["query_positions"] == [[3]]
            pointer = int(graph_before["query_cache_ptr"])
        else:
            pointer = int(graph_before["mapping_ptr"])
        if stable_ptr is None:
            stable_ptr = pointer
        else:
            assert pointer == stable_ptr

        before_len = int(graph_before["row_lens"][0])
        after_len = int(graph_after_lens[0])
        triggered = (
            after_len < before_len
            if method != "omnikv"
            else int(graph_before["active_context_lens"][0]) < before_len
        )
        trigger_count += int(triggered)
        if method == "rkv":
            assert graph.manager._rkv_query_positions[0].tolist() == [[-1]]

        steps.append(
            {
                "step": step + 1,
                "token": int(graph_token_ids.item()),
                "logits_sha256": _tensor_sha256(graph_logits),
                "latent_sha256": _tensor_sha256(graph.storage.latent_cache),
                "rope_sha256": _tensor_sha256(graph.storage.rope_cache),
                "before_lens": graph_before["row_lens"],
                "after_lens": graph_after_lens,
                "triggered": triggered,
                "runtime_state": graph_before,
            }
        )
        eager.sequence.append_token(int(eager_token_ids.item()))
        graph.sequence.append_token(int(graph_token_ids.item()))

    if method in {"snapkv", "h2o"}:
        assert trigger_count == 0
    else:
        assert trigger_count > 0
    if method == "h2o":
        assert graph.manager._h2o_counters["decode_evictions"] == 0
    evidence = {
        "method": method,
        "harness_scope": "tiny_random_sparse_method_component",
        "real_checkpoint": False,
        "graph_active": any(
            state.graph is not None for state in graph.runner._graphs.values()
        ),
        "graph_count": sum(
            state.graph is not None for state in graph.runner._graphs.values()
        ),
        "capture_count": graph.runner.capture_count,
        "replay_count": graph.runner.replay_count,
        "eager_static_count": graph.runner.eager_static_count,
        "force_eager_count": graph.runner.force_eager_count,
        "fallback": graph.runner.force_eager_count > 0,
        "trigger_count": trigger_count,
        "stable_runtime_ptr": stable_ptr,
        "steps": steps,
    }
    assert evidence["graph_active"] is True
    assert evidence["graph_count"] == 1
    assert evidence["capture_count"] == 1
    assert evidence["replay_count"] == 2
    assert evidence["eager_static_count"] == 0
    assert evidence["force_eager_count"] == 0
    assert evidence["fallback"] is False
    print("GLM_METHOD_CUDA_GRAPH_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
