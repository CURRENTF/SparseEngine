from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    apply_rotary_pos_emb_interleave,
)

from sparseengine.config import QuantizationConfig
from sparseengine.debug.tiny_random import (
    build_tiny_random_hf_model,
    initialize_sparse_model,
)
from sparseengine.distributed import ParallelContext, ParallelGroup
from sparseengine.distributed.moe_communication import AllReduceMoeCommunication
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.layers.rotary_embedding import apply_interleaved_rotary_emb
from sparseengine.models.glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteRouter,
    Glm4MoeLiteSparseMoeBlock,
    build_glm4_moe_lite_mla_attention,
)
from sparseengine.models.qwen3 import Qwen3MLP
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.mla_attention import MlaAttentionOpSpec
from sparseengine.operators.moe import TritonMoeProvider
from sparseengine.operators.moe_router import GlmBiasedSigmoidRouterProvider
from sparseengine.platforms import device_runtime


def _config(**overrides) -> Glm4MoeLiteConfig:
    values = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "n_shared_experts": 1,
        "n_routed_experts": 64,
        "num_experts_per_tok": 4,
        "routed_scaling_factor": 1.8,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "kv_lora_rank": 512,
        "q_lora_rank": 768,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "qk_nope_head_dim": 192,
        "max_position_embeddings": 32,
        "dtype": torch.bfloat16,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
        },
    }
    values.update(overrides)
    config = Glm4MoeLiteConfig(**values)
    config.mlp_chunk_size = 8
    config.quantization_config = QuantizationConfig.disabled()
    config.decode_graph = False
    return config


def _tp_context(tp_rank: int = 0, tp_size: int = 1) -> ParallelContext:
    ranks = tuple(range(tp_size))
    return ParallelContext(
        world=ParallelGroup(None, ranks, tp_rank, tp_size),
        moe_tp=ParallelGroup(None, ranks, tp_rank, tp_size),
        attn_tp=ParallelGroup(None, ranks, tp_rank, tp_size),
        moe_ep=ParallelGroup(None, (tp_rank,), 0, 1),
        attn_dp=ParallelGroup(None, (tp_rank,), 0, 1),
    )


def _ep_context(ep_rank: int = 0, ep_size: int = 1) -> ParallelContext:
    ranks = tuple(range(ep_size))
    return ParallelContext(
        world=ParallelGroup(object(), ranks, ep_rank, ep_size),
        moe_tp=ParallelGroup(None, (ep_rank,), 0, 1),
        attn_tp=ParallelGroup(None, (ep_rank,), 0, 1),
        moe_ep=ParallelGroup(object(), ranks, ep_rank, ep_size),
        attn_dp=ParallelGroup(None, (ep_rank,), 0, 1),
    )


def _fake_mla(tp_size: int = 1):
    spec = MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=tp_size,
        cuda_graph=False,
    )
    return SimpleNamespace(
        spec=spec,
        provider=SimpleNamespace(name="test"),
        hidden_size=64,
        projection_chunk_size=8,
    )


def test_glm_releases_shared_mla_cache_runtime_binding() -> None:
    cache_manager = object()
    mla_attention = SimpleNamespace(release_cache_runtime_bindings=Mock())
    model = object.__new__(Glm4MoeLiteForCausalLM)
    nn.Module.__init__(model)
    model.model = SimpleNamespace(mla_attention=mla_attention)

    model.release_cache_runtime_bindings(cache_manager)

    mla_attention.release_cache_runtime_bindings.assert_called_once_with(
        cache_manager
    )


@contextmanager
def _construction_context(context: ParallelContext):
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "sparseengine.models.glm4_moe_lite.get_parallel_context",
                return_value=context,
            )
        )
        stack.enter_context(
            patch("sparseengine.layers.linear.get_parallel_context", return_value=context)
        )
        stack.enter_context(
            patch(
                "sparseengine.layers.embed_head.get_parallel_context",
                return_value=context,
            )
        )
        stack.enter_context(
            patch(
                "sparseengine.models.glm4_moe_lite.resolve_moe_provider",
                return_value=TritonMoeProvider(),
            )
        )
        stack.enter_context(
            patch(
                "sparseengine.models.glm4_moe_lite.resolve_moe_router_provider",
                return_value=GlmBiasedSigmoidRouterProvider(),
            )
        )
        stack.enter_context(torch.device("cpu"))
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            yield
        finally:
            torch.set_default_dtype(previous_dtype)


def _model(config=None, *, tp_rank: int = 0, tp_size: int = 1):
    config = _config() if config is None else config
    context = _tp_context(tp_rank, tp_size)
    with _construction_context(context):
        return Glm4MoeLiteForCausalLM(
            config,
            mla_attention=_fake_mla(tp_size),
            mlp_chunk_size=config.mlp_chunk_size,
            decode_graph=config.decode_graph,
        )


def test_glm_topology_reuses_one_mla_object_and_qwen_dense_mlp() -> None:
    model = _model()

    assert len(model.model.layers) == 2
    assert isinstance(model.model.layers[0].mlp, Qwen3MLP)
    assert isinstance(model.model.layers[1].mlp, Glm4MoeLiteSparseMoeBlock)
    assert model.model.layers[0].self_attn.mla_attention is model.model.mla_attention
    assert model.model.layers[1].self_attn.mla_attention is model.model.mla_attention
    assert model.model.rotary_emb.interleaved is True
    assert model.model.layers[1].mlp.experts.op_spec.routing_method == "biased_sigmoid"


@pytest.mark.parametrize(
    ("sparse_method", "requires_scores"),
    [
        (None, False),
        ("streamingllm", False),
        ("rkv", False),
        ("h2o", False),
        ("omnikv", True),
    ],
)
def test_glm_runtime_kwargs_bind_shared_operators(
    sparse_method,
    requires_scores,
) -> None:
    config = _config()
    context = _tp_context(tp_size=2)
    runtime = SimpleNamespace(
        decode_graph=True,
        max_model_len=32768,
        max_num_seqs_in_batch=4,
        max_decoding_seqs=8,
        max_num_batched_tokens=64,
        moe_backend="all-reduce",
        mla_prefill_workspace_bytes=1024,
        mla_prefill_history_chunk_size=17,
        mlp_chunk_size=16,
        tiny_random=False,
        sparse_method=sparse_method,
        attention_cache_layout="mla_latent",
    )
    mla = object()
    all_reduce = object()
    collective_runtime = Mock()
    collective_runtime.request_moe_collectives.return_value = all_reduce
    with patch(
        "sparseengine.models.glm4_moe_lite.build_glm4_moe_lite_mla_attention",
        return_value=mla,
    ) as build_mla:
        kwargs = Glm4MoeLiteForCausalLM.build_runtime_kwargs(
            config,
            engine_config=runtime,
            parallel_context=context,
            collective_runtime=collective_runtime,
            device=torch.device("cuda", 1),
            max_decode_tokens=8,
        )

    assert kwargs == {
        "mla_attention": mla,
        "mlp_chunk_size": 16,
        "decode_graph": True,
        "parallel_collectives": all_reduce,
    }
    build_mla.assert_called_once_with(
        config,
        device=torch.device("cuda", 1),
        max_batch_size=8,
        prefill_workspace_bytes=1024,
        history_chunk_size=runtime.mla_prefill_history_chunk_size,
        decode_graph=True,
        context_capacity=32768,
        projection_chunk_size=16,
        score_output=(
            AttentionScoreKind.RAW_QK_PER_HEAD
            if requires_scores
            else AttentionScoreKind.NONE
        ),
    )
    collective_runtime.request_moe_collectives.assert_called_once_with(
        attention_max_rows=8,
        moe_max_rows=16,
        max_local_tokens=64,
        backend="all-reduce",
        num_experts=config.n_routed_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=64,
        dtype=torch.bfloat16,
    )


def test_glm_decode_graph_mla_spec_owns_context_capacity() -> None:
    config = _config()
    bound = object()
    with (
        patch(
            "sparseengine.models.glm4_moe_lite.get_parallel_context",
            return_value=_tp_context(tp_size=2),
        ),
        patch(
            "sparseengine.models.glm4_moe_lite.MLAAttention.bind",
            return_value=bound,
        ) as bind,
    ):
        actual = build_glm4_moe_lite_mla_attention(
            config,
            device="cpu",
            max_batch_size=8,
            prefill_workspace_bytes=1024,
            decode_graph=True,
            context_capacity=32768,
            projection_chunk_size=16,
        )

    assert actual is bound
    spec = bind.call_args.kwargs["spec"]
    assert spec.cuda_graph
    assert spec.context_capacity == 32768


def test_glm_interleaved_rope_matches_transformers() -> None:
    torch.manual_seed(13)
    q = torch.randn(1, 3, 5, 64)
    k = torch.randn(1, 1, 5, 64)
    angles = torch.randn(1, 5, 32)
    cos_half = angles.cos()
    sin_half = angles.sin()
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)

    expected_q, expected_k = apply_rotary_pos_emb_interleave(q, k, cos, sin)
    actual_q = apply_interleaved_rotary_emb(
        q,
        cos_half.unsqueeze(1),
        sin_half.unsqueeze(1),
    )
    actual_k = apply_interleaved_rotary_emb(
        k,
        cos_half.unsqueeze(1),
        sin_half.unsqueeze(1),
    )

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)


def test_glm_tp_projection_slices_follow_local_heads() -> None:
    config = _config()
    context = _tp_context(tp_rank=2, tp_size=4)
    with _construction_context(context):
        attention = Glm4MoeLiteAttention(
            config,
            _fake_mla(tp_size=4),
            projection_chunk_size=config.mlp_chunk_size,
        )

    q_source = (
        torch.arange(20 * 256 * 768).remainder(127).to(torch.bfloat16)
    ).view(20 * 256, 768)
    kv_source = (
        torch.arange(20 * 448 * 512).remainder(127).to(torch.bfloat16)
    ).view(20 * 448, 512)
    o_source = (
        torch.arange(64 * 20 * 256).remainder(127).to(torch.bfloat16)
    ).view(64, 20 * 256)
    attention.q_b_proj.weight_loader(attention.q_b_proj.weight, q_source)
    attention.kv_b_proj.weight_loader(attention.kv_b_proj.weight, kv_source)
    attention.o_proj.weight_loader(attention.o_proj.weight, o_source)

    assert attention.local_heads == 5
    assert torch.equal(attention.q_b_proj.weight, q_source[2560:3840])
    assert torch.equal(attention.kv_b_proj.weight, kv_source[4480:6720])
    assert torch.equal(attention.o_proj.weight, o_source[:, 2560:3840])


def test_glm_qkv_a_projection_loads_and_executes_as_one_gemm() -> None:
    config = _config()
    context = _tp_context()
    with _construction_context(context):
        attention = Glm4MoeLiteAttention(
            config,
            _fake_mla(),
            projection_chunk_size=config.mlp_chunk_size,
        )

    torch.manual_seed(23)
    q_weight = torch.randn(config.q_lora_rank, config.hidden_size)
    kv_output_size = config.kv_lora_rank + config.qk_rope_head_dim
    kv_weight = torch.randn(kv_output_size, config.hidden_size)
    projection = attention.fused_qkv_a_proj
    projection.weight_loader(projection.weight, q_weight, 0)
    projection.weight_loader(projection.weight, kv_weight, 1)

    q_weight = q_weight.to(dtype=projection.weight.dtype)
    kv_weight = kv_weight.to(dtype=projection.weight.dtype)
    hidden_states = torch.randn(
        5,
        config.hidden_size,
        dtype=projection.weight.dtype,
    )
    actual_q, actual_kv = projection(hidden_states).split(
        [config.q_lora_rank, kv_output_size],
        dim=-1,
    )
    torch.testing.assert_close(actual_q, F.linear(hidden_states, q_weight))
    torch.testing.assert_close(actual_kv, F.linear(hidden_states, kv_weight))
    assert Glm4MoeLiteForCausalLM.packed_modules_mapping[
        "self_attn.q_a_proj"
    ] == ("self_attn.fused_qkv_a_proj", 0)
    assert Glm4MoeLiteForCausalLM.packed_modules_mapping[
        "self_attn.kv_a_proj_with_mqa"
    ] == ("self_attn.fused_qkv_a_proj", 1)


def test_glm_decode_absorption_and_value_reconstruction_match_linear_algebra() -> None:
    config = _config()
    context = _tp_context()
    with _construction_context(context):
        attention = Glm4MoeLiteAttention(
            config,
            _fake_mla(),
            projection_chunk_size=config.mlp_chunk_size,
        )
    torch.manual_seed(19)
    attention.kv_b_proj.weight.data.normal_(mean=0.0, std=0.02)
    q_nope = torch.randn(3, 20, 192, dtype=torch.bfloat16)
    latent_output = torch.randn(3, 20, 512, dtype=torch.bfloat16)
    weight = attention.kv_b_proj.weight.view(20, 448, 512)

    absorbed = attention._decode_absorbed_query(q_nope)
    reconstructed = attention._reconstruct_decode_values(latent_output)

    expected_absorbed = torch.einsum(
        "bhd,hdr->bhr",
        q_nope,
        weight[:, :192],
    )
    expected_reconstructed = torch.einsum(
        "bhr,hvr->bhv",
        latent_output,
        weight[:, 192:],
    )
    torch.testing.assert_close(absorbed, expected_absorbed)
    torch.testing.assert_close(reconstructed, expected_reconstructed)


def test_glm_router_uses_bias_only_for_selection_and_scales_weights() -> None:
    torch.manual_seed(23)
    with patch(
        "sparseengine.models.glm4_moe_lite.resolve_moe_router_provider",
        return_value=GlmBiasedSigmoidRouterProvider(),
    ):
        router = Glm4MoeLiteRouter(_config())
    router.weight.data.normal_(mean=0.0, std=0.1)
    router.e_score_correction_bias.data.zero_()
    router.e_score_correction_bias.data[:4] = 100.0

    def torch_topk(_spec, logits, correction_bias, *, routed_scaling_factor):
        scores = logits.sigmoid()
        ids = torch.topk(scores + correction_bias, 4, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * routed_scaling_factor, ids

    router.provider.run = torch_topk
    hidden_states = torch.randn(7, 64, dtype=torch.bfloat16)
    logits, weights, ids = router(hidden_states)
    original_scores = logits.sigmoid()
    expected = original_scores.gather(1, ids)
    expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-20) * 1.8

    assert all(set(row.tolist()) == {0, 1, 2, 3} for row in ids)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(dim=-1), torch.full((7,), 1.8))


def test_tiny_transformers_weights_load_through_strict_glm_mapping() -> None:
    config = _config()
    context = _tp_context()
    with _construction_context(context):
        model = Glm4MoeLiteForCausalLM(
            config,
            mla_attention=_fake_mla(),
            mlp_chunk_size=config.mlp_chunk_size,
            decode_graph=config.decode_graph,
        )
        initialize_sparse_model(model, config, seed=29)
    reference = build_tiny_random_hf_model(config, seed=29)
    reference_experts = reference.model.layers[1].mlp.experts
    target_experts = model.model.layers[1].mlp.experts

    torch.testing.assert_close(
        model.model.layers[0].self_attn.q_b_proj.weight.cpu(),
        reference.model.layers[0].self_attn.q_b_proj.weight,
    )
    torch.testing.assert_close(
        target_experts.w13_weight[:, :16].cpu(),
        reference_experts.gate_up_proj[:, :16],
    )
    torch.testing.assert_close(
        target_experts.w13_weight[:, 16:].cpu(),
        reference_experts.gate_up_proj[:, 16:],
    )
    torch.testing.assert_close(
        target_experts.w2_weight.cpu(),
        reference_experts.down_proj,
    )
    assert len(target_experts._loaded_expert_shards) == 64 * 3
    assert model.model.layers[1].mlp.gate.e_score_correction_bias.dtype == torch.float32


@pytest.mark.parametrize(("ep_rank", "ep_size"), [(1, 2), (3, 4)])
def test_glm_ep_loader_keeps_only_local_experts(
    ep_rank: int,
    ep_size: int,
) -> None:
    config = _config()
    context = _ep_context(ep_rank, ep_size)
    with _construction_context(context):
        model = Glm4MoeLiteForCausalLM(
            config,
            mla_attention=_fake_mla(),
            mlp_chunk_size=config.mlp_chunk_size,
            decode_graph=config.decode_graph,
        )
        initialize_sparse_model(model, config, seed=31)

    experts = model.model.layers[1].mlp.experts
    expected_local = 64 // ep_size
    assert experts.local_expert_start == ep_rank * expected_local
    assert experts.local_expert_end == (ep_rank + 1) * expected_local
    assert len(experts._loaded_expert_shards) == expected_local * 3


@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_glm_sparse_moe_reduces_routed_and_shared_partials_over_world(ep_size: int) -> None:
    context = _ep_context(ep_rank=0, ep_size=ep_size)
    block = object.__new__(Glm4MoeLiteSparseMoeBlock)
    nn.Module.__init__(block)
    block.parallel_context = context
    block.moe_communication = AllReduceMoeCommunication(context.world.all_reduce)
    block.parallel_collectives = None
    block.mlp_chunk_size = 8
    block.shared_experts = nn.Identity()
    block._routed_chunk = lambda hidden_states: hidden_states.clone()
    block.moe_execution = block._create_moe_execution()
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    with patch.object(dist, "all_reduce") as all_reduce:
        output = block(hidden_states)

    torch.testing.assert_close(output, hidden_states * 2)
    if ep_size == 1:
        all_reduce.assert_not_called()
    else:
        all_reduce.assert_called_once()
        assert all_reduce.call_args.kwargs["group"] is context.world.process_group


@pytest.mark.parametrize(("is_prefill", "expected_reductions"), [(False, 1), (True, 1)])
def test_glm_sparse_moe_reduces_pure_tp_over_world(
    is_prefill: bool,
    expected_reductions: int,
) -> None:
    moe_tp_process_group = object()
    context = ParallelContext(
        world=ParallelGroup(moe_tp_process_group, (0, 1), 0, 2),
        attn_tp=ParallelGroup(moe_tp_process_group, (0, 1), 0, 2),
        moe_ep=ParallelGroup(object(), (0,), 0, 1),
        attn_dp=ParallelGroup(object(), (0,), 0, 1),
        moe_tp=ParallelGroup(moe_tp_process_group, (0, 1), 0, 2),
    )
    block = object.__new__(Glm4MoeLiteSparseMoeBlock)
    nn.Module.__init__(block)
    block.parallel_context = context
    block.moe_communication = AllReduceMoeCommunication(context.world.all_reduce)
    block.parallel_collectives = None
    block.mlp_chunk_size = 8
    block.shared_experts = nn.Identity()
    block._routed_chunk = lambda hidden_states: hidden_states.clone()
    block.moe_execution = block._create_moe_execution()
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    with (
        patch.object(dist, "all_reduce") as all_reduce,
        patch(
            "sparseengine.models.glm4_moe_lite.get_context",
            return_value=SimpleNamespace(is_prefill=is_prefill),
        ),
    ):
        output = block(hidden_states)

    torch.testing.assert_close(output, hidden_states * 2)
    assert all_reduce.call_count == expected_reductions
    assert all(
        call.kwargs["group"] is context.world.process_group
        for call in all_reduce.call_args_list
    )


def test_glm_tp1_prefill_uses_fused_routed_and_shared_path() -> None:
    context = _ep_context()
    block = object.__new__(Glm4MoeLiteSparseMoeBlock)
    nn.Module.__init__(block)
    block.parallel_context = context
    block.moe_communication = AllReduceMoeCommunication(context.world.all_reduce)
    block.parallel_collectives = None
    block.mlp_chunk_size = 8
    block.experts = SimpleNamespace(
        fuses_shared_prefill=True,
        fuses_shared_decode=True,
        shared_fusion_token_limit=None,
    )
    block.shared_experts = None
    block._routed_and_shared_chunk = lambda hidden_states: hidden_states * 3
    block._routed_chunk = Mock(side_effect=AssertionError("routed-only path used"))
    block._shared_chunk = Mock(side_effect=AssertionError("shared path used"))
    block.moe_execution = block._create_moe_execution()
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    with patch(
        "sparseengine.models.glm4_moe_lite.get_context",
        return_value=SimpleNamespace(is_prefill=True),
    ):
        output = block(hidden_states)

    torch.testing.assert_close(output, hidden_states * 3)
    block._routed_chunk.assert_not_called()
    block._shared_chunk.assert_not_called()


def test_glm_sparse_moe_reduces_hybrid_tp_ep_shards_over_outer_world() -> None:
    world_process_group = object()
    context = ParallelContext(
        world=ParallelGroup(world_process_group, (0, 1, 2, 3), 0, 4),
        attn_tp=ParallelGroup(world_process_group, (0, 1, 2, 3), 0, 4),
        moe_ep=ParallelGroup(object(), (0, 2), 0, 2),
        attn_dp=ParallelGroup(None, (0,), 0, 1),
        moe_tp=ParallelGroup(object(), (0, 1), 0, 2),
    )
    block = object.__new__(Glm4MoeLiteSparseMoeBlock)
    nn.Module.__init__(block)
    block.parallel_context = context
    block.moe_communication = AllReduceMoeCommunication(context.world.all_reduce)
    block.parallel_collectives = None
    block.mlp_chunk_size = 8
    block.shared_experts = nn.Identity()
    block._routed_chunk = lambda hidden_states: hidden_states.clone()
    block.moe_execution = block._create_moe_execution()
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    with patch.object(dist, "all_reduce") as all_reduce:
        output = block(hidden_states)

    torch.testing.assert_close(output, hidden_states * 2)
    all_reduce.assert_called_once()
    assert all_reduce.call_args.kwargs["group"] is world_process_group


def test_glm_hybrid_tp_ep_shared_expert_defers_reduction_to_moe_block() -> None:
    world_process_group = object()
    context = ParallelContext(
        world=ParallelGroup(world_process_group, (0, 1), 0, 2),
        attn_tp=ParallelGroup(world_process_group, (0, 1), 0, 2),
        moe_ep=ParallelGroup(world_process_group, (0, 1), 0, 2),
        attn_dp=ParallelGroup(None, (0,), 0, 1),
        moe_tp=ParallelGroup(None, (0,), 0, 1),
    )
    config = _config()
    with _construction_context(context):
        block = Glm4MoeLiteSparseMoeBlock(
            config,
            mlp_chunk_size=config.mlp_chunk_size,
            decode_graph=False,
        )

    assert block.shared_experts is not None
    assert block.shared_experts.down_proj.reduce_results is False


@pytest.mark.parametrize("reduced_scale", [1, 2])
def test_glm_moe_debug_contract_populates_model_runner_summaries(reduced_scale) -> None:
    context = _ep_context()
    block = object.__new__(Glm4MoeLiteSparseMoeBlock)
    nn.Module.__init__(block)
    block.parallel_context = context
    block.moe_communication = AllReduceMoeCommunication(
        lambda x: x * reduced_scale
    )
    block.parallel_collectives = None
    block.mlp_chunk_size = 8

    class _Gate(nn.Module):
        def forward(self, hidden_states):
            tokens = int(hidden_states.shape[0])
            logits = torch.arange(64, dtype=torch.float32).expand(tokens, -1)
            topk_ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.long).expand(
                tokens, -1
            )
            topk_weights = torch.full((tokens, 4), 0.25, dtype=torch.float32)
            return logits, topk_weights, topk_ids

    class _Experts(nn.Module):
        local_expert_start = 0
        local_expert_end = 16

        def forward(self, hidden_states, topk_ids, topk_weights):
            del topk_ids, topk_weights
            return hidden_states * 2

    block.gate = _Gate()
    block.experts = _Experts()
    block.shared_experts = nn.Identity()
    block.moe_execution = block._create_moe_execution()
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    with (
        patch.dict(os.environ, {"SPARSEENGINE_DEBUG_MOE": "1"}),
        patch.object(
            device_runtime,
            "is_stream_capturing",
            return_value=True,
        ),
    ):
        output = block(hidden_states)

    torch.testing.assert_close(output, hidden_states * (2 * reduced_scale + 1))
    assert isinstance(block.debug_last_local_hit_count, torch.Tensor)
    torch.testing.assert_close(block.debug_last_local_output, hidden_states * 2)
    torch.testing.assert_close(block.debug_last_routed_output, hidden_states * (2 * reduced_scale))
    torch.testing.assert_close(block.debug_last_output, output)

    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(mlp=nn.Identity()),
                SimpleNamespace(mlp=block),
            ]
        )
    )
    runner.parallel_context = context
    runner.sparse_controller = SimpleNamespace(debug_state_summary=dict)
    runner.prefix_cache_coordinator = None
    runner.debug_last_logits = torch.ones((1, 8), dtype=torch.float32)
    runner.rank = 0
    runner.world_size = 1
    runner.device = torch.device("cpu")

    summary = runner.debug_sparse_state_summary()
    assert set(summary["moe_synced"]) == {"1"}
    assert set(summary["moe_local"]) == {"1"}
    assert summary["moe_local"]["1"]["local_expert_start"] == 0
    assert summary["moe_local"]["1"]["local_expert_end"] == 16
    assert summary["moe_local"]["1"]["local_hit_count"] == 8
    assert summary["moe_synced"]["1"]["output"]["shape"] == [2, 4]
    assert summary["parallel"]["configured"] == {
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
        "data_parallel_size": 1,
        "world_size": 1,
    }
    assert summary["parallel"]["effective"]["moe_ep"] == {
        "rank": 0,
        "size": 1,
        "ranks": [0],
    }

    cpu_states = runner.debug_moe_states_cpu()
    assert cpu_states is not None
    assert set(cpu_states) == {1}
    torch.testing.assert_close(cpu_states[1]["output"], output)
    consistency = runner.debug_replica_consistency()
    assert consistency is not None
    assert set(consistency["moe_layers"]) == {"1"}
    assert consistency["moe_layers"]["1"]["topk_ids_mismatch"] is False
