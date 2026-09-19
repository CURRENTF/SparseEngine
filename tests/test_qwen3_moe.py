from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig

from sparseengine.config import QuantizationConfig
from sparseengine.distributed import ParallelContext, ParallelGroup
from sparseengine.layers.layernorm import RMSNorm
from sparseengine.models.qwen3 import Qwen3Attention
from sparseengine.models.qwen3 import build_qwen3_prefill_attention_op
from sparseengine.models.qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoePackedExperts,
    Qwen3MoeSparseMoeBlock,
)
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.moe import (
    FlashInferCutlassFp8MoeProvider,
    TritonMoeProvider,
)
from sparseengine.quantization.fp8 import fp8_blockwise_linear_reference
from sparseengine.utils.loader import load_model


class _TestRouterProvider:
    name = "test_router"

    def run(self, spec, router_logits, correction_bias=None, **_kwargs):
        assert correction_bias is None
        probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        weights, ids = probabilities.topk(spec.top_k, dim=-1)
        if spec.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(router_logits.dtype), ids


@pytest.fixture(autouse=True)
def _bind_test_router_provider():
    with patch(
        "sparseengine.models.qwen3_moe.resolve_moe_router_provider",
        return_value=_TestRouterProvider(),
    ):
        yield


def _config(**overrides) -> Qwen3MoeConfig:
    values = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
        "hidden_act": "silu",
        "attention_bias": False,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    return Qwen3MoeConfig(**values)


def _fp8_config(**overrides) -> Qwen3MoeConfig:
    values = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 128,
    }
    values.update(overrides)
    config = _config(**values)
    config.quantization_config = QuantizationConfig(
        enabled=True,
        quant_method="fp8",
        weight_dtype="e4m3",
        activation_scheme="dynamic",
        weight_block_size=(128, 128),
        model_name="Qwen3MoE",
    )
    return config


def _ep_context(ep_rank: int, ep_size: int) -> ParallelContext:
    ranks = tuple(range(ep_size))
    return ParallelContext(
        world=ParallelGroup(None, ranks, ep_rank, ep_size),
        moe_tp=ParallelGroup(None, (ep_rank,), 0, 1),
        attn_tp=ParallelGroup(None, (ep_rank,), 0, 1),
        moe_ep=ParallelGroup(None, ranks, ep_rank, ep_size),
        attn_dp=ParallelGroup(None, (ep_rank,), 0, 1),
    )


def _tp_context(tp_rank: int, tp_size: int) -> ParallelContext:
    ranks = tuple(range(tp_size))
    return ParallelContext(
        world=ParallelGroup(None, ranks, tp_rank, tp_size),
        moe_tp=ParallelGroup(None, ranks, tp_rank, tp_size),
        attn_tp=ParallelGroup(None, ranks, tp_rank, tp_size),
        moe_ep=ParallelGroup(None, (tp_rank,), 0, 1),
        attn_dp=ParallelGroup(None, (tp_rank,), 0, 1),
    )


def _hybrid_context(world_rank: int) -> ParallelContext:
    ranks = (0, 1, 2, 3)
    moe_tp_ranks = (0, 1) if world_rank < 2 else (2, 3)
    moe_ep_ranks = (0, 2) if world_rank % 2 == 0 else (1, 3)
    return ParallelContext(
        world=ParallelGroup(None, ranks, world_rank, 4),
        attn_tp=ParallelGroup(None, ranks, world_rank, 4),
        moe_ep=ParallelGroup(
            None, moe_ep_ranks, moe_ep_ranks.index(world_rank), 2
        ),
        attn_dp=ParallelGroup(None, (world_rank,), 0, 1),
        moe_tp=ParallelGroup(
            None, moe_tp_ranks, moe_tp_ranks.index(world_rank), 2
        ),
    )


def _instantiate_model(config, context, full_attention_provider=None):
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "sparseengine.models.qwen3_moe.get_parallel_context",
                return_value=context,
            )
        )
        stack.enter_context(
            patch("sparseengine.models.qwen3.get_parallel_context", return_value=context)
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
        fp8_enabled = bool(
            getattr(getattr(config, "quantization_config", None), "enabled", False)
        )
        stack.enter_context(
            patch(
                "sparseengine.models.qwen3_moe.resolve_moe_provider",
                return_value=(
                    FlashInferCutlassFp8MoeProvider()
                    if fp8_enabled
                    else TritonMoeProvider()
                ),
            )
        )
        if fp8_enabled:
            stack.enter_context(
                patch(
                    "sparseengine.layers.linear.QuantizationRegistry."
                    "resolve_linear_provider",
                    return_value=Mock(return_value=None),
                )
            )
        return Qwen3MoeForCausalLM(
            config,
            full_attention_provider=full_attention_provider,
        )


@pytest.mark.parametrize(
    ("tp_size", "expected_q_heads", "expected_kv_heads"),
    [(1, 32, 4), (2, 16, 2)],
)
def test_qwen3_moe_builds_real_prefill_shape_spec(
    tp_size,
    expected_q_heads,
    expected_kv_heads,
):
    config = _config(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
    )
    prepared = SimpleNamespace(name="prepared")
    with patch(
        "sparseengine.models.attention_runtime.prepare_prefill_attention_op",
        return_value=prepared,
    ) as prepare:
        actual = build_qwen3_prefill_attention_op(
            config,
            engine_config=SimpleNamespace(sparse_method=""),
            parallel_context=_tp_context(0, tp_size),
            device=torch.device("cuda", 0),
        )

    spec = prepare.call_args.args[0]
    assert actual is prepared
    assert spec.num_query_heads == expected_q_heads
    assert spec.num_kv_heads == expected_kv_heads
    assert spec.head_dim == 128
    assert spec.page_size == 1
    assert not spec.layer_varying_page_table
    assert spec.score_output is AttentionScoreKind.NONE


@pytest.mark.parametrize(
    ("sparse_method", "expected_layer_varying"),
    [
        ("", False),
        ("omnikv", False),
        ("quest", False),
        ("snapkv", True),
        ("h2o", True),
    ],
)
def test_qwen3_prefill_builder_preserves_orthogonal_flashprefill_semantics(
    sparse_method,
    expected_layer_varying,
):
    engine_config = SimpleNamespace(
        sparse_method=sparse_method,
        sparse_prefill_score_mode=None,
        h2o_prefill_score_window=0,
        prefill_sparse_method="flashprefill_v2",
        flashprefill_v2_k_block_m=128,
        flashprefill_v2_k_block_n=128,
        flashprefill_v2_abs_threshold=0.1,
        flashprefill_v2_attention_sink_blocks=2,
        flashprefill_v2_window_blocks=4,
        flashprefill_v2_last_query_blocks=8,
        flashprefill_v2_min_sparse_q_len=4096,
        flashprefill_v2_use_mean_correction=True,
    )
    prepared = SimpleNamespace(name="flashprefill_v2")
    with patch(
        "sparseengine.models.attention_runtime.prepare_prefill_attention_op",
        return_value=prepared,
    ) as prepare:
        actual = build_qwen3_prefill_attention_op(
            _config(),
            engine_config=engine_config,
            parallel_context=_tp_context(0, 1),
            device=torch.device("cuda", 0),
        )

    spec = prepare.call_args.args[0]
    assert actual is prepared
    assert spec.prefill_sparse_method == "flashprefill_v2"
    assert spec.flashprefill_v2 is not None
    assert spec.flashprefill_v2.k_block_n == 128
    assert spec.flashprefill_v2.min_sparse_q_len == 4096
    assert spec.layer_varying_page_table is expected_layer_varying
    assert spec.score_output is AttentionScoreKind.NONE
    assert spec.return_softmax_lse is False


def test_qwen3_moe_shares_and_closes_full_attention_provider():
    prepared = SimpleNamespace(name="sgl_fa3")
    decode = SimpleNamespace(name="triton_decode")
    full_attention = SimpleNamespace(
        prefill_op=prepared,
        decode_op=decode,
        prefill_name=prepared.name,
        decode_name=decode.name,
        close=Mock(),
    )

    def bind(model):
        for layer in model.layers:
            layer.self_attn.attn.full_attention_provider = full_attention
            layer.self_attn.attn.prefill_op = prepared
            layer.self_attn.attn.decode_op = decode
        return len(model.layers)

    full_attention.bind = bind
    model = _instantiate_model(
        _config(num_hidden_layers=2),
        _tp_context(0, 1),
        full_attention_provider=full_attention,
    )

    assert all(
        layer.self_attn.attn.prefill_op is prepared
        for layer in model.model.layers
    )
    model.close_runtime_operators()
    full_attention.close.assert_called_once_with()


def test_qwen3_sparse_prefill_uses_resolved_provider():
    prepared = SimpleNamespace(name="prepared")
    with patch(
        "sparseengine.models.attention_runtime.prepare_prefill_attention_op",
        return_value=prepared,
    ) as prepare:
        actual = build_qwen3_prefill_attention_op(
            _config(),
            engine_config=SimpleNamespace(sparse_method="snapkv"),
            parallel_context=_tp_context(0, 1),
            device=torch.device("cuda", 0),
        )

    assert actual is prepared
    spec = prepare.call_args.args[0]
    assert spec.score_output is AttentionScoreKind.NONE
    assert spec.layer_varying_page_table


def _rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    normalized = x_float * torch.rsqrt(
        x_float.square().mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen3_rmsnorm_does_not_modify_input(dtype):
    pytest.importorskip("flashinfer")
    norm = RMSNorm(128).cuda().to(dtype)
    x = torch.randn(3, 128, device="cuda", dtype=dtype)
    original = x.clone()

    actual = norm(x)

    assert torch.equal(x, original)
    torch.testing.assert_close(
        actual,
        _rmsnorm_reference(x, norm.weight, norm.eps),
        rtol=1.0e-2,
        atol=3.0e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_rmsnorm_matches_fp32_reference(dtype, with_residual):
    pytest.importorskip("flashinfer")
    torch.manual_seed(27)
    norm = RMSNorm(128).cuda().to(dtype)
    norm.weight.data.normal_(mean=1.0, std=0.2)
    x = torch.randn(7, 128, device="cuda", dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None

    if residual is None:
        expected = _rmsnorm_reference(x, norm.weight, norm.eps)
        actual = norm(x)
    else:
        merged = x.float() + residual.float()
        expected_residual = merged.to(dtype)
        expected = _rmsnorm_reference(
            merged,
            norm.weight,
            norm.eps,
        ).to(dtype)
        actual, actual_residual = norm(x, residual)
        assert torch.equal(actual_residual, expected_residual)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1.0e-2,
        atol=3.0e-2,
    )


def test_qwen3_attention_passes_raw_key_without_clone():
    class FixedProjection(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, _):
            return self.output

    class PairIdentity(torch.nn.Module):
        def forward(self, _positions, query, key):
            return query, key

    class AttentionIdentity(torch.nn.Module):
        def forward(self, query, _key, _value):
            return query

    class CacheRecorder:
        def save_raw_kv_if_needed(self, _layer_idx, key, _value):
            self.raw_key = key
            self.saved_raw_key = key.clone()

        def save_rope_kv_if_needed(self, _layer_idx, _key, _value):
            pass

    attention = Qwen3Attention.__new__(Qwen3Attention)
    torch.nn.Module.__init__(attention)
    qkv = torch.randn(2, 16)
    expected_raw_key = qkv[:, 8:12].view(2, 1, 4)
    attention.qkv_proj = FixedProjection(qkv)
    attention.o_proj = torch.nn.Identity()
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    attention.rotary_emb = PairIdentity()
    attention.attn = AttentionIdentity()
    attention.q_size = 8
    attention.kv_size = 4
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 4
    attention.qkv_bias = False
    attention.proj_chunk_size = 16
    cache = CacheRecorder()
    context = SimpleNamespace(cache_manager=cache, now_layer_idx=0)

    with patch("sparseengine.models.qwen3.get_context", return_value=context):
        attention(torch.arange(2), torch.empty(2, 8))

    assert cache.raw_key.data_ptr() == expected_raw_key.data_ptr()
    assert torch.equal(cache.saved_raw_key, expected_raw_key)


def test_moe_block_uses_bound_router_provider():
    config = _config()
    context = _ep_context(0, 1)
    with (
        patch("sparseengine.models.qwen3_moe.get_parallel_context", return_value=context),
        patch(
            "sparseengine.models.qwen3_moe.resolve_moe_provider",
            return_value=TritonMoeProvider(),
        ),
    ):
        block = Qwen3MoeSparseMoeBlock(config)
    assert block.gate.op_spec.num_experts == config.num_experts
    assert block.gate.op_spec.top_k == config.num_experts_per_tok
    hidden_states = torch.randn(3, config.hidden_size)
    expected = torch.randn_like(hidden_states)

    with (
        patch.object(block.gate, "forward", return_value=(
            torch.empty(3, config.num_experts),
            torch.empty(3, config.num_experts_per_tok),
            torch.empty(3, config.num_experts_per_tok, dtype=torch.int64),
        )),
        patch.object(block.experts, "forward", return_value=expected) as triton_forward,
    ):
        actual = block(hidden_states)

    assert torch.equal(actual, expected)
    triton_forward.assert_called_once()


def test_moe_block_reduces_in_activation_dtype():
    config = _config()
    context = _ep_context(0, 1)
    with (
        patch("sparseengine.models.qwen3_moe.get_parallel_context", return_value=context),
        patch(
            "sparseengine.models.qwen3_moe.resolve_moe_provider",
            return_value=TritonMoeProvider(),
        ),
    ):
        block = Qwen3MoeSparseMoeBlock(config)
    hidden_states = torch.randn(3, config.hidden_size, dtype=torch.bfloat16)
    local_output = torch.randn(3, config.hidden_size, dtype=torch.bfloat16)

    with (
        patch.object(
            block.gate,
            "forward",
            return_value=(
                torch.empty(3, config.num_experts, dtype=torch.bfloat16),
                torch.empty(3, config.num_experts_per_tok, dtype=torch.bfloat16),
                torch.empty(3, config.num_experts_per_tok, dtype=torch.int64),
            ),
        ),
        patch.object(block.experts, "forward", return_value=local_output),
    ):
        output = block(hidden_states)

    assert output.dtype == hidden_states.dtype
    assert torch.equal(output, local_output)


def test_moe_block_honors_mlp_chunk_size():
    config = _config()
    config.mlp_chunk_size = 2
    context = _ep_context(0, 1)
    with (
        patch("sparseengine.models.qwen3_moe.get_parallel_context", return_value=context),
        patch(
            "sparseengine.models.qwen3_moe.resolve_moe_provider",
            return_value=TritonMoeProvider(),
        ),
    ):
        block = Qwen3MoeSparseMoeBlock(config)

    hidden_states = torch.randn(5, config.hidden_size)
    chunk_sizes = []

    def gate_forward(chunk):
        chunk_sizes.append(int(chunk.shape[0]))
        return (
            torch.empty(chunk.shape[0], config.num_experts),
            torch.empty(chunk.shape[0], config.num_experts_per_tok),
            torch.empty(
                chunk.shape[0],
                config.num_experts_per_tok,
                dtype=torch.int64,
            ),
        )

    with (
        patch.object(block.gate, "forward", side_effect=gate_forward),
        patch.object(
            block.experts,
            "forward",
            side_effect=lambda chunk, _ids, _weights: chunk,
        ),
    ):
        output = block(hidden_states)

    assert chunk_sizes == [2, 2, 1]
    assert torch.equal(output, hidden_states)


def test_moe_chunking_does_not_concatenate_debug_metadata_when_disabled():
    config = _config()
    config.mlp_chunk_size = 2
    context = _ep_context(0, 1)
    with (
        patch(
            "sparseengine.models.qwen3_moe.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparseengine.models.qwen3_moe.resolve_moe_provider",
            return_value=TritonMoeProvider(),
        ),
    ):
        block = Qwen3MoeSparseMoeBlock(config)
    hidden_states = torch.randn(5, config.hidden_size)
    real_cat = torch.cat

    with (
        patch.object(
            block.gate,
            "forward",
            side_effect=lambda chunk: (
                torch.empty(chunk.shape[0], config.num_experts),
                torch.empty(chunk.shape[0], config.num_experts_per_tok),
                torch.empty(
                    chunk.shape[0],
                    config.num_experts_per_tok,
                    dtype=torch.int64,
                ),
            ),
        ),
        patch.object(
            block.experts,
            "forward",
            side_effect=lambda chunk, _ids, _weights: chunk,
        ),
        patch(
            "sparseengine.models.qwen3_moe.torch.cat",
            wraps=real_cat,
        ) as cat,
    ):
        output = block(hidden_states)

    assert torch.equal(output, hidden_states)
    assert cat.call_count == 1




def test_moe_warmup_uses_requested_tokens_and_balanced_local_assignments():
    config = _config(num_experts_per_tok=3)
    context = _ep_context(1, 2)
    model = _instantiate_model(config, context)
    experts = model.model.layers[0].mlp.experts
    expected = torch.zeros(5, config.hidden_size)

    with (
        patch.object(experts, "forward", return_value=expected) as forward,
        patch(
            "sparseengine.models.qwen3_moe.device_runtime.synchronize"
        ) as synchronize,
    ):
        model.warmup_moe(num_tokens=5)

    hidden_states, topk_ids, topk_weights = forward.call_args.args
    assert hidden_states.shape == (5, config.hidden_size)
    assert topk_ids.tolist() == [
        [2, 3, 2],
        [3, 2, 3],
        [2, 3, 2],
        [3, 2, 3],
        [2, 3, 2],
    ]
    assert torch.allclose(topk_weights, torch.full((5, 3), 1 / 3))
    synchronize.assert_called_once()


def test_moe_warmup_rejects_non_positive_token_count():
    config = _config()
    context = _ep_context(0, 1)
    model = _instantiate_model(config, context)

    with pytest.raises(ValueError, match="num_tokens must be > 0"):
        model.warmup_moe(num_tokens=0)


def test_packed_expert_weight_mapping():
    torch.manual_seed(1)
    config = _config()
    context = _ep_context(0, 1)
    with (
        patch("sparseengine.models.qwen3_moe.get_parallel_context", return_value=context),
        patch(
            "sparseengine.models.qwen3_moe.resolve_moe_provider",
            return_value=TritonMoeProvider(),
        ),
    ):
        experts = Qwen3MoePackedExperts(config)

    source_weights = {}
    for expert_id in range(config.num_experts):
        for projection, shape in {
            "gate_proj": (config.moe_intermediate_size, config.hidden_size),
            "up_proj": (config.moe_intermediate_size, config.hidden_size),
            "down_proj": (config.hidden_size, config.moe_intermediate_size),
        }.items():
            weight = torch.randn(shape)
            source_weights[(expert_id, projection)] = weight
            experts.load_expert_weight(expert_id, projection, weight)
    experts.validate_loaded_weights()

    for expert_id in range(config.num_experts):
        assert torch.equal(
            experts.w13_weight[expert_id, : config.moe_intermediate_size],
            source_weights[(expert_id, "gate_proj")],
        )
        assert torch.equal(
            experts.w13_weight[expert_id, config.moe_intermediate_size :],
            source_weights[(expert_id, "up_proj")],
        )
        assert torch.equal(
            experts.w2_weight[expert_id],
            source_weights[(expert_id, "down_proj")],
        )


@pytest.mark.parametrize("tp_size", [2, 4])
def test_packed_expert_tp_shards_cover_each_projection_exactly(tp_size):
    torch.manual_seed(31)
    config = _config(moe_intermediate_size=8)
    source_weights = {
        projection: torch.randn(shape)
        for projection, shape in {
            "gate_proj": (8, config.hidden_size),
            "up_proj": (8, config.hidden_size),
            "down_proj": (config.hidden_size, 8),
        }.items()
    }
    shards = []
    for tp_rank in range(tp_size):
        context = _tp_context(tp_rank, tp_size)
        with (
            patch(
                "sparseengine.models.qwen3_moe.get_parallel_context",
                return_value=context,
            ),
            patch(
                "sparseengine.models.qwen3_moe.resolve_moe_provider",
                return_value=TritonMoeProvider(),
            ),
        ):
            experts = Qwen3MoePackedExperts(config)
        for projection, weight in source_weights.items():
            experts.load_expert_weight(0, projection, weight)
        shards.append(experts)

    assert all(experts.num_local_experts == config.num_experts for experts in shards)
    local_intermediate_size = config.moe_intermediate_size // tp_size
    assert all(
        experts.intermediate_size == local_intermediate_size for experts in shards
    )
    assert torch.equal(
        torch.cat(
            [
                experts.w13_weight[0, :local_intermediate_size]
                for experts in shards
            ]
        ),
        source_weights["gate_proj"],
    )
    assert torch.equal(
        torch.cat(
            [
                experts.w13_weight[0, local_intermediate_size:]
                for experts in shards
            ]
        ),
        source_weights["up_proj"],
    )
    assert torch.equal(
        torch.cat([experts.w2_weight[0] for experts in shards], dim=1),
        source_weights["down_proj"],
    )


def test_packed_experts_own_rank_local_checkpoint_slices():
    config = _config(
        moe_intermediate_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
    )
    model = _instantiate_model(config, _tp_context(1, 2))
    prefix = "model.layers.0.mlp.experts.0."

    gate_target = model.resolve_special_weight(prefix + "gate_proj.expert_weight")
    up_target = model.resolve_special_weight(prefix + "up_proj.expert_weight")
    down_target = model.resolve_special_weight(prefix + "down_proj.expert_weight")

    assert gate_target.module.rank_local_weight_slice(
        (8, 8), loaded_shard_id=gate_target.shard_id
    ) == (slice(4, 8), slice(None))
    assert up_target.module.rank_local_weight_slice(
        (8, 8), loaded_shard_id=up_target.shard_id
    ) == (slice(4, 8), slice(None))
    assert down_target.module.rank_local_weight_slice(
        (8, 8), loaded_shard_id=down_target.shard_id
    ) == (slice(None), slice(4, 8))
    assert not hasattr(model, "rank_local_special_weight_slice")


def test_moe_block_reduces_hybrid_partial_output_over_outer_world():
    config = _config(moe_intermediate_size=8)
    context = _tp_context(0, 2)
    with (
        patch("sparseengine.models.qwen3_moe.get_parallel_context", return_value=context),
        patch(
            "sparseengine.models.qwen3_moe.resolve_moe_provider",
            return_value=TritonMoeProvider(),
        ),
    ):
        block = Qwen3MoeSparseMoeBlock(config)
    hidden_states = torch.randn(3, config.hidden_size, dtype=torch.bfloat16)
    local_output = torch.randn_like(hidden_states)

    with (
        patch.object(
            block.gate,
            "forward",
            return_value=(
                torch.empty(3, config.num_experts, dtype=torch.bfloat16),
                torch.empty(3, config.num_experts_per_tok, dtype=torch.bfloat16),
                torch.empty(3, config.num_experts_per_tok, dtype=torch.int64),
            ),
        ),
        patch.object(block.experts, "forward", return_value=local_output),
        patch.object(
            block.moe_communication,
            "_reduce",
            return_value=local_output,
        ) as reduce,
    ):
        output = block(hidden_states)

    assert output is local_output
    reduce.assert_called_once_with(local_output)


def test_hybrid_expert_shards_cover_ep_and_moe_tp_dimensions():
    config = _config(moe_intermediate_size=8)
    source = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    shards = []
    for world_rank in range(4):
        with (
            patch(
                "sparseengine.models.qwen3_moe.get_parallel_context",
                return_value=_hybrid_context(world_rank),
            ),
            patch(
                "sparseengine.models.qwen3_moe.resolve_moe_provider",
                return_value=TritonMoeProvider(),
            ),
        ):
            experts = Qwen3MoePackedExperts(config)
        experts.load_expert_weight(experts.local_expert_start, "gate_proj", source)
        shards.append(experts)

    assert [(item.local_expert_start, item.local_expert_end) for item in shards] == [
        (0, 2),
        (0, 2),
        (2, 4),
        (2, 4),
    ]
    assert torch.equal(shards[0].w13_weight[0, :4], source[:4])
    assert torch.equal(shards[1].w13_weight[0, :4], source[4:])
    assert torch.equal(shards[2].w13_weight[0, :4], source[:4])
    assert torch.equal(shards[3].w13_weight[0, :4], source[4:])


def test_packed_fp8_expert_weight_and_scale_mapping():
    torch.manual_seed(2)
    config = _fp8_config()
    context = _ep_context(0, 1)
    with (
        patch("sparseengine.models.qwen3_moe.get_parallel_context", return_value=context),
        patch(
            "sparseengine.models.qwen3_moe.resolve_moe_provider",
            return_value=FlashInferCutlassFp8MoeProvider(),
        ),
    ):
        experts = Qwen3MoePackedExperts(config)

    source_weights = {}
    source_scales = {}
    for expert_id in range(config.num_experts):
        for projection, shape in {
            "gate_proj": (config.moe_intermediate_size, config.hidden_size),
            "up_proj": (config.moe_intermediate_size, config.hidden_size),
            "down_proj": (config.hidden_size, config.moe_intermediate_size),
        }.items():
            weight = torch.randn(shape).clamp(-4.0, 4.0).to(torch.float8_e4m3fn)
            scale = torch.rand(1, 1, dtype=torch.bfloat16) + 0.1
            source_weights[(expert_id, projection)] = weight
            source_scales[(expert_id, projection)] = scale
            experts.load_expert_weight(
                expert_id,
                projection,
                weight,
                scale,
            )
    experts.validate_loaded_weights()

    for expert_id in range(config.num_experts):
        assert torch.equal(
            experts.w13_weight[expert_id, : config.moe_intermediate_size],
            source_weights[(expert_id, "up_proj")],
        )
        assert torch.equal(
            experts.w13_weight[expert_id, config.moe_intermediate_size :],
            source_weights[(expert_id, "gate_proj")],
        )
        assert torch.equal(
            experts.w13_scale_inv[expert_id, :1],
            source_scales[(expert_id, "up_proj")],
        )
        assert torch.equal(
            experts.w13_scale_inv[expert_id, 1:],
            source_scales[(expert_id, "gate_proj")],
        )
        assert torch.equal(
            experts.w2_weight[expert_id],
            source_weights[(expert_id, "down_proj")],
        )
        assert torch.equal(
            experts.w2_scale_inv[expert_id],
            source_scales[(expert_id, "down_proj")],
        )


def test_fp8_expert_loader_rejects_missing_scale_and_unaligned_shapes():
    context = _ep_context(0, 1)
    config = _fp8_config()
    with (
        patch("sparseengine.models.qwen3_moe.get_parallel_context", return_value=context),
        patch(
            "sparseengine.models.qwen3_moe.resolve_moe_provider",
            return_value=FlashInferCutlassFp8MoeProvider(),
        ),
    ):
        experts = Qwen3MoePackedExperts(config)

    weight = torch.randn(128, 128).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="Missing FP8 weight_scale_inv"):
        experts.load_expert_weight(0, "gate_proj", weight, None)

    invalid_config = _fp8_config(moe_intermediate_size=96)
    with (
        patch("sparseengine.models.qwen3_moe.get_parallel_context", return_value=context),
        patch(
            "sparseengine.models.qwen3_moe.resolve_moe_provider",
            return_value=FlashInferCutlassFp8MoeProvider(),
        ),
        pytest.raises(ValueError, match="aligned to 128"),
    ):
        Qwen3MoePackedExperts(invalid_config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashinfer_fp8_experts_match_torch_reference():
    pytest.importorskip("flashinfer")
    torch.manual_seed(11)
    config = _fp8_config(num_experts=2, num_experts_per_tok=2)
    context = _ep_context(0, 1)
    with patch("sparseengine.models.qwen3_moe.get_parallel_context", return_value=context):
        experts = Qwen3MoePackedExperts(config).cuda()

    source = {}
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    for expert_id in range(config.num_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight = torch.randn(
                128,
                128,
                device="cuda",
                dtype=torch.float32,
            ) * 0.05
            scale = (
                (weight.abs().amax() / fp8_max)
                .clamp_min(1.0e-12)
                .view(1, 1)
                .to(torch.bfloat16)
            )
            quantized = (weight / scale).to(torch.float8_e4m3fn)
            experts.load_expert_weight(
                expert_id,
                projection,
                quantized,
                scale,
            )
            source[(expert_id, projection)] = (quantized, scale.float())

    hidden_states = torch.randn(
        5,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.2
    topk_ids = torch.tensor(
        [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.55, 0.45], [0.5, 0.5]],
        device="cuda",
        dtype=torch.float32,
    )

    actual = experts(hidden_states, topk_ids, topk_weights)
    expected = torch.zeros_like(hidden_states, dtype=torch.float32)
    for token_idx in range(hidden_states.shape[0]):
        token = hidden_states[token_idx : token_idx + 1]
        for route_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, route_idx])
            gate_weight, gate_scale = source[(expert_id, "gate_proj")]
            up_weight, up_scale = source[(expert_id, "up_proj")]
            down_weight, down_scale = source[(expert_id, "down_proj")]
            gate = fp8_blockwise_linear_reference(
                token,
                gate_weight,
                gate_scale,
            )
            up = fp8_blockwise_linear_reference(
                token,
                up_weight,
                up_scale,
            )
            activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                torch.bfloat16
            )
            routed = fp8_blockwise_linear_reference(
                activated,
                down_weight,
                down_scale,
            )
            expected[token_idx].add_(
                routed[0].float() * topk_weights[token_idx, route_idx]
            )

    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=0.15,
        atol=0.08,
    )


def test_model_maps_only_local_experts_and_validates_all_weights():
    torch.manual_seed(3)
    config = _config()
    model = _instantiate_model(config, _ep_context(1, 2))

    for expert_id in range(config.num_experts):
        for projection, shape in {
            "gate_proj": (config.moe_intermediate_size, config.hidden_size),
            "up_proj": (config.moe_intermediate_size, config.hidden_size),
            "down_proj": (config.hidden_size, config.moe_intermediate_size),
        }.items():
            source_name = f"model.layers.0.mlp.experts.{expert_id}.{projection}.weight"
            target_name = model.map_weight_name(source_name)
            if expert_id < 2:
                assert target_name is None
            else:
                assert target_name is not None
                assert model.load_special_weight(
                    target_name,
                    torch.randn(shape),
                    None,
                ) == 1

    packed_names = {
        name
        for name, _ in model.named_parameters()
        if name.endswith(".mlp.experts.w13_weight")
        or name.endswith(".mlp.experts.w2_weight")
    }
    dense_names = {name for name, _ in model.named_parameters()} - packed_names
    model.validate_loaded_weights(dense_names)
    assert len(model._intentionally_skipped_expert_weights) == 6


def test_missing_local_expert_weight_fails_validation():
    config = _config()
    model = _instantiate_model(config, _ep_context(0, 2))
    experts = model.model.layers[0].mlp.experts
    with pytest.raises(ValueError, match="Missing local Qwen3MoE expert weights"):
        experts.validate_loaded_weights()


def test_checkpoint_loader_loads_local_experts_and_skips_remote(tmp_path):
    torch.manual_seed(5)
    config = _config()
    context = _ep_context(1, 2)
    template = _instantiate_model(config, context)
    checkpoint = {}
    for name, parameter in template.named_parameters():
        if name.endswith(".mlp.experts.w13_weight") or name.endswith(
            ".mlp.experts.w2_weight"
        ):
            continue
        value = torch.randn(parameter.shape, dtype=parameter.dtype)
        if name.endswith(".self_attn.qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            q_size = config.num_attention_heads * config.head_dim
            kv_size = config.num_key_value_heads * config.head_dim
            checkpoint[prefix + "q_proj.weight"] = value[:q_size].clone()
            checkpoint[prefix + "k_proj.weight"] = value[q_size : q_size + kv_size].clone()
            checkpoint[prefix + "v_proj.weight"] = value[q_size + kv_size :].clone()
        else:
            checkpoint[name] = value

    local_sources = {}
    for expert_id in range(config.num_experts):
        for projection, shape in {
            "gate_proj": (config.moe_intermediate_size, config.hidden_size),
            "up_proj": (config.moe_intermediate_size, config.hidden_size),
            "down_proj": (config.hidden_size, config.moe_intermediate_size),
        }.items():
            name = f"model.layers.0.mlp.experts.{expert_id}.{projection}.weight"
            checkpoint[name] = torch.randn(shape)
            if expert_id >= 2:
                local_sources[(expert_id, projection)] = checkpoint[name]
    save_file(checkpoint, tmp_path / "model.safetensors")

    target = _instantiate_model(config, context)
    load_model(target, str(tmp_path), tp_rank=0, tp_size=1)

    experts = target.model.layers[0].mlp.experts
    assert torch.equal(experts.w13_weight[0, : config.moe_intermediate_size], local_sources[(2, "gate_proj")])
    assert torch.equal(experts.w13_weight[0, config.moe_intermediate_size :], local_sources[(2, "up_proj")])
    assert torch.equal(experts.w2_weight[1], local_sources[(3, "down_proj")])
    assert len(target._intentionally_skipped_expert_weights) == 6


def test_checkpoint_loader_reads_tp_local_expert_projection_shards(tmp_path):
    torch.manual_seed(41)
    config = _config(
        moe_intermediate_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
    )
    source_model = _instantiate_model(config, _ep_context(0, 1))
    checkpoint = {}
    for name, parameter in source_model.named_parameters():
        if name.endswith(".mlp.experts.w13_weight") or name.endswith(
            ".mlp.experts.w2_weight"
        ):
            continue
        value = torch.randn(parameter.shape, dtype=parameter.dtype)
        if name.endswith(".self_attn.qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            q_size = config.num_attention_heads * config.head_dim
            kv_size = config.num_key_value_heads * config.head_dim
            checkpoint[prefix + "q_proj.weight"] = value[:q_size].clone()
            checkpoint[prefix + "k_proj.weight"] = value[
                q_size : q_size + kv_size
            ].clone()
            checkpoint[prefix + "v_proj.weight"] = value[q_size + kv_size :].clone()
        else:
            checkpoint[name] = value

    source_experts = {}
    for expert_id in range(config.num_experts):
        for projection, shape in {
            "gate_proj": (8, config.hidden_size),
            "up_proj": (8, config.hidden_size),
            "down_proj": (config.hidden_size, 8),
        }.items():
            name = f"model.layers.0.mlp.experts.{expert_id}.{projection}.weight"
            checkpoint[name] = torch.randn(shape)
            source_experts[(expert_id, projection)] = checkpoint[name]
    save_file(checkpoint, tmp_path / "model.safetensors")

    target = _instantiate_model(config, _tp_context(1, 2))
    load_model(target, str(tmp_path), tp_rank=1, tp_size=2)

    experts = target.model.layers[0].mlp.experts
    for expert_id in range(config.num_experts):
        assert torch.equal(
            experts.w13_weight[expert_id, :4],
            source_experts[(expert_id, "gate_proj")][4:8],
        )
        assert torch.equal(
            experts.w13_weight[expert_id, 4:],
            source_experts[(expert_id, "up_proj")][4:8],
        )
        assert torch.equal(
            experts.w2_weight[expert_id],
            source_experts[(expert_id, "down_proj")][:, 4:8],
        )


def test_checkpoint_loader_loads_local_fp8_experts_and_scales(tmp_path):
    torch.manual_seed(7)
    config = _fp8_config()
    context = _ep_context(1, 2)
    template = _instantiate_model(config, context)
    checkpoint = {}
    for name, parameter in template.named_parameters():
        if name.endswith(".mlp.experts.w13_weight") or name.endswith(
            ".mlp.experts.w2_weight"
        ):
            continue
        if name.endswith(".self_attn.qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            for projection in ("q", "k", "v"):
                source_name = prefix + f"{projection}_proj.weight"
                checkpoint[source_name] = (
                    torch.randn(128, 128)
                    .clamp(-4.0, 4.0)
                    .to(torch.float8_e4m3fn)
                )
                checkpoint[
                    source_name[: -len(".weight")] + ".weight_scale_inv"
                ] = (torch.rand(1, 1) + 0.1).to(torch.bfloat16)
        elif name.endswith(".self_attn.o_proj.weight"):
            checkpoint[name] = (
                torch.randn(parameter.shape)
                .clamp(-4.0, 4.0)
                .to(torch.float8_e4m3fn)
            )
            checkpoint[name[: -len(".weight")] + ".weight_scale_inv"] = (
                (torch.rand(1, 1) + 0.1).to(torch.bfloat16)
            )
        else:
            checkpoint[name] = torch.randn(parameter.shape, dtype=parameter.dtype)

    local_sources = {}
    for expert_id in range(config.num_experts):
        for projection, shape in {
            "gate_proj": (128, 128),
            "up_proj": (128, 128),
            "down_proj": (128, 128),
        }.items():
            name = (
                f"model.layers.0.mlp.experts.{expert_id}."
                f"{projection}.weight"
            )
            checkpoint[name] = (
                torch.randn(shape).clamp(-4.0, 4.0).to(torch.float8_e4m3fn)
            )
            scale_name = name[: -len(".weight")] + ".weight_scale_inv"
            checkpoint[scale_name] = (torch.rand(1, 1) + 0.1).to(
                torch.bfloat16
            )
            if expert_id >= 2:
                local_sources[(expert_id, projection)] = (
                    checkpoint[name],
                    checkpoint[scale_name],
                )
    save_file(checkpoint, tmp_path / "model.safetensors")

    target = _instantiate_model(config, context)
    load_model(target, str(tmp_path), tp_rank=0, tp_size=1)

    experts = target.model.layers[0].mlp.experts
    assert torch.equal(
        experts.w13_weight[0, :128],
        local_sources[(2, "up_proj")][0],
    )
    assert torch.equal(
        experts.w13_weight[0, 128:],
        local_sources[(2, "gate_proj")][0],
    )
    assert torch.equal(
        experts.w13_scale_inv[0, :1],
        local_sources[(2, "up_proj")][1],
    )
    assert torch.equal(
        experts.w13_scale_inv[0, 1:],
        local_sources[(2, "gate_proj")][1],
    )
    assert torch.equal(
        experts.w2_weight[1],
        local_sources[(3, "down_proj")][0],
    )
    assert len(target._intentionally_skipped_expert_weights) == 6
    assert len(target._intentionally_skipped_expert_scales) == 6
