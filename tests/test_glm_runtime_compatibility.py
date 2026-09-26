from __future__ import annotations

import pytest
from sparseengine.engine.cache_manager.storage import CacheLayout
from glm_test_helpers import _glm_config


def test_glm_config_selects_mla_latent_layout():
    config = _glm_config()

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value


def test_mla_prefill_workspace_budget_must_be_positive():
    with pytest.raises(ValueError, match="mla_prefill_workspace_bytes"):
        _glm_config(mla_prefill_workspace_bytes=0)


def test_glm_config_accepts_latent_quest_without_prefix_cache():
    config = _glm_config(
        sparse_method="quest",
        decode_graph=True,
    )

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.sparse_method == "quest"
    assert config.decode_graph is True
    assert config.enable_prefix_caching is False


def test_glm_config_rejects_flashprefill_v2_for_mla_latent_storage():
    with pytest.raises(
        NotImplementedError,
        match="flashprefill_v2.*requires explicit KV cache storage",
    ):
        _glm_config(
            sparse_method="quest",
            prefill_sparse_method="flashprefill_v2",
            flashprefill_v2_abs_threshold=0.1,
        )


@pytest.mark.parametrize("tp,dp", [(1, 1), (2, 1), (2, 2)])
def test_glm_latent_quest_accepts_prefix_cache_with_graph(tp, dp):
    config = _glm_config(
        sparse_method="quest",
        enable_prefix_caching=True,
        decode_graph=True,
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        expert_parallel_size=tp * dp,
    )
    assert config.resolved_prefix_cache_mode == "radix"
    assert config.prefix_cache_block_size == config.quest_chunk_size
    assert config.decode_graph is True


def test_glm_latent_quest_prefix_offload_accepts_graph():
    config = _glm_config(
        sparse_method="quest",
        enable_prefix_caching=True,
        enable_prefix_cache_offload=True,
        prefix_cache_host_size_gb=1,
        decode_graph=True,
    )
    assert config.enable_prefix_cache_offload
    assert config.resolved_prefix_cache_mode == "radix"
    assert config.decode_graph


@pytest.mark.parametrize("expert_parallel_size", [2, 4])
def test_glm_config_rejects_ep_that_expands_the_attention_world(expert_parallel_size):
    with pytest.raises(ValueError, match="must be divisible by MoE EP"):
        _glm_config(tensor_parallel_size=1, expert_parallel_size=expert_parallel_size)


@pytest.mark.parametrize(
    (
        "tensor_parallel_size",
        "expert_parallel_size",
        "world_size",
        "moe_tp_size",
    ),
    [
        (2, 2, 2, 1),
        (4, 2, 4, 2),
        (4, 4, 4, 1),
    ],
)
def test_glm_config_accepts_outer_tp_moe_ep_layout(
    tensor_parallel_size,
    expert_parallel_size,
    world_size,
    moe_tp_size,
):
    config = _glm_config(
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
    )

    assert config.world_size == world_size
    assert config.moe_tp_size == moe_tp_size


def test_glm_hybrid_checks_routed_width_against_moe_tp_not_outer_tp():
    config = _glm_config(
        tensor_parallel_size=4,
        expert_parallel_size=2,
        hf_overrides={"moe_intermediate_size": 6},
    )

    assert config.moe_tp_size == 2


def test_glm_hybrid_rejects_routed_width_not_divisible_by_moe_tp():
    with pytest.raises(ValueError, match="divisible by MoE TP"):
        _glm_config(
            tensor_parallel_size=4,
            expert_parallel_size=2,
            hf_overrides={"moe_intermediate_size": 5},
        )


def test_glm_config_rejects_nondivisible_outer_tp_moe_ep_layout():
    with pytest.raises(ValueError, match="must be divisible by MoE EP"):
        _glm_config(tensor_parallel_size=2, expert_parallel_size=4)


def test_glm_config_rejects_dp_without_matching_expert_ranks():
    with pytest.raises(ValueError, match="EP=world size"):
        _glm_config(data_parallel_size=2)


def test_glm_config_defaults_to_bounded_vanilla_startup_graph_capture():
    config = _glm_config(decode_graph=True)

    assert config.decode_graph_startup_capture is True
    assert config.decode_graph_startup_capture_limit == 32


def test_glm_config_rejects_disabling_default_startup_graph_capture():
    with pytest.raises(
        ValueError,
        match="requires startup capture",
    ):
        _glm_config(
            decode_graph=True,
            decode_graph_startup_capture=False,
        )


def test_glm_config_uses_shared_sparse_startup_capture_budget():
    config = _glm_config(
        decode_graph=True,
        sparse_method="snapkv",
    )

    assert config.decode_graph_startup_capture is True
    assert config.decode_graph_startup_capture_limit == 32


def test_glm_config_rejects_startup_capture_without_cuda_graph():
    with pytest.raises(ValueError, match="requires decode_graph=True"):
        _glm_config(decode_graph=False, decode_graph_startup_capture=True)


def test_glm_config_rejects_disabling_sparse_startup_capture():
    with pytest.raises(
        ValueError,
        match="requires startup capture",
    ):
        _glm_config(
            decode_graph=True,
            decode_graph_startup_capture=False,
            sparse_method="snapkv",
        )


def test_glm_config_rejects_startup_budget_smaller_than_batch_plan():
    with pytest.raises(ValueError, match="must cover every batch bucket"):
        _glm_config(
            decode_graph=True,
            decode_graph_startup_capture=True,
            decode_graph_capture_sizes=[1, 2, 3, 4, 5],
            decode_graph_startup_capture_limit=4,
            max_decoding_seqs=5,
        )


def test_hybrid_attention_rejects_only_unimplemented_moe_tp():
    ep_size = 2
    from sparseengine.distributed import ParallelTopology
    from sparseengine.models.spec import resolve_model_spec

    topology = ParallelTopology(attn_tp_size=2, moe_ep_size=ep_size, attn_dp_size=2)
    with pytest.raises(ValueError, match="engine currently supports DP attention only"):
        resolve_model_spec("glm4_moe_lite").validate_parallel_execution(topology)
    with pytest.raises(ValueError, match="engine currently supports DP attention only"):
        _glm_config(tensor_parallel_size=2, data_parallel_size=2, expert_parallel_size=ep_size)


def test_glm_dense_mlp_width_uses_attention_tp_even_with_pure_ep_experts():
    with pytest.raises(ValueError, match="attention TP.*intermediate_size"):
        _glm_config(tensor_parallel_size=4, expert_parallel_size=4,
                    hf_overrides={"intermediate_size": 6})


@pytest.mark.parametrize("model_type,dp_size,tp_size", [
    ("glm4_moe_lite", 2, 2), ("qwen3_moe", 2, 4),
    ("minimax_m2", 4, 2), ("glm4_moe_lite", 3, 2),
])
def test_hybrid_attention_execution_has_no_four_gpu_limit(dp_size, tp_size, model_type):
    from sparseengine.distributed import ParallelTopology
    from sparseengine.models.spec import resolve_model_spec

    topology = ParallelTopology(tp_size, dp_size * tp_size, dp_size)
    resolve_model_spec(model_type).validate_parallel_execution(topology)
