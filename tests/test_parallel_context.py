from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist

import sparseengine.platforms as platforms
from sparseengine.config import Config, RuntimeLayout
from sparseengine.distributed import (
    ParallelContext,
    ParallelGroup,
    ParallelTopology,
    parallel_group_ranks,
)
from sparseengine.distributed.parallel_context import (
    get_parallel_context,
    init_parallel_context,
    reset_parallel_context,
)
from sparseengine.engine.cache_manager.base import CacheManager
from sparseengine.layers.embed_head import VocabParallelEmbedding
from sparseengine.layers.linear import ColumnParallelLinear, RowParallelLinear
from sparseengine.platforms.cpu import CpuPlatform


def _replicated_ep_context(world_rank: int = 2, world_size: int = 4) -> ParallelContext:
    return ParallelContext(
        world=ParallelGroup(None, tuple(range(world_size)), world_rank, world_size),
        moe_tp=ParallelGroup(None, (world_rank,), 0, 1),
        attn_tp=ParallelGroup(None, (world_rank,), 0, 1),
        moe_ep=ParallelGroup(None, tuple(range(world_size)), world_rank, world_size),
        attn_dp=ParallelGroup(None, (world_rank,), 0, 1),
    )


class _MinimalCacheManager(CacheManager):
    def allocate_kv_cache(self):
        raise NotImplementedError

    def get_layer_batch_states(self, layer_idx):
        raise NotImplementedError

    def get_layer_kv_cache(self, layer_idx):
        raise NotImplementedError

    def get_layer_store_view(self, layer_idx):
        raise NotImplementedError

    def get_layer_compute_tensors(self, layer_idx, selection=None):
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx):
        raise NotImplementedError

    @property
    def num_free_slots(self):
        return 0

    def free_seq(self, seq_id):
        raise NotImplementedError

    def free_part_slots(self, layer_idx, seq, keep_indices):
        raise NotImplementedError

    def _prepare_prefill(self, seqs):
        raise NotImplementedError

    def _prepare_decode(self, seqs):
        raise NotImplementedError


def _hf_config(model_type: str = "qwen3_moe", *, num_experts: int = 8):
    return SimpleNamespace(
        model_type=model_type,
        dtype=torch.bfloat16,
        max_position_embeddings=32768,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        num_experts=num_experts,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


@pytest.mark.parametrize("sizes", [(1, 1, 1), (1, 4, 4), (4, 2, 1), (4, 4, 2), (3, 3, 2)])
def test_parallel_topology_stage_coordinates_and_group_partitions(sizes):
    # Each physical rank belongs to one group per axis, including crossing
    # attention/MoE boundaries that the engine cannot execute yet.
    topology = ParallelTopology(*sizes)
    groups = parallel_group_ranks(topology)
    assert topology.world_size == topology.attn_dp_size * topology.attn_tp_size
    assert topology.world_size == topology.moe_ep_size * topology.moe_tp_size
    for dimension, partitions in groups.items():
        assert sorted(rank for group in partitions for rank in group) == list(range(topology.world_size))
        assert all(len(group) == getattr(topology, dimension + "_size") for group in partitions)
    for rank in range(topology.world_size):
        dp, tp = topology.attn_ranks(rank)
        ep, mtp = topology.moe_ranks(rank)
        assert dp * topology.attn_tp_size + tp == rank
        assert ep * topology.moe_tp_size + mtp == rank
        assert all(topology.attn_ranks(peer)[0] == dp for peer in groups["attn_tp"][dp])
        assert all(topology.attn_ranks(peer)[1] == tp for peer in groups["attn_dp"][tp])
        assert all(topology.moe_ranks(peer)[0] == ep for peer in groups["moe_tp"][ep])
        assert all(topology.moe_ranks(peer)[1] == mtp for peer in groups["moe_ep"][mtp])


@pytest.mark.parametrize("sizes", [(0, 1, 1), (4, 3, 1), (1, 2, 1), (1, 1, -1), (1.5, 1, 1)])
def test_parallel_topology_rejects_invalid_sizes(sizes):
    with pytest.raises(ValueError):
        ParallelTopology(*sizes)


@pytest.mark.parametrize("rank", [-1, 4])
def test_stage_coordinates_reject_out_of_world_rank(rank):
    topology = ParallelTopology(2, 2, 2)
    with pytest.raises(ValueError, match="world_rank"):
        topology.attn_ranks(rank)
    with pytest.raises(ValueError, match="world_rank"):
        topology.moe_ranks(rank)


def test_hybrid_moe_parallel_context_uses_explicit_groups():
    reset_parallel_context()
    with (
        patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=4),
            patch.object(dist, "get_rank", return_value=2),
            patch.object(dist, "get_backend", return_value=dist.Backend.GLOO),
            patch.object(dist, "new_group", side_effect=lambda _ranks: object()),
    ):
        context = init_parallel_context(
            topology=ParallelTopology(4, 2, 1),
        )
    assert context.attn_tp.ranks == (0, 1, 2, 3)
    assert context.attn_tp_rank == 2
    assert context.moe_tp.ranks == (2, 3)
    assert context.moe_tp_rank == 0
    assert context.moe_ep.ranks == (0, 2)
    assert context.moe_ep_rank == 1
    reset_parallel_context()


def test_parallel_context_lifecycle_and_local_groups():
    reset_parallel_context()
    fake_groups = []

    def new_group(ranks):
        group = object()
        fake_groups.append((tuple(ranks), group))
        return group

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=4),
        patch.object(dist, "get_rank", return_value=2),
        patch.object(dist, "get_backend", return_value=dist.Backend.GLOO),
        patch.object(dist, "new_group", side_effect=new_group),
    ):
        topology = ParallelTopology(2, 2, 2)
        context = init_parallel_context(topology=topology)
        assert context.world_rank == 2
        assert context.attn_tp_rank == 0
        assert context.attn_tp_size == 2
        assert context.moe_ep_rank == 1
        assert context.moe_ep.ranks == (0, 2)
        assert context.attn_dp_rank == 1
        assert context.attn_dp.ranks == (0, 2)
        assert get_parallel_context() is context
        with pytest.raises(RuntimeError, match="already initialized"):
            init_parallel_context(topology=topology)

    assert sorted(ranks for ranks, _ in fake_groups) == sorted([
        (0, 1),
        (2, 3),
        (0, 2),
        (1, 3),
    ])
    reset_parallel_context()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_parallel_context()


def test_parallel_context_rejects_world_size_mismatch():
    reset_parallel_context()
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=2),
        patch.object(dist, "get_rank", return_value=0),
    ):
        with pytest.raises(ValueError, match="does not match"):
            init_parallel_context(topology=ParallelTopology(4, 4, 1))


def test_ep_broadcast_uses_source_world_rank():
    context = _replicated_ep_context(world_rank=2, world_size=4)
    tensor = torch.tensor([1.0])

    with patch.object(dist, "broadcast", return_value=None) as broadcast:
        returned = context.moe_ep.broadcast(tensor, src_rank=1)

    assert returned is tensor
    broadcast.assert_called_once_with(
        tensor,
        src=1,
        group=context.moe_ep.process_group,
    )


def test_ep_broadcast_rejects_invalid_source_rank():
    context = _replicated_ep_context()

    with pytest.raises(ValueError, match="Broadcast source"):
        context.moe_ep.broadcast(torch.tensor([1.0]), src_rank=4)


@pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.MAX])
def test_parallel_context_collectives_are_always_in_place_torch_operations(op):
    world_group = object()
    context = ParallelContext(
        world=ParallelGroup(world_group, (0, 1, 2, 3), 0, 4),
        moe_tp=ParallelGroup(world_group, (0, 1, 2, 3), 0, 4),
        attn_tp=ParallelGroup(world_group, (0, 1, 2, 3), 0, 4),
        moe_ep=ParallelGroup(None, (0,), 0, 1),
        attn_dp=ParallelGroup(None, (0,), 0, 1),
    )
    tensor = torch.ones(2, 3072, dtype=torch.bfloat16)

    with patch.object(dist, "all_reduce") as all_reduce:
        returned = context.world.all_reduce(tensor, op=op)

    assert returned is tensor
    all_reduce.assert_called_once_with(
        tensor,
        op=op,
        group=world_group,
    )


def test_qwen3_moe_parallel_config_validation(tmp_path):
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        config = Config(model=str(tmp_path), tensor_parallel_size=2, expert_parallel_size=2)
    assert config.world_size == 2
    assert config.weight_loading_workers_per_rank == 1

    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        hybrid = Config(
            model=str(tmp_path), tensor_parallel_size=2, expert_parallel_size=2
        )
    assert hybrid.world_size == 2
    assert hybrid.attn_tp_size == 2
    assert hybrid.moe_ep_size == 2
    assert hybrid.moe_tp_size == 1

    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        config = Config(model=str(tmp_path), tensor_parallel_size=2)
    assert config.tensor_parallel_size == 2
    assert config.expert_parallel_size == 1

    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            Config(model=str(tmp_path), tensor_parallel_size=4)

    fp16 = _hf_config()
    fp16.dtype = torch.float16
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=fp16):
        with pytest.raises(NotImplementedError, match="attention TP supports BF16"):
            Config(model=str(tmp_path), tensor_parallel_size=2)

    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        with pytest.raises(ValueError, match="must be divisible by MoE EP"):
            Config(model=str(tmp_path), tensor_parallel_size=3, expert_parallel_size=2)

    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config(num_experts=7)):
        with pytest.raises(ValueError, match="divisible"):
            Config(model=str(tmp_path), tensor_parallel_size=2, expert_parallel_size=2)

    invalid_layout = _hf_config()
    invalid_layout.decoder_sparse_step = 0
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=invalid_layout):
        with pytest.raises(NotImplementedError, match="every decoder layer"):
            Config(model=str(tmp_path))

    invalid_dtype = _hf_config()
    invalid_dtype.dtype = torch.float32
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=invalid_dtype):
        with pytest.raises(NotImplementedError, match="BF16/FP16 checkpoints"):
            Config(model=str(tmp_path))


def test_qwen3_moe_snapkv_tp_supports_chain_cache_with_decode_graph(tmp_path):
    with (
        patch(
            "sparseengine.configs.runtime.AutoConfig.from_pretrained",
            return_value=_hf_config(),
        ),
        patch("sparseengine.configs.cuda_graph.log_once") as log_once,
    ):
        config = Config(
            model=str(tmp_path),
            sparse_method="snapkv",
            tensor_parallel_size=2,
            enable_prefix_caching=True,
            prefix_cache_mode="chain",
            decode_graph=True,
            decode_graph_capture_sampling=False,
        )

    assert config.resolved_prefix_cache_mode == "chain"
    assert config.enable_prefix_caching is True
    assert any(
        "TP-local sparse selection" in call.args[0]
        for call in log_once.call_args_list
    )


def test_qwen3_moe_vanilla_tp_decode_graph_omits_sparse_selection_warning(
    tmp_path,
):
    with (
        patch(
            "sparseengine.configs.runtime.AutoConfig.from_pretrained",
            return_value=_hf_config(),
        ),
        patch("sparseengine.configs.cuda_graph.log_once") as log_once,
    ):
        config = Config(
            model=str(tmp_path),
            sparse_method="vanilla",
            tensor_parallel_size=2,
            decode_graph=True,
        )

    assert config.sparse_method == ""
    assert not any(
        "TP-local sparse selection" in call.args[0]
        for call in log_once.call_args_list
    )


def test_qwen3_moe_fp8_config_validation(tmp_path):
    hf_config = _hf_config()
    hf_config.architectures = ["Qwen3MoeForCausalLM"]
    hf_config.hidden_size = 128
    hf_config.moe_intermediate_size = 128
    raw_quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "lm_head",
            "model.layers.0.mlp.gate",
            "model.layers.1.mlp.gate",
        ],
    }
    hf_config.quantization_config = raw_quantization_config
    with patch(
        "sparseengine.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        config = Config(model=str(tmp_path), tensor_parallel_size=2, expert_parallel_size=2)
    assert config.quantization_config.enabled

    hf_config.quantization_config = {
        **raw_quantization_config,
        "modules_to_not_convert": ["lm_head"],
    }
    with patch(
        "sparseengine.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        with pytest.raises(ValueError, match="router gate"):
            Config(model=str(tmp_path), tensor_parallel_size=2, expert_parallel_size=2)


def test_qwen3_dense_fp8_config_validation(tmp_path):
    hf_config = _hf_config("qwen3")
    hf_config.architectures = ["Qwen3ForCausalLM"]
    hf_config.hidden_size = 4096
    hf_config.intermediate_size = 12288
    hf_config.head_dim = 128
    hf_config.num_attention_heads = 32
    hf_config.num_key_value_heads = 8
    hf_config.quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    with patch(
        "sparseengine.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        config = Config(model=str(tmp_path), tensor_parallel_size=8)
    assert config.quantization_config.enabled
    assert config.quantization_config.activation_dtype == "bfloat16"

    hf_config.intermediate_size = 12160
    with patch(
        "sparseengine.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        with pytest.raises(ValueError, match="TP-local dense projection"):
            Config(model=str(tmp_path), tensor_parallel_size=8)


def test_qwen3_dense_fp8_rejects_wrong_architecture(tmp_path):
    hf_config = _hf_config("qwen3")
    hf_config.architectures = ["Qwen3MoeForCausalLM"]
    hf_config.quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    with patch(
        "sparseengine.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        with pytest.raises(ValueError, match="Qwen3ForCausalLM"):
            Config(model=str(tmp_path))


def test_dense_config_rejects_expert_or_data_parallelism(tmp_path):
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config("qwen3")):
        with pytest.raises(ValueError, match="does not support expert parallelism"):
            Config(model=str(tmp_path), tensor_parallel_size=2, expert_parallel_size=2)


def test_dense_layers_use_tp_group_in_replicated_ep_topology():
    context = _replicated_ep_context()
    with (
        patch("sparseengine.layers.linear.get_parallel_context", return_value=context),
        patch("sparseengine.layers.embed_head.get_parallel_context", return_value=context),
    ):
        column = ColumnParallelLinear(8, 16)
        row = RowParallelLinear(8, 16)
        embedding = VocabParallelEmbedding(32, 8)

    assert column.weight.shape == (16, 8)
    assert row.weight.shape == (16, 8)
    assert embedding.weight.shape == (32, 8)


def test_vocab_parallel_embedding_reduces_results():
    reduced = torch.randn(2, 4)
    context = SimpleNamespace(
        attn_tp_rank=0,
        attn_tp_size=2,
        attn_tp=SimpleNamespace(all_reduce=Mock(return_value=reduced)),
    )
    with patch(
        "sparseengine.layers.embed_head.get_parallel_context",
        return_value=context,
    ):
        embedding = VocabParallelEmbedding(8, 4)

    output = embedding(torch.tensor([0, 5]))

    assert output is reduced
    context.attn_tp.all_reduce.assert_called_once()


def test_cache_kv_heads_depend_on_tp_not_ep():
    context = _replicated_ep_context()
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=4,
            num_attention_heads=8,
            hidden_size=32,
            head_dim=4,
        ),
        runtime_layout=RuntimeLayout.dense(2),
        max_model_len=128,
        max_num_seqs_in_gpu=2,
        max_num_seqs_in_batch=2,
    )
    with patch.object(platforms, "_current_platform", CpuPlatform()):
        manager = _MinimalCacheManager(config, context)

    assert manager.world_size == 4
    assert manager.tp_size == 1
    assert manager.ep_size == 4
    assert manager.num_kv_heads == 4
