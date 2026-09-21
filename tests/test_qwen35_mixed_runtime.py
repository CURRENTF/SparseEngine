import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sparseengine.platforms as platforms
from sparseengine.platforms import device_runtime
from sparseengine.config import Config, RuntimeLayout
from sparseengine.distributed import ParallelContext, ParallelGroup, ParallelTopology
from sparseengine.engine.cache_manager.base import (
    CacheManager,
    LayerBatchStates,
    resolve_joint_prefix_capacity,
)
from sparseengine.engine.cache_manager.methods.quest import QuestCacheManager, QuestPrefixBlockPayload
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.cache_manager.standard import (
    StandardCacheManager,
    StandardPrefixBlockPayload,
)
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.prefix_cache import (
    PrefixCacheBlock,
    PrefixTransferKind,
    RadixPrefixIndex,
)
from sparseengine.engine.prefix_cache_coordinator import MixedPrefixBlockPayload, PrefixCacheCoordinator
from sparseengine.engine.cache_manager.prefix_cache_mixin import PrefixLookupCache
from sparseengine.engine.recurrent_state_manager import (
    RecurrentPrefixPayload,
    RecurrentStateManager,
    RecurrentStateSpec,
    RecurrentTensorSpec,
)
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.scheduler import Scheduler
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import LayerBatchSparseState, SparseController
from sparseengine.engine.sparse_methods.dynamic import OmniKVRuntime
from sparseengine.models.qwen3_5 import (
    Qwen35ForCausalLM,
    Qwen35LinearAttention,
    Qwen35LinearConv1D,
    Qwen35RMSNorm,
    _get_rotary_dim,
)
from sparseengine.models.qwen3_5_moe import (
    Qwen35MoePackedExperts,
    Qwen35MoeRouter,
    Qwen35MoeSparseMoeBlock,
)
from sparseengine.models.spec import resolve_model_spec
from sparseengine.platforms.cpu import CpuPlatform
from sparseengine.sampling_params import SamplingParams
from sparseengine.utils.loader import _target_weight_name_for_model, _validate_all_quantized_weights_loaded


def _single_process_parallel_context() -> ParallelContext:
    group = ParallelGroup(process_group=None, ranks=(0,), rank=0, size=1)
    return ParallelContext(world=group, moe_tp=group, attn_tp=group, moe_ep=group, attn_dp=group)


def test_qwen35_runtime_passes_h2o_score_contract_to_full_attention_builder():
    config = object()
    engine_config = SimpleNamespace(
        sparse_method="h2o",
        sparse_prefill_score_mode="logits",
        h2o_prefill_score_window=0,
        max_decoding_seqs=8,
        decode_graph=False,
    )
    context = SimpleNamespace(attn_tp_size=1)
    full_attention = object()
    gated_delta_rule = object()
    with (
        patch(
            "sparseengine.models.qwen3_5.build_mha_full_attention_provider",
            return_value=full_attention,
        ) as build_attention,
        patch(
            "sparseengine.models.qwen3_5.build_gated_delta_rule_op",
            return_value=gated_delta_rule,
        ),
    ):
        kwargs = Qwen35ForCausalLM.build_runtime_kwargs(
            config,
            engine_config=engine_config,
            parallel_context=context,
            device=torch.device("cpu"),
        )

    assert kwargs["full_attention_provider"] is full_attention
    build_attention.assert_called_once_with(
        config,
        sparse_method="h2o",
        attention_tp_size=1,
        device=torch.device("cpu"),
        max_batch_size=8,
        cuda_graph=False,
        runtime_config=engine_config,
    )


def test_qwen35_runtime_closes_attention_if_gdn_prepare_fails():
    config = object()
    engine_config = SimpleNamespace(
        sparse_method="vanilla",
        max_decoding_seqs=8,
        decode_graph=False,
    )
    context = SimpleNamespace(attn_tp_size=1)
    full_attention = Mock(name="full_attention")
    with (
        patch(
            "sparseengine.models.qwen3_5.build_mha_full_attention_provider",
            return_value=full_attention,
        ),
        patch(
            "sparseengine.models.qwen3_5.build_gated_delta_rule_op",
            side_effect=RuntimeError("GDN prepare failed"),
        ),
        pytest.raises(RuntimeError, match="GDN prepare failed"),
    ):
        Qwen35ForCausalLM.build_runtime_kwargs(
            config,
            engine_config=engine_config,
            parallel_context=context,
            device=torch.device("cpu"),
        )

    full_attention.close.assert_called_once_with()


def test_qwen35_fp8_expert_validation_uses_per_projection_weights():
    loaded_shards = {
        (expert_id, projection)
        for expert_id in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    experts = SimpleNamespace(
        fp8_enabled=True,
        local_expert_start=0,
        local_expert_end=2,
        _loaded_packed_projections=set(),
        _loaded_expert_shards=loaded_shards,
    )

    Qwen35MoePackedExperts.validate_loaded_weights(experts)

    experts._loaded_expert_shards.remove((1, "down_proj"))
    with pytest.raises(ValueError, match="Missing local Qwen3.6 expert weights"):
        Qwen35MoePackedExperts.validate_loaded_weights(experts)


def _qwen35_outer_config(*, num_layers: int = 64, full_layers: tuple[int, ...] | None = None):
    if full_layers is None:
        full_layers = tuple(range(0, num_layers, 4))
    full = set(full_layers)
    layer_types = ["full_attention" if idx in full else "linear_attention" for idx in range(num_layers)]
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        num_hidden_layers=num_layers,
        num_kv_layers=len(full_layers),
        layer_types=layer_types,
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        rms_norm_eps=1.0e-6,
        hidden_act="silu",
        attention_bias=False,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        max_position_embeddings=131072,
        dtype=torch.float16,
        tie_word_embeddings=False,
        quantization_config={
            "quant_method": "fp8",
            "weight_dtype": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )
    return SimpleNamespace(model_type="qwen3_5", text_config=text_config)


def _make_config(tmp_path, **kwargs):
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=_qwen35_outer_config()):
        return Config(model=str(tmp_path), **kwargs)


def test_linear_attention_fuses_qkvz_and_ba_projections():
    config = _qwen35_outer_config(num_layers=1, full_layers=()).text_config
    config.quantization_config = None
    config.runtime_recurrent_state_dtype = torch.float32
    parallel_context = _single_process_parallel_context()
    with (
        patch(
            "sparseengine.models.qwen3_5.get_parallel_context",
            return_value=parallel_context,
        ),
        patch(
            "sparseengine.layers.linear.get_parallel_context",
            return_value=parallel_context,
        ),
    ):
        attention = Qwen35LinearAttention(config)

    torch.manual_seed(0)
    attention.in_proj_qkvz.weight.data.normal_()
    attention.in_proj_ba.weight.data.normal_()
    hidden_states = torch.randn(3, config.hidden_size)
    qkvz = torch.nn.functional.linear(
        hidden_states,
        attention.in_proj_qkvz.weight,
    )
    q, k, v, z = qkvz.split(
        [
            attention.tp_key_dim,
            attention.tp_key_dim,
            attention.tp_value_dim,
            attention.tp_value_dim,
        ],
        dim=-1,
    )
    expected_b, expected_a = torch.nn.functional.linear(
        hidden_states,
        attention.in_proj_ba.weight,
    ).chunk(2, dim=-1)

    mixed_qkv, actual_z, actual_b, actual_a = attention._project_qkvzba(
        hidden_states
    )

    torch.testing.assert_close(mixed_qkv, torch.cat((q, k, v), dim=-1))
    torch.testing.assert_close(
        actual_z,
        z.view(-1, attention.num_v_heads, attention.head_v_dim),
    )
    torch.testing.assert_close(actual_b, expected_b)
    torch.testing.assert_close(actual_a, expected_a)
    assert Qwen35ForCausalLM.packed_modules_mapping["in_proj_z"] == (
        "in_proj_qkvz",
        3,
    )


def test_qwen35_moe_fuses_router_and_shared_gate_projection():
    router = Qwen35MoeRouter.__new__(Qwen35MoeRouter)
    torch.nn.Module.__init__(router)
    router.num_experts, router.hidden_size = 4, 3
    router.weight = torch.nn.Parameter(torch.empty(5, 3))
    router.op_spec = object()
    topk_weights, topk_ids = torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32)
    router.provider = Mock()
    router.provider.run.return_value = (topk_weights, topk_ids)
    expert_weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    shared_weight = torch.tensor([[2.0, 3.0, 5.0]])
    router.weight_loader(router.weight, expert_weight)
    router.weight_loader(router.weight, shared_weight, "shared")
    hidden_states = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    actual_weights, actual_ids, shared_logits = router(hidden_states)

    assert actual_weights is topk_weights
    assert actual_ids is topk_ids
    torch.testing.assert_close(
        shared_logits,
        torch.nn.functional.linear(hidden_states, shared_weight),
    )
    router.provider.run.assert_called_once()
    assert Qwen35ForCausalLM.packed_modules_mapping["shared_expert_gate"] == (
        "gate",
        "shared",
    )


def test_qwen35_moe_reduces_routed_and_shared_outputs_together():
    class ReturnValue(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, *_):
            return self.value

    hidden_states = torch.randn(2, 4)
    local_output, shared_output = torch.randn_like(hidden_states), torch.randn_like(hidden_states)
    gate_logits = torch.randn(2, 1)
    block = Qwen35MoeSparseMoeBlock.__new__(Qwen35MoeSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.shared_expert = ReturnValue(shared_output)
    block.gate = ReturnValue((torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32), gate_logits))
    block.experts = ReturnValue(local_output)
    block.parallel_context = Mock()
    block.parallel_context.world.size = 2
    block.parallel_context.world.ranks = (0, 1)
    block.parallel_context.attn_tp.ranks = (0, 1)
    block.parallel_context.world.all_reduce.side_effect = lambda outputs: outputs + 1

    with patch(
        "sparseengine.models.qwen3_5_moe.gated_shared_add",
        side_effect=lambda routed, shared, gate: routed + shared * gate.sigmoid(),
    ):
        from sparseengine.distributed.moe_communication import AllReduceMoeCommunication
        block.mlp_chunk_size = 16
        block.experts.fp8_enabled = False
        block.moe_communication = AllReduceMoeCommunication(block.parallel_context.world.all_reduce)
        block.moe_execution = block._create_moe_execution()
        actual = block._forward_chunk(hidden_states)

    packed = block.parallel_context.world.all_reduce.call_args.args[0]
    torch.testing.assert_close(packed, torch.stack((local_output, shared_output)))
    torch.testing.assert_close(
        actual,
        local_output + 1 + (shared_output + 1) * gate_logits.sigmoid(),
    )




def test_qwen35_moe_skips_single_rank_output_packing():
    hidden_states = torch.randn(2, 4)
    block = Qwen35MoeSparseMoeBlock.__new__(Qwen35MoeSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.shared_expert = Mock(return_value=hidden_states + 1)
    block.gate = Mock(
        return_value=(
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int32),
            torch.zeros(2, 1),
        )
    )
    block.experts = Mock(return_value=hidden_states + 2)
    block.parallel_context = Mock()
    block.parallel_context.world.size = 1

    with patch(
        "sparseengine.models.qwen3_5_moe.gated_shared_add",
        side_effect=lambda routed, shared, gate: routed + shared * gate.sigmoid(),
    ):
        from sparseengine.distributed.moe_communication import AllReduceMoeCommunication
        block.mlp_chunk_size = 16
        block.experts.fp8_enabled = False
        block.moe_communication = AllReduceMoeCommunication(block.parallel_context.world.all_reduce)
        block.moe_execution = block._create_moe_execution()
        actual = block._forward_chunk(hidden_states)

    block.parallel_context.world.all_reduce.assert_not_called()
    torch.testing.assert_close(
        actual, hidden_states + 2 + 0.5 * (hidden_states + 1)
    )


class _ResidentAdmissionCache:
    def decode_window_budgets(self):
        return {"slots": self.num_free_slots}

    def decode_window_costs(self, seq, tokens):
        return {"slots": tokens}

    def prefill_capacity_after_decode_reservations(self, free_slots, reserved, *, admission):
        return free_slots - reserved.get("slots", 0)

    def prefill_private_slots_for(self, seq):
        return 0

    def __init__(self):
        self.num_free_slots = 1_000_000
        self.freed_seq_ids = []
        self.prefix_blocks = 0

    def prefill_batched_tokens_margin(self):
        return 0

    def remaining_prefill_tokens(self, seq):
        return int(seq.num_prompt_tokens - max(seq.num_prefilled_tokens, seq.prefix_cache_hit_len))

    def prefill_execution_mode(self, seq):
        del seq
        return "chunked"

    def prefill_batch_compatibility_key(self, seq):
        del seq
        return None

    def reset_prefill_execution_state(self, seq_id):
        del seq_id

    def complete_prefill_execution(self, seq):
        del seq

    def reserved_prefill_slots(self, waiting, engine_prefill_chunk_size):
        return 0

    def should_schedule_full_prefill(self, seq):
        return False

    def requires_full_prefill_step(self, seq):
        return False

    def requires_long_prefill_offload(self, seq):
        return False

    def prefill_step_free_slots(self):
        return self.num_free_slots

    def prefill_step_free_slots_for(self, seq):
        return self.num_free_slots

    def min_final_prefill_chunk_size(self, seq):
        return 0

    def prefill_step_reservation_cost(self, seq, scheduled_tokens):
        return int(scheduled_tokens)

    def decode_step_free_slots(self):
        return self.num_free_slots

    def decode_step_free_slots_for(self, seq):
        return self.num_free_slots

    def decode_step_reservation_cost(self, seq):
        return 1

    def prompt_admission_free_slots(self):
        return self.num_free_slots

    def prompt_admission_budgets(self, waiting, engine_prefill_chunk_size):
        return {"slots": self.num_free_slots}

    def prompt_admission_costs(self, seq):
        return {"slots": int(seq.num_prompt_tokens - seq.prefix_cache_hit_len)}

    def prompt_admission_cost(self, seq):
        return int(seq.num_prompt_tokens - seq.prefix_cache_hit_len)

    def prompt_logical_reservation_cost(self, seq):
        return int(seq.num_prompt_tokens - seq.prefix_cache_hit_len)

    def prompt_admission_failure_action(self):
        return "defer"

    def on_prompt_admitted(self, seq, costs):
        return None

    def refresh_prefix_cache_hit(self, seq):
        return None

    def clear_prefix_cache_hit(self, seq):
        seq.clear_prefix_cache_hit()

    def free_seq(self, seq_id):
        self.freed_seq_ids.append(int(seq_id))

    def reset_after_warmup(self):
        return None

    def free_slot_stats(self):
        return {"free_slots": self.num_free_slots}

    def debug_live_seq_slots(self):
        return {}


def _resident_scheduler_config():
    return SimpleNamespace(
        max_num_seqs_in_batch=2,
        max_num_batched_tokens=4,
        max_decoding_seqs=2,
        max_num_seqs_in_gpu=2,
        engine_prefill_chunk_size=2,
        prefill_schedule_policy="all_chunked",
        eos=-1,
        eos_token_ids=(),
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=4,
        sparse_method="",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "max_num_seqs_in_batch": 8,
            "max_decoding_seqs": 8,
            "max_num_seqs_in_gpu": 7,
        },
        {
            "max_num_seqs_in_batch": 8,
            "max_decoding_seqs": 10,
            "max_num_seqs_in_gpu": 9,
        },
    ],
)
def test_config_rejects_resident_capacity_below_step_or_decode_limit(tmp_path, kwargs):
    with pytest.raises(ValueError, match="max_num_seqs_in_gpu must be >="):
        _make_config(tmp_path, **kwargs)


def test_resident_admission_waits_at_capacity_and_partial_prefill_keeps_progressing():
    config = _resident_scheduler_config()
    cache = _ResidentAdmissionCache()
    runtime_state = RuntimeState(config, cache)
    scheduler = Scheduler(config, runtime_state)
    seqs = [Sequence([1, 2, 3, 4], SamplingParams(max_tokens=4)) for _ in range(3)]
    for seq in seqs:
        scheduler.add(seq)

    first_batch, is_prefill, _ = scheduler.schedule()
    assert is_prefill
    assert {seq.seq_id for seq in first_batch} == {seqs[0].seq_id, seqs[1].seq_id}
    assert runtime_state.free_slot_stats()["resident_sequences"] == 2
    scheduler.postprocess(first_batch, [10, 11], is_prefill=True)

    second_batch, is_prefill, _ = scheduler.schedule()
    assert is_prefill
    assert {seq.seq_id for seq in second_batch} == {seqs[0].seq_id, seqs[1].seq_id}
    assert seqs[2] in scheduler.waiting
    assert runtime_state.free_slot_stats()["resident_sequences"] == 2
    assert runtime_state.prompt_admission_costs(seqs[0])["resident_seqs"] == 0
    scheduler.postprocess(second_batch, [12, 13], is_prefill=True)

    decode_batch, is_prefill, _ = scheduler.schedule()
    assert not is_prefill
    assert {seq.seq_id for seq in decode_batch} == {seqs[0].seq_id, seqs[1].seq_id}
    assert seqs[2] in scheduler.waiting

    scheduler.decoding.remove(seqs[0])
    runtime_state.free_seq(seqs[0].seq_id)
    admitted_batch, is_prefill, _ = scheduler.schedule()
    assert is_prefill
    assert [seq.seq_id for seq in admitted_batch] == [seqs[2].seq_id]
    assert runtime_state.free_slot_stats()["resident_sequences"] == 2


def test_resident_ownership_releases_on_abort_free_and_warmup_reset():
    config = _resident_scheduler_config()
    cache = _ResidentAdmissionCache()
    runtime_state = RuntimeState(config, cache)
    scheduler = Scheduler(config, runtime_state)
    seq = Sequence([1, 2], SamplingParams(max_tokens=2))
    runtime_state.on_prompt_admitted(seq, {"slots": 2, "resident_seqs": 1})
    seq.num_prefilled_tokens = 1
    scheduler.waiting.append(seq)

    assert scheduler.abort(seq.seq_id)
    runtime_state.free_seq(seq.seq_id)
    assert runtime_state.free_slot_stats()["resident_sequences"] == 0
    assert cache.freed_seq_ids == [seq.seq_id]

    runtime_state.on_prompt_admitted(seq, {"slots": 2, "resident_seqs": 1})
    runtime_state.reset_after_warmup()
    assert runtime_state.free_slot_stats()["resident_sequences"] == 0


def test_prefix_history_does_not_consume_resident_slots_but_hit_request_does():
    config = _resident_scheduler_config()
    cache = _ResidentAdmissionCache()
    cache.prefix_blocks = 100
    runtime_state = RuntimeState(config, cache)
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=2))
    seq.prefix_cache_hit_len = 2

    assert runtime_state.prompt_admission_budgets(deque(), 2)["resident_seqs"] == 2
    costs = runtime_state.prompt_admission_costs(seq)
    assert costs["slots"] == 2
    assert costs["resident_seqs"] == 1
    runtime_state.on_prompt_admitted(seq, costs)
    assert runtime_state.prompt_admission_budgets(deque(), 2)["resident_seqs"] == 1


@pytest.mark.parametrize(
    ("world_size", "conv_shape", "recurrent_shape", "snapshot_bytes"),
    [
        (1, (10_240, 3), (48, 128, 128), 78_446_592),
        (2, (5_120, 3), (24, 128, 128), 39_223_296),
    ],
)
def test_qwen36_recurrent_spec_has_exact_tp_local_bytes(
    world_size,
    conv_shape,
    recurrent_shape,
    snapshot_bytes,
):
    config = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        dtype=torch.bfloat16,
    )

    spec = Qwen35ForCausalLM.recurrent_state_spec(config, world_size)

    assert spec.tensor_specs == (
        RecurrentTensorSpec("conv_state", conv_shape, torch.bfloat16),
        RecurrentTensorSpec("recurrent_state", recurrent_shape, torch.bfloat16),
    )
    assert spec.bytes_per_layer == snapshot_bytes // 48
    assert spec.bytes_for_layers(48) == snapshot_bytes


def test_qwen36_recurrent_pool_rejects_explicit_cap_below_required_bytes():
    state_spec = Qwen35ForCausalLM.recurrent_state_spec(
        SimpleNamespace(
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            dtype=torch.bfloat16,
        ),
        world_size=1,
    )
    config = SimpleNamespace(
        enable_prefix_caching=False,
        recurrent_state_max_bytes=1 << 30,
        max_num_seqs_in_batch=16,
        max_decoding_seqs=16,
        max_num_seqs_in_gpu=16,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "required_bytes=1333592064.*limit_bytes=1073741824.*"
            "rows=17.*active_rows=16.*bytes_per_row=78446592"
        ),
    ):
        RecurrentStateManager.resolve_capacity(config, state_spec, num_recurrent_layers=48)


def test_qwen36_recurrent_capacity_counts_rows_and_scratch_exactly():
    state_spec = Qwen35ForCausalLM.recurrent_state_spec(
        SimpleNamespace(
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            dtype=torch.bfloat16,
        ),
        world_size=1,
    )
    prefix_off = SimpleNamespace(
        enable_prefix_caching=False,
        recurrent_state_max_bytes=1 << 30,
        max_num_seqs_in_batch=4,
        max_decoding_seqs=4,
        max_num_seqs_in_gpu=12,
    )
    prefix_on = SimpleNamespace(
        enable_prefix_caching=True,
        recurrent_state_max_bytes=1 << 30,
        max_num_seqs_in_batch=4,
        max_decoding_seqs=4,
        max_num_seqs_in_gpu=12,
    )

    off_capacity = RecurrentStateManager.resolve_capacity(
        prefix_off,
        state_spec,
        num_recurrent_layers=48,
    )
    on_capacity = RecurrentStateManager.resolve_capacity(
        prefix_on,
        state_spec,
        num_recurrent_layers=48,
    )

    assert off_capacity.row_capacity == 12
    assert off_capacity.total_rows == 13
    assert off_capacity.pool_bytes == 1_019_805_696
    assert on_capacity.row_capacity == 12
    assert on_capacity.total_rows == 13
    assert on_capacity.pool_bytes == 1_019_805_696


def _deprecation_messages(mock_log_once):
    return [
        call.args[0]
        for call in mock_log_once.call_args_list
        if call.args and "prefix_cache_max_recurrent_bytes is deprecated" in call.args[0]
    ]


def test_recurrent_budget_accepts_deprecated_prefix_cache_alias(tmp_path):
    with patch("sparseengine.configs.prefix_cache.log_once") as mock_log_once:
        config = _make_config(tmp_path, prefix_cache_max_recurrent_bytes=2 << 30)

    assert config.recurrent_state_max_bytes == 2 << 30
    assert config.prefix_cache_max_recurrent_bytes == 2 << 30
    assert len(_deprecation_messages(mock_log_once)) == 1


def test_recurrent_budget_rejects_conflicting_new_and_deprecated_names(tmp_path):
    with pytest.raises(ValueError, match="conflicting recurrent state budgets"):
        _make_config(
            tmp_path,
            recurrent_state_max_bytes=2 << 30,
            prefix_cache_max_recurrent_bytes=3 << 30,
        )


def test_deprecated_recurrent_budget_caps_prefix_on_live_pool(tmp_path):
    config = _make_config(
        tmp_path,
        enable_prefix_caching=True,
        prefix_cache_max_recurrent_bytes=1,
    )
    state_spec = Qwen35ForCausalLM.recurrent_state_spec(config.hf_config, world_size=1)

    assert config.recurrent_state_max_bytes == 1
    with pytest.raises(RuntimeError, match="required_bytes=.*limit_bytes=1"):
        RecurrentStateManager.resolve_capacity(
            config,
            state_spec,
            num_recurrent_layers=48,
        )


@pytest.mark.parametrize("manager_cls", [StandardCacheManager, QuestCacheManager])
@pytest.mark.parametrize("max_num_seqs_in_batch", [2, 8])
def test_mixed_cache_manager_rows_do_not_exceed_recurrent_pool(
    manager_cls,
    max_num_seqs_in_batch,
):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=64,
            num_key_value_heads=4,
            num_attention_heads=24,
            hidden_size=5120,
            head_dim=256,
        ),
        runtime_layout=RuntimeLayout.from_config(
            _qwen35_outer_config().text_config,
            require_mixed=True,
        ),
        max_model_len=121_000,
        max_num_seqs_in_batch=max_num_seqs_in_batch,
        max_decoding_seqs=12,
        max_num_seqs_in_gpu=12,
        recurrent_state_row_capacity=12,
    )
    manager = object.__new__(manager_cls)

    with patch.object(platforms, "_current_platform", CpuPlatform()):
        CacheManager.__init__(manager, config, _single_process_parallel_context())

    assert manager.max_buffer_rows == 12


def test_dense_cache_manager_keeps_redundant_row_capacity():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            num_attention_heads=1,
            hidden_size=32,
            head_dim=32,
        ),
        runtime_layout=RuntimeLayout.dense(2),
        max_model_len=128,
        max_num_seqs_in_batch=8,
        max_decoding_seqs=12,
        max_num_seqs_in_gpu=24,
    )
    manager = object.__new__(StandardCacheManager)

    with patch.object(platforms, "_current_platform", CpuPlatform()):
        CacheManager.__init__(manager, config, _single_process_parallel_context())

    assert manager.max_buffer_rows == 24


def test_qwen36_prefix_on_matches_kv_and_recurrent_block_capacity():
    kv_bytes_per_block = 268_435_456
    recurrent_bytes_per_block = 78_446_592
    available_bytes = 5 * (kv_bytes_per_block + recurrent_bytes_per_block) + 123

    capacity = resolve_joint_prefix_capacity(
        available_bytes=available_bytes,
        kv_bytes_per_block=kv_bytes_per_block,
        recurrent_bytes_per_block=recurrent_bytes_per_block,
        requested_max_blocks=None,
    )

    assert capacity.block_capacity == 5
    assert capacity.kv_allocatable_bytes == 5 * kv_bytes_per_block
    assert capacity.recurrent_capacity_bytes == 5 * recurrent_bytes_per_block
    assert capacity.unallocated_bytes == 123
    assert (
        available_bytes - capacity.kv_allocatable_bytes
        == capacity.recurrent_capacity_bytes + capacity.unallocated_bytes
    )


def test_joint_prefix_capacity_applies_explicit_block_cap_to_both_payloads():
    capacity = resolve_joint_prefix_capacity(
        available_bytes=10_000,
        kv_bytes_per_block=600,
        recurrent_bytes_per_block=400,
        requested_max_blocks=3,
    )

    assert capacity.block_capacity == 3
    assert capacity.kv_allocatable_bytes == 1_800
    assert capacity.recurrent_capacity_bytes == 1_200
    assert capacity.unallocated_bytes == 7_000


def _budget_test_manager(
    *,
    free_bytes,
    peak_bytes,
    current_bytes,
    prefix_on=False,
    prefix_mode="radix",
    recurrent_pool_bytes=1_000,
    recurrent_peak_before_bytes=None,
    recurrent_peak_after_bytes=None,
):
    manager = object.__new__(StandardCacheManager)
    manager.config = SimpleNamespace(
        hf_config=SimpleNamespace(
            hidden_size=10,
            intermediate_size=10,
            dtype=torch.float16,
        ),
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=1,
        engine_prefill_chunk_size=1,
        long_prefill_offload_threshold=1,
        prefill_schedule_policy="long_bs1full_short_batch",
        enable_prefix_caching=prefix_on,
        resolved_prefix_cache_mode=prefix_mode if prefix_on else "disabled",
        prefix_cache_block_size=4,
        prefix_cache_max_blocks=None,
        recurrent_state_pool_bytes=recurrent_pool_bytes,
        recurrent_state_allocator_peak_before_bytes=(
            peak_bytes
            if recurrent_peak_before_bytes is None
            else recurrent_peak_before_bytes
        ),
        recurrent_state_allocator_peak_after_bytes=(
            peak_bytes
            if recurrent_peak_after_bytes is None
            else recurrent_peak_after_bytes
        ),
        prefix_recurrent_bytes_per_block=400 if prefix_on else 0,
    )
    manager.world_size = 1
    manager.tp_size = 1
    manager.device = torch.device("cpu")
    manager.num_kv_heads = 1
    manager.head_dim = 10
    manager.num_kv_layers = 2
    manager.platform = SimpleNamespace(
        get_available_memory=lambda _device_id: (free_bytes, 10_000_000),
        get_allocator_stats=lambda _device: SimpleNamespace(
            peak_allocated_bytes=peak_bytes,
            current_allocated_bytes=current_bytes,
        ),
    )
    return manager


def test_kv_available_bytes_decrease_by_exact_eager_recurrent_pool_bytes():
    baseline = _budget_test_manager(
        free_bytes=8_000_000,
        peak_bytes=1_000_000,
        current_bytes=1_000_000,
        recurrent_pool_bytes=0,
    )
    with_pool = _budget_test_manager(
        free_bytes=7_999_000,
        peak_bytes=1_001_000,
        current_bytes=1_001_000,
        recurrent_peak_before_bytes=1_000_000,
        recurrent_peak_after_bytes=1_001_000,
    )

    baseline_available, _ = baseline._get_available_slots_info()
    pooled_available, _ = with_pool._get_available_slots_info()

    assert baseline_available - pooled_available == 1_000


def test_kv_available_bytes_deduct_pool_when_allocator_peak_does_not_rise():
    baseline = _budget_test_manager(
        free_bytes=8_000_000,
        peak_bytes=2_000_000,
        current_bytes=1_000_000,
        recurrent_pool_bytes=0,
    )
    with_pool = _budget_test_manager(
        free_bytes=7_999_000,
        peak_bytes=2_000_000,
        current_bytes=1_001_000,
        recurrent_peak_before_bytes=2_000_000,
        recurrent_peak_after_bytes=2_000_000,
    )

    baseline_available, _ = baseline._get_available_slots_info()
    pooled_available, _ = with_pool._get_available_slots_info()

    assert baseline_available - pooled_available == 1_000


def test_model_runner_resets_inherited_allocator_peak_before_model_construction():
    events = []

    class LifecyclePlatform(CpuPlatform):
        inherited_peak = 62_000_000_000
        current = 0

        def validate_inference(self):
            return None

        def init_backend(self):
            return None

        def get_distributed_backend(self):
            return "gloo"

        def reset_peak_memory_stats(self, device):
            events.append(("reset_peak", device))
            self.inherited_peak = self.current

    platform = LifecyclePlatform()
    observed_peak_at_construction = []

    def stop_at_model_construction(_config):
        observed_peak_at_construction.append(platform.inherited_peak)
        raise RuntimeError("stop after model construction boundary")

    config = SimpleNamespace(
        enable_profiler=False,
        world_size=1,
        attn_dp_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=1,
        data_parallel_size=1,
        parallel_topology=ParallelTopology(1, 1, 1),
        mlp_chunk_size=16384,
        decode_graph=False,
        decode_graph_capture_sizes=None,
        max_decoding_seqs=64,
        max_num_seqs_in_batch=32,
        hf_config=SimpleNamespace(model_type="qwen2", dtype=torch.float32),
        model_spec=resolve_model_spec("qwen2"),
    )
    with (
        patch.object(platforms, "_current_platform", platform),
        patch("sparseengine.engine.model_runner.dist.is_initialized", return_value=True),
        patch(
            "sparseengine.engine.model_runner.init_parallel_context",
            return_value=_single_process_parallel_context(),
        ),
        patch("sparseengine.engine.model_runner.Qwen2ForCausalLM", side_effect=stop_at_model_construction),
        pytest.raises(RuntimeError, match="stop after model construction boundary"),
    ):
        ModelRunner(config, rank=0, event=[])

    assert events == [("reset_peak", torch.device("cpu"))]
    assert observed_peak_at_construction == [0]


def test_cache_budget_resolves_joint_prefix_capacity_before_kv_allocation():
    manager = _budget_test_manager(
        free_bytes=8_000_000,
        peak_bytes=1_000_000,
        current_bytes=1_000_000,
        prefix_on=True,
    )

    kv_allocatable_bytes, slot_bytes_per_layer = manager._get_available_slots_info()

    assert slot_bytes_per_layer == 40
    assert manager.config.prefix_kv_bytes_per_block == 320
    assert manager.config.prefix_kv_block_capacity == 9_720
    assert manager.config.prefix_cache_max_blocks == 9_720
    assert manager.config.prefix_recurrent_capacity_bytes == 3_888_000
    assert kv_allocatable_bytes == 3_110_400


def test_chain_cache_budget_does_not_reserve_radix_prefix_blocks():
    chain = _budget_test_manager(
        free_bytes=8_000_000,
        peak_bytes=1_000_000,
        current_bytes=1_000_000,
        prefix_on=True,
        prefix_mode="chain",
    )
    without_prefix = _budget_test_manager(
        free_bytes=8_000_000,
        peak_bytes=1_000_000,
        current_bytes=1_000_000,
        prefix_on=False,
    )

    chain_available, _ = chain._get_available_slots_info()
    without_prefix_available, _ = without_prefix._get_available_slots_info()

    assert chain_available == without_prefix_available
    assert not hasattr(chain.config, "prefix_kv_block_capacity")


def test_quest_joint_prefix_kv_bytes_include_page_metadata():
    manager = object.__new__(QuestCacheManager)
    manager.config = SimpleNamespace(prefix_cache_block_size=4096)
    manager.page_size = 16
    manager.num_kv_layers = 16
    manager.num_kv_heads = 4
    manager.head_dim = 256
    manager.hf_config = SimpleNamespace(dtype=torch.bfloat16)
    payload = QuestPrefixBlockPayload(
        block_slot=None,
        token_slots=torch.arange(4096, dtype=torch.int32),
        block_start=0,
        block_end=4096,
        block_slots=torch.arange(256, dtype=torch.int32),
    )

    reserved_bytes = manager._kv_allocation_bytes_per_prefix_block(4096)

    assert reserved_bytes == 285_212_672
    assert manager.prefix_kv_payload_nbytes(payload) == reserved_bytes


def test_runtime_layout_maps_qwen35_full_layers_to_compact_kv_indices():
    layout = RuntimeLayout.from_config(_qwen35_outer_config().text_config, require_mixed=True)

    assert layout.num_layers == 64
    assert layout.num_kv_layers == 16
    assert layout.full_attention_layer_indices[:4] == (0, 4, 8, 12)
    assert layout.linear_attention_layer_indices[:3] == (1, 2, 3)
    assert layout.kv_layer_index(0) == 0
    assert layout.kv_layer_index(4) == 1
    with pytest.raises(RuntimeError, match="linear_attention"):
        layout.kv_layer_index(1)


def test_omnikv_observation_layers_follow_compact_qwen_kv_order(tmp_path):
    outer_config = _qwen35_outer_config(full_layers=tuple(range(3, 64, 4)))
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(
            model=str(tmp_path),
            sparse_method="omnikv",
            full_attention_layers="3,11,23,31,35,47,59",
        )

    assert cfg.obs_layer_ids == [3, 11, 23, 35, 47, 59]


def test_qwen35_pyramidkv_ratios_follow_compact_kv_order(tmp_path):
    outer_config = _qwen35_outer_config(full_layers=tuple(range(3, 64, 4)))
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(
            model=str(tmp_path),
            sparse_method="pyramidkv",
            pyramidkv_start_ratio=0.5,
            pyramidkv_least_ratio=0.1,
        )

    assert len(cfg.pyramid_layer_ratios) == cfg.runtime_layout.num_kv_layers == 16
    assert cfg.pyramid_layer_ratios[0] == pytest.approx(0.5)
    assert cfg.pyramid_layer_ratios[-1] == pytest.approx(0.1)

    controller = SparseController(cfg, SimpleNamespace())
    first_kv_layer = cfg.runtime_layout.kv_idx_to_layer_idx[0]
    last_kv_layer = cfg.runtime_layout.kv_idx_to_layer_idx[-1]
    assert (
        controller.runtime._get_layer_budget(first_kv_layer, is_prefill=True)
        == 4672
    )
    assert (
        controller.runtime._get_layer_budget(last_kv_layer, is_prefill=True)
        == 1395
    )


def test_qwen35_pyramidkv_allocates_slots_only_for_kv_layers(tmp_path):
    outer_config = _qwen35_outer_config(num_layers=8, full_layers=(3, 7))
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(
            model=str(tmp_path),
            sparse_method="pyramidkv",
            pyramidkv_start_ratio=0.5,
            pyramidkv_least_ratio=0.1,
        )

    manager = object.__new__(SnapKVCacheManager)
    manager.config = cfg
    manager.hf_config = cfg.hf_config
    manager.runtime_layout = cfg.runtime_layout
    manager.num_layers = cfg.runtime_layout.num_layers
    manager.num_kv_layers = cfg.runtime_layout.num_kv_layers
    manager.num_kv_heads = 1
    manager.head_dim = 1
    manager.max_buffer_rows = 1
    manager.max_model_len = 8
    manager.device = torch.device("cpu")
    manager.pyramidkv_prefill_staging_num_slots = 0
    manager.pyramidkv_prefill_staging_kv_cache = None
    manager._get_available_slots_info = lambda: (1_000, 8)
    manager._pyramidkv_can_use_full_prefill_staging = lambda: False

    SnapKVCacheManager.allocate_kv_cache(manager)

    assert len(manager.kv_cache) == 2
    assert [cfg.num_kvcache_slots[idx] for idx in (0, 1, 2, 4, 5, 6)] == [0] * 6
    assert cfg.num_kvcache_slots[3] > cfg.num_kvcache_slots[7] > 0


def test_qwen35_snapkv_initializes_compact_kv_metadata(tmp_path):
    outer_config = _qwen35_outer_config(num_layers=8, full_layers=(3, 7))
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(
            model=str(tmp_path),
            sparse_method="snapkv",
            max_num_seqs_in_batch=2,
            max_decoding_seqs=2,
        )
    cfg.max_model_len = 8

    def allocate_small_cache(manager):
        manager.config.num_kvcache_slots = 16
        manager.kv_cache = torch.empty(0)

    with (
        patch.object(platforms, "_current_platform", CpuPlatform()),
        patch.object(SnapKVCacheManager, "allocate_kv_cache", allocate_small_cache),
    ):
        manager = SnapKVCacheManager(cfg, _single_process_parallel_context())

    assert manager.buffer_req_to_token_slots_tensor.shape == (2, 6, 8)
    assert manager.free_slots_stack_tensor.shape == (2, 16)
    assert (
        manager.buffer_req_to_token_slots[3].data_ptr()
        == manager.buffer_req_to_token_slots_tensor[0].data_ptr()
    )
    assert (
        manager.buffer_req_to_token_slots[7].data_ptr()
        == manager.buffer_req_to_token_slots_tensor[1].data_ptr()
    )
    assert manager.buffer_req_to_token_slots[0] is None
    assert manager.free_slots_stack[0] is None

    seq_a = Sequence([1, 2])
    seq_b = Sequence([3, 4])
    seq_a.seq_id = 10
    seq_b.seq_id = 11
    seq_a.current_chunk_size = 2
    seq_b.current_chunk_size = 2
    layers_slot_mapping = torch.full((8, 4), -1, dtype=torch.int32)

    assert manager._allocate_prefill_batch_same_size_all_layers(
        [seq_a, seq_b],
        layers_slot_mapping,
    )
    assert manager._num_free_slots[3] == manager._num_free_slots[7] == 12
    assert manager._num_free_slots[0] == 0
    assert manager.buffer_req_to_token_slots_tensor[:, :, :2].tolist() == [
        [[14, 15], [12, 13], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[14, 15], [12, 13], [0, 0], [0, 0], [0, 0], [0, 0]],
    ]
    assert layers_slot_mapping[3].tolist() == [14, 15, 12, 13]
    assert layers_slot_mapping[7].tolist() == [14, 15, 12, 13]
    assert layers_slot_mapping[0].tolist() == [-1, -1, -1, -1]

    selected_slots, next_lens, _, wrote_slot_output = manager._allocate_decode_batch_all_layers(
        [seq_a.seq_id, seq_b.seq_id]
    )
    assert not wrote_slot_output
    assert selected_slots[3].tolist() == [10, 11]
    assert selected_slots[7].tolist() == [10, 11]
    assert selected_slots[0].tolist() == [-1, -1]
    assert next_lens[3].tolist() == [3, 3]
    assert next_lens[7].tolist() == [3, 3]
    assert manager.buffer_req_to_token_slots_tensor[:, :2, 2].tolist() == [
        [10, 11],
        [10, 11],
    ]


def test_qwen35_pyramidkv_projects_legacy_transformer_layer_ratios(tmp_path):
    outer_config = _qwen35_outer_config(num_layers=8, full_layers=(3, 7))
    legacy_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(
            model=str(tmp_path),
            sparse_method="pyramidkv",
            pyramid_layer_ratios=legacy_ratios,
        )

    assert cfg.pyramid_layer_ratios == [legacy_ratios[3], legacy_ratios[7]]


def test_qwen35_prefix_block_rejects_unaligned_sizes(tmp_path):
    with pytest.raises(ValueError, match="4096\\*N"):
        _make_config(tmp_path, enable_prefix_caching=True, prefix_cache_block_size=2048)
    with pytest.raises(ValueError, match="4096\\*N"):
        _make_config(tmp_path, enable_prefix_caching=True, prefix_cache_block_size=4097)


def test_qwen35_deltakv_requires_compatible_checkpoint_even_when_missing_allowed(tmp_path):
    with pytest.raises(ValueError, match="DeltaKV for Qwen3.5 requires"):
        _make_config(
            tmp_path,
            sparse_method="deltakv",
            allow_missing_deltakv_path=True,
            deltakv_latent_quant_bits=0,
        )


def test_qwen35_raw_config_fallback_when_transformers_autoconfig_is_unknown(tmp_path):
    outer_config = _qwen35_outer_config()
    text_config = dict(vars(outer_config.text_config))
    text_config["dtype"] = "float16"
    with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "qwen3_5",
                "text_config": text_config,
            },
            f,
        )

    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", side_effect=ValueError("unknown model")):
        cfg = Config(model=str(tmp_path))

    assert cfg.outer_hf_config.model_type == "qwen3_5"
    assert cfg.hf_config.model_type == "qwen3_5_text"
    assert cfg.runtime_layout.num_kv_layers == 16


def test_qwen35_rejects_unquantized_fp16_checkpoint(tmp_path):
    outer_config = _qwen35_outer_config()
    outer_config.text_config.quantization_config = None

    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=outer_config):
        with pytest.raises(ValueError, match="requires BF16 weights"):
            Config(model=str(tmp_path))


def test_qwen35_rejects_unsupported_quantization(tmp_path):
    outer_config = _qwen35_outer_config()
    outer_config.text_config.quantization_config = {"quant_method": "awq"}

    with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=outer_config):
        with pytest.raises(NotImplementedError, match="quant_method='awq'"):
            Config(model=str(tmp_path))


def test_qwen35_linear_conv1d_matches_hf_biasless_checkpoint():
    with patch(
        "sparseengine.models.qwen3_5.get_parallel_context",
        return_value=_single_process_parallel_context(),
    ):
        conv = Qwen35LinearConv1D(conv_dim=16, kernel_size=4, qk_dim=4, v_dim=8)

    assert conv.bias is None
    assert "bias" not in dict(conv.named_parameters())


def test_qwen35_rotary_dim_uses_partial_rotary_factor():
    config = SimpleNamespace(rope_parameters={"partial_rotary_factor": 0.25})

    assert _get_rotary_dim(config, 256) == 64
    assert _get_rotary_dim(SimpleNamespace(partial_rotary_factor=0.5), 256) == 128
    assert _get_rotary_dim(SimpleNamespace(qk_rope_head_dim=96), 256) == 96


def _qwen35_rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    normalized = x_float * torch.rsqrt(
        x_float.square().mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * (1.0 + weight.float())).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen35_rmsnorm_uses_hf_offset_weight_semantics(dtype):
    pytest.importorskip("flashinfer")
    torch.manual_seed(43)
    norm = Qwen35RMSNorm(128, eps=1.0e-6).cuda().to(dtype)
    norm.weight.data.normal_(mean=0.0, std=0.2)
    x = torch.randn(7, 128, device="cuda", dtype=dtype)
    original = x.clone()

    actual = norm(x)

    torch.testing.assert_close(
        actual,
        _qwen35_rmsnorm_reference(x, norm.weight, norm.eps),
        rtol=1.0e-2,
        atol=3.0e-2,
    )
    assert torch.equal(x, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen35_fused_rmsnorm_uses_hf_offset_weight_semantics(dtype):
    pytest.importorskip("flashinfer")
    torch.manual_seed(47)
    norm = Qwen35RMSNorm(128, eps=1.0e-6).cuda().to(dtype)
    norm.weight.data.normal_(mean=0.0, std=0.2)
    x = torch.randn(7, 128, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    merged = x.float() + residual.float()
    expected_residual = merged.to(dtype)
    expected = _qwen35_rmsnorm_reference(
        merged,
        norm.weight,
        norm.eps,
    ).to(dtype)

    actual, actual_residual = norm(x, residual)

    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=3.0e-2)
    assert torch.equal(actual_residual, expected_residual)
    assert actual is x
    assert actual_residual is residual


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen35_rmsnorm_supports_strided_qkv_views(dtype):
    pytest.importorskip("flashinfer")
    torch.manual_seed(53)
    norm = Qwen35RMSNorm(128, eps=1.0e-6).cuda().to(dtype)
    projection = torch.randn(7, 14336, device="cuda", dtype=dtype)
    query = projection[:, : 48 * 128].view(7, 48, 128)

    actual = norm(query)

    torch.testing.assert_close(
        actual,
        _qwen35_rmsnorm_reference(query, norm.weight, norm.eps),
        rtol=1.0e-2,
        atol=3.0e-2,
    )


def test_qwen35_decode_state_padding_preserves_real_rows_for_static_batch():
    from sparseengine.models.qwen3_5 import Qwen35LinearAttention

    conv = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    recurrent = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)

    padded_conv, padded_recurrent = Qwen35LinearAttention._pad_decode_states_for_static_batch(
        conv,
        recurrent,
        token_batch=4,
        real_batch=2,
    )

    assert padded_conv.shape == (4, 3, 4)
    assert padded_recurrent.shape == (4, 3, 2, 2)
    assert torch.equal(padded_conv[:2], conv)
    assert torch.equal(padded_recurrent[:2], recurrent)
    assert torch.equal(padded_conv[2], conv[0])
    assert torch.equal(padded_recurrent[3], recurrent[0])


def test_linear_layer_kv_hook_fails_fast_without_allocating_cache():
    manager = object.__new__(StandardCacheManager)
    manager.runtime_layout = RuntimeLayout.from_config(_qwen35_outer_config().text_config, require_mixed=True)

    with pytest.raises(RuntimeError, match="linear_attention"):
        manager.get_layer_buffer_req_to_token_slots(1)


def test_snapkv_prefill_context_lens_stays_dense_for_mixed_layout():
    manager = object.__new__(SnapKVCacheManager)
    manager.device = torch.device("cpu")
    manager.num_layers = 8
    manager.runtime_layout = RuntimeLayout.from_config(
        _qwen35_outer_config(num_layers=8, full_layers=(3, 7)).text_config,
        require_mixed=True,
    )
    manager.layer_batch_states = [LayerBatchStates() for _ in range(manager.num_layers)]
    manager.seq_id_to_row = [{} for _ in range(manager.num_layers)]
    for layer_idx in manager.kv_transformer_layer_indices():
        manager.seq_id_to_row[layer_idx] = {10: 0, 11: 1}
    manager._clear_prefill_attention_scores = lambda seq_id: None
    manager._should_use_pyramidkv_long_prefill_offload_staging = lambda seqs: False
    manager._should_use_pyramidkv_full_prefill_staging = lambda seqs: False

    def fake_batched_alloc(seqs, layers_slot_mapping):
        del seqs
        layers_slot_mapping[3].copy_(torch.arange(layers_slot_mapping.shape[1], dtype=torch.int32))
        layers_slot_mapping[7].copy_(torch.arange(layers_slot_mapping.shape[1], dtype=torch.int32) + 100)
        return True

    manager._allocate_prefill_batch_same_size_all_layers = fake_batched_alloc
    seqs = [
        SimpleNamespace(seq_id=10, current_chunk_size=5, num_prefilled_tokens=0, token_ids=[1, 2, 3, 4, 5]),
        SimpleNamespace(seq_id=11, current_chunk_size=5, num_prefilled_tokens=0, token_ids=[6, 7, 8, 9, 10]),
    ]

    input_ids, positions, cu_seqlens_q = SnapKVCacheManager._prepare_prefill(manager, seqs)

    assert input_ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert positions.tolist() == list(range(5)) + list(range(5))
    assert cu_seqlens_q.tolist() == [0, 5, 10]
    assert manager.layer_batch_states[3].context_lens.tolist() == [5, 5]
    assert manager.layer_batch_states[7].context_lens.tolist() == [5, 5]
    assert manager.layer_batch_states[0].context_lens is None


def test_pyramidkv_full_prefill_staging_populates_context_lens():
    manager = object.__new__(SnapKVCacheManager)
    manager.device = torch.device("cpu")
    manager.num_layers = 1
    manager.layer_batch_states = [LayerBatchStates()]
    manager.seq_id_to_row = [{}]
    manager.row_seq_lens = [[0]]
    manager.pyramidkv_prefill_staging_num_slots = 4
    manager._pyramidkv_prefill_staging_seq_offsets = {}
    manager._pyramidkv_prefill_staging_materialized_layers = set()
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._clear_prefill_attention_scores = lambda seq_id: None
    manager._should_use_pyramidkv_long_prefill_offload_staging = lambda seqs: False
    manager._should_use_pyramidkv_full_prefill_staging = lambda seqs: True

    def fake_get_free_row(layer_idx, seq_id):
        manager.seq_id_to_row[layer_idx][seq_id] = 0
        return 0

    manager._get_free_row = fake_get_free_row
    seq = SimpleNamespace(
        seq_id=10,
        current_chunk_size=2,
        num_prefilled_tokens=0,
        token_ids=[1, 2],
    )

    input_ids, positions, cu_seqlens_q = SnapKVCacheManager._prepare_prefill(
        manager,
        [seq],
    )

    assert input_ids.tolist() == [1, 2]
    assert positions.tolist() == [0, 1]
    assert cu_seqlens_q.tolist() == [0, 2]
    assert manager.layer_batch_states[0].context_lens.tolist() == [2]
    assert manager.layer_batch_states[0].max_context_len == 2
    assert manager._pyramidkv_prefill_staging_context_lens.tolist() == [2]


def test_omnikv_uses_first_target_kv_layer_for_slot_table_in_mixed_layout():
    class FakeCacheManager:
        def __init__(self):
            self.requested_layers = []

        def get_layer_buffer_req_to_token_slots(self, layer_idx):
            self.requested_layers.append(int(layer_idx))
            return torch.arange(20, dtype=torch.int32).reshape(1, 20)

    runtime = object.__new__(OmniKVRuntime)
    runtime.sparse_method = "omnikv"
    runtime.debug_dynamic_selection = {}
    runtime.debug_dynamic_selection_detail = False
    runtime.dynamic_deltakv_topk_tiebreak = False
    runtime.device = torch.device("cpu")
    runtime.cache_manager = FakeCacheManager()
    runtime.num_sink = 0
    runtime.num_recent = 1
    runtime.decode_keep_tokens = 2
    from sparseengine.operators.omnikv_selection import (
        OmniKVSelectionSpec, prepare_omnikv_selection,
    )
    runtime._omnikv_selection_provider = prepare_omnikv_selection(
        OmniKVSelectionSpec(0, 2), device=runtime.device,
    )
    runtime.config = SimpleNamespace(decode_graph=False)
    runtime.layer_batch_sparse_states = {
        3: LayerBatchSparseState(),
        7: LayerBatchSparseState(),
    }
    obs_state = runtime.layer_batch_sparse_states[3]
    obs_state.attn_score = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    obs_state.context_lens = torch.tensor([10], dtype=torch.int32)
    obs_state.req_indices = torch.tensor([0], dtype=torch.int32)

    def fake_build(topk_indices, topk_lens, hist_lens, recent_chunk_lens, slot_table, req_indices, num_sink, max_s, *, context_lens):
        del topk_indices, topk_lens, hist_lens, recent_chunk_lens, slot_table, req_indices, num_sink
        keep = torch.zeros((1, max_s), dtype=torch.int32)
        slots = torch.zeros((1, max_s), dtype=torch.int32)
        lens = torch.tensor([max_s], dtype=torch.int32)
        return keep, slots, lens

    context = SimpleNamespace(is_prefill=False)
    with patch(
        "sparseengine.engine.sparse_methods.dynamic.build_omnikv_keep_and_slots",
        side_effect=fake_build,
    ):
        runtime._update_dynamic_indices(3, [7], context)

    assert runtime.cache_manager.requested_layers == [7]
    assert runtime.layer_batch_sparse_states[7].active_slots is not None


def test_standard_mixed_prefix_payload_preserves_block_range():
    manager = object.__new__(StandardCacheManager)
    manager.seq_id_to_row = {7: 0}
    manager.row_seq_lens = [8192]
    manager.buffer_req_to_token_slots = torch.arange(8192, dtype=torch.int32).reshape(1, 8192)
    seq = SimpleNamespace(seq_id=7)

    payload = manager.build_prefix_kv_payload(seq, 4096, 8192)

    assert payload.block_start == 4096
    assert payload.block_end == 8192
    assert payload.token_slots[0].item() == 4096


def test_standard_mixed_prefix_promotion_bulk_allocates_one_slot_tensor():
    manager = object.__new__(StandardCacheManager)
    allocations = []
    manager._take_prefix_device_slots = lambda count: (
        allocations.append(count) or torch.arange(count, dtype=torch.int32)
    )
    payloads = [
        StandardPrefixBlockPayload(token_slots=None, block_start=0, block_end=4),
        StandardPrefixBlockPayload(token_slots=None, block_start=4, block_end=8),
    ]

    manager.allocate_prefix_kv_payloads_device(payloads)

    assert allocations == [8]
    assert payloads[0].token_slots.tolist() == [0, 1, 2, 3]
    assert payloads[1].token_slots.tolist() == [4, 5, 6, 7]
    assert (
        payloads[0].token_slots.untyped_storage().data_ptr()
        == payloads[1].token_slots.untyped_storage().data_ptr()
    )


def test_quest_mixed_prefix_promotion_bulk_allocates_one_page_tensor():
    manager = object.__new__(QuestCacheManager)
    manager.page_size = 2
    manager.page_offsets_i32 = torch.arange(2, dtype=torch.int32)
    allocations = []
    manager._take_prefix_device_pages = lambda count: (
        allocations.append(count) or torch.tensor([3, 5, 7, 9], dtype=torch.int32)
    )
    payloads = [
        QuestPrefixBlockPayload(
            block_slot=None,
            token_slots=None,
            block_start=0,
            block_end=4,
        ),
        QuestPrefixBlockPayload(
            block_slot=None,
            token_slots=None,
            block_start=4,
            block_end=8,
        ),
    ]

    manager.allocate_prefix_kv_payloads_device(payloads)

    assert allocations == [4]
    assert payloads[0].block_slots.tolist() == [3, 5]
    assert payloads[0].token_slots.tolist() == [6, 7, 10, 11]
    assert payloads[1].block_slots.tolist() == [7, 9]
    assert payloads[1].token_slots.tolist() == [14, 15, 18, 19]


@pytest.mark.parametrize("manager_type", [StandardCacheManager, QuestCacheManager])
def test_mixed_prefix_attach_preflight_rejects_missing_row_capacity(manager_type):
    manager = object.__new__(manager_type)
    manager.seq_id_to_row = {}
    manager.row_seq_lens = []
    manager.free_rows = deque()

    with pytest.raises(RuntimeError, match="No free rows"):
        manager.validate_prefix_kv_attach(SimpleNamespace(seq_id=7))


class _BoundaryCacheManager:
    def prefill_step_free_slots_for(self, seq):
        del seq
        return 9999


class _BoundaryCoordinator:
    block_size = 4096

    def step_reclaimable_slots(self):
        return 0


def test_mixed_runtime_caps_prefill_chunks_at_recurrent_snapshot_boundary():
    runtime_state = RuntimeState(
        config=None,
        cache_manager=_BoundaryCacheManager(),
        prefix_cache_coordinator=_BoundaryCoordinator(),
    )
    seq = SimpleNamespace(num_prompt_tokens=5000, num_prefilled_tokens=0, prefix_cache_hit_len=0)

    assert runtime_state.prefill_step_free_slots_for(seq) == 4096

    seq.num_prefilled_tokens = 4096
    assert runtime_state.prefill_step_free_slots_for(seq) == 904


def test_mixed_admission_counts_inflight_d2h_before_pressure_prompt():
    prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"mixed-reclaimable")
    block_id = prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload="kv",
            recurrent_payload="recurrent",
            token_count=4,
            accounting_bytes=8,
            recurrent_bytes=4,
        ),
        token_ids=(1, 2, 3, 4),
    )
    prefix_cache.insert_block(block)
    prefix_cache.begin_d2h(block)
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.prefix_cache = prefix_cache
    coordinator.block_size = 4
    coordinator.offload_controller = object()
    cache_manager = _ResidentAdmissionCache()
    cache_manager.num_free_slots = 4
    runtime_state = RuntimeState(
        config=SimpleNamespace(max_num_seqs_in_gpu=4),
        cache_manager=cache_manager,
        prefix_cache_coordinator=coordinator,
    )

    assert coordinator.evictable_slots() == 0
    assert coordinator.step_reclaimable_slots() == 4
    assert coordinator.admission_reclaimable_slots() == 4
    assert runtime_state.prefill_step_free_slots() == 8
    assert runtime_state.decode_step_free_slots() == 8
    assert runtime_state.prompt_admission_free_slots() == 8
    assert runtime_state.prompt_admission_budgets(deque(), 4)["slots"] == 8
    hit_seq = SimpleNamespace(
        seq_id=7,
        num_prompt_tokens=5,
        prefix_cache_hit_len=4,
        prefix_cache_hit_block_count=1,
        prefix_cache_hit_last_block_id=block_id,
    )
    assert runtime_state.prompt_admission_costs(hit_seq)["slots"] == 5

    prefix_cache.acquire_block_ref(block)
    assert runtime_state.prompt_admission_free_slots() == 4


@pytest.fixture
def mixed_decode_capacity():
    # Real radix ownership and scheduler decisions; only tensor allocation and
    # forward computation are replaced with CPU resource accounting.
    released = []

    class Cache(_ResidentAdmissionCache):
        def free_prefix_kv_payload(self, payload):
            self.num_free_slots += payload
            released.append("kv")

        def prepare_step(self, seqs, is_prefill):
            assert not is_prefill
            assert self.num_free_slots >= len(seqs)
            self.num_free_slots -= len(seqs)

    cache = Cache()
    cache.num_free_slots = 0
    config = _resident_scheduler_config()
    config.decode_reservation_tokens = 2
    index = RadixPrefixIndex(block_size=4, fingerprint=b"decode-reclaimable")
    block = PrefixCacheBlock(
        stable_block_id=index.stable_block_id([1, 2, 3, 4], None),
        parent_block_id=None, block_size=4, logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload=4, recurrent_payload="state", token_count=4,
            accounting_bytes=8, recurrent_bytes=4,
        ),
        token_ids=(1, 2, 3, 4),
    )
    index.insert_block(block)
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.prefix_cache = index
    coordinator.block_size = 4
    coordinator.offload_controller = None
    coordinator.cache_manager = cache
    coordinator.recurrent_state_manager = SimpleNamespace(
        free_prefix_recurrent_payload=lambda payload: released.append("recurrent"),
    )
    runtime = RuntimeState(config, cache, prefix_cache_coordinator=coordinator)
    scheduler = Scheduler(config, runtime)
    for _ in range(2):
        seq = Sequence([8, 9], SamplingParams(max_tokens=8))
        seq.num_prefilled_tokens = 2
        seq.append_token(10)
        scheduler.decoding.append(seq)
    return cache, runtime, scheduler, block, released


def test_mixed_decode_reserves_reclaimable_prefix_once_and_evicts_before_forward(mixed_decode_capacity):
    cache, runtime, scheduler, block, released = mixed_decode_capacity
    assert runtime.reserve_decode_windows(scheduler.decoding, scheduler.waiting) is None
    seqs, is_prefill, preempted = scheduler.schedule()
    assert len(seqs) == 2 and not is_prefill and not preempted
    assert runtime.decode_reservations.outstanding() == {"slots": 4}
    assert runtime.prompt_admission_budgets(deque(), 2)["slots"] == 0
    assert runtime.prompt_admission_free_slots() == 0
    assert runtime.prefill_step_free_slots() == 0
    assert not released and cache.num_free_slots == 0
    extra = Sequence([8, 9], SamplingParams(max_tokens=8))
    extra.num_prefilled_tokens = 2
    extra.append_token(10)
    assert runtime.reserve_decode_windows([*seqs, extra], []) is extra
    assert extra.seq_id not in runtime.decode_reservations.requests
    runtime.prepare_step(seqs, is_prefill=False)
    assert released == ["kv", "recurrent"]
    assert block.stable_block_id not in runtime.prefix_cache_coordinator.prefix_cache.blocks
    scheduler.postprocess(seqs, [11, 11], is_prefill=False)
    assert cache.num_free_slots == 2
    assert runtime.decode_reservations.outstanding() == {"slots": 2}
    assert runtime.prompt_admission_budgets(deque(), 2)["slots"] == 0


@pytest.mark.parametrize("protection", ["referenced", "negative_priority"])
def test_mixed_decode_cannot_reserve_protected_prefix(mixed_decode_capacity, protection):
    _, runtime, scheduler, block, released = mixed_decode_capacity
    if protection == "referenced":
        runtime.prefix_cache_coordinator.prefix_cache.acquire_block_ref(block)
    else:
        block.eviction_priority = -1
    assert runtime.reserve_decode_windows(scheduler.decoding, scheduler.waiting) is scheduler.decoding[0]
    assert not runtime.decode_reservations.requests
    assert not released


@pytest.mark.parametrize("inflight_d2h", [False, True])
def test_mixed_sole_decode_shortens_window_to_reclaimable_capacity(mixed_decode_capacity, inflight_d2h):
    _, runtime, scheduler, block, _ = mixed_decode_capacity
    if inflight_d2h:
        coordinator = runtime.prefix_cache_coordinator
        coordinator.offload_controller = object()
        coordinator.prefix_cache.begin_d2h(block)
        assert coordinator.evictable_slots() == 0
    scheduler.decoding.pop()
    runtime.decode_reservations.window = 8
    seqs, is_prefill, preempted = scheduler.schedule()
    assert len(seqs) == 1 and not is_prefill and not preempted
    assert runtime.decode_reservations.outstanding() == {"slots": 4}


def test_mixed_decode_preserves_pending_prefill_when_physical_budget_is_clamped(mixed_decode_capacity):
    cache, runtime, scheduler, _, released = mixed_decode_capacity
    partial = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=8))
    partial.num_prefilled_tokens = 2
    cache.reserved_prefill_slots = lambda waiting, step: sum(
        seq.num_prompt_tokens - seq.num_prefilled_tokens for seq in waiting
    )
    # Free GPU slots are zero, so the cache's clamped admission budget is zero
    # with or without this partial prefill. It still owns two future slots.
    assert runtime.reserve_decode_windows(scheduler.decoding, [partial]) is scheduler.decoding[1]
    assert runtime.decode_reservations.outstanding() == {"slots": 2}
    assert not released


def test_mixed_cpu_only_hit_reclaims_other_device_block_before_promotion():
    prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"mixed-promotion-pressure")
    pressure_kv = SimpleNamespace(token_slots=torch.arange(4, dtype=torch.int32))
    pressure_recurrent = RecurrentPrefixPayload(
        token_count=4,
        layer_states={1: {"state": torch.ones(2)}},
    )
    pressure_id = prefix_cache.stable_block_id([9, 9, 9, 9], None)
    pressure_block = PrefixCacheBlock(
        stable_block_id=pressure_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload=pressure_kv,
            recurrent_payload=pressure_recurrent,
            token_count=4,
            accounting_bytes=24,
            recurrent_bytes=8,
            host_block_index=1,
        ),
        token_ids=(9, 9, 9, 9),
    )
    pressure_block.residency.host_present = True
    prefix_cache.insert_block(pressure_block)

    target_kv = SimpleNamespace(token_slots=None)
    target_recurrent = RecurrentPrefixPayload(token_count=4, layer_states={})
    target_id = prefix_cache.stable_block_id([1, 2, 3, 4], None)
    target_block = PrefixCacheBlock(
        stable_block_id=target_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload=target_kv,
            recurrent_payload=target_recurrent,
            token_count=4,
            accounting_bytes=24,
            recurrent_bytes=8,
            host_block_index=0,
        ),
        token_ids=(1, 2, 3, 4),
    )
    target_block.residency.device_present = False
    target_block.residency.host_present = True
    prefix_cache.insert_block(target_block)

    class FakeCacheManager:
        def __init__(self):
            self.num_free_slots = 0
            self.allocations = []
            self.attachments = []

        def free_prefix_kv_payload_device(self, payload):
            assert isinstance(payload.token_slots, torch.Tensor)
            payload.token_slots = None
            self.num_free_slots += 4

        def allocate_prefix_kv_payload_device(self, payload):
            assert self.num_free_slots >= 4
            payload.token_slots = torch.arange(4, 8, dtype=torch.int32)
            self.num_free_slots -= 4
            self.allocations.append(payload)

        def allocate_prefix_kv_payloads_device(self, payloads):
            for payload in payloads:
                self.allocate_prefix_kv_payload_device(payload)

        def attach_prefix_kv_payloads(self, seq, payloads):
            self.attachments.extend((seq.seq_id, payload) for payload in payloads)

        def validate_prefix_kv_attach(self, seq):
            assert seq.seq_id not in {attached_seq for attached_seq, _ in self.attachments}

    class FakeController:
        def __init__(self):
            self.operations = {}

        def poll(self):
            return (0, 0)

        def free_device_recurrent(self, block):
            block.payload.recurrent_payload.layer_states = {}

        def allocate_device_recurrent(self, blocks):
            for block in blocks:
                block.payload.recurrent_payload.layer_states = {
                    1: {"state": torch.empty(2)}
                }

        def submit_h2d(self, blocks):
            operation = SimpleNamespace(auxiliary_layer_events={1: object()})
            for block in blocks:
                prefix_cache.begin_h2d(block)
                self.operations[block.stable_block_id] = operation
            return operation

        def h2d_operation_for_block(self, block):
            return self.operations.get(block.stable_block_id)

    recurrent_attachments = []
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.enabled = True
    coordinator.block_size = 4
    coordinator.prefix_cache = prefix_cache
    coordinator.cache_manager = FakeCacheManager()
    coordinator.recurrent_state_manager = SimpleNamespace(
        attach_prefix_recurrent_payload=lambda seq, payload, readiness_events: (
            recurrent_attachments.append((seq.seq_id, payload, readiness_events))
        )
    )
    coordinator.offload_controller = FakeController()
    coordinator.seq_id_to_prefix_blocks = {}
    coordinator._step_h2d_operations = {}
    seq = SimpleNamespace(
        seq_id=7,
        prefix_cache_hit_len=4,
        prefix_cache_hit_block_count=1,
        prefix_cache_hit_last_block_id=target_id,
    )

    coordinator.attach_prefix_cache_hits([seq])

    assert pressure_block.residency.device_present is False
    assert pressure_block.residency.host_present is True
    assert pressure_kv.token_slots is None
    assert target_block.residency.transfer == PrefixTransferKind.H2D
    assert target_block.ref_count == 1
    assert coordinator.cache_manager.num_free_slots == 0
    assert coordinator.cache_manager.allocations == [target_kv]
    assert coordinator.cache_manager.attachments == [(seq.seq_id, target_kv)]
    assert recurrent_attachments[0][:2] == (seq.seq_id, target_recurrent)


@pytest.mark.parametrize("manager_type", [StandardCacheManager, QuestCacheManager])
@pytest.mark.parametrize("row_preexisted", [False, True])
def test_mixed_recurrent_attach_failure_rolls_back_kv_row_and_prefix_ref(
    manager_type,
    row_preexisted,
):
    manager = object.__new__(manager_type)
    manager.device = torch.device("cpu")
    manager.max_model_len = 8
    manager.seq_id_to_row = {7: 0} if row_preexisted else {}
    manager.free_rows = deque([1] if row_preexisted else [0, 1])
    manager.row_seq_lens = [0, 0]
    manager.buffer_req_to_token_slots = torch.zeros((2, 8), dtype=torch.int32)

    if manager_type is StandardCacheManager:
        manager.config = SimpleNamespace(prefix_cache_block_size=4)
        manager.seq_id_to_cached_ranges = {}
        manager._num_free_slots = 4
        payload = StandardPrefixBlockPayload(
            token_slots=torch.tensor([4, 5, 6, 7], dtype=torch.int32),
            block_start=0,
            block_end=4,
        )
        free_capacity_before = manager._num_free_slots
    else:
        manager.page_size = 2
        manager.page_offsets_i32 = torch.arange(2, dtype=torch.int32)
        manager.num_pages = 8
        manager.seq_id_to_cached_pages = {}
        manager.buffer_req_to_page_slots = torch.full((2, 4), -1, dtype=torch.int32)
        manager._num_free_pages = 2
        payload = QuestPrefixBlockPayload(
            block_slot=None,
            block_slots=torch.tensor([2, 3], dtype=torch.int32),
            token_slots=torch.tensor([4, 5, 6, 7], dtype=torch.int32),
            block_start=0,
            block_end=4,
        )
        free_capacity_before = manager._num_free_pages

    prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"mixed-attach-rollback")
    block_id = prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload=payload,
            recurrent_payload=RecurrentPrefixPayload(
                token_count=4,
                layer_states={1: {"state": torch.ones(2)}},
            ),
            token_count=4,
            accounting_bytes=24,
            recurrent_bytes=8,
        ),
        token_ids=(1, 2, 3, 4),
    )
    prefix_cache.insert_block(block)

    def fail_recurrent_attach(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected recurrent attach failure")

    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.block_size = 4
    coordinator.prefix_cache = prefix_cache
    coordinator.cache_manager = manager
    coordinator.recurrent_state_manager = SimpleNamespace(
        attach_prefix_recurrent_payload=fail_recurrent_attach,
    )
    coordinator.offload_controller = None
    coordinator.seq_id_to_prefix_blocks = {}
    coordinator._step_h2d_operations = {}
    seq = SimpleNamespace(
        seq_id=7,
        prefix_cache_hit_len=4,
        prefix_cache_hit_block_count=1,
        prefix_cache_hit_last_block_id=block_id,
    )

    with pytest.raises(RuntimeError, match="injected recurrent attach failure"):
        coordinator.attach_prefix_cache_hits([seq])

    assert block.ref_count == 0
    assert coordinator.seq_id_to_prefix_blocks == {}
    assert not coordinator._step_h2d_operations
    assert manager.seq_id_to_row == ({7: 0} if row_preexisted else {})
    assert list(manager.free_rows) == ([1] if row_preexisted else [0, 1])
    assert manager.row_seq_lens == [0, 0]
    assert torch.count_nonzero(manager.buffer_req_to_token_slots).item() == 0
    assert torch.equal(payload.token_slots, torch.tensor([4, 5, 6, 7], dtype=torch.int32))
    if manager_type is StandardCacheManager:
        assert manager.seq_id_to_cached_ranges == {}
        assert manager._num_free_slots == free_capacity_before
    else:
        assert manager.seq_id_to_cached_pages == {}
        assert manager._num_free_pages == free_capacity_before
        assert torch.all(manager.buffer_req_to_page_slots == -1).item()


def test_mixed_runtime_uses_prefix_hit_as_virtual_prefill_boundary():
    runtime_state = RuntimeState(
        config=None,
        cache_manager=_BoundaryCacheManager(),
        prefix_cache_coordinator=_BoundaryCoordinator(),
    )
    seq = SimpleNamespace(num_prompt_tokens=5000, num_prefilled_tokens=0, prefix_cache_hit_len=4096)

    assert runtime_state.prefill_step_free_slots_for(seq) == 904


def test_mixed_runtime_drops_step_h2d_references_after_forward():
    coordinator = object.__new__(PrefixCacheCoordinator)
    operation = SimpleNamespace(auxiliary_tensors=(torch.ones(2),))
    coordinator._step_h2d_operations = {id(operation): operation}
    coordinator.record_step_tokens = lambda seqs, is_prefill: None
    coordinator.commit_pending_blocks = lambda seqs: None
    cache_manager = SimpleNamespace(on_forward_end=lambda seqs, is_prefill: None)
    runtime_state = RuntimeState(
        config=None,
        cache_manager=cache_manager,
        prefix_cache_coordinator=coordinator,
    )

    runtime_state.on_forward_end([], is_prefill=True)

    assert not coordinator._step_h2d_operations


def test_mixed_prefix_rejects_prefill_chunk_crossing_recurrent_snapshot_boundary():
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.prefix_cache = object()
    coordinator.block_size = 4096
    coordinator.capacity_limited_seq_ids = set()
    seq = SimpleNamespace(
        seq_id=11,
        num_prefilled_tokens=0,
        current_chunk_size=5000,
        token_ids=list(range(5000)),
    )

    with pytest.raises(RuntimeError, match="snapshot boundaries"):
        coordinator.record_step_tokens([seq], is_prefill=True)


def test_runtime_warmup_reset_clears_coordinator_before_kv_allocator():
    calls = []
    cache_manager = SimpleNamespace(reset_after_warmup=lambda: calls.append("kv"))
    recurrent_manager = SimpleNamespace(reset_after_warmup=lambda: calls.append("recurrent"))
    coordinator = SimpleNamespace(reset_after_warmup=lambda: calls.append("coordinator"))
    runtime_state = RuntimeState(
        config=None,
        cache_manager=cache_manager,
        recurrent_state_manager=recurrent_manager,
        prefix_cache_coordinator=coordinator,
    )

    runtime_state.reset_after_warmup()

    assert calls == ["coordinator", "kv", "recurrent"]


def test_mixed_prefix_warmup_reset_clears_dummy_blocks_and_accounting():
    freed = []
    config = SimpleNamespace(
        enable_prefix_caching=True,
        prefix_cache_block_size=4,
        prefix_recurrent_capacity_bytes=8,
        prefix_cache_max_blocks=2,
        model="test",
        hf_config=SimpleNamespace(model_type="test", dtype=torch.float16),
        tensor_parallel_size=1,
        sparse_method="",
        prefix_cache_salt="",
    )
    coordinator = PrefixCacheCoordinator(
        config,
        SimpleNamespace(
            free_prefix_kv_payload=lambda payload: freed.append(("kv", payload))
        ),
        SimpleNamespace(
            free_prefix_recurrent_payload=lambda payload: freed.append(("recurrent", payload))
        ),
    )
    block_id = coordinator.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    coordinator.prefix_cache.insert_block(
        PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=None,
            block_size=4,
            logical_block_idx=0,
            payload=MixedPrefixBlockPayload("kv-warmup", "state-warmup", 4, 8, 4),
            token_ids=(1, 2, 3, 4),
        )
    )
    coordinator.runtime_states[99] = object()
    coordinator.capacity_limited_seq_ids.add(99)
    coordinator.skipped_capacity_blocks = 1

    summary = coordinator.debug_state_summary()
    assert summary is not None
    assert summary["fingerprint"] == coordinator.prefix_cache.fingerprint.hex()
    assert summary["stats"]["prefix_cache_live_blocks"] == 1
    assert summary["blocks"] == [
        {
            "stable_block_id": block_id.hex(),
            "parent_block_id": None,
            "logical_block_idx": 0,
            "token_ids": [1, 2, 3, 4],
            "ref_count": 0,
            "last_access": 1,
            "eviction_priority": 0,
            "device_present": True,
            "host_present": False,
            "transfer": None,
        }
    ]

    coordinator.reset_after_warmup()

    assert len(coordinator.prefix_cache) == 0
    assert coordinator.runtime_states == {}
    assert coordinator.pending_recurrent_bytes == 0
    assert coordinator.capacity_limited_seq_ids == set()
    assert coordinator.skipped_capacity_blocks == 0
    assert freed == [("kv", "kv-warmup"), ("recurrent", "state-warmup")]


def test_mixed_offload_reset_drains_and_rebinds_controller():
    config = SimpleNamespace(
        prefix_cache_block_size=4,
        prefix_cache_max_blocks=2,
        model="test",
        hf_config=SimpleNamespace(model_type="test", dtype=torch.float16),
        tensor_parallel_size=1,
        sparse_method="",
        prefix_cache_salt="",
    )
    old_prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"old", max_blocks=2)
    calls = []
    controller = SimpleNamespace(
        prefix_cache=old_prefix_cache,
        synchronize_all=lambda: calls.append("synchronize"),
        reset=lambda: calls.append("reset"),
    )
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.config = config
    coordinator.block_size = 4
    coordinator.prefix_cache = old_prefix_cache
    coordinator.offload_controller = controller
    coordinator.cache_manager = SimpleNamespace()
    coordinator.recurrent_state_manager = SimpleNamespace()
    coordinator.seq_id_to_prefix_blocks = {}
    coordinator.seq_id_to_materialized_blocks = {}
    coordinator.runtime_states = {}
    coordinator.pending_blocks = {}
    coordinator.pending_duplicate_refs = {}
    coordinator.pending_block_ids = set()
    coordinator.pending_recurrent_bytes = 0
    coordinator.capacity_limited_seq_ids = set()
    coordinator.skipped_capacity_blocks = 0

    coordinator.reset_after_warmup()

    assert calls == ["synchronize", "reset"]
    assert coordinator.prefix_cache is not old_prefix_cache
    assert controller.prefix_cache is coordinator.prefix_cache


def test_runtime_state_does_not_proxy_undeclared_cache_manager_api():
    runtime_state = RuntimeState(
        config=None,
        cache_manager=SimpleNamespace(method_specific_api=lambda: "leaked"),
    )

    with pytest.raises(AttributeError, match="method_specific_api"):
        runtime_state.method_specific_api()


def test_generic_loader_only_applies_model_declared_weight_rules():
    generic_model = SimpleNamespace()
    assert (
        _target_weight_name_for_model(generic_model, "model.language_model.layers.0.weight")
        == "model.language_model.layers.0.weight"
    )
    assert _target_weight_name_for_model(generic_model, "visual.encoder.weight") == "visual.encoder.weight"

    qwen35_model = object.__new__(Qwen35ForCausalLM)
    assert (
        _target_weight_name_for_model(qwen35_model, "model.language_model.layers.0.weight")
        == "model.layers.0.weight"
    )
    assert _target_weight_name_for_model(qwen35_model, "visual.encoder.weight") is None


def test_mixed_prefix_recurrent_byte_budget_evicts_freeable_block():
    freed = []
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.block_size = 4
    coordinator.max_recurrent_bytes = 8
    coordinator.cache_manager = SimpleNamespace(free_prefix_kv_payload=lambda payload: freed.append(("kv", payload)))
    coordinator.recurrent_state_manager = SimpleNamespace(
        free_prefix_recurrent_payload=lambda payload: freed.append(("recurrent", payload))
    )
    coordinator.prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"test")
    block_id = coordinator.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload="kv-0",
            recurrent_payload="state-0",
            token_count=4,
            accounting_bytes=6,
            recurrent_bytes=6,
        ),
        token_ids=(1, 2, 3, 4),
    )
    coordinator.prefix_cache.insert_block(block)

    assert coordinator._evict_for_insert(1, incoming_recurrent_bytes=6)

    assert len(coordinator.prefix_cache) == 0
    assert freed == [("kv", "kv-0"), ("recurrent", "state-0")]


def test_mixed_offload_release_demotes_both_payloads_then_host_evicts():
    freed_device_kv = []
    freed_host = []
    prefix_cache = RadixPrefixIndex(
        block_size=4,
        fingerprint=b"mixed-offload-manager",
        max_blocks=1,
    )

    class FakeController:
        def __init__(self):
            self.prefix_cache = prefix_cache
            self.host_pool = SimpleNamespace(free_blocks=4, capacity_blocks=4)
            self.submitted = []

        def poll(self):
            return (0, 0)

        def wait_oldest_d2h(self):
            return False

        def submit_d2h(self, blocks):
            for block in blocks:
                self.prefix_cache.begin_d2h(block)
            self.submitted.append(list(blocks))

        def free_device_recurrent(self, block):
            block.payload.recurrent_payload.layer_states = {}

        def free_host_payloads(self, blocks):
            for block in blocks:
                freed_host.append(block.stable_block_id)
                block.payload.host_block_index = None

    kv_payload = SimpleNamespace(token_slots=torch.arange(4, dtype=torch.int32))
    recurrent_payload = RecurrentPrefixPayload(
        token_count=4,
        layer_states={1: {"state": torch.ones(2)}},
    )
    block_id = prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload=kv_payload,
            recurrent_payload=recurrent_payload,
            token_count=4,
            accounting_bytes=24,
            recurrent_bytes=8,
        ),
        token_ids=(1, 2, 3, 4),
        ref_count=1,
    )
    prefix_cache.insert_block(block)
    controller = FakeController()
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.block_size = 4
    coordinator.max_recurrent_bytes = 8
    coordinator.prefix_cache = prefix_cache
    coordinator.offload_controller = controller
    coordinator.cache_manager = SimpleNamespace(
        free_prefix_kv_payload_device=lambda payload: (
            freed_device_kv.append(payload.token_slots.clone()),
            setattr(payload, "token_slots", None),
        ),
    )
    coordinator.recurrent_state_manager = SimpleNamespace(
        free_prefix_recurrent_payload=lambda payload: None
    )
    coordinator.seq_id_to_prefix_blocks = {}
    coordinator.seq_id_to_materialized_blocks = {7: {block.stable_block_id: block}}
    coordinator.prefix_lookup_cache = PrefixLookupCache()
    coordinator.runtime_states = {}
    coordinator.pending_blocks = {}
    coordinator.pending_duplicate_refs = {}
    coordinator.pending_block_ids = set()
    coordinator.pending_recurrent_bytes = 0
    coordinator.capacity_limited_seq_ids = set()

    coordinator.release_seq(7)

    assert controller.submitted == [[block]]
    assert block.residency.transfer == PrefixTransferKind.D2H
    block.payload.host_block_index = 0
    prefix_cache.finish_d2h(block)

    coordinator.evict_for_slots(4)

    assert prefix_cache.get_block(block_id) is block
    assert block.residency.device_present is False
    assert block.residency.host_present is True
    assert block.payload.kv_payload.token_slots is None
    assert block.payload.recurrent_payload.layer_states == {}
    assert len(freed_device_kv) == 1

    assert coordinator._evict_for_insert(1, incoming_recurrent_bytes=8)
    assert len(prefix_cache) == 0
    assert freed_host == [block_id]


def test_mixed_prefix_capacity_skips_insert_when_live_chain_is_referenced():
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.block_size = 4
    coordinator.max_recurrent_bytes = 8
    coordinator.cache_manager = SimpleNamespace(free_prefix_kv_payload=lambda payload: None)
    coordinator.recurrent_state_manager = SimpleNamespace(free_prefix_recurrent_payload=lambda payload: None)
    coordinator.prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"test")
    block_id = coordinator.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload="kv-0",
            recurrent_payload="state-0",
            token_count=4,
            accounting_bytes=6,
            recurrent_bytes=6,
        ),
        token_ids=(1, 2, 3, 4),
        ref_count=1,
    )
    coordinator.prefix_cache.insert_block(block)

    assert not coordinator._evict_for_insert(1, incoming_recurrent_bytes=6)
    assert coordinator.prefix_cache.get_block(block_id) is block


def test_mixed_prefix_referenced_leaf_protects_its_ancestor_chain():
    freed = []
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.block_size = 4
    coordinator.max_recurrent_bytes = 8
    coordinator.cache_manager = SimpleNamespace(
        free_prefix_kv_payload=lambda payload: freed.append(("kv", payload))
    )
    coordinator.recurrent_state_manager = SimpleNamespace(
        free_prefix_recurrent_payload=lambda payload: freed.append(("recurrent", payload))
    )
    coordinator.prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"test")
    root_id = coordinator.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    leaf_id = coordinator.prefix_cache.stable_block_id([5, 6, 7, 8], root_id)
    for block_id, parent_id, logical_idx, tokens, ref_count in (
        (root_id, None, 0, (1, 2, 3, 4), 0),
        (leaf_id, root_id, 1, (5, 6, 7, 8), 1),
    ):
        coordinator.prefix_cache.insert_block(
            PrefixCacheBlock(
                stable_block_id=block_id,
                parent_block_id=parent_id,
                block_size=4,
                logical_block_idx=logical_idx,
                payload=MixedPrefixBlockPayload(
                    f"kv-{logical_idx}",
                    f"state-{logical_idx}",
                    4,
                    4,
                    4,
                ),
                token_ids=tokens,
                ref_count=ref_count,
            )
        )

    assert not coordinator._evict_for_insert(1, incoming_recurrent_bytes=4)
    assert set(coordinator.prefix_cache.blocks) == {root_id, leaf_id}
    assert freed == []


def test_mixed_prefix_free_validates_all_payloads_before_releasing_any_resource():
    freed = []
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.cache_manager = SimpleNamespace(
        free_prefix_kv_payload=lambda payload: freed.append(("kv", payload))
    )
    coordinator.recurrent_state_manager = SimpleNamespace(
        free_prefix_recurrent_payload=lambda payload: freed.append(("recurrent", payload))
    )
    valid = SimpleNamespace(payload=MixedPrefixBlockPayload("kv", "state", 4, 4, 4))
    invalid = SimpleNamespace(payload=object())

    with pytest.raises(RuntimeError, match="invalid payload"):
        coordinator._free_blocks([valid, invalid])

    assert freed == []


def _make_pending_capacity_coordinator(*, max_recurrent_bytes: int):
    clone_calls = []
    cache_manager = SimpleNamespace(
        build_prefix_kv_payload=lambda seq, start, end: (seq.seq_id, start, end),
        prefix_kv_payload_nbytes=lambda payload: 10,
        free_prefix_kv_payload=lambda payload: None,
        mark_materialized_prefix_kv_payload=lambda seq, payload: None,
    )
    recurrent_manager = SimpleNamespace(
        prefix_recurrent_snapshot_nbytes=lambda: 6,
        build_prefix_recurrent_payload=lambda seq, end: clone_calls.append((seq.seq_id, end))
        or f"state-{seq.seq_id}-{end}",
        prefix_recurrent_payload_nbytes=lambda payload: 6,
        free_prefix_recurrent_payload=lambda payload: None,
    )
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.block_size = 4
    coordinator.max_recurrent_bytes = max_recurrent_bytes
    coordinator.cache_manager = cache_manager
    coordinator.recurrent_state_manager = recurrent_manager
    coordinator.prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"test")
    coordinator.prefix_lookup_cache = PrefixLookupCache()
    coordinator.runtime_states = {}
    coordinator.pending_blocks = {}
    coordinator.pending_duplicate_refs = {}
    coordinator.pending_block_ids = set()
    coordinator.pending_recurrent_bytes = 0
    coordinator.capacity_limited_seq_ids = set()
    coordinator.skipped_capacity_blocks = 0
    coordinator.seq_id_to_prefix_blocks = {}
    coordinator.seq_id_to_materialized_blocks = {}
    return coordinator, clone_calls


def test_mixed_prefix_enforces_capacity_before_recurrent_snapshot_clone():
    coordinator, clone_calls = _make_pending_capacity_coordinator(max_recurrent_bytes=6)
    block_id = coordinator.prefix_cache.stable_block_id([9, 9, 9, 9], None)
    coordinator.prefix_cache.insert_block(
        PrefixCacheBlock(
            stable_block_id=block_id,
            parent_block_id=None,
            block_size=4,
            logical_block_idx=0,
            payload=MixedPrefixBlockPayload("kv", "state", 4, 16, 6),
            token_ids=(9, 9, 9, 9),
            ref_count=1,
        )
    )
    seq = SimpleNamespace(seq_id=7, prefix_cache_hit_block_count=0, prefix_cache_hit_last_block_id=None)

    coordinator._record_tokens(seq, [1, 2, 3, 4])

    assert clone_calls == []
    assert coordinator.pending_blocks[7] == []
    assert coordinator.pending_recurrent_bytes == 0
    assert coordinator.capacity_limited_seq_ids == {7}
    assert coordinator.skipped_capacity_blocks == 1


def test_mixed_prefix_pending_snapshots_count_and_duplicate_does_not_clone():
    coordinator, clone_calls = _make_pending_capacity_coordinator(max_recurrent_bytes=12)
    seq1 = SimpleNamespace(seq_id=1, prefix_cache_hit_block_count=0, prefix_cache_hit_last_block_id=None)
    seq2 = SimpleNamespace(seq_id=2, prefix_cache_hit_block_count=0, prefix_cache_hit_last_block_id=None)
    seq3 = SimpleNamespace(seq_id=3, prefix_cache_hit_block_count=0, prefix_cache_hit_last_block_id=None)
    duplicate = SimpleNamespace(seq_id=4, prefix_cache_hit_block_count=0, prefix_cache_hit_last_block_id=None)

    coordinator._record_tokens(seq1, [1, 2, 3, 4])
    coordinator._record_tokens(seq2, [5, 6, 7, 8])
    coordinator._record_tokens(duplicate, [1, 2, 3, 4])
    coordinator._record_tokens(seq3, [9, 10, 11, 12])

    assert clone_calls == [(1, 4), (2, 4)]
    assert coordinator.pending_recurrent_bytes == 12
    assert len(coordinator.pending_block_ids) == 2
    assert coordinator.pending_blocks[4] == []
    assert coordinator.capacity_limited_seq_ids == {3}


def test_mixed_prefix_duplicate_pending_block_holds_reference_after_commit():
    coordinator, clone_calls = _make_pending_capacity_coordinator(max_recurrent_bytes=6)
    owner = SimpleNamespace(seq_id=1, prefix_cache_hit_block_count=0, prefix_cache_hit_last_block_id=None)
    duplicate = SimpleNamespace(seq_id=2, prefix_cache_hit_block_count=0, prefix_cache_hit_last_block_id=None)

    coordinator._record_tokens(owner, [1, 2, 3, 4])
    coordinator._record_tokens(duplicate, [1, 2, 3, 4])
    coordinator.commit_pending_blocks([owner, duplicate])

    block = next(iter(coordinator.prefix_cache.blocks.values()))
    assert clone_calls == [(1, 4)]
    assert block.ref_count == 2
    coordinator.release_seq(owner.seq_id)
    assert block.ref_count == 1
    assert not coordinator.prefix_cache.can_evict(block)
    coordinator.release_seq(duplicate.seq_id)
    assert block.ref_count == 0


def test_mixed_prefix_snapshot_error_rolls_back_pending_accounting():
    coordinator, _ = _make_pending_capacity_coordinator(max_recurrent_bytes=6)
    coordinator.recurrent_state_manager.build_prefix_recurrent_payload = (
        lambda seq, end: (_ for _ in ()).throw(RuntimeError("clone failed"))
    )
    seq = SimpleNamespace(seq_id=1, prefix_cache_hit_block_count=0, prefix_cache_hit_last_block_id=None)

    with pytest.raises(RuntimeError, match="clone failed"):
        coordinator._record_tokens(seq, [1, 2, 3, 4])

    assert coordinator.pending_recurrent_bytes == 0
    assert coordinator.pending_block_ids == set()
    assert coordinator.pending_blocks[1] == []


def test_mixed_prefix_mark_failure_rolls_back_radix_ref_and_allocator_ownership():
    coordinator, _ = _make_pending_capacity_coordinator(max_recurrent_bytes=12)
    allocator = SimpleNamespace(
        free_slots=0,
        cached_ranges=set(),
        freed_kv_payloads=[],
    )

    def fail_mark(seq, payload):
        allocator.cached_ranges.add((seq.seq_id, payload))
        raise RuntimeError("mark failed")

    def rollback_mark(seq, payload):
        allocator.cached_ranges.discard((seq.seq_id, payload))

    coordinator.cache_manager.mark_materialized_prefix_kv_payload = fail_mark
    coordinator.cache_manager.rollback_materialized_prefix_kv_payload = rollback_mark
    coordinator.cache_manager.free_prefix_kv_payload = (
        lambda payload: allocator.freed_kv_payloads.append(payload)
    )
    freed_recurrent = []
    coordinator.recurrent_state_manager.free_prefix_recurrent_payload = (
        lambda payload: freed_recurrent.append(payload)
    )
    seq = SimpleNamespace(
        seq_id=1,
        prefix_cache_hit_block_count=0,
        prefix_cache_hit_last_block_id=None,
    )
    coordinator._record_tokens(seq, [1, 2, 3, 4, 5, 6, 7, 8])

    with pytest.raises(RuntimeError, match="mark failed"):
        coordinator.commit_pending_blocks([seq])

    assert len(coordinator.prefix_cache) == 0
    assert not coordinator.seq_id_to_materialized_blocks.get(seq.seq_id)
    assert coordinator.pending_recurrent_bytes == 0
    assert coordinator.pending_block_ids == set()
    assert allocator.cached_ranges == set()
    assert allocator.free_slots == 0
    assert allocator.freed_kv_payloads == []
    assert freed_recurrent == ["state-1-4", "state-1-8"]
    assert coordinator.prefix_cache.stats()["prefix_cache_committed_blocks"] == 0


def test_recurrent_state_manager_reuses_preallocated_rows_for_decode():
    config = SimpleNamespace(
        runtime_layout=SimpleNamespace(
            linear_attention_layer_indices=(1,),
            is_linear_attention=lambda layer_idx: int(layer_idx) == 1,
        ),
        enable_prefix_caching=True,
        max_num_seqs_in_batch=2,
        max_decoding_seqs=4,
        max_num_seqs_in_gpu=6,
        recurrent_state_max_bytes=None,
        prefix_cache_block_size=4,
    )
    manager = RecurrentStateManager(
        config,
        _single_process_parallel_context(),
        device=torch.device("cpu"),
        state_spec=RecurrentStateSpec(
            name="test recurrent",
            tensor_specs=(
                RecurrentTensorSpec("conv_state", (2, 3), torch.float16),
                RecurrentTensorSpec("recurrent_state", (2, 2, 2), torch.float16),
            ),
        ),
    )
    assert manager.row_capacity == 6
    assert manager.pool_bytes == 196
    assert manager.layer_buffers[1]["conv_state"].shape == (7, 2, 3)
    assert manager.layer_buffers[1]["recurrent_state"].shape == (7, 2, 2, 2)
    seq = SimpleNamespace(seq_id=7)
    manager.prepare_step([seq], is_prefill=True)
    conv = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    recurrent = torch.arange(8, dtype=torch.float16).reshape(2, 2, 2)
    manager.set_layer_state(7, 1, {"conv_state": conv, "recurrent_state": recurrent})
    first_ptr = manager.get_layer_state(7, 1)["conv_state"].data_ptr()

    manager.set_layer_state(7, 1, {"conv_state": conv + 1, "recurrent_state": recurrent + 1})
    manager.prepare_decode_static([seq], token_batch=4, device=torch.device("cpu"))
    state_buffers, state_indices = manager.get_decode_layer_state(
        [seq],
        layer_idx=1,
        token_batch=4,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    conv_pool = state_buffers["conv_state"]
    recurrent_pool = state_buffers["recurrent_state"]

    assert manager.get_layer_state(7, 1)["conv_state"].data_ptr() == first_ptr
    assert conv_pool.shape == (7, 2, 3)
    assert recurrent_pool.shape == (7, 2, 2, 2)
    assert conv_pool.dtype == torch.float16
    assert recurrent_pool.dtype == torch.float16
    assert state_indices.tolist() == [0, 6, 6, 6]
    assert torch.equal(conv_pool[0], conv + 1)
    assert torch.count_nonzero(conv_pool[6]).item() == 0

    manager.free_seq(7)
    replacement = SimpleNamespace(seq_id=8)
    manager.prepare_step([replacement], is_prefill=True)
    manager.set_layer_state(8, 1, {"conv_state": conv + 2, "recurrent_state": recurrent + 2})
    manager.prepare_decode_static([replacement], token_batch=4, device=torch.device("cpu"))

    assert manager.decode_state_indices[4].data_ptr() == state_indices.data_ptr()
    assert manager.decode_state_indices[4].tolist() == [1, 6, 6, 6]

    manager.reset_after_warmup()

    assert manager.decode_state_indices[4].data_ptr() == state_indices.data_ptr()


def test_recurrent_prefix_h2d_waits_and_attaches_one_layer_at_first_access(monkeypatch):
    config = SimpleNamespace(
        runtime_layout=SimpleNamespace(
            linear_attention_layer_indices=(1, 3),
            is_linear_attention=lambda layer_idx: int(layer_idx) in {1, 3},
        ),
        enable_prefix_caching=True,
        max_num_seqs_in_batch=1,
        max_decoding_seqs=1,
        max_num_seqs_in_gpu=2,
        recurrent_state_max_bytes=None,
        prefix_cache_block_size=4,
    )
    manager = RecurrentStateManager(
        config,
        _single_process_parallel_context(),
        device=torch.device("cpu"),
        state_spec=RecurrentStateSpec(
            name="layer-ready-test",
            tensor_specs=(RecurrentTensorSpec("state", (2,), torch.float16),),
        ),
    )
    payload = RecurrentPrefixPayload(
        token_count=4,
        layer_states={
            1: {"state": torch.tensor([1.0, 2.0], dtype=torch.float16)},
            3: {"state": torch.tensor([3.0, 4.0], dtype=torch.float16)},
        },
    )
    waits = []
    monkeypatch.setattr(
        device_runtime,
        "wait_event",
        lambda event, device=None: waits.append((event, device)),
    )
    seq = SimpleNamespace(seq_id=9)
    manager.attach_prefix_recurrent_payload(
        seq,
        payload,
        readiness_events={1: "layer-1-ready", 3: "layer-3-ready"},
    )

    first = manager.get_layer_state(seq.seq_id, 1)

    assert waits == [("layer-1-ready", torch.device("cpu"))]
    assert torch.equal(first["state"], payload.layer_states[1]["state"])
    assert seq.seq_id in manager.pending_prefix_payloads
    assert torch.count_nonzero(manager.layer_buffers[3]["state"][0]).item() == 0

    later = manager.get_layer_state(seq.seq_id, 3)

    assert waits[-1] == ("layer-3-ready", torch.device("cpu"))
    assert torch.equal(later["state"], payload.layer_states[3]["state"])
    assert seq.seq_id not in manager.pending_prefix_payloads


def test_recurrent_state_manager_uses_model_declared_state_schema():
    config = SimpleNamespace(
        runtime_layout=SimpleNamespace(
            linear_attention_layer_indices=(0,),
            is_linear_attention=lambda layer_idx: int(layer_idx) == 0,
        ),
        enable_prefix_caching=True,
        max_num_seqs_in_batch=1,
        max_decoding_seqs=1,
        max_num_seqs_in_gpu=3,
        recurrent_state_max_bytes=None,
        prefix_cache_block_size=4,
    )
    manager = RecurrentStateManager(
        config,
        _single_process_parallel_context(),
        device=torch.device("cpu"),
        state_spec=RecurrentStateSpec(
            name="single-state model",
            tensor_specs=(RecurrentTensorSpec("ssm_state", (4,), torch.float16),),
        ),
    )
    seq = SimpleNamespace(seq_id=3)
    state = torch.arange(4, dtype=torch.float16)

    manager.set_layer_state(seq.seq_id, 0, {"ssm_state": state})
    manager.prepare_decode_static([seq], token_batch=1, device=torch.device("cpu"))
    state_buffers, state_indices = manager.get_decode_layer_state(
        [seq],
        layer_idx=0,
        token_batch=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )

    assert tuple(state_buffers) == ("ssm_state",)
    assert torch.equal(state_buffers["ssm_state"][0], state)
    assert state_indices.tolist() == [0]
    with pytest.raises(RuntimeError, match="schema mismatch"):
        manager.set_layer_state(seq.seq_id, 0, {"conv_state": state})


def test_quest_mixed_prefix_payload_spans_multiple_pages():
    manager = object.__new__(QuestCacheManager)
    manager.page_size = 2
    manager.device = torch.device("cpu")
    manager.num_pages = 8
    manager.page_offsets_i32 = torch.arange(2, dtype=torch.int32)
    manager.seq_id_to_row = {7: 0}
    manager.row_seq_lens = torch.tensor([4, 0], dtype=torch.int32).numpy()
    manager.buffer_req_to_token_slots = torch.zeros((2, 8), dtype=torch.int32)
    manager.buffer_req_to_token_slots[0, :4] = torch.tensor([0, 1, 4, 5], dtype=torch.int32)
    manager.buffer_req_to_page_slots = torch.full((2, 4), -1, dtype=torch.int32)
    manager.seq_id_to_cached_pages = {}
    manager.free_rows = deque([1])
    manager._num_free_pages = 6
    manager.free_pages_stack = torch.empty(8, dtype=torch.int32)

    payload = manager.build_prefix_kv_payload(SimpleNamespace(seq_id=7), 0, 4)
    manager.seq_id_to_row.pop(7)
    manager.attach_prefix_kv_payloads(SimpleNamespace(seq_id=8), [payload])

    assert payload.block_slot is None
    assert payload.block_slots.tolist() == [0, 2]
    assert manager.buffer_req_to_page_slots[1, :2].tolist() == [0, 2]
    assert manager.buffer_req_to_token_slots[1, :4].tolist() == [0, 1, 4, 5]
    assert manager.row_seq_lens[1] == 4

    manager.free_prefix_kv_payload(payload)

    assert manager._num_free_pages == 8
    assert manager.free_pages_stack[6:8].tolist() == [0, 2]


def test_quantized_loader_rejects_unloaded_fp8_modules():
    class FakeQuantizedLinear(torch.nn.Module):
        quantized = True
        _quantized_weight_loaded = False
        _quantized_loaded_ranges = [(0, 128)]

    model = torch.nn.Module()
    model.proj = FakeQuantizedLinear()

    with pytest.raises(ValueError, match="Missing FP8 weight loads"):
        _validate_all_quantized_weights_loaded(model)


def test_decode_window_failure_preempts_and_releases_lease_before_replay():
    config = _resident_scheduler_config()
    config.decode_reservation_tokens = 4
    cache = _ResidentAdmissionCache()
    cache.num_free_slots = 6
    runtime = RuntimeState(config, cache)
    scheduler = Scheduler(config, runtime)
    seqs = [Sequence([1, 2], SamplingParams(max_tokens=20)) for _ in range(2)]
    for seq in seqs:
        seq.num_prefilled_tokens = 2
        seq.append_token(3)
        scheduler.decoding.append(seq)
    scheduled, is_prefill, victims = scheduler.schedule()
    assert not scheduled and not is_prefill
    assert victims == [seqs[1]]
    assert seqs[1].is_recompute_replay
    assert list(scheduler.decoding) == [seqs[0]]
    assert runtime.decode_reservations.outstanding() == {'slots': 4}
    runtime.free_seq(seqs[1].seq_id)
    scheduled, is_prefill, victims = scheduler.schedule()
    assert scheduled == [seqs[0]] and not is_prefill and not victims
    # Recovery must not immediately reconsume the capacity just released.
    assert seqs[1] in scheduler.waiting
    runtime.free_seq(seqs[0].seq_id)
    assert runtime.decode_reservations.outstanding() == {}
