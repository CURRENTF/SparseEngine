from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import pytest

from sparseengine.engine.cache_manager.storage import CacheLayout
from sparseengine.engine.cache_manager.standard import StandardCacheManager
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.startup import (
    KVCapacityPlan,
    StartupMemoryProfile,
    feasible_startup_graph_plan,
    profiling_kv_budget_bytes,
    profiling_kv_slots,
    profiling_prefill_chunk_lengths,
)
from sparseengine.models.layout import RuntimeLayout


def _config(*, sparse_method: str = "", prefill_sparse_method=None):
    layout = RuntimeLayout.dense(4)
    layout = RuntimeLayout(
        **{
            **layout.__dict__,
            "kv_num_heads": (8, 8, 8, 8),
            "kv_head_dims": (128, 128, 128, 128),
        }
    )
    return SimpleNamespace(
        sparse_method=sparse_method,
        prefill_sparse_method=prefill_sparse_method,
        attention_cache_layout=CacheLayout.EXPLICIT_KV,
        runtime_layout=layout,
        parallel_topology=SimpleNamespace(attn_tp_size=2),
        hf_config=SimpleNamespace(
            dtype=torch.bfloat16,
            num_key_value_heads=8,
            num_attention_heads=32,
            hidden_size=4096,
            head_dim=128,
        ),
        quest_chunk_size=16,
        max_num_batched_tokens=16,
        max_num_seqs_in_batch=4,
        max_num_seqs_in_gpu=8,
        max_decoding_seqs=4,
        engine_prefill_chunk_size=8,
        max_model_len=32,
        decode_graph_startup_capture=False,
        sink_keep_tokens=2,
        decode_keep_tokens=8,
        recent_keep_tokens=6,
    )


def test_capacity_plan_uses_larger_runtime_peak_and_external_headroom():
    profile = StartupMemoryProfile(
        total_bytes=1000,
        persistent_bytes=300,
        runtime_persistent_bytes=0,
        profile_persistent_growth_bytes=0,
        prefill_transient_bytes=120,
        decode_transient_bytes=80,
        cuda_graph_bytes=50,
    )

    plan = KVCapacityPlan.from_profile(profile, 0.9)

    assert plan.target_bytes == 900
    assert plan.safety_headroom_bytes == 100
    assert plan.runtime_transient_bytes == 120
    assert plan.local_kv_budget_bytes == 430


def test_explicit_profiling_budget_uses_tp_local_kv_shape():
    config = _config()

    budget = profiling_kv_budget_bytes(config, 10)

    expected_bytes_per_slot = 4 * 2 * 4 * 128 * 2
    expected_row_mapping = 8 * 32 * 4
    assert budget == 10 * (expected_bytes_per_slot + 4) + expected_row_mapping


def test_explicit_profiling_budget_uses_declared_head_dim_without_layout_shapes():
    config = _config()
    config.runtime_layout = RuntimeLayout.dense(4)

    budget = profiling_kv_budget_bytes(config, 10)

    expected_bytes_per_slot = 4 * 2 * 4 * 128 * 2
    expected_row_mapping = 8 * 32 * 4
    assert budget == 10 * (expected_bytes_per_slot + 4) + expected_row_mapping


def test_quest_profiling_budget_includes_page_metadata():
    config = _config(sparse_method="quest")

    budget = profiling_kv_budget_bytes(config, 17)

    bytes_per_slot = 4 * 2 * 4 * 128 * 2
    fixed_metadata = 8 * 32 * 4 + 8 * 2 * 4 + 16 * (4 + 8)
    assert budget == 32 * bytes_per_slot + 2 * (bytes_per_slot + 4) + fixed_metadata


def test_h2o_prefill_only_uses_h2o_physical_cache_budget():
    config = _config(prefill_sparse_method="h2o_prefill")

    budget = profiling_kv_budget_bytes(config, 10)

    from sparseengine.engine.cache_manager.methods.snapkv import resolve_snapkv_cache_capacity

    # Regression: a doubled startup budget made high-concurrency H2O OOM
    # before profiling, despite its pruned resident rows fitting in memory.
    slots, _, _ = resolve_snapkv_cache_capacity(
        available_bytes=budget, slot_bytes_per_layer=2 * 4 * 128 * 2,
        num_kv_layers=4, max_buffer_rows=8, max_model_len=32,
    )
    assert slots == 10


def test_snapkv_profiling_budget_matches_storage_and_metadata_tensor_bytes():
    # Regression: a doubled temporary KV payload OOMed Qwen's MoE warmup at
    # batch32 even though the required graph-family KV storage fitted VRAM.
    config = _config(sparse_method="snapkv")
    slots = 10
    tensors = []
    for heads, dim in config.runtime_layout.local_kv_shapes(2):
        tensors.extend([
            torch.empty(2, slots, heads, dim, dtype=config.hf_config.dtype),
            torch.empty(slots, dtype=torch.int32),
            torch.empty(config.max_num_seqs_in_gpu, config.max_model_len, dtype=torch.int32),
        ])
    assert profiling_kv_budget_bytes(config, slots) == sum(t.numel() * t.element_size() for t in tensors)


def test_snapkv_mla_profiling_budget_preserves_replicated_latent_width():
    config = _config(sparse_method="snapkv")
    config.attention_cache_layout = CacheLayout.MLA_LATENT
    config.hf_config.kv_lora_rank = 512
    config.hf_config.qk_rope_head_dim = 64
    slots = 10
    tensors = []
    for _ in range(config.runtime_layout.num_kv_layers):
        tensors.extend([
            torch.empty(slots, 512, dtype=config.hf_config.dtype),
            torch.empty(slots, 64, dtype=config.hf_config.dtype),
            torch.empty(slots, dtype=torch.int32),
            torch.empty(config.max_num_seqs_in_gpu, config.max_model_len, dtype=torch.int32),
        ])
    assert profiling_kv_budget_bytes(config, slots) == sum(t.numel() * t.element_size() for t in tensors)


@pytest.mark.parametrize("token_budget, resident_rows", [(17, 8), (2, 8), (100, 2)])
def test_prefill_profile_balances_chunks_within_token_and_row_limits(token_budget, resident_rows):
    config = _config()
    config.max_num_batched_tokens = token_budget
    config.max_num_seqs_in_gpu = resident_rows
    lengths = profiling_prefill_chunk_lengths(config)

    assert len(lengths) == min(config.max_num_seqs_in_batch, resident_rows, token_budget)
    assert 0 < min(lengths) <= max(lengths) <= config.engine_prefill_chunk_size
    assert max(lengths) - min(lengths) <= 1
    assert sum(lengths) <= token_budget
    if sum(lengths) < token_budget:
        assert all(length == config.engine_prefill_chunk_size for length in lengths)


def test_profiling_kv_slots_cover_runtime_steps_not_maximum_context():
    config = _config()
    config.max_model_len = 1_000_000

    assert profiling_kv_slots(config) == 24


def test_quest_profiling_slots_round_each_request_to_a_page():
    config = _config(sparse_method="quest")

    assert profiling_kv_slots(config) == 64


def test_production_graph_plan_skips_families_larger_than_final_kv():
    config = _config(sparse_method="quest")
    config.sink_keep_tokens = 2
    config.decode_keep_tokens = 8
    config.recent_keep_tokens = 6
    plan = [(12, 32), (5, 32), (4, 32), (2, 32)]

    class AdmissionOracle:
        def startup_batch_fits(self, prompt_lengths, *, max_tokens):
            full_layers = sum(int(length) + int(max_tokens) for length in prompt_lengths)
            centers = sum((int(length) + 7) // 8 for length in prompt_lengths)
            return full_layers <= 32 and centers <= 4

    feasible, skipped = feasible_startup_graph_plan(
        config,
        plan,
        AdmissionOracle(),
    )

    assert feasible == [(4, 32), (2, 32)]
    assert skipped == [(12, 32), (5, 32)]


def test_explicit_budget_still_resolves_mixed_kv_and_recurrent_prefix_capacity():
    manager = object.__new__(StandardCacheManager)
    manager.config = SimpleNamespace(
        resolved_prefix_cache_mode="radix",
        prefix_recurrent_bytes_per_block=40,
        prefix_cache_max_blocks=None,
        prefix_cache_block_size=4,
        hf_config=SimpleNamespace(),
    )
    manager.allocation_budget_bytes = 1_000
    manager.attention_cache_bytes_per_slot_per_layer = lambda: 10
    manager._kv_allocation_bytes_per_prefix_block = lambda _slot_bytes: 60

    available, slot_bytes = manager._get_available_slots_info()

    assert available == 600
    assert slot_bytes == 10
    assert manager.config.prefix_cache_max_blocks == 10
    assert manager.config.prefix_recurrent_capacity_bytes == 400
    assert manager.config.prefix_kv_block_capacity == 10


def test_startup_batch_feasibility_uses_all_memory_oracle_budgets():
    class MultiBudgetManager:
        def scheduler_capacity_snapshot(self):
            return nullcontext()

        def prompt_admission_budgets(self, _waiting, _chunk_size):
            return {"full_layers": 100, "centers": 2}

        def prompt_admission_costs(self, seq):
            return {
                "full_layers": int(seq.num_prompt_tokens + seq.max_tokens),
                "centers": 1,
            }

        def prompt_logical_reservation_cost(self, seq):
            return int(seq.num_prompt_tokens)

        def prompt_admission_free_slots(self):
            return 100

    runtime = RuntimeState(
        SimpleNamespace(engine_prefill_chunk_size=8, max_num_seqs_in_gpu=8),
        MultiBudgetManager(),
    )

    assert runtime.startup_batch_fits((16, 16), max_tokens=2)
    assert not runtime.startup_batch_fits((16, 16, 16), max_tokens=2)


@pytest.mark.parametrize("free_by_layer,expected", [([4, 4], True), ([4, 3], False)])
def test_startup_decode_checks_all_h2o_layers_without_allocating(free_by_layer, expected):
    # A later layer can lack the fourth append even when the first layer fits.
    from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager

    manager = object.__new__(H2OCacheManager)
    manager._num_free_slots = list(free_by_layer)
    manager.kv_transformer_layer_indices = lambda: (0, 1)
    runtime = RuntimeState(SimpleNamespace(), manager)
    seqs = [SimpleNamespace(seq_id=i) for i in range(4)]

    assert runtime.startup_decode_batch_fits(seqs) is expected
    assert manager._num_free_slots == free_by_layer


@pytest.mark.parametrize("free_pages,lengths,expected", [
    (1, [15, 16], True),
    (1, [16, 32], False),
    (0, [15], True),
    (0, [16], False),
])
def test_startup_decode_uses_page_append_costs(free_pages, lengths, expected):
    # One token may consume a whole new page, or no shared capacity at all.
    # A len(seqs) <= free_slots check would get both boundary cases wrong.
    from sparseengine.engine.cache_manager.quantized import QuantizedCacheManager
    from sparseengine.engine.cache_manager.quantized_pages import QuantizedPagePool

    manager = object.__new__(QuantizedCacheManager)
    manager.page_size = 16
    manager.config = SimpleNamespace(max_decoding_seqs=4)
    manager.prefix_cache = None
    manager._scheduler_capacity_snapshot_depth = 0
    manager.page_pool = QuantizedPagePool(
        free_pages + sum((length + 15) // 16 for length in lengths), 16, 64,
    )
    for seq_id, length in enumerate(lengths):
        manager.page_pool.append(seq_id, length)
    before_free = list(manager.page_pool.free)
    before_pages = {key: list(pages) for key, pages in manager.page_pool.pages.items()}
    runtime = RuntimeState(SimpleNamespace(), manager)

    assert runtime.startup_decode_batch_fits(
        [SimpleNamespace(seq_id=i) for i in range(len(lengths))],
    ) is expected
    assert manager.page_pool.lengths == dict(enumerate(lengths))
    assert manager.page_pool.free == before_free
    assert manager.page_pool.pages == before_pages


def test_startup_decode_rejects_request_local_capacity_limit():
    manager = SimpleNamespace(
        decode_step_free_slots=lambda: 32,
        decode_step_free_slots_for=lambda seq: 0 if seq.seq_id == 1 else 32,
        decode_step_reservation_cost=lambda seq: 1,
    )
    runtime = RuntimeState(SimpleNamespace(), manager)

    assert not runtime.startup_decode_batch_fits(
        [SimpleNamespace(seq_id=0), SimpleNamespace(seq_id=1)],
    )


@pytest.mark.parametrize("cache_tokens", [0, 32])
def test_omnikv_profiling_budget_can_admit_its_profile_workload(cache_tokens):
    # Native-KV profiling budgets omitted fixed offload pools, causing startup
    # to fail before it could measure production capacity when LRU was enabled.
    from sparseengine.engine.cache_manager.methods.omnikv.capacity import plan_omnikv_pools

    config = _config(sparse_method="omnikv")
    config.enable_omnikv_offload = True
    config.full_attention_layers = [0]
    config.omnikv_offload_cache_tokens = cache_tokens
    slots = profiling_kv_slots(config)
    budget = profiling_kv_budget_bytes(config, slots)
    plan = plan_omnikv_pools(config, [0], 4, 8, 2 * 4 * 128 * 2)
    assert (budget - plan.fixed_bytes) // plan.slot_bytes >= slots


def test_omnikv_cache_cannot_evict_tokens_selected_in_the_same_step():
    from sparseengine.engine.cache_manager.methods.omnikv.capacity import plan_omnikv_pools

    config = _config(sparse_method="omnikv")
    config.omnikv_offload_cache_tokens = 1
    with pytest.raises(ValueError, match="cover the full selected-token budget"):
        plan_omnikv_pools(config, [0], 4, 8, 2048)


@pytest.mark.parametrize("latent_bits,reserve_ratio", [(4, 0.1), (0, 0.0)])
def test_deltakv_profile_budget_funds_fixed_pools_and_profile_requests(latent_bits, reserve_ratio):
    # Regression: an otherwise idle GPU failed before profiling because the
    # temporary budget omitted reconstruction scratch for eight decode rows.
    # Meta tensors exercise the real allocator without allocating GPU memory.
    from sparseengine.configs.groups import DeltaKVConfig
    from sparseengine.engine.cache_manager.methods.deltakv_less_memory_cuda_graph import DeltaKVLessMemoryCudaGraphCacheManager

    config = _config(sparse_method="deltakv")
    for key, value in vars(DeltaKVConfig()).items():
        setattr(config, key, value)
    config.runtime_layout = RuntimeLayout.dense(32)
    config.parallel_topology.attn_tp_size = 1
    config.full_attention_layers = [0, 2, 7, 13, 16, 26]
    config.max_model_len = 131072
    config.max_num_batched_tokens = 65536
    config.engine_prefill_chunk_size = 8192
    config.max_num_seqs_in_batch = 4
    config.max_decoding_seqs = 8
    config.max_num_seqs_in_gpu = 12
    config.sink_keep_tokens = 64
    config.recent_keep_tokens = 256
    config.decode_keep_tokens = 4096
    config.decode_graph = True
    config.decode_graph_capture_sizes = [1, 2, 4, 8]
    config.deltakv_latent_dim = 512
    config.deltakv_latent_quant_bits = latent_bits
    config.deltakv_latent_quant_group_size = 32
    config.full_layer_kv_quant_bits = 0
    config.deltakv_full_pool_reserve_ratio = reserve_ratio
    slots = profiling_kv_slots(config)
    budget = profiling_kv_budget_bytes(config, slots)
    config.startup_cache_phase = "profiling"
    manager = object.__new__(DeltaKVLessMemoryCudaGraphCacheManager)
    manager.config = config
    manager.hf_config = config.hf_config
    manager.device = torch.device("meta")
    manager.num_kv_heads = 8
    manager.head_dim = 128
    manager.max_model_len = config.max_model_len
    manager.max_buffer_rows = config.max_num_seqs_in_gpu
    manager.full_layer_ids = config.full_attention_layers
    manager.deltakv_layer_ids = [i for i in range(32) if i not in config.full_attention_layers]
    manager._get_available_slots_info = lambda: (budget, 2 * 8 * 128 * 2)
    manager.allocate_kv_cache()

    # An independent workload bound: keep the entire profiling prompt in raw
    # form alongside the allocator's reserved decode scratch, and allow its
    # latent representation plus all full-layer tokens.
    assert manager.deltakv_full_num_slots - manager._deltakv_decode_reconstruct_full_reserve >= slots
    assert manager.deltakv_latent_num_slots >= slots
    assert manager.full_num_slots >= slots
    # Sum actual KV payloads and workspaces, including quantization scales,
    # rather than duplicating the sizing formula. Index/row metadata is
    # measured separately by startup's runtime-persistent memory profile.
    payload_names = {
        "full_kv_cache", "deltakv_full_kv_cache", "deltakv_materialized_kv_cache",
        "deltakv_prefill_staging_kv_cache", "deltakv_prefill_staging_pre_rope_k_cache",
        "deltakv_latent_cache", "deltakv_latent_scales", "deltakv_latent_mins",
    }
    tensors = [value for key, value in vars(manager).items()
               if key in payload_names and isinstance(value, torch.Tensor)]
    allocated = sum(t.numel() * t.element_size() for t in tensors)
    assert allocated <= budget

    # The production rebuild must reserve reconstruction before sizing the
    # variable pools too; it must not retain the profiling-only prompt reserve.
    config.startup_cache_phase = "production"
    manager.allocate_kv_cache()
    assert manager.deltakv_full_num_slots > manager._deltakv_decode_reconstruct_full_reserve
    assert manager._deltakv_centers_capacity > 0
    tensors = [value for key, value in vars(manager).items()
               if key in payload_names and isinstance(value, torch.Tensor)]
    assert sum(t.numel() * t.element_size() for t in tensors) <= budget
