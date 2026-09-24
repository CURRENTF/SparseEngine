import os
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sparseengine.config import Config
from sparseengine.configs.cuda_graph import (
    _default_decode_cuda_graph_capture_sizes,
    _resolve_decode_static_batch_capacity,
)
from sparseengine.engine.cache_manager.base import LayerBatchStates
from sparseengine.engine.cache_manager.standard import StandardCacheManager
from sparseengine.engine.cache_manager.methods.deltakv import DeltaKVCacheManager
from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager
from sparseengine.engine.cache_manager.methods.deltakv_less_memory_cuda_graph import (
    DeltaKVLessMemoryCudaGraphCacheManager,
)
from sparseengine.engine.cache_manager.methods.snapkv import SnapKVCacheManager
from sparseengine.engine.decode_cuda_graph import DecodeCudaGraphRunner
from sparseengine.engine.llm_engine import (
    LLMEngine,
    _moe_workspace_warmup_token_counts,
)
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.prefill import (
    PREFILL_EXECUTION_CHUNKED,
    PREFILL_EXECUTION_FULL,
    PREFILL_EXECUTION_RAW_OFFLOAD,
)
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.scheduler import Scheduler
from sparseengine.engine.sequence import Sequence
from sparseengine.layers.sampler import Sampler
from sparseengine.sampling_params import SamplingParams
from sparseengine.engine.sparse_controller import SparseController
from sparseengine.engine.sparse_methods import AttentionEndEvent
from sparseengine.engine.sparse_methods.snapkv import PyramidKVRuntime
from sparseengine.method_registry import (
    PREFILL_POLICY_ALL_CHUNKED,
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
)


class FakeMemoryOracle:
    def reserve_decode_windows(self, decoding, waiting):
        return None

    def prefill_private_slots_for(self, seq):
        return 0

    def __init__(
        self,
        free_slots=1_000_000,
        *,
        step_free_slots=None,
        force_full_prefill=False,
        force_whole_prefill=False,
        prefix_hit_len=0,
        prefix_hit_blocks=0,
        long_prefill_offload=False,
        min_final_prefill_chunk_size=0,
        execution_mode=None,
    ):
        self._free_slots = int(free_slots)
        self._step_free_slots = int(step_free_slots) if step_free_slots is not None else int(free_slots)
        self._force_full_prefill = bool(force_full_prefill)
        self._force_whole_prefill = bool(force_whole_prefill)
        self.prefix_hit_len = int(prefix_hit_len)
        self.prefix_hit_blocks = int(prefix_hit_blocks)
        self._long_prefill_offload = bool(long_prefill_offload)
        self._min_final_prefill_chunk_size = int(min_final_prefill_chunk_size)
        self._execution_mode = execution_mode
        self.refresh_calls = 0
        self.clear_calls = 0

    @property
    def num_free_slots(self):
        return self._free_slots

    def prefill_step_free_slots(self):
        return self._step_free_slots

    def should_schedule_full_prefill(self, seq):
        return self._force_full_prefill and int(seq.num_prefilled_tokens) == 0

    def requires_full_prefill_step(self, seq):
        return self._force_whole_prefill and int(seq.num_prefilled_tokens) == 0

    def requires_long_prefill_offload(self, seq):
        return self._long_prefill_offload

    def prefill_step_free_slots_for(self, seq):
        return self._free_slots

    def prefill_step_reservation_cost(self, seq, scheduled_tokens):
        return int(scheduled_tokens)

    def min_final_prefill_chunk_size(self, seq):
        return self._min_final_prefill_chunk_size

    def decode_step_free_slots(self):
        return self._free_slots

    def decode_step_free_slots_for(self, seq):
        return self._free_slots

    def decode_step_reservation_cost(self, seq):
        return 1

    def reserved_prefill_slots(self, waiting, engine_prefill_chunk_size):
        return 0

    def remaining_prefill_tokens(self, seq):
        virtual_prefilled = max(seq.num_prefilled_tokens, seq.prefix_cache_hit_len)
        return int(seq.num_prompt_tokens - virtual_prefilled)

    def prefill_execution_mode(self, seq):
        del seq
        if self._execution_mode is not None:
            return self._execution_mode
        if self._long_prefill_offload:
            return PREFILL_EXECUTION_RAW_OFFLOAD
        if self._force_full_prefill or self._force_whole_prefill:
            return PREFILL_EXECUTION_FULL
        return PREFILL_EXECUTION_CHUNKED

    def prefill_batch_compatibility_key(self, seq):
        del seq
        return None

    def reset_prefill_execution_state(self, seq_id):
        del seq_id

    def complete_prefill_execution(self, seq):
        del seq

    def prefill_batched_tokens_margin(self):
        return 0

    def prompt_admission_budgets(self, waiting, engine_prefill_chunk_size):
        return {"slots": self._free_slots}

    def prompt_admission_free_slots(self):
        return self._free_slots

    def prompt_admission_costs(self, seq):
        return {"slots": int(seq.num_prompt_tokens - seq.prefix_cache_hit_len)}

    def prompt_admission_failure_action(self):
        return "raise"

    def on_prompt_admitted(self, seq, costs):
        return None

    def prompt_logical_reservation_cost(self, seq):
        return int(seq.num_prompt_tokens - seq.prefix_cache_hit_len)

    def refresh_prefix_cache_hit(self, seq):
        self.refresh_calls += 1
        seq.clear_prefix_cache_hit()
        if self.prefix_hit_len <= 0:
            return
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = self.prefix_hit_len
        seq.prefix_cache_hit_block_count = self.prefix_hit_blocks
        seq.prefix_cache_hit_last_block_id = b"test"
        seq.prefix_cache_block_size = 4
        seq.prefix_cache_method = ""

    def clear_prefix_cache_hit(self, seq):
        self.clear_calls += 1
        seq.clear_prefix_cache_hit()

    def free_slot_stats(self):
        return {"free_slots": int(self._free_slots), "step_free_slots": int(self._step_free_slots)}


class CacheManagerPolicyOracle(FakeMemoryOracle):
    def __init__(self, cache_manager, **kwargs):
        super().__init__(**kwargs)
        self.cache_manager = cache_manager

    def prefill_execution_mode(self, seq):
        return self.cache_manager.prefill_execution_mode(seq)

    def prefill_batch_compatibility_key(self, seq):
        return self.cache_manager.prefill_batch_compatibility_key(seq)

    def min_final_prefill_chunk_size(self, seq):
        return self.cache_manager.min_final_prefill_chunk_size(seq)

    def reset_prefill_execution_state(self, seq_id):
        self.cache_manager.reset_prefill_execution_state(seq_id)

    def complete_prefill_execution(self, seq):
        self.cache_manager.complete_prefill_execution(seq)


def make_scheduler(policy, *, method="", chunk=5, max_tokens=10, oracle=None):
    cfg = SimpleNamespace(
        max_num_seqs_in_batch=4,
        max_num_batched_tokens=max_tokens,
        max_decoding_seqs=16,
        engine_prefill_chunk_size=chunk,
        prefill_schedule_policy=policy,
        eos=-1,
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=4,
        observation_window_size=2,
        sparse_method=method,
    )
    if oracle is None:
        oracle = FakeMemoryOracle(
            execution_mode=(
                PREFILL_EXECUTION_FULL
                if policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH
                else PREFILL_EXECUTION_CHUNKED
            )
        )
    return Scheduler(cfg, oracle)


def make_sparse_controller_config():
    return SimpleNamespace(
        sparse_method="deltakv",
        obs_layer_ids=[0],
        full_attention_layers=[0],
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=8,
            num_attention_heads=1,
        ),
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float32",
    )


def test_private_page_tail_progress_does_not_spend_another_requests_pages():
    """Full shared pools can still fill owned tails; tails cannot fund new pages."""
    from sparseengine.engine.cache_manager.quantized import QuantizedCacheManager
    from sparseengine.engine.cache_manager.quantized_pages import QuantizedPagePool

    first, second = Sequence(list(range(32))), Sequence(list(range(64)))
    pool = QuantizedPagePool(2, 32, 128)
    pool.append(first.seq_id, 31)
    pool.append(second.seq_id, 1)
    first.num_prefilled_tokens, second.num_prefilled_tokens = 31, 1
    manager = object.__new__(QuantizedCacheManager)
    manager.page_pool, manager.page_size = pool, 32

    class PageOracle(FakeMemoryOracle):
        def prefill_private_slots_for(self, seq):
            return manager.prefill_private_slots_for(seq)

        def prefill_step_free_slots_for(self, seq):
            return manager.prefill_step_free_slots_for(seq)

        def prefill_step_reservation_cost(self, seq, scheduled_tokens):
            return manager.prefill_step_reservation_cost(seq, scheduled_tokens)

    scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED, method="kivi", chunk=32, max_tokens=64,
                               oracle=PageOracle(free_slots=0, execution_mode=PREFILL_EXECUTION_CHUNKED))
    scheduler.waiting.extend([first, second])
    scheduled, is_prefill, _ = scheduler.schedule()
    assert is_prefill and set(seq.seq_id for seq in scheduled) == {first.seq_id, second.seq_id}
    assert sum(seq.current_chunk_size for seq in scheduled) == 32
    for seq in scheduled:
        assert pool.append_cost(seq.seq_id, seq.current_chunk_size) == 0
        pool.append(seq.seq_id, seq.current_chunk_size)


def test_shared_prefix_admission_cost_is_charged_once_per_batch():
    class SharedPrefixOracle(FakeMemoryOracle):
        def __init__(self):
            super().__init__(free_slots=18, step_free_slots=18)
            self.admitted = []

        def prompt_admission_costs(self, seq):
            return {"slots": 17}

        def prompt_admission_shared_costs(self, seq):
            return {"slots": {b"shared-prefix": 16}}

        def prompt_logical_reservation_cost(self, seq):
            return 17

        def on_prompt_admitted(self, seq, costs):
            self.admitted.append((seq.seq_id, dict(costs)))

    oracle = SharedPrefixOracle()
    scheduler = make_scheduler(
        PREFILL_POLICY_ALL_CHUNKED,
        chunk=1,
        max_tokens=2,
        oracle=oracle,
    )
    seqs = [Sequence([1], SamplingParams(max_tokens=1)) for _ in range(2)]
    scheduler.waiting.extend(seqs)

    scheduled, is_prefill, preempted = scheduler.schedule()

    assert is_prefill and not preempted
    assert scheduled == seqs
    assert oracle.admitted == [(seq.seq_id, {"slots": 17}) for seq in seqs]


def test_private_tail_candidate_scan_is_linear_when_global_pool_is_full():
    class CountingTailOracle(FakeMemoryOracle):
        def __init__(self, tail_seq_id):
            super().__init__(free_slots=0, step_free_slots=0)
            self.tail_seq_id = tail_seq_id
            self.private_queries = 0

        def prefill_private_slots_for(self, seq):
            self.private_queries += 1
            return 1 if seq.seq_id == self.tail_seq_id else 0

        def prefill_step_free_slots_for(self, seq):
            return self.prefill_private_slots_for(seq)

    seqs = [Sequence([1, 2], SamplingParams(max_tokens=1)) for _ in range(128)]
    for seq in seqs:
        seq.num_prefilled_tokens = 1
    oracle = CountingTailOracle(seqs[-1].seq_id)
    scheduler = make_scheduler(
        PREFILL_POLICY_ALL_CHUNKED, chunk=1, max_tokens=1, oracle=oracle
    )
    scheduler.waiting.extend(seqs)

    scheduled, is_prefill, _ = scheduler.schedule()

    assert is_prefill and scheduled == [seqs[-1]]
    assert oracle.private_queries <= 2 * len(seqs)


def identity_runtime_layout(num_layers):
    return SimpleNamespace(
        kv_idx_to_layer_idx=tuple(range(num_layers)),
        kv_layer_index=lambda layer_idx: int(layer_idx),
        is_full_attention=lambda layer_idx: 0 <= int(layer_idx) < int(num_layers),
    )


def make_scheduler_with_oracle(
    policy,
    oracle,
    *,
    method="",
    chunk=5,
    max_tokens=10,
    prefix_cache_hit_refresher=None,
):
    cfg = SimpleNamespace(
        max_num_seqs_in_batch=4,
        max_num_batched_tokens=max_tokens,
        max_decoding_seqs=16,
        engine_prefill_chunk_size=chunk,
        prefill_schedule_policy=policy,
        eos=-1,
        sink_keep_tokens=1,
        recent_keep_tokens=1,
        decode_keep_tokens=4,
        observation_window_size=2,
        sparse_method=method,
    )
    return Scheduler(
        cfg,
        oracle,
        prefix_cache_hit_refresher=prefix_cache_hit_refresher,
    )


def seq_with_len(n):
    return Sequence(list(range(n)))


class PrefillPolicyRegistryTest(unittest.TestCase):
    def test_prefill_reuses_cache_manager_metadata_across_layers(self):
        context_lens = torch.tensor([4, 7], dtype=torch.int32)
        req_indices = torch.tensor([0, 1], dtype=torch.int32)
        batch_state = LayerBatchStates(
            context_lens=context_lens,
            max_context_len=7,
            req_indices=req_indices,
        )

        class SharedMetadataManager:
            device = torch.device("cpu")

            def get_layer_batch_states(self, layer_idx):
                del layer_idx
                return batch_state

        cfg = make_sparse_controller_config()
        cfg.sparse_method = ""
        cfg.obs_layer_ids = []
        cfg.full_attention_layers = [0, 1]
        cfg.hf_config.num_hidden_layers = 2
        controller = SparseController(cfg, SharedMetadataManager())

        controller.prepare_forward([], is_prefill=True)

        for layer_idx in range(2):
            state = controller.layer_batch_sparse_states[layer_idx]
            self.assertIs(state.context_lens, context_lens)
            self.assertIs(state.req_indices, req_indices)

    def test_scheduler_stops_on_any_request_eos_token(self):
        scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED)
        seq = Sequence([1, 2], SamplingParams(max_tokens=8, eos_token_ids=(10, 11)))
        seq.current_chunk_size = 2

        scheduler.postprocess([seq], [11], is_prefill=True)

        self.assertTrue(seq.is_finished)
        self.assertNotIn(seq, scheduler.decoding)

    def test_deltakv_topk_tiebreak_env_can_enable(self):
        with patch.dict(os.environ, {"SPARSEENGINE_DELTAKV_DETERMINISTIC_TOPK_TIEBREAK": "1"}, clear=True):
            controller = SparseController(make_sparse_controller_config(), SimpleNamespace())

        self.assertTrue(controller.runtime.dynamic_deltakv_topk_tiebreak)
        self.assertTrue(controller.sparse_config["dynamic_deltakv_topk_tiebreak"])

    def test_deltakv_topk_tiebreak_env_rejects_invalid_value(self):
        with patch.dict(os.environ, {"SPARSEENGINE_DELTAKV_DETERMINISTIC_TOPK_TIEBREAK": "maybe"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SPARSEENGINE_DELTAKV_DETERMINISTIC_TOPK_TIEBREAK"):
                SparseController(make_sparse_controller_config(), SimpleNamespace())

    def test_pyramidkv_decode_trigger_uses_shared_interval(self):
        cfg = make_sparse_controller_config()
        cfg.sparse_method = "pyramidkv"
        cfg.sink_keep_tokens = 64
        cfg.recent_keep_tokens = 512
        cfg.decode_keep_tokens = 4096
        cfg.decode_eviction_interval = 37
        controller = SparseController(cfg, SimpleNamespace())

        low_layer_budget = 64 + 68 + 512
        self.assertEqual(
            controller.runtime._snapkv_decode_trigger_len(low_layer_budget),
            low_layer_budget + 37,
        )

    def test_snapkv_decode_trigger_uses_shared_interval(self):
        cfg = make_sparse_controller_config()
        cfg.sparse_method = "snapkv"
        cfg.sink_keep_tokens = 64
        cfg.recent_keep_tokens = 512
        cfg.decode_keep_tokens = 4096
        cfg.decode_eviction_interval = 37
        controller = SparseController(cfg, SimpleNamespace())

        budget = 64 + 4096 + 512
        self.assertEqual(
            controller.runtime._snapkv_decode_trigger_len(budget),
            budget + 37,
        )

    def test_streamingllm_decode_eviction_batches_layer_compaction(self):
        class FakeStreamingManager:
            device = torch.device("cpu")

            def __init__(self):
                self.layer_calls = []

            def decode_kv_lens_for_layer(self, layer_idx, seqs):
                return [12 for _seq in seqs]

            def free_part_slots_batch_layers(self, layer_indices, seqs, keep_indices):
                self.layer_calls.append((list(layer_indices), list(seqs), keep_indices.clone()))

        cfg = make_sparse_controller_config()
        cfg.sparse_method = "streamingllm"
        cfg.hf_config.num_hidden_layers = 3
        cfg.sink_keep_tokens = 2
        cfg.recent_keep_tokens = 3
        manager = FakeStreamingManager()
        controller = SparseController(cfg, manager)

        seq_a = Sequence([1])
        seq_b = Sequence([2])
        seq_a.seq_id = 10
        seq_b.seq_id = 11
        for layer_idx in range(3):
            state = controller.layer_batch_sparse_states[layer_idx]
            state.context_lens = torch.tensor([12, 12], dtype=torch.int32)
            state.max_context_len = 12

        controller.runtime._streamingllm_decode_eviction([seq_a, seq_b])

        self.assertEqual(len(manager.layer_calls), 1)
        layer_indices, seqs, keep_indices = manager.layer_calls[0]
        self.assertEqual(layer_indices, [0, 1, 2])
        self.assertEqual([seq.seq_id for seq in seqs], [10, 11])
        self.assertEqual(tuple(keep_indices.shape), (3, 2, 5))
        self.assertEqual(keep_indices[0, 0].tolist(), [0, 1, 9, 10, 11])
        self.assertTrue(torch.equal(keep_indices[0], keep_indices[1]))

    def test_streamingllm_prefill_eviction_batches_layer_compaction(self):
        class FakeStreamingManager:
            device = torch.device("cpu")

            def __init__(self):
                self.layer_calls = []

            def free_part_slots_batch_layers(self, layer_indices, seqs, keep_indices):
                self.layer_calls.append((list(layer_indices), list(seqs), keep_indices.clone()))

        cfg = make_sparse_controller_config()
        cfg.sparse_method = "streamingllm"
        cfg.hf_config.num_hidden_layers = 2
        cfg.sink_keep_tokens = 2
        cfg.recent_keep_tokens = 3
        manager = FakeStreamingManager()
        controller = SparseController(cfg, manager)

        seq_a = Sequence(list(range(12)))
        seq_b = Sequence(list(range(12)))
        seq_a.seq_id = 20
        seq_b.seq_id = 21
        seq_a.num_prefilled_tokens = 8
        seq_b.num_prefilled_tokens = 8
        seq_a.current_chunk_size = 4
        seq_b.current_chunk_size = 4
        for layer_idx in range(2):
            state = controller.layer_batch_sparse_states[layer_idx]
            state.context_lens = torch.tensor([12, 12], dtype=torch.int32)
            state.max_context_len = 12

        controller.runtime._streamingllm_prefill_eviction([seq_a, seq_b])

        self.assertEqual(len(manager.layer_calls), 1)
        layer_indices, seqs, keep_indices = manager.layer_calls[0]
        self.assertEqual(layer_indices, [0, 1])
        self.assertEqual([seq.seq_id for seq in seqs], [20, 21])
        self.assertEqual(tuple(keep_indices.shape), (2, 2, 5))
        self.assertEqual(keep_indices[0, 0].tolist(), [0, 1, 9, 10, 11])

    def test_streamingllm_prefill_eviction_finalizes_only_final_sequences(self):
        class FakeStreamingManager:
            device = torch.device("cpu")

            def __init__(self):
                self.calls = []

            def free_part_slots_batch_layers(self, layer_indices, seqs, keep_indices):
                self.calls.append((list(layer_indices), list(seqs), keep_indices.clone()))

            def free_part_slots(self, layer_idx, seq, keep_indices):
                self.calls.append(([layer_idx], [seq], keep_indices[None, :].clone()))

        cfg = make_sparse_controller_config()
        cfg.sparse_method = "streamingllm"
        cfg.hf_config.num_hidden_layers = 1
        cfg.sink_keep_tokens = 2
        cfg.recent_keep_tokens = 3
        manager = FakeStreamingManager()
        controller = SparseController(cfg, manager)
        final_seq = seq_with_len(12)
        partial_seq = seq_with_len(16)
        final_seq.current_chunk_size = 4
        final_seq.num_prefilled_tokens = 8
        partial_seq.current_chunk_size = 4
        partial_seq.num_prefilled_tokens = 8
        controller.layer_batch_sparse_states[0].context_lens = torch.tensor(
            [12, 12], dtype=torch.int32
        )

        controller.runtime._streamingllm_prefill_eviction(
            [final_seq, partial_seq]
        )

        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(manager.calls[0][1], [final_seq])

    def test_snapkv_prefill_eviction_finalizes_only_final_sequences(self):
        class FakeSnapManager:
            device = torch.device("cpu")

            def __init__(self):
                self.popped = []
                self.freed = []

            def pop_prefill_attention_score(self, layer_idx, seq):
                self.popped.append((layer_idx, seq.seq_id))
                return torch.arange(8, dtype=torch.float32)

            def free_part_slots(self, layer_idx, seq, keep_indices):
                self.freed.append((layer_idx, seq.seq_id, keep_indices.clone()))

        cfg = make_sparse_controller_config()
        cfg.sparse_method = "snapkv"
        cfg.sink_keep_tokens = 1
        cfg.recent_keep_tokens = 1
        cfg.decode_keep_tokens = 3
        cfg.pool_kernel_size = 1
        cfg.snapkv_num_full_layers = 0
        cfg.pyramid_layer_ratios = None
        manager = FakeSnapManager()
        controller = SparseController(cfg, manager)
        final_seq = seq_with_len(8)
        partial_seq = seq_with_len(12)
        final_seq.current_chunk_size = 4
        final_seq.num_prefilled_tokens = 4
        partial_seq.current_chunk_size = 4
        partial_seq.num_prefilled_tokens = 4

        controller.runtime._snapkv_prefill_eviction([final_seq, partial_seq])

        self.assertEqual(manager.popped, [(0, final_seq.seq_id)])
        self.assertEqual(len(manager.freed), 1)
        self.assertEqual(manager.freed[0][1], final_seq.seq_id)

    def test_snapkv_prefill_score_reuses_cpu_metadata_without_tensor_item(self):
        from sparseengine.engine.cache_manager import (
            AttentionViewMeta,
            ExplicitKVPayload,
            PrefillComputeView,
        )
        from sparseengine.utils.context import get_context, reset_context, set_context

        manager = object.__new__(SnapKVCacheManager)
        manager.config = SimpleNamespace(
            sparse_method="snapkv",
            sparse_prefill_score_mode="probability",
            sparse_attn_score_dtype="float32",
            sink_keep_tokens=1,
            recent_keep_tokens=1,
        )
        manager._prefill_attn_score_accumulators = {}
        manager._prefill_context_lens_cpu_by_layer = {0: (6,), 1: (6,)}
        manager._prefill_score_metadata_cache = {}
        manager._prefill_step_score_buffers = {}

        seq = seq_with_len(6)
        seq.num_prefilled_tokens = 4
        seq.current_chunk_size = 2
        manager._prefill_score_rows = lambda layer_idx, seqs: [
            (0, seqs[0], 4, 6)
        ]
        view = PrefillComputeView(
            meta=AttentionViewMeta(
                active_slots=torch.arange(6, dtype=torch.int32)[None, :],
                req_indices=torch.tensor([0], dtype=torch.int32),
                context_lens=torch.tensor([6], dtype=torch.int32),
                max_context_len=6,
            ),
            payload=ExplicitKVPayload(
                k_cache=torch.empty((6, 1, 1)),
                v_cache=torch.empty((6, 1, 1)),
            ),
        )
        score_buffer_ids = []
        metadata_ids = []

        def fake_run_prefill_score(
            q,
            k_cache,
            step_score,
            meta,
            b_start_loc,
            prompt_cache_lens,
            max_query_len,
            score_starts,
            score_ends,
            **kwargs,
        ):
            del q, k_cache, meta, b_start_loc, max_query_len, kwargs
            score_buffer_ids.append(step_score.data_ptr())
            metadata_ids.append(
                (id(prompt_cache_lens), id(score_starts), id(score_ends))
            )
            step_score[0, :6] = torch.arange(6, dtype=torch.float32)

        reset_context()
        set_context(is_prefill=True, cache_manager=manager, seqs=[seq])
        try:
            with (
                patch.object(
                    manager,
                    "_run_prefill_score",
                    side_effect=fake_run_prefill_score,
                ),
                patch.object(
                    torch.Tensor,
                    "item",
                    side_effect=AssertionError("unexpected tensor.item()"),
                ),
            ):
                for layer_idx in (0, 1):
                    manager.collect_prefill_attention_score(
                        layer_idx,
                        torch.empty((2, 1, 1)),
                        view,
                        b_start_loc=torch.tensor([0], dtype=torch.int32),
                        chunk_lens=torch.tensor([2], dtype=torch.int32),
                    )
        finally:
            reset_context()

        self.assertEqual(score_buffer_ids[0], score_buffer_ids[1])
        self.assertEqual(metadata_ids[0], metadata_ids[1])
        self.assertEqual(
            manager._prefill_attn_score_accumulators[(0, seq.seq_id)].tolist(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )

    def test_pyramid_materialization_uses_cpu_context_lengths_once_per_layer(self):
        from sparseengine.utils.context import get_context, reset_context, set_context

        class FakePyramidManager:
            def __init__(self):
                self.cpu_len_calls = 0
                self.materialized = []

            def has_prefill_staging_view(self, layer_idx):
                return layer_idx == 0

            def requires_long_prefill_offload(self, seq):
                del seq
                return True

            def prefill_staging_context_lens_cpu(self, layer_idx):
                self.cpu_len_calls += 1
                self.asserted_layer = layer_idx
                return (7, 5)

            def get_prefill_staging_view(self, layer_idx):
                raise AssertionError(
                    f"unexpected CUDA staging-length read for layer={layer_idx}"
                )

            def materialize_prefill_staging_layer_batch(self, layer_idx, values):
                self.materialized.append((layer_idx, values))

        manager = FakePyramidManager()
        runtime = object.__new__(PyramidKVRuntime)
        runtime.sparse_method = "pyramidkv"
        runtime.cache_manager = manager
        runtime.device = torch.device("cpu")
        runtime.config = SimpleNamespace()
        runtime._is_kv_layer = lambda layer_idx: layer_idx == 0
        runtime._get_layer_budget = lambda layer_idx, is_prefill: None
        seq_a = seq_with_len(7)
        seq_b = seq_with_len(5)
        seq_a.current_chunk_size = 2
        seq_a.num_prefilled_tokens = 5
        seq_b.current_chunk_size = 2
        seq_b.num_prefilled_tokens = 3

        reset_context()
        set_context(is_prefill=True, cache_manager=manager, seqs=[seq_a, seq_b])
        try:
            runtime.on_attention_end(
                AttentionEndEvent(
                    layer_idx=0,
                    forward_context=get_context(),
                )
            )
        finally:
            reset_context()

        self.assertEqual(manager.cpu_len_calls, 1)
        self.assertEqual(len(manager.materialized), 1)
        self.assertEqual(
            [int(indices.numel()) for _, indices in manager.materialized[0][1]],
            [7, 5],
        )


class StandardCacheManagerAdmissionTest(unittest.TestCase):
    def test_prompt_admission_tracks_row_budget(self):
        manager = object.__new__(StandardCacheManager)
        manager._num_free_slots = 100
        manager.free_rows = deque([0, 1])

        budgets = manager.prompt_admission_budgets(deque(), engine_prefill_chunk_size=16)
        costs = manager.prompt_admission_costs(seq_with_len(10))

        self.assertEqual(budgets["slots"], 100)
        self.assertEqual(budgets["rows"], 2)
        self.assertEqual(costs["slots"], 10)
        self.assertEqual(costs["rows"], 1)


class PrefillPolicyConfigTest(unittest.TestCase):
    def test_pyramid_forces_decode_eviction_and_rejects_nonpositive_interval(self):
        from sparseengine.configs.sparse import _normalize_snapkv

        config = SimpleNamespace(
            sparse_method="pyramidkv", snapkv_decode_eviction=False,
            decode_eviction_interval=1024, observation_window_size=32,
            snapkv_num_full_layers=0,
        )
        _normalize_snapkv(config)
        self.assertTrue(config.snapkv_decode_eviction)
        config.decode_eviction_interval = 0
        with self.assertRaisesRegex(ValueError, "decode_eviction_interval"):
            _normalize_snapkv(config)

    def hf_config(self):
        return SimpleNamespace(
            model_type="qwen2",
            dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
        )

    def make_config(self, **kwargs):
        sparse_method = str(kwargs.get("sparse_method", ""))
        if sparse_method.startswith(("omnikv", "deltakv")):
            kwargs.setdefault("full_attention_layers", [0])
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            with patch("sparseengine.configs.runtime.AutoConfig.from_pretrained", return_value=self.hf_config()):
                return Config(model=str(model_dir), **kwargs)

    def test_prefill_token_limits_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "max_num_batched_tokens must be > 0"):
            self.make_config(sparse_method="vanilla", max_num_batched_tokens=0)
        with self.assertRaisesRegex(ValueError, "engine_prefill_chunk_size must be > 0"):
            self.make_config(sparse_method="vanilla", engine_prefill_chunk_size=0)

    def test_invalid_policy_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unsupported prefill_schedule_policy"):
            self.make_config(sparse_method="snapkv", prefill_schedule_policy="old_chunk_mode")

    def test_deltakv_sparse_decode_backend_explicit_custom_does_not_require_flash_attn(self):
        with patch("sparseengine.configs.delta._flash_attn_available", return_value=False):
            cfg = self.make_config(
                sparse_method="deltakv-less-memory",
                allow_missing_deltakv_path=True,
                deltakv_latent_quant_bits=0,
                deltakv_sparse_decode_backend="custom",
            )

        self.assertEqual(cfg.deltakv_sparse_decode_backend, "custom")

    def test_deltakv_sparse_decode_backend_explicit_fa2_requires_flash_attn(self):
        with patch("sparseengine.configs.delta._flash_attn_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "requires the flash_attn package"):
                self.make_config(
                    sparse_method="deltakv-less-memory",
                    allow_missing_deltakv_path=True,
                    deltakv_latent_quant_bits=0,
                    deltakv_sparse_decode_backend="fa2",
                )

    def test_deltakv_sparse_decode_backend_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "deltakv_sparse_decode_backend"):
            self.make_config(
                sparse_method="deltakv-less-memory",
                allow_missing_deltakv_path=True,
                deltakv_latent_quant_bits=0,
                deltakv_sparse_decode_backend="flash",
            )

    def test_deltakv_legacy_graph_method_does_not_enable_graph(self):
        cfg = self.make_config(
            sparse_method="deltakv-less-memory-cudagraph",
            decode_graph=False,
            allow_missing_deltakv_path=True,
            deltakv_latent_quant_bits=0,
        )
        self.assertEqual(cfg.sparse_method, "deltakv")
        self.assertFalse(cfg.decode_graph)

    def test_decode_cuda_graph_capture_sampling_requires_graph(self):
        with self.assertRaisesRegex(ValueError, "requires decode_graph"):
            self.make_config(
                sparse_method="omnikv",
                decode_graph=False,
                decode_graph_capture_sampling=True,
            )

    def test_decode_batch_defaults_to_prefill_batch_limit(self):
        cfg = self.make_config(
            sparse_method="omnikv",
            decode_graph=True,
            max_num_seqs_in_batch=8,
        )

        self.assertEqual(cfg.max_num_seqs_in_batch, 8)
        self.assertEqual(cfg.max_decoding_seqs, 8)
        self.assertIn(8, cfg.decode_graph_capture_sizes)

    def test_explicit_decode_batch_override_updates_resident_capacity(self):
        cfg = self.make_config(
            sparse_method="vanilla",
            max_num_seqs_in_batch=8,
            max_decoding_seqs=64,
        )

        self.assertEqual(cfg.max_num_seqs_in_batch, 8)
        self.assertEqual(cfg.max_decoding_seqs, 64)
        self.assertGreaterEqual(cfg.max_num_seqs_in_gpu, 64)

    def test_decode_cuda_graph_auto_capture_sizes_use_decode_limit(self):
        for max_num_seqs_in_batch, max_decoding_seqs in (
            (1, 64),
            (8, 64),
            (64, 16),
        ):
            with self.subTest(
                max_num_seqs_in_batch=max_num_seqs_in_batch,
                max_decoding_seqs=max_decoding_seqs,
            ):
                cfg = self.make_config(
                    sparse_method="omnikv",
                    decode_graph=True,
                    max_num_seqs_in_batch=max_num_seqs_in_batch,
                    max_decoding_seqs=max_decoding_seqs,
                )
                capture_sizes = cfg.decode_graph_capture_sizes
                self.assertEqual(capture_sizes, sorted(set(capture_sizes)))
                self.assertEqual(capture_sizes[-1], max_decoding_seqs)
                self.assertTrue(
                    all(0 < size <= max_decoding_seqs for size in capture_sizes)
                )
                self.assertTrue(cfg.decode_graph)

    def test_decode_cuda_graph_auto_capture_sizes_are_bounded_for_large_limits(self):
        for max_decoding_seqs in (64, 80, 128, 256, 1024):
            with self.subTest(max_decoding_seqs=max_decoding_seqs):
                sizes = _default_decode_cuda_graph_capture_sizes(max_decoding_seqs)
                self.assertLessEqual(len(sizes), 32)
                self.assertEqual(sizes[:8], list(range(1, 9)))
                self.assertEqual(sizes[-1], max_decoding_seqs)
                self.assertEqual(sizes, sorted(set(sizes)))

    def test_decode_static_batch_capacity_uses_reachable_padding_bucket(self):
        cases = (
            ([1, 2, 4, 8, 16, 32, 64], 64, 64),
            ([1, 4, 8, 32, 64], 32, 32),
        )
        for capture_sizes, max_decode, expected in cases:
            with self.subTest(
                capture_sizes=capture_sizes,
                max_decode=max_decode,
            ):
                self.assertEqual(
                    _resolve_decode_static_batch_capacity(
                        capture_sizes,
                        max_decoding_seqs=max_decode,
                    ),
                    expected,
                )

    def test_legacy_platform_aliases_are_not_config_fields(self):
        fields = Config.__dataclass_fields__
        for name in (
            "decode_cuda_graph",
            "decode_cuda_graph_capture_sizes",
            "decode_cuda_graph_capture_sampling",
            "device_memory_utilization",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, fields)

    def test_decode_graph_uses_canonical_fields(self):
        cfg = self.make_config(
            sparse_method="omnikv",
            decode_graph=True,
            decode_graph_capture_sizes="1,4",
            decode_graph_capture_sampling=True,
            max_decoding_seqs=4,
            gpu_memory_utilization=0.7,
        )
        self.assertTrue(cfg.decode_graph)
        self.assertTrue(cfg.decode_graph_capture_sampling)
        self.assertEqual(cfg.decode_graph_capture_sizes, [1, 4])
        self.assertEqual(cfg.gpu_memory_utilization, 0.7)

    def test_auto_capture_greedy_sampling_scope(self):
        runner = object.__new__(ModelRunner)
        seqs = [
            SimpleNamespace(temperature=0.0),
            SimpleNamespace(temperature=0.0),
        ]

        runner.config = SimpleNamespace(
            decode_graph_capture_sampling=False,
            tensor_parallel_size=1,
            enable_prefix_caching=False,
            sparse_method="",
        )
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))

        runner.config.sparse_method = "omnikv"
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))

        runner.config.sparse_method = "quest"
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))

        runner.config.sparse_method = ""
        seqs[0].temperature = 0.7
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))

        seqs[0].temperature = 0.0
        runner.config.enable_prefix_caching = True
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))

        runner.config.enable_prefix_caching = False
        runner.config.tensor_parallel_size = 2
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))

        runner.config.decode_graph_capture_sampling = True
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))
        runner.config.tensor_parallel_size = 1
        self.assertTrue(runner._auto_capture_greedy_sampling(seqs))

        seqs[0].temperature = 0.7
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))
        seqs[0].temperature = 0.0

        seqs[0].presence_penalty = 0.1
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))
        seqs[0].presence_penalty = 0.0
        seqs[0].repetition_penalty = 0.9
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))
        seqs[0].repetition_penalty = 1.0
        self.assertTrue(runner._auto_capture_greedy_sampling(seqs))

        seqs[0].should_publish_sample = False
        self.assertFalse(runner._auto_capture_greedy_sampling(seqs))

    def test_decode_cuda_graph_capture_sampling_rejects_penalties(self):
        runner = object.__new__(DecodeCudaGraphRunner)

        for seq in (
            SimpleNamespace(
                temperature=0.0,
                presence_penalty=0.1,
                repetition_penalty=1.0,
            ),
            SimpleNamespace(
                temperature=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.1,
            ),
        ):
            with self.subTest(seq=seq):
                with self.assertRaisesRegex(ValueError, "sampling penalties"):
                    runner.run([seq], capture_sampling=True)

    def test_recompute_rows_are_excluded_from_sampling(self):
        runner = object.__new__(ModelRunner)
        sampled_shapes = []

        def sampler(logits, temperatures, top_ps, top_ks, *, all_greedy, all_unfiltered, max_top_k):
            sampled_shapes.append(tuple(logits.shape))
            self.assertTrue(all_greedy)
            return logits.argmax(dim=-1)

        runner.sampler = sampler
        replay = SimpleNamespace(should_publish_sample=False, temperature=0.0)
        live = SimpleNamespace(should_publish_sample=True, temperature=0.0)
        logits = torch.tensor([[9.0, 1.0], [2.0, 8.0]])

        token_ids = runner._sample_model_outputs(logits, [replay, live])

        self.assertEqual(token_ids, [0, 1])
        self.assertEqual(sampled_shapes, [(1, 2)])

    def test_penalized_logits_drive_greedy_sampling_and_logprobs(self):
        runner = object.__new__(ModelRunner)
        runner.sampler = Sampler()
        seq = Sequence(
            [0],
            SamplingParams(
                temperature=0.0,
                max_tokens=3,
                presence_penalty=0.5,
                repetition_penalty=2.0,
                logprobs=1,
            ),
        )
        seq.append_token(2)
        raw_logits = torch.tensor([[4.0, 3.0, 3.5]])

        sampling_logits = runner._apply_sampling_penalties(raw_logits, [seq])
        token_ids = runner._sample_model_outputs(sampling_logits, [seq])
        token_logprobs, top_logprobs = runner._collect_logprobs(
            sampling_logits,
            token_ids,
            [seq],
        )

        torch.testing.assert_close(
            sampling_logits,
            torch.tensor([[2.0, 3.0, 1.25]]),
        )
        self.assertEqual(token_ids, [1])
        expected_logprob = torch.log_softmax(sampling_logits, dim=-1)[0, 1].item()
        self.assertAlmostEqual(token_logprobs[0], expected_logprob)
        self.assertEqual(list(top_logprobs[0]), [1])

    def test_decode_cuda_graph_capture_sizes_must_include_decode_limit(self):
        with self.assertRaisesRegex(ValueError, "cover max_decoding_seqs"):
            self.make_config(
                sparse_method="omnikv",
                decode_graph=True,
                max_num_seqs_in_batch=6,
                max_decoding_seqs=6,
                decode_graph_capture_sizes=[1, 2, 4],
            )

        with self.assertRaisesRegex(ValueError, "exact batch bucket"):
            self.make_config(
                sparse_method="omnikv",
                decode_graph=True,
                max_num_seqs_in_batch=4,
                max_decoding_seqs=6,
                decode_graph_capture_sizes=[1, 2, 4, 8],
            )

        cfg = self.make_config(
            sparse_method="omnikv",
            decode_graph=True,
            max_num_seqs_in_batch=4,
            max_decoding_seqs=64,
            decode_graph_capture_sizes=[1, 2, 4, 64],
        )
        self.assertEqual(cfg.decode_graph_capture_sizes, [1, 2, 4, 64])


class DecodeCudaGraphWarmupPolicyTest(unittest.TestCase):
    def make_config(self, method="deltakv", decode_graph=True):
        return SimpleNamespace(
            sparse_method=method,
            decode_graph=decode_graph,
            data_parallel_size=1,
        )

    def test_startup_batch_uses_distinct_prompts_and_requested_shapes(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=32),
        )
        prompts = []
        pending = 0

        def add_request(prompt, sampling_params):
            nonlocal pending
            prompts.append((list(prompt), sampling_params.max_tokens))
            pending += 1

        def step():
            nonlocal pending
            pending = 0

        engine.add_request = add_request
        engine.is_finished = lambda: pending == 0
        engine.step = step

        next_offset = engine._run_startup_batch(
            (8, 3, 1),
            SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True),
            4,
        )

        self.assertEqual(next_offset, 7)
        self.assertEqual([prompt[0] for prompt, _ in prompts], [4, 5, 6])
        self.assertEqual(
            [len(prompt) for prompt, _ in prompts],
            [8, 3, 1],
        )
        self.assertEqual([max_tokens for _, max_tokens in prompts], [2, 2, 2])

    def test_graph_warmup_propagates_failure(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=32),
        )
        pending = 0

        def add_request(_prompt, _sampling_params):
            nonlocal pending
            pending += 1

        engine.add_request = add_request
        engine.is_finished = lambda: pending == 0
        engine.step = lambda: (_ for _ in ()).throw(RuntimeError("warmup failed"))

        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            engine._run_startup_batch(
                (1,),
                SamplingParams(max_tokens=1, temperature=0.0),
                0,
            )

    def test_startup_batch_rejects_exhausted_dummy_token_range(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=4),
        )
        with self.assertRaisesRegex(ValueError, "distinct leading token"):
            engine._run_startup_batch(
                (1, 1),
                SamplingParams(max_tokens=1, temperature=0.0),
                3,
            )

    def test_moe_workspace_warmup_profiles_decode_and_maximum_mlp_shapes(self):
        config = self.make_config(method="omnikv")
        config.max_decoding_seqs = 24
        config.max_num_batched_tokens = 56_214
        config.mlp_chunk_size = 16_384
        config.model_spec = SimpleNamespace(num_experts_field="num_experts")

        self.assertEqual(
            _moe_workspace_warmup_token_counts(config),
            (24, 16_384),
        )

    def test_dense_model_skips_moe_workspace_warmup(self):
        config = self.make_config(method="vanilla")
        config.model_spec = SimpleNamespace(num_experts_field=None)

        self.assertEqual(_moe_workspace_warmup_token_counts(config), ())

    def test_engine_runs_each_moe_workspace_shape_after_regular_warmup(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            data_parallel_size=1,
            max_decoding_seqs=24,
            max_num_batched_tokens=56_214,
            mlp_chunk_size=16_384,
            model_spec=SimpleNamespace(num_experts_field="num_experts"),
        )
        calls = []
        engine.model_runner = SimpleNamespace(
            call=lambda method, *args: calls.append((method, *args))
        )

        engine._warmup_moe_workspaces()

        self.assertEqual(
            calls,
            [
                ("warmup_moe_workspace", 24),
                ("warmup_moe_workspace", 16_384),
            ],
        )

    def test_moe_workspace_oom_fails_startup(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            data_parallel_size=1,
            max_decoding_seqs=24,
            max_num_batched_tokens=56_214,
            mlp_chunk_size=16_384,
            model_spec=SimpleNamespace(num_experts_field="num_experts"),
        )

        def fail_on_workspace(_method, _num_tokens):
            raise torch.OutOfMemoryError("warmup workspace OOM")

        engine.model_runner = SimpleNamespace(call=fail_on_workspace)

        with self.assertRaisesRegex(torch.OutOfMemoryError, "warmup workspace OOM"):
            engine._warmup_moe_workspaces()


class DeltaKVLessMemoryCudaGraphReserveTest(unittest.TestCase):
    def make_manager(self, decode_graph=True):
        manager = object.__new__(DeltaKVLessMemoryCudaGraphCacheManager)
        manager.config = SimpleNamespace(decode_graph=decode_graph)
        return manager

    def test_regular_less_memory_has_no_graph_workspace_reserve(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)

        self.assertEqual(manager._extra_workspace_reserve_bytes(), 0)

    def test_non_graph_mode_does_not_reserve_capture_memory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                self.make_manager(decode_graph=False)._decode_cuda_graph_memory_reserve_bytes(),
                0,
            )

    def test_graph_mode_uses_profiled_global_reserve_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                self.make_manager(decode_graph=True)._decode_cuda_graph_memory_reserve_bytes(),
                0,
            )

    def test_explicit_graph_reserve_remains_available_for_experiments(self):
        with patch.dict(
            os.environ,
            {"SPARSEENGINE_DELTAKV_CUDAGRAPH_RESERVE_BYTES": "4096"},
            clear=True,
        ):
            self.assertEqual(
                self.make_manager(decode_graph=True)._decode_cuda_graph_memory_reserve_bytes(),
                4096,
            )

    def test_graph_reserve_env_must_be_non_negative_integer(self):
        manager = self.make_manager(decode_graph=True)

        for value in ("-1", "large"):
            with patch.dict(os.environ, {"SPARSEENGINE_DELTAKV_CUDAGRAPH_RESERVE_BYTES": value}, clear=True):
                with self.assertRaisesRegex(ValueError, "SPARSEENGINE_DELTAKV_CUDAGRAPH_RESERVE_BYTES"):
                    manager._decode_cuda_graph_memory_reserve_bytes()


class SchedulerPrefillPolicyTest(unittest.TestCase):
    def test_batched_prefix_refresh_preserves_eligibility_and_classification(self):
        oracle = FakeMemoryOracle(prefix_hit_len=4)
        scheduler = make_scheduler_with_oracle(PREFILL_POLICY_ALL_CHUNKED, oracle)
        fresh = seq_with_len(12)
        partial = seq_with_len(12)
        partial.num_prefilled_tokens = 2
        scheduler.add(fresh)
        scheduler.add(partial)
        batches = []

        def refresh(seqs):
            batches.append(list(seqs))
            for seq in seqs:
                oracle.refresh_prefix_cache_hit(seq)

        scheduler.prefix_cache_hits_refresher = refresh
        scheduler.prefix_cache_hit_refresher = lambda _seq: self.fail("individual RPC")
        scheduler._prefill_mode_order()
        self.assertEqual(batches, [[fresh]])
        self.assertEqual(fresh.prefix_cache_hit_len, 4)
        self.assertEqual(partial.num_prefilled_tokens, 2)
        scheduler.max_decoding_seqs = 0
        scheduler._prefill_mode_order()
        self.assertEqual(batches, [[fresh]])

    def test_batched_prefix_refresh_skips_fresh_prompts_during_recompute(self):
        scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED)
        fresh = seq_with_len(12)
        replay = seq_with_len(12)
        replay.append_token(1)
        replay.start_recompute_replay()
        scheduler.add(fresh)
        scheduler.add(replay)
        scheduler.prefix_cache_hits_refresher = lambda _seqs: self.fail(
            "fresh prompt must wait for recompute replay"
        )
        self.assertTrue(scheduler._prefill_mode_order())

    def test_engine_wires_batch_rpc_only_when_prefix_cache_enabled(self):
        from sparseengine.engine.llm_engine import LLMEngine

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                engine = object.__new__(LLMEngine)
                oracle = FakeMemoryOracle()
                engine.config = make_scheduler(PREFILL_POLICY_ALL_CHUNKED).config
                engine.config.enable_prefix_caching = enabled
                calls = []
                engine.model_runner = SimpleNamespace(
                    runtime_state=oracle,
                    call=lambda method, seqs: calls.append((method, seqs)),
                )
                scheduler = engine._create_scheduler()
                seqs = [seq_with_len(8), seq_with_len(12)]
                for seq in seqs:
                    scheduler.add(seq)
                scheduler._prefill_mode_order()
                engine._refresh_prefix_cache_hits([])
                self.assertEqual(calls, [("refresh_prefix_cache_hits", seqs)] if enabled else [])

    def test_batched_prefix_refresh_matches_individual_schedule(self):
        outcomes = []
        for batched in (False, True):
            oracle = FakeMemoryOracle(prefix_hit_len=4)
            scheduler = make_scheduler_with_oracle(PREFILL_POLICY_ALL_CHUNKED, oracle)
            if batched:
                scheduler.prefix_cache_hits_refresher = lambda seqs: [
                    oracle.refresh_prefix_cache_hit(seq) for seq in seqs
                ]
            seqs = [seq_with_len(n) for n in (12, 8, 15)]
            for seq in seqs:
                scheduler.add(seq)
            scheduled, is_prefill, _ = scheduler.schedule()
            outcomes.append((is_prefill, [
                (seqs.index(seq), seq.num_prefilled_tokens, seq.current_chunk_size)
                for seq in scheduled
            ]))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_pyramid_modes_use_attached_residual_at_boundary(self):
        pyramid = object.__new__(SnapKVCacheManager)
        pyramid.config = SimpleNamespace(
            sparse_method="pyramidkv",
            pyramid_layer_ratios=[1.0],
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=2,
            long_prefill_offload_threshold=5,
        )
        pyramid.pyramidkv_prefill_staging_kv_cache = torch.empty(
            (2, 32, 1, 1), dtype=torch.float32
        )

        for residual, expected in (
            (4, PREFILL_EXECUTION_FULL),
            (5, PREFILL_EXECUTION_FULL),
            (6, PREFILL_EXECUTION_RAW_OFFLOAD),
        ):
            with self.subTest(residual=residual):
                fresh = seq_with_len(residual)
                self.assertEqual(pyramid.prefill_execution_mode(fresh), expected)

                resumed = seq_with_len(10 + residual)
                resumed.num_prefilled_tokens = 10
                resumed.chain_status = "resumed"
                resumed.chain_reused_tokens = 10
                self.assertEqual(pyramid.prefill_execution_mode(resumed), expected)

    def test_deltakv_fresh_modes_use_residual_boundary(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=2,
            long_prefill_offload_threshold=5,
        )

        for residual, expected in (
            (4, PREFILL_EXECUTION_FULL),
            (5, PREFILL_EXECUTION_FULL),
            (6, PREFILL_EXECUTION_RAW_OFFLOAD),
        ):
            with self.subTest(residual=residual):
                self.assertEqual(
                    manager.prefill_execution_mode(seq_with_len(residual)),
                    expected,
                )

    def test_long_raw_offload_mode_stays_sticky_across_scheduler_steps(self):
        managers = []
        pyramid = object.__new__(SnapKVCacheManager)
        pyramid.config = SimpleNamespace(
            sparse_method="pyramidkv",
            pyramid_layer_ratios=[1.0],
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=2,
            long_prefill_offload_threshold=5,
            observation_window_size=0,
        )
        pyramid.pyramidkv_prefill_staging_kv_cache = torch.empty(
            (2, 8, 1, 1), dtype=torch.float32
        )
        managers.append(("pyramidkv", pyramid))

        deltakv = object.__new__(DeltaKVLessMemoryCacheManager)
        deltakv.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=2,
            long_prefill_offload_threshold=5,
        )
        managers.append(("deltakv", deltakv))

        for method, manager in managers:
            with self.subTest(method=method):
                scheduler = make_scheduler(
                    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
                    method=method,
                    chunk=2,
                    max_tokens=8,
                    oracle=CacheManagerPolicyOracle(manager),
                )
                seq = seq_with_len(8)
                scheduler.add(seq)
                modes = []
                chunks = []
                while scheduler.waiting:
                    scheduled, is_prefill, _ = scheduler.schedule()
                    self.assertTrue(is_prefill)
                    modes.append(scheduler.prefill_execution_mode_for_batch(scheduled))
                    chunks.append(int(seq.current_chunk_size))
                    scheduler.postprocess(scheduled, [99], is_prefill=True)

                self.assertEqual(modes, [PREFILL_EXECUTION_RAW_OFFLOAD] * 4)
                self.assertEqual(chunks, [2, 2, 2, 2])
                self.assertEqual(
                    getattr(manager, "_raw_offload_prefill_phases", {}),
                    {},
                )

    def test_deltakv_resumed_raw_offload_fails_before_row_mutation(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=2,
            long_prefill_offload_threshold=5,
        )
        manager.row_seq_lens = np.array([10], dtype=np.int32)
        manager.seq_id_to_row = {0: 0}
        seq = seq_with_len(16)
        seq.num_prefilled_tokens = 10
        seq.chain_status = "resumed"
        seq.chain_reused_tokens = 10

        with self.assertRaisesRegex(
            RuntimeError,
            "DeltaKV does not support attached-prefix prefill",
        ):
            manager.prefill_execution_mode(seq)

        self.assertEqual(int(manager.row_seq_lens[0]), 10)
        self.assertEqual(seq.num_prefilled_tokens, 10)

    def test_scheduler_refreshes_prefix_hit_before_execution_mode_classification(self):
        events = []

        class AttachAwareOracle(FakeMemoryOracle):
            def refresh_prefix_cache_hit(self, seq):
                events.append("refresh")
                super().refresh_prefix_cache_hit(seq)

            def prefill_execution_mode(self, seq):
                events.append(
                    ("mode", self.remaining_prefill_tokens(seq))
                )
                return (
                    PREFILL_EXECUTION_FULL
                    if self.remaining_prefill_tokens(seq) <= 5
                    else PREFILL_EXECUTION_RAW_OFFLOAD
                )

        oracle = AttachAwareOracle(prefix_hit_len=8)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            oracle,
            method="pyramidkv",
            chunk=2,
            max_tokens=12,
        )
        seq = seq_with_len(12)
        scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(seq.num_prefilled_tokens, 8)
        self.assertEqual(seq.current_chunk_size, 4)
        self.assertEqual(events[0], "refresh")
        self.assertTrue(all(event == "refresh" or event == ("mode", 4) for event in events))

    def test_full_mode_batches_complete_residuals_without_chunking(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="pyramidkv",
            chunk=2,
            max_tokens=10,
            oracle=FakeMemoryOracle(execution_mode=PREFILL_EXECUTION_FULL),
        )
        seq_a = seq_with_len(5)
        seq_b = seq_with_len(4)
        scheduler.add(seq_a)
        scheduler.add(seq_b)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq_a, seq_b])
        self.assertEqual([seq.current_chunk_size for seq in scheduled], [5, 4])

    def test_pyramidkv_separates_fresh_staged_and_resumed_resident_full_batches(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.config = SimpleNamespace(
            sparse_method="pyramidkv",
            pyramid_layer_ratios=[1.0],
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=2,
            long_prefill_offload_threshold=5,
            observation_window_size=0,
        )
        manager.pyramidkv_prefill_staging_num_slots = 16
        manager.pyramidkv_prefill_staging_kv_cache = torch.empty(
            (2, 16, 1, 1), dtype=torch.float32
        )
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="pyramidkv",
            chunk=2,
            max_tokens=10,
            oracle=CacheManagerPolicyOracle(manager),
        )
        fresh = seq_with_len(4)
        resumed = seq_with_len(14)
        resumed.num_prefilled_tokens = 10
        resumed.chain_status = "resumed"
        resumed.chain_reused_tokens = 10
        scheduler.add(fresh)
        scheduler.add(resumed)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [fresh])
        self.assertEqual(fresh.current_chunk_size, 4)
        self.assertIsNone(resumed.current_chunk_size)
        self.assertTrue(
            manager._should_use_pyramidkv_full_prefill_staging(scheduled)
        )

    def test_all_chunked_batches_sparse_mixed_lengths(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_ALL_CHUNKED,
            method="quest",
        )
        long_seq = seq_with_len(20)
        short_seq = seq_with_len(4)
        scheduler.add(long_seq)
        scheduler.add(short_seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [long_seq, short_seq])
        self.assertEqual(long_seq.current_chunk_size, 5)
        self.assertEqual(short_seq.current_chunk_size, 4)

    def test_vanilla_decode_batches_across_sparse_long_text_boundary(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_ALL_CHUNKED,
            method="",
        )
        short_seq = seq_with_len(4)
        long_seq = seq_with_len(20)
        short_seq.num_prefilled_tokens = short_seq.num_prompt_tokens
        long_seq.num_prefilled_tokens = long_seq.num_prompt_tokens
        scheduler.decoding.extend((short_seq, long_seq))

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [short_seq, long_seq])

    def test_h2o_decode_does_not_starve_long_rows_behind_short_rows(self):
        # H2O has one physical-row attention topology. The old generic sparse
        # partition ran only the short row, even with room for every request.
        for affinity in (0, 2):
            with self.subTest(affinity=affinity):
                scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED, method="h2o")
                scheduler.favor_min_decoding_seqs = affinity
                scheduler.phase = "decode"
                seqs = [seq_with_len(length) for length in (2, 4, 20)]
                for seq in seqs:
                    seq.num_prefilled_tokens = seq.num_prompt_tokens
                scheduler.decoding.extend(seqs)

                scheduled, is_prefill, preempted = scheduler.schedule()

                self.assertFalse(is_prefill)
                self.assertEqual(scheduled, seqs)
                self.assertEqual(preempted, [])
    def test_decode_batch_limit_is_independent_from_prefill_batch_limit(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_ALL_CHUNKED,
            method="",
        )
        seqs = [seq_with_len(4) for _ in range(6)]
        for seq in seqs:
            seq.num_prefilled_tokens = seq.num_prompt_tokens
        scheduler.decoding.extend(seqs)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, seqs)

    def test_prefill_batch_still_uses_prefill_sequence_limit(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_ALL_CHUNKED,
            method="",
            max_tokens=100,
        )
        seqs = [seq_with_len(4) for _ in range(6)]
        for seq in seqs:
            scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, seqs[:4])

    def test_sparse_decode_mixes_lengths_without_starvation(self):
        # Short arrivals must not exclude existing long requests. Repeat after
        # the short row crosses the former budget threshold and is replaced.
        for method in ("quest", "omnikv", "deltakv", "pyramidkv", "streamingllm", "snapkv", "rkv", "skipkv", "kivi", "turboquant", "fp8_kv"):
            with self.subTest(method=method):
                scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED, method=method)
                seqs = [seq_with_len(n) for n in (20, 2, 6)]
                for seq in seqs:
                    seq.num_prefilled_tokens = seq.num_prompt_tokens
                scheduler.decoding.extend(seqs)
                for _ in range(8):
                    batch, is_prefill, preempted = scheduler.schedule()
                    self.assertFalse(is_prefill)
                    self.assertEqual(batch, seqs)
                    self.assertEqual(preempted, [])
                    scheduler.postprocess(batch, [0] * len(batch), False)

    def test_all_chunked_caps_each_prefill_by_chunk_size(self):
        scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED, method="", chunk=5, max_tokens=20)
        seq_a = seq_with_len(20)
        seq_b = seq_with_len(12)
        scheduler.add(seq_a)
        scheduler.add(seq_b)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertTrue(all(seq.current_chunk_size <= 5 for seq in scheduled))

    def test_all_chunked_uses_batch_cap_when_it_is_below_chunk_size(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_ALL_CHUNKED,
            method="",
            chunk=8,
            max_tokens=4,
        )
        seq = seq_with_len(8)
        scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(seq.current_chunk_size, 4)

    def test_all_chunked_reserves_minimum_final_prefill_chunk(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_ALL_CHUNKED,
            chunk=5,
            max_tokens=10,
            oracle=FakeMemoryOracle(min_final_prefill_chunk_size=2),
        )
        seq = seq_with_len(10)
        seq.num_prefilled_tokens = 4
        scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(seq.current_chunk_size, 4)
        self.assertEqual(seq.num_prompt_tokens - seq.num_prefilled_tokens - seq.current_chunk_size, 2)

    def test_all_chunked_preserves_snapkv_window_at_reported_boundaries(self):
        cases = (
            (16_966, 8_771, 16_934),
            (15_056, 6_845, 15_024),
        )
        for prompt_len, num_prefilled_tokens, expected_chunk_end in cases:
            with self.subTest(prompt_len=prompt_len):
                scheduler = make_scheduler(
                    PREFILL_POLICY_ALL_CHUNKED,
                    chunk=8_192,
                    max_tokens=65_536,
                    oracle=FakeMemoryOracle(min_final_prefill_chunk_size=32),
                )
                seq = seq_with_len(prompt_len)
                seq.num_prefilled_tokens = num_prefilled_tokens
                scheduler.add(seq)

                scheduled, is_prefill, _ = scheduler.schedule()

                self.assertTrue(is_prefill)
                self.assertEqual(scheduled, [seq])
                self.assertEqual(seq.num_prefilled_tokens + seq.current_chunk_size, expected_chunk_end)
                self.assertEqual(seq.num_prompt_tokens - expected_chunk_end, 32)

    def test_all_chunked_defers_final_window_when_batch_tail_is_too_small(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_ALL_CHUNKED,
            chunk=5,
            max_tokens=5,
            oracle=FakeMemoryOracle(min_final_prefill_chunk_size=2),
        )
        first = seq_with_len(10)
        first.num_prefilled_tokens = 6
        final_window = seq_with_len(10)
        final_window.num_prefilled_tokens = 8
        scheduler.add(first)
        scheduler.add(final_window)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [first])
        self.assertIsNone(final_window.current_chunk_size)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [final_window])
        self.assertEqual(final_window.current_chunk_size, 2)

    def test_all_chunked_reports_final_window_larger_than_effective_step_budget(self):
        # Config normally rejects this static mismatch; the scheduler still
        # needs a clear failure when a synthetic oracle or live capacity hits it.
        scheduler = make_scheduler(
            PREFILL_POLICY_ALL_CHUNKED,
            chunk=64,
            max_tokens=32,
            oracle=FakeMemoryOracle(min_final_prefill_chunk_size=64),
        )
        seq = seq_with_len(128)
        seq.num_prefilled_tokens = 64
        scheduler.add(seq)

        with self.assertRaisesRegex(
            RuntimeError,
            "Final prefill score window cannot fit in the effective step budget",
        ):
            scheduler.schedule()

    def test_snapkv_final_prefill_window_applies_when_final_physical_context_exceeds_budget(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.num_kv_layers = 2
        manager.config = SimpleNamespace(
            sparse_method="snapkv",
            observation_window_size=2,
            snapkv_num_full_layers=0,
            sink_keep_tokens=1,
            decode_keep_tokens=4,
            recent_keep_tokens=1,
            engine_prefill_chunk_size=5,
        )

        self.assertEqual(manager.min_final_prefill_chunk_size(seq_with_len(12)), 2)
        self.assertEqual(manager.min_final_prefill_chunk_size(seq_with_len(11)), 2)
        self.assertEqual(manager.min_final_prefill_chunk_size(seq_with_len(6)), 0)

        manager.config.snapkv_num_full_layers = 2
        self.assertEqual(manager.min_final_prefill_chunk_size(seq_with_len(12)), 0)

        manager.config.sparse_method = "pyramidkv"
        manager.config.snapkv_num_full_layers = 0
        manager.config.pyramid_layer_ratios = [1.0, 0.5]
        manager.runtime_layout = identity_runtime_layout(2)
        self.assertEqual(manager.min_final_prefill_chunk_size(seq_with_len(12)), 2)

    def test_pyramidkv_scheduler_preserves_final_score_window(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.num_kv_layers = 1
        manager.runtime_layout = identity_runtime_layout(1)
        manager.config = SimpleNamespace(
            sparse_method="pyramidkv",
            pyramid_layer_ratios=[1.0],
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            long_prefill_offload_threshold=5,
            engine_prefill_chunk_size=4,
            observation_window_size=3,
            snapkv_num_full_layers=0,
            sink_keep_tokens=1,
            decode_keep_tokens=4,
            recent_keep_tokens=1,
        )
        manager.pyramidkv_prefill_staging_kv_cache = torch.empty(
            (2, 10, 1, 1), dtype=torch.float32
        )
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="pyramidkv",
            chunk=4,
            max_tokens=10,
            oracle=CacheManagerPolicyOracle(manager),
        )
        seq = seq_with_len(10)
        scheduler.add(seq)
        chunks = []

        while scheduler.waiting:
            scheduled, is_prefill, _ = scheduler.schedule()
            self.assertTrue(is_prefill)
            self.assertEqual(scheduled, [seq])
            chunks.append(int(seq.current_chunk_size))
            scheduler.postprocess(scheduled, [99], is_prefill=True)

        self.assertEqual(chunks, [4, 3, 3])

    def test_pyramidkv_resumed_raw_offload_preserves_final_score_window(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.num_kv_layers = 1
        manager.runtime_layout = identity_runtime_layout(1)
        manager.config = SimpleNamespace(
            sparse_method="pyramidkv",
            pyramid_layer_ratios=[1.0],
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            long_prefill_offload_threshold=64,
            engine_prefill_chunk_size=32,
            observation_window_size=8,
            snapkv_num_full_layers=0,
            sink_keep_tokens=0,
            decode_keep_tokens=100,
            recent_keep_tokens=0,
        )
        manager.pyramidkv_prefill_staging_kv_cache = torch.empty(
            (2, 256, 1, 1), dtype=torch.float32
        )
        manager.chain_physical_kv_len = lambda layer_idx, seq_id: 30
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="pyramidkv",
            chunk=32,
            max_tokens=256,
            oracle=CacheManagerPolicyOracle(manager),
        )
        seq = seq_with_len(190)
        seq.chain_status = "resumed"
        seq.chain_reused_tokens = 90
        seq.num_prefilled_tokens = 90
        scheduler.add(seq)
        chunks = []

        while scheduler.waiting:
            scheduled, is_prefill, _ = scheduler.schedule()
            self.assertTrue(is_prefill)
            self.assertEqual(scheduled, [seq])
            chunks.append(int(seq.current_chunk_size))
            if seq.is_last_chunk_prefill:
                manager._pyramidkv_prefill_staging_context_lens_cpu_by_layer = {
                    0: (130,)
                }
                self.assertEqual(
                    manager._prefill_score_rows(0, [seq]),
                    [(0, seq, 122, 130)],
                )
            scheduler.postprocess(scheduled, [99], is_prefill=True)

        self.assertEqual(chunks, [32, 32, 28, 8])

    def test_snapkv_scoring_accepts_preserved_reported_final_windows(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(1)
        manager.config = SimpleNamespace(
            sparse_method="snapkv",
            observation_window_size=32,
            snapkv_num_full_layers=0,
            sink_keep_tokens=64,
            decode_keep_tokens=4_096,
            recent_keep_tokens=512,
        )

        for prompt_len in (16_966, 15_056):
            with self.subTest(prompt_len=prompt_len):
                seq = seq_with_len(prompt_len)
                seq.num_prefilled_tokens = prompt_len - 32
                seq.current_chunk_size = 32

                rows = manager._prefill_score_rows(0, [seq])

                self.assertEqual(rows, [(0, seq, prompt_len - 32, prompt_len)])

    def test_runtime_state_forwards_minimum_final_prefill_chunk(self):
        cache_manager = SimpleNamespace(min_final_prefill_chunk_size=lambda seq: seq.num_prompt_tokens)
        runtime_state = RuntimeState(SimpleNamespace(), cache_manager)

        self.assertEqual(runtime_state.min_final_prefill_chunk_size(seq_with_len(7)), 7)

    def test_long_bs1full_policy_chunks_long_as_single_offload_prefill(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv",
            chunk=5,
            max_tokens=10,
            oracle=FakeMemoryOracle(long_prefill_offload=True),
        )
        long_a = seq_with_len(20)
        long_b = seq_with_len(30)
        scheduler.add(long_a)
        scheduler.add(long_b)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [long_a])
        self.assertEqual(long_a.current_chunk_size, 5)
        self.assertEqual(long_b.current_chunk_size, None)

    def test_long_bs1full_policy_keeps_chunk_boundary_in_short_mode(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv",
            chunk=5,
            max_tokens=10,
            oracle=FakeMemoryOracle(long_prefill_offload=False),
        )
        boundary_seq = seq_with_len(5)
        scheduler.add(boundary_seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [boundary_seq])
        self.assertEqual(boundary_seq.current_chunk_size, 5)

    def test_long_bs1full_policy_batches_short_chunked_prefill(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv",
            chunk=5,
            max_tokens=10,
        )
        short_a = seq_with_len(5)
        short_b = seq_with_len(4)
        scheduler.add(short_a)
        scheduler.add(short_b)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [short_a, short_b])
        self.assertEqual(short_a.current_chunk_size, 5)
        self.assertEqual(short_b.current_chunk_size, 4)

    def test_deltakv_short_prefill_defers_when_step_free_cannot_fit_whole_prompt(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv-less-memory",
            chunk=8192,
            max_tokens=16384,
            oracle=FakeMemoryOracle(
                step_free_slots=32,
                force_whole_prefill=True,
                long_prefill_offload=False,
            ),
        )
        short_seq = seq_with_len(8192)
        decode_seq = seq_with_len(3)
        decode_seq.num_prefilled_tokens = decode_seq.num_prompt_tokens
        scheduler.add(short_seq)
        scheduler.decoding.append(decode_seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [decode_seq])
        self.assertEqual(short_seq.current_chunk_size, None)
        self.assertEqual(list(scheduler.waiting), [short_seq])

    def test_decode_only_schedule_skips_prompt_and_prefill_capacity_queries(self):
        class CountingOracle(FakeMemoryOracle):
            def __init__(self):
                super().__init__()
                self.prompt_calls = 0
                self.prefill_calls = 0

            def prompt_admission_free_slots(self):
                self.prompt_calls += 1
                return super().prompt_admission_free_slots()

            def prompt_admission_budgets(
                self,
                waiting,
                engine_prefill_chunk_size,
            ):
                self.prompt_calls += 1
                return super().prompt_admission_budgets(
                    waiting,
                    engine_prefill_chunk_size,
                )

            def prefill_step_free_slots(self):
                self.prefill_calls += 1
                return super().prefill_step_free_slots()

        oracle = CountingOracle()
        scheduler = make_scheduler(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle=oracle,
        )
        decode_seq = seq_with_len(8)
        decode_seq.num_prefilled_tokens = decode_seq.num_prompt_tokens
        scheduler.decoding.append(decode_seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [decode_seq])
        self.assertEqual(oracle.prompt_calls, 0)
        self.assertEqual(oracle.prefill_calls, 0)

    def test_snapkv_remaining_prefill_no_longer_reserves_score_window_chunk(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.config = SimpleNamespace(engine_prefill_chunk_size=5, observation_window_size=2, sparse_method="snapkv")
        seq = seq_with_len(20)

        self.assertEqual(SnapKVCacheManager.remaining_prefill_tokens(manager, seq), 20)

        seq.current_chunk_size = 5
        seq.num_prefilled_tokens = 5
        self.assertEqual(SnapKVCacheManager.remaining_prefill_tokens(manager, seq), 15)

    def test_snapkv_batch_free_part_slots_compacts_rows_and_releases_slots(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(1)
        manager.device = torch.device("cpu")
        manager._uniform_decode_metadata = True
        manager.buffer_req_to_token_slots_tensor = torch.empty((1, 2, 6), dtype=torch.int32)
        manager.seq_id_to_row = [{10: 0, 11: 1}]
        manager.row_seq_lens = [np.array([6, 6], dtype=np.int32)]
        manager.buffer_req_to_token_slots = [
            torch.tensor(
                [
                    [100, 101, 102, 103, 104, 105],
                    [200, 201, 202, 203, 204, 205],
                ],
                dtype=torch.int32,
            )
        ]
        manager.free_slots_stack = [torch.zeros((16,), dtype=torch.int32)]
        manager._num_free_slots = [0]

        seq_a = Sequence([1])
        seq_b = Sequence([2])
        seq_a.seq_id = 10
        seq_b.seq_id = 11

        manager.free_part_slots_batch(
            0,
            [seq_a, seq_b],
            torch.tensor([[5, 0, 2], [3, 1, 5]], dtype=torch.long),
        )

        self.assertFalse(manager._uniform_decode_metadata)
        self.assertEqual(manager.row_seq_lens[0].tolist(), [3, 3])
        self.assertEqual(manager.buffer_req_to_token_slots[0][0].tolist(), [100, 102, 105, 0, 0, 0])
        self.assertEqual(manager.buffer_req_to_token_slots[0][1].tolist(), [201, 203, 205, 0, 0, 0])
        self.assertEqual(manager._num_free_slots[0], 6)
        self.assertEqual(
            sorted(manager.free_slots_stack[0][:6].tolist()),
            [101, 103, 104, 200, 202, 204],
        )

    def test_snapkv_layer_batch_free_part_slots_compacts_rows_and_releases_slots(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(2)
        manager.device = torch.device("cpu")
        manager._uniform_decode_metadata = True
        manager.seq_id_to_row = [{10: 0, 11: 1}, {10: 0, 11: 1}]
        manager.row_seq_lens = [
            np.array([6, 6], dtype=np.int32),
            np.array([6, 6], dtype=np.int32),
        ]
        manager.buffer_req_to_token_slots_tensor = torch.tensor(
            [
                [
                    [100, 101, 102, 103, 104, 105],
                    [200, 201, 202, 203, 204, 205],
                ],
                [
                    [300, 301, 302, 303, 304, 305],
                    [400, 401, 402, 403, 404, 405],
                ],
            ],
            dtype=torch.int32,
        )
        manager.buffer_req_to_token_slots = [
            manager.buffer_req_to_token_slots_tensor[0],
            manager.buffer_req_to_token_slots_tensor[1],
        ]
        manager.free_slots_stack_tensor = torch.zeros((2, 16), dtype=torch.int32)
        manager.free_slots_stack = [
            manager.free_slots_stack_tensor[0],
            manager.free_slots_stack_tensor[1],
        ]
        manager._num_free_slots = [0, 0]

        seq_a = Sequence([1])
        seq_b = Sequence([2])
        seq_a.seq_id = 10
        seq_b.seq_id = 11

        manager.free_part_slots_batch_layers(
            [0, 1],
            [seq_a, seq_b],
            torch.tensor(
                [
                    [[5, 0, 2], [3, 1, 5]],
                    [[4, 0, 1], [2, 0, 5]],
                ],
                dtype=torch.long,
            ),
        )

        self.assertFalse(manager._uniform_decode_metadata)
        self.assertEqual(manager.row_seq_lens[0].tolist(), [3, 3])
        self.assertEqual(manager.row_seq_lens[1].tolist(), [3, 3])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[0, 0].tolist(), [100, 102, 105, 0, 0, 0])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[0, 1].tolist(), [201, 203, 205, 0, 0, 0])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[1, 0].tolist(), [300, 301, 304, 0, 0, 0])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[1, 1].tolist(), [400, 402, 405, 0, 0, 0])
        self.assertEqual(manager._num_free_slots, [6, 6])
        self.assertEqual(
            sorted(manager.free_slots_stack_tensor[0, :6].tolist()),
            [101, 103, 104, 200, 202, 204],
        )
        self.assertEqual(
            sorted(manager.free_slots_stack_tensor[1, :6].tolist()),
            [302, 303, 305, 401, 403, 404],
        )

    def test_snapkv_prefix_recent_layer_batch_compacts_contiguous_middle(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(2)
        manager.device = torch.device("cpu")
        manager._uniform_decode_metadata = True
        manager.seq_id_to_row = [{10: 0, 11: 1}, {10: 0, 11: 1}]
        manager.row_seq_lens = [
            np.array([8, 8], dtype=np.int32),
            np.array([8, 8], dtype=np.int32),
        ]
        manager.buffer_req_to_token_slots_tensor = torch.tensor(
            [
                [
                    [100, 101, 102, 103, 104, 105, 106, 107],
                    [200, 201, 202, 203, 204, 205, 206, 207],
                ],
                [
                    [300, 301, 302, 303, 304, 305, 306, 307],
                    [400, 401, 402, 403, 404, 405, 406, 407],
                ],
            ],
            dtype=torch.int32,
        )
        manager.buffer_req_to_token_slots = [
            manager.buffer_req_to_token_slots_tensor[0],
            manager.buffer_req_to_token_slots_tensor[1],
        ]
        manager.free_slots_stack_tensor = torch.zeros((2, 16), dtype=torch.int32)
        manager.free_slots_stack = [
            manager.free_slots_stack_tensor[0],
            manager.free_slots_stack_tensor[1],
        ]
        manager._num_free_slots = [0, 0]

        seq_a = Sequence([1])
        seq_b = Sequence([2])
        seq_a.seq_id = 10
        seq_b.seq_id = 11

        manager.free_prefix_recent_slots_batch_layers(
            [0, 1],
            [seq_a, seq_b],
            kv_len=8,
            sink_keep_tokens=2,
            recent_keep_tokens=3,
        )

        self.assertFalse(manager._uniform_decode_metadata)
        self.assertEqual(manager.row_seq_lens[0].tolist(), [5, 5])
        self.assertEqual(manager.row_seq_lens[1].tolist(), [5, 5])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[0, 0].tolist(), [100, 101, 105, 106, 107, 0, 0, 0])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[0, 1].tolist(), [200, 201, 205, 206, 207, 0, 0, 0])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[1, 0].tolist(), [300, 301, 305, 306, 307, 0, 0, 0])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[1, 1].tolist(), [400, 401, 405, 406, 407, 0, 0, 0])
        self.assertEqual(manager._num_free_slots, [6, 6])
        self.assertEqual(
            sorted(manager.free_slots_stack_tensor[0, :6].tolist()),
            [102, 103, 104, 202, 203, 204],
        )
        self.assertEqual(
            sorted(manager.free_slots_stack_tensor[1, :6].tolist()),
            [302, 303, 304, 402, 403, 404],
        )

    def test_snapkv_prefill_batch_all_layers_preserves_stack_order(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(2)
        manager.device = torch.device("cpu")
        manager.num_layers = 2
        manager.max_model_len = 8
        manager.seq_id_to_row = [{}, {}]
        manager.free_rows = [deque([0, 1]), deque([0, 1])]
        manager.row_seq_lens = [
            np.zeros((2,), dtype=np.int32),
            np.zeros((2,), dtype=np.int32),
        ]
        manager.free_slots_stack_tensor = torch.stack(
            [
                torch.arange(20, dtype=torch.int32),
                torch.arange(100, 120, dtype=torch.int32),
            ],
            dim=0,
        )
        manager.free_slots_stack = [
            manager.free_slots_stack_tensor[0],
            manager.free_slots_stack_tensor[1],
        ]
        manager._num_free_slots = [20, 20]
        manager.buffer_req_to_token_slots_tensor = torch.zeros((2, 2, 8), dtype=torch.int32)
        manager.buffer_req_to_token_slots = [
            manager.buffer_req_to_token_slots_tensor[0],
            manager.buffer_req_to_token_slots_tensor[1],
        ]

        seq_a = Sequence([1, 2])
        seq_b = Sequence([3, 4])
        seq_a.seq_id = 10
        seq_b.seq_id = 11
        seq_a.current_chunk_size = 2
        seq_b.current_chunk_size = 2
        layers_slot_mapping = torch.empty((2, 4), dtype=torch.int32)

        used_fast_path = manager._allocate_prefill_batch_same_size_all_layers(
            [seq_a, seq_b],
            layers_slot_mapping,
        )

        self.assertTrue(used_fast_path)
        self.assertEqual(manager._num_free_slots, [16, 16])
        self.assertEqual(manager.row_seq_lens[0].tolist(), [2, 2])
        self.assertEqual(manager.row_seq_lens[1].tolist(), [2, 2])
        self.assertEqual(layers_slot_mapping[0].tolist(), [18, 19, 16, 17])
        self.assertEqual(layers_slot_mapping[1].tolist(), [118, 119, 116, 117])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[0, 0, :2].tolist(), [18, 19])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[0, 1, :2].tolist(), [16, 17])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[1, 0, :2].tolist(), [118, 119])
        self.assertEqual(manager.buffer_req_to_token_slots_tensor[1, 1, :2].tolist(), [116, 117])

    def test_pyramidkv_resumed_full_prefill_appends_to_physical_row(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(1)
        manager.config = SimpleNamespace(sparse_method="pyramidkv")
        manager.device = torch.device("cpu")
        manager.num_layers = 1
        manager.max_model_len = 16
        manager.seq_id_to_row = [{10: 0}]
        manager.free_rows = [deque()]
        manager.row_seq_lens = [np.array([3], dtype=np.int32)]
        manager.buffer_req_to_token_slots_tensor = None
        manager.buffer_req_to_token_slots = [
            torch.tensor([[1, 2, 3] + [0] * 13], dtype=torch.int32)
        ]
        manager.free_slots_stack_tensor = None
        manager.free_slots_stack = [torch.arange(20, 40, dtype=torch.int32)]
        manager._num_free_slots = [20]
        manager.layer_batch_states = [SimpleNamespace()]
        manager._prefill_attn_score_accumulators = {}
        manager._pyramidkv_prefill_staging_active = False
        manager._pyramidkv_can_use_full_prefill_staging = lambda: True
        manager._should_use_pyramidkv_long_prefill_offload_staging = lambda seqs: False
        manager._should_use_pyramidkv_full_prefill_staging = lambda seqs: False

        seq = Sequence(list(range(10)), SamplingParams(max_tokens=1))
        seq.seq_id = 10
        seq.chain_status = "resumed"
        seq.chain_reused_tokens = 8
        seq.num_prefilled_tokens = 8
        seq.current_chunk_size = 2

        input_ids, positions, _ = SnapKVCacheManager._prepare_prefill(
            manager,
            [seq],
        )

        self.assertEqual(manager.row_seq_lens[0].tolist(), [5])
        self.assertEqual(manager.layer_batch_states[0].context_lens.tolist(), [5])
        self.assertEqual(input_ids.tolist(), [8, 9])
        self.assertEqual(positions.tolist(), [8, 9])

    def test_pyramidkv_batch_materialize_updates_rows_and_kv(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(1)
        manager.config = SimpleNamespace(sparse_method="pyramidkv")
        manager.device = torch.device("cpu")
        manager.num_layers = 1
        manager.num_kv_layers = 1
        manager.max_model_len = 8
        manager._pyramidkv_prefill_staging_active = True
        manager._pyramidkv_prefill_staging_materialized_layers = set()
        manager.seq_id_to_row = [{10: 0, 11: 1}]
        manager.row_seq_lens = [np.zeros((2,), dtype=np.int32)]
        manager.free_slots_stack = [torch.arange(8, dtype=torch.int32)]
        manager._num_free_slots = [8]
        manager.buffer_req_to_token_slots = [torch.zeros((2, 8), dtype=torch.int32)]

        k_cache = torch.zeros((8, 1, 1), dtype=torch.float32)
        v_cache = torch.zeros((8, 1, 1), dtype=torch.float32)
        manager.kv_cache = [(k_cache, v_cache)]
        k_stage = torch.arange(8, dtype=torch.float32).view(8, 1, 1) + 10
        v_stage = torch.arange(8, dtype=torch.float32).view(8, 1, 1) + 20
        manager.pyramidkv_prefill_staging_kv_cache = (k_stage, v_stage)

        seq_a = Sequence([1])
        seq_b = Sequence([2])
        seq_a.seq_id = 10
        seq_b.seq_id = 11
        manager._pyramidkv_prefill_staging_seq_offsets = {10: 0, 11: 4}

        manager.materialize_prefill_staging_layer_batch(
            0,
            [
                (seq_a, torch.tensor([0, 2], dtype=torch.long)),
                (seq_b, torch.tensor([1, 3], dtype=torch.long)),
            ],
        )

        self.assertFalse(manager._pyramidkv_prefill_staging_active)
        self.assertEqual(manager.row_seq_lens[0].tolist(), [2, 2])
        self.assertEqual(manager.buffer_req_to_token_slots[0][0, :2].tolist(), [6, 7])
        self.assertEqual(manager.buffer_req_to_token_slots[0][1, :2].tolist(), [4, 5])
        self.assertEqual(k_cache[[6, 7, 4, 5], 0, 0].tolist(), [10.0, 12.0, 15.0, 17.0])
        self.assertEqual(v_cache[[6, 7, 4, 5], 0, 0].tolist(), [20.0, 22.0, 25.0, 27.0])

    def test_snapkv_eager_decode_falls_back_when_layer_metadata_diverges(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(2)
        manager.device = torch.device("cpu")
        manager.num_layers = 2
        manager.num_kv_layers = 2
        manager.max_model_len = 8
        manager._uniform_decode_metadata = False
        manager._decode_static_state_binding_key = None
        manager.seq_id_to_row = [{10: 0}, {10: 1}]
        manager.free_rows = [deque([1]), deque([0])]
        manager.row_seq_lens = [
            np.array([2, 0], dtype=np.int32),
            np.array([0, 3], dtype=np.int32),
        ]
        manager.free_slots_stack_tensor = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32
        )
        manager.free_slots_stack = [
            manager.free_slots_stack_tensor[0],
            manager.free_slots_stack_tensor[1],
        ]
        manager._num_free_slots = [4, 3]
        manager.buffer_req_to_token_slots_tensor = torch.zeros((2, 2, 8), dtype=torch.int32)
        manager.buffer_req_to_token_slots = [
            manager.buffer_req_to_token_slots_tensor[0],
            manager.buffer_req_to_token_slots_tensor[1],
        ]
        manager.layer_batch_states = [SimpleNamespace(), SimpleNamespace()]

        cap = 1
        manager._decode_buf_capacity = cap
        manager._pinned_input_ids = torch.empty(cap, dtype=torch.int64)
        manager._pinned_positions = torch.empty(cap, dtype=torch.int64)
        manager._cuda_input_ids = torch.empty(cap, dtype=torch.int64)
        manager._cuda_positions = torch.empty(cap, dtype=torch.int64)
        manager._pinned_layers_context_lens = torch.empty((2, cap), dtype=torch.int32)
        manager._cuda_layers_context_lens = torch.empty((2, cap), dtype=torch.int32)
        manager._cuda_layers_slot_mapping = torch.empty((2, cap), dtype=torch.int32)
        manager._pinned_layers_req_indices = torch.empty((2, cap), dtype=torch.int32)
        manager._cuda_layers_req_indices = torch.empty((2, cap), dtype=torch.int32)
        manager._static_rows_gpu = torch.empty(cap, dtype=torch.long)
        manager._static_cols_gpu = torch.empty(cap, dtype=torch.long)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=2))
        seq.seq_id = 10
        seq.num_prefilled_tokens = seq.num_prompt_tokens
        seq.append_token(4)

        manager._prepare_decode([seq])

        self.assertEqual(manager._num_free_slots, [3, 2])
        self.assertEqual(manager.row_seq_lens[0].tolist(), [3, 0])
        self.assertEqual(manager.row_seq_lens[1].tolist(), [0, 4])
        self.assertEqual(manager.buffer_req_to_token_slots[0][0, 2].item(), 13)
        self.assertEqual(manager.buffer_req_to_token_slots[1][1, 3].item(), 22)
        self.assertEqual(manager.layer_batch_states[0].slot_mapping.tolist(), [13])
        self.assertEqual(manager.layer_batch_states[1].slot_mapping.tolist(), [22])
        self.assertEqual(manager.layer_batch_states[0].context_lens.tolist(), [3])
        self.assertEqual(manager.layer_batch_states[1].context_lens.tolist(), [4])

    def test_pyramidkv_long_prefill_offload_candidate_uses_chunked_staging(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.config = SimpleNamespace(
            sparse_method="pyramidkv",
            pyramid_layer_ratios=[1.0],
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=1024,
            long_prefill_offload_threshold=1024,
            max_num_batched_tokens=2048,
        )
        manager.pyramidkv_prefill_staging_num_slots = 4096
        manager.pyramidkv_prefill_staging_kv_cache = torch.empty((2, 4096, 1, 1), dtype=torch.float32)

        seq = seq_with_len(2048)
        seq.current_chunk_size = 1024
        self.assertTrue(SnapKVCacheManager.requires_long_prefill_offload(manager, seq))
        self.assertFalse(SnapKVCacheManager.requires_full_prefill_step(manager, seq))
        self.assertFalse(SnapKVCacheManager._should_use_pyramidkv_full_prefill_staging(manager, [seq]))
        self.assertTrue(SnapKVCacheManager._should_use_pyramidkv_long_prefill_offload_staging(manager, [seq]))

    def test_long_prefill_offload_threshold_is_independent_from_chunk_size(self):
        pyramid = object.__new__(SnapKVCacheManager)
        pyramid.config = SimpleNamespace(
            sparse_method="pyramidkv",
            pyramid_layer_ratios=[1.0],
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=4096,
            long_prefill_offload_threshold=8192,
            max_num_batched_tokens=128000,
        )
        pyramid.pyramidkv_prefill_staging_num_slots = 128356
        pyramid.pyramidkv_prefill_staging_kv_cache = torch.empty((2, 1, 1, 1), dtype=torch.float32)

        deltakv = object.__new__(DeltaKVCacheManager)
        deltakv.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=4096,
            long_prefill_offload_threshold=8192,
            max_num_batched_tokens=128000,
        )

        boundary_seq = seq_with_len(8192)
        long_seq = seq_with_len(8193)
        self.assertFalse(SnapKVCacheManager.requires_long_prefill_offload(pyramid, boundary_seq))
        self.assertFalse(DeltaKVCacheManager.requires_long_prefill_offload(deltakv, boundary_seq))
        self.assertTrue(SnapKVCacheManager.requires_long_prefill_offload(pyramid, long_seq))
        self.assertFalse(SnapKVCacheManager.requires_full_prefill_step(pyramid, long_seq))
        self.assertTrue(DeltaKVCacheManager.requires_long_prefill_offload(deltakv, long_seq))

    def test_pyramidkv_long_prefill_offload_restores_prefix_to_staging(self):
        from sparseengine.engine.cache_manager.raw_kv_offload import RawKVOffloadBuffer

        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(1)
        manager.config = SimpleNamespace(sparse_method="pyramidkv")
        manager.device = torch.device("cpu")
        manager.num_layers = 1
        manager.seq_id_to_row = [{10: 0}]
        manager.raw_kv_offload_buffer = RawKVOffloadBuffer(pin_memory=False, mode="chunked")
        manager.pyramidkv_prefill_staging_kv_cache = torch.zeros((2, 4, 1, 1), dtype=torch.float32)
        manager._pyramidkv_prefill_staging_active = True
        manager._pyramidkv_long_prefill_offload_step_active = True
        manager._pyramidkv_long_prefill_offload_seq_id = 10
        manager._pyramidkv_long_prefill_offload_start = 0
        manager._pyramidkv_long_prefill_offload_end = 2
        manager._pyramidkv_long_prefill_offload_total_len = 4
        manager._pyramidkv_long_prefill_offload_is_last_chunk = False

        manager.pyramidkv_prefill_staging_kv_cache[0, :2, 0, 0] = torch.tensor([1.0, 2.0])
        manager.pyramidkv_prefill_staging_kv_cache[1, :2, 0, 0] = torch.tensor([11.0, 12.0])
        SnapKVCacheManager._offload_pyramidkv_long_prefill_layer(manager, 0)

        manager.pyramidkv_prefill_staging_kv_cache.zero_()
        manager._pyramidkv_long_prefill_offload_start = 2
        manager._pyramidkv_long_prefill_offload_end = 4
        SnapKVCacheManager.before_prefill_layer_attention(manager, 0, None)

        self.assertEqual(manager.pyramidkv_prefill_staging_kv_cache[0, :2, 0, 0].tolist(), [1.0, 2.0])
        self.assertEqual(manager.pyramidkv_prefill_staging_kv_cache[1, :2, 0, 0].tolist(), [11.0, 12.0])

    def test_pyramidkv_resumed_offload_keeps_resident_prefix_and_offloads_only_residual(self):
        from sparseengine.engine.cache_manager.raw_kv_offload import RawKVOffloadBuffer

        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(1)
        manager.config = SimpleNamespace(sparse_method="pyramidkv")
        manager.device = torch.device("cpu")
        manager.num_layers = 1
        manager.seq_id_to_row = [{10: 0}]
        manager.row_seq_lens = [np.array([2], dtype=np.int32)]
        manager.buffer_req_to_token_slots = [
            torch.tensor([[4, 5, 0, 0, 0, 0]], dtype=torch.int32)
        ]
        k_cache = torch.zeros((8, 1, 1), dtype=torch.float32)
        v_cache = torch.zeros((8, 1, 1), dtype=torch.float32)
        k_cache[4:6, 0, 0] = torch.tensor([101.0, 102.0])
        v_cache[4:6, 0, 0] = torch.tensor([201.0, 202.0])
        manager.kv_cache = [(k_cache, v_cache)]
        manager.raw_kv_offload_buffer = RawKVOffloadBuffer(
            pin_memory=False,
            mode="chunked",
        )
        manager.pyramidkv_prefill_staging_kv_cache = torch.zeros(
            (2, 8, 1, 1), dtype=torch.float32
        )
        manager._pyramidkv_prefill_staging_active = True
        manager._pyramidkv_long_prefill_offload_step_active = True
        manager._pyramidkv_long_prefill_offload_seq_id = 10
        manager._pyramidkv_long_prefill_offload_residual_start = 10
        manager._pyramidkv_long_prefill_offload_resident_prefix_lens = {0: 2}
        manager._pyramidkv_long_prefill_offload_start = 10
        manager._pyramidkv_long_prefill_offload_end = 12
        manager._pyramidkv_long_prefill_offload_total_len = 16
        manager._pyramidkv_long_prefill_offload_is_last_chunk = False
        manager.pyramidkv_prefill_staging_kv_cache[0, 2:4, 0, 0] = torch.tensor(
            [1.0, 2.0]
        )
        manager.pyramidkv_prefill_staging_kv_cache[1, 2:4, 0, 0] = torch.tensor(
            [11.0, 12.0]
        )

        SnapKVCacheManager._offload_pyramidkv_long_prefill_layer(manager, 0)
        entry = manager.raw_kv_offload_buffer._entries[
            (0, 0, "pyramidkv_post_rope")
        ]
        self.assertEqual(entry.capacity, 6)
        self.assertEqual(entry.filled_until, 2)

        manager.pyramidkv_prefill_staging_kv_cache.zero_()
        manager._pyramidkv_long_prefill_offload_start = 12
        manager._pyramidkv_long_prefill_offload_end = 14
        SnapKVCacheManager.before_prefill_layer_attention(manager, 0, None)

        self.assertEqual(
            manager.pyramidkv_prefill_staging_kv_cache[0, :4, 0, 0].tolist(),
            [101.0, 102.0, 1.0, 2.0],
        )
        self.assertEqual(
            manager.pyramidkv_prefill_staging_kv_cache[1, :4, 0, 0].tolist(),
            [201.0, 202.0, 11.0, 12.0],
        )

    def test_pyramidkv_long_prefill_offload_uses_staged_prefetch(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.config = SimpleNamespace(sparse_method="pyramidkv")
        manager.device = torch.device("cpu")
        manager.seq_id_to_row = [{10: 0}]
        manager.pyramidkv_prefill_staging_kv_cache = torch.zeros((2, 4, 1, 1), dtype=torch.float32)
        manager._pyramidkv_prefill_staging_active = True
        manager._pyramidkv_long_prefill_offload_step_active = True
        manager._pyramidkv_long_prefill_offload_seq_id = 10
        manager._pyramidkv_long_prefill_offload_start = 2
        manager.has_prefill_staging_view = lambda layer_idx: True
        manager._pyramidkv_consume_long_prefill_offload_staged_prefetch = lambda **kwargs: True
        manager.raw_kv_offload_buffer = SimpleNamespace(
            copy_prefix_to=lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected synchronous restore"))
        )

        SnapKVCacheManager.before_prefill_layer_attention(manager, 0, None)

    def test_pyramidkv_long_prefill_offload_prefetch_waits_for_current_stream_before_staging_write(self):
        manager = object.__new__(SnapKVCacheManager)
        manager.runtime_layout = identity_runtime_layout(2)
        manager.device = "cuda:0"
        manager.num_layers = 2
        manager._pyramidkv_long_prefill_offload_prefetch_stream = None
        manager._pyramidkv_long_prefill_offload_prefetch_states = {}
        manager._pyramidkv_long_prefill_offload_seq_id = 10
        manager.seq_id_to_row = [{10: 0}, {10: 5}]
        manager._pyramidkv_long_prefill_offload_prefetch_enabled = lambda: True
        manager.pyramidkv_prefill_staging_kv_cache = torch.empty((2, 4, 1, 1))

        calls = []
        created_events = []

        class FakeEvent:
            def __init__(self):
                self.name = f"event{len(created_events)}"
                created_events.append(self)

            def record(self, stream=None):
                calls.append(("record", self.name, getattr(stream, "name", None)))

        class FakeStream:
            def __init__(self, device=None, *, name="prefetch"):
                self.device = device
                self.name = name

            def wait_event(self, event):
                calls.append(("wait", self.name, event.name))

        class FakeStreamContext:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                calls.append(("enter", self.stream.name))
                return self.stream

            def __exit__(self, exc_type, exc, tb):
                calls.append(("exit", self.stream.name))
                return False

        current_stream = [FakeStream(device=manager.device, name="current")]

        def fake_record_event(event, device=None):
            del device
            event.record(current_stream[0])

        class FakeRuntimeStreamContext(FakeStreamContext):
            def __enter__(self):
                current_stream[0] = self.stream
                return super().__enter__()

            def __exit__(self, exc_type, exc, tb):
                try:
                    return super().__exit__(exc_type, exc, tb)
                finally:
                    current_stream[0] = FakeStream(device=manager.device, name="current")

        def fake_copy_prefix_to(**kwargs):
            calls.append(("copy_prefix_to", int(kwargs["layer_idx"]), int(kwargs["row_idx"]), int(kwargs["end"])))

        manager.raw_kv_offload_buffer = SimpleNamespace(copy_prefix_to=fake_copy_prefix_to)

        with (
            patch(
                "sparseengine.engine.cache_manager.methods.snapkv.device_runtime.new_event",
                lambda device=None: FakeEvent(),
            ),
            patch(
                "sparseengine.engine.cache_manager.methods.snapkv.device_runtime.new_stream",
                lambda device=None: FakeStream(device=device),
            ),
            patch("sparseengine.engine.cache_manager.methods.snapkv.device_runtime.record_event", fake_record_event),
            patch(
                "sparseengine.engine.cache_manager.methods.snapkv.device_runtime.stream_context",
                lambda stream: FakeRuntimeStreamContext(stream),
            ),
            patch(
                "sparseengine.engine.cache_manager.methods.snapkv.device_runtime.stream_wait_event",
                lambda stream, event: stream.wait_event(event),
            ),
        ):
            SnapKVCacheManager._pyramidkv_schedule_next_long_prefill_offload_prefetch(
                manager,
                layer_idx=0,
                end=2,
            )

        self.assertEqual(
            calls,
            [
                ("record", "event0", "current"),
                ("enter", "prefetch"),
                ("wait", "prefetch", "event0"),
                ("copy_prefix_to", 1, 5, 2),
                ("record", "event1", "prefetch"),
                ("exit", "prefetch"),
            ],
        )
        key = (1, 5, "pyramidkv_post_rope", 2)
        state = manager._pyramidkv_long_prefill_offload_prefetch_states[key]
        self.assertIs(state["staging_available_event"], created_events[0])
        self.assertIs(state["event"], created_events[1])

    def test_deltakv_short_prefill_fails_fast_when_no_work_can_free_slots(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv",
            chunk=8192,
            max_tokens=16384,
            oracle=FakeMemoryOracle(step_free_slots=32, force_whole_prefill=True),
        )
        short_seq = seq_with_len(8192)
        scheduler.add(short_seq)

        with self.assertRaisesRegex(RuntimeError, "atomic prefill step"):
            scheduler.schedule()

    def test_full_prefill_hook_routes_short_bucket_as_single_full_prefill(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv-less-memory",
            chunk=8192,
            max_tokens=16384,
            oracle=FakeMemoryOracle(step_free_slots=32, force_full_prefill=True),
        )
        short_seq = seq_with_len(8192)
        scheduler.add(short_seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [short_seq])
        self.assertEqual(short_seq.current_chunk_size, 8192)

    def test_whole_prefill_hook_keeps_batched_short_prefills(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="pyramidkv",
            chunk=4096,
            max_tokens=10_000,
            oracle=FakeMemoryOracle(step_free_slots=10_000, force_whole_prefill=True),
        )
        scheduler.decode_keep_tokens = 4096
        seq_a = seq_with_len(4000)
        seq_b = seq_with_len(4000)
        scheduler.add(seq_a)
        scheduler.add(seq_b)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq_a, seq_b])
        self.assertEqual(seq_a.current_chunk_size, 4000)
        self.assertEqual(seq_b.current_chunk_size, 4000)

    def test_prefix_cache_hit_reduces_prefill_work_for_fresh_prompt(self):
        oracle = FakeMemoryOracle(prefix_hit_len=8, prefix_hit_blocks=2)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
        )
        seq = seq_with_len(20)
        scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(oracle.refresh_calls, 1)
        self.assertTrue(seq.prefix_cache_enabled)
        self.assertEqual(seq.num_prefilled_tokens, 8)
        self.assertEqual(seq.current_chunk_size, 5)

    def test_prefix_cache_lookup_uses_scheduler_refresher(self):
        oracle = FakeMemoryOracle(prefix_hit_len=8, prefix_hit_blocks=2)
        refresh_calls = []

        def refresh(seq):
            refresh_calls.append(seq.seq_id)
            seq.prefix_cache_enabled = True
            seq.prefix_cache_hit_len = 4
            seq.prefix_cache_hit_block_count = 1
            seq.prefix_cache_hit_last_block_id = b"world"
            seq.prefix_cache_block_size = 4

        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
            prefix_cache_hit_refresher=refresh,
        )
        seq = seq_with_len(20)
        scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(refresh_calls, [seq.seq_id])
        self.assertEqual(oracle.refresh_calls, 0)
        self.assertEqual(seq.num_prefilled_tokens, 4)

    def test_prefix_cache_lookup_skips_recompute_replay(self):
        oracle = FakeMemoryOracle(prefix_hit_len=8, prefix_hit_blocks=2)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
        )
        seq = seq_with_len(20)
        seq.append_token(99)
        seq.start_recompute_replay()
        scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(oracle.refresh_calls, 0)
        self.assertFalse(seq.prefix_cache_enabled)
        self.assertEqual(seq.num_prefilled_tokens, 0)

    def test_decode_preemption_after_generation_starts_recompute_replay(self):
        oracle = FakeMemoryOracle(free_slots=0)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
        )
        seq = seq_with_len(8)
        seq.num_prefilled_tokens = seq.num_prompt_tokens
        seq.append_token(99)
        survivor = seq_with_len(8)
        survivor.num_prefilled_tokens = survivor.num_prompt_tokens
        survivor.append_token(89)
        scheduler.decoding.append(seq)
        scheduler.decoding.append(survivor)

        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertEqual(scheduled, [])
        self.assertFalse(is_prefill)
        self.assertEqual(preempted, [seq])
        self.assertTrue(seq.is_recompute_replay)
        self.assertEqual(seq.num_prefilled_tokens, 0)
        self.assertEqual(list(scheduler.waiting), [seq])
        self.assertEqual(list(scheduler.decoding), [survivor])
        self.assertEqual(scheduler.total_recompute_replays, 1)

    def test_decode_recompute_rejects_sole_request_that_cannot_make_progress(self):
        oracle = FakeMemoryOracle(free_slots=0)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
        )
        seq = seq_with_len(8)
        seq.num_prefilled_tokens = seq.num_prompt_tokens
        seq.append_token(99)
        scheduler.decoding.append(seq)

        with self.assertRaisesRegex(RuntimeError, "sole remaining decode"):
            scheduler.schedule()

    def test_sole_decode_without_new_progress_cannot_yield_to_replay(self):
        oracle = FakeMemoryOracle(free_slots=0)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
        )
        waiting = seq_with_len(8)
        waiting.append_token(90)
        waiting.start_recompute_replay()
        active = seq_with_len(8)
        active.num_prefilled_tokens = active.num_prompt_tokens
        active.append_token(91)
        active.decode_progress_checkpoint = active.num_completion_tokens
        scheduler.waiting.append(waiting)
        scheduler.decoding.append(active)

        with self.assertRaisesRegex(RuntimeError, "forward progress"):
            scheduler.schedule()

    def test_sole_decode_with_new_progress_can_yield_to_replay(self):
        oracle = FakeMemoryOracle(free_slots=0)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
        )
        waiting = seq_with_len(8)
        waiting.append_token(90)
        waiting.start_recompute_replay()
        active = seq_with_len(8)
        active.num_prefilled_tokens = active.num_prompt_tokens
        active.append_token(91)
        active.decode_progress_checkpoint = active.num_completion_tokens
        active.append_token(92)
        scheduler.waiting.append(waiting)
        scheduler.decoding.append(active)

        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertEqual(scheduled, [])
        self.assertFalse(is_prefill)
        self.assertEqual(preempted, [active])
        self.assertEqual(list(scheduler.decoding), [])
        self.assertEqual(list(scheduler.waiting), [waiting, active])
        self.assertTrue(active.is_recompute_replay)

    def test_recompute_replay_waits_while_other_decode_is_active(self):
        oracle = FakeMemoryOracle(free_slots=8)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=8,
            max_tokens=32,
        )
        replay = seq_with_len(8)
        replay.append_token(90)
        replay.append_token(91)
        replay.start_recompute_replay()
        active = seq_with_len(8)
        active.num_prefilled_tokens = active.num_prompt_tokens
        active.append_token(80)
        scheduler.waiting.append(replay)
        scheduler.decoding.append(active)

        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [active])
        self.assertEqual(preempted, [])
        self.assertEqual(list(scheduler.waiting), [replay])
        self.assertEqual(list(scheduler.decoding), [active])

    def test_recompute_replay_prefill_isolated_from_fresh_prompt(self):
        oracle = FakeMemoryOracle(free_slots=64)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=8,
            max_tokens=32,
        )
        replay = seq_with_len(8)
        replay.append_token(90)
        replay.start_recompute_replay()
        fresh = seq_with_len(4)
        scheduler.waiting.extend([replay, fresh])

        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [replay])
        self.assertEqual(preempted, [])
        self.assertEqual(list(scheduler.waiting), [fresh])
        self.assertEqual(replay.current_chunk_size, 8)

    def test_partial_prefill_progresses_when_cold_replay_cannot_fit(self):
        class ReservedPrefillOracle(FakeMemoryOracle):
            def reserved_prefill_slots(self, waiting, engine_prefill_chunk_size):
                return sum(
                    seq.num_prompt_tokens - seq.num_prefilled_tokens
                    for seq in waiting
                    if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens
                )

            def prompt_admission_budgets(self, waiting, engine_prefill_chunk_size):
                return {"slots": self.num_free_slots - self.reserved_prefill_slots(
                    waiting, engine_prefill_chunk_size,
                )}

            def prompt_admission_failure_action(self):
                return "defer"

        oracle = ReservedPrefillOracle(free_slots=20)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="snapkv",
            chunk=8,
            max_tokens=32,
        )
        fresh = seq_with_len(4)
        replay = seq_with_len(16)
        replay.append_token(90)
        replay.start_recompute_replay()
        partial = seq_with_len(12)
        partial.num_prefilled_tokens = 4
        scheduler.waiting.extend([fresh, replay, partial])

        # The replay needs 16 slots, but the partial prefill owns eight of the
        # 20 free slots for its remaining prompt. It must be able to advance.
        self.assertEqual(oracle.prompt_admission_budgets(scheduler.waiting, 8)["slots"], 12)
        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [partial])
        self.assertEqual(partial.current_chunk_size, 8)
        self.assertEqual(preempted, [])
        self.assertEqual(list(scheduler.waiting), [fresh, replay])

    def test_multiple_replay_prefills_are_serialized(self):
        oracle = FakeMemoryOracle(free_slots=64)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=8,
            max_tokens=32,
        )
        first = seq_with_len(15)
        first.max_tokens = 2
        second = seq_with_len(23)
        for seq, token_id in ((first, 90), (second, 91)):
            seq.append_token(token_id)
            seq.start_recompute_replay()
        scheduler.waiting.extend([first, second])

        scheduled, is_prefill, _ = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [first])
        scheduler.postprocess([first], [999], is_prefill=True)
        self.assertEqual(first.num_prefilled_tokens, 8)
        self.assertEqual(second.num_prefilled_tokens, 0)

        scheduled, is_prefill, _ = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [first])
        scheduler.postprocess([first], [999], is_prefill=True)
        self.assertEqual(first.num_prefilled_tokens, 15)
        self.assertEqual(second.num_prefilled_tokens, 0)

        scheduled, is_prefill, _ = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first])
        scheduler.postprocess([first], [999], is_prefill=False)
        self.assertTrue(first.is_finished)

        scheduled, is_prefill, _ = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [second])
        self.assertEqual(second.num_prefilled_tokens, 0)

    def test_recompute_replay_rebuilds_prompt_and_completion_without_appending(self):
        oracle = FakeMemoryOracle(free_slots=64)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=8,
            max_tokens=32,
        )
        seq = seq_with_len(8)
        for token_id in (90, 91, 92):
            seq.append_token(token_id)
        original_tokens = list(seq.token_ids)
        seq.start_recompute_replay()
        scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(seq.replay_input_token_ids, list(range(8)))
        scheduler.postprocess([seq], [999], is_prefill=True)
        self.assertEqual(seq.token_ids, original_tokens)
        self.assertTrue(seq.is_recompute_replay)

        for expected_token, expected_position in ((90, 8), (91, 9)):
            scheduled, is_prefill, _ = scheduler.schedule()
            self.assertFalse(is_prefill)
            self.assertEqual(scheduled, [seq])
            self.assertEqual(seq.decode_input_token, expected_token)
            self.assertEqual(seq.decode_input_position, expected_position)
            self.assertFalse(seq.should_publish_sample)
            scheduler.postprocess([seq], [998], is_prefill=False)
            self.assertEqual(seq.token_ids, original_tokens)

        self.assertFalse(seq.is_recompute_replay)
        self.assertTrue(seq.should_publish_sample)
        scheduled, is_prefill, _ = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(seq.decode_input_token, 92)
        self.assertEqual(seq.decode_input_position, 10)
        scheduler.postprocess([seq], [93], is_prefill=False)
        self.assertEqual(seq.token_ids, original_tokens + [93])

    def test_preemption_releases_chain_runtime_without_invalidating_identity(self):
        engine = object.__new__(LLMEngine)
        calls = []
        engine.model_runner = SimpleNamespace(
            call=lambda method, *args: calls.append((method, args))
        )
        seq = seq_with_len(8)
        seq.chain_id = "chain-a"
        engine._active_chain_sequences = {seq.seq_id: seq}

        engine._release_preempted_sequences([seq])

        self.assertEqual(calls, [("free_slots_batch", ([seq.seq_id],))])
        self.assertIs(engine._active_chain_sequences[seq.seq_id], seq)

    def test_decode_runs_partial_batch_before_considering_preemption(self):
        oracle = FakeMemoryOracle(free_slots=3)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=8,
            max_tokens=32,
        )
        seqs = [seq_with_len(8) for _ in range(4)]
        for seq in seqs:
            seq.current_chunk_size = seq.num_prompt_tokens

        # Follow the real state transition: final prefill emits the first
        # completion token before the request enters the decoding queue.
        scheduler.postprocess(seqs, [99, 99, 99, 99], is_prefill=True)
        self.assertTrue(all(seq.num_completion_tokens == 1 for seq in seqs))

        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, seqs[:3])
        self.assertEqual(preempted, [])
        self.assertEqual(list(scheduler.decoding), [seqs[3], *seqs[:3]])
        self.assertEqual(scheduler.total_preemptions, 0)

    def test_partial_decode_retries_blocked_sequence_before_preempting_progressed_one(self):
        oracle = FakeMemoryOracle(free_slots=3)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=8,
            max_tokens=32,
        )
        seqs = [seq_with_len(8) for _ in range(4)]
        for seq in seqs:
            seq.current_chunk_size = seq.num_prompt_tokens
        scheduler.postprocess(seqs, [99, 99, 99, 99], is_prefill=True)

        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, seqs[:3])
        self.assertEqual(preempted, [])

        oracle._free_slots = 0
        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [])
        self.assertEqual(preempted, [seqs[3]])
        self.assertEqual(list(scheduler.waiting), [seqs[3]])
        self.assertEqual(list(scheduler.decoding), seqs[:3])

    def test_decode_preempts_when_no_candidate_can_use_partial_capacity(self):
        class PartialPageOracle(FakeMemoryOracle):
            def decode_step_free_slots(self):
                return 1

            def decode_step_free_slots_for(self, seq):
                return 0

        oracle = PartialPageOracle(free_slots=0)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="quest",
            chunk=5,
            max_tokens=20,
        )
        seq = seq_with_len(8)
        seq.num_prefilled_tokens = seq.num_prompt_tokens
        scheduler.decoding.append(seq)

        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertEqual(scheduled, [])
        self.assertFalse(is_prefill)
        self.assertEqual(preempted, [seq])
        self.assertEqual(list(scheduler.decoding), [])
        self.assertEqual(list(scheduler.waiting), [seq])
        self.assertEqual(scheduler.total_preemptions, 1)

    def test_prefill_fails_fast_when_no_candidate_can_use_partial_capacity(self):
        class PartialPageOracle(FakeMemoryOracle):
            def prefill_step_free_slots(self):
                return 1

            def prefill_step_free_slots_for(self, seq):
                return 0

        oracle = PartialPageOracle(free_slots=0, step_free_slots=1)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="quest",
            chunk=5,
            max_tokens=20,
        )
        scheduler.add(seq_with_len(8))

        with self.assertRaisesRegex(RuntimeError, "No prefill candidate can use"):
            scheduler.schedule()

    def test_full_prefill_staging_candidate_uses_candidate_budget(self):
        class StagingOracle(FakeMemoryOracle):
            def __init__(self):
                super().__init__(
                    free_slots=73,
                    step_free_slots=73,
                    force_full_prefill=True,
                    force_whole_prefill=True,
                )

            def prefill_step_free_slots_for(self, seq):
                return 32768

            def prefill_step_reservation_cost(self, seq, scheduled_tokens):
                return 0

            def prompt_admission_costs(self, seq):
                return {"slots": 0}

            def prompt_logical_reservation_cost(self, seq):
                return 0

        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            StagingOracle(),
            method="deltakv",
            chunk=32768,
            max_tokens=65536,
        )
        scheduler.add(seq_with_len(16000))

        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(preempted, [])
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0].current_chunk_size, 16000)

    def test_sequence_setstate_round_trips_prefix_cache_block_metadata(self):
        seq = seq_with_len(4)
        seq.current_chunk_size = 4
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = 8
        seq.prefix_cache_hit_block_count = 2
        seq.prefix_cache_hit_last_block_id = b"block"
        seq.prefix_cache_block_size = 4
        seq.prefix_cache_method = "omnikv"
        state = seq.__getstate__()
        self.assertIsInstance(state[15], tuple)
        restored = object.__new__(Sequence)
        restored.__setstate__(state)

        self.assertTrue(restored.prefix_cache_enabled)
        self.assertEqual(restored.prefix_cache_hit_len, 8)
        self.assertEqual(restored.prefix_cache_hit_block_count, 2)
        self.assertEqual(restored.prefix_cache_hit_last_block_id, b"block")
        self.assertIsInstance(restored.token_ids, list)
        self.assertEqual(restored.prompt_token_ids, tuple(seq.token_ids))

    def test_sequence_ipc_carries_recompute_prefill_and_decode_inputs(self):
        seq = seq_with_len(4)
        seq.append_token(90)
        seq.append_token(91)
        seq.start_recompute_replay()
        seq.current_chunk_size = 2

        restored_prefill = object.__new__(Sequence)
        prefill_state = seq.__getstate__()
        self.assertIsInstance(prefill_state[15], list)
        restored_prefill.__setstate__(prefill_state)
        self.assertTrue(restored_prefill.is_recompute_prefill)
        self.assertEqual(restored_prefill.replay_input_token_ids, [0, 1])

        seq.num_prefilled_tokens = seq.num_prompt_tokens
        restored_decode = object.__new__(Sequence)
        restored_decode.__setstate__(seq.__getstate__())
        self.assertTrue(restored_decode.is_recompute_decode)
        self.assertEqual(restored_decode.decode_input_token, 90)
        self.assertEqual(restored_decode.decode_input_position, 4)


class DeltaKVFullPrefillStagingTest(unittest.TestCase):
    def _make_raw_deltakv_prefill_manager(self):
        from sparseengine.engine.cache_manager.raw_kv_offload import RawKVOffloadBuffer

        max_model_len = 16
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.device = torch.device("cpu")
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=2,
            long_prefill_offload_threshold=5,
            max_num_batched_tokens=16,
            sink_keep_tokens=1,
            recent_keep_tokens=1,
            deltakv_center_ratio=0.5,
        )
        manager.deltakv_layer_ids = [1]
        manager.deltakv_layer_to_idx = {1: 0}
        manager.full_layer_to_idx = {0: 0}
        manager.deltakv_prefill_staging_num_slots = max_model_len
        manager.free_slots_stack_full = torch.arange(100, 100 + max_model_len, dtype=torch.int32)
        manager._num_free_slots_full = max_model_len
        manager.full_layer_slots_map = torch.zeros((1, max_model_len), dtype=torch.int32)
        manager.full_layer_slot_to_pos = None
        manager.free_slots_stack_deltakv_full = torch.arange(16, 16 + max_model_len, dtype=torch.int32)
        manager._num_free_slots_deltakv_full = max_model_len
        manager._deltakv_temp_full_reserve = 0
        manager._deltakv_static_temp_slots_reserved_total = 0
        manager.deltakv_slot_to_pos = torch.full((64,), -1, dtype=torch.int32)
        manager.sparse_layer_raw_slots_map = torch.full((1, max_model_len), -1, dtype=torch.int32)
        manager.free_slots_stack_deltakv_latent = torch.arange(max_model_len, dtype=torch.int32)
        manager._num_free_slots_deltakv_latent = max_model_len
        manager.sparse_layer_latent_slots_map = torch.full((1, max_model_len), -1, dtype=torch.int32)
        manager.seq_id_to_row = {}
        manager.free_rows = deque([0])
        manager.row_seq_lens = np.zeros((1,), dtype=np.int32)
        manager.row_deltakv_compressed_lens = np.zeros((1,), dtype=np.int32)
        manager.row_deltakv_compressed_lens_gpu = torch.zeros((1,), dtype=torch.int32)
        manager.row_deltakv_center_slots = [[None, None]]
        manager.full_layer_batch_states = SimpleNamespace()
        manager.deltakv_layer_batch_states = SimpleNamespace()
        manager._deltakv_prefill_staging_active = False
        manager.deltakv_prefill_staging_kv_cache = torch.zeros(
            (2, max_model_len, 1, 1), dtype=torch.float32
        )
        manager.deltakv_prefill_staging_pre_rope_k_cache = torch.zeros(
            (max_model_len, 1, 1), dtype=torch.float32
        )
        manager.raw_kv_offload_buffer = RawKVOffloadBuffer(
            pin_memory=False,
            mode="chunked",
        )
        manager._deltakv_full_prefill_plans = {}
        manager._deltakv_full_prefill_compressed_layers = set()
        manager._deltakv_long_prefill_offload_row_idx = None
        manager._deltakv_long_prefill_offload_start = 0
        manager._deltakv_long_prefill_offload_end = 0
        manager._deltakv_long_prefill_offload_total_len = 0
        manager._deltakv_long_prefill_offload_is_last_chunk = False
        return manager

    def test_raw_full_layer_short_prefill_uses_persistent_slots(self):
        manager = self._make_raw_deltakv_prefill_manager()
        seq = seq_with_len(2)
        seq.current_chunk_size = 2

        DeltaKVCacheManager._prepare_prefill(manager, [seq])

        row_idx = manager.seq_id_to_row[seq.seq_id]
        persistent_slots = manager.full_layer_slots_map[row_idx, :2].clone()
        torch.testing.assert_close(manager.full_layer_batch_states.slot_mapping, persistent_slots)
        self.assertNotEqual(
            manager.full_layer_batch_states.slot_mapping.tolist(),
            torch.arange(2, dtype=torch.int32).tolist(),
        )
        torch.testing.assert_close(
            manager.deltakv_layer_batch_states.slot_mapping,
            manager.sparse_layer_raw_slots_map[row_idx, :2],
        )

    def test_raw_full_layer_long_offload_staging_uses_persistent_slots(self):
        manager = self._make_raw_deltakv_prefill_manager()
        seq = seq_with_len(6)
        seq.current_chunk_size = 2

        DeltaKVCacheManager._prepare_prefill(manager, [seq])

        row_idx = manager.seq_id_to_row[seq.seq_id]
        persistent_slots = manager.full_layer_slots_map[row_idx, :2].clone()
        torch.testing.assert_close(manager.full_layer_batch_states.slot_mapping, persistent_slots)
        self.assertNotEqual(
            manager.full_layer_batch_states.slot_mapping.tolist(),
            torch.arange(2, dtype=torch.int32).tolist(),
        )
        torch.testing.assert_close(
            manager.deltakv_layer_batch_states.slot_mapping,
            torch.arange(2, dtype=torch.int32),
        )

    def test_fresh_raw_offload_tracks_chunk_rows_and_finalizes_once(self):
        manager = self._make_raw_deltakv_prefill_manager()
        manager._full_layer_kivi_enabled = lambda: False
        manager._deltakv_restore_sparse_prefix_to_staging = (
            lambda layer_idx, start: None
        )
        manager._debug_track_deltakv_full_slots = lambda *args, **kwargs: None
        calls = []

        def compress(layer_idx):
            calls.append(("compress", int(layer_idx)))
            manager._deltakv_full_prefill_compressed_layers.add(int(layer_idx))

        manager._deltakv_compress_full_prefill_layer = compress
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv",
            chunk=2,
            max_tokens=8,
            oracle=CacheManagerPolicyOracle(manager),
        )
        seq = seq_with_len(8)
        scheduler.add(seq)
        observed_row_lens = []
        observed_modes = []
        while scheduler.waiting:
            scheduled, is_prefill, _ = scheduler.schedule()
            self.assertTrue(is_prefill)
            observed_modes.append(
                scheduler.prefill_execution_mode_for_batch(scheduled)
            )
            manager.prepare_step(scheduled, is_prefill=True)
            row_idx = manager.seq_id_to_row[seq.seq_id]
            observed_row_lens.append(int(manager.row_seq_lens[row_idx]))
            self.assertEqual(
                int(manager.deltakv_layer_batch_states.context_lens[0]),
                int(seq.num_prefilled_tokens + seq.current_chunk_size),
            )
            if seq.num_prefilled_tokens > 0:
                manager.before_prefill_layer_attention(1, None)
            start = int(seq.num_prefilled_tokens)
            end = start + int(seq.current_chunk_size)
            manager.deltakv_prefill_staging_pre_rope_k_cache[start:end] = 1
            manager.deltakv_prefill_staging_kv_cache[1, start:end] = 2
            manager.on_layer_attention_end(1)
            manager.on_forward_end(scheduled, is_prefill=True)
            scheduler.postprocess(scheduled, [99], is_prefill=True)

        self.assertEqual(observed_modes, [PREFILL_EXECUTION_RAW_OFFLOAD] * 4)
        self.assertEqual(observed_row_lens, [2, 4, 6, 8])
        self.assertEqual(calls, [("compress", 1)])
        self.assertFalse(manager._deltakv_prefill_staging_active)
        self.assertEqual(manager._deltakv_full_prefill_plans, {})
        self.assertEqual(manager.raw_kv_offload_buffer._entries, {})
        self.assertEqual(int(manager.row_seq_lens[row_idx]), 8)

    def test_deltakv_sparse_decode_backend_controls_fa2_view(self):
        from sparseengine.engine.cache_manager import (
            AttentionViewMeta,
            DecodeComputeView,
            ExplicitKVPayload,
        )
        from sparseengine.engine.cache_manager.methods.deltakv_base import DeltaKVCacheTritonManagerV4
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        q = torch.empty((1, 1, 4), dtype=torch.float32)
        selection = SimpleNamespace(
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([2], dtype=torch.int32),
            attn_score=None,
            max_context_len=2,
        )

        cases = (
            ("fa2", False, "flash_attn_contiguous"),
            ("custom", False, "dense"),
            ("fa2", True, "dense"),
        )
        for backend, staging_active, expected in cases:
            with self.subTest(backend=backend, staging_active=staging_active):
                manager = object.__new__(DeltaKVLessMemoryCacheManager)
                manager.config = SimpleNamespace(deltakv_sparse_decode_backend=backend)
                manager.full_layer_to_idx = {}
                manager._full_layer_kivi_enabled = lambda: False
                manager.deltakv_layer_to_idx = {1: 0}
                manager.has_prefill_staging_view = lambda layer_idx, active=staging_active: active
                view = DecodeComputeView(
                    meta=AttentionViewMeta(
                        active_slots=torch.tensor([[0, 1]], dtype=torch.int32),
                        req_indices=selection.req_indices,
                        context_lens=selection.context_lens,
                    ),
                    payload=ExplicitKVPayload(
                        k_cache=torch.empty((2, 1, 4), dtype=torch.float32),
                        v_cache=torch.empty((2, 1, 4), dtype=torch.float32),
                        backend="dense",
                    ),
                )

                with patch.object(DeltaKVCacheTritonManagerV4, "build_decode_compute_view", return_value=view):
                    out = DeltaKVLessMemoryCacheManager.build_decode_compute_view(
                        manager,
                        1,
                        q,
                        selection,
                        num_heads=1,
                        num_kv_heads=1,
                    )

                self.assertIsInstance(out.payload, ExplicitKVPayload)
                self.assertEqual(out.payload.backend, expected)

    def test_static_decode_resets_deltakv_view_cache_before_validation(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager._deltakv_view_cache_key = (1, 1, 2, 1, 4)
        manager._deltakv_view_cache_value = object()
        reset_calls = []

        def reset_view_cache():
            reset_calls.append(True)
            manager._deltakv_view_cache_key = None
            manager._deltakv_view_cache_value = None

        manager._deltakv_reset_view_cache = reset_view_cache

        with self.assertRaisesRegex(ValueError, "non-empty real decode batch"):
            DeltaKVCacheManager.prepare_decode_static(
                manager,
                [],
                torch.empty((1,), dtype=torch.int64),
                torch.empty((1,), dtype=torch.int64),
                torch.empty((1,), dtype=torch.int32),
                torch.empty((1,), dtype=torch.int32),
                torch.empty((1,), dtype=torch.int32),
            )

        self.assertEqual(reset_calls, [True])
        self.assertIsNone(manager._deltakv_view_cache_key)
        self.assertIsNone(manager._deltakv_view_cache_value)

    def test_base_deltakv_has_no_middle_full_prefill_staging(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=5,
            long_prefill_offload_threshold=5,
            max_num_batched_tokens=64,
        )
        manager.deltakv_layer_ids = [0]
        seq = seq_with_len(20)

        seq.current_chunk_size = 5
        self.assertFalse(DeltaKVCacheManager._should_use_full_prefill_staging(manager, [seq]))
        self.assertTrue(DeltaKVCacheManager.requires_long_prefill_offload(manager, seq))

    def test_deltakv_short_atomic_prefill_requirement_is_cache_manager_owned(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=8192,
            long_prefill_offload_threshold=8192,
            max_num_batched_tokens=16384,
        )
        manager.deltakv_layer_ids = [0]

        short_seq = seq_with_len(8192)
        self.assertTrue(DeltaKVCacheManager.requires_full_prefill_step(manager, short_seq))

        long_seq = seq_with_len(9000)
        self.assertFalse(DeltaKVCacheManager.requires_full_prefill_step(manager, long_seq))
        self.assertTrue(DeltaKVCacheManager.requires_long_prefill_offload(manager, long_seq))

    def test_full_prefill_plan_keeps_only_persistent_final_representation(self):
        plan = DeltaKVCacheManager._deltakv_full_prefill_plan_cpu(
            20,
            sink=2,
            recent=4,
            cluster_step=4,
        )

        self.assertEqual(plan.evict_start, 2)
        self.assertEqual(plan.evict_end, 14)
        self.assertEqual(plan.center_positions, (2, 6, 10))
        self.assertIn(3, plan.latent_positions)
        self.assertLess(len(plan.keep_positions), plan.total_len)
        self.assertNotIn(3, plan.keep_positions)

    def test_prompt_admission_counts_sparse_raw_keep_positions_after_graph_reserve(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.config = SimpleNamespace(
            sink_keep_tokens=8,
            recent_keep_tokens=128,
            deltakv_center_ratio=0.1,
        )
        manager._num_free_slots_full = 100_000
        manager._num_free_slots_deltakv_full = 16_988
        manager._deltakv_temp_full_reserve = 16_384
        manager._deltakv_centers_capacity = 10_000
        manager._deltakv_centers_reserved_total = 0
        manager._deltakv_latent_reserved_by_seq = {}
        manager._num_free_slots_deltakv_latent = 100_000

        seq = seq_with_len(8192)
        budgets = DeltaKVCacheManager.prompt_admission_budgets(manager, deque(), engine_prefill_chunk_size=2048)
        costs = DeltaKVCacheManager.prompt_admission_costs(manager, seq)

        self.assertEqual(costs["deltakv_raw"], 1050)
        self.assertEqual(budgets["deltakv_raw"], 604)
        self.assertLess(budgets["deltakv_raw"], costs["deltakv_raw"])

        manager._deltakv_static_temp_slots_reserved_total = 12_000
        budgets = DeltaKVCacheManager.prompt_admission_budgets(manager, deque(), engine_prefill_chunk_size=2048)
        self.assertEqual(budgets["deltakv_raw"], 12_604)

    def test_prompt_admission_reserves_full_layers_across_long_offload_chunks(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.device = torch.device("cpu")
        manager.config = SimpleNamespace(
            sink_keep_tokens=1,
            recent_keep_tokens=1,
            deltakv_center_ratio=0.5,
            max_model_len=64,
            max_num_seqs_in_batch=4,
        )
        manager._num_free_slots_full = 12
        manager.free_slots_stack_full = torch.arange(12, dtype=torch.int32)
        manager.full_layer_slots_map = torch.zeros((1, 64), dtype=torch.int32)
        manager.full_layer_slot_to_pos = None
        manager.seq_id_to_row = {}
        manager.free_rows = deque([0])
        manager.row_seq_lens = np.zeros((1,), dtype=np.int32)
        manager._num_free_slots_deltakv_full = 100
        manager._deltakv_temp_full_reserve = 0
        manager._deltakv_static_temp_slots_reserved_total = 0
        manager._deltakv_centers_capacity = 100
        manager._deltakv_centers_reserved_total = 0
        manager._deltakv_centers_reserved_by_seq = {}
        manager._deltakv_latent_reserved_total = 0
        manager._deltakv_latent_reserved_by_seq = {}
        manager._full_layer_kivi_reserved_total = 0
        manager._full_layer_kivi_reserved_by_seq = {}
        manager._full_layers_reserved_total = 0
        manager._full_layers_reserved_by_seq = {}

        manager._num_free_slots_deltakv_latent = 100
        manager.row_deltakv_compressed_lens = np.zeros((1,), dtype=np.int32)
        seq = seq_with_len(6)
        seq.max_tokens = 2
        costs = DeltaKVCacheManager.prompt_admission_costs(manager, seq)

        DeltaKVCacheManager.on_prompt_admitted(manager, seq, costs)
        DeltaKVCacheManager._allocate_full(manager, seq.seq_id, 2)

        self.assertEqual(manager._full_layers_reserved_by_seq[seq.seq_id], 4)
        self.assertEqual(manager._full_layers_reserved_total, 4)
        seq.num_prefilled_tokens = 2
        budgets = DeltaKVCacheManager.prompt_admission_budgets(manager, deque([seq]), engine_prefill_chunk_size=2)
        self.assertEqual(budgets["full_layers"], 6)

        DeltaKVCacheManager._release_prompt_admission_reservations(manager, seq.seq_id)
        self.assertNotIn(seq.seq_id, manager._full_layers_reserved_by_seq)
        self.assertEqual(manager._full_layers_reserved_total, 0)

    def test_temp_deltakv_full_allocation_does_not_alias_free_stack(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.free_slots_stack_deltakv_full = torch.arange(16, dtype=torch.int32)
        manager._num_free_slots_deltakv_full = 16
        manager._deltakv_temp_full_reserve = 0

        slots = DeltaKVCacheManager._allocate_temp_deltakv_full(manager, 4)

        self.assertEqual(manager._num_free_slots_deltakv_full, 12)
        torch.testing.assert_close(slots, torch.tensor([12, 13, 14, 15], dtype=torch.int32))

        manager.free_slots_stack_deltakv_full[12:16] = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        torch.testing.assert_close(slots, torch.tensor([12, 13, 14, 15], dtype=torch.int32))

    def test_layer_attention_end_triggers_layer_local_staging_compression(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager._deltakv_prefill_staging_active = True
        manager.deltakv_layer_to_idx = {0: 0}
        manager.deltakv_layer_ids = [0]
        manager._deltakv_full_prefill_compressed_layers = set()
        manager._deltakv_full_prefill_plans = {}
        calls = []

        def compress(layer_idx):
            calls.append(layer_idx)
            manager._deltakv_full_prefill_compressed_layers.add(layer_idx)

        manager._deltakv_compress_full_prefill_layer = compress

        DeltaKVCacheManager.on_layer_attention_end(manager, 0)

        self.assertEqual(calls, [0])
        self.assertFalse(manager._deltakv_prefill_staging_active)

    def test_finish_full_prefill_staging_clears_completed_plans(self):
        manager = object.__new__(DeltaKVCacheManager)
        released_rows = []
        manager.raw_kv_offload_buffer = SimpleNamespace(
            release_row=lambda row_idx: released_rows.append(int(row_idx))
        )
        manager._deltakv_prefill_staging_active = True
        manager._deltakv_full_prefill_compressed_layers = {0}
        manager._deltakv_full_prefill_plans = {
            3: {
                "row_idx": 3,
                "keep_slots": torch.empty((0,), dtype=torch.int32),
                "keep_pos": torch.empty((0,), dtype=torch.int32),
            }
        }
        manager.deltakv_slot_to_pos = torch.empty((0,), dtype=torch.int32)

        DeltaKVCacheManager._deltakv_finish_full_prefill_staging(manager)

        self.assertEqual(released_rows, [3])
        self.assertFalse(manager._deltakv_prefill_staging_active)
        self.assertEqual(manager._deltakv_full_prefill_plans, {})
        self.assertEqual(manager._deltakv_full_prefill_compressed_layers, set())

    def test_less_memory_finish_full_prefill_staging_clears_kivi_plans(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        released_rows = []
        manager.raw_kv_offload_buffer = SimpleNamespace(
            release_row=lambda row_idx: released_rows.append(int(row_idx))
        )
        manager._deltakv_prefill_staging_active = True
        manager._deltakv_full_prefill_compressed_layers = {1}
        manager._deltakv_full_prefill_plans = {
            4: {
                "row_idx": 4,
                "keep_slots": torch.empty((0,), dtype=torch.int32),
                "keep_pos": torch.empty((0,), dtype=torch.int32),
            }
        }
        manager.deltakv_slot_to_pos = torch.empty((0,), dtype=torch.int32)
        manager._full_layer_kivi_full_prefill_plans = {4: {"row_idx": 4}}
        manager._full_layer_kivi_full_prefill_materialized_layers = {0}
        manager._deltakv_clear_long_prefill_offload_prefetch = lambda: None

        DeltaKVLessMemoryCacheManager._deltakv_finish_full_prefill_staging(manager)

        self.assertEqual(released_rows, [4])
        self.assertFalse(manager._deltakv_prefill_staging_active)
        self.assertEqual(manager._deltakv_full_prefill_plans, {})
        self.assertEqual(manager._deltakv_full_prefill_compressed_layers, set())
        self.assertEqual(manager._full_layer_kivi_full_prefill_plans, {})
        self.assertEqual(manager._full_layer_kivi_full_prefill_materialized_layers, set())


class DeltaKVLessMemoryStorageContractTest(unittest.TestCase):
    def test_sparse_rope_to_key_applies_only_key_rope(self):
        from sparseengine.layers.rotary_embedding import RotaryEmbedding

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        key = torch.arange(8, dtype=torch.float32).view(2, 1, 4)
        positions = torch.tensor([0, 1], dtype=torch.long)
        calls = []

        def fake_apply_rotary_emb(x, cos, sin):
            calls.append((x, cos, sin))
            return x + 7

        manager.rotary_emb = RotaryEmbedding(
            head_size=4,
            rotary_dim=4,
            max_position_embeddings=2,
            base=10000.0,
            backend="torch",
        )
        with patch("sparseengine.engine.cache_manager.methods.deltakv_base.apply_rotary_emb", fake_apply_rotary_emb):
            out = DeltaKVLessMemoryCacheManager._apply_sparse_rope_to_key(manager, positions, key)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], key)
        torch.testing.assert_close(out, key + 7)

    def test_rotary_embedding_forward_uses_compiled_path(self):
        from sparseengine.layers.rotary_embedding import RotaryEmbedding

        rotary_emb = RotaryEmbedding(
            head_size=4,
            rotary_dim=4,
            max_position_embeddings=2,
            base=10000.0,
            backend="torch",
        )
        positions = torch.tensor([0, 1], dtype=torch.long)
        query = torch.zeros((2, 1, 4), dtype=torch.float32)
        key = torch.ones((2, 1, 4), dtype=torch.float32)
        calls = []

        def fake_apply_rotary_emb(x, cos, sin):
            calls.append((x, cos, sin))
            return x + len(calls)

        unwrapped_forward = rotary_emb.compiled_forward.__wrapped__

        def eager_forward(*args):
            return unwrapped_forward(rotary_emb, *args)

        with (
            patch("sparseengine.layers.rotary_embedding.apply_rotary_emb", fake_apply_rotary_emb),
            patch.object(rotary_emb, "compiled_forward", wraps=eager_forward) as compiled,
        ):
            query_out, key_out = rotary_emb(positions, query, key)

        compiled.assert_called_once_with(positions, query, key)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], query)
        self.assertIs(calls[1][0], key)
        torch.testing.assert_close(query_out, query + 1)
        torch.testing.assert_close(key_out, key + 2)

    def test_long_prefill_offload_sparse_restore_applies_rope_helper(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.device = "cpu"
        manager.config = SimpleNamespace(engine_prefill_chunk_size=2)
        manager._deltakv_long_prefill_offload_step_active = True
        manager._deltakv_long_prefill_offload_start = 3
        manager._deltakv_long_prefill_offload_row_idx = 0
        manager.has_prefill_staging_view = lambda layer_idx: True
        manager._deltakv_long_prefill_offload_kind = lambda layer_idx: "sparse_pre_rope"
        manager._deltakv_consume_long_prefill_offload_staged_prefetch = lambda **kwargs: True
        manager.deltakv_layer_to_idx = {1: 0}
        manager.deltakv_prefill_staging_pre_rope_k_cache = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        manager.deltakv_prefill_staging_kv_cache = torch.zeros((2, 4, 1, 4), dtype=torch.float32)
        manager._apply_sparse_k_norm_if_needed = lambda l_idx, k: k + 1
        rope_calls = []

        def fake_apply_sparse_rope_to_key(pos, key):
            rope_calls.append((pos, key.clone()))
            return key + 100

        manager._apply_sparse_rope_to_key = fake_apply_sparse_rope_to_key

        DeltaKVLessMemoryCacheManager.before_prefill_layer_attention(manager, 1, None)

        self.assertEqual(len(rope_calls), 2)
        torch.testing.assert_close(rope_calls[0][0], torch.tensor([0, 1], dtype=torch.long))
        torch.testing.assert_close(rope_calls[1][0], torch.tensor([2], dtype=torch.long))
        expected_normed = manager.deltakv_prefill_staging_pre_rope_k_cache[:3] + 1
        torch.testing.assert_close(rope_calls[0][1], expected_normed[:2])
        torch.testing.assert_close(rope_calls[1][1], expected_normed[2:3])
        torch.testing.assert_close(manager.deltakv_prefill_staging_kv_cache[0, :3], expected_normed + 100)

    def test_long_prefill_offload_prefetch_waits_for_current_stream_before_staging_write(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.device = "cuda:0"
        manager._deltakv_long_prefill_offload_prefetch_stream = None
        manager._deltakv_long_prefill_offload_prefetch_states = {}
        manager._deltakv_long_prefill_offload_layer_order = lambda: [0, 1]
        manager._deltakv_long_prefill_offload_kind = lambda layer_idx: "sparse_pre_rope"
        manager._deltakv_long_prefill_offload_prefetch_enabled = lambda: True
        manager.deltakv_prefill_staging_pre_rope_k_cache = torch.empty((4, 1, 1))
        manager.deltakv_prefill_staging_kv_cache = torch.empty((2, 4, 1, 1))

        calls = []
        created_events = []

        class FakeEvent:
            def __init__(self):
                self.name = f"event{len(created_events)}"
                created_events.append(self)

            def record(self, stream=None):
                calls.append(("record", self.name, getattr(stream, "name", None)))

        class FakeStream:
            def __init__(self, device=None, *, name="prefetch"):
                self.device = device
                self.name = name

            def wait_event(self, event):
                calls.append(("wait", self.name, event.name))

        class FakeStreamContext:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                calls.append(("enter", self.stream.name))
                return self.stream

            def __exit__(self, exc_type, exc, tb):
                calls.append(("exit", self.stream.name))
                return False

        current_stream = [FakeStream(device=manager.device, name="current")]

        def fake_record_event(event, device=None):
            del device
            event.record(current_stream[0])

        class FakeRuntimeStreamContext(FakeStreamContext):
            def __enter__(self):
                current_stream[0] = self.stream
                return super().__enter__()

            def __exit__(self, exc_type, exc, tb):
                try:
                    return super().__exit__(exc_type, exc, tb)
                finally:
                    current_stream[0] = FakeStream(device=manager.device, name="current")

        def fake_copy_prefix_to(**kwargs):
            calls.append(("copy_prefix_to", int(kwargs["layer_idx"]), int(kwargs["end"])))

        manager.raw_kv_offload_buffer = SimpleNamespace(copy_prefix_to=fake_copy_prefix_to)

        with (
            patch(
                "sparseengine.engine.cache_manager.methods.deltakv_less_memory.device_runtime.new_event",
                lambda device=None: FakeEvent(),
            ),
            patch(
                "sparseengine.engine.cache_manager.methods.deltakv_less_memory.device_runtime.new_stream",
                lambda device=None: FakeStream(device=device),
            ),
            patch(
                "sparseengine.engine.cache_manager.methods.deltakv_less_memory.device_runtime.record_event",
                fake_record_event,
            ),
            patch(
                "sparseengine.engine.cache_manager.methods.deltakv_less_memory.device_runtime.stream_context",
                lambda stream: FakeRuntimeStreamContext(stream),
            ),
            patch(
                "sparseengine.engine.cache_manager.methods.deltakv_less_memory.device_runtime.stream_wait_event",
                lambda stream, event: stream.wait_event(event),
            ),
        ):
            DeltaKVLessMemoryCacheManager._deltakv_schedule_next_long_prefill_offload_prefetch(
                manager,
                layer_idx=0,
                row_idx=3,
                end=2,
            )

        self.assertEqual(
            calls,
            [
                ("record", "event0", "current"),
                ("enter", "prefetch"),
                ("wait", "prefetch", "event0"),
                ("copy_prefix_to", 1, 2),
                ("record", "event1", "prefetch"),
                ("exit", "prefetch"),
            ],
        )
        key = (1, 3, "sparse_pre_rope", 2)
        state = manager._deltakv_long_prefill_offload_prefetch_states[key]
        self.assertIs(state["staging_available_event"], created_events[0])
        self.assertIs(state["event"], created_events[1])

    def test_compressor_residual_quant_group_size_uses_payload_dim(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.head_dim = 128
        manager.config = SimpleNamespace(deltakv_latent_quant_group_size=0, use_compression=True)

        self.assertEqual(DeltaKVLessMemoryCacheManager._quant_group_size(manager, 1024), 1024)

        manager.config.deltakv_latent_quant_group_size = 32
        self.assertEqual(DeltaKVLessMemoryCacheManager._quant_group_size(manager, 1024), 32)

    def test_context_does_not_own_attention_transients(self):
        from sparseengine.utils.context import get_context, reset_context, set_context

        reset_context()
        set_context(is_prefill=True)

        ctx = get_context()
        for name in (
            "pre_qk_norm_k",
            "pre_rope_k",
            "pre_rope_v",
            "full_layer_k_post_rope_for_store",
            "full_layer_q_post_rope_for_score",
        ):
            self.assertFalse(hasattr(ctx, name), name)
        reset_context()

    def test_delta_quant_full_kivi_stages_first_prefill_even_below_chunk_threshold(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=32768,
            long_prefill_offload_threshold=32768,
            enable_full_layer_kivi_quant=True,
            full_layer_kv_quant_bits=4,
        )
        manager.deltakv_layer_ids = [2]

        seq = seq_with_len(11766)
        seq.current_chunk_size = 11766

        self.assertTrue(DeltaKVLessMemoryCacheManager._should_use_full_prefill_staging(manager, [seq]))
        peer = seq_with_len(4096)
        peer.current_chunk_size = 4096
        self.assertTrue(
            DeltaKVLessMemoryCacheManager._should_use_full_prefill_staging(
                manager,
                [seq, peer],
            )
        )

        partial = seq_with_len(11766)
        partial.current_chunk_size = 4096
        self.assertFalse(DeltaKVLessMemoryCacheManager._should_use_full_prefill_staging(manager, [partial]))
        self.assertFalse(
            DeltaKVLessMemoryCacheManager._should_use_full_prefill_staging(
                manager,
                [seq, partial],
            )
        )

    def test_delta_quant_full_kivi_requests_full_prefill_when_step_slots_are_too_small(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=32768,
            long_prefill_offload_threshold=32768,
            enable_full_layer_kivi_quant=True,
            full_layer_kv_quant_bits=4,
        )
        manager.deltakv_layer_ids = [2]
        manager.deltakv_prefill_staging_num_slots = 32768
        manager.prefill_step_free_slots = lambda: 32

        seq = seq_with_len(11766)
        self.assertTrue(DeltaKVLessMemoryCacheManager.should_schedule_full_prefill(manager, seq))
        self.assertEqual(
            DeltaKVLessMemoryCacheManager.prefill_step_free_slots_for(manager, seq),
            32768,
        )
        self.assertEqual(
            DeltaKVLessMemoryCacheManager.prefill_step_reservation_cost(manager, seq, 11766),
            0,
        )

        tiny = seq_with_len(16)
        self.assertTrue(DeltaKVLessMemoryCacheManager.should_schedule_full_prefill(manager, tiny))

    def test_delta_quant_full_kivi_does_not_force_full_prefill_for_offload_candidate(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=1024,
            long_prefill_offload_threshold=1024,
            max_num_batched_tokens=2048,
            enable_full_layer_kivi_quant=True,
            full_layer_kv_quant_bits=4,
        )
        manager.deltakv_layer_ids = [2]
        manager.deltakv_prefill_staging_num_slots = 4096
        manager.prefill_step_free_slots = lambda: 32

        seq = seq_with_len(2048)
        seq.current_chunk_size = 1024
        self.assertFalse(DeltaKVLessMemoryCacheManager.should_schedule_full_prefill(manager, seq))
        self.assertFalse(DeltaKVLessMemoryCacheManager._should_use_full_prefill_staging(manager, [seq]))
        self.assertTrue(DeltaKVLessMemoryCacheManager._should_use_long_prefill_offload_staging(manager, [seq]))

    def test_delta_quant_raw_overhead_does_not_depend_on_prefill_chunk(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        max_seqs = 8
        sink = 8
        recent = 32
        top_decode = 1342

        persistent = DeltaKVLessMemoryCacheManager._resident_sparse_raw_overhead_slots(
            max_seqs,
            sink,
            recent,
        )
        scratch = DeltaKVLessMemoryCacheManager._decode_reconstruct_scratch_slots(
            max_seqs,
            top_decode,
            sink,
            recent,
        )

        self.assertEqual(persistent, max_seqs * (sink + 2 * recent + 1))
        self.assertEqual(scratch, max_seqs * 2 * top_decode)

    def test_deltakv_storage_hooks_keep_sparse_raw_and_full_postrope(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        stores = []
        rope_hooks = []
        slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
        manager._store_layer_kv = lambda layer_idx, k, v: stores.append((layer_idx, k, v)) or slot_mapping
        manager.on_kv_stored = (
            lambda layer_idx, k, slots, **kwargs: rope_hooks.append((layer_idx, k, slots, kwargs))
        )

        raw_k = torch.full((2, 1, 4), 3.0)
        raw_v = torch.full((2, 1, 4), 4.0)
        postrope_k = torch.ones((2, 1, 4))
        value = torch.full((2, 1, 4), 2.0)

        DeltaKVCacheManager.save_raw_kv_if_needed(manager, 1, raw_k, raw_v)
        DeltaKVCacheManager.save_rope_kv_if_needed(manager, 1, postrope_k, value)

        self.assertEqual(len(stores), 1)
        self.assertEqual(stores[0][0], 1)
        self.assertIs(stores[0][1], raw_k)
        self.assertIs(stores[0][2], raw_v)
        self.assertEqual(len(rope_hooks), 0)

        DeltaKVCacheManager.save_rope_kv_if_needed(manager, 0, postrope_k, value)

        self.assertEqual(len(stores), 2)
        self.assertEqual(stores[1][0], 0)
        self.assertIs(stores[1][1], postrope_k)
        self.assertIs(stores[1][2], value)
        self.assertEqual(len(rope_hooks), 1)

    def test_deltakv_materializes_raw_sparse_view_before_attention(self):
        from sparseengine.layers.rotary_embedding import apply_rotary_emb
        from sparseengine.utils.context import reset_context, set_context

        reset_context()
        manager = object.__new__(DeltaKVCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.num_kv_heads = 1
        manager.head_dim = 4
        manager.deltakv_full_kv_cache = torch.empty((2, 1, 4, 1, 4), dtype=torch.float32)
        manager.deltakv_full_kv_cache[0, 0] = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0]],
                [[9.0, 9.0, 9.0, 9.0]],
                [[0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0]],
            ]
        )
        manager.deltakv_full_kv_cache[1, 0] = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        manager.deltakv_slot_to_pos = torch.tensor([2, 1, -1, -1], dtype=torch.int32)
        manager._deltakv_postrope_slot_marker = torch.zeros((1, 4), dtype=torch.int32)
        cos = torch.tensor(
            [
                [1.0, 1.0],
                [0.9, 0.8],
                [0.7, 0.6],
            ],
            dtype=torch.float32,
        )
        sin = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            dtype=torch.float32,
        )
        manager.cos_sin_cache = torch.cat([cos, sin], dim=-1).unsqueeze(1)
        manager._allocate_temp_deltakv_full = lambda size: torch.tensor([2, 3], dtype=torch.int32)[:size]

        set_context(is_prefill=False, cache_manager=manager)
        active_slots = torch.tensor([[0, 1]], dtype=torch.int32)
        context_lens = torch.tensor([2], dtype=torch.int32)
        out_active, temp_slots = DeltaKVCacheManager._materialize_deltakv_active_postrope_view(
            manager,
            1,
            active_slots.clone(),
            context_lens,
            already_postrope_slots=torch.tensor([1], dtype=torch.int32),
        )

        self.assertTrue(torch.equal(out_active, torch.tensor([[2, 1]], dtype=torch.int32)))
        self.assertEqual(int(temp_slots.numel()), 0)
        raw_k = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
        cos_sin = manager.cos_sin_cache[torch.tensor([2])]
        expected_cos, expected_sin = cos_sin.chunk(2, dim=-1)
        expected_k = apply_rotary_emb(raw_k, expected_cos, expected_sin)
        torch.testing.assert_close(manager.deltakv_full_kv_cache[0, 0, 2], expected_k[0])
        torch.testing.assert_close(manager.deltakv_full_kv_cache[1, 0, 2], manager.deltakv_full_kv_cache[1, 0, 0])
        self.assertEqual(int(manager.deltakv_slot_to_pos[2]), 2)
        reset_context()

    def test_delta_quant_sparse_store_uses_raw_space_only_outside_staging(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False

        postrope_k = torch.ones((2, 1, 4))
        value = torch.full((2, 1, 4), 2.0)
        raw_k = torch.full((2, 1, 4), 3.0)
        raw_v = torch.full((2, 1, 4), 4.0)

        store_k, store_v = DeltaKVLessMemoryCacheManager.get_layer_store_tensors(
            manager,
            1,
            k_post_rope=postrope_k,
            v=value,
            pre_rope_k=raw_k,
            pre_rope_v=raw_v,
        )
        self.assertIs(store_k, raw_k)
        self.assertIs(store_v, raw_v)

        full_k, full_v = DeltaKVLessMemoryCacheManager.get_layer_store_tensors(
            manager,
            0,
            k_post_rope=postrope_k,
            v=value,
            pre_rope_k=raw_k,
            pre_rope_v=raw_v,
        )
        self.assertIs(full_k, postrope_k)
        self.assertIs(full_v, value)

        manager.has_prefill_staging_view = lambda layer_idx: True
        staging_k, staging_v = DeltaKVLessMemoryCacheManager.get_layer_store_tensors(
            manager,
            1,
            k_post_rope=postrope_k,
            v=value,
            pre_rope_k=raw_k,
            pre_rope_v=raw_v,
        )
        self.assertIs(staging_k, postrope_k)
        self.assertIs(staging_v, value)

    def test_delta_quant_sparse_store_uses_explicit_pre_rope_state(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager
        from sparseengine.utils.context import reset_context, set_context

        reset_context()
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False

        postrope_k = torch.ones((2, 1, 4))
        value = torch.full((2, 1, 4), 2.0)
        raw_k = torch.full((2, 1, 4), 3.0)
        raw_v = torch.full((2, 1, 4), 4.0)

        set_context(is_prefill=True, cache_manager=manager)
        store_k, store_v = DeltaKVLessMemoryCacheManager.get_layer_store_tensors(
            manager,
            1,
            k_post_rope=postrope_k,
            v=value,
            pre_rope_k=raw_k,
            pre_rope_v=raw_v,
        )

        self.assertIs(store_k, raw_k)
        self.assertIs(store_v, raw_v)
        reset_context()

    def test_delta_quant_storage_hooks_separate_raw_and_rope_paths(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False
        manager._prefill_pre_rope_stage_active = lambda: False
        slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
        stores = []
        raw_hooks = []
        rope_hooks = []
        manager._store_layer_kv = lambda layer_idx, k, v: stores.append((layer_idx, k, v)) or slot_mapping
        manager.on_pre_rope_kv_stored = (
            lambda layer_idx, k, v, slots: raw_hooks.append((layer_idx, k, v, slots))
        )
        manager.on_kv_stored = (
            lambda layer_idx, k, slots, **kwargs: rope_hooks.append((layer_idx, k, slots, kwargs))
        )

        postrope_k = torch.ones((2, 1, 4))
        value = torch.full((2, 1, 4), 2.0)
        raw_k = torch.full((2, 1, 4), 3.0)
        raw_v = torch.full((2, 1, 4), 4.0)

        DeltaKVLessMemoryCacheManager.save_raw_kv_if_needed(manager, 1, raw_k, raw_v)
        DeltaKVLessMemoryCacheManager.save_rope_kv_if_needed(manager, 1, postrope_k, value)

        self.assertEqual(len(stores), 1)
        self.assertEqual(stores[0][0], 1)
        self.assertIs(stores[0][1], raw_k)
        self.assertIs(stores[0][2], raw_v)
        self.assertEqual(len(raw_hooks), 1)
        self.assertEqual(len(rope_hooks), 0)

        DeltaKVLessMemoryCacheManager.save_rope_kv_if_needed(manager, 0, postrope_k, value)

        self.assertEqual(len(stores), 2)
        self.assertEqual(stores[1][0], 0)
        self.assertIs(stores[1][1], postrope_k)
        self.assertIs(stores[1][2], value)
        self.assertEqual(len(rope_hooks), 1)

    def test_full_layer_kivi_prefill_compute_uses_high_precision_staging(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager
        from sparseengine.utils.context import reset_context

        reset_context()
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.full_layer_to_idx = {0: 0}
        manager._full_layer_kivi_enabled = lambda: True
        manager.has_prefill_staging_view = lambda layer_idx: True

        postrope_k = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        value = torch.arange(100, 116, dtype=torch.float32).view(4, 1, 4)
        manager.deltakv_prefill_staging_kv_cache = [postrope_k.clone(), value.clone()]

        k_compute, v_compute = DeltaKVLessMemoryCacheManager.get_layer_compute_tensors(manager, 0)

        self.assertIs(k_compute, manager.deltakv_prefill_staging_kv_cache[0])
        self.assertIs(v_compute, manager.deltakv_prefill_staging_kv_cache[1])
        torch.testing.assert_close(manager.deltakv_prefill_staging_kv_cache[0], postrope_k)
        torch.testing.assert_close(manager.deltakv_prefill_staging_kv_cache[1], value)
        reset_context()

    def test_delta_quant_materializes_raw_sparse_cache_for_attention(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager
        from sparseengine.layers.rotary_embedding import apply_rotary_emb

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False
        manager.num_kv_heads = 1
        manager.head_dim = 4
        manager.deltakv_materialized_compute_num_slots = 4
        manager.deltakv_full_kv_cache = torch.empty((2, 1, 4, 1, 4), dtype=torch.float32)
        manager.deltakv_full_kv_cache[0, 0] = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0]],
                [[0.5, 0.5, 1.0, 0.0]],
                [[0.0, 1.0, 0.5, 0.5]],
                [[1.0, 1.0, 1.0, 1.0]],
            ]
        )
        manager.deltakv_full_kv_cache[1, 0] = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        manager.deltakv_slot_to_pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        cos = torch.tensor(
            [
                [1.0, 1.0],
                [0.9, 0.8],
                [0.7, 0.6],
                [0.5, 0.4],
            ],
            dtype=torch.float32,
        )
        sin = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            dtype=torch.float32,
        )
        manager.cos_sin_cache = torch.cat([cos, sin], dim=-1).unsqueeze(1)
        manager.deltakv_materialized_kv_cache = torch.empty((2, 4, 1, 4), dtype=torch.float32)
        manager._deltakv_materialized_active_slots = None
        manager._deltakv_materialized_local_req = None

        active_slots = torch.tensor([[2, 1]], dtype=torch.int32)
        context_lens = torch.tensor([2], dtype=torch.int32)
        k_cache, v_cache, local_active, local_req, out_lens = DeltaKVLessMemoryCacheManager.get_layer_compute_view(
            manager,
            1,
            active_slots=active_slots,
            req_indices=torch.tensor([7], dtype=torch.int32),
            context_lens=context_lens,
            selection=None,
        )

        raw_k = manager.deltakv_full_kv_cache[0, 0, active_slots.reshape(-1).long()]
        raw_v = manager.deltakv_full_kv_cache[1, 0, active_slots.reshape(-1).long()]
        pos = manager.deltakv_slot_to_pos[active_slots.reshape(-1).long()].long()
        cos_sin = manager.cos_sin_cache[pos]
        expected_cos, expected_sin = cos_sin.chunk(2, dim=-1)
        expected_k = apply_rotary_emb(raw_k, expected_cos, expected_sin)

        self.assertTrue(torch.equal(local_active, torch.tensor([[0, 1]], dtype=torch.int32)))
        self.assertTrue(torch.equal(local_req, torch.tensor([0], dtype=torch.int32)))
        self.assertIs(out_lens, context_lens)
        self.assertTrue(torch.allclose(k_cache, expected_k))
        self.assertTrue(torch.equal(v_cache, raw_v))

    def test_delta_quant_materialized_view_does_not_rerope_postrope_slots(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager
        from sparseengine.layers.rotary_embedding import apply_rotary_emb

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False
        manager.num_kv_heads = 1
        manager.head_dim = 4
        manager.deltakv_materialized_compute_num_slots = 4
        manager.deltakv_full_kv_cache = torch.empty((2, 1, 4, 1, 4), dtype=torch.float32)
        manager.deltakv_full_kv_cache[0, 0] = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0]],
                [[0.5, 0.5, 1.0, 0.0]],
                [[0.0, 1.0, 0.5, 0.5]],
                [[1.0, 1.0, 1.0, 1.0]],
            ]
        )
        manager.deltakv_full_kv_cache[1, 0] = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        manager.deltakv_slot_to_pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        cos = torch.tensor(
            [
                [1.0, 1.0],
                [0.9, 0.8],
                [0.7, 0.6],
                [0.5, 0.4],
            ],
            dtype=torch.float32,
        )
        sin = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            dtype=torch.float32,
        )
        manager.cos_sin_cache = torch.cat([cos, sin], dim=-1).unsqueeze(1)
        manager.deltakv_materialized_kv_cache = torch.empty((2, 4, 1, 4), dtype=torch.float32)
        manager._deltakv_materialized_active_slots = None
        manager._deltakv_materialized_local_req = None
        manager._deltakv_postrope_slot_mask = torch.zeros((1, 4), dtype=torch.bool)
        manager._deltakv_postrope_slot_mask[0, 1] = True

        active_slots = torch.tensor([[2, 1]], dtype=torch.int32)
        context_lens = torch.tensor([2], dtype=torch.int32)
        k_cache, _, _, _, _ = DeltaKVLessMemoryCacheManager.get_layer_compute_view(
            manager,
            1,
            active_slots=active_slots,
            req_indices=torch.tensor([7], dtype=torch.int32),
            context_lens=context_lens,
            selection=None,
        )

        raw_k = manager.deltakv_full_kv_cache[0, 0, active_slots.reshape(-1).long()]
        pos = manager.deltakv_slot_to_pos[active_slots.reshape(-1).long()].long()
        cos_sin = manager.cos_sin_cache[pos]
        expected_cos, expected_sin = cos_sin.chunk(2, dim=-1)
        expected_raw_rope = apply_rotary_emb(raw_k, expected_cos, expected_sin)
        expected = expected_raw_rope.clone()
        expected[1] = raw_k[1]

        self.assertTrue(torch.allclose(k_cache, expected))


class DeltaKVStaticDecodeRobustnessTest(unittest.TestCase):
    def test_no_graph_static_workspace_accepts_auto_capture_sizes(self):
        manager = object.__new__(DeltaKVLessMemoryCudaGraphCacheManager)
        manager.config = SimpleNamespace(
            max_decoding_seqs=16,
            decode_graph_capture_sizes="auto",
        )

        self.assertEqual(manager._decode_graph_capture_size_capacity(3), 16)

    def test_kivi_short_batch_prefill_does_not_use_singleton_staging_slots(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            engine_prefill_chunk_size=8192,
            long_prefill_offload_threshold=8192,
        )
        manager.deltakv_layer_ids = [1]
        manager._full_layer_kivi_enabled = lambda: True
        manager._deltakv_less_memory_prepare_full_prefill_staging = False
        seq = seq_with_len(1024)
        seq.current_chunk_size = 1024

        self.assertFalse(manager._should_stage_full_layer_kivi_prefill(seq, 1024))

        manager._deltakv_less_memory_prepare_full_prefill_staging = True
        self.assertTrue(manager._should_stage_full_layer_kivi_prefill(seq, 1024))

    def test_kivi_short_batch_prefill_uses_disjoint_staging_ranges(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.device = torch.device("cpu")
        manager.deltakv_prefill_staging_num_slots = 8
        manager.seq_id_to_row = {}
        manager.free_rows = deque([0, 1])
        manager.row_seq_lens = np.zeros((2,), dtype=np.int32)
        manager.full_layer_slots_map = torch.full((2, 8), -1, dtype=torch.int32)
        first = seq_with_len(3)
        second = seq_with_len(2)
        first.current_chunk_size = 3
        second.current_chunk_size = 2
        manager._deltakv_less_memory_prepare_seqs = [first, second]
        manager._deltakv_less_memory_full_prefill_staging_offset = 0
        manager._should_stage_full_layer_kivi_prefill = lambda seq, size: True
        manager._should_use_long_prefill_offload_staging = lambda seqs: False

        first_slots = manager._allocate_full(first.seq_id, 3)
        second_slots = manager._allocate_full(second.seq_id, 2)

        self.assertTrue(torch.equal(first_slots, torch.tensor([0, 1, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(second_slots, torch.tensor([3, 4], dtype=torch.int32)))
        self.assertEqual(manager._deltakv_less_memory_full_prefill_staging_offset, 5)

    def test_kivi_short_batch_finalize_reads_each_plan_staging_range(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.device = torch.device("cpu")
        manager.head_dim = 8
        manager.config = SimpleNamespace(
            mlp_chunk_size=8,
            full_layer_kivi_group_size=8,
        )
        manager.full_layer_ids = [0]
        manager.full_layer_to_idx = {0: 0}
        manager.deltakv_prefill_staging_kv_cache = torch.stack(
            [
                torch.arange(6, dtype=torch.float32).view(6, 1, 1),
                (100 + torch.arange(6, dtype=torch.float32)).view(6, 1, 1),
            ]
        )
        manager.full_kv_cache = torch.zeros((2, 1, 4, 1, 1), dtype=torch.float32)
        manager._full_layer_kivi_full_prefill_materialized_layers = set()
        manager._full_layer_kivi_full_prefill_plans = {
            0: {
                "staging_start": 0,
                "keep_pos": torch.tensor([0, 1], dtype=torch.int32),
                "keep_slots": torch.tensor([0, 1], dtype=torch.int32),
                "block_slots": torch.empty((0,), dtype=torch.int32),
            },
            1: {
                "staging_start": 3,
                "keep_pos": torch.tensor([0, 1], dtype=torch.int32),
                "keep_slots": torch.tensor([2, 3], dtype=torch.int32),
                "block_slots": torch.empty((0,), dtype=torch.int32),
            },
        }

        manager._full_layer_kivi_materialize_full_prefill_layer(0)

        torch.testing.assert_close(
            manager.full_kv_cache[0, 0, :, 0, 0],
            torch.tensor([0.0, 1.0, 3.0, 4.0]),
        )
        torch.testing.assert_close(
            manager.full_kv_cache[1, 0, :, 0, 0],
            torch.tensor([100.0, 101.0, 103.0, 104.0]),
        )


if __name__ == "__main__":
    unittest.main()
