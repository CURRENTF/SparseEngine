import json
from types import SimpleNamespace
import unittest

import torch

from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import SparseController
from sparseengine.engine.sparse_methods import SparseStepContext
from sparseengine.utils.context import get_context, reset_context, set_context


class WorkerInfoTest(unittest.TestCase):
    def test_h2o_fusion_choice_survives_worker_metadata_serialization(self):
        """Keep fused and baseline runs distinguishable in exported metadata."""
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            model="model",
            hf_config=SimpleNamespace(),
            sparse_method="h2o",
            h2o_decode_eviction=True,
        )
        for enabled in (False, True):
            with self.subTest(fusion=enabled):
                engine.config.h2o_decode_score_fusion = enabled
                info = json.loads(json.dumps(engine.worker_info()))
                self.assertIs(
                    info["benchmark_config"]["h2o_decode_score_fusion"], enabled
                )

    def test_prefix_offload_capacity_config_is_reported_json_safely(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            model="model",
            hf_config=SimpleNamespace(model_type="test", vocab_size=32_000),
            sparse_method="",
            enable_prefix_caching=True,
            prefix_cache_block_size=16,
            prefix_cache_max_blocks=4_096,
            prefix_cache_requested_max_blocks=8_192,
            enable_prefix_cache_offload=True,
            prefix_cache_host_size_gb=6.5,
            recurrent_state_max_bytes=1 << 30,
            prefix_cache_max_recurrent_bytes=None,
            recurrent_state_pool_bytes=768 << 20,
            recurrent_state_bytes_per_row=16_384,
            recurrent_state_row_capacity=49_152,
            prefix_recurrent_bytes_per_block=16_384,
            prefix_recurrent_capacity_bytes=256 << 20,
            prefix_kv_bytes_per_block=65_536,
            prefix_kv_block_capacity=4_096,
            kv_allocatable_bytes=512 << 20,
            num_kvcache_slots=262_144,
            max_num_seqs_in_gpu=64,
        )

        worker_info = engine.worker_info()
        benchmark_config = worker_info["benchmark_config"]

        self.assertIs(benchmark_config["enable_prefix_cache_offload"], True)
        self.assertEqual(benchmark_config["prefix_cache_host_size_gb"], 6.5)
        self.assertEqual(benchmark_config["recurrent_state_max_bytes"], 1 << 30)
        self.assertIsNone(benchmark_config["prefix_cache_max_recurrent_bytes"])
        self.assertEqual(benchmark_config["prefix_cache_requested_max_blocks"], 8_192)
        self.assertEqual(benchmark_config["prefix_cache_max_blocks"], 4_096)
        self.assertEqual(benchmark_config["recurrent_state_pool_bytes"], 768 << 20)
        self.assertEqual(benchmark_config["recurrent_state_bytes_per_row"], 16_384)
        self.assertEqual(benchmark_config["recurrent_state_row_capacity"], 49_152)
        self.assertEqual(benchmark_config["prefix_recurrent_bytes_per_block"], 16_384)
        self.assertEqual(benchmark_config["prefix_recurrent_capacity_bytes"], 256 << 20)
        self.assertEqual(benchmark_config["prefix_kv_bytes_per_block"], 65_536)
        self.assertEqual(benchmark_config["prefix_kv_block_capacity"], 4_096)
        self.assertEqual(benchmark_config["kv_allocatable_bytes"], 512 << 20)
        self.assertEqual(benchmark_config["num_kvcache_slots"], 262_144)
        self.assertEqual(worker_info["max_num_seqs_in_gpu"], 64)
        json.dumps(worker_info)


def make_controller(
    method="snapkv",
    *,
    layers=2,
    kv_len=6,
    graph=False,
    graph_capacity=None,
    keep=2,
    score_dtype="float32",
    sink=1,
    recent=1,
    pyramid_ratios=None,
    eviction=False,
    interval=2,
):
    layout = SimpleNamespace(
        kv_layer_index=lambda layer: int(layer),
        is_full_attention=lambda layer: 0 <= int(layer) < layers,
    )
    config = SimpleNamespace(
        sparse_method=method,
        obs_layer_ids=[],
        full_attention_layers=[],
        hf_config=SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=8,
            num_attention_heads=2,
            dtype=torch.float32,
        ),
        runtime_layout=layout,
        sink_keep_tokens=sink,
        recent_keep_tokens=recent,
        decode_keep_tokens=keep,
        sparse_attn_score_dtype=score_dtype,
        tensor_parallel_size=1,
        snapkv_num_full_layers=0,
        snapkv_decode_eviction=eviction,
        decode_eviction_interval=interval,
        pyramid_layer_ratios=(
            pyramid_ratios
            if pyramid_ratios is not None
            else ([1.0] * layers if method == "pyramidkv" else None)
        ),
        decode_graph=graph,
        pool_kernel_size=1,
    )

    class Manager:
        device = torch.device("cpu")

        def __init__(self):
            self.compactions = []
            self.scored = []
            self.cleared = []
            if graph:
                self._decode_static_max_context_len = int(
                    graph_capacity if graph_capacity is not None else kv_len
                )

        def get_layer_batch_states(self, layer):
            del layer
            return SimpleNamespace(
                context_lens=torch.tensor([kv_len], dtype=torch.int32),
                max_context_len=kv_len,
                req_indices=torch.tensor([0], dtype=torch.int32),
            )

        def decode_kv_lens_for_layer(self, layer, seqs):
            del layer
            return [kv_len for _seq in seqs]

        def decode_query_scores(self, layer, seq, kv_len):
            self.scored.append((layer, seq.seq_id, kv_len))
            return torch.arange(kv_len, dtype=torch.float32)

        def clear_decode_query_history(self, layer, seq_id):
            self.cleared.append((layer, seq_id))

        def free_part_slots(self, layer, seq, keep_indices):
            self.compactions.append((layer, seq.seq_id, keep_indices.clone()))

    manager = Manager()
    controller = SparseController(config, manager)
    seqs = [Sequence([1])]
    set_context(
        False,
        cache_manager=manager,
        seqs=seqs,
    )
    controller.prepare_forward(seqs, is_prefill=False)
    return controller, manager, seqs


class SnapKVDecodeScoreLifecycleTest(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_disabled_snapkv_does_not_score_or_compact(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                controller, manager, seqs = make_controller(graph=graph)
                self.assertIsNone(controller.layer_batch_sparse_states[0].attn_score)
                self.assertFalse(controller.runtime.needs_attention_score(
                    0, SparseStepContext(seqs, False, get_context()),
                ))
                controller.post_forward(seqs, is_prefill=False)
                self.assertEqual(manager.scored, [])
                self.assertEqual(manager.compactions, [])

    def test_snapkv_scores_only_at_physical_eviction_boundary(self):
        below, manager, seqs = make_controller(
            "snapkv", layers=1, kv_len=5, keep=2, eviction=True,
            interval=2, graph=True, graph_capacity=16,
        )
        self.assertIsNone(below.layer_batch_sparse_states[0].attn_score)
        below.post_forward(seqs, is_prefill=False)
        self.assertEqual(manager.scored, [])

        due, manager, seqs = make_controller(
            "snapkv", layers=1, kv_len=6, keep=2, eviction=True,
            interval=2, graph=True, graph_capacity=16,
        )
        self.assertIsNone(due.get_decode_selection(
            0, torch.empty((1, 2, 4)),
        ).attn_score)
        due.post_forward(seqs, is_prefill=False)
        self.assertEqual(manager.scored, [(0, seqs[0].seq_id, 6)])
        self.assertEqual(len(manager.compactions), 1)
        self.assertEqual(manager.compactions[0][2].numel(), 4)

    def test_pyramidkv_scores_only_layers_at_their_physical_boundary(self):
        common = {
            "method": "pyramidkv", "layers": 2, "graph": True,
            "keep": 4_096, "sink": 64, "recent": 512,
            "pyramid_ratios": [0.6, 0.01], "interval": 68,
        }
        below, manager, seqs = make_controller(
            **common, kv_len=700, graph_capacity=1_024,
        )
        self.assertEqual(below.runtime._get_layer_budget(1, False), 644)
        self.assertEqual(below.runtime._snapkv_decode_trigger_len(644), 712)
        self.assertIsNone(below.layer_batch_sparse_states[1].attn_score)
        below.post_forward(seqs, is_prefill=False)
        self.assertEqual(manager.scored, [])

        due, manager, seqs = make_controller(
            **common, kv_len=712, graph_capacity=1_024,
        )
        due.post_forward(seqs, is_prefill=False)
        self.assertEqual(manager.scored, [(1, seqs[0].seq_id, 712)])
        self.assertEqual(len(manager.compactions), 1)
        self.assertEqual(manager.compactions[0][0], 1)
        self.assertEqual(manager.compactions[0][2].numel(), 644)

    def test_decode_score_shape_failure_is_explicit(self):
        controller, manager, seqs = make_controller("pyramidkv", layers=1)
        manager.decode_query_scores = lambda _layer, _seq, kv_len: torch.zeros(kv_len - 1)
        with self.assertRaisesRegex(RuntimeError, r"decode query scores must be \[B, L\]"):
            controller.post_forward(seqs, is_prefill=False)
        self.assertEqual(manager.compactions, [])

    def test_recompute_replay_does_not_pollute_decode_observations(self):
        controller, manager, seqs = make_controller(
            "snapkv", layers=1, eviction=True,
        )
        seqs[0].recompute_replay_cursor = 0
        controller.post_forward(seqs, is_prefill=False)
        self.assertEqual(manager.cleared, [(0, seqs[0].seq_id)])
        self.assertEqual(manager.scored, [])
        self.assertEqual(manager.compactions, [])


if __name__ == "__main__":
    unittest.main()
