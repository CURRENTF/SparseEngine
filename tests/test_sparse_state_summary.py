import os
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparseengine.engine.cache_manager.base import _debug_tensor_summary
from sparseengine.distributed import ParallelContext, ParallelGroup
from sparseengine.engine.model_runner import ModelRunner
from sparseengine.engine.sparse_controller import LayerBatchSparseState
from sparseengine.engine.sparse_methods.passthrough import PassThroughRuntime


def test_debug_tensor_summary_is_order_sensitive_and_deterministic():
    first = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    same = first.clone()
    reordered = torch.tensor([[1, 3], [2, 4]], dtype=torch.int32)

    assert _debug_tensor_summary(first) == _debug_tensor_summary(same)
    assert _debug_tensor_summary(first)["sha256"] != _debug_tensor_summary(reordered)[
        "sha256"
    ]


def test_debug_tensor_summary_supports_bfloat16_scalars():
    summary = _debug_tensor_summary(torch.tensor(1.5, dtype=torch.bfloat16))

    assert summary["shape"] == []
    assert summary["dtype"] == "torch.bfloat16"
    assert summary["numel"] == 1
    assert len(summary["sha256"]) == 64


def test_sparse_controller_summary_captures_selection_and_cache_state():
    runtime = object.__new__(PassThroughRuntime)
    runtime.sparse_method = "quest"
    runtime.layer_batch_sparse_states = {
        0: LayerBatchSparseState(
            active_indices=torch.tensor([[0, 2]], dtype=torch.int64),
            active_slots=torch.tensor([[7, 9]], dtype=torch.int32),
            context_lens=torch.tensor([2], dtype=torch.int32),
            max_context_len=2,
        ),
        1: LayerBatchSparseState(),
    }
    runtime.debug_dynamic_selection = {"quest": {"0": {"calls": 1}}}
    runtime.cache_manager = SimpleNamespace(
        debug_state_summary=lambda: {
            "cache_manager_class": "FakeCacheManager",
            "free_slot_stats": {"free_slots": 12},
        }
    )

    summary = runtime.debug_state_summary()

    assert summary["sparse_method"] == "quest"
    assert list(summary["layers"]) == ["0"]
    assert summary["layers"]["0"]["tensors"]["active_indices"]["shape"] == [1, 2]
    assert summary["dynamic_selection"]["quest"]["0"]["calls"] == 1
    assert summary["cache"]["free_slot_stats"] == {"free_slots": 12}


def test_model_runner_gathers_one_debug_summary_per_world_rank():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.rank = 0
    process_group = object()
    world_group = ParallelGroup(
        process_group=process_group,
        ranks=(0, 1),
        rank=0,
        size=2,
    )
    singleton_group = ParallelGroup(
        process_group=None,
        ranks=(0,),
        rank=0,
        size=1,
    )
    runner.parallel_context = ParallelContext(
        world=world_group,
        attn_tp=world_group,
        moe_ep=world_group,
        attn_dp=singleton_group,
        moe_tp=singleton_group,
    )
    runner.sparse_controller = SimpleNamespace(
        debug_state_summary=lambda: {"sparse_method": "", "layers": {}}
    )
    runner.config = SimpleNamespace(decode_graph=True)
    runner.decode_graph_runner = SimpleNamespace(
        capture_count=1,
        replay_count=3,
        eager_static_count=0,
        force_eager_count=0,
        _graphs={"graph": object()},
        last_state_key=SimpleNamespace(
            method="snapkv",
            batch_size=2,
            graph_path_id="unified",
            capture_sampling=False,
        ),
        graph_plan=lambda: {
            "batch_sizes": [2],
            "startup_plan_sealed": True,
            "cached_graph_keys": [],
        },
    )

    def gather(output, local, group):
        assert group is runner.parallel_context.attn_tp.process_group
        output[:] = [local, {"world_rank": 1, "ep_rank": 1, "state": local["state"]}]

    with (
        patch.object(runner, "_sync_tp_rpc_status") as sync_status,
        patch("sparseengine.engine.model_runner.dist.all_gather_object", side_effect=gather),
    ):
        summaries = runner.debug_sparse_state_summaries()

    assert [summary["world_rank"] for summary in summaries] == [0, 1]
    assert summaries[0]["state"] == summaries[1]["state"]
    expected_graph = {
        "enabled": True,
        "capture_count": 1,
        "replay_count": 3,
        "eager_static_count": 0,
        "force_eager_count": 0,
        "eviction_count": 0,
        "recapture_count": 0,
        "cached_graph_count": 1,
        "graph_plan": {
            "batch_sizes": [2],
            "startup_plan_sealed": True,
            "cached_graph_keys": [],
        },
        "last_state_key": {
            "method": "snapkv",
            "batch_size": 2,
            "graph_path_id": "unified",
            "capture_sampling": False,
        },
    }
    assert {key: summaries[0]["decode_graph"][key] for key in expected_graph} == expected_graph
    sync_status.assert_called_once_with("debug_sparse_state_summaries", None)


def test_tp_debug_replica_consistency_marks_vocab_sharded_logits_not_applicable():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.parallel_context = ParallelContext(
        world=ParallelGroup(process_group=object(), ranks=(0, 1), rank=1, size=2),
        moe_tp=ParallelGroup(process_group=object(), ranks=(0, 1), rank=1, size=2),
        attn_tp=ParallelGroup(process_group=object(), ranks=(0, 1), rank=1, size=2),
        moe_ep=ParallelGroup(process_group=None, ranks=(1,), rank=0, size=1),
        attn_dp=ParallelGroup(process_group=None, ranks=(1,), rank=0, size=1),
    )
    runner.model = SimpleNamespace(model=SimpleNamespace(layers=()))

    consistency = runner.debug_replica_consistency()

    assert consistency == {
        "last_logits_max_abs": None,
        "last_logits_tolerance_ratio": None,
        "last_logits_comparison": "not_applicable_tp_vocab_sharded",
        "moe_layers": {},
    }


def test_nonzero_rank_debug_logits_rpc_returns_none_before_tensor_access():
    runner = object.__new__(ModelRunner)
    runner.parallel_context = SimpleNamespace(attn_tp_rank=1)

    assert runner.debug_last_logits_cpu() is None


def test_run_model_does_not_capture_non_tensor_tp_logits():
    runner = object.__new__(ModelRunner)

    class FakeModel:
        def __call__(self, input_ids, positions):
            return torch.ones(1)

        def compute_logits(self, hidden_states):
            return None

    runner.model = FakeModel()
    with patch.dict(os.environ, {"SPARSEENGINE_DEBUG_RUNTIME": "1"}):
        logits = runner.run_model(torch.ones(1), torch.ones(1), is_prefill=False)

    assert logits is None
    assert not hasattr(runner, "debug_last_logits")


def test_debug_logits_can_be_refreshed_after_cuda_graph_replay():
    runner = object.__new__(ModelRunner)
    capture_value = torch.tensor([[1.0, 2.0]])
    replay_value = torch.tensor([[3.0, 4.0]])

    with patch.dict(os.environ, {"SPARSEENGINE_DEBUG_RUNTIME": "1"}):
        runner._record_debug_logits(capture_value)
        runner._record_debug_logits(replay_value)

    torch.testing.assert_close(runner.debug_last_logits, replay_value)
    assert runner.debug_last_logits.data_ptr() != replay_value.data_ptr()
