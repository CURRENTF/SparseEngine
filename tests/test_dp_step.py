"""DP steps must agree before GPU work, while caches remain replica-local."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sparseengine.engine.dp_step import coordinate_dp_step
from sparseengine.utils.context import get_context, reset_context


def test_mixed_prefill_uses_common_transport_capacity_without_padding_attention_to_prefill():
    graph = SimpleNamespace(
        _select_graph_batch_size=lambda count: 4, dp_batch_capacity=None
    )
    runner = SimpleNamespace(
        decode_graph_runner=graph,
        cache_manager=SimpleNamespace(),
        parallel_context=SimpleNamespace(attn_dp_rank=0),
        dp_control_group=object(),
        dp_control_buffer=torch.empty(3, dtype=torch.int64),
    )

    def reduce(control, op, group):
        assert control.tolist() == [4, 0, 0]
        control.copy_(torch.tensor([4, 37, 1]))

    try:
        with patch("sparseengine.engine.dp_step.dist.all_reduce", side_effect=reduce):
            assert coordinate_dp_step(runner, [object()], False)
        assert get_context().moe_token_capacity == 37
        assert get_context().moe_token_sizes == (4, 37)
        assert graph.dp_batch_capacity is None
    finally:
        reset_context()


def test_idle_replica_joins_active_graph_without_consulting_stale_cache_eager_flag():
    force_eager = Mock(side_effect=AssertionError("idle cache state consulted"))
    graph = SimpleNamespace(_select_graph_batch_size=Mock(), dp_batch_capacity=None)
    runner = SimpleNamespace(
        decode_graph_runner=graph,
        cache_manager=SimpleNamespace(decode_graph_force_eager=force_eager),
        parallel_context=SimpleNamespace(attn_dp_rank=0),
        dp_control_group=object(),
        dp_control_buffer=torch.empty(3, dtype=torch.int64),
    )

    def reduce(control, op, group):
        assert control.tolist() == [0, 0, 0]
        control.copy_(torch.tensor([0, 4, 0]))

    try:
        with patch("sparseengine.engine.dp_step.dist.all_reduce", side_effect=reduce):
            assert not coordinate_dp_step(runner, [], False)
        assert graph.dp_batch_capacity == get_context().moe_token_capacity == 4
        assert get_context().moe_token_sizes is None
    finally:
        reset_context()
