from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sparseengine.distributed.collective_runtime import (
    ParallelCollectiveRuntime,
    ParallelCollectiveState,
)
from sparseengine.distributed.parallel_context import (
    ParallelContext,
    ParallelGroup,
    init_parallel_context,
    reset_parallel_context,
)
from sparseengine.distributed.topology import ParallelTopology
from sparseengine.utils.context import reset_context, set_context


def _group(ranks=(0,), rank=0, process_group=None):
    return ParallelGroup(process_group, tuple(ranks), rank, len(ranks))


def _context(*, world=None, attention=None):
    world = _group() if world is None else world
    attention = world if attention is None else attention
    singleton = _group()
    return ParallelContext(
        world=world,
        attn_tp=attention,
        moe_ep=singleton,
        attn_dp=singleton,
        moe_tp=attention,
    )


def _prepared_op(name="fake", metadata=None):
    return SimpleNamespace(
        name=name,
        collect_local_cuda_graph_metadata=Mock(return_value=metadata),
        graph_metadata_summary=Mock(return_value=(0, 0)),
        register_cuda_graph_buffers=Mock(),
        close=Mock(),
        run=Mock(side_effect=lambda tensor: tensor),
    )


def _hybrid_collective_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    runtime = None
    try:
        context = init_parallel_context(topology=ParallelTopology(2, 2, 2))
        # Verify every stage axis against its global member identities. Nonzero
        # group-local sources catch accidental use of local ranks as world ranks.
        for group in (context.attn_tp, context.attn_dp, context.moe_tp, context.moe_ep):
            value = torch.tensor([float(rank)])
            group.all_reduce(value)
            assert value.item() == sum(group.ranks)
            value.fill_(rank)
            group.broadcast(value, src_rank=group.size - 1)
            assert value.item() == group.ranks[-1]
            gathered = group.gather(torch.tensor([rank]), dst_rank=group.size - 1)
            if group.rank == group.size - 1:
                assert [item.item() for item in gathered] == list(group.ranks)
            else:
                assert gathered is None
        runtime = ParallelCollectiveRuntime(
            context,
            cuda_graph=True,
            device_index=0,
        )
        collectives = runtime.request_decode_collectives(
            attention_max_rows=1,
            moe_max_rows=1,
            hidden_size=1,
            dtype=torch.float32,
        )
        runtime.prepare()

        runtime.begin_cuda_graph_capture()
        runtime.collect_local_cuda_graph_metadata()
        runtime.exchange_cuda_graph_metadata()
        runtime.register_cuda_graph_buffers()
        runtime.mark_cuda_graph_replayable()
        runtime.assert_cuda_graph_replayable()

        set_context(False)
        local = torch.tensor([[float(rank + 1)]])
        attention = collectives.attention.run(local.clone())
        moe = collectives.moe.run(local.clone())
        assert attention.item() == (3.0 if rank < 2 else 7.0)
        assert moe.item() == 10.0
    finally:
        if runtime is not None:
            runtime.close()
        reset_context()
        reset_parallel_context()
        dist.destroy_process_group()


def test_hybrid_subgroup_and_world_lifecycle_uses_one_global_order(tmp_path):
    rendezvous = tmp_path / "hybrid-collective-rendezvous"
    mp.spawn(
        _hybrid_collective_worker,
        args=(4, f"file://{rendezvous}"),
        nprocs=4,
        join=True,
    )


def test_same_group_requests_share_one_operator_with_largest_capacity():
    world = _group((0, 1), process_group=object())
    context = _context(world=world, attention=world)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)

    collectives = runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=16,
        hidden_size=2048,
        dtype=torch.bfloat16,
    )
    op = _prepared_op()
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=op,
    ) as prepare:
        runtime.prepare()

    assert collectives.attention is collectives.moe
    prepare.assert_called_once_with(
        context.world,
        max_rows=16,
        hidden_size=2048,
        dtype=torch.bfloat16,
        cuda_graph=True,
        device_index=0,
    )


def test_attention_subgroup_and_world_group_prepare_distinct_operators():
    world = _group((0, 1, 2, 3), process_group=object())
    attention = _group((0, 1), process_group=object())
    context = _context(world=world, attention=attention)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=4)
    collectives = runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=16,
        hidden_size=2048,
        dtype=torch.bfloat16,
    )

    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        side_effect=(_prepared_op("attention"), _prepared_op("world")),
    ) as prepare:
        runtime.prepare()

    assert collectives.attention is not collectives.moe
    assert [call.args[0] for call in prepare.call_args_list] == [attention, world]


def test_expert_parallel_attention_identity_only_prepares_world_operator():
    world = _group((0, 1), process_group=object())
    attention = _group()
    context = _context(world=world, attention=attention)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=4)
    collectives = runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=16,
        hidden_size=2048,
        dtype=torch.bfloat16,
    )
    op = _prepared_op("world")

    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=op,
    ) as prepare:
        runtime.prepare()

    tensor = torch.ones((1, 2048), dtype=torch.bfloat16)
    assert collectives.attention.run(tensor) is tensor
    assert collectives.attention.name == "identity"
    assert collectives.moe.name == "world"
    prepare.assert_called_once_with(
        world,
        max_rows=16,
        hidden_size=2048,
        dtype=torch.bfloat16,
        cuda_graph=True,
        device_index=4,
    )


def test_single_rank_collective_is_identity_without_prepared_operator():
    context = _context()
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    collectives = runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=16,
        hidden_size=128,
        dtype=torch.float16,
    )

    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce"
    ) as prepare:
        runtime.prepare()

    tensor = torch.ones((2, 128), dtype=torch.float16)
    assert collectives.attention is collectives.moe
    assert collectives.attention.run(tensor) is tensor
    assert collectives.attention.name == "identity"
    assert not runtime.has_graph_collectives
    prepare.assert_not_called()


def test_cuda_graph_replay_is_blocked_until_registration_finishes():
    world = _group((0, 1), process_group=object())
    context = _context(world=world, attention=world)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=128,
        dtype=torch.float16,
    )
    op = _prepared_op()
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=op,
    ):
        runtime.prepare()

    runtime.begin_cuda_graph_capture()
    with pytest.raises(RuntimeError, match="blocked"):
        runtime.assert_cuda_graph_replayable()
    runtime.assert_can_capture()
    runtime.collect_local_cuda_graph_metadata()
    with patch.object(
        runtime,
        "_all_gather_object",
        side_effect=lambda local, group: [local] * group.size,
    ):
        runtime.exchange_cuda_graph_metadata()
    runtime.register_cuda_graph_buffers()
    with pytest.raises(RuntimeError, match="blocked"):
        runtime.assert_cuda_graph_replayable()
    runtime.mark_cuda_graph_replayable()
    runtime.assert_cuda_graph_replayable()
    with pytest.raises(RuntimeError, match="capture is closed"):
        runtime.assert_can_capture()

    op.collect_local_cuda_graph_metadata.assert_called_once_with()
    op.register_cuda_graph_buffers.assert_called_once_with([None, None])


def test_completed_collectives_can_reset_for_production_graph_recapture():
    world = _group((0, 1), process_group=object())
    context = _context(world=world, attention=world)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    handle = runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=128,
        dtype=torch.float16,
    ).attention
    profiling_op = _prepared_op("profiling")
    production_op = _prepared_op("production")
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        side_effect=(profiling_op, production_op),
    ) as prepare:
        runtime.prepare()
        runtime.begin_cuda_graph_capture()
        runtime.collect_local_cuda_graph_metadata()
        with patch.object(
            runtime,
            "_all_gather_object",
            side_effect=lambda local, group: [local] * group.size,
        ):
            runtime.exchange_cuda_graph_metadata()
        runtime.register_cuda_graph_buffers()
        runtime.mark_cuda_graph_replayable()

        runtime.reset_for_cuda_graph_recapture()

    assert runtime.state is ParallelCollectiveState.PREPARED
    assert handle.name == "production"
    assert prepare.call_count == 2
    profiling_op.close.assert_called_once_with()
    production_op.close.assert_not_called()


def test_uncaptured_collectives_need_no_reset_before_runtime_rebuild():
    world = _group((0, 1), process_group=object())
    runtime = ParallelCollectiveRuntime(
        _context(world=world, attention=world),
        cuda_graph=True,
        device_index=0,
    )
    runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=128,
        dtype=torch.float16,
    )
    op = _prepared_op()
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=op,
    ):
        runtime.prepare()

    runtime.reset_for_cuda_graph_recapture()

    assert runtime.state is ParallelCollectiveState.PREPARED
    op.close.assert_not_called()


def test_handle_uses_plain_collective_for_prefill_and_prepared_op_for_decode():
    world = _group((0, 1), process_group=object())
    context = _context(world=world, attention=world)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=False, device_index=0)
    handle = runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=4,
        dtype=torch.float32,
    ).attention
    op = _prepared_op()
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=op,
    ):
        runtime.prepare()
    tensor = torch.ones((2, 4))

    with (
        patch(
            "sparseengine.distributed.collective_runtime.get_context",
            return_value=SimpleNamespace(is_prefill=True),
        ),
        patch.object(
            ParallelGroup,
            "all_reduce",
            return_value=tensor,
        ) as eager_reduce,
    ):
        assert handle.run(tensor) is tensor
    eager_reduce.assert_called_once_with(tensor)
    op.run.assert_not_called()

    with patch(
        "sparseengine.distributed.collective_runtime.get_context",
        return_value=SimpleNamespace(is_prefill=False),
    ):
        assert handle.run(tensor) is tensor
    op.run.assert_called_once_with(tensor)


def test_provider_mismatch_stops_before_any_subgroup_exchange():
    world = _group((0, 1), process_group=object())
    context = _context(world=world, attention=world)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=128,
        dtype=torch.float16,
    )
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=_prepared_op("provider-a"),
    ):
        runtime.prepare()
    runtime.begin_cuda_graph_capture()
    runtime.collect_local_cuda_graph_metadata()
    calls = []

    def gather(local, group):
        calls.append((local, group))
        entry = local[0]
        peer = (entry[:-1] + ("provider-b",),)
        return [local, peer]

    with (
        patch.object(runtime, "_all_gather_object", side_effect=gather),
        pytest.raises(RuntimeError, match="incompatible all-reduce schedules"),
    ):
        runtime.exchange_cuda_graph_metadata()

    assert len(calls) == 1
    assert calls[0][1] is world
    assert runtime.state is ParallelCollectiveState.CAPTURED


def test_inconsistent_subgroup_membership_stops_before_subgroup_exchange():
    world = _group((0, 1, 2, 3), process_group=object())
    attention = _group((0, 1), process_group=object())
    context = _context(world=world, attention=attention)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=128,
        dtype=torch.float16,
    )
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        side_effect=(_prepared_op("attention"), _prepared_op("world")),
    ):
        runtime.prepare()
    runtime.begin_cuda_graph_capture()
    runtime.collect_local_cuda_graph_metadata()
    calls = []

    def gather(local, group):
        calls.append((local, group))
        assert group is world
        attention_entry, world_entry = local

        def peer_schedule(attention_ranks):
            return (
                (
                    attention_entry[0],
                    attention_ranks,
                    *attention_entry[2:],
                ),
                world_entry,
            )

        return [
            peer_schedule((0, 1)),
            peer_schedule((1, 2)),
            peer_schedule((2, 3)),
            peer_schedule((2, 3)),
        ]

    with (
        patch.object(runtime, "_all_gather_object", side_effect=gather),
        pytest.raises(RuntimeError, match="incompatible all-reduce schedules"),
    ):
        runtime.exchange_cuda_graph_metadata()

    assert len(calls) == 1
    assert runtime.state is ParallelCollectiveState.CAPTURED


def test_metadata_count_mismatch_is_reported_after_required_exchange():
    world = _group((0, 1), process_group=object())
    context = _context(world=world, attention=world)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=128,
        dtype=torch.float16,
    )
    op = _prepared_op()
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=op,
    ):
        runtime.prepare()
    runtime.begin_cuda_graph_capture()
    runtime.collect_local_cuda_graph_metadata()

    calls = []

    def gather(local, group):
        calls.append((local, group))
        if len(calls) == 1:
            return [local, local]
        peer_fingerprint = local[0][:-1] + ((1, 1),)
        peer = (peer_fingerprint, local[1])
        return [local, peer]

    with patch.object(runtime, "_all_gather_object", side_effect=gather):
        with pytest.raises(RuntimeError, match="incompatible"):
            runtime.exchange_cuda_graph_metadata()

    assert len(calls) == 2
    assert runtime.state is ParallelCollectiveState.CAPTURED
    op.register_cuda_graph_buffers.assert_not_called()


def test_local_registration_failure_never_marks_runtime_replayable():
    world = _group((0, 1), process_group=object())
    context = _context(world=world, attention=world)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=128,
        dtype=torch.float16,
    )
    op = _prepared_op()
    op.register_cuda_graph_buffers.side_effect = RuntimeError("registration failed")
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=op,
    ):
        runtime.prepare()
    runtime.begin_cuda_graph_capture()
    runtime.collect_local_cuda_graph_metadata()
    with patch.object(
        runtime,
        "_all_gather_object",
        side_effect=lambda local, group: [local] * group.size,
    ):
        runtime.exchange_cuda_graph_metadata()

    with pytest.raises(RuntimeError, match="registration failed"):
        runtime.register_cuda_graph_buffers()
    assert runtime.state is ParallelCollectiveState.EXCHANGED
    with pytest.raises(RuntimeError, match="registered collective buffers"):
        runtime.mark_cuda_graph_replayable()


def test_local_metadata_failure_cannot_start_registration_exchange():
    world = _group((0, 1), process_group=object())
    context = _context(world=world, attention=world)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=128,
        dtype=torch.float16,
    )
    op = _prepared_op()
    op.collect_local_cuda_graph_metadata.side_effect = RuntimeError("local failure")
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=op,
    ):
        runtime.prepare()
    runtime.begin_cuda_graph_capture()

    with pytest.raises(RuntimeError, match="local failure"):
        runtime.collect_local_cuda_graph_metadata()
    assert runtime.state is ParallelCollectiveState.CAPTURING
    with pytest.raises(RuntimeError, match="completed local capture"):
        runtime.exchange_cuda_graph_metadata()


def test_local_metadata_validation_failure_cannot_start_exchange():
    world = _group((0, 1), process_group=object())
    context = _context(world=world, attention=world)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    runtime.request_decode_collectives(
        attention_max_rows=8,
        moe_max_rows=8,
        hidden_size=128,
        dtype=torch.float16,
    )
    op = _prepared_op()
    op.graph_metadata_summary.side_effect = RuntimeError("invalid local metadata")
    with patch(
        "sparseengine.distributed.collective_runtime.prepare_parallel_all_reduce",
        return_value=op,
    ):
        runtime.prepare()
    runtime.begin_cuda_graph_capture()

    with pytest.raises(RuntimeError, match="invalid local metadata"):
        runtime.collect_local_cuda_graph_metadata()
    assert runtime.state is ParallelCollectiveState.CAPTURING
    with pytest.raises(RuntimeError, match="completed local capture"):
        runtime.exchange_cuda_graph_metadata()
