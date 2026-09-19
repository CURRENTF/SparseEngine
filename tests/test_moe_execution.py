from unittest.mock import Mock

import pytest
import torch
from torch import nn

from sparsevllm.distributed.moe_communication import AllReduceMoeCommunication
from sparsevllm.operators.moe_execution import MoeExecutionPlan, prepare_model_moe_execution


def test_fused_execution_does_not_execute_shared_twice():
    routed, shared = Mock(), Mock()
    fused = Mock(side_effect=lambda x: x * 3)
    plan = MoeExecutionPlan(routed=routed, shared=shared, fused=fused,
                            communication=AllReduceMoeCommunication(lambda x: x),
                            chunk_size=2, fuse_prefill=True, fuse_decode=True)
    x = torch.arange(20.).reshape(5, 4)
    torch.testing.assert_close(plan(x, is_prefill=False), 3 * x)
    assert fused.call_count == 3
    routed.assert_not_called()
    shared.assert_not_called()


@pytest.mark.parametrize('separate', [False, True])
def test_composition_preserves_reduction_order(separate):
    # Deliberately nonlinear rounding distinguishes sum-before-reduce from
    # reducing each branch. This checks the contract, not a tuning constant.
    reduce = lambda x: (x * 1.3).round()
    x = torch.tensor([[.4, .7]])
    plan = MoeExecutionPlan(routed=lambda x: x, shared=lambda x: x * .7,
                            communication=AllReduceMoeCommunication(reduce),
                            chunk_size=4, reduce_decode_branches_separately=separate)
    expected = reduce(x) + reduce(x * .7) if separate else reduce(x + x * .7)
    torch.testing.assert_close(plan(x, is_prefill=False), expected)
    torch.testing.assert_close(plan(x, is_prefill=True), reduce(x + x * .7))


def test_fusion_limit_uses_separate_branches_above_limit():
    fused = Mock(side_effect=AssertionError('unexpected fusion'))
    plan = MoeExecutionPlan(routed=lambda x: x, shared=lambda x: x * 2,
                            fused=fused, fuse_decode=True, fusion_token_limit=2,
                            communication=AllReduceMoeCommunication(lambda x: x),
                            chunk_size=2)
    x = torch.ones(3, 4)
    torch.testing.assert_close(plan(x, is_prefill=False), x * 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_parallel_eager_and_shared_pool_graphs_against_reference(dtype):
    torch.manual_seed(34)
    device = torch.device('cuda', 0)
    a = torch.randn(128, 64, device=device, dtype=dtype) / 16
    b = torch.randn(64, 128, device=device, dtype=dtype) / 8
    c = torch.randn(128, 64, device=device, dtype=dtype) / 16
    d = torch.randn(64, 128, device=device, dtype=dtype) / 8
    model = nn.Module()
    for i in range(2):
        layer = nn.Module()
        layer.moe_execution = MoeExecutionPlan(
            routed=lambda x: torch.relu(x @ a) @ b,
            shared=lambda x: torch.sigmoid(x @ c) @ d,
            communication=AllReduceMoeCommunication(lambda x: x), chunk_size=32)
        model.add_module(str(i), layer)
    stream = prepare_model_moe_execution(model, device)
    assert stream is not None
    plans = [m.moe_execution for m in model.children()]
    assert plans[0].stream is plans[1].stream
    pool = torch.cuda.graph_pool_handle()
    graphs = []
    def reference(x):
        return (torch.relu(x.float() @ a.float()) @ b.float()
                + torch.sigmoid(x.float() @ c.float()) @ d.float())
    tolerance = 1e-5 if dtype == torch.float32 else .025
    with torch.inference_mode():
        # The same layer/events participate in multiple batch-capacity graphs.
        for batch in [1, 16]:
            plan = plans[0]
            x = torch.randn(batch, 128, device=device, dtype=dtype)
            for _ in range(3):
                y = plan(x, is_prefill=False)
            torch.cuda.synchronize()
            torch.testing.assert_close(y.float(), reference(x), atol=tolerance, rtol=tolerance)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                y = plan(x, is_prefill=False)
            graphs.append((g, x, y))
        # Reuse pool in capture order, consume output before the next graph;
        # repeatedly change inputs and create allocator pressure without a
        # device synchronize between submissions.
        for iteration in range(12):
            results = []
            for g, x, y in graphs:
                x.fill_(iteration / 10)
                g.replay()
                results.append((y.clone(), reference(x)))
                scratch = torch.empty(8192, device=device, dtype=dtype)
                scratch.fill_(13)
            for output, expected in results:
                torch.testing.assert_close(output.float(), expected, atol=tolerance, rtol=tolerance)


def test_serial_reference_and_plan_cannot_rebind():
    plan = MoeExecutionPlan(routed=lambda x: x, shared=lambda x: x,
                            communication=AllReduceMoeCommunication(lambda x: x),
                            chunk_size=2)
    plan.prepare(None)
    assert plan.stream is None
    torch.testing.assert_close(plan(torch.ones(2, 3), is_prefill=False), torch.full((2, 3), 2.))
    with pytest.raises(RuntimeError, match="once"):
        plan.prepare(None)


def test_finish_receives_chunked_routing_metadata_and_local_shared_branch():
    x = torch.arange(15.).reshape(5, 3)
    communication = Mock()
    finish = Mock(side_effect=lambda routed, shared: routed[0] + shared * routed[1])
    plan = MoeExecutionPlan(
        routed=lambda x: (x * 2, x[:, :1]), shared=lambda x: x * 3,
        communication=communication, chunk_size=2, finish=finish,
    )
    torch.testing.assert_close(plan(x, is_prefill=False), x * 2 + x * 3 * x[:, :1])
    communication.combine_local_branches.assert_not_called()
    assert finish.call_count == 1


@pytest.mark.parametrize("has_independent_decode", [False, True])
def test_model_prepares_only_reachable_parallel_branches(monkeypatch, has_independent_decode):
    from sparsevllm.operators import moe_execution

    runtime = moe_execution.device_runtime
    new_stream, new_event, bind_lane = Mock(), Mock(), Mock()
    monkeypatch.setattr(runtime, "supports_streams", lambda device: True)
    monkeypatch.setattr(runtime, "new_stream", new_stream)
    monkeypatch.setattr(runtime, "new_event", new_event)
    monkeypatch.setattr(moe_execution, "bind_module_workspace_lane", bind_lane)
    model = nn.Module()
    # Prefill is independent but serial, so it does not need an auxiliary lane.
    for name, fused, limit in [("fused", True, None)] + (
        [("bounded", True, 2), ("independent", False, None)]
        if has_independent_decode else []
    ):
        layer = nn.Module()
        layer.shared = nn.Identity()
        layer.moe_execution = MoeExecutionPlan(
            routed=lambda x: x, shared=layer.shared,
            fused=lambda x: x * 2, fuse_decode=fused, fusion_token_limit=limit,
            communication=AllReduceMoeCommunication(lambda x: x),
            chunk_size=None, shared_modules=(layer.shared,),
        )
        model.add_module(name, layer)
    stream = prepare_model_moe_execution(model, "cuda")
    assert model.fused.moe_execution.stream is None
    assert model.fused.moe_execution.input_ready is None
    assert model.fused.moe_execution.shared_ready is None
    x = torch.ones(3, 4)
    torch.testing.assert_close(model.fused.moe_execution(x, is_prefill=False), x * 2)
    torch.testing.assert_close(model.fused.moe_execution(x, is_prefill=True), x * 2)
    if has_independent_decode:
        new_stream.assert_called_once_with("cuda")
        assert model.bounded.moe_execution.stream is stream
        assert model.independent.moe_execution.stream is stream
        assert new_event.call_count == 4
        assert [call.args[0] for call in bind_lane.call_args_list] == [
            model.bounded.shared, model.independent.shared,
        ]
    else:
        assert stream is None
        new_stream.assert_not_called()
        new_event.assert_not_called()
        bind_lane.assert_not_called()
