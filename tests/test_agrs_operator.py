"""Protect provider eligibility and shared-workspace lifetime, not tuning values."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from sparseengine.distributed.collective_runtime import ParallelCollectiveRuntime
from sparseengine.distributed.parallel_context import ParallelContext, ParallelGroup
from sparseengine.operators import agrs
from sparseengine.operators.registry import OpResolver
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


def _spec():
    return agrs.AgRsOpSpec(2, 2048, torch.bfloat16, 8, "nccl", True, True)


def _caps():
    return DeviceCaps(
        PlatformEnum.CUDA,
        "cuda",
        0,
        "NVIDIA H100",
        compute_capability=(9, 0),
        supports_graph_capture=True,
    )


def test_unavailable_collective_capability_resolves_before_launch(monkeypatch):
    # Upstream selection must not bypass peer/multicast eligibility.
    monkeypatch.setattr(agrs, "_mixed_comm_available", lambda: True)
    resolver = OpResolver(agrs.AGRS_REGISTRY)
    assert isinstance(
        resolver.resolve(_spec(), _caps()).provider, agrs.FlashInferMixedAgRsProvider
    )
    assert isinstance(
        resolver.resolve(
            replace(_spec(), multicast_and_peer_access=False), _caps()
        ).provider,
        agrs.TorchAgRsProvider,
    )
    monkeypatch.setattr(agrs, "_mixed_comm_available", lambda: False)
    assert isinstance(
        resolver.resolve(_spec(), _caps()).provider, agrs.TorchAgRsProvider
    )


def test_supported_upstream_binding_needs_no_device_or_shape_profile(monkeypatch):
    # Exercise a legal contract outside the old H100/DP2/BF16 decode profile.
    # Selection is not a numerical or performance assertion.
    monkeypatch.setattr(agrs, "_mixed_comm_available", lambda: True)
    spec = replace(
        _spec(),
        world_size=4,
        hidden_size=4096,
        dtype=torch.float16,
        max_rows=2048,
    )
    caps = replace(_caps(), device_name="test")
    assert isinstance(
        OpResolver(agrs.AGRS_REGISTRY).resolve(spec, caps).provider,
        agrs.FlashInferMixedAgRsProvider,
    )


def test_broken_installed_dependency_does_not_select_nccl(monkeypatch):
    def broken():
        raise ImportError("broken installed CUDA binding")

    monkeypatch.setattr(agrs, "_mixed_comm_available", broken)
    with pytest.raises(ImportError, match="broken installed CUDA binding"):
        OpResolver(agrs.AGRS_REGISTRY).resolve(_spec(), _caps())


def test_agrs_workspace_survives_graph_replacement_and_closes_once(monkeypatch):
    # Graph input addresses change on recapture; transport workspace must not be
    # closed while an idle graph still references it, or allocated once per layer.
    world = ParallelGroup(object(), (0, 1), 0, 2)
    local = ParallelGroup(None, (0,), 0, 1)
    context = ParallelContext(world=world, attn_tp=local, moe_ep=world, attn_dp=world, moe_tp=local)
    runtime = ParallelCollectiveRuntime(context, cuda_graph=True, device_index=0)
    handle = runtime.request_dp_collectives(
        max_rows=8, max_local_tokens=1024, hidden_size=2048, dtype=torch.bfloat16
    ).moe_transport
    with pytest.raises(RuntimeError, match="not prepared"):
        handle.dispatch(torch.empty(1, 2048), capacity=1)
    op = Mock()
    prepare = Mock(return_value=op)
    monkeypatch.setattr(
        "sparseengine.distributed.moe_communication.prepare_parallel_agrs", prepare
    )
    runtime.prepare()
    monkeypatch.setattr(
        runtime, "_all_gather_object", lambda local, group: [local] * group.size
    )
    for _ in range(2):
        runtime.begin_cuda_graph_capture()
        runtime.collect_local_cuda_graph_metadata()
        runtime.exchange_cuda_graph_metadata()
        runtime.register_cuda_graph_buffers()
        runtime.mark_cuda_graph_replayable()
        runtime.reset_for_cuda_graph_recapture()
    assert handle.op is op
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["max_rows"] >= 1024
    op.close.assert_not_called()
    runtime.close()
    runtime.close()
    op.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        handle.dispatch(torch.empty(1, 2048), capacity=1)


def test_prepared_contract_rejects_before_provider_launch():
    provider = Mock()
    op = agrs.PreparedAgRsOp(_spec(), provider)
    local = torch.empty(2, 2048, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="prepared"):
        op.all_gather(torch.empty_like(local), local)
    provider.all_gather.assert_not_called()
    op.close()
    with pytest.raises(RuntimeError, match="closed"):
        op.all_gather(torch.empty(4, 2048, dtype=torch.bfloat16), local)
