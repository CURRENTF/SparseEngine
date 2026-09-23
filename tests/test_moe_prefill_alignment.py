from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparseengine.operators.moe import MoeOpSpec, TritonMoeProvider
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum
from sparseengine.kernels.external.sgl.moe import sgl_moe_alignment_support
from sparseengine.utils.context import get_context


def spec(experts=8, topk=2, ep=1, dtype=torch.bfloat16):
    return MoeOpSpec(experts, experts // ep, 32, 24, topk, dtype, dtype, None, ep, False)


def caps():
    return DeviceCaps(PlatformEnum.CUDA, "cuda", 0, "CUDA device", supports_triton=True)


def test_missing_alignment_dependency_keeps_triton():
    with patch("sparseengine.kernels.external.sgl.moe.sgl_moe_alignment_support", return_value=(False, "absent")):
        provider = TritonMoeProvider.bind(spec(), caps())
    assert provider._prefill_alignment is None
    assert provider.binding_metadata()["prefill_alignment_reason"] == "absent"


def test_broken_alignment_dependency_is_not_silently_ignored():
    with patch("sparseengine.kernels.external.sgl.moe.sgl_moe_alignment_support", side_effect=RuntimeError("broken API")):
        with pytest.raises(RuntimeError, match="broken API"):
            TritonMoeProvider.bind(spec(), caps())


def test_fp8_keeps_its_existing_preparation():
    with patch("sparseengine.kernels.external.sgl.moe.sgl_moe_alignment_support") as check:
        provider = TritonMoeProvider.bind(replace(spec(), weight_dtype=torch.float8_e4m3fn), caps())
    check.assert_not_called()
    assert provider._prefill_alignment is None


@pytest.mark.parametrize("prefill", [False, True])
def test_alignment_is_selected_only_for_prefill(monkeypatch, prefill):
    with patch("sparseengine.kernels.external.sgl.moe.sgl_moe_alignment_support", return_value=(True, "available")):
        provider = TritonMoeProvider.bind(spec(), caps())
    monkeypatch.setattr(get_context(), "is_prefill", prefill)
    with patch("sparseengine.kernels.triton.moe.fused_moe") as run:
        provider.run(spec(), None, None, None, None, None, None, None,
                     local_expert_start=0, tp_rank=0, ep_rank=0)
    assert run.call_args.kwargs["alignment_impl"] is (provider._prefill_alignment if prefill else None)


@pytest.mark.skipif(not torch.cuda.is_available() or not sgl_moe_alignment_support()[0], reason="CUDA and SGL required")
@pytest.mark.parametrize("experts,topk,ep", [(8, 2, 1), (64, 4, 2), (128, 8, 1), (256, 8, 2)])
@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prepared_prefill_moe_matches_independent_reference(monkeypatch, experts, topk, ep, tokens, dtype):
    from test_sgl_moe import _torch_local_moe

    torch.manual_seed(938)
    s = spec(experts, topk, ep, dtype)
    start = 0 if ep == 1 else s.num_local_experts
    x = torch.randn(tokens, s.hidden_size, device="cuda", dtype=dtype) * .1
    w13 = torch.randn(s.num_local_experts, 2 * s.intermediate_size, s.hidden_size, device="cuda", dtype=dtype) * .1
    w2 = torch.randn(s.num_local_experts, s.hidden_size, s.intermediate_size, device="cuda", dtype=dtype) * .1
    ids = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)])
    weights = torch.rand(tokens, topk, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    expected = _torch_local_moe(x, w13, w2, ids, weights, start)
    provider = TritonMoeProvider.bind(s, caps())
    assert provider._prefill_alignment is not None
    monkeypatch.setattr(get_context(), "is_prefill", True)
    with torch.inference_mode():
        actual = provider.run(s, x, ids, weights, w13, w2, None, None,
                              local_expert_start=start, tp_rank=0, ep_rank=ep - 1)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-2)

@pytest.mark.parametrize("assignments,local_end,expected", [
    (1, 8, "original"), (64, 8, "sgl"),
    (262144, 4, "original"), (262144, 8, "sgl"),
])
def test_alignment_preserves_tiny_and_large_sharded_algorithms(assignments, local_end, expected):
    from sparseengine.operators.moe import _prefill_moe_align_block_size

    ids = SimpleNamespace(numel=lambda: assignments)
    with patch("sparseengine.kernels.triton.moe._prepare_expert_assignment", return_value="original"), patch("sparseengine.operators.moe._sgl_moe_align_block_size", return_value="sgl"):
        actual = _prefill_moe_align_block_size(ids, block_size=16, num_experts=8,
                                              local_expert_start=0, local_expert_end=local_end)
    assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available() or not sgl_moe_alignment_support()[0], reason="CUDA and SGL required")
def test_large_sharded_prefill_preserves_full_moe_output(monkeypatch):
    test_prepared_prefill_moe_matches_independent_reference(
        monkeypatch, 128, 8, 2, 16384, torch.bfloat16,
    )
