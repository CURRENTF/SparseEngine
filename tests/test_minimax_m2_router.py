import unittest

import pytest
import torch

from sparseengine.kernels.triton.moe_biased_sigmoid import topk_biased_sigmoid
from sparseengine.operators.moe_router import (
    MiniMaxBiasedSigmoidRouterProvider,
    MoeRouterOpSpec,
)


_MINIMAX_ROUTER = MiniMaxBiasedSigmoidRouterProvider()
_MINIMAX_SPEC = MoeRouterOpSpec(256, 8, torch.float32, True, True, "biased_sigmoid")


def _reference(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    routing_weights = torch.sigmoid(logits.float())
    scores = routing_weights + correction_bias
    _, ids = torch.topk(scores, 8, dim=-1, sorted=False)
    weights = routing_weights.gather(1, ids)
    weights /= weights.sum(dim=-1, keepdim=True)
    return weights, ids


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_topk_biased_sigmoid_matches_minimax_reference():
    torch.manual_seed(27)
    logits = torch.randn(4096, 256, dtype=torch.float32, device="cuda") * 3
    correction_bias = torch.randn(256, dtype=torch.float32, device="cuda") * 0.1
    expected_weights, expected_ids = _reference(logits, correction_bias)

    weights, ids = _MINIMAX_ROUTER.run(_MINIMAX_SPEC, logits, correction_bias)
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_topk_biased_sigmoid_matches_glm_64_expert_reference():
    torch.manual_seed(29)
    logits = torch.randn(1024, 64, dtype=torch.float32, device="cuda") * 3
    correction_bias = torch.randn(64, dtype=torch.float32, device="cuda") * 0.1
    routing_weights = torch.sigmoid(logits)
    expected_ids = torch.topk(
        routing_weights + correction_bias,
        4,
        dim=-1,
        sorted=False,
    ).indices
    expected_weights = routing_weights.gather(1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20

    weights, ids = topk_biased_sigmoid(
        logits,
        correction_bias,
        top_k=4,
    )
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_topk_biased_sigmoid_matches_nonfinite_reference():
    logits = torch.zeros(6, 256, dtype=torch.float32, device="cuda")
    logits[0, 3] = float("nan")
    logits[1, 7] = float("inf")
    logits[2, 9] = -float("inf")
    logits[3] = 100
    logits[4] = -100
    correction_bias = torch.zeros(256, dtype=torch.float32, device="cuda")
    correction_bias[5] = float("nan")
    expected_weights, expected_ids = _reference(logits, correction_bias)

    weights, ids = _MINIMAX_ROUTER.run(_MINIMAX_SPEC, logits, correction_bias)
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids)
    assert torch.equal(torch.isnan(weights), torch.isnan(expected_weights))
    assert torch.equal(
        torch.nan_to_num(weights), torch.nan_to_num(expected_weights)
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_topk_biased_sigmoid_matches_unsorted_tie_order():
    logits = torch.zeros(4, 256, dtype=torch.float32, device="cuda")
    logits[1, :9] = 1
    logits[2, 120:129] = 1
    logits[3] = (torch.arange(256, device="cuda") // 8).float()
    correction_bias = torch.zeros(256, dtype=torch.float32, device="cuda")
    expected_weights, expected_ids = _reference(logits, correction_bias)

    weights, ids = _MINIMAX_ROUTER.run(_MINIMAX_SPEC, logits, correction_bias)
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_correction_bias_selects_experts_but_does_not_change_weights():
    logits = torch.linspace(-4, 4, 256, dtype=torch.float32, device="cuda")[None]
    correction_bias = torch.zeros(256, dtype=torch.float32, device="cuda")
    correction_bias[:8] = 2
    expected_weights, expected_ids = _reference(logits, correction_bias)
    weights, ids = _MINIMAX_ROUTER.run(_MINIMAX_SPEC, logits, correction_bias)
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids)
    assert set(ids[0].tolist()) == set(range(8))
    assert torch.equal(weights, expected_weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_topk_biased_sigmoid_is_cuda_graph_capturable():
    torch.manual_seed(41)
    logits = torch.randn(8, 256, dtype=torch.float32, device="cuda")
    correction_bias = torch.randn(256, dtype=torch.float32, device="cuda") * 0.1
    _MINIMAX_ROUTER.run(_MINIMAX_SPEC, logits, correction_bias)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = _MINIMAX_ROUTER.run(_MINIMAX_SPEC, logits, correction_bias)
    logits.copy_(torch.randn_like(logits))
    expected_weights, expected_ids = _reference(logits, correction_bias)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids)
    assert torch.allclose(weights, expected_weights, atol=2e-7, rtol=2e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_topk_biased_sigmoid_rejects_unsupported_inputs():
    bias = torch.zeros(256, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="num_experts=256"):
        _MINIMAX_ROUTER.run(_MINIMAX_SPEC, torch.zeros(2, 128, device="cuda"), bias)
    with pytest.raises(TypeError, match="FP32"):
        _MINIMAX_ROUTER.run(
            _MINIMAX_SPEC,
            torch.zeros(2, 256, dtype=torch.bfloat16, device="cuda"),
            bias,
        )
    with pytest.raises(ValueError, match="contiguous"):
        _MINIMAX_ROUTER.run(_MINIMAX_SPEC, torch.zeros(256, 2, device="cuda").T, bias)
