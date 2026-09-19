import unittest

import pytest
import torch
import torch.nn.functional as F

from sparseengine.kernels.triton.minimax_m2_router import (
    minimax_m2_router,
    topk_biased_sigmoid,
)
from sparseengine.kernels.triton.moe_biased_sigmoid import (
    topk_biased_sigmoid as generic_topk_biased_sigmoid,
)


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

    weights, ids = topk_biased_sigmoid(logits, correction_bias)
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

    weights, ids = generic_topk_biased_sigmoid(
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

    weights, ids = topk_biased_sigmoid(logits, correction_bias)
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

    weights, ids = topk_biased_sigmoid(logits, correction_bias)
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_correction_bias_selects_experts_but_does_not_change_weights():
    logits = torch.linspace(-4, 4, 256, dtype=torch.float32, device="cuda")[None]
    correction_bias = torch.zeros(256, dtype=torch.float32, device="cuda")
    correction_bias[:8] = 2
    expected_weights, expected_ids = _reference(logits, correction_bias)
    weights, ids = topk_biased_sigmoid(logits, correction_bias)
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids)
    assert set(ids[0].tolist()) == set(range(8))
    assert torch.equal(weights, expected_weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_minimax_router_matches_fp32_gate_and_reference():
    torch.manual_seed(31)
    hidden_states = torch.randn(17, 3072, dtype=torch.float32, device="cuda")
    gate_weight = torch.randn(256, 3072, dtype=torch.float32, device="cuda") * 0.02
    correction_bias = torch.randn(256, dtype=torch.float32, device="cuda") * 0.1
    expected_logits = F.linear(hidden_states, gate_weight)
    expected_weights, expected_ids = _reference(expected_logits, correction_bias)

    logits, weights, ids = minimax_m2_router(
        hidden_states,
        gate_weight,
        correction_bias,
    )
    torch.cuda.synchronize()

    assert torch.equal(logits, expected_logits)
    assert torch.equal(ids, expected_ids)
    assert torch.allclose(weights, expected_weights, atol=2e-7, rtol=2e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
def test_topk_biased_sigmoid_is_cuda_graph_capturable():
    torch.manual_seed(41)
    logits = torch.randn(8, 256, dtype=torch.float32, device="cuda")
    correction_bias = torch.randn(256, dtype=torch.float32, device="cuda") * 0.1
    topk_biased_sigmoid(logits, correction_bias)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = topk_biased_sigmoid(logits, correction_bias)
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
        topk_biased_sigmoid(torch.zeros(2, 128, device="cuda"), bias)
    with pytest.raises(TypeError, match="FP32"):
        topk_biased_sigmoid(
            torch.zeros(2, 256, dtype=torch.bfloat16, device="cuda"),
            bias,
        )
    with pytest.raises(ValueError, match="contiguous"):
        topk_biased_sigmoid(torch.zeros(256, 2, device="cuda").T, bias)
