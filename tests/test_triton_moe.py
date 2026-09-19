import unittest

import pytest
import torch
import torch.nn.functional as F

from sparseengine.kernels.triton.gate_up_swiglu import h20_gate_up_swiglu
from sparseengine.kernels.triton.gemma4_moe import fused_gemma4_moe
from sparseengine.kernels.triton.moe import (
    _prepare_expert_assignment,
    append_shared_expert_route,
    fused_moe,
    fused_moe_gate_up_swiglu,
    moe_align_block_size,
)
from sparseengine.kernels.triton.moe_topk import topk_softmax
from sparseengine.kernels.triton.silu_and_mul import _resolve_silu_launch_config
from sparseengine.operators.gated_shared_add import gated_shared_add


def test_silu_launch_config_uses_decode_tile_only_for_small_rows():
    assert _resolve_silu_launch_config(160) == (32, 128, 4)
    assert _resolve_silu_launch_config(257) == (128, 128, None)


def _is_h20() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA H20"


@pytest.mark.skipif(not _is_h20(), reason="requires NVIDIA H20")
@pytest.mark.parametrize("intermediate_size", [256, 512])
def test_h20_gate_up_swiglu_matches_torch(intermediate_size):
    torch.manual_seed(0)
    inputs = torch.randn(1, 2048, dtype=torch.bfloat16, device="cuda")
    weight = 0.02 * torch.randn(
        2 * intermediate_size,
        2048,
        dtype=torch.bfloat16,
        device="cuda",
    )
    projected = torch.nn.functional.linear(inputs, weight)
    gate, up = projected.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate.float()) * up.float()

    actual = h20_gate_up_swiglu(inputs, weight)

    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 1024])
def test_gated_shared_add_matches_torch(num_tokens):
    torch.manual_seed(num_tokens)
    routed = torch.randn((num_tokens, 2048), device="cuda", dtype=torch.bfloat16)
    shared = torch.randn_like(routed)
    padded_gate = torch.randn((num_tokens, 257), device="cuda", dtype=torch.bfloat16)
    gate_logits = padded_gate[:, -1:]

    actual = gated_shared_add(routed, shared, gate_logits)
    expected = routed + torch.sigmoid(gate_logits) * shared

    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.005)


def _pytorch_topk_reference(
    logits: torch.Tensor,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    weights, ids = torch.topk(probabilities, 8, dim=-1)
    if norm_topk_prob:
        weights /= weights.sum(dim=-1, keepdim=True)
    return weights, ids


def _oracle_local_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_start: int,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    local_expert_end = local_expert_start + int(w13_weight.shape[0])
    for local_expert_id in range(int(w13_weight.shape[0])):
        global_expert_id = local_expert_start + local_expert_id
        token_ids, topk_slots = torch.where(topk_ids == global_expert_id)
        if token_ids.numel() == 0:
            continue
        assert local_expert_start <= global_expert_id < local_expert_end
        gate_up = F.linear(hidden_states[token_ids], w13_weight[local_expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate) * up, w2_weight[local_expert_id])
        expert_output *= topk_weights[token_ids, topk_slots, None]
        output.index_add_(0, token_ids, expert_output.to(output.dtype))
    return output


def _oracle_gemma4_moe(
    hidden_states,
    w13_weight,
    w2_weight,
    topk_ids,
    topk_weights,
    local_expert_start,
):
    output = torch.zeros_like(hidden_states)
    for local_id in range(w13_weight.shape[0]):
        global_id = local_expert_start + local_id
        token_ids, routes = torch.where(topk_ids == global_id)
        if token_ids.numel() == 0:
            continue
        gate, up = F.linear(hidden_states[token_ids], w13_weight[local_id]).chunk(2, -1)
        routed = F.linear(F.gelu(gate, approximate="tanh") * up, w2_weight[local_id])
        output.index_add_(0, token_ids, routed * topk_weights[token_ids, routes, None])
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tokens", [1, 19])
def test_gemma4_moe_matches_torch(num_tokens):
    torch.manual_seed(71 + num_tokens)
    hidden_size, intermediate_size, num_experts, top_k = 64, 32, 7, 3
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    w13_weight = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.1
    w2_weight = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.1
    ids = torch.randint(
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device="cuda",
    )
    weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    expected = _oracle_gemma4_moe(
        hidden_states, w13_weight, w2_weight, ids, weights, 0
    )
    actual = fused_gemma4_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        ids,
        weights,
        num_experts=num_experts,
        local_expert_start=0,
    )
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_append_shared_expert_route_matches_cat_and_replays_graph():
    ids = torch.tensor(
        [[3, 1, 7, 2], [9, 8, 4, 6]],
        dtype=torch.int32,
        device="cuda",
    )
    weights = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        dtype=torch.float32,
        device="cuda",
    )
    actual_ids, actual_weights = append_shared_expert_route(
        ids,
        weights,
        shared_expert_id=64,
    )
    expected_ids = torch.cat(
        (ids, torch.full_like(ids[:, :1], 64)),
        dim=1,
    )
    expected_weights = torch.cat(
        (weights, torch.ones_like(weights[:, :1])),
        dim=1,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_ids, expected_ids)
    assert torch.equal(actual_weights, expected_weights)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ids, graph_weights = append_shared_expert_route(
            ids,
            weights,
            shared_expert_id=64,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_ids, expected_ids)
    assert torch.equal(graph_weights, expected_weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_moe_align_block_size_filters_ep_experts_and_pads_blocks():
    topk_ids = torch.tensor(
        [[2, 0], [3, 2], [5, 2], [0, 1]],
        dtype=torch.int64,
        device="cuda",
    )
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        4,
        6,
        local_expert_start=2,
        local_expert_end=4,
    )
    torch.cuda.synchronize()

    invalid = topk_ids.numel()
    assert int(num_tokens_post_padded.item()) == 8
    assert expert_ids[:2].tolist() == [0, 1]
    assert all(expert_id == -1 for expert_id in expert_ids[2:].tolist())
    assert sorted(sorted_ids[:4].tolist()) == sorted([0, 3, 5, invalid])
    assert sorted(sorted_ids[4:8].tolist()) == sorted([2, invalid, invalid, invalid])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_naive_assignment_filters_ep_experts_in_one_kernel():
    alignment = _prepare_expert_assignment(
        torch.tensor([[3, 12]], dtype=torch.int64, device="cuda"),
        block_size=1,
        num_experts=16,
        local_expert_start=8,
        local_expert_end=16,
    )
    torch.cuda.synchronize()

    assert alignment.naive
    assert alignment.expert_ids.tolist() == [-1, 4]
    assert alignment.num_tokens_post_padded.item() == 2


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_moe_alignment_covers_hotspot_and_empty_rank(dtype):
    for topk_ids, local_start, local_end in (
        (torch.zeros((128, 8), dtype=dtype, device="cuda"), 0, 64),
        (torch.zeros((16, 8), dtype=dtype, device="cuda"), 64, 128),
    ):
        sorted_ids, expert_ids, num_padded = moe_align_block_size(
            topk_ids,
            16,
            128,
            local_expert_start=local_start,
            local_expert_end=local_end,
        )
        torch.cuda.synchronize()
        expected = 1024 if local_start == 0 else 0
        assert int(num_padded.item()) == expected
        assert int((expert_ids >= 0).sum().item()) == expected // 16
        valid = sorted_ids[sorted_ids < topk_ids.numel()]
        assert int(valid.numel()) == expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("norm_topk_prob", [False, True])
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_matches_pytorch(dtype, norm_topk_prob):
    torch.manual_seed(21)
    base = torch.arange(128, dtype=dtype, device="cuda") / 16 - 4
    logits = torch.stack(
        [base[torch.randperm(128, device="cuda")] for _ in range(257)]
    )
    expected_probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    expected_weights, expected_ids = torch.topk(expected_probs, 8, dim=-1)
    if norm_topk_prob:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    actual_weights, actual_ids = topk_softmax(
        logits,
        top_k=8,
        norm_topk_prob=norm_topk_prob,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_ids, expected_ids.to(torch.int32))
    tolerance = 2e-2 if dtype == torch.bfloat16 else 4e-3
    assert torch.allclose(
        actual_weights.float(),
        expected_weights,
        atol=tolerance,
        rtol=tolerance,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_accepts_any_valid_experts_for_ties():
    logits = torch.zeros(1, 128, dtype=torch.bfloat16, device="cuda")
    weights, ids = topk_softmax(logits, top_k=8, norm_topk_prob=True)
    torch.cuda.synchronize()

    assert bool(((ids >= 0) & (ids < 128)).all())
    assert int(torch.unique(ids).numel()) == 8
    assert torch.allclose(
        weights.float(),
        torch.full((1, 8), 1 / 8, device="cuda"),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_accepts_padded_row_stride():
    logits = torch.randn(3, 257, dtype=torch.bfloat16, device="cuda")[:, :256]
    expected_weights, expected_ids = _pytorch_topk_reference(logits, True)

    weights, ids = topk_softmax(logits, top_k=8, norm_topk_prob=True)
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids.to(torch.int32))
    assert torch.allclose(weights.float(), expected_weights, atol=2e-2, rtol=2e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_is_stable_for_extreme_finite_logits():
    logits = torch.full((2, 128), -100, dtype=torch.bfloat16, device="cuda")
    logits[:, :8] = torch.arange(100, 92, -1, dtype=torch.bfloat16, device="cuda")
    expected_weights, expected_ids = _pytorch_topk_reference(logits, True)

    weights, ids = topk_softmax(logits, top_k=8, norm_topk_prob=True)
    torch.cuda.synchronize()

    assert torch.equal(ids, expected_ids.to(torch.int32))
    assert torch.allclose(weights.float(), expected_weights, atol=2e-2, rtol=2e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_nonfinite_inputs_keep_ids_in_range_and_propagate_nan():
    logits = torch.zeros(2, 128, dtype=torch.bfloat16, device="cuda")
    logits[0, 3] = float("nan")
    logits[1, 7] = float("inf")

    weights, ids = topk_softmax(logits, top_k=8, norm_topk_prob=False)
    torch.cuda.synchronize()

    assert bool(((ids >= 0) & (ids < 128)).all())
    assert not bool(torch.isfinite(weights).all())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_topk_softmax_rejects_unsupported_shape_and_layout():
    with pytest.raises(ValueError, match="num_experts=128"):
        topk_softmax(
            torch.zeros(2, 64, dtype=torch.bfloat16, device="cuda"),
            top_k=8,
            norm_topk_prob=True,
        )
    non_contiguous = torch.zeros(128, 2, dtype=torch.bfloat16, device="cuda").T
    with pytest.raises(ValueError, match="contiguous"):
        topk_softmax(non_contiguous, top_k=8, norm_topk_prob=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_naive_decode_matches_oracle(dtype):
    torch.manual_seed(11)
    device = torch.device("cuda")
    num_experts = 16
    hidden_size = 37
    intermediate_size = 23
    hidden_states = torch.randn(1, hidden_size, device=device, dtype=dtype)
    w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.tensor([[3, 12]], dtype=torch.int64, device=device)
    topk_weights = torch.tensor([[0.65, 0.35]], dtype=dtype, device=device)

    expected = _oracle_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        0,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=0,
    )
    torch.cuda.synchronize()

    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_aligned_prefill_matches_oracle_with_padding():
    torch.manual_seed(12)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 19
    num_experts = 7
    top_k = 3
    hidden_size = 45
    intermediate_size = 29
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device=device,
    )
    topk_ids[:5] = 0
    topk_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    expected = _oracle_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        0,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=0,
    )
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "top_k"),
    [(1, 16, 2), (4, 8, 3), (19, 7, 3)],
)
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_fused_gate_up_swiglu_matches_generic_pipeline(
    num_tokens,
    num_experts,
    top_k,
):
    torch.manual_seed(43 + num_tokens)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_size = 64
    intermediate_size = 32
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w13_weight = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    w2_weight = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    expected = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=0,
    )
    actual = fused_moe_gate_up_swiglu(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=0,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_fused_gate_up_swiglu_matches_qwen3_tp_ep_shape():
    torch.manual_seed(59)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_states = torch.randn(4, 2048, device=device, dtype=dtype)
    w13_weight = torch.randn(1, 768, 2048, device=device, dtype=dtype) * 0.02
    w2_weight = torch.randn(1, 2048, 384, device=device, dtype=dtype) * 0.02
    topk_ids = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [3, 0]],
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(4, 2, device=device, dtype=dtype)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    kwargs = {"num_experts": 128, "local_expert_start": 0}

    expected = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        **kwargs,
    )
    actual = fused_moe_gate_up_swiglu(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        **kwargs,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_ep_local_output_matches_oracle_and_ignores_remote_experts():
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 13
    num_experts = 8
    local_expert_start = 2
    num_local_experts = 4
    top_k = 2
    hidden_size = 32
    intermediate_size = 17
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w13_weight = (
        torch.randn(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)

    expected = _oracle_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        local_expert_start,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
    )
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_tp_partial_sum_matches_unsharded_oracle():
    torch.manual_seed(37)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 11
    num_experts = 8
    top_k = 3
    hidden_size = 64
    intermediate_size = 32
    tp_size = 2
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    gate = torch.randn(
        num_experts,
        intermediate_size,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.1
    up = torch.randn_like(gate) * 0.1
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.1
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    expected = _oracle_local_moe(
        hidden_states,
        torch.cat((gate, up), dim=1),
        down,
        topk_ids,
        topk_weights,
        0,
    )
    partial_outputs = []
    for tp_rank in range(tp_size):
        gate_shard = gate.chunk(tp_size, dim=1)[tp_rank]
        up_shard = up.chunk(tp_size, dim=1)[tp_rank]
        down_shard = down.chunk(tp_size, dim=2)[tp_rank].contiguous()
        partial_outputs.append(
            fused_moe(
                hidden_states,
                torch.cat((gate_shard, up_shard), dim=1),
                down_shard,
                topk_ids,
                topk_weights,
                num_experts=num_experts,
                local_expert_start=0,
            )
        )
    actual = torch.stack(partial_outputs).sum(dim=0)
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=4e-2, rtol=4e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_tp_ep_partial_sum_matches_unsharded_oracle():
    torch.manual_seed(41)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_experts = 8
    hidden_size = 64
    intermediate_size = 32
    ep_size = 2
    moe_tp_size = 2
    hidden_states = torch.randn(4, hidden_size, device=device, dtype=dtype)
    gate = torch.randn(
        num_experts, intermediate_size, hidden_size, device=device, dtype=dtype
    ) * 0.1
    up = torch.randn_like(gate) * 0.1
    down = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * 0.1
    topk_ids = torch.tensor(
        [[0, 1, 2], [4, 5, 7], [0, 4, 1], [3, 7, 4]],
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(4, 3, device=device, dtype=dtype)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    expected = _oracle_local_moe(
        hidden_states,
        torch.cat((gate, up), dim=1),
        down,
        topk_ids,
        topk_weights,
        0,
    )
    partial_outputs = []
    local_expert_count = num_experts // ep_size
    for ep_rank in range(ep_size):
        expert_slice = slice(
            ep_rank * local_expert_count,
            (ep_rank + 1) * local_expert_count,
        )
        for moe_tp_rank in range(moe_tp_size):
            gate_shard = gate[expert_slice].chunk(moe_tp_size, dim=1)[moe_tp_rank]
            up_shard = up[expert_slice].chunk(moe_tp_size, dim=1)[moe_tp_rank]
            down_shard = (
                down[expert_slice]
                .chunk(moe_tp_size, dim=2)[moe_tp_rank]
                .contiguous()
            )
            partial_outputs.append(
                fused_moe(
                    hidden_states,
                    torch.cat((gate_shard, up_shard), dim=1),
                    down_shard,
                    topk_ids,
                    topk_weights,
                    num_experts=num_experts,
                    local_expert_start=ep_rank * local_expert_count,
                )
            )
    actual = torch.stack(partial_outputs).sum(dim=0)
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=4e-2, rtol=4e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_can_preserve_fp32_topk_sum():
    torch.manual_seed(15)
    device = torch.device("cuda")
    hidden_states = torch.randn(5, 31, device=device, dtype=torch.bfloat16)
    w13_weight = torch.randn(4, 38, 31, device=device, dtype=torch.bfloat16)
    w2_weight = torch.randn(4, 31, 19, device=device, dtype=torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 3], [1, 2], [2, 0], [3, 1], [0, 2]],
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(5, 2, device=device, dtype=torch.bfloat16)

    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=4,
        local_expert_start=0,
        output_dtype=torch.float32,
    )
    torch.cuda.synchronize()

    assert actual.dtype == torch.float32


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_returns_zero_when_all_assignments_are_remote():
    torch.manual_seed(14)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_states = torch.randn(9, 24, device=device, dtype=dtype)
    w13_weight = torch.randn(4, 26, 24, device=device, dtype=dtype)
    w2_weight = torch.randn(4, 24, 13, device=device, dtype=dtype)
    topk_ids = torch.zeros(9, 2, dtype=torch.int64, device=device)
    topk_weights = torch.full((9, 2), 0.5, dtype=dtype, device=device)

    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=8,
        local_expert_start=4,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual).item() == 0
