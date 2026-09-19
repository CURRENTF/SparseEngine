import pytest
import torch
import torch.nn.functional as F

from sparseengine.operators.fp8_linear import (
    FlashInferGroupwiseSm120Fp8LinearProvider,
    Fp8LinearSpec,
    _sm120_activation_workspace,
    resolve_fp8_linear_provider,
)
from sparseengine.platforms import current_platform
from sparseengine.quantization.fp8 import fp8_blockwise_linear_reference
from sparseengine.kernels.triton.fp8_blockwise import fp8_blockwise_matmul
from sparseengine.kernels.triton.moe import fused_moe_fp8
from sparseengine.kernels.triton.minimax_m2_moe import fused_minimax_m2_moe_fp8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def _fp8_weight(shape, device):
    return (
        torch.randn(shape, device=device, dtype=torch.float32)
        .clamp(-3.0, 3.0)
        .to(torch.float8_e4m3fn)
    )


def _assert_fp8_pipeline_close(actual, expected):
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    relative_l2 = torch.linalg.vector_norm(
        actual_fp32 - expected_fp32
    ) / torch.linalg.vector_norm(expected_fp32)
    cosine = F.cosine_similarity(
        actual_fp32.flatten(),
        expected_fp32.flatten(),
        dim=0,
    )
    assert relative_l2.item() < 2.0e-2
    assert cosine.item() > 0.9998


@pytest.mark.parametrize(
    ("tokens", "out_features", "in_features"),
    [(1, 128, 128), (7, 256, 384), (19, 129, 257)],
)
def test_fp8_blockwise_matmul_matches_reference(
    tokens,
    out_features,
    in_features,
):
    torch.manual_seed(tokens + out_features + in_features)
    device = torch.device("cuda")
    inputs = torch.randn(
        tokens,
        in_features,
        device=device,
        dtype=torch.bfloat16,
    )
    weight = _fp8_weight((out_features, in_features), device)
    scales = (
        torch.rand(
            (out_features + 127) // 128,
            (in_features + 127) // 128,
            device=device,
        )
        + 0.25
    )

    actual = fp8_blockwise_matmul(inputs, weight, scales)
    expected = fp8_blockwise_linear_reference(inputs, weight, scales).to(
        torch.bfloat16
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-1)


def test_resolved_fp8_linear_provider_matches_reference():
    torch.manual_seed(29)
    device = torch.device("cuda")
    inputs = torch.randn(3, 128, device=device, dtype=torch.bfloat16)
    weight = _fp8_weight((128, 128), device)
    scales = torch.rand(1, 1, device=device) + 0.25
    provider = resolve_fp8_linear_provider(
        (128, 128),
        input_features=weight.shape[1],
        output_features=weight.shape[0],
    )

    actual = provider(inputs, weight, scales)
    expected = fp8_blockwise_linear_reference(inputs, weight, scales).to(
        torch.bfloat16
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-1)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="profiled FP8 Linear dispatch requires SM120",
)
def test_resolved_sm120_fp8_linear_matches_reference_across_batch_sizes():
    torch.manual_seed(20260821)
    device = torch.device("cuda")
    weight = _fp8_weight((5120, 2048), device)
    scales = torch.rand(40, 16, device=device) + 0.25
    provider = resolve_fp8_linear_provider(
        (128, 128),
        input_features=2048,
        output_features=5120,
    )

    for tokens in (1, 512, 513, 515):
        inputs = torch.randn(
            tokens,
            2048,
            device=device,
            dtype=torch.bfloat16,
        )
        actual = provider(inputs, weight, scales)
        expected = fp8_blockwise_linear_reference(inputs, weight, scales)
        _assert_fp8_pipeline_close(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="FlashInfer groupwise FP8 Linear requires SM120",
)
def test_flashinfer_sm120_fp8_linear_cuda_graph_replay():
    torch.manual_seed(20260821)
    device = torch.device("cuda")
    inputs = torch.randn(3, 384, device=device, dtype=torch.bfloat16)
    weight = _fp8_weight((256, 384), device)
    scales = torch.rand(2, 3, device=device) + 0.25
    spec = Fp8LinearSpec(
        block_shape=(128, 128),
        input_features=int(weight.shape[1]),
        output_features=int(weight.shape[0]),
    )
    caps = current_platform.get_device_caps(torch.cuda.current_device())
    assert FlashInferGroupwiseSm120Fp8LinearProvider.supports(
        spec, caps
    ).supported
    provider = FlashInferGroupwiseSm120Fp8LinearProvider.bind(
        spec,
        caps,
    )

    provider(inputs, weight, scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = provider(inputs, weight, scales)
    inputs.copy_(torch.randn_like(inputs))
    graph.replay()
    torch.cuda.synchronize()

    expected = fp8_blockwise_linear_reference(inputs, weight, scales)
    _assert_fp8_pipeline_close(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="FlashInfer groupwise FP8 Linear requires SM120",
)
def test_sm120_fp8_linear_reuses_geometric_activation_workspace():
    device = torch.device("cuda")
    first = torch.empty((513, 640), device=device, dtype=torch.bfloat16)
    second = torch.empty((600, 640), device=device, dtype=torch.bfloat16)

    first_quantized, first_scales = _sm120_activation_workspace(first)
    second_quantized, second_scales = _sm120_activation_workspace(second)

    assert first_quantized.untyped_storage().data_ptr() == (
        second_quantized.untyped_storage().data_ptr()
    )
    assert first_scales.untyped_storage().data_ptr() == (
        second_scales.untyped_storage().data_ptr()
    )
    assert first_quantized.shape == (513, 640)
    assert second_quantized.shape == (600, 640)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="FlashInfer block-scale FP8 Linear requires SM90",
)
def test_resolved_sm90_fp8_linear_cuda_graph_replay():
    torch.manual_seed(20260822)
    device = torch.device("cuda")
    inputs = torch.randn(3, 384, device=device, dtype=torch.bfloat16)
    weight = _fp8_weight((256, 384), device)
    scales = torch.rand(2, 3, device=device) + 0.25
    provider = resolve_fp8_linear_provider(
        (128, 128),
        input_features=int(weight.shape[1]),
        output_features=int(weight.shape[0]),
    )

    provider(inputs, weight, scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = provider(inputs, weight, scales)
    inputs.copy_(torch.randn_like(inputs))
    graph.replay()
    torch.cuda.synchronize()

    expected = fp8_blockwise_linear_reference(inputs, weight, scales)
    _assert_fp8_pipeline_close(actual, expected)


def test_resolver_uses_triton_for_non_sm90_aligned_shape():
    torch.manual_seed(31)
    device = torch.device("cuda")
    inputs = torch.randn(3, 257, device=device, dtype=torch.bfloat16)
    weight = _fp8_weight((129, 257), device)
    scales = torch.rand(2, 3, device=device) + 0.25
    provider = resolve_fp8_linear_provider(
        (128, 128),
        input_features=weight.shape[1],
        output_features=weight.shape[0],
    )

    actual = provider(inputs, weight, scales)
    expected = fp8_blockwise_linear_reference(inputs, weight, scales).to(
        torch.bfloat16
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-1)


def _reference_moe(
    hidden_states,
    w13_weight,
    w2_weight,
    w13_scale,
    w2_scale,
    topk_ids,
    topk_weights,
    gate_up_order,
):
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    intermediate = w2_weight.shape[-1]
    for token in range(hidden_states.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            packed = fp8_blockwise_linear_reference(
                hidden_states[token : token + 1],
                w13_weight[expert],
                w13_scale[expert],
            )
            first, second = packed.split(intermediate, dim=-1)
            gate, up = (
                (first, second)
                if gate_up_order == "gate_up"
                else (second, first)
            )
            activated = F.silu(gate) * up
            expert_output = fp8_blockwise_linear_reference(
                activated.to(hidden_states.dtype),
                w2_weight[expert],
                w2_scale[expert],
            )
            output[token].add_(
                expert_output[0] * topk_weights[token, slot].float()
            )
    return output.to(hidden_states.dtype)


@pytest.mark.parametrize("gate_up_order", ["gate_up", "up_gate"])
@pytest.mark.parametrize("activation_dtype", [torch.bfloat16, torch.float16])
def test_fp8_moe_matches_reference(gate_up_order, activation_dtype):
    torch.manual_seed(17)
    device = torch.device("cuda")
    tokens, experts, top_k = 5, 4, 2
    hidden, intermediate = 128, 128
    hidden_states = torch.randn(
        tokens,
        hidden,
        device=device,
        dtype=activation_dtype,
    )
    w13_weight = _fp8_weight((experts, 2 * intermediate, hidden), device)
    w2_weight = _fp8_weight((experts, hidden, intermediate), device)
    w13_scale = torch.rand(
        experts,
        2 * intermediate // 128,
        hidden // 128,
        device=device,
    ) + 0.25
    w2_scale = torch.rand(
        experts,
        hidden // 128,
        intermediate // 128,
        device=device,
    ) + 0.25
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(
        tokens,
        top_k,
        device=device,
        dtype=torch.float32,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    actual = fused_moe_fp8(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
        num_experts=experts,
        local_expert_start=0,
        gate_up_order=gate_up_order,
    )
    expected = _reference_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
        gate_up_order,
    )

    _assert_fp8_pipeline_close(actual, expected)


@pytest.mark.parametrize("tokens", [1, 2, 8])
def test_qwen_fp8_decode_routes_preserve_packed_weights_across_graph_replays(tokens):
    """Catch wrong gate/up packing or stale per-request routing during graph replay.

    The small generic MoE oracle does not cover the Qwen expert dimensions,
    shared layout between providers, or changing routes in a captured graph.
    """
    from sparseengine.operators.moe import (
        FlashInferCutlassFp8MoeProvider,
        MoeOpSpec,
        TritonUpGateFp8MoeProvider,
    )

    providers = [TritonUpGateFp8MoeProvider()]
    if torch.cuda.get_device_capability() == (9, 0):
        providers.append(FlashInferCutlassFp8MoeProvider())
    torch.manual_seed(42)
    experts, hidden, intermediate, top_k = 128, 2048, 768, 8
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w13 = _fp8_weight((experts, 2 * intermediate, hidden), device)
    w2 = _fp8_weight((experts, hidden, intermediate), device)
    s13 = torch.rand(experts, 2 * intermediate // 128, hidden // 128, device=device) * 0.02 + 0.01
    s2 = torch.rand(experts, hidden // 128, intermediate // 128, device=device) * 0.02 + 0.01
    ids = torch.stack([torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]).int()
    weights = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1)
    spec = MoeOpSpec(
        experts, experts, hidden, intermediate, top_k,
        torch.bfloat16, torch.float8_e4m3fn, (128, 128), 1, True,
        scale_dtype=torch.float32,
    )
    graphs = []
    for provider in providers:
        provider.prepare(spec, device=device, tp_rank=0, ep_rank=0)

        def run():
            return provider.run(
                spec, x, ids, weights, w13, w2, s13, s2,
                local_expert_start=0, tp_rank=0, ep_rank=0,
            )

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        graphs.append((graph, output, provider))
    for _ in range(3):
        x.normal_()
        ids.copy_(torch.stack([torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]))
        weights.copy_(torch.softmax(torch.randn_like(weights), dim=-1))
        expected = _reference_moe(x, w13, w2, s13, s2, ids, weights, "up_gate")
        for graph, output, _provider in graphs:
            graph.replay()
            assert torch.isfinite(output).all()
            relative_l2 = torch.linalg.vector_norm(output.float() - expected.float())
            relative_l2 /= torch.linalg.vector_norm(expected.float())
            assert relative_l2 < 0.05
            assert F.cosine_similarity(output.float(), expected.float()).min() > 0.995


@pytest.mark.parametrize(
    ("hidden", "intermediate"),
    [(128, 128), (3072, 384)],
)
def test_minimax_fused_gate_up_matches_generic_fp8_pipeline(hidden, intermediate):
    torch.manual_seed(71)
    device = torch.device("cuda")
    tokens, experts, top_k = 4, 4, 2
    hidden_states = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w13_weight = _fp8_weight((experts, 2 * intermediate, hidden), device)
    w2_weight = _fp8_weight((experts, hidden, intermediate), device)
    w13_scale = torch.rand(
        experts,
        2 * intermediate // 128,
        hidden // 128,
        device=device,
    ) + 0.25
    w2_scale = torch.rand(
        experts,
        hidden // 128,
        intermediate // 128,
        device=device,
    ) + 0.25
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    arguments = (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
    )
    kwargs = {"num_experts": experts, "local_expert_start": 0}

    expected = fused_moe_fp8(*arguments, **kwargs, gate_up_order="gate_up")
    actual = fused_minimax_m2_moe_fp8(*arguments, **kwargs)
    torch.cuda.synchronize()

    _assert_fp8_pipeline_close(actual, expected)
