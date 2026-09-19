"""Scalar-scale checkpoints must preserve weights across packing and TP slicing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sparseengine.layers.linear import (
    AbsorbedColumnParallelLinear,
    MergedReplicatedLinear,
    RowParallelLinear,
)
from sparseengine.quantization.config import QuantizationConfig
from sparseengine.quantization.fp8 import expand_fp8_tensor_scale
from sparseengine.utils.loader import _read_safetensors_shard, load_model


def _quantization():
    return QuantizationConfig.from_hf_config({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "is_checkpoint_fp8_serialized": True,
    }, model_name="GLM-4.7-Flash")


def _weight(shape):
    return torch.randn(shape).clamp(-4, 4).to(torch.float8_e4m3fn)


def test_missing_generation_config_preserves_all_model_stop_ids(tmp_path):
    from sparseengine.engine.llm_engine import _resolve_eos_token_ids

    config = SimpleNamespace(eos_token_id=[154820, 154827, 154829])
    assert _resolve_eos_token_ids(str(tmp_path), config, 154820) == (154820, 154827, 154829)


def test_existing_generation_config_overrides_model_stop_ids(tmp_path):
    from transformers import GenerationConfig
    from sparseengine.engine.llm_engine import _resolve_eos_token_ids

    GenerationConfig(eos_token_id=[5, 6]).save_pretrained(tmp_path)
    assert _resolve_eos_token_ids(str(tmp_path), SimpleNamespace(eos_token_id=7), 6) == (5, 6)


def test_malformed_generation_config_does_not_use_model_defaults(tmp_path):
    from sparseengine.engine.llm_engine import _resolve_eos_token_ids

    (tmp_path / "generation_config.json").write_text("invalid JSON")
    with pytest.raises(OSError):
        _resolve_eos_token_ids(str(tmp_path), SimpleNamespace(eos_token_id=7), 6)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (4, 2)])
def test_scalar_checkpoint_load_preserves_merged_and_tp_weights(tmp_path, tp_size, tp_rank):
    torch.manual_seed(23)
    quantization = _quantization()
    # Round-tripping must not reinterpret the serialized checkpoint as block-wise.
    assert QuantizationConfig.from_hf_config(quantization.to_dict()).checkpoint_scale_layout == "per_tensor"
    context = SimpleNamespace(attn_tp_size=tp_size, attn_tp_rank=tp_rank)
    with (
        patch("sparseengine.layers.linear.get_parallel_context", return_value=context),
        patch("sparseengine.layers.linear.QuantizationRegistry.resolve_linear_provider"),
    ):
        model = nn.Module()
        model.packed_modules_mapping = {"q_a": ("qkv_a", 0), "kv_a": ("qkv_a", 1)}
        model.qkv_a = MergedReplicatedLinear(128, [768, 576], quantization=quantization)
        # Local output width has a partial block at TP4, as GLM kv_b does.
        model.kv_b = AbsorbedColumnParallelLinear(128, 896, quantization=quantization)
        model.out = RowParallelLinear(512, 128, quantization=quantization)
    checkpoint = {}
    for name, shape, scale in (
        ("q_a", (768, 128), 0.125),
        ("kv_a", (576, 128), 0.25),
        ("kv_b", (896, 128), 0.5),
        ("out", (128, 512), 0.0625),
    ):
        checkpoint[f"{name}.weight"] = _weight(shape)
        checkpoint[f"{name}.weight_scale"] = torch.tensor(scale, dtype=torch.bfloat16)
    save_file(checkpoint, tmp_path / "model.safetensors")
    load_model(model, str(tmp_path))

    expected_merged = torch.cat([checkpoint["q_a.weight"].float() * 0.125,
                                 checkpoint["kv_a.weight"].float() * 0.25])
    expanded = model.qkv_a.weight_scale_inv.repeat_interleave(128, 0)[:1344]
    torch.testing.assert_close(model.qkv_a.weight.float() * expanded, expected_merged, rtol=0, atol=0)
    expected_kv = checkpoint["kv_b.weight"].float().chunk(tp_size, 0)[tp_rank] * 0.5
    torch.testing.assert_close(model.kv_b.absorbed_weight, expected_kv.bfloat16(), rtol=0, atol=0)
    expected_out = checkpoint["out.weight"].float().chunk(tp_size, 1)[tp_rank]
    torch.testing.assert_close(model.out.weight.float(), expected_out, rtol=0, atol=0)
    torch.testing.assert_close(model.out.weight_scale_inv, torch.full_like(model.out.weight_scale_inv, 0.0625))


@pytest.mark.parametrize("bad_scale", [torch.tensor(float("nan")), torch.tensor(0.), torch.ones(2)])
def test_invalid_tensor_scale_fails_before_weight_copy(bad_scale):
    with pytest.raises(ValueError, match="weight_scale"):
        expand_fp8_tensor_scale(_weight((128, 128)), bad_scale)


def test_duplicate_scale_formats_are_rejected(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"proj.weight": _weight((128, 128)),
               "proj.weight_scale": torch.tensor(0.5),
               "proj.weight_scale_inv": torch.ones(1, 1)}, path)
    with pytest.raises(ValueError, match="both tensor and block"):
        _read_safetensors_shard(str(path))


def test_skipped_expert_scales_follow_weight_ownership(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"remote.weight": _weight((128, 128)),
               "remote.weight_scale": torch.tensor(0.5)}, path)
    model = nn.Module()
    model.map_weight_name = lambda name: None if name == "remote.weight" else name
    shard = _read_safetensors_shard(str(path), model)
    assert not shard.tensors
    assert "remote.weight_scale_inv" in shard.metadata


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (4, 1)])
def test_glm_checkpoint_loads_dense_shared_and_routed_projections(tmp_path, tp_size, tp_rank):
    from tests.test_glm4_moe_lite import _config, _construction_context, _tp_context
    from sparseengine.debug.tiny_random import build_tiny_random_hf_model
    from sparseengine.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
    from sparseengine.operators.mla_attention import MlaAttentionOpSpec

    config = _config(hidden_size=128, intermediate_size=512, moe_intermediate_size=512,
                     q_lora_rank=128, kv_lora_rank=128, num_attention_heads=4,
                     num_key_value_heads=4, qk_nope_head_dim=64, v_head_dim=128,
                     n_routed_experts=4, num_experts_per_tok=2)
    reference = build_tiny_random_hf_model(config, seed=37)
    config.quantization_config = _quantization()
    mla = SimpleNamespace(hidden_size=128, projection_chunk_size=8,
                          spec=MlaAttentionOpSpec(4, 128, 64, 128, 128,
                                                  torch.bfloat16, torch.bfloat16, tp_size, False))
    with _construction_context(_tp_context(tp_rank, tp_size)), patch(
        "sparseengine.layers.linear.QuantizationRegistry.resolve_linear_provider"
    ):
        model = Glm4MoeLiteForCausalLM(config, mla_attention=mla,
                                      mlp_chunk_size=8, decode_graph=False)
    checkpoint = {}
    for name, weight in model.iter_tiny_reference_weights(reference.state_dict()):
        if weight.ndim == 2 and name.endswith(".weight") and not any(
            part in name for part in ("embed_tokens", "lm_head", ".gate.")
        ):
            checkpoint[name] = (weight.float() / 0.015625).to(torch.float8_e4m3fn)
            checkpoint[name.removesuffix(".weight") + ".weight_scale"] = torch.tensor(0.015625, dtype=torch.bfloat16)
        else:
            checkpoint[name] = weight.contiguous()
    save_file(checkpoint, tmp_path / "model.safetensors")
    load_model(model, str(tmp_path))
    expert = model.model.layers[1].mlp.experts
    expected = checkpoint["model.layers.1.mlp.experts.0.gate_proj.weight"].float().chunk(tp_size, 0)[tp_rank]
    torch.testing.assert_close(expert.w13_weight[0, :expert.intermediate_size].float(), expected, rtol=0, atol=0)
    assert len(expert._loaded_expert_shards) == config.n_routed_experts * 3
    assert model.model.layers[0].mlp.gate_up_proj._quantized_weight_loaded
    assert model.model.layers[1].mlp.shared_experts.gate_up_proj._quantized_weight_loaded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an idle CUDA GPU")
def test_glm_fp8_projection_graph_matches_independent_quantized_math():
    from sparseengine.operators.fp8_linear import resolve_fp8_linear_provider

    torch.manual_seed(29)
    x = torch.randn(3, 2048, device="cuda", dtype=torch.bfloat16)
    weight = _weight((1344, 2048)).cuda()
    scalar = torch.tensor(0.03125, device="cuda", dtype=torch.bfloat16)
    scales = expand_fp8_tensor_scale(weight, scalar).float().contiguous()
    provider = resolve_fp8_linear_provider((128, 128), input_features=2048, output_features=1344)
    # Independent W8A8 block arithmetic; no production quantizer or GEMM in oracle.
    def reference(inputs):
        blocks = inputs.float().reshape(3, 16, 128)
        a_scale = (blocks.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-10)
        restored = ((blocks / a_scale).to(torch.float8_e4m3fn).float() * a_scale).reshape(3, 2048)
        return (restored @ (weight.float() * scalar.float()).T).bfloat16()
    for _ in range(3):
        actual = provider(x, weight, scales)
    torch.testing.assert_close(actual, reference(x), atol=0.05, rtol=0.04)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = provider(x, weight, scales)
    for _ in range(2):
        x.normal_()
        graph.replay()
        torch.testing.assert_close(captured, reference(x), atol=0.05, rtol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an idle CUDA GPU")
def test_tensor_fp8_linear_preserves_merged_scales_and_graph_inputs():
    """Catch accidental fused-scale collapse and stale activation quantization on replay."""
    from sparseengine.operators.fp8_linear import resolve_fp8_linear_provider

    torch.manual_seed(91)
    weight = _weight((1344, 2048)).cuda()
    scales = torch.full((11, 16), 0.03125, device="cuda")
    scales[6:] = 0.125
    provider = resolve_fp8_linear_provider(
        (128, 128), input_features=2048, output_features=1344,
        weight_layout_id="tensor_scales_in_block_grid")
    provider.prepare_weights(weight, scales)
    x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(1344, device="cuda", dtype=torch.bfloat16)
    restored_weight = weight.float() * scales[:, :1].repeat_interleave(128, 0)[:1344]

    def reference():
        scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        restored = (x.float() / scale).to(torch.float8_e4m3fn).float() * scale
        return (restored @ restored_weight.T + bias.float()).bfloat16()

    for _ in range(3):
        out = provider(x, weight, scales, bias)
    torch.testing.assert_close(out, reference(), atol=0.0625, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = provider(x, weight, scales, bias)
    for factor in (0., 1., 32.):
        x.normal_().mul_(factor)
        graph.replay()
        torch.testing.assert_close(out, reference(), atol=2. if factor == 32 else 0.0625, rtol=0.02)
    bad_scales = scales.clone()
    bad_scales[0, 1] *= 2
    with pytest.raises(ValueError, match="varying along K"):
        provider.prepare_weights(weight, bad_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an idle CUDA GPU")
@pytest.mark.parametrize("tensor_scales", [False, True])
def test_glm_fp8_experts_graph_matches_quantized_reference(tensor_scales):
    from tests.test_triton_fp8_operators import _reference_moe, _assert_fp8_pipeline_close
    from sparseengine.operators.moe import MoeOpSpec, resolve_moe_provider

    torch.manual_seed(41)
    spec = MoeOpSpec(num_experts=64, num_local_experts=64, hidden_size=2048,
                     intermediate_size=1536, top_k=4, activation_dtype=torch.bfloat16,
                     weight_dtype=torch.float8_e4m3fn, block_shape=None if tensor_scales else (128, 128),
                     ep_size=1, cuda_graph=True, routing_method="biased_sigmoid",
                     scale_dtype=torch.float32, max_num_tokens=3)
    provider = resolve_moe_provider(spec)
    x = torch.randn(3, 2048, dtype=torch.bfloat16, device="cuda")
    gate_up = torch.randn(64, 3072, 2048, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    down = torch.randn(64, 2048, 1536, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    scale13 = torch.full((64, 24, 16), 0.015625, device="cuda")
    scale2 = torch.full((64, 16, 12), 0.015625, device="cuda")
    ids = torch.tensor([[0, 7, 31, 63], [7, 8, 31, 42], [0, 2, 9, 60]], device="cuda", dtype=torch.int32)
    routes = torch.rand(3, 4, device="cuda")
    routes = routes / routes.sum(-1, keepdim=True) * 1.8
    provider.prepare(spec, device=x.device, tp_rank=0, ep_rank=0)
    def reference():
        if not tensor_scales:
            return _reference_moe(x, gate_up, down, scale13, scale2, ids, routes, provider.gate_up_order)
        def quantize(value):
            scale = value.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
            return (value.float() / scale).to(torch.float8_e4m3fn).float() * scale
        expected = torch.zeros_like(x)
        for token in range(x.shape[0]):
            terms = []
            for slot in range(ids.shape[1]):
                expert = int(ids[token, slot])
                packed = (quantize(x[token:token+1]) @ (gate_up[expert].float() * .015625).T).bfloat16()
                first, second = packed.chunk(2, -1)
                gate, up = (first, second) if provider.gate_up_order == "gate_up" else (second, first)
                activated = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
                out = (quantize(activated) @ (down[expert].float() * .015625).T) * routes[token, slot]
                terms.append(out.bfloat16())
            expected[token] = torch.stack(terms).float().sum(0).bfloat16()
        return expected
    def run():
        return provider.run(spec, x, ids, routes, gate_up, down, scale13, scale2,
                            local_expert_start=0, tp_rank=0, ep_rank=0)
    for _ in range(3):
        actual = run()
    expected = reference()
    _assert_fp8_pipeline_close(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    x.normal_()
    graph.replay()
    expected = reference()
    _assert_fp8_pipeline_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an idle CUDA GPU")
def test_packed_tensor_shared_expert_matches_separate_paths_on_replay():
    """Protect packed scale slicing and the shared-expert prefill/decode split."""
    from sparseengine.models.glm4_moe_lite import Glm4MoeLitePackedExperts
    from tests.test_glm4_moe_lite import _tp_context
    torch.manual_seed(53)
    config = SimpleNamespace(n_routed_experts=64, num_experts_per_tok=4, n_shared_experts=1,
                             hidden_size=2048, moe_intermediate_size=1536,
                             dtype=torch.bfloat16, quantization_config=_quantization(), moe_max_num_tokens=4)
    with (torch.device("cuda"),
          patch("sparseengine.models.glm4_moe_lite.get_parallel_context", return_value=_tp_context()),
          patch("sparseengine.models.glm4_moe_lite.use_packed_shared_experts", return_value=True)):
        experts = Glm4MoeLitePackedExperts(config, decode_graph=True)
    experts.w13_weight.data.copy_(torch.randn_like(experts.w13_weight, dtype=torch.bfloat16).to(torch.float8_e4m3fn))
    experts.w2_weight.data.copy_(torch.randn_like(experts.w2_weight, dtype=torch.bfloat16).to(torch.float8_e4m3fn))
    experts.w13_scale_inv.fill_(.015625)
    experts.w13_scale_inv[:,12:].fill_(.03125)
    experts.w2_scale_inv.fill_(.015625)
    experts.provider.prepare_shared_expert(experts.op_spec, experts.w13_weight[-1], experts.w2_weight[-1],
                                          experts.w13_scale_inv[-1], experts.w2_scale_inv[-1])
    x = torch.randn(4,2048,device="cuda",dtype=torch.bfloat16)
    ids = torch.arange(16,device="cuda",dtype=torch.int32).view(4,4)
    routes = torch.full((4,4),.45,device="cuda")
    for _ in range(3):
        experts.forward_routed_and_shared(x,ids,routes)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = experts.forward_routed_and_shared(x,ids,routes)
    for _ in range(2):
        x.normal_()
        graph.replay()
        expected = experts(x,ids,routes) + experts.forward_shared(x)
        relative = (actual.float()-expected.float()).norm()/expected.float().norm()
        assert relative < .02
