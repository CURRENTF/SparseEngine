"""Branch scratch ownership and real FP8 MoE fork/join correctness."""
from unittest.mock import patch

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from sparseengine.operators.fp8_linear import (
    Fp8LinearProvider, Fp8LinearDispatchRoute, Sm120Fp8LinearDispatchPlan,
    _sm120_activation_buffers,
)
from sparseengine.operators.moe_execution import prepare_model_moe_execution
from sparseengine.operators.workspace import bind_module_workspace_lane, close_workspace_manager


def test_activation_scratch_reuses_within_lane_but_not_across_branches():
    x = torch.empty(5, 256, dtype=torch.bfloat16)
    first = _sm120_activation_buffers(x, lane="test_routed")
    next_layer = _sm120_activation_buffers(x[:4], rows=7, lane="test_routed")
    shared = _sm120_activation_buffers(x, lane="test_shared")
    for left, reused, right in zip(first, next_layer, shared):
        assert left.data_ptr() == reused.data_ptr()
        assert left.data_ptr() != right.data_ptr()
        left.fill_(2)
        right.fill_(3)
        assert torch.all(left.float() == 2)


def test_shared_lane_reaches_linear_dispatch_routes_and_packed_projections():
    from sparseengine.layers.linear import LinearBase
    from sparseengine.layers.packed_moe import PackedMoeExperts
    from sparseengine.operators.moe import MoeProvider

    # Exercise real ownership hooks without requiring GPU provider resolution.
    plan = object.__new__(Sm120Fp8LinearDispatchPlan)
    Fp8LinearProvider.__init__(plan)
    plan.routes = (Fp8LinearDispatchRoute(0, None, Fp8LinearProvider()),)
    linear = object.__new__(LinearBase)
    nn.Module.__init__(linear)
    linear.quant_provider = plan
    packed = object.__new__(PackedMoeExperts)
    nn.Module.__init__(packed)
    packed.provider = MoeProvider()
    projection = Fp8LinearProvider()
    packed.provider._shared_projections = ((projection, None, None),)
    bind_module_workspace_lane(nn.ModuleList([linear, packed]), "shared")
    assert plan.routes[0].provider.workspace_lane == "shared"
    assert projection.workspace_lane == "shared"
    plan.routes[0].provider._workspace_used = True
    with pytest.raises(RuntimeError, match="before warmup"):
        plan.bind_workspace_lane("late")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("model_kind", ["glm_block", "glm_tensor", "glm_packed", "qwen"])
def test_quantized_model_branches_overlap_eager_and_graph(tmp_path, model_kind, record_property):
    from sparseengine.distributed import init_parallel_context, reset_parallel_context, ParallelTopology
    from sparseengine.models.glm4_moe_lite import Glm4MoeLiteSparseMoeBlock
    from sparseengine.models.qwen3_5_moe import Qwen35MoeSparseMoeBlock
    from sparseengine.quantization.config import QuantizationConfig
    from sparseengine.quantization.fp8 import fp8_blockwise_linear_reference
    from tests.test_glm4_moe_lite import _config
    from tests.test_triton_fp8_operators import _reference_moe

    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}",
                                        rank=0, world_size=1)
    init_parallel_context(topology=ParallelTopology(1, 1, 1))
    try:
        tensor_scales = model_kind in {"glm_tensor", "glm_packed"}
        quant = {"quant_method": "fp8", "activation_scheme": "dynamic"}
        quant.update({"is_checkpoint_fp8_serialized": True} if tensor_scales
                     else {"weight_block_size": [128, 128]})
        config = _config(hidden_size=256, moe_intermediate_size=128)
        config.quantization_config = QuantizationConfig.from_hf_config(quant)
        config.moe_max_num_tokens = 8
        config.decode_graph = True
        config.hidden_act = "silu"
        torch.manual_seed(67)
        with torch.device("cuda"):
            if model_kind == "qwen":
                config.num_experts = 128
                config.num_experts_per_tok = 8
                config.shared_expert_intermediate_size = 128
                config.norm_topk_prob = True
                block = Qwen35MoeSparseMoeBlock(config)
            else:
                with patch("sparseengine.models.glm4_moe_lite.use_packed_shared_experts",
                           return_value=model_kind == "glm_packed"):
                    block = Glm4MoeLiteSparseMoeBlock(config, mlp_chunk_size=8, decode_graph=True)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.copy_(torch.randn_like(parameter, dtype=torch.float32).mul_(.2).to(parameter.dtype))
            for name, buffer in block.named_buffers():
                if "scale_inv" in name:
                    buffer.fill_(.125)
            # Router requires BF16 inputs on Qwen; keep GLM's FP32 router.
            if model_kind == "qwen":
                block.gate.weight.data = block.gate.weight.data.bfloat16()
            shared = getattr(block, "shared_experts", getattr(block, "shared_expert", None))
            if shared is not None:
                for module in shared.modules():
                    if getattr(module, "quant_provider", None) is not None:
                        module.quant_provider.prepare_weights(module.weight, module.weight_scale_inv)
            else:
                experts = block.experts
                idx = experts.shared_expert_id
                experts.provider.prepare_shared_expert(
                    experts.op_spec, experts.w13_weight[idx], experts.w2_weight[idx],
                    experts.w13_scale_inv[idx], experts.w2_scale_inv[idx])
        prepare_model_moe_execution(block, torch.device("cuda"))
        assert block.moe_execution.stream is not None
        record_property("device", torch.cuda.get_device_name())
        record_property("routed_provider", block.experts.provider.name)
        record_property("shared_providers", ",".join(
            module.quant_provider.name for module in shared.modules()
            if getattr(module, "quant_provider", None) is not None
        ) if shared is not None else ",".join(
            provider.name for provider, _, _ in block.experts.provider._shared_projections))
        # Force the unfused path for the packed owner by using > its fusion limit.
        batches = (1, 7) if model_kind != "glm_packed" else (5, 7)

        def linear_reference(x, weight, scales):
            if not tensor_scales:
                return fp8_blockwise_linear_reference(x, weight, scales)
            # Tensor-scaled weights use per-token, rather than per-block, inputs.
            scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
            quantized = (x.float() / scale).to(torch.float8_e4m3fn).float() * scale
            channel_scales = scales[:, 0].repeat_interleave(128)[:weight.shape[0]]
            return F.linear(quantized, weight.float() * channel_scales[:, None]).bfloat16()

        def reference(x):
            if model_kind == "qwen":
                logits = F.linear(x, block.gate.weight).float()
                probs = logits[:, :-1].softmax(-1)
                routes, ids = probs.topk(config.num_experts_per_tok, dim=-1)
                routes = routes / routes.sum(-1, keepdim=True)
                gate_logits = logits[:, -1:]
            else:
                probs = F.linear(x.float(), block.gate.weight.float()).sigmoid()
                ids = (probs + block.gate.e_score_correction_bias).topk(config.num_experts_per_tok, dim=-1).indices
                routes = probs.gather(1, ids)
                routes = routes / routes.sum(-1, keepdim=True) * config.routed_scaling_factor
            experts = block.experts
            if tensor_scales:
                routed = torch.zeros_like(x, dtype=torch.float32)
                for token in range(x.shape[0]):
                    for slot in range(ids.shape[1]):
                        expert = int(ids[token, slot])
                        packed = linear_reference(x[token:token+1], experts.w13_weight[expert],
                                                  experts.w13_scale_inv[expert])
                        gate, up = packed.float().chunk(2, -1)
                        if experts.provider.gate_up_order == "up_gate":
                            gate, up = up, gate
                        term = linear_reference((F.silu(gate) * up).bfloat16(),
                                                experts.w2_weight[expert], experts.w2_scale_inv[expert])
                        routed[token] += term[0].float() * routes[token, slot]
            else:
                routed = _reference_moe(x, experts.w13_weight, experts.w2_weight,
                                        experts.w13_scale_inv, experts.w2_scale_inv,
                                        ids, routes, experts.provider.gate_up_order)
            if shared is None:
                idx = experts.shared_expert_id
                w13, w2 = experts.w13_weight[idx], experts.w2_weight[idx]
                s13, s2 = experts.w13_scale_inv[idx], experts.w2_scale_inv[idx]
            else:
                w13, w2 = shared.gate_up_proj.weight, shared.down_proj.weight
                s13, s2 = shared.gate_up_proj.weight_scale_inv, shared.down_proj.weight_scale_inv
            gate, up = linear_reference(x, w13, s13).float().chunk(2, -1)
            dense = linear_reference((F.silu(gate) * up).bfloat16(), w2, s2)
            if model_kind == "qwen":
                dense = dense * gate_logits.sigmoid()
            return routed.float() + dense.float()

        with torch.inference_mode():
            for batch in batches:
                x = torch.randn(batch, 256, device="cuda", dtype=torch.bfloat16)
                for _ in range(3):
                    block(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = block(x)
                for _ in range(4):
                    x.normal_()
                    eager = block(x)
                    graph.replay()
                    expected = reference(x)
                    torch.testing.assert_close(output, eager, rtol=0, atol=0)
                    relative = (output.float() - expected).norm() / expected.norm()
                    assert relative < .06
                    assert F.cosine_similarity(output.float().flatten(), expected.flatten(), dim=0) > .998
                graph.reset()
    finally:
        close_workspace_manager()
        reset_parallel_context()
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_branch_scratch_isolation_across_streams_and_graph_replays():
    """Exercise the SM120 scratch allocator on any CUDA device, without its GEMM."""
    from sparseengine.distributed.moe_communication import AllReduceMoeCommunication
    from sparseengine.operators.moe_execution import MoeExecutionPlan

    class ScratchBranch(nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset
            self.lane = "scratch_test_main"

        def bind_workspace_lane(self, lane):
            self.lane = lane

        def forward(self, x):
            scratch, _, _ = _sm120_activation_buffers(x, lane=self.lane)
            torch.add(x, self.offset, out=scratch)
            return scratch.square()

    model = nn.Module()
    model.routed = ScratchBranch(1)
    model.shared = ScratchBranch(-3)
    model.moe_execution = MoeExecutionPlan(
        routed=model.routed, shared=model.shared, shared_modules=(model.shared,),
        communication=AllReduceMoeCommunication(lambda x: x), chunk_size=None)
    prepare_model_moe_execution(model, torch.device("cuda"))
    with torch.inference_mode():
        for batch in (1, 7):
            x = torch.randn(batch, 256, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                model.moe_execution(x, is_prefill=False)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = model.moe_execution(x, is_prefill=False)
            for _ in range(12):
                x.normal_()
                graph.replay()
                torch.testing.assert_close(output, (x + 1).square() + (x - 3).square(), rtol=0, atol=0)
            graph.reset()
