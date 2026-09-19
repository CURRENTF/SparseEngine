from types import SimpleNamespace

import torch

from sparseengine.layers.packed_moe import PackedMoeExperts
from sparseengine.models.minimax_m2 import MiniMaxM2PackedExperts
from sparseengine.models.qwen3_moe import Qwen3MoePackedExperts
from sparseengine.operators.moe import TritonMoeProvider


def _parallel_context(*, tp_rank=0, tp_size=1, ep_rank=0, ep_size=1):
    return SimpleNamespace(
        moe_tp_rank=tp_rank,
        moe_tp_size=tp_size,
        moe_ep_rank=ep_rank,
        moe_ep_size=ep_size,
    )


def _experts(**overrides) -> PackedMoeExperts:
    values = {
        "num_experts": 4,
        "hidden_size": 8,
        "intermediate_size": 8,
        "top_k": 2,
        "activation_dtype": torch.bfloat16,
        "fp8_enabled": False,
        "cuda_graph": False,
        "routing_method": "biased_sigmoid",
        "model_label": "TestGLM",
        "provider_resolver": lambda spec: TritonMoeProvider(),
        "parallel_context": _parallel_context(),
    }
    values.update(overrides)
    return PackedMoeExperts(**values)


def test_model_packed_experts_use_shared_physical_module() -> None:
    assert issubclass(Qwen3MoePackedExperts, PackedMoeExperts)
    assert issubclass(MiniMaxM2PackedExperts, PackedMoeExperts)


def test_packed_experts_accept_glm_router_contract_without_owning_router() -> None:
    experts = _experts(num_experts=64, top_k=4)

    assert experts.op_spec.routing_method == "biased_sigmoid"
    assert experts.op_spec.top_k == 4
    assert experts.w13_weight.shape == (64, 16, 8)
    assert experts.w2_weight.shape == (64, 8, 8)


def test_packed_experts_load_and_validate_rank_local_projections() -> None:
    experts = _experts(
        num_experts=4,
        parallel_context=_parallel_context(ep_rank=1, ep_size=2),
    )
    for global_expert_id in range(2, 4):
        for projection, shape in (
            ("gate_proj", (8, 8)),
            ("up_proj", (8, 8)),
            ("down_proj", (8, 8)),
        ):
            experts.load_expert_weight(
                global_expert_id,
                projection,
                torch.full(shape, float(global_expert_id), dtype=torch.bfloat16),
            )

    experts.validate_loaded_weights()
    assert experts.local_expert_start == 2
    assert experts.local_expert_end == 4
