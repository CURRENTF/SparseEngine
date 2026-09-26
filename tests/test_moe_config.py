import pytest
import torch

from sparseengine.kernels.triton.moe_config import resolve_moe_gemm_config


def _unknown_shape_config():
    return resolve_moe_gemm_config(
        dtype=torch.float16,
        num_tokens=8,
        top_k=2,
        num_local_experts=16,
        hidden_size=64,
        intermediate_size=32,
        stage="w13",
        device_name="unprofiled device",
        device_capability=(9, 0),
    )


def test_unknown_shape_uses_a_deterministic_valid_config():
    first = _unknown_shape_config()
    second = _unknown_shape_config()

    assert first == second
    assert first.block_m > 0
    assert first.block_n > 0
    assert first.block_k > 0
    assert first.num_warps > 0
    assert first.num_stages > 0


def test_moe_config_rejects_unknown_stage():
    arguments = dict(
        dtype=torch.bfloat16,
        num_tokens=16,
        top_k=8,
        num_local_experts=64,
        hidden_size=2048,
        intermediate_size=768,
        device_name="NVIDIA H20",
        device_capability=(9, 0),
    )

    with pytest.raises(ValueError, match="stage"):
        resolve_moe_gemm_config(**arguments, stage="w3")


@pytest.mark.parametrize("stage", ["w13", "w2"])
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16])
def test_h20_ep2_profile_accepts_measured_decode_capacities(stage, num_tokens):
    from sparseengine.kernels.triton.moe_config import (
        MoeGemmShape,
        _glm_sm90_tp2_ep2_config,
    )

    shape = MoeGemmShape("h20", (9, 0), torch.bfloat16, 4, 32, 2048, 1536)
    config = _glm_sm90_tp2_ep2_config(shape, num_tokens=num_tokens, stage=stage)
    assert config is not None
    assert all(value > 0 for value in config.as_triton_kwargs().values())


@pytest.mark.parametrize(
    "change,num_tokens,stage",
    [
        ({}, 17, "w13"),
        ({}, 32, "w2"),
        ({}, 4, "gate_up_swiglu"),
        ({"hardware": "unprofiled"}, 4, "w13"),
        ({"capability": (8, 0)}, 4, "w13"),
        ({"dtype": torch.float16}, 4, "w13"),
        ({"num_local_experts": 64}, 4, "w13"),
        ({"intermediate_size": 768}, 4, "w13"),
        ({"top_k": 8}, 4, "w13"),
    ],
)
def test_h20_ep2_profile_misses_preserve_portfolio(change, num_tokens, stage):
    from dataclasses import replace

    from sparseengine.kernels.triton.moe_config import (
        MoeGemmShape,
        _glm_sm90_tp2_ep2_config,
    )

    shape = MoeGemmShape("h20", (9, 0), torch.bfloat16, 4, 32, 2048, 1536)
    assert _glm_sm90_tp2_ep2_config(
        replace(shape, **change), num_tokens=num_tokens, stage=stage
    ) is None


@pytest.mark.parametrize("stage", ["w13", "w2"])
@pytest.mark.parametrize("num_tokens", range(1, 17))
def test_h20_packed_decode_profile_resolves_measured_interval(stage, num_tokens):
    from sparseengine.kernels.triton.moe_config import (
        MoeGemmShape,
        _glm_h20_packed_decode_config,
    )

    shape = MoeGemmShape("h20", (9, 0), torch.bfloat16, 5, 65, 2048, 1536)
    config = _glm_h20_packed_decode_config(shape, num_tokens=num_tokens, stage=stage)
    assert config is not None
    resolved = resolve_moe_gemm_config(
        dtype=shape.dtype,
        num_tokens=num_tokens,
        top_k=shape.top_k,
        num_local_experts=shape.num_local_experts,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        stage=stage,
        device_name="NVIDIA H20",
        device_capability=shape.capability,
    )
    assert resolved == config


@pytest.mark.parametrize(
    "change,num_tokens,stage",
    [
        ({}, 0, "w13"),
        ({}, 17, "w13"),
        ({}, 32, "w2"),
        ({}, 4, "gate_up_swiglu"),
        ({"hardware": "h100"}, 4, "w13"),
        ({"capability": (8, 0)}, 4, "w13"),
        ({"dtype": torch.float16}, 4, "w13"),
        ({"num_local_experts": 64}, 4, "w13"),
        ({"hidden_size": 4096}, 4, "w13"),
        ({"intermediate_size": 768}, 4, "w13"),
        ({"top_k": 4}, 4, "w13"),
    ],
)
def test_h20_packed_decode_profile_misses(change, num_tokens, stage):
    from dataclasses import replace

    from sparseengine.kernels.triton.moe_config import (
        MoeGemmShape,
        _glm_h20_packed_decode_config,
    )

    shape = MoeGemmShape("h20", (9, 0), torch.bfloat16, 5, 65, 2048, 1536)
    assert _glm_h20_packed_decode_config(
        replace(shape, **change), num_tokens=num_tokens, stage=stage
    ) is None
