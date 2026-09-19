import pytest
import torch

from sparseengine.quantization.fp8 import fp8_blockwise_linear_reference
from sparseengine.quantization.fp8_tp import Fp8ExpertTpShard


def test_qwen_tp4_uses_block_aligned_physical_shards():
    plans = [Fp8ExpertTpShard(768, rank, 4) for rank in range(4)]

    assert [plan.logical_size for plan in plans] == [192] * 4
    assert [plan.physical_size for plan in plans] == [256] * 4
    assert [(plan.aligned_start, plan.aligned_stop) for plan in plans] == [
        (0, 256),
        (128, 384),
        (384, 640),
        (512, 768),
    ]


@pytest.mark.parametrize("down_projection", [False, True])
def test_fp8_tp_slice_and_padding_preserve_logical_partition(down_projection):
    hidden_size = 128
    global_size = 768
    weight_shape = (
        (hidden_size, global_size) if down_projection else (global_size, hidden_size)
    )
    scale_shape = (
        (hidden_size // 128, global_size // 128)
        if down_projection
        else (global_size // 128, hidden_size // 128)
    )
    weight = torch.ones(weight_shape).to(torch.float8_e4m3fn)
    scale = torch.ones(scale_shape, dtype=torch.bfloat16)

    for rank in range(4):
        plan = Fp8ExpertTpShard(global_size, rank, 4)
        weight_slice = plan.checkpoint_slice(
            weight_shape,
            hidden_size=hidden_size,
            down_projection=down_projection,
            is_scale=False,
        )
        scale_slice = plan.checkpoint_slice(
            scale_shape,
            hidden_size=hidden_size,
            down_projection=down_projection,
            is_scale=True,
        )
        local_weight, local_scale = plan.prepare_projection(
            weight[weight_slice],
            scale[scale_slice],
            hidden_size=hidden_size,
            down_projection=down_projection,
        )

        intermediate_axis = 1 if down_projection else 0
        logical_sum = local_weight.float().sum(dim=1 - intermediate_axis)
        expected_nonzero = plan.logical_size if down_projection else plan.physical_size
        assert torch.count_nonzero(logical_sum).item() == expected_nonzero
        assert tuple(local_scale.shape) == ((1, 2) if down_projection else (2, 1))


def test_aligned_minimax_tp_shard_does_not_copy_weight():
    plan = Fp8ExpertTpShard(1536, 1, 4)
    weight = torch.ones(384, 128).to(torch.float8_e4m3fn)
    scale = torch.ones(3, 1)

    local_weight, local_scale = plan.prepare_projection(
        weight,
        scale,
        hidden_size=128,
        down_projection=False,
    )

    assert local_weight.data_ptr() == weight.data_ptr()
    assert local_scale.data_ptr() == scale.data_ptr()


def test_overlapping_qwen_tp_blocks_preserve_fp8_expert_math():
    torch.manual_seed(23)
    hidden_size, intermediate_size = 128, 768
    inputs = torch.randn(2, hidden_size, dtype=torch.bfloat16)
    gate = (
        torch.randn(intermediate_size, hidden_size).clamp(-4, 4).to(torch.float8_e4m3fn)
    )
    up = torch.randn_like(gate.float()).clamp(-4, 4).to(torch.float8_e4m3fn)
    down = (
        torch.randn(hidden_size, intermediate_size).clamp(-4, 4).to(torch.float8_e4m3fn)
    )
    gate_scale = torch.rand(6, 1) + 0.1
    up_scale = torch.rand(6, 1) + 0.1
    down_scale = torch.rand(1, 6) + 0.1
    full_gate = fp8_blockwise_linear_reference(inputs, gate, gate_scale)
    full_up = fp8_blockwise_linear_reference(inputs, up, up_scale)
    full_activation = (torch.nn.functional.silu(full_gate) * full_up).to(torch.bfloat16)
    expected = fp8_blockwise_linear_reference(
        full_activation,
        down,
        down_scale,
    )
    actual = torch.zeros_like(expected)

    for rank in range(4):
        plan = Fp8ExpertTpShard(intermediate_size, rank, 4)
        local_gate, local_gate_scale = plan.prepare_projection(
            gate,
            gate_scale,
            hidden_size=hidden_size,
            down_projection=False,
        )
        local_up, local_up_scale = plan.prepare_projection(
            up,
            up_scale,
            hidden_size=hidden_size,
            down_projection=False,
        )
        local_down, local_down_scale = plan.prepare_projection(
            down,
            down_scale,
            hidden_size=hidden_size,
            down_projection=True,
        )
        local_gate_output = fp8_blockwise_linear_reference(
            inputs,
            local_gate,
            local_gate_scale,
        )
        local_up_output = fp8_blockwise_linear_reference(
            inputs,
            local_up,
            local_up_scale,
        )
        local_activation = (
            torch.nn.functional.silu(local_gate_output) * local_up_output
        ).to(torch.bfloat16)
        assert torch.equal(
            local_activation,
            full_activation[:, plan.aligned_start : plan.aligned_stop],
        )
        actual.add_(
            fp8_blockwise_linear_reference(
                local_activation,
                local_down,
                local_down_scale,
            )
        )

    normalized_max_error = float(
        (actual.float() - expected.float()).abs().max() / expected.float().abs().max()
    )
    assert normalized_max_error < 5.0e-3
