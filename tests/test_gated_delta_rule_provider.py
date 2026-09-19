import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from sparseengine.kernels.external.flashinfer.gdn import (
    _GDN_PREFILL_REQUIRED_ARGUMENTS,
    _gdn_prefill_op,
    flashinfer_chunk_gated_delta_rule,
    flashinfer_gdn_prefill_support,
)
from sparseengine.models.gdn_runtime import (
    bind_gated_delta_rule_op,
    build_gated_delta_rule_op,
)
from sparseengine.operators.gated_delta_rule import (
    GATED_DELTA_RULE_REGISTRY,
    FlashInferGatedDeltaRuleProvider,
    GatedDeltaRuleOpSpec,
    PreparedGatedDeltaRuleOp,
    TritonGatedDeltaRuleProvider,
)
from sparseengine.operators.registry import OpResolver
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


def _spec(**overrides) -> GatedDeltaRuleOpSpec:
    values = {
        "num_key_heads": 2,
        "num_value_heads": 4,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "activation_dtype": torch.bfloat16,
        "recurrent_state_dtype": torch.float32,
        "cuda_graph_decode": True,
    }
    values.update(overrides)
    return GatedDeltaRuleOpSpec(**values)


def _cuda_caps(
    compute_capability: tuple[int, int],
    **overrides,
) -> DeviceCaps:
    values = {
        "platform": PlatformEnum.CUDA,
        "device_type": "cuda",
        "device_index": 0,
        "device_name": f"test SM{compute_capability[0]}{compute_capability[1]}",
        "compute_capability": compute_capability,
        "runtime_version": "13.0",
        "supports_graph_capture": True,
        "supports_triton": True,
        "supports_bfloat16": True,
    }
    values.update(overrides)
    return DeviceCaps(**values)


def test_gdn_spec_rejects_nonintegral_gva_head_relationship():
    with pytest.raises(ValueError, match="divisible by key heads"):
        _spec(num_key_heads=16, num_value_heads=17)


@pytest.mark.parametrize(
    "compute_capability",
    [(9, 0), (10, 0), (10, 3), (12, 0), (12, 1)],
)
def test_flashinfer_gdn_is_upstream_default_on_declared_architectures(
    compute_capability,
):
    with patch(
        "sparseengine.operators.gated_delta_rule.flashinfer_gdn_prefill_support",
        return_value=(True, "FlashInfer GDN available"),
    ) as support:
        resolved = OpResolver(GATED_DELTA_RULE_REGISTRY).resolve(
            _spec(),
            _cuda_caps(compute_capability),
        )

    assert resolved.provider.name == "flashinfer_gdn_prefill_triton_decode"
    assert resolved.report.selection_basis == "upstream_default"
    support.assert_called_once_with(compute_capability)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_sm120_qwen35_gva_topologies_select_flashinfer(tp_size):
    with patch(
        "sparseengine.operators.gated_delta_rule.flashinfer_gdn_prefill_support",
        return_value=(True, "FlashInfer GDN available"),
    ):
        resolved = OpResolver(GATED_DELTA_RULE_REGISTRY).resolve(
            _spec(
                num_key_heads=16 // tp_size,
                num_value_heads=48 // tp_size,
                recurrent_state_dtype=torch.bfloat16,
            ),
            _cuda_caps((12, 0)),
        )

    assert resolved.provider.name == "flashinfer_gdn_prefill_triton_decode"
    assert resolved.report.selection_basis == "upstream_default"


@pytest.mark.parametrize(
    ("spec", "caps", "reason"),
    [
        (_spec(key_head_dim=64, value_head_dim=64), _cuda_caps((12, 0)), "dim 128"),
        (_spec(), _cuda_caps((8, 9)), "SM90, SM100, SM103, SM120, or SM121"),
    ],
)
def test_flashinfer_gdn_rejects_contracts_outside_declared_support(
    spec,
    caps,
    reason,
):
    with patch(
        "sparseengine.operators.gated_delta_rule.flashinfer_gdn_prefill_support",
        return_value=(True, "FlashInfer GDN available"),
    ):
        resolved = OpResolver(GATED_DELTA_RULE_REGISTRY).resolve(spec, caps)

    assert resolved.provider.name == "triton_gated_delta_rule"
    assert reason in dict(resolved.rejected)["flashinfer_gdn_prefill_triton_decode"]


def test_flashinfer_sm100_gdn_requires_cuda_13():
    result = FlashInferGatedDeltaRuleProvider.supports(
        _spec(),
        _cuda_caps((10, 0), runtime_version="12.9"),
    )

    assert not result.supported
    assert "CUDA runtime >= 13.0" in result.reason


def test_flashinfer_gdn_project_minimum_accepts_additive_optional_parameter():
    parameter_names = (*_GDN_PREFILL_REQUIRED_ARGUMENTS, "_cp_chunk_len")
    public_function = Mock()
    public_function.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in parameter_names
        ]
    )
    module = SimpleNamespace(
        chunk_gated_delta_rule=public_function,
        chunk_gated_delta_rule_sm100=Mock(),
    )
    _gdn_prefill_op.cache_clear()
    try:
        with (
            patch(
                "sparseengine.kernels.external.flashinfer.gdn.flashinfer_kernel_support",
                return_value=(
                    True,
                    "flashinfer-python 0.6.15 GDN prefill is available",
                ),
            ),
            patch(
                "sparseengine.kernels.external.flashinfer.gdn.importlib.import_module",
                return_value=module,
            ),
        ):
            supported, reason = flashinfer_gdn_prefill_support((10, 0))
            function, _ = _gdn_prefill_op((10, 0))
    finally:
        _gdn_prefill_op.cache_clear()

    assert function is public_function
    assert supported
    assert "0.6.15" in reason


def test_flashinfer_gdn_rejects_unsupported_activation_dtype():
    result = FlashInferGatedDeltaRuleProvider.supports(
        _spec(activation_dtype=torch.float32),
        _cuda_caps((12, 0)),
    )

    assert not result.supported
    assert "BF16 or FP16" in result.reason


def test_flashinfer_prefill_adapter_converts_log_gate_and_state_contract():
    provider = FlashInferGatedDeltaRuleProvider()
    packed_qk = torch.randn(1, 3, 2, 256, dtype=torch.bfloat16)
    q = packed_qk[..., :128]
    k = packed_qk[..., 128:]
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    v = torch.randn(1, 3, 4, 128, dtype=torch.bfloat16)
    g = torch.randn(1, 3, 4, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn_like(g))
    state = torch.randn(2, 4, 128, 128, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32)
    output = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    final_state = torch.randn(2, 4, 128, 128, dtype=torch.float32)

    with (
        patch(
            "sparseengine.operators.gated_delta_rule.l2norm_fwd",
            side_effect=lambda tensor: tensor,
        ) as normalize,
        patch(
            "sparseengine.operators.gated_delta_rule.flashinfer_chunk_gated_delta_rule",
            return_value=(output, final_state),
        ) as kernel,
    ):
        actual_output, actual_state = provider.run_prefill(
            _spec(recurrent_state_dtype=torch.bfloat16),
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            cu_seqlens=cu_seqlens,
        )

    assert actual_output.shape == (1, 3, 4, 128)
    assert actual_state.dtype == torch.float32
    torch.testing.assert_close(actual_state, final_state.transpose(-1, -2))
    assert actual_state.is_contiguous()
    assert all(call.args[0].is_contiguous() for call in normalize.call_args_list)
    call = kernel.call_args
    torch.testing.assert_close(call.args[3], torch.exp(g.squeeze(0)))
    assert call.args[5].dtype == torch.float32
    torch.testing.assert_close(
        call.args[5],
        state.to(torch.float32).transpose(-1, -2),
    )
    assert call.args[5].is_contiguous()
    assert call.args[6] is cu_seqlens


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_flashinfer_adapter_normalizes_sequence_offsets(dtype, strided):
    offsets = torch.tensor([0, 1, 3], dtype=dtype)
    if strided:
        offsets = torch.stack((offsets, offsets), dim=1)[:, 0]
    q = torch.zeros(3, 2, 128, dtype=torch.bfloat16)
    state = torch.zeros(2, 2, 128, 128, dtype=torch.float32)
    gates = torch.ones(3, 2)
    kernel = Mock(return_value=(q, state))
    with (
        patch("sparseengine.kernels.external.flashinfer.gdn._gdn_prefill_op",
              return_value=(kernel, "available")),
        patch("sparseengine.kernels.external.flashinfer.gdn.torch.cuda.get_device_capability",
              return_value=(9, 0)),
    ):
        flashinfer_chunk_gated_delta_rule(q, q, q, gates, gates, state, offsets)
    passed = kernel.call_args.kwargs["cu_seqlens"]
    assert passed.dtype == torch.int64
    assert passed.is_contiguous()
    assert passed.device == offsets.device
    assert passed.tolist() == offsets.tolist()
    assert offsets.dtype == dtype
    if dtype == torch.int64 and not strided:
        assert passed is offsets
    assert kernel.call_args.kwargs["use_cp"] == "auto"


def test_flashinfer_prefill_adapter_rejects_non_fp32_final_state():
    q = torch.randn(1, 2, 128, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    state = torch.randn(1, 4, 128, 128, dtype=torch.float32)
    gate = torch.randn(1, 4, dtype=torch.float32)
    kernel = Mock(
        return_value=(v, torch.empty_like(state, dtype=torch.bfloat16))
    )

    with (
        patch(
            "sparseengine.kernels.external.flashinfer.gdn._gdn_prefill_op",
            return_value=(kernel, "available"),
        ),
        patch(
            "sparseengine.kernels.external.flashinfer.gdn."
            "torch.cuda.get_device_capability",
            return_value=(12, 0),
        ),
        pytest.raises(RuntimeError, match="FP32 final_state"),
    ):
        flashinfer_chunk_gated_delta_rule(
            q,
            q,
            v,
            gate,
            gate,
            state,
            torch.tensor([0, 1], dtype=torch.int32),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()
    not in {(9, 0), (10, 0), (10, 3), (12, 0), (12, 1)},
    reason="requires CUDA SM90, SM100, SM103, SM120, or SM121",
)
@pytest.mark.parametrize("use_cp", [False, True, "auto"])
@pytest.mark.parametrize("token_count", [48, 1057])
def test_flashinfer_prefill_matches_independent_triton_provider(use_cp, token_count):
    supported, reason = flashinfer_gdn_prefill_support(
        torch.cuda.get_device_capability()
    )
    if not supported:
        pytest.skip(reason)

    torch.manual_seed(20260824)
    num_key_heads, num_value_heads, head_dim = 2, 4, 128
    q = torch.randn(
        1,
        token_count,
        num_key_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn(
        1,
        token_count,
        num_value_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.2
    g = -(
        torch.rand(
            1,
            token_count,
            num_value_heads,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.2
        + 0.01
    )
    beta = torch.sigmoid(torch.randn_like(g))
    initial_state = torch.randn(
        2,
        num_value_heads,
        head_dim,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    ) * 0.01
    cu_seqlens = torch.tensor(
        [0, 17, token_count], device="cuda", dtype=torch.int32
    )
    spec = _spec(
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=head_dim,
        value_head_dim=head_dim,
    )

    # Exercise both real upstream kernels, independent of the auto heuristic.
    with patch(
        "sparseengine.operators.gated_delta_rule.flashinfer_chunk_gated_delta_rule",
        side_effect=lambda *args, **kwargs: flashinfer_chunk_gated_delta_rule(
            *args, **kwargs, use_cp=use_cp),
    ):
        actual_output, actual_state = (
            FlashInferGatedDeltaRuleProvider().run_prefill(
                spec,
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state.clone(),
                cu_seqlens=cu_seqlens,
            )
        )
    expected_output, expected_state = TritonGatedDeltaRuleProvider().run_prefill(
        spec,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), rtol=0.05, atol=2e-4
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), rtol=0.02, atol=2e-3
    )


def test_bound_decode_calls_fused_raw_gating_kernel_without_dispatch():
    provider = TritonGatedDeltaRuleProvider()
    q = torch.randn(4, 1, 2, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(4, 1, 4, 128, dtype=torch.bfloat16)
    state = torch.randn(9, 4, 128, 128, dtype=torch.float32)
    state_indices = torch.tensor([1, 3, 5, 8], dtype=torch.int32)
    A_log = torch.randn(4, dtype=torch.float32)
    dt_bias = torch.randn(4, dtype=torch.float32)
    a = torch.randn(4, 4, dtype=torch.bfloat16)
    b = torch.randn(4, 4, dtype=torch.bfloat16)
    output = torch.randn(4, 1, 4, 128, dtype=torch.bfloat16)

    with patch(
        "sparseengine.operators.gated_delta_rule.fused_recurrent_gated_delta_rule",
        return_value=(output, state),
    ) as kernel:
        actual = provider.run_decode(
            _spec(),
            q=q,
            k=k,
            v=v,
            initial_state=state,
            state_indices=state_indices,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
        )

    assert actual is output
    assert kernel.call_args.kwargs == {
        "q": q,
        "k": k,
        "v": v,
        "initial_state": state,
        "inplace_final_state": True,
        "ssm_state_indices": state_indices,
        "use_qk_l2norm_in_kernel": True,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "a_raw": a,
        "b_raw": b,
    }


def test_triton_prefill_provider_expands_qk_to_value_heads():
    provider = TritonGatedDeltaRuleProvider()
    q = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
    k = q + 100
    v = torch.randn(1, 4, 6, 3)
    g = torch.randn(1, 4, 6)
    beta = torch.randn(1, 4, 6)
    state = torch.randn(1, 6, 3, 3)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    output = torch.randn_like(v)

    with patch(
        "sparseengine.operators.gated_delta_rule.chunk_gated_delta_rule",
        return_value=(output, state),
    ) as kernel:
        actual = provider.run_prefill(
            _spec(
                num_key_heads=2,
                num_value_heads=6,
                key_head_dim=3,
                value_head_dim=3,
            ),
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            cu_seqlens=cu_seqlens,
        )

    assert actual[0] is output
    assert actual[1] is state
    repeated_q = kernel.call_args.kwargs["q"]
    repeated_k = kernel.call_args.kwargs["k"]
    assert repeated_q.shape == (1, 4, 6, 3)
    assert repeated_k.shape == (1, 4, 6, 3)
    assert torch.equal(repeated_q[:, :, 0], q[:, :, 0])
    assert torch.equal(repeated_q[:, :, 1], q[:, :, 0])
    assert torch.equal(repeated_q[:, :, 2], q[:, :, 0])
    assert torch.equal(repeated_q[:, :, 3], q[:, :, 1])
    assert torch.equal(repeated_k[:, :, 5], k[:, :, 1])


def test_prepared_gdn_operator_rejects_calls_after_close():
    provider = Mock(name="provider")
    provider.name = "test"
    prepared = PreparedGatedDeltaRuleOp(_spec(), provider)

    prepared.close()

    provider.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.run_decode()


def test_prepared_gdn_operator_forwards_auxiliary_pipeline_without_dispatch():
    provider = Mock(name="provider")
    provider.name = "test"
    provider.run_gating.return_value = ("g", "beta")
    provider.run_prefill_conv.return_value = "conv"
    provider.prepare_decode_inputs.return_value = ("q", "k", "v", "z", "a", "b")
    provider.run_gated_rmsnorm.return_value = "norm"
    prepared = PreparedGatedDeltaRuleOp(_spec(), provider)

    assert prepared.run_gating(A_log=1, a=2, b=3, dt_bias=4) == ("g", "beta")
    assert prepared.run_prefill_conv(mixed_qkv=1) == "conv"
    assert prepared.prepare_decode_inputs(mixed_qkv=1) == (
        "q", "k", "v", "z", "a", "b"
    )
    assert prepared.run_gated_rmsnorm(x=1) == "norm"
    provider.run_gating.assert_called_once_with(A_log=1, a=2, b=3, dt_bias=4)


def test_model_runtime_builds_and_binds_one_shared_gdn_operator():
    config = SimpleNamespace(
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        dtype=torch.bfloat16,
        quantization_config=None,
    )
    prepared = Mock(name="prepared_gdn")
    root = nn.Module()
    root.first = nn.Module()
    root.first.is_gated_delta_rule_layer = True
    root.second = nn.Module()
    root.second.is_gated_delta_rule_layer = True
    root.second.bind_gated_delta_rule_op = Mock()

    with patch(
        "sparseengine.models.gdn_runtime.prepare_gated_delta_rule_op",
        return_value=prepared,
    ) as prepare:
        actual = build_gated_delta_rule_op(
            config,
            attention_tp_size=2,
            device=torch.device("cuda", 3),
            cuda_graph=True,
        )

    assert actual is prepared
    spec = prepare.call_args.args[0]
    assert (spec.num_key_heads, spec.num_value_heads) == (2, 4)
    assert spec.recurrent_state_dtype == torch.bfloat16
    assert prepare.call_args.kwargs == {"device_index": 3}
    assert bind_gated_delta_rule_op(root, prepared) == 2
    assert root.first.gated_delta_rule_op is prepared
    root.second.bind_gated_delta_rule_op.assert_called_once_with(prepared)
