from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.kernels.external.support import (
    ExternalKernelFamilyError,
    KernelFamilyHealth,
    KernelFamilyState,
    RequiredExternalKernelFamilyError,
)
from sparseengine.operators.activation import (
    SILU_AND_MUL_REGISTRY,
    SiluAndMulSpec,
    TorchSiluAndMulProvider,
    TritonSiluAndMulProvider,
)
from sparseengine.operators.fp8_linear import (
    FP8_LINEAR_REGISTRY,
    FlashInferGroupwiseSm120Fp8LinearProvider,
    FlashInferSm90Fp8LinearProvider,
    Fp8LinearSpec,
)
from sparseengine.operators.gate_up_swiglu import (
    GateUpSwiGLUOpSpec,
    NativeGateUpSwiGLUProvider,
)
from sparseengine.operators.moe import (
    FlashInferCutlassFp8MoeProvider,
    MOE_REGISTRY,
    MoeOpSpec,
    Qwen3Fp8MoeDispatchPlan,
    SglDerivedTritonMoeProvider,
    SglTritonGlmMoeProvider,
    resolve_moe_provider,
)
from sparseengine.operators.quest_selection import (
    H100ExactQuestPagedViewDispatch,
    QUEST_PAGE_SELECTION_REGISTRY,
    FlashInferQuestPageSelectionProvider,
    QuestPageSelectionOpSpec,
    TritonExactQuestPageSelectionProvider,
)
from sparseengine.operators.registry import NoProviderError, OpResolver
from sparseengine.platforms import DeviceCaps, PlatformEnum
from sparseengine.quantization.config import QuantizationConfig
from sparseengine.quantization.registry import QuantizationRegistry


@pytest.fixture(autouse=True)
def _mock_sgl_fp8_quantization_contract():
    with (
        patch(
            "sparseengine.kernels.external.sgl.moe.sgl_fp8_group_quantization_support",
            return_value=(True, "sgl quant available"),
        ),
        patch(
            "sparseengine.kernels.external.flashinfer.fp8_linear.flashinfer_sm90_fp8_linear_support",
            return_value=(True, "flashinfer sm90 available"),
        ),
        patch(
            "sparseengine.kernels.external.flashinfer.fp8_linear.flashinfer_sm120_groupwise_fp8_linear_support",
            return_value=(True, "flashinfer sm120 available"),
        ),
        patch(
            "sparseengine.kernels.external.flashinfer.moe.flashinfer_cutlass_fp8_moe_support",
            return_value=(True, "flashinfer moe available"),
        ),
        patch(
            "sparseengine.kernels.external.flashinfer.support.flashinfer_kernel_health",
            return_value=KernelFamilyHealth(
                family="flashinfer-python",
                state=KernelFamilyState.READY,
                version="0.6.15.post1",
                reason="ready",
            ),
        ),
        patch(
            "sparseengine.kernels.external.sgl.support.sgl_kernel_health",
            return_value=KernelFamilyHealth(
                family="sglang-kernel",
                state=KernelFamilyState.READY,
                version="0.4.5",
                reason="ready",
            ),
        ),
        patch("torch.__version__", "2.11.0"),
        patch("triton.__version__", "3.6.0"),
    ):
        yield


def _broken_sgl_family(feature: str) -> ExternalKernelFamilyError:
    return ExternalKernelFamilyError(
        KernelFamilyHealth(
            family="sglang-kernel",
            state=KernelFamilyState.BROKEN,
            version="0.4.5",
            reason="binary failed to load: undefined symbol",
        ),
        feature=feature,
    )


def _missing_sgl_family(feature: str) -> RequiredExternalKernelFamilyError:
    return RequiredExternalKernelFamilyError(
        KernelFamilyHealth(
            family="sglang-kernel",
            state=KernelFamilyState.ABSENT,
            version=None,
            reason="sglang-kernel is not installed",
        ),
        feature=feature,
    )


def _broken_flashinfer_family(feature: str) -> ExternalKernelFamilyError:
    return ExternalKernelFamilyError(
        KernelFamilyHealth(
            family="flashinfer-python",
            state=KernelFamilyState.BROKEN,
            version="0.6.15.post1",
            reason="package failed to load: undefined symbol",
        ),
        feature=feature,
    )


def _missing_flashinfer_family(feature: str) -> RequiredExternalKernelFamilyError:
    return RequiredExternalKernelFamilyError(
        KernelFamilyHealth(
            family="flashinfer-python",
            state=KernelFamilyState.ABSENT,
            version=None,
            reason="flashinfer-python is not installed",
        ),
        feature=feature,
    )


def _cuda_caps(
    capability: tuple[int, int],
    *,
    native_fp8: bool = True,
    runtime_version: str | None = "13.0",
    device_name: str | None = None,
) -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name=device_name or f"SM{capability[0]}{capability[1]}",
        compute_capability=capability,
        runtime_version=runtime_version,
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
        supports_native_fp8=native_fp8,
    )


def _non_cuda_caps(platform: PlatformEnum) -> DeviceCaps:
    return DeviceCaps(
        platform=platform,
        device_type="cuda" if platform == PlatformEnum.ROCM else "cpu",
        device_index=0,
        device_name=platform.name,
        supports_triton=platform == PlatformEnum.ROCM,
        supports_bfloat16=True,
        supports_native_fp8=platform == PlatformEnum.ROCM,
    )


@pytest.mark.parametrize(
    "device_name",
    ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"],
)
def test_quest_fused_paged_view_is_the_default_h100_profile(
    device_name: str,
) -> None:
    spec = QuestPageSelectionOpSpec(
        score_dtype=torch.bfloat16,
        cuda_graph=True,
    )
    with patch(
        "sparseengine.operators.quest_selection."
        "flashinfer_top_k_page_table_transform_support",
        return_value=(True, "available"),
    ):
        resolved = OpResolver(QUEST_PAGE_SELECTION_REGISTRY).resolve(
            spec,
            _cuda_caps((9, 0), device_name=device_name),
            op_spec=spec,
        )

    assert isinstance(resolved.provider, H100ExactQuestPagedViewDispatch)
    assert resolved.report.selection_basis == "profile_override"


@pytest.mark.parametrize(
    ("graph_mode", "width", "expected_route"),
    [(True, 256, "fused"), (True, 2048, "flashinfer"),
     (False, 2048, "fused")],
)
def test_quest_profile_routes_dense_fallback_by_execution_mode(
    graph_mode: bool, width: int, expected_route: str,
) -> None:
    spec = QuestPageSelectionOpSpec(score_dtype=torch.bfloat16, cuda_graph=graph_mode)
    provider = H100ExactQuestPagedViewDispatch(op_spec=spec)
    outputs = tuple(object() for _ in range(5))
    with (
        patch.object(provider.fused, "select_and_finalize_paged_view", return_value=outputs) as fused,
        patch.object(provider.fallback, "select_and_finalize_paged_view", return_value=outputs) as fallback,
        patch("sparseengine.operators.quest_selection.device_runtime.is_stream_capturing", return_value=False),
    ):
        result = provider.select_and_finalize_paged_view(
            torch.empty(1, width, dtype=torch.bfloat16),
            torch.empty(1, width, dtype=torch.int32),
            torch.tensor([width - 1], dtype=torch.int32),
            torch.tensor([width], dtype=torch.int32),
            torch.tensor([width * 16], dtype=torch.int32),
            k=1, page_size=16, token_budget=16,
            outputs=outputs, use_dense_fallback=True,
        )
    assert result is outputs
    assert fused.call_count == (expected_route == "fused")
    assert fallback.call_count == (expected_route == "flashinfer")

def test_quest_triton_exact_atomic_capability_is_portable_cuda() -> None:
    spec = QuestPageSelectionOpSpec(
        score_dtype=torch.bfloat16,
        cuda_graph=True,
    )

    assert TritonExactQuestPageSelectionProvider.supports(
        spec,
        _cuda_caps((8, 0), device_name="NVIDIA A100-SXM4-80GB"),
    ).supported
    assert not TritonExactQuestPageSelectionProvider.supports(
        spec,
        _non_cuda_caps(PlatformEnum.ROCM),
    ).supported


def _moe_spec(
    *,
    activation_dtype=torch.bfloat16,
    weight_dtype=torch.float8_e4m3fn,
    block_shape=(128, 128),
    hidden_size=256,
    intermediate_size=128,
    num_local_experts=4,
    num_experts=8,
    top_k=2,
    ep_size=2,
    tp_size=1,
    routing_method="softmax",
    scale_dtype=None,
    cuda_graph=True,
    activation="silu",
    max_num_tokens=1,
) -> MoeOpSpec:
    return MoeOpSpec(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        activation_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        block_shape=block_shape,
        ep_size=ep_size,
        cuda_graph=cuda_graph,
        tp_size=tp_size,
        routing_method=routing_method,
        scale_dtype=scale_dtype,
        activation=activation,
        max_num_tokens=max_num_tokens,
    )


def _linear_spec(
    *,
    input_features=256,
    output_features=128,
    activation_dtype=torch.bfloat16,
    block_shape=(128, 128),
) -> Fp8LinearSpec:
    return Fp8LinearSpec(
        block_shape=block_shape,
        input_features=input_features,
        output_features=output_features,
        activation_dtype=activation_dtype,
    )


def _gate_up_spec(**overrides) -> GateUpSwiGLUOpSpec:
    values = {
        "hidden_size": 2048,
        "intermediate_size": 512,
        "tp_size": 1,
        "activation_dtype": torch.bfloat16,
        "weight_dtype": torch.bfloat16,
        "cuda_graph": True,
    }
    values.update(overrides)
    return GateUpSwiGLUOpSpec(**values)


def test_native_gate_up_provider_matches_swiglu_semantics():
    torch.manual_seed(0)
    inputs = torch.randn(3, 8)
    projection = torch.nn.Linear(8, 12, bias=False)
    with torch.inference_mode():
        projected = projection(inputs)
        gate, up = projected.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * up
        actual = NativeGateUpSwiGLUProvider().run(
            _gate_up_spec(
                hidden_size=8,
                intermediate_size=6,
                activation_dtype=torch.float32,
                weight_dtype=torch.float32,
                cuda_graph=False,
            ),
            inputs,
            projection,
        )

    torch.testing.assert_close(actual, expected)


def test_silu_and_mul_provider_respects_platform_capability():
    cuda_provider = OpResolver(SILU_AND_MUL_REGISTRY).resolve(
        SiluAndMulSpec(activation_dtype=torch.bfloat16),
        _cuda_caps((9, 0)),
        op_spec=SiluAndMulSpec(activation_dtype=torch.bfloat16),
    ).provider
    cpu_provider = OpResolver(SILU_AND_MUL_REGISTRY).resolve(
        SiluAndMulSpec(activation_dtype=torch.float32),
        _non_cuda_caps(PlatformEnum.CPU),
        op_spec=SiluAndMulSpec(activation_dtype=torch.float32),
    ).provider

    assert isinstance(cuda_provider, TritonSiluAndMulProvider)
    assert isinstance(cpu_provider, TorchSiluAndMulProvider)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_experts": 0},
        {"num_local_experts": 3},
        {"ep_size": 0},
        {"tp_size": 0},
        {"hidden_size": 0},
        {"intermediate_size": -1},
        {"top_k": 0},
        {"top_k": 9},
        {"block_shape": (128, 0)},
        {"routing_method": "unknown"},
    ],
)
def test_moe_spec_rejects_inconsistent_semantics(overrides):
    values = {
        "num_experts": 8,
        "num_local_experts": 4,
        "hidden_size": 256,
        "intermediate_size": 128,
        "top_k": 2,
        "activation_dtype": torch.bfloat16,
        "weight_dtype": torch.float8_e4m3fn,
        "block_shape": (128, 128),
        "ep_size": 2,
        "cuda_graph": True,
        "tp_size": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        MoeOpSpec(**values)


def test_generic_swiglu_moe_registry_rejects_gelu_tanh_contract():
    with pytest.raises(NoProviderError, match="requires SiLU activation"):
        OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(activation="gelu_tanh"),
            _cuda_caps((9, 0)),
        )


@pytest.mark.parametrize(
    ("activation_dtype_name", "activation_dtype"),
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ],
)
def test_quantization_registry_preserves_model_activation_dtype(
    activation_dtype_name,
    activation_dtype,
):
    quantization = QuantizationConfig(
        enabled=True,
        quant_method="fp8",
        weight_dtype="e4m3",
        activation_scheme="dynamic",
        weight_block_size=(128, 128),
        model_name="Llama",
        activation_dtype=activation_dtype_name,
    )
    caps = _cuda_caps((9, 0))
    platform = SimpleNamespace(
        is_cuda_alike=lambda: False,
        get_device_caps=lambda _device_index: caps,
    )

    with patch(
        "sparseengine.operators.fp8_linear.platforms",
        SimpleNamespace(current_platform=platform),
    ):
        provider = QuantizationRegistry.resolve_linear_provider(
            quantization,
            input_features=128,
            output_features=128,
        )

    assert provider.spec.activation_dtype == activation_dtype


def test_quantization_config_normalizes_model_activation_dtype():
    quantization = QuantizationConfig.from_hf_config(
        {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
        model_name="Qwen2",
        activation_dtype="torch.float16",
    )

    assert quantization.activation_dtype == "float16"


def test_fp8_linear_does_not_hide_broken_flashinfer_family_on_sm90():
    error = _broken_flashinfer_family("SM90 block-scale FP8 Linear")
    with (
        patch(
            "sparseengine.kernels.external.flashinfer.fp8_linear.flashinfer_sm90_fp8_linear_support",
            side_effect=error,
        ),
        pytest.raises(ExternalKernelFamilyError, match="undefined symbol"),
    ):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            _cuda_caps((9, 0)),
        )


def test_flashinfer_linear_does_not_mask_missing_jit_artifact():
    flashinfer_call = Mock(
        side_effect=RuntimeError(
            "Assertion failed: !cubin.empty() || isPathValid(path_)"
        )
    )
    provider = FlashInferSm90Fp8LinearProvider()
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scale = torch.ones(1, 1)

    with (
        patch(
            "sparseengine.kernels.external.flashinfer.fp8_linear.flashinfer_fp8_blockscale_gemm_sm90",
            flashinfer_call,
        ),
        pytest.raises(RuntimeError, match="cubin.empty"),
    ):
        provider(x, weight, scale)

    assert flashinfer_call.call_count == 1


def test_flashinfer_linear_does_not_mask_other_runtime_failures():
    flashinfer_call = Mock(side_effect=RuntimeError("invalid scale layout"))
    provider = FlashInferSm90Fp8LinearProvider()
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scale = torch.ones(1, 1)

    with (
        patch(
            "sparseengine.kernels.external.flashinfer.fp8_linear.flashinfer_fp8_blockscale_gemm_sm90",
            flashinfer_call,
        ),
        pytest.raises(RuntimeError, match="invalid scale layout"),
    ):
        provider(x, weight, scale)


def test_flashinfer_sm120_linear_composes_quantizer_and_gemm():
    quantize_call = Mock()
    gemm_call = Mock(return_value=torch.full((2, 128), 3.0, dtype=torch.bfloat16))
    provider = FlashInferGroupwiseSm120Fp8LinearProvider()
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scale = torch.ones(1, 1)

    with (
        patch(
            "sparseengine.kernels.external.sgl.moe.sgl_per_token_group_quant_8bit",
            quantize_call,
        ),
        patch(
            "sparseengine.kernels.external.flashinfer.fp8_linear.flashinfer_fp8_nt_groupwise_sm120",
            gemm_call,
        ),
    ):
        output = provider(x, weight, scale)

    assert quantize_call.call_count == 1
    assert gemm_call.call_count == 1
    assert torch.equal(output, torch.full_like(output, 3.0))


def test_fp8_linear_reports_unsupported_pre_fp8_device():
    with pytest.raises(RuntimeError, match="native FP8 tensor cores"):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            _cuda_caps((8, 0), native_fp8=False),
        )


@pytest.mark.parametrize(
    "spec",
    [
        _linear_spec(block_shape=(64, 128)),
        _linear_spec(activation_dtype=torch.float32),
    ],
)
def test_fp8_linear_rejects_unsupported_operation_specs(spec):
    with pytest.raises(RuntimeError, match="No block-scaled FP8 Linear provider"):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(spec, _cuda_caps((9, 0)))


@pytest.mark.parametrize("platform", [PlatformEnum.CPU, PlatformEnum.ROCM])
def test_fp8_linear_rejects_non_cuda_platforms(platform):
    with pytest.raises(RuntimeError, match="requires CUDA"):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            _non_cuda_caps(platform),
        )


def test_fp8_linear_rejects_cuda_without_triton_or_flashinfer_feature():
    caps = _cuda_caps((9, 0))
    caps = DeviceCaps(**{**caps.__dict__, "supports_triton": False})
    with (
        patch(
            "sparseengine.kernels.external.flashinfer.fp8_linear.flashinfer_sm90_fp8_linear_support",
            return_value=(False, "SM90 feature unavailable"),
        ),
        pytest.raises(RuntimeError, match="does not support Triton"),
    ):
        OpResolver(FP8_LINEAR_REGISTRY).resolve(
            _linear_spec(),
            caps,
        )


def test_sgl_triton_moe_does_not_hide_broken_alignment() -> None:
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=384,
        num_local_experts=128,
        num_experts=128,
        top_k=8,
        ep_size=1,
        tp_size=2,
    )

    with patch(
        "sparseengine.kernels.external.sgl.moe.sgl_moe_alignment_support",
        side_effect=_broken_sgl_family("MoE alignment"),
    ):
        with pytest.raises(ExternalKernelFamilyError, match="undefined symbol"):
            OpResolver(MOE_REGISTRY).resolve(
                spec,
                _cuda_caps(
                    (9, 0),
                    native_fp8=False,
                    device_name="NVIDIA H100 80GB HBM3",
                ),
            )


def test_sgl_triton_moe_fails_when_required_alignment_is_missing() -> None:
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=384,
        num_local_experts=128,
        num_experts=128,
        top_k=8,
        ep_size=1,
        tp_size=2,
    )

    with (
        patch(
            "sparseengine.kernels.external.sgl.moe.sgl_moe_alignment_support",
            side_effect=_missing_sgl_family("MoE alignment"),
        ),
        pytest.raises(
            RequiredExternalKernelFamilyError,
            match=r'pip install -e "\.\[cu129\]"',
        ),
    ):
        OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps(
                (9, 0),
                native_fp8=False,
                device_name="NVIDIA H100 80GB HBM3",
            ),
        )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"activation_dtype": torch.float32}, "requires BF16 activations"),
        ({"block_shape": (128, 128)}, "unquantized expert weights"),
    ],
)
def test_sgl_triton_moe_rejects_unsupported_specs(overrides, reason):
    values = dict(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=384,
        num_local_experts=128,
        num_experts=128,
        top_k=8,
        ep_size=1,
        tp_size=2,
    )
    values.update(overrides)
    if "activation_dtype" in overrides and "weight_dtype" not in overrides:
        values["weight_dtype"] = overrides["activation_dtype"]
    support = SglDerivedTritonMoeProvider.supports(
        _moe_spec(**values),
        _cuda_caps(
            (9, 0),
            native_fp8=False,
            device_name="NVIDIA H100 80GB HBM3",
        ),
    )

    assert not support.supported
    assert reason in support.reason


@pytest.mark.parametrize(
    "device_name",
    ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"],
)
def test_glm_tp1_fused_shared_profile_selects_sgl_triton(device_name) -> None:
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=1536,
        num_local_experts=65,
        num_experts=65,
        top_k=5,
        ep_size=1,
        tp_size=1,
        routing_method="biased_sigmoid",
    )

    resolved = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps(
            (9, 0),
            native_fp8=False,
            device_name=device_name,
        ),
    )

    assert isinstance(resolved.provider, SglTritonGlmMoeProvider)
    assert resolved.report.selected_profile == "sgl_triton_glm_tp1_bf16_profile"
    assert resolved.report.selection_basis == "profile_override"


def test_minimax_m2_fused_moe_rejects_graph_on_unsupported_device():
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=384,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=4,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )
    caps = _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3")
    caps = DeviceCaps(**{**caps.__dict__, "supports_graph_capture": False})

    resolved = OpResolver(MOE_REGISTRY).resolve(spec, caps)

    assert (
        "triton_minimax_m2_fused",
        "device does not support CUDA Graph capture",
    ) in resolved.rejected


def test_minimax_m2_fused_moe_requires_native_fp8():
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=384,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=4,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="native FP8 tensor cores"):
        OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps(
                (9, 0),
                native_fp8=False,
                device_name="NVIDIA H100 80GB HBM3",
            ),
        )


def test_minimax_m2_tp4_ep1_keeps_exact_triton_profile() -> None:
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=384,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=4,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )

    resolved = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
    )

    assert resolved.provider.name == "triton_minimax_m2_fused"
    assert resolved.report.selected_profile == "triton_minimax_m2_profile"


def test_minimax_m2_tp4_ep1_benchmark_override_forces_flashinfer() -> None:
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=384,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=4,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )

    resolved = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
        force_atomic_provider="flashinfer_cutlass_fp8_sm90",
    )

    assert resolved.provider.name == "flashinfer_cutlass_fp8_sm90"
    assert resolved.report.selected_profile is None
    assert resolved.report.selection_basis == "benchmark_override"


def test_moe_benchmark_provider_override_is_wired() -> None:
    spec = _moe_spec()
    caps = _cuda_caps((9, 0))

    with patch(
        "sparseengine.operators.moe.platforms.current_platform.get_device_caps",
        return_value=caps,
    ):
        provider = resolve_moe_provider(
            spec,
            device_index=0,
            force_atomic_provider="flashinfer_cutlass_fp8_sm90",
        )

    assert provider.name == "flashinfer_cutlass_fp8_sm90"


@pytest.mark.parametrize(
    ("tp_size", "ep_size", "num_local_experts", "intermediate_size"),
    [(2, 2, 128, 768), (1, 4, 64, 1536), (1, 8, 32, 1536)],
)
def test_minimax_m2_mixed_tp_ep_uses_flashinfer_upstream_default(
    tp_size: int,
    ep_size: int,
    num_local_experts: int,
    intermediate_size: int,
) -> None:
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts=256,
        top_k=8,
        ep_size=ep_size,
        tp_size=tp_size,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )

    resolved = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
    )

    assert resolved.provider.name == "flashinfer_cutlass_fp8_sm90"
    assert resolved.report.selected_profile is None
    assert resolved.report.selection_basis == "upstream_default"


@pytest.mark.parametrize(
    ("tp_size", "ep_size", "num_local_experts", "intermediate_size"),
    [(2, 2, 128, 768), (1, 4, 64, 1536), (1, 8, 32, 1536)],
)
def test_minimax_mixed_ep_fails_when_flashinfer_missing(
    tp_size: int,
    ep_size: int,
    num_local_experts: int,
    intermediate_size: int,
) -> None:
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts=256,
        top_k=8,
        ep_size=ep_size,
        tp_size=tp_size,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )

    with (
        patch(
            "sparseengine.kernels.external.flashinfer.moe."
            "flashinfer_cutlass_fp8_moe_support",
            side_effect=_missing_flashinfer_family("SM90 CUTLASS FP8 MoE"),
        ),
        pytest.raises(
            RequiredExternalKernelFamilyError,
            match=r'pip install -e "\.\[cu130\]"',
        ),
    ):
        OpResolver(MOE_REGISTRY).resolve(
            spec,
            _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
        )


def test_minimax_ep4_uses_triton_when_flashinfer_unsupported() -> None:
    spec = _moe_spec(
        hidden_size=3072,
        intermediate_size=1536,
        num_local_experts=64,
        num_experts=256,
        top_k=8,
        ep_size=4,
        tp_size=1,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )

    resolved = OpResolver(MOE_REGISTRY).resolve(
        spec,
        _cuda_caps(
            (9, 0),
            runtime_version="12.7",
            device_name="NVIDIA H100 80GB HBM3",
        ),
    )

    assert resolved.provider.name == "triton_minimax_m2_fused"
    assert resolved.report.selected_profile is None
    assert resolved.report.selection_basis == "semantic_fallback"
    assert "requires CUDA runtime >= 12.8" in dict(resolved.rejected)[
        "flashinfer_cutlass_fp8_sm90"
    ]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"routing_method": "softmax"}, "biased-sigmoid routing"),
        ({"scale_dtype": torch.bfloat16}, "FP32 expert scales"),
        ({"hidden_size": 2048}, "MiniMax M2.7 expert shape"),
    ],
)
def test_minimax_m2_fused_moe_falls_back_locally(overrides, reason):
    values = dict(
        hidden_size=3072,
        intermediate_size=384,
        num_local_experts=256,
        num_experts=256,
        top_k=8,
        ep_size=1,
        tp_size=4,
        routing_method="biased_sigmoid",
        scale_dtype=torch.float32,
    )
    values.update(overrides)
    resolved = OpResolver(MOE_REGISTRY).resolve(
        _moe_spec(**values),
        _cuda_caps((9, 0), device_name="NVIDIA H100 80GB HBM3"),
    )

    rejection = dict(resolved.rejected)["triton_minimax_m2_fused"]
    assert reason in rejection


def test_hopper_fused_moe_falls_back_for_missing_graph_support():
    spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
        hidden_size=2048,
        intermediate_size=384,
        num_local_experts=64,
        num_experts=128,
        top_k=8,
        ep_size=2,
        tp_size=2,
    )
    caps = _cuda_caps(
        (9, 0),
        native_fp8=False,
        device_name="NVIDIA H100 80GB HBM3",
    )
    caps = DeviceCaps(**{**caps.__dict__, "supports_graph_capture": False})

    resolved = OpResolver(MOE_REGISTRY).resolve(spec, caps)

    assert (
        "triton_hopper_fused",
        "device does not support CUDA Graph capture",
    ) in resolved.rejected


def test_triton_fp8_moe_does_not_hide_broken_sgl_quantization() -> None:
    with patch(
        "sparseengine.kernels.external.sgl.moe.sgl_fp8_group_quantization_support",
        side_effect=_broken_sgl_family("per-token FP8 group quantization"),
    ):
        with pytest.raises(ExternalKernelFamilyError, match="undefined symbol"):
            OpResolver(MOE_REGISTRY).resolve(_moe_spec(), _cuda_caps((12, 0)))


def test_fp8_moe_fails_when_flashinfer_is_missing():
    with (
        patch(
            "sparseengine.kernels.external.flashinfer.moe."
            "flashinfer_cutlass_fp8_moe_support",
            side_effect=_missing_flashinfer_family("SM90 CUTLASS FP8 MoE"),
        ),
        pytest.raises(RequiredExternalKernelFamilyError, match="flashinfer-python"),
    ):
        OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(),
            _cuda_caps((9, 0)),
        )


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (_moe_spec(block_shape=(64, 128)), "block_shape"),
        (_moe_spec(hidden_size=129), "128-aligned"),
        (
            _moe_spec(activation_dtype=torch.float32),
            "BF16 or FP16 activations",
        ),
    ],
)
def test_fp8_moe_rejects_unsupported_operation_specs(spec, reason):
    with pytest.raises(RuntimeError, match=reason):
        OpResolver(MOE_REGISTRY).resolve(spec, _cuda_caps((10, 0)))


def test_fp8_moe_rejects_pre_fp8_cuda():
    with pytest.raises(RuntimeError, match="native FP8 tensor cores"):
        OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(),
            _cuda_caps((8, 0), native_fp8=False),
        )


@pytest.mark.parametrize(
    ("tp_size", "ep_size", "num_local_experts"),
    [(4, 1, 8), (2, 2, 4), (1, 4, 2)],
)
def test_fp8_moe_uses_flashinfer_for_all_tp_ep_topologies(
    tp_size: int,
    ep_size: int,
    num_local_experts: int,
):
    with patch(
        "sparseengine.kernels.external.flashinfer.moe."
        "flashinfer_cutlass_fp8_moe_support",
        return_value=(True, "available"),
    ):
        resolved = OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(
                tp_size=tp_size,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
            ),
            _cuda_caps((9, 0)),
        )

    assert resolved.provider.name == "flashinfer_cutlass_fp8_sm90"
    assert resolved.report.selection_basis == "upstream_default"


def test_flashinfer_moe_prepare_reserves_shared_topology_workspace() -> None:
    provider = FlashInferCutlassFp8MoeProvider()
    spec = _moe_spec(
        tp_size=2,
        ep_size=2,
        num_local_experts=4,
        max_num_tokens=512,
    )
    manager = Mock()
    lease = Mock()
    manager.reserve_bytes.return_value = lease

    with (
        patch(
            "sparseengine.kernels.external.flashinfer.moe."
            "flashinfer_cutlass_fused_moe_workspace_size",
            return_value=8192,
        ) as workspace_size,
        patch(
            "sparseengine.operators.workspace.get_workspace_manager",
            return_value=manager,
        ),
    ):
        provider.prepare(
            spec,
            device=torch.device("cuda", 1),
            tp_rank=1,
            ep_rank=1,
        )

    assert workspace_size.call_args.kwargs["max_num_tokens"] == 512
    assert workspace_size.call_args.kwargs["tp_size"] == 2
    assert workspace_size.call_args.kwargs["tp_rank"] == 1
    assert workspace_size.call_args.kwargs["ep_size"] == 2
    assert workspace_size.call_args.kwargs["ep_rank"] == 1
    manager.reserve_bytes.assert_called_once_with(
        8192,
        label="flashinfer_cutlass_fp8_sm90",
        lane="flashinfer_cutlass_moe",
    )


def test_moe_dispatch_prepares_flashinfer_for_its_route_capacity() -> None:
    spec = _moe_spec(
        hidden_size=2048,
        intermediate_size=768,
        num_local_experts=128,
        num_experts=128,
        top_k=8,
        ep_size=1,
        tp_size=1,
        max_num_tokens=16_384,
    )
    provider = Qwen3Fp8MoeDispatchPlan(spec)
    manager = Mock()
    manager.reserve_bytes.return_value = Mock()

    with (
        patch(
            "sparseengine.kernels.external.flashinfer.moe."
            "flashinfer_cutlass_fused_moe_workspace_size",
            return_value=8192,
        ) as workspace_size,
        patch(
            "sparseengine.operators.workspace.get_workspace_manager",
            return_value=manager,
        ),
    ):
        provider.prepare(
            spec,
            device=torch.device("cuda", 0),
            tp_rank=0,
            ep_rank=0,
        )

    assert workspace_size.call_args.kwargs["max_num_tokens"] == 128


@pytest.mark.parametrize("platform", [PlatformEnum.CPU, PlatformEnum.ROCM])
def test_moe_rejects_non_cuda_platforms(platform):
    with pytest.raises(RuntimeError, match="requires CUDA"):
        OpResolver(MOE_REGISTRY).resolve(
            _moe_spec(
                activation_dtype=torch.bfloat16,
                weight_dtype=torch.bfloat16,
                block_shape=None,
            ),
            _non_cuda_caps(platform),
        )


def test_provider_owns_packed_gate_up_layout():
    triton_spec = _moe_spec(
        activation_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        block_shape=None,
    )
    triton = OpResolver(MOE_REGISTRY).resolve(
        triton_spec,
        _cuda_caps((9, 0)),
    ).provider
    flashinfer_spec = _moe_spec()
    with patch(
        "sparseengine.kernels.external.flashinfer.moe."
        "flashinfer_cutlass_fp8_moe_support",
        return_value=(True, "available"),
    ):
        flashinfer = OpResolver(MOE_REGISTRY).resolve(
            flashinfer_spec,
            _cuda_caps((9, 0)),
        ).provider

    triton_w13 = torch.zeros(4, 256, 256, dtype=torch.bfloat16)
    triton_w2 = torch.zeros(4, 256, 128, dtype=torch.bfloat16)
    gate = torch.full((128, 256), 1, dtype=torch.bfloat16)
    up = torch.full((128, 256), 2, dtype=torch.bfloat16)
    for projection, weight in (("gate", gate), ("up", up)):
        triton.load_expert_projection(
            triton_spec,
            local_expert_id=0,
            projection=projection,
            loaded_weight=weight,
            loaded_scale=None,
            w13_weight=triton_w13,
            w2_weight=triton_w2,
            w13_scale_inv=None,
            w2_scale_inv=None,
        )
    assert torch.equal(triton_w13[0, :128], gate)
    assert torch.equal(triton_w13[0, 128:], up)

    flashinfer_w13 = torch.zeros(4, 256, 256, dtype=torch.float8_e4m3fn)
    flashinfer_w2 = torch.zeros(4, 256, 128, dtype=torch.float8_e4m3fn)
    flashinfer_s13 = torch.zeros(4, 2, 2)
    flashinfer_s2 = torch.zeros(4, 2, 1)
    gate_fp8 = torch.full((128, 256), 1).to(torch.float8_e4m3fn)
    up_fp8 = torch.full((128, 256), 2).to(torch.float8_e4m3fn)
    gate_scale = torch.full((1, 2), 3.0)
    up_scale = torch.full((1, 2), 4.0)
    for projection, weight, scale in (
        ("gate", gate_fp8, gate_scale),
        ("up", up_fp8, up_scale),
    ):
        flashinfer.load_expert_projection(
            flashinfer_spec,
            local_expert_id=0,
            projection=projection,
            loaded_weight=weight,
            loaded_scale=scale,
            w13_weight=flashinfer_w13,
            w2_weight=flashinfer_w2,
            w13_scale_inv=flashinfer_s13,
            w2_scale_inv=flashinfer_s2,
        )
    assert torch.equal(flashinfer_w13[0, :128], up_fp8)
    assert torch.equal(flashinfer_w13[0, 128:], gate_fp8)
    assert torch.equal(flashinfer_s13[0, :1], up_scale)
    assert torch.equal(flashinfer_s13[0, 1:], gate_scale)
