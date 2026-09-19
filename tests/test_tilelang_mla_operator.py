from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from importlib import metadata
from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.kernels.external.support import ExternalKernelFamilyError
from sparseengine.engine.cache_manager import (
    AttentionViewMeta,
    DecodeComputeView,
    MlaLatentPayload,
)
from sparseengine.kernels.tilelang.mla.runtime import (
    TileMlaDecodeKernel,
    TileMlaLaunchConfig,
    TileMlaLaunchPlan,
    tilelang_mla_support,
)
from sparseengine.kernels.triton.mla import MlaDecodeWorkspace
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.mla_attention import (
    MLA_ATTENTION_REGISTRY,
    MlaAttentionOpSpec,
    MlaTileLangScoreProvider,
    MlaTritonProvider,
)
from sparseengine.operators.registry import (
    OpResolver,
)
from sparseengine.platforms import DeviceCaps, PlatformEnum


def _spec(*, tp_size: int = 2) -> MlaAttentionOpSpec:
    return MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=tp_size,
        cuda_graph=True,
        score_output=AttentionScoreKind.RAW_QK_PER_HEAD,
        context_capacity=65536,
    )


def _h100_caps(**overrides) -> DeviceCaps:
    values = {
        "platform": PlatformEnum.CUDA,
        "device_type": "cuda",
        "device_index": 0,
        "device_name": "NVIDIA H100 80GB HBM3",
        "compute_capability": (9, 0),
        "runtime_version": "13.0",
        "supports_graph_capture": True,
        "supports_torch_compile": True,
        "supports_triton": True,
        "supports_pin_memory": True,
        "supports_bfloat16": True,
        "supports_native_fp8": True,
        "multi_processor_count": 132,
    }
    values.update(overrides)
    return DeviceCaps(**values)


def _cpu_workspace() -> MlaDecodeWorkspace:
    return MlaDecodeWorkspace(
        block_size=torch.empty(1, dtype=torch.int32),
        batch_start_indices=torch.empty(2, dtype=torch.int32),
        mid_output=torch.empty(20, 1, 512, dtype=torch.float32),
        mid_logsumexp=torch.empty(20, 1, dtype=torch.float32),
    )


def _view(*, score: torch.Tensor | None) -> DecodeComputeView:
    active_slots = torch.full((3, 64), -1, dtype=torch.int32)
    active_slots[2, :3] = torch.tensor([5, 2, 7], dtype=torch.int32)
    return DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=active_slots,
            req_indices=torch.tensor([2, -1], dtype=torch.int32),
            context_lens=torch.tensor([3, 0], dtype=torch.int32),
            max_context_len=64,
            attn_score=score,
        ),
        payload=MlaLatentPayload(
            latent_cache=torch.empty(8, 1, 512, dtype=torch.bfloat16),
            rope_cache=torch.empty(8, 1, 64, dtype=torch.bfloat16),
        ),
    )


def test_tilelang_support_does_not_import_runtime() -> None:
    def version(package: str) -> str:
        return {"tilelang": "0.1.9", "apache-tvm-ffi": "0.1.10"}[package]

    with patch.object(metadata, "version", side_effect=version):
        assert tilelang_mla_support() == (
            True,
            "tilelang 0.1.9, apache-tvm-ffi 0.1.10",
        )


def test_tilelang_runtime_import_does_not_require_tilelang() -> None:
    code = """
import sys

class RejectTileLang:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tilelang" or fullname.startswith("tilelang."):
            raise ModuleNotFoundError("blocked optional tilelang import")

sys.meta_path.insert(0, RejectTileLang())
from sparseengine.kernels.tilelang.mla.runtime import tilelang_mla_support
assert "tilelang" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("0.1.8", False),
        ("0.1.9", True),
        ("0.1.13", False),
        ("0.2.0+cu130", False),
    ],
)
def test_tilelang_support_version_boundary(version: str, supported: bool) -> None:
    def installed(package: str) -> str:
        return version if package == "tilelang" else "0.1.10"

    with patch.object(metadata, "version", side_effect=installed):
        if supported:
            assert tilelang_mla_support()[0]
        else:
            with pytest.raises(ExternalKernelFamilyError, match="validated dependency pair"):
                tilelang_mla_support()


def test_tilelang_support_reports_missing_package() -> None:
    with patch.object(
        metadata,
        "version",
        side_effect=metadata.PackageNotFoundError,
    ):
        assert tilelang_mla_support() == (False, "tilelang is not installed")


def test_tilelang_support_rejects_unvalidated_dependency_pair() -> None:
    def version(package: str) -> str:
        return {
            "tilelang": "0.1.9",
            "apache-tvm-ffi": "0.1.13.post2",
        }[package]

    with (
        patch.object(metadata, "version", side_effect=version),
        pytest.raises(ExternalKernelFamilyError, match="validated dependency pair"),
    ):
        tilelang_mla_support()


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("0.1.1", False),
        ("0.1.2", False),
        ("0.1.10", True),
        ("0.1.13.post2", False),
        ("0.2.0", False),
    ],
)
def test_tvm_ffi_support_version_boundary(version: str, supported: bool) -> None:
    def installed(package: str) -> str:
        return "0.1.9" if package == "tilelang" else version

    with patch.object(metadata, "version", side_effect=installed):
        if supported:
            assert tilelang_mla_support()[0]
        else:
            with pytest.raises(ExternalKernelFamilyError, match="validated dependency pair"):
                tilelang_mla_support()


@pytest.mark.parametrize(("tp_size", "local_heads"), [(1, 20), (2, 10), (4, 5)])
def test_tilelang_provider_binds_rank_local_head_count(
    tp_size: int,
    local_heads: int,
) -> None:
    workspace = _cpu_workspace()
    with (
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
        patch("sparseengine.operators.mla_attention.SglFa3DecodeKernel"),
        patch(
            "sparseengine.operators.mla_attention.TileMlaDecodeKernel"
        ) as tilelang_cls,
    ):
        MlaTileLangScoreProvider(
            op_spec=_spec(tp_size=tp_size),
            device="cpu",
            max_batch_size=2,
            sm_count=_h100_caps().multi_processor_count,
        )

    tilelang_cls.assert_called_once()
    kwargs = tilelang_cls.call_args.kwargs
    assert kwargs["device"] == torch.device("cpu")
    assert kwargs["softmax_scale"] == 256**-0.5
    assert kwargs["valid_heads"] == local_heads
    plan = kwargs["launch_plan"]
    assert isinstance(plan, TileMlaLaunchPlan)
    assert plan.context_capacity == 65536
    assert plan.local_q_heads == local_heads
    assert all(config.score_mode == "per_head" for config in plan.configs)


def test_missing_tilelang_binds_score_capable_triton_provider() -> None:
    workspace = _cpu_workspace()
    with (
        patch(
            "sparseengine.operators.mla_attention.sgl_fa3_device_support",
            return_value=(True, "sgl test"),
        ),
        patch(
            "sparseengine.operators.mla_attention.tilelang_mla_support",
            return_value=(False, "tilelang missing"),
        ),
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
        patch("sparseengine.operators.mla_attention.SglFa3DecodeKernel"),
    ):
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            _spec(),
            _h100_caps(),
            op_spec=_spec(),
            device="cpu",
            max_batch_size=2,
        )
    assert type(resolved.provider) is MlaTritonProvider
    assert (
        "sgl_fa3_sm90",
        "does not satisfy the prepared score-output contract",
    ) in resolved.rejected
    assert (
        "tilelang_score",
        "tilelang missing",
    ) in resolved.rejected


def test_tilelang_mla_profile_uses_hardware_family_not_tp_or_capacity() -> None:
    spec = _spec(tp_size=4)
    workspace = _cpu_workspace()
    with (
        patch(
            "sparseengine.operators.mla_attention.sgl_fa3_device_support",
            return_value=(True, "sgl test"),
        ),
        patch(
            "sparseengine.operators.mla_attention.tilelang_mla_support",
            return_value=(True, "tilelang test"),
        ),
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
    ):
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            spec,
            _h100_caps(device_name="NVIDIA H20"),
            op_spec=spec,
            device="cpu",
            max_batch_size=2,
        )

    assert type(resolved.provider) is MlaTritonProvider
    assert resolved.report.selected_profile is None
    assert resolved.report.profiles[0].matched is False
    assert resolved.report.candidates[2].supported is True


@pytest.mark.parametrize(
    "device_name",
    ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"],
)
def test_tilelang_mla_h100_family_profile_overrides_default_portfolio(
    device_name: str,
) -> None:
    spec = _spec()
    workspace = _cpu_workspace()
    with (
        patch(
            "sparseengine.operators.mla_attention.sgl_fa3_device_support",
            return_value=(True, "sgl test"),
        ),
        patch(
            "sparseengine.operators.mla_attention.tilelang_mla_support",
            return_value=(True, "tilelang test"),
        ),
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
        patch("sparseengine.operators.mla_attention.SglFa3DecodeKernel"),
        patch("sparseengine.operators.mla_attention.TileMlaDecodeKernel"),
    ):
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            spec,
            _h100_caps(device_name=device_name),
            op_spec=spec,
            device="cpu",
            max_batch_size=2,
        )

    assert type(resolved.provider) is MlaTileLangScoreProvider
    assert resolved.report.selected_profile == "tilelang_score_h100_profile"
    assert resolved.report.selection_basis == "profile_override"


def test_decode_graph_score_contract_binds_static_tilelang_plan() -> None:
    spec = replace(
        _spec(),
    )
    workspace = _cpu_workspace()
    with (
        patch(
            "sparseengine.operators.mla_attention.sgl_fa3_device_support",
            return_value=(True, "sgl test"),
        ),
        patch("sparseengine.operators.mla_attention.SglFa3DecodeKernel"),
        patch("sparseengine.operators.mla_attention.TileMlaDecodeKernel"),
        patch(
            "sparseengine.operators.mla_attention.tilelang_mla_support",
            return_value=(True, "tilelang test"),
        ),
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
    ):
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            spec,
            _h100_caps(),
            op_spec=spec,
            device="cpu",
            max_batch_size=2,
        )

    assert type(resolved.provider) is MlaTileLangScoreProvider
    assert resolved.provider.supports_decode_graph
    assert resolved.provider.tilelang_launch_plan.context_capacity == 65536


def test_tilelang_bound_plan_is_not_reselected_by_replay_context() -> None:
    """Context changes must not resize the graph's split workspace."""
    plan = TileMlaLaunchPlan.build(
        context_capacity=65536,
        local_q_heads=10,
        max_batch_size=32,
        need_score=True,
        score_mode="per_head",
        sm_count=_h100_caps().multi_processor_count,
    )
    runner = TileMlaDecodeKernel(device="cpu", softmax_scale=0.0625, launch_plan=plan)
    for length in (1, 512, 8192, plan.context_capacity):
        assert runner._config_for(batch_size=17, context_capacity=length, need_score=True) is plan.configs[16]
    with pytest.raises(ValueError, match="exceeds the static launch plan"):
        runner._config_for(batch_size=17, context_capacity=plan.context_capacity + 1, need_score=True)


def test_tilelang_launch_configs_are_independent_of_context_capacity() -> None:
    """Storage capacity must not become a hidden context tuning bucket."""
    kwargs = dict(local_q_heads=10, max_batch_size=32, need_score=True,
                  score_mode="per_head", sm_count=_h100_caps().multi_processor_count)
    short = TileMlaLaunchPlan.build(context_capacity=1024, **kwargs)
    long = TileMlaLaunchPlan.build(context_capacity=131072, **kwargs)
    assert short.configs == long.configs


def test_tilelang_bind_rejects_missing_sm_count_before_allocating() -> None:
    with patch("sparseengine.operators.mla_attention.allocate_mla_decode_workspace") as allocate:
        with pytest.raises(ValueError, match="SM count"):
            MlaTileLangScoreProvider.bind(_spec(), _h100_caps(multi_processor_count=None),
                                          op_spec=_spec(), device="cpu", max_batch_size=2)
    allocate.assert_not_called()


def test_decode_graph_reduced_score_contract_binds_static_triton_provider() -> None:
    spec = replace(
        _spec(),
        score_output=AttentionScoreKind.RAW_QK_REDUCED,
    )
    with (
        patch(
            "sparseengine.operators.mla_attention.sgl_fa3_device_support",
            return_value=(True, "sgl test"),
        ),
        patch(
            "sparseengine.operators.mla_attention.tilelang_mla_support",
            return_value=(True, "tilelang test"),
        ),
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=_cpu_workspace(),
        ),
    ):
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            spec,
            _h100_caps(),
            op_spec=spec,
            device="cpu",
            max_batch_size=2,
        )

    assert type(resolved.provider) is MlaTritonProvider
    assert (
        "tilelang_score",
        "requires the RAW_QK_PER_HEAD decode score contract",
    ) in resolved.rejected


def _provider_with_mocks() -> tuple[MlaTileLangScoreProvider, Mock, Mock]:
    fa3 = Mock(return_value=torch.empty(2, 10, 512, dtype=torch.bfloat16))
    tilelang = Mock(return_value=torch.empty(2, 10, 512, dtype=torch.bfloat16))
    tilelang.runtime_metadata.return_value = {
        "compiled_variant_count": 1,
        "compiled_variants": [],
    }
    with (
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=_cpu_workspace(),
        ),
        patch(
            "sparseengine.operators.mla_attention.SglFa3DecodeKernel",
            return_value=fa3,
        ),
        patch(
            "sparseengine.operators.mla_attention.TileMlaDecodeKernel",
            return_value=tilelang,
        ),
    ):
        provider = MlaTileLangScoreProvider(
            op_spec=_spec(),
            device="cpu",
            max_batch_size=2,
            sm_count=_h100_caps().multi_processor_count,
        )
    return provider, fa3, tilelang


def test_per_head_score_path_routes_to_tilelang_with_caller_owned_score() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    score = torch.full((2, 10, 64), -1e20, dtype=torch.float32)
    view = _view(score=score)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch(
        "sparseengine.operators.mla_attention.validate_mla_decode_metadata"
    ):
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_called_once_with(
        q_latent,
        q_rope,
        view.payload.latent_cache,
        view.payload.rope_cache,
        view.meta.active_slots,
        view.meta.req_indices,
        view.meta.context_lens,
        output,
        attn_score=score,
        max_context_len=64,
    )
    assert provider.runtime_kernel_stats() == {
        "kernel_paths": {
            "tilelang_score": {
                "cuda_graph_capture_dispatches": 0,
                "eager_dispatches": 1,
            }
        },
        "fallback_reasons": {},
        "tilelang": {
            "compiled_variant_count": 1,
            "compiled_variants": [],
        },
    }


def test_noncontiguous_glm_queries_route_to_tilelang() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    score = torch.full((2, 10, 64), -1e20, dtype=torch.float32)
    view = _view(score=score)
    q_latent = torch.empty(10, 2, 512, dtype=torch.bfloat16).transpose(0, 1)
    q_rope = torch.empty(10, 2, 64, dtype=torch.bfloat16).transpose(0, 1)
    output = torch.empty(q_latent.shape, dtype=q_latent.dtype)

    assert not q_latent.is_contiguous()
    assert not q_rope.is_contiguous()
    with patch(
        "sparseengine.operators.mla_attention.validate_mla_decode_metadata"
    ):
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_called_once()


def test_runtime_kernel_stats_distinguish_cuda_graph_capture() -> None:
    provider, _, tilelang = _provider_with_mocks()
    view = _view(score=torch.empty(2, 10, 64, dtype=torch.float32))
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with (
        patch(
            "sparseengine.operators.mla_attention.validate_mla_decode_metadata"
        ),
        patch(
            "sparseengine.operators.mla_attention.device_runtime.is_stream_capturing",
            return_value=True,
        ),
    ):
        provider.run(q_latent, q_rope, view, output)

    tilelang.assert_called_once()
    assert provider.runtime_kernel_stats()["kernel_paths"]["tilelang_score"] == {
        "cuda_graph_capture_dispatches": 1,
        "eager_dispatches": 0,
    }


def test_no_score_path_remains_fa3() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    view = _view(score=None)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch(
        "sparseengine.operators.mla_attention.validate_mla_decode_metadata"
    ):
        provider.run(q_latent, q_rope, view, output)

    tilelang.assert_not_called()
    fa3.assert_called_once()
    assert fa3.call_args.kwargs["num_splits"] == 0


@pytest.mark.parametrize(
    ("score", "message"),
    [
        (torch.empty(2, 64, dtype=torch.float32), "RAW_QK_PER_HEAD"),
        (torch.empty(2, 10, 64, dtype=torch.bfloat16), "must use FP32"),
        (torch.empty(2, 9, 64, dtype=torch.float32), "head count"),
    ],
)
def test_unsupported_score_contract_fails_instead_of_falling_back(
    score,
    message: str,
) -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    view = _view(score=score)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with pytest.raises((TypeError, ValueError), match=message):
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_not_called()
    assert provider.runtime_kernel_stats()["fallback_reasons"] == {}


def test_score_capacity_smaller_than_declared_context_fails() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    view = _view(score=torch.empty(2, 10, 64, dtype=torch.float32))
    object.__setattr__(view.meta, "max_context_len", 128)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with pytest.raises(ValueError, match="must cover max_context_len"):
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_not_called()
    assert provider.runtime_kernel_stats()["fallback_reasons"] == {}


def test_score_capacity_may_include_padding_beyond_active_slots() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    view = _view(score=torch.empty(2, 10, 64, dtype=torch.float32))
    object.__setattr__(
        view.meta,
        "active_slots",
        torch.zeros((2, 4), dtype=torch.int32),
    )
    object.__setattr__(view.meta, "max_context_len", 4)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch(
        "sparseengine.operators.mla_attention.validate_mla_decode_metadata"
    ):
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_called_once()


def test_noncontiguous_active_slots_fail_instead_of_falling_back() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    view = _view(score=torch.empty(2, 10, 64, dtype=torch.float32))
    backing = torch.full((3, 128), -1, dtype=torch.int32)
    backing[2, :6:2] = torch.tensor([5, 2, 7], dtype=torch.int32)
    object.__setattr__(view.meta, "active_slots", backing[:, ::2])
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with pytest.raises(ValueError, match="noncontiguous:active_slots"):
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_not_called()
    assert provider.runtime_kernel_stats()["fallback_reasons"] == {}


def test_noncontiguous_per_head_score_routes_to_tilelang_staging() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    score = torch.empty(2, 10, 128, dtype=torch.float32)[:, :, ::2]
    view = _view(score=score)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch(
        "sparseengine.operators.mla_attention.validate_mla_decode_metadata"
    ):
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_called_once()
    assert tilelang.call_args.kwargs["attn_score"] is score


def test_tilelang_runner_rejects_score_capacity_smaller_than_context() -> None:
    runner = TileMlaDecodeKernel(
        device="cpu",
        softmax_scale=0.0625,
        fixed_config=TileMlaLaunchConfig(1, score_mode="per_head"),
    )
    view = _view(score=torch.empty(2, 10, 63, dtype=torch.float32))
    with pytest.raises(ValueError, match="must fit"):
        runner(
            torch.empty(2, 10, 512, dtype=torch.bfloat16),
            torch.empty(2, 10, 64, dtype=torch.bfloat16),
            view.payload.latent_cache,
            view.payload.rope_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            torch.empty(2, 10, 512, dtype=torch.bfloat16),
            attn_score=view.meta.attn_score,
            max_context_len=64,
        )


def test_tilelang_runner_rejects_unsupported_local_head_count() -> None:
    with pytest.raises(ValueError, match="valid_heads must be one of"):
        TileMlaDecodeKernel(
            device="cpu",
            softmax_scale=0.0625,
            valid_heads=7,
        )
