import builtins
import gc
import importlib
import os
import sys
import tempfile
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.engine.cache_manager.base import (
    AttentionViewMeta,
    ExplicitKVPayload,
    PrefillComputeView,
)
from sparseengine.kernels.external.support import (
    ExternalKernelFamilyError,
    KernelFamilyHealth,
    KernelFamilyState,
)
from sparseengine.layers.attention import Attention
from sparseengine.method_registry import (
    PrefillScoreCollectionKind,
    sparse_prefill_attention_contract,
)
from sparseengine.models.attention_runtime import build_mha_prefill_attention_spec
from sparseengine.operators.prefill_attention import (
    PREFILL_ATTENTION_REGISTRY,
    FlashInferFa2Sm120PagedPrefillAttentionProvider,
    FlashInferPagedPrefillAttentionProvider,
    PreparedPrefillAttentionOp,
    PrefillAttentionOpSpec,
    SglFa3PagedPrefillAttentionProvider,
    TilelangGqaPagedPrefillAttentionProvider,
    TritonPagedPrefillAttentionProvider,
    _FlashInferPagedPrefillState,
    _resolve_prefill_attention_provider,
)
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.registry import NoProviderError, OpResolver
from sparseengine.platforms import DeviceCaps, PlatformEnum


def _spec(**overrides) -> PrefillAttentionOpSpec:
    values = {
        "num_query_heads": 12,
        "num_kv_heads": 2,
        "head_dim": 128,
        "activation_dtype": torch.bfloat16,
        "softmax_scale": 128**-0.5,
        "causal": True,
        "page_size": 1,
        "score_output": AttentionScoreKind.NONE,
        "layer_varying_page_table": False,
    }
    values.update(overrides)
    return PrefillAttentionOpSpec(**values)


def test_prefill_softmax_lse_requirement_is_part_of_kernel_contract():
    spec = _spec(return_softmax_lse=True)

    assert spec.kernel_request.requires_softmax_lse
    assert not TritonPagedPrefillAttentionProvider.supports(
        spec,
        _h100_caps(),
    ).supported


@pytest.mark.parametrize(
    ("device_kind", "head_dim", "activation_dtype"),
    [
        ("sm120", 128, torch.bfloat16),
        ("sm120", 256, torch.bfloat16),
        ("h100", 128, torch.float16),
    ],
)
def test_optional_h2o_lse_falls_back_during_provider_resolution(
    device_kind,
    head_dim,
    activation_dtype,
):
    caps = _sm120_caps() if device_kind == "sm120" else _h100_caps()
    spec = _spec(
        head_dim=head_dim,
        softmax_scale=head_dim**-0.5,
        activation_dtype=activation_dtype,
        layer_varying_page_table=True,
        return_softmax_lse=True,
        allow_softmax_lse_fallback=True,
        prefill_sparse_method="h2o_prefill",
    )
    platform = SimpleNamespace(get_device_caps=lambda _index: caps)

    with patch(
        "sparseengine.platforms.get_current_platform",
        return_value=platform,
    ):
        provider, execution_spec = _resolve_prefill_attention_provider(
            spec, device_index=0
        )

    assert provider.name == "triton_paged_prefill"
    assert spec.return_softmax_lse
    assert not execution_spec.return_softmax_lse


def test_hard_prefill_lse_requirement_does_not_fall_back():
    spec = _spec(
        head_dim=256,
        softmax_scale=256**-0.5,
        layer_varying_page_table=True,
        return_softmax_lse=True,
    )
    platform = SimpleNamespace(get_device_caps=lambda _index: _sm120_caps())

    with (
        patch(
            "sparseengine.platforms.get_current_platform",
            return_value=platform,
        ),
        pytest.raises(NoProviderError),
    ):
        _resolve_prefill_attention_provider(spec, device_index=0)


def _h100_caps(**overrides) -> DeviceCaps:
    values = {
        "platform": PlatformEnum.CUDA,
        "device_type": "cuda",
        "device_index": 0,
        "device_name": "NVIDIA H100 80GB HBM3",
        "compute_capability": (9, 0),
        "runtime_version": "13.0",
        "supports_graph_capture": True,
        "supports_triton": True,
        "supports_bfloat16": True,
        "supports_native_fp8": True,
    }
    values.update(overrides)
    return DeviceCaps(**values)


def _sm120_caps(**overrides) -> DeviceCaps:
    values = {
        "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "compute_capability": (12, 0),
    }
    values.update(overrides)
    return _h100_caps(**values)


@pytest.fixture(autouse=True)
def _mock_flashinfer_paged_prefill_contract():
    with patch(
        "sparseengine.operators.prefill_attention.flashinfer_paged_prefill_support",
        return_value=(True, "flashinfer prefill available"),
    ):
        yield


@pytest.mark.parametrize(
    ("method", "main_score", "collection"),
    [
        ("", AttentionScoreKind.NONE, PrefillScoreCollectionKind.NONE),
        ("snapkv", AttentionScoreKind.NONE, PrefillScoreCollectionKind.METHOD_OWNED_POSTHOC_REDUCED),
        ("pyramidkv", AttentionScoreKind.NONE, PrefillScoreCollectionKind.METHOD_OWNED_POSTHOC_REDUCED),
        ("h2o", AttentionScoreKind.NONE, PrefillScoreCollectionKind.METHOD_OWNED_POSTHOC_REDUCED),
        ("omnikv", AttentionScoreKind.NONE, PrefillScoreCollectionKind.NONE),
        ("deltakv", AttentionScoreKind.NONE, PrefillScoreCollectionKind.NONE),
    ],
)
def test_sparse_prefill_contract_separates_main_and_posthoc_scores(
    method, main_score, collection
):
    contract = sparse_prefill_attention_contract(method)

    assert contract.main_score_kind is main_score
    assert contract.score_collection is collection


@pytest.mark.parametrize("method", ["omnikv", "quest"])
def test_dense_prefill_sparse_methods_share_one_physical_page_table(method):
    contract = sparse_prefill_attention_contract(method)

    assert not contract.layer_varying_page_table


@pytest.mark.parametrize("method", ["streamingllm", "snapkv", "deltakv"])
def test_per_layer_physical_cache_methods_keep_layer_varying_prefill(method):
    contract = sparse_prefill_attention_contract(method)

    assert contract.layer_varying_page_table


@pytest.mark.parametrize("method", ["omnikv", "quest"])
def test_sm120_dense_prefill_sparse_method_selects_flashinfer_fa2(method):
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
    )
    runtime_config = SimpleNamespace(
        prefill_sparse_method="",
        sparse_prefill_score_mode=None,
        h2o_prefill_score_window=0,
    )
    spec = build_mha_prefill_attention_spec(
        config,
        sparse_method=method,
        attention_tp_size=1,
        runtime_config=runtime_config,
    )

    resolved = OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
        spec,
        _sm120_caps(),
    )

    assert not spec.layer_varying_page_table
    assert resolved.provider.name == "flashinfer_paged_prefill_fa2_sm120"
    assert resolved.report.selection_basis == "upstream_default"


def test_h2o_full_query_logits_request_fused_reduced_prefill_score():
    contract = sparse_prefill_attention_contract(
        "h2o",
        sparse_prefill_score_mode="logits",
        h2o_prefill_score_window=0,
    )

    assert contract.main_score_kind is AttentionScoreKind.RAW_QK_REDUCED
    assert (
        contract.score_collection
        is PrefillScoreCollectionKind.MAIN_ATTENTION_REDUCED
    )


@pytest.mark.parametrize(
    ("sparse_method", "prefill_sparse_method"),
    [("", "h2o_prefill"), ("h2o", "")],
)
def test_independent_h2o_axes_keep_prompt_score_and_page_table_contract(
    sparse_method,
    prefill_sparse_method,
):
    contract = sparse_prefill_attention_contract(
        sparse_method,
        prefill_sparse_method=prefill_sparse_method,
        sparse_prefill_score_mode="probability",
    )

    assert contract.layer_varying_page_table
    assert contract.main_score_kind is AttentionScoreKind.NONE
    assert (
        contract.score_collection
        is PrefillScoreCollectionKind.METHOD_OWNED_POSTHOC_REDUCED
    )


@pytest.mark.parametrize(
    ("sparse_method", "prefill_sparse_method"),
    [("", "h2o_prefill"), ("h2o", "")],
)
def test_independent_h2o_axes_request_lse_for_probability_scoring(
    sparse_method,
    prefill_sparse_method,
):
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
    )
    runtime_config = SimpleNamespace(
        prefill_sparse_method=prefill_sparse_method,
        sparse_prefill_score_mode="probability",
        h2o_prefill_score_window=0,
    )

    spec = build_mha_prefill_attention_spec(
        config,
        sparse_method=sparse_method,
        attention_tp_size=1,
        runtime_config=runtime_config,
    )

    assert spec.return_softmax_lse
    assert spec.allow_softmax_lse_fallback


def test_h2o_flashprefill_uses_method_owned_posthoc_prefill_scoring():
    contract = sparse_prefill_attention_contract(
        "h2o",
        prefill_sparse_method="flashprefill_v2",
        sparse_prefill_score_mode="logits",
        h2o_prefill_score_window=0,
    )

    assert contract.main_score_kind is AttentionScoreKind.NONE
    assert (
        contract.score_collection
        is PrefillScoreCollectionKind.METHOD_OWNED_POSTHOC_REDUCED
    )


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (_spec(num_query_heads=32, num_kv_heads=4, head_dim=64), "head_dim"),
        (
            _spec(
                num_query_heads=32,
                num_kv_heads=4,
                activation_dtype=torch.float16,
            ),
            "activation dtype",
        ),
        (
            _spec(
                num_query_heads=32,
                num_kv_heads=4,
                score_output=AttentionScoreKind.RAW_QK_PER_HEAD,
            ),
            "RAW_QK_PER_HEAD",
        ),
        (
            _spec(num_query_heads=32, num_kv_heads=4, cuda_graph=True),
            "CUDA Graph",
        ),
    ],
)
def test_sgl_fa3_provider_rejects_unvalidated_contracts(spec, reason):
    result = SglFa3PagedPrefillAttentionProvider.supports(spec, _h100_caps())

    assert not result.supported
    assert reason in result.reason


@pytest.mark.parametrize(
    ("spec", "caps", "reason"),
    [
        (_spec(head_dim=64), _h100_caps(), "head_dim"),
        (_spec(activation_dtype=torch.float16), _h100_caps(), "activation dtype"),
        (_spec(page_size=16), _h100_caps(), "page_size=1"),
        (
            _spec(score_output=AttentionScoreKind.RAW_QK_PER_HEAD),
            _h100_caps(),
            "RAW_QK_PER_HEAD",
        ),
        (
            _spec(layer_varying_page_table=True),
            _h100_caps(),
            "layer-varying",
        ),
        (
            _spec(),
            _h100_caps(compute_capability=(8, 0), device_name="NVIDIA A100"),
            "compute capability",
        ),
    ],
)
def test_flashinfer_provider_rejects_unsupported_contracts(spec, caps, reason):
    result = FlashInferPagedPrefillAttentionProvider.supports(spec, caps)
    assert not result.supported
    assert reason in result.reason


@pytest.mark.parametrize(
    ("spec", "caps", "reason"),
    [
        (
            _spec(
                num_query_heads=24,
                num_kv_heads=4,
                head_dim=192,
                softmax_scale=192**-0.5,
            ),
            _sm120_caps(),
            "head_dim",
        ),
        (
            _spec(
                num_query_heads=24,
                num_kv_heads=4,
                head_dim=256,
                softmax_scale=256**-0.5,
                activation_dtype=torch.float16,
            ),
            _sm120_caps(),
            "activation dtype",
        ),
        (
            _spec(
                num_query_heads=24,
                num_kv_heads=4,
                head_dim=256,
                softmax_scale=256**-0.5,
                score_output=AttentionScoreKind.RAW_QK_REDUCED,
            ),
            _sm120_caps(),
            "RAW_QK_REDUCED",
        ),
        (
            _spec(
                num_query_heads=24,
                num_kv_heads=4,
                head_dim=256,
                softmax_scale=256**-0.5,
                layer_varying_page_table=True,
            ),
            _sm120_caps(),
            "layer-varying",
        ),
    ],
)
def test_flashinfer_fa2_sm120_rejects_unsupported_contracts(spec, caps, reason):
    result = FlashInferFa2Sm120PagedPrefillAttentionProvider.supports(spec, caps)

    assert not result.supported
    assert reason in result.reason


def test_flashinfer_fa2_sm120_support_is_not_limited_by_local_device_profile():
    result = FlashInferFa2Sm120PagedPrefillAttentionProvider.supports(
        _spec(
            num_query_heads=32,
            num_kv_heads=8,
            head_dim=64,
            softmax_scale=64**-0.5,
        ),
        _sm120_caps(device_name="NVIDIA GeForce RTX 5090"),
    )

    assert result.supported


@patch(
    "sparseengine.kernels.tilelang.gqa.runtime.tilelang_gqa_device_support",
    return_value=(False, "tilelang is not installed"),
)
@patch(
    "sparseengine.operators.prefill_attention.sgl_fa3_device_support",
    return_value=(False, "sglang-kernel is not installed"),
)
def test_resolver_does_not_hide_broken_flashinfer_family(_sgl, _tl):
    health = KernelFamilyHealth(
        family="flashinfer-python",
        state=KernelFamilyState.BROKEN,
        version="0.6.14",
        reason="requires flashinfer-python>=0.6.15,<0.7",
    )
    with (
        patch(
            "sparseengine.operators.prefill_attention.flashinfer_paged_prefill_support",
            side_effect=ExternalKernelFamilyError(health, feature="FA3 paged prefill"),
        ),
        pytest.raises(ExternalKernelFamilyError, match="0.6.15"),
    ):
        OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
            _spec(), _h100_caps()
        )


@patch(
    "sparseengine.kernels.tilelang.gqa.runtime.tilelang_gqa_device_support",
    return_value=(False, "tilelang is not installed"),
)
@patch(
    "sparseengine.operators.prefill_attention.sgl_fa3_device_support",
    return_value=(False, "unsupported sglang-kernel FA3 fwd schema"),
)
def test_resolver_falls_back_to_triton_for_variant_table_when_sgl_abi_rejected(
    _support, _tl
):
    resolved = OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
        _spec(
            num_query_heads=32,
            num_kv_heads=4,
            layer_varying_page_table=True,
        ),
        _h100_caps(),
    )

    assert (
        "sgl_fa3_paged_prefill_sm90",
        "unsupported sglang-kernel FA3 fwd schema",
    ) in resolved.rejected


def test_tilelang_provider_rejects_per_head_score_contract():
    result = TilelangGqaPagedPrefillAttentionProvider.supports(
        _spec(
            num_query_heads=16,
            num_kv_heads=2,
            score_output=AttentionScoreKind.RAW_QK_PER_HEAD,
        ),
        _h100_caps(),
    )

    assert not result.supported
    assert "RAW_QK_PER_HEAD" in result.reason


@patch(
    "sparseengine.kernels.tilelang.gqa.runtime.tilelang_gqa_device_support",
    return_value=(True, "dependencies available"),
)
def test_tilelang_atomic_support_is_not_narrowed_to_the_profiled_h100(_support):
    result = TilelangGqaPagedPrefillAttentionProvider.supports(
        _spec(
            num_query_heads=16,
            num_kv_heads=2,
            score_output=AttentionScoreKind.RAW_QK_REDUCED,
        ),
        _sm120_caps(),
    )

    assert result.supported


@patch(
    "sparseengine.kernels.tilelang.gqa.runtime.tilelang_gqa_device_support",
    return_value=(True, "validated pair"),
)
def test_unprofiled_tilelang_score_provider_does_not_override_triton(_support):
    resolved = OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
        _spec(
            num_query_heads=16,
            num_kv_heads=2,
            score_output=AttentionScoreKind.RAW_QK_REDUCED,
            layer_varying_page_table=True,
        ),
        _h100_caps(),
    )

    assert resolved.provider.name == "triton_paged_prefill"
    assert resolved.report.selected_profile is None


def test_tilelang_installed_with_unvalidated_pair_is_broken(monkeypatch):
    from sparseengine.kernels.tilelang import support

    versions = {"tilelang": "0.2.0", "apache-tvm-ffi": "0.1.10"}
    monkeypatch.setattr(support.metadata, "version", versions.__getitem__)

    with pytest.raises(ExternalKernelFamilyError, match="validated dependency pair"):
        support.tilelang_dependency_support()


def test_tilelang_support_probe_does_not_import_compiler(monkeypatch):
    module_name = "sparseengine.kernels.tilelang.gqa.runtime"
    package_name = "sparseengine.kernels.tilelang.gqa"
    sys.modules.pop(module_name, None)
    sys.modules.pop(package_name, None)
    compiler_modules_before = {
        name for name in sys.modules if name == "tilelang" or name.startswith("tilelang.")
    }
    monkeypatch.delenv("TMPDIR", raising=False)
    original_tempdir = tempfile.tempdir
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tilelang" or name.startswith("tilelang."):
            raise AssertionError(f"support probe imported compiler module {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runtime = importlib.import_module(module_name)

    assert callable(runtime.tilelang_gqa_device_support)
    compiler_modules_after = {
        name for name in sys.modules if name == "tilelang" or name.startswith("tilelang.")
    }
    assert compiler_modules_after == compiler_modules_before
    assert "TMPDIR" not in os.environ
    assert tempfile.tempdir is original_tempdir


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (_spec(causal=False), "causal attention"),
        (_spec(page_size=16), "page_size=1"),
        (_spec(softmax_scale=0.125), "default head-dimension scale"),
    ],
)
def test_triton_provider_rejects_unimplemented_attention_semantics(spec, reason):
    result = TritonPagedPrefillAttentionProvider.supports(spec, _h100_caps())

    assert not result.supported
    assert reason in result.reason


def test_resolver_rejects_page_sizes_without_a_valid_kernel():
    with pytest.raises(RuntimeError, match="No paged prefill attention provider"):
        OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
            _spec(page_size=16),
            _h100_caps(),
        )


def test_paged_prefill_rejects_non_int32_page_table_before_kernel_launch():
    provider = TritonPagedPrefillAttentionProvider()
    view = SimpleNamespace(active_slots=torch.zeros(1, 2, dtype=torch.int64))

    with pytest.raises(TypeError, match="int32 physical-slot page table"):
        provider.run(
            _spec(),
            torch.empty(1, 12, 128),
            view,
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            chunk_lens=torch.tensor([1], dtype=torch.int32),
            max_context_len=1,
            layer_idx=0,
        )


def test_prepared_prefill_close_releases_provider_state():
    class State:
        pass

    provider = FlashInferPagedPrefillAttentionProvider()
    state = State()
    state_ref = weakref.ref(state)
    provider._state = state
    op = PreparedPrefillAttentionOp(_spec(), provider)
    del state

    op.close()
    gc.collect()

    assert provider._state is None
    assert state_ref() is None
    with pytest.raises(RuntimeError, match="closed"):
        op.run(None, None)


def test_flashinfer_prefill_passes_kv_cache_as_page_views_without_copying():
    wrapper = SimpleNamespace(run=Mock())
    provider = FlashInferPagedPrefillAttentionProvider()
    provider._state = SimpleNamespace(
        plan_scope=None,
        plan=Mock(),
        wrapper=wrapper,
    )
    q = torch.empty(1, 12, 128, dtype=torch.bfloat16)
    k_cache = torch.empty(5, 2, 128, dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    view = SimpleNamespace(
        k_cache=k_cache,
        v_cache=v_cache,
        active_slots=torch.tensor([[4, 1]], dtype=torch.int32),
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([2], dtype=torch.int32),
        attn_score=None,
    )

    plan_scope = object()
    provider._state.plan_scope = plan_scope
    with patch(
        "sparseengine.utils.context.get_context",
        return_value=SimpleNamespace(attention_validation_scope=plan_scope),
    ):
        provider.run(
            _spec(),
            q,
            view,
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            chunk_lens=torch.tensor([1], dtype=torch.int32),
            max_context_len=2,
            layer_idx=1,
        )

    provider._state.plan.assert_not_called()
    paged_k, paged_v = wrapper.run.call_args.args[1]
    assert paged_k.shape == (5, 1, 2, 128)
    assert paged_v.shape == (5, 1, 2, 128)
    assert paged_k.data_ptr() == k_cache.data_ptr()
    assert paged_v.data_ptr() == v_cache.data_ptr()


def test_flashinfer_prefill_plans_on_first_full_attention_call_per_step():
    wrapper = SimpleNamespace(
        plan=Mock(),
        run=Mock(),
        workspace_size=Mock(return_value=(16, 16)),
        reset_workspace_buffer=Mock(),
    )
    state = object.__new__(_FlashInferPagedPrefillState)
    state.backend = "fa2"
    state.wrapper = wrapper
    state.workspace = torch.empty(16, dtype=torch.uint8)
    state.int_workspace = torch.empty(16, dtype=torch.uint8)
    state.plan_scope = None
    provider = FlashInferFa2Sm120PagedPrefillAttentionProvider()
    provider._state = state
    spec = _spec(
        num_query_heads=24,
        num_kv_heads=4,
        head_dim=256,
        softmax_scale=256**-0.5,
    )
    q = torch.empty(1, 24, 256, dtype=torch.bfloat16)
    view = SimpleNamespace(
        k_cache=torch.empty(5, 4, 256, dtype=torch.bfloat16),
        v_cache=torch.empty(5, 4, 256, dtype=torch.bfloat16),
        active_slots=torch.tensor([[4, 1]], dtype=torch.int32),
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([2], dtype=torch.int32),
        attn_score=None,
    )
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32)
    chunk_lens = torch.tensor([1], dtype=torch.int32)
    context = SimpleNamespace(attention_validation_scope=object())

    with patch("sparseengine.utils.context.get_context", return_value=context):
        provider.run(
            spec,
            q,
            view,
            qo_indptr=qo_indptr,
            chunk_lens=chunk_lens,
            max_context_len=2,
            layer_idx=3,
        )
        provider.run(
            spec,
            q,
            view,
            qo_indptr=qo_indptr,
            chunk_lens=chunk_lens,
            max_context_len=2,
            layer_idx=7,
        )
        first_scope = context.attention_validation_scope
        context.attention_validation_scope = object()
        view.active_slots = torch.tensor([[2, 3]], dtype=torch.int32)
        view.context_lens = torch.tensor([1], dtype=torch.int32)
        provider.run(
            spec,
            q,
            view,
            qo_indptr=qo_indptr,
            chunk_lens=chunk_lens,
            max_context_len=1,
            layer_idx=3,
        )

    assert wrapper.plan.call_count == 2
    assert wrapper.workspace_size.call_count == 2
    wrapper.reset_workspace_buffer.assert_not_called()
    torch.testing.assert_close(
        wrapper.plan.call_args_list[0].args[2],
        torch.tensor([4, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        wrapper.plan.call_args_list[1].args[2],
        torch.tensor([2], dtype=torch.int32),
    )
    assert first_scope is not context.attention_validation_scope
    assert state.plan_scope is context.attention_validation_scope
    assert wrapper.run.call_count == 3


def test_flashinfer_fa3_plans_without_workspace_sizing():
    wrapper = SimpleNamespace(
        plan=Mock(),
        workspace_size=Mock(side_effect=NotImplementedError),
        reset_workspace_buffer=Mock(),
    )
    state = object.__new__(_FlashInferPagedPrefillState)
    state.backend = "fa3"
    state.wrapper = wrapper
    state.workspace = torch.empty(16, dtype=torch.uint8)
    state.int_workspace = None
    state.plan_scope = None

    state.plan(
        _spec(),
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
        active_slots=torch.tensor([[4, 1]], dtype=torch.int32),
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([2], dtype=torch.int32),
        max_context_len=2,
        plan_scope=object(),
    )

    wrapper.workspace_size.assert_not_called()
    wrapper.reset_workspace_buffer.assert_not_called()
    wrapper.plan.assert_called_once()


def test_flashinfer_prefill_grows_workspace_before_planning():
    wrapper = SimpleNamespace(
        plan=Mock(),
        workspace_size=Mock(side_effect=[(64, 32), (48, 16)]),
        reset_workspace_buffer=Mock(),
    )
    state = object.__new__(_FlashInferPagedPrefillState)
    state.backend = "fa2"
    state.wrapper = wrapper
    state.workspace = torch.empty(32, dtype=torch.uint8)
    state.int_workspace = torch.empty(16, dtype=torch.uint8)
    state.plan_scope = None
    spec = _spec()
    kwargs = {
        "qo_indptr": torch.tensor([0, 1], dtype=torch.int32),
        "active_slots": torch.tensor([[4, 1]], dtype=torch.int32),
        "req_indices": torch.tensor([0], dtype=torch.int32),
        "context_lens": torch.tensor([2], dtype=torch.int32),
        "max_context_len": 2,
    }

    state.plan(spec, plan_scope=object(), **kwargs)

    assert state.workspace.numel() == 64
    assert state.int_workspace.numel() == 32
    wrapper.reset_workspace_buffer.assert_called_once_with(
        state.workspace,
        state.int_workspace,
    )

    state.plan(spec, plan_scope=object(), **kwargs)

    assert state.workspace.numel() == 64
    assert state.int_workspace.numel() == 32
    assert wrapper.reset_workspace_buffer.call_count == 1


def test_sgl_fa3_prefill_passes_token_page_view_without_copying():
    kernel = Mock()
    kernel.run_explicit_varlen.side_effect = lambda *args, **_kwargs: args[6]
    provider = SglFa3PagedPrefillAttentionProvider()
    provider._kernel = kernel
    q = torch.empty(5, 32, 128, dtype=torch.bfloat16)
    k_cache = torch.empty(23, 4, 128, dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    active_slots = torch.tensor(
        [[7, 3, 11, 1, 9, 0], [6, 4, 10, 2, 8, 5]], dtype=torch.int32
    )
    req_indices = torch.tensor([1, 0], dtype=torch.int32)
    context_lens = torch.tensor([6, 5], dtype=torch.int32)
    view = SimpleNamespace(
        k_cache=k_cache,
        v_cache=v_cache,
        active_slots=active_slots,
        req_indices=req_indices,
        context_lens=context_lens,
        attn_score=None,
    )
    qo_indptr = torch.tensor([0, 3, 5], dtype=torch.int32)

    actual = provider.run(
        _spec(num_query_heads=32, num_kv_heads=4),
        q,
        view,
        qo_indptr=qo_indptr,
        chunk_lens=torch.tensor([3, 2], dtype=torch.int32),
        max_context_len=6,
        layer_idx=0,
    )

    call = kernel.run_explicit_varlen.call_args
    assert call.args[0].data_ptr() == q.data_ptr()
    assert call.args[1].data_ptr() == k_cache.data_ptr()
    assert call.args[2].data_ptr() == v_cache.data_ptr()
    assert call.args[3].data_ptr() == active_slots.data_ptr()
    assert call.args[4].data_ptr() == req_indices.data_ptr()
    assert call.args[5].data_ptr() == context_lens.data_ptr()
    assert call.kwargs["cu_seqlens_q"].data_ptr() == qo_indptr.data_ptr()
    assert call.kwargs["max_seqlen_q"] == 3
    assert actual.shape == q.shape


def test_attention_forward_runs_sgl_before_posthoc_cache_manager_hooks():
    events = []
    kernel = Mock()

    def run_sgl(*args, **_kwargs):
        events.append("sgl")
        return args[6]

    kernel.run_explicit_varlen.side_effect = run_sgl
    provider = SglFa3PagedPrefillAttentionProvider()
    provider._kernel = kernel
    prepared = PreparedPrefillAttentionOp(
        _spec(num_query_heads=32, num_kv_heads=4), provider
    )
    q = torch.empty(2, 32, 128, dtype=torch.bfloat16)
    k_cache = torch.empty(7, 4, 128, dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    view = PrefillComputeView(
        payload=ExplicitKVPayload(k_cache=k_cache, v_cache=v_cache),
        meta=AttentionViewMeta(
            active_slots=torch.tensor([[4, 1]], dtype=torch.int32),
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([2], dtype=torch.int32),
            max_context_len=2,
        ),
    )
    cache_manager = SimpleNamespace(
        before_prefill_layer_attention=Mock(),
        build_prefill_compute_view=Mock(return_value=view),
        collect_prefill_attention_score=Mock(
            side_effect=lambda *_args, **_kwargs: events.append("collect_score")
        ),
        record_prefill_query=Mock(
            side_effect=lambda *_args, **_kwargs: events.append("record_query")
        ),
        on_layer_attention_end=Mock(),
    )
    sparse_controller = SimpleNamespace(
        get_prefill_selection=Mock(return_value=object()),
        on_layer_attention_end=Mock(),
    )
    context = SimpleNamespace(
        is_prefill=True,
        cache_manager=cache_manager,
        sparse_controller=sparse_controller,
        now_layer_idx=0,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        attention_validation_scope=object(),
    )
    attention = Attention(32, 128, 128**-0.5, 4, prefill_op=prepared)
    attention.attention_backend = SimpleNamespace(
        maybe_run_fake_prefill=Mock(return_value=None),
        debug_check_prefill_bounds=Mock(),
    )

    with (
        patch("sparseengine.layers.attention.get_context", return_value=context),
        patch("sparseengine.utils.context.get_context", return_value=context),
    ):
        actual = attention(q, torch.empty_like(k_cache[:2]), torch.empty_like(v_cache[:2]))

    assert actual.shape == q.shape
    kernel.run_explicit_varlen.assert_called_once()
    cache_manager.collect_prefill_attention_score.assert_called_once()
    cache_manager.record_prefill_query.assert_called_once()
    assert events == ["sgl", "collect_score", "record_query"]


def _torch_prefill_oracle(q, logical_k, logical_v, q_lens, kv_lens):
    outputs = []
    q_cursor = 0
    for q_len, kv_len, k, v in zip(q_lens, kv_lens, logical_k, logical_v):
        q_seq = q[q_cursor : q_cursor + q_len].transpose(0, 1).float()
        q_cursor += q_len
        query_groups = q.shape[1] // k.shape[1]
        k = k.transpose(0, 1).float().repeat_interleave(query_groups, dim=0)
        v = v.transpose(0, 1).float().repeat_interleave(query_groups, dim=0)
        q_positions = kv_len - q_len + torch.arange(q_len, device=q.device)
        k_positions = torch.arange(kv_len, device=q.device)
        allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        scores = torch.matmul(q_seq, k.transpose(-1, -2)) * (q.shape[2] ** -0.5)
        scores.masked_fill_(~allowed.unsqueeze(0), -torch.inf)
        output = torch.matmul(torch.softmax(scores, dim=-1), v)
        outputs.append(output.transpose(0, 1).to(torch.bfloat16))
    return torch.cat(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashinfer_page_size_one_matches_noncontiguous_torch_oracle():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("The specialized provider requires SM90.")
    pytest.importorskip("flashinfer")
    torch.manual_seed(20260809)
    q_lens = [3, 2]
    kv_lens = [5, 6]
    q = torch.randn(5, 12, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(23, 2, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    pages = torch.randperm(23, device="cuda")[:11]
    rows = torch.zeros(2, 6, device="cuda", dtype=torch.int32)
    rows[0, :5] = pages[:5].to(torch.int32)
    rows[1, :6] = pages[5:].to(torch.int32)
    logical_k = [k_cache[pages[:5]], k_cache[pages[5:]]]
    logical_v = [v_cache[pages[:5]], v_cache[pages[5:]]]
    view = SimpleNamespace(
        k_cache=k_cache,
        v_cache=v_cache,
        active_slots=rows,
        req_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        context_lens=torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
        attn_score=None,
    )
    provider = FlashInferPagedPrefillAttentionProvider()
    spec = _spec()
    provider.prepare(spec)
    actual = provider.run(
        spec,
        q,
        view,
        qo_indptr=torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32),
        chunk_lens=torch.tensor(q_lens, device="cuda", dtype=torch.int32),
        max_context_len=6,
        layer_idx=0,
    )
    expected = _torch_prefill_oracle(
        q, logical_k, logical_v, q_lens, kv_lens
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
    provider.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads", "head_dim"),
    [
        (32, 8, 64),
        (32, 4, 128),
        (24, 4, 256),
    ],
)
@pytest.mark.parametrize("sparse_method", ["", "omnikv", "quest"])
def test_flashinfer_fa2_sm120_matches_noncontiguous_torch_oracle(
    num_query_heads,
    num_kv_heads,
    head_dim,
    sparse_method,
):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("The specialized provider requires SM120.")
    pytest.importorskip("flashinfer")
    torch.manual_seed(20260822)
    q_lens = [3, 2]
    kv_lens = [5, 6]
    q = torch.randn(
        5, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        23, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    pages = torch.randperm(23, device="cuda")[:11]
    rows = torch.zeros(2, 6, device="cuda", dtype=torch.int32)
    rows[0, :5] = pages[:5].to(torch.int32)
    rows[1, :6] = pages[5:].to(torch.int32)
    logical_k = [k_cache[pages[:5]], k_cache[pages[5:]]]
    logical_v = [v_cache[pages[:5]], v_cache[pages[5:]]]
    view = SimpleNamespace(
        k_cache=k_cache,
        v_cache=v_cache,
        active_slots=rows,
        req_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        context_lens=torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
        attn_score=None,
    )
    spec = build_mha_prefill_attention_spec(
        SimpleNamespace(
            dtype=torch.bfloat16,
            hidden_size=num_query_heads * head_dim,
            num_attention_heads=num_query_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        ),
        sparse_method=sparse_method,
        attention_tp_size=1,
        runtime_config=SimpleNamespace(
            prefill_sparse_method="",
            sparse_prefill_score_mode=None,
            h2o_prefill_score_window=0,
        ),
    )
    resolved = OpResolver(PREFILL_ATTENTION_REGISTRY).resolve(
        spec,
        _sm120_caps(),
    )
    provider = resolved.provider
    assert provider.name == "flashinfer_paged_prefill_fa2_sm120"
    provider.prepare(spec)
    actual = provider.run(
        spec,
        q,
        view,
        qo_indptr=torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32),
        chunk_lens=torch.tensor(q_lens, device="cuda", dtype=torch.int32),
        max_context_len=6,
        layer_idx=0,
    )
    expected = _torch_prefill_oracle(q, logical_k, logical_v, q_lens, kv_lens)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
    provider.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("query_len", [64, 256])
def test_flashinfer_fa2_sm120_grows_workspace_for_short_q_long_prefix(query_len):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("The specialized provider requires SM120.")
    pytest.importorskip("flashinfer")
    context_len = 16_384
    spec = _spec(num_query_heads=32, num_kv_heads=4)
    q = torch.zeros(
        query_len, 32, 128, device="cuda", dtype=torch.bfloat16
    )
    k_cache = torch.zeros(
        context_len, 4, 128, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.zeros_like(k_cache)
    view = SimpleNamespace(
        k_cache=k_cache,
        v_cache=v_cache,
        active_slots=torch.arange(
            context_len, device="cuda", dtype=torch.int32
        ).unsqueeze(0),
        req_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        context_lens=torch.tensor(
            [context_len], device="cuda", dtype=torch.int32
        ),
        attn_score=None,
    )
    provider = FlashInferFa2Sm120PagedPrefillAttentionProvider()
    provider.prepare(spec)
    assert provider._state is not None
    initial_workspace_bytes = provider._state.workspace.numel()

    try:
        actual = provider.run(
            spec,
            q,
            view,
            qo_indptr=torch.tensor(
                [0, query_len], device="cuda", dtype=torch.int32
            ),
            chunk_lens=torch.tensor(
                [query_len], device="cuda", dtype=torch.int32
            ),
            max_context_len=context_len,
            layer_idx=0,
        )

        assert provider._state.workspace.numel() > initial_workspace_bytes
        assert torch.count_nonzero(actual).item() == 0
    finally:
        provider.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_page_size_one_matches_noncontiguous_torch_oracle():
    torch.manual_seed(20260809)
    q_lens = [3, 2]
    kv_lens = [5, 6]
    q = torch.randn(5, 12, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(23, 2, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    pages = torch.randperm(23, device="cuda")[:11]
    rows = torch.zeros(2, 6, device="cuda", dtype=torch.int32)
    rows[0, :5] = pages[:5].to(torch.int32)
    rows[1, :6] = pages[5:].to(torch.int32)
    logical_k = [k_cache[pages[:5]], k_cache[pages[5:]]]
    logical_v = [v_cache[pages[:5]], v_cache[pages[5:]]]
    view = SimpleNamespace(
        k_cache=k_cache,
        v_cache=v_cache,
        active_slots=rows,
        req_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        context_lens=torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
        attn_score=None,
    )
    provider = TritonPagedPrefillAttentionProvider()

    actual = provider.run(
        _spec(),
        q,
        view,
        qo_indptr=torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32),
        chunk_lens=torch.tensor(q_lens, device="cuda", dtype=torch.int32),
        max_context_len=6,
        layer_idx=0,
    )
    expected = _torch_prefill_oracle(
        q,
        logical_k,
        logical_v,
        q_lens,
        kv_lens,
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
