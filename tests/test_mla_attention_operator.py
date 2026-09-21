from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import pytest
import torch

from sparseengine.engine.cache_manager import (
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
    MlaLatentPayload,
)
from sparseengine.kernels.external.sgl.fa3 import sgl_fa3_device_support
from sparseengine.kernels.triton.mla import (
    MlaDecodeWorkspace,
)
from sparseengine.operators.mla_attention import (
    MLA_ATTENTION_REGISTRY,
    MlaAttentionOpSpec,
    MlaSglFa3Provider,
    MlaTritonProvider,
)
from sparseengine.operators.attention_capabilities import AttentionScoreKind
from sparseengine.operators.registry import OpResolver
from sparseengine.platforms import DeviceCaps, PlatformEnum


def _spec(**overrides) -> MlaAttentionOpSpec:
    values = {
        "num_q_heads": 20,
        "kv_lora_rank": 512,
        "rope_dim": 64,
        "qk_head_dim": 256,
        "value_head_dim": 256,
        "activation_dtype": torch.bfloat16,
        "cache_dtype": torch.bfloat16,
        "tp_size": 4,
        "cuda_graph": False,
        "context_capacity": 65536,
    }
    values.update(overrides)
    return MlaAttentionOpSpec(**values)


def _h100_caps(**overrides) -> DeviceCaps:
    values = {
        "platform": PlatformEnum.CUDA,
        "device_type": "cuda",
        "device_index": 0,
        "device_name": "NVIDIA H100 80GB HBM3",
        "compute_capability": (9, 0),
        "runtime_version": "12.9",
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


def _cpu_workspace(batch_size: int, head_count: int) -> MlaDecodeWorkspace:
    return MlaDecodeWorkspace(
        block_size=torch.empty(1, dtype=torch.int32),
        batch_start_indices=torch.empty(batch_size, dtype=torch.int32),
        mid_output=torch.empty(head_count, 1, 512, dtype=torch.float32),
        mid_logsumexp=torch.empty(head_count, 1, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_q_heads": 0},
        {"kv_lora_rank": 0},
        {"rope_dim": -1},
        {"qk_head_dim": 0},
        {"value_head_dim": 0},
        {"tp_size": 0},
        {"num_q_heads": 20, "tp_size": 3},
        {"context_capacity": 0},
    ],
)
def test_mla_attention_spec_rejects_invalid_dimensions(overrides) -> None:
    with pytest.raises(ValueError):
        _spec(**overrides)


def test_mla_attention_scale_uses_qk_head_dimension() -> None:
    spec = _spec()

    assert spec.softmax_scale == pytest.approx(256**-0.5)
    assert spec.softmax_scale != pytest.approx((512 + 64) ** -0.5)


def test_online_h2o_requests_an_executable_mla_score_contract():
    from sparseengine.method_registry import sparse_decode_attention_score_kind
    from sparseengine.operators.attention_capabilities import AttentionScoreKind

    kind = sparse_decode_attention_score_kind(
        "h2o", h2o_decode_eviction=True, attention_cache_layout="mla_latent",
    )
    spec = _spec(score_output=kind, tp_size=1)
    assert kind is AttentionScoreKind.RAW_QK_REDUCED
    assert MlaTritonProvider.supports(spec, _h100_caps()).supported
    # A score-free provider must not bind and silently drop the online scores.
    assert not MlaSglFa3Provider.supports(spec, _h100_caps()).supported


def test_mla_triton_atomic_support_is_not_narrowed_by_device_name() -> None:
    result = MlaTritonProvider.supports(
        _spec(tp_size=4),
        _h100_caps(device_name="unprofiled SM90 GPU"),
    )

    assert result.supported


def test_mla_triton_atomic_supports_sm120() -> None:
    result = MlaTritonProvider.supports(
        _spec(tp_size=1),
        _h100_caps(
            device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            compute_capability=(12, 0),
            runtime_version="13.0",
        ),
    )

    assert result.supported


def test_mla_resolver_selects_triton_on_sm120() -> None:
    spec = _spec(tp_size=1)
    caps = _h100_caps(
        device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        compute_capability=(12, 0),
        runtime_version="13.0",
    )
    workspace = _cpu_workspace(batch_size=1, head_count=20)
    with (
        patch(
            "sparseengine.operators.mla_attention.sgl_fa3_device_support",
            return_value=(False, "FA3 unsupported on SM120"),
        ),
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
    ):
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            spec,
            caps,
            op_spec=spec,
            device="cpu",
            max_batch_size=1,
        )

    assert type(resolved.provider) is MlaTritonProvider
    assert resolved.provider.name == "triton_mla"
    assert resolved.report.selection_basis == "semantic_fallback"


def test_decode_graph_mla_requires_static_capacity() -> None:
    spec = _spec(
        cuda_graph=True,
        context_capacity=None,
    )

    result = MlaTritonProvider.supports(spec, _h100_caps())

    assert not result.supported
    assert "static context capacity" in result.reason


def test_decode_graph_mla_launch_config_ignores_runtime_context() -> None:
    spec = _spec(
        tp_size=2,
        cuda_graph=True,
        context_capacity=32768,
    )
    workspace = _cpu_workspace(batch_size=32, head_count=10)
    with patch(
        "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ):
        provider = MlaTritonProvider(
            op_spec=spec,
            device="cpu",
            max_batch_size=32,
            sm_count=132,
        )
    launch_config = provider.launch_config
    with patch(
        "sparseengine.operators.mla_attention.select_glm_mla_decode_config",
        return_value=launch_config,
    ) as select:
        first = provider._launch_config_for(
            batch_size=32,
            max_context_len=1,
            active_slot_width=64,
        )
        second = provider._launch_config_for(
            batch_size=32,
            max_context_len=32000,
            active_slot_width=65536,
        )

    assert first is launch_config
    assert second is launch_config
    select.assert_not_called()


def test_mla_binding_rejects_missing_sm_before_allocation() -> None:
    spec = _spec(tp_size=2)
    workspace = _cpu_workspace(batch_size=8, head_count=10)
    with patch(
        "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ) as allocate:
        with pytest.raises(ValueError, match="positive SM count"):
            MlaTritonProvider.bind(
                spec, _h100_caps(multi_processor_count=None),
                op_spec=spec, device="cpu", max_batch_size=8,
            )
    allocate.assert_not_called()


def test_sgl_mla_accepts_graph_stable_score_free_contract() -> None:
    spec = _spec(
        cuda_graph=True,
        context_capacity=32768,
    )
    with patch(
        "sparseengine.operators.mla_attention.sgl_fa3_device_support",
        return_value=(True, "sgl test"),
    ):
        result = MlaSglFa3Provider.supports(spec, _h100_caps())

    assert result.supported
    assert MlaSglFa3Provider.supports_decode_graph


def test_decode_graph_mla_resolver_prefers_sgl_fa3() -> None:
    spec = _spec(
        cuda_graph=True,
        context_capacity=32768,
    )
    workspace = _cpu_workspace(batch_size=8, head_count=5)
    with (
        patch(
            "sparseengine.operators.mla_attention.sgl_fa3_device_support",
            return_value=(True, "sgl test"),
        ),
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
        patch("sparseengine.operators.mla_attention.SglFa3DecodeKernel"),
    ):
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            spec,
            _h100_caps(),
            op_spec=spec,
            device="cpu",
            max_batch_size=8,
        )

    assert type(resolved.provider) is MlaSglFa3Provider
    assert resolved.report.selection_basis == "upstream_default"


@pytest.mark.parametrize(
    ("spec_overrides", "caps_overrides", "reason"),
    [
        ({}, {"platform": PlatformEnum.CPU}, "requires platform"),
        ({}, {"supports_triton": False}, "does not support Triton"),
        ({}, {"supports_bfloat16": False}, "does not support BF16"),
        (
            {"cuda_graph": True},
            {"supports_graph_capture": False},
            "graph capture support",
        ),
        ({"activation_dtype": torch.float16}, {}, "activation dtype"),
        ({"cache_dtype": torch.float16}, {}, "BF16 cache"),
        ({"kv_lora_rank": 256}, {}, "GLM MLA shape"),
        ({"tp_size": 5}, {}, "tensor parallel size"),
    ],
)
def test_mla_resolver_rejects_unvalidated_contracts(
    spec_overrides,
    caps_overrides,
    reason,
) -> None:
    with pytest.raises(RuntimeError, match=reason):
        OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            _spec(**spec_overrides),
            _h100_caps(**caps_overrides),
        )


def test_mla_provider_rejects_explicit_kv_before_kernel() -> None:
    spec = _spec(tp_size=1)
    workspace = _cpu_workspace(batch_size=1, head_count=20)
    with patch(
        "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ):
        provider = MlaTritonProvider(
            op_spec=spec,
            device="cpu",
            max_batch_size=1,
        )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor([[0]], dtype=torch.int32),
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([1], dtype=torch.int32),
        ),
        payload=ExplicitKVPayload(
            k_cache=torch.empty(1, 1, 256, dtype=torch.bfloat16),
            v_cache=torch.empty(1, 1, 256, dtype=torch.bfloat16),
        ),
    )

    with (
        patch("sparseengine.operators.mla_attention.run_mla_decode") as kernel,
        pytest.raises(TypeError, match="MlaLatentPayload"),
    ):
        provider.run(
            torch.empty(1, 20, 512, dtype=torch.bfloat16),
            torch.empty(1, 20, 64, dtype=torch.bfloat16),
            view,
            torch.empty(1, 20, 512, dtype=torch.bfloat16),
        )
    kernel.assert_not_called()


@pytest.mark.parametrize("causal", [False, True])
def test_sgl_provider_returns_chunk_output_and_lse(causal) -> None:
    spec = _spec(tp_size=4)
    workspace = _cpu_workspace(batch_size=1, head_count=5)
    fa3 = Mock()
    with (
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
        patch(
            "sparseengine.operators.mla_attention.SglFa3DecodeKernel",
            return_value=fa3,
        ),
    ):
        provider = MlaSglFa3Provider(
            op_spec=spec,
            device="cpu",
            max_batch_size=1,
        )
    q = torch.empty(2, 5, 256, dtype=torch.bfloat16)
    k = torch.empty(4, 5, 256, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 4], dtype=torch.int32)
    lse = torch.empty(5, 2, dtype=torch.float32)
    fa3.run_contiguous_explicit_varlen.side_effect = lambda q, k, v, out, **kw: (out, lse)

    output, actual_lse = provider.run_prefill_chunk(
        q, k, v, cu_seqlens_q, cu_seqlens_k, 2, 4, causal=causal
    )

    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert actual_lse is lse
    fa3.run_contiguous_explicit_varlen.assert_called_once_with(
        q,
        k,
        v,
        output,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=2,
        max_seqlen_k=4,
        causal=causal,
        return_softmax_lse=True,
    )
    fa3.run_explicit_varlen.assert_not_called()


def test_atomic_sgl_provider_rejects_late_score_request() -> None:
    provider = object.__new__(MlaSglFa3Provider)
    view = SimpleNamespace(meta=SimpleNamespace(attn_score=torch.empty(1)))

    with pytest.raises(RuntimeError, match="score-free operation"):
        provider.run(None, None, view, None)


@pytest.mark.parametrize(
    "score_output",
    [AttentionScoreKind.RAW_QK_PER_HEAD, AttentionScoreKind.RAW_QK_REDUCED],
)
def test_score_producing_triton_decode_keeps_independent_compressed_prefill(
    score_output,
) -> None:
    spec = _spec(score_output=score_output)
    workspace = _cpu_workspace(batch_size=4, head_count=5)
    with patch(
        "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ):
        provider = MlaTritonProvider(
            op_spec=spec,
            device="cpu",
            max_batch_size=4,
        )
    compressed = Mock()
    compressed.kernel_path = "triton_mla_prefill_latent"
    provider._compressed_prefill = compressed
    provider._compressed_prefill_support_reason = "test provider"
    plan = SimpleNamespace(
        query_starts=(0, 16, 32, 48, 64),
        cu_q=torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32),
        scope=object(),
        meta=SimpleNamespace(is_sparse=False),
    )
    assert provider.use_compressed_prefill(plan, None)

    q_latent = torch.empty(64, 5, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(64, 5, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)
    lse = torch.empty(5, 64)
    compressed.run.return_value = (output, lse)
    view = SimpleNamespace(
        payload=SimpleNamespace(
            rope_cache=torch.empty(1, 1, 64, dtype=torch.bfloat16),
            latent_cache=torch.empty(1, 1, 512, dtype=torch.bfloat16),
        ),
        meta=SimpleNamespace(
            active_slots=torch.zeros(4, 1, dtype=torch.int32),
            req_indices=torch.arange(4, dtype=torch.int32),
            context_lens=torch.ones(4, dtype=torch.int32),
        ),
    )
    actual, actual_lse = provider.run_compressed_prefill(
        q_latent, q_rope, view, plan,
    )

    assert actual is output
    assert actual_lse is lse
    compressed.run.assert_called_once_with(q_latent, q_rope, view, plan)


def test_mla_provider_run_does_not_resolve_or_allocate() -> None:
    spec = _spec(tp_size=4)
    workspace = _cpu_workspace(batch_size=2, head_count=5)
    with patch(
        "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ):
        provider = MlaTritonProvider(
            op_spec=spec,
            device="cpu",
            max_batch_size=2,
        )
    payload = MlaLatentPayload(
        latent_cache=torch.empty(4, 1, 512, dtype=torch.bfloat16),
        rope_cache=torch.empty(4, 1, 64, dtype=torch.bfloat16),
    )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            req_indices=torch.tensor([0, -1], dtype=torch.int32),
            context_lens=torch.tensor([2, 0], dtype=torch.int32),
        ),
        payload=payload,
    )
    q_nope_absorbed = torch.empty(2, 5, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 5, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_nope_absorbed)
    validation_scope = object()

    with (
        patch.object(OpResolver, "resolve") as resolve,
        patch(
            "sparseengine.operators.mla_attention.allocate_mla_decode_workspace"
        ) as allocate,
        patch(
            "sparseengine.operators.mla_attention.run_mla_decode",
            return_value=output,
        ) as kernel,
        patch(
            "sparseengine.operators.mla_attention.validate_mla_decode_metadata"
        ) as validate,
    ):
        actual = provider.run(
            q_nope_absorbed,
            q_rope,
            view,
            output,
            validation_scope=validation_scope,
        )
        provider.run(
            q_nope_absorbed,
            q_rope,
            view,
            output,
            validation_scope=validation_scope,
        )
        provider.run(
            q_nope_absorbed,
            q_rope,
            view,
            output,
            validation_scope=object(),
        )

    assert actual is output
    resolve.assert_not_called()
    allocate.assert_not_called()
    assert validate.call_count == 2
    validate.assert_called_with(
        view.meta.active_slots,
        view.meta.req_indices,
        view.meta.context_lens,
        cache_slot_count=4,
        max_context_len=None,
        valid_batch_size=None,
    )
    assert kernel.call_count == 3
    kernel.assert_called_with(
        q_nope_absorbed,
        q_rope,
        payload.latent_cache,
        payload.rope_cache,
        view.meta.active_slots,
        view.meta.req_indices,
        view.meta.context_lens,
        output,
        workspace,
        softmax_scale=spec.softmax_scale,
        attn_score=None,
        max_context_len=None,
        config=provider.launch_config,
        validate_metadata=False,
    )


def test_mla_provider_validates_each_metadata_identity_once_per_scope() -> None:
    spec = _spec(tp_size=4)
    workspace = _cpu_workspace(batch_size=1, head_count=5)
    with patch(
        "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ):
        provider = MlaTritonProvider(
            op_spec=spec,
            device="cpu",
            max_batch_size=1,
        )

    payload = MlaLatentPayload(
        latent_cache=torch.empty(4, 1, 512, dtype=torch.bfloat16),
        rope_cache=torch.empty(4, 1, 64, dtype=torch.bfloat16),
    )

    def view(slots: list[int]) -> DecodeComputeView:
        return DecodeComputeView(
            meta=AttentionViewMeta(
                active_slots=torch.tensor([slots], dtype=torch.int32),
                req_indices=torch.tensor([0], dtype=torch.int32),
                context_lens=torch.tensor([len(slots)], dtype=torch.int32),
            ),
            payload=payload,
        )

    view_a = view([0, 1])
    view_b = view([2, 3])
    q_nope_absorbed = torch.empty(1, 5, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(1, 5, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_nope_absorbed)
    validation_scope = object()

    with (
        patch(
            "sparseengine.operators.mla_attention.run_mla_decode",
            return_value=output,
        ),
        patch(
            "sparseengine.operators.mla_attention.validate_mla_decode_metadata"
        ) as validate,
    ):
        for decode_view in (view_a, view_b, view_a, view_b):
            provider.run(
                q_nope_absorbed,
                q_rope,
                decode_view,
                output,
                validation_scope=validation_scope,
            )

    assert validate.call_count == 2


def test_mla_provider_rejects_batch_larger_than_workspace() -> None:
    spec = _spec(tp_size=4)
    workspace = _cpu_workspace(batch_size=1, head_count=5)
    with patch(
        "sparseengine.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ):
        provider = MlaTritonProvider(
            op_spec=spec,
            device="cpu",
            max_batch_size=1,
        )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor([[0], [1]], dtype=torch.int32),
            req_indices=torch.tensor([0, 1], dtype=torch.int32),
            context_lens=torch.tensor([1, 1], dtype=torch.int32),
        ),
        payload=MlaLatentPayload(
            latent_cache=torch.empty(2, 1, 512, dtype=torch.bfloat16),
            rope_cache=torch.empty(2, 1, 64, dtype=torch.bfloat16),
        ),
    )

    with pytest.raises(ValueError, match="exceeds the bound workspace"):
        provider.run(
            torch.empty(2, 5, 512, dtype=torch.bfloat16),
            torch.empty(2, 5, 64, dtype=torch.bfloat16),
            view,
            torch.empty(2, 5, 512, dtype=torch.bfloat16),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the MLA provider integration test",
)
def test_mla_provider_runs_static_padded_batch() -> None:
    spec = _spec(tp_size=4)
    provider = MlaTritonProvider(
        op_spec=spec,
        device="cuda",
        max_batch_size=2,
    )
    q_nope_absorbed = torch.randn(
        2,
        5,
        512,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_rope = torch.randn(2, 5, 64, dtype=torch.bfloat16, device="cuda")
    payload = MlaLatentPayload(
        latent_cache=torch.randn(4, 1, 512, dtype=torch.bfloat16, device="cuda"),
        rope_cache=torch.randn(4, 1, 64, dtype=torch.bfloat16, device="cuda"),
    )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor(
                [[0, 2], [1, 3]],
                dtype=torch.int32,
                device="cuda",
            ),
            req_indices=torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
            context_lens=torch.tensor([2, 0], dtype=torch.int32, device="cuda"),
        ),
        payload=payload,
    )
    output = torch.empty_like(q_nope_absorbed)

    provider.run(q_nope_absorbed, q_rope, view, output)
    torch.cuda.synchronize()

    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    assert bool(torch.isfinite(output).all().item())


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not sgl_fa3_device_support(torch.cuda.current_device())[0],
    reason="CUDA and a validated sglang-kernel are required",
)
@torch.inference_mode()
def test_glm_production_provider_replays_across_1k_boundary() -> None:
    torch.manual_seed(20260825)
    device = torch.device("cuda")
    capacity = 1025
    spec = _spec(
        tp_size=1,
        cuda_graph=True,
        context_capacity=capacity,
    )
    provider = MlaSglFa3Provider(
        op_spec=spec,
        device=device,
        max_batch_size=1,
    )

    q_nope_absorbed = 0.125 * torch.randn(
        1,
        spec.local_q_heads,
        spec.kv_lora_rank,
        dtype=torch.bfloat16,
        device=device,
    )
    q_rope = 0.125 * torch.randn(
        1,
        spec.local_q_heads,
        spec.rope_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    payload = MlaLatentPayload(
        latent_cache=0.125
        * torch.randn(
            capacity,
            1,
            spec.kv_lora_rank,
            dtype=torch.bfloat16,
            device=device,
        ),
        rope_cache=0.125
        * torch.randn(
            capacity,
            1,
            spec.rope_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
    )
    active_slots = torch.arange(
        capacity,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    context_lens = torch.tensor([1023], dtype=torch.int32, device=device)
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=active_slots,
            req_indices=torch.zeros(1, dtype=torch.int32, device=device),
            context_lens=context_lens,
            max_context_len=capacity,
        ),
        payload=payload,
    )
    output = torch.empty_like(q_nope_absorbed)
    validation_scope = object()

    provider.run(
        q_nope_absorbed,
        q_rope,
        view,
        output,
        validation_scope=validation_scope,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        provider.run(
            q_nope_absorbed,
            q_rope,
            view,
            output,
            validation_scope=validation_scope,
        )

    static_ptrs = {
        "active_slots": active_slots.data_ptr(),
        "context_lens": context_lens.data_ptr(),
        "output": output.data_ptr(),
    }
    captured_plans = provider.fa3._captured_scheduler_plans
    assert len(captured_plans) == 1
    scheduler_metadata_ptr = captured_plans[0].metadata.data_ptr()

    for context_len in (1023, 1024, 1025):
        context_lens.fill_(context_len)
        graph.replay()
        torch.cuda.synchronize()

        latent = payload.latent_cache[:context_len, 0].float()
        rope = payload.rope_cache[:context_len, 0].float()
        logits = torch.einsum(
            "hd,ld->hl",
            q_nope_absorbed[0].float(),
            latent,
        ) + torch.einsum("hd,ld->hl", q_rope[0].float(), rope)
        probabilities = torch.softmax(logits * spec.softmax_scale, dim=-1)
        expected = torch.einsum("hl,ld->hd", probabilities, latent).to(torch.bfloat16)
        torch.testing.assert_close(output[0], expected, rtol=3e-2, atol=3e-2)

        assert active_slots.data_ptr() == static_ptrs["active_slots"]
        assert context_lens.data_ptr() == static_ptrs["context_lens"]
        assert output.data_ptr() == static_ptrs["output"]
        assert len(provider.fa3._captured_scheduler_plans) == 1
        assert (
            provider.fa3._captured_scheduler_plans[0].metadata.data_ptr()
            == scheduler_metadata_ptr
        )
