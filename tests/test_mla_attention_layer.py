from __future__ import annotations

import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.engine.cache_manager import (
    AttentionKeyComputeView,
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
    MlaLatentPayload,
    PrefillComputeView,
)
from sparseengine.engine.cache_manager.base import CacheManager
from sparseengine.layers.mla_attention import (
    MLAAttention,
)
from sparseengine.operators.mla_attention import (
    MlaAttentionOpSpec,
    MlaAttentionProvider,
)
from sparseengine.utils.context import get_context, reset_context, set_context


class _TestProvider(MlaAttentionProvider):
    name = "test"

    def __init__(
        self,
        spec: MlaAttentionOpSpec,
        *,
        device: torch.device | str,
        max_batch_size: int,
    ) -> None:
        self.spec = spec
        self.device = torch.device(device)
        self.max_batch_size = int(max_batch_size)


def _spec(tp_size: int = 4) -> MlaAttentionOpSpec:
    return MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=tp_size,
        cuda_graph=False,
    )


def _attention(
    *,
    device: torch.device | str = "cpu",
    tp_size: int = 4,
    max_batch_size: int = 4,
    budget: int = 64 * 1024 * 1024,
) -> MLAAttention:
    spec = _spec(tp_size)
    resolved_device = torch.device("cuda:0") if str(device) == "cuda" else device
    return MLAAttention(
        spec=spec,
        provider=_TestProvider(
            spec,
            device=resolved_device,
            max_batch_size=max_batch_size,
        ),
        prefill_workspace_bytes=budget,
        hidden_size=64,
        projection_chunk_size=8,
    )


def _view(
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
) -> PrefillComputeView:
    return PrefillComputeView(
        meta=AttentionViewMeta(
            active_slots=active_slots,
            req_indices=request_indices,
            context_lens=context_lens,
            max_context_len=int(context_lens.max().item()),
        ),
        payload=MlaLatentPayload(
            latent_cache=latent_cache,
            rope_cache=rope_cache,
        ),
    )


def test_mla_binds_key_materializer_once_per_manager_and_layer() -> None:
    attention = _attention()
    project_latent = Mock()
    first_manager = SimpleNamespace(register_attention_key_materializer=Mock())
    second_manager = SimpleNamespace(register_attention_key_materializer=Mock())

    attention._ensure_key_materializer(first_manager, 0, project_latent)
    attention._ensure_key_materializer(first_manager, 0, project_latent)
    attention._ensure_key_materializer(first_manager, 1, project_latent)
    attention._ensure_key_materializer(second_manager, 0, project_latent)

    assert first_manager.register_attention_key_materializer.call_count == 2
    second_manager.register_attention_key_materializer.assert_called_once()


def test_mla_restores_original_materializer_after_history_profiling() -> None:
    # Registration mocks cannot catch the duplicate callback rejected when
    # startup returns to its original manager after a synthetic-history probe.
    class Manager:
        register_attention_key_materializer = (
            CacheManager.register_attention_key_materializer
        )

        def kv_layer_index(self, layer_idx):
            return layer_idx

    attention = _attention()
    project_latent = Mock()
    original = Manager()
    probe = Manager()
    probe_ref = weakref.ref(probe)

    for layer_idx in (0, 1):
        attention._ensure_key_materializer(original, layer_idx, project_latent)
    original_callbacks = original._attention_key_materializers.copy()
    for layer_idx in (0, 1):
        attention._ensure_key_materializer(probe, layer_idx, project_latent)
    attention.release_cache_runtime_bindings(probe)
    del probe
    assert probe_ref() is None

    for layer_idx, callback in original_callbacks.items():
        attention._ensure_key_materializer(original, layer_idx, project_latent)
        assert original._attention_key_materializers[layer_idx] is callback
        with pytest.raises(RuntimeError, match="already bound"):
            original.register_attention_key_materializer(layer_idx, Mock())


def test_mla_releases_only_bindings_for_the_retiring_cache_runtime() -> None:
    attention = _attention()
    project_latent = Mock()
    first_manager = SimpleNamespace(register_attention_key_materializer=Mock())
    second_manager = SimpleNamespace(register_attention_key_materializer=Mock())

    attention._ensure_key_materializer(first_manager, 0, project_latent)
    attention._ensure_key_materializer(first_manager, 1, project_latent)
    attention._ensure_key_materializer(second_manager, 0, project_latent)
    attention.release_cache_runtime_bindings(first_manager)
    attention.release_cache_runtime_bindings(first_manager)

    attention._ensure_key_materializer(second_manager, 0, project_latent)
    attention._ensure_key_materializer(first_manager, 1, project_latent)
    second_manager.register_attention_key_materializer.assert_called_once()
    assert first_manager.register_attention_key_materializer.call_count == 3


def test_mla_runtime_release_drops_cached_history_mapping() -> None:
    # The startup history cache must not survive through the shared MLA plan
    # after its cache manager is retired.
    attention = _attention()
    meta = AttentionViewMeta(
        active_slots=torch.tensor([[0, 1]], dtype=torch.int32),
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([2], dtype=torch.int32),
    )
    mapping = weakref.ref(meta.active_slots)
    view = PrefillComputeView(
        meta, MlaLatentPayload(torch.empty(2, 1, 512), torch.empty(2, 1, 64))
    )
    attention.chunked_prefill.prepare(
        view, torch.tensor([0, 1], dtype=torch.int32), object()
    )
    del view
    del meta
    assert mapping() is not None
    attention.release_cache_runtime_bindings(object())
    assert mapping() is None


def test_mla_prefill_rejects_wrong_payload_before_gather() -> None:
    attention = _attention()
    view = PrefillComputeView(
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

    with pytest.raises(TypeError, match="MlaLatentPayload"):
        attention._require_mla_payload(view, operation="chunked prefill")


def test_mla_attention_bind_resolves_provider_once() -> None:
    spec = _spec()
    provider = _TestProvider(spec, device="cpu", max_batch_size=8)

    with patch(
        "sparseengine.layers.mla_attention.resolve_mla_attention_provider",
        return_value=provider,
    ) as resolve:
        attention = MLAAttention.bind(
            spec=spec,
            device="cpu",
            max_batch_size=8,
            prefill_workspace_bytes=1024,
            hidden_size=64,
            projection_chunk_size=8,
        )

    assert attention.provider is provider
    resolve.assert_called_once_with(
        spec,
        device="cpu",
        max_batch_size=8,
    )


def test_set_context_starts_a_new_attention_validation_scope() -> None:
    reset_context()
    initial_scope = get_context().attention_validation_scope
    set_context(False)
    first_step_scope = get_context().attention_validation_scope
    set_context(False)
    second_step_scope = get_context().attention_validation_scope

    assert first_step_scope is not initial_scope
    assert second_step_scope is not first_step_scope


def test_mla_decode_passes_valid_batch_size_to_provider() -> None:
    class _RecordingProvider(_TestProvider):
        def run(
            self,
            q_nope_absorbed,
            q_rope,
            view,
            output,
            *,
            validation_scope=None,
            valid_batch_size=None,
        ):
            self.valid_batch_size = valid_batch_size
            self.output_is_contiguous = output.is_contiguous()
            return output

    spec = _spec(tp_size=4)
    provider = _RecordingProvider(spec, device="cpu", max_batch_size=4)
    attention = MLAAttention(
        spec=spec,
        provider=provider,
        prefill_workspace_bytes=64 * 1024 * 1024,
        hidden_size=64,
        projection_chunk_size=8,
    )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.arange(16, dtype=torch.int32).view(4, 4),
            req_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
            context_lens=torch.tensor([4, 4, 4, 4], dtype=torch.int32),
        ),
        payload=MlaLatentPayload(
            latent_cache=torch.empty(16, 1, 512, dtype=torch.bfloat16),
            rope_cache=torch.empty(16, 1, 64, dtype=torch.bfloat16),
        ),
    )
    q_nope_absorbed = torch.empty(5, 4, 512, dtype=torch.bfloat16).transpose(0, 1)
    q_rope = torch.empty(5, 4, 64, dtype=torch.bfloat16).transpose(0, 1)

    set_context(False, seqs=[object(), object(), object()])
    try:
        output = attention.run_decode(q_nope_absorbed, q_rope, view)
    finally:
        reset_context()

    assert output.shape == q_nope_absorbed.shape
    assert not q_nope_absorbed.is_contiguous()
    assert output.is_contiguous()
    assert provider.output_is_contiguous
    assert provider.valid_batch_size == 3


def test_mla_materializes_actual_keys_for_permuted_slots() -> None:
    torch.manual_seed(47)
    attention = _attention(tp_size=4)
    latent_cache = torch.randn(7, 1, 512, dtype=torch.bfloat16)
    rope_cache = torch.randn(7, 1, 64, dtype=torch.bfloat16)
    slots = torch.tensor([[5, 1], [6, 2]], dtype=torch.int32)
    view = AttentionKeyComputeView(
        active_slots=slots,
        payload=MlaLatentPayload(
            latent_cache=latent_cache,
            rope_cache=rope_cache,
        ),
    )
    weights = torch.randn(
        attention.spec.local_q_heads,
        448,
        512,
        dtype=torch.bfloat16,
    )

    def project_latent(latent: torch.Tensor) -> torch.Tensor:
        return (
            torch.einsum(
                "tr,hor->tho",
                latent.float(),
                weights.float(),
            )
            .to(torch.bfloat16)
            .flatten(1)
        )

    actual = attention.materialize_expanded_keys(
        view,
        project_latent=project_latent,
    )

    flat_slots = slots.long().flatten()
    latent = latent_cache[flat_slots, 0]
    projected = torch.einsum(
        "tr,hor->tho",
        latent.float(),
        weights.float(),
    ).to(torch.bfloat16)
    expected = torch.cat(
        (
            projected[..., :192],
            rope_cache[flat_slots, 0][:, None, :].expand(
                -1,
                attention.spec.local_q_heads,
                -1,
            ),
        ),
        dim=-1,
    ).view(2, 2, attention.spec.local_q_heads, 256)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("split_scratch", [0, 2 * 1024 * 1024])
def test_chunked_prefill_budget_fails_before_projection(split_scratch):
    # A deliberately small cap must reject before allocating expanded KV.
    attention = _attention(budget=1024 * 1024 if split_scratch else 1)
    # The base tensors fit 1 MiB; provider scratch alone must trigger rejection.
    attention.chunked_prefill._partial_workspace = lambda **shape: split_scratch
    view = _view(
        torch.empty(2, 1, 512, dtype=torch.bfloat16),
        torch.empty(2, 1, 64, dtype=torch.bfloat16),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    manager = SimpleNamespace(
        register_attention_key_materializer=Mock(),
        store_attention_payload=Mock(return_value=None),
        on_kv_stored=Mock(),
        before_prefill_layer_attention=Mock(),
        build_prefill_compute_view=Mock(return_value=view),
        prefill_score_request=Mock(return_value=None),
    )
    set_context(True, torch.tensor([0, 1], dtype=torch.int32), manager, seqs=[])
    get_context().sparse_controller = SimpleNamespace(
        get_prefill_selection=Mock(return_value=None)
    )
    project = Mock()
    try:
        with pytest.raises(MemoryError, match="workspace exceeds budget"):
            attention.run_cached_attention(
                torch.empty(1, 5, 256),
                torch.empty(1, 5, 192),
                torch.empty(1, 5, 64),
                torch.empty(1, 512),
                torch.empty(1, 64),
                project_latent=project,
                absorb_query=Mock(),
                reconstruct_values=Mock(),
            )
        project.assert_not_called()
    finally:
        reset_context()


@pytest.mark.parametrize('budget', [1, 1024 * 1024])
@pytest.mark.parametrize('score_mode', [None, 'logits', 'probability'])
def test_compressed_prefill_preserves_prefill_hooks_and_budget(budget, score_mode):
    from sparseengine.operators.mla_attention import MlaSglFa3Provider
    from sparseengine.engine.cache_manager.base import PrefillScoreRequest
    attention = _attention(budget=budget)
    attention._use_compressed_prefill = MlaSglFa3Provider.use_compressed_prefill.__get__(attention.provider)
    view = _view(torch.empty(2, 1, 512, dtype=torch.bfloat16),
                 torch.empty(2, 1, 64, dtype=torch.bfloat16),
                 torch.tensor([[1, 0]], dtype=torch.int32),
                 torch.tensor([0], dtype=torch.int32),
                 torch.tensor([2], dtype=torch.int32))
    manager = SimpleNamespace(**{name: Mock() for name in (
        'register_attention_key_materializer', 'store_attention_payload',
        'on_kv_stored', 'before_prefill_layer_attention',
        'collect_prefill_attention_score', 'record_prefill_query',
        'record_decode_query', 'on_layer_attention_end')})
    manager.build_prefill_compute_view = Mock(return_value=view)
    request = PrefillScoreRequest(((1, 2),), score_mode) if score_mode else None
    manager.prefill_score_request = Mock(return_value=request)
    scores = torch.ones(1, 2) if score_mode else None
    attention.chunked_prefill.score_compressed = Mock(return_value=scores)
    controller = SimpleNamespace(get_prefill_selection=Mock(return_value=None), on_layer_attention_end=Mock())
    set_context(True, torch.tensor([0, 1], dtype=torch.int32), manager, seqs=[])
    get_context().sparse_controller = controller
    output = torch.ones(1, 5, 256, dtype=torch.bfloat16)
    attention.provider.run_compressed_prefill = Mock(return_value=(
        torch.ones(1, 5, 512, dtype=torch.bfloat16), torch.zeros(5, 1)))
    project = Mock(side_effect=AssertionError('compressed prefill projected history'))
    def run():
        return attention.run_cached_attention(
            torch.empty_like(output), torch.empty(1, 5, 192), torch.empty(1, 5, 64),
            torch.empty(1, 512), torch.empty(1, 64), project_latent=project,
            absorb_query=Mock(return_value=torch.ones(1, 5, 512)),
            reconstruct_values=Mock(return_value=output))
    try:
        if budget == 1:
            with pytest.raises(MemoryError, match='compressed prefill workspace'):
                run()
            attention.provider.run_compressed_prefill.assert_not_called()
            attention.chunked_prefill.score_compressed.assert_not_called()
        else:
            assert run() is output
            assert get_context().is_prefill
            assert not attention.chunked_prefill.plan.history_prepared
            manager.record_prefill_query.assert_called_once()
            manager.collect_prefill_attention_score.assert_called_once()
            assert manager.collect_prefill_attention_score.call_args.args[2].token_scores is scores
            assert attention.chunked_prefill.score_compressed.call_args.args[4] is request
            manager.record_decode_query.assert_not_called()
            manager.on_layer_attention_end.assert_called_once()
            controller.on_layer_attention_end.assert_called_once()
        project.assert_not_called()
    finally:
        reset_context()
