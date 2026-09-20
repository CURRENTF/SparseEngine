from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.cache_manager.base import (
    AttentionViewMeta, CacheManager, MlaLatentWrite, PrefillComputeView,
)
from sparseengine.engine.cache_manager.standard import StandardCacheManager
from sparseengine.engine.cache_manager.storage import MlaLatentStorage


@pytest.mark.parametrize("changed", [None, "rows", "lengths", "slots", "temporary", "dtype", "storage"])
def test_only_untransformed_physical_current_tokens_are_reused(monkeypatch, changed):
    storage = MlaLatentStorage(kv_lora_rank=512, rope_dim=64, dtype=torch.bfloat16)
    storage.allocate(num_layers=1, num_slots=10, device=torch.device("cpu"))
    slots = torch.arange(10, dtype=torch.int32).reshape(2, 5)
    rows = torch.tensor([1, 0], dtype=torch.int32)
    lengths = torch.tensor([5, 3], dtype=torch.int32)
    meta = AttentionViewMeta(slots, rows, lengths)
    payload = storage.layer_payload(0)
    latent = torch.randn(3, 512, dtype=torch.bfloat16)
    rope = torch.randn(3, 64, dtype=torch.bfloat16)
    if changed == "rows":
        meta = replace(meta, req_indices=rows.clone())
    elif changed == "lengths":
        meta = replace(meta, context_lens=lengths.clone())
    elif changed == "slots":
        meta = replace(meta, active_slots=slots.clone())
    elif changed == "temporary":
        meta = replace(meta, temp_slots=torch.tensor([0]))
    elif changed == "dtype":
        latent = latent.float()
    elif changed == "storage":
        payload = replace(payload, latent_cache=payload.latent_cache.clone())
    view = PrefillComputeView(meta, payload)
    monkeypatch.setattr(CacheManager, "build_prefill_compute_view", lambda *a: view)
    manager = object.__new__(StandardCacheManager)
    manager.attention_cache_storage = storage
    manager.buffer_req_to_token_slots = slots
    manager.layer_batch_state = SimpleNamespace(req_indices=rows, context_lens=lengths)
    manager.kv_layer_index = lambda layer: layer
    result = manager.build_prefill_compute_view(0, latent, rope, None)
    if changed is not None:
        assert result.current_mla is None
    else:
        assert result.current_mla.latent.data_ptr() == latent.data_ptr()
        assert result.current_mla.rope.data_ptr() == rope.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_current_reuse_preserves_ragged_attention_and_gathers_only_history():
    from test_mla_chunked_prefill import make_case, partial_provider
    from sparseengine.operators.mla_prefill import ChunkedMlaPrefill
    spec, q, view, cu, project, absorb = make_case()
    runner = ChunkedMlaPrefill(spec, partial_provider("fa3", spec, q.device, 3), 17)
    scope = object()
    plan = runner.prepare(view, cu, scope)
    original = runner.run(q, view, cu, scope, project, absorb)
    latent, rope = runner.gather(view.payload, plan.current_slots)
    current = MlaLatentWrite(latent[:, None], rope[:, None])
    calls = []
    gather = runner.gather

    def record(payload, slots):
        calls.append(slots.numel())
        return gather(payload, slots)

    runner.gather = record
    actual = runner.run(q, replace(view, current_mla=current), cu, scope, project, absorb)
    assert len(calls) == len(plan.history_chunks)
    assert sum(calls) == sum(plan.contexts) - q.shape[0]
    torch.testing.assert_close(actual[0], original[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], original[1], atol=0, rtol=0)
