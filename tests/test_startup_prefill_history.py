from types import SimpleNamespace

import pytest
import torch

from glm_test_helpers import _glm_config, _single_rank_parallel_context
from sparseengine.engine.cache_manager.storage import (
    ExplicitKVStorage,
    HeterogeneousExplicitKVStorage,
    MlaLatentStorage,
)
from sparseengine.engine.startup import prefill_history
from sparseengine.platforms.cpu import CpuPlatform
from sparseengine.platforms.interface import AllocatorStats
from sparseengine.utils.context import get_context, set_context


@pytest.mark.parametrize("layout", ["explicit", "mla", "heterogeneous"])
def test_history_storage_reuses_layers_without_aliasing_token_slots(layout):
    # A full-depth history allocation would defeat the low-memory startup probe;
    # aliased token slots would instead corrupt the suffix/history separation.
    shapes = ((2, 16), (1, 32), (2, 16))
    if layout == "explicit":
        storage = ExplicitKVStorage(num_kv_heads=2, head_dim=16, dtype=torch.bfloat16)
    elif layout == "mla":
        storage = MlaLatentStorage(kv_lora_rank=512, rope_dim=64, dtype=torch.bfloat16)
    else:
        storage = HeterogeneousExplicitKVStorage(layer_shapes=shapes, dtype=torch.bfloat16)
    manager = object.__new__(prefill_history.PrefillHistoryCacheManager)
    manager.config = SimpleNamespace(num_kvcache_slots=32)
    manager.device = torch.device("cpu")
    manager.num_kv_layers = len(shapes)
    manager.attention_cache_storage = storage
    manager.allocate_kv_cache()

    first = storage.layer_payload(0)
    last = storage.layer_payload(2)
    first_tensor = first.latent_cache if layout == "mla" else first.k_cache
    last_tensor = last.latent_cache if layout == "mla" else last.k_cache
    assert first_tensor.data_ptr() == last_tensor.data_ptr()
    assert first_tensor[0].data_ptr() != first_tensor[1].data_ptr()
    first_tensor[-4:].fill_(3)
    assert torch.count_nonzero(last_tensor[:-4]) == 0
    assert torch.all(last_tensor[-4:] == 3)
    if layout == "heterogeneous":
        assert storage.layer_payload(1).k_cache.shape[1:] == shapes[1]
        assert storage.layer_payload(1).k_cache.data_ptr() != first_tensor.data_ptr()


@pytest.mark.parametrize(
    "token_budget, fail_forward, max_model_len",
    [(64, False, 128), (16, True, 128), (64, False, 3)],
)
def test_history_probe_runs_one_full_batch_and_restores_runtime(
    monkeypatch, token_budget, fail_forward, max_model_len,
):
    # Exercise the actual dense allocator/prepare path without GPU arithmetic.
    # Existing startup tests do not cover seeding history or restoring a probe
    # runtime after a model failure.
    class Platform(CpuPlatform):
        peak = 100

        def reset_peak_memory_stats(self, device=None):
            self.peak = 100

        def get_allocator_stats(self, device=None):
            return AllocatorStats(current_allocated_bytes=100, peak_allocated_bytes=self.peak)

    platform = Platform()
    monkeypatch.setattr("sparseengine.platforms._current_platform", platform)
    monkeypatch.setattr("sparseengine.engine.startup.profiling.release_unused_device_memory", lambda _: None)
    config = _glm_config(sparse_method="h2o", enable_prefix_caching=False)
    config.max_num_batched_tokens = token_budget
    config.max_model_len = max_model_len
    original_slots = config.num_kvcache_slots
    previous = object(), object(), object()
    model = SimpleNamespace(sparse_controller=previous[1], layers=[])
    released = []
    runner = SimpleNamespace(
        config=config,
        parallel_context=_single_rank_parallel_context(),
        recurrent_state_manager=None,
        platform=platform,
        device=torch.device("cpu"),
        cache_manager=previous[0],
        sparse_controller=previous[1],
        runtime_state=previous[2],
        model=SimpleNamespace(model=model, release_cache_runtime_bindings=lambda manager: released.append(manager)),
    )
    calls = []

    def run(seqs, is_prefill):
        manager = runner.cache_manager
        assert manager.config is not config
        assert manager.config.sparse_method == ""
        assert model.sparse_controller is runner.sparse_controller
        assert is_prefill
        input_ids, positions, cu_seqlens = runner.runtime_state.prepare_step(seqs, is_prefill)
        set_context(True, cu_seqlens_q=cu_seqlens, cache_manager=manager, seqs=seqs)
        get_context().sparse_controller = runner.sparse_controller
        runner.sparse_controller.prepare_forward(seqs, is_prefill)
        assert input_ids.numel() == token_budget
        assert len(seqs) == min(config.max_num_seqs_in_batch, config.max_num_seqs_in_gpu, token_budget)
        assert seqs[0].num_prefilled_tokens + seqs[0].current_chunk_size == config.max_model_len - 1
        assert all(seq.num_prefilled_tokens == 0 for seq in seqs[1:])
        all_slots = []
        context_lens = []
        for index, seq in enumerate(seqs):
            end = seq.num_prefilled_tokens + seq.current_chunk_size
            assert positions[cu_seqlens[index]:cu_seqlens[index + 1]].tolist() == list(
                range(seq.num_prefilled_tokens, end)
            )
            row = manager.seq_id_to_row[seq.seq_id]
            all_slots.append(manager.buffer_req_to_token_slots[row, :end])
            context_lens.append(end)
        slots = torch.cat(all_slots)
        assert slots.unique().numel() == slots.numel()
        assert manager.layer_batch_state.context_lens.tolist() == context_lens
        assert manager.num_free_slots == 0
        assert slots.numel() == config.max_model_len - 1 + sum(
            seq.current_chunk_size for seq in seqs[1:]
        )
        calls.append(input_ids.numel())
        platform.peak = 350
        if fail_forward:
            raise RuntimeError("model failed")

    runner.run = run
    if fail_forward:
        with pytest.raises(RuntimeError, match="model failed"):
            prefill_history.profile_prefill_history(runner)
    else:
        result = prefill_history.profile_prefill_history(runner)
        assert result.measurement.transient_peak_bytes == 250
    assert len(calls) == 1
    assert calls[0] == token_budget
    assert (runner.cache_manager, runner.sparse_controller, runner.runtime_state) == previous
    assert model.sparse_controller is previous[1]
    assert config.sparse_method == "h2o"
    assert config.num_kvcache_slots == original_slots
    assert get_context().cache_manager is None
    assert len(released) == 1
    assert not released[0].seq_id_to_row
