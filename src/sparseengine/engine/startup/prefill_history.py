from __future__ import annotations

import copy

import torch

from sparseengine.engine.cache_manager.standard import StandardCacheManager
from sparseengine.engine.cache_manager.storage.low_rank_kv import LowRankKVStorage
from sparseengine.engine.cache_manager.storage import (
    ExplicitKVStorage,
    HeterogeneousExplicitKVStorage,
    MlaLatentStorage,
)
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.sparse_controller import SparseController
from sparseengine.sampling_params import SamplingParams
from sparseengine.utils.context import reset_context
from sparseengine.utils.log import logger

from .capacity import profiling_prefill_chunk_lengths
from .profiling import StartupMemoryProfiler


class PrefillHistoryCacheManager(StandardCacheManager):
    """Startup-only dense cache with synthetic history shared across layers."""

    def allocate_kv_cache(self) -> None:
        slots = int(self.config.num_kvcache_slots)
        storage = self.attention_cache_storage
        if isinstance(storage, LowRankKVStorage):
            storage.allocate_shared_history(num_slots=slots, device=self.device)
        elif isinstance(storage, HeterogeneousExplicitKVStorage):
            caches = {
                shape: torch.zeros(2, slots, *shape, dtype=storage.dtype, device=self.device)
                for shape in dict.fromkeys(storage.layer_shapes)
            }
            storage.kv_cache = [caches[shape] for shape in storage.layer_shapes]
        else:
            storage.allocate(num_layers=1, num_slots=slots, device=self.device)
            for tensor in storage.accounting_tensors():
                tensor.zero_()
            # Layers execute sequentially and overwrite only the current chunk.
            # The synthetic history stays zero, so one physical layer suffices.
            if isinstance(storage, ExplicitKVStorage):
                storage.kv_cache = storage.kv_cache.expand(
                    -1, self.num_kv_layers, -1, -1, -1,
                )
            elif isinstance(storage, MlaLatentStorage):
                storage.latent_cache = storage.latent_cache.expand(
                    self.num_kv_layers, -1, -1, -1,
                )
                storage.rope_cache = storage.rope_cache.expand(
                    self.num_kv_layers, -1, -1, -1,
                )
        self.kv_cache = getattr(storage, "kv_cache", None)

    def seed_history(self, seq: Sequence) -> None:
        self._allocate(seq.seq_id, int(seq.num_prefilled_tokens))


@torch.inference_mode()
def profile_prefill_history(runner):
    """Measure a full token batch with one maximum-context request in one step.

    Dense synthetic history covers history-dependent attention allocations.
    Sparse compression/scoring and concurrent long requests rely on utilization
    headroom rather than method-specific history reconstruction.
    """
    config = copy.copy(runner.config)
    context_len = int(config.max_model_len) - 1
    chunk_lengths = profiling_prefill_chunk_lengths(config)
    prompt_lengths = (context_len, *chunk_lengths[1:])
    config.enable_prefix_caching = False
    config.enable_prefix_cache_offload = False
    config.resolved_prefix_cache_mode = "disabled"
    config.startup_cache_phase = "profiling"
    config.num_kvcache_slots = sum(prompt_lengths)
    history_factory = getattr(runner.cache_manager, "create_prefill_history_manager", None)
    if callable(history_factory):
        manager = history_factory(config, runner.parallel_context)
    else:
        config.sparse_method = ""
        config.prefill_sparse_method = None
        manager = PrefillHistoryCacheManager(config, runner.parallel_context)
    bind_layers = getattr(manager, "set_model_layers", None)
    if callable(bind_layers):
        bind_layers(runner.model.model.layers)
    controller = SparseController(config, manager)
    runtime = RuntimeState(config, manager, runner.recurrent_state_manager)
    seqs = []
    for prompt_len, chunk_len in zip(prompt_lengths, chunk_lengths):
        seq = Sequence([0] * prompt_len, SamplingParams(max_tokens=1, temperature=0.0))
        seq.num_prefilled_tokens = prompt_len - chunk_len
        seq.current_chunk_size = chunk_len
        seqs.append(seq)
    if seqs[0].num_prefilled_tokens:
        manager.seed_history(seqs[0])

    model = runner.model.model
    previous = runner.cache_manager, runner.sparse_controller, runner.runtime_state
    previous_controller = model.sparse_controller
    profiler = StartupMemoryProfiler(runner.platform, runner.device)
    try:
        runner.cache_manager, runner.sparse_controller, runner.runtime_state = (
            manager, controller, runtime,
        )
        model.sparse_controller = controller
        controller.set_modules(model.layers)
        logger.info(
            "Startup profile phase=prefill tokens={} batch={} max_context={} "
            "history={} max_chunk={} visible_tokens={}.",
            sum(chunk_lengths), len(seqs), context_len,
            seqs[0].num_prefilled_tokens, max(chunk_lengths), sum(prompt_lengths),
        )
        # Keep the synthetic cache alive across both snapshots so its persistent
        # bytes are not counted as transient model memory.
        profiler.begin("prefill")
        runner.run(seqs, is_prefill=True)
        result = profiler.finish("prefill")
        logger.info(
            "Startup prefill transient_peak={:.2f} GiB.",
            result.measurement.transient_peak_bytes / 1024**3,
        )
        return result
    finally:
        runner.cache_manager, runner.sparse_controller, runner.runtime_state = previous
        model.sparse_controller = previous_controller
        reset_context()
        for seq in seqs:
            runtime.free_seq(seq.seq_id)
        runtime.reset_after_warmup()
        release_bindings = getattr(runner.model, "release_cache_runtime_bindings", None)
        if callable(release_bindings):
            release_bindings(manager)
