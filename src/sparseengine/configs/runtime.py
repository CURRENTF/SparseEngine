"""Top-level configuration composition and initialization orchestration."""

from dataclasses import dataclass, field
from typing import Any

from sparseengine.configs.bootstrap import normalize_bootstrap
from sparseengine.configs.cuda_graph import (
    _resolve_decode_cuda_graph_capture_sizes,
    normalize_decode_cuda_graph,
)
from sparseengine.configs.delta import (
    normalize_deltakv_storage,
    validate_deltakv_runtime,
)
from sparseengine.configs.full_attention_profiles import (
    resolve_auto_full_attention_layers,
)
from sparseengine.configs.groups import (
    DecodeCudaGraphConfig,
    DeltaKVConfig,
    KVQuantConfig,
    ObservabilityConfig,
    PrefillSparseMethodConfig,
    PrefixCacheConfig,
    SparseMethodConfig,
)
from sparseengine.configs.model import (
    AutoConfig,
    load_and_validate_model,
)
from sparseengine.configs.platform import normalize_platform
from sparseengine.configs.prefix_cache import (
    finalize_prefix_cache,
    normalize_prefix_cache,
)
from sparseengine.configs.scheduling import normalize_scheduling
from sparseengine.configs.sparse import (
    finalize_sparse_layout,
    normalize_prefill_sparse_method,
    normalize_sparse_method_name,
    normalize_sparse_methods,
)
from sparseengine.distributed import ParallelTopology
from sparseengine.method_registry import PREFILL_POLICY_AUTO
from sparseengine.models.layout import RuntimeLayout
from sparseengine.models.spec import ModelSpec
from sparseengine.quantization import QuantizationConfig
from sparseengine.utils.log import logger


@dataclass
class Config(
    PrefixCacheConfig,
    DecodeCudaGraphConfig,
    SparseMethodConfig,
    PrefillSparseMethodConfig,
    DeltaKVConfig,
    KVQuantConfig,
    ObservabilityConfig,
):
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs_in_batch: int = 32  # 不能设置太大
    max_model_len: int | None = None
    max_model_len_auto: bool = field(default=False, init=False)
    # None preserves the legacy shared batch limit. Set explicitly to allow
    # decode and prefill to use different per-step sequence limits.
    async_scheduling: bool | None = None
    async_max_inflight: int = 2
    decode_reservation_tokens: int = 1024
    max_decoding_seqs: int | None = None
    favor_min_decoding_seqs: int | None = None
    max_num_seqs_in_gpu: int | None = None

    engine_prefill_chunk_size: int | None = 8192
    long_prefill_offload_threshold: int = 64 * 1024
    mlp_chunk_size: int = 16384
    mla_prefill_workspace_bytes: int = 6 * 1024**3
    mla_prefill_history_chunk_size: int = 16384
    prefill_schedule_policy: str = PREFILL_POLICY_AUTO
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    moe_backend: str | None = None
    # Soft host-side I/O budget shared across ranks; every rank retains at
    # least one synchronous loading path when the budget is smaller.
    weight_loading_workers: int = 1
    enable_multimodal: bool = False
    hf_config: AutoConfig | None = None
    outer_hf_config: Any | None = None
    runtime_layout: RuntimeLayout | None = None
    attention_cache_layout: str = field(default="explicit_kv", init=False)
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig.disabled)
    model_spec: ModelSpec = field(init=False, repr=False)
    parallel_topology: ParallelTopology = field(init=False, repr=False)
    tiny_random: bool = False
    tiny_random_config: str | None = None
    tiny_random_seed: int = 0
    tiny_random_overrides: dict[str, int] = field(default_factory=dict, init=False)
    eos: int = -1
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)
    num_kvcache_slots: int | list = -1

    @property
    def attn_tp_size(self) -> int:
        return self.parallel_topology.attn_tp_size

    @property
    def attn_dp_size(self) -> int:
        return self.parallel_topology.attn_dp_size

    @property
    def moe_tp_size(self) -> int:
        return self.parallel_topology.moe_tp_size

    @property
    def moe_ep_size(self) -> int:
        return self.parallel_topology.moe_ep_size

    @property
    def world_size(self) -> int:
        return self.parallel_topology.world_size

    @property
    def weight_loading_workers_per_rank(self) -> int:
        return max(1, self.weight_loading_workers // self.world_size)

    def limit_auto_max_model_len(self, capacity: int) -> None:
        if not self.max_model_len_auto:
            return
        resolved = min(int(self.max_model_len), int(capacity))
        if resolved <= 0:
            raise RuntimeError(f"Runtime capacity must be positive, got {capacity}.")
        if resolved == self.max_model_len:
            return
        log = (
            logger.debug
            if getattr(self, "startup_cache_phase", "production") == "profiling"
            else logger.info
        )
        log(
            "Limiting auto max_model_len from model capacity {} to runtime capacity {}.",
            self.max_model_len,
            resolved,
        )
        self.max_model_len = resolved

    def __post_init__(self):
        from sparseengine.utils.compilation_guard import validate_compilation_limit

        validate_compilation_limit(self.runtime_compilation_limit)
        normalize_bootstrap(self)
        normalize_sparse_method_name(self)
        normalize_prefill_sparse_method(self)
        normalize_prefix_cache(self)
        normalize_scheduling(self)
        normalize_deltakv_storage(self)
        normalize_platform(self)
        load_and_validate_model(self)
        from sparseengine.configs.moe_communication import validate_moe_backend

        validate_moe_backend(self)
        from sparseengine.configs.kv_quant import validate_quantized_kv

        validate_quantized_kv(self)
        normalize_decode_cuda_graph(self)
        normalize_sparse_methods(self)
        finalize_prefix_cache(self)
        validate_deltakv_runtime(self)
        resolve_auto_full_attention_layers(self)
        finalize_sparse_layout(self)
        async_compatible = (
            self.attn_dp_size == 1 and not self.enable_multimodal
            and not self.enable_prefix_cache_offload and self.prefix_cache_mode != "chain"
            and not self.runtime_layout.linear_attention_layer_indices
        )
        if self.async_scheduling is None:
            from sparseengine.platforms import device_runtime
            self.async_scheduling = async_compatible and device_runtime.supports_streams()
        if self.async_scheduling:
            if self.async_max_inflight < 2:
                raise ValueError("async_max_inflight must be at least 2")
            if not async_compatible:
                raise ValueError("async_scheduling requires DP=1, radix/no prefix, "
                                 "no prefix offload, and text-only non-recurrent models")

        logger.info(
            "Runtime config: model={} sparse_method={} prefill_sparse_method={} "
            "cache_method={} tp={} ep={} dp={} moe_backend={} "
            "max_model_len={} max_batched_tokens={} prefill_chunk={} "
            "max_prefill_batch={} max_decode_batch={} favor_min_decoding_seqs={} gpu_utilization={:.3f} "
            "decode_graph={} async_scheduling={}.",
            self.model,
            self.sparse_method or "vanilla",
            self.prefill_sparse_method or "none",
            self.resolved_cache_sparse_method or "standard",
            self.tensor_parallel_size,
            self.expert_parallel_size,
            self.data_parallel_size,
            self.moe_backend,
            self.max_model_len,
            self.max_num_batched_tokens,
            self.engine_prefill_chunk_size,
            self.max_num_seqs_in_batch,
            self.max_decoding_seqs,
            self.favor_min_decoding_seqs,
            self.gpu_memory_utilization,
            self.decode_graph,
            self.async_scheduling,
        )
        logger.debug("Full runtime config: {}", str(self).replace("\n", " "))
        setattr(self.hf_config, "runtime_layout", self.runtime_layout)
