"""Composable groups of user-facing SparseEngine configuration fields."""

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class PrefixCacheConfig:
    """Prefix-cache capacity, mode, and host-offload settings."""

    enable_prefix_caching: bool = False
    prefix_cache_mode: str = "auto"
    resolved_prefix_cache_mode: str = field(default="disabled", init=False)
    prefix_cache_block_size: int | None = None
    prefix_cache_max_blocks: int | None = None
    chain_cache_max_tombstones: int = 1024
    enable_prefix_cache_offload: bool = False
    prefix_cache_host_size_gb: float | None = None
    recurrent_state_max_bytes: int | None = None
    prefix_cache_max_recurrent_bytes: int | None = None
    prefix_cache_salt: str = ""


@dataclass(kw_only=True)
class DecodeCudaGraphConfig:
    """Decode CUDA Graph capture and compatibility settings."""

    decode_graph: bool = True
    decode_graph_capture_sampling: bool = False
    decode_graph_capture_sizes: str | int | list[int] | tuple[int, ...] | None = "auto"
    decode_graph_startup_capture: bool | None = None
    decode_graph_startup_capture_limit: int | None = None
    sparse_attn_score_dtype: str = "float32"

@dataclass(kw_only=True)
class SparseMethodConfig:
    """Shared and method-specific sparse-attention settings."""

    sparse_method: str = ""
    palu_checkpoint_path: str | None = None
    palu_manifest: dict | None = field(default=None, init=False)
    enable_omnikv_offload: bool = False
    omnikv_offload_cache_tokens: int | None = None
    # Physical cache/runtime ownership can also come from a prefill method.
    # This derived field is normalized before cache and prefix-cache setup.
    resolved_cache_sparse_method: str = field(default="", init=False)
    sink_keep_tokens: int = 64
    recent_keep_tokens: int = 512
    decode_keep_tokens: int = 4096

    obs_layer_ids: list[int] = field(default=None, init=False)
    full_attention_layers: str | list[int] = "auto"
    resolved_full_attention_profile: str | None = field(default=None, init=False)

    quest_chunk_size: int = 16
    quest_token_budget: int = field(init=False)
    quest_skip_layers: int = 2

    snapkv_window_size: int = 32
    snapkv_num_full_layers: int = 0
    sparse_prefill_score_mode: str | None = None

    kvzip_token_budget: int = 4096
    kvzip_score_chunk_size: int = 2048
    kvzip_prev_postfix_size: int = 64

    h2o_decode_budget: int = 4096
    h2o_decode_eviction: bool = False
    h2o_decode_score_fusion: bool = True
    h2o_decode_eviction_interval: int = 128
    h2o_prefill_budget: int = 8192
    h2o_recent_ratio: float = 0.5
    h2o_prefill_score_window: int = 128

    rkv_compression_interval: int = 128
    rkv_observation_tokens: int = 8
    rkv_alpha: float = 0.1
    rkv_kernel_size: int = 7
    rkv_score_chunk_mb: int = 512
    rkv_similarity_threshold: float | None = None
    rkv_recent_similar_keep: int = 1
    rkv_max_redundancy_tokens: int | None = None
    rkv_redundancy_window: int = 0

    skipkv_compression_interval: int = 128
    skipkv_alpha: float = 0.1
    skipkv_similarity_threshold: float = 0.95
    skipkv_segment_size: int = 32
    skipkv_max_redundancy_tokens: int = 4096
    skipkv_redundancy_window: int = 64
    skipkv_enable_sentence_scoring: bool = True
    skipkv_sentence_score_weight: float = 1.0
    skipkv_sentence_min_tokens: int = 4
    skipkv_sentence_max_tokens: int = 256
    skipkv_sentence_embedding_layer: int = -1
    skipkv_max_tracked_sentences: int = 256
    skipkv_enable_activation_steering: bool = False
    skipkv_steering_vector_path: str | None = None
    skipkv_steering_layer: int = -1
    skipkv_steering_alpha: float = 0.0
    skipkv_steering_alpha_increment: float = 0.0
    skipkv_steering_alpha_max: float = 0.0

    pyramid_layer_ratios: list[float] | None = None
    pyramidkv_start_layer: int = 0
    pyramidkv_start_ratio: float = 0.6
    pyramidkv_least_layer: int | None = None
    pyramidkv_least_ratio: float = 0.01


@dataclass(kw_only=True)
class PrefillSparseMethodConfig:
    """Prefill acceleration settings, independent of KV/decode methods."""

    # None preserves the legacy H2O combined default. Explicit "" disables
    # prefill acceleration even when sparse_method="h2o" (decode-only H2O).
    prefill_sparse_method: str | None = None
    omnikv_prefill_full_attention_layers: str | list[int] | None = "auto"
    resolved_prefill_full_attention_profile: str | None = field(default=None, init=False)
    omnikv_prefill_keep_tokens: int = 4096
    omnikv_prefill_sink_keep_tokens: int = 8
    omnikv_prefill_recent_keep_tokens: int = 128
    flashprefill_v2_k_block_m: int = 128
    flashprefill_v2_k_block_n: int = 128
    flashprefill_v2_abs_threshold: float | None = None
    flashprefill_v2_attention_sink_blocks: int = 2
    flashprefill_v2_window_blocks: int = 4
    flashprefill_v2_last_query_blocks: int = 8
    flashprefill_v2_min_sparse_q_len: int = 4096
    flashprefill_v2_use_mean_correction: bool = True


@dataclass(kw_only=True)
class KVQuantConfig:
    kv_quant_page_size: int = 32
    kivi_bits: int = 4
    turboquant_bits: int = 4
    turboquant_seed: int = 0


@dataclass(kw_only=True)
class DeltaKVConfig:
    """DeltaKV compressor, quantization, and kernel settings."""

    deltakv_checkpoint_path: str | None = None
    deltakv_neighbor_count: int = 4
    deltakv_center_ratio: float = 0.1
    cluster_metric: str = "l2"
    cluster_on_kv: bool = True
    use_compression: bool = True
    deltakv_latent_dim: int = 128
    deltakv_latent_quant_bits: int = 4
    deltakv_latent_quant_group_size: int = 0
    enable_sparse_ref_fp8: bool = False

    full_layer_kv_quant_bits: int = 0
    full_layer_cluster_ratio: float = 0.0
    full_layer_kivi_group_size: int = 32
    full_layer_kivi_residual_length: int = 32
    full_layer_kivi_decode_block_seq: int = 256
    full_layer_kivi_decode_block_n: int = 16
    full_layer_kivi_decode_num_warps: int = 2
    full_layer_kivi_decode_num_stages: int = 3
    enable_full_layer_kivi_quant: bool = True
    enable_full_layer_kivi_fused_decode: bool = False
    enable_full_layer_kivi_grouped_decode: bool = False
    enable_full_layer_kivi_dense_decode: bool = False
    pool_kernel_size: int = 1

    use_nonlinear_compressor: bool = True
    compressor_intermediate_size: int = 2048
    compressor_linear_bias: bool = True
    compressor_down_type: str = "auto"
    compressor_up_type: str = "auto"
    compressor_down_intermediate_size: int = -1
    compressor_up_intermediate_size: int = -1

    deltakv_full_pool_reserve_ratio: float = 0.1
    deltakv_cache_capacity_margin: float = 1.05
    deltakv_center_capacity_margin: float = 1.5
    deltakv_triton_gather_heads_per_program: int = 4
    deltakv_triton_reconstruct_heads_per_program: int = 4
    deltakv_triton_materialize_block_tokens: int = 16
    deltakv_sparse_decode_backend: str = "auto"
    deltakv_cluster_gather_chunk_size: int = 16384

    allow_missing_deltakv_path: bool = False


@dataclass(kw_only=True)
class ObservabilityConfig:
    """Diagnostics and config-loading behavior."""

    enable_profiler: bool = False
    validate_runtime_invariants: bool = False
    throughput_log_interval_s: float = 10.0
    runtime_compilation_limit: int = 500
