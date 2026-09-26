"""Sparse-method normalization and layout-dependent validation."""

import torch

from sparseengine.configs.common import (
    _coerce_bool_config,
    _normalize_float_attr,
    _normalize_int_attr,
    _normalize_positive_int,
)
from sparseengine.method_registry import (
    CANONICAL_PREFILL_SPARSE_METHODS,
    PREFILL_SPARSE_METHOD_COMPATIBILITY,
    SKIPKV_ASSET_MODEL_NAMES,
    SUPPORTED_SPARSE_METHODS,
    normalize_sparse_method,
    omnikv_prefill_layer_indices,
    resolve_cache_sparse_method,
    resolve_prefill_sparse_method,
    resolve_sparse_prefill_score_mode,
)
from sparseengine.utils.log import log_once, logger
from sparseengine.models.layout import resolve_attention_qk_head_dim


def normalize_sparse_method_name(config) -> None:
    config.sparse_method = normalize_sparse_method(config.sparse_method)
    config.enable_omnikv_offload = _coerce_bool_config(
        "enable_omnikv_offload", config.enable_omnikv_offload
    )
    if config.enable_omnikv_offload and config.sparse_method != "omnikv":
        raise ValueError("enable_omnikv_offload requires sparse_method='omnikv'.")
    cache_tokens = config.omnikv_offload_cache_tokens
    if cache_tokens is not None and (
        isinstance(cache_tokens, bool) or not isinstance(cache_tokens, int) or cache_tokens < 0
    ):
        raise ValueError("omnikv_offload_cache_tokens must be a non-negative integer or None.")
    if cache_tokens and not config.enable_omnikv_offload:
        raise ValueError("omnikv_offload_cache_tokens requires enable_omnikv_offload.")
    if config.sparse_method not in SUPPORTED_SPARSE_METHODS:
        supported = ", ".join(repr(method) for method in sorted(SUPPORTED_SPARSE_METHODS) if method)
        raise ValueError(
            f"Unsupported sparse_method={config.sparse_method!r}. "
            f"Supported methods: '', {supported}."
        )
    for name in ("sink_keep_tokens", "decode_keep_tokens", "recent_keep_tokens"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer token count, got {value!r}."
            )


def normalize_prefill_sparse_method(config) -> None:
    sparse_method = normalize_sparse_method(getattr(config, "sparse_method", ""))
    method = resolve_prefill_sparse_method(
        getattr(config, "prefill_sparse_method", None),
        sparse_method=sparse_method,
    )
    if method not in CANONICAL_PREFILL_SPARSE_METHODS:
        supported = ", ".join(
            repr(name) for name in sorted(CANONICAL_PREFILL_SPARSE_METHODS) if name
        )
        raise ValueError(
            f"Unsupported prefill_sparse_method={method!r}. "
            f"Supported methods: '', {supported}."
        )
    compatible_sparse_methods = PREFILL_SPARSE_METHOD_COMPATIBILITY[method]
    if sparse_method not in compatible_sparse_methods:
        choices = ", ".join(
            "'vanilla'" if name == "" else repr(name)
            for name in sorted(compatible_sparse_methods)
        )
        raise ValueError(
            f"prefill_sparse_method={method!r} is incompatible with "
            f"sparse_method={sparse_method!r}; supported cache/decode methods: "
            f"{choices}."
        )
    config.prefill_sparse_method = method
    config.resolved_cache_sparse_method = resolve_cache_sparse_method(
        sparse_method,
        prefill_sparse_method=method,
    )

    for name in (
        "flashprefill_v2_k_block_m",
        "flashprefill_v2_k_block_n",
    ):
        _normalize_positive_int(config, name, fallback=0)
    if config.flashprefill_v2_k_block_m % 16:
        raise ValueError(
            "flashprefill_v2_k_block_m must be a multiple of 16, got "
            f"{config.flashprefill_v2_k_block_m}."
        )
    block_n = int(config.flashprefill_v2_k_block_n)
    if block_n % 64 or block_n & (block_n - 1):
        raise ValueError(
            "flashprefill_v2_k_block_n must be a power-of-two multiple of 64, "
            f"got {block_n}."
        )
    threshold = config.flashprefill_v2_abs_threshold
    if threshold is None:
        if method == "flashprefill_v2":
            raise ValueError(
                "prefill_sparse_method='flashprefill_v2' requires an explicit "
                "flashprefill_v2_abs_threshold calibrated for the model."
            )
    else:
        _normalize_float_attr(config, "flashprefill_v2_abs_threshold")
        if not 0.0 <= config.flashprefill_v2_abs_threshold <= 1.0:
            raise ValueError(
                "flashprefill_v2_abs_threshold must be in [0, 1], got "
                f"{config.flashprefill_v2_abs_threshold}."
            )
    for name in (
        "flashprefill_v2_attention_sink_blocks",
        "flashprefill_v2_window_blocks",
        "flashprefill_v2_last_query_blocks",
        "flashprefill_v2_min_sparse_q_len",
    ):
        _normalize_int_attr(config, name, fallback=0)
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative, got {getattr(config, name)}.")
    config.flashprefill_v2_use_mean_correction = _coerce_bool_config(
        "flashprefill_v2_use_mean_correction",
        config.flashprefill_v2_use_mean_correction,
    )


def _normalize_quest(config) -> None:
    if isinstance(config.full_attention_layers, str):
        layers = config.full_attention_layers.strip()
        if layers.lower() != "auto":
            config.full_attention_layers = (
                [] if not layers else [int(x) for x in layers.split(",")]
            )

    if config.quest_chunk_size <= 0:
        raise ValueError("quest_chunk_size 必须 > 0")
    config.quest_token_budget = 0
    if config.sparse_method == "quest":
        config.quest_token_budget = (
            config.sink_keep_tokens
            + config.decode_keep_tokens
            + config.recent_keep_tokens
        )
        if config.quest_token_budget <= 0:
            raise ValueError(
                "QuEST derived token budget must be > 0: "
                "sink_keep_tokens + decode_keep_tokens + recent_keep_tokens "
                f"= {config.quest_token_budget}."
            )
    if config.quest_skip_layers < 0:
        raise ValueError("quest_skip_layers 不能 < 0")


def _normalize_snapkv(config) -> None:
    config.snapkv_decode_eviction = _coerce_bool_config(
        "snapkv_decode_eviction", config.snapkv_decode_eviction
    )
    _normalize_positive_int(config, "observation_window_size", fallback=0)
    _normalize_positive_int(config, "decode_eviction_interval", fallback=0)
    if config.sparse_method == "pyramidkv":
        config.snapkv_decode_eviction = True
    _normalize_int_attr(config, "snapkv_num_full_layers")
    if config.snapkv_num_full_layers != 0:
        raise ValueError(
            "snapkv_num_full_layers is unsupported and must be 0, got "
            f"{config.snapkv_num_full_layers}."
        )


def _normalize_h2o(config) -> None:
    _normalize_positive_int(config, "h2o_decode_budget", fallback=0)
    _normalize_positive_int(config, "h2o_decode_eviction_interval", fallback=0)
    _normalize_int_attr(config, "h2o_prefill_budget", fallback=0)
    if config.h2o_prefill_budget < config.h2o_decode_budget:
        raise ValueError(
            "h2o_prefill_budget must be >= h2o_decode_budget, "
            f"got prefill={config.h2o_prefill_budget} decode={config.h2o_decode_budget}."
        )
    _normalize_float_attr(config, "h2o_recent_ratio")
    if not 0.0 < config.h2o_recent_ratio < 1.0:
        raise ValueError(
            f"h2o_recent_ratio must be in (0, 1), got {config.h2o_recent_ratio}."
        )
    _normalize_int_attr(config, "h2o_prefill_score_window", fallback=0)
    score_mode = getattr(config, "sparse_prefill_score_mode", "probability")
    if score_mode == "logits":
        if config.h2o_prefill_score_window < 0:
            raise ValueError(
                "h2o_prefill_score_window must be non-negative in logits "
                f"mode (0 means the full chunk), got {config.h2o_prefill_score_window}."
            )
    elif not 0 <= config.h2o_prefill_score_window <= 128:
        raise ValueError(
            "h2o_prefill_score_window must be in [0, 128] in probability mode "
            "(0 means the full current chunk), got "
            f"{config.h2o_prefill_score_window}."
        )


def _normalize_sparse_prefill_score(config) -> None:
    config.h2o_decode_score_fusion = _coerce_bool_config(
        "h2o_decode_score_fusion", getattr(config, "h2o_decode_score_fusion", True)
    )
    config.h2o_decode_eviction = _coerce_bool_config(
        "h2o_decode_eviction", getattr(config, "h2o_decode_eviction", False)
    )
    if config.h2o_decode_eviction:
        if normalize_sparse_method(config.sparse_method) != "h2o":
            raise ValueError("h2o_decode_eviction=True requires sparse_method='h2o'.")
        if getattr(config, "attention_cache_layout", "explicit_kv") == "mla_latent":
            # TODO(h2o-mla-parity): normalize each head before reduction instead
            # of approximating attention mass with softmax(head-max raw QK).
            log_once(
                "TODO(h2o-mla-parity): MLA H2O decode uses softmax(scale * "
                "RAW_QK_REDUCED), reducing heads before normalization. This "
                "approximation is not fully aligned with original H2O "
                "attention-probability accumulation; prefill scoring is unchanged.",
                level="WARNING",
            )
    cache_method = resolve_cache_sparse_method(
        config.sparse_method,
        prefill_sparse_method=config.prefill_sparse_method,
    )
    mode = resolve_sparse_prefill_score_mode(
        cache_method,
        config.sparse_prefill_score_mode,
    )
    allowed = {"probability", "logits"}
    if mode not in allowed:
        raise ValueError(
            "sparse_prefill_score_mode must be one of "
            f"{sorted(allowed)}, got {config.sparse_prefill_score_mode!r}."
        )
    if config.h2o_decode_eviction:
        if mode != "probability":
            log_once(
                "h2o_decode_eviction=True forces sparse_prefill_score_mode='probability' "
                "instead of mixing prefill logits with decode probabilities.",
                level="WARNING",
            )
        mode = "probability"
    if mode != "probability" and cache_method not in {
        "snapkv",
        "pyramidkv",
        "h2o",
    }:
        raise ValueError(
            "sparse_prefill_score_mode='logits' only applies to "
            f"SnapKV/PyramidKV/H2O, got method={cache_method!r}."
        )
    if mode == "logits" and config.sparse_attn_score_dtype != "float32":
        raise ValueError(
            "sparse_prefill_score_mode='logits' requires "
            "sparse_attn_score_dtype='float32', got "
            f"{config.sparse_attn_score_dtype!r}."
        )
    config.sparse_prefill_score_mode = mode
    if mode == "probability" and cache_method == "h2o":
        # TODO(h2o-probability-efficiency): avoid duplicate QK work in H2O
        # probability scoring, including the path with reused attention LSE,
        # while preserving score semantics.
        log_once(
            "TODO(h2o-probability-efficiency): H2O selects probability "
            "scoring with a known efficiency issue: separate score kernels "
            "recompute QK and reduce probabilities in addition to main attention. "
            "Reusing attention LSE, when available, does not eliminate this extra "
            "QK work. Probability scoring remains enabled with unchanged semantics.",
            level="WARNING",
        )

def _normalize_rkv(config) -> None:
    _normalize_positive_int(config, "rkv_compression_interval", fallback=0)
    _normalize_positive_int(config, "rkv_observation_tokens", fallback=0)
    if config.rkv_observation_tokens > config.rkv_compression_interval:
        raise ValueError(
            "rkv_observation_tokens must be <= rkv_compression_interval so the query cache "
            "can be refreshed between decode evictions, "
            f"got observation={config.rkv_observation_tokens} interval={config.rkv_compression_interval}."
        )
    _normalize_float_attr(config, "rkv_alpha")
    if not 0.0 <= config.rkv_alpha <= 1.0:
        raise ValueError(f"rkv_alpha must be in [0, 1], got {config.rkv_alpha}.")
    if config.rkv_similarity_threshold is None:
        config.rkv_similarity_threshold = 0.5 if config.sparse_method == "rkv" else 0.8
    _normalize_float_attr(config, "rkv_similarity_threshold")
    if not 0.0 <= config.rkv_similarity_threshold <= 1.0:
        raise ValueError(
            "rkv_similarity_threshold must be in [0, 1], "
            f"got {config.rkv_similarity_threshold}."
        )
    _normalize_int_attr(config, "rkv_recent_similar_keep")
    if config.rkv_recent_similar_keep < 0:
        raise ValueError(
            f"rkv_recent_similar_keep must be >= 0, got {config.rkv_recent_similar_keep}."
        )
    _normalize_positive_int(config, "rkv_kernel_size", fallback=0)
    _normalize_positive_int(config, "rkv_score_chunk_mb", fallback=0)
    if config.rkv_kernel_size % 2 != 1:
        raise ValueError("rkv_kernel_size must be odd.")
    if config.sparse_method == "rkv":
        budget = config.sink_keep_tokens + config.decode_keep_tokens + config.recent_keep_tokens
        if budget <= config.rkv_observation_tokens:
            raise ValueError("R-KV total retention budget must exceed rkv_observation_tokens.")
        if config.rkv_similarity_threshold != 0.5 or config.rkv_recent_similar_keep != 1:
            raise ValueError("R-KV vLLM scoring requires threshold=0.5 and the last similar representative; "
                             "remove legacy rkv_similarity_threshold/rkv_recent_similar_keep overrides.")
        if config.rkv_redundancy_window != 0 or config.rkv_max_redundancy_tokens is not None:
            raise ValueError("R-KV vLLM scoring uses the full resident domain with bounded row tiles; "
                             "remove rkv_redundancy_window/rkv_max_redundancy_tokens overrides "
                             "and configure rkv_score_chunk_mb instead.")

def _normalize_skipkv(config) -> None:
    _normalize_positive_int(config, "skipkv_compression_interval", fallback=0)
    _normalize_float_attr(config, "skipkv_alpha")
    if config.skipkv_alpha < 0.0:
        raise ValueError(f"skipkv_alpha must be >= 0, got {config.skipkv_alpha}.")
    _normalize_float_attr(config, "skipkv_similarity_threshold")
    if not 0.0 <= config.skipkv_similarity_threshold <= 1.0:
        raise ValueError(
            "skipkv_similarity_threshold must be in [0, 1], "
            f"got {config.skipkv_similarity_threshold}."
        )
    _normalize_positive_int(config, "skipkv_segment_size", fallback=0)
    _normalize_positive_int(config, "skipkv_max_redundancy_tokens", fallback=0)
    _normalize_positive_int(config, "skipkv_redundancy_window", fallback=0)
    if config.skipkv_redundancy_window > config.skipkv_max_redundancy_tokens:
        raise ValueError(
            "skipkv_redundancy_window must be <= skipkv_max_redundancy_tokens, "
            f"got window={config.skipkv_redundancy_window} max={config.skipkv_max_redundancy_tokens}."
        )
    config.skipkv_enable_sentence_scoring = bool(config.skipkv_enable_sentence_scoring)
    _normalize_float_attr(config, "skipkv_sentence_score_weight")
    if config.skipkv_sentence_score_weight < 0.0:
        raise ValueError(
            "skipkv_sentence_score_weight must be >= 0, "
            f"got {config.skipkv_sentence_score_weight}."
        )
    _normalize_positive_int(config, "skipkv_sentence_min_tokens", fallback=0)
    _normalize_int_attr(config, "skipkv_sentence_max_tokens", fallback=0)
    if config.skipkv_sentence_max_tokens < config.skipkv_sentence_min_tokens:
        raise ValueError(
            "skipkv_sentence_max_tokens must be >= skipkv_sentence_min_tokens, "
            f"got max={config.skipkv_sentence_max_tokens} min={config.skipkv_sentence_min_tokens}."
        )
    _normalize_int_attr(config, "skipkv_sentence_embedding_layer")
    _normalize_positive_int(config, "skipkv_max_tracked_sentences", fallback=0)
    config.skipkv_enable_activation_steering = bool(config.skipkv_enable_activation_steering)
    _normalize_int_attr(config, "skipkv_steering_layer")
    _normalize_float_attr(config, "skipkv_steering_alpha")
    _normalize_float_attr(config, "skipkv_steering_alpha_increment")
    _normalize_float_attr(config, "skipkv_steering_alpha_max")
    if config.skipkv_enable_activation_steering and not config.skipkv_steering_vector_path:
        raise ValueError(
            "skipkv_enable_activation_steering=True requires skipkv_steering_vector_path. "
            "Official SkipKV support is limited to the released steering vectors for "
            f"{', '.join(sorted(SKIPKV_ASSET_MODEL_NAMES))}."
        )


def _validate_prefill_sparse_method_model_compatibility(config) -> None:
    if config.prefill_sparse_method == "omnikv_prefill":
        if config.attention_cache_layout not in {"explicit_kv", "mla_latent"}:
            raise NotImplementedError(
                "omnikv_prefill requires explicit KV or MLA latent cache storage."
            )
        if config.sparse_attn_score_dtype != "float32":
            raise ValueError("omnikv_prefill raw-QK accumulation requires float32 scores.")
        for name in (
            "omnikv_prefill_keep_tokens",
            "omnikv_prefill_sink_keep_tokens",
            "omnikv_prefill_recent_keep_tokens",
        ):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        return
    if config.prefill_sparse_method != "flashprefill_v2":
        return
    cache_layout = str(config.attention_cache_layout)
    if cache_layout != "explicit_kv":
        raise NotImplementedError(
            "prefill_sparse_method='flashprefill_v2' requires explicit KV cache "
            f"storage; model {config.model_spec.name!r} uses "
            f"attention_cache_layout={cache_layout!r}."
        )


def _normalize_kvzip(config) -> None:
    if config.sparse_method != "kvzip":
        return
    for name in ("kvzip_token_budget", "kvzip_score_chunk_size", "kvzip_prev_postfix_size"):
        value = getattr(config, name)
        minimum = 0 if name == "kvzip_prev_postfix_size" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if config.attention_cache_layout != "explicit_kv":
        raise NotImplementedError("KVzip requires explicit KV storage.")
    if resolve_attention_qk_head_dim(config.hf_config) not in {16, 32, 64, 128, 256}:
        raise NotImplementedError("KVzip scoring requires head dimension 16, 32, 64, 128 or 256.")
    if config.sparse_attn_score_dtype != "float32":
        raise ValueError("KVzip requires sparse_attn_score_dtype='float32'.")
    if config.kvzip_score_chunk_size > config.engine_prefill_chunk_size:
        raise ValueError("kvzip_score_chunk_size must be <= engine_prefill_chunk_size.")


def _normalize_retroinfer(config) -> None:
    if config.sparse_method != "retroinfer":
        return
    if config.attention_cache_layout != "explicit_kv":
        raise ValueError("RetroInfer GPU-only requires uniform explicit KV storage.")
    layout = config.runtime_layout
    if layout.linear_attention_layer_indices or layout.heterogeneous_kv:
        raise ValueError("RetroInfer GPU-only requires uniform explicit KV layers.")
    if layout.layer_idx_to_kv_idx != tuple(range(layout.num_layers)):
        raise ValueError("RetroInfer GPU-only requires independent per-layer KV caches.")
    if any(
        str(layer_type) == "sliding_attention"
        for layer_type in (getattr(config.hf_config, "layer_types", None) or ())
    ):
        raise ValueError("RetroInfer GPU-only requires full causal attention, not sliding attention.")
    if config.hf_config.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("RetroInfer GPU-only requires FP16 or BF16 KV.")
    if config.enable_prefix_caching or config.enable_prefix_cache_offload:
        raise ValueError("RetroInfer GPU-only v1 does not support prefix caching.")
    if config.decode_graph:
        raise ValueError("RetroInfer GPU-only v1 requires decode_graph=False.")
    if config.async_scheduling is True:
        raise ValueError("RetroInfer GPU-only v1 requires async_scheduling=False.")
    config.async_scheduling = False
    for name in (
        "retroinfer_sink_tokens", "retroinfer_recent_tokens",
        "retroinfer_avg_cluster_size", "retroinfer_min_index_tokens",
        "retroinfer_update_tokens",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    for name in ("retroinfer_retrieval_ratio", "retroinfer_estimation_ratio"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    if config.retroinfer_retrieval_ratio <= 0:
        raise ValueError("RetroInfer retrieval ratio must be positive.")
    if config.retroinfer_retrieval_ratio + config.retroinfer_estimation_ratio > 1:
        raise ValueError("RetroInfer retrieval and estimation ratios must sum to at most 1.")
    if config.retroinfer_update_tokens % config.retroinfer_avg_cluster_size:
        raise ValueError("RetroInfer update size must be divisible by average cluster size.")
    if config.retroinfer_min_index_tokens % config.retroinfer_avg_cluster_size:
        raise ValueError("RetroInfer minimum index size must be divisible by average cluster size.")


def normalize_sparse_methods(config) -> None:
    _normalize_retroinfer(config)
    _normalize_kvzip(config)
    _validate_prefill_sparse_method_model_compatibility(config)
    if (
        getattr(config.hf_config, "model_type", "") == "gemma4_text"
        and int(getattr(config.hf_config, "num_kv_shared_layers", 0) or 0)
        and config.sparse_method == "streamingllm"
    ):
        raise NotImplementedError(
            "Gemma 4 StreamingLLM requires independent per-layer KV caches; "
            "KV-sharing variants support vanilla and OmniKV."
        )
    _normalize_quest(config)
    _normalize_snapkv(config)
    # _normalize_sparse_prefill_score must run before _normalize_h2o to validate
    # and canonicalize config.sparse_prefill_score_mode before H2O window checks.
    _normalize_sparse_prefill_score(config)
    _normalize_h2o(config)
    _normalize_rkv(config)
    _normalize_skipkv(config)

def finalize_sparse_layout(config) -> None:
    configured_full_layers = {int(layer) for layer in config.full_attention_layers}
    kv_layers = tuple(int(layer) for layer in config.runtime_layout.kv_idx_to_layer_idx)
    kv_positions = {layer: index for index, layer in enumerate(kv_layers)}
    if config.prefill_sparse_method == "omnikv_prefill":
        prefill_layers = omnikv_prefill_layer_indices(config)
        layers = config.omnikv_prefill_full_attention_layers
        if isinstance(layers, str):
            layers = [int(item.strip()) for item in layers.split(",") if item.strip()]
        if (
            not isinstance(layers, list)
            or not layers
            or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers)
            or len(set(layers)) != len(layers)
            or not prefill_layers
            or not set(layers).issubset(prefill_layers)
            or prefill_layers[0] not in layers
        ):
            raise ValueError(
                "omnikv_prefill_full_attention_layers must explicitly list unique "
                "global KV layer indices including the first KV layer eligible "
                "for global prefill; sliding-window layers are excluded."
            )
        config.omnikv_prefill_full_attention_layers = sorted(layers)
    unknown_full_layers = sorted(configured_full_layers - set(kv_layers))
    if unknown_full_layers and config.sparse_method in {"omnikv", "deltakv"}:
        raise ValueError(
            "full_attention_layers must contain KV/full-attention layer indices for "
            f"{config.sparse_method}; non-KV layers={unknown_full_layers}."
        )
    config.obs_layer_ids = []
    for layer in config.full_attention_layers:
        layer = int(layer)
        kv_position = kv_positions.get(layer)
        if kv_position is None or kv_position + 1 >= len(kv_layers):
            continue
        if kv_layers[kv_position + 1] not in configured_full_layers:
            config.obs_layer_ids.append(layer)

    # PyramidKV 配置验证与智能生成
    if 'pyramidkv' == config.sparse_method:
        num_layers = int(config.runtime_layout.num_layers)
        num_kv_layers = int(config.runtime_layout.num_kv_layers)
        if config.pyramid_layer_ratios is None:
            start_l = int(config.pyramidkv_start_layer)
            least_l = (
                int(config.pyramidkv_least_layer)
                if config.pyramidkv_least_layer is not None
                else num_kv_layers - 1
            )
            start_r = float(config.pyramidkv_start_ratio)
            least_r = float(config.pyramidkv_least_ratio)
            if not 0 <= start_l < num_kv_layers:
                raise ValueError(
                    f"pyramidkv_start_layer must be a KV layer position in [0, {num_kv_layers}), "
                    f"got {start_l}."
                )
            if not start_l <= least_l < num_kv_layers:
                raise ValueError(
                    "pyramidkv_least_layer must be a KV layer position between "
                    f"start_layer={start_l} and {num_kv_layers - 1}, got {least_l}."
                )

            ratios = [1.0] * num_kv_layers
            for i in range(start_l, num_kv_layers):
                if i <= least_l:
                    if least_l > start_l:
                        ratio = start_r - (start_r - least_r) * (i - start_l) / (least_l - start_l)
                    else:
                        ratio = least_r
                    ratios[i] = ratio
                else:
                    ratios[i] = least_r
            config.pyramid_layer_ratios = ratios
            logger.info(f"PyramidKV 自动生成 KV layer_ratios = {[f'{r:.3f}' for r in ratios]}")
        else:
            ratios = [float(ratio) for ratio in config.pyramid_layer_ratios]
            if len(ratios) == num_layers and num_layers != num_kv_layers:
                ratios = [ratios[layer_idx] for layer_idx in config.runtime_layout.kv_idx_to_layer_idx]
            config.pyramid_layer_ratios = ratios

    if config.pyramid_layer_ratios is not None:
        # PyramidKV 模式自动启用 SnapKV 逻辑
        if 'pyramidkv' != config.sparse_method:
            raise ValueError('sparse_method 应为 pyramidkv')

        num_kv_layers = int(config.runtime_layout.num_kv_layers)
        if len(config.pyramid_layer_ratios) != num_kv_layers:
            raise ValueError(
                f"pyramid_layer_ratios length ({len(config.pyramid_layer_ratios)}) must equal "
                f"the number of KV/full-attention layers ({num_kv_layers})."
            )

        if any(r <= 0 or r > 1.0 for r in config.pyramid_layer_ratios):
            raise ValueError("pyramid_layer_ratios 的所有值必须在 (0, 1.0] 范围内")
