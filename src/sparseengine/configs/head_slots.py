"""Validation for independent, page-size-one KV-head storage."""

import math

import torch


def normalize_head_slot_layout(config) -> None:
    if config.kv_cache_layout == "auto":
        config.kv_cache_layout = "head_slots" if config.sparse_method == "adakv" else "token"
    if config.sparse_method == "adakv" and config.kv_cache_layout != "head_slots":
        raise ValueError("AdaKV requires independent kv_cache_layout='head_slots'.")


def validate_head_slots(config) -> None:
    if config.kv_cache_layout not in {"token", "head_slots"}:
        raise ValueError("kv_cache_layout must be 'token' or 'head_slots'.")
    if config.kv_cache_layout == "token":
        if config.head_slot_budgets is not None:
            raise ValueError("Head-slot budgets require kv_cache_layout='head_slots'.")
        return
    if config.sparse_method not in {"snapkv", "adakv", "kvzip"} or config.prefill_sparse_method:
        raise ValueError("head_slots requires SnapKV, AdaKV or native KVzip with dense prefill.")
    if (config.enable_prefix_cache_offload
            or (config.enable_prefix_caching and config.resolved_prefix_cache_mode != "chain")
            or (config.sparse_method == "snapkv" and not config.enable_prefix_caching)):
        raise ValueError("head_slots supports exclusive chain caching without offload; "
                         "AdaKV also supports independent requests without prefix caching.")
    if config.sparse_method == "kvzip":
        if config.enable_prefix_caching or config.enable_prefix_cache_offload:
            raise ValueError("Native KVzip head slots do not support prefix caching.")
        if config.head_slot_budgets is not None:
            raise ValueError("KVzip selects a global head budget; do not set head_slot_budgets.")
    if config.world_size != 1:
        raise ValueError("head_slots requires TP=EP=DP=1.")
    hf, layout = config.hf_config, config.runtime_layout
    if (config.attention_cache_layout != "explicit_kv"
            or layout.heterogeneous_kv or layout.linear_attention_layer_indices
            or layout.num_kv_layers != layout.num_layers):
        raise ValueError("head_slots requires homogeneous explicit KV attention storage.")
    if hf.dtype != torch.bfloat16 or config.quantization_config.enabled:
        raise ValueError("head_slots requires an unquantized BF16 model and KV cache.")
    # The model implementations below have ordinary causal RoPE attention. Also
    # check the actual checkpoint contract; a name alone does not establish it.
    if hf.model_type not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("head_slots currently supports ordinary Llama/Qwen attention.")
    if (getattr(hf, "sliding_window", None) not in (None, 0)
            or getattr(hf, "use_sliding_window", False)
            or getattr(hf, "attn_logit_softcapping", None)
            or getattr(hf, "alibi", False)
            or any("sliding" in str(t) for t in (getattr(hf, "layer_types", None) or []))):
        raise ValueError("head_slots does not support sliding windows or position/logit biases.")
    if hf.num_attention_heads % hf.num_key_value_heads:
        raise ValueError("head_slots requires integral GQA groups.")
    if config.sparse_method == "kvzip":
        return
    if type(config.observation_window_size) is not int or not 1 <= config.observation_window_size <= 128:
        raise ValueError("head_slots observation window must be in [1, 128].")
    if type(config.pool_kernel_size) is not int or config.pool_kernel_size < 1:
        raise ValueError("head_slots pool_kernel_size must be a positive integer.")
    if config.snapkv_decode_eviction:
        raise ValueError("head_slots compresses at final prefill/chain turn boundaries; decode eviction is unsupported.")
    if (config.snapkv_num_full_layers or config.full_attention_layers
            or config.pyramid_layer_ratios is not None):
        raise ValueError("Use head_slot_budgets for every head/layer, without full-layer overrides.")
    budgets = config.head_slot_budgets
    if config.sparse_method == "adakv":
        if budgets is not None:
            raise ValueError("AdaKV allocates actual head budgets; do not set head_slot_budgets.")
        alpha = config.adakv_alpha
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha) or not 0 < alpha <= 1):
            raise ValueError("adakv_alpha must be in (0, 1] to preserve nonempty heads.")
        budget = config.sink_keep_tokens + config.decode_keep_tokens + config.recent_keep_tokens
        if not 1 <= budget <= config.max_model_len:
            raise ValueError("AdaKV average total head budget must be in [1, max_model_len].")
        if config.sparse_prefill_score_mode != "probability":
            raise ValueError("AdaKV requires probability scores.")
    if budgets is not None:
        if (not isinstance(budgets, list) or len(budgets) != layout.num_kv_layers
                or any(not isinstance(row, list) or len(row) != hf.num_key_value_heads
                       for row in budgets)):
            raise ValueError("head_slot_budgets must have shape [num_layers, num_kv_heads].")
        if any(type(b) is not int or not 1 <= b <= config.max_model_len
               for row in budgets for b in row):
            raise ValueError("Every head-slot budget must be an integer in [1, max_model_len].")
