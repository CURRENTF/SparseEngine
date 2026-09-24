from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from sparseengine.distributed.topology import ParallelTopology
from sparseengine.models.spec import MODEL_SPECS
from sparseengine.operators.attention_capabilities import AttentionScoreKind

PREFILL_POLICY_ALL_CHUNKED = "all_chunked"
PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH = "long_bs1full_short_batch"
PREFILL_POLICY_AUTO = "auto"

SUPPORTED_PREFILL_POLICIES = {
    PREFILL_POLICY_ALL_CHUNKED,
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
}

METHOD_ALIASES = {
    None: "",
    "": "",
    "vanilla": "",
    "attention-sink": "streamingllm",
    "attention_sink": "streamingllm",
    "r-kv": "rkv",
    "r_kv": "rkv",
    "skip-kv": "skipkv",
    "skip_kv": "skipkv",
    # DeltaKV now has one public runtime.  The old names stay as aliases so old
    # config files still load, but all code routes through sparse_method="deltakv".
    "deltakv-less-memory": "deltakv",
    "deltakv_less_memory": "deltakv",
    "deltakv-less-memory-cudagraph": "deltakv",
    "deltakv_less_memory_cudagraph": "deltakv",
}

SPARSE_AUXILIARY_PREFILL_PROMPTS = {
    "kvzip": "\nReconstruct the following context span exactly:\n",
}
SPARSE_METHOD_MODEL_TYPES = {
    "palu": frozenset({"llama", "qwen3"}),
    "kvzip": frozenset({"qwen2", "qwen3", "llama"}),
}

QUANTIZED_KV_METHODS = frozenset({"kivi", "turboquant", "fp8_kv"})

CANONICAL_SPARSE_METHODS = {
    "palu",
    "",
    "streamingllm",
    "snapkv",
    "kvzip",
    "h2o",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
    "skipkv",
    "deltakv",
    *QUANTIZED_KV_METHODS,
}

SUPPORTED_SPARSE_METHODS = set(CANONICAL_SPARSE_METHODS)
SUPPORTED_SPARSE_METHOD_ALIASES = {str(k) for k in METHOD_ALIASES if k is not None and str(k)}

PREFILL_SPARSE_METHOD_ALIASES = {
    None: "",
    "": "",
    "dense": "",
    "h2o-prefill": "h2o_prefill",
    "h2o_prefill": "h2o_prefill",
    "flash-prefill-v2": "flashprefill_v2",
    "flash_prefill_v2": "flashprefill_v2",
    "flashprefill-v2": "flashprefill_v2",
    "flashprefill_v2": "flashprefill_v2",
}

CANONICAL_PREFILL_SPARSE_METHODS = {
    "",
    "h2o_prefill",
    "flashprefill_v2",
    "omnikv_prefill",
}

PREFILL_SPARSE_METHOD_COMPATIBILITY = {
    "": frozenset(CANONICAL_SPARSE_METHODS),
    "h2o_prefill": frozenset({"", "h2o"}),
    "omnikv_prefill": frozenset({"", "omnikv"}),
    "flashprefill_v2": frozenset({"", "omnikv", "quest", "snapkv", "h2o"}),
}


def normalize_prefill_sparse_method(method: str | None) -> str:
    if method in PREFILL_SPARSE_METHOD_ALIASES:
        return PREFILL_SPARSE_METHOD_ALIASES[method]
    return str(method).strip().lower().replace("-", "_")


def resolve_prefill_sparse_method(
    method: str | None,
    *,
    sparse_method: str | None,
) -> str:
    """Resolve the prefill algorithm, including cache-method defaults."""

    # A missing value keeps old `sparse_method="h2o"` configurations combined.
    # An explicit empty string is different: it requests decode-only H2O.
    if method is None and normalize_sparse_method(sparse_method) == "h2o":
        return "h2o_prefill"
    return normalize_prefill_sparse_method(method)


def resolve_cache_sparse_method(
    sparse_method: str | None,
    *,
    prefill_sparse_method: str | None,
) -> str:
    """Resolve the method that owns physical KV state and step lifecycle."""

    normalized = normalize_sparse_method(sparse_method)
    resolved_prefill = resolve_prefill_sparse_method(
        prefill_sparse_method,
        sparse_method=normalized,
    )
    if resolved_prefill == "h2o_prefill":
        return "h2o"
    return normalized


def omnikv_prefill_layer_indices(config) -> tuple[int, ...]:
    """Logical KV consumers eligible for global cross-layer prefill selection."""
    layout = getattr(config, "runtime_layout", None)
    layer_types = getattr(config.hf_config, "layer_types", None)
    return tuple(
        layer for layer in range(config.hf_config.num_hidden_layers)
        if (layout is None or layout.is_full_attention(layer))
        and (layer_types is None or layer_types[layer] != "sliding_attention")
    )


def prefill_sparse_method_fingerprint(config) -> dict[str, object]:
    """Return prefill semantics that can change cached hidden-state-derived KV."""

    method = resolve_prefill_sparse_method(
        getattr(config, "prefill_sparse_method", ""),
        sparse_method=getattr(config, "sparse_method", ""),
    )
    payload: dict[str, object] = {"prefill_sparse_method": method}
    if method == "omnikv_prefill":
        for field_name in (
            "omnikv_prefill_full_attention_layers",
            "omnikv_prefill_keep_tokens",
            "omnikv_prefill_sink_keep_tokens",
            "omnikv_prefill_recent_keep_tokens",
            "sparse_attn_score_dtype",
            "engine_prefill_chunk_size",
            "max_num_batched_tokens",
        ):
            payload[field_name] = getattr(config, field_name, None)
    if method == "flashprefill_v2":
        for field_name in (
            "flashprefill_v2_k_block_m",
            "flashprefill_v2_k_block_n",
            "flashprefill_v2_abs_threshold",
            "flashprefill_v2_attention_sink_blocks",
            "flashprefill_v2_window_blocks",
            "flashprefill_v2_last_query_blocks",
            "flashprefill_v2_min_sparse_q_len",
            "flashprefill_v2_use_mean_correction",
        ):
            payload[field_name] = getattr(config, field_name, None)
    return payload

PREFIX_CACHE_SUPPORTED_METHODS = {
    "",
    "streamingllm",
    "omnikv",
    "quest",
    "snapkv",
    "h2o",
    "pyramidkv",
    "rkv",
    "skipkv",
}

H2O_SUPPORTED_MODEL_TYPES = frozenset(MODEL_SPECS) - {"gemma4"}

SKIPKV_ASSET_MODEL_NAMES = frozenset(
    {
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-14B",
    }
)


@dataclass(frozen=True)
class ModelRuntimeCompatibility:
    sparse_methods: frozenset[str]
    prefix_cache_methods: frozenset[str]
    decode_graph_methods: frozenset[str] = frozenset()


class PrefillScoreCollectionKind(Enum):
    NONE = auto()
    METHOD_OWNED_POSTHOC_REDUCED = auto()
    MAIN_ATTENTION_REDUCED = auto()
    MAIN_ATTENTION_PER_HEAD = auto()


@dataclass(frozen=True)
class SparsePrefillAttentionContract:
    main_score_kind: AttentionScoreKind
    score_collection: PrefillScoreCollectionKind
    layer_varying_page_table: bool
    optional_score_output: bool = False


_PREFILL_POSTHOC_SCORE_METHODS = frozenset(
    {"snapkv", "pyramidkv", "h2o"}
)

# Prefill provider planning may be reused across transformer layers only when
# every layer reads the same physical slot table. OmniKV keeps Standard's
# shared table, and QuEST does not apply its query-aware page selection until
# decode. SnapKV-family managers and DeltaKV own per-layer physical tables.
_PREFILL_LAYER_VARYING_PAGE_TABLE = {
    "palu": False,
    **dict.fromkeys(QUANTIZED_KV_METHODS, False),
    "": False,
    "streamingllm": True,
    "snapkv": True,
    "kvzip": True,
    "h2o": True,
    "pyramidkv": True,
    "omnikv": False,
    "quest": False,
    "rkv": True,
    "skipkv": True,
    "deltakv": True,
}
if set(_PREFILL_LAYER_VARYING_PAGE_TABLE) != CANONICAL_SPARSE_METHODS:
    raise RuntimeError(
        "Prefill page-table contracts must cover every canonical sparse method."
    )

# Static method score contracts let providers bind before CUDA Graph capture
# instead of changing the score-producing implementation during replay.
_DECODE_ATTENTION_SCORE_KINDS = {
    "omnikv": AttentionScoreKind.RAW_QK_PER_HEAD,
    "skipkv": AttentionScoreKind.RAW_QK_PER_HEAD,
    "deltakv": AttentionScoreKind.RAW_QK_PER_HEAD,
}


def resolve_sparse_prefill_score_mode(
    method: str | None,
    configured_mode: str | None,
) -> str:
    """Resolve the method-owned prefill score mode before provider binding."""

    normalized = normalize_sparse_method(method)
    if configured_mode is None:
        return "logits" if normalized in {"snapkv", "h2o"} else "probability"
    return str(configured_mode).strip().lower()


def sparse_prefill_attention_contract(
    method: str | None,
    *,
    prefill_sparse_method: str | None = None,
    sparse_prefill_score_mode: str | None = None,
    h2o_prefill_score_window: int = 128,
) -> SparsePrefillAttentionContract:
    normalized = normalize_sparse_method(method)
    if normalized not in CANONICAL_SPARSE_METHODS:
        raise ValueError(f"Unknown sparse method {normalized!r}.")
    resolved_prefill_method = resolve_prefill_sparse_method(
        prefill_sparse_method,
        sparse_method=normalized,
    )
    if resolved_prefill_method == "omnikv_prefill":
        return SparsePrefillAttentionContract(
            main_score_kind=AttentionScoreKind.RAW_QK_PER_HEAD,
            score_collection=PrefillScoreCollectionKind.MAIN_ATTENTION_PER_HEAD,
            layer_varying_page_table=True,
            optional_score_output=True,
        )
    cache_method = resolve_cache_sparse_method(
        normalized,
        prefill_sparse_method=resolved_prefill_method,
    )
    h2o_score_collection = cache_method == "h2o"
    layer_varying_page_table = _PREFILL_LAYER_VARYING_PAGE_TABLE[cache_method]
    fused_h2o_score = (
        h2o_score_collection
        and resolved_prefill_method != "flashprefill_v2"
        and resolve_sparse_prefill_score_mode(
            cache_method,
            sparse_prefill_score_mode,
        )
        == "logits"
        and int(h2o_prefill_score_window) == 0
    )
    if fused_h2o_score:
        return SparsePrefillAttentionContract(
            main_score_kind=AttentionScoreKind.RAW_QK_REDUCED,
            score_collection=PrefillScoreCollectionKind.MAIN_ATTENTION_REDUCED,
            layer_varying_page_table=layer_varying_page_table,
        )
    collection = (
        PrefillScoreCollectionKind.METHOD_OWNED_POSTHOC_REDUCED
        if cache_method in _PREFILL_POSTHOC_SCORE_METHODS
        else PrefillScoreCollectionKind.NONE
    )
    return SparsePrefillAttentionContract(
        main_score_kind=AttentionScoreKind.NONE,
        score_collection=collection,
        layer_varying_page_table=layer_varying_page_table,
    )


def h2o_uses_fused_prefill_score(config) -> bool:
    return (
        resolve_cache_sparse_method(
            getattr(config, "sparse_method", None),
            prefill_sparse_method=getattr(config, "prefill_sparse_method", None),
        )
        == "h2o"
        and resolve_prefill_sparse_method(
            getattr(config, "prefill_sparse_method", None),
            sparse_method=getattr(config, "sparse_method", None),
        )
        != "flashprefill_v2"
        and resolve_sparse_prefill_score_mode(
            "h2o",
            getattr(config, "sparse_prefill_score_mode", None),
        )
        == "logits"
        and int(getattr(config, "h2o_prefill_score_window", 128)) == 0
    )


def sparse_decode_attention_requires_scores(
    method: str | None,
    *,
    h2o_decode_eviction: bool = False,
) -> bool:
    """Return whether a prepared decode implementation must support scores."""

    return (
        sparse_decode_attention_score_kind(
            method, h2o_decode_eviction=h2o_decode_eviction,
        )
        is not AttentionScoreKind.NONE
    )


def sparse_decode_attention_score_kind(
    method: str | None,
    *,
    h2o_decode_eviction: bool = False,
    attention_cache_layout: str = "explicit_kv",
) -> AttentionScoreKind:
    """Return the score representation consumed by sparse decode logic.

    OmniKV, SkipKV, and DeltaKV normalize each head in ``SparseController``
    before reducing across heads, so providers must preserve raw per-head QK.
    SnapKV and PyramidKV score cached query windows after the decode step.
    Explicit-KV H2O consumes probabilities summed over heads. MLA H2O uses
    an explicit head-reduced-logit softmax approximation in its runtime.
    """

    normalized = normalize_sparse_method(method)
    if normalized not in CANONICAL_SPARSE_METHODS:
        raise ValueError(f"Unknown sparse method {normalized!r}.")
    if normalized == "h2o" and h2o_decode_eviction:
        if attention_cache_layout == "mla_latent":
            return AttentionScoreKind.RAW_QK_REDUCED
        return AttentionScoreKind.ATTENTION_PROBABILITY_REDUCED
    return _DECODE_ATTENTION_SCORE_KINDS.get(
        normalized,
        AttentionScoreKind.NONE,
    )


_MOE_SPARSE_METHODS = frozenset(
    {"", "streamingllm", "snapkv", "h2o", "pyramidkv", "omnikv", "quest", "rkv"}
)

DENSE_MODEL_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=frozenset(CANONICAL_SPARSE_METHODS),
    prefix_cache_methods=frozenset(PREFIX_CACHE_SUPPORTED_METHODS),
    decode_graph_methods=frozenset(CANONICAL_SPARSE_METHODS),
)

QWEN3_MOE_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=_MOE_SPARSE_METHODS | QUANTIZED_KV_METHODS,
    prefix_cache_methods=frozenset(
        {"", "omnikv", "quest", "snapkv", "h2o", "pyramidkv", "rkv"}
    ),
    decode_graph_methods=_MOE_SPARSE_METHODS | QUANTIZED_KV_METHODS,
)

QWEN3_MOE_TP_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=_MOE_SPARSE_METHODS | QUANTIZED_KV_METHODS,
    prefix_cache_methods=frozenset({"", "snapkv"}),
    decode_graph_methods=_MOE_SPARSE_METHODS | QUANTIZED_KV_METHODS,
)

QWEN3_MOE_TP_COMPATIBILITY = QWEN3_MOE_TP_EP_COMPATIBILITY

QWEN35_MOE_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=_MOE_SPARSE_METHODS,
    prefix_cache_methods=frozenset({""}),
    decode_graph_methods=_MOE_SPARSE_METHODS,
)

MINIMAX_M2_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=_MOE_SPARSE_METHODS,
    prefix_cache_methods=frozenset({"", "omnikv", "quest", "snapkv"}),
    decode_graph_methods=_MOE_SPARSE_METHODS,
)

MINIMAX_M2_TP_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=MINIMAX_M2_EP_COMPATIBILITY.sparse_methods,
    prefix_cache_methods=MINIMAX_M2_EP_COMPATIBILITY.prefix_cache_methods,
    decode_graph_methods=MINIMAX_M2_EP_COMPATIBILITY.decode_graph_methods,
)

GLM4_MOE_LITE_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=frozenset(
        {"", "streamingllm", "snapkv", "h2o", "omnikv", "quest", "rkv"}
    ),
    prefix_cache_methods=frozenset(
        {"", "streamingllm", "snapkv", "h2o", "omnikv", "quest", "rkv"}
    ),
    decode_graph_methods=frozenset(
        {"", "streamingllm", "snapkv", "h2o", "omnikv", "quest", "rkv"}
    ),
)

GEMMA4_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=frozenset({"", "streamingllm", "omnikv"}),
    prefix_cache_methods=frozenset({"", "streamingllm", "omnikv"}),
    decode_graph_methods=frozenset({"", "streamingllm", "omnikv"}),
)

MODEL_RUNTIME_COMPATIBILITY = {
    **{
        model_type: DENSE_MODEL_COMPATIBILITY
        for model_type in ("qwen2", "qwen3", "qwen3_5", "llama")
    },
    "qwen3_moe": QWEN3_MOE_EP_COMPATIBILITY,
    "qwen3_5_moe": QWEN35_MOE_COMPATIBILITY,
    "minimax_m2": MINIMAX_M2_EP_COMPATIBILITY,
    "glm4_moe_lite": GLM4_MOE_LITE_EP_COMPATIBILITY,
    "gemma4": GEMMA4_COMPATIBILITY,
}

DECODE_CUDA_GRAPH_SUPPORTED_METHODS = set(CANONICAL_SPARSE_METHODS)
TP_DECODE_CUDA_GRAPH_SUPPORTED_METHODS = {
    "kvzip",
    *QUANTIZED_KV_METHODS,
    "",
    "streamingllm",
    "snapkv",
    "h2o",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
    "skipkv",
}


def decode_graph_path_id(method: str) -> str:
    """One graph-stable decode family for all request lengths."""
    return "unified" if method else "dense"


_DEFAULT_PREFILL_POLICY_BY_METHOD = {
    "palu": PREFILL_POLICY_ALL_CHUNKED,
    **dict.fromkeys(QUANTIZED_KV_METHODS, PREFILL_POLICY_ALL_CHUNKED),
    "": PREFILL_POLICY_ALL_CHUNKED,
    "streamingllm": PREFILL_POLICY_ALL_CHUNKED,
    "snapkv": PREFILL_POLICY_ALL_CHUNKED,
    "kvzip": PREFILL_POLICY_ALL_CHUNKED,
    "h2o": PREFILL_POLICY_ALL_CHUNKED,
    "pyramidkv": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "omnikv": PREFILL_POLICY_ALL_CHUNKED,
    "quest": PREFILL_POLICY_ALL_CHUNKED,
    "rkv": PREFILL_POLICY_ALL_CHUNKED,
    "skipkv": PREFILL_POLICY_ALL_CHUNKED,
    "deltakv": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
}

PREFILL_POLICY_BY_METHOD = {
    **_DEFAULT_PREFILL_POLICY_BY_METHOD,
    "vanilla": PREFILL_POLICY_ALL_CHUNKED,
    "attention-sink": PREFILL_POLICY_ALL_CHUNKED,
    "attention_sink": PREFILL_POLICY_ALL_CHUNKED,
    "r-kv": PREFILL_POLICY_ALL_CHUNKED,
    "r_kv": PREFILL_POLICY_ALL_CHUNKED,
    "skip-kv": PREFILL_POLICY_ALL_CHUNKED,
    "skip_kv": PREFILL_POLICY_ALL_CHUNKED,
    "deltakv-less-memory": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv_less_memory": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv-less-memory-cudagraph": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv_less_memory_cudagraph": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
}


def normalize_sparse_method(method: str | None) -> str:
    if method is None:
        return ""
    normalized = str(method).strip().lower()
    return METHOD_ALIASES.get(normalized, normalized)


def validate_sparse_method_assets(method: str | None, model_path: str) -> None:
    if normalize_sparse_method(method) != "skipkv":
        return
    model_name = str(model_path).rstrip("/").split("/")[-1]
    if model_name not in SKIPKV_ASSET_MODEL_NAMES:
        supported = ", ".join(sorted(SKIPKV_ASSET_MODEL_NAMES))
        raise ValueError(
            "SkipKV is supported only for models with released steering assets: "
            f"{supported}. Got model basename {model_name!r}."
        )


def is_deltakv_method(method: str | None) -> bool:
    return normalize_sparse_method(method) == "deltakv"


def is_decode_cuda_graph_supported(method: str | None) -> bool:
    return normalize_sparse_method(method) in DECODE_CUDA_GRAPH_SUPPORTED_METHODS


def is_tp_decode_cuda_graph_supported(method: str | None) -> bool:
    return normalize_sparse_method(method) in TP_DECODE_CUDA_GRAPH_SUPPORTED_METHODS


def validate_model_runtime_compatibility(
    *,
    model_type: str,
    sparse_method: str | None,
    topology: ParallelTopology,
    decode_graph: bool,
    enable_prefix_caching: bool,
    decode_sparse_method: str | None = None,
) -> ModelRuntimeCompatibility:
    model_type = str(model_type or "").strip().lower()
    method = normalize_sparse_method(sparse_method)
    decode_method = normalize_sparse_method(
        sparse_method if decode_sparse_method is None else decode_sparse_method
    )
    model_types = SPARSE_METHOD_MODEL_TYPES.get(method)
    if model_types is not None and model_type not in model_types:
        raise NotImplementedError(
            f"{method} requires one of {sorted(model_types)}; "
            f"the auxiliary reconstruction lifecycle for {model_type!r} is not implemented."
        )
    compatibility = MODEL_RUNTIME_COMPATIBILITY.get(model_type)
    if model_type == "qwen3_moe" and topology.attn_tp_size > 1:
        compatibility = QWEN3_MOE_TP_EP_COMPATIBILITY
    if compatibility is None:
        raise NotImplementedError(
            f"Unsupported SparseEngine model_type={model_type!r}."
        )

    if bool(decode_graph) and decode_method not in compatibility.decode_graph_methods:
        supported = ", ".join(
            "'vanilla'" if item == "" else repr(item)
            for item in sorted(compatibility.decode_graph_methods)
        )
        raise ValueError(
            f"{model_type} v1 decode_graph is validated only for {supported}; "
            f"got method={decode_method!r}."
        )
    if method not in compatibility.sparse_methods:
        supported = ", ".join(
            "'vanilla'" if item == "" else repr(item)
            for item in sorted(compatibility.sparse_methods)
        )
        raise ValueError(
            f"Unsupported {model_type} sparse method "
            f"{method!r}; validated methods: {supported}."
        )
    if bool(enable_prefix_caching) and method not in compatibility.prefix_cache_methods:
        supported = ", ".join(
            "'vanilla'" if item == "" else repr(item)
            for item in sorted(compatibility.prefix_cache_methods)
        )
        raise ValueError(
            f"{model_type} prefix caching is validated only for {supported}; got method={method!r}."
        )
    return compatibility


def get_default_prefill_schedule_policy(method: str | None) -> str:
    normalized = normalize_sparse_method(method)
    if normalized not in _DEFAULT_PREFILL_POLICY_BY_METHOD:
        supported = ", ".join(repr(name) for name in sorted(CANONICAL_SPARSE_METHODS) if name)
        aliases = ", ".join(repr(name) for name in sorted(SUPPORTED_SPARSE_METHOD_ALIASES))
        raise ValueError(
            f"Unsupported sparse_method={method!r}. Supported methods: '', {supported}. "
            f"Supported aliases: {aliases}."
        )
    return _DEFAULT_PREFILL_POLICY_BY_METHOD[normalized]


def resolve_prefill_schedule_policy(method: str | None, policy: str | None) -> str:
    default_policy = get_default_prefill_schedule_policy(method)
    if policy is None:
        return default_policy

    requested = str(policy).strip().lower()
    if requested in {"", PREFILL_POLICY_AUTO}:
        return default_policy
    if requested not in SUPPORTED_PREFILL_POLICIES:
        supported = ", ".join(repr(name) for name in sorted(SUPPORTED_PREFILL_POLICIES))
        raise ValueError(
            f"Unsupported prefill_schedule_policy={policy!r}. Supported policies: {supported}, "
            f"or {PREFILL_POLICY_AUTO!r}."
        )
    if requested != default_policy:
        raise ValueError(
            "prefill_schedule_policy must match the registry default for reproducibility. "
            f"method={normalize_sparse_method(method)!r} requested={requested!r} default={default_policy!r}."
        )
    return requested
