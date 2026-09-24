from __future__ import annotations

import torch
from torch import nn

from sparseengine.method_registry import (
    normalize_sparse_method,
    QUANTIZED_KV_METHODS,
    resolve_cache_sparse_method,
    resolve_prefill_sparse_method,
    resolve_sparse_prefill_score_mode,
    sparse_decode_attention_requires_scores,
    sparse_prefill_attention_contract,
)
from sparseengine.operators.decode_attention import (
    DecodeAttentionOpSpec,
    PreparedDecodeAttentionOp,
    prepare_decode_attention_op,
)
from sparseengine.operators.full_attention import (
    FullAttentionOpSpec,
    FullAttentionProvider,
    prepare_full_attention_provider,
)
from sparseengine.operators.moe import model_activation_dtype
from sparseengine.operators.prefill_attention import (
    FlashPrefillV2Semantics,
    PreparedPrefillAttentionOp,
    PrefillAttentionOpSpec,
    prepare_prefill_attention_op,
)


def resolve_mha_head_dim(config) -> int:
    explicit = getattr(config, "head_dim", None)
    if explicit is not None:
        head_dim = int(explicit)
        if head_dim <= 0:
            raise ValueError(f"MHA head_dim must be positive, got {head_dim}.")
        return head_dim

    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    if num_attention_heads <= 0 or hidden_size % num_attention_heads:
        raise ValueError(
            "Cannot infer MHA head_dim from hidden_size / num_attention_heads: "
            f"hidden_size={hidden_size} num_attention_heads={num_attention_heads}."
        )
    return hidden_size // num_attention_heads


def _resolve_mha_local_shape(
    config,
    *,
    attention_tp_size: int,
) -> tuple[int, int, int, torch.dtype]:
    tp_size = int(attention_tp_size)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    if query_heads % tp_size or kv_heads % tp_size:
        raise ValueError(
            "MHA query and KV heads must be divisible by attention TP size: "
            f"query_heads={query_heads} kv_heads={kv_heads} tp_size={tp_size}."
        )
    return (
        query_heads // tp_size,
        kv_heads // tp_size,
        resolve_mha_head_dim(config),
        model_activation_dtype(config),
    )


def build_mha_prefill_attention_spec(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    runtime_config=None,
) -> PrefillAttentionOpSpec:
    query_heads, kv_heads, head_dim, activation_dtype = _resolve_mha_local_shape(
        config,
        attention_tp_size=attention_tp_size,
    )
    normalized_method = normalize_sparse_method(sparse_method)
    score_config = config if runtime_config is None else runtime_config
    prefill_sparse_method = resolve_prefill_sparse_method(
        getattr(score_config, "prefill_sparse_method", None),
        sparse_method=normalized_method,
    )
    cache_method = resolve_cache_sparse_method(
        normalized_method,
        prefill_sparse_method=prefill_sparse_method,
    )
    score_mode = resolve_sparse_prefill_score_mode(
        cache_method,
        getattr(score_config, "sparse_prefill_score_mode", None),
    )
    contract = sparse_prefill_attention_contract(
        normalized_method,
        prefill_sparse_method=prefill_sparse_method,
        sparse_prefill_score_mode=score_mode,
        h2o_prefill_score_window=getattr(
            score_config, "h2o_prefill_score_window", 128
        ),
    )
    flashprefill_v2 = None
    if prefill_sparse_method == "flashprefill_v2":
        flashprefill_v2 = FlashPrefillV2Semantics(
            k_block_m=int(score_config.flashprefill_v2_k_block_m),
            k_block_n=int(score_config.flashprefill_v2_k_block_n),
            abs_threshold=float(score_config.flashprefill_v2_abs_threshold),
            attention_sink_blocks=int(
                score_config.flashprefill_v2_attention_sink_blocks
            ),
            window_blocks=int(score_config.flashprefill_v2_window_blocks),
            last_query_blocks=int(
                score_config.flashprefill_v2_last_query_blocks
            ),
            min_sparse_q_len=int(
                score_config.flashprefill_v2_min_sparse_q_len
            ),
            use_mean_correction=bool(
                score_config.flashprefill_v2_use_mean_correction
            ),
        )
    return PrefillAttentionOpSpec(
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        activation_dtype=activation_dtype,
        softmax_scale=head_dim**-0.5,
        causal=True,
        page_size=1,
        score_output=contract.main_score_kind,
        optional_score_output=contract.optional_score_output,
        layer_varying_page_table=contract.layer_varying_page_table,
        return_softmax_lse=(
            cache_method == "h2o"
            and prefill_sparse_method != "flashprefill_v2"
            and score_mode == "probability"
        ),
        allow_softmax_lse_fallback=(
            cache_method == "h2o"
            and prefill_sparse_method != "flashprefill_v2"
            and score_mode == "probability"
        ),
        prefill_sparse_method=prefill_sparse_method,
        flashprefill_v2=flashprefill_v2,
    )


def build_mha_prefill_attention_op(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
    runtime_config=None,
) -> PreparedPrefillAttentionOp:
    return prepare_prefill_attention_op(
        build_mha_prefill_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            runtime_config=runtime_config,
        ),
        device_index=int(device.index or 0),
    )


def build_mha_decode_attention_spec(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    max_batch_size: int,
    cuda_graph: bool,
    runtime_config=None,
) -> DecodeAttentionOpSpec:
    query_heads, kv_heads, head_dim, activation_dtype = _resolve_mha_local_shape(
        config,
        attention_tp_size=attention_tp_size,
    )
    normalized_method = normalize_sparse_method(sparse_method)
    cache_method = resolve_cache_sparse_method(
        normalized_method,
        prefill_sparse_method=getattr(
            runtime_config,
            "prefill_sparse_method",
            None,
        ),
    )
    requires_decode_scores = sparse_decode_attention_requires_scores(
        normalized_method,
        h2o_decode_eviction=getattr(runtime_config, "h2o_decode_eviction", False),
    )
    return DecodeAttentionOpSpec(
        kv_storage_format=cache_method if cache_method in QUANTIZED_KV_METHODS else "dense",
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        activation_dtype=activation_dtype,
        softmax_scale=head_dim**-0.5,
        max_batch_size=int(max_batch_size),
        causal=True,
        page_size=(
            int(getattr(runtime_config, "quest_chunk_size", 16))
            if normalized_method == "quest"
            else 1
        ),
        # Score demand can change between decode steps for sparse methods.
        # Bind the score-capable implementation up front instead of
        # switching providers in the runtime path.
        may_require_attention_scores=requires_decode_scores,
        layer_varying_page_table=bool(cache_method),
        cuda_graph=bool(cuda_graph),
        h2o_layerwise_probability_scores=(
            normalized_method == "h2o" and requires_decode_scores
            and not getattr(runtime_config, "h2o_decode_score_fusion", True)
        ),
        h2o_headwise_logits=(
            normalized_method == "h2o" and requires_decode_scores
            and getattr(runtime_config, "h2o_decode_score_fusion", True)
        ),
        context_capacity=int(getattr(runtime_config, "max_model_len", 0) or 0)
        or None,
        sparse_context_budget=(
            int(getattr(runtime_config, "quest_token_budget", 2080))
            if normalized_method == "quest"
            else None
        ),
        may_use_full_layer_kivi_int4=(
            normalized_method == "deltakv"
            and int(
                getattr(runtime_config, "full_layer_kv_quant_bits", 0) or 0
            )
            == 4
            and bool(
                getattr(runtime_config, "enable_full_layer_kivi_quant", True)
            )
        ),
        full_layer_kivi_decode_block_seq=int(
            getattr(runtime_config, "full_layer_kivi_decode_block_seq", 256)
            or 256
        ),
        full_layer_kivi_decode_block_n=int(
            getattr(runtime_config, "full_layer_kivi_decode_block_n", 16) or 16
        ),
        full_layer_kivi_decode_num_warps=int(
            getattr(runtime_config, "full_layer_kivi_decode_num_warps", 2) or 2
        ),
        full_layer_kivi_decode_num_stages=int(
            getattr(runtime_config, "full_layer_kivi_decode_num_stages", 3) or 3
        ),
    )


def build_mha_decode_attention_op(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
    max_batch_size: int,
    cuda_graph: bool,
    runtime_config=None,
) -> PreparedDecodeAttentionOp:
    return prepare_decode_attention_op(
        build_mha_decode_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            max_batch_size=max_batch_size,
            cuda_graph=cuda_graph,
            runtime_config=runtime_config,
        ),
        device_index=int(device.index or 0),
    )


def build_mha_full_attention_provider(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
    max_batch_size: int,
    cuda_graph: bool,
    runtime_config=None,
) -> FullAttentionProvider:
    if normalize_sparse_method(sparse_method) == "palu":
        from sparseengine.operators.palu_attention import PaluFullAttentionProvider

        return PaluFullAttentionProvider(config, runtime_config, device=device)
    spec = FullAttentionOpSpec(
        prefill=build_mha_prefill_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            runtime_config=runtime_config,
        ),
        decode=build_mha_decode_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            max_batch_size=max_batch_size,
            cuda_graph=cuda_graph,
            runtime_config=runtime_config,
        ),
    )
    return prepare_full_attention_provider(
        spec,
        device_index=int(device.index or 0),
    )


def bind_mha_full_attention_provider(
    model: nn.Module,
    provider: FullAttentionProvider,
) -> int:
    return provider.bind(model)


__all__ = [
    "bind_mha_full_attention_provider",
    "build_mha_decode_attention_op",
    "build_mha_decode_attention_spec",
    "build_mha_full_attention_provider",
    "build_mha_prefill_attention_op",
    "build_mha_prefill_attention_spec",
    "resolve_mha_head_dim",
]
