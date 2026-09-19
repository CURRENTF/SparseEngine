from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch

from benchmark.multimodal.model_adapters.qwen3_vl_pruning import (
    _select_divprune,
    _select_divprune_official,
    _select_fastv_keep_indices,
    _slice_attention_mask,
    _slice_position_embeddings,
)


@dataclass(frozen=True)
class LlavaOneVisionPruningConfig:
    method: str
    keep_ratio: float
    density_neighbors: int = 4
    temporal_segments: int = 8
    context_fraction: float = 0.15625
    fastv_layer: int = 3


def _slice_cache_position(cache_position: torch.Tensor | None, keep_indices: torch.Tensor, seq_len: int) -> torch.Tensor | None:
    if cache_position is None:
        return None
    if not torch.is_tensor(cache_position):
        raise RuntimeError(f"LLaVA-OV pruning supports tensor/None cache_position, got {type(cache_position)!r}.")
    if cache_position.ndim == 1 and cache_position.shape[0] == seq_len:
        return cache_position.index_select(0, keep_indices)
    if cache_position.ndim == 2 and cache_position.shape[-1] == seq_len:
        return cache_position.index_select(-1, keep_indices)
    return cache_position


def _visual_mask_from_input_ids(model: Any, input_ids: torch.Tensor | None) -> torch.Tensor | None:
    if input_ids is None:
        return None
    image_token_id = getattr(model.config, "image_token_id", None)
    video_token_id = getattr(model.config, "video_token_id", None)
    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if image_token_id is not None:
        visual_mask |= input_ids == int(image_token_id)
    if video_token_id is not None:
        visual_mask |= input_ids == int(video_token_id)
    return visual_mask


def _bind_model_forward_inputs(original_forward: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> inspect.BoundArguments:
    signature = inspect.signature(original_forward)
    return signature.bind_partial(*args, **kwargs)


def _stash_visual_metadata(model: Any, original_forward: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    bound = _bind_model_forward_inputs(original_forward, args, kwargs)
    input_ids = bound.arguments.get("input_ids")
    model._sparseengine_last_visual_pos_masks = _visual_mask_from_input_ids(model, input_ids)
    pixel_values_videos = bound.arguments.get("pixel_values_videos")
    if torch.is_tensor(pixel_values_videos) and pixel_values_videos.ndim >= 2:
        model._sparseengine_last_video_frames = int(pixel_values_videos.shape[1])
    else:
        model._sparseengine_last_video_frames = None


def _build_keep_indices(
    inputs_embeds: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    cfg: LlavaOneVisionPruningConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs_embeds.shape[0] != 1:
        raise RuntimeError("LLaVA-OV pruning adapters currently require batch_size=1.")
    visual_mask = visual_pos_masks[0].to(device=inputs_embeds.device, dtype=torch.bool)
    visual_idx = torch.nonzero(visual_mask, as_tuple=False).flatten()
    if visual_idx.numel() == 0:
        keep = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        return keep, torch.empty((0,), dtype=torch.long, device=inputs_embeds.device)

    keep_visual = max(1, int(round(int(visual_idx.numel()) * float(cfg.keep_ratio))))
    visual_features = inputs_embeds[0].index_select(0, visual_idx)
    if cfg.method == "divprune":
        selected_visual_local = _select_divprune(visual_features, keep_visual)
    elif cfg.method == "divprune_official":
        selected_visual_local = _select_divprune_official(visual_features, keep_visual)
    else:
        raise RuntimeError(f"Unsupported LLaVA-OV prefill pruning method: {cfg.method}")

    selected_visual = visual_idx.index_select(0, selected_visual_local)
    keep_mask = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device)
    keep_mask[visual_idx] = False
    keep_mask[selected_visual] = True
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    return keep_indices, selected_visual_local.sort().values


def _select_visionzip_tokens(
    visual_features: torch.Tensor,
    cfg: LlavaOneVisionPruningConfig,
) -> tuple[torch.Tensor, dict[str, int]]:
    if visual_features.ndim != 2:
        raise RuntimeError(f"LLaVA-OV VisionZip expects [visual_tokens, hidden], got {tuple(visual_features.shape)}.")
    visual_tokens = int(visual_features.shape[0])
    if visual_tokens == 0:
        raise RuntimeError("LLaVA-OV VisionZip received an empty visual span.")
    total_keep = max(1, min(visual_tokens, int(round(visual_tokens * float(cfg.keep_ratio)))))
    if total_keep >= visual_tokens:
        return visual_features, {
            "original_visual_tokens": visual_tokens,
            "kept_visual_tokens": visual_tokens,
            "dominant_tokens": visual_tokens,
            "contextual_tokens": 0,
        }

    contextual_tokens = 0
    if total_keep > 1:
        contextual_tokens = max(1, int(round(total_keep * float(cfg.context_fraction))))
        contextual_tokens = min(contextual_tokens, total_keep - 1)
    dominant_tokens = total_keep - contextual_tokens

    features_fp32 = visual_features.float()
    salience = features_fp32.norm(dim=-1)
    dominant_local = salience.topk(dominant_tokens, largest=True, sorted=False).indices.sort().values
    dominant_mask = torch.zeros(visual_tokens, dtype=torch.bool, device=visual_features.device)
    dominant_mask[dominant_local] = True
    dominant_features = visual_features.index_select(0, dominant_local)

    if contextual_tokens == 0:
        compressed = dominant_features
    else:
        residual_features = visual_features[~dominant_mask]
        if residual_features.shape[0] <= contextual_tokens:
            contextual_features = residual_features
        else:
            metric = residual_features.float()
            metric = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            step = max(1, int(metric.shape[0]) // int(contextual_tokens))
            target_local = torch.arange(0, metric.shape[0], step, device=metric.device)[:contextual_tokens]
            if target_local.numel() != contextual_tokens:
                raise RuntimeError(
                    f"LLaVA-OV VisionZip target selection produced {target_local.numel()} "
                    f"targets for contextual_tokens={contextual_tokens}."
                )
            target_metric = metric.index_select(0, target_local)
            all_residual = torch.arange(metric.shape[0], device=metric.device)
            merge_mask = ~torch.isin(all_residual, target_local)
            merge_metric = metric[merge_mask]
            target_hidden = residual_features.index_select(0, target_local)
            if merge_metric.numel() == 0:
                contextual_features = target_hidden
            else:
                similarity = torch.matmul(merge_metric, target_metric.transpose(0, 1))
                assignment = similarity.argmax(dim=1)
                assign_one_hot = torch.zeros(
                    merge_metric.shape[0],
                    contextual_tokens,
                    dtype=features_fp32.dtype,
                    device=visual_features.device,
                )
                assign_one_hot.scatter_(1, assignment.unsqueeze(-1), 1.0)
                hidden_to_merge = residual_features[merge_mask].float()
                counts = assign_one_hot.sum(dim=0).clamp(min=1.0).unsqueeze(-1)
                aggregated = torch.matmul(assign_one_hot.transpose(0, 1), hidden_to_merge) / counts
                contextual_features = target_hidden + aggregated.to(target_hidden.dtype)
        compressed = torch.cat([dominant_features, contextual_features.to(visual_features.dtype)], dim=0)

    if int(compressed.shape[0]) != total_keep:
        raise RuntimeError(
            f"LLaVA-OV VisionZip produced {compressed.shape[0]} visual tokens, expected {total_keep}."
        )
    return compressed, {
        "original_visual_tokens": visual_tokens,
        "kept_visual_tokens": total_keep,
        "dominant_tokens": dominant_tokens,
        "contextual_tokens": contextual_tokens,
    }


def _ensure_single_contiguous_visual_span(visual_mask: torch.Tensor) -> tuple[int, int]:
    visual_idx = torch.nonzero(visual_mask, as_tuple=False).flatten()
    if visual_idx.numel() == 0:
        raise RuntimeError("LLaVA-OV VisionZip requires at least one visual token.")
    start = int(visual_idx[0].item())
    end = int(visual_idx[-1].item()) + 1
    expected = torch.arange(start, end, device=visual_idx.device)
    if visual_idx.numel() != expected.numel() or not torch.equal(visual_idx, expected):
        raise RuntimeError(
            "LLaVA-OV VisionZip currently expects one contiguous visual token span. "
            f"Got {visual_idx.numel()} visual tokens between positions {start} and {end - 1}."
        )
    return start, end


def _replace_visual_span_2d_mask(
    mask: torch.Tensor | None,
    start: int,
    end: int,
    compressed_len: int,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        raise RuntimeError(f"LLaVA-OV VisionZip supports tensor/None attention masks, got {type(mask)!r}.")
    if mask.ndim != 2 or mask.shape[0] != 1:
        raise RuntimeError(f"LLaVA-OV VisionZip expects a [1, seq] attention mask, got {tuple(mask.shape)}.")
    visual_values = mask[:, start:end]
    if visual_values.numel() == 0:
        raise RuntimeError("LLaVA-OV VisionZip cannot replace an empty attention-mask visual span.")
    fill_value = visual_values.max(dim=1, keepdim=True).values
    compressed_mask = fill_value.expand(mask.shape[0], compressed_len)
    return torch.cat([mask[:, :start], compressed_mask, mask[:, end:]], dim=1)


def _replace_visual_span_position_ids(
    position_ids: torch.Tensor | None,
    new_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if position_ids is None:
        return None
    if not torch.is_tensor(position_ids):
        raise RuntimeError(f"LLaVA-OV VisionZip supports tensor/None position_ids, got {type(position_ids)!r}.")
    if position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise RuntimeError(f"LLaVA-OV VisionZip expects [1, seq] position_ids, got {tuple(position_ids.shape)}.")
    return torch.arange(new_len, device=device, dtype=position_ids.dtype).unsqueeze(0)


def _replace_visual_span_cache_position(
    cache_position: torch.Tensor | None,
    new_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if cache_position is None:
        return None
    if not torch.is_tensor(cache_position):
        raise RuntimeError(f"LLaVA-OV VisionZip supports tensor/None cache_position, got {type(cache_position)!r}.")
    if cache_position.ndim != 1:
        raise RuntimeError(f"LLaVA-OV VisionZip expects 1D cache_position, got {tuple(cache_position.shape)}.")
    return torch.arange(new_len, device=device, dtype=cache_position.dtype)


def _maybe_apply_visionzip(
    owner: Any,
    cfg: LlavaOneVisionPruningConfig,
    kwargs: dict[str, Any],
) -> None:
    inputs_embeds = kwargs.get("inputs_embeds")
    past_key_values = kwargs.get("past_key_values")
    visual_pos_masks = getattr(owner, "_sparseengine_last_visual_pos_masks", None)
    prefill_active = (
        inputs_embeds is not None
        and visual_pos_masks is not None
        and inputs_embeds.shape[1] > 1
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
        and cfg.keep_ratio < 1.0
    )
    if not prefill_active:
        return
    if inputs_embeds.shape[0] != 1 or visual_pos_masks.shape[0] != 1:
        raise RuntimeError("LLaVA-OV VisionZip currently requires batch_size=1.")

    visual_mask = visual_pos_masks[0].to(device=inputs_embeds.device, dtype=torch.bool)
    start, end = _ensure_single_contiguous_visual_span(visual_mask)
    visual_features = inputs_embeds[0, start:end, :]
    compressed_visual, stats = _select_visionzip_tokens(visual_features, cfg)
    new_inputs_embeds = torch.cat(
        [
            inputs_embeds[:, :start, :],
            compressed_visual.unsqueeze(0).to(dtype=inputs_embeds.dtype),
            inputs_embeds[:, end:, :],
        ],
        dim=1,
    )
    new_len = int(new_inputs_embeds.shape[1])
    compressed_len = int(compressed_visual.shape[0])
    new_visual_mask = torch.cat(
        [
            visual_pos_masks[:, :start].to(device=inputs_embeds.device),
            torch.ones((1, compressed_len), dtype=torch.bool, device=inputs_embeds.device),
            visual_pos_masks[:, end:].to(device=inputs_embeds.device),
        ],
        dim=1,
    )

    kwargs["inputs_embeds"] = new_inputs_embeds
    kwargs["attention_mask"] = _replace_visual_span_2d_mask(kwargs.get("attention_mask"), start, end, compressed_len)
    kwargs["position_ids"] = _replace_visual_span_position_ids(kwargs.get("position_ids"), new_len, inputs_embeds.device)
    kwargs["cache_position"] = _replace_visual_span_cache_position(kwargs.get("cache_position"), new_len, inputs_embeds.device)
    owner._sparseengine_last_visual_pos_masks = new_visual_mask
    owner._sparseengine_last_prune_stats = {
        "method": cfg.method,
        "keep_ratio": cfg.keep_ratio,
        "context_fraction": cfg.context_fraction,
        "original_seq_len": int(inputs_embeds.shape[1]),
        "pruned_seq_len": new_len,
        **stats,
    }


def _maybe_prune_language_inputs(
    owner: Any,
    cfg: LlavaOneVisionPruningConfig,
    kwargs: dict[str, Any],
) -> None:
    inputs_embeds = kwargs.get("inputs_embeds")
    past_key_values = kwargs.get("past_key_values")
    visual_pos_masks = getattr(owner, "_sparseengine_last_visual_pos_masks", None)
    prefill_active = (
        inputs_embeds is not None
        and visual_pos_masks is not None
        and inputs_embeds.shape[1] > 1
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
        and cfg.keep_ratio < 1.0
    )
    if not prefill_active:
        return
    seq_len = int(inputs_embeds.shape[1])
    keep_indices, selected_visual_local = _build_keep_indices(
        inputs_embeds,
        visual_pos_masks,
        cfg,
    )
    kwargs["inputs_embeds"] = inputs_embeds.index_select(1, keep_indices)
    kwargs["attention_mask"] = _slice_attention_mask(kwargs.get("attention_mask"), keep_indices, seq_len)
    position_ids = kwargs.get("position_ids")
    if position_ids is not None:
        kwargs["position_ids"] = position_ids.index_select(-1, keep_indices)
    kwargs["cache_position"] = _slice_cache_position(kwargs.get("cache_position"), keep_indices, seq_len)
    owner._sparseengine_last_visual_pos_masks = visual_pos_masks.index_select(1, keep_indices)
    owner._sparseengine_last_prune_stats = {
        "method": cfg.method,
        "keep_ratio": cfg.keep_ratio,
        "original_seq_len": seq_len,
        "pruned_seq_len": int(kwargs["inputs_embeds"].shape[1]),
        "original_visual_tokens": int(visual_pos_masks.sum().item()),
        "kept_visual_tokens": int(selected_visual_local.numel()),
    }


def apply_llava_onevision_prefill_pruning(model: Any, cfg: LlavaOneVisionPruningConfig) -> dict[str, Any]:
    if not (0.0 < float(cfg.keep_ratio) <= 1.0):
        raise ValueError("LLaVA-OV pruning keep_ratio must be in (0, 1].")
    if cfg.method not in {"divprune", "divprune_official"}:
        raise ValueError(
            f"LLaVA-OV prefill pruning supports divprune/divprune_official, got {cfg.method!r}."
        )

    original_model_forward = model.model.forward
    language_model = model.model.language_model
    original_language_forward = language_model.forward
    owner = model.model

    def model_forward_with_visual_stash(self, *args, **kwargs):
        _stash_visual_metadata(self, original_model_forward, args, kwargs)
        return original_model_forward(*args, **kwargs)

    def language_forward_with_pruning(self, *args, **kwargs):
        if args:
            raise RuntimeError("LLaVA-OV pruning adapter expects language_model.forward keyword arguments.")
        _maybe_prune_language_inputs(owner, cfg, kwargs)
        return original_language_forward(*args, **kwargs)

    model.model.forward = MethodType(model_forward_with_visual_stash, model.model)
    language_model.forward = MethodType(language_forward_with_pruning, language_model)
    return {
        "method": cfg.method,
        "selection_policy": (
            "official_divprune_greedy_max_min_projected_visual_tokens"
            if cfg.method == "divprune_official"
            else "max_min_diversity_visual_tokens"
        ),
        "keep_ratio": cfg.keep_ratio,
        "divprune_source_repo": "vbdi/divprune@799e2d9" if cfg.method == "divprune_official" else None,
        "divprune_hf_feature_source": "prefill_projected_visual_embeddings" if cfg.method == "divprune_official" else None,
        "supports_batch_generation": False,
        "hook": "llava_onevision_qwen2_language_model_prefill_inputs_embeds",
    }


def apply_llava_onevision_visionzip(model: Any, cfg: LlavaOneVisionPruningConfig) -> dict[str, Any]:
    if not (0.0 < float(cfg.keep_ratio) <= 1.0):
        raise ValueError("LLaVA-OV VisionZip keep_ratio must be in (0, 1].")
    if cfg.method != "visionzip":
        raise ValueError(f"LLaVA-OV VisionZip adapter got unsupported method {cfg.method!r}.")
    if not (0.0 <= float(cfg.context_fraction) < 1.0):
        raise ValueError("LLaVA-OV VisionZip context_fraction must be in [0, 1).")

    original_model_forward = model.model.forward
    language_model = model.model.language_model
    original_language_forward = language_model.forward
    owner = model.model

    def model_forward_with_visual_stash(self, *args, **kwargs):
        _stash_visual_metadata(self, original_model_forward, args, kwargs)
        return original_model_forward(*args, **kwargs)

    def language_forward_with_visionzip(self, *args, **kwargs):
        if args:
            raise RuntimeError("LLaVA-OV VisionZip adapter expects language_model.forward keyword arguments.")
        _maybe_apply_visionzip(owner, cfg, kwargs)
        return original_language_forward(*args, **kwargs)

    model.model.forward = MethodType(model_forward_with_visual_stash, model.model)
    language_model.forward = MethodType(language_forward_with_visionzip, language_model)
    return {
        "method": cfg.method,
        "selection_policy": "visionzip_hf_projected_visual_norm_dominant_contextual_merge",
        "keep_ratio": cfg.keep_ratio,
        "context_fraction": cfg.context_fraction,
        "source_repo": "dvlab-research/VisionZip@8f86b55",
        "source_paper": "VisionZip: Longer is Better but Not Necessary in Vision Language Models",
        "hf_port_note": (
            "LLaVA-OneVision HF port applies VisionZip-style dominant/contextual token compression "
            "on projected visual embeddings; it is not the official LLaVA-1.5 CLIP-attention code path."
        ),
        "supports_batch_generation": False,
        "hook": "llava_onevision_qwen2_language_model_prefill_projected_visual_embeddings",
    }


def apply_llava_onevision_fastv(model: Any, cfg: LlavaOneVisionPruningConfig) -> dict[str, Any]:
    if not (0.0 < float(cfg.keep_ratio) <= 1.0):
        raise ValueError("LLaVA-OV FastV keep_ratio must be in (0, 1].")

    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
        repeat_kv,
    )

    original_model_forward = model.model.forward
    owner = model.model
    language_model = model.model.language_model
    if not (0 <= int(cfg.fastv_layer) < len(language_model.layers)):
        raise ValueError(
            f"LLaVA-OV FastV layer must be in [0, {len(language_model.layers) - 1}], got {cfg.fastv_layer}."
        )

    def model_forward_with_visual_stash(self, *args, **kwargs):
        _stash_visual_metadata(self, original_model_forward, args, kwargs)
        return original_model_forward(*args, **kwargs)

    def attention_forward_with_fastv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if (
            getattr(owner, "_sparseengine_fastv_prefill_active", False)
            and self.layer_idx == int(cfg.fastv_layer) - 1
            and hidden_states.shape[1] > 1
        ):
            score_key_states = repeat_kv(key_states, self.num_key_value_groups)
            score = torch.matmul(
                query_states[:, :, -1:, :].float(),
                score_key_states.transpose(-2, -1).float(),
            ) * float(self.scaling)
            if torch.is_tensor(attention_mask) and attention_mask.ndim == 4:
                score = score + attention_mask[:, :, -1:, :].float()
            owner._sparseengine_fastv_attention_scores = torch.softmax(score, dim=-1).mean(dim=1)[0, 0].detach()

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if torch.is_tensor(attention_mask) and attention_mask.ndim == 4:
            if attention_mask.shape[-1] != key_states.shape[-2]:
                attention_mask = attention_mask[..., -key_states.shape[-2] :]
            if attention_mask.shape[-2] != query_states.shape[-2]:
                attention_mask = attention_mask[..., -query_states.shape[-2] :, :]
        elif torch.is_tensor(attention_mask) and attention_mask.ndim == 2 and attention_mask.shape[-1] != key_states.shape[-2]:
            attention_mask = attention_mask[..., -key_states.shape[-2] :]

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    for layer in language_model.layers:
        layer.self_attn.forward = MethodType(attention_forward_with_fastv, layer.self_attn)

    def language_forward_with_fastv(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **flash_attn_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise RuntimeError("LLaVA-OV FastV currently requires batch_size=1.")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        visual_pos_masks = getattr(owner, "_sparseengine_last_visual_pos_masks", None)
        prefill_active = (
            visual_pos_masks is not None
            and inputs_embeds.shape[1] > 1
            and (past_key_values is None or past_key_values.get_seq_length() == 0)
            and cfg.keep_ratio < 1.0
        )

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        owner._sparseengine_fastv_prefill_active = bool(prefill_active)
        owner._sparseengine_fastv_attention_scores = None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if prefill_active and layer_idx == int(cfg.fastv_layer):
                scores = getattr(owner, "_sparseengine_fastv_attention_scores", None)
                if scores is None:
                    raise RuntimeError("LLaVA-OV FastV did not collect attention scores before the pruning layer.")
                seq_len = int(hidden_states.shape[1])
                original_visual_tokens = int(visual_pos_masks.sum().item())
                keep_indices, selected_visual_local = _select_fastv_keep_indices(
                    scores.to(hidden_states.device),
                    visual_pos_masks.to(hidden_states.device),
                    cfg.keep_ratio,
                )
                hidden_states = hidden_states.index_select(1, keep_indices)
                position_ids = position_ids.index_select(-1, keep_indices)
                cache_position = _slice_cache_position(cache_position, keep_indices, seq_len)
                position_embeddings = _slice_position_embeddings(position_embeddings, keep_indices, seq_len)
                visual_pos_masks = visual_pos_masks.index_select(1, keep_indices)
                owner._sparseengine_last_visual_pos_masks = visual_pos_masks
                causal_mask_mapping = {
                    name: _slice_attention_mask(mask, keep_indices, seq_len)
                    for name, mask in causal_mask_mapping.items()
                }
                owner._sparseengine_last_prune_stats = {
                    "method": cfg.method,
                    "keep_ratio": cfg.keep_ratio,
                    "fastv_layer": int(cfg.fastv_layer),
                    "original_seq_len": seq_len,
                    "pruned_seq_len": int(hidden_states.shape[1]),
                    "original_visual_tokens": original_visual_tokens,
                    "kept_visual_tokens": int(selected_visual_local.numel()),
                }

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        owner._sparseengine_fastv_prefill_active = False
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    model.model.forward = MethodType(model_forward_with_visual_stash, model.model)
    language_model.forward = MethodType(language_forward_with_fastv, language_model)
    return {
        "method": cfg.method,
        "selection_policy": "fastv_last_token_attention_topk_visual_tokens",
        "keep_ratio": cfg.keep_ratio,
        "fastv_layer": int(cfg.fastv_layer),
        "supports_batch_generation": False,
        "hook": "llava_onevision_qwen2_language_model_layer_attention_kv_pruning",
    }
