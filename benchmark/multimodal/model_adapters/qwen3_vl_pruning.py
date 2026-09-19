from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch


@dataclass(frozen=True)
class Qwen3VLPruningConfig:
    method: str
    keep_ratio: float
    density_neighbors: int = 4
    temporal_segments: int = 8
    context_fraction: float = 0.35
    fastv_layer: int = 3


def _select_divprune(features: torch.Tensor, keep: int) -> torch.Tensor:
    count = int(features.shape[0])
    keep = max(1, min(int(keep), count))
    if keep >= count:
        return torch.arange(count, device=features.device)

    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    distances = 1.0 - normalized @ normalized.transpose(0, 1)
    selected = torch.empty((keep,), dtype=torch.long, device=features.device)
    min_dist = None
    for idx in range(keep):
        if idx == 0:
            scores = torch.topk(distances, 2, dim=0, largest=False).values[1]
        else:
            current = distances.index_select(0, selected[:idx])
            scores = current.min(dim=0).values
            scores[selected[:idx]] = -1.0
        chosen = torch.argmax(scores)
        selected[idx] = chosen
        if min_dist is not None:
            min_dist = torch.minimum(min_dist, distances[chosen])
    return selected.sort().values


def _select_divprune_official(features: torch.Tensor, keep: int) -> torch.Tensor:
    count = int(features.shape[0])
    keep = max(1, min(int(keep), count))
    if keep >= count:
        return torch.arange(count, device=features.device)

    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    distances = 1.0 - normalized @ normalized.transpose(0, 1)
    selected = torch.empty((keep,), dtype=torch.long, device=features.device)
    for idx in range(keep):
        if idx == 0:
            scores = torch.topk(distances, 2, dim=0, largest=False).values[1]
        else:
            selected_rows = selected.index_select(0, torch.arange(idx, device=features.device))
            scores = distances.index_select(0, selected_rows).min(dim=0).values
        selected[idx] = torch.argmax(scores)
    return selected.sort().values


def _density_centers(features: torch.Tensor, keep: int, neighbors: int) -> torch.Tensor:
    count = int(features.shape[0])
    keep = max(0, min(int(keep), count))
    if keep <= 0:
        return torch.empty((0,), dtype=torch.long, device=features.device)
    if keep >= count:
        return torch.arange(count, device=features.device)

    work = features.float()
    dim = max(1, int(work.shape[-1]))
    distances = torch.cdist(work, work) / (dim**0.5)
    k = max(1, min(int(neighbors), count))
    nearest = torch.topk(distances, k=k, dim=-1, largest=False).values
    density = (-(nearest**2).mean(dim=-1)).exp()
    density = density + torch.arange(count, device=features.device, dtype=density.dtype) * 1e-12
    higher_density = density[:, None] > density[None, :]
    dist_max = distances.max()
    parent_dist = torch.where(higher_density, distances, dist_max).min(dim=-1).values
    score = density * parent_dist
    return torch.topk(score, k=keep, largest=True).indices.sort().values


def _infer_video_frame_shape(video_grid_thw: torch.Tensor | None, visual_count: int) -> tuple[int, int]:
    if video_grid_thw is None or video_grid_thw.numel() == 0:
        return 1, visual_count
    grid = video_grid_thw.detach().cpu().long()
    total = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
    if total != visual_count or int(grid.shape[0]) != 1:
        return 1, visual_count
    frames = max(1, int(grid[0, 0].item()))
    per_frame = max(1, visual_count // frames)
    if frames * per_frame != visual_count:
        return 1, visual_count
    return frames, per_frame


def _select_fastvid(
    visual_features: torch.Tensor,
    text_features: torch.Tensor,
    keep: int,
    video_grid_thw: torch.Tensor | None,
    cfg: Qwen3VLPruningConfig,
) -> torch.Tensor:
    total = int(visual_features.shape[0])
    keep = max(1, min(int(keep), total))
    if keep >= total:
        return torch.arange(total, device=visual_features.device)

    frames, per_frame = _infer_video_frame_shape(video_grid_thw, total)
    if frames <= 1:
        return _density_centers(visual_features, keep, cfg.density_neighbors)

    visual = visual_features.reshape(frames, per_frame, -1)
    frame_global = torch.nn.functional.normalize(visual.float().mean(dim=1), dim=-1)
    if frames == 1:
        segment_sizes = [1]
    else:
        adjacent_sim = (frame_global[:-1] * frame_global[1:]).sum(dim=-1)
        cuts_by_count = torch.topk(
            adjacent_sim,
            k=max(0, min(cfg.temporal_segments - 1, frames - 1)),
            largest=False,
        ).indices
        cuts = torch.unique(cuts_by_count.sort().values)
        segment_sizes = []
        start = 0
        for cut in cuts.tolist():
            end = int(cut) + 1
            if end > start:
                segment_sizes.append(end - start)
            start = end
        if start < frames:
            segment_sizes.append(frames - start)
        if not segment_sizes:
            segment_sizes = [frames]

    text_anchor = torch.nn.functional.normalize(text_features.float().mean(dim=0, keepdim=True), dim=-1)
    selected_chunks = []
    frame_start = 0
    remaining_keep = keep
    remaining_frames = frames
    for seg_idx, segment_size in enumerate(segment_sizes):
        seg_tokens = segment_size * per_frame
        seg_quota = min(remaining_keep, max(1, round(keep * (segment_size / frames))))
        if seg_idx == len(segment_sizes) - 1:
            seg_quota = remaining_keep
        remaining_keep -= seg_quota
        remaining_frames -= segment_size

        segment = visual[frame_start : frame_start + segment_size].reshape(seg_tokens, -1)
        salient = max(0, min(seg_quota, int(round(seg_quota * (1.0 - cfg.context_fraction)))))
        context = max(0, seg_quota - salient)
        relevance = (
            torch.nn.functional.normalize(segment.float(), dim=-1) @ text_anchor.squeeze(0)
        )
        salient_idx = (
            torch.topk(relevance, k=salient, largest=True).indices
            if salient > 0
            else torch.empty((0,), dtype=torch.long, device=segment.device)
        )
        if context > 0:
            all_idx = torch.arange(seg_tokens, device=segment.device)
            if salient_idx.numel() > 0:
                mask = torch.ones(seg_tokens, dtype=torch.bool, device=segment.device)
                mask[salient_idx] = False
                remaining_idx = all_idx[mask]
            else:
                remaining_idx = all_idx
            context_local = _density_centers(segment.index_select(0, remaining_idx), context, cfg.density_neighbors)
            context_idx = remaining_idx.index_select(0, context_local)
        else:
            context_idx = torch.empty((0,), dtype=torch.long, device=segment.device)
        selected = torch.cat([salient_idx, context_idx]).unique().sort().values
        selected_chunks.append(selected + frame_start * per_frame)
        frame_start += segment_size

    selected = torch.cat(selected_chunks).unique().sort().values
    if selected.numel() < keep:
        all_idx = torch.arange(total, device=visual_features.device)
        mask = torch.ones(total, dtype=torch.bool, device=visual_features.device)
        mask[selected] = False
        fill = all_idx[mask][: keep - selected.numel()]
        selected = torch.cat([selected, fill]).sort().values
    return selected[:keep].sort().values


def _build_keep_indices(
    inputs_embeds: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    video_grid_thw: torch.Tensor | None,
    cfg: Qwen3VLPruningConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs_embeds.shape[0] != 1:
        raise RuntimeError("Qwen3-VL pruning adapters currently require batch_size=1.")
    visual_mask = visual_pos_masks[0].to(dtype=torch.bool)
    visual_idx = torch.nonzero(visual_mask, as_tuple=False).flatten()
    if visual_idx.numel() == 0:
        keep = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        return keep, torch.arange(0, device=inputs_embeds.device)

    keep_visual = max(1, int(round(int(visual_idx.numel()) * float(cfg.keep_ratio))))
    visual_features = inputs_embeds[0].index_select(0, visual_idx)
    text_idx = torch.nonzero(~visual_mask, as_tuple=False).flatten()
    text_features = inputs_embeds[0].index_select(0, text_idx)
    if cfg.method == "divprune":
        selected_visual_local = _select_divprune(visual_features, keep_visual)
    elif cfg.method == "divprune_official":
        selected_visual_local = _select_divprune_official(visual_features, keep_visual)
    elif cfg.method == "fastvid":
        selected_visual_local = _select_fastvid(visual_features, text_features, keep_visual, video_grid_thw, cfg)
    else:
        raise RuntimeError(f"Unsupported Qwen3-VL pruning method: {cfg.method}")

    selected_visual = visual_idx.index_select(0, selected_visual_local)
    keep_mask = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device)
    keep_mask[visual_idx] = False
    keep_mask[selected_visual] = True
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    return keep_indices, selected_visual_local.sort().values


def _slice_tensor_sequence(value: torch.Tensor, keep_indices: torch.Tensor, seq_len: int) -> torch.Tensor:
    if value.shape[-1] != seq_len:
        raise RuntimeError(
            f"Cannot slice Qwen3-VL FastV tensor with last dim {value.shape[-1]} for seq_len={seq_len}."
        )
    return value.index_select(-1, keep_indices)


def _slice_attention_mask(mask: Any, keep_indices: torch.Tensor, seq_len: int) -> Any:
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        raise RuntimeError(f"Qwen3-VL FastV supports tensor/None attention masks, got {type(mask)!r}.")
    if mask.ndim == 2:
        return _slice_tensor_sequence(mask, keep_indices, seq_len)
    if mask.ndim == 4:
        out = mask
        if out.shape[-1] == seq_len:
            out = out.index_select(-1, keep_indices)
        if out.shape[-2] == seq_len:
            out = out.index_select(-2, keep_indices)
        return out
    raise RuntimeError(f"Qwen3-VL FastV does not support attention_mask.ndim={mask.ndim}.")


def _slice_position_embeddings(
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    keep_indices: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    if cos.shape[1] != seq_len or sin.shape[1] != seq_len:
        raise RuntimeError(
            f"Cannot slice Qwen3-VL FastV RoPE with cos/sin seq dims {cos.shape[1]}/{sin.shape[1]} "
            f"for seq_len={seq_len}."
        )
    return cos.index_select(1, keep_indices), sin.index_select(1, keep_indices)


def _select_fastv_keep_indices(
    attention_scores: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    keep_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if visual_pos_masks.shape[0] != 1:
        raise RuntimeError("Qwen3-VL FastV currently requires batch_size=1.")
    visual_mask = visual_pos_masks[0].to(dtype=torch.bool)
    visual_idx = torch.nonzero(visual_mask, as_tuple=False).flatten()
    if visual_idx.numel() == 0:
        keep = torch.arange(attention_scores.shape[0], device=attention_scores.device)
        return keep, torch.empty((0,), dtype=torch.long, device=attention_scores.device)

    keep_visual = max(1, int(round(int(visual_idx.numel()) * float(keep_ratio))))
    keep_visual = min(keep_visual, int(visual_idx.numel()))
    visual_scores = attention_scores.index_select(0, visual_idx)
    selected_visual_local = torch.topk(visual_scores, k=keep_visual, largest=True).indices.sort().values
    selected_visual = visual_idx.index_select(0, selected_visual_local)

    keep_mask = torch.ones(attention_scores.shape[0], dtype=torch.bool, device=attention_scores.device)
    keep_mask[visual_idx] = False
    keep_mask[selected_visual] = True
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    return keep_indices, selected_visual_local


def apply_qwen3_vl_prefill_pruning(model: Any, cfg: Qwen3VLPruningConfig) -> dict[str, Any]:
    if not (0.0 < float(cfg.keep_ratio) <= 1.0):
        raise ValueError("Qwen3-VL pruning keep_ratio must be in (0, 1].")
    if cfg.method not in {"divprune", "divprune_official", "fastvid"}:
        raise ValueError(f"Qwen3-VL prefill pruning supports divprune/divprune_official/fastvid, got {cfg.method!r}.")
    original_model_forward = model.model.forward
    language_model = model.model.language_model
    original_forward = language_model.forward
    owner = model.model

    def model_forward_with_grid_stash(self, *args, **kwargs):
        self._sparseengine_last_video_grid_thw = kwargs.get("video_grid_thw")
        return original_model_forward(*args, **kwargs)

    def forward_with_pruning(self, *args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        visual_pos_masks = kwargs.get("visual_pos_masks")
        past_key_values = kwargs.get("past_key_values")
        if (
            inputs_embeds is not None
            and visual_pos_masks is not None
            and inputs_embeds.shape[1] > 1
            and (past_key_values is None or past_key_values.get_seq_length() == 0)
            and cfg.keep_ratio < 1.0
        ):
            keep_indices, selected_visual_local = _build_keep_indices(
                inputs_embeds,
                visual_pos_masks,
                getattr(owner, "_sparseengine_last_video_grid_thw", None),
                cfg,
            )
            kwargs["inputs_embeds"] = inputs_embeds.index_select(1, keep_indices)
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None and attention_mask.ndim == 2:
                kwargs["attention_mask"] = attention_mask.index_select(1, keep_indices)
            position_ids = kwargs.get("position_ids")
            if position_ids is not None:
                kwargs["position_ids"] = position_ids.index_select(-1, keep_indices)
            kwargs["visual_pos_masks"] = visual_pos_masks.index_select(1, keep_indices)
            deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
            if deepstack_visual_embeds is not None:
                kwargs["deepstack_visual_embeds"] = [
                    embed.index_select(0, selected_visual_local) for embed in deepstack_visual_embeds
                ]
            owner._sparseengine_last_prune_stats = {
                "method": cfg.method,
                "keep_ratio": cfg.keep_ratio,
                "original_seq_len": int(inputs_embeds.shape[1]),
                "pruned_seq_len": int(kwargs["inputs_embeds"].shape[1]),
                "original_visual_tokens": int(visual_pos_masks.sum().item()),
                "kept_visual_tokens": int(selected_visual_local.numel()),
            }
        return original_forward(*args, **kwargs)

    model.model.forward = MethodType(model_forward_with_grid_stash, model.model)
    language_model.forward = MethodType(forward_with_pruning, language_model)
    return {
        "method": cfg.method,
        "selection_policy": (
            "official_divprune_greedy_max_min_projected_visual_tokens"
            if cfg.method == "divprune_official"
            else (
                "max_min_diversity_visual_tokens"
                if cfg.method == "divprune"
                else "dynamic_temporal_density_spatiotemporal_pruning"
            )
        ),
        "keep_ratio": cfg.keep_ratio,
        "divprune_source_repo": "vbdi/divprune@799e2d9" if cfg.method == "divprune_official" else None,
        "divprune_hf_feature_source": "prefill_projected_visual_embeddings" if cfg.method == "divprune_official" else None,
        "supports_batch_generation": False,
        "hook": "qwen3_vl_language_model_prefill_inputs_embeds",
    }


def apply_qwen3_vl_fastv(model: Any, cfg: Qwen3VLPruningConfig) -> dict[str, Any]:
    if not (0.0 < float(cfg.keep_ratio) <= 1.0):
        raise ValueError("Qwen3-VL FastV keep_ratio must be in (0, 1].")

    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        create_causal_mask,
        eager_attention_forward,
        repeat_kv,
    )

    owner = model.model
    language_model = model.model.language_model
    if not (0 <= int(cfg.fastv_layer) < len(language_model.layers)):
        raise ValueError(
            f"Qwen3-VL FastV layer must be in [0, {len(language_model.layers) - 1}], got {cfg.fastv_layer}."
        )
    original_model_forward = model.model.forward

    def model_forward_with_grid_stash(self, *args, **kwargs):
        self._sparseengine_last_video_grid_thw = kwargs.get("video_grid_thw")
        return original_model_forward(*args, **kwargs)

    def attention_forward_with_fastv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
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

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if torch.is_tensor(attention_mask) and attention_mask.ndim == 4:
            if attention_mask.shape[-1] != key_states.shape[-2]:
                attention_mask = attention_mask[..., -key_states.shape[-2] :]
            if attention_mask.shape[-2] != query_states.shape[-2]:
                attention_mask = attention_mask[..., -query_states.shape[-2] :, :]
        elif torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            if attention_mask.shape[-1] != key_states.shape[-2]:
                attention_mask = attention_mask[..., -key_states.shape[-2] :]

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    for layer in language_model.layers:
        layer.self_attn.forward = MethodType(attention_forward_with_fastv, layer.self_attn)

    def forward_with_fastv(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        prefill_active = (
            inputs_embeds is not None
            and visual_pos_masks is not None
            and inputs_embeds.shape[1] > 1
            and (past_key_values is None or past_key_values.get_seq_length() == 0)
            and cfg.keep_ratio < 1.0
        )

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rotary_position_ids = position_ids

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rotary_position_ids)
        owner._sparseengine_fastv_prefill_active = bool(prefill_active)
        owner._sparseengine_fastv_attention_scores = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if prefill_active and layer_idx == int(cfg.fastv_layer):
                scores = getattr(owner, "_sparseengine_fastv_attention_scores", None)
                if scores is None:
                    raise RuntimeError(
                        "Qwen3-VL FastV did not collect attention scores before the pruning layer."
                    )
                seq_len = int(hidden_states.shape[1])
                original_visual_tokens = int(visual_pos_masks.sum().item())
                keep_indices, selected_visual_local = _select_fastv_keep_indices(
                    scores.to(hidden_states.device),
                    visual_pos_masks,
                    cfg.keep_ratio,
                )
                hidden_states = hidden_states.index_select(1, keep_indices)
                attention_mask = _slice_attention_mask(attention_mask, keep_indices, seq_len)
                if text_position_ids is not None:
                    text_position_ids = text_position_ids.index_select(-1, keep_indices)
                rotary_position_ids = rotary_position_ids.index_select(-1, keep_indices)
                position_embeddings = _slice_position_embeddings(position_embeddings, keep_indices, seq_len)
                visual_pos_masks = visual_pos_masks.index_select(1, keep_indices)
                if deepstack_visual_embeds is not None:
                    deepstack_visual_embeds = [
                        embed.index_select(0, selected_visual_local) for embed in deepstack_visual_embeds
                    ]
                owner._sparseengine_last_prune_stats = {
                    "method": cfg.method,
                    "keep_ratio": cfg.keep_ratio,
                    "fastv_layer": int(cfg.fastv_layer),
                    "original_seq_len": seq_len,
                    "pruned_seq_len": int(hidden_states.shape[1]),
                    "original_visual_tokens": original_visual_tokens,
                    "kept_visual_tokens": int(selected_visual_local.numel()),
                }

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        owner._sparseengine_fastv_prefill_active = False
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    model.model.forward = MethodType(model_forward_with_grid_stash, model.model)
    language_model.forward = MethodType(forward_with_fastv, language_model)
    return {
        "method": cfg.method,
        "selection_policy": "fastv_last_token_attention_topk_visual_tokens",
        "keep_ratio": cfg.keep_ratio,
        "fastv_layer": int(cfg.fastv_layer),
        "supports_batch_generation": False,
        "hook": "qwen3_vl_language_model_layer_attention_kv_pruning",
    }
