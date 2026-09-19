from __future__ import annotations

import torch
from torch import nn
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel,
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
)

from sparseengine.multimodal.runtime import MultiModalState


class Gemma4MultimodalEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        text_config = config.text_config
        self.vision_tower = (
            Gemma4VisionModel(config.vision_config)
            if config.vision_config is not None
            else None
        )
        self.embed_vision = (
            Gemma4MultimodalEmbedder(config.vision_config, text_config)
            if config.vision_config is not None
            else None
        )
        self.audio_tower = (
            Gemma4AudioModel(config.audio_config)
            if config.audio_config is not None
            else None
        )
        self.embed_audio = (
            Gemma4MultimodalEmbedder(config.audio_config, text_config)
            if config.audio_config is not None
            else None
        )

    def _vision_features(
        self,
        pixels: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        video: bool,
    ) -> torch.Tensor:
        if self.vision_tower is None or self.embed_vision is None:
            raise ValueError("This Gemma 4 checkpoint has no vision tower.")
        device = next(self.vision_tower.parameters()).device
        if video:
            pixels = pixels.flatten(0, 1)
            position_ids = position_ids.flatten(0, 1)
        output = self.vision_tower(
            pixel_values=pixels.to(device=device, dtype=self.vision_tower.dtype),
            pixel_position_ids=position_ids.to(device),
            return_dict=True,
        )
        return self.embed_vision(output.last_hidden_state)

    @torch.inference_mode()
    def encode(
        self,
        input_ids: list[int],
        tensors: dict[str, torch.Tensor],
    ) -> MultiModalState:
        del input_ids
        device = next(self.parameters()).device
        type_ids = tensors["mm_token_type_ids"].squeeze(0).to(device)
        embeddings = {}
        if "pixel_values" in tensors:
            embeddings[1] = self._vision_features(
                tensors["pixel_values"], tensors["image_position_ids"], video=False
            )
        if "pixel_values_videos" in tensors:
            embeddings[2] = self._vision_features(
                tensors["pixel_values_videos"], tensors["video_position_ids"], video=True
            )
        if "input_features" in tensors:
            if self.audio_tower is None or self.embed_audio is None:
                raise ValueError("This Gemma 4 checkpoint has no audio tower.")
            output = self.audio_tower(
                tensors["input_features"].to(device=device, dtype=self.audio_tower.dtype),
                tensors["input_features_mask"].to(device),
                return_dict=True,
            )
            features = self.embed_audio(output.last_hidden_state)
            embeddings[3] = features[output.attention_mask.to(device=features.device)]
        return MultiModalState(type_ids, embeddings, None, 0)


__all__ = ["Gemma4MultimodalEncoder"]
