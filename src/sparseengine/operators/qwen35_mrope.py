from __future__ import annotations

import torch
from torch import nn

from sparseengine.layers.rotary_embedding import (
    apply_rotary_emb,
    get_rope,
)


class Qwen35MRotaryEmbedding(nn.Module):
    """Dedicated Qwen3.5 M-RoPE path; 1-D text decode keeps FlashInfer RoPE."""

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        sections: list[int],
    ) -> None:
        super().__init__()
        if sum(sections) != rotary_dim // 2:
            raise ValueError(
                f"Qwen3.5 M-RoPE sections must sum to {rotary_dim // 2}, got {sections}."
            )
        self.rotary_dim = int(rotary_dim)
        self.sections = tuple(int(section) for section in sections)
        self.text_rope = get_rope(
            rotary_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=base,
        )

    def _multimodal_cos_sin(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.text_rope.cos_sin_cache[positions]
        cos, sin = cache.chunk(2, dim=-1)
        merged_cos = cos[0].clone()
        merged_sin = sin[0].clone()
        h_end = self.sections[1] * 3
        w_end = self.sections[2] * 3
        merged_cos[..., 1:h_end:3] = cos[1, ..., 1:h_end:3]
        merged_sin[..., 1:h_end:3] = sin[1, ..., 1:h_end:3]
        merged_cos[..., 2:w_end:3] = cos[2, ..., 2:w_end:3]
        merged_sin[..., 2:w_end:3] = sin[2, ..., 2:w_end:3]
        return merged_cos, merged_sin

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            if self.rotary_dim == query.shape[-1] == key.shape[-1]:
                return self.text_rope(positions, query, key)
            cos_sin = self.text_rope.cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
        elif positions.ndim == 2 and positions.shape[0] == 3:
            cos, sin = self._multimodal_cos_sin(positions)
        else:
            raise ValueError(
                f"Qwen3.5 positions must be [tokens] or [3, tokens], got {tuple(positions.shape)}."
            )
        query_rot = apply_rotary_emb(query[..., : self.rotary_dim], cos, sin)
        key_rot = apply_rotary_emb(key[..., : self.rotary_dim], cos, sin)
        return (
            torch.cat((query_rot, query[..., self.rotary_dim :]), dim=-1),
            torch.cat((key_rot, key[..., self.rotary_dim :]), dim=-1),
        )


__all__ = ["Qwen35MRotaryEmbedding"]
