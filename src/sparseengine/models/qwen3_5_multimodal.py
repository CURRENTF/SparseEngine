from __future__ import annotations

import itertools

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from sparseengine.multimodal.runtime import MultiModalState


def _vision_positions(
    start: int,
    grid: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    t, h, w = [int(value) for value in grid.tolist()]
    h //= spatial_merge_size
    w //= spatial_merge_size
    temporal = torch.arange(t, device=grid.device) + start
    height = torch.arange(h, device=grid.device) + start
    width = torch.arange(w, device=grid.device) + start
    values = torch.meshgrid(temporal, height, width, indexing="ij")
    return torch.stack(values).reshape(3, -1)


def qwen35_mrope_positions(
    type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        ).clone()
        video_grid_thw[:, 0] = 1
    grids = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }
    current = 0
    segments = []
    for modality, group in itertools.groupby(
        enumerate(type_ids.tolist()), lambda item: item[1]
    ):
        group = list(group)
        length = group[-1][0] - group[0][0] + 1
        if modality == 0:
            segment = torch.arange(length, device=type_ids.device).expand(3, -1) + current
            current += length
        else:
            grid_iter = grids.get(int(modality))
            if grid_iter is None:
                raise ValueError(f"Missing Qwen3.5 grid for modality={modality}.")
            grid = next(grid_iter)
            segment = _vision_positions(current, grid, spatial_merge_size)
            current += max(int(grid[1]), int(grid[2])) // spatial_merge_size
        if int(segment.shape[1]) != length:
            raise ValueError(
                "Qwen3.5 M-RoPE span/grid mismatch: "
                f"modality={modality} span={length} positions={segment.shape[1]}."
            )
        segments.append(segment)
    positions = torch.cat(segments, dim=1)
    return positions, int(positions.max().item() + 1 - type_ids.numel())


class Qwen35MultimodalEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.spatial_merge_size = int(config.vision_config.spatial_merge_size)

    @torch.inference_mode()
    def encode(
        self,
        input_ids: list[int],
        tensors: dict[str, torch.Tensor],
    ) -> MultiModalState:
        device = next(self.parameters()).device
        type_ids = tensors["mm_token_type_ids"].squeeze(0).to(device)
        embeddings = {}
        inputs = (
            (1, "pixel_values", "image_grid_thw"),
            (2, "pixel_values_videos", "video_grid_thw"),
        )
        for modality, pixels_name, grid_name in inputs:
            pixels = tensors.get(pixels_name)
            if pixels is None:
                continue
            grid = tensors.get(grid_name)
            if grid is None:
                raise ValueError(f"{pixels_name} requires {grid_name}.")
            output = self.visual(
                pixels.to(device=device, dtype=self.visual.dtype),
                grid_thw=grid.to(device),
                return_dict=True,
            )
            embeddings[modality] = output.pooler_output
        positions, delta = qwen35_mrope_positions(
            type_ids,
            None if tensors.get("image_grid_thw") is None else tensors["image_grid_thw"].to(device),
            None if tensors.get("video_grid_thw") is None else tensors["video_grid_thw"].to(device),
            self.spatial_merge_size,
        )
        return MultiModalState(type_ids, embeddings, positions, delta)


__all__ = ["Qwen35MultimodalEncoder", "qwen35_mrope_positions"]
