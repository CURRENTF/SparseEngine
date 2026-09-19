from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import torch


@dataclass
class MultiModalState:
    type_ids: torch.Tensor
    embeddings: dict[int, torch.Tensor]
    position_ids: torch.Tensor | None
    position_delta: int
    feature_ranges: dict[int, list[tuple[int, int, int]]] = field(
        default_factory=dict, init=False
    )
    image_ranges: list[tuple[int, int]] = field(default_factory=list, init=False)


class MultiModalRuntime:
    """Rank-local encoder feature cache keyed by the engine sequence id."""

    def __init__(self, model, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.states: dict[int, MultiModalState] = {}

    @property
    def enabled(self) -> bool:
        return callable(getattr(self.model, "encode_multimodal", None))

    def register(
        self,
        seq_id: int,
        input_ids: list[int],
        tensors: dict[str, torch.Tensor],
    ) -> int:
        if not self.enabled:
            raise NotImplementedError(
                f"{type(self.model).__name__} has no native multimodal encoder."
            )
        seq_id = int(seq_id)
        if seq_id in self.states:
            raise RuntimeError(f"Multimodal state already exists for seq_id={seq_id}.")
        encoded = self.model.encode_multimodal(input_ids, tensors)
        if not isinstance(encoded, MultiModalState):
            raise TypeError(
                "encode_multimodal() must return MultiModalState, "
                f"got {type(encoded).__name__}."
            )
        if encoded.type_ids.ndim != 1 or encoded.type_ids.numel() != len(input_ids):
            raise ValueError(
                "Multimodal token types must align with the prompt: "
                f"types={tuple(encoded.type_ids.shape)} tokens={len(input_ids)}."
            )
        # Token types are immutable request metadata. Compute host ranges once
        # so chunk preparation needs neither device reductions nor ragged gathers.
        encoded.feature_ranges = {int(modality): [] for modality in encoded.embeddings}
        encoded.image_ranges = []
        counts = dict.fromkeys(encoded.feature_ranges, 0)
        host_types = encoded.type_ids.cpu().numpy()
        boundaries = np.concatenate(
            (
                [0],
                np.flatnonzero(host_types[1:] != host_types[:-1]) + 1,
                [len(host_types)],
            )
        )
        for start, end in pairwise(boundaries):
            if start == end:
                continue
            start, end = int(start), int(end)
            modality = int(host_types[start])
            if modality in encoded.feature_ranges:
                encoded.feature_ranges[modality].append((start, end, counts[modality]))
                counts[modality] += end - start
            if modality in (1, 2):
                if encoded.image_ranges and encoded.image_ranges[-1][1] == start:
                    encoded.image_ranges[-1] = (encoded.image_ranges[-1][0], end)
                else:
                    encoded.image_ranges.append((start, end))
        for modality, features in encoded.embeddings.items():
            expected = counts[int(modality)]
            if features.ndim != 2 or int(features.shape[0]) != expected:
                raise ValueError(
                    "Multimodal feature/token mismatch: "
                    f"modality={modality} features={tuple(features.shape)} tokens={expected}."
                )
        self.states[seq_id] = encoded
        return int(encoded.position_delta)

    def free(self, seq_id: int) -> None:
        self.states.pop(int(seq_id), None)

    def free_batch(self, seq_ids: list[int]) -> None:
        for seq_id in seq_ids:
            self.free(seq_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_mask: torch.Tensor,
    ) -> torch.Tensor:
        forward = getattr(self.model, "forward_multimodal", None)
        if not callable(forward):
            raise NotImplementedError(
                f"{type(self.model).__name__} has no multimodal forward path."
            )
        return forward(input_ids, positions, inputs_embeds, multimodal_mask)

    def prepare(
        self,
        seqs,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        if not is_prefill or not any(int(seq.seq_id) in self.states for seq in seqs):
            return None, positions, None

        inputs_embeds = self.model.embed_input_ids(input_ids)
        multimodal_mask = torch.zeros(
            input_ids.shape, dtype=torch.bool, device=input_ids.device
        )
        position_rows = []
        batch_offset = 0
        image_groups = torch.zeros_like(input_ids, dtype=torch.int32)
        next_image_group = 1
        for seq in seqs:
            chunk_size = int(seq.current_chunk_size)
            start = int(seq.num_prefilled_tokens)
            end = start + chunk_size
            state = self.states.get(int(seq.seq_id))
            if state is None:
                position_rows.append(
                    positions[batch_offset : batch_offset + chunk_size].expand(3, -1)
                )
                batch_offset += chunk_size
                continue

            position_rows.append(
                state.position_ids[:, start:end]
                if state.position_ids is not None
                else positions[batch_offset : batch_offset + chunk_size].expand(3, -1)
            )
            for modality, features in state.embeddings.items():
                for run_start, run_end, feature_offset in state.feature_ranges[
                    int(modality)
                ]:
                    left, right = max(start, run_start), min(end, run_end)
                    if left >= right:
                        continue
                    chunk_start = batch_offset + left - start
                    chunk_end = batch_offset + right - start
                    feature_start = feature_offset + left - run_start
                    inputs_embeds[chunk_start:chunk_end].copy_(
                        features[feature_start : feature_start + right - left]
                    )
                    multimodal_mask[chunk_start:chunk_end] = True

            for run_start, run_end in state.image_ranges:
                left, right = max(start, run_start), min(end, run_end)
                if left < right:
                    image_groups[
                        batch_offset + left - start : batch_offset + right - start
                    ] = next_image_group
                    next_image_group += 1
            batch_offset += chunk_size

        from sparseengine.utils.context import get_context

        get_context().multimodal_image_groups = (
            image_groups
            if next_image_group > 1
            and bool(getattr(self.model, "multimodal_bidirectional", False))
            else None
        )
        use_mrope = any(
            state.position_ids is not None
            for seq in seqs
            if (state := self.states.get(int(seq.seq_id))) is not None
        )
        return (
            inputs_embeds,
            torch.cat(position_rows, dim=1) if use_mrope else positions,
            multimodal_mask,
        )


__all__ = ["MultiModalRuntime", "MultiModalState"]
