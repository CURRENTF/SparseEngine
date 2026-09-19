from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def _aligned_offset(offset: int, alignment: int) -> int:
    return (int(offset) + int(alignment) - 1) // int(alignment) * int(alignment)


@dataclass
class DecodeGraphHostInputs:
    """Contiguous host staging with typed views for one decode graph state."""

    storage: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    context_lens: torch.Tensor
    request_indices: torch.Tensor
    active_mask: torch.Tensor
    sequence_ids: torch.Tensor
    _input_ids_np: np.ndarray = field(init=False, repr=False)
    _positions_np: np.ndarray = field(init=False, repr=False)
    _context_lens_np: np.ndarray = field(init=False, repr=False)
    _request_indices_np: np.ndarray = field(init=False, repr=False)
    _active_mask_np: np.ndarray = field(init=False, repr=False)
    _sequence_ids_np: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._input_ids_np = self.input_ids.numpy()
        self._positions_np = self.positions.numpy()
        self._context_lens_np = self.context_lens.numpy()
        self._request_indices_np = self.request_indices.numpy()
        self._active_mask_np = self.active_mask.numpy()
        self._sequence_ids_np = self.sequence_ids.numpy()

    @classmethod
    def allocate(
        cls,
        batch_capacity: int,
        *,
        pin_memory: bool,
    ) -> DecodeGraphHostInputs:
        batch_capacity = int(batch_capacity)
        if batch_capacity <= 0:
            raise ValueError(
                "Decode graph host staging capacity must be positive, got "
                f"{batch_capacity}."
            )

        specs = (
            ("input_ids", torch.int64),
            ("positions", torch.int64),
            ("context_lens", torch.int32),
            ("request_indices", torch.int32),
            ("active_mask", torch.bool),
            ("sequence_ids", torch.int64),
        )
        offsets: dict[str, tuple[int, torch.dtype]] = {}
        offset = 0
        for name, dtype in specs:
            item_size = torch.empty((), dtype=dtype).element_size()
            offset = _aligned_offset(offset, item_size)
            offsets[name] = (offset, dtype)
            offset += batch_capacity * item_size

        storage = torch.empty(
            offset,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=bool(pin_memory),
        )

        def typed_view(name: str) -> torch.Tensor:
            field_offset, dtype = offsets[name]
            item_size = torch.empty((), dtype=dtype).element_size()
            return storage.narrow(
                0,
                field_offset,
                batch_capacity * item_size,
            ).view(dtype)

        return cls(
            storage=storage,
            input_ids=typed_view("input_ids"),
            positions=typed_view("positions"),
            context_lens=typed_view("context_lens"),
            request_indices=typed_view("request_indices"),
            active_mask=typed_view("active_mask"),
            sequence_ids=typed_view("sequence_ids"),
        )

    @property
    def batch_capacity(self) -> int:
        return int(self.input_ids.numel())

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_ids,
            self.positions,
            self.context_lens,
            self.request_indices,
            self.active_mask,
        )

    def pack_requests(self, seqs: list[object]) -> np.ndarray:
        """Pack Python-owned request facts once and return active sequence ids."""

        real_batch_size = len(seqs)
        if real_batch_size > self.batch_capacity:
            raise ValueError(
                "Decode request batch exceeds host staging capacity: "
                f"real={real_batch_size} capacity={self.batch_capacity}."
            )
        for index, seq in enumerate(seqs):
            self._input_ids_np[index] = int(seq.decode_input_token)
            self._positions_np[index] = int(seq.decode_input_position)
            self._sequence_ids_np[index] = int(seq.seq_id)
        return self._sequence_ids_np[:real_batch_size]

    def pack_cache_facts(
        self,
        *,
        context_lens: np.ndarray,
        request_indices: np.ndarray,
        real_batch_size: int,
        padding_active: bool,
    ) -> None:
        """Pack cache-owned facts and define every host-visible padding row."""

        real_batch_size = int(real_batch_size)
        if real_batch_size <= 0 or real_batch_size > self.batch_capacity:
            raise ValueError(
                "Decode graph host staging requires a non-empty active prefix: "
                f"real={real_batch_size} capacity={self.batch_capacity}."
            )
        context_lens = np.asarray(context_lens)
        request_indices = np.asarray(request_indices)
        if context_lens.shape != (real_batch_size,):
            raise ValueError(
                "Decode context lengths must match the active batch: "
                f"shape={context_lens.shape} real={real_batch_size}."
            )
        if request_indices.shape != (real_batch_size,):
            raise ValueError(
                "Decode request indices must match the active batch: "
                f"shape={request_indices.shape} real={real_batch_size}."
            )

        self._context_lens_np[:real_batch_size] = context_lens
        self._request_indices_np[:real_batch_size] = request_indices
        self._active_mask_np[:real_batch_size] = True

        if real_batch_size == self.batch_capacity:
            return
        padding = slice(real_batch_size, self.batch_capacity)
        self._input_ids_np[padding] = self._input_ids_np[0]
        self._positions_np[padding] = self._positions_np[0]
        self._context_lens_np[padding] = self._context_lens_np[0]
        self._request_indices_np[padding] = self._request_indices_np[0]
        self._active_mask_np[padding] = bool(padding_active)
