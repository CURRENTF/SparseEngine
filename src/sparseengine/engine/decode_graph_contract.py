from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from sparseengine.engine.decode_graph_staging import DecodeGraphHostInputs


@dataclass(frozen=True)
class DecodeGraphPaddingContract:
    """Safe values used for inactive rows in a fixed-capacity decode graph."""

    write_slot: int = -1
    active: bool = False
    mirror_first_real_row_for_reads: bool = True


@dataclass(frozen=True)
class DecodeGraphContract:
    """Capture-time facts that define one decode graph family."""

    method: str
    topology_path_id: str
    batch_capacity: int
    context_capacity: int
    capture_sampling: bool = False
    dynamic_context_lens: bool = True
    padding: DecodeGraphPaddingContract = field(
        default_factory=DecodeGraphPaddingContract
    )

    def __post_init__(self) -> None:
        if self.batch_capacity <= 0 or self.context_capacity <= 0:
            raise ValueError(
                "Decode graph batch and context capacities must be positive, got "
                f"batch={self.batch_capacity} context={self.context_capacity}."
            )
        if self.topology_path_id not in {"dense", "unified"}:
            raise ValueError(
                "Decode graph topology_path_id must be 'dense' or 'unified'; "
                f"got {self.topology_path_id!r}. Length-specific decode paths "
                "are not supported."
            )
        if not self.dynamic_context_lens:
            raise ValueError(
                "decode graphs require device-resident dynamic context lengths."
            )

    @property
    def capability_level(self) -> str:
        return "strict"


@dataclass
class DecodeGraphInputs:
    """Typed, address-stable public inputs shared by decode participants."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    context_lens: torch.Tensor
    request_indices: torch.Tensor
    write_slot_mapping: torch.Tensor
    active_mask: torch.Tensor
    host: DecodeGraphHostInputs

    @classmethod
    def allocate(
        cls,
        contract: DecodeGraphContract,
        *,
        device: torch.device,
        pin_memory: bool,
    ) -> DecodeGraphInputs:
        batch = int(contract.batch_capacity)

        def device_buffer(dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(batch, dtype=dtype, device=device)

        inputs = cls(
            input_ids=device_buffer(torch.int64),
            positions=device_buffer(torch.int64),
            context_lens=device_buffer(torch.int32),
            request_indices=device_buffer(torch.int32),
            write_slot_mapping=device_buffer(torch.int32),
            active_mask=device_buffer(torch.bool),
            host=DecodeGraphHostInputs.allocate(
                batch,
                pin_memory=pin_memory,
            ),
        )
        inputs.validate(contract)
        return inputs

    @property
    def batch_capacity(self) -> int:
        return int(self.input_ids.numel())

    def device_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_ids,
            self.positions,
            self.context_lens,
            self.request_indices,
            self.write_slot_mapping,
            self.active_mask,
        )

    def keepalive_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.device_tensors() + self.host.tensors()

    def data_ptrs(self) -> tuple[int, ...]:
        return tuple(int(tensor.data_ptr()) for tensor in self.device_tensors())

    def validate(self, contract: DecodeGraphContract) -> None:
        expected = int(contract.batch_capacity)
        tensors = self.device_tensors()
        if any(tensor.ndim != 1 or tensor.numel() != expected for tensor in tensors):
            raise ValueError(
                "Decode graph public inputs must be one-dimensional and match "
                f"batch_capacity={expected}."
            )
        device = self.input_ids.device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("Decode graph public inputs must share one device.")
        expected_dtypes = (
            torch.int64,
            torch.int64,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.bool,
        )
        actual_dtypes = tuple(tensor.dtype for tensor in tensors)
        if actual_dtypes != expected_dtypes:
            raise TypeError(
                "Decode graph public input dtypes do not match the contract: "
                f"expected={expected_dtypes} actual={actual_dtypes}."
            )
        host_tensors = self.host.tensors()
        if any(tensor.device.type != "cpu" for tensor in host_tensors):
            raise ValueError("Decode graph host mirrors must reside on CPU.")
        if tuple(tensor.dtype for tensor in host_tensors) != (
            torch.int64,
            torch.int64,
            torch.int32,
            torch.int32,
            torch.bool,
        ):
            raise TypeError("Decode graph host mirror dtypes do not match public inputs.")
        if any(tensor.ndim != 1 or tensor.numel() != expected for tensor in host_tensors):
            raise ValueError(
                "Decode graph host mirrors must match the graph batch capacity."
            )


@dataclass
class CacheDecodeGraphState:
    """Per-graph cache participant state owned by one cache manager."""

    contract: DecodeGraphContract
    inputs: DecodeGraphInputs


@dataclass
class DecodeGraphState:
    """Typed public graph state plus participant-owned private state."""

    contract: DecodeGraphContract
    inputs: DecodeGraphInputs
    runtime_state: object | None = None

    def keepalive_tensors(self) -> list[torch.Tensor]:
        tensors = list(self.inputs.keepalive_tensors())
        participant = self.runtime_state
        keepalive = getattr(participant, "graph_keepalive_tensors", None)
        if callable(keepalive):
            tensors.extend(keepalive())
        return tensors

    def close(self) -> None:
        participant = self.runtime_state
        close = getattr(participant, "close", None)
        if callable(close):
            close()
        self.runtime_state = None


@runtime_checkable
class DecodeGraphParticipant(Protocol):
    """Minimal lifecycle implemented by graph metadata owners."""

    def prepare_out_graph(self, seqs: list[object]) -> None: ...

    def prepare_in_graph(self) -> None: ...

    def graph_keepalive_tensors(self) -> Iterable[torch.Tensor]: ...
