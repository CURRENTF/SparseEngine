from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from sparseengine.config import Config
from sparseengine.distributed import ParallelContext
from sparseengine.engine.sequence import Sequence
from sparseengine.platforms import device_runtime


@dataclass
class RecurrentPrefixPayload:
    token_count: int
    layer_states: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)


@dataclass(frozen=True)
class RecurrentTensorSpec:
    """Exact TP-local geometry for one persistent recurrent tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        name = str(self.name)
        shape = tuple(int(dim) for dim in self.shape)
        if not name or not shape or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"Invalid recurrent tensor spec name={name!r} shape={shape!r}."
            )
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(
                f"Recurrent tensor dtype must be torch.dtype, got {type(self.dtype).__name__}."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", shape)

    @property
    def nbytes(self) -> int:
        elements = 1
        for dim in self.shape:
            elements *= int(dim)
        return int(elements * torch.empty((), dtype=self.dtype).element_size())


@dataclass(frozen=True)
class RecurrentStateSpec:
    """Model-provided schema for one recurrent layer's persistent tensors."""

    name: str
    tensor_specs: tuple[RecurrentTensorSpec, ...]

    def __post_init__(self) -> None:
        tensor_specs = tuple(self.tensor_specs)
        names = tuple(spec.name for spec in tensor_specs)
        if not self.name or not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(
                f"Invalid recurrent state spec name={self.name!r} state_names={names!r}."
            )
        if any(not isinstance(spec, RecurrentTensorSpec) for spec in tensor_specs):
            raise TypeError("RecurrentStateSpec.tensor_specs must contain RecurrentTensorSpec values.")
        object.__setattr__(self, "tensor_specs", tensor_specs)

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.tensor_specs)

    @property
    def bytes_per_layer(self) -> int:
        return int(sum(spec.nbytes for spec in self.tensor_specs))

    def bytes_for_layers(self, num_layers: int) -> int:
        return int(self.bytes_per_layer * int(num_layers))


@dataclass(frozen=True)
class RecurrentCapacity:
    row_capacity: int
    total_rows: int
    bytes_per_row: int
    pool_bytes: int
    supported_active_sequences: int


class RecurrentStateManager:
    """Preallocated per-sequence rows for a model-declared recurrent schema."""

    def __init__(
        self,
        config: Config,
        parallel_context: ParallelContext,
        *,
        device: torch.device,
        platform=None,
        state_spec: RecurrentStateSpec,
    ):
        self.config = config
        self.parallel_context = parallel_context
        self.rank = parallel_context.world_rank
        self.world_size = parallel_context.world_size
        self.tp_rank = parallel_context.attn_tp_rank
        self.tp_size = parallel_context.attn_tp_size
        self.ep_rank = parallel_context.moe_ep_rank
        self.ep_size = parallel_context.moe_ep_size
        self.device = torch.device(device)
        self.runtime_layout = config.runtime_layout
        self.state_spec = state_spec
        recurrent_layers = tuple(
            int(layer_idx)
            for layer_idx in self.runtime_layout.linear_attention_layer_indices
        )
        capacity = self.resolve_capacity(
            config,
            state_spec,
            num_recurrent_layers=len(recurrent_layers),
        )
        self.row_capacity = capacity.row_capacity
        self.bytes_per_row = capacity.bytes_per_row
        self.pool_bytes = capacity.pool_bytes
        if self.row_capacity <= 0:
            raise ValueError(f"Recurrent state row capacity must be positive, got {self.row_capacity}.")
        self.scratch_row = self.row_capacity
        self.seq_id_to_row: dict[int, int] = {}
        self.row_to_seq_id: list[int | None] = [None] * self.row_capacity
        self.free_rows: deque[int] = deque(range(self.row_capacity))
        allocator_peak_before = 0
        if platform is not None:
            allocator_peak_before = int(
                platform.get_allocator_stats(self.device).peak_allocated_bytes
            )
        self.layer_buffers: dict[int, dict[str, torch.Tensor]] = {
            layer_idx: {
                tensor_spec.name: torch.zeros(
                    (capacity.total_rows, *tensor_spec.shape),
                    dtype=tensor_spec.dtype,
                    device=self.device,
                )
                for tensor_spec in state_spec.tensor_specs
            }
            for layer_idx in recurrent_layers
        }
        allocator_peak_after = allocator_peak_before
        if platform is not None:
            allocator_peak_after = int(
                platform.get_allocator_stats(self.device).peak_allocated_bytes
            )
        self.decode_state_indices: dict[int, torch.Tensor] = {}
        self.pending_prefix_payloads: dict[
            int, tuple[RecurrentPrefixPayload, dict[int, object]]
        ] = {}
        config.recurrent_state_pool_bytes = int(self.pool_bytes)
        config.recurrent_state_bytes_per_row = int(self.bytes_per_row)
        config.recurrent_state_row_capacity = int(self.row_capacity)
        config.prefix_recurrent_bytes_per_block = int(self.bytes_per_row)
        config.recurrent_state_allocator_peak_before_bytes = int(
            allocator_peak_before
        )
        config.recurrent_state_allocator_peak_after_bytes = int(
            allocator_peak_after
        )

    @staticmethod
    def resolve_capacity(
        config: Config,
        state_spec: RecurrentStateSpec,
        *,
        num_recurrent_layers: int,
    ) -> RecurrentCapacity:
        num_recurrent_layers = int(num_recurrent_layers)
        if num_recurrent_layers <= 0:
            raise ValueError(
                f"num_recurrent_layers must be positive, got {num_recurrent_layers}."
            )
        bytes_per_row = state_spec.bytes_for_layers(num_recurrent_layers)
        row_capacity = int(config.max_num_seqs_in_gpu)
        total_rows = row_capacity + 1
        pool_bytes = int(total_rows * bytes_per_row)
        budget_bytes = getattr(config, "recurrent_state_max_bytes", None)
        if budget_bytes is not None and pool_bytes > int(budget_bytes):
            raise RuntimeError(
                f"{state_spec.name} live recurrent-state pool exceeds recurrent_state_max_bytes: "
                f"required_bytes={pool_bytes} limit_bytes={int(budget_bytes)} "
                f"rows={total_rows} active_rows={row_capacity} "
                f"bytes_per_row={bytes_per_row}."
            )
        return RecurrentCapacity(
            row_capacity=int(row_capacity),
            total_rows=int(total_rows),
            bytes_per_row=int(bytes_per_row),
            pool_bytes=pool_bytes,
            supported_active_sequences=int(row_capacity),
        )

    def _allocate_row(self, seq_id: int) -> int:
        seq_id = int(seq_id)
        existing = self.seq_id_to_row.get(seq_id)
        if existing is not None:
            return int(existing)
        if not self.free_rows:
            raise RuntimeError(
                "Recurrent state row capacity exhausted: "
                f"capacity={self.row_capacity} active_sequences={len(self.seq_id_to_row)}."
            )
        row = int(self.free_rows.popleft())
        self.seq_id_to_row[seq_id] = row
        self.row_to_seq_id[row] = seq_id
        for buffers in self.layer_buffers.values():
            for buffer in buffers.values():
                buffer[row].zero_()
        return row

    def _ensure_buffer(self, layer_idx: int, name: str, value: torch.Tensor) -> torch.Tensor:
        layer_idx = int(layer_idx)
        name = str(name)
        buffers = self.layer_buffers.get(layer_idx)
        if buffers is None:
            raise RuntimeError(
                f"Recurrent state was requested for undeclared layer_idx={layer_idx}."
            )
        buffer = buffers.get(name)
        expected_shape = (self.row_capacity + 1, *value.shape)
        if buffer is None:
            raise RuntimeError(
                f"Recurrent state was requested for undeclared tensor name={name!r}."
            )
        if tuple(buffer.shape) != tuple(expected_shape):
            raise RuntimeError(
                "Recurrent state shape changed after row pool allocation: "
                f"layer_idx={layer_idx} name={name!r} "
                f"expected={tuple(buffer.shape[1:])} got={tuple(value.shape)}."
            )
        if buffer.dtype != value.dtype or buffer.device != value.device:
            raise RuntimeError(
                "Recurrent state dtype/device changed after row pool allocation: "
                f"layer_idx={layer_idx} name={name!r} "
                f"expected={buffer.dtype}/{buffer.device} got={value.dtype}/{value.device}."
            )
        return buffer

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool) -> None:
        del is_prefill
        for seq in seqs:
            self._allocate_row(int(seq.seq_id))

    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        *,
        token_batch: int,
        device: torch.device,
    ) -> None:
        self.prepare_step(seqs, is_prefill=False)
        token_batch = int(token_batch)
        if token_batch < len(seqs):
            raise RuntimeError(
                f"{self.state_spec.name} static decode has fewer token rows than real sequences: "
                f"token_batch={token_batch} real_batch={len(seqs)}."
            )
        rows = [self.seq_id_to_row[int(seq.seq_id)] for seq in seqs]
        rows.extend([self.scratch_row] * (token_batch - len(seqs)))
        state_indices = self.decode_state_indices.get(token_batch)
        if state_indices is None:
            state_indices = torch.empty(token_batch, dtype=torch.int32, device=device)
            self.decode_state_indices[token_batch] = state_indices
        elif state_indices.device != device:
            raise RuntimeError(
                f"{self.state_spec.name} decode state-index buffer changed device: "
                f"expected={state_indices.device} got={device}."
            )
        state_indices.copy_(torch.tensor(rows, dtype=torch.int32, device=device))

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool) -> None:
        del seqs, is_prefill

    def free_seq(self, seq_id: int) -> None:
        seq_id = int(seq_id)
        self.pending_prefix_payloads.pop(seq_id, None)
        row = self.seq_id_to_row.pop(seq_id, None)
        if row is None:
            return
        if self.row_to_seq_id[row] != seq_id:
            raise RuntimeError(
                "Recurrent state row ownership is inconsistent: "
                f"seq_id={seq_id} row={row} owner={self.row_to_seq_id[row]}."
            )
        self.row_to_seq_id[row] = None
        self.free_rows.append(int(row))

    def reset_after_warmup(self) -> None:
        self.pending_prefix_payloads.clear()
        self.seq_id_to_row.clear()
        self.row_to_seq_id = [None] * self.row_capacity
        self.free_rows = deque(range(self.row_capacity))
        for buffers in self.layer_buffers.values():
            for buffer in buffers.values():
                buffer.zero_()

    def get_layer_state(self, seq_id: int, layer_idx: int) -> dict[str, torch.Tensor] | None:
        self._materialize_pending_prefix_layer(int(seq_id), int(layer_idx))
        row = self.seq_id_to_row.get(int(seq_id))
        buffers = self.layer_buffers.get(int(layer_idx))
        if row is None or not buffers:
            return None
        return {name: buffer[row] for name, buffer in buffers.items()}

    def set_layer_state(
        self,
        seq_id: int,
        layer_idx: int,
        states: dict[str, torch.Tensor],
    ) -> None:
        layer_idx = int(layer_idx)
        if self.runtime_layout is not None and not self.runtime_layout.is_linear_attention(layer_idx):
            raise RuntimeError(f"layer_idx={layer_idx} is full_attention and has no recurrent state")
        missing = [name for name in self.state_spec.state_names if name not in states]
        unexpected = [name for name in states if name not in self.state_spec.state_names]
        if missing or unexpected:
            raise RuntimeError(
                f"{self.state_spec.name} recurrent state schema mismatch: "
                f"missing={missing} unexpected={unexpected}."
            )
        for name, value in states.items():
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Recurrent state {name!r} must be a tensor, got {type(value).__name__}."
                )
        row = self._allocate_row(int(seq_id))
        for name, value in states.items():
            buffer = self._ensure_buffer(layer_idx, str(name), value)
            buffer[row].copy_(value.detach())

    def get_decode_layer_state(
        self,
        seqs: list[Sequence],
        *,
        layer_idx: int,
        token_batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        layer_idx = int(layer_idx)
        token_batch = int(token_batch)
        real_batch = len(seqs)
        for seq in seqs:
            self._materialize_pending_prefix_layer(int(seq.seq_id), layer_idx)
        if token_batch < real_batch:
            raise RuntimeError(
                f"{self.state_spec.name} decode has fewer token rows than real sequences: "
                f"token_batch={token_batch} real_batch={real_batch}."
            )
        buffers = self.layer_buffers.get(layer_idx)
        missing = [name for name in self.state_spec.state_names if not buffers or name not in buffers]
        if missing:
            raise RuntimeError(
                f"{self.state_spec.name} decode requires initialized recurrent state for "
                f"layer_idx={layer_idx}; missing={missing}."
            )
        state_buffers = {name: buffers[name] for name in self.state_spec.state_names}
        tensor_specs = {
            tensor_spec.name: tensor_spec
            for tensor_spec in self.state_spec.tensor_specs
        }
        for name, buffer in state_buffers.items():
            expected_dtype = tensor_specs[name].dtype
            if buffer.dtype != expected_dtype or buffer.device != device:
                raise RuntimeError(
                    f"{self.state_spec.name} decode recurrent pool does not match its model schema: "
                    f"layer_idx={layer_idx} name={name} pool={buffer.dtype}/{buffer.device} "
                    f"expected={expected_dtype}/{device}."
                )
        activation_state_name = (
            "conv_state"
            if "conv_state" in tensor_specs
            else (
                self.state_spec.state_names[0]
                if len({spec.dtype for spec in self.state_spec.tensor_specs}) == 1
                else None
            )
        )
        if (
            activation_state_name is not None
            and tensor_specs[activation_state_name].dtype != dtype
        ):
            raise RuntimeError(
                f"{self.state_spec.name} decode activation dtype does not match "
                f"{activation_state_name}: activation={dtype} "
                f"state={tensor_specs[activation_state_name].dtype}."
            )
        expected_rows: list[int] = []
        for seq in seqs:
            row = self.seq_id_to_row.get(int(seq.seq_id))
            if row is None:
                raise RuntimeError(
                    f"{self.state_spec.name} decode requires recurrent state for every sequence: "
                    f"missing_seq_id={seq.seq_id}."
                )
            expected_rows.append(int(row))
        for buffer in state_buffers.values():
            buffer[self.scratch_row].zero_()
        state_indices = self.decode_state_indices.get(token_batch)
        if state_indices is None:
            raise RuntimeError(
                f"{self.state_spec.name} decode state-index buffer was not prepared for the static batch: "
                f"token_batch={token_batch}."
            )
        expected_rows.extend([self.scratch_row] * (token_batch - real_batch))
        if state_indices.device.type == "cpu" and state_indices.tolist() != expected_rows:
            raise RuntimeError(
                f"{self.state_spec.name} decode state-index buffer is stale: "
                f"expected={expected_rows} got={state_indices.tolist()}."
            )
        return state_buffers, state_indices

    def build_prefix_recurrent_payload(self, seq: Sequence, token_count: int) -> RecurrentPrefixPayload:
        token_count = int(token_count)
        if token_count <= 0:
            raise ValueError(f"token_count must be > 0, got {token_count}.")
        block_size = int(getattr(self.config, "prefix_cache_block_size", 0) or 0)
        if block_size > 0 and token_count % block_size != 0:
            raise ValueError(
                "Recurrent prefix snapshots are only valid at prefix block boundaries: "
                f"seq_id={seq.seq_id} token_count={token_count} block_size={block_size}."
            )
        row = self.seq_id_to_row.get(int(seq.seq_id))
        if row is None:
            raise RuntimeError(f"Cannot snapshot recurrent state for unknown seq_id={seq.seq_id}.")
        payload_states: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx, buffers in self.layer_buffers.items():
            if self.runtime_layout is not None and not self.runtime_layout.is_linear_attention(layer_idx):
                continue
            payload_states[layer_idx] = {
                name: buffer[row].detach().clone()
                for name, buffer in buffers.items()
            }
        return RecurrentPrefixPayload(token_count=token_count, layer_states=payload_states)

    def attach_prefix_recurrent_payload(
        self,
        seq: Sequence,
        payload: RecurrentPrefixPayload | None,
        *,
        readiness_events: dict[int, object] | None = None,
    ) -> None:
        if payload is None:
            return
        if int(payload.token_count) <= 0:
            raise RuntimeError(f"Invalid recurrent prefix payload token_count={payload.token_count}.")
        block_size = int(getattr(self.config, "prefix_cache_block_size", 0) or 0)
        if block_size > 0 and int(payload.token_count) % block_size != 0:
            raise RuntimeError(
                "Cannot attach recurrent prefix payload from a non-boundary snapshot: "
                f"seq_id={seq.seq_id} token_count={payload.token_count} block_size={block_size}."
            )
        if readiness_events is not None:
            expected_layers = set(int(layer_idx) for layer_idx in payload.layer_states)
            if set(int(layer_idx) for layer_idx in readiness_events) != expected_layers:
                raise RuntimeError(
                    "Recurrent prefix readiness events do not match payload layers: "
                    f"expected={sorted(expected_layers)} got={sorted(readiness_events)}."
                )
            self._allocate_row(int(seq.seq_id))
            self.pending_prefix_payloads[int(seq.seq_id)] = (
                payload,
                {int(layer_idx): event for layer_idx, event in readiness_events.items()},
            )
            return
        for layer_idx, state in payload.layer_states.items():
            self.set_layer_state(int(seq.seq_id), int(layer_idx), state)

    def _materialize_pending_prefix_layer(self, seq_id: int, layer_idx: int) -> None:
        pending = self.pending_prefix_payloads.get(int(seq_id))
        if pending is None:
            return
        payload, readiness_events = pending
        state = payload.layer_states.get(int(layer_idx))
        if state is None:
            return
        event = readiness_events.pop(int(layer_idx), None)
        if event is None:
            raise RuntimeError(
                f"Recurrent prefix layer={layer_idx} is missing its H2D readiness event."
            )
        if device_runtime.is_stream_capturing():
            raise RuntimeError(
                "Recurrent prefix H2D readiness waits are forbidden during graph capture."
            )
        device_runtime.wait_event(event, device=self.device)
        self.set_layer_state(int(seq_id), int(layer_idx), state)
        if not readiness_events:
            self.pending_prefix_payloads.pop(int(seq_id), None)

    def free_prefix_recurrent_payload(self, payload: RecurrentPrefixPayload | None) -> None:
        del payload

    def prefix_recurrent_snapshot_nbytes(self) -> int:
        return int(self.bytes_per_row)

    def prefix_recurrent_payload_nbytes(self, payload: RecurrentPrefixPayload | None) -> int:
        if payload is None:
            return 0
        total = 0
        for state in payload.layer_states.values():
            for tensor in state.values():
                if torch.is_tensor(tensor):
                    total += int(tensor.numel() * tensor.element_size())
        return int(total)
