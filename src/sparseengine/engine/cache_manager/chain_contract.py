"""Chain's exclusive-writer contract, not a radix adapter or a second ledger.

The coordinator owns logical history and admission promises. The manager owns
rows, physical slots and method state. All calls are rank-local; the runner
establishes TP agreement and execution completion before terminal operations.
See docs/development/cache-contracts.md for the cross-component call contract.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    import torch
    from .chain_offload import ChainMethodState

# Layer order is kv_transformer_layer_indices(), NOT range(num_layers).
# Additional physical high-water mark, rows, current slot deficits, row deficit.
ChainCapacity = tuple[tuple[int, ...], int, tuple[int, ...], int]


class ChainManager(Protocol):
    def kv_transformer_layer_indices(self) -> Sequence[int]: ...

    def chain_capacity_deficits(
        self, *, suffix_tokens: int, generation_tokens: int = 0,
        existing_slots_by_layer: tuple[int, ...] = (),
        outstanding_reserved_slots_by_layer: tuple[int, ...] = (),
        outstanding_reserved_rows: int = 0, needs_resident_row: bool,
    ) -> ChainCapacity:
        """Read-only physical peak, including restore when no row is resident.

        A generation horizon includes its final sampled, unprocessed token.
        Do not allocate, evict, pin a row, change method state or reserve here.
        Independent pools retain their layer vector. A shared pool repeats its
        demand on the layer axis; its physical charge is never summed by layer.
        """
        ...

    def chain_has_residency(self, seq_id: int) -> bool:
        """Whether ANY row exists, including a partially prepared failed step."""
        ...

    def chain_physical_residency(self, seq_id: int) -> tuple[int, ...]:
        """Actual occupied lengths in KV-layer order; missing layers are errors."""
        ...

    def chain_physical_kv_len(self, layer_idx: int, seq_id: int) -> int: ...

    def on_chain_turn_finished(self, seq_id: int, processed_token_count: int) -> None:
        """After fencing and owner validation, finalize method state, not history.

        Must not publish IDLE or release another request's resources. Failure
        makes this turn unusable; caller invalidates it rather than resuming it.
        """
        ...

    def free_seq(self, seq_id: int) -> None:
        """After fencing: release all request-local state; repeated calls are safe.

        Does not choose a chain victim, mutate the chain index or publish IDLE.
        """
        ...


class ChainOffloadManager(ChainManager, Protocol):
    def chain_storage_tensors(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    def chain_token_slots(self, layer: int, seq_id: int) -> torch.Tensor:
        """Borrow the occupied slot-ID view. Async consumers must freeze it."""
        ...

    def allocate_chain_restore(
        self, seq_id: int, lengths_by_layer: tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        """Reserve all restore rows/slots, or leave no rows for this seq_id.

        Reject existing residency, wrong layer count and invalid lengths before
        mutation. Return layer -> occupied slot view. No KV transfer, sidecar
        restore, index mutation or host-accounting change is permitted here.
        """
        ...

    def snapshot_chain_method_state(self, seq_id: int) -> ChainMethodState: ...
    def restore_chain_method_state(self, seq_id: int, state: ChainMethodState) -> None: ...


def validate_chain_manager(manager, base_type: type, *, offload: bool) -> None:
    """Factory-only wiring gate. This does not certify semantic conformance.

    No wrapper, observer, signature inspection or per-step check is installed.
    Shared inherited implementations are valid; unimplemented base stubs are not.
    """
    required = (
        "chain_capacity_deficits", "chain_has_residency",
        "chain_physical_residency", "chain_physical_kv_len", "free_seq",
    )
    if offload:
        required += (
            "create_chain_offload", "chain_storage_tensors", "chain_token_slots",
            "allocate_chain_restore", "snapshot_chain_method_state",
            "restore_chain_method_state",
        )
    cls = type(manager)
    for name in required:
        implementation = getattr(cls, name, None)
        if not callable(implementation) or implementation is getattr(base_type, name, None):
            raise TypeError(f"{cls.__name__}: chain contract requires {name}().")
