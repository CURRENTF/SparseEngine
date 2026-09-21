"""Radix's shared immutable-block contract; independent of Chain's lifecycle."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch
    from sparseengine.engine.sequence import Sequence


class RadixManager(Protocol):
    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        """Set a speculative logical hit, not ownership or guaranteed capacity.

        Lookup may update bounded memoization and access statistics. It must not
        pin blocks, allocate a row, submit promotion or move num_prefilled_tokens.
        """
        ...

    def _attach_prefix_cache_if_needed(self, seq: Sequence) -> None:
        """Revalidate and pin the hit, promote if needed, then publish row aliases.

        Repeated attachment is a no-op. On failure this request owns no new
        aliases/refs. A successfully submitted promotion may remain index-owned;
        never free its destination before its completion fence.
        """
        ...

    def _record_prefix_materialization(
        self, seq: Sequence, token_ids: list[int], slots: torch.Tensor,
    ) -> None:
        """Record private, completed-token candidates; freeze mutable slot IDs.

        In async execution publication is deferred until the submitted step is
        retired successfully. This operation does not make a block reusable.
        """
        ...

    def publish_pending_prefix_blocks(self, seqs: list[Sequence]) -> None:
        """Transfer complete blocks to the index and retain request references.

        Existing logical IDs keep their original payload. Duplicate recompute
        allocations remain private, not aliases of that payload. Successfully
        published blocks may survive a later publication failure; cleanup must
        retain exactly the index-owned slots and release request-only slots.
        """
        ...

    def free_seq(self, seq_id: int) -> None:
        """Release private storage and refs, never another reader's shared KV."""
        ...

    def reset_prefix_cache(self) -> None:
        """Only after drain and zero references; return all index-owned storage."""
        ...


class MixedRadixKVManager(Protocol):
    """KV half of PrefixCacheCoordinator's atomic KV+recurrent attachment.

    This is not the manager-local RadixManager attach path. The coordinator owns
    block references and recurrent rollback; the manager owns physical KV.
    """
    def build_prefix_kv_payload(self, seq: Sequence, block_start: int, block_end: int) -> object: ...
    def attach_prefix_kv_payloads(self, seq: Sequence, payloads: list[object]) -> None: ...
    def validate_prefix_kv_attach(self, seq: Sequence) -> bool: ...
    def rollback_prefix_kv_attach(
        self, seq: Sequence, payloads: list[object], *, row_preexisted: bool,
    ) -> None: ...
    def free_prefix_kv_payload(self, payload: object) -> None: ...
    def mark_materialized_prefix_kv_payload(self, seq: Sequence, payload: object) -> None: ...


def validate_radix_manager(manager, base_type: type, *, mixed: bool) -> None:
    """One cold wiring check; the conformance harness checks observable behavior."""
    if mixed:
        required = (
            "build_prefix_kv_payload", "attach_prefix_kv_payloads",
            "validate_prefix_kv_attach", "rollback_prefix_kv_attach",
            "free_prefix_kv_payload", "mark_materialized_prefix_kv_payload",
        )
    else:
        required = (
            "refresh_prefix_cache_hit", "_attach_prefix_cache_if_needed",
            "_record_prefix_materialization", "publish_pending_prefix_blocks",
            "reset_prefix_cache", "free_seq",
        )
    cls = type(manager)
    for name in required:
        implementation = getattr(cls, name, None)
        if not callable(implementation) or implementation is getattr(base_type, name, None):
            raise TypeError(f"{cls.__name__}: radix contract requires {name}().")
