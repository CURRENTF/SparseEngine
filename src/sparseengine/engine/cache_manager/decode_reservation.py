"""CPU-side future decode capacity; physical allocation stays method-owned."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sparseengine.engine.sequence import Sequence


def _submitted_output_count(seq: Sequence) -> int:
    # Async submission has already allocated KV before its output is collected.
    # Collection transfers pending outputs to completed outputs without new KV.
    return seq.num_completion_tokens + seq.num_pending_outputs


@dataclass
class DecodeReservation:
    sequence: Sequence
    end: int


class DecodeReservations:
    def __init__(self, cache_manager, window: int):
        if type(window) is not int or window <= 0:
            raise ValueError("Decode reservation window must be a positive integer.")
        self.cache_manager = cache_manager
        self.window = window
        self.requests: dict[int, DecodeReservation] = {}

    def release(self, seq_id: int) -> None:
        self.requests.pop(int(seq_id), None)

    def outstanding(self, *, exclude: int | None = None) -> dict[str, int]:
        total: dict[str, int] = {}
        for seq_id, reservation in self.requests.items():
            if seq_id == exclude:
                continue
            seq = reservation.sequence
            remaining = max(0, reservation.end - _submitted_output_count(seq))
            if remaining:
                costs = self.cache_manager.decode_window_costs(seq, remaining)
                for name, cost in costs.items():
                    total[name] = total.get(name, 0) + int(cost)
        return total

    def needs_acquisition(self, seq: Sequence) -> bool:
        reservation = self.requests.get(seq.seq_id)
        submitted = _submitted_output_count(seq)
        if reservation is not None and submitted < reservation.end:
            return False
        return submitted < seq.max_tokens and not seq.is_recompute_replay

    def acquire(self, seq: Sequence, *, allow_short: bool = False,
                prefill_reserve: dict[str, int] | None = None,
                budgets: dict[str, int] | None = None) -> bool:
        return self.acquire_many(
            (seq,), allow_short=allow_short, prefill_reserve=prefill_reserve,
            budgets=budgets,
        ) is None

    def acquire_many(self, seqs: Iterable[Sequence], *, allow_short: bool = False,
                     prefill_reserve: dict[str, int] | None = None,
                     budgets: dict[str, int] | None = None) -> Sequence | None:
        # Physical residency cannot change during this synchronous acquisition.
        # Rebuild on every call: decode/eviction changes the cost of live windows.
        outstanding = None
        for seq in seqs:
            if not self.needs_acquisition(seq):
                continue
            if outstanding is None:
                if budgets is None:
                    budgets = self.cache_manager.decode_window_budgets()
                outstanding = self.outstanding()
            # A renewal's old window is exhausted, so contributes zero here.
            # Preserve earlier acquisitions when a later request cannot fit.
            if not self._acquire(seq, outstanding, budgets, prefill_reserve, allow_short):
                return seq
        return None

    def _acquire(self, seq: Sequence, outstanding: dict[str, int],
                 budgets: dict[str, int], prefill_reserve: dict[str, int] | None,
                 allow_short: bool) -> bool:
        submitted = _submitted_output_count(seq)
        remaining = seq.max_tokens - submitted
        tokens = min(self.window, remaining)
        costs = self.cache_manager.decode_window_costs(seq, tokens)

        def fits(costs: dict[str, int]) -> bool:
            return all(
                cost + outstanding.get(name, 0) + (prefill_reserve or {}).get(name, 0) <= budgets[name]
                for name, cost in costs.items()
            )

        if not fits(costs):
            if not allow_short or not fits(self.cache_manager.decode_window_costs(seq, 1)):
                return False
            # A sole request must not deadlock merely because a whole window
            # does not fit. Cost hooks are monotone upper bounds over a horizon.
            lo, hi = 1, tokens
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if fits(self.cache_manager.decode_window_costs(seq, mid)):
                    lo = mid
                else:
                    hi = mid - 1
            tokens = lo
            costs = self.cache_manager.decode_window_costs(seq, tokens)
        self.requests[seq.seq_id] = DecodeReservation(seq, submitted + tokens)
        for name, cost in costs.items():
            outstanding[name] = outstanding.get(name, 0) + int(cost)
        return True
