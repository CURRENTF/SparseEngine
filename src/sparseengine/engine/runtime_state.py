from __future__ import annotations

import json
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager
from typing import Protocol

import torch

from sparseengine.config import Config
from sparseengine.engine.cache_manager.decode_reservation import DecodeReservations
from sparseengine.engine.chain_cache import (
    ChainAdmissionPlan,
    ChainBusyError,
    ChainCacheCoordinator,
    ChainOwnerMismatchError,
    ChainState,
)
from sparseengine.engine.decode_graph_contract import (
    CacheDecodeGraphState,
    DecodeGraphState,
)
from sparseengine.engine.prefix_cache_coordinator import PrefixCacheCoordinator
from sparseengine.engine.recurrent_state_manager import RecurrentStateManager
from sparseengine.engine.sequence import Sequence
from sparseengine.sampling_params import SamplingParams
from sparseengine.utils.log import logger
from sparseengine.utils.profiler import cpu_timing


class MemoryOracle(Protocol):
    """Explicit scheduler-facing memory and admission interface."""

    @property
    def num_free_slots(self) -> int: ...

    def reserve_decode_windows(self, decoding, waiting) -> Sequence | None: ...
    def prefill_batched_tokens_margin(self) -> int: ...
    def remaining_prefill_tokens(self, seq: Sequence) -> int: ...
    def prefill_execution_mode(self, seq: Sequence) -> str: ...
    def prefill_batch_compatibility_key(self, seq: Sequence) -> object: ...
    def reset_prefill_execution_state(self, seq_id: int) -> None: ...
    def complete_prefill_execution(self, seq: Sequence) -> None: ...
    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], engine_prefill_chunk_size: int) -> int: ...
    def should_schedule_full_prefill(self, seq: Sequence) -> bool: ...
    def requires_full_prefill_step(self, seq: Sequence) -> bool: ...
    def requires_long_prefill_offload(self, seq: Sequence) -> bool: ...
    def prefill_step_free_slots(self) -> int: ...
    def prefill_step_free_slots_for(self, seq: Sequence) -> int: ...
    def prefill_private_slots_for(self, seq: Sequence) -> int: ...
    def min_final_prefill_chunk_size(self, seq: Sequence) -> int: ...
    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int: ...
    def decode_step_free_slots(self) -> int: ...
    def decode_step_free_slots_for(self, seq: Sequence) -> int: ...
    def decode_step_reservation_cost(self, seq: Sequence) -> int: ...
    def prompt_admission_free_slots(self) -> int: ...
    def prompt_admission_budgets(self, waiting_seqs: deque[Sequence], engine_prefill_chunk_size: int) -> dict[str, int]: ...
    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]: ...
    def prompt_logical_reservation_cost(self, seq: Sequence) -> int: ...
    def prompt_admission_failure_action(self) -> str: ...
    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]) -> None: ...
    def refresh_prefix_cache_hit(self, seq: Sequence) -> None: ...
    def clear_prefix_cache_hit(self, seq: Sequence) -> None: ...
    def scheduler_capacity_snapshot(self) -> ContextManager[None]: ...
    def free_slot_stats(self) -> dict[str, int]: ...
    def debug_live_seq_slots(self) -> dict[int, int]: ...
    def startup_batch_fits(
        self,
        prompt_lengths: tuple[int, ...],
        *,
        max_tokens: int,
    ) -> bool: ...


@dataclass
class RuntimeDecodeGraphState:
    """Per-graph runtime participant that delegates to semantic owners."""

    owner: RuntimeState
    cache: CacheDecodeGraphState
    operator_states: tuple[tuple[object, object], ...] = ()

    def prepare_out_graph(self, seqs: list[Sequence]) -> None:
        self.owner._evict_mixed_prefix_for_step(seqs, is_prefill=False)
        self.owner.cache_manager.prepare_decode_graph_step(seqs, self.cache)
        for participant, state in self.operator_states:
            participant.prepare_decode_graph_out(state)
        if self.owner.recurrent_state_manager is not None:
            inputs = self.cache.inputs
            self.owner.recurrent_state_manager.prepare_decode_static(
                seqs,
                token_batch=inputs.batch_capacity,
                device=inputs.input_ids.device,
            )

    def prepare_in_graph(self) -> None:
        self.owner.cache_manager.prepare_decode_graph_in(self.cache)
        for participant, state in self.operator_states:
            participant.prepare_decode_graph_in(state)

    def graph_keepalive_tensors(self) -> list[torch.Tensor]:
        tensors = self.owner.cache_manager.decode_graph_state_keepalive_tensors(
            self.cache
        )
        for participant, state in self.operator_states:
            tensors.extend(participant.decode_graph_keepalive_tensors(state))
        return tensors

    def close(self) -> None:
        for participant, state in reversed(self.operator_states):
            participant.close_decode_graph_state(state)


class RuntimeState:
    """Single lifecycle entrypoint for KV, recurrent state, and mixed prefix cache."""

    def __init__(
        self,
        config: Config,
        cache_manager,
        recurrent_state_manager: RecurrentStateManager | None = None,
        prefix_cache_coordinator: PrefixCacheCoordinator | None = None,
        chain_cache_coordinator: ChainCacheCoordinator | None = None,
        decode_graph_participants: tuple[object, ...] = (),
    ):
        self.config = config
        self.cache_manager = cache_manager
        self.recurrent_state_manager = recurrent_state_manager
        self.prefix_cache_coordinator = prefix_cache_coordinator
        self.chain_cache_coordinator = chain_cache_coordinator
        self.decode_graph_participants = tuple(decode_graph_participants)
        self._resident_seq_ids: set[int] = set()
        self.decode_reservations = DecodeReservations(
            cache_manager,
            getattr(config, "decode_reservation_tokens", Config.decode_reservation_tokens),
        )
        if chain_cache_coordinator is not None:
            chain_cache_coordinator.decode_reservations = self.decode_reservations

    @cpu_timing.timed
    def reserve_decode_windows(self, decoding, waiting) -> Sequence | None:
        # Capacity queries can traverse the entire prefix cache. Existing
        # windows need no renewal budget, even when another prompt is waiting.
        if not any(self.decode_reservations.needs_acquisition(seq) for seq in decoding):
            return None
        step = int(self.config.engine_prefill_chunk_size)
        budgets = dict(self.cache_manager.decode_window_budgets())
        prefill = self.cache_manager.prompt_admission_budgets(waiting, step)
        scalar = int(self.reserved_prefill_slots(waiting, step))
        pending = {
            name: (
                max(0, free - int(prefill[name])) if name in prefill
                else scalar if name.startswith("layer_") else 0
            )
            for name, free in budgets.items()
        }
        coordinator = self.chain_cache_coordinator
        if coordinator is not None:
            # Partial prefills are already covered above. New chain requests
            # also own capacity before their first prefill chunk is scheduled.
            partial_ids = {int(seq.seq_id) for seq in waiting
                           if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens}
            chain_slots, _ = coordinator._outstanding_active_reservations(
                exclude_seq_ids=partial_ids,
            )
            for layer, slots in zip(self.cache_manager.kv_transformer_layer_indices(), chain_slots):
                name = f"layer_{layer}"
                if name in pending:
                    pending[name] += int(slots)
            if "slots" in pending:
                pending["slots"] += max(chain_slots, default=0)
        extra = self._mixed_prefix_step_reclaimable_slots()
        if extra > 0:
            if "slots" not in budgets:
                raise RuntimeError("Mixed prefix decode reservations require a slots budget.")
            budgets["slots"] += extra
            # A clamped physical budget can hide pending prefill demand when
            # the remaining capacity is still held by reclaimable prefixes.
            pending["slots"] = max(pending["slots"], scalar)
        return self.decode_reservations.acquire_many(
            decoding, allow_short=len(decoding) == 1, prefill_reserve=pending,
            budgets=budgets,
        )

    @property
    def num_free_slots(self) -> int:
        return int(self.cache_manager.num_free_slots)

    def _step_required_slots(self, seqs: list[Sequence], is_prefill: bool) -> int:
        if is_prefill:
            return int(sum(int(seq.current_chunk_size or 0) for seq in seqs))
        return int(len(seqs))

    def _mixed_prefix_step_reclaimable_slots(self) -> int:
        if self.prefix_cache_coordinator is None:
            return 0
        return int(self.prefix_cache_coordinator.step_reclaimable_slots())

    def _mixed_prefix_admission_reclaimable_slots(self) -> int:
        if self.prefix_cache_coordinator is None:
            return 0
        return int(self.prefix_cache_coordinator.admission_reclaimable_slots())

    def _mixed_prefix_boundary_limit(self, seq: Sequence) -> int:
        if self.prefix_cache_coordinator is None:
            return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        block_size = int(self.prefix_cache_coordinator.block_size)
        if block_size <= 0:
            return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        start = max(
            int(seq.num_prefilled_tokens),
            int(getattr(seq, "prefix_cache_hit_len", 0) or 0),
        )
        remaining = int(seq.num_prompt_tokens) - start
        if remaining <= 0:
            return 0
        to_boundary = block_size - (start % block_size)
        if to_boundary == 0:
            to_boundary = block_size
        return int(min(remaining, to_boundary))

    def _evict_mixed_prefix_for_step(self, seqs: list[Sequence], is_prefill: bool) -> None:
        if self.prefix_cache_coordinator is None:
            return
        needed = self._step_required_slots(seqs, is_prefill)
        if needed <= 0:
            return
        free = int(getattr(self.cache_manager, "num_free_slots"))
        if free < needed:
            self.prefix_cache_coordinator.evict_for_slots(needed - free)

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        if is_prefill and self.prefix_cache_coordinator is not None:
            self.prefix_cache_coordinator.attach_prefix_cache_hits(seqs)
        self._evict_mixed_prefix_for_step(seqs, is_prefill)
        result = self.cache_manager.prepare_step(seqs, is_prefill)
        if self.recurrent_state_manager is not None:
            self.recurrent_state_manager.prepare_step(seqs, is_prefill)
        return result

    def prepare_decode_static(self, seqs: list[Sequence], *args):
        self._evict_mixed_prefix_for_step(seqs, is_prefill=False)
        result = self.cache_manager.prepare_decode_static(seqs, *args)
        if self.recurrent_state_manager is not None:
            if not args or not hasattr(args[0], "shape") or not hasattr(args[0], "device"):
                raise RuntimeError("Static recurrent decode requires the graph input tensor.")
            self.recurrent_state_manager.prepare_decode_static(
                seqs,
                token_batch=int(args[0].shape[0]),
                device=args[0].device,
            )
        return result

    def init_decode_graph_state(
        self,
        graph_state: DecodeGraphState,
    ) -> RuntimeDecodeGraphState:
        if graph_state.runtime_state is not None:
            raise RuntimeError("Decode graph runtime state was initialized twice.")
        cache_state = self.cache_manager.init_decode_graph_state(
            graph_state.contract,
            graph_state.inputs,
        )
        operator_states: list[tuple[object, object]] = []
        try:
            for participant in self.decode_graph_participants:
                operator_states.append(
                    (
                        participant,
                        participant.init_decode_graph_state(
                            graph_state.contract,
                            graph_state.inputs,
                        ),
                    )
                )
        except BaseException:
            for participant, state in reversed(operator_states):
                participant.close_decode_graph_state(state)
            raise
        state = RuntimeDecodeGraphState(
            owner=self,
            cache=cache_state,
            operator_states=tuple(operator_states),
        )
        graph_state.runtime_state = state
        return state

    def prepare_decode_graph_step(
        self,
        seqs: list[Sequence],
        graph_state: DecodeGraphState,
    ):
        participant = graph_state.runtime_state
        if participant is None:
            participant = self.init_decode_graph_state(graph_state)
        if not isinstance(participant, RuntimeDecodeGraphState):
            raise TypeError(
                "Decode graph runtime participant has an unexpected type: "
                f"{type(participant).__name__}."
            )
        participant.prepare_out_graph(seqs)
        return graph_state.inputs.input_ids, graph_state.inputs.positions, None

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool) -> None:
        self.cache_manager.on_forward_end(seqs, is_prefill)
        if is_prefill and self.chain_cache_coordinator is not None:
            for seq in seqs:
                if seq.chain_id and seq.num_prefilled_tokens + int(seq.current_chunk_size or 0) >= seq.num_prompt_tokens:
                    record = self.chain_cache_coordinator.index.lookup(seq.chain_id)
                    record.reserved_slots_by_layer = ()
                    record.reserved_rows = 0
        if self.recurrent_state_manager is not None:
            self.recurrent_state_manager.on_forward_end(seqs, is_prefill)
        if self.prefix_cache_coordinator is not None:
            try:
                self.prefix_cache_coordinator.record_step_tokens(seqs, is_prefill)
                self.prefix_cache_coordinator.commit_pending_blocks(seqs)
            finally:
                self.prefix_cache_coordinator.finish_step()

    def free_seq(self, seq_id: int) -> None:
        self._free_seq_payload(seq_id)

    def _free_seq_payload(self, seq_id: int) -> None:
        self.decode_reservations.release(seq_id)
        try:
            self.cache_manager.free_seq(seq_id)
        finally:
            # A post-detach write-through failure must not strand recurrent
            # state or coordinator refs. The runner fences execution first;
            # cleanup errors propagate, never silently resume a failed worker.
            try:
                if self.prefix_cache_coordinator is not None:
                    self.prefix_cache_coordinator.release_seq(seq_id)
            finally:
                try:
                    if self.recurrent_state_manager is not None:
                        self.recurrent_state_manager.free_seq(seq_id)
                finally:
                    self._resident_seq_ids.discard(int(seq_id))

    def chain_admission_plan(
        self,
        chain_id: str,
        seq_id: int,
        token_ids: list[int],
    ) -> ChainAdmissionPlan:
        if self.chain_cache_coordinator is None:
            raise RuntimeError("Chain prefix cache is not enabled for this runtime.")
        return self.chain_cache_coordinator.plan_admission(
            chain_id=str(chain_id),
            seq_id=int(seq_id),
            token_ids=[int(token_id) for token_id in token_ids],
        )

    def chain_validate_admission_plan(
        self,
        expected: ChainAdmissionPlan,
        input_token_count: int,
        input_prefix_digest: bytes,
    ) -> ChainAdmissionPlan:
        if self.chain_cache_coordinator is None:
            raise RuntimeError("Chain prefix cache is not enabled for this runtime.")
        return self.chain_cache_coordinator.validate_admission_plan(
            expected,
            input_token_count=int(input_token_count),
            input_prefix_digest=bytes(input_prefix_digest),
        )

    def chain_apply_admission(self, plan: ChainAdmissionPlan) -> dict[str, object]:
        if self.chain_cache_coordinator is None:
            raise RuntimeError("Chain prefix cache is not enabled for this runtime.")
        local_victims: list[tuple[str, int]] = []
        for chain_id in plan.victim_chain_ids:
            record = self.chain_cache_coordinator.index.lookup(chain_id)
            local_victims.append((str(chain_id), int(record.seq_id)))
        record = self.chain_cache_coordinator.apply_admission(plan)
        for chain_id in plan.demote_chain_ids:
            victim = self.chain_cache_coordinator.index.lookup(chain_id)
            self._resident_seq_ids.discard(int(victim.seq_id))
        for _chain_id, victim_seq_id in local_victims:
            self._free_seq_payload(victim_seq_id)
        try:
            self.chain_cache_coordinator.prepare_resumed_chain(record)
        except Exception:
            # A failed restore retains the CPU snapshot and must not leave an
            # ACTIVE writer or an outstanding reservation that can never run.
            if plan.status == "resumed":
                self.chain_cache_coordinator.index._set_record_state(record, ChainState.IDLE)
                record.reserved_slots_by_layer = ()
                record.reserved_rows = 0
            raise
        if plan.status == "resumed":
            self._resident_seq_ids.add(int(record.seq_id))
        return {
            "chain_id": record.chain_id,
            "seq_id": int(record.seq_id),
            "state": record.state.value,
            "victim_chain_ids": list(plan.victim_chain_ids),
            "demote_chain_ids": list(plan.demote_chain_ids),
        }

    def chain_finish(
        self,
        chain_id: str,
        seq_id: int,
        processed_token_digest: bytes,
        processed_token_count: int,
    ) -> dict[str, object]:
        if self.chain_cache_coordinator is None:
            raise RuntimeError("Chain prefix cache is not enabled for this runtime.")
        coordinator = self.chain_cache_coordinator
        # Reject stale owners before a method can compact or clear its sidecars.
        record = coordinator.index.lookup(str(chain_id))
        if int(record.seq_id) != int(seq_id):
            raise ChainOwnerMismatchError(
                f"Chain owner mismatch for {chain_id!r}: "
                f"resident_seq_id={record.seq_id}, finished_seq_id={int(seq_id)}.",
                chain_id=str(chain_id),
            )
        if record.state is not ChainState.ACTIVE:
            raise ChainBusyError("Only an ACTIVE chain turn may finish.", chain_id=str(chain_id))
        try:
            self.cache_manager.on_chain_turn_finished(
                int(seq_id), int(processed_token_count),
            )
            self.decode_reservations.release(seq_id)
            record = coordinator.finish_values(
                chain_id=str(chain_id), seq_id=int(seq_id),
                processed_token_digest=bytes(processed_token_digest),
                processed_token_count=int(processed_token_count),
            )
            coordinator.save_finished_chain(record)
        except Exception:
            # Method finalization and physical-residency publication can fail
            # too, not just D2H submission. Never expose their half-finished turn.
            coordinator.invalidate(record.chain_id)
            self._free_seq_payload(int(record.seq_id))
            raise
        return {
            "chain_id": record.chain_id,
            "seq_id": int(record.seq_id),
            "state": record.state.value,
            "processed_token_count": int(record.processed_token_count),
            "physical_slots_by_layer": list(record.physical_slots_by_layer),
        }

    def chain_reclaim_idle(
        self, chain_id: str, expected_seq_id: int, demote: bool,
    ) -> dict[str, object]:
        coordinator = self.chain_cache_coordinator
        if coordinator is None:
            raise RuntimeError("Chain prefix cache is not enabled for this runtime.")
        record = coordinator.index.lookup(chain_id)
        if int(record.seq_id) != int(expected_seq_id):
            raise ChainOwnerMismatchError(
                f"Chain owner changed before reclaim: {chain_id!r}.", chain_id=chain_id,
            )
        if record.state is not ChainState.IDLE:
            raise ChainBusyError("Cannot reclaim an ACTIVE chain.", chain_id=chain_id)
        if record.resident_rows <= 0 or not self.cache_manager.chain_has_residency(record.seq_id):
            raise RuntimeError(f"Chain {chain_id!r} has no resident KV to reclaim.")
        if demote:
            if coordinator.offload is None or not coordinator.offload.wait(record.seq_id).valid:
                raise RuntimeError("Cannot demote a chain without a valid CPU snapshot.")
        elif coordinator.offload is not None:
            coordinator.offload.drop(record.seq_id)
        self._free_seq_payload(int(record.seq_id))
        if demote:
            record.resident_rows = 0
            logger.info("chain_demoted {}", json.dumps({
                "chain_id": chain_id, "seq_id": int(record.seq_id),
                "cause": "idle_reclaim",
            }))
        else:
            coordinator.index.evict(chain_id)
        return {"chain_id": chain_id, "seq_id": int(record.seq_id), "demoted": bool(demote)}

    def chain_invalidate(
        self,
        chain_id: str,
        *,
        expected_seq_id: int | None = None,
    ) -> dict[str, object]:
        if self.chain_cache_coordinator is None:
            raise RuntimeError("Chain prefix cache is not enabled for this runtime.")
        record = self.chain_cache_coordinator.index.lookup(str(chain_id))
        if expected_seq_id is not None and int(record.seq_id) != int(expected_seq_id):
            raise ChainOwnerMismatchError(
                f"Chain owner mismatch for {chain_id!r}: "
                f"resident_seq_id={record.seq_id}, expected_seq_id={expected_seq_id}.",
                chain_id=str(chain_id),
            )
        record = self.chain_cache_coordinator.invalidate(str(chain_id))
        # A queued/CPU-only/failed-prepare request may own reservations and
        # method metadata without owning any physical row. Cleanup is idempotent.
        self._free_seq_payload(int(record.seq_id))
        return {
            "chain_id": record.chain_id,
            "seq_id": int(record.seq_id),
            "state": "tombstoned",
        }

    def chain_routing_match(self, chain_id: str) -> dict[str, object]:
        if self.chain_cache_coordinator is None:
            return {
                "present": False,
                "state": None,
                "tombstone": False,
                "enabled": False,
            }
        return {
            **self.chain_cache_coordinator.routing_match(str(chain_id)),
            "enabled": True,
        }

    @torch.inference_mode()
    def reset_after_warmup(self) -> None:
        self.decode_reservations.requests.clear()
        if self.prefix_cache_coordinator is not None:
            self.prefix_cache_coordinator.reset_after_warmup()
        if self.chain_cache_coordinator is not None:
            if self.chain_cache_coordinator.offload is not None:
                self.chain_cache_coordinator.offload.reset()
            resident_chain_seq_ids = sorted(
                {
                    int(record.seq_id)
                    for record in self.chain_cache_coordinator.index.records.values()
                    if int(record.seq_id) in self._resident_seq_ids
                }
            )
            for seq_id in resident_chain_seq_ids:
                self._free_seq_payload(seq_id)
            self.chain_cache_coordinator.reset()
        reset_cache = getattr(self.cache_manager, "reset_after_warmup", None)
        if callable(reset_cache):
            reset_cache()
        else:
            reset_prefix_cache = getattr(self.cache_manager, "reset_prefix_cache", None)
            if callable(reset_prefix_cache):
                reset_prefix_cache()
        if self.recurrent_state_manager is not None:
            self.recurrent_state_manager.reset_after_warmup()
        self._resident_seq_ids.clear()

    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        if getattr(seq, "multimodal_digest", None) is not None:
            self.clear_prefix_cache_hit(seq)
            return
        if self.prefix_cache_coordinator is not None:
            self.prefix_cache_coordinator.refresh_prefix_cache_hit(seq)
            return
        self.cache_manager.refresh_prefix_cache_hit(seq)

    def prefix_cache_lookup_batch(self, *, first: bool, last: bool) -> ContextManager[None]:
        owner = (self.prefix_cache_coordinator
                 if self.prefix_cache_coordinator is not None else self.cache_manager)
        cache = getattr(owner, "prefix_lookup_cache", None)
        return cache.batch(first=first, last=last) if cache is not None else nullcontext()

    def clear_prefix_cache_hit(self, seq: Sequence) -> None:
        self.cache_manager.clear_prefix_cache_hit(seq)

    def scheduler_capacity_snapshot(self) -> ContextManager[None]:
        snapshot = getattr(
            self.cache_manager,
            "scheduler_capacity_snapshot",
            None,
        )
        return snapshot() if callable(snapshot) else nullcontext()

    def startup_batch_fits(
        self,
        prompt_lengths: tuple[int, ...],
        *,
        max_tokens: int,
    ) -> bool:
        """Check startup admission through the same budgets as Scheduler."""
        if not prompt_lengths or any(int(length) <= 0 for length in prompt_lengths):
            raise ValueError(
                f"Startup prompt lengths must be positive, got {prompt_lengths!r}."
            )
        sampling_params = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=0.0,
            ignore_eos=True,
        )
        seqs = [
            Sequence([0] * int(prompt_len), sampling_params)
            for prompt_len in prompt_lengths
        ]
        waiting = deque(seqs)
        chunk_size = int(self.config.engine_prefill_chunk_size)
        with self.scheduler_capacity_snapshot():
            budgets = self.prompt_admission_budgets(waiting, chunk_size)
            aggregate_costs: dict[str, int] = {}
            logical_cost = 0
            for seq in seqs:
                for name, cost in self.prompt_admission_costs(seq).items():
                    aggregate_costs[name] = (
                        int(aggregate_costs.get(name, 0)) + int(cost)
                    )
                logical_cost += int(self.prompt_logical_reservation_cost(seq))
            if any(
                int(aggregate_costs.get(name, 0)) > int(budgets.get(name, 0))
                for name in aggregate_costs
            ):
                return False
            return logical_cost <= int(self.prompt_admission_free_slots())

    def startup_decode_batch_fits(self, seqs: list[Sequence]) -> bool:
        """Check a parked batch's next append without reserving or mutating KV."""
        if not seqs:
            raise ValueError("Startup decode capacity check requires a non-empty batch.")
        with self.scheduler_capacity_snapshot():
            remaining = self.decode_step_free_slots()
            for seq in seqs:
                cost = self.decode_step_reservation_cost(seq)
                if min(remaining, self.decode_step_free_slots_for(seq)) < cost:
                    return False
                remaining -= cost
        return True

    def prefill_step_free_slots(self) -> int:
        free_slots = int(self.cache_manager.prefill_step_free_slots())
        reserved = self.decode_reservations.outstanding()
        if reserved:
            free_slots = self.cache_manager.prefill_capacity_after_decode_reservations(
                free_slots, reserved, admission=False,
            )
        return max(0, free_slots + self._mixed_prefix_step_reclaimable_slots())

    def prefill_batched_tokens_margin(self) -> int:
        return int(self.cache_manager.prefill_batched_tokens_margin())

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        return int(self.cache_manager.remaining_prefill_tokens(seq))

    def prefill_execution_mode(self, seq: Sequence) -> str:
        if getattr(seq, "multimodal_full_prefill", False):
            return "full"
        return str(self.cache_manager.prefill_execution_mode(seq))

    def prefill_batch_compatibility_key(self, seq: Sequence) -> object:
        return self.cache_manager.prefill_batch_compatibility_key(seq)

    def reset_prefill_execution_state(self, seq_id: int) -> None:
        self.cache_manager.reset_prefill_execution_state(int(seq_id))

    def complete_prefill_execution(self, seq: Sequence) -> None:
        self.cache_manager.complete_prefill_execution(seq)

    @cpu_timing.timed
    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], engine_prefill_chunk_size: int) -> int:
        return int(self.cache_manager.reserved_prefill_slots(waiting_seqs, engine_prefill_chunk_size))

    def should_schedule_full_prefill(self, seq: Sequence) -> bool:
        return bool(self.cache_manager.should_schedule_full_prefill(seq))

    def requires_full_prefill_step(self, seq: Sequence) -> bool:
        return bool(self.cache_manager.requires_full_prefill_step(seq))

    def requires_long_prefill_offload(self, seq: Sequence) -> bool:
        return bool(self.cache_manager.requires_long_prefill_offload(seq))

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        free_slots = int(
            self.cache_manager.prefill_step_free_slots_for(seq)
            + self._mixed_prefix_step_reclaimable_slots()
        )
        if self.prefix_cache_coordinator is None:
            return free_slots
        return int(min(free_slots, self._mixed_prefix_boundary_limit(seq)))

    def prefill_private_slots_for(self, seq: Sequence) -> int:
        return int(self.cache_manager.prefill_private_slots_for(seq))

    def min_final_prefill_chunk_size(self, seq: Sequence) -> int:
        return int(self.cache_manager.min_final_prefill_chunk_size(seq))

    def decode_step_free_slots(self) -> int:
        return int(
            self.cache_manager.decode_step_free_slots()
            + self._mixed_prefix_step_reclaimable_slots()
        )

    def decode_step_free_slots_for(self, seq: Sequence) -> int:
        return int(
            self.cache_manager.decode_step_free_slots_for(seq)
            + self._mixed_prefix_step_reclaimable_slots()
        )

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        return int(self.cache_manager.prefill_step_reservation_cost(seq, scheduled_tokens))

    def decode_step_reservation_cost(self, seq: Sequence) -> int:
        return int(self.cache_manager.decode_step_reservation_cost(seq))

    def prompt_admission_free_slots(self) -> int:
        free_slots = int(self.cache_manager.prompt_admission_free_slots())
        reserved = self.decode_reservations.outstanding()
        if reserved:
            free_slots = self.cache_manager.prefill_capacity_after_decode_reservations(
                free_slots, reserved, admission=True,
            )
        return max(0, free_slots + self._mixed_prefix_admission_reclaimable_slots())

    def prompt_admission_budgets(self, waiting_seqs, engine_prefill_chunk_size: int) -> dict[str, int]:
        budgets = dict(self.cache_manager.prompt_admission_budgets(waiting_seqs, engine_prefill_chunk_size))
        extra = self._mixed_prefix_admission_reclaimable_slots()
        if extra > 0:
            if "slots" in budgets:
                budgets["slots"] = int(budgets["slots"]) + extra
            elif len(budgets) == 1:
                key = next(iter(budgets))
                budgets[key] = int(budgets[key]) + extra
            else:
                raise RuntimeError(
                    "Mixed prefix admission accounting cannot add evictable slots to "
                    f"multi-budget cache manager budgets={budgets}."
                )
        # Decode reservations may be backed by reclaimable prefix slots too.
        # Add that capacity before subtracting reservations and clamping.
        reserved = self.decode_reservations.outstanding()
        for name in budgets:
            if name == "slots" and reserved:
                budgets[name] = max(
                    0, self.cache_manager.prefill_capacity_after_decode_reservations(
                        int(budgets[name]), reserved, admission=True,
                    ),
                )
            else:
                budgets[name] = max(0, int(budgets[name]) - reserved.get(name, 0))
        budgets["resident_seqs"] = max(
            0,
            int(self.config.max_num_seqs_in_gpu) - len(self._resident_seq_ids),
        )
        return budgets

    def prompt_admission_cost(self, seq: Sequence) -> int:
        cost = int(self.cache_manager.prompt_admission_cost(seq))
        if self.prefix_cache_coordinator is not None:
            cost += int(self.prefix_cache_coordinator.prefix_hit_evictable_slots(seq))
        return cost

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        cost = int(self.cache_manager.prompt_logical_reservation_cost(seq))
        if self.prefix_cache_coordinator is not None:
            cost += int(self.prefix_cache_coordinator.prefix_hit_evictable_slots(seq))
        return cost

    def prompt_admission_failure_action(self) -> str:
        return str(self.cache_manager.prompt_admission_failure_action())

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]) -> None:
        self.cache_manager.on_prompt_admitted(seq, costs)
        if int(costs.get("resident_seqs", 0) or 0) > 0:
            self._resident_seq_ids.add(int(seq.seq_id))

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        costs = dict(self.cache_manager.prompt_admission_costs(seq))
        costs["resident_seqs"] = (
            0 if int(seq.seq_id) in self._resident_seq_ids else 1
        )
        if self.prefix_cache_coordinator is None:
            return costs
        extra = int(self.prefix_cache_coordinator.prefix_hit_evictable_slots(seq))
        if extra <= 0:
            return costs
        if "slots" in costs:
            costs["slots"] = int(costs["slots"]) + extra
        elif len(costs) == 2 and "resident_seqs" in costs:
            key = next(key for key in costs if key != "resident_seqs")
            costs[key] = int(costs[key]) + extra
        else:
            raise RuntimeError(
                "Mixed prefix admission accounting cannot add hit-evictable slots to "
                f"multi-budget cache manager costs={costs}."
            )
        return costs

    def prefix_cache_inspect(self, token_ids: list[int], *, include_subtree: bool = False) -> dict[str, object]:
        if self.prefix_cache_coordinator is not None:
            return self.prefix_cache_coordinator.inspect(token_ids, include_subtree=include_subtree)
        return self.cache_manager.prefix_cache_inspect(token_ids, include_subtree=include_subtree)

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        if self.prefix_cache_coordinator is not None:
            return self.prefix_cache_coordinator.match(token_ids)
        return self.cache_manager.prefix_cache_match(token_ids)

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        if self.prefix_cache_coordinator is not None:
            return self.prefix_cache_coordinator.delete_subtree(token_ids)
        return self.cache_manager.prefix_cache_delete_subtree(token_ids)

    def prefix_cache_set_eviction_priority(self, token_ids: list[int], *, priority: int) -> dict[str, object]:
        if self.prefix_cache_coordinator is not None:
            return self.prefix_cache_coordinator.set_eviction_priority(token_ids, priority=priority)
        return self.cache_manager.prefix_cache_set_eviction_priority(token_ids, priority=priority)

    def free_slot_stats(self) -> dict[str, int]:
        stats = self.cache_manager.free_slot_stats()
        stats["decode_reservation_tokens"] = self.decode_reservations.window
        stats["resident_sequences"] = int(len(self._resident_seq_ids))
        stats["resident_sequence_capacity"] = int(self.config.max_num_seqs_in_gpu)
        stats["free_resident_sequence_slots"] = max(
            0,
            int(self.config.max_num_seqs_in_gpu) - len(self._resident_seq_ids),
        )
        if self.prefix_cache_coordinator is not None:
            stats.update(self.prefix_cache_coordinator.stats())
        if self.chain_cache_coordinator is not None:
            stats.update(self.chain_cache_coordinator.stats())
        return stats

    def debug_live_seq_slots(self) -> dict[int, int]:
        return dict(self.cache_manager.debug_live_seq_slots())
