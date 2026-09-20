"""CPU scheduling ahead of device completion, with ordered result retirement."""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence as ReadOnlySequence
from dataclasses import dataclass

import time

from sparseengine.utils.profiler import cpu_timing
from sparseengine.engine.async_scheduling.execution import AsyncDrainRequired
from sparseengine.engine.sequence import Sequence, SequenceStatus
from sparseengine.sampling_params import resolve_eos_token_ids
from sparseengine.utils.profiler import profiler


class _TokenHistoryView(ReadOnlySequence):
    """Bounded view of append-only history; snapshot cost is context independent."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.length = len(tokens)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, stride = index.indices(self.length)
            if stride == 1:
                return self.tokens[start:stop]
            return [self.tokens[i] for i in range(start, stop, stride)]
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        return self.tokens[index]


def execution_snapshot(seq: Sequence) -> Sequence:
    # copy.copy uses Sequence's compact IPC state and loses rank-0 history.
    result = object.__new__(Sequence)
    result.__dict__ = dict(seq.__dict__)
    result.num_tokens += seq.num_pending_outputs
    result.num_pending_outputs = 0
    result.token_ids = _TokenHistoryView(seq.token_ids)
    return result


@dataclass
class PendingStep:
    ticket: int
    seqs: list[Sequence]
    snapshots: list[Sequence]
    is_prefill: bool
    publishes: list[bool]


class AsyncScheduler:
    def __init__(self, engine):
        self.engine = engine
        self.pending = deque()
        self.retiring: set[int] = set()
        self.discarded: set[int] = set()
        self.next_ticket = 0
        controller = getattr(engine.model_runner, "sparse_controller", None)
        self.requires_committed_token_history = bool(
            getattr(controller, "requires_committed_token_history", False))

    def abort(self, seq_id):
        scheduler = self.engine.scheduler
        owns = scheduler.abort(seq_id)
        self.discarded.add(seq_id)
        if owns or any(seq_id == s.seq_id for p in self.pending for s in p.seqs):
            self.retiring.add(seq_id)
        self._drain_discarded()
        self._retire_ready()

    def _drain_discarded(self):
        while self.pending and all(s.seq_id in self.discarded for s in self.pending[0].seqs):
            pending = self.pending[0]
            self.engine.model_runner.call("collect_async", pending.ticket, sorted(self.discarded))
            self.pending.popleft()
            for seq, publish in zip(pending.seqs, pending.publishes):
                if publish:
                    seq.num_pending_outputs -= 1

    def _retire_ready(self):
        referenced = {s.seq_id for p in self.pending for s in p.seqs}
        ready = self.retiring - referenced
        if ready:
            self.engine.model_runner.call("retire_async", sorted(ready))
            self.retiring.difference_update(ready)
            self.discarded.difference_update(ready)

    @cpu_timing.timed
    def _submit(self):
        engine = self.engine
        scheduler = engine.scheduler
        if getattr(engine, "_pending_prefix_prune_ids", ()):
            return False
        if self.requires_committed_token_history and any(
            any(step.publishes) for step in self.pending
        ):
            return False
        # A request whose final output is in flight needs no further work.
        if not scheduler.waiting and not any(
            s.num_completion_tokens + s.num_pending_outputs < s.max_tokens
            for s in scheduler.decoding
        ):
            return False
        scheduler._async_inflight = len(self.pending)
        try:
            with profiler.record("async_schedule"):
                seqs, is_prefill, preempted = scheduler.schedule()
        except AsyncDrainRequired:
            return False
        finally:
            scheduler._async_inflight = 0
        engine._release_preempted_sequences(preempted)
        if not seqs:
            return False
        if any(s.is_recompute_replay for s in seqs):
            # Recompute consumes accepted historical tokens, not device feedback.
            # Return ownership to the ordinary scheduler after draining.
            if is_prefill:
                scheduler.waiting.extendleft(reversed(seqs))
            return False
        snapshots = [execution_snapshot(s) for s in seqs]
        publishes = [not is_prefill or s.is_last_chunk_prefill for s in snapshots]
        ticket = self.next_ticket
        self.next_ticket += 1
        try:
            engine.model_runner.call("submit_async", ticket, snapshots, is_prefill)
        except BaseException:
            # Keep partially allocated requests visible to cancellation after a
            # failed submission, including the first prefill chunk.
            if is_prefill:
                scheduler.waiting.extendleft(reversed(seqs))
            raise
        for seq, publish in zip(seqs, publishes):
            if publish:
                seq.num_pending_outputs = seq.num_pending_outputs + 1
            if is_prefill:
                seq.num_prefilled_tokens += seq.current_chunk_size
                if seq.num_prefilled_tokens < seq.num_prompt_tokens:
                    seq.status = SequenceStatus.WAITING
                    scheduler._prefill_wait_since.setdefault(seq.seq_id, time.monotonic())
                    scheduler.waiting.appendleft(seq)
                else:
                    scheduler.memory_oracle.complete_prefill_execution(seq)
                    seq.status = SequenceStatus.RUNNING
                    scheduler.decoding.append(seq)
        self.pending.append(PendingStep(ticket, seqs, snapshots, is_prefill, publishes))
        return True

    @cpu_timing.timed
    def step(self):
        engine = self.engine
        with profiler.record("step"):
            while len(self.pending) < engine.config.async_max_inflight:
                if not self._submit():
                    break
            if not self.pending:
                # Capacity/recompute boundaries use the established synchronous
                # recovery path, with no outstanding GPU storage references.
                return engine._step_sync()
            pending = self.pending[0]
            tokens, logs = engine.model_runner.call(
                "collect_async", pending.ticket, sorted(self.discarded)
            )
            self.pending.popleft()
            sampled, tops = logs if logs is not None else ([None]*len(tokens), [None]*len(tokens))
            engine.last_step_token_outputs = []
            engine.last_step_logprob_outputs = []
            engine.last_step_prompt_cache_hits = [
                (s.seq_id, s.prefix_cache_hit_len) for s in pending.snapshots
            ] if pending.is_prefill else []
            finished = []
            for seq, token, logprob, top, publish in zip(
                pending.seqs, tokens, sampled, tops, pending.publishes
            ):
                if publish:
                    seq.num_pending_outputs -= 1
                if not publish or seq.seq_id in self.discarded:
                    continue
                seq.append_token(token, logprob, top)
                engine.last_step_token_outputs.append((seq.seq_id, [token]))
                engine.last_step_logprob_outputs.append((seq.seq_id, [logprob], [top]))
                eos = resolve_eos_token_ids(seq.eos_token_ids, engine.scheduler.eos_token_ids)
                if (not seq.ignore_eos and token in eos) or seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    if seq in engine.scheduler.decoding:
                        engine.scheduler.decoding.remove(seq)
                    self.discarded.add(seq.seq_id)
                    self.retiring.add(seq.seq_id)
                    finished.append((seq.seq_id, seq.completion_token_ids,
                                     seq.completion_token_logprobs, seq.completion_top_logprobs))
            self._drain_discarded()
            self._retire_ready()
            count = sum(s.current_chunk_size for s in pending.snapshots) if pending.is_prefill else -len(tokens)
            engine._throughput_logger.record_step(count)
            scheduler = engine.scheduler
            prefills, decodes = len(scheduler.waiting), len(scheduler.decoding)
            modes = scheduler.prefill_execution_mode_counts()
            batch = ("pf-" + scheduler.prefill_execution_mode_for_batch(pending.snapshots)
                     if pending.is_prefill else "decode")
            engine._throughput_logger.record_state(
                prefills + decodes, prefills, decodes,
                modes["chunked"], modes["full"], modes["raw_offload"], batch,
            )
            return finished, count
