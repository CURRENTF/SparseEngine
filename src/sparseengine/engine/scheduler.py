import os
import time
from collections import deque
from collections.abc import Callable, Iterable

from sparseengine.config import Config
from sparseengine.engine.prefill import (
    PREFILL_EXECUTION_CHUNKED,
    PREFILL_EXECUTION_FULL,
    PREFILL_EXECUTION_RAW_OFFLOAD,
    validate_prefill_execution_mode,
)
from sparseengine.engine.sequence import Sequence, SequenceStatus
from sparseengine.engine.runtime_state import MemoryOracle
from sparseengine.sampling_params import resolve_eos_token_ids
from sparseengine.utils.log import logger
from sparseengine.utils.profiler import cpu_timing


class Scheduler:
    """
    请求调度器，负责管理待处理 (waiting) 和正在运行 (running) 的序列。
    主要职责：
    1. 决定每一轮 (step) GPU 应该处理哪些序列。
    2. 实现分块 Prefill (Chunked Prefill) 以处理长序列。
    3. 管理逻辑显存额度，并在显存不足时触发抢占 (Preemption/Eviction)。
    """

    def __init__(
        self,
        config: Config,
        memory_oracle: MemoryOracle,
        prefix_cache_hit_refresher: Callable[[Sequence], None] | None = None,
        decode_capacity_reclaimer: Callable[[Sequence], Sequence | None] | None = None,
        *,
        prefix_cache_hits_refresher: Callable[[list[Sequence]], None] | None = None,
        prefill_capacity_reclaimer: Callable[[Sequence], bool] | None = None,
    ):
        self.config = config
        self.max_num_seqs_in_batch = config.max_num_seqs_in_batch
        self.max_num_batched_tokens = config.max_num_batched_tokens
        logger.debug(f'set max_num_batched_tokens = {config.max_num_batched_tokens} in Scheduler')
        self.max_decoding_seqs = config.max_decoding_seqs
        self.favor_min_decoding_seqs = getattr(config, "favor_min_decoding_seqs", 0)
        self.phase = "prefill"
        self._prefill_wait_since: dict[int, float] = {}
        self._phase_started_at = time.monotonic()
        self.last_phase_decision: dict = {}
        self._phase_reason = "prefill_priority"

        self.engine_prefill_chunk_size = config.engine_prefill_chunk_size
        self.prefill_schedule_policy = config.prefill_schedule_policy
        self.eos = config.eos
        self.eos_token_ids = resolve_eos_token_ids(
            configured_eos_token_ids=getattr(
                config, "eos_token_ids", ()
            ),
            fallback_eos_token_id=self.eos,
        )

        
        # memory_oracle 引用 Rank 0 的 CacheManager，作为全局显存余量参考。
        # 对多层异构预算，采用更保守的可用空间估计。
        self.memory_oracle = memory_oracle
        self.decode_capacity_reclaimer = decode_capacity_reclaimer
        self.prefill_capacity_reclaimer = prefill_capacity_reclaimer
        self.prefix_cache_hits_refresher = prefix_cache_hits_refresher
        self.prefix_cache_hit_refresher = (
            memory_oracle.refresh_prefix_cache_hit
            if prefix_cache_hit_refresher is None
            else prefix_cache_hit_refresher
        )
        
        self.waiting: deque[Sequence] = deque()
        self.decoding: deque[Sequence] = deque()
        self._admission_defer_warned_seq_ids: set[int] = set()
        self.total_preemptions = 0
        self.total_recompute_replays = 0

    def _prefill_execution_mode(self, seq: Sequence) -> str:
        return validate_prefill_execution_mode(
            self.memory_oracle.prefill_execution_mode(seq)
        )

    def _prefill_batch_key(self, seq: Sequence) -> tuple[str, object]:
        mode = self._prefill_execution_mode(seq)
        compatibility = self.memory_oracle.prefill_batch_compatibility_key(seq)
        try:
            hash(compatibility)
        except TypeError as exc:
            raise TypeError(
                "prefill_batch_compatibility_key must be hashable: "
                f"seq_id={seq.seq_id} key={compatibility!r}."
            ) from exc
        return mode, compatibility

    def prefill_execution_mode_counts(self) -> dict[str, int]:
        """Observe queued modes using their last refreshed scheduling metadata."""
        counts = {
            PREFILL_EXECUTION_CHUNKED: 0,
            PREFILL_EXECUTION_FULL: 0,
            PREFILL_EXECUTION_RAW_OFFLOAD: 0,
        }
        for seq in self.waiting:
            counts[self._prefill_execution_mode(seq)] += 1
        return counts

    def prefill_execution_mode_for_batch(self, seqs: list[Sequence]) -> str:
        """Return the single execution mode guaranteed by prefill bucketing."""
        batch_keys = {self._prefill_batch_key(seq) for seq in seqs}
        if len(batch_keys) != 1:
            raise RuntimeError(
                "Scheduler produced an incompatible prefill batch: "
                f"keys={batch_keys!r} seq_ids={[seq.seq_id for seq in seqs]}."
            )
        mode, _compatibility = batch_keys.pop()
        return mode

    def _refresh_prefill_metadata(self, seq: Sequence) -> None:
        if seq.num_prefilled_tokens == 0 and seq.num_completion_tokens == 0:
            self.prefix_cache_hit_refresher(seq)

    @staticmethod
    def _prefill_priority(seq: Sequence, replay_pending: bool) -> int:
        if not replay_pending:
            return 0
        # While replay blocks fresh admission, a partial prefill already owns
        # KV and reserves the capacity needed to finish its remaining prompt.
        partial = 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens
        if partial:
            return 1 if seq.is_recompute_replay else 0
        return 2 if seq.is_recompute_replay else 3

    def _prefill_mode_order(self) -> list[tuple[int, str, object]]:
        replay_pending = any(seq.is_recompute_replay for seq in self.waiting)
        candidates: list[Sequence] = []
        for seq in self.waiting:
            if replay_pending and self._prefill_priority(seq, replay_pending) == 3:
                continue
            # Match candidate admission before prefix refresh, which broadcasts
            # an RPC to every TP rank. Blocked fresh prompts cannot run yet.
            if len(self.decoding) >= self.max_decoding_seqs and seq.num_prefilled_tokens == 0:
                continue
            candidates.append(seq)
        if self.prefix_cache_hits_refresher is not None:
            fresh = [
                seq for seq in candidates
                if seq.num_prefilled_tokens == 0 and seq.num_completion_tokens == 0
            ]
            if fresh:
                self.prefix_cache_hits_refresher(fresh)
        modes: list[tuple[int, str, object]] = []
        for seq in sorted(candidates, key=lambda seq: self._prefill_priority(seq, replay_pending)):
            if self.prefix_cache_hits_refresher is None:
                self._refresh_prefill_metadata(seq)
            batch_key = (self._prefill_priority(seq, replay_pending), *self._prefill_batch_key(seq))
            if batch_key not in modes:
                modes.append(batch_key)
        return modes

    def _pop_next_prefill_seq(
        self,
        target_priority: int,
        target_mode: str,
        target_compatibility: object,
        *,
        replay_pending: bool,
        skipped: deque[Sequence],
    ) -> Sequence | None:
        while self.waiting:
            seq = self.waiting[0]
            priority = self._prefill_priority(seq, replay_pending)
            eligible = priority == target_priority and (not replay_pending or priority != 3)
            eligible = eligible and not (
                len(self.decoding) >= self.max_decoding_seqs
                and seq.num_prefilled_tokens == 0
            )
            if eligible and self._prefill_batch_key(seq) == (target_mode, target_compatibility):
                return self.waiting.popleft()
            skipped.append(self.waiting.popleft())
        return None

    def is_finished(self):
        """判断所有请求是否已处理完成"""
        return len(self.waiting) == 0 and len(self.decoding) == 0

    def add(self, seq: Sequence):
        """将新请求加入等待队列"""
        if self.is_finished():
            self.phase = "prefill"
            self._phase_started_at = time.monotonic()
        self._prefill_wait_since.setdefault(seq.seq_id, time.monotonic())
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> bool:
        """Remove a request from scheduler queues.

        Returns True when the sequence may own KV slots and the caller should
        notify ModelRunner.free_slots(seq_id).
        """
        return self.abort_many((seq_id,)).get(int(seq_id), False)

    def abort_many(self, seq_ids: Iterable[int]) -> dict[int, bool]:
        """Remove requests with one stable pass over each scheduler queue."""
        requested = {int(seq_id) for seq_id in seq_ids}
        if not requested:
            return {}
        for seq_id in requested:
            self._prefill_wait_since.pop(seq_id, None)
        removed: dict[int, bool] = {}
        for queue in (self.waiting, self.decoding):
            retained = deque()
            while queue:
                seq = queue.popleft()
                seq_id = int(seq.seq_id)
                if seq_id not in requested:
                    retained.append(seq)
                    continue
                may_own_slots = (
                    seq.status == SequenceStatus.RUNNING
                    or seq.num_prefilled_tokens > 0
                    or queue is self.decoding
                )
                seq.status = SequenceStatus.FINISHED
                self._admission_defer_warned_seq_ids.discard(seq_id)
                self.memory_oracle.reset_prefill_execution_state(seq_id)
                removed[seq_id] = bool(removed.get(seq_id, False) or may_own_slots)
            queue.extend(retained)
        return removed

    def request_may_own_slots(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.decoding):
            for seq in queue:
                if seq.seq_id == seq_id:
                    return bool(
                        seq.status == SequenceStatus.RUNNING
                        or seq.num_prefilled_tokens > 0
                        or queue is self.decoding
                    )
        return False

    def _reserved_prefill_tokens(self) -> int:
        return int(self.memory_oracle.reserved_prefill_slots(self.waiting, self.engine_prefill_chunk_size))

    def _can_continue_prefill_batch(
        self,
        *,
        target_mode: str,
        scheduled_seqs: list[Sequence],
        step_free_count: int,
        num_batched_tokens: int,
        num_batched_seqs: int,
        margin_batched_tokens: int,
    ) -> bool:
        if not self.waiting:
            return False
        if scheduled_seqs and scheduled_seqs[0].is_recompute_replay:
            # A replay rebuilds cache for an already accepted request. Keep the
            # whole prefill step isolated so a fresh or partial prompt cannot
            # inflate its compute workspace or consume the slots it needs.
            return False
        if target_mode == PREFILL_EXECUTION_RAW_OFFLOAD:
            return not scheduled_seqs and step_free_count > 0
        return (
            num_batched_tokens <= self.max_num_batched_tokens - margin_batched_tokens
            and num_batched_seqs < self.max_num_seqs_in_batch
        )

    def _prefill_step_tokens(
        self,
        *,
        seq: Sequence,
        mode: str,
        remaining_prefill_tokens: int,
        num_batched_tokens: int,
        step_free_count: int,
    ) -> int:
        if mode == PREFILL_EXECUTION_FULL:
            available = min(
                self.max_num_batched_tokens - num_batched_tokens,
                step_free_count,
            )
            if remaining_prefill_tokens <= available:
                return int(remaining_prefill_tokens)
            return 0
        if mode in {
            PREFILL_EXECUTION_CHUNKED,
            PREFILL_EXECUTION_RAW_OFFLOAD,
        }:
            return min(
                remaining_prefill_tokens,
                self.engine_prefill_chunk_size,
                self.max_num_batched_tokens - num_batched_tokens,
                step_free_count,
            )
        raise ValueError(f"Unknown prefill execution mode={mode!r}")

    def _respect_min_final_prefill_chunk(
        self,
        seq: Sequence,
        remaining_prefill_tokens: int,
        proposed_tokens: int,
    ) -> int:
        min_final = int(self.memory_oracle.min_final_prefill_chunk_size(seq))
        if min_final < 0:
            raise ValueError(
                "min_final_prefill_chunk_size must be non-negative, "
                f"got {min_final} for seq_id={seq.seq_id}."
            )
        remaining_after = int(remaining_prefill_tokens) - int(proposed_tokens)
        if min_final == 0 or remaining_after <= 0 or remaining_after >= min_final:
            return int(proposed_tokens)
        return max(0, int(remaining_prefill_tokens) - min_final)

    def _raise_prompt_admission_failure(
        self,
        seq: Sequence,
        failed_budget: str,
        need: int,
        free: int,
        *,
        physical_free_count: int,
        reserved_prefill: int,
        logical_free_count: int,
        admission_budgets: dict[str, int],
    ):
        raise RuntimeError(
            "Insufficient KV cache slots to admit prompt. "
            f"cache_manager={type(self.memory_oracle).__name__} prompt_len={seq.num_prompt_tokens} "
            f"failed_budget={failed_budget} need={need} free={free} budgets={admission_budgets} "
            f"free_slots={physical_free_count} reserved_prefill={reserved_prefill} "
            f"logical_free={logical_free_count}"
        )

    def _preempt_decode_victim(
        self,
        victim: Sequence,
        scheduled_seqs: list[Sequence],
        preempted_seqs: list[Sequence],
        *,
        physical_free_count: int,
        reserved_prefill: int,
    ) -> tuple[list[Sequence], bool, list[Sequence]]:
        if getattr(self, "_async_inflight", 0):
            from sparseengine.engine.async_scheduling.execution import AsyncDrainRequired
            # Every caller removes the victim before entering this helper.
            # Draining must preserve its ownership and cancellation visibility.
            self.decoding.appendleft(victim)
            self.decoding.extendleft(reversed(scheduled_seqs))
            raise AsyncDrainRequired("Preemption requires completed in-flight KV users")
        has_waiting_replay = any(
            seq.is_recompute_replay for seq in self.waiting
        )
        made_progress_since_handoff = (
            int(victim.num_completion_tokens)
            > int(victim.decode_progress_checkpoint)
        )
        if (
            victim.num_completion_tokens > 0
            and not self.decoding
            and not (
                has_waiting_replay and made_progress_since_handoff
            )
        ):
            # The caller popped the candidate, but a rejected preemption must
            # retain scheduler ownership so cancellation can reclaim its KV.
            self.decoding.appendleft(victim)
            self.decoding.extendleft(reversed(scheduled_seqs))
            raise RuntimeError(
                "KV cache is too small for the sole remaining decode request "
                "to make forward progress. Recompute replay would rebuild the "
                "same cache state and loop forever. "
                f"seq_id={victim.seq_id} prompt_len={victim.num_prompt_tokens} "
                f"num_tokens={victim.num_tokens} completion_tokens={victim.num_completion_tokens} "
                f"decode_progress_checkpoint={victim.decode_progress_checkpoint} "
                f"free_slots={physical_free_count} reserved_prefill={reserved_prefill}."
            )
        debug_slots = os.getenv("SPARSEENGINE_DEBUG_SLOTS", "0") == "1"
        if debug_slots:
            logger.info(
                "preempt seq_id={} prompt_len={} num_tokens={} prefetched={} free_slots_before={} waiting_before={} decoding_before={}",
                victim.seq_id,
                int(victim.num_prompt_tokens),
                int(victim.num_tokens),
                int(victim.num_prefilled_tokens),
                int(self.memory_oracle.num_free_slots),
                len(self.waiting),
                len(self.decoding),
            )
        victim.status = SequenceStatus.WAITING
        if victim.num_completion_tokens > 0:
            victim.start_recompute_replay()
            self.total_recompute_replays += 1
            logger.warning(
                "recompute_replay_start seq_id={} prompt_tokens={} completion_tokens={} active_decodes={}",
                victim.seq_id,
                victim.num_prompt_tokens,
                victim.num_completion_tokens,
                len(self.decoding),
            )
        else:
            victim.num_prefilled_tokens = 0  # 重置进度，下次回来重新跑 Prefill
        for survivor in self.decoding:
            survivor.decode_progress_checkpoint = int(
                survivor.num_completion_tokens
            )
        self.memory_oracle.clear_prefix_cache_hit(victim)
        self.memory_oracle.reset_prefill_execution_state(victim.seq_id)
        # Requeue to the tail. While a replay is pending, prefill scheduling is
        # suspended until active decodes drain; the victim therefore cannot
        # immediately consume the slots it just released.
        self._prefill_wait_since.setdefault(victim.seq_id, time.monotonic())
        self.waiting.append(victim)
        # Any decode sequences already popped into `scheduled_seqs` in this round
        # have not been executed yet. Put them back before returning, otherwise
        # they disappear from scheduler queues while still occupying KV slots.
        if scheduled_seqs:
            self.decoding.extendleft(reversed(scheduled_seqs))
            scheduled_seqs.clear()
        preempted_seqs.append(victim)
        self.total_preemptions += 1
        logger.warning(f'驱逐请求 id = {victim.seq_id} | slots={self.memory_oracle.free_slot_stats()}')
        return [], False, preempted_seqs

    def _eligible_decode_batch(self) -> list[Sequence]:
        """Mirror decode selection without popping or preempting."""
        if not self.decoding:
            return []
        free = max(0, int(self.memory_oracle.decode_step_free_slots()))
        batch = []
        for seq in self.decoding:
            if seq.num_completion_tokens + seq.num_pending_outputs >= seq.max_tokens:
                continue
            cost = int(self.memory_oracle.decode_step_reservation_cost(seq))
            if min(free, int(self.memory_oracle.decode_step_free_slots_for(seq))) < cost:
                if free <= 0:
                    break
                continue
            free -= cost
            batch.append(seq)
            if len(batch) == self.max_decoding_seqs:
                break
        return batch

    @cpu_timing.timed
    def schedule(self) -> tuple[list[Sequence], bool, list[Sequence]]:
        now = time.monotonic()
        for seq in self.waiting:
            self._prefill_wait_since.setdefault(seq.seq_id, now)
        waiting_before = list(self.waiting)
        decoding_before = list(self.decoding)
        self.last_phase_decision = {}
        try:
            snapshot = getattr(self.memory_oracle, "scheduler_capacity_snapshot", None)
            if callable(snapshot):
                with snapshot():
                    result = self._schedule_impl()
            else:
                result = self._schedule_impl()
        except Exception:
            # A popped prefill candidate must remain abortable after hook failure.
            owned = {seq.seq_id for seq in (*self.waiting, *self.decoding)}
            for seq in waiting_before:
                if seq.seq_id not in owned:
                    self.waiting.append(seq)
            for seq in decoding_before:
                if seq.seq_id not in owned and seq not in self.waiting:
                    self.decoding.append(seq)
            raise
        seqs, is_prefill, _ = result
        if seqs:
            phase = "prefill" if is_prefill else "decode"
            waits = []
            if is_prefill:
                for seq in seqs:
                    since = self._prefill_wait_since.pop(seq.seq_id, now)
                    waits.append({"seq_id": seq.seq_id, "partial": seq.num_prefilled_tokens > 0,
                                  "seconds": now - since})
            switched = phase != self.phase
            self.last_phase_decision = {
                "phase": phase, "reason": self._phase_reason,
                "switched": switched, "previous_phase_seconds": now - self._phase_started_at,
                "decode_batch": 0 if is_prefill else len(seqs), "prefill_waits": waits,
            }
            if switched:
                self._phase_started_at = now
            self.phase = phase
            logger.debug("scheduler_phase {}", self.last_phase_decision)
        return result

    def _schedule_impl(self, *, allow_prefill_reclaim: bool = True) -> tuple[list[Sequence], bool, list[Sequence]]:
        """
        核心调度逻辑。
        返回：(本次要运行的序列列表, 是否是 Prefill 阶段, 本次被抢占的序列列表)
        
        注意：目前为了简化算子实现，单次 step 不支持 Prefill 和 Decode 混合。
        """
        scheduled_seqs = []
        preempted_seqs = []
        num_batched_seqs = 0
        num_batched_tokens = 0
        decode_reservation_failure = self.memory_oracle.reserve_decode_windows(self.decoding, self.waiting)
        if decode_reservation_failure is not None and self.decode_capacity_reclaimer is not None:
            decode_reservation_failure = self.decode_capacity_reclaimer(decode_reservation_failure)
        if decode_reservation_failure is not None and len(self.decoding) > 1:
            self.decoding.remove(decode_reservation_failure)
            return self._preempt_decode_victim(
                decode_reservation_failure, scheduled_seqs, preempted_seqs,
                physical_free_count=self.memory_oracle.num_free_slots,
                reserved_prefill=self._reserved_prefill_tokens(),
            )

        # Decide decode affinity before querying unused prefill capacity.
        self._phase_reason = "prefill_priority"
        overdue = False
        if self.favor_min_decoding_seqs:
            overdue = any(time.monotonic() - since >= 60.0
                          for since in self._prefill_wait_since.values())
            self._phase_reason = (
                "prefill_wait_timeout" if overdue else
                "prefill_continue" if self.phase == "prefill" else "decode_below_threshold"
            )
            if self.phase == "decode" and not overdue and decode_reservation_failure is None:
                batch = self._eligible_decode_batch()
                if batch and (not self.waiting or len(batch) >= self.favor_min_decoding_seqs):
                    self._phase_reason = "no_prefill" if not self.waiting else "decode_affinity"
                    return batch, False, []

        # 逻辑可用空间计数器，用于在本轮调度中预估显存占用
        physical_free_count = self.memory_oracle.num_free_slots
        if self.waiting:
            reserved_prefill = self._reserved_prefill_tokens()
            prompt_logical_free_count = max(
                0,
                int(self.memory_oracle.prompt_admission_free_slots())
                - reserved_prefill,
            )
            step_free_count = int(
                self.memory_oracle.prefill_step_free_slots()
            )
            admission_budgets = dict(
                self.memory_oracle.prompt_admission_budgets(
                    self.waiting,
                    self.engine_prefill_chunk_size,
                )
            )
            margin_batched_tokens = (
                self.memory_oracle.prefill_batched_tokens_margin()
            )
        else:
            reserved_prefill = 0
            prompt_logical_free_count = 0
            step_free_count = int(physical_free_count)
            admission_budgets = {}
            margin_batched_tokens = 0
        decode_logical_free_count = max(0, int(self.memory_oracle.decode_step_free_slots()))
        deferred_prompt_failure: tuple[Sequence, str, int, int] | None = None
        blocked_prefill_step_failure: tuple[Sequence, int, int] | None = None
        blocked_prefill_capacity_failure: tuple[Sequence, int, int, int] | None = None
        blocked_final_prefill_window_failure: tuple[Sequence, int, int] | None = None
        charged_shared_resources: dict[str, set[object]] = {}

        if overdue:
            # Preserve admission accounting order before prioritizing old work.
            self.waiting = deque(sorted(
                self.waiting, key=lambda seq: self._prefill_wait_since[seq.seq_id]
            ))

        # --- 阶段 1: Prefill 调度 ---
        # Affinity may already have selected decode above.
        prefill_mode_order: list[tuple[int, str, object]] = []
        replay_waiting = [seq for seq in self.waiting if seq.is_recompute_replay]
        if self.waiting and not (replay_waiting and self.decoding):
            # A preempted request may only rebuild after the surviving decode
            # set drains. Rebuilding immediately would consume the KV capacity
            # that preemption just released and can make the scheduler alternate
            # between replay victims without producing another token.
            prefill_mode_order = self._prefill_mode_order()

        # No replay is created during prefill selection, and accepting one ends
        # this step. Keep the snapshot local to this scheduling invocation.
        replay_pending = any(seq.is_recompute_replay for seq in self.waiting)
        for target_priority, target_mode, target_compatibility in prefill_mode_order:
            if scheduled_seqs:
                break
            bucket_scan_budget = len(self.waiting)
            skipped_prefill = deque()
            try:
                while (
                    bucket_scan_budget > 0
                    and self._can_continue_prefill_batch(
                        target_mode=target_mode,
                        scheduled_seqs=scheduled_seqs,
                        step_free_count=step_free_count,
                        num_batched_tokens=num_batched_tokens,
                        num_batched_seqs=num_batched_seqs,
                        margin_batched_tokens=margin_batched_tokens,
                    )
                ):
                    seq = self._pop_next_prefill_seq(
                        target_priority,
                        target_mode,
                        target_compatibility,
                        replay_pending=replay_pending,
                        skipped=skipped_prefill,
                    )
                    if seq is None:
                        break
                    bucket_scan_budget -= 1
                    remaining_prefill_tokens = self.memory_oracle.remaining_prefill_tokens(seq)
                    candidate_step_free_count = int(self.memory_oracle.prefill_step_free_slots_for(seq))
                    uses_full_prefill_staging = bool(
                        self.memory_oracle.should_schedule_full_prefill(seq)
                    )
                    if (
                        target_mode != PREFILL_EXECUTION_RAW_OFFLOAD
                        and not uses_full_prefill_staging
                    ):
                        candidate_step_free_count = min(
                            int(step_free_count) + self.memory_oracle.prefill_private_slots_for(seq),
                            int(candidate_step_free_count),
                        )

                    # 异常处理：如果由于某种原因已经 prefill 完却还在 waiting 队列
                    if remaining_prefill_tokens <= 0:
                        raise ValueError('BUG：理论上不应该在 waiting 里')

                    # 确定本次 Chunk 的大小
                    can_prefill_tokens = self._prefill_step_tokens(
                        seq=seq,
                        mode=target_mode,
                        remaining_prefill_tokens=remaining_prefill_tokens,
                        num_batched_tokens=num_batched_tokens,
                        step_free_count=candidate_step_free_count,
                    )
                    proposed_prefill_tokens = int(can_prefill_tokens)
                    can_prefill_tokens = self._respect_min_final_prefill_chunk(
                        seq,
                        remaining_prefill_tokens,
                        can_prefill_tokens,
                    )

                    if can_prefill_tokens <= 0:
                        min_final = int(
                            self.memory_oracle.min_final_prefill_chunk_size(seq)
                        )
                        if (
                            min_final > 0
                            and remaining_prefill_tokens <= min_final
                            and proposed_prefill_tokens < remaining_prefill_tokens
                            and blocked_final_prefill_window_failure is None
                        ):
                            blocked_final_prefill_window_failure = (
                                seq,
                                int(min_final),
                                int(proposed_prefill_tokens),
                            )
                        if candidate_step_free_count <= 0 and step_free_count > 0:
                            if blocked_prefill_capacity_failure is None:
                                blocked_prefill_capacity_failure = (
                                    seq,
                                    int(remaining_prefill_tokens),
                                    int(candidate_step_free_count),
                                    int(step_free_count),
                                )
                        if target_mode == PREFILL_EXECUTION_FULL:
                            available = min(
                                self.max_num_batched_tokens - num_batched_tokens,
                                candidate_step_free_count,
                            )
                            blocked_prefill_step_failure = (seq, int(remaining_prefill_tokens), int(available))
                        logger.debug(f'{can_prefill_tokens=} 结束 schedule prefill 请求')
                        self.waiting.append(seq)
                        continue

                    # 逻辑显存分配检查：如果是新序列的起始，检查是否能容纳完整的 Prompt 长度。
                    # 采用保守策略：预先逻辑占位整个 Prompt，即使后续可能会有稀疏逐出。
                    # 只要我想尽可能地持续生成某个序列，那就应该提前都申请出来
                    if seq.num_prefilled_tokens == 0:
                        raw_costs = self.memory_oracle.prompt_admission_costs(seq)
                        shared_costs_fn = getattr(
                            self.memory_oracle, "prompt_admission_shared_costs", None
                        )
                        shared_costs = (
                            shared_costs_fn(seq) if callable(shared_costs_fn) else {}
                        )
                        costs = dict(raw_costs)
                        for name, resources in shared_costs.items():
                            already_charged = charged_shared_resources.get(name, set())
                            costs[name] = int(costs.get(name, 0)) - sum(
                                int(cost)
                                for resource_id, cost in resources.items()
                                if resource_id in already_charged
                            )
                            if costs[name] < 0:
                                raise RuntimeError(
                                    "Shared prompt admission costs exceed the raw budget cost: "
                                    f"budget={name} raw={raw_costs.get(name, 0)} adjusted={costs[name]}."
                                )
                        failed = None
                        for name, need in costs.items():
                            free = int(admission_budgets.get(name, 0) or 0)
                            if free < int(need):
                                failed = (name, int(need), free)
                                break
                        if failed is not None:
                            action = self.memory_oracle.prompt_admission_failure_action()
                            name, need, free = failed
                            if action == "defer":
                                if deferred_prompt_failure is None:
                                    deferred_prompt_failure = (seq, name, need, free)
                                if seq.seq_id not in self._admission_defer_warned_seq_ids:
                                    logger.warning(
                                        "Prompt admission deferred because the current batch/KV budget is saturated. "
                                        f"seq_id={seq.seq_id} prompt_len={seq.num_prompt_tokens} "
                                        f"failed_budget={name} need={need} free={free} "
                                        f"waiting={len(self.waiting) + 1} decoding={len(self.decoding)} "
                                        f"scheduled_prefill={len(scheduled_seqs)} free_slots={physical_free_count} "
                                        f"reserved_prefill={reserved_prefill}. "
                                        "This usually means batch size is too large for the current KV budget."
                                    )
                                    if os.getenv("SPARSEENGINE_DEBUG_SLOTS", "0") == "1" and len(self.decoding) == 0:
                                        live_seq_slots = self.memory_oracle.debug_live_seq_slots()
                                        live_seq_items = sorted(
                                            ((int(seq_id), int(n_slots)) for seq_id, n_slots in live_seq_slots.items()),
                                            key=lambda x: (-x[1], x[0]),
                                        )[:16]
                                        waiting_seq_ids_all = [int(s.seq_id) for s in self.waiting]
                                        decoding_seq_ids_all = [int(s.seq_id) for s in self.decoding]
                                        scheduled_seq_ids_all = [int(s.seq_id) for s in scheduled_seqs]
                                        known_seq_ids = (
                                            set(waiting_seq_ids_all)
                                            | set(decoding_seq_ids_all)
                                            | set(scheduled_seq_ids_all)
                                        )
                                        zombie_seq_ids = sorted(
                                            int(seq_id)
                                            for seq_id in live_seq_slots
                                            if int(seq_id) not in known_seq_ids
                                        )[:16]
                                        logger.info(
                                            "defer_with_no_decoding seq_id={} need={} free={} free_slots={} reserved_prefill={} "
                                            "scheduled_prefill={} waiting_seq_ids={} scheduled_seq_ids={} zombie_seq_ids={} "
                                            "live_seq_slots={}",
                                            seq.seq_id,
                                            int(need),
                                            int(free),
                                            int(physical_free_count),
                                            int(reserved_prefill),
                                            len(scheduled_seqs),
                                            waiting_seq_ids_all[:16],
                                            scheduled_seq_ids_all[:16],
                                            zombie_seq_ids,
                                            live_seq_items,
                                        )
                                    self._admission_defer_warned_seq_ids.add(seq.seq_id)
                                self.waiting.append(seq)
                                continue
                            self._raise_prompt_admission_failure(
                                seq,
                                name,
                                need,
                                free,
                                physical_free_count=physical_free_count,
                                reserved_prefill=reserved_prefill,
                                logical_free_count=prompt_logical_free_count,
                                admission_budgets=admission_budgets,
                            )
                        self._admission_defer_warned_seq_ids.discard(seq.seq_id)
                        for name, need in costs.items():
                            admission_budgets[name] = int(admission_budgets.get(name, 0) or 0) - int(need)
                        # Admission hooks may acquire residency before raising.
                        # Keep cancellation responsible for releasing that ownership.
                        seq.status = SequenceStatus.RUNNING
                        self.memory_oracle.on_prompt_admitted(seq, raw_costs)
                        if int(getattr(seq, "prefix_cache_hit_len", 0) or 0) > 0:
                            seq.num_prefilled_tokens = int(seq.prefix_cache_hit_len)
                        logical_need = int(
                            self.memory_oracle.prompt_logical_reservation_cost(seq)
                        )
                        logical_need -= sum(
                            int(cost)
                            for name, resources in shared_costs.items()
                            for resource_id, cost in resources.items()
                            if resource_id in charged_shared_resources.get(name, set())
                        )
                        if prompt_logical_free_count < logical_need:
                            # Fail fast: admission budgets should already account for reserved prefill headroom.
                            # Reaching this branch usually means a cache-manager-specific budget mismatch.
                            raise RuntimeError(
                                "Prompt admission budget mismatch after reservation check. "
                                f"cache_manager={type(self.memory_oracle).__name__} prompt_len={seq.num_prompt_tokens} "
                                f"logical_need={logical_need} logical_free={prompt_logical_free_count} "
                                f"budgets={admission_budgets} costs={costs} "
                                f"free_slots={physical_free_count} reserved_prefill={reserved_prefill}"
                            )
                        prompt_logical_free_count -= int(logical_need)
                        for name, resources in shared_costs.items():
                            charged_shared_resources.setdefault(name, set()).update(resources)

                    # 设置当前 Chunk 属性并标记状态
                    logger.debug(f'Add chunk prefill with {can_prefill_tokens} tokens.')
                    seq.current_chunk_size = can_prefill_tokens
                    num_batched_seqs += 1
                    num_batched_tokens += can_prefill_tokens
                    prefill_reservation_cost = self.memory_oracle.prefill_step_reservation_cost(
                        seq,
                        can_prefill_tokens,
                    )
                    step_free_count = max(0, step_free_count - int(prefill_reservation_cost))
                    seq.status = SequenceStatus.RUNNING
                    scheduled_seqs.append(seq)
                    if target_mode == PREFILL_EXECUTION_RAW_OFFLOAD:
                        break

            finally:
                self.waiting.extendleft(reversed(skipped_prefill))

        # 如果有 Prefill 请求被选中，直接返回，本次 step 只跑 Prefill。
        if scheduled_seqs:
            return scheduled_seqs, True, []

        # A partial prefill may release capacity through compaction. Give it
        # the opportunity to finish before declaring a sole decode unable to run.
        if decode_reservation_failure is not None:
            self.decoding.remove(decode_reservation_failure)
            return self._preempt_decode_victim(
                decode_reservation_failure, scheduled_seqs, preempted_seqs,
                physical_free_count=physical_free_count,
                reserved_prefill=reserved_prefill,
            )

        self._phase_reason = "prefill_unavailable" if self.waiting else "no_prefill"
        # --- 阶段 2: Decode 调度 ---
        # 只有在没有 Prefill 任务时才处理增量生成任务。
        decode_scan_budget = len(self.decoding)
        blocked_decode_victim: Sequence | None = None
        while (
            self.decoding
            and decode_scan_budget > 0
            and num_batched_seqs < self.max_decoding_seqs
        ):
            seq = self.decoding.popleft()
            decode_scan_budget -= 1
            if seq.num_completion_tokens + seq.num_pending_outputs >= seq.max_tokens:
                self.decoding.append(seq)
                continue

            # 检查逻辑空间是否够塞下一个新 Token (Decode 步进)
            candidate_decode_free = min(
                int(decode_logical_free_count),
                int(self.memory_oracle.decode_step_free_slots_for(seq)),
            )
            decode_reservation_cost = int(self.memory_oracle.decode_step_reservation_cost(seq))
            if candidate_decode_free < decode_reservation_cost:
                if decode_logical_free_count > 0:
                    if blocked_decode_victim is None:
                        blocked_decode_victim = seq
                    self.decoding.append(seq)
                    continue
                if scheduled_seqs:
                    # The current step still has useful work. Keep this request
                    # queued and run the partial decode batch before considering
                    # preemption; otherwise a fourth request can fail the whole
                    # step after three requests have already reserved its last
                    # three writable KV slots.
                    if blocked_decode_victim is None:
                        blocked_decode_victim = seq
                    self.decoding.append(seq)
                    break
                # 显存耗尽，触发驱逐/抢占逻辑
                # 策略：牺牲当前 seq，并立刻返回，让上层先释放槽位再进入下一轮调度。
                # 这样可以避免在一次 schedule() 调用中反复驱逐多个请求造成抖动。
                return self._preempt_decode_victim(
                    seq,
                    scheduled_seqs,
                    preempted_seqs,
                    physical_free_count=physical_free_count,
                    reserved_prefill=reserved_prefill,
                )
            else:
                # Reserve the cache-manager-specific decode capacity for this step.
                decode_logical_free_count -= decode_reservation_cost
                num_batched_seqs += 1
                scheduled_seqs.append(seq)
                # logger.debug('Add a decode req.')

        if not scheduled_seqs:
            if blocked_decode_victim is not None:
                try:
                    self.decoding.remove(blocked_decode_victim)
                except ValueError:
                    pass
                return self._preempt_decode_victim(
                    blocked_decode_victim,
                    scheduled_seqs,
                    preempted_seqs,
                    physical_free_count=physical_free_count,
                    reserved_prefill=reserved_prefill,
                )
            if (
                not self.decoding and self.waiting and allow_prefill_reclaim
                and self.prefill_capacity_reclaimer is not None
                and (deferred_prompt_failure is not None or step_free_count <= 0)
            ):
                if getattr(self, "_async_inflight", 0):
                    from sparseengine.engine.async_scheduling.execution import AsyncDrainRequired
                    raise AsyncDrainRequired("Prefill capacity reclaim requires completed in-flight KV users")
                # Zero capacity skips the prefill scan, so no admission failure
                # is recorded even when IDLE chains can make the prompt fit.
                seq = deferred_prompt_failure[0] if deferred_prompt_failure else self.waiting[0]
                if self.prefill_capacity_reclaimer(seq):
                    return self._schedule_impl(allow_prefill_reclaim=False)
            if blocked_prefill_step_failure is not None and not self.decoding:
                seq, need, free = blocked_prefill_step_failure
                raise RuntimeError(
                    "Prefill candidate requires an atomic prefill step but cannot fit. "
                    f"cache_manager={type(self.memory_oracle).__name__} "
                    f"seq_id={seq.seq_id} prompt_len={seq.num_prompt_tokens} "
                    f"remaining_prefill_tokens={need} available_step_tokens={free} "
                    f"engine_prefill_chunk_size={self.engine_prefill_chunk_size} "
                    f"max_num_batched_tokens={self.max_num_batched_tokens}. "
                    "Increase the raw KV budget / max_num_batched_tokens or reduce short-batch size."
                )
            if blocked_final_prefill_window_failure is not None and not self.decoding:
                seq, min_final, available = blocked_final_prefill_window_failure
                raise RuntimeError(
                    "Final prefill score window cannot fit in the effective step budget. "
                    f"cache_manager={type(self.memory_oracle).__name__} "
                    f"seq_id={seq.seq_id} prompt_len={seq.num_prompt_tokens} "
                    f"remaining_prefill_tokens={seq.num_prompt_tokens - seq.num_prefilled_tokens} "
                    f"required_final_window={min_final} available_step_tokens={available} "
                    f"engine_prefill_chunk_size={self.engine_prefill_chunk_size} "
                    f"max_num_batched_tokens={self.max_num_batched_tokens}."
                )
            if blocked_prefill_capacity_failure is not None and not self.decoding:
                seq, need, seq_free, global_free = blocked_prefill_capacity_failure
                raise RuntimeError(
                    "No prefill candidate can use the remaining cache capacity. "
                    f"cache_manager={type(self.memory_oracle).__name__} seq_id={seq.seq_id} "
                    f"prompt_len={seq.num_prompt_tokens} remaining_prefill_tokens={need} "
                    f"candidate_step_free={seq_free} global_step_free={global_free} "
                    f"free_slots={physical_free_count} reserved_prefill={reserved_prefill} "
                    f"waiting={len(self.waiting)} decoding={len(self.decoding)}. "
                    "This usually means the only remaining capacity belongs to another sequence's partial page; "
                    "reduce concurrency or free a decode sequence first."
                )
            if deferred_prompt_failure is not None and not self.decoding:
                seq, name, need, free = deferred_prompt_failure
                if os.getenv("SPARSEENGINE_DEBUG_SLOTS", "0") == "1":
                    waiting_seq_ids_all = [int(s.seq_id) for s in self.waiting]
                    decoding_seq_ids_all = [int(s.seq_id) for s in self.decoding]
                    scheduled_seq_ids_all = [int(s.seq_id) for s in scheduled_seqs]
                    live_seq_slots = self.memory_oracle.debug_live_seq_slots()
                    live_seq_items = sorted(
                        ((int(seq_id), int(n_slots)) for seq_id, n_slots in live_seq_slots.items()),
                        key=lambda x: (-x[1], x[0]),
                    )[:16]
                    waiting_prompt_lens = [int(s.num_prompt_tokens) for s in list(self.waiting)[:8]]
                    known_seq_ids = (
                        set(waiting_seq_ids_all)
                        | set(decoding_seq_ids_all)
                        | set(scheduled_seq_ids_all)
                    )
                    zombie_seq_ids = sorted(
                        int(seq_id)
                        for seq_id in live_seq_slots
                        if int(seq_id) not in known_seq_ids
                    )[:16]
                    logger.info(
                        "deferred_deadlock seq_id={} failed_budget={} need={} free={} free_slots={} reserved_prefill={} "
                        "waiting_prompt_lens={} waiting_seq_ids={} decoding_seq_ids={} scheduled_seq_ids={} "
                        "zombie_seq_ids={} live_seq_slots={}",
                        seq.seq_id,
                        name,
                        int(need),
                        int(free),
                        int(physical_free_count),
                        int(reserved_prefill),
                        waiting_prompt_lens,
                        waiting_seq_ids_all[:16],
                        decoding_seq_ids_all[:16],
                        scheduled_seq_ids_all[:16],
                        zombie_seq_ids,
                        live_seq_items,
                    )
                raise RuntimeError(
                    "All prompt admissions were deferred and no runnable work remains. "
                    f"cache_manager={type(self.memory_oracle).__name__} seq_id={seq.seq_id} "
                    f"prompt_len={seq.num_prompt_tokens} failed_budget={name} need={need} free={free} "
                    f"free_slots={physical_free_count} reserved_prefill={reserved_prefill} "
                    f"waiting={len(self.waiting)} decoding={len(self.decoding)}. "
                    "Reduce batch size/max_num_seqs_in_batch/max_num_batched_tokens, "
                    "or shorten the prompt / generation budget."
                )
            return [], False, preempted_seqs
            
        # 将被选中的 Decode 序列放回 running 队列以保持顺序
        self.decoding.extendleft(reversed(scheduled_seqs))
        if blocked_decode_victim is not None:
            # Retry the sequence that could not join this partial batch first.
            # If no slot was freed by the completed step, it is the appropriate
            # preemption victim; evicting a sequence that just made progress
            # would create avoidable completion replay work.
            try:
                self.decoding.remove(blocked_decode_victim)
            except ValueError:
                pass
            self.decoding.appendleft(blocked_decode_victim)
        return scheduled_seqs, False, preempted_seqs

    @cpu_timing.timed
    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
        token_logprobs: list[float | None] | None = None,
        top_logprobs: list[dict[int, float] | None] | None = None,
        *,
        retain_finished: bool = False,
    ):
        """
        模型运行后的后处理工作。
        1. 更新 Token 序列。
        2. 更新 Prefill 进度。
        3. 处理序列完成状态 (EOS 或 Max Tokens)。
        """
        token_logprobs = token_logprobs or [None] * len(seqs)
        top_logprobs = top_logprobs or [None] * len(seqs)
        if is_prefill:
            for seq, token_id, token_logprob, top_logprob in zip(seqs, token_ids, token_logprobs, top_logprobs):
                seq.num_prefilled_tokens += seq.current_chunk_size
                # 检查 Chunked Prefill 是否完成
                if seq.num_prefilled_tokens < seq.num_prompt_tokens:
                    # 没跑完，塞回等待队列头部下次继续
                    seq.status = SequenceStatus.WAITING
                    self._prefill_wait_since.setdefault(seq.seq_id, time.monotonic())
                    self.waiting.appendleft(seq)
                else:
                    self.memory_oracle.complete_prefill_execution(seq)
                    # Prefill 彻底结束，进入正常生成流程
                    seq.status = SequenceStatus.RUNNING
                    self.decoding.append(seq)
                    if seq.is_recompute_replay:
                        # Prefill rebuilt the original prompt KV. Its sampled
                        # token is discarded because the accepted completion
                        # history is authoritative.
                        if seq.replay_decode_token_count == 0:
                            seq.finish_recompute_replay()
                            logger.info(
                                "recompute_replay_complete seq_id={} replay_decode_tokens=0",
                                seq.seq_id,
                            )
                        continue
                    # 记录模型生成的第一个 Token
                    seq.append_token(token_id, token_logprob, top_logprob)
                    # 检查是否命中结束条件
                    request_eos = resolve_eos_token_ids(
                        seq.eos_token_ids,
                        self.eos_token_ids,
                    )
                    if (not seq.ignore_eos and token_id in request_eos) or seq.num_completion_tokens == seq.max_tokens:
                        seq.status = SequenceStatus.FINISHED
                        if not retain_finished:
                            self.decoding.remove(seq)
            return

        # 处理 Decode 步骤
        for seq, token_id, token_logprob, top_logprob in zip(seqs, token_ids, token_logprobs, top_logprobs):
            if seq.is_recompute_decode:
                # This forward rebuilt KV for an already accepted completion
                # token. Ignore the sampled output and advance through history.
                replay_token_count = seq.replay_decode_token_count
                seq.advance_recompute_replay()
                if not seq.is_recompute_replay:
                    logger.info(
                        "recompute_replay_complete seq_id={} replay_decode_tokens={}",
                        seq.seq_id,
                        replay_token_count,
                    )
                continue
            seq.append_token(token_id, token_logprob, top_logprob)
            request_eos = resolve_eos_token_ids(
                seq.eos_token_ids,
                self.eos_token_ids,
            )
            if (not seq.ignore_eos and token_id in request_eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                if not retain_finished and seq in self.decoding:
                    self.decoding.remove(seq)
