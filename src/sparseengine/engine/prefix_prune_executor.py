"""Forward scoring and batch execution for online prefix pruning."""

from collections import deque

import torch
import torch.distributed as dist

from sparseengine.engine.prefix_prune import (
    normalize_prefix_prune_ranges,
    validate_prefix_prune_request,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.utils.context import get_context, reset_context


def _prefix_prune_score_forward(
    runner,
    *,
    token_ids: list[int],
    prefix_hit_len: int,
    protected_prefix_len: int,
    candidate_start: int,
    temp_seq_id: int,
    protected_token_ids: list[int] | None = None,
) -> torch.Tensor:
    return runner._prefix_prune_score_forward_batch([dict(
        token_ids=token_ids, prefix_hit_len=prefix_hit_len,
        protected_prefix_len=protected_prefix_len, candidate_start=candidate_start,
        temp_seq_id=temp_seq_id, protected_token_ids=protected_token_ids,
    )])[0]


def _prefix_prune_score_forward_batch(runner, requests) -> list[torch.Tensor]:
    manager = runner.cache_manager
    prefix_cache = manager.prefix_cache
    block_size = int(manager.prefix_cache_block_size)
    if prefix_cache is None or block_size <= 0:
        raise RuntimeError("prefix cache is not enabled on this model worker.")
    asynchronous = getattr(runner, "_async_execution", None)
    if asynchronous is not None:
        asynchronous.prepare_synchronous_execution()
    seqs, scoring, protected = [], [], {}
    for request in requests:
        token_ids = request["token_ids"]
        prepared = request.get("prefix")
        prefix_hit_len = request["prefix_hit_len"]
        protected_prefix_len = request["protected_prefix_len"]
        candidate_start = request["candidate_start"]
        temp_seq_id = request["temp_seq_id"]
        query_end = len(token_ids) + (prefix_hit_len if prepared is not None else 0)
        if query_end > int(runner.config.max_model_len):
            raise ValueError(
                "prefix-prune scoring context exceeds max_model_len: "
                f"context={query_end} max_model_len={runner.config.max_model_len}. "
                "Reduce observation_tokens, score_chunk_size, or prev_postfix_size."
            )
        if prefix_hit_len <= candidate_start or prefix_hit_len >= query_end:
            raise ValueError(
                "prefix-prune score forward requires candidate tokens followed by queries: "
                f"candidate_start={candidate_start} hit={prefix_hit_len} end={query_end}."
            )
        if prepared is not None:
            hit_len = prefix_hit_len
            hit_blocks = len(prepared.blocks)
            last_block_id = prepared.blocks[-1].stable_block_id
            protected_blocks = ()  # Already pinned by the maintenance iterator.
        else:
            block_ids = prefix_cache.block_ids_for_tokens(
                token_ids[:prefix_hit_len], max_tokens=prefix_hit_len
            )
            hit_len, last_block_id, hit_blocks = prefix_cache.match_longest_block_ids(
                block_ids
            )
            if hit_len != prefix_hit_len or last_block_id is None:
                raise RuntimeError(
                    "prefix-prune score forward cannot attach the requested cached prefix: "
                    f"requested={prefix_hit_len} matched={hit_len}."
                )
            protection_tokens = request.get("protected_token_ids")
            if protection_tokens is not None:
                protected_block_ids = prefix_cache.block_ids_for_tokens(
                    protection_tokens, max_tokens=len(protection_tokens),
                )
                protected_hit, protected_last, protected_count = (
                    prefix_cache.match_longest_block_ids(protected_block_ids)
                )
            elif protected_prefix_len == prefix_hit_len:
                protected_hit, protected_last, protected_count = hit_len, last_block_id, hit_blocks
            else:
                protected_block_ids = prefix_cache.block_ids_for_tokens(
                    token_ids[:protected_prefix_len], max_tokens=protected_prefix_len
                )
                protected_hit, protected_last, protected_count = (
                    prefix_cache.match_longest_block_ids(protected_block_ids)
                )
            if protected_hit < protected_prefix_len or protected_last is None:
                raise RuntimeError(
                    "prefix-prune target changed before its scoring forward: "
                    f"protected={protected_prefix_len} matched={protected_hit}."
                )
            protected_blocks = prefix_cache.get_chain(
                protected_last, protected_count
            )
        seq = Sequence([int(token_id) for token_id in token_ids])
        seq.seq_id = int(temp_seq_id)
        if prepared is not None:
            seq.num_tokens = seq.num_prompt_tokens = query_end
            seq.prefix_prune_prefix = prepared
        seq.num_prefilled_tokens = int(prefix_hit_len)
        seq.current_chunk_size = query_end - prefix_hit_len
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = int(prefix_hit_len)
        seq.prefix_cache_hit_block_count = int(hit_blocks)
        seq.prefix_cache_hit_last_block_id = last_block_id
        seq.prefix_cache_block_size = block_size
        seq.prefix_cache_method = str(getattr(runner.config, "sparse_method", "") or "")
        seqs.append(seq)
        scoring.append(dict(seq_id=seq.seq_id, candidate_start=candidate_start,
                            query_start=prefix_hit_len, query_end=query_end))
        for block in protected_blocks:
            protected[id(block)] = block
    acquired = []
    try:
        for block in protected.values():
            prefix_cache.acquire_block_ref(block)
            acquired.append(block)
        manager.begin_prefix_prune_scoring_batch(scoring)
        input_ids, positions = runner.prepare_step(seqs, True)
        for seq, request, state in zip(seqs, requests, manager._prefix_prune_scoring["batch"], strict=True):
            prefix = request.get("prefix")
            if prefix is None:
                continue
            if prefix.token_slots is None:
                manager.attach_prefix_prune_prefix(seq, prefix)
            start = int(prefix.token_slots.numel())
            length = int(seq.current_chunk_size)
            state["physical_window"] = (start, start + length, prefix.candidate_start)
            if prefix.logical_positions is not None:
                state["logical_positions"] = torch.cat((
                    prefix.logical_positions,
                    torch.arange(prefix.logical_length, prefix.logical_length + length,
                                 dtype=torch.long, device=runner.device),
                ))
        ctx = get_context()
        ctx.sparse_controller = runner.sparse_controller
        runner.sparse_controller.prepare_forward(seqs, True)
        runner.model(input_ids, positions)
        runner.sparse_controller.post_forward(seqs, True)
        return manager.finish_prefix_prune_scoring_batch()
    finally:
        manager.abort_prefix_prune_scoring()
        reset_context()
        for seq in seqs:
            if seq.seq_id in manager.seq_id_to_row:
                runner.runtime_state.free_seq(seq.seq_id)
        for block in acquired:
            prefix_cache.release_block_ref(block)


def prefix_cache_prune(
    runner,
    token_ids: list[int],
    range_start: int | None = None,
    range_end: int | None = None,
    keep_tokens: int | None = None,
    policy: str | None = None,
    prune_id: str = "",
    allow_recompress: bool = False,
    observation_tokens: int = 64,
    score_chunk_size: int = 2048,
    prev_postfix_size: int = 64,
    kvzip_replay_prefix_ids: list[int] | None = None,
    temp_seq_id: int = -1,
    ranges: list[tuple[int, int]] | None = None,
    *,
    prune_batch,
) -> dict[str, object]:
    token_ids = [int(token_id) for token_id in token_ids]
    block_size = int(runner.config.prefix_cache_block_size)
    intervals = normalize_prefix_prune_ranges(
        token_count=len(token_ids), block_size=block_size,
        range_start=range_start, range_end=range_end, ranges=ranges,
    )
    validate_prefix_prune_request(
        token_count=len(token_ids), block_size=block_size, ranges=intervals,
        keep_tokens=keep_tokens, policy=policy,
    )
    runner.cache_manager.validate_prefix_prune_keep_tokens(keep_tokens)
    runner.cache_manager.validate_prefix_cache_prune_target(
        token_ids, ranges=intervals, allow_recompress=allow_recompress,
    )
    range_start, range_end = intervals[0][0], intervals[-1][1]
    width = sum(right - left for left, right in intervals)
    if policy == "snapkv_global":
        observation_tokens = max(1, int(observation_tokens))
        query_start = max(range_start + block_size, range_end - observation_tokens)
        query_start = (query_start // block_size) * block_size
        protected_count = sum(
            right - max(left, query_start)
            for left, right in intervals if right > query_start
        )
        if protected_count > keep_tokens or query_start >= range_end:
            raise ValueError(
                "SnapKV observation window exceeds the global keep budget: "
                f"observation={protected_count} keep_tokens={keep_tokens}."
            )
        score = runner._prefix_prune_score_forward(
            token_ids=token_ids[:range_end], prefix_hit_len=query_start,
            protected_prefix_len=range_end, candidate_start=range_start,
            temp_seq_id=temp_seq_id, protected_token_ids=token_ids,
        )
        # Pack only requested tokens before the collective and selection.
        packed = torch.cat([score[left:right] for left, right in intervals])
        runner.parallel_context.world.all_reduce(packed, op=dist.ReduceOp.MAX)
        # Sorted disjoint intervals put all protected positions at the end.
        keep_indices = runner.cache_manager.select_prefix_prune_keep_indices(
            packed,
            keep_tokens=keep_tokens,
            protected_suffix_tokens=protected_count,
        )
    elif policy == "kvzip_global":
        if keep_tokens == 0:
            # The mask is known without reconstruction when everything is dropped.
            keep_indices = torch.empty(0, dtype=torch.long, device=runner.device)
        else:
            outcomes = prune_batch(runner, [dict(
                token_ids=token_ids, ranges=intervals, keep_tokens=keep_tokens,
                allow_recompress=allow_recompress, score_chunk_size=max(1, int(score_chunk_size)),
                prev_postfix_size=max(0, int(prev_postfix_size)), prune_id=prune_id,
            )], list(kvzip_replay_prefix_ids or []))
            if "error" in outcomes[0]:
                raise RuntimeError(outcomes[0]["error"])
            return outcomes[0]["result"]
    else:
        raise ValueError(f"unsupported prefix prune policy: {policy!r}.")

    return runner.cache_manager.prefix_cache_prune(
        token_ids, ranges=intervals, keep_indices=keep_indices,
        policy=policy, prune_id=prune_id, allow_recompress=allow_recompress,
    )


def prefix_cache_prune_batch(runner, jobs, replay_prefix):
    iterator = _prefix_cache_prune_batch_iter(runner, jobs, replay_prefix)
    try:
        while True:
            next(iterator)
    except StopIteration as done:
        return done.value
    finally:
        iterator.close()


def prefix_cache_prune_batch_start(runner, jobs, replay_prefix):
    if getattr(runner, "_prefix_prune_iterator", None) is not None:
        raise RuntimeError("Another prefix-prune batch is already running.")
    runner._prefix_prune_iterator = runner._prefix_cache_prune_batch_iter(jobs, replay_prefix)


def prefix_cache_prune_batch_step(runner):
    iterator = getattr(runner, "_prefix_prune_iterator", None)
    if iterator is None:
        raise RuntimeError("No prefix-prune batch is running.")
    try:
        next(iterator)
    except StopIteration as done:
        runner._prefix_prune_iterator = None
        return done.value
    except BaseException:
        runner.prefix_cache_prune_batch_abort()
        raise
    return None


def prefix_cache_prune_batch_abort(runner):
    iterator = getattr(runner, "_prefix_prune_iterator", None)
    runner._prefix_prune_iterator = None
    if iterator is not None:
        iterator.close()


def _prefix_cache_prune_batch_iter(runner, jobs, replay_prefix):
    """Yield only after a complete forward has released its temporary rows."""
    from sparseengine.engine.cache_manager.prefix_prune_scoring import PrefixPrunePrefix

    manager = runner.cache_manager
    cache = manager.prefix_cache
    block_size = int(runner.config.prefix_cache_block_size)
    states, protected, affected_ids = [], {}, set()
    outcomes = [None] * len(jobs)
    for index, job in enumerate(jobs):
        try:
            intervals = normalize_prefix_prune_ranges(
                token_count=len(job["token_ids"]), block_size=block_size,
                ranges=job["ranges"],
            )
            validate_prefix_prune_request(
                token_count=len(job["token_ids"]), block_size=block_size,
                ranges=intervals, keep_tokens=job["keep_tokens"], policy="kvzip_global",
            )
            manager.validate_prefix_prune_keep_tokens(job["keep_tokens"])
            affected = manager.validate_prefix_cache_prune_target(
                job["token_ids"], ranges=intervals,
                allow_recompress=job["allow_recompress"],
            )
            ids = {id(block) for block in affected}
            if affected_ids.intersection(ids):
                raise RuntimeError("Overlapping prune targets cannot share a scoring batch.")
            end = intervals[-1][1]
            # Preserve the complete cached route, including non-candidate
            # tails which the caller will reuse after pruning.
            block_ids = cache.block_ids_for_tokens(
                job["token_ids"], max_tokens=len(job["token_ids"]),
            )
            _, last, count = cache.match_longest_block_ids(block_ids)
            chain = cache.get_chain(last, count)
            chunks = deque()
            if job["keep_tokens"]:
                if not replay_prefix:
                    raise ValueError("KVzip requires a non-empty reconstruction prompt.")
                for left, right in intervals:
                    for start in range(left, right, job["score_chunk_size"]):
                        stop = min(right, start + job["score_chunk_size"])
                        prev = max(left, start - job["prev_postfix_size"])
                        chunks.append((prev, stop))
                max_query = max(len(replay_prefix) + stop - prev for prev, stop in chunks)
                if end + max_query > runner.config.max_model_len:
                    raise ValueError("Prefix-prune scoring context exceeds max_model_len.")
            prefix_blocks = chain[:end // block_size]
            state = dict(index=index, job=job, intervals=intervals,
                         prefix=PrefixPrunePrefix(prefix_blocks, end, intervals[0][0]),
                         prefix_blocks=prefix_blocks,
                         chunks=chunks, forwards=0, max_batch=0,
                         aggregate=torch.zeros(sum(r-l for l,r in intervals),
                                               dtype=torch.float32, device=runner.device))
            states.append(state)
            affected_ids.update(ids)
            protected.update((id(block), block) for block in chain)
        except (ValueError, RuntimeError) as exc:
            outcomes[index] = {"error": f"{type(exc).__name__}: {exc}"}
    acquired = []
    try:
        # Protect every pending job, including those not in the next forward.
        for block in protected.values():
            cache.acquire_block_ref(block)
            acquired.append(block)
        pending = deque(state for state in states if state["chunks"])
        next_seq_id = -1_000_000_000
        token_limit = min(runner.config.max_num_batched_tokens,
                          runner.config.engine_prefill_chunk_size or runner.config.max_num_batched_tokens)
        while pending:
            requests, owners = [], []
            query_tokens = 0
            while pending and len(requests) < runner.config.max_num_seqs_in_batch:
                owner = pending[0]
                prev, stop = owner["chunks"][0]
                replay = replay_prefix + owner["job"]["token_ids"][prev:stop]
                if requests and query_tokens + len(replay) > token_limit:
                    break
                pending.popleft()
                owner["chunks"].popleft()
                end = owner["prefix"].logical_length
                requests.append(dict(token_ids=replay, prefix=owner["prefix"],
                                     prefix_hit_len=end, protected_prefix_len=end,
                                     candidate_start=owner["intervals"][0][0],
                                     temp_seq_id=next_seq_id))
                next_seq_id -= 1
                owners.append((owner, (prev, stop)))
                query_tokens += len(replay)
                if owner["chunks"]:
                    pending.append(owner)
            reservation = [(r["temp_seq_id"], len(r["token_ids"]))
                           for r in requests]
            prefix_blocks = {r["temp_seq_id"]: (owner["prefix_blocks"]
                                              if owner["prefix"].token_slots is None else ())
                             for r, (owner, _) in zip(requests, owners, strict=True)}
            with manager.reserve_prefill_slots(reservation, prefix_blocks=prefix_blocks) as admitted:
                # Return unadmitted chunks in their original per-task order.
                for owner, chunk in reversed(owners[admitted:]):
                    owner["chunks"].appendleft(chunk)
                    if not any(item is owner for item in pending):
                        pending.appendleft(owner)
                requests = requests[:admitted]
                owners = [owner for owner, _ in owners[:admitted]]
                scores = runner._prefix_prune_score_forward_batch(requests)
            for owner, score in zip(owners, scores, strict=True):
                packed = torch.cat([score[l:r] for l, r in owner["intervals"]])
                torch.maximum(owner["aggregate"], packed, out=owner["aggregate"])
                owner["forwards"] += 1
                owner["max_batch"] = max(owner["max_batch"], len(requests))
            del requests, scores, owners, score, packed
            yield
    finally:
        for block in acquired:
            cache.release_block_ref(block)
    for state in states:
        job = state["job"]
        aggregate = state["aggregate"]
        if job["keep_tokens"]:
            runner.parallel_context.world.all_reduce(aggregate, op=dist.ReduceOp.MAX)
        indices = manager.select_prefix_prune_keep_indices(
            aggregate, keep_tokens=job["keep_tokens"],
        )
        try:
            result = manager.prefix_cache_prune(
                job["token_ids"], ranges=state["intervals"], keep_indices=indices,
                policy="kvzip_global", prune_id=job["prune_id"],
                allow_recompress=job["allow_recompress"],
            )
            result.update(scoring_chunks=state["forwards"],
                          max_scoring_batch=state["max_batch"], scoring_jobs=len(states))
            outcomes[state["index"]] = {"result": result}
        except (ValueError, RuntimeError) as exc:
            outcomes[state["index"]] = {"error": f"{type(exc).__name__}: {exc}"}
    return outcomes
