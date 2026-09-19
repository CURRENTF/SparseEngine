"""Single-node DP attention frontend with one scheduler/cache owner per TP replica."""

from __future__ import annotations

import atexit
import os
import signal
import threading
import time
import traceback
from dataclasses import replace
from multiprocessing.connection import wait

import torch.multiprocessing as mp
from transformers import AutoTokenizer

from sparseengine.config import Config
from sparseengine.engine.chain_cache import ChainOwnerMismatchError, ChainRoutingSnapshot
from sparseengine.engine.model_runner import select_master_port
from sparseengine.utils.log import logger


class DPAttentionPrefixSnapshot:
    def __init__(self, replicas):
        self.replicas = replicas

    def match(self, token_ids):
        return max(
            (replica.match(token_ids) for replica in self.replicas),
            key=lambda result: int(result.get("matched_tokens", 0)),
        )


def _worker_main(connection, model, kwargs, rank, port):
    from sparseengine.engine.llm_engine import LLMEngine

    # A replica owns TP children; fatal teardown must also stop those workers.
    os.setsid()
    engine = None
    try:
        engine = LLMEngine(model, _dp_worker=(rank, port), **kwargs)
        previous_prefix = previous_chain = None

        def snapshots():
            nonlocal previous_prefix, previous_chain
            prefix = engine.prefix_cache_routing_snapshot()
            chain = engine.chain_cache_routing_snapshot()
            update = (
                engine.worker_routing_load(),
                prefix if prefix is not previous_prefix else None,
                chain if chain != previous_chain else None,
            )
            previous_prefix, previous_chain = prefix, chain
            return update

        connection.send((True, engine.config, snapshots()))
        while True:
            method, args, call_kwargs = connection.recv()
            if method == "exit":
                engine.exit()
                connection.send((True, None, None))
                return
            try:
                result = getattr(engine, method)(*args, **call_kwargs)
                if method == "step":
                    result = (
                        result,
                        engine.last_step_token_outputs,
                        engine.last_step_prompt_cache_hits,
                        engine.last_step_logprob_outputs,
                        engine.is_finished(),
                    )
                connection.send((True, result, snapshots()))
            except Exception as exc:  # noqa: BLE001 - serialize control failures to the parent.
                connection.send((False, (exc, traceback.format_exc()), None))
    except BaseException as exc:  # noqa: BLE001 - propagate worker startup/fatal failures.
        connection.send((False, (exc, traceback.format_exc()), None))
    finally:
        connection.close()


class DPAttentionEngine:
    """Route requests; keep admission, scheduling and cache accounting local.

    Every step is submitted to all workers, including idle workers. Control
    operations address only their owner and do not issue expert collectives.
    """

    def __init__(self, model, **kwargs):
        self.config = Config(model, **kwargs)
        topology = self.config.parallel_topology
        if topology.attn_dp_size > 1 and (
            topology.attn_tp_size > 1
            or topology.moe_ep_size > 1
            or topology.moe_tp_size > 1
        ):
            logger.warning(
                "DP combined with TP/EP has known performance issues in the "
                "current implementation (attention DP={}, TP={}; MoE EP={}, TP={}). "
                "Throughput and latency may be degraded.",
                topology.attn_dp_size,
                topology.attn_tp_size,
                topology.moe_ep_size,
                topology.moe_tp_size,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.multimodal_processor = None
        self._connections = []
        self.ps = []
        self._closed = False
        self._rpc_lock = threading.RLock()
        self._owners = {}
        self._global_ids = {}
        self._chain_owners = {}
        self._chain_seq_ids = set()
        self._chain_sequences = {}
        self._next_id = 0
        self._active = [set() for _ in range(self.config.data_parallel_size)]
        self._routing_loads = {}
        self._prefix_snapshots = {}
        self._chain_snapshots = {}
        self.last_step_token_outputs = []
        self.last_step_prompt_cache_hits = []
        self.last_step_logprob_outputs = []
        ctx = mp.get_context("spawn")
        port = select_master_port()
        try:
            for rank in range(self.config.data_parallel_size):
                parent, child = ctx.Pipe()
                process = ctx.Process(
                    target=_worker_main, args=(child, model, kwargs, rank, port)
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self.ps.append(process)
            configs = self._receive(range(len(self.ps)), fatal=True)
            self.config = configs[0]
        except BaseException:
            self._terminate()
            raise
        atexit.register(self.exit)

    def _receive(self, ranks, *, fatal=False):
        pending = {self._connections[rank]: rank for rank in ranks}
        results = {}
        errors = []
        deadline = time.monotonic() + float(
            os.getenv("SPARSEENGINE_DP_RPC_TIMEOUT_S", "1800")
        )
        while pending:
            ready = wait(list(pending), timeout=1)
            if not ready and (
                time.monotonic() > deadline or any(not p.is_alive() for p in self.ps)
            ):
                self._terminate()
                raise RuntimeError(
                    "DP worker exited or timed out while completing a collective step."
                )
            for connection in ready:
                rank = pending.pop(connection)
                try:
                    ok, result, snapshots = connection.recv()
                except EOFError as exc:
                    self._terminate()
                    raise RuntimeError(
                        f"DP worker {rank} exited unexpectedly."
                    ) from exc
                if not ok:
                    error, trace = result
                    # A failed forward may leave peers inside NCCL. Tear down the
                    # entire worker group rather than reusing inconsistent state.
                    if fatal:
                        self._terminate()
                    logger.error("DP rank {} failed:\n{}", rank, trace)
                    if fatal:
                        raise error
                    errors.append(error)
                    continue
                if snapshots is not None:
                    load, prefix, chain = snapshots
                    self._routing_loads[rank] = load
                    if prefix is not None:
                        self._prefix_snapshots[rank] = prefix
                    if chain is not None:
                        self._chain_snapshots[rank] = chain
                        known = (
                            chain.active_chain_ids
                            | chain.idle_chain_ids
                            | chain.tombstone_chain_ids
                        )
                        expired = [
                            key
                            for key, owner in self._chain_owners.items()
                            if owner == rank and key not in known
                        ]
                        for key in expired:
                            self._chain_owners.pop(key)
                            seq_id = self._chain_sequences.pop(key, None)
                            if seq_id is not None:
                                self._chain_seq_ids.discard(seq_id)
                                self._release_request_identity(seq_id)
                results[rank] = result
        if errors:
            raise errors[0]
        return {rank: results[rank] for rank in sorted(results)}

    def _call(self, ranks, method, *args, **kwargs):
        with self._rpc_lock:
            if self._closed:
                raise RuntimeError("DP attention engine is closed.")
            ranks = tuple(ranks)
            for rank in ranks:
                self._connections[rank].send((method, args, kwargs))
            return self._receive(ranks, fatal=method == "step")

    def admit_request(
        self, prompt, sampling_params, chain_id=None, chain_append_only=False
    ):
        rank = self._chain_owners.get(str(chain_id or "").strip())
        if rank is None:
            loads = [len(active) for active in self._active]
            candidates = [rank for rank, load in enumerate(loads) if load == min(loads)]
            if self.config.resolved_prefix_cache_mode == "radix":
                token_ids = (
                    self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
                )
                matches = self._call(candidates, "prefix_cache_match", token_ids)
                rank = max(
                    candidates,
                    key=lambda item: int(matches[item].get("matched_tokens", 0)),
                )
            else:
                rank = candidates[0]
        admission = self._call(
            (rank,),
            "admit_request",
            prompt,
            sampling_params,
            chain_id,
            chain_append_only,
        )[rank]
        local_id = admission.seq_id
        identity = (rank, local_id)
        seq_id = self._global_ids.get(identity)
        if seq_id is None:
            seq_id = self._next_id
            self._next_id += 1
            self._global_ids[identity] = seq_id
            self._owners[seq_id] = identity
        self._active[rank].add(seq_id)
        if admission.chain_id:
            self._chain_owners[admission.chain_id] = rank
            self._chain_seq_ids.add(seq_id)
            self._chain_sequences[admission.chain_id] = seq_id
        return replace(admission, seq_id=seq_id)

    def add_request(self, prompt, sampling_params):
        return self.admit_request(prompt, sampling_params).seq_id

    def abort_request(self, seq_id, disposition="invalidate"):
        if seq_id not in self._owners:
            return None
        rank, local_id = self._owners[seq_id]
        result = self._call((rank,), "abort_request", local_id, disposition)[rank]
        self._active[rank].discard(seq_id)
        if disposition == "invalidate":
            self._chain_seq_ids.discard(seq_id)
        self._release_request_identity(seq_id)
        return result

    def step(self):
        records = self._call(range(len(self.ps)), "step")
        finished = []
        self.last_step_prefill_tokens = self.last_step_decode_tokens = 0
        self.last_step_token_outputs = []
        self.last_step_prompt_cache_hits = []
        self.last_step_logprob_outputs = []
        for rank, (result, tokens, hits, logprobs, idle) in records.items():

            def translate(rows, rank=rank):
                return [(self._global_ids[(rank, row[0])], *row[1:]) for row in rows]

            outputs, count = result
            outputs = translate(outputs)
            finished.extend(outputs)
            self.last_step_prefill_tokens += max(0, count)
            self.last_step_decode_tokens += max(0, -count)
            self.last_step_token_outputs.extend(translate(tokens))
            self.last_step_prompt_cache_hits.extend(translate(hits))
            self.last_step_logprob_outputs.extend(translate(logprobs))
            for row in outputs:
                self._active[rank].discard(row[0])
                self._release_request_identity(row[0])
        return finished, self.last_step_prefill_tokens or -self.last_step_decode_tokens

    def _release_request_identity(self, seq_id):
        identity = self._owners.get(seq_id)
        if identity is not None and seq_id not in self._chain_seq_ids:
            self._owners.pop(seq_id)
            self._global_ids.pop(identity)

    def is_finished(self):
        return not any(self._active)

    def generate(self, prompts, sampling_params, use_tqdm=True):
        from sparseengine.engine.llm_engine import LLMEngine

        return LLMEngine.generate(self, prompts, sampling_params, use_tqdm)

    def chain_cache_routing_match(self, chain_id):
        rank = self._chain_owners.get(chain_id, 0)
        return self._call((rank,), "chain_cache_routing_match", chain_id)[rank]

    def invalidate_chain(self, chain_id):
        rank = self._chain_owners.get(chain_id, 0)
        seq_id = self._chain_sequences.get(chain_id)
        if seq_id is not None and seq_id in self._owners:
            return self.discard_chain(chain_id, expected_seq_id=seq_id)
        return self._call((rank,), "invalidate_chain", chain_id)[rank]

    def prefix_cache_match(self, token_ids):
        matches = self._call(range(len(self.ps)), "prefix_cache_match", token_ids)
        return max(
            matches.values(), key=lambda match: int(match.get("matched_tokens", 0))
        )

    def prefix_cache_routing_snapshot(self):
        return DPAttentionPrefixSnapshot(list(self._prefix_snapshots.values()))

    def prefix_cache_inspect(self, token_ids, include_subtree=False):
        return {
            "replicas": self._call(
                range(len(self.ps)), "prefix_cache_inspect", token_ids, include_subtree
            )
        }

    def prefix_cache_delete_subtree(self, token_ids):
        return {
            "replicas": self._call(
                range(len(self.ps)), "prefix_cache_delete_subtree", token_ids
            )
        }

    def prefix_cache_set_eviction_priority(self, token_ids, priority):
        return {
            "replicas": self._call(
                range(len(self.ps)),
                "prefix_cache_set_eviction_priority",
                token_ids,
                priority,
            )
        }

    def prefix_cache_prune_start(self, *args, **kwargs):
        raise NotImplementedError(
            "Score-based prefix pruning is not supported with DP attention."
        )

    def chain_cache_routing_snapshot(self):
        snapshots = list(self._chain_snapshots.values())
        return ChainRoutingSnapshot(
            enabled=all(snapshot.enabled for snapshot in snapshots),
            active_chain_ids=frozenset().union(
                *(snapshot.active_chain_ids for snapshot in snapshots)
            ),
            idle_chain_ids=frozenset().union(
                *(snapshot.idle_chain_ids for snapshot in snapshots)
            ),
            tombstone_chain_ids=frozenset().union(
                *(snapshot.tombstone_chain_ids for snapshot in snapshots)
            ),
        )

    def discard_chain(self, chain_id, *, expected_seq_id):
        owner = self._owners.get(expected_seq_id)
        if owner is None:
            if self._chain_sequences.get(chain_id, expected_seq_id) != expected_seq_id:
                raise ChainOwnerMismatchError(
                    "Chain sequence identity does not match.", chain_id=chain_id
                )
            return False
        rank, local_id = owner
        if rank != self._chain_owners.get(chain_id):
            raise ChainOwnerMismatchError(
                "Chain belongs to a different DP replica.", chain_id=chain_id
            )
        result = self._call(
            (rank,), "discard_chain", chain_id, expected_seq_id=local_id
        )[rank]
        if result:
            self._active[rank].discard(expected_seq_id)
            self._chain_seq_ids.discard(expected_seq_id)
            self._release_request_identity(expected_seq_id)
        return result

    def worker_info(self, served_model_name=None, tags=None):
        infos = self._call(range(len(self.ps)), "worker_info", served_model_name, tags)
        result = dict(infos[0])
        result["replicas"] = list(infos.values())
        return result

    def worker_routing_load(self):
        loads = list(self._routing_loads.values())
        return {key: sum(load[key] for load in loads) for key in loads[0]}

    def worker_load(self):
        loads = list(self._call(range(len(self.ps)), "worker_load").values())
        result = {
            key: sum(load[key] for load in loads) for key in loads[0] if key != "cache"
        }
        result["cache"] = {
            key: sum(load["cache"][key] for load in loads) for key in loads[0]["cache"]
        }
        result["replicas"] = loads
        return result

    def debug_sparse_state_summaries(self, synchronize=False):
        summaries = self._call(
            range(len(self.ps)), "debug_sparse_state_summaries", synchronize
        )
        return [summary for local in summaries.values() for summary in local]

    def operator_runtime_stats(self):
        reports = self._call(range(len(self.ps)), "operator_runtime_stats")
        return [report for local in reports.values() for report in local]

    def debug_last_logits(self):
        return self._call(range(len(self.ps)), "debug_last_logits")

    def debug_set_next_decode_token(self, seq_id, token_id):
        rank, local_id = self._owners[seq_id]
        self._call((rank,), "debug_set_next_decode_token", local_id, token_id)

    def _terminate(self):
        self._closed = True
        for process in self.ps:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if process.is_alive():
                    process.kill()
        for process in self.ps:
            process.join(timeout=5)
        for connection in self._connections:
            connection.close()

    def exit(self):
        if self._closed:
            return
        try:
            self._call(range(len(self.ps)), "exit")
            for process in self.ps:
                process.join(timeout=10)
        finally:
            self._terminate()
            atexit.unregister(self.exit)
