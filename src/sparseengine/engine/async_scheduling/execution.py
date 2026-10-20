"""Ordered CUDA submissions with device token feedback and delayed host results.

Only this module owns the output-copy stream. Model/cache work stays ordered on
the runner stream; a copied result is never an alias of a graph's next output.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from sparseengine.utils.profiler import cpu_timing
from sparseengine.engine.decode_graph_staging import DecodeGraphHostInputs
from sparseengine.platforms import device_runtime
from sparseengine.engine.async_scheduling.output_pool import HostOutputPool
from sparseengine.operators.async_state import prepare_async_state_provider


class AsyncDrainRequired(RuntimeError):
    """Scheduling must retire outstanding work before preempting storage."""


@dataclass
class DeviceLogprobs:
    sampled: torch.Tensor
    values: torch.Tensor | None = None
    indices: torch.Tensor | None = None

    def copy_with(self, copy):
        return DeviceLogprobs(copy(self.sampled),
                              copy(self.values) if self.values is not None else None,
                              copy(self.indices) if self.indices is not None else None)


@dataclass
class DeviceResult:
    seqs: list
    is_prefill: bool
    event: object
    tokens: torch.Tensor
    inputs: torch.Tensor | None
    logprobs: DeviceLogprobs | None
    prefix_records: list
    keepalive: list
    output_buffers: list


class AsyncExecution:
    def __init__(self, runner):
        if not device_runtime.supports_streams(runner.device):
            raise ValueError("Asynchronous execution currently requires CUDA")
        self.runner = runner
        self.copy_stream = device_runtime.new_stream(runner.device)
        self.results: dict[int, DeviceResult] = {}
        self._request_rows: dict[int, int] = {}
        self._penalty_rows: dict[int, int] = {}
        self._feedback_ready: set[int] = set()
        self._last_batch_ids = None
        self._last_batch_tokens = None
        self._free_feedback: list[int] = []
        self._free_penalties: list[int] = []
        self._next_feedback = self._next_penalty = 0
        self._feedback = torch.empty(0, dtype=torch.long, device=runner.device)
        self._penalty_flags = torch.empty((1, 1), dtype=torch.uint8, device=runner.device)
        self._penalty_parameters = None
        self._step_metadata = None
        self._metadata_cache = OrderedDict()
        self._is_prefill = False
        self.state_provider = prepare_async_state_provider(torch.device(runner.device))
        self.output_pool = HostOutputPool()
        self.input_tokens = None
        self.keepalive = []
        self.submitted = self.completed = self.peak_inflight = 0
        self.host_input_pool: dict[int, list[DecodeGraphHostInputs]] = {}
        self.host_tensor_pool: dict[tuple, list[torch.Tensor]] = {}

    def acquire_host_tensor(self, template):
        return self._acquire_host(tuple(template.shape), template.dtype, template.is_pinned())

    def _acquire_host(self, shape, dtype, pinned=True):
        key = (shape, dtype, pinned)
        pool = self.host_tensor_pool.setdefault(key, [])
        tensor = pool.pop() if pool else torch.empty(shape, dtype=dtype, pin_memory=pinned)
        self.keepalive.append(tensor)
        return tensor

    def _upload(self, values, dtype=torch.int64):
        array = np.asarray(values)
        host = self._acquire_host(array.shape, dtype)
        host.numpy()[...] = array
        uploaded = host.to(device=self.runner.device, non_blocking=True)
        self.keepalive.append(uploaded)
        return uploaded

    def _ensure_feedback(self, seqs):
        for seq in seqs:
            if seq.seq_id not in self._request_rows:
                if self._free_feedback:
                    slot = self._free_feedback.pop()
                else:
                    slot = self._next_feedback
                    self._next_feedback += 1
                self._request_rows[seq.seq_id] = slot
        if self._next_feedback > self._feedback.numel():
            capacity = 1 << (self._next_feedback - 1).bit_length()
            previous = self._feedback
            self.keepalive.append(previous)
            self._feedback = torch.empty(capacity, dtype=torch.long, device=self.runner.device)
            if previous.numel():
                self._feedback[:previous.numel()].copy_(previous)

    def _metadata(self, seqs):
        if self._step_metadata is None:
            ids = tuple(s.seq_id for s in seqs)
            publish = tuple(not self._is_prefill or s.is_last_chunk_prefill for s in seqs)
            reusable = all(sid in self._feedback_ready for sid in ids)
            key = (ids, publish)
            if reusable and key in self._metadata_cache:
                self._step_metadata = self._metadata_cache[key]
                self._metadata_cache.move_to_end(key)
                return self._step_metadata
            self._ensure_feedback(seqs)
            self._step_metadata = self._upload([
                [self._request_rows[s.seq_id] for s in seqs],
                [s.decode_input_token for s in seqs],
                [s.seq_id in self._feedback_ready for s in seqs],
                [self._penalty_rows.get(s.seq_id, -1) for s in seqs],
                publish,
            ])
            if reusable:
                self._metadata_cache[key] = self._step_metadata
                limit = max(2, getattr(getattr(self.runner, "config", None), "async_max_inflight", 2))
                while len(self._metadata_cache) > limit:
                    self._metadata_cache.popitem(last=False)
        return self._step_metadata

    def acquire_host_inputs(self, capacity):
        pool = self.host_input_pool.setdefault(capacity, [])
        inputs = pool.pop() if pool else DecodeGraphHostInputs.allocate(capacity, pin_memory=True)
        self.keepalive.append(inputs)
        return inputs

    @cpu_timing.timed
    def prepare_inputs(self, input_ids, seqs):
        if input_ids.numel() < len(seqs) or not seqs:
            raise ValueError("Async feedback needs a nonempty batch fitting the input capacity")
        if tuple(s.seq_id for s in seqs) == self._last_batch_ids:
            # An unchanged order already has a private packed device result.
            # Keep the request table for later reorder/join/leave transitions.
            values = self._last_batch_tokens
            input_ids[:len(seqs)].copy_(values)
            if input_ids.numel() > len(seqs):
                input_ids[len(seqs):].copy_(values[0].expand(input_ids.numel() - len(seqs)))
            self.input_tokens = values
            return
        metadata = self._metadata(seqs)
        # Private selected tokens survive overwrites of graph input storage.
        values = torch.empty(len(seqs), dtype=torch.long, device=input_ids.device)
        self.state_provider.gather(self._feedback, metadata, input_ids, values)
        self.input_tokens = values

    def _ensure_penalties(self, logits, seqs):
        new = [s for s in seqs if s.has_sampling_penalty and s.seq_id not in self._penalty_rows]
        if not new:
            return
        vocab = logits.shape[-1]
        if self._penalty_parameters is not None and vocab != self._penalty_flags.shape[1]:
            raise ValueError("Async penalty vocabulary changed within one runner")
        for seq in new:
            if self._free_penalties:
                row = self._free_penalties.pop()
            else:
                row = self._next_penalty
                self._next_penalty += 1
            self._penalty_rows[seq.seq_id] = row
        previous = self._penalty_flags
        if self._penalty_parameters is None or self._next_penalty > previous.shape[0]:
            capacity = 1 << (self._next_penalty - 1).bit_length()
            self._penalty_flags = torch.empty((capacity, vocab), dtype=torch.uint8, device=logits.device)
            parameters = torch.empty((2, capacity), dtype=torch.float32, device=logits.device)
            if self._penalty_parameters is not None:
                self.keepalive.extend((previous, self._penalty_parameters))
                self._penalty_flags[:previous.shape[0]].copy_(previous)
                parameters[:, :previous.shape[0]].copy_(self._penalty_parameters)
            self._penalty_parameters = parameters
        rows = self._upload([self._penalty_rows[s.seq_id] for s in new])
        self._penalty_flags.index_fill_(0, rows, 0)
        parameters = self._upload([[s.repetition_penalty for s in new],
                                   [s.presence_penalty for s in new]], torch.float32)
        self._penalty_parameters.index_copy_(1, rows, parameters)
        for source, flag in (("prompt_token_ids", 1), ("completion_token_ids", 3)):
            indices = []
            for seq in new:
                tokens = getattr(seq, source)
                if any(t < 0 or t >= vocab for t in tokens):
                    raise ValueError("Sampling penalty history contains an out-of-vocabulary token")
                indices.extend(self._penalty_rows[seq.seq_id] * vocab + t for t in set(tokens))
            if indices:
                self._penalty_flags.view(-1).index_fill_(0, self._upload(indices), flag)
        self._step_metadata = None
        self._metadata_cache.clear()

    @cpu_timing.timed
    def apply_penalties(self, logits, seqs):
        if not any(s.has_sampling_penalty for s in seqs):
            return logits
        if logits.ndim != 2 or logits.shape[0] != len(seqs):
            raise ValueError("Sampling penalties require one logits row per request")
        self._ensure_penalties(logits, seqs)
        metadata = self._metadata(seqs)
        result = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
        self.state_provider.penalize(logits, self._penalty_flags, metadata[3],
                                     self._penalty_parameters, result)
        return result

    @cpu_timing.timed
    def submit(self, ticket, seqs, is_prefill):
        if ticket in self.results:
            raise RuntimeError(f"Duplicate asynchronous submission {ticket}")
        runner = self.runner
        self.input_tokens = None
        self.keepalive = []
        self._step_metadata = None
        self._is_prefill = bool(is_prefill)
        records = []
        runner.cache_manager._async_prefix_records = records
        runner._async_submitting = True
        runner.cache_manager._step_host_allocator = self
        runner.decode_graph_runner.async_input_provider = self
        try:
            tokens, logprobs = runner.run(seqs, is_prefill)
        finally:
            runner._async_submitting = False
            runner.cache_manager._step_host_allocator = None
            runner.cache_manager._async_prefix_records = None
            runner.decode_graph_runner.async_input_provider = None
        # Every TP rank needs the same device feedback for its next embedding.
        if runner.parallel_context.attn_tp_rank == 0:
            tokens = tokens.clone()
        else:
            tokens = torch.empty(len(seqs), dtype=torch.long, device=runner.device)
        if runner.parallel_context.attn_tp_size > 1:
            dist.broadcast(tokens, src=0, group=runner.parallel_context.attn_tp.process_group)
        metadata = self._metadata(seqs)
        self.state_provider.commit(tokens, metadata[0], metadata[3], metadata[4],
                                    self._feedback, self._penalty_flags)
        self._feedback_ready.update(s.seq_id for s in seqs if not is_prefill or s.is_last_chunk_prefill)
        if all(not is_prefill or s.is_last_chunk_prefill for s in seqs):
            self._last_batch_ids = tuple(s.seq_id for s in seqs)
            self._last_batch_tokens = tokens
        else:
            self._last_batch_ids = self._last_batch_tokens = None
        # Snapshots are kept alive until DMA completes, including H2D staging.
        keepalive = [tokens, self.input_tokens, logprobs, metadata, self._feedback,
                     self._penalty_flags, self._penalty_parameters, *self.keepalive]
        main = device_runtime.current_stream(runner.device)
        self.copy_stream.wait_stream(main)
        output_buffers = []
        def copy_host(tensor):
            host, buffer = self.output_pool.acquire(tensor)
            output_buffers.append(buffer)
            host.copy_(tensor, non_blocking=True)
            return host
        with device_runtime.stream_context(self.copy_stream):
            host_tokens = copy_host(tokens)
            host_inputs = copy_host(self.input_tokens) if records and self.input_tokens is not None else None
            host_logprobs = logprobs.copy_with(copy_host) if logprobs is not None else None
            event = device_runtime.new_event(runner.device)
            event.record(self.copy_stream)
        self.results[ticket] = DeviceResult(seqs, is_prefill, event, host_tokens,
                                           host_inputs, host_logprobs, records, keepalive, output_buffers)
        self.submitted += 1
        self.peak_inflight = max(self.peak_inflight, len(self.results))

    @cpu_timing.timed
    def collect(self, ticket, discarded=()):
        result = self.results[ticket]
        device_runtime.synchronize_event(result.event)  # This result, never the whole stream.
        ignored = set(discarded)
        input_tokens = result.inputs.tolist() if result.inputs is not None else None
        input_by_id = dict(zip((s.seq_id for s in result.seqs), input_tokens or []))
        cache = self.runner.cache_manager
        for seq, tokens, slots in result.prefix_records:
            if seq.seq_id not in ignored:
                if not result.is_prefill:
                    tokens = [input_by_id[seq.seq_id]]
                cache._record_frozen_prefix_materialization(seq, tokens, slots)
        if result.prefix_records:
            live = [s for s in result.seqs if s.seq_id not in ignored]
            cache.publish_pending_prefix_blocks(live)
        tokens = result.tokens.tolist()
        logs = None
        if result.logprobs:
            sampled, values, indices = result.logprobs.sampled, result.logprobs.values, result.logprobs.indices
            sampled = sampled.tolist()
            tops = [None] * len(result.seqs)
            if values is not None:
                values, indices = values.tolist(), indices.tolist()
                for row, seq in enumerate(result.seqs):
                    count = int(seq.logprobs or 0)
                    if count:
                        tops[row] = dict(zip(indices[row][:count], values[row][:count]))
            logs = sampled, tops
        self.completed += 1
        del self.results[ticket]
        self.output_pool.release(result.output_buffers)
        for buffer in result.keepalive:
            if isinstance(buffer, DecodeGraphHostInputs):
                self.host_input_pool[buffer.batch_capacity].append(buffer)
            elif isinstance(buffer, torch.Tensor) and buffer.device.type == "cpu":
                key = (tuple(buffer.shape), buffer.dtype, buffer.is_pinned())
                self.host_tensor_pool.setdefault(key, []).append(buffer)
        return tokens, logs

    def stats(self):
        return {"submitted": self.submitted, "completed": self.completed,
                "inflight": len(self.results), "peak_inflight": self.peak_inflight,
                "device_feedback_requests": len(self._request_rows),
                "penalty_requests": len(self._penalty_rows),
                "state_provider": self.state_provider.name,
                "feedback_capacity": self._feedback.numel(),
                "penalty_state_bytes": self._penalty_flags.numel(),
                "output_pool_allocations": self.output_pool.allocations,
                "output_pool_reuses": self.output_pool.reuses,
                "output_pool_cached_bytes": self.output_pool.cached_bytes,
                "output_pool_live_bytes": self.output_pool.live_bytes}

    @cpu_timing.timed
    def prepare_synchronous_execution(self):
        if self.results:
            raise RuntimeError("Synchronous execution requires all asynchronous results to retire")
        # A synchronous recovery step can advance any live request. Re-seed
        # feedback and penalties from accepted history on the next submission.
        self._request_rows.clear()
        self._penalty_rows.clear()
        self._feedback_ready.clear()
        self._last_batch_ids = self._last_batch_tokens = None
        self._free_feedback.clear()
        self._free_penalties.clear()
        self._next_feedback = self._next_penalty = 0
        self._step_metadata = None
        self._metadata_cache.clear()


    def assert_releasable(self, seq_ids) -> None:
        """Control-boundary guard; a completed event is not a retired result.

        Reuse the existing result ownership, not a second per-request counter.
        Do not wait here: the driver must retire the affected tickets first.
        Unrelated requests may remain in flight.
        """
        if not self.results:
            return
        targets = set(seq_ids)
        for ticket, result in self.results.items():
            if any(seq.seq_id in targets for seq in result.seqs):
                raise AsyncDrainRequired(
                    f"Cache release requires asynchronous ticket {ticket} to retire."
                )

    def forget(self, seq_ids):
        seq_ids = tuple(seq_ids)
        self.assert_releasable(seq_ids)
        self._last_batch_ids = self._last_batch_tokens = None
        for seq_id in seq_ids:
            row = self._request_rows.pop(seq_id, None)
            if row is not None:
                self._free_feedback.append(row)
            row = self._penalty_rows.pop(seq_id, None)
            if row is not None:
                self._free_penalties.append(row)
            self._feedback_ready.discard(seq_id)
        self._step_metadata = None
        self._metadata_cache.clear()
