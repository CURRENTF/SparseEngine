"""Ordered CUDA submissions with device token feedback and delayed host results.

Only this module owns the output-copy stream. Model/cache work stays ordered on
the runner stream; a copied result is never an alias of a graph's next output.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from sparseengine.utils.profiler import cpu_timing
from sparseengine.engine.decode_graph_staging import DecodeGraphHostInputs
from sparseengine.platforms import device_runtime


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


class AsyncExecution:
    def __init__(self, runner):
        if not device_runtime.supports_streams(runner.device):
            raise ValueError("Asynchronous execution currently requires CUDA")
        self.runner = runner
        self.copy_stream = device_runtime.new_stream(runner.device)
        self.results: dict[int, DeviceResult] = {}
        self.last_tokens: dict[int, torch.Tensor] = {}
        self.penalties: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.input_tokens = None
        self.keepalive = []
        self.submitted = self.completed = self.peak_inflight = 0
        self.host_input_pool: dict[int, list[DecodeGraphHostInputs]] = {}
        self.host_tensor_pool: dict[tuple, list[torch.Tensor]] = {}

    def acquire_host_tensor(self, template):
        key = (tuple(template.shape), template.dtype, template.is_pinned())
        pool = self.host_tensor_pool.setdefault(key, [])
        tensor = pool.pop() if pool else torch.empty_like(
            template, pin_memory=template.is_pinned())
        self.keepalive.append(tensor)
        return tensor

    def acquire_host_inputs(self, capacity):
        pool = self.host_input_pool.setdefault(capacity, [])
        inputs = pool.pop() if pool else DecodeGraphHostInputs.allocate(capacity, pin_memory=True)
        self.keepalive.append(inputs)
        return inputs

    @cpu_timing.timed
    def prepare_inputs(self, input_ids, seqs):
        # Request identity, not the previous batch row, defines feedback.
        missing = any(s.seq_id not in self.last_tokens for s in seqs)
        initial = torch.tensor([s.decode_input_token for s in seqs], device=input_ids.device) if missing else None
        values = torch.stack([
            self.last_tokens[s.seq_id] if s.seq_id in self.last_tokens else initial[i]
            for i, s in enumerate(seqs)
        ])
        input_ids[:len(seqs)].copy_(values)
        if input_ids.numel() > len(seqs):
            input_ids[len(seqs):].copy_(values[0].expand(input_ids.numel()-len(seqs)))
        self.input_tokens = values

    @cpu_timing.timed
    def apply_penalties(self, logits, seqs):
        if not any(s.has_sampling_penalty for s in seqs):
            return logits
        result = logits.float().clone()
        for row, seq in enumerate(seqs):
            if not seq.has_sampling_penalty:
                continue
            if seq.seq_id not in self.penalties:
                present = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
                repeated = torch.zeros_like(present)
                prompt = torch.tensor(seq.prompt_token_ids, dtype=torch.long, device=logits.device)
                repeated[prompt] = True
                # Re-entering after a drained control/recompute boundary.
                history = torch.tensor(seq.completion_token_ids, dtype=torch.long, device=logits.device)
                present[history] = True
                repeated[history] = True
                self.penalties[seq.seq_id] = present, repeated
            present, repeated = self.penalties[seq.seq_id]
            values = result[row]
            if seq.repetition_penalty != 1.0:
                penalized = torch.where(values > 0, values / seq.repetition_penalty,
                                        values * seq.repetition_penalty)
                values.copy_(torch.where(repeated, penalized, values))
            if seq.presence_penalty != 0.0:
                values.sub_(present.to(values.dtype) * seq.presence_penalty)
        return result

    @cpu_timing.timed
    def submit(self, ticket, seqs, is_prefill):
        if ticket in self.results:
            raise RuntimeError(f"Duplicate asynchronous submission {ticket}")
        runner = self.runner
        self.input_tokens = None
        self.keepalive = []
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
        for i, seq in enumerate(seqs):
            if not is_prefill or seq.is_last_chunk_prefill:
                self.last_tokens[seq.seq_id] = tokens[i]
                if seq.seq_id in self.penalties:
                    present, repeated = self.penalties[seq.seq_id]
                    present[tokens[i]] = True
                    repeated[tokens[i]] = True
        # Snapshots are kept alive until DMA completes, including H2D staging.
        keepalive = [tokens, self.input_tokens, logprobs, *self.keepalive]
        main = device_runtime.current_stream(runner.device)
        self.copy_stream.wait_stream(main)
        def copy_host(tensor):
            host = torch.empty_like(tensor, device='cpu', pin_memory=True)
            host.copy_(tensor, non_blocking=True)
            return host
        with device_runtime.stream_context(self.copy_stream):
            host_tokens = copy_host(tokens)
            host_inputs = copy_host(self.input_tokens) if self.input_tokens is not None else None
            host_logprobs = logprobs.copy_with(copy_host) if logprobs is not None else None
            event = device_runtime.new_event(runner.device)
            event.record(self.copy_stream)
        self.results[ticket] = DeviceResult(seqs, is_prefill, event, host_tokens,
                                           host_inputs, host_logprobs, records, keepalive)
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
                cache._record_prefix_materialization(seq, tokens, slots)
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
                "device_feedback_requests": len(self.last_tokens),
                "penalty_requests": len(self.penalties)}

    @cpu_timing.timed
    def prepare_synchronous_execution(self):
        if self.results:
            raise RuntimeError("Synchronous execution requires all asynchronous results to retire")
        # A synchronous recovery step can advance any live request. Re-seed
        # feedback and penalties from accepted history on the next submission.
        self.last_tokens.clear()
        self.penalties.clear()

    def forget(self, seq_ids):
        for seq_id in seq_ids:
            self.last_tokens.pop(seq_id, None)
            self.penalties.pop(seq_id, None)
