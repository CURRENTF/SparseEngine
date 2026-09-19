from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import queue
import traceback
from pathlib import Path
from typing import Any

import torch


METHOD_CHOICES = (
    "vanilla",
    "streamingllm",
    "attention-sink",
    "attention_sink",
    "snapkv",
    "kvzip",
    "pyramidkv",
    "h2o",
    "rkv",
    "skipkv",
    "quest",
    "omnikv",
    "deltakv",
    "deltakv-less-memory",
    "deltakv-less-memory-cudagraph",
    "kivi",
    "turboquant",
    "fp8_kv",
)

GLM_GRAPH_METHODS = frozenset(
    {"vanilla", "streamingllm", "snapkv", "h2o", "omnikv", "rkv"}
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_arg(value: str) -> dict[str, Any]:
    if value is None:
        return {}
    value = str(value).strip()
    if value.startswith("@"):
        value = Path(value[1:]).expanduser().read_text(encoding="utf-8")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--hyper_params must be a JSON object.")
    return parsed


def _sparse_kwargs(method: str) -> dict[str, Any]:
    return {"sparse_method": "vanilla" if method == "vanilla" else method}


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> dict[str, float | int]:
    k = min(int(k), int(a.numel()), int(b.numel()))
    a_top = set(a.topk(k).indices.tolist())
    b_top = set(b.topk(k).indices.tolist())
    intersection = len(a_top & b_top)
    return {"intersection": intersection, "ratio": float(intersection / k if k else 1.0)}


def _compare_logits(
    eager: torch.Tensor,
    graph: torch.Tensor,
    *,
    atol: float = 0.05,
    rtol: float = 0.05,
) -> dict[str, Any]:
    if eager.shape != graph.shape:
        raise ValueError(
            "Logit shape mismatch: "
            f"eager={tuple(eager.shape)} graph={tuple(graph.shape)}"
        )
    diff = (eager - graph).abs()
    tolerance = float(atol) + float(rtol) * eager.abs()
    tolerance_ratio = diff / tolerance.clamp_min(torch.finfo(torch.float32).eps)
    result: dict[str, Any] = {
        "shape": list(eager.shape),
        "atol": float(atol),
        "rtol": float(rtol),
        "within_tolerance": bool(torch.all(diff <= tolerance).item()),
        "max_tolerance_ratio": float(tolerance_ratio.max().item()),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "argmax_match": eager.argmax(dim=-1).tolist() == graph.argmax(dim=-1).tolist(),
        "eager_argmax": eager.argmax(dim=-1).tolist(),
        "graph_argmax": graph.argmax(dim=-1).tolist(),
        "rows": [],
        "topk_overlap": {},
    }
    for row in range(eager.shape[0]):
        row_diff = diff[row]
        result["rows"].append(
            {
                "row": row,
                "max_abs_diff": float(row_diff.max().item()),
                "mean_abs_diff": float(row_diff.mean().item()),
                "argmax_match": int(eager[row].argmax().item()) == int(graph[row].argmax().item()),
                "eager_argmax": int(eager[row].argmax().item()),
                "graph_argmax": int(graph[row].argmax().item()),
                "top5": _topk_overlap(eager[row], graph[row], 5),
                "top10": _topk_overlap(eager[row], graph[row], 10),
            }
        )
    for k in (1, 5, 10, 50):
        row_scores = [
            _topk_overlap(eager[row], graph[row], k)
            for row in range(eager.shape[0])
        ]
        result["topk_overlap"][str(k)] = {
            "min_ratio": min(item["ratio"] for item in row_scores),
            "avg_ratio": sum(item["ratio"] for item in row_scores) / len(row_scores),
        }
    return result


def _tensor_summary(tensor: torch.Tensor | None, *, limit: int = 16) -> dict[str, Any] | None:
    if tensor is None:
        return None
    detached = tensor.detach()
    flat = detached.flatten()
    preview = flat[:limit].cpu()
    out: dict[str, Any] = {
        "shape": [int(x) for x in detached.shape],
        "dtype": str(detached.dtype),
        "numel": int(flat.numel()),
        "sha256": _tensor_sha256(detached),
    }
    if flat.numel() == 0:
        out.update({"sum": 0, "min": None, "max": None, "preview": []})
        return out
    if detached.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
        torch.bool,
    ):
        values = flat.to(torch.int64)
        out.update(
            {
                "sum": int(values.sum().item()),
                "min": int(values.min().item()),
                "max": int(values.max().item()),
                "preview": [int(x) for x in preview.to(torch.int64).tolist()],
            }
        )
    else:
        values = flat.float()
        out.update(
            {
                "sum": float(values.sum().item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "preview": [float(x) for x in preview.float().tolist()],
            }
        )
    return out


def _canonical_method(method: str) -> str:
    aliases = {
        "attention-sink": "streamingllm",
        "attention_sink": "streamingllm",
    }
    return aliases.get(str(method), str(method))


def _install_method_instrumentation(llm) -> dict[str, int]:
    """Count method-path calls made after engine warmup.

    These wrappers are diagnostic-only and live inside the isolated worker.
    They make post-forward eviction/compaction calls auditable without changing
    the cache-manager hot path or relying on log text.
    """

    calls: dict[str, int] = {}
    targets = (
        (
            "cache",
            llm.model_runner.cache_manager,
            (
                "free_prefix_recent_slots_batch_layers",
                "free_part_slots_batch_layers",
                "free_part_slots_batch",
                "free_part_slots",
                "evict_after_decode",
                "update_decode_attention_scores_all_layers",
                "accumulate_decode_headwise_logits",
                "rkv_joint_scores",
                "materialize_attention_keys",
            ),
        ),
        (
            "controller",
            llm.model_runner.sparse_controller,
            ("_update_dynamic_omnikv_indices",),
        ),
        (
            "runtime",
            llm.model_runner.sparse_controller.runtime,
            ("_update_dynamic_indices",),
        ),
    )
    for prefix, target, names in targets:
        for name in names:
            original = getattr(target, name, None)
            if not callable(original):
                continue
            key = f"{prefix}.{name}"
            calls[key] = 0

            def wrapped(*args, _key=key, _original=original, **kwargs):
                calls[_key] += 1
                return _original(*args, **kwargs)

            setattr(target, name, wrapped)
    return calls


def _graph_counter_snapshot(runner) -> dict[str, int]:
    return {
        "capture_count": int(runner.capture_count),
        "replay_count": int(runner.replay_count),
        "eager_static_count": int(runner.eager_static_count),
        "force_eager_count": int(runner.force_eager_count),
        "graph_count": sum(
            state.graph is not None for state in runner._graphs.values()
        ),
    }


def _start_graph_measurement(llm) -> dict[str, int]:
    """Snapshot warmup counters without invalidating its CUDA graph pool.

    A captured graph owns the allocator private pool represented by
    ``runner.graph_pool``. Clearing the last graph while a captured allocation
    is still referenced leaves that handle non-reusable in PyTorch. Preserve
    the warmup graphs and report business-request activity as counter deltas.
    """

    runner = llm.model_runner.decode_graph_runner
    return _graph_counter_snapshot(runner)


def _graph_runtime_summary(
    llm,
    *,
    use_graph: bool,
    counters_before: dict[str, int],
) -> dict[str, Any]:
    runner = llm.model_runner.decode_graph_runner
    graph_states = [state for state in runner._graphs.values() if state.graph is not None]
    counters_after = _graph_counter_snapshot(runner)
    counter_delta = {
        name: int(counters_after[name]) - int(counters_before[name])
        for name in (
            "capture_count",
            "replay_count",
            "eager_static_count",
            "force_eager_count",
        )
    }
    return {
        "requested": bool(use_graph),
        "config_enabled": bool(llm.config.decode_graph),
        "model": str(llm.config.model),
        "model_type": str(getattr(llm.config.hf_config, "model_type", "")),
        "configured_sparse_method": str(
            getattr(llm.config, "sparse_method", "") or "vanilla"
        ),
        "runner_method": str(runner.method or "vanilla"),
        "graph_active": bool(graph_states),
        "graph_count": len(graph_states),
        "capture_count": int(runner.capture_count),
        "replay_count": int(runner.replay_count),
        "eager_static_count": int(runner.eager_static_count),
        "force_eager_count": int(runner.force_eager_count),
        "counters_before": dict(counters_before),
        "counters_after": counters_after,
        "counter_delta": counter_delta,
        "fallback": bool(
            counter_delta["force_eager_count"]
            or (use_graph and not graph_states)
        ),
        "graph_keys": [
            {
                "method": str(state.key.method or "vanilla"),
                "batch_size": int(state.key.batch_size),
                "context_capacity": int(state.capture_context_capacity),
                "graph_path_id": str(state.key.graph_path_id),
                "capture_sampling": bool(state.key.capture_sampling),
            }
            for state in graph_states
        ],
    }


def _validate_graph_runtime(summary: dict[str, Any]) -> None:
    failures = []
    delta = summary.get("counter_delta", summary)
    if not summary.get("config_enabled"):
        failures.append("decode_graph config is disabled")
    if not summary.get("graph_active") or int(summary.get("graph_count", 0)) <= 0:
        failures.append("no captured CUDA Graph is active")
    if int(summary.get("capture_count", 0)) <= 0:
        failures.append("no CUDA Graph capture exists in the engine lifetime")
    if int(delta.get("replay_count", 0)) <= 0:
        failures.append("business-request replay_count did not increase")
    if int(delta.get("eager_static_count", 0)) != 0:
        failures.append("graph run executed eager-static decode")
    if int(delta.get("force_eager_count", 0)) != 0:
        failures.append("graph run forced eager decode")
    if summary.get("fallback"):
        failures.append("graph runtime reported fallback")
    if failures:
        raise RuntimeError("CUDA Graph runtime gate failed: " + "; ".join(failures))


def _validate_eager_runtime(summary: dict[str, Any]) -> None:
    failures = []
    delta = summary.get("counter_delta", summary)
    if summary.get("config_enabled"):
        failures.append("eager control unexpectedly enabled decode_graph")
    if summary.get("graph_active") or int(summary.get("graph_count", 0)) != 0:
        failures.append("eager control retained a captured CUDA Graph")
    if int(delta.get("eager_static_count", 0)) <= 0:
        failures.append("eager control did not execute eager-static decode")
    if int(delta.get("capture_count", 0)) != 0 or int(
        delta.get("replay_count", 0)
    ) != 0:
        failures.append("eager control captured or replayed a CUDA Graph")
    if int(delta.get("force_eager_count", 0)) != 0:
        failures.append("eager control unexpectedly used force-eager routing")
    if failures:
        raise RuntimeError("Eager runtime gate failed: " + "; ".join(failures))


def _capture_selection_trace(
    llm,
    *,
    step: int,
    stage: str,
    logical_context_len: int,
    use_graph: bool,
) -> dict[str, Any]:
    sparse_controller = llm.model_runner.sparse_controller
    cache_manager = llm.model_runner.cache_manager
    layers: dict[str, Any] = {}
    for layer_idx, state in sparse_controller.layer_batch_sparse_states.items():
        active = state.active_compressed_indices
        attn_score = state.attn_score
        should_record = (
            active is not None
            or state.active_indices is not None
            or state.active_slots is not None
            or attn_score is not None
            or int(layer_idx) in set(getattr(sparse_controller, "obs_layer_ids", []))
        )
        if not should_record:
            continue
        layers[str(int(layer_idx))] = {
            "active_indices": _tensor_summary(state.active_indices),
            "active_slots": _tensor_summary(state.active_slots),
            "active_compressed_indices": _tensor_summary(active),
            "attn_score": _tensor_summary(attn_score, limit=8),
            "context_lens": _tensor_summary(state.context_lens),
            "req_indices": _tensor_summary(state.req_indices),
            "max_context_len": (
                None
                if state.max_context_len is None
                else int(state.max_context_len)
            ),
        }

    compressed_lens = getattr(cache_manager, "_deltakv_decode_static_compressed_lens", None)
    rkv_materializer_layers: list[int] = []
    layer_indices = getattr(cache_manager, "kv_transformer_layer_indices", lambda: ())()
    has_materializer = getattr(cache_manager, "has_attention_key_materializer", None)
    if callable(has_materializer):
        rkv_materializer_layers = [
            int(layer_idx)
            for layer_idx in layer_indices
            if has_materializer(int(layer_idx))
        ]
    return {
        "step": int(step),
        "stage": str(stage),
        "logical_context_len": int(logical_context_len),
        "attention_cache_layout": str(llm.config.attention_cache_layout),
        "use_graph": bool(use_graph),
        "compressed_lens": _tensor_summary(compressed_lens),
        "layers": layers,
        "dynamic_selection": sparse_controller.debug_state_summary()[
            "dynamic_selection"
        ],
        "cache": cache_manager.debug_state_summary(),
        "rkv_materializer_layers": rkv_materializer_layers,
    }


def _live_row_lengths(trace: dict[str, Any]) -> list[int]:
    live_rows = trace.get("cache", {}).get("live_rows", {})
    return [
        int(record["row_len"])
        for records in live_rows.values()
        for record in records
    ]


def _has_physical_compaction(traces: list[dict[str, Any]]) -> bool:
    for trace in traces:
        if "logical_context_lens" in trace:
            logical = trace["logical_context_lens"]
            if any(
                int(record["row_len"]) < logical[str(record["seq_id"])]
                for records in trace.get("cache", {}).get("live_rows", {}).values()
                for record in records
                if str(record["seq_id"]) in logical
            ):
                return True
            continue
        row_lengths = _live_row_lengths(trace)
        if row_lengths and min(row_lengths) < int(trace["logical_context_len"]):
            return True
    return False


def _has_omnikv_selection(traces: list[dict[str, Any]]) -> bool:
    for trace in traces:
        logical_context_len = int(trace["logical_context_len"])
        for layer in trace.get("layers", {}).values():
            active_slots = layer.get("active_slots")
            context_lens = layer.get("context_lens")
            if (
                active_slots is not None
                and int(active_slots.get("numel", 0)) > 0
                and context_lens is not None
                and context_lens.get("max") is not None
                and int(context_lens["max"]) < logical_context_len
            ):
                return True
    return False


def _latest_h2o_state(traces: list[dict[str, Any]]) -> dict[str, Any]:
    for trace in reversed(traces):
        h2o = trace.get("cache", {}).get("h2o")
        if isinstance(h2o, dict):
            return h2o
    return {}


def _build_method_trigger_evidence(
    method: str,
    traces: list[dict[str, Any]],
    method_calls: dict[str, int],
) -> dict[str, Any]:
    method = _canonical_method(method)
    physical_compaction = _has_physical_compaction(traces)
    positive_calls = {key: int(value) for key, value in method_calls.items() if int(value) > 0}
    evidence: dict[str, Any] = {
        "method": method,
        "required": method in (GLM_GRAPH_METHODS | {"kvzip"}) - {"vanilla"},
        "triggered": method == "vanilla",
        "trigger_kind": "dense_baseline" if method == "vanilla" else "",
        "physical_compaction": physical_compaction,
        "method_calls": positive_calls,
    }
    if method in {"streamingllm", "snapkv", "kvzip"}:
        compaction_calls = sum(
            count
            for name, count in positive_calls.items()
            if "free_" in name
        )
        evidence.update(
            triggered=bool(physical_compaction and compaction_calls > 0),
            trigger_kind="physical_eviction_compaction",
            compaction_call_count=int(compaction_calls),
        )
    elif method == "h2o":
        h2o = _latest_h2o_state(traces)
        counters = h2o.get("counters", {})
        eviction_count = sum(
            int(counters.get(name, 0))
            for name in (
                "intermediate_prefill_evictions",
                "final_prefill_evictions",
                "decode_evictions",
            )
        )
        dropped_tokens = int(counters.get("dropped_tokens", 0))
        ring_fallback_rows = int(
            h2o.get("ring_counters", {}).get("fallback_rows", 0)
        )
        evidence.update(
            triggered=bool(eviction_count > 0 and dropped_tokens > 0),
            trigger_kind="score_eviction_compaction",
            h2o_counters=counters,
            h2o_ring_counters=h2o.get("ring_counters", {}),
            eviction_count=eviction_count,
            dropped_tokens=dropped_tokens,
            internal_fallback_rows=ring_fallback_rows,
        )
    elif method == "omnikv":
        selection_calls = int(
            positive_calls.get("controller._update_dynamic_omnikv_indices", 0)
        ) + int(positive_calls.get("runtime._update_dynamic_indices", 0))
        selected = _has_omnikv_selection(traces)
        captured_replay = any(bool(trace.get("use_graph")) for trace in traces)
        evidence.update(
            triggered=bool(
                selected and (selection_calls > 0 or captured_replay)
            ),
            trigger_kind="dynamic_topk_selection",
            selection_call_count=selection_calls,
            compressed_selection_observed=selected,
            execution_mode=(
                "captured_replay"
                if captured_replay and selection_calls == 0
                else "python_eager_or_capture"
            ),
        )
    elif method == "rkv":
        score_calls = sum(
            count
            for name, count in positive_calls.items()
            if "rkv_joint_scores" in name
        )
        materializer_calls = int(
            positive_calls.get("cache.materialize_attention_keys", 0)
        )
        materializer_layers = sorted(
            {
                int(layer_idx)
                for trace in traces
                for layer_idx in trace.get("rkv_materializer_layers", [])
            }
        )
        evidence.update(
            triggered=bool(
                physical_compaction
                and score_calls > 0
                and materializer_calls > 0
                and (
                    materializer_layers
                    or all(trace.get("attention_cache_layout") == "explicit_kv" for trace in traces)
                )
            ),
            trigger_kind="query_scored_eviction_compaction",
            query_score_call_count=int(score_calls),
            materializer_call_count=materializer_calls,
            materializer_layers=materializer_layers,
        )
    return evidence


def _validate_method_trigger(evidence: dict[str, Any]) -> None:
    if evidence.get("required") and not evidence.get("triggered"):
        raise RuntimeError(
            "Sparse method trigger gate failed: "
            + json.dumps(evidence, sort_keys=True)
        )


def _same_provider_eager_model_runner(*args, **kwargs):
    """Spawn-safe diagnostic adapter: bind Graph providers, execute eagerly."""
    from sparseengine.engine.model_runner import ModelRunner

    original_run = ModelRunner.run

    def run_eager(self, *run_args, **run_kwargs):
        # Each TP worker owns its own deserialized config. Changing the leader's
        # config alone would leave follower ranks trying to replay a graph.
        self.config.decode_graph = False
        self.config.decode_graph_startup_capture = False
        return original_run(self, *run_args, **run_kwargs)

    ModelRunner.run = run_eager
    return ModelRunner(*args, **kwargs)


def _run_decode_logits(
    *,
    model_path: str,
    method: str,
    prompt_lens: list[int | list[int]],
    batch_size: int,
    max_tokens: int,
    hyper_params: dict[str, Any],
    use_graph: bool,
    same_provider_eager: bool = False,
    trace_selection: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    from sparseengine import LLM, SamplingParams

    rounds = [
        [lengths] * batch_size if isinstance(lengths, int) else lengths
        for lengths in prompt_lens
    ]
    if not rounds or any(
        len(lengths) != batch_size or any(length <= 0 for length in lengths)
        for lengths in rounds
    ):
        raise ValueError("Each prompt-length round must contain one positive length per request.")
    construct_with_graph = bool(use_graph or same_provider_eager)
    if os.getenv("SPARSEENGINE_DEBUG_SKIP_ENGINE_WARMUP", "0") == "1":
        LLM._warmup = lambda self: None
    elif same_provider_eager and not use_graph:
        import sparseengine.engine.llm_engine as engine_module

        engine_module.ModelRunner = _same_provider_eager_model_runner
        original_warmup = LLM._warmup

        def eager_warmup(self):
            # Providers are already prepared with Graph-compatible contracts.
            # Keep capacity profiling, but do not capture or replay any graphs.
            self.config.decode_graph = False
            self.config.decode_graph_startup_capture = False
            original_warmup(self)

        LLM._warmup = eager_warmup

    decode_capacity = int(hyper_params.get("max_decoding_seqs", batch_size))
    if decode_capacity < batch_size:
        raise ValueError("max_decoding_seqs must cover the oracle's real batch")
    engine_kwargs = {
        **hyper_params,
        **_sparse_kwargs(method),
        "max_model_len": int(hyper_params.get(
            "max_model_len", max(map(max, rounds)) + max_tokens + 100,
        )),
        "max_num_seqs_in_batch": batch_size,
        "max_decoding_seqs": decode_capacity,
        "decode_graph": construct_with_graph,
        "decode_graph_capture_sampling": False,
        "throughput_log_interval_s": 0.0,
    }
    llm = LLM(model_path, **engine_kwargs)
    if same_provider_eager and not use_graph:
        # Keep the graph-selected provider, but execute every decode step through
        # DecodeCudaGraphRunner.run_eager_static() as the graph-independent oracle.
        llm.config.decode_graph = False
    graph_counters_before = _start_graph_measurement(llm)
    method_calls = _install_method_instrumentation(llm)
    captured: list[torch.Tensor] = []
    trace: list[dict[str, Any]] = []
    runtime_step = 0
    generated_token_outputs: list[dict[str, Any]] = []
    decode_batches: list[list[int]] = []
    original_schedule = llm.scheduler.schedule
    scheduled_lengths: dict[str, int] = {}

    def record_schedule():
        seqs, is_prefill, preempted = original_schedule()
        scheduled_lengths.clear()
        scheduled_lengths.update({
            str(seq.seq_id): (
                seq.num_prefilled_tokens + seq.current_chunk_size if is_prefill else len(seq)
            )
            for seq in seqs
        })
        if seqs and not is_prefill:
            decode_batches.append([len(seq) for seq in seqs])
        return seqs, is_prefill, preempted

    llm.scheduler.schedule = record_schedule

    if not use_graph:
        runner = llm.model_runner
        original_run_model = runner.run_model

        def wrapped_run_model(input_ids, positions, is_prefill):
            logits = original_run_model(input_ids, positions, is_prefill)
            if not is_prefill:
                from sparseengine.utils.context import get_context

                # Eager-static invokes this hook before trimming its padded
                # model output; token events contain only live request rows.
                real_rows = len(get_context().seqs)
                captured.append(logits[:real_rows].detach().float().cpu())
            return logits

        runner.run_model = wrapped_run_model
        if runner.decode_graph_runner is not None:
            runner.decode_graph_runner.run_model = wrapped_run_model
    else:
        graph_runner = llm.model_runner.decode_graph_runner
        original_graph_run = graph_runner.run

        def wrapped_graph_run(*args, **kwargs):
            logits, token_ids = original_graph_run(*args, **kwargs)
            if logits is not None:
                captured.append(logits.detach().float().cpu())
            return logits, token_ids

        graph_runner.run = wrapped_graph_run

    try:
        for round_idx, request_lengths in enumerate(rounds):
            prompt_len = max(request_lengths)
            round_decode_step = 0
            round_prefilled_tokens = 0
            prompt_token_ids = []
            for batch_idx in range(batch_size):
                # Use deterministic non-uniform prompts so sparse selection and
                # graph-state reuse bugs are not hidden by identical rows.
                base = 100 + 997 * round_idx + 131 * batch_idx
                prompt_token_ids.append([
                    base + (pos % 127) for pos in range(request_lengths[batch_idx])
                ])
            sampling_params = [
                SamplingParams(temperature=0.0, top_p=1.0, ignore_eos=True, max_tokens=max_tokens)
                for _ in range(batch_size)
            ]
            request_indices = {
                int(llm.add_request(prompt, params)): request_idx
                for request_idx, (prompt, params) in enumerate(
                    zip(prompt_token_ids, sampling_params)
                )
            }

            while not llm.is_finished():
                _, num_tokens = llm.step()
                runtime_step += 1
                if num_tokens > 0:
                    round_prefilled_tokens += int(num_tokens) // int(batch_size)
                    stage = "prefill"
                    logical_context_len = min(
                        int(prompt_len),
                        int(round_prefilled_tokens),
                    )
                elif num_tokens < 0:
                    round_decode_step += 1
                    stage = "decode"
                    logical_context_len = int(prompt_len) + int(round_decode_step)
                else:
                    stage = "idle"
                    logical_context_len = int(prompt_len) + int(round_decode_step)

                generated_token_outputs.append(
                    {
                        "step": int(runtime_step),
                        "round": int(round_idx),
                        "stage": stage,
                        "token_outputs": [
                            {
                                "seq_id": int(seq_id),
                                "request_idx": request_indices[int(seq_id)],
                                "token_ids": [int(token_id) for token_id in token_ids],
                            }
                            for seq_id, token_ids in llm.last_step_token_outputs
                        ],
                    }
                )
                if num_tokens != 0:
                    trace.append(
                        _capture_selection_trace(
                            llm,
                            step=runtime_step,
                            stage=stage,
                            logical_context_len=logical_context_len,
                            use_graph=use_graph,
                        )
                    )
                    trace[-1]["logical_context_lens"] = dict(scheduled_lengths)

        graph_runtime = _graph_runtime_summary(
            llm,
            use_graph=use_graph,
            counters_before=graph_counters_before,
        )
        method_evidence = _build_method_trigger_evidence(
            method,
            trace,
            method_calls,
        )
        runtime_evidence = {
            "graph": graph_runtime,
            "method_trigger": method_evidence,
            "method_calls": {key: int(value) for key, value in method_calls.items()},
            "generated_token_outputs": generated_token_outputs,
            "decode_batches": decode_batches,
        }
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    if not captured:
        raise RuntimeError("No decode logits captured. Use max_tokens >= 3.")
    del trace_selection
    return (
        _canonicalize_decode_logits(torch.cat(captured, dim=0), runtime_evidence),
        trace,
        runtime_evidence,
    )


def _run_decode_logits_worker(result_queue, kwargs: dict[str, Any]):
    try:
        logits, trace, runtime = _run_decode_logits(**kwargs)
        result_queue.put(
            (
                "ok",
                {
                    "logits": logits.numpy(),
                    "trace": trace,
                    "runtime": runtime,
                },
            )
        )
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _run_decode_logits_isolated(
    **kwargs,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_run_decode_logits_worker, args=(result_queue, kwargs))
    process.start()
    try:
        status, payload = result_queue.get(timeout=900)
    except queue.Empty as exc:
        process.terminate()
        process.join(timeout=30)
        raise TimeoutError("Timed out waiting for decode logits worker.") from exc
    process.join()
    if process.exitcode != 0 or status != "ok":
        raise RuntimeError(
            "Decode logits worker failed with "
            f"exitcode={process.exitcode}:\n{payload}"
        )
    return (
        torch.from_numpy(payload["logits"]),
        payload["trace"],
        payload["runtime"],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full eager and decode-CUDA-Graph logits and prove that "
            "the requested sparse runtime path executed."
        )
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--method",
        default="vanilla",
        choices=METHOD_CHOICES,
    )
    parser.add_argument("--prompt_len", type=int, default=2048)
    parser.add_argument(
        "--second_prompt_len",
        type=int,
        default=None,
        help="Run a second generate() on the same LLM instance to test graph reuse.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--prompt_length_rounds",
        help=(
            "JSON list of per-request length lists, for example [[1,3,129],[257,5,33]]. "
            "Overrides prompt_len and second_prompt_len; each round must match batch_size."
        ),
    )
    parser.add_argument("--max_tokens", type=int, default=3)
    parser.add_argument("--hyper_params", default="{}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--trace_selection", action="store_true")
    parser.add_argument(
        "--same_provider_eager",
        action="store_true",
        help=(
            "Construct the eager control with decode_graph enabled so it binds "
            "the same provider, skip startup capture, then execute eager-static."
        ),
    )
    return parser


def _generated_token_ids(runtime: dict[str, Any]) -> list[int]:
    requests: dict[tuple[int, int], list[int]] = {}
    for step in runtime["generated_token_outputs"]:
        for record in step["token_outputs"]:
            key = (int(step["round"]), int(record["request_idx"]))
            requests.setdefault(key, []).extend(map(int, record["token_ids"]))
    return [
        token_id for key in sorted(requests) for token_id in requests[key]
    ]


def _canonicalize_decode_logits(
    logits: torch.Tensor, runtime: dict[str, Any]
) -> torch.Tensor:
    # Admission and physical row order may differ after Graph startup. Compare
    # the same submitted request and decode position, not execution batch order.
    positions: dict[tuple[int, int], int] = {}
    row_keys = []
    for step in runtime["generated_token_outputs"]:
        if step["stage"] != "decode":
            continue
        for record in step["token_outputs"]:
            if len(record["token_ids"]) != 1:
                raise RuntimeError("Decode logit comparison requires one token per row.")
            key = (int(step["round"]), int(record["request_idx"]))
            position = positions.get(key, 0)
            row_keys.append((*key, position))
            positions[key] = position + 1
    if len(row_keys) != logits.shape[0]:
        raise RuntimeError(
            f"Decode logit rows do not match token events: {logits.shape[0]} vs {len(row_keys)}."
        )
    order = sorted(range(len(row_keys)), key=row_keys.__getitem__)
    return logits[order]


def _save_full_logits_artifact(
    path: Path,
    *,
    eager: torch.Tensor,
    graph: torch.Tensor,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scope": "all_decode_rows_and_full_vocabulary",
            "eager": eager.contiguous(),
            "graph": graph.contiguous(),
        },
        path,
    )
    return {
        "path": str(path.resolve()),
        "scope": "all_decode_rows_and_full_vocabulary",
        "artifact_sha256": _file_sha256(path),
        "eager": _tensor_summary(eager),
        "graph": _tensor_summary(graph),
    }


def _validation_error(validator, payload: dict[str, Any]) -> str | None:
    try:
        validator(payload)
    except RuntimeError as exc:
        return str(exc)
    return None


def main(argv: list[str] | None = None):
    args = _build_parser().parse_args(argv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_current_output = False
    hyper_params = None

    try:
        if args.max_tokens < 3:
            raise ValueError(
                "--max_tokens must be >= 3 to force at least one decode step."
            )
        if args.prompt_len <= 0 or args.batch_size <= 0:
            raise ValueError("--prompt_len and --batch_size must be positive.")
        if args.second_prompt_len is not None and args.second_prompt_len <= 0:
            raise ValueError("--second_prompt_len must be positive when provided.")
        if args.atol < 0 or args.rtol < 0:
            raise ValueError("--atol and --rtol must be non-negative.")

        hyper_params = _load_json_arg(args.hyper_params)
        prompt_lens = [args.prompt_len]
        if args.second_prompt_len is not None:
            prompt_lens.append(args.second_prompt_len)
        if args.prompt_length_rounds is not None:
            prompt_lens = json.loads(args.prompt_length_rounds)
            if not isinstance(prompt_lens, list) or not prompt_lens or any(
                not isinstance(lengths, list)
                or len(lengths) != args.batch_size
                or any(type(length) is not int or length <= 0 for length in lengths)
                for lengths in prompt_lens
            ):
                raise ValueError("--prompt_length_rounds requires one positive integer per request in each round.")
        eager_logits, eager_trace, eager_runtime = _run_decode_logits_isolated(
            model_path=args.model_path,
            method=args.method,
            prompt_lens=prompt_lens,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            hyper_params=hyper_params,
            use_graph=False,
            same_provider_eager=args.same_provider_eager,
            trace_selection=args.trace_selection,
        )
        graph_logits, graph_trace, graph_runtime = _run_decode_logits_isolated(
            model_path=args.model_path,
            method=args.method,
            prompt_lens=prompt_lens,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            hyper_params=hyper_params,
            use_graph=True,
            same_provider_eager=args.same_provider_eager,
            trace_selection=args.trace_selection,
        )

        logits_artifact_path = output_path.with_name(
            output_path.stem + ".full_logits.pt"
        )
        logits_artifact = _save_full_logits_artifact(
            logits_artifact_path,
            eager=eager_logits,
            graph=graph_logits,
        )
        comparison = _compare_logits(
            eager_logits,
            graph_logits,
            atol=args.atol,
            rtol=args.rtol,
        )
        eager_token_ids = _generated_token_ids(eager_runtime)
        graph_token_ids = _generated_token_ids(graph_runtime)
        token_ids_match = eager_token_ids == graph_token_ids
        eager_runtime_error = _validation_error(
            _validate_eager_runtime,
            eager_runtime["graph"],
        )
        graph_runtime_error = _validation_error(
            _validate_graph_runtime,
            graph_runtime["graph"],
        )
        eager_method_error = _validation_error(
            _validate_method_trigger,
            eager_runtime["method_trigger"],
        )
        graph_method_error = _validation_error(
            _validate_method_trigger,
            graph_runtime["method_trigger"],
        )
        gates = {
            "full_logits_within_tolerance": bool(
                comparison["within_tolerance"]
            ),
            "argmax_match": bool(comparison["argmax_match"]),
            "generated_token_ids_match": token_ids_match,
            "eager_runtime_contract": eager_runtime_error is None,
            "graph_runtime_contract": graph_runtime_error is None,
            "graph_capture_observed": bool(
                graph_runtime["graph"]["capture_count"] > 0
            ),
            "graph_replay_observed": bool(
                graph_runtime["graph"]["counter_delta"]["replay_count"] > 0
            ),
            "no_eager_or_force_eager_fallback": bool(
                graph_runtime["graph"]["counter_delta"]["eager_static_count"]
                == 0
                and graph_runtime["graph"]["counter_delta"][
                    "force_eager_count"
                ]
                == 0
                and not graph_runtime["graph"]["fallback"]
            ),
            "eager_method_triggered": bool(
                eager_method_error is None
            ),
            "graph_method_triggered": bool(
                graph_method_error is None
            ),
        }
        if args.prompt_length_rounds is not None and any(
            len(set(lengths)) > 1 for lengths in prompt_lens
        ):
            gates["mixed_decode_batch_observed"] = all(
                any(len(set(lengths)) > 1 for lengths in runtime["decode_batches"])
                for runtime in (eager_runtime, graph_runtime)
            )
        gates["no_serving_recapture"] = graph_runtime["graph"]["counter_delta"]["capture_count"] == 0
        passed = all(gates.values())
        output = {
            "status": "success" if passed else "failed",
            "method": args.method,
            "prompt_lens": prompt_lens,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "same_provider_eager": bool(args.same_provider_eager),
            "hyper_params": hyper_params,
            "comparison": comparison,
            "generated_token_ids": {
                "eager": eager_token_ids,
                "graph": graph_token_ids,
                "match": token_ids_match,
            },
            "full_logits_artifact": logits_artifact,
            "runtime": {
                "eager": eager_runtime,
                "graph": graph_runtime,
            },
            "gates": gates,
            "gate_errors": {
                key: value
                for key, value in {
                    "eager_runtime_contract": eager_runtime_error,
                    "graph_runtime_contract": graph_runtime_error,
                    "eager_method_triggered": eager_method_error,
                    "graph_method_triggered": graph_method_error,
                }.items()
                if value is not None
            },
        }
        if args.trace_selection:
            output["selection_trace"] = {
                "eager": eager_trace,
                "graph": graph_trace,
            }
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        wrote_current_output = True
        print(json.dumps({"status": output["status"], "gates": gates}, indent=2))
        if not passed:
            raise RuntimeError(
                "Eager-vs-CUDA-Graph validation gates failed: "
                + json.dumps(gates, sort_keys=True)
            )
    except BaseException:
        if not wrote_current_output:
            output_path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "method": args.method,
                        "same_provider_eager": bool(args.same_provider_eager),
                        "arguments": vars(args),
                        "hyper_params": hyper_params,
                        "error": traceback.format_exc(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
