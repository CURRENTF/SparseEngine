"""Reproducible vLLM latency baseline for SparseEngine comparisons.

Run this script with an isolated vLLM environment. It intentionally imports
vLLM inside ``main`` so the SparseEngine project environment does not need vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import sys
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def _synchronize_worker(worker) -> None:
    import torch
    torch.cuda.synchronize()


def _worker_peak_memory(worker) -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 1024**3


def _install_window_graph_observer(worker):
    """Worker-local CPU counters; no synchronization in the forward hook."""
    from vllm.v1.worker import gpu_model_runner
    if hasattr(worker, "_paper_window_graphs"):
        raise RuntimeError("Window observer already installed")
    worker._paper_window_graphs = {"replay_count": 0, "eager_decode_count": 0}
    original = gpu_model_runner.set_forward_context

    def context(*args, **kwargs):
        mode = kwargs["cudagraph_runtime_mode"]
        key = "eager_decode_count" if mode.name == "NONE" else "replay_count"
        worker._paper_window_graphs[key] += 1
        return original(*args, **kwargs)

    gpu_model_runner.set_forward_context = context


def _window_worker_boundary(worker):
    import torch
    from vllm.compilation.counter import compilation_counter
    torch.cuda.synchronize()
    return {**worker._paper_window_graphs,
            "capture_count": compilation_counter.num_cudagraph_captured}


def _run_decode_window(engine, core, bs, length, args, row, raw_outputs, steps, case_dir):
    """Drain via the existing engine loop, without disabling its async scheduler."""
    from benchmark.efficiency.metrics import PipelinedDecodeWindow, decode_window_fields
    scheduler = core.scheduler
    original_schedule = scheduler.schedule
    original_update = scheduler.update_from_output
    original_has_requests = scheduler.has_requests
    graph_stats = []
    tickets = {}
    draining = False
    peak = 0
    preemptions = 0
    # Recent vLLM assigns private internal IDs; Tangram's pinned older V1 uses
    # the caller ID directly. Persist the actual mapping, never strip suffixes.
    request_id_map = {getattr(state, "external_req_id", internal): internal
                      for internal, state in engine.output_processor.request_states.items()}
    if len(request_id_map) != bs:
        raise RuntimeError("Missing or duplicate external-to-internal request mapping")
    row["request_id_map"] = request_id_map
    core.model_executor.collective_rpc(_install_window_graph_observer)

    def synchronize():
        graph_stats[:] = core.model_executor.collective_rpc(_window_worker_boundary)
        if len(graph_stats) != row["resolved_parallel_topology"]["tensor_parallel_size"]:
            raise RuntimeError("Window boundary did not cover every TP/EP worker")

    window = PipelinedDecodeWindow(bs, args.decode_window_steps,
        args.decode_warmup_steps_after_full, synchronize=synchronize,
        clock=perf_counter, graph_stats=lambda: [dict(r) for r in graph_stats])

    def schedule(*a, **kw):
        nonlocal peak, preemptions
        before = {k: (r.num_computed_tokens, r.num_prompt_tokens)
                  for k, r in scheduler.requests.items()}
        out = original_schedule(*a, **kw)
        counts = out.num_scheduled_tokens
        preemptions += len(out.preempted_req_ids or ())
        if preemptions:
            raise RuntimeError("Full decode batch capacity exceeded: scheduler preemption")
        pure = bool(counts) and all(before[k][0] >= before[k][1] for k in counts)
        if pure:
            peak = max(peak, len(counts))
        ticket = window.submit(is_decode=pure, request_ids=list(counts),
            tokens=sum(counts.values()), admission_complete=len(counts) == bs and pure,
            context_lengths=[before[k][0] + counts[k] for k in counts])
        tickets[id(out)] = (ticket, pure, list(counts))
        return out

    def update(out, result, *a, **kw):
        ticket, pure, ids = tickets.pop(id(out))
        if pure and ticket is not None:
            actual = sum(len(result.sampled_token_ids[result.req_id_to_index[k]]) for k in ids)
        else:
            actual = None
        ret = original_update(out, result, *a, **kw)
        window.complete(ticket, decode_tokens=actual)
        return ret

    scheduler.schedule = schedule
    scheduler.update_from_output = update
    scheduler.has_requests = lambda: False if draining else original_has_requests()
    try:
        while engine.has_unfinished_requests() or core.batch_queue:
            draining = window.needs_boundary
            if draining and not core.batch_queue:
                window.boundary()
                draining = False
            outputs = engine.step()
            for output in outputs:
                if output.finished:
                    tokens = list(output.outputs[0].token_ids)
                    raw_outputs.append(dict(request_id=request_id_map[output.request_id],
                        external_request_id=output.request_id, token_ids=tokens,
                        status="success" if len(tokens) == args.output_len else "model_failed"))
        window.boundary()
        if peak != bs:
            raise RuntimeError("Full decode batch capacity exceeded: incomplete full residency")
        result = window.require_result()
        if len(raw_outputs) != bs or any(r["status"] != "success" for r in raw_outputs):
            raise RuntimeError("Incomplete output: every request must finish the requested output length")
        row.update(status="SUCCESS", require_full_decode_batch=True,
            actual_decode_peak=peak, completed_requests=len(raw_outputs),
            scheduler_preemptions=preemptions, full_admission_reached=peak == bs,
            decode_warmup_steps_after_full=args.decode_warmup_steps_after_full,
            avg_bs=bs,
            mem=max(core.model_executor.collective_rpc(_worker_peak_memory)),
            actual_async_scheduling=bool(core.async_scheduling),
            **decode_window_fields(result))
        _write_json(case_dir / "window.json", result)
    finally:
        steps.extend(window.records)
        _write_jsonl(case_dir / "window_steps.jsonl", window.records)
        scheduler.schedule = original_schedule
        scheduler.update_from_output = original_update
        scheduler.has_requests = original_has_requests


def benchmark_decode_stage(method, length, bs, args, results_dict):
    """Synchronized V1 step diagnostic, called by the canonical microbench CLI.

    The in-process core exposes the scheduled token work before postprocessing
    changes request state. Mixed prefill/decode steps and falling-batch tails
    are never included in the fixed-concurrency decode rate.
    """
    from benchmark.efficiency.metrics import stage_throughput

    row = {"engine": "vllm", "method": method, "length": length,
           "batch_size": bs, "output_len": args.output_len, "status": "FAILED"}
    steps, raw_outputs = [], []
    llm = None
    case_dir = Path(args.output_dir) / f"{method}-{length}-{bs}" if args.output_dir else None
    try:
        window_mode = bool(getattr(args, "decode_window_steps", 0))
        extra = dict(getattr(args, "engine_kwargs_dict", {}))
        label = getattr(args, "backend_label", None)
        if method != "vanilla" and not (
            method == "snapkv" and label == "tangram-snapkv"
            and extra.get("compression_scorer") == "snapkv"
            and extra.get("compression_budget_tokens", 0) > 0
        ):
            raise ValueError("Non-vanilla vLLM stage requires explicit Tangram SnapKV configuration and label")
        if not (args.synchronize_step_timing or window_mode) or not args.require_full_decode_batch:
            raise ValueError("vLLM stage baseline requires synchronized full-batch timing")
        if window_mode and (args.synchronize_step_timing or case_dir is None
                            or args.decode_warmup_steps_after_full < 1):
            raise ValueError("Window timing requires positive warmup, artifacts and no per-step sync")
        if args.max_decode_steps_after_full or args.decode_warmup_steps_after_full < 0:
            raise ValueError("Stage baseline requires untruncated outputs and non-negative warmup")
        if args.admission_wave_size or args.wave_decode_gap_steps or args.require_prefix_cache_hit:
            raise ValueError("vLLM stage baseline does not support admission waves or required prefix hits")
        if length <= 0 or bs <= 0 or args.output_len <= 1:
            raise ValueError("Stage baseline requires positive input/batch and output_len > 1")
        if args.max_model_len_override is not None and args.max_model_len_override < length + args.output_len:
            raise ValueError("max_model_len_override must cover length + output_len")
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
            raise ValueError("Set VLLM_ENABLE_V1_MULTIPROCESSING=0 for the inspectable synchronized V1 core")
        hp = args.hyper_params_dict
        allowed = {"tensor_parallel_size", "expert_parallel_size", "data_parallel_size",
                   "gpu_memory_utilization", "max_num_batched_tokens", "decode_graph",
                   "engine_prefill_chunk_size", "enable_prefix_caching"}
        unknown = set(hp) - allowed
        if unknown:
            raise ValueError(f"Unmapped vLLM stage parameters: {sorted(unknown)}")
        if hp.get("data_parallel_size", 1) != 1 or hp.get("enable_prefix_caching", False):
            raise ValueError("Stage baseline requires DP1 and disabled prefix caching")
        tp = hp.get("tensor_parallel_size", 1)
        if hp.get("expert_parallel_size", 1) not in (1, tp):
            raise ValueError("vLLM stage baseline requires EP1 or EP equal to TP")
        if hp.get("engine_prefill_chunk_size", hp.get("max_num_batched_tokens", 8192)) != hp.get("max_num_batched_tokens", 8192):
            raise ValueError("vLLM has no independent engine_prefill_chunk_size; it must equal max_num_batched_tokens")
        import vllm
        from vllm import LLM, SamplingParams

        config = dict(model=args.model_path, tensor_parallel_size=hp.get("tensor_parallel_size", 1),
                      enable_expert_parallel=hp.get("expert_parallel_size", 1) > 1,
                      gpu_memory_utilization=hp.get("gpu_memory_utilization", 0.9),
                      max_model_len=args.max_model_len_override or length + args.output_len,
                      max_num_seqs=bs, max_num_batched_tokens=hp.get("max_num_batched_tokens", 8192),
                      enable_prefix_caching=False, enable_chunked_prefill=True,
                      async_scheduling=window_mode, enforce_eager=not hp.get("decode_graph", True),
                      seed=42, disable_log_stats=True,
                      compilation_config={"cudagraph_capture_sizes": [bs], "max_cudagraph_capture_size": bs})
        allowed_extra = {"dtype", "compression_scorer", "compression_budget_scope",
                         "compression_budget_tokens", "compression_n_sink_tokens",
                         "compression_window_size", "compression_chunk_size", "compression_scorer_options"}
        if set(extra) - allowed_extra:
            raise ValueError(f"Unmapped or protected vLLM stage options: {sorted(set(extra) - allowed_extra)}")
        if extra.get("compression_chunk_size", config["max_num_batched_tokens"]) != config["max_num_batched_tokens"]:
            raise ValueError("Tangram compression chunk must equal max_num_batched_tokens")
        config.update(extra)
        if label == "tangram-snapkv":
            # This pinned fork rejects persistent score storage with one row,
            # even for a single request. Reserve two rows; still submit only bs.
            config["max_num_seqs"] = max(2, bs)
            row["constructor_capacity_note"] = "Tangram persistent-score guard requires at least two workspace rows; measured concurrency is unchanged"
        row["backend_label"] = label or "vllm-vanilla"
        row["engine_hyper_params"] = config
        row["resolved_parallel_topology"] = {
            "tensor_parallel_size": tp,
            "expert_parallel_size": tp if config["enable_expert_parallel"] else 1,
            "data_parallel_size": 1,
        }
        row["vllm_version"] = vllm.__version__
        row["package_source"] = str(Path(vllm.__file__).resolve())
        if window_mode:
            from benchmark.efficiency.paper import record_package_source
            row["package_identity"] = record_package_source(vllm, case_dir)
        llm = LLM(**config)
        engine = llm.llm_engine
        core = engine.engine_core.engine_core
        if not window_mode and (core.batch_queue is not None or core.async_scheduling):
            raise RuntimeError("Stage diagnostic requires synchronous, non-pipelined EngineCore.step")
        scheduler = core.scheduler
        original_schedule = scheduler.schedule
        scheduled = {}

        def observe_schedule(*schedule_args, **schedule_kwargs):
            before = {key: (request.num_computed_tokens, request.num_prompt_tokens)
                      for key, request in scheduler.requests.items()}
            output = original_schedule(*schedule_args, **schedule_kwargs)
            counts = output.num_scheduled_tokens
            pure = bool(counts) and all(
                before[key][0] >= before[key][1]
                for key in counts
            )
            scheduled.update(tokens=sum(counts.values()), pure_decode=pure,
                             pure_prefill=bool(counts) and all(
                                 before[key][0] < before[key][1]
                                 for key in counts),
                             active=len(counts), preempted=len(output.preempted_req_ids or ()))
            return output

        scheduler.schedule = observe_schedule
        params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                ignore_eos=True, max_tokens=args.output_len, detokenize=False)
        for index in range(bs):
            engine.add_request(str(index), {"prompt_token_ids": [100] * length}, params)
        if window_mode:
            scheduler.schedule = original_schedule
            _run_decode_window(engine, core, bs, length, args, row, raw_outputs, steps, case_dir)
            return
        core.model_executor.collective_rpc(_synchronize_worker)
        full_steps = peak = preemptions = 0
        started = perf_counter()
        first_token_s = None
        while engine.has_unfinished_requests():
            scheduled.clear()
            begin = perf_counter()
            outputs = engine.step()
            core.model_executor.collective_rpc(_synchronize_worker)
            elapsed = perf_counter() - begin
            if not scheduled:
                raise RuntimeError("V1 step bypassed the inspected scheduler")
            preemptions += scheduled["preempted"]
            if preemptions:
                raise RuntimeError("Full decode batch capacity exceeded: scheduler preemption")
            if scheduled["pure_decode"]:
                peak = max(peak, scheduled["active"])
            full = scheduled["pure_decode"] and scheduled["active"] == bs
            if full:
                if scheduled["tokens"] != bs:
                    raise RuntimeError("Expected one computed decode token per request")
                full_steps += 1
            steps.append({**scheduled, "elapsed_s": elapsed,
                          "measured": full and full_steps > args.decode_warmup_steps_after_full})
            for output in outputs:
                if output.outputs and output.outputs[0].token_ids and first_token_s is None:
                    first_token_s = perf_counter() - started
                if output.finished:
                    tokens = list(output.outputs[0].token_ids)
                    raw_outputs.append({"request_id": output.request_id, "token_ids": tokens,
                                        "status": "success" if len(tokens) == args.output_len else "model_failed"})
                    if peak != bs:
                        raise RuntimeError("Full decode batch capacity exceeded: requests finished before full decode admission")
        if len(raw_outputs) != bs or any(item["status"] != "success" for item in raw_outputs):
            raise RuntimeError("Incomplete output: every request must finish the requested output length")
        measured = [step for step in steps if step["measured"]]
        if not measured:
            raise RuntimeError("No full-batch decode steps remain after warmup")
        elapsed = sum(step["elapsed_s"] for step in measured)
        tokens = sum(step["tokens"] for step in measured)
        throughput = stage_throughput(tokens, elapsed)
        prefill_steps = [step for step in steps if step["pure_prefill"]]
        prefill_throughput = stage_throughput(sum(step["tokens"] for step in prefill_steps),
                                            sum(step["elapsed_s"] for step in prefill_steps))
        if prefill_throughput is None:
            raise RuntimeError("No separately measured prefill work")
        peak_memory = max(core.model_executor.collective_rpc(_worker_peak_memory))
        row.update(status="SUCCESS", stage_metrics_status="success",
                   stage_timing_scope="sum_synchronized_llm_steps",
                   synchronization_boundary="LLMEngine.step plus all-worker CUDA synchronization RPC",
                   measurement_scope="full_batch_pure_decode_steps", require_full_decode_batch=True,
                   synchronize_step_timing=True, decode_stage_tokens=tokens,
                   decode_stage_elapsed_s=elapsed, decode_stage_throughput_tps=throughput,
                   actual_decode_peak=peak, completed_requests=len(raw_outputs),
                   full_admission_reached=peak == bs, scheduler_preemptions=preemptions,
                   measured_decode_steps_after_full=len(measured),
                   decode_warmup_steps_after_full=args.decode_warmup_steps_after_full,
                   decode_tp=throughput, prefill_tp=prefill_throughput, avg_bs=bs, mem=peak_memory,
                   ttft=first_token_s, itl=elapsed / len(measured) * 1000,
                   request_metrics_status="not_measured")
    except Exception as error:
        row.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if case_dir is not None:
            _write_jsonl(case_dir / "steps.jsonl", steps)
            _write_jsonl(case_dir / "raw_outputs.jsonl", raw_outputs)
        results_dict[(method, length, bs)] = row
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def _parse_positive_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return values


def _env_snapshot() -> dict[str, str]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "VLLM_ALL2ALL_BACKEND",
        "VLLM_USE_V1",
        "NCCL_DEBUG",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-token vLLM baseline compatible with microbench.py."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-sizes", type=_parse_positive_ints, default=[1, 2, 4])
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=1252)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--prompt-token-id", type=int, default=100)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "input_len",
        "output_len",
        "num_warmups",
        "num_iters",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.input_len + args.output_len > args.max_model_len:
        raise ValueError(
            "max_model_len must cover input_len + output_len: "
            f"{args.max_model_len} < {args.input_len + args.output_len}"
        )
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")


def main() -> int:
    args = _build_parser().parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = list(args.batch_sizes)
    engine_config = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "enable_expert_parallel": bool(args.enable_expert_parallel),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_seqs": max(batch_sizes),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "enable_prefix_caching": False,
        "language_model_only": True,
        "seed": 0,
        "disable_log_stats": True,
        "compilation_config": {
            "cudagraph_capture_sizes": batch_sizes,
            "max_cudagraph_capture_size": max(batch_sizes),
        },
    }
    run_info = {
        "benchmark": "vllm_microbench",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": shlex.join(sys.argv),
        "engine_config": engine_config,
        "input_len": int(args.input_len),
        "output_len": int(args.output_len),
        "batch_sizes": batch_sizes,
        "num_warmups": int(args.num_warmups),
        "num_iters": int(args.num_iters),
        "prompt_token_id": int(args.prompt_token_id),
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
        },
        "env": _env_snapshot(),
    }
    _write_json(output_dir / "run_info.json", run_info)

    performance_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    raw_output_rows: list[dict[str, Any]] = []
    llm = None
    try:
        import torch
        import transformers
        import vllm
        from vllm import LLM, SamplingParams

        run_info["versions"] = {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "flashinfer_python": version("flashinfer-python"),
        }
        _write_json(output_dir / "run_info.json", run_info)

        llm = LLM(**engine_config)
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=int(args.output_len),
            detokenize=False,
        )

        for batch_size in batch_sizes:
            prompts = [
                {"prompt_token_ids": [int(args.prompt_token_id)] * args.input_len}
                for _ in range(batch_size)
            ]
            for _ in range(args.num_warmups):
                warmup_outputs = llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                if len(warmup_outputs) != batch_size:
                    raise RuntimeError(
                        f"warmup returned {len(warmup_outputs)} requests, "
                        f"expected {batch_size}"
                    )

            latencies: list[float] = []
            for iteration in range(args.num_iters):
                started = perf_counter()
                outputs = llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                latency = perf_counter() - started
                latencies.append(latency)
                if len(outputs) != batch_size:
                    raise RuntimeError(
                        f"iteration {iteration} returned {len(outputs)} requests, "
                        f"expected {batch_size}"
                    )

                for sample_index, output in enumerate(outputs):
                    if len(output.outputs) != 1:
                        raise RuntimeError(
                            f"iteration {iteration} sample {sample_index} returned "
                            f"{len(output.outputs)} sequences, expected 1"
                        )
                    token_ids = list(output.outputs[0].token_ids)
                    status = (
                        "success"
                        if len(token_ids) == args.output_len
                        else "model_failed"
                    )
                    sample_row = {
                        "batch_size": batch_size,
                        "iteration": iteration,
                        "sample_index": sample_index,
                        "status": status,
                        "input_tokens": int(args.input_len),
                        "output_tokens": len(token_ids),
                    }
                    per_sample_rows.append(sample_row)
                    raw_output_rows.append({**sample_row, "token_ids": token_ids})
                    if status != "success":
                        raise RuntimeError(
                            f"iteration {iteration} sample {sample_index} produced "
                            f"{len(token_ids)} tokens, expected {args.output_len}"
                        )

            mean_latency = statistics.fmean(latencies)
            performance_rows.append(
                {
                    "batch_size": batch_size,
                    "status": "success",
                    "latencies_s": latencies,
                    "e2e_latency_s_mean": mean_latency,
                    "e2e_latency_s_median": statistics.median(latencies),
                    "input_tok_s": batch_size * args.input_len / mean_latency,
                    "output_tok_s": batch_size * args.output_len / mean_latency,
                    "total_tok_s": (
                        batch_size * (args.input_len + args.output_len) / mean_latency
                    ),
                }
            )

        aggregate = {
            "benchmark": "vllm_microbench",
            "status": "success",
            "num_cases": len(performance_rows),
            "records": performance_rows,
        }
    except Exception as error:
        aggregate = {
            "benchmark": "vllm_microbench",
            "status": "model_failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "records": performance_rows,
        }
        _write_jsonl(output_dir / "raw_outputs.jsonl", raw_output_rows)
        _write_jsonl(output_dir / "per_sample_results.jsonl", per_sample_rows)
        _write_jsonl(output_dir / "performance.jsonl", performance_rows)
        _write_json(output_dir / "aggregate_metrics.json", aggregate)
        raise
    finally:
        if llm is not None:
            del llm

    _write_jsonl(output_dir / "raw_outputs.jsonl", raw_output_rows)
    _write_jsonl(output_dir / "per_sample_results.jsonl", per_sample_rows)
    _write_jsonl(output_dir / "performance.jsonl", performance_rows)
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
