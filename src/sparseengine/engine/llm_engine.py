import atexit
import gc
import os
import pickle
import uuid
from collections import deque
from dataclasses import fields
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter
import threading
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig, Qwen2Tokenizer
import torch
import torch.multiprocessing as mp
import sparseengine.platforms as platforms
from sparseengine.utils.code_revision import code_revision_info
from sparseengine.utils.log import logger
import sys
import time

from sparseengine.configs.cuda_graph import build_decode_cuda_graph_startup_plan

from sparseengine.config import Config
from sparseengine.kernels.external.required import (
    validate_required_cuda_kernel_metadata,
)
from sparseengine.kernels.external.flashinfer.jit_cache import resolve_trtllm_cache_root
from sparseengine.method_registry import (
    SPARSE_AUXILIARY_PREFILL_PROMPTS,
    decode_graph_path_id,
)
from sparseengine.platforms.interface import PlatformEnum
from sparseengine.sampling_params import SamplingParams
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.scheduler import Scheduler
from sparseengine.engine.model_runner import ModelRunner, make_tp_shm_name, select_master_port
from sparseengine.engine.input_processor import tokenize_text_prompt
from sparseengine.engine.startup import (
    build_startup_capacity_decision,
    log_startup_capacity_decision,
    log_startup_completion,
    validate_production_kv_records,
)
from sparseengine.multimodal.inputs import (
    MultiModalInputProcessor,
    MultiModalPrompt,
    is_multimodal_prompt,
)
from sparseengine.engine.prefix_cache import PrefixCacheRoutingSnapshot
from sparseengine.engine.prefix_prune import (
    PrefixPruneJob,
    validate_prefix_prune_request,
)
from sparseengine.engine.chain_cache import (
    ChainCacheIndex,
    ChainRoutingSnapshot,
    ChainModeError,
    ChainNotFoundError,
    ChainOwnerMismatchError,
    ChainPrefixMismatchError,
    RequestAdmission,
    stable_token_digest,
)
from sparseengine.utils.profiler import profiler

def _moe_workspace_warmup_token_counts(config: Config) -> tuple[int, ...]:
    if config.model_spec.num_experts_field is None:
        return ()

    max_batched_tokens = int(config.max_num_batched_tokens)
    mlp_chunk_size = int(config.mlp_chunk_size)
    if max_batched_tokens <= 0 or mlp_chunk_size <= 0:
        raise ValueError(
            "MoE workspace warmup requires positive max_num_batched_tokens and "
            f"mlp_chunk_size, got {max_batched_tokens} and {mlp_chunk_size}."
        )

    token_replicas = int(config.data_parallel_size)
    max_moe_tokens = min(max_batched_tokens * token_replicas, mlp_chunk_size)
    decode_tokens = min(
        max_moe_tokens,
        max(1, int(config.max_decoding_seqs)) * token_replicas,
    )
    return tuple(dict.fromkeys((decode_tokens, max_moe_tokens)))


class _ThroughputIntervalLogger:
    def __init__(self, interval_s: float, rank: int = 0):
        self._interval_s = float(interval_s)
        self._rank = int(rank)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prefill_tokens = 0
        self._decode_tokens = 0
        self._prefill_steps = 0
        self._decode_batch_counts: dict[int, int] = {}
        self._running_seqs = 0
        self._prefill_seqs = 0
        self._decode_seqs = 0
        self._prefill_chunked_seqs = 0
        self._prefill_full_seqs = 0
        self._prefill_raw_offload_seqs = 0
        self._last_batch = "idle"
        self._last_report_t = perf_counter()

    def start(self):
        if self._interval_s <= 0:
            return
        if self._thread is not None:
            return
        with self._lock:
            self._last_report_t = perf_counter()
        self._thread = threading.Thread(target=self._run, name="sengine-throughput-logger", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=self._interval_s + 1.0)

    def record_step(self, num_tokens: int):
        if num_tokens == 0:
            return
        with self._lock:
            if num_tokens > 0:
                self._prefill_tokens += int(num_tokens)
                self._prefill_steps += 1
            else:
                self._decode_tokens += int(-num_tokens)
                batch_size = int(-num_tokens)
                self._decode_batch_counts[batch_size] = self._decode_batch_counts.get(batch_size, 0) + 1

    def record_state(
        self,
        running_seqs: int,
        prefill_seqs: int,
        decode_seqs: int,
        prefill_chunked_seqs: int,
        prefill_full_seqs: int,
        prefill_raw_offload_seqs: int,
        last_batch: str,
    ):
        with self._lock:
            self._running_seqs = int(running_seqs)
            self._prefill_seqs = int(prefill_seqs)
            self._decode_seqs = int(decode_seqs)
            self._prefill_chunked_seqs = int(prefill_chunked_seqs)
            self._prefill_full_seqs = int(prefill_full_seqs)
            self._prefill_raw_offload_seqs = int(prefill_raw_offload_seqs)
            self._last_batch = str(last_batch)

    def _run(self):
        while not self._stop.wait(self._interval_s):
            now = perf_counter()
            with self._lock:
                prefill_tokens = self._prefill_tokens
                decode_tokens = self._decode_tokens
                prefill_steps = self._prefill_steps
                decode_batch_counts = self._decode_batch_counts
                self._prefill_steps = 0
                self._decode_batch_counts = {}
                running_seqs = self._running_seqs
                prefill_seqs = self._prefill_seqs
                decode_seqs = self._decode_seqs
                prefill_chunked_seqs = self._prefill_chunked_seqs
                prefill_full_seqs = self._prefill_full_seqs
                prefill_raw_offload_seqs = self._prefill_raw_offload_seqs
                last_batch = self._last_batch
                self._prefill_tokens = 0
                self._decode_tokens = 0
                last_t = self._last_report_t
                self._last_report_t = now

            dt = max(now - last_t, 1e-9)
            prefill_tp = prefill_tokens / dt
            decode_tp = decode_tokens / dt
            logger.info(
                "Avg TP (last {dt:.1f}s): dp_rank={rank} prefill_tp={prefill_tp:.0f} tok/s, decode_tp={decode_tp:.0f} tok/s "
                "| seq(run/prf/dc)={running_seqs}/{prefill_seqs}/{decode_seqs} "
                "| prf(chunked/full/raw_offload)={prefill_chunked_seqs}/{prefill_full_seqs}/{prefill_raw_offload_seqs} "
                "| last_batch={last_batch} "
                "| prefill_steps={prefill_steps} decode_batch_steps={decode_batch_counts} "
                "(prefill_tokens={prefill_tokens}, decode_tokens={decode_tokens})",
                dt=dt,
                rank=self._rank,
                prefill_tokens=prefill_tokens,
                prefill_tp=prefill_tp,
                decode_tokens=decode_tokens,
                decode_tp=decode_tp,
                running_seqs=running_seqs,
                prefill_seqs=prefill_seqs,
                decode_seqs=decode_seqs,
                prefill_chunked_seqs=prefill_chunked_seqs,
                prefill_full_seqs=prefill_full_seqs,
                prefill_raw_offload_seqs=prefill_raw_offload_seqs,
                last_batch=last_batch,
                prefill_steps=prefill_steps,
                decode_batch_counts=dict(sorted(decode_batch_counts.items())),
            )

def _resolve_eos_token_ids(model_path, hf_config, tokenizer_eos_token_id):
    if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "generation_config.json")):
        # A generation config is optional in HF checkpoints. GLM FP8 publishes
        # its complete EOS list in config.json instead; preserve every stop ID.
        eos_values = getattr(hf_config, "eos_token_id", None)
        logger.info("No generation_config.json in {}; use model config EOS metadata.", model_path)
    else:
        eos_values = GenerationConfig.from_pretrained(model_path).eos_token_id
    if eos_values is None:
        eos_values = []
    elif isinstance(eos_values, int):
        eos_values = [eos_values]
    else:
        eos_values = list(eos_values)
    if tokenizer_eos_token_id is not None:
        eos_values.append(int(tokenizer_eos_token_id))
    return tuple(dict.fromkeys(int(token_id) for token_id in eos_values))


class LLMEngine:
    """
    SparseEngine 推理引擎的核心入口类。
    负责协调 Tokenizer、调度器 (Scheduler) 和模型执行器 (ModelRunner)。
    管理多进程张量并行 (Tensor Parallelism) 的生命周期。
    """

    def __new__(cls, model, **kwargs):
        config_fields = {field.name for field in fields(Config) if field.init}
        ignored_keys = sorted(set(kwargs) - config_fields - {"_dp_worker"})
        if ignored_keys:
            raise ValueError(
                f"Unknown SparseEngine config keys: {ignored_keys}. "
                "Runtime parameter aliases and unknown keys are not accepted."
            )
        if int(kwargs.get("data_parallel_size", 1)) > 1 and kwargs.get("_dp_worker") is None:
            from sparseengine.engine.dp_engine import DPAttentionEngine
            return DPAttentionEngine(model, **kwargs)
        return super().__new__(cls)

    def __init__(self, model, *, _dp_worker=None, **kwargs):
        # 1. 初始化配置
        config = Config(model, **kwargs)
        self.config = config
        trtllm_cache_root = None
        if platforms.get_current_platform().enum is PlatformEnum.CUDA:
            validate_required_cuda_kernel_metadata()
            # Pass the original root explicitly: later engines spawn children
            # after rank zero has already installed its process-local env path.
            trtllm_cache_root = str(resolve_trtllm_cache_root())
        
        # 初始化 Profiler
        profiler.set_enabled(config.enable_profiler)
        
        # 2. 启动 world worker 进程；TP/EP/DP 语义由 ParallelContext 管理。
        master_port = select_master_port() if _dp_worker is None else _dp_worker[1]
        logger.info("Using distributed master port: {}", master_port)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        replica_rank = 0 if _dp_worker is None else _dp_worker[0]
        leader_rank = replica_rank * config.attn_tp_size
        tp_shm_name = make_tp_shm_name() if config.attn_tp_size > 1 else None
        for i in range(1, config.attn_tp_size):
            event = (ctx.Event(), ctx.Event())
            # 为每一个非零 Rank 启动一个独立的 ModelRunner 进程
            process = ctx.Process(
                target=ModelRunner,
                args=(config, leader_rank + i, event, tp_shm_name, master_port, trtllm_cache_root),
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # 3. 初始化主进程的 ModelRunner (Rank 0)
        # 注意：必须先初始化 ModelRunner 以便在本地 GPU 分配 KV Cache 账本
        self.model_runner = ModelRunner(
            config, leader_rank, self.events,
            tp_shm_name, master_port, trtllm_cache_root,
        )
        
        # 加载分词器
        self.tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.multimodal_processor = (
            MultiModalInputProcessor(config.model)
            if config.enable_multimodal
            and callable(getattr(self.model_runner.model, "encode_multimodal", None))
            else None
        )
        config.eos_token_ids = _resolve_eos_token_ids(
            config.model, config.hf_config, self.tokenizer.eos_token_id
        )
        config.eos = config.eos_token_ids[0] if config.eos_token_ids else -1
        auxiliary_prompt = SPARSE_AUXILIARY_PREFILL_PROMPTS.get(config.sparse_method)
        self.model_runner.call(
            "set_tokenizer_metadata",
            self._build_delimiter_token_ids(self.tokenizer),
            self._build_non_execution_token_ids(self.tokenizer),
            self.tokenizer.encode(
                auxiliary_prompt, add_special_tokens=False,
            ) if auxiliary_prompt else [],
        )
        
        # 4. 初始化调度器
        # 关键设计：将 Rank 0 的 CacheManager 传给 Scheduler。
        # Scheduler 通过它来感知全局显存的余量，从而做出调度和抢占决策。
        self.scheduler = self._create_scheduler()
        
        self._exited = False
        self._throughput_logger = _ThroughputIntervalLogger(config.throughput_log_interval_s, rank=replica_rank)
        self.last_step_token_outputs: list[tuple[int, list[int]]] = []
        self.last_step_prompt_cache_hits: list[tuple[int, int]] = []
        self.last_step_logprob_outputs: list[
            tuple[int, list[float | None], list[dict[int, float] | None]]
        ] = []
        self._active_chain_sequences: dict[int, Sequence] = {}
        self._prefix_prune_jobs: dict[str, PrefixPruneJob] = {}
        self._pending_prefix_prune_ids: deque[str] = deque()
        # 注册退出钩子，确保程序崩溃或结束时能正确释放多进程资源
        self._atexit_callback = self.exit
        atexit.register(self._atexit_callback)

        # Startup requests only exercise kernels and capacity profiles. They
        # must not consume persistent ChainCache identities across warmup
        # batches or CUDA Graph families.
        self._startup_warmup_active = True
        try:
            self._warmup()
        finally:
            self._startup_warmup_active = False
        self.model_runner.call("arm_runtime_compilation_guard")
        if os.getenv("SPARSEENGINE_PROFILER_RESET_AFTER_WARMUP", "0") == "1":
            profiler.reset()
        self._throughput_logger.start()
        if config.async_scheduling:
            from sparseengine.engine.async_scheduling.scheduler import AsyncScheduler
            self._async_scheduler = AsyncScheduler(self)

    @staticmethod
    def _build_delimiter_token_ids(tokenizer) -> list[int]:
        # Match SkipKV's official newline-oriented split set. Plain "." or "?"
        # would trigger steering far more often than the paper implementation.
        delimiter_texts = [
            "\n",
            ".\n",
            ")\n",
            "\n\n",
            ".\n\n",
            ")\n\n",
            "?\n\n",
        ]
        token_ids: set[int] = set()
        for text in delimiter_texts:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                ids = []
            if ids:
                token_ids.add(int(ids[-1]))
        return sorted(token_ids)

    @staticmethod
    def _build_non_execution_token_ids(tokenizer) -> list[int]:
        marker_texts = [
            "Alternatively",
            "Wait",
            "again",
        ]
        token_ids: set[int] = set()
        for text in marker_texts:
            candidates = {text, " " + text, text.lower(), " " + text.lower()}
            for candidate in candidates:
                try:
                    ids = tokenizer.encode(candidate, add_special_tokens=False)
                except Exception:
                    ids = []
                if ids:
                    token_ids.add(int(ids[-1]))
        return sorted(token_ids)

    def _create_scheduler(self) -> Scheduler:
        return Scheduler(
            self.config,
            self.model_runner.runtime_state,
            prefix_cache_hit_refresher=(
                self._refresh_prefix_cache_hit
                if self.config.enable_prefix_caching
                else None
            ),
            prefix_cache_hits_refresher=(
                self._refresh_prefix_cache_hits
                if self.config.enable_prefix_caching
                else None
            ),
            decode_capacity_reclaimer=self._reclaim_idle_chains_for_decode,
        )

    def _reclaim_idle_chains_for_decode(self, failure: Sequence) -> Sequence | None:
        runtime = self.model_runner.runtime_state
        coordinator = runtime.chain_cache_coordinator
        if coordinator is None:
            return failure
        # Only the driver chooses victims. Complete the same physical release
        # on every rank before retrying the scheduler's reservation decision.
        for record in coordinator.index.idle_resident_lru():
            demote = coordinator.offload is not None and record.seq_id in coordinator.offload.snapshots
            self.model_runner.call("chain_reclaim_idle", record.chain_id, int(record.seq_id), demote)
            logger.info(
                "Reclaimed IDLE chain for decode capacity: chain_id={} seq_id={} demoted={} free_slots={}",
                record.chain_id, record.seq_id, demote, runtime.num_free_slots,
            )
            failure = runtime.reserve_decode_windows(self.scheduler.decoding, self.scheduler.waiting)
            if failure is None:
                break
        return failure

    def _run_startup_batch(
        self,
        prompt_lengths: tuple[int, ...],
        sampling_params: SamplingParams,
        prompt_offset: int,
    ) -> int:
        vocab_size = int(self.config.hf_config.vocab_size)
        if prompt_offset + len(prompt_lengths) > vocab_size:
            raise ValueError(
                "Startup requires one distinct leading token per dummy prompt: "
                f"end={prompt_offset + len(prompt_lengths)} vocab_size={vocab_size}."
            )
        for request_idx, prompt_len in enumerate(prompt_lengths):
            dummy_prompt = [prompt_offset + request_idx] + [0] * (int(prompt_len) - 1)
            self.add_request(dummy_prompt, sampling_params)
        while not self.is_finished():
            self.step()
        return prompt_offset + len(prompt_lengths)

    def _prepare_startup_capture_batch(
        self,
        sampling_params: SamplingParams,
        prompt_offset: int,
        *,
        batch_size: int,
        prompt_len: int,
        sequential_prefill: bool = False,
    ) -> tuple[list[Sequence] | None, int]:
        seq_ids = []
        parked: list[Sequence] = []
        prepared = False

        def park_prefilled() -> None:
            while self.scheduler.waiting:
                self.step()
                while self.scheduler.decoding:
                    parked.append(self.scheduler.decoding.popleft())
            while self.scheduler.decoding:
                parked.append(self.scheduler.decoding.popleft())

        try:
            for request_idx in range(batch_size):
                if sequential_prefill:
                    records = self.model_runner.call(
                        "startup_capture_prefill_fits", prompt_len,
                    )
                    if not all(record["fits"] for record in records):
                        return None, prompt_offset + len(seq_ids)
                dummy_prompt = [prompt_offset + request_idx] + [0] * (prompt_len - 1)
                seq_ids.append(self.add_request(dummy_prompt, sampling_params))
                if sequential_prefill:
                    # Park after final-prefill compaction. The next admission
                    # sees actual retained KV, not a sum of all prompt peaks.
                    park_prefilled()
            if not sequential_prefill:
                park_prefilled()
            if len(parked) != batch_size:
                raise RuntimeError(
                    "Startup decode CUDA Graph prefill did not park the requested "
                    f"batch: expected={batch_size}, actual={len(parked)}."
                )
            if {int(seq.seq_id) for seq in parked} != set(seq_ids):
                raise RuntimeError("Startup decode CUDA Graph prefill parked unexpected sequences.")
            # Capture prepares one decode append for every parked request.
            # Serial prefill can fit even when that full-batch peak cannot.
            records = self.model_runner.call("startup_capture_decode_fits", parked)
            if not all(record["fits"] for record in records):
                logger.info(
                    "Startup CUDA Graph batch exceeds decode append capacity: "
                    "batch={} rank_checks={}.", batch_size, records,
                )
                return None, prompt_offset + len(seq_ids)
            prepared = True
            return parked, prompt_offset + batch_size
        finally:
            if not prepared:
                self.scheduler.decoding.extend(parked)
                for seq_id in seq_ids:
                    self.abort_request(int(seq_id))

    def _capture_startup_decode_graphs(
        self,
        prompt_offset: int,
        *,
        respect_runtime_capacity: bool = False,
    ) -> int:
        if not bool(getattr(self.config, "decode_graph_startup_capture", False)):
            return prompt_offset
        startup_plan = build_decode_cuda_graph_startup_plan(self.config)
        sequential_plan = set()
        if respect_runtime_capacity:
            plan_records = self.model_runner.call(
                "resolve_startup_decode_graph_plan",
                startup_plan,
            )
            feasible_by_rank = [
                {tuple(entry) for entry in record["feasible"]}
                for record in plan_records
            ]
            sequential_plan = {
                entry
                for entry in startup_plan
                if not all(tuple(entry) in feasible for feasible in feasible_by_rank)
            }
        if not startup_plan:
            raise RuntimeError(
                "Production KV capacity cannot capture any configured decode CUDA Graph."
            )

        required_prompts = sum(batch_size for batch_size, _ in startup_plan)
        if prompt_offset + required_prompts > int(self.config.hf_config.vocab_size):
            raise ValueError(
                "Startup CUDA Graph capture requires distinct leading tokens: "
                f"end={prompt_offset + required_prompts} "
                f"vocab_size={self.config.hf_config.vocab_size}."
            )

        self.model_runner.call("begin_decode_cuda_graph_capture")
        logger.info(
            "Startup CUDA Graph capture: graphs={} sequential_prefill_candidates={}.",
            len(startup_plan), len(sequential_plan),
        )
        logger.debug("Startup CUDA Graph capture plan: {}.", startup_plan)
        capture_params = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
        skipped_plan = []
        for entry in startup_plan:
            batch_size, _ = entry
            prompt_len = 1
            parked, prompt_offset = self._prepare_startup_capture_batch(
                capture_params,
                prompt_offset,
                batch_size=batch_size,
                prompt_len=prompt_len,
                sequential_prefill=entry in sequential_plan,
            )
            if parked is None:
                skipped_plan.append(entry)
                logger.info(
                    "Startup CUDA Graph family exceeds KV capacity during "
                    "prefill or decode preparation: batch={} path={!r}.",
                    batch_size, decode_graph_path_id(str(self.config.sparse_method or "")),
                )
                continue
            try:
                self.model_runner.call("capture_decode_cuda_graph_warmup", parked)
            finally:
                self.scheduler.decoding.extend(parked)
                for seq in parked:
                    self.abort_request(int(seq.seq_id))

        graph_runner = self.model_runner.decode_graph_runner
        captured = {
            (
                int(key.batch_size),
                int(state.capture_context_capacity),
                key.graph_path_id,
            )
            for key, state in graph_runner._graphs.items()
            if state.graph is not None
            and key.method == str(self.config.sparse_method or "")
            and not key.capture_sampling
        }
        method = str(self.config.sparse_method or "")
        expected = {
            (batch_size, context_capacity, decode_graph_path_id(method))
            for batch_size, context_capacity in startup_plan
            if (batch_size, context_capacity) not in skipped_plan
        }
        if not expected:
            raise RuntimeError(
                "Production KV capacity cannot capture any configured decode CUDA Graph."
            )
        missing = sorted(expected - captured)
        if missing:
            raise RuntimeError(
                "Startup CUDA Graph capture did not materialize its plan: "
                f"missing={missing}."
            )
        self.model_runner.call("collect_decode_cuda_graph_metadata")
        self.model_runner.call("exchange_decode_cuda_graph_metadata")
        self.model_runner.call("register_decode_cuda_graph_buffers")
        self.model_runner.call("seal_decode_cuda_graph_startup_plan")
        logger.info(
            "Startup CUDA Graph capture complete: cached={} capture_count={} "
            "replay_count={} skipped_for_kv_capacity={}.",
            len(captured),
            graph_runner.capture_count,
            graph_runner.replay_count,
            len(skipped_plan),
        )
        return prompt_offset

    def _warmup(self):
        logger.info("Startup profiling begins with a temporary KV runtime.")
        prompt_offset = 0
        compile_prompt_len = min(
            128,
            int(self.config.engine_prefill_chunk_size),
            int(self.config.max_model_len) - 1,
        )
        prompt_offset = self._run_startup_batch(
            (compile_prompt_len,),
            SamplingParams(max_tokens=1, temperature=0.0),
            prompt_offset,
        )
        self._after_warmup_debug_cleanup()
        self._warmup_moe_workspaces()
        self._after_warmup_debug_cleanup()

        prefill_records = self.model_runner.call("profile_startup_prefill")

        logger.info("Startup profile phase=cuda_graph.")
        self.model_runner.call("begin_startup_memory_profile", "cuda_graph")
        prompt_offset = self._capture_startup_decode_graphs(
            prompt_offset, respect_runtime_capacity=True,
        )
        self._after_warmup_debug_cleanup()
        graph_records = self.model_runner.call(
            "finish_startup_memory_profile",
            "cuda_graph",
        )

        decode_batch = int(self.config.max_decoding_seqs)
        logger.info("Startup profile phase=decode batch={}.", decode_batch)
        self.model_runner.call("begin_startup_memory_profile", "decode")
        prompt_offset = self._run_startup_batch(
            (1,) * decode_batch,
            SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True),
            prompt_offset,
        )
        self._after_warmup_debug_cleanup()
        decode_records = self.model_runner.call(
            "finish_startup_memory_profile",
            "decode",
        )

        self.scheduler = None
        persistent_records = self.model_runner.call("release_profiling_cache_runtime")
        decision = build_startup_capacity_decision(
            prefill_records=prefill_records,
            graph_records=graph_records,
            decode_records=decode_records,
            persistent_records=persistent_records,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        log_startup_capacity_decision(decision)

        production_records = self.model_runner.call(
            "build_production_cache_runtime",
            decision.selected_kv_budget_bytes,
        )
        validate_production_kv_records(production_records)
        self.scheduler = self._create_scheduler()
        prompt_offset = self._capture_startup_decode_graphs(
            prompt_offset=0,
            respect_runtime_capacity=True,
        )
        self._after_warmup_debug_cleanup()

        post_capture_batch = int(self.config.max_decoding_seqs)
        logger.info(
            "Startup post-capture warmup: batch={} max_tokens=2.",
            post_capture_batch,
        )
        self._run_startup_batch(
            (1,) * post_capture_batch,
            SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True),
            prompt_offset,
        )
        self._after_warmup_debug_cleanup()
        final_records = self.model_runner.call("capture_startup_memory_snapshot")
        log_startup_completion(production_records, final_records, decision)
        self.model_runner.call("log_operator_implementations")
        logger.info("Startup completed; production runtime is ready.")

    def _warmup_moe_workspaces(self) -> None:
        token_counts = _moe_workspace_warmup_token_counts(self.config)
        if not token_counts:
            return
        logger.info(
            "Startup persistent MoE workspace warmup: token_counts={}.",
            token_counts,
        )
        for num_tokens in token_counts:
            self.model_runner.call("warmup_moe_workspace", num_tokens)

    def _after_warmup_debug_cleanup(self):
        self.model_runner.call("reset_after_warmup")

    @staticmethod
    def _cleanup_model_runner_shared_memory(model_runner):
        shm = getattr(model_runner, "shm", None)
        if shm is None:
            return
        try:
            shm.close()
        except Exception as exc:
            logger.warning("Failed to close ModelRunner shared memory during shutdown: {}", repr(exc))
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Failed to unlink ModelRunner shared memory during shutdown: {}", repr(exc))

    def exit(self):
        """优雅地退出所有子进程并清理共享内存"""
        atexit_callback = getattr(self, "_atexit_callback", None)
        if atexit_callback is not None:
            atexit.unregister(atexit_callback)
            del self._atexit_callback
        if self._exited:
            return
        self._exited = True

        profiler.print_stats()
        if hasattr(self, "_throughput_logger"):
            self._throughput_logger.stop()
        runner_exit_completed, runner_platform = self._shutdown_runtime()
        if runner_exit_completed:
            # Collect only after _shutdown_runtime() returns. Its worker-thread
            # closure temporarily owns ModelRunner, so collecting inside that
            # frame can leave cyclic model/cache objects alive until exit().
            gc.collect()
            if runner_platform is not None:
                runner_platform.empty_cache()

    def _shutdown_runtime(self):
        """Stop the runner/workers and drop engine-owned runtime references."""
        runner_exit_completed = True
        runner_platform = None
        if hasattr(self, "model_runner"):
            model_runner = self.model_runner
            runner_platform = getattr(model_runner, "platform", None)
            timeout_s = float(os.getenv("SPARSEENGINE_ENGINE_EXIT_TIMEOUT_S", "10"))
            errors: list[BaseException] = []

            def call_model_runner_exit():
                try:
                    model_runner.call("exit")
                except BaseException as exc:  # pragma: no cover - surfaced by warning below.
                    errors.append(exc)

            exit_thread = threading.Thread(
                target=call_model_runner_exit,
                name="sparseengine-engine-exit",
                daemon=True,
            )
            exit_thread.start()
            exit_thread.join(timeout=max(0.0, timeout_s))
            if exit_thread.is_alive():
                runner_exit_completed = False
                logger.warning(
                    "Timed out waiting {:.1f}s for ModelRunner exit RPC; terminating workers.",
                    timeout_s,
                )
                self._cleanup_model_runner_shared_memory(model_runner)
            elif errors:
                logger.warning("ModelRunner exit RPC failed during shutdown: {}", repr(errors[0]))
                self._cleanup_model_runner_shared_memory(model_runner)
            errors.clear()
            del self.model_runner
        if hasattr(self, "scheduler"):
            del self.scheduler
        if hasattr(self, "ps"):
            join_timeout_s = float(os.getenv("SPARSEENGINE_WORKER_JOIN_TIMEOUT_S", "5"))
            for p in self.ps:
                # The exit RPC has already asked each worker to leave its loop.
                # Give it time to release distributed/Event resources before using
                # terminate(), which can leave multiprocessing semaphores registered.
                p.join(timeout=max(0.0, join_timeout_s))
                if p.is_alive():
                    logger.warning(
                        "Worker process pid={} did not stop after the exit RPC; terminating.",
                        p.pid,
                    )
                    p.terminate()
                    p.join(timeout=max(0.0, join_timeout_s))
                if p.is_alive():
                    logger.warning(
                        "Worker process pid={} did not stop after terminate; killing.",
                        p.pid,
                    )
                    p.kill()
                    p.join(timeout=max(0.0, join_timeout_s))
                close = getattr(p, "close", None)
                if callable(close) and not p.is_alive():
                    close()
        if hasattr(self, "events"):
            self.events.clear()
        return runner_exit_completed, runner_platform

    def _tokenize_prompt(self, prompt: str | list[int]) -> list[int]:
        tokenizer = self.tokenizer if isinstance(prompt, str) else None
        return tokenize_text_prompt(tokenizer, prompt)

    def admit_request(
        self,
        prompt: str | list[int] | MultiModalPrompt | dict,
        sampling_params: SamplingParams,
        chain_id: str | None = None,
        chain_append_only: bool = False,
    ) -> RequestAdmission:
        """Validate and synchronously admit one request.

        In chain mode the returned seq_id is the resident sequence identity and
        remains stable across turns. The caller's request identity is separate.
        """
        multimodal = None
        if is_multimodal_prompt(prompt):
            if self.multimodal_processor is None:
                raise NotImplementedError(
                    "Multimodal input is disabled or unsupported by this model."
                )
            if chain_id or chain_append_only:
                raise ChainModeError("Multimodal requests do not support chain mode.")
            multimodal = self.multimodal_processor.process(prompt)
            prompt = multimodal.token_ids
        mode = (
            "disabled"
            if bool(getattr(self, "_startup_warmup_active", False))
            else str(
                getattr(self.config, "resolved_prefix_cache_mode", "disabled")
            )
        )
        normalized_chain_id = str(chain_id or "").strip()
        existing = None
        if mode == "chain" and normalized_chain_id:
            coordinator = (
                self.model_runner.runtime_state.chain_cache_coordinator
            )
            if coordinator is None:
                raise RuntimeError(
                    "Config resolved chain prefix caching but the runtime has no "
                    "ChainCacheCoordinator."
                )
            existing = coordinator.index.records.get(normalized_chain_id)
            if existing is None:
                coordinator.index.lookup(normalized_chain_id)
        if chain_append_only:
            if existing is None:
                raise ChainNotFoundError(
                    "chain_append_only requires an existing chain.",
                    chain_id=normalized_chain_id or None,
                )
            suffix_token_ids = (
                [
                    int(token_id)
                    for token_id in self.tokenizer.encode(
                        prompt,
                        add_special_tokens=False,
                    )
                ]
                if isinstance(prompt, str)
                else [int(token_id) for token_id in prompt]
            )
            if not suffix_token_ids:
                raise ChainPrefixMismatchError(
                    "A chain append must contain at least one suffix token.",
                    chain_id=normalized_chain_id,
                )
            prompt = [
                int(token_id) for token_id in existing.token_ids
            ] + suffix_token_ids
        else:
            prompt = self._tokenize_prompt(prompt)
        prompt_len = len(prompt)
        max_tokens = sampling_params.max_tokens
        if prompt_len + max_tokens > self.config.max_model_len:
            raise ValueError(
                "Prompt length + max_tokens exceeds max_model_len: "
                f"{prompt_len} + {max_tokens} > {self.config.max_model_len}. "
                "Reduce prompt/decoding length or increase max_model_len if the model supports it."
            )
        logger.debug(f'add prompt with {len(prompt)} tokens.')
        seq = Sequence(prompt, sampling_params)
        if multimodal is not None:
            seq.multimodal_digest = multimodal.digest
            seq.multimodal_full_prefill = (
                getattr(self.config.hf_config, "use_bidirectional_attention", None)
                == "vision"
            )
            payload = pickle.dumps(multimodal.tensors, protocol=pickle.HIGHEST_PROTOCOL)
            payload_shm = SharedMemory(create=True, size=len(payload))
            try:
                payload_shm.buf[: len(payload)] = payload
                seq.multimodal_position_delta = int(
                    self.model_runner.call(
                        "register_multimodal_shared",
                        int(seq.seq_id),
                        list(prompt),
                        payload_shm.name,
                        len(payload),
                    )
                )
            except Exception as register_error:
                try:
                    self.model_runner.call("free_multimodal", int(seq.seq_id))
                except Exception as cleanup_error:
                    logger.error(
                        "Failed to roll back multimodal seq_id={} after registration "
                        "error {}: {}",
                        seq.seq_id,
                        type(register_error).__name__,
                        cleanup_error,
                    )
                raise
            finally:
                payload_shm.close()
                payload_shm.unlink()
        if mode != "chain":
            if normalized_chain_id:
                raise ChainModeError(
                    "chain_id requires enable_prefix_caching=True with "
                    "prefix_cache_mode='chain'.",
                    chain_id=normalized_chain_id,
                )
            try:
                self.scheduler.add(seq)
            except Exception:
                if multimodal is not None:
                    self.model_runner.call("free_multimodal", int(seq.seq_id))
                raise
            return RequestAdmission(
                seq_id=int(seq.seq_id),
                chain_id=None,
                chain_status="disabled",
                reused_tokens=0,
                prefilled_tokens=prompt_len,
                prompt_token_ids=list(prompt),
            )

        coordinator = self.model_runner.runtime_state.chain_cache_coordinator
        if coordinator is None:
            raise RuntimeError(
                "Config resolved chain prefix caching but the runtime has no "
                "ChainCacheCoordinator."
            )
        created = not normalized_chain_id
        if created:
            normalized_chain_id = ChainCacheIndex.new_chain_id()
        if existing is not None:
            seq.seq_id = int(existing.seq_id)
        recreated = False
        try:
            plan = self.model_runner.runtime_state.chain_admission_plan(
                normalized_chain_id,
                int(seq.seq_id),
                prompt,
            )
        except ChainPrefixMismatchError as exc:
            if existing is None or chain_append_only:
                raise
            replaced_chain_id = normalized_chain_id
            replaced_seq_id = int(existing.seq_id)
            self.model_runner.call(
                "chain_invalidate",
                replaced_chain_id,
                replaced_seq_id,
            )
            normalized_chain_id = ChainCacheIndex.new_chain_id()
            logger.warning(
                "Recreating chain after strict token-prefix mismatch: "
                "old_chain_id={} new_chain_id={} input_tokens={} reason={}",
                replaced_chain_id,
                normalized_chain_id,
                prompt_len,
                str(exc),
            )
            plan = self.model_runner.runtime_state.chain_admission_plan(
                normalized_chain_id,
                int(seq.seq_id),
                prompt,
            )
            recreated = True
        if plan.status == "resumed" and prompt_len <= int(plan.reused_tokens):
            raise ChainPrefixMismatchError(
                "A resumed chain request must include at least one suffix token "
                "beyond the processed boundary.",
                chain_id=normalized_chain_id,
            )
        self.model_runner.call(
            "chain_validate_admission_plan",
            plan,
            prompt_len,
            # Planning already checked the exact input prefix against this
            # digest. Other ranks still validate it against their own records.
            coordinator.index.lookup(normalized_chain_id).processed_token_digest
            if plan.status == "resumed" else stable_token_digest((), count=0),
        )
        self.model_runner.call("chain_apply_admission", plan)
        chain_status = "recreated" if recreated else str(plan.status)
        seq.chain_id = normalized_chain_id
        seq.chain_status = chain_status
        seq.chain_reused_tokens = int(plan.reused_tokens)
        seq.num_prefilled_tokens = int(plan.reused_tokens)
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = int(plan.reused_tokens)
        seq.prefix_cache_method = str(
            getattr(
                self.config,
                "resolved_cache_sparse_method",
                self.config.sparse_method,
            )
            or ""
        )
        try:
            self.scheduler.add(seq)
        except Exception:
            self.model_runner.call(
                "chain_invalidate",
                normalized_chain_id,
                int(seq.seq_id),
            )
            raise
        self._active_chain_sequences[int(seq.seq_id)] = seq
        return RequestAdmission(
            seq_id=int(seq.seq_id),
            chain_id=normalized_chain_id,
            chain_status=chain_status,
            reused_tokens=int(plan.reused_tokens),
            prefilled_tokens=prompt_len - int(plan.reused_tokens),
            prompt_token_ids=list(prompt),
        )

    def add_request(
        self,
        prompt: str | list[int] | MultiModalPrompt | dict,
        sampling_params: SamplingParams,
    ):
        """Backward-compatible request API returning only seq_id."""
        return self.admit_request(prompt, sampling_params).seq_id

    def _refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        self.model_runner.call("refresh_prefix_cache_hit", seq)

    def _refresh_prefix_cache_hits(self, seqs: list[Sequence]) -> None:
        if seqs:
            self.model_runner.call("refresh_prefix_cache_hits", seqs)

    def abort_request(self, seq_id: int, disposition: str = "invalidate"):
        """Abort a queued or running request and release any owned KV slots."""
        disposition = str(disposition)
        if disposition != "invalidate":
            raise ValueError(
                "abort_request only supports 'invalidate'; interrupted chain "
                "state cannot be retained safely, got "
                f"{disposition!r}."
            )
        if hasattr(self, "_async_scheduler"):
            self._async_scheduler.abort(int(seq_id))
            return
        chain_seq = self._active_chain_sequences.get(int(seq_id))
        multimodal = any(
            seq.seq_id == seq_id and seq.multimodal_digest is not None
            for queue in (
                getattr(self.scheduler, "waiting", ()),
                getattr(self.scheduler, "decoding", ()),
            )
            for seq in queue
        )
        should_free = self.scheduler.abort(seq_id)
        if chain_seq is not None:
            self._active_chain_sequences.pop(int(seq_id), None)
            self.model_runner.call(
                "chain_invalidate",
                str(chain_seq.chain_id),
                int(chain_seq.seq_id),
            )
            return
        coordinator = (
            self.model_runner.runtime_state.chain_cache_coordinator
        )
        if coordinator is not None:
            chain_id = coordinator.index.seq_id_to_chain_id.get(
                int(seq_id)
            )
            if chain_id is not None:
                self.model_runner.call(
                    "chain_invalidate",
                    str(chain_id),
                    int(seq_id),
                )
                return
        if should_free:
            self.model_runner.call("free_slots", seq_id)
        elif multimodal:
            self.model_runner.call("free_multimodal", seq_id)

    def chain_cache_routing_match(self, chain_id: str) -> dict[str, object]:
        return self.model_runner.runtime_state.chain_routing_match(
            str(chain_id)
        )

    def invalidate_chain(self, chain_id: str) -> None:
        coordinator = (
            self.model_runner.runtime_state.chain_cache_coordinator
        )
        if coordinator is None:
            raise ChainModeError(
                "Chain prefix cache is not enabled.",
                chain_id=str(chain_id),
            )
        record = coordinator.index.lookup(str(chain_id))
        self._active_chain_sequences.pop(int(record.seq_id), None)
        self.model_runner.call(
            "chain_invalidate",
            str(chain_id),
            int(record.seq_id),
        )

    def discard_chain(
        self,
        chain_id: str,
        *,
        expected_seq_id: int,
    ) -> bool:
        coordinator = (
            self.model_runner.runtime_state.chain_cache_coordinator
        )
        if coordinator is None:
            raise ChainModeError(
                "Chain prefix cache is not enabled.",
                chain_id=str(chain_id),
            )
        record = coordinator.index.records.get(str(chain_id))
        if record is None:
            return False
        if int(record.seq_id) != int(expected_seq_id):
            raise ChainOwnerMismatchError(
                f"Chain owner mismatch for {chain_id!r}: "
                f"resident_seq_id={record.seq_id}, "
                f"expected_seq_id={int(expected_seq_id)}.",
                chain_id=str(chain_id),
            )
        seq_id = int(record.seq_id)
        self.scheduler.abort(seq_id)
        self._active_chain_sequences.pop(seq_id, None)
        self.model_runner.call(
            "chain_invalidate",
            str(chain_id),
            seq_id,
        )
        return True

    def chain_cache_routing_snapshot(self) -> ChainRoutingSnapshot:
        coordinator = (
            self.model_runner.runtime_state.chain_cache_coordinator
        )
        if coordinator is None:
            return ChainRoutingSnapshot(enabled=False)
        return coordinator.index.routing_snapshot()

    def prefix_cache_inspect(
        self,
        token_ids: list[int],
        include_subtree: bool = False,
    ) -> dict[str, object]:
        return self.model_runner.call(
            "prefix_cache_inspect",
            [int(token_id) for token_id in token_ids],
            bool(include_subtree),
        )

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        return self.model_runner.call(
            "prefix_cache_match",
            [int(token_id) for token_id in token_ids],
        )

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        return self.model_runner.call(
            "prefix_cache_delete_subtree",
            [int(token_id) for token_id in token_ids],
        )

    def prefix_cache_set_eviction_priority(
        self,
        token_ids: list[int],
        priority: int,
    ) -> dict[str, object]:
        return self.model_runner.call(
            "prefix_cache_set_eviction_priority",
            [int(token_id) for token_id in token_ids],
            int(priority),
        )

    def prefix_cache_prune_start(
        self,
        token_ids: list[int],
        range_start: int,
        range_end: int,
        keep_tokens: int,
        policy: str,
        allow_recompress: bool = False,
        observation_tokens: int = 64,
        score_chunk_size: int = 2048,
        prev_postfix_size: int = 64,
    ) -> dict[str, object]:
        token_ids = [int(token_id) for token_id in token_ids]
        if str(self.config.sparse_method or "") == "quest":
            raise RuntimeError(
                "QuEST prefix cache/offload remains supported, but physical prefix "
                "pruning is intentionally unsupported."
            )
        normalized_policy = validate_prefix_prune_request(
            token_count=len(token_ids),
            range_start=int(range_start),
            range_end=int(range_end),
            keep_tokens=int(keep_tokens),
            block_size=int(self.config.prefix_cache_block_size),
            policy=str(policy),
        )
        if not bool(self.config.enable_prefix_caching):
            raise RuntimeError("prefix cache must be enabled before a prune job can start.")
        if int(observation_tokens) <= 0:
            raise ValueError("observation_tokens must be positive.")
        if int(score_chunk_size) <= 0:
            raise ValueError("score_chunk_size must be positive.")
        if int(prev_postfix_size) < 0:
            raise ValueError("prev_postfix_size must be non-negative.")
        prune_id = uuid.uuid4().hex
        job = PrefixPruneJob(
            prune_id=prune_id,
            token_ids=token_ids,
            range_start=int(range_start),
            range_end=int(range_end),
            keep_tokens=int(keep_tokens),
            policy=normalized_policy,
            allow_recompress=bool(allow_recompress),
            observation_tokens=int(observation_tokens),
            score_chunk_size=int(score_chunk_size),
            prev_postfix_size=int(prev_postfix_size),
        )
        self._prefix_prune_jobs[prune_id] = job
        self._pending_prefix_prune_ids.append(prune_id)
        return job.to_dict()

    def prefix_cache_prune_status(self, prune_id: str) -> dict[str, object]:
        job = self._prefix_prune_jobs.get(str(prune_id))
        if job is None:
            raise RuntimeError(f"unknown prefix prune id: {prune_id!r}.")
        return job.to_dict()

    def run_pending_prefix_prune(self) -> bool:
        if not self._pending_prefix_prune_ids:
            return False
        prune_id = self._pending_prefix_prune_ids.popleft()
        job = self._prefix_prune_jobs[prune_id]
        job.status = "running"
        job.started_at = time.time()
        try:
            replay_prefix_ids = None
            if job.policy == "kvzip_global":
                replay_prefix_ids = [
                    int(token_id)
                    for token_id in self.tokenizer.encode(
                        "\nReconstruct the following context span exactly:\n",
                        add_special_tokens=False,
                    )
                ]
                if not replay_prefix_ids:
                    raise RuntimeError(
                        "KVzip reconstruction prompt tokenized to an empty sequence."
                    )
            job.result = self.model_runner.call(
                "prefix_cache_prune",
                job.token_ids,
                job.range_start,
                job.range_end,
                job.keep_tokens,
                job.policy,
                job.prune_id,
                job.allow_recompress,
                job.observation_tokens,
                job.score_chunk_size,
                job.prev_postfix_size,
                replay_prefix_ids,
                -1_000_000_000 - len(self._prefix_prune_jobs) * 100_000,
            )
            job.status = "completed"
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            message = str(exc).lower()
            job.status = (
                "blocked"
                if "idle" in message or "referenced" in message or "in-flight" in message
                else "failed"
            )
            logger.error("Prefix prune job {} {}: {}", prune_id, job.status, job.error)
        finally:
            job.finished_at = time.time()
        return True

    def debug_sparse_state_summaries(self, synchronize: bool = False) -> list[dict[str, object]]:
        summaries = self.model_runner.call("debug_sparse_state_summaries", synchronize)
        expected = self.config.attn_tp_size
        if not isinstance(summaries, list) or len(summaries) != expected:
            raise RuntimeError(
                "Sparse-state summary did not return one record per attention TP rank: "
                f"expected={expected}, got={summaries!r}."
            )
        return summaries

    def operator_runtime_stats(self) -> list[dict[str, object]]:
        stats = self.model_runner.call("operator_runtime_stats")
        expected = self.config.attn_tp_size
        if not isinstance(stats, list) or len(stats) != expected:
            raise RuntimeError(
                "Operator runtime stats did not return one record per attention TP rank: "
                f"expected={expected}, got={stats!r}."
            )
        return stats

    def debug_last_logits(self) -> torch.Tensor:
        logits = self.model_runner.call("debug_last_logits_cpu")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(f"Rank 0 did not return debug logits: {logits!r}.")
        return logits

    def debug_set_next_decode_token(self, seq_id: int, token_id: int) -> None:
        """Teacher-force the unprocessed token for independent logit validation."""
        if not 0 <= int(token_id) < int(self.config.hf_config.vocab_size):
            raise ValueError("Teacher-forced token is outside the vocabulary.")
        seq = next(seq for seq in self.scheduler.decoding if seq.seq_id == seq_id)
        if seq.chain_id or seq.has_sampling_penalty:
            raise ValueError("Teacher-forced validation requires no chain or sampling penalties.")
        seq.token_ids[-1] = int(token_id)
        seq.last_token = int(token_id)

    def debug_hidden_states(self) -> dict[int, torch.Tensor]:
        snapshots = self.model_runner.call("debug_hidden_states_cpu")
        if not isinstance(snapshots, dict) or not all(
            isinstance(layer_idx, int) and isinstance(tensor, torch.Tensor)
            for layer_idx, tensor in snapshots.items()
        ):
            raise RuntimeError(
                f"Rank 0 did not return hidden-state snapshots: {snapshots!r}."
            )
        return snapshots

    def debug_moe_states(self) -> dict[int, dict[str, torch.Tensor]]:
        snapshots = self.model_runner.call("debug_moe_states_cpu")
        if not isinstance(snapshots, dict):
            raise RuntimeError(f"Rank 0 did not return MoE snapshots: {snapshots!r}.")
        return snapshots

    def worker_info(
        self,
        served_model_name: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, object]:
        config = self.config
        benchmark_config_keys = (
            "gpu_memory_utilization",
            "num_kvcache_slots",
            "max_num_batched_tokens",
            "prefill_schedule_policy",
            "engine_prefill_chunk_size",
            "mla_prefill_history_chunk_size",
            "mla_prefill_workspace_bytes",
            "long_prefill_offload_threshold",
            "sink_keep_tokens",
            "recent_keep_tokens",
            "decode_keep_tokens",
            "full_attention_layers",
            "resolved_full_attention_profile",
            "obs_layer_ids",
            "snapkv_window_size",
            "snapkv_num_full_layers",
            "sparse_prefill_score_mode",
            "prefill_sparse_method",
            "flashprefill_v2_k_block_m",
            "flashprefill_v2_k_block_n",
            "flashprefill_v2_abs_threshold",
            "flashprefill_v2_attention_sink_blocks",
            "flashprefill_v2_window_blocks",
            "flashprefill_v2_last_query_blocks",
            "flashprefill_v2_min_sparse_q_len",
            "flashprefill_v2_use_mean_correction",
            "h2o_decode_budget",
            "h2o_decode_eviction",
            "h2o_decode_score_fusion",
            "h2o_decode_eviction_interval",
            "h2o_prefill_budget",
            "h2o_recent_ratio",
            "h2o_prefill_score_window",
            "pool_kernel_size",
            "sparse_attn_score_dtype",
            "pyramid_layer_ratios",
            "pyramidkv_start_layer",
            "pyramidkv_start_ratio",
            "pyramidkv_least_layer",
            "pyramidkv_least_ratio",
            "quest_chunk_size",
            "quest_token_budget",
            "quest_skip_layers",
            "deltakv_checkpoint_path",
            "deltakv_center_ratio",
            "deltakv_latent_dim",
            "deltakv_latent_quant_bits",
            "deltakv_latent_quant_group_size",
            "decode_graph",
            "decode_graph_capture_sampling",
            "decode_graph_capture_sizes",
            "decode_graph_startup_capture",
            "decode_graph_startup_capture_limit",
            "enable_prefix_caching",
            "prefix_cache_mode",
            "resolved_prefix_cache_mode",
            "chain_cache_max_tombstones",
            "prefix_cache_block_size",
            "prefix_cache_requested_max_blocks",
            "prefix_cache_max_blocks",
            "enable_prefix_cache_offload",
            "prefix_cache_host_size_gb",
            "recurrent_state_max_bytes",
            "prefix_cache_max_recurrent_bytes",
            "recurrent_state_pool_bytes",
            "recurrent_state_bytes_per_row",
            "recurrent_state_row_capacity",
            "prefix_recurrent_bytes_per_block",
            "prefix_recurrent_capacity_bytes",
            "prefix_kv_bytes_per_block",
            "prefix_kv_block_capacity",
            "kv_allocatable_bytes",
        )

        def jsonable(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [jsonable(item) for item in value]
            if isinstance(value, dict):
                return {
                    str(key): jsonable(item)
                    for key, item in value.items()
                }
            raise TypeError(
                "Worker benchmark metadata is not JSON serializable: "
                f"type={type(value).__name__} value={value!r}."
            )

        return {
            "served_model_name": served_model_name or str(config.model),
            "model": str(config.model),
            "model_type": str(getattr(config.hf_config, "model_type", "")),
            "vocab_size": int(
                getattr(config.hf_config, "vocab_size", 0) or 0
            ),
            "sparse_method": str(getattr(config, "sparse_method", "") or ""),
            "prefill_sparse_method": str(
                getattr(config, "prefill_sparse_method", "") or ""
            ),
            "world_size": int(getattr(config, "world_size", 1)),
            "tensor_parallel_size": int(getattr(config, "tensor_parallel_size", 1)),
            "expert_parallel_size": int(getattr(config, "expert_parallel_size", 1)),
            "data_parallel_size": int(getattr(config, "data_parallel_size", 1)),
            "max_model_len": int(getattr(config, "max_model_len", 0) or 0),
            "max_num_seqs_in_batch": int(getattr(config, "max_num_seqs_in_batch", 0) or 0),
            "max_decoding_seqs": int(getattr(config, "max_decoding_seqs", 0) or 0),
            "max_num_seqs_in_gpu": int(getattr(config, "max_num_seqs_in_gpu", 0) or 0),
            "prefix_cache_enabled": bool(getattr(config, "enable_prefix_caching", False)),
            "prefix_cache_mode": str(
                getattr(config, "resolved_prefix_cache_mode", "disabled")
            ),
            "prefix_cache_block_size": getattr(config, "prefix_cache_block_size", None),
            "code_revision": code_revision_info(),
            "benchmark_config": {
                key: jsonable(getattr(config, key))
                for key in benchmark_config_keys
                if hasattr(config, key)
            },
            "tags": sorted(str(tag) for tag in (tags or []) if str(tag)),
        }

    def worker_routing_load(self) -> dict[str, object]:
        scheduler = self.scheduler
        waiting = len(scheduler.waiting)
        decoding = len(scheduler.decoding)
        return {
            "waiting_requests": int(waiting),
            "decoding_requests": int(decoding),
            "active_requests": int(waiting + decoding),
            "total_preemptions": int(getattr(scheduler, "total_preemptions", 0)),
            "total_recompute_replays": int(
                getattr(scheduler, "total_recompute_replays", 0)
            ),
            "max_num_seqs_in_batch": int(getattr(scheduler, "max_num_seqs_in_batch", 0)),
            "max_decoding_seqs": int(getattr(scheduler, "max_decoding_seqs", 0)),
            "max_num_seqs_in_gpu": int(getattr(scheduler.config, "max_num_seqs_in_gpu", 0)),
        }

    def worker_load(self) -> dict[str, object]:
        result = self.worker_routing_load()
        cache_stats = self.model_runner.runtime_state.free_slot_stats()
        result["cache"] = {
            str(key): int(value)
            for key, value in cache_stats.items()
            if isinstance(value, int)
        }
        return result

    def prefix_cache_routing_snapshot(self) -> PrefixCacheRoutingSnapshot:
        runtime_state = self.model_runner.runtime_state
        owner = (
            runtime_state.prefix_cache_coordinator
            if runtime_state.prefix_cache_coordinator is not None
            else runtime_state.cache_manager
        )
        method = str(self.config.sparse_method or "")
        prefix_cache = getattr(owner, "prefix_cache", None)
        if prefix_cache is not None:
            return prefix_cache.routing_snapshot(method)

        supported = hasattr(owner, "prefix_cache")
        return PrefixCacheRoutingSnapshot(
            supported=supported,
            enabled=False,
            method=method,
            reason=(
                "prefix cache is not enabled for this runtime."
                if supported
                else "prefix cache is not supported by this cache manager."
            ),
        )

    def _release_preempted_sequences(self, preempted_seqs: list[Sequence]) -> None:
        preempted_seq_ids = [int(seq.seq_id) for seq in preempted_seqs]
        if not preempted_seq_ids:
            return
        # Preemption is transient: retain the logical request and chain
        # identity, release only runtime KV/recurrent state, and let
        # scheduler-driven recompute rebuild it later.
        self.model_runner.call("free_slots_batch", preempted_seq_ids)

    def step(self):
        asynchronous = getattr(self, "_async_scheduler", None)
        return asynchronous.step() if asynchronous is not None else self._step_sync()

    def _step_sync(self):
        """
        执行单个推理步进（一个 Batch）。
        包含：调度、抢占处理、模型前向计算、状态更新、资源回收。
        """
        with profiler.record("step"):
            self.last_step_token_outputs = []
            self.last_step_prompt_cache_hits = []
            self.last_step_logprob_outputs = []
            # 1. 调度：决定哪些序列进入本次 Batch
            with profiler.record("schedule"):
                seqs, is_prefill, preempted_seqs = self.scheduler.schedule()
            if is_prefill:
                self.last_step_prompt_cache_hits = [
                    (int(seq.seq_id), int(seq.prefix_cache_hit_len))
                    for seq in seqs
                ]
            prefill_batch_mode = (
                self.scheduler.prefill_execution_mode_for_batch(seqs)
                if seqs and is_prefill
                else None
            )
            
            # 2. 显式处理抢占 (Eviction)：
            # 如果有序列被调度器踢出，立即广播指令让所有 Rank 释放其占用的物理槽位
            with profiler.record("preempt_free"):
                self._release_preempted_sequences(preempted_seqs)
                
            if not seqs:
                if self.config.attn_dp_size > 1:
                    self.model_runner.call("run", [], False)
                # No progress can be made; avoid infinite busy-looping in callers.
                if preempted_seqs or self.is_finished():
                    prefill_seqs = len(self.scheduler.waiting)
                    decode_seqs = len(self.scheduler.decoding)
                    prefill_modes = self.scheduler.prefill_execution_mode_counts()
                    self._throughput_logger.record_state(
                        prefill_seqs + decode_seqs,
                        prefill_seqs,
                        decode_seqs,
                        prefill_modes["chunked"],
                        prefill_modes["full"],
                        prefill_modes["raw_offload"],
                        "idle",
                    )
                    return [], 0
                # Most commonly: a prompt is larger than KV cache capacity (for methods that keep all tokens),
                # or scheduling constraints prevent any chunk from being placed.
                raise RuntimeError(
                    "Scheduler returned no runnable sequences and no preemptions; "
                    "this would hang the generation loop. "
                    f"method={self.config.sparse_method} free_slots={self.model_runner.runtime_state.num_free_slots} "
                    f"waiting={len(self.scheduler.waiting)} decoding={len(self.scheduler.decoding)}"
                )
                
            # 3. 跨进程广播并执行推理：
            # Rank 0 会驱动所有 Rank 进程同步运行本地的 ModelRunner.run
            with profiler.record("model_run_call"):
                try:
                    token_ids, logprob_outputs = self.model_runner.call(
                        "run", seqs, is_prefill
                    )
                except Exception:
                    if is_prefill:
                        # Prefills leave waiting during selection. Restore
                        # ownership until cleanup succeeds, including the first
                        # chunk whose num_prefilled_tokens is still zero.
                        self.scheduler.waiting.extendleft(reversed(seqs))
                    for seq in seqs:
                        chain_seq = self._active_chain_sequences.get(int(seq.seq_id))
                        try:
                            if chain_seq is None:
                                self.model_runner.call("free_slots", int(seq.seq_id))
                            else:
                                self.model_runner.call(
                                    "chain_invalidate",
                                    str(chain_seq.chain_id),
                                    int(chain_seq.seq_id),
                                )
                        except Exception:
                            logger.exception(
                                "Failed to reclaim seq_id={} after model failure.",
                                seq.seq_id,
                            )
                            continue
                        # Commit removal only after the physical release. Keep
                        # failed cleanup visible to the serving cancellation path.
                        self._active_chain_sequences.pop(int(seq.seq_id), None)
                        self.scheduler.abort(int(seq.seq_id))
                    raise
            token_logprobs, top_logprobs = (
                logprob_outputs if logprob_outputs is not None else (None, None)
            )

            token_outputs: list[tuple[int, list[int]]] = []
            logprob_step_outputs: list[
                tuple[int, list[float | None], list[dict[int, float] | None]]
            ] = []
            step_token_logprobs = token_logprobs or [None] * len(seqs)
            step_top_logprobs = top_logprobs or [None] * len(seqs)
            for seq, token_id, token_logprob, top_logprob in zip(
                seqs,
                token_ids,
                step_token_logprobs,
                step_top_logprobs,
            ):
                if (
                    seq.should_publish_sample
                    and (not is_prefill or seq.is_last_chunk_prefill)
                ):
                    token_outputs.append((seq.seq_id, [int(token_id)]))
                    logprob_step_outputs.append((seq.seq_id, [token_logprob], [top_logprob]))
            
            # 4. 逻辑后处理：更新序列的 Token 列表和状态机
            with profiler.record("postprocess"):
                self.scheduler.postprocess(
                    seqs,
                    token_ids,
                    is_prefill,
                    token_logprobs=token_logprobs,
                    top_logprobs=top_logprobs,
                )
            self.last_step_token_outputs = token_outputs
            self.last_step_logprob_outputs = logprob_step_outputs
            
            # 5. 完成序列的资源回收：
            # 遍历序列，如果已达到 EOS 或最大长度，则通知所有进程释放物理槽位
            with profiler.record("finished_free"):
                finished_outputs = []
                finished_seq_ids = []
                for seq in seqs:
                    if seq.is_finished:
                        chain_seq = self._active_chain_sequences.get(
                            int(seq.seq_id), None
                        )
                        if chain_seq is None:
                            finished_seq_ids.append(int(seq.seq_id))
                        else:
                            processed_token_count = max(
                                0, int(chain_seq.num_tokens) - 1
                            )
                            coordinator = (
                                self.model_runner.runtime_state
                                .chain_cache_coordinator
                            )
                            if coordinator is None:
                                raise RuntimeError(
                                    "Finished a chain request without a chain "
                                    "cache coordinator."
                                )
                            prepared = coordinator.prepare_processed_tokens(
                                chain_id=str(chain_seq.chain_id),
                                seq_id=int(chain_seq.seq_id),
                                token_ids=chain_seq.token_ids,
                                processed_token_count=processed_token_count,
                            )
                            self.model_runner.call(
                                "chain_finish",
                                prepared.chain_id,
                                prepared.seq_id,
                                prepared.processed_token_digest,
                                prepared.processed_token_count,
                            )
                            coordinator.remember_prepared_tokens(prepared)
                            self._active_chain_sequences.pop(int(seq.seq_id), None)
                        finished_outputs.append(
                            (
                                seq.seq_id,
                                seq.completion_token_ids,
                                seq.completion_token_logprobs,
                                seq.completion_top_logprobs,
                            )
                        )
                if finished_seq_ids:
                    self.model_runner.call("finish_slots_batch", finished_seq_ids)
        
        # 计算吞吐量统计数据 (正数表示 Prefill，负数表示 Decode)
        num_tokens = sum(seq.current_chunk_size for seq in seqs) if is_prefill else -len(seqs)
        self._throughput_logger.record_step(num_tokens)
        prefill_seqs = len(self.scheduler.waiting)
        decode_seqs = len(self.scheduler.decoding)
        prefill_modes = self.scheduler.prefill_execution_mode_counts()
        if is_prefill:
            if prefill_batch_mode is None:
                raise RuntimeError("Missing execution mode for a scheduled prefill batch.")
            last_batch = f"pf-{prefill_batch_mode}"
        else:
            last_batch = "decode"
        self._throughput_logger.record_state(
            prefill_seqs + decode_seqs,
            prefill_seqs,
            decode_seqs,
            prefill_modes["chunked"],
            prefill_modes["full"],
            prefill_modes["raw_offload"],
            last_batch,
        )
        return finished_outputs, num_tokens

    def is_finished(self):
        """检查是否所有请求都已处理完毕"""
        pending = getattr(getattr(self, "_async_scheduler", None), "pending", ())
        return not pending and self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[MultiModalPrompt] | list[dict],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        高层 API：批量输入 Prompt，阻塞直至全部生成完成。
        返回包含生成的 text 和 token_ids 的字典列表。
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 提交所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        # 主推理循环
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            # 更新吞吐量统计
            if use_tqdm:
                dt = perf_counter() - t
                if num_tokens > 0:
                    prefill_throughput = num_tokens / dt
                else:
                    decode_throughput = -num_tokens / dt
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集已完成的输出
            for seq_id, token_ids, _token_logprobs, _top_logprobs in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 按照请求提交顺序排序并解码
        results = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        results = [{"text": self.tokenizer.decode(tids, skip_special_tokens=True), "token_ids": tids} for tids in results]
        
        if use_tqdm:
            pbar.close()
        return results
