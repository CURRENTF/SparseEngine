import copy
import inspect
import os
import pickle
import socket
import time
import uuid
import torch
import torch.distributed as dist
from sparseengine.utils.log import logger
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from sparseengine.config import (
    Config,
    _resolve_decode_cuda_graph_capture_sizes,
)
from sparseengine.configs.cuda_graph import (
    _decode_cuda_graph_max_real_batch_size,
    _resolve_decode_static_batch_capacity,
)
from sparseengine.distributed import (
    ParallelCollectiveRuntime,
    init_parallel_context,
    reset_parallel_context,
)
from sparseengine.distributed.topology import parallel_group_ranks
from sparseengine.engine.sparse_methods.base import AuxiliaryPrefillRequest
from sparseengine.engine.sequence import Sequence
from sparseengine.engine.async_scheduling.execution import AsyncExecution, DeviceLogprobs
from sparseengine.models.qwen2 import Qwen2ForCausalLM
from sparseengine.models.llama import LlamaForCausalLM
from sparseengine.layers.sampler import Sampler
from sparseengine.kernels.external.required import (
    validate_required_cuda_kernel_families,
)
from sparseengine.kernels.external.flashinfer.jit_cache import configure_trtllm_cache
from sparseengine.operators import registry as operator_registry
from sparseengine.operators.decode_attention import (
    collect_decode_graph_participants,
    validate_decode_graph_model,
)
from sparseengine.operators.workspace import (
    close_workspace_manager,
    init_workspace_manager,
    lock_workspace_manager,
)
from sparseengine.utils.context import set_context, get_context, reset_context
from sparseengine.utils.loader import load_model, sync_deltakv_config_from_checkpoint
from sparseengine.utils.process_title import set_engine_process_title

from sparseengine.engine.cache_manager import CacheManager
from sparseengine.engine.cache_manager.base import _debug_tensor_summary
from sparseengine.engine.decode_cuda_graph import DecodeCudaGraphRunner
from sparseengine.engine.prefix_cache_coordinator import PrefixCacheCoordinator
from sparseengine.engine.prefix_prune import (
    normalize_prefix_prune_ranges,
    validate_prefix_prune_request,
)
from sparseengine.engine.chain_cache import ChainAdmissionPlan, ChainCacheCoordinator
from sparseengine.engine.recurrent_state_manager import RecurrentStateManager, RecurrentStateSpec
from sparseengine.engine.runtime_state import RuntimeState
from sparseengine.engine.startup import (
    CacheRuntimeBuildMeasurement,
    DeviceMemorySnapshot,
    StartupMemoryProfiler,
    feasible_startup_graph_plan,
    profiling_kv_budget_bytes,
    profiling_kv_slots,
    release_unused_device_memory,
)
from sparseengine.multimodal.runtime import MultiModalRuntime
from sparseengine.engine.sparse_controller import SparseController
from sparseengine.models.spec import ModelSpec
import sparseengine.platforms as platforms
from sparseengine.utils.profiler import cpu_timing, profiler

try:
    from sparseengine.models.qwen3 import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = None

try:
    from sparseengine.models.qwen3_moe import Qwen3MoeForCausalLM
except ImportError:
    Qwen3MoeForCausalLM = None

try:
    from sparseengine.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
except ImportError:
    Glm4MoeLiteForCausalLM = None

try:
    from sparseengine.models.minimax_m2 import MiniMaxM2ForCausalLM
except ImportError:
    MiniMaxM2ForCausalLM = None

try:
    from sparseengine.models.qwen3_5 import Qwen35ForCausalLM
except ImportError:
    Qwen35ForCausalLM = None

try:
    from sparseengine.models.qwen3_5_moe import Qwen35MoeForCausalLM
except ImportError:
    Qwen35MoeForCausalLM = None

try:
    from sparseengine.models.gemma4 import Gemma4ForCausalLM
except ImportError:
    Gemma4ForCausalLM = None


_INIT_PROCESS_GROUP_SUPPORTS_DEVICE_ID = (
    "device_id" in inspect.signature(dist.init_process_group).parameters
)


def _init_process_group(
    *,
    backend: str,
    init_method: str,
    world_size: int,
    rank: int,
    device: torch.device,
) -> None:
    kwargs = {
        "backend": backend,
        "init_method": init_method,
        "world_size": world_size,
        "rank": rank,
    }
    if _INIT_PROCESS_GROUP_SUPPORTS_DEVICE_ID:
        kwargs["device_id"] = device
    dist.init_process_group(**kwargs)


def _close_runtime_bindings(runtime_bindings: dict) -> None:
    closed_ids: set[int] = set()
    for binding in runtime_bindings.values():
        if id(binding) in closed_ids:
            continue
        close = getattr(binding, "close", None)
        if callable(close):
            close()
            closed_ids.add(id(binding))


def _create_model(hf_config, model_spec: ModelSpec, **runtime_kwargs):
    class_name = model_spec.runtime_class_name
    model_class = globals().get(class_name)
    if model_class is None:
        raise ImportError(f"{class_name} is unavailable for {model_spec.name}.")
    builder = getattr(model_class, "build_runtime_kwargs", None)
    model_runtime_kwargs = (
        builder(hf_config, **runtime_kwargs) if callable(builder) else {}
    )
    model = None
    try:
        model = model_class(hf_config, **model_runtime_kwargs)
        engine_config = runtime_kwargs.get("engine_config")
        configure_multimodal = getattr(model, "configure_multimodal", None)
        if (
            callable(configure_multimodal)
            and bool(getattr(engine_config, "enable_multimodal", False))
            and getattr(engine_config, "outer_hf_config", hf_config) is not hf_config
        ):
            configure_multimodal(engine_config.outer_hf_config)
        return model
    except BaseException:
        close_runtime_operators = (
            None
            if model is None
            else getattr(model, "close_runtime_operators", None)
        )
        if callable(close_runtime_operators):
            close_runtime_operators()
        else:
            _close_runtime_bindings(model_runtime_kwargs)
        raise


TP_SHM_NAME_PREFIX = "sparseengine_"
TP_SHM_SIZE = 2**20
TP_RUN_STATUS_PENDING = 0
TP_RUN_STATUS_SUCCESS = 1
TP_RUN_STATUS_FAILED = 2
DEFAULT_MASTER_PORT = 2333
PREFIX_CACHE_CONTROL_RPC_METHODS = {
    "prefix_cache_inspect",
    "prefix_cache_match",
    "prefix_cache_delete_subtree",
    "prefix_cache_set_eviction_priority",
    "prefix_cache_prune",
    "prefix_cache_prune_batch",
}
DECODE_GRAPH_HOST_STATUS_SYNC_METHODS = {
    "begin_decode_cuda_graph_capture",
    "capture_decode_cuda_graph_warmup",
    "collect_decode_cuda_graph_metadata",
    "exchange_decode_cuda_graph_metadata",
    "register_decode_cuda_graph_buffers",
    "seal_decode_cuda_graph_startup_plan",
}
STARTUP_HOST_STATUS_SYNC_METHODS = {
    "arm_runtime_compilation_guard",
    "begin_startup_memory_profile",
    "build_production_cache_runtime",
    "capture_startup_memory_snapshot",
    "finish_startup_memory_profile",
    "profile_startup_prefill",
    "release_profiling_cache_runtime",
    "resolve_startup_decode_graph_plan",
    "startup_capture_prefill_fits",
    "startup_capture_decode_fits",
}
RECOVERABLE_TP_CONTROL_RPC_METHODS = PREFIX_CACHE_CONTROL_RPC_METHODS | {
    "chain_validate_admission_plan",
    "finish_slots_batch",
    "free_multimodal",
    "free_slots",
    "free_slots_batch",
    "register_multimodal_shared",
} | DECODE_GRAPH_HOST_STATUS_SYNC_METHODS | STARTUP_HOST_STATUS_SYNC_METHODS
TP_RPC_STATUS_SYNC_METHODS = PREFIX_CACHE_CONTROL_RPC_METHODS | {
    "chain_admission_plan",
    "chain_apply_admission",
    "chain_reclaim_idle",
    "chain_finish",
    "chain_invalidate",
    "chain_validate_admission_plan",
    "debug_hidden_states_cpu",
    "debug_moe_states_cpu",
    "free_slots",
    "free_slots_batch",
    "finish_slots_batch",
    "free_multimodal",
    "log_operator_implementations",
    "refresh_prefix_cache_hit",
    "refresh_prefix_cache_hits",
    "reset_after_warmup",
    "run",
    "submit_async",
    "collect_async",
    "retire_async",
    "register_multimodal_shared",
    "warmup_moe_workspace",
} | DECODE_GRAPH_HOST_STATUS_SYNC_METHODS | STARTUP_HOST_STATUS_SYNC_METHODS


class _RPCPayloadTooLarge(RuntimeError):
    """The command was rejected before any worker was signalled."""


def make_tp_shm_name() -> str:
    return f"{TP_SHM_NAME_PREFIX}{os.getpid()}_{uuid.uuid4().hex}"


def select_master_port() -> int:
    configured = os.getenv("SPARSEENGINE_MASTER_PORT")
    if configured is not None:
        try:
            port = int(configured)
        except ValueError as exc:
            raise ValueError(
                f"SPARSEENGINE_MASTER_PORT must be an integer, got {configured!r}."
            ) from exc
        if not 1 <= port <= 65535:
            raise ValueError(f"SPARSEENGINE_MASTER_PORT must be in [1, 65535], got {port}.")
    else:
        port = DEFAULT_MASTER_PORT

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # torch.distributed's TCPStore may leave the just-finished master port in
        # TIME_WAIT. Match the store's reusable-listener semantics so a serial
        # benchmark can safely reuse its explicitly assigned port, while an
        # active listener still fails the bind below.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            if configured is not None:
                raise RuntimeError(
                    f"SPARSEENGINE_MASTER_PORT={port} is already in use."
                ) from exc
            sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ModelRunner:
    """
    负责模型执行的类。每个 GPU Rank 进程都拥有一个 ModelRunner 实例。
    主要职责：权重加载、显存分配 (KV Cache)、槽位管理 (Rank-Local)、前向计算。
    """

    def __init__(
        self,
        config: Config,
        rank: int,
        event: tuple[Event, Event] | list[tuple[Event, Event]],
        tp_shm_name: str | None = None,
        master_port: int | None = None,
        trtllm_cache_root: str | None = None,
    ):
        self.config = config
        self._slot_release_states: dict[int, str] = {}
        # Inference-only engine: disable autograd graph construction globally in this process.
        # (This is process-local; must be set inside every spawned TP worker.)
        torch.set_grad_enabled(False)
        profiler.set_rank(rank)
        profiler.set_enabled(config.enable_profiler and rank == 0)
        hf_config = config.hf_config
        self.world_size = config.world_size
        self.rank = rank
        self.event = event
        self.tp_shm_name = tp_shm_name
        self.platform = platforms.current_platform
        if self.platform.enum is platforms.PlatformEnum.CUDA:
            cache_path = configure_trtllm_cache(rank, trtllm_cache_root)
            logger.info("Worker {} TensorRT-LLM DeepGEMM cache: {}", rank, cache_path)
        self.platform.validate_inference()
        self.platform.init_backend()
        self.device = self.platform.get_device(rank)

        # 初始化分布式环境并绑定对应的设备
        self.platform.set_device(self.device)
        if self.platform.enum is platforms.PlatformEnum.CUDA:
            validate_required_cuda_kernel_families()
        if not dist.is_initialized():
            master_port = select_master_port() if master_port is None else master_port
            _init_process_group(
                backend=self.platform.get_distributed_backend(),
                init_method=f"tcp://localhost:{master_port}",
                world_size=self.world_size,
                rank=rank,
                device=self.device,
            )
        self.parallel_context = init_parallel_context(
            topology=config.parallel_topology,
        )
        # Prefix lookup agreement contains host metadata only. Using the CUDA
        # group would enqueue object collectives behind in-flight model work
        # and synchronously copy their results back to the scheduler.
        self.tp_control_group = None
        for ranks in parallel_group_ranks(config.parallel_topology)["attn_tp"]:
            if len(ranks) > 1:
                group = dist.new_group(list(ranks), backend="gloo")
                if rank in ranks:
                    self.tp_control_group = group
        self.dp_control_group = (
            dist.new_group(backend="gloo") if config.attn_dp_size > 1 else None
        )
        self.dp_control_buffer = (
            torch.empty(config.attn_dp_size + 1, dtype=torch.int64, device="cpu")
            if config.attn_dp_size > 1
            else None
        )
        self.dp_idle_graphs = {}
        self.dp_idle_replay_count = 0
        self.dp_idle_eager_count = 0
        set_engine_process_title(self.parallel_context)
        # CUDA allocator peaks are process-global and survive LLMEngine.exit().
        # Start a new lifecycle before model construction so KV sizing observes
        # only this engine's model load and persistent allocations.
        self.platform.reset_peak_memory_stats(self.device)
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device)
        setattr(hf_config, "mlp_chunk_size", config.mlp_chunk_size)
        setattr(
            hf_config,
            "moe_max_num_tokens",
            min(
                int(
                    getattr(
                        config,
                        "max_num_batched_tokens",
                        config.mlp_chunk_size,
                    )
                ) * config.attn_dp_size,
                int(config.mlp_chunk_size),
            ),
        )
        setattr(
            hf_config,
            "decode_graph",
            bool(getattr(config, "decode_graph", False)),
        )
        max_real_decode_batch_size = _decode_cuda_graph_max_real_batch_size(
            max_decoding_seqs=config.max_decoding_seqs,
        )
        decode_static_capture_sizes = _resolve_decode_cuda_graph_capture_sizes(
            config.decode_graph_capture_sizes,
            max_real_decode_batch_size,
        )
        self.collective_runtime = ParallelCollectiveRuntime(
            self.parallel_context,
            cuda_graph=config.decode_graph,
            device_index=int(self.device.index or 0),
        )
        init_workspace_manager(self.device)
        try:
            self.model = _create_model(
                hf_config,
                config.model_spec,
                engine_config=config,
                parallel_context=self.parallel_context,
                collective_runtime=self.collective_runtime,
                device=self.device,
                max_decode_tokens=_resolve_decode_static_batch_capacity(
                    decode_static_capture_sizes,
                    max_decoding_seqs=config.max_decoding_seqs,
                ),
            )
            self.collective_runtime.prepare()
            if (
                self.collective_runtime.has_graph_collectives
                and not config.decode_graph_startup_capture
            ):
                raise ValueError(
                    "Distributed decode CUDA Graph collectives require "
                    "decode_graph_startup_capture=True so every graph buffer is "
                    "registered before the first replay."
                )
        except BaseException:
            model = getattr(self, "model", None)
            close_runtime_operators = getattr(model, "close_runtime_operators", None)
            if callable(close_runtime_operators):
                close_runtime_operators()
            self.collective_runtime.close()
            close_workspace_manager()
            raise
        if self.config.decode_graph:
            validate_decode_graph_model(self.model)
        if config.tiny_random:
            from sparseengine.debug.tiny_random import initialize_sparse_model

            initialize_sparse_model(
                self.model,
                hf_config,
                seed=config.tiny_random_seed,
                quantized=config.quantization_config.enabled,
            )
        else:
            load_model(
                self.model,
                config.model,
                tp_rank=self.parallel_context.attn_tp_rank,
                tp_size=self.parallel_context.attn_tp_size,
                num_threads=config.weight_loading_workers_per_rank,
                show_progress=self.parallel_context.world_rank == 0,
                progress_rank=0 if self.parallel_context.world_rank == 0 else None,
            )
        self.model.eval()
        from sparseengine.operators.moe_execution import prepare_model_moe_execution

        self.moe_aux_stream = prepare_model_moe_execution(
            self.model, self.device,
        )
        self.multimodal_runtime = MultiModalRuntime(self.model, self.device)
        self._prefill_inputs_embeds = None
        self._prefill_multimodal_mask = None
        warmup_moe = getattr(self.model, "warmup_moe", None)
        if callable(warmup_moe):
            warmup_moe()
        lock_workspace_manager()
        
        self.sampler = Sampler()
        self._tokenizer_metadata: tuple[tuple[int, ...], tuple[int, ...] | None] | None = None

        # DeltaKV cache allocation depends on latent dimension / compressor architecture.
        # Sync those fields from the compressor checkpoint before creating CacheManager.
        sync_deltakv_config_from_checkpoint(config)
        
        has_linear_layers = bool(getattr(config.runtime_layout, "linear_attention_layer_indices", ()))
        state_spec_provider = getattr(self.model, "recurrent_state_spec", None)
        if has_linear_layers and not callable(state_spec_provider):
            raise RuntimeError(
                f"Model {type(self.model).__name__} declares linear-attention layers but does not "
                "provide recurrent_state_spec()."
            )
        state_spec = (
            state_spec_provider(config.hf_config, self.parallel_context.attn_tp_size)
            if has_linear_layers
            else None
        )
        if state_spec is not None and not isinstance(state_spec, RecurrentStateSpec):
            raise TypeError(
                f"recurrent_state_spec() must return RecurrentStateSpec, got {type(state_spec).__name__}."
            )
        self.recurrent_state_manager = None
        if state_spec is not None:
            self.recurrent_state_manager = RecurrentStateManager(
                config,
                self.parallel_context,
                device=self.device,
                platform=self.platform,
                state_spec=state_spec,
            )
        release_unused_device_memory(self.platform)
        self.startup_memory_profiler = StartupMemoryProfiler(
            self.platform,
            self.device,
        )
        self.profiling_kv_slots = profiling_kv_slots(config)
        self.profiling_kv_budget_bytes = profiling_kv_budget_bytes(
            config,
            self.profiling_kv_slots,
        )
        profiling_config = copy.copy(config)
        profiling_config.startup_cache_phase = "profiling"
        self._build_cache_runtime(
            profiling_config,
            allocation_budget_bytes=self.profiling_kv_budget_bytes,
        )
        self.cache_runtime_phase = "profiling"

        # Uninstall the construction-only DeviceContext. Setting "cpu" keeps
        # a Python interception mode on every Torch call in eager prefill.
        torch.set_default_device(None)
        torch.set_default_dtype(default_dtype)

        # TP 场景下的多进程指令同步
        if self.parallel_context.attn_tp_size > 1:
            if not self.tp_shm_name:
                raise ValueError("tp_shm_name is required when attention TP > 1.")
            if self.parallel_context.attn_tp_rank == 0:
                # Rank 0 创建共享内存用于发送方法调用指令
                self.shm = SharedMemory(name=self.tp_shm_name, create=True, size=TP_SHM_SIZE)
                self.parallel_context.attn_tp.barrier(
                    device_ids=self.platform.barrier_device_ids(rank)
                )
            else:
                # 其他 Rank 监听共享内存中的方法调用指令
                self.parallel_context.attn_tp.barrier(
                    device_ids=self.platform.barrier_device_ids(rank)
                )
                self.shm = SharedMemory(name=self.tp_shm_name)
                self.loop()

    def _build_cache_runtime(
        self,
        runtime_config: Config,
        *,
        allocation_budget_bytes: int,
    ) -> None:
        before_manager = DeviceMemorySnapshot.capture(self.platform, self.device)
        self.cache_manager = CacheManager.create(
            runtime_config,
            self.parallel_context,
            allocation_budget_bytes=int(allocation_budget_bytes),
        )
        release_unused_device_memory(self.platform)
        after_manager = DeviceMemorySnapshot.capture(self.platform, self.device)
        prefix_cache_mode = str(
            getattr(runtime_config, "resolved_prefix_cache_mode", "disabled")
        )
        self.prefix_cache_coordinator = (
            PrefixCacheCoordinator(
                runtime_config,
                self.cache_manager,
                self.recurrent_state_manager,
            )
            if self.recurrent_state_manager is not None and prefix_cache_mode == "radix"
            else None
        )
        self.chain_cache_coordinator = (
            ChainCacheCoordinator(runtime_config, self.cache_manager)
            if prefix_cache_mode == "chain"
            else None
        )
        if self.prefix_cache_coordinator is not None:
            self.cache_manager.prefix_cache_coordinator = self.prefix_cache_coordinator
        self.runtime_state = RuntimeState(
            runtime_config,
            self.cache_manager,
            self.recurrent_state_manager,
            self.prefix_cache_coordinator,
            self.chain_cache_coordinator,
            decode_graph_participants=collect_decode_graph_participants(self.model),
        )

        self.sparse_controller = SparseController(runtime_config, self.cache_manager)
        self.sparse_controller.bind_auxiliary_prefill(self._run_auxiliary_prefill)
        self._restore_tokenizer_metadata()
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.model.model.sparse_controller = self.sparse_controller
            if self.recurrent_state_manager is not None:
                self.model.model.recurrent_state_manager = self.recurrent_state_manager
            self.sparse_controller.set_modules(self.model.model.layers)
            if hasattr(self.cache_manager, "set_model_layers"):
                self.cache_manager.set_model_layers(self.model.model.layers)

        self.load_deltakv_compressors()

        max_real_decode_batch_size = _decode_cuda_graph_max_real_batch_size(
            max_decoding_seqs=runtime_config.max_decoding_seqs,
        )
        decode_static_capture_sizes = _resolve_decode_cuda_graph_capture_sizes(
            runtime_config.decode_graph_capture_sizes,
            max_real_decode_batch_size,
        )
        # Decode graph keys are captured lazily and replayed in workload order,
        # which is not necessarily their capture order.  Do not force those
        # independent graphs into one shared allocator pool: PyTorch only
        # permits pool sharing when replay follows capture order.
        self.cuda_graph_pool = None
        self.decode_graph_runner = DecodeCudaGraphRunner(
            runtime_state=self.runtime_state,
            cache_manager=self.cache_manager,
            recurrent_state_manager=self.recurrent_state_manager,
            sparse_controller=self.sparse_controller,
            run_model=self.run_model,
            method=runtime_config.sparse_method,
            capture_sizes=decode_static_capture_sizes,
            graph_pool=self.cuda_graph_pool,
            collective_runtime=self.collective_runtime,
        )
        release_unused_device_memory(self.platform)
        after_runtime = DeviceMemorySnapshot.capture(self.platform, self.device)
        self.cache_runtime_build_measurement = (
            CacheRuntimeBuildMeasurement.from_snapshots(
                before_manager,
                after_manager,
                after_runtime,
                allocation_budget_bytes=int(allocation_budget_bytes),
            )
        )

    def _gather_startup_record(self, local_record):
        if self.world_size == 1:
            return [local_record]
        records = [None] * self.world_size
        dist.all_gather_object(
            records,
            local_record,
            group=self.parallel_context.world.process_group,
        )
        return records if self.parallel_context.attn_tp_rank == 0 else None

    def begin_startup_memory_profile(self, phase: str) -> None:
        self.startup_memory_profiler.begin(phase)

    def finish_startup_memory_profile(self, phase: str):
        result = self.startup_memory_profiler.finish(phase)
        return self._gather_startup_record(
            {
                "world_rank": int(self.rank),
                "before": result.before,
                "after": result.after,
                "measurement": result.measurement,
            }
        )

    def profile_startup_prefill(self):
        from sparseengine.engine.startup.prefill_history import profile_prefill_history

        result = profile_prefill_history(self)
        return self._gather_startup_record(
            {"world_rank": int(self.rank), "measurement": result.measurement}
        )

    def release_profiling_cache_runtime(self):
        if self.cache_runtime_phase != "profiling":
            raise RuntimeError(
                "Profiling cache runtime can be released only once, got "
                f"phase={self.cache_runtime_phase!r}."
            )
        self.platform.synchronize()
        self.reset_after_warmup()
        release_unused_device_memory(self.platform)
        pre_graph_release_snapshot = DeviceMemorySnapshot.capture(
            self.platform,
            self.device,
        )
        self.decode_graph_runner.clear_captured_graphs()
        self.dp_idle_graphs.clear()
        if self.config.decode_graph:
            self.collective_runtime.reset_for_cuda_graph_recapture()
        release_unused_device_memory(self.platform)
        post_graph_release_snapshot = DeviceMemorySnapshot.capture(
            self.platform,
            self.device,
        )
        release_model_bindings = getattr(
            self.model,
            "release_cache_runtime_bindings",
            None,
        )
        if callable(release_model_bindings):
            release_model_bindings(self.cache_manager)
        model = getattr(self.model, "model", None)
        if model is not None and getattr(model, "sparse_controller", None) is self.sparse_controller:
            model.sparse_controller = None
        self.decode_graph_runner = None
        self.sparse_controller = None
        self.runtime_state = None
        self.prefix_cache_coordinator = None
        self.chain_cache_coordinator = None
        self.cache_manager = None
        release_unused_device_memory(self.platform)
        self.cache_runtime_phase = "released"
        snapshot = DeviceMemorySnapshot.capture(self.platform, self.device)
        return self._gather_startup_record(
            {
                "world_rank": int(self.rank),
                "snapshot": snapshot,
                "pre_graph_release_snapshot": pre_graph_release_snapshot,
                "post_graph_release_snapshot": post_graph_release_snapshot,
                "profiling_kv_budget_bytes": int(self.profiling_kv_budget_bytes),
                "runtime_build": self.cache_runtime_build_measurement,
            }
        )

    def build_production_cache_runtime(self, allocation_budget_bytes: int):
        if self.cache_runtime_phase != "released":
            raise RuntimeError(
                "Production cache runtime requires a released profiling runtime, got "
                f"phase={self.cache_runtime_phase!r}."
            )
        release_unused_device_memory(self.platform)
        self._build_cache_runtime(
            self.config,
            allocation_budget_bytes=int(allocation_budget_bytes),
        )
        self.platform.synchronize()
        self.cache_runtime_phase = "production"
        return self._gather_startup_record(
            {
                "world_rank": int(self.rank),
                "num_kvcache_slots": self.config.num_kvcache_slots,
            }
        )

    def capture_startup_memory_snapshot(self):
        release_unused_device_memory(self.platform)
        snapshot = DeviceMemorySnapshot.capture(self.platform, self.device)
        return self._gather_startup_record(
            {
                "world_rank": int(self.rank),
                "snapshot": snapshot,
            }
        )

    def resolve_startup_decode_graph_plan(
        self,
        startup_plan: list[tuple[int, int]],
    ):
        feasible, skipped = feasible_startup_graph_plan(
            self.config,
            startup_plan,
            self.runtime_state,
        )
        return self._gather_startup_record(
            {
                "world_rank": int(self.rank),
                "feasible": feasible,
                "skipped": skipped,
            }
        )

    def startup_capture_prefill_fits(self, prompt_len: int):
        return self._gather_startup_record(
            {
                "world_rank": int(self.rank),
                "fits": self.runtime_state.startup_batch_fits(
                    (int(prompt_len),), max_tokens=2,
                ),
            }
        )

    def startup_capture_decode_fits(self, seqs: list[Sequence]):
        return self._gather_startup_record(
            {
                "world_rank": int(self.rank),
                "fits": self.runtime_state.startup_decode_batch_fits(seqs),
            }
        )

    def arm_runtime_compilation_guard(self):
        from sparseengine.utils.compilation_guard import RuntimeCompilationGuard

        if getattr(self, "_compilation_guard", None) is None:
            self._compilation_guard = RuntimeCompilationGuard(
                self.config.runtime_compilation_limit, self.rank,
            )
        self._compilation_guard.arm()

    def exit(self):
        """释放资源并注销分布式进程组"""
        guard = getattr(self, "_compilation_guard", None)
        if guard is not None:
            guard.close()
        self.platform.set_device(self.device)
        # Graph replay is asynchronous on every rank. Drain and release captured
        # NCCL work before entering the shutdown barrier or destroying its group.
        self.platform.synchronize()
        self.dp_idle_graphs.clear()
        if self.config.decode_graph and self.decode_graph_runner is not None:
            self.decode_graph_runner.clear_captured_graphs()
            self.platform.synchronize()
        close_runtime_operators = getattr(self.model, "close_runtime_operators", None)
        if callable(close_runtime_operators):
            close_runtime_operators()
            self.platform.synchronize()
        self.collective_runtime.close()
        self.platform.synchronize()
        close_workspace_manager()
        if self.parallel_context.attn_tp_size > 1:
            self.shm.close()
            self.parallel_context.attn_tp.barrier(
                device_ids=self.platform.barrier_device_ids(self.rank)
            )
            if self.parallel_context.attn_tp_rank == 0:
                self.shm.unlink()
        reset_parallel_context()
        dist.destroy_process_group()

    def loop(self):
        """子进程的主循环：监听共享内存，解析并执行来自 Rank 0 的方法指令"""
        while True:
            method_name, args = self.read_shm()
            try:
                self.call(method_name, *args)
            except Exception as exc:
                if method_name in RECOVERABLE_TP_CONTROL_RPC_METHODS:
                    logger.error(
                        "TP worker recoverable control RPC {} failed: {}: {}",
                        method_name,
                        type(exc).__name__,
                        exc,
                    )
                else:
                    raise
            if method_name == "exit":
                break

    def read_shm(self):
        """反序列化共享内存中的方法名和参数"""
        assert self.parallel_context.attn_tp_size > 1 and self.parallel_context.attn_tp_rank > 0
        command_event, _ = self.event
        command_event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        command_event.clear()
        return method_name, args

    @cpu_timing.timed
    def write_shm(self, method_name, *args, wait_for_read: bool = True):
        """序列化方法名 and 参数并写入共享内存"""
        assert self.parallel_context.attn_tp_size > 1 and self.parallel_context.attn_tp_rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        command_capacity = len(self.shm.buf) - self.parallel_context.attn_tp_size
        if n + 4 > command_capacity:
            raise _RPCPayloadTooLarge(
                f"Shared memory command is too large: {n + 4} > {command_capacity}"
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for rank, (command_event, completion_event) in enumerate(self.event, start=1):
            completion_event.clear()
            self.shm.buf[self._run_status_offset(rank)] = TP_RUN_STATUS_PENDING
            command_event.set()
        if not wait_for_read:
            return
        timeout_s = float(os.getenv("SPARSEENGINE_TP_RPC_ACK_TIMEOUT_S", "30"))
        deadline = time.monotonic() + timeout_s
        for command_event, _ in self.event:
            while command_event.is_set():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for TP worker to read shared-memory RPC "
                        f"{method_name!r} after {timeout_s:.1f}s."
                    )
                time.sleep(0.0001)

    def call(self, method_name, *args):
        """RPC 风格的调用：如果是 Rank 0 则先广播指令，然后所有进程执行本地逻辑"""
        synchronizes_status = method_name in TP_RPC_STATUS_SYNC_METHODS
        if self.parallel_context.attn_tp_size > 1 and self.parallel_context.attn_tp_rank == 0:
            # A status-synchronized RPC already waits for every worker before
            # the shared command buffer can be reused.  Let rank 0 begin its
            # local work immediately instead of polling for a separate read ACK.
            try:
                self.write_shm(
                    method_name,
                    *args,
                    wait_for_read=not synchronizes_status,
                )
            except _RPCPayloadTooLarge:
                # Split only an unpublished read-only lookup batch. Never retry
                # a command that may have already run on any rank.
                if method_name != "refresh_prefix_cache_hits":
                    raise
                seqs = args[0]
                first, last = args[1] if len(args) > 1 else (True, True)
                if len(seqs) == 1:
                    return [self.call("refresh_prefix_cache_hit", seqs[0], (first, last))]
                if not seqs:
                    raise
                midpoint = len(seqs) // 2
                return (
                    self.call(method_name, seqs[:midpoint], (first, False))
                    + self.call(method_name, seqs[midpoint:], (False, last))
                )
        method = getattr(self, method_name, None)
        # Ensure *all* runner-side ops (including sparse post-processing like DeltaKV eviction)
        # run without autograd bookkeeping to avoid large activation graphs / OOM.
        if synchronizes_status:
            local_error: BaseException | None = None
            result = None
            try:
                with torch.inference_mode():
                    result = method(*args)
            except BaseException as exc:
                local_error = exc
            if method_name in {"submit_async", "collect_async", "retire_async"}:
                self._sync_tp_host_status(method_name, local_error, synchronize=False)
                if local_error is not None:
                    raise local_error
                if (
                    method_name == "retire_async"
                    and self.parallel_context.attn_tp_rank == 0
                ):
                    # Host-status workers only know their local outcome. Commit
                    # release idempotency state after rank 0 observes all ranks.
                    self.call("commit_slot_releases", args[0])
                return result
            if method_name == "refresh_prefix_cache_hits":
                self._sync_prefix_cache_batch_result(result, local_error)
                return result
            if method_name == "run" and self.config.decode_graph:
                self._sync_tp_run_status(local_error)
            elif method_name in (
                DECODE_GRAPH_HOST_STATUS_SYNC_METHODS
                | STARTUP_HOST_STATUS_SYNC_METHODS
            ):
                self._sync_tp_host_status(method_name, local_error)
            else:
                self._sync_tp_rpc_status(method_name, local_error)
            if local_error is not None:
                raise local_error
            if method_name == "refresh_prefix_cache_hit":
                self._sync_prefix_cache_lookup_result(result)
            elif method_name.startswith("chain_"):
                self._sync_chain_cache_result(method_name, result)
            if method_name in {"free_slots", "free_slots_batch", "finish_slots_batch"}:
                seq_ids = [int(args[0])] if method_name == "free_slots" else [int(v) for v in args[0]]
                self.commit_slot_releases(seq_ids)
            return result
        with torch.inference_mode():
            return method(*args)

    def _run_status_offset(self, rank: int) -> int:
        if not 0 < rank < self.parallel_context.attn_tp_size:
            raise ValueError(f"Invalid TP worker rank {rank} for attention TP={self.parallel_context.attn_tp_size}.")
        return len(self.shm.buf) - self.parallel_context.attn_tp_size + rank

    def _synchronize_tp_run_stream(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        else:
            self.platform.synchronize()

    def _sync_tp_run_status(self, local_error: BaseException | None) -> None:
        self._sync_tp_host_status("run", local_error)

    @cpu_timing.timed
    def _sync_tp_host_status(
        self,
        method_name: str,
        local_error: BaseException | None,
        *, synchronize: bool = True,
    ) -> None:
        if self.parallel_context.attn_tp_size <= 1:
            return

        sync_error: BaseException | None = None
        if local_error is None and synchronize:
            try:
                # Preserve the old all-reduce + item error boundary without
                # launching a per-token NCCL collective.
                self._synchronize_tp_run_stream()
            except BaseException as exc:
                sync_error = exc

        if self.parallel_context.attn_tp_rank > 0:
            _, completion_event = self.event
            status = (
                TP_RUN_STATUS_FAILED
                if local_error is not None or sync_error is not None
                else TP_RUN_STATUS_SUCCESS
            )
            self.shm.buf[self._run_status_offset(self.parallel_context.attn_tp_rank)] = status
            completion_event.set()
            if sync_error is not None:
                raise sync_error
            return

        timeout_s = float(os.getenv("SPARSEENGINE_TP_RPC_STATUS_TIMEOUT_S", "300"))
        deadline = time.monotonic() + timeout_s
        failed_ranks: list[int] = []
        for rank, (_, completion_event) in enumerate(self.event, start=1):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not completion_event.wait(timeout=remaining):
                raise TimeoutError(
                    f"Timed out waiting for TP worker {rank} to complete {method_name!r} "
                    f"after {timeout_s:.1f}s."
                )
            status = int(self.shm.buf[self._run_status_offset(rank)])
            completion_event.clear()
            if status == TP_RUN_STATUS_FAILED:
                failed_ranks.append(rank)
            elif status != TP_RUN_STATUS_SUCCESS:
                raise RuntimeError(
                    f"TP worker {rank} returned invalid run status {status}."
                )

        if sync_error is not None:
            raise sync_error
        if failed_ranks and local_error is None:
            ranks = ", ".join(str(rank) for rank in failed_ranks)
            raise RuntimeError(
                f"TP worker rank(s) {ranks} failed during {method_name}."
            )

    @cpu_timing.timed
    def _sync_tp_rpc_status(
        self,
        method_name: str,
        local_error: BaseException | None,
    ) -> None:
        if self.parallel_context.attn_tp_size <= 1 or not dist.is_initialized():
            return
        failed = torch.tensor(
            [1 if local_error is not None else 0],
            dtype=torch.int32,
            device=self.device,
        )
        self.parallel_context.attn_tp.all_reduce(failed, op=dist.ReduceOp.MAX)
        if int(failed.item()) != 0 and local_error is None:
            raise RuntimeError(f"At least one attention TP worker failed during {method_name}.")

    def _sync_prefix_cache_control_rpc_status(
        self,
        method_name: str,
        local_error: BaseException | None,
    ) -> None:
        self._sync_tp_rpc_status(method_name, local_error)

    def _sync_prefix_cache_lookup_result(self, local_result: dict[str, object]) -> None:
        if self.parallel_context.attn_tp_size <= 1:
            return
        results = [None] * self.parallel_context.attn_tp_size
        dist.all_gather_object(
            results,
            local_result,
            group=self.parallel_context.attn_tp.process_group,
        )
        if any(result != results[0] for result in results[1:]):
            raise RuntimeError(
                "Prefix-cache lookup diverged across attention TP ranks: "
                f"results={results!r}."
            )

    def _sync_prefix_cache_batch_result(
        self,
        result: list[dict[str, object]] | None,
        local_error: BaseException | None,
    ) -> None:
        # Gather error status and ordered lookup results together. All ranks
        # enter the same collective even if one failed partway through a batch.
        if self.parallel_context.attn_tp_size > 1:
            payload = (
                None if local_error is None else f"{type(local_error).__name__}: {local_error}",
                result,
            )
            results = [None] * self.parallel_context.attn_tp_size
            dist.all_gather_object(
                results, payload, group=self.tp_control_group,
            )
            failures = [
                (rank, value[0]) for rank, value in enumerate(results)
                if value[0] is not None
            ]
            if failures:
                if local_error is not None:
                    raise local_error
                raise RuntimeError(f"Attention TP prefix-cache batch lookup failed: {failures!r}.")
            if any(value[1] != results[0][1] for value in results[1:]):
                raise RuntimeError(
                    f"Prefix-cache lookup diverged across attention TP ranks: results={results!r}."
                )
        if local_error is not None:
            raise local_error

    def _sync_chain_cache_result(self, method_name: str, local_result) -> None:
        if self.parallel_context.attn_tp_size <= 1:
            return
        results = [None] * self.parallel_context.attn_tp_size
        dist.all_gather_object(
            results,
            local_result,
            group=self.parallel_context.attn_tp.process_group,
        )
        if any(result != results[0] for result in results[1:]):
            raise RuntimeError(
                f"Chain-cache {method_name} diverged across attention TP ranks: "
                f"results={results!r}."
            )

    def load_deltakv_compressors(self):
        """加载 DeltaKV 压缩器权重"""
        method = str(self.config.sparse_method or "")
        if not method.startswith('deltakv') or self.config.deltakv_checkpoint_path is None:
            return
        
        logger.info(f"Loading DeltaKV compressors from {self.config.deltakv_checkpoint_path}")
        from sparseengine.utils.loader import load_deltakv_compressors_to_cache_manager

        load_deltakv_compressors_to_cache_manager(self.cache_manager, self.config.deltakv_checkpoint_path)

    def reset_after_warmup(self) -> None:
        reset_after_warmup = getattr(self.runtime_state, "reset_after_warmup", None)
        if callable(reset_after_warmup):
            reset_after_warmup()
        else:
            reset_cache = getattr(self.cache_manager, "reset_after_warmup", None)
            if callable(reset_cache):
                reset_cache()
            else:
                reset_prefix_cache = getattr(self.cache_manager, "reset_prefix_cache", None)
                if callable(reset_prefix_cache):
                    reset_prefix_cache()

        if os.getenv("SPARSEENGINE_DELTAKV_CLEAR_GRAPHS_AFTER_WARMUP", "0") == "1":
            self.decode_graph_runner.clear_captured_graphs()
        if os.getenv("SPARSEENGINE_DELTAKV_CLEAR_ATTN_SCORE_BUFFERS_AFTER_WARMUP", "0") == "1":
            self.sparse_controller.clear_decode_attn_score_buffers()

    def log_operator_implementations(self) -> None:
        if self.parallel_context.world_rank == 0:
            operator_registry.log_operator_implementations()

    def operator_runtime_stats(self) -> list[dict[str, object]] | None:
        local_error: BaseException | None = None
        local_stats = None
        try:
            local_stats = {
                "world_rank": int(self.parallel_context.world_rank),
                "bindings": operator_registry.operator_binding_reports(),
                "operators": operator_registry.operator_runtime_stats(),
                "moe_communication": self.collective_runtime.moe_transport_stats(),
                "async_execution": self._async_execution.stats() if hasattr(self, "_async_execution") else None,
            }
        except BaseException as exc:
            local_error = exc
        self._sync_tp_rpc_status("operator_runtime_stats", local_error)
        if local_error is not None:
            raise local_error
        if self.parallel_context.attn_tp_size == 1:
            return [local_stats]
        stats = [None] * self.parallel_context.attn_tp_size
        dist.all_gather_object(
            stats,
            local_stats,
            group=self.parallel_context.attn_tp.process_group,
        )
        return stats if self.parallel_context.attn_tp_rank == 0 else None

    def warmup_moe_workspace(self, num_tokens: int) -> None:
        warmup_moe = getattr(self.model, "warmup_moe", None)
        if not callable(warmup_moe):
            raise RuntimeError(
                f"Model {type(self.model).__name__} does not provide warmup_moe()."
            )
        warmup_moe(num_tokens=int(num_tokens))

    def _assert_cache_release(self, seq_ids) -> None:
        # Terminal/control path only; never add a synchronization to decode.
        from sparseengine.platforms import device_runtime

        if device_runtime.is_stream_capturing():
            raise RuntimeError("Cache ownership cannot change during graph capture.")
        asynchronous = getattr(self, "_async_execution", None)
        if asynchronous is not None:
            asynchronous.assert_releasable(seq_ids)

    def free_slots(self, seq_id: int):
        """通知 CacheManager 释放该序列占用的物理显存位子"""
        self._assert_cache_release((seq_id,))
        with profiler.record("model_free_slots"):
            if os.getenv("SPARSEENGINE_DEBUG_SLOTS", "0") == "1":
                before = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots seq_id={} before={}", seq_id, before)
            self._release_runtime_slots_once(seq_id)
            if hasattr(self, "_async_execution"):
                self._async_execution.forget([seq_id])
            self.multimodal_runtime.free(seq_id)
            if os.getenv("SPARSEENGINE_DEBUG_SLOTS", "0") == "1":
                after = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots seq_id={} after={}", seq_id, after)

    def free_slots_batch(self, seq_ids: list[int]):
        """Release cache slots for a batch of finished/preempted sequences."""
        with profiler.record("model_free_slots_batch"):
            seq_ids = [int(seq_id) for seq_id in seq_ids]
            if not seq_ids:
                return
            self._assert_cache_release(seq_ids)
            if os.getenv("SPARSEENGINE_DEBUG_SLOTS", "0") == "1":
                before = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots_batch seq_ids={} before={}", seq_ids, before)
            for seq_id in seq_ids:
                self._release_runtime_slots_once(seq_id)
            if hasattr(self, "_async_execution"):
                self._async_execution.forget(seq_ids)
            if os.getenv("SPARSEENGINE_DEBUG_SLOTS", "0") == "1":
                after = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots_batch seq_ids={} after={}", seq_ids, after)

    def finish_slots_batch(self, seq_ids: list[int]):
        self.free_slots_batch(seq_ids)
        self.multimodal_runtime.free_batch(seq_ids)

    def _release_runtime_slots_once(self, seq_id: int) -> None:
        states = getattr(self, "_slot_release_states", None)
        if states is None:
            states = {}
            self._slot_release_states = states
        state = states.get(int(seq_id))
        if state == "released":
            return
        if state == "failed":
            raise RuntimeError(
                "Cache release previously failed on this worker after its mutation "
                f"boundary became unknown; restart the worker before reusing seq_id={seq_id}."
            )
        try:
            self.runtime_state.free_seq(int(seq_id))
        except BaseException:
            states[int(seq_id)] = "failed"
            raise
        states[int(seq_id)] = "released"

    def commit_slot_releases(self, seq_ids: list[int]) -> None:
        states = getattr(self, "_slot_release_states", None)
        if states is None:
            return
        for seq_id in seq_ids:
            states.pop(int(seq_id), None)

    def free_multimodal(self, seq_id: int):
        self.multimodal_runtime.free(seq_id)

    def register_multimodal_shared(
        self,
        seq_id: int,
        input_ids: list[int],
        shared_name: str,
        payload_size: int,
    ) -> int:
        payload_shm = SharedMemory(name=shared_name)
        try:
            tensors = pickle.loads(bytes(payload_shm.buf[: int(payload_size)]))
        finally:
            payload_shm.close()
        return self.multimodal_runtime.register(seq_id, input_ids, tensors)

    def chain_admission_plan(
        self,
        chain_id: str,
        seq_id: int,
        token_ids: list[int],
    ) -> ChainAdmissionPlan:
        return self.runtime_state.chain_admission_plan(
            chain_id,
            seq_id,
            token_ids,
        )

    def chain_apply_admission(
        self,
        plan: ChainAdmissionPlan,
    ) -> dict[str, object]:
        coordinator = self.runtime_state.chain_cache_coordinator
        seq_ids = [plan.seq_id]
        if coordinator is not None:
            seq_ids.extend(coordinator.index.lookup(cid).seq_id
                           for cid in (*plan.victim_chain_ids, *plan.demote_chain_ids))
        self._assert_cache_release(seq_ids)
        return self.runtime_state.chain_apply_admission(plan)

    def chain_validate_admission_plan(
        self,
        plan: ChainAdmissionPlan,
        input_token_count: int,
        input_prefix_digest: bytes,
    ) -> ChainAdmissionPlan:
        return self.runtime_state.chain_validate_admission_plan(
            plan,
            input_token_count,
            input_prefix_digest,
        )

    def chain_reclaim_idle(
        self, chain_id: str, expected_seq_id: int, demote: bool,
    ) -> dict[str, object]:
        self._assert_cache_release((expected_seq_id,))
        return self.runtime_state.chain_reclaim_idle(chain_id, expected_seq_id, demote)

    def chain_finish(
        self,
        chain_id: str,
        seq_id: int,
        processed_token_digest: bytes,
        processed_token_count: int,
    ) -> dict[str, object]:
        self._assert_cache_release((seq_id,))
        return self.runtime_state.chain_finish(
            chain_id,
            seq_id,
            processed_token_digest,
            processed_token_count,
        )

    def chain_invalidate(
        self,
        chain_id: str,
        expected_seq_id: int | None = None,
    ) -> dict[str, object]:
        coordinator = self.runtime_state.chain_cache_coordinator
        if coordinator is not None:
            self._assert_cache_release((coordinator.index.lookup(str(chain_id)).seq_id,))
        return self.runtime_state.chain_invalidate(
            chain_id,
            expected_seq_id=expected_seq_id,
        )

    def set_tokenizer_metadata(
        self,
        delimiter_token_ids: list[int],
        non_execution_token_ids: list[int] | None = None,
        auxiliary_prefill_token_ids: list[int] | None = None,
    ):
        self._auxiliary_prefill_token_ids = list(auxiliary_prefill_token_ids or [])
        self._tokenizer_metadata = (
            tuple(int(x) for x in delimiter_token_ids),
            (
                None
                if non_execution_token_ids is None
                else tuple(int(x) for x in non_execution_token_ids)
            ),
        )
        self._restore_tokenizer_metadata()

    def _restore_tokenizer_metadata(self) -> None:
        metadata = getattr(self, "_tokenizer_metadata", None)
        if metadata is None:
            return
        setter = getattr(self.sparse_controller, "set_tokenizer_metadata", None)
        if setter is not None:
            delimiter_token_ids, non_execution_token_ids = metadata
            setter(
                auxiliary_prefill_token_ids=getattr(self, "_auxiliary_prefill_token_ids", []),
                delimiter_token_ids=list(delimiter_token_ids),
                non_execution_token_ids=(
                    None
                    if non_execution_token_ids is None
                    else list(non_execution_token_ids)
                ),
            )

    def _run_auxiliary_prefill(self, request: AuxiliaryPrefillRequest) -> None:
        # Keep business sampling, token history and completion state untouched.
        seq = copy.copy(request.seq)
        seq.token_ids = list(request.token_ids)
        seq.num_tokens = seq.num_prompt_tokens = request.prefix_len + len(seq.token_ids)
        seq.num_prefilled_tokens = int(request.prefix_len)
        seq.current_chunk_size = len(request.token_ids)
        context = get_context()
        saved_context = vars(context).copy()
        try:
            input_ids, positions = self.prepare_step([seq], True)
            context.sparse_controller = self.sparse_controller
            self.sparse_controller.prepare_forward([seq], True)
            self.model(input_ids, positions)
        finally:
            vars(context).clear()
            vars(context).update(saved_context)

    def prefix_cache_inspect(
        self,
        token_ids: list[int],
        include_subtree: bool = False,
    ) -> dict[str, object]:
        return self.runtime_state.prefix_cache_inspect(
            [int(token_id) for token_id in token_ids],
            include_subtree=bool(include_subtree),
        )

    def refresh_prefix_cache_hit(
        self, seq: Sequence, lookup_batch: tuple[bool, bool] | None = None,
    ) -> dict[str, object]:
        if lookup_batch is not None:
            first, last = lookup_batch
            with self.runtime_state.prefix_cache_lookup_batch(first=first, last=last):
                return self.refresh_prefix_cache_hit(seq)
        self.runtime_state.refresh_prefix_cache_hit(seq)
        return {
            "enabled": bool(seq.prefix_cache_enabled),
            "hit_len": int(seq.prefix_cache_hit_len),
            "hit_block_count": int(seq.prefix_cache_hit_block_count),
            "hit_last_block_id": seq.prefix_cache_hit_last_block_id,
            "block_size": int(seq.prefix_cache_block_size),
            "method": str(seq.prefix_cache_method),
        }

    def refresh_prefix_cache_hits(
        self, seqs: list[Sequence], lookup_batch: tuple[bool, bool] = (True, True),
    ) -> list[dict[str, object]]:
        first, last = lookup_batch
        with self.runtime_state.prefix_cache_lookup_batch(first=first, last=last):
            return [self.refresh_prefix_cache_hit(seq) for seq in seqs]

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        return self.runtime_state.prefix_cache_match(
            [int(token_id) for token_id in token_ids],
        )

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        return self.runtime_state.prefix_cache_delete_subtree(
            [int(token_id) for token_id in token_ids],
        )

    def prefix_cache_set_eviction_priority(
        self,
        token_ids: list[int],
        priority: int,
    ) -> dict[str, object]:
        return self.runtime_state.prefix_cache_set_eviction_priority(
            [int(token_id) for token_id in token_ids],
            priority=int(priority),
        )

    @cpu_timing.timed
    def _prefix_prune_score_forward(
        self,
        *,
        token_ids: list[int],
        prefix_hit_len: int,
        protected_prefix_len: int,
        candidate_start: int,
        temp_seq_id: int,
        protected_token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        return self._prefix_prune_score_forward_batch([dict(
            token_ids=token_ids, prefix_hit_len=prefix_hit_len,
            protected_prefix_len=protected_prefix_len, candidate_start=candidate_start,
            temp_seq_id=temp_seq_id, protected_token_ids=protected_token_ids,
        )])[0]

    @cpu_timing.timed
    def _prefix_prune_score_forward_batch(self, requests) -> list[torch.Tensor]:
        manager = self.cache_manager
        prefix_cache = manager.prefix_cache
        block_size = int(manager.prefix_cache_block_size)
        if prefix_cache is None or block_size <= 0:
            raise RuntimeError("prefix cache is not enabled on this model worker.")
        asynchronous = getattr(self, "_async_execution", None)
        if asynchronous is not None:
            asynchronous.prepare_synchronous_execution()
        seqs, scoring, protected = [], [], {}
        for request in requests:
            token_ids = request["token_ids"]
            prefix_hit_len = request["prefix_hit_len"]
            protected_prefix_len = request["protected_prefix_len"]
            candidate_start = request["candidate_start"]
            temp_seq_id = request["temp_seq_id"]
            query_end = len(token_ids)
            if query_end > int(self.config.max_model_len):
                raise ValueError(
                    "prefix-prune scoring context exceeds max_model_len: "
                    f"context={query_end} max_model_len={self.config.max_model_len}. "
                    "Reduce observation_tokens, score_chunk_size, or prev_postfix_size."
                )
            if prefix_hit_len <= candidate_start or prefix_hit_len >= query_end:
                raise ValueError(
                    "prefix-prune score forward requires candidate tokens followed by queries: "
                    f"candidate_start={candidate_start} hit={prefix_hit_len} end={query_end}."
                )
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
            seq.num_prefilled_tokens = int(prefix_hit_len)
            seq.current_chunk_size = query_end - prefix_hit_len
            seq.prefix_cache_enabled = True
            seq.prefix_cache_hit_len = int(prefix_hit_len)
            seq.prefix_cache_hit_block_count = int(hit_blocks)
            seq.prefix_cache_hit_last_block_id = last_block_id
            seq.prefix_cache_block_size = block_size
            seq.prefix_cache_method = str(getattr(self.config, "sparse_method", "") or "")
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
            input_ids, positions = self.prepare_step(seqs, True)
            ctx = get_context()
            ctx.sparse_controller = self.sparse_controller
            self.sparse_controller.prepare_forward(seqs, True)
            self.model(input_ids, positions)
            self.sparse_controller.post_forward(seqs, True)
            return manager.finish_prefix_prune_scoring_batch()
        finally:
            manager.abort_prefix_prune_scoring()
            reset_context()
            for seq in seqs:
                if seq.seq_id in manager.seq_id_to_row:
                    self.runtime_state.free_seq(seq.seq_id)
            for block in acquired:
                prefix_cache.release_block_ref(block)

    @torch.no_grad()
    @cpu_timing.timed
    def prefix_cache_prune(
        self,
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
    ) -> dict[str, object]:
        token_ids = [int(token_id) for token_id in token_ids]
        block_size = int(self.config.prefix_cache_block_size)
        intervals = normalize_prefix_prune_ranges(
            token_count=len(token_ids), block_size=block_size,
            range_start=range_start, range_end=range_end, ranges=ranges,
        )
        validate_prefix_prune_request(
            token_count=len(token_ids), block_size=block_size, ranges=intervals,
            keep_tokens=keep_tokens, policy=policy,
        )
        self.cache_manager.validate_prefix_prune_keep_tokens(keep_tokens)
        self.cache_manager.validate_prefix_cache_prune_target(
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
            score = self._prefix_prune_score_forward(
                token_ids=token_ids[:range_end], prefix_hit_len=query_start,
                protected_prefix_len=range_end, candidate_start=range_start,
                temp_seq_id=temp_seq_id, protected_token_ids=token_ids,
            )
            # Pack only requested tokens before the collective and selection.
            packed = torch.cat([score[left:right] for left, right in intervals])
            self.parallel_context.world.all_reduce(packed, op=dist.ReduceOp.MAX)
            # Sorted disjoint intervals put all protected positions at the end.
            keep_indices = self.cache_manager.select_prefix_prune_keep_indices(
                packed,
                keep_tokens=keep_tokens,
                protected_suffix_tokens=protected_count,
            )
        elif policy == "kvzip_global":
            if keep_tokens == 0:
                # The mask is known without reconstruction when everything is dropped.
                keep_indices = torch.empty(0, dtype=torch.long, device=self.device)
            else:
                replay_prefix = [int(token_id) for token_id in (kvzip_replay_prefix_ids or [])]
                if not replay_prefix:
                    raise ValueError("KVzip prefix pruning requires non-empty replay prompt token ids.")
                score_chunk_size = max(1, int(score_chunk_size))
                prev_postfix_size = max(0, int(prev_postfix_size))
                aggregate = torch.zeros((width,), dtype=torch.float32, device=self.device)
                chunk_number = 0
                cached_prefix = token_ids[:range_end]
                for left, right in intervals:
                    for start in range(left, right, score_chunk_size):
                        end = min(right, start + score_chunk_size)
                        previous = token_ids[max(left, start - prev_postfix_size) : start]
                        replay_ids = replay_prefix + previous + token_ids[start:end]
                        step_score = self._prefix_prune_score_forward(
                            token_ids=cached_prefix + replay_ids,
                            prefix_hit_len=range_end, protected_prefix_len=range_end,
                            candidate_start=range_start, temp_seq_id=temp_seq_id - chunk_number,
                            protected_token_ids=token_ids,
                        )
                        packed = torch.cat([step_score[l:r] for l, r in intervals])
                        torch.maximum(aggregate, packed, out=aggregate)
                        chunk_number += 1
                self.parallel_context.world.all_reduce(aggregate, op=dist.ReduceOp.MAX)
                keep_indices = self.cache_manager.select_prefix_prune_keep_indices(
                    aggregate, keep_tokens=keep_tokens,
                )
        else:
            raise ValueError(f"unsupported prefix prune policy: {policy!r}.")

        return self.cache_manager.prefix_cache_prune(
            token_ids, ranges=intervals, keep_indices=keep_indices,
            policy=policy, prune_id=prune_id, allow_recompress=allow_recompress,
        )

    @torch.no_grad()
    @cpu_timing.timed
    def prefix_cache_prune_batch(self, jobs, replay_prefix):
        """Score independent jobs round-robin, filling spare rows with more chunks."""
        from collections import deque

        manager = self.cache_manager
        cache = manager.prefix_cache
        block_size = int(self.config.prefix_cache_block_size)
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
                prefix = job["token_ids"][:end]
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
                    if end + max_query > self.config.max_model_len:
                        raise ValueError("Prefix-prune scoring context exceeds max_model_len.")
                state = dict(index=index, job=job, intervals=intervals, prefix=prefix,
                             prefix_blocks=chain[:end // block_size],
                             chunks=chunks, forwards=0, max_batch=0,
                             aggregate=torch.zeros(sum(r-l for l,r in intervals),
                                                   dtype=torch.float32, device=self.device))
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
            token_limit = min(self.config.max_num_batched_tokens,
                              self.config.engine_prefill_chunk_size or self.config.max_num_batched_tokens)
            while pending:
                requests, owners = [], []
                query_tokens = 0
                while pending and len(requests) < self.config.max_num_seqs_in_batch:
                    owner = pending[0]
                    prev, stop = owner["chunks"][0]
                    replay = replay_prefix + owner["job"]["token_ids"][prev:stop]
                    if requests and query_tokens + len(replay) > token_limit:
                        break
                    pending.popleft()
                    owner["chunks"].popleft()
                    end = len(owner["prefix"])
                    requests.append(dict(token_ids=owner["prefix"] + replay,
                                         prefix_hit_len=end, protected_prefix_len=end,
                                         candidate_start=owner["intervals"][0][0],
                                         temp_seq_id=next_seq_id))
                    next_seq_id -= 1
                    owners.append((owner, (prev, stop)))
                    query_tokens += len(replay)
                    if owner["chunks"]:
                        pending.append(owner)
                reservation = [(r["temp_seq_id"], len(r["token_ids"]) - r["prefix_hit_len"])
                               for r in requests]
                prefix_blocks = {r["temp_seq_id"]: owner["prefix_blocks"]
                                 for r, (owner, _) in zip(requests, owners, strict=True)}
                with manager.reserve_prefill_slots(reservation, prefix_blocks=prefix_blocks) as admitted:
                    # Return unadmitted chunks in their original per-task order.
                    for owner, chunk in reversed(owners[admitted:]):
                        owner["chunks"].appendleft(chunk)
                        if not any(item is owner for item in pending):
                            pending.appendleft(owner)
                    requests = requests[:admitted]
                    owners = [owner for owner, _ in owners[:admitted]]
                    scores = self._prefix_prune_score_forward_batch(requests)
                for owner, score in zip(owners, scores, strict=True):
                    packed = torch.cat([score[l:r] for l, r in owner["intervals"]])
                    torch.maximum(owner["aggregate"], packed, out=owner["aggregate"])
                    owner["forwards"] += 1
                    owner["max_batch"] = max(owner["max_batch"], len(requests))
        finally:
            for block in acquired:
                cache.release_block_ref(block)
        for state in states:
            job = state["job"]
            aggregate = state["aggregate"]
            if job["keep_tokens"]:
                self.parallel_context.world.all_reduce(aggregate, op=dist.ReduceOp.MAX)
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

    def debug_sparse_state_summary(self) -> dict[str, object]:
        def parallel_group_summary(group) -> dict[str, object] | None:
            if group is None:
                return None
            return {
                "rank": int(group.rank),
                "size": int(group.size),
                "ranks": [int(rank) for rank in group.ranks],
            }

        parallel_context = self.parallel_context
        config = getattr(self, "config", None)
        moe_synced = {}
        moe_local = {}
        model = getattr(getattr(self, "model", None), "model", None)
        layers = getattr(model, "layers", ())
        selected_layers = {0, len(layers) - 1} if layers else set()
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in selected_layers:
                continue
            block = getattr(layer, "mlp", None)
            topk_ids = getattr(block, "debug_last_topk_ids", None)
            topk_weights = getattr(block, "debug_last_topk_weights", None)
            output = getattr(block, "debug_last_output", None)
            if topk_ids is None or topk_weights is None or output is None:
                continue
            moe_synced[str(layer_idx)] = {
                "topk_ids": _debug_tensor_summary(topk_ids),
                "topk_weights": _debug_tensor_summary(topk_weights),
                "output": _debug_tensor_summary(output),
            }
            experts = block.experts
            moe_local[str(layer_idx)] = {
                "local_expert_start": int(experts.local_expert_start),
                "local_expert_end": int(experts.local_expert_end),
                "local_hit_count": int(block.debug_last_local_hit_count),
                "local_output": _debug_tensor_summary(block.debug_last_local_output),
            }
        state = self.sparse_controller.debug_state_summary()
        prefix_cache_coordinator = getattr(
            self,
            "prefix_cache_coordinator",
            None,
        )
        if prefix_cache_coordinator is not None:
            state["mixed_prefix_cache"] = (
                prefix_cache_coordinator.debug_state_summary()
            )
        graph_runner = getattr(self, "decode_graph_runner", None)
        graph_key = getattr(graph_runner, "last_state_key", None)
        graph_summary = {
            "dp_idle_graph_count": len(getattr(self, "dp_idle_graphs", {})),
            "dp_idle_replay_count": int(getattr(self, "dp_idle_replay_count", 0)),
            "dp_idle_eager_count": int(getattr(self, "dp_idle_eager_count", 0)),
            "enabled": bool(
                getattr(config, "decode_graph", False)
            ),
            "capture_count": int(
                getattr(graph_runner, "capture_count", 0)
            ),
            "replay_count": int(
                getattr(graph_runner, "replay_count", 0)
            ),
            "eager_static_count": int(
                getattr(graph_runner, "eager_static_count", 0)
            ),
            "force_eager_count": int(
                getattr(graph_runner, "force_eager_count", 0)
            ),
            "eviction_count": int(
                getattr(graph_runner, "eviction_count", 0)
            ),
            "recapture_count": int(
                getattr(graph_runner, "recapture_count", 0)
            ),
            "cached_graph_count": len(
                getattr(graph_runner, "_graphs", {})
            ),
            "graph_plan": (
                graph_runner.graph_plan()
                if callable(getattr(graph_runner, "graph_plan", None))
                else None
            ),
            "last_state_key": (
                {
                    "method": str(graph_key.method or ""),
                    "batch_size": int(graph_key.batch_size),
                    "graph_path_id": str(graph_key.graph_path_id),
                    "capture_sampling": bool(graph_key.capture_sampling),
                }
                if graph_key is not None
                else None
            ),
        }
        return {
            "world_rank": self.parallel_context.world_rank,
            "ep_rank": self.parallel_context.moe_ep_rank,
            "parallel": {
                "configured": {
                    "tensor_parallel_size": int(
                        getattr(config, "tensor_parallel_size", parallel_context.attn_tp_size)
                    ),
                    "expert_parallel_size": int(
                        getattr(config, "expert_parallel_size", parallel_context.moe_ep_size)
                    ),
                    "data_parallel_size": int(
                        getattr(config, "data_parallel_size", parallel_context.attn_dp_size)
                    ),
                    "world_size": int(
                        getattr(config, "world_size", parallel_context.world_size)
                    ),
                },
                "effective": {
                    "world": parallel_group_summary(parallel_context.world),
                    "attn_tp": parallel_group_summary(parallel_context.attn_tp),
                    "moe_ep": parallel_group_summary(parallel_context.moe_ep),
                    "moe_tp": parallel_group_summary(
                        parallel_context.moe_tp
                    ),
                    "attn_dp": parallel_group_summary(parallel_context.attn_dp),
                },
            },
            "state": state,
            "decode_graph": graph_summary,
            "last_logits": (
                _debug_tensor_summary(self.debug_last_logits)
                if hasattr(self, "debug_last_logits")
                else None
            ),
            "moe_synced": moe_synced,
            "moe_local": moe_local,
        }

    def debug_last_logits_cpu(self) -> torch.Tensor | None:
        if self.parallel_context.attn_tp_rank != 0:
            return None
        logits = getattr(self, "debug_last_logits", None)
        if logits is None:
            raise RuntimeError(
                "No debug logits are available. Set SPARSEENGINE_DEBUG_RUNTIME=1 before engine startup."
            )
        return logits.detach().cpu()

    def debug_hidden_states_cpu(self) -> dict[int, torch.Tensor] | None:
        model = getattr(getattr(self, "model", None), "model", None)
        snapshots = getattr(model, "debug_last_hidden_states", None)
        if snapshots is None:
            raise RuntimeError(
                "No hidden-state snapshots are available. Set "
                "SPARSEENGINE_DEBUG_HIDDEN_LAYERS before model execution."
            )
        if self.parallel_context.attn_tp_rank != 0:
            return None
        return {
            int(layer_idx): tensor.detach().cpu()
            for layer_idx, tensor in snapshots.items()
        }

    def debug_moe_states_cpu(self) -> dict[int, dict[str, torch.Tensor]] | None:
        model = getattr(getattr(self, "model", None), "model", None)
        layers = getattr(model, "layers", ())
        snapshots = {}
        for layer_idx, layer in enumerate(layers):
            block = getattr(layer, "mlp", None)
            if block is None or not hasattr(block, "experts"):
                continue
            required = {
                "input": getattr(block, "debug_last_input", None),
                "topk_ids": getattr(block, "debug_last_topk_ids", None),
                "topk_weights": getattr(block, "debug_last_topk_weights", None),
                "output": getattr(block, "debug_last_output", None),
            }
            missing = [name for name, tensor in required.items() if tensor is None]
            if missing:
                raise RuntimeError(
                    f"Layer {layer_idx} is missing MoE debug tensors {missing}. Set "
                    "SPARSEENGINE_DEBUG_MOE before model execution."
                )
            snapshots[layer_idx] = {
                name: tensor.detach().cpu()
                for name, tensor in required.items()
            }
        return snapshots if self.parallel_context.attn_tp_rank == 0 else None

    def _debug_float_error_from_tp_leader(
        self,
        tensor: torch.Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> tuple[float, float]:
        if self.parallel_context.attn_tp_size == 1:
            return 0.0, 0.0
        reference = tensor.detach().clone()
        dist.broadcast(
            reference,
            src=self.parallel_context.attn_tp.ranks[0],
            group=self.parallel_context.attn_tp.process_group,
        )
        difference = (tensor.detach().float() - reference.float()).abs()
        max_abs = difference.max()
        tolerance_ratio = (
            difference / (float(atol) + float(rtol) * reference.float().abs())
        ).max()
        self.parallel_context.attn_tp.all_reduce(max_abs, op=dist.ReduceOp.MAX)
        self.parallel_context.attn_tp.all_reduce(tolerance_ratio, op=dist.ReduceOp.MAX)
        return float(max_abs.item()), float(tolerance_ratio.item())

    def _debug_any_mismatch_from_tp_leader(self, tensor: torch.Tensor) -> bool:
        if self.parallel_context.attn_tp_size == 1:
            return False
        reference = tensor.detach().clone()
        dist.broadcast(
            reference,
            src=self.parallel_context.attn_tp.ranks[0],
            group=self.parallel_context.attn_tp.process_group,
        )
        mismatch = torch.tensor(
            [int(not torch.equal(tensor.detach(), reference))],
            dtype=torch.int32,
            device=self.device,
        )
        self.parallel_context.attn_tp.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        return bool(mismatch.item())

    def debug_replica_consistency(self) -> dict[str, object] | None:
        logits = getattr(self, "debug_last_logits", None)
        if self.parallel_context.attn_tp_size > 1:
            result: dict[str, object] = {
                "last_logits_max_abs": None,
                "last_logits_tolerance_ratio": None,
                "last_logits_comparison": "not_applicable_tp_vocab_sharded",
                "moe_layers": {},
            }
        else:
            if logits is None:
                return None
            logits_max_abs, logits_tolerance_ratio = (
                self._debug_float_error_from_tp_leader(
                    logits,
                    atol=0.05,
                    rtol=0.05,
                )
            )
            result = {
                "last_logits_max_abs": logits_max_abs,
                "last_logits_tolerance_ratio": logits_tolerance_ratio,
                "last_logits_comparison": "compared",
                "moe_layers": {},
            }
        model = getattr(getattr(self, "model", None), "model", None)
        layers = getattr(model, "layers", ())
        for layer_idx in sorted({0, len(layers) - 1} if layers else set()):
            block = getattr(layers[layer_idx], "mlp", None)
            if not hasattr(block, "debug_last_topk_ids"):
                continue
            topk_weights_max_abs, topk_weights_tolerance_ratio = (
                self._debug_float_error_from_tp_leader(
                    block.debug_last_topk_weights,
                    atol=0.01,
                    rtol=0.01,
                )
            )
            output_max_abs, output_tolerance_ratio = (
                self._debug_float_error_from_tp_leader(
                    block.debug_last_output,
                    atol=0.05,
                    rtol=0.05,
                )
            )
            result["moe_layers"][str(layer_idx)] = {
                "topk_ids_mismatch": self._debug_any_mismatch_from_tp_leader(
                    block.debug_last_topk_ids
                ),
                "topk_weights_max_abs": topk_weights_max_abs,
                "topk_weights_tolerance_ratio": topk_weights_tolerance_ratio,
                "output_max_abs": output_max_abs,
                "output_tolerance_ratio": output_tolerance_ratio,
            }
        return result

    def debug_sparse_state_summaries(self, synchronize: bool = False) -> list[dict[str, object]] | None:
        local_error: BaseException | None = None
        local_summary = None
        try:
            if synchronize:
                self.platform.synchronize()
            local_summary = self.debug_sparse_state_summary()
            local_summary["replica_consistency"] = (
                self.debug_replica_consistency()
            )
        except BaseException as exc:
            local_error = exc
        self._sync_tp_rpc_status("debug_sparse_state_summaries", local_error)
        if local_error is not None:
            raise local_error
        if self.parallel_context.attn_tp_size == 1:
            return [local_summary]
        summaries = [None] * self.parallel_context.attn_tp_size
        dist.all_gather_object(
            summaries,
            local_summary,
            group=self.parallel_context.attn_tp.process_group,
        )
        return summaries if self.parallel_context.attn_tp_rank == 0 else None

    @cpu_timing.timed
    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        """准备前向上下文并设置 Context"""
        input_ids, positions, cu_seqlens_q = self.runtime_state.prepare_step(seqs, is_prefill)
        set_context(
            is_prefill,
            cu_seqlens_q=cu_seqlens_q,
            cache_manager=self.cache_manager,
            seqs=seqs,
            recurrent_state_manager=self.recurrent_state_manager,
        )
        (
            self._prefill_inputs_embeds,
            positions,
            self._prefill_multimodal_mask,
        ) = self.multimodal_runtime.prepare(seqs, input_ids, positions, is_prefill)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样超参数"""
        size = len(seqs)
        buffers = getattr(self, "_sampling_buffers", None)
        upload_done = getattr(self, "_sampling_upload_done", None)
        if getattr(self, "_async_submitting", False):
            buffers = None
            upload_done = None
        if upload_done is not None:
            upload_done.synchronize()
        if buffers is None or buffers[0].shape[1] < size:
            capacity = max(1, 1 << (size - 1).bit_length())
            pin_memory = torch.device(self.device).type == "cuda" and self.platform.supports_pin_memory()
            host_float = torch.zeros((2, capacity), dtype=torch.float32, pin_memory=pin_memory)
            host_int = torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory)
            buffers = (
                host_float, host_int,
                torch.empty_like(host_float, device=self.device),
                torch.empty_like(host_int, device=self.device),
            )
            self._sampling_buffers = buffers
        host_float, host_int, device_float, device_int = buffers
        host_float.numpy()[0, :size] = [seq.temperature for seq in seqs]
        host_float.numpy()[1, :size] = [seq.top_p for seq in seqs]
        host_int.numpy()[:size] = [seq.top_k for seq in seqs]
        non_blocking = host_float.is_pinned()
        device_float.copy_(host_float, non_blocking=non_blocking)
        device_int[:size].copy_(host_int[:size], non_blocking=non_blocking)
        if non_blocking:
            # The next call may overwrite host storage only after DMA completes.
            if upload_done is None:
                upload_done = torch.cuda.Event()
                self._sampling_upload_done = upload_done
            upload_done.record(torch.cuda.current_stream(self.device))
        if getattr(self, "_async_submitting", False):
            self._async_execution.keepalive.extend(buffers)
        return device_float[0, :size], device_float[1, :size], device_int[:size]

    def _auto_capture_greedy_sampling(self, seqs: list[Sequence]) -> bool:
        if any(self._has_sampling_penalty(seq) for seq in seqs):
            return False
        if not self.config.decode_graph_capture_sampling:
            return False
        if self.config.tensor_parallel_size != 1:
            return False
        if self.config.enable_prefix_caching:
            return False
        return all(
            bool(getattr(seq, "should_publish_sample", True))
            and getattr(seq, "benchmark_forced_token_ids", None) is None
            and seq.temperature <= 1e-10
            for seq in seqs
        )

    @staticmethod
    def _has_sampling_penalty(seq: Sequence) -> bool:
        return (
            float(getattr(seq, "presence_penalty", 0.0)) != 0.0
            or float(getattr(seq, "repetition_penalty", 1.0)) != 1.0
        )

    @cpu_timing.timed
    def _apply_sampling_penalties(
        self,
        logits: torch.Tensor,
        seqs: list[Sequence],
    ) -> torch.Tensor:
        if getattr(self, "_async_submitting", False):
            return self._async_execution.apply_penalties(logits, seqs)
        if not any(self._has_sampling_penalty(seq) for seq in seqs):
            return logits

        vocab_size = int(logits.shape[-1])
        presence_penalties = [
            float(getattr(seq, "presence_penalty", 0.0)) for seq in seqs
        ]
        repetition_penalties = [
            float(getattr(seq, "repetition_penalty", 1.0)) for seq in seqs
        ]
        presence_token_ids = [
            seq.presence_penalty_token_ids_tensor(
                device=logits.device,
                vocab_size=vocab_size,
            )
            if presence_penalty != 0.0
            else None
            for seq, presence_penalty in zip(seqs, presence_penalties)
        ]
        repetition_token_ids = [
            seq.repetition_penalty_token_ids_tensor(
                device=logits.device,
                vocab_size=vocab_size,
            )
            if repetition_penalty != 1.0
            else None
            for seq, repetition_penalty in zip(seqs, repetition_penalties)
        ]
        return self.sampler.apply_penalties(
            logits,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
            presence_token_ids=presence_token_ids,
            repetition_token_ids=repetition_token_ids,
        )

    @cpu_timing.timed
    def _sample_model_outputs(
        self,
        logits: torch.Tensor,
        seqs: list[Sequence],
        graph_token_ids: torch.Tensor | None = None,
        *,
        return_device_tokens: bool = False,
    ) -> list[int] | torch.Tensor:
        """Sample only outputs that belong to live generation.

        Recompute replay forwards rebuild KV for already accepted tokens. Their
        logits must neither advance the sampling RNG nor escape to serving.
        """
        publish_indices = [
            idx
            for idx, seq in enumerate(seqs)
            if bool(getattr(seq, "should_publish_sample", True))
        ]
        token_ids = [0] * len(seqs)
        if not publish_indices:
            return torch.zeros(len(seqs), dtype=torch.long, device=logits.device) if return_device_tokens else token_ids
        if graph_token_ids is not None:
            if any(getattr(seq, "benchmark_forced_token_ids", None) is not None for seq in seqs):
                raise RuntimeError("Benchmark forced tokens require sampling outside the CUDA Graph")
            if len(publish_indices) != len(seqs):
                raise RuntimeError(
                    "CUDA graph sampling cannot mix recompute replay and live outputs."
                )
            return graph_token_ids if return_device_tokens else [int(token_id) for token_id in graph_token_ids.tolist()]

        # Most batches publish every row. Advanced indexing would still copy
        # the entire vocabulary matrix and upload an index tensor each step.
        if len(publish_indices) == len(seqs):
            publish_seqs = seqs
            publish_logits = logits
        else:
            publish_seqs = [seqs[idx] for idx in publish_indices]
            publish_logits = logits[publish_indices]
        all_greedy = all(seq.temperature <= 1e-10 for seq in publish_seqs)
        temperatures = None
        top_ps = None
        top_ks = None
        if not all_greedy:
            temperatures, top_ps, top_ks = self.prepare_sample(publish_seqs)
        stochastic_seqs = [seq for seq in publish_seqs if seq.temperature > 1e-10]
        max_top_k = None
        if stochastic_seqs and all(
            seq.top_p == 1.0 and 0 < seq.top_k < publish_logits.shape[-1]
            for seq in stochastic_seqs
        ):
            max_top_k = max(seq.top_k for seq in stochastic_seqs)
        sampled = self.sampler(
            publish_logits,
            temperatures,
            top_ps,
            top_ks,
            all_greedy=all_greedy,
            max_top_k=max_top_k,
            all_unfiltered=not all_greedy and all(
                seq.top_p == 1.0 and (seq.top_k == 0 or seq.top_k >= publish_logits.shape[-1])
                for seq in stochastic_seqs
            ),
        )
        forced_indices = []
        forced_values = []
        for idx, seq in enumerate(publish_seqs):
            forced = getattr(seq, "benchmark_forced_token_ids", None)
            if forced is None:
                continue
            offset = int(seq.num_completion_tokens)
            if not 0 <= offset < len(forced):
                raise RuntimeError(
                    f"Forced-token cursor is outside the recorded completion: seq_id={seq.seq_id} offset={offset}"
                )
            value = int(forced[offset])
            if value >= int(logits.shape[-1]):
                raise ValueError(f"Forced token {value} is outside the model vocabulary")
            forced_indices.append(idx)
            forced_values.append(value)
        if forced_indices:
            sampled[forced_indices] = torch.tensor(
                forced_values, dtype=sampled.dtype, device=sampled.device
            )
        if return_device_tokens:
            if len(publish_indices) == len(seqs):
                return sampled
            result = torch.zeros(len(seqs), dtype=sampled.dtype, device=sampled.device)
            result[publish_indices] = sampled
            return result
        for idx, token_id in zip(publish_indices, sampled.tolist()):
            token_ids[idx] = int(token_id)
        return token_ids

    @staticmethod
    def _mask_recompute_logprobs(
        seqs: list[Sequence],
        outputs: tuple[list[float | None], list[dict[int, float] | None]] | None,
    ):
        if outputs is None:
            return None
        if isinstance(outputs, DeviceLogprobs):
            return outputs
        token_logprobs, top_logprobs = outputs
        if token_logprobs is None or top_logprobs is None:
            return outputs
        for idx, seq in enumerate(seqs):
            if not bool(getattr(seq, "should_publish_sample", True)):
                token_logprobs[idx] = None
                top_logprobs[idx] = None
        return token_logprobs, top_logprobs

    def seal_decode_cuda_graph_startup_plan(self):
        self.collective_runtime.mark_cuda_graph_replayable()
        self.decode_graph_runner.seal_startup_plan()

    def begin_decode_cuda_graph_capture(self) -> None:
        self.collective_runtime.begin_cuda_graph_capture()

    def collect_decode_cuda_graph_metadata(self) -> None:
        self.collective_runtime.collect_local_cuda_graph_metadata()

    def exchange_decode_cuda_graph_metadata(self) -> None:
        self.collective_runtime.exchange_cuda_graph_metadata()

    def register_decode_cuda_graph_buffers(self) -> None:
        self.collective_runtime.register_cuda_graph_buffers()

    def capture_decode_cuda_graph_warmup(self, seqs: list[Sequence]) -> None:
        """Capture one planned graph without advancing scheduler sequence state."""
        try:
            if self.parallel_context.attn_dp_size > 1:
                self.decode_graph_runner.dp_batch_capacity = None
            self.decode_graph_runner.run(
                seqs,
                capture_sampling=False,
                replay_after_capture=False,
            )
            if self.parallel_context.attn_dp_size > 1:
                capacity = self.decode_graph_runner.last_state_key.batch_size
                self.capture_dp_idle_graph(capacity)
        finally:
            reset_context()

    def _forward_dp_idle(self) -> None:
        # No fabricated request or KV row: idle replicas execute only the
        # expert transport/compute schedule, with zero owner-local tokens.
        hidden = torch.empty(
            (0, int(self.config.hf_config.hidden_size)),
            dtype=self.config.hf_config.dtype, device=self.device,
        )
        self.model.forward_idle_experts(hidden)

    def capture_dp_idle_graph(self, capacity: int) -> None:
        if capacity in self.dp_idle_graphs:
            return
        get_context().moe_token_capacity = capacity
        self.dp_idle_graphs[capacity] = self.decode_graph_runner.capture_idle_experts(
            self._forward_dp_idle, self.device
        )

    def submit_async(self, ticket: int, seqs: list[Sequence], is_prefill: bool):
        if not hasattr(self, "_async_execution"):
            self._async_execution = AsyncExecution(self)
        with profiler.record("async_submit"):
            self._async_execution.submit(ticket, seqs, is_prefill)

    def collect_async(self, ticket: int, discarded=()):
        with profiler.record("async_collect"):
            return self._async_execution.collect(ticket, discarded)

    def retire_async(self, seq_ids):
        self.finish_slots_batch(seq_ids)
        self._async_execution.forget(seq_ids)

    def run_dp_idle(self, *, use_graph: bool) -> None:
        capacity = get_context().moe_token_capacity
        if not capacity:
            return
        if use_graph:
            graph = self.dp_idle_graphs.get(capacity)
            if graph is None:
                raise RuntimeError(f"No startup-captured idle DP graph for capacity {capacity}.")
            graph.replay()
            self.dp_idle_replay_count += 1
        else:
            self._forward_dp_idle()
            self.dp_idle_eager_count += 1

    @torch.inference_mode()
    @cpu_timing.timed
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """物理执行逻辑：统一使用 Eager 模式"""
        _stage = 'prefill' if is_prefill else 'decode'
        with profiler.record(f"model_run_model_{_stage}"):
            if is_prefill and self._prefill_inputs_embeds is not None:
                hidden_states = self.multimodal_runtime.forward(
                    input_ids,
                    positions,
                    self._prefill_inputs_embeds,
                    self._prefill_multimodal_mask,
                )
            else:
                hidden_states = self.model(input_ids, positions)
            logits = self.model.compute_logits(hidden_states)
        self._record_debug_logits(logits)
        return logits

    def _record_debug_logits(self, logits: torch.Tensor | None) -> None:
        if (
            os.getenv("SPARSEENGINE_DEBUG_RUNTIME", "0") == "1"
            and isinstance(logits, torch.Tensor)
        ):
            # A clone captured inside run_model keeps the capture-time value on
            # CUDA Graph replay.  Refresh outside replay so debug/validation
            # observes the business step that actually completed.
            self.debug_last_logits = logits.detach().clone()

    def run_logits_for_compare(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor | None:
        """Debug logits-alignment path: execute one step and return rank-0 logits without sampling."""
        try:
            if is_prefill:
                ctx = get_context()
                input_ids, positions = self.prepare_step(seqs, is_prefill)
                with profiler.record("model_sparse_prepare"):
                    ctx.sparse_controller = self.sparse_controller
                    self.sparse_controller.prepare_forward(seqs, is_prefill)
                logits = self.run_model(input_ids, positions, is_prefill)
            else:
                logits = self.decode_graph_runner.run_eager_static(seqs)
            with profiler.record("model_sparse_post"):
                self.sparse_controller.post_forward(seqs, is_prefill)
            return logits if self.parallel_context.attn_tp_rank == 0 else None
        finally:
            reset_context()

    @cpu_timing.timed
    def _post_sparse_forward(self, seqs: list[Sequence], is_prefill: bool) -> None:
        # State transitions belong to this submission, before another step can
        # overwrite scores or prepare inputs from its KV allocation metadata.
        with profiler.record("model_sparse_post"):
            with profiler.record("sparse_post_forward"):
                self.sparse_controller.post_forward(seqs, is_prefill)
            with profiler.record("cache_on_forward_end"):
                self.runtime_state.on_forward_end(seqs, is_prefill)

    def _collect_logprobs(
        self,
        logits: torch.Tensor,
        token_ids: list[int] | torch.Tensor,
        seqs: list[Sequence],
    ) -> tuple[list[float | None], list[dict[int, float] | None]] | tuple[None, None]:
        if not any(seq.logprobs is not None for seq in seqs):
            return None if getattr(self, "_async_submitting", False) else (None, None)

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_tensor = torch.as_tensor(token_ids, device=log_probs.device, dtype=torch.long)
        sampled = log_probs.gather(1, token_tensor.unsqueeze(1)).squeeze(1)
        if getattr(self, "_async_submitting", False):
            count = max(int(seq.logprobs or 0) for seq in seqs)
            if count:
                values, indices = torch.topk(log_probs, min(count, log_probs.shape[-1]), dim=-1)
                return DeviceLogprobs(sampled, values, indices)
            return DeviceLogprobs(sampled)
        sampled_logprobs: list[float | None] = sampled.detach().cpu().tolist()

        max_top_logprobs = max(int(seq.logprobs or 0) for seq in seqs)
        top_logprobs: list[dict[int, float] | None]
        if max_top_logprobs <= 0:
            top_logprobs = [None] * len(seqs)
        else:
            top_values, top_indices = torch.topk(
                log_probs,
                k=min(max_top_logprobs, log_probs.shape[-1]),
                dim=-1,
            )
            values_cpu = top_values.detach().cpu().tolist()
            indices_cpu = top_indices.detach().cpu().tolist()
            top_logprobs = []
            for row, seq in enumerate(seqs):
                requested = int(seq.logprobs or 0)
                if requested <= 0:
                    top_logprobs.append(None)
                    continue
                values = values_cpu[row][:requested]
                indices = indices_cpu[row][:requested]
                top_logprobs.append({int(token_id): float(value) for token_id, value in zip(indices, values)})
        return sampled_logprobs, top_logprobs

    @cpu_timing.timed
    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> tuple[list[int], tuple[list[float | None], list[dict[int, float] | None]] | None]:
        """单步执行主逻辑"""
        asynchronous = getattr(self, "_async_execution", None)
        if asynchronous is not None and not getattr(self, "_async_submitting", False):
            asynchronous.prepare_synchronous_execution()
        dp_eager = False
        if self.parallel_context.attn_dp_size > 1:
            from sparseengine.engine.dp_step import coordinate_dp_step
            dp_eager = coordinate_dp_step(self, seqs, is_prefill)
            if not seqs:
                try:
                    self.run_dp_idle(use_graph=self.config.decode_graph and not dp_eager)
                    return [], None
                finally:
                    reset_context()
        name = "model_run_prefill" if is_prefill else "model_run_decode"
        with profiler.record(name):
            if not is_prefill:
                try:
                    if self.config.decode_graph and not dp_eager:
                        logits, graph_token_ids = self.decode_graph_runner.run(
                            seqs,
                            capture_sampling=self._auto_capture_greedy_sampling(seqs),
                        )
                    else:
                        logits = self.decode_graph_runner.run_eager_static(seqs)
                        graph_token_ids = None
                    self._record_debug_logits(logits)
                    if self.parallel_context.attn_tp_rank != 0:
                        self._post_sparse_forward(seqs, is_prefill)
                        return None, None
                    self._post_sparse_forward(seqs, is_prefill)
                    with profiler.record("model_sampler"):
                        sampling_logits = self._apply_sampling_penalties(logits, seqs)
                        token_ids = self._sample_model_outputs(
                            sampling_logits,
                            seqs,
                            graph_token_ids=graph_token_ids,
                            return_device_tokens=True,
                        )
                    logprob_outputs = self._mask_recompute_logprobs(
                        seqs,
                        self._collect_logprobs(sampling_logits, token_ids, seqs),
                    )
                    return (token_ids if getattr(self, "_async_submitting", False) else token_ids.tolist()), logprob_outputs
                finally:
                    reset_context()

            # 1. 准备前向上下文
            ctx = get_context()
            input_ids, positions = self.prepare_step(seqs, is_prefill)
            
            # 2. 准备稀疏化状态
            with profiler.record("model_sparse_prepare"):
                ctx.sparse_controller = self.sparse_controller
                self.sparse_controller.prepare_forward(seqs, is_prefill)
            
            # 3. 前向计算
            logits = self.run_model(input_ids, positions, is_prefill)
            
            # 4. Token 采样 (仅 Rank 0)
            with profiler.record("model_sampler"):
                if self.parallel_context.attn_tp_rank == 0:
                    sampling_logits = self._apply_sampling_penalties(logits, seqs)
                    token_ids = self._sample_model_outputs(sampling_logits, seqs, return_device_tokens=True)
                else:
                    sampling_logits = None
                    token_ids = None
            logprob_outputs = (
                self._mask_recompute_logprobs(
                    seqs,
                    self._collect_logprobs(sampling_logits, token_ids, seqs),
                )
                if self.parallel_context.attn_tp_rank == 0
                else None
            )

            if token_ids is not None and not getattr(self, "_async_submitting", False):
                token_ids = token_ids.tolist()

            # 5. 后置稀疏处理 (如 SnapKV 驱逐)
            self._post_sparse_forward(seqs, is_prefill)

            reset_context()
            return token_ids, logprob_outputs
