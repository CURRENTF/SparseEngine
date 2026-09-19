from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

import sparseengine.platforms as platforms
from sparseengine.configs.cuda_graph import _select_decode_cuda_graph_batch_size
from sparseengine.engine.decode_graph_contract import (
    DecodeGraphContract,
    DecodeGraphInputs,
    DecodeGraphState,
)
from sparseengine.engine.sequence import Sequence
from sparseengine.method_registry import decode_graph_path_id
from sparseengine.utils.context import get_context, set_context
from sparseengine.utils.profiler import cpu_timing, profiler


@dataclass(frozen=True)
class DecodeCudaGraphKey:
    method: str
    batch_size: int
    capture_sampling: bool
    graph_path_id: str = ""


@dataclass
class DecodeCudaGraphState:
    key: DecodeCudaGraphKey
    capture_context_capacity: int = 0
    decode_state: DecodeGraphState | None = None
    graph: torch.cuda.CUDAGraph | None = None
    logits: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None
    keepalive: list[object] = field(default_factory=list)
    sparse_state_refs: dict[int, dict[str, object]] = field(default_factory=dict)


class DecodeCudaGraphRunner:
    """Fixed-shape decode runner, optionally backed by CUDA Graph replay.

    The runner owns graph-stable decode metadata tensors. Cache managers still
    allocate real KV slots every step, but write the per-step metadata into these
    stable buffers before the model forward. CUDA Graph is an execution mode on
    top of the same static-compatible decode path; eager decode uses the same
    preparation and view-building route without capture/replay.
    """

    def __init__(
        self,
        *,
        runtime_state,
        cache_manager,
        recurrent_state_manager,
        sparse_controller,
        run_model: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        method: str,
        capture_sizes: list[int],
        graph_pool=None,
        collective_runtime=None,
    ):
        self.runtime_state = runtime_state
        self.cache_manager = cache_manager
        self.recurrent_state_manager = recurrent_state_manager
        self.sparse_controller = sparse_controller
        self.run_model = run_model
        self.method = str(method or "")
        self.platform = platforms.current_platform
        self.capture_sizes = sorted(set(int(size) for size in capture_sizes))
        if not self.capture_sizes or any(size <= 0 for size in self.capture_sizes):
            raise ValueError(f"decode_graph capture_sizes must be positive, got {capture_sizes}.")
        self._graphs: dict[DecodeCudaGraphKey, DecodeCudaGraphState] = {}
        self.last_state_key: DecodeCudaGraphKey | None = None
        self.last_real_batch_size: int | None = None
        self.dp_batch_capacity: int | None = None
        self.graph_pool = graph_pool
        self.collective_runtime = collective_runtime
        self.capture_count = 0
        self.replay_count = 0
        self.eager_static_count = 0
        self.force_eager_count = 0
        self.recapture_count = 0
        self._captured_keys: set[DecodeCudaGraphKey] = set()
        self.startup_plan_sealed = False

    @staticmethod
    def capture_idle_experts(forward: Callable[[], None], device):
        """Capture the expert collective schedule without attention/cache rows."""
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        return graph

    def seal_startup_plan(self):
        self.startup_plan_sealed = True

    def clear_captured_graphs(self):
        for state in list(self._graphs.values()):
            self._release_graph_state(state)
        self._graphs.clear()
        self.last_state_key = None
        self.last_real_batch_size = None
        self.dp_batch_capacity = None

    @staticmethod
    def _release_graph_state(state: DecodeCudaGraphState):
        state.graph = None
        if state.decode_state is not None:
            state.decode_state.close()
        state.decode_state = None
        state.logits = None
        state.token_ids = None
        state.keepalive.clear()
        state.sparse_state_refs.clear()


    def _select_graph_batch_size(self, real_batch_size: int) -> int:
        selector = getattr(self.cache_manager, "select_decode_cuda_graph_batch_size", None)
        if selector is not None:
            selected = selector(int(real_batch_size), list(self.capture_sizes))
            if selected is not None:
                return int(selected)

        return _select_decode_cuda_graph_batch_size(
            real_batch_size,
            self.capture_sizes,
        )

    def _select_state(
        self,
        *,
        method: str,
        batch_size: int,
        context_capacity: int,
        capture_sampling: bool,
        graph_path_id: str = "",
    ) -> DecodeCudaGraphState:
        graph_path_id = str(graph_path_id) or decode_graph_path_id(method)
        key = DecodeCudaGraphKey(
            method=method,
            batch_size=batch_size,
            capture_sampling=capture_sampling,
            graph_path_id=graph_path_id,
        )
        state = self._graphs.get(key)
        if state is not None:
            if context_capacity > state.capture_context_capacity:
                raise RuntimeError(
                    "decode CUDA Graph request exceeded captured topology-path "
                    f"capacity: requested={context_capacity}, "
                    f"captured={state.capture_context_capacity}, "
                    f"path={graph_path_id!r}."
                )
            return state

        if self.startup_plan_sealed:
            raise RuntimeError(
                "decode CUDA Graph has no startup-captured graph for "
                f"batch_size={batch_size}, path={graph_path_id!r}."
            )

        state = DecodeCudaGraphState(
            key=key,
            capture_context_capacity=int(context_capacity),
        )
        device = getattr(
            self.cache_manager,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        contract = DecodeGraphContract(
            method=str(method),
            topology_path_id=graph_path_id,
            batch_capacity=int(batch_size),
            context_capacity=int(context_capacity),
            capture_sampling=bool(capture_sampling),
        )
        platform = getattr(self, "platform", None)
        pin_memory = bool(
            device.type != "cpu"
            and platform is not None
            and platform.supports_pin_memory()
        )
        inputs = DecodeGraphInputs.allocate(
            contract,
            device=device,
            pin_memory=pin_memory,
        )
        state.decode_state = DecodeGraphState(contract=contract, inputs=inputs)
        self._graphs[key] = state
        return state

    @cpu_timing.timed
    def _prepare_static_step(
        self,
        state: DecodeCudaGraphState,
        seqs: list[Sequence],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepare_decode_graph_step = getattr(
            self.runtime_state,
            "prepare_decode_graph_step",
            None,
        )
        if prepare_decode_graph_step is None:
            raise TypeError(
                "decode_graph requires runtime_state.prepare_decode_graph_step()."
            )
        graph_state = state.decode_state
        if graph_state is None:
            raise RuntimeError("Decode graph state was released before preparation.")
        graph_state.inputs.validate(graph_state.contract)

        self.cache_manager.set_decode_static_max_context_len(
            int(state.capture_context_capacity)
        )
        provider = getattr(self, "async_input_provider", None)
        if provider is not None:
            graph_state.inputs.host = provider.acquire_host_inputs(state.key.batch_size)
        input_ids, positions, _ = prepare_decode_graph_step(
            seqs,
            graph_state,
        )
        provider = getattr(self, "async_input_provider", None)
        if provider is not None:
            provider.prepare_inputs(input_ids, seqs)

        set_context(
            False,
            cu_seqlens_q=None,
            cache_manager=self.cache_manager,
            seqs=seqs,
            recurrent_state_manager=self.recurrent_state_manager,
        )
        self.cache_manager.set_decode_static_max_context_len(
            int(state.capture_context_capacity)
        )

        self.sparse_controller.prepare_decode_replay(seqs)
        return input_ids, positions

    def graph_plan(self) -> dict[str, object]:
        return {
            "batch_sizes": list(self.capture_sizes),
            "startup_plan_sealed": self.startup_plan_sealed,
            "cached_graph_keys": [
                {
                    "method": key.method,
                    "batch_size": key.batch_size,
                    "context_capacity": state.capture_context_capacity,
                    "graph_path_id": key.graph_path_id,
                    "capture_sampling": key.capture_sampling,
                }
                for key, state in self._graphs.items()
                if state.graph is not None
            ],
        }

    def _graph_path_id(self) -> str:
        resolver = getattr(self.cache_manager, "decode_graph_path_id", None)
        if callable(resolver):
            return str(resolver())
        return decode_graph_path_id(self.method)

    def _graph_path_capacity(
        self, seqs: list[Sequence]
    ) -> int:
        resolver = getattr(
            self.cache_manager,
            "decode_graph_path_capacity",
            None,
        )
        if not callable(resolver):
            raise TypeError(
                "decode CUDA Graph requires a topology-path capacity resolver."
            )
        capacity = int(resolver())
        validator = getattr(
            self.cache_manager,
            "validate_decode_graph_path_capacity",
            None,
        )
        if callable(validator):
            validator(seqs, capacity=capacity)
        return capacity

    def _snapshot_sparse_state_refs(self) -> dict[int, dict[str, object]]:
        refs: dict[int, dict[str, object]] = {}
        for layer_idx, sparse_state in self.sparse_controller.layer_batch_sparse_states.items():
            refs[int(layer_idx)] = {
                "attn_score": sparse_state.attn_score,
                "active_indices": sparse_state.active_indices,
                "active_slots": sparse_state.active_slots,
                "req_indices": sparse_state.req_indices,
                "context_lens": sparse_state.context_lens,
                "max_context_len": sparse_state.max_context_len,
                "active_compressed_indices": sparse_state.active_compressed_indices,
                "global_req_indices": sparse_state.global_req_indices,
                "deltakv_free_temp_slots": sparse_state.deltakv_free_temp_slots,
            }
        return refs

    def _restore_sparse_state_refs(self, state: DecodeCudaGraphState):
        """Restore Python sparse-state pointers captured by this graph.

        CUDA Graph replay uses the tensor addresses captured during warmup. A
        real request's prefill can overwrite SparseController Python fields
        before decode; restoring here keeps post-forward sparse eviction reading
        the same stable tensors that prepare_decode_static updates in place.
        """
        for layer_idx, refs in state.sparse_state_refs.items():
            sparse_state = self.sparse_controller.layer_batch_sparse_states[layer_idx]
            for name, value in refs.items():
                setattr(sparse_state, name, value)

    def _reset_graph_input_attn_scores(self, refs: dict[int, dict[str, object]]):
        """Reset graph-input score buffers inside capture/replay.

        prepare_forward() allocates and initializes decode attn_score tensors
        before graph capture. Replay does not re-run that Python setup, so score
        buffers used by captured observation-layer kernels must be reset by a
        captured fill before each replay.
        """
        reset_contiguous = getattr(
            self.sparse_controller,
            "reset_decode_attn_scores_for_graph",
            None,
        )
        if callable(reset_contiguous) and reset_contiguous(refs):
            return
        for layer_refs in refs.values():
            attn_score = layer_refs.get("attn_score")
            if isinstance(attn_score, torch.Tensor):
                attn_score.fill_(-1e20)

    def _capture(
        self,
        state: DecodeCudaGraphState,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> DecodeCudaGraphState:
        if self.collective_runtime is not None:
            self.collective_runtime.assert_can_capture()
        if not self.platform.supports_graph_capture():
            raise RuntimeError(f"Platform {self.platform.name!r} does not support decode CUDA graph capture.")
        graph_state = state.decode_state
        if graph_state is None:
            raise RuntimeError("Decode graph state was released before capture.")
        ctx = get_context()
        ctx.sparse_controller = self.sparse_controller

        with profiler.record("decode_graph_warmup"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
            participant = graph_state.runtime_state
            if participant is not None:
                participant.prepare_in_graph()
            logits = self.run_model(input_ids, positions, is_prefill=False)
            if state.key.capture_sampling:
                if logits is None:
                    raise RuntimeError("decode_graph capture_sampling requires rank-0 logits.")
                _ = logits.argmax(dim=-1)
        self.platform.synchronize()

        # In runtime-invariant mode, eager warmup consumes the prevalidated
        # storage scope. Establish a fresh scope for capture.
        self.cache_manager.validate_decode_cuda_graph_slot_mappings()

        with profiler.record("decode_graph_capture"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
            # Dynamic score paths can replace Python state fields during the
            # captured forward. Keep both input and post-forward refs alive;
            # SnapKV-family fused 2D score buffers are retained by the controller.
            graph_input_sparse_state_refs = self._snapshot_sparse_state_refs()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=self.graph_pool):
                    participant = graph_state.runtime_state
                    if participant is not None:
                        participant.prepare_in_graph()
                    self._reset_graph_input_attn_scores(graph_input_sparse_state_refs)
                    logits = self.run_model(input_ids, positions, is_prefill=False)
                    if state.key.capture_sampling:
                        if logits is None:
                            raise RuntimeError("decode_graph capture_sampling requires rank-0 logits.")
                        token_ids = logits.argmax(dim=-1)
                    else:
                        token_ids = None
            except Exception as exc:
                raise RuntimeError(f"decode_graph capture failed: {exc!r}") from exc

        state.graph = graph
        state.logits = logits
        state.token_ids = token_ids
        state.sparse_state_refs = self._snapshot_sparse_state_refs()

        keepalive: list[object] = [
            ctx,
            logits,
            ctx.decode_mid_o,
            ctx.decode_mid_o_logexpsum,
        ]
        keepalive.extend(graph_state.keepalive_tensors())
        if token_ids is not None:
            keepalive.append(token_ids)
        for sparse_refs_by_layer in (graph_input_sparse_state_refs, state.sparse_state_refs):
            for refs in sparse_refs_by_layer.values():
                for value in refs.values():
                    if isinstance(value, torch.Tensor):
                        keepalive.append(value)
        sparse_keepalive = getattr(self.sparse_controller, "decode_graph_keepalive_tensors", None)
        if sparse_keepalive is not None:
            keepalive.extend(sparse_keepalive())
        state.keepalive = keepalive
        captured_keys = getattr(self, "_captured_keys", None)
        if captured_keys is None:
            captured_keys = set()
            self._captured_keys = captured_keys
        if state.key in captured_keys:
            self.recapture_count = int(getattr(self, "recapture_count", 0)) + 1
        else:
            captured_keys.add(state.key)
        self.capture_count += 1
        return state

    def run(
        self,
        seqs: list[Sequence],
        *,
        capture_sampling: bool = False,
        replay_after_capture: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not seqs:
            raise ValueError("decode_graph requires a non-empty decode batch.")
        if capture_sampling and any(seq.temperature > 1e-10 for seq in seqs):
            raise ValueError("decode_graph capture_sampling currently supports greedy decode only.")
        if capture_sampling and any(
            float(getattr(seq, "presence_penalty", 0.0)) != 0.0
            or float(getattr(seq, "repetition_penalty", 1.0)) != 1.0
            for seq in seqs
        ):
            raise ValueError(
                "decode_graph capture_sampling cannot apply sampling penalties."
            )

        real_batch_size = len(seqs)
        force_eager = getattr(self.cache_manager, "decode_graph_force_eager", None)
        if force_eager is not None and force_eager():
            self.force_eager_count += 1
            return self.run_eager_static(seqs), None

        graph_batch_size = self.dp_batch_capacity or self._select_graph_batch_size(real_batch_size)
        graph_path_id = self._graph_path_id()
        context_capacity = self._graph_path_capacity(seqs)
        state = self._select_state(
            method=self.method,
            batch_size=graph_batch_size,
            context_capacity=context_capacity,
            capture_sampling=bool(capture_sampling),
            graph_path_id=graph_path_id,
        )
        self.last_state_key = state.key
        self.last_real_batch_size = real_batch_size
        input_ids, positions = self._prepare_static_step(state, seqs)

        if state.graph is None:
            state = self._capture(state, seqs, input_ids, positions)
            self._restore_sparse_state_refs(state)
            if not replay_after_capture:
                return None, None
            if self.collective_runtime is not None:
                self.collective_runtime.assert_cuda_graph_replayable()
            with profiler.record("decode_graph_replay_after_capture"):
                state.graph.replay()
            self.replay_count += 1
            logits = state.logits[:real_batch_size] if state.logits is not None else None
            token_ids = state.token_ids[:real_batch_size] if state.token_ids is not None else None
            return logits, token_ids

        if not replay_after_capture:
            raise RuntimeError(
                "decode_graph capture-only warmup selected an already-captured graph: "
                f"key={state.key}."
            )

        self._restore_sparse_state_refs(state)
        if self.collective_runtime is not None:
            self.collective_runtime.assert_cuda_graph_replayable()
        with profiler.record("decode_graph_replay"):
            state.graph.replay()
        self.replay_count += 1
        logits = state.logits[:real_batch_size] if state.logits is not None else None
        token_ids = state.token_ids[:real_batch_size] if state.token_ids is not None else None
        return logits, token_ids

    def run_eager_static(self, seqs: list[Sequence]) -> torch.Tensor | None:
        """Run decode eagerly through the same static-compatible path used by graphs."""
        if not seqs:
            raise ValueError("static decode requires a non-empty decode batch.")
        self.eager_static_count += 1

        real_batch_size = len(seqs)
        graph_batch_size = self.dp_batch_capacity or self._select_graph_batch_size(real_batch_size)
        graph_path_id = self._graph_path_id()
        context_capacity = self._graph_path_capacity(seqs)
        state = self._select_state(
            method=self.method,
            batch_size=graph_batch_size,
            context_capacity=context_capacity,
            capture_sampling=False,
            graph_path_id=graph_path_id,
        )
        self.last_state_key = state.key
        self.last_real_batch_size = real_batch_size
        input_ids, positions = self._prepare_static_step(state, seqs)
        graph_state = state.decode_state
        if graph_state is None:
            raise RuntimeError("Decode graph state was released before static execution.")

        ctx = get_context()
        ctx.sparse_controller = self.sparse_controller
        with profiler.record("model_sparse_prepare"):
            self.sparse_controller.prepare_forward(seqs, is_prefill=False)
        participant = graph_state.runtime_state
        if participant is not None:
            participant.prepare_in_graph()
        logits = self.run_model(input_ids, positions, is_prefill=False)
        if logits is None:
            return None
        return logits[:real_batch_size]
