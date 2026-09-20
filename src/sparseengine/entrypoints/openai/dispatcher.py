import asyncio
import os
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from sparseengine.engine.input_processor import tokenize_text_prompt
from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer
from sparseengine.entrypoints.openai.sampling import _find_stop_index
from sparseengine.entrypoints.openai.sampling import _safe_stream_text_len
from sparseengine.llm import LLM
from sparseengine.sampling_params import SamplingParams
from sparseengine.sampling_params import resolve_eos_token_ids
from sparseengine.utils.log import logger


@dataclass
class RequestHandle:
    output_queue: asyncio.Queue
    cancelled: threading.Event
    terminal: threading.Event = field(default_factory=threading.Event)
    request_token: str = field(default_factory=lambda: uuid.uuid4().hex)
    seq_id: int | None = None
    chain_id: str | None = None
    chain_status: str = "disabled"
    reused_tokens: int = 0
    prefilled_tokens: int = 0
    prompt_token_ids: list[int] | None = None
    admission_future: asyncio.Future | None = None
    admission_error: BaseException | None = None


@dataclass
class _QueuedRequest:
    prompt: str | list[int]
    sampling_params: SamplingParams
    index: int
    stop: list[str]
    loop: asyncio.AbstractEventLoop
    output_queue: asyncio.Queue
    cancelled: threading.Event
    handle: RequestHandle
    admission_future: asyncio.Future
    chain_id: str | None = None
    chain_append_only: bool = False
    stream: bool = True
    preparation_error: Exception | None = None


@dataclass
class _ActiveRequest:
    index: int
    loop: asyncio.AbstractEventLoop
    output_queue: asyncio.Queue
    prompt_token_ids: list[int]
    max_tokens: int
    stop: list[str]
    completion_token_ids: list[int]
    completion_token_logprobs: list[float | None]
    completion_top_logprobs: list[dict[int, float] | None]
    detokenizer: IncrementalDetokenizer
    stream: bool = True
    eos_token_ids: frozenset[int] = field(default_factory=frozenset)
    ignore_eos: bool = False
    terminal: threading.Event = field(default_factory=threading.Event)
    emitted_text_len: int = 0
    emitted_raw_text_len: int = 0
    pending_token_ids: list[int] = field(default_factory=list)
    pending_token_logprobs: list[float | None] = field(default_factory=list)
    pending_top_logprobs: list[dict[int, float] | None] = field(default_factory=list)
    chain_id: str | None = None
    chain_status: str = "disabled"
    reused_tokens: int = 0
    prefilled_tokens: int = 0
    request_token: str | None = None
    prompt_token_count: int | None = None
    handle: RequestHandle | None = None


@dataclass
class _ControlRequest:
    operation: str
    kwargs: dict[str, Any]
    loop: asyncio.AbstractEventLoop
    output_queue: asyncio.Queue


@dataclass(frozen=True)
class _AbortRequest:
    seq_id: int
    disposition: str
    chain_id: str | None
    request_token: str
    terminal: threading.Event | None = None
    force: bool = False
    loop: asyncio.AbstractEventLoop | None = None
    output_queue: asyncio.Queue | None = None


_WAKEUP = object()


class AsyncEngineDispatcher:
    def __init__(self, engine: LLM):
        self.engine = engine
        # Serialize preparation/submission without blocking the engine thread or
        # the ASGI loop. A dedicated worker also bounds tokenizer concurrency
        # when a submitting task is cancelled while encode is still running.
        self._input_tokenizer = None
        self._prepare_lock = asyncio.Lock()
        self._prepare_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sparseengine-input"
        )
        self._pending: queue.Queue[_QueuedRequest | object | None] = queue.Queue()
        self._aborts: queue.Queue[
            _AbortRequest | tuple[int, str, str | None] | tuple[int, str] | int
        ] = queue.Queue()
        self._latest_request_tokens: dict[int, str] = {}
        self._controls: queue.Queue[_ControlRequest] = queue.Queue()
        self._closing = threading.Event()
        self._failed_message: str | None = None
        self._fatal_callback: Callable[[str], None] | None = None
        self._state_lock = threading.Lock()
        self._routing_snapshot_lock = threading.Lock()
        self._worker_routing_load_snapshot: dict[str, Any] | None = None
        self._prefix_cache_routing_snapshot: Any | None = None
        self._chain_cache_routing_snapshot: Any | None = None
        self._refresh_routing_snapshots()
        self._thread = threading.Thread(target=self._run, name="sparseengine-openai-dispatcher", daemon=True)
        self._thread.start()

    @property
    def failure_message(self) -> str | None:
        with self._state_lock:
            return self._failed_message

    @property
    def is_ready(self) -> bool:
        with self._state_lock:
            return self._terminal_message_locked() is None

    @property
    def chain_cache_enabled(self) -> bool:
        config = getattr(self.engine, "config", None)
        return (
            str(
                getattr(
                    config, "resolved_prefix_cache_mode", "disabled"
                )
            )
            == "chain"
        )

    @property
    def admission_ack_enabled(self) -> bool:
        return hasattr(
            getattr(self.engine, "config", None),
            "resolved_prefix_cache_mode",
        )

    def set_fatal_callback(self, callback: Callable[[str], None]) -> None:
        with self._state_lock:
            self._fatal_callback = callback
            failure_message = self._failed_message
        if failure_message is not None:
            callback(failure_message)

    def _mark_failed(self, message: str) -> Callable[[str], None] | None:
        with self._state_lock:
            self._failed_message = message
            return self._fatal_callback

    def _terminal_message_locked(self) -> str | None:
        if self._failed_message is not None:
            return self._failed_message
        if self._closing.is_set():
            return "SparseEngine server is shutting down."
        if not self._thread.is_alive():
            return "SparseEngine dispatcher thread is not running."
        return None

    async def submit(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        index: int,
        stop: list[str] | None = None,
        chain_id: str | None = None,
        chain_append_only: bool = False,
        *,
        stream: bool = True,
    ) -> RequestHandle:
        output_queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()
        handle = RequestHandle(output_queue=output_queue, cancelled=cancelled)
        loop = asyncio.get_running_loop()
        admission_future = loop.create_future()
        handle.admission_future = admission_future
        queued = _QueuedRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            index=index,
            stop=list(stop or []),
            loop=loop,
            output_queue=output_queue,
            cancelled=cancelled,
            handle=handle,
            admission_future=admission_future,
            chain_id=chain_id,
            chain_append_only=bool(chain_append_only),
            stream=stream,
        )
        async with self._prepare_lock:
            with self._state_lock:
                terminal_message = self._terminal_message_locked()
            if terminal_message is None and isinstance(prompt, str):
                try:
                    queued.prompt = await loop.run_in_executor(
                        self._prepare_executor,
                        self._prepare_text_prompt,
                        prompt,
                        chain_append_only,
                    )
                except Exception as exc:
                    # Keep the existing admission error/final notification path.
                    queued.preparation_error = exc
            with self._state_lock:
                terminal_message = self._terminal_message_locked()
                if terminal_message is None:
                    self._pending.put(queued)
        if terminal_message is not None:
            handle.admission_error = RuntimeError(terminal_message)
            handle.terminal.set()
            admission_future.set_result(handle)
            output_queue.put_nowait(
                {"type": "error", "message": terminal_message}
            )
        return handle

    def _prepare_text_prompt(self, prompt: str, chain_append_only: bool) -> list[int]:
        if self._input_tokenizer is None:
            # Fast tokenizers can mutate padding/truncation settings during
            # encode. Do not share that backend with engine detokenization.
            self._input_tokenizer = deepcopy(self.engine.tokenizer)
        tokenizer = self._input_tokenizer
        if chain_append_only:
            # A suffix must never acquire a new BOS token. Empty suffix handling
            # and chain validation remain owned by engine admission.
            return tokenizer.encode(prompt, add_special_tokens=False)
        return tokenize_text_prompt(tokenizer, prompt)

    async def submit_admitted(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        index: int,
        stop: list[str] | None = None,
        chain_id: str | None = None,
        chain_append_only: bool = False,
        *,
        stream: bool = True,
    ) -> RequestHandle:
        handle = await self.submit(
            prompt,
            sampling_params,
            index,
            stop,
            chain_id=chain_id,
            chain_append_only=chain_append_only,
            stream=stream,
        )
        if handle.admission_future is not None:
            try:
                await asyncio.shield(handle.admission_future)
            except asyncio.CancelledError:
                self.cancel(handle)
                raise
        if handle.admission_error is not None:
            raise handle.admission_error
        return handle

    async def control(self, operation: str, **kwargs: Any) -> Any:
        output_queue: asyncio.Queue = asyncio.Queue()
        queued = _ControlRequest(
            operation=operation,
            kwargs=dict(kwargs),
            loop=asyncio.get_running_loop(),
            output_queue=output_queue,
        )
        with self._state_lock:
            terminal_message = self._terminal_message_locked()
            if terminal_message is None:
                self._controls.put(queued)
                self._pending.put(_WAKEUP)
        if terminal_message is not None:
            raise RuntimeError(terminal_message)
        result = await output_queue.get()
        if result["type"] == "error":
            raise RuntimeError(result["message"])
        return result["value"]

    def worker_routing_load_snapshot(self) -> dict[str, Any]:
        self._raise_if_unavailable()
        with self._routing_snapshot_lock:
            snapshot = self._worker_routing_load_snapshot
        if snapshot is None:
            raise RuntimeError(
                "Engine does not expose worker_routing_load for routing snapshots."
            )
        return {**snapshot, "snapshot": True}

    def prefix_cache_routing_match(
        self,
        token_ids: list[int],
    ) -> dict[str, Any]:
        self._raise_if_unavailable()
        with self._routing_snapshot_lock:
            snapshot = self._prefix_cache_routing_snapshot
        if snapshot is None:
            raise RuntimeError(
                "Engine does not expose a prefix-cache routing snapshot."
            )
        result = snapshot.match([int(token_id) for token_id in token_ids])
        if not isinstance(result, dict):
            raise RuntimeError(
                "Prefix-cache routing snapshot returned a non-object result: "
                f"{type(result).__name__}."
            )
        return result

    def chain_cache_routing_match(self, chain_id: str) -> dict[str, Any]:
        self._raise_if_unavailable()
        with self._routing_snapshot_lock:
            snapshot = self._chain_cache_routing_snapshot
        if snapshot is None:
            return {
                "enabled": False,
                "present": False,
                "state": None,
                "tombstone": False,
            }
        result = snapshot.match(str(chain_id))
        if not isinstance(result, dict):
            raise RuntimeError(
                "Chain-cache routing snapshot returned a non-object result: "
                f"{type(result).__name__}."
            )
        return result

    def _raise_if_unavailable(self) -> None:
        with self._state_lock:
            terminal_message = self._terminal_message_locked()
        if terminal_message is not None:
            raise RuntimeError(terminal_message)

    def _refresh_routing_snapshots(self) -> None:
        worker_routing_load_fn = getattr(
            self.engine,
            "worker_routing_load",
            None,
        )
        if not callable(worker_routing_load_fn):
            worker_routing_load_fn = getattr(self.engine, "worker_load", None)
        prefix_snapshot_fn = getattr(
            self.engine,
            "prefix_cache_routing_snapshot",
            None,
        )
        chain_snapshot_fn = getattr(
            self.engine,
            "chain_cache_routing_snapshot",
            None,
        )
        worker_routing_load = (
            worker_routing_load_fn()
            if callable(worker_routing_load_fn)
            else None
        )
        prefix_snapshot = (
            prefix_snapshot_fn()
            if callable(prefix_snapshot_fn)
            else None
        )
        chain_snapshot = (
            chain_snapshot_fn()
            if callable(chain_snapshot_fn)
            else None
        )
        if (
            worker_routing_load is not None
            and not isinstance(worker_routing_load, dict)
        ):
            raise RuntimeError(
                "worker_routing_load must return an object for routing "
                f"snapshots, got {type(worker_routing_load).__name__}."
            )
        with self._routing_snapshot_lock:
            if worker_routing_load is not None:
                self._worker_routing_load_snapshot = dict(
                    worker_routing_load
                )
            if prefix_snapshot is not None:
                self._prefix_cache_routing_snapshot = prefix_snapshot
            if chain_snapshot is not None:
                self._chain_cache_routing_snapshot = chain_snapshot

    def cancel(
        self,
        handle: RequestHandle,
        disposition: str = "invalidate",
    ):
        if handle.terminal.is_set():
            return
        handle.cancelled.set()
        if handle.seq_id is not None:
            self._aborts.put(
                _AbortRequest(
                    seq_id=int(handle.seq_id),
                    disposition=str(disposition),
                    chain_id=handle.chain_id,
                    request_token=handle.request_token,
                    terminal=handle.terminal,
                )
            )
            self._pending.put(_WAKEUP)

    async def discard(self, handle: RequestHandle) -> None:
        handle.cancelled.set()
        if handle.seq_id is None:
            return
        output_queue: asyncio.Queue = asyncio.Queue()
        abort = _AbortRequest(
            seq_id=int(handle.seq_id),
            disposition="invalidate",
            chain_id=handle.chain_id,
            request_token=handle.request_token,
            terminal=handle.terminal,
            force=True,
            loop=asyncio.get_running_loop(),
            output_queue=output_queue,
        )
        with self._state_lock:
            terminal_message = self._terminal_message_locked()
            if terminal_message is None:
                self._aborts.put(abort)
                self._pending.put(_WAKEUP)
        if terminal_message is not None:
            raise RuntimeError(terminal_message)
        timeout_s = float(
            os.getenv("SPARSEENGINE_OPENAI_DISCARD_TIMEOUT_S", "30")
        )
        try:
            result = await asyncio.wait_for(
                output_queue.get(),
                timeout=max(0.1, timeout_s),
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Timed out while discarding a partially admitted request."
            ) from exc
        if result["type"] == "error":
            raise RuntimeError(result["message"])

    def close(self):
        with self._state_lock:
            if not self._closing.is_set():
                self._closing.set()
                self._pending.put(None)
        self._prepare_executor.shutdown(wait=False, cancel_futures=True)
        timeout_s = float(os.getenv("SPARSEENGINE_OPENAI_SHUTDOWN_TIMEOUT_S", "5"))
        self._thread.join(timeout=max(0.0, timeout_s))
        if self._thread.is_alive():
            logger.warning(
                "OpenAI dispatcher did not stop within {:.1f}s; forcing engine shutdown.",
                timeout_s,
            )
        self.engine.exit()

    def _put(self, request: _ActiveRequest | _QueuedRequest, item: dict[str, Any]):
        request.loop.call_soon_threadsafe(request.output_queue.put_nowait, item)

    def _put_control(self, request: _ControlRequest, item: dict[str, Any]):
        request.loop.call_soon_threadsafe(request.output_queue.put_nowait, item)

    def _resolve_admission(
        self,
        item: _QueuedRequest,
        error: BaseException | None = None,
    ) -> None:
        item.handle.admission_error = error

        def resolve() -> None:
            if not item.admission_future.done():
                item.admission_future.set_result(item.handle)

        item.loop.call_soon_threadsafe(resolve)

    def _run(self):
        active: dict[int, _ActiveRequest] = {}
        stopping = False
        fatal_callback: Callable[[str], None] | None = None
        try:
            while not stopping:
                self._drain_controls()
                run_prefix_prune = getattr(
                    self.engine, "run_pending_prefix_prune", None
                )
                if callable(run_prefix_prune):
                    run_prefix_prune()
                if self._drain_aborts(active):
                    self._refresh_routing_snapshots()
                if not active:
                    item = self._pending.get()
                    if item is None:
                        break
                    if item is _WAKEUP:
                        continue
                    self._admit(item, active)

                while True:
                    try:
                        item = self._pending.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        stopping = True
                        continue
                    if item is _WAKEUP:
                        continue
                    self._admit(item, active)

                self._drain_controls()
                if self._drain_aborts(active):
                    self._refresh_routing_snapshots()
                if not active:
                    self._refresh_routing_snapshots()
                    continue

                self._refresh_routing_snapshots()
                finished_outputs, _num_tokens = self.engine.step()
                self._refresh_routing_snapshots()
                self._update_prompt_cache_usage(active)
                self._publish_token_deltas(active)
                self._publish_finished(active, finished_outputs)
        except Exception as exc:
            failed_message = f"{type(exc).__name__}: {exc}"
            fatal_callback = self._mark_failed(failed_message)
            logger.exception("OpenAI dispatcher stopped after a fatal error: {}", failed_message)
        finally:
            terminal_message = self.failure_message or "SparseEngine server is shutting down."
            try:
                self._fail_active_requests(active, terminal_message)
                self._drain_queued_requests(terminal_message)
                self._fail_pending_aborts(terminal_message)
            finally:
                if fatal_callback is not None:
                    try:
                        fatal_callback(terminal_message)
                    except Exception:
                        logger.exception("OpenAI dispatcher fatal callback failed")

    def _drain_controls(self):
        while True:
            try:
                item = self._controls.get_nowait()
            except queue.Empty:
                return
            if self._closing.is_set():
                self._put_control(item, {"type": "error", "message": "SparseEngine server is shutting down."})
                continue
            if self._failed_message is not None:
                self._put_control(item, {"type": "error", "message": self._failed_message})
                continue
            try:
                method = getattr(self.engine, item.operation)
                value = method(**item.kwargs)
            except Exception as exc:
                self._put_control(item, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                continue
            try:
                self._refresh_routing_snapshots()
            except Exception as exc:
                self._put_control(item, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                raise
            self._put_control(item, {"type": "result", "value": value})

    def _admit(self, item: _QueuedRequest, active: dict[int, _ActiveRequest]):
        if item.cancelled.is_set():
            item.handle.terminal.set()
            self._resolve_admission(item, asyncio.CancelledError())
            return
        if self._closing.is_set():
            message = "SparseEngine server is shutting down."
            item.handle.terminal.set()
            self._put(item, {"type": "error", "message": message})
            self._resolve_admission(item, RuntimeError(message))
            return
        if self._failed_message is not None:
            item.handle.terminal.set()
            self._put(item, {"type": "error", "message": self._failed_message})
            self._resolve_admission(
                item,
                RuntimeError(self._failed_message),
            )
            return
        try:
            detokenizer = IncrementalDetokenizer(self.engine.tokenizer)
            if item.preparation_error is not None:
                raise item.preparation_error
            admit = getattr(self.engine, "admit_request", None)
            if callable(admit):
                admission_kwargs = {"chain_id": item.chain_id}
                if item.chain_append_only:
                    admission_kwargs["chain_append_only"] = True
                admission = admit(
                    item.prompt,
                    item.sampling_params,
                    **admission_kwargs,
                )
                seq_id = int(admission.seq_id)
                item.handle.chain_id = admission.chain_id
                item.handle.chain_status = str(admission.chain_status)
                item.handle.reused_tokens = int(admission.reused_tokens)
                item.handle.prefilled_tokens = int(admission.prefilled_tokens)
            else:
                seq_id = self.engine.add_request(
                    item.prompt, item.sampling_params
                )
            item.handle.seq_id = seq_id
            if item.cancelled.is_set():
                self.engine.abort_request(seq_id)
                item.handle.terminal.set()
                self._resolve_admission(item, asyncio.CancelledError())
                return
            admitted_prompt_token_ids = (
                getattr(admission, "prompt_token_ids", None) if callable(admit) else None
            )
            prompt_token_ids = (
                list(admitted_prompt_token_ids)
                if admitted_prompt_token_ids is not None
                else list(item.prompt)
                if isinstance(item.prompt, list)
                else self.engine.tokenizer.encode(item.prompt)
            )
            item.handle.prompt_token_ids = prompt_token_ids
            engine_config = getattr(self.engine, "config", None)
            eos_token_ids = resolve_eos_token_ids(
                getattr(item.sampling_params, "eos_token_ids", ()),
                getattr(engine_config, "eos_token_ids", ()),
                fallback_eos_token_id=getattr(
                    engine_config, "eos", -1
                ),
            )
            active[seq_id] = _ActiveRequest(
                index=item.index,
                loop=item.loop,
                output_queue=item.output_queue,
                prompt_token_ids=prompt_token_ids,
                max_tokens=item.sampling_params.max_tokens,
                stop=item.stop,
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=detokenizer,
                stream=item.stream,
                eos_token_ids=eos_token_ids,
                ignore_eos=bool(
                    getattr(item.sampling_params, "ignore_eos", False)
                ),
                terminal=item.handle.terminal,
                chain_id=item.handle.chain_id,
                chain_status=item.handle.chain_status,
                reused_tokens=item.handle.reused_tokens,
                prefilled_tokens=item.handle.prefilled_tokens,
                request_token=item.handle.request_token,
                prompt_token_count=(
                    int(admission.reused_tokens)
                    + int(admission.prefilled_tokens)
                    if callable(admit)
                    else len(prompt_token_ids)
                ),
                handle=item.handle,
            )
            latest_request_tokens = getattr(
                self, "_latest_request_tokens", None
            )
            if latest_request_tokens is None:
                latest_request_tokens = {}
                self._latest_request_tokens = latest_request_tokens
            latest_request_tokens[seq_id] = item.handle.request_token
            self._resolve_admission(item)
        except Exception as exc:
            if item.handle.seq_id is not None:
                try:
                    self.engine.abort_request(
                        item.handle.seq_id,
                        disposition="invalidate",
                    )
                except TypeError:
                    self.engine.abort_request(item.handle.seq_id)
                except Exception:
                    logger.exception(
                        "Failed to abort request {} after admission setup failed.",
                        item.handle.seq_id,
                    )
            item.handle.terminal.set()
            latest_request_tokens = getattr(
                self, "_latest_request_tokens", {}
            )
            if (
                item.handle.seq_id is not None
                and latest_request_tokens.get(int(item.handle.seq_id))
                == item.handle.request_token
            ):
                latest_request_tokens.pop(int(item.handle.seq_id), None)
            self._put(
                item,
                {
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            )
            self._resolve_admission(item, exc)

    @staticmethod
    def _response_parser_raw_text(
        request: _ActiveRequest,
        token_ids: list[int],
        raw_text: str,
        visible_text_len: int,
    ) -> str:
        if (
            request.ignore_eos
            or not request.eos_token_ids
            or not token_ids
            or int(token_ids[-1]) not in request.eos_token_ids
        ):
            parser_raw_text = raw_text
        else:
            content_end = len(token_ids)
            while (
                content_end > 0
                and int(token_ids[content_end - 1])
                in request.eos_token_ids
            ):
                content_end -= 1
            parser_raw_text = request.detokenizer.tokenizer.decode(
                token_ids[:content_end],
                skip_special_tokens=False,
            )

        if not raw_text.startswith(parser_raw_text):
            raise RuntimeError(
                "Response-parser raw text is not a prefix of detokenized "
                f"raw text: parser={parser_raw_text!r} raw={raw_text!r}."
            )
        if not request.stop:
            return parser_raw_text
        raw_text_len = request.detokenizer.raw_offset_for_visible_prefix(
            visible_text_len,
            raw_text_limit=len(parser_raw_text),
        )
        return parser_raw_text[:raw_text_len]

    def _update_prompt_cache_usage(self, active: dict[int, _ActiveRequest]) -> None:
        for seq_id, hit_tokens in getattr(
            self.engine, "last_step_prompt_cache_hits", []
        ):
            request = active.get(seq_id)
            if request is None:
                continue
            request.reused_tokens = max(request.reused_tokens, int(hit_tokens))
            prompt_tokens = request.prompt_token_count or len(request.prompt_token_ids)
            request.prefilled_tokens = max(0, prompt_tokens - request.reused_tokens)
            if request.handle is not None:
                request.handle.reused_tokens = request.reused_tokens
                request.handle.prefilled_tokens = request.prefilled_tokens

    def _publish_token_deltas(self, active: dict[int, _ActiveRequest]):
        logprob_outputs = {
            seq_id: (token_logprobs, top_logprobs)
            for seq_id, token_logprobs, top_logprobs in getattr(
                self.engine,
                "last_step_logprob_outputs",
                [],
            )
        }
        for seq_id, token_ids in self.engine.last_step_token_outputs:
            request = active.get(seq_id)
            if request is None:
                continue
            token_logprobs, top_logprobs = logprob_outputs.get(
                seq_id,
                ([None] * len(token_ids), [None] * len(token_ids)),
            )
            request.completion_token_ids.extend(token_ids)
            request.completion_token_logprobs.extend(token_logprobs)
            request.completion_top_logprobs.extend(top_logprobs)
            request.pending_token_ids.extend(token_ids)
            request.pending_token_logprobs.extend(token_logprobs)
            request.pending_top_logprobs.extend(top_logprobs)
            request.detokenizer.push(token_ids)
            full_text = request.detokenizer.text
            stop_index = _find_stop_index(full_text, request.stop)
            visible_text = full_text if stop_index is None else full_text[:stop_index]
            emit_len = (
                len(visible_text)
                if stop_index is not None
                else _safe_stream_text_len(visible_text, request.stop)
            )
            parser_raw_text = self._response_parser_raw_text(
                request,
                request.completion_token_ids,
                request.detokenizer.raw_text,
                emit_len,
            )
            if len(parser_raw_text) < request.emitted_raw_text_len:
                raise RuntimeError(
                    "Response-parser boundary precedes emitted text: "
                    f"emitted={request.emitted_raw_text_len} "
                    f"boundary={len(parser_raw_text)}."
                )
            raw_text_delta = parser_raw_text[
                request.emitted_raw_text_len:
            ]
            text = visible_text[request.emitted_text_len:emit_len]
            request.emitted_text_len = emit_len
            if stop_index is not None:
                final = request.detokenizer.finish(request.completion_token_ids)
                final_stop_index = _find_stop_index(
                    final.text,
                    request.stop,
                )
                if final_stop_index is None:
                    raise RuntimeError(
                        "Incremental stop match disappeared during final "
                        "detokenization."
                    )
                final_parser_raw_text = self._response_parser_raw_text(
                    request,
                    request.completion_token_ids,
                    final.raw_text,
                    final_stop_index,
                )
                try:
                    self.engine.abort_request(
                        seq_id, disposition="invalidate"
                    )
                except TypeError:
                    self.engine.abort_request(seq_id)
                if request.chain_id is not None:
                    request.chain_status = "invalidated"
                self._refresh_routing_snapshots()
            if text or raw_text_delta:
                self._publish_pending_token_event(request, text, raw_text_delta)
            if stop_index is not None:
                if (
                    len(final_parser_raw_text)
                    < request.emitted_raw_text_len
                ):
                    raise RuntimeError(
                        "Final response-parser text is shorter than its "
                        f"stream: emitted={request.emitted_raw_text_len} "
                        f"final={len(final_parser_raw_text)}."
                    )
                active.pop(seq_id, None)
                self._mark_request_terminal(seq_id, request)
                self._put(
                    request,
                    {
                        "type": "final",
                        "index": request.index,
                        "text": visible_text,
                        "raw_text": final_parser_raw_text,
                        "text_delta": visible_text[request.emitted_text_len:],
                        "finish_reason": "stop",
                        "prompt_tokens": (
                            request.prompt_token_count
                            if request.prompt_token_count is not None
                            else len(request.prompt_token_ids)
                        ),
                        "completion_tokens": len(request.completion_token_ids),
                        "token_ids": request.completion_token_ids,
                        "token_logprobs": request.completion_token_logprobs,
                        "top_logprobs": request.completion_top_logprobs,
                        "chain_id": request.chain_id,
                        "chain_status": request.chain_status,
                        "reused_tokens": request.reused_tokens,
                        "prefilled_tokens": request.prefilled_tokens,
                    },
                )

    def _publish_pending_token_event(
        self,
        request: _ActiveRequest,
        text: str,
        raw_text_delta: str,
    ):
        if request.stream:
            self._put(
                request,
                {
                    "type": "token",
                    "index": request.index,
                    "text": text,
                    "raw_text_delta": raw_text_delta,
                    "token_ids": list(request.pending_token_ids),
                    "token_logprobs": list(request.pending_token_logprobs),
                    "top_logprobs": list(request.pending_top_logprobs),
                    "chain_id": request.chain_id,
                    "chain_status": request.chain_status,
                    "reused_tokens": request.reused_tokens,
                    "prefilled_tokens": request.prefilled_tokens,
                },
            )
        request.pending_token_ids.clear()
        request.pending_token_logprobs.clear()
        request.pending_top_logprobs.clear()
        request.emitted_raw_text_len += len(raw_text_delta)

    def _mark_request_terminal(
        self,
        seq_id: int,
        request: _ActiveRequest,
    ) -> None:
        request.terminal.set()
        latest_request_tokens = getattr(
            self, "_latest_request_tokens", {}
        )
        if latest_request_tokens.get(int(seq_id)) == request.request_token:
            latest_request_tokens.pop(int(seq_id), None)

    @staticmethod
    def _resolve_abort(
        abort: _AbortRequest,
        *,
        error: BaseException | None = None,
    ) -> None:
        if abort.loop is None or abort.output_queue is None:
            return
        result = (
            {"type": "error", "message": f"{type(error).__name__}: {error}"}
            if error is not None
            else {"type": "result", "value": None}
        )
        abort.loop.call_soon_threadsafe(
            abort.output_queue.put_nowait,
            result,
        )

    def _drain_aborts(self, active: dict[int, _ActiveRequest]) -> bool:
        changed = False
        while True:
            try:
                abort = self._aborts.get_nowait()
            except queue.Empty:
                return changed
            chain_id = None
            request_token = None
            if isinstance(abort, _AbortRequest):
                seq_id = int(abort.seq_id)
                disposition = str(abort.disposition)
                chain_id = abort.chain_id
                request_token = abort.request_token
                if (
                    abort.terminal is not None
                    and abort.terminal.is_set()
                    and not abort.force
                ):
                    continue
            elif isinstance(abort, tuple) and len(abort) == 3:
                seq_id, disposition, chain_id = abort
            elif isinstance(abort, tuple):
                seq_id, disposition = abort
            else:
                seq_id, disposition = abort, "invalidate"
            latest_request_tokens = getattr(
                self, "_latest_request_tokens", {}
            )
            latest_request_token = latest_request_tokens.get(int(seq_id))
            if (
                request_token is not None
                and latest_request_token is not None
                and request_token != latest_request_token
            ):
                if isinstance(abort, _AbortRequest):
                    if abort.terminal is not None:
                        abort.terminal.set()
                    self._resolve_abort(abort)
                continue
            if seq_id in active:
                active_request = active[seq_id]
                if (
                    request_token is not None
                    and active_request.request_token is not None
                    and request_token != active_request.request_token
                ):
                    if isinstance(abort, _AbortRequest):
                        if abort.terminal is not None:
                            abort.terminal.set()
                        self._resolve_abort(abort)
                    continue
                try:
                    try:
                        self.engine.abort_request(
                            seq_id, disposition=disposition
                        )
                    except TypeError:
                        self.engine.abort_request(seq_id)
                except Exception as exc:
                    if isinstance(abort, _AbortRequest):
                        self._resolve_abort(abort, error=exc)
                    raise
                active.pop(seq_id)
                self._mark_request_terminal(seq_id, active_request)
                if isinstance(abort, _AbortRequest):
                    self._resolve_abort(abort)
                changed = True
            else:
                if isinstance(abort, _AbortRequest) and abort.force:
                    try:
                        if chain_id:
                            discard_chain = getattr(
                                self.engine,
                                "discard_chain",
                                None,
                            )
                            if not callable(discard_chain):
                                raise RuntimeError(
                                    "Engine does not expose discard_chain."
                                )
                            discarded = discard_chain(
                                chain_id,
                                expected_seq_id=seq_id,
                            )
                            changed = bool(discarded) or changed
                    except Exception as exc:
                        self._resolve_abort(abort, error=exc)
                    else:
                        self._resolve_abort(abort)
                    continue
                abort_request = getattr(
                    self.engine, "abort_request", None
                )
                if callable(abort_request):
                    try:
                        abort_request(
                            seq_id, disposition=disposition
                        )
                    except TypeError:
                        abort_request(seq_id)
                    if latest_request_token == request_token:
                        latest_request_tokens.pop(int(seq_id), None)
                    if isinstance(abort, _AbortRequest):
                        if abort.terminal is not None:
                            abort.terminal.set()
                        self._resolve_abort(abort)
                    changed = True
                elif disposition == "invalidate" and chain_id:
                    invalidate_chain = getattr(
                        self.engine, "invalidate_chain", None
                    )
                    if callable(invalidate_chain):
                        invalidate_chain(chain_id)
                        if latest_request_token == request_token:
                            latest_request_tokens.pop(int(seq_id), None)
                        if isinstance(abort, _AbortRequest):
                            if abort.terminal is not None:
                                abort.terminal.set()
                            self._resolve_abort(abort)
                        changed = True

    def _fail_pending_aborts(self, message: str) -> None:
        while True:
            try:
                abort = self._aborts.get_nowait()
            except queue.Empty:
                return
            if not isinstance(abort, _AbortRequest):
                continue
            if abort.terminal is not None:
                abort.terminal.set()
            self._resolve_abort(abort, error=RuntimeError(message))

    def _fail_active_requests(self, active: dict[int, _ActiveRequest], message: str):
        for seq_id, request in list(active.items()):
            active.pop(seq_id, None)
            self._mark_request_terminal(seq_id, request)
            try:
                self._put(request, {"type": "error", "message": message})
            except Exception:
                logger.exception("Failed to notify OpenAI request {} during dispatcher shutdown", seq_id)
            try:
                self.engine.abort_request(seq_id)
            except Exception:
                logger.exception("Failed to abort OpenAI request {} during dispatcher shutdown", seq_id)

    def _drain_queued_requests(self, message: str):
        while True:
            try:
                item = self._pending.get_nowait()
            except queue.Empty:
                break
            if item is not None and item is not _WAKEUP:
                try:
                    error = RuntimeError(message)
                    self._resolve_admission(item, error)
                    item.handle.terminal.set()
                    self._put(
                        item, {"type": "error", "message": message}
                    )
                except Exception:
                    logger.exception("Failed to notify queued OpenAI request during dispatcher shutdown")
        while True:
            try:
                item = self._controls.get_nowait()
            except queue.Empty:
                return
            try:
                self._put_control(item, {"type": "error", "message": message})
            except Exception:
                logger.exception("Failed to notify queued OpenAI control request during dispatcher shutdown")

    def _publish_finished(
        self,
        active: dict[int, _ActiveRequest],
        finished_outputs: list[
            tuple[
                int,
                list[int],
                list[float | None],
                list[dict[int, float] | None],
            ]
        ],
    ):
        for seq_id, completion_token_ids, token_logprobs, top_logprobs in finished_outputs:
            request = active.get(seq_id)
            if request is None:
                continue
            observed = len(request.completion_token_ids)
            final = request.detokenizer.finish(completion_token_ids)
            request.pending_token_ids.extend(completion_token_ids[observed:])
            request.pending_token_logprobs.extend(token_logprobs[observed:])
            request.pending_top_logprobs.extend(top_logprobs[observed:])
            active.pop(seq_id, None)
            request.completion_token_ids = list(completion_token_ids)
            request.completion_token_logprobs = list(token_logprobs)
            request.completion_top_logprobs = list(top_logprobs)
            text = final.text
            stop_index = _find_stop_index(text, request.stop)
            parser_raw_text = self._response_parser_raw_text(
                request,
                completion_token_ids,
                final.raw_text,
                stop_index if stop_index is not None else len(text),
            )
            if len(parser_raw_text) < request.emitted_raw_text_len:
                raise RuntimeError(
                    "Final response-parser text is shorter than its stream: "
                    f"emitted={request.emitted_raw_text_len} "
                    f"final={len(parser_raw_text)}."
                )
            parser_raw_text_delta = parser_raw_text[
                request.emitted_raw_text_len:
            ]
            ended_by_eos = (
                not request.ignore_eos
                and bool(completion_token_ids)
                and int(completion_token_ids[-1])
                in request.eos_token_ids
            )
            finish_reason = (
                "stop"
                if ended_by_eos
                or len(completion_token_ids) < request.max_tokens
                else "length"
            )
            if stop_index is not None:
                text = text[:stop_index]
                finish_reason = "stop"
                if request.chain_id is not None:
                    try:
                        self.engine.abort_request(
                            seq_id, disposition="invalidate"
                        )
                    except TypeError:
                        self.engine.abort_request(seq_id)
                    request.chain_status = "invalidated"
                    self._refresh_routing_snapshots()
            text_delta = text[request.emitted_text_len:]
            has_pending_logprobs = any(
                value is not None for value in request.pending_token_logprobs
            ) or any(value is not None for value in request.pending_top_logprobs)
            if request.pending_token_ids and (
                text_delta or parser_raw_text_delta or has_pending_logprobs
            ):
                self._publish_pending_token_event(
                    request,
                    text_delta,
                    parser_raw_text_delta,
                )
                request.emitted_text_len = len(text)
            self._mark_request_terminal(seq_id, request)
            self._put(
                request,
                {
                    "type": "final",
                    "index": request.index,
                    "text": text,
                    "raw_text": parser_raw_text,
                    "text_delta": text[request.emitted_text_len:],
                    "finish_reason": finish_reason,
                    "prompt_tokens": (
                        request.prompt_token_count
                        if request.prompt_token_count is not None
                        else len(request.prompt_token_ids)
                    ),
                    "completion_tokens": len(completion_token_ids),
                    "token_ids": completion_token_ids,
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs,
                    "chain_id": request.chain_id,
                    "chain_status": request.chain_status,
                    "reused_tokens": request.reused_tokens,
                    "prefilled_tokens": request.prefilled_tokens,
                },
            )
