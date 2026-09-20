import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
from unittest.mock import AsyncMock


class _FastTokenizerAdapter:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.is_fast = True
        self.backend_tokenizer = tokenizer._tokenizer
        self.chat_template = None
        self.bos_token = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids, skip_special_tokens=True):
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class _TransformersResponseTokenizer:
    response_template = None

    def __init__(self, *, xml_tools=False, minimax_tools=False, glm_tools=False):
        self.chat_template = "<think><tool_call>"
        if xml_tools:
            self.chat_template += "<function=name><parameter=key>"
        if minimax_tools:
            self.chat_template = (
                '<think><minimax:tool_call><invoke name="tool">'
                '<parameter name="key">'
            )
        if glm_tools:
            self.chat_template = (
                "<|assistant|><think><tool_call><arg_key><arg_value>"
            )

    def parse_response(self, response, schema, *, prefix=None):
        from transformers.utils.chat_parsing import parse_response

        return parse_response(response, schema, prefix=prefix)

    def get_response_parser(self, response_template=None, *, prefix=None):
        from transformers.utils.chat_parsing import ResponseParser

        return ResponseParser(response_template, prefix=prefix)


def _transformers_response_parser(
    *, xml_tools=False, minimax_tools=False, glm_tools=False
):
    from sparseengine.entrypoints.openai.serving.response_parsing import TransformersResponseParser

    parser = TransformersResponseParser.from_tokenizer(
        _TransformersResponseTokenizer(
            xml_tools=xml_tools,
            minimax_tools=minimax_tools,
            glm_tools=glm_tools,
        )
    )
    assert parser is not None
    return parser


def _byte_level_tokenizer(*, special_tokens=()):
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        ["训练数据：中文，日本語，한국어，café，🙂。"],
        vocab_size=256,
        min_frequency=100,
        special_tokens=list(special_tokens),
    )
    return _FastTokenizerAdapter(tokenizer)


async def _dispatcher_items_for_token_ids(
    tokenizer,
    token_ids,
    *,
    stop=(),
    incremental=True,
    stream=True,
):
    from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
    from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

    class Engine:
        def __init__(self):
            self.tokenizer = tokenizer
            self.last_step_token_outputs = []
            self.last_step_logprob_outputs = []
            self.aborted = []

        def abort_request(self, seq_id):
            self.aborted.append(seq_id)

        def exit(self):
            pass

    engine = Engine()
    dispatcher = AsyncEngineDispatcher(engine)
    output_queue = asyncio.Queue()
    active = {
        7: _ActiveRequest(
            index=0,
            loop=asyncio.get_running_loop(),
            output_queue=output_queue,
            prompt_token_ids=[10],
            max_tokens=len(token_ids),
            stop=list(stop),
            completion_token_ids=[],
            completion_token_logprobs=[],
            completion_top_logprobs=[],
            detokenizer=IncrementalDetokenizer(tokenizer),
            stream=stream,
        )
    }
    items = []
    try:
        if incremental:
            for token_id in token_ids:
                engine.last_step_token_outputs = [(7, [token_id])]
                engine.last_step_logprob_outputs = [(7, [None], [None])]
                dispatcher._publish_token_deltas(active)
                await asyncio.sleep(0)
                while not output_queue.empty():
                    items.append(output_queue.get_nowait())
                if 7 not in active:
                    break

        if 7 in active:
            dispatcher._publish_finished(
                active,
                [(7, token_ids, [None] * len(token_ids), [None] * len(token_ids))],
            )
            await asyncio.sleep(0)
            while not output_queue.empty():
                items.append(output_queue.get_nowait())
    finally:
        dispatcher.close()
    return items


async def _dispatcher_items_for_text(text, *, stop=()):
    tokenizer = _byte_level_tokenizer()
    return await _dispatcher_items_for_token_ids(
        tokenizer,
        tokenizer.encode(text),
        stop=stop,
    )


class _TestRequest:
    def __init__(self, app):
        self.app = app

    async def is_disconnected(self):
        return False


def _route_endpoint(app, path):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
        if hasattr(route, "effective_route_contexts"):
            for context in route.effective_route_contexts():
                original_route = context.original_route
                if getattr(original_route, "path", None) == path:
                    return original_route.endpoint
    raise AssertionError(f"route not found: {path}")


def _response_sse_events(chunks):
    text = "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in chunks)
    events = []
    for frame in text.split("\n\n"):
        if not frame:
            continue
        lines = frame.splitlines()
        data_lines = [line for line in lines if line.startswith("data: ")]
        if not data_lines:
            continue
        data = data_lines[-1].removeprefix("data: ")
        if data == "[DONE]":
            events.append(("[DONE]", None))
            continue
        event_lines = [line for line in lines if line.startswith("event: ")]
        event_type = event_lines[-1].removeprefix("event: ") if event_lines else None
        payload = json.loads(data)
        events.append((event_type, payload))
    return events


@unittest.skipIf(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("pydantic") is None,
    "OpenAI API server dependencies are not installed",
)
class OpenAIAPIServerTest(unittest.IsolatedAsyncioTestCase):
    def test_response_parser_cli_replaces_reasoning_parser(self):
        from sparseengine.entrypoints.openai import api_server

        parser = api_server.build_arg_parser()
        args = parser.parse_args(
            ["--model", "/tmp/model", "--response-parser", "minimax_m2"]
        )

        self.assertEqual(args.response_parser, "minimax_m2")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--model", "/tmp/model", "--reasoning-parser", "minimax_m2"]
            )

        for alias in ("auto", "glm47"):
            args = parser.parse_args(
                ["--model", "/tmp/model", "--response-parser", alias]
            )
            self.assertEqual(args.response_parser, alias)

    def test_incremental_detokenizer_waits_for_complete_unicode(self):
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训练")
        detokenizer = IncrementalDetokenizer(tokenizer)

        deltas = [detokenizer.push([token_id]) for token_id in token_ids]
        final = detokenizer.finish(token_ids)

        self.assertEqual("".join(delta.text for delta in deltas), "训练")
        self.assertEqual("".join(delta.raw_text for delta in deltas), "训练")
        self.assertNotIn("�", "".join(delta.text for delta in deltas))
        self.assertEqual(final.text, "训练")
        self.assertEqual(final.raw_text, "训练")
        self.assertEqual(final.text_delta, "")
        self.assertEqual(final.raw_text_delta, "")

    def test_incremental_detokenizer_handles_multilingual_batches(self):
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        text = "中文，日本語，한국어，café，🙂。"
        token_ids = tokenizer.encode(text)
        detokenizer = IncrementalDetokenizer(tokenizer)

        midpoint = len(token_ids) // 2
        deltas = [
            detokenizer.push(token_ids[:midpoint]),
            detokenizer.push(token_ids[midpoint:]),
        ]

        self.assertEqual("".join(delta.text for delta in deltas), text)
        self.assertEqual(detokenizer.finish(token_ids).text, text)

    def test_incremental_detokenizer_keeps_raw_special_tokens(self):
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer(special_tokens=["<special>"])
        token_id = tokenizer._tokenizer.token_to_id("<special>")
        detokenizer = IncrementalDetokenizer(tokenizer)

        delta = detokenizer.push([token_id])
        final = detokenizer.finish([token_id])

        self.assertEqual(delta.text, "")
        self.assertEqual(delta.raw_text, "<special>")
        self.assertEqual(final.text, "")
        self.assertEqual(final.raw_text, "<special>")

    def test_incremental_detokenizer_flushes_invalid_final_bytes(self):
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训")[:2]
        detokenizer = IncrementalDetokenizer(tokenizer)

        delta = detokenizer.push(token_ids)
        final = detokenizer.finish(token_ids)

        self.assertEqual(delta.text, "")
        self.assertEqual(final.text, "�")
        self.assertEqual(final.text_delta, "�")

    def test_incremental_detokenizer_reports_unpublished_final_suffix(self):
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训练")
        detokenizer = IncrementalDetokenizer(tokenizer)

        final = detokenizer.finish(token_ids)

        self.assertEqual(final.text, "训练")
        self.assertEqual(final.raw_text, "训练")
        self.assertEqual(final.text_delta, "训练")
        self.assertEqual(final.raw_text_delta, "训练")

    def test_incremental_detokenizer_rejects_non_fast_tokenizer(self):
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        class SlowTokenizer:
            is_fast = False

        with self.assertRaisesRegex(TypeError, "fast tokenizer backend"):
            IncrementalDetokenizer(SlowTokenizer())

    async def test_dispatcher_rejects_slow_tokenizer_before_admission(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        class SlowTokenizer:
            is_fast = False

        class Engine:
            tokenizer = SlowTokenizer()

            def __init__(self):
                self.added = False

            def add_request(self, _prompt, _sampling_params):
                self.added = True
                return 1

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit(
                "prompt",
                type("Sampling", (), {"max_tokens": 1})(),
                0,
            )
            item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(item["type"], "error")
        self.assertIn("fast tokenizer backend", item["message"])
        self.assertFalse(engine.added)

    async def test_dispatcher_fatal_failure_marks_unready_and_notifies_supervisor(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.aborted = []

            def worker_routing_load(self):
                return {"active_requests": 1}

            def prefix_cache_routing_snapshot(self):
                return type(
                    "Snapshot",
                    (),
                    {"match": lambda _self, _token_ids: {"matched_tokens": 0}},
                )()

            def add_request(self, _prompt, _sampling_params):
                return 17

            def step(self):
                raise RuntimeError("fatal engine step")

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                pass

        notified = threading.Event()
        dispatcher = AsyncEngineDispatcher(Engine())
        dispatcher.set_fatal_callback(lambda _message: notified.set())
        try:
            sampling = type("Sampling", (), {"max_tokens": 1})()
            handle = await dispatcher.submit("prompt", sampling, 0)
            item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
            self.assertTrue(await asyncio.to_thread(notified.wait, 1))
        finally:
            dispatcher.close()

        self.assertEqual(item["type"], "error")
        self.assertIn("fatal engine step", item["message"])
        self.assertFalse(dispatcher.is_ready)
        self.assertIn("fatal engine step", dispatcher.failure_message)
        with self.assertRaisesRegex(RuntimeError, "fatal engine step"):
            dispatcher.worker_routing_load_snapshot()
        with self.assertRaisesRegex(RuntimeError, "fatal engine step"):
            dispatcher.prefix_cache_routing_match([1, 2, 3])

    async def test_dispatcher_fatal_failure_releases_concurrent_control(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        step_started = threading.Event()
        release_step = threading.Event()

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer

            def add_request(self, _prompt, _sampling_params):
                return 17

            def step(self):
                step_started.set()
                release_step.wait(1)
                raise RuntimeError("fatal engine step")

            def worker_load(self):
                return {"active_requests": 1}

            def abort_request(self, _seq_id):
                pass

            def exit(self):
                pass

        dispatcher = AsyncEngineDispatcher(Engine())
        try:
            sampling = type("Sampling", (), {"max_tokens": 1})()
            handle = await dispatcher.submit("prompt", sampling, 0)
            self.assertTrue(await asyncio.to_thread(step_started.wait, 1))
            control_task = asyncio.create_task(dispatcher.control("worker_load"))
            await asyncio.sleep(0)
            self.assertEqual(dispatcher._controls.qsize(), 1)
            release_step.set()

            item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
            with self.assertRaisesRegex(RuntimeError, "fatal engine step"):
                await asyncio.wait_for(control_task, timeout=1)
        finally:
            release_step.set()
            dispatcher.close()

        self.assertEqual(item["type"], "error")
        self.assertEqual(dispatcher._controls.qsize(), 0)

    async def test_dispatcher_refreshes_snapshots_before_control_result(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        class Snapshot:
            def __init__(self, generation):
                self.generation = generation

            def match(self, _token_ids):
                return {"generation": self.generation}

        class Engine:
            tokenizer = object()

            def __init__(self):
                self.generation = 0

            def mutate(self):
                self.generation = 1
                return "mutated"

            def worker_routing_load(self):
                return {"generation": self.generation}

            def prefix_cache_routing_snapshot(self):
                return Snapshot(self.generation)

            def exit(self):
                pass

        dispatcher = AsyncEngineDispatcher(Engine())
        published_snapshot_states = []
        original_put_control = dispatcher._put_control

        def put_after_observing_snapshots(request, item):
            if item["type"] == "result":
                load = dispatcher.worker_routing_load_snapshot()
                match = dispatcher.prefix_cache_routing_match([1])
                published_snapshot_states.append(
                    (load["generation"], match["generation"])
                )
            original_put_control(request, item)

        dispatcher._put_control = put_after_observing_snapshots
        try:
            result = await asyncio.wait_for(
                dispatcher.control("mutate"),
                timeout=1,
            )
        finally:
            dispatcher.close()

        self.assertEqual(result, "mutated")
        self.assertEqual(published_snapshot_states, [(1, 1)])

    async def test_dispatcher_snapshot_refresh_failure_is_fatal(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        class Engine:
            tokenizer = object()

            def __init__(self):
                self.fail_snapshot = False

            def mutate(self):
                self.fail_snapshot = True

            def worker_routing_load(self):
                if self.fail_snapshot:
                    raise RuntimeError("snapshot refresh failed")
                return {"active_requests": 0}

            def exit(self):
                pass

        failed = threading.Event()
        dispatcher = AsyncEngineDispatcher(Engine())
        dispatcher.set_fatal_callback(lambda _message: failed.set())
        try:
            with self.assertRaisesRegex(RuntimeError, "snapshot refresh failed"):
                await asyncio.wait_for(
                    dispatcher.control("mutate"),
                    timeout=1,
                )
            self.assertTrue(await asyncio.to_thread(failed.wait, 1))
        finally:
            dispatcher.close()

        self.assertFalse(dispatcher.is_ready)
        self.assertIn("snapshot refresh failed", dispatcher.failure_message)

    async def test_dispatcher_abort_failure_notifies_supervisor_and_rejects_new_work(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        step_started = threading.Event()
        release_step = threading.Event()

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def add_request(self, _prompt, _sampling_params):
                return 17

            def step(self):
                step_started.set()
                release_step.wait(1)
                return [], 0

            def abort_request(self, _seq_id):
                raise RuntimeError("abort failed")

            def exit(self):
                pass

        notified = threading.Event()
        dispatcher = AsyncEngineDispatcher(Engine())
        dispatcher.set_fatal_callback(lambda _message: notified.set())
        try:
            sampling = type("Sampling", (), {"max_tokens": 1})()
            handle = await dispatcher.submit("prompt", sampling, 0)
            self.assertTrue(await asyncio.to_thread(step_started.wait, 1))
            dispatcher.cancel(handle)
            release_step.set()
            self.assertTrue(await asyncio.to_thread(notified.wait, 1))

            item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
            rejected = await dispatcher.submit("later", sampling, 1)
            rejected_item = await asyncio.wait_for(rejected.output_queue.get(), timeout=1)
            with self.assertRaisesRegex(RuntimeError, "abort failed"):
                await asyncio.wait_for(dispatcher.control("worker_load"), timeout=1)
        finally:
            release_step.set()
            dispatcher.close()

        self.assertEqual(item["type"], "error")
        self.assertIn("abort failed", item["message"])
        self.assertEqual(rejected_item["type"], "error")
        self.assertIn("abort failed", rejected_item["message"])
        self.assertFalse(dispatcher.is_ready)

    async def test_dispatcher_refreshes_snapshot_after_last_active_abort(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        step_started = threading.Event()
        release_step = threading.Event()
        abort_called = threading.Event()
        refreshed_after_abort = threading.Event()

        class Engine:
            last_step_token_outputs = []
            last_step_logprob_outputs = []

            def __init__(self):
                self.tokenizer = tokenizer
                self.active_requests = 0

            def worker_routing_load(self):
                if abort_called.is_set() and self.active_requests == 0:
                    refreshed_after_abort.set()
                return {"active_requests": self.active_requests}

            def add_request(self, _prompt, _sampling_params):
                self.active_requests = 1
                return 17

            def step(self):
                step_started.set()
                release_step.wait(1)
                return [], 0

            def abort_request(self, _seq_id):
                self.active_requests = 0
                abort_called.set()

            def exit(self):
                pass

        dispatcher = AsyncEngineDispatcher(Engine())
        try:
            sampling = type("Sampling", (), {"max_tokens": 1})()
            handle = await dispatcher.submit("prompt", sampling, 0)
            self.assertTrue(await asyncio.to_thread(step_started.wait, 1))
            dispatcher.cancel(handle)
            release_step.set()
            self.assertTrue(await asyncio.to_thread(abort_called.wait, 1))
            self.assertTrue(
                await asyncio.to_thread(refreshed_after_abort.wait, 1)
            )
            self.assertEqual(
                dispatcher.worker_routing_load_snapshot()["active_requests"],
                0,
            )
        finally:
            release_step.set()
            dispatcher.close()

    async def test_dispatcher_stop_abort_refreshes_snapshots_before_events(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        output_token_ids = tokenizer.encode("visibleSTOP")

        class Snapshot:
            def __init__(self, active_requests):
                self.active_requests = active_requests

            def match(self, _token_ids):
                return {"active_requests": self.active_requests}

        class Engine:
            last_step_logprob_outputs = []

            def __init__(self):
                self.tokenizer = tokenizer
                self.active_requests = 0
                self.last_step_token_outputs = []

            def worker_routing_load(self):
                return {"active_requests": self.active_requests}

            def prefix_cache_routing_snapshot(self):
                return Snapshot(self.active_requests)

            def add_request(self, _prompt, _sampling_params):
                self.active_requests = 1
                return 17

            def step(self):
                self.last_step_token_outputs = [(17, output_token_ids)]
                return [], 0

            def abort_request(self, _seq_id):
                self.active_requests = 0

            def exit(self):
                pass

        dispatcher = AsyncEngineDispatcher(Engine())
        published_snapshot_states = []
        original_put = dispatcher._put

        def put_after_observing_snapshots(request, item):
            if item["type"] in {"token", "final"}:
                load = dispatcher.worker_routing_load_snapshot()
                match = dispatcher.prefix_cache_routing_match([1])
                published_snapshot_states.append(
                    (
                        item["type"],
                        load["active_requests"],
                        match["active_requests"],
                    )
                )
            original_put(request, item)

        dispatcher._put = put_after_observing_snapshots
        try:
            sampling = type(
                "Sampling",
                (),
                {"max_tokens": len(output_token_ids)},
            )()
            handle = await dispatcher.submit(
                "prompt",
                sampling,
                0,
                stop=["STOP"],
            )
            items = []
            while not items or items[-1]["type"] != "final":
                items.append(
                    await asyncio.wait_for(handle.output_queue.get(), timeout=1)
                )
        finally:
            dispatcher.close()

        self.assertEqual([item["type"] for item in items], ["token", "final"])
        self.assertEqual(
            published_snapshot_states,
            [("token", 0, 0), ("final", 0, 0)],
        )
        self.assertEqual(
            dispatcher._worker_routing_load_snapshot["active_requests"],
            0,
        )

    async def test_dispatcher_stop_abort_refresh_failure_releases_request(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        output_token_ids = tokenizer.encode("visibleSTOP")

        class Engine:
            last_step_logprob_outputs = []

            def __init__(self):
                self.tokenizer = tokenizer
                self.active_requests = 0
                self.fail_snapshot = False
                self.last_step_token_outputs = []

            def worker_routing_load(self):
                if self.fail_snapshot:
                    raise RuntimeError("stop snapshot refresh failed")
                return {"active_requests": self.active_requests}

            def add_request(self, _prompt, _sampling_params):
                self.active_requests = 1
                return 17

            def step(self):
                self.last_step_token_outputs = [(17, output_token_ids)]
                return [], 0

            def abort_request(self, _seq_id):
                self.active_requests = 0
                self.fail_snapshot = True

            def exit(self):
                pass

        failed = threading.Event()
        dispatcher = AsyncEngineDispatcher(Engine())
        dispatcher.set_fatal_callback(lambda _message: failed.set())
        try:
            sampling = type(
                "Sampling",
                (),
                {"max_tokens": len(output_token_ids)},
            )()
            handle = await dispatcher.submit(
                "prompt",
                sampling,
                0,
                stop=["STOP"],
            )
            item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
            self.assertTrue(await asyncio.to_thread(failed.wait, 1))
        finally:
            dispatcher.close()

        self.assertEqual(item["type"], "error")
        self.assertIn("stop snapshot refresh failed", item["message"])
        self.assertFalse(dispatcher.is_ready)

    def test_health_is_readiness_aware_but_livez_stays_available(self):
        from sparseengine.entrypoints.openai import api_server

        class Engine:
            tokenizer = object()
            config = type("Config", (), {"sparse_method": ""})()

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        health_endpoint = _route_endpoint(app, "/health")
        ready_endpoint = _route_endpoint(app, "/readyz")
        live_endpoint = _route_endpoint(app, "/livez")
        try:
            self.assertEqual(health_endpoint(_TestRequest(app)).status_code, 200)
            self.assertEqual(ready_endpoint(_TestRequest(app)).status_code, 200)
            app.state.dispatcher._failed_message = "OutOfMemoryError: CUDA out of memory"
            self.assertEqual(health_endpoint(_TestRequest(app)).status_code, 503)
            self.assertEqual(ready_endpoint(_TestRequest(app)).status_code, 503)
            self.assertEqual(live_endpoint().status_code, 200)
        finally:
            app.state.dispatcher.close()

    def test_cli_server_exits_nonzero_after_fatal_dispatcher_failure(self):
        from sparseengine.entrypoints.openai import api_server

        class Dispatcher:
            failure_message = None
            callback = None

            def set_fatal_callback(self, callback):
                self.callback = callback

        dispatcher = Dispatcher()
        app = type(
            "App",
            (),
            {"state": type("State", (), {"dispatcher": dispatcher})()},
        )()
        servers = []

        class Server:
            def __init__(self, _config):
                self.should_exit = False
                servers.append(self)

            def run(self):
                dispatcher.failure_message = "OutOfMemoryError: CUDA out of memory"
                dispatcher.callback(dispatcher.failure_message)

        with patch("uvicorn.Config", return_value=object()), patch("uvicorn.Server", Server):
            exit_code = api_server._run_server(app, host="127.0.0.1", port=18000)

        self.assertEqual(exit_code, 1)
        self.assertTrue(servers[0].should_exit)

    def test_incremental_detokenizer_rejects_final_token_mismatch(self):
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        first = tokenizer.encode("训")
        second = tokenizer.encode("练")
        detokenizer = IncrementalDetokenizer(tokenizer)
        detokenizer.push(first)

        with self.assertRaisesRegex(RuntimeError, "token history mismatch"):
            detokenizer.finish(second)

    def test_incremental_detokenizers_keep_request_state_isolated(self):
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        first_ids = tokenizer.encode("训练")
        second_ids = tokenizer.encode("中文")
        first = IncrementalDetokenizer(tokenizer)
        second = IncrementalDetokenizer(tokenizer)

        first_delta = first.push(first_ids[:3])
        second_delta = second.push(second_ids)
        final_delta = first.push(first_ids[3:])

        self.assertEqual(first_delta.text, "训")
        self.assertEqual(second_delta.text, "中文")
        self.assertEqual(final_delta.text, "练")
        self.assertEqual(first.finish(first_ids).text, "训练")
        self.assertEqual(second.finish(second_ids).text, "中文")

    def test_incremental_detokenizer_rejects_canonical_text_mismatch(self):
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("a")
        detokenizer = IncrementalDetokenizer(tokenizer)
        detokenizer.push(token_ids)
        tokenizer.decode = lambda _ids, skip_special_tokens=True: "different"

        with self.assertRaisesRegex(RuntimeError, "not a prefix"):
            detokenizer.finish(token_ids)

    async def test_completion_response_collects_usage_and_sorts_choices(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _completion_response

        queue_0 = asyncio.Queue()
        queue_1 = asyncio.Queue()
        await queue_0.put(
            {
                "type": "final",
                "index": 1,
                "text": "second",
                "finish_reason": "length",
                "prompt_tokens": 2,
                "completion_tokens": 3,
            }
        )
        await queue_1.put(
            {
                "type": "final",
                "index": 0,
                "text": "first",
                "finish_reason": "stop",
                "prompt_tokens": 5,
                "completion_tokens": 7,
            }
        )

        handles = [
            RequestHandle(output_queue=queue_0, cancelled=threading.Event()),
            RequestHandle(output_queue=queue_1, cancelled=threading.Event()),
        ]

        response = await _completion_response("cmpl-test", 123, "model-a", handles)

        self.assertEqual([choice["text"] for choice in response["choices"]], ["first", "second"])
        self.assertEqual(response["usage"], {
            "prompt_tokens": 7,
            "completion_tokens": 10,
            "total_tokens": 17,
            "prompt_tokens_details": {"cached_tokens": 0},
        })

    async def test_dispatcher_records_prompt_cache_hits(self):
        from sparseengine.entrypoints.openai.api_server import (
            AsyncEngineDispatcher,
            RequestHandle,
            _ActiveRequest,
        )
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        handle = RequestHandle(
            output_queue=asyncio.Queue(),
            cancelled=threading.Event(),
        )
        request = _ActiveRequest(
            index=0,
            loop=asyncio.get_running_loop(),
            output_queue=handle.output_queue,
            prompt_token_ids=[1, 2, 3, 4, 5, 6],
            max_tokens=1,
            stop=[],
            completion_token_ids=[],
            completion_token_logprobs=[],
            completion_top_logprobs=[],
            detokenizer=IncrementalDetokenizer(tokenizer),
            handle=handle,
        )
        dispatcher = object.__new__(AsyncEngineDispatcher)
        dispatcher.engine = type(
            "Engine", (), {"last_step_prompt_cache_hits": [(7, 4)]}
        )()

        dispatcher._update_prompt_cache_usage({7: request})

        self.assertEqual((request.reused_tokens, request.prefilled_tokens), (4, 2))
        self.assertEqual((handle.reused_tokens, handle.prefilled_tokens), (4, 2))

    def test_sse_serializes_openai_data_frame(self):
        from sparseengine.entrypoints.openai.api_server import _sse
        from sparseengine.entrypoints.openai.serving.chat import (
            _chat_stream_chunk,
        )

        frame = _sse({"text": "hello"})

        self.assertTrue(frame.startswith("data: "))
        self.assertTrue(frame.endswith("\n\n"))
        self.assertEqual(json.loads(frame.removeprefix("data: ")), {"text": "hello"})
        non_chain = json.loads(
            _chat_stream_chunk(
                "chatcmpl-test",
                123,
                "model",
                0,
                {"content": "x"},
            ).removeprefix("data: ")
        )
        self.assertNotIn("chain_id", non_chain)

    def test_deltakv_serving_method_fails_fast(self):
        from sparseengine.entrypoints.openai.api_server import _validate_serving_method

        with self.assertRaisesRegex(ValueError, "not supported"):
            _validate_serving_method({"sparse_method": "deltakv"})
        with self.assertRaisesRegex(ValueError, "not supported"):
            _validate_serving_method({"sparse_method": "deltakv-standalone"})

    def test_completion_request_rejects_unknown_fields(self):
        from pydantic import ValidationError

        from sparseengine.entrypoints.openai.api_server import ChatCompletionRequest, CompletionRequest

        with self.assertRaises(ValidationError):
            CompletionRequest(model="m", prompt="p", suffix="ignored")
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "assistant", "content": "p", "tool_calls": []}],
                metadata={"trace": "x"},
            )

    def test_chat_request_accepts_claw_eval_tool_metadata(self):
        from sparseengine.entrypoints.openai.api_server import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="m",
            messages=[
                {"role": "user", "content": "p"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ],
            stream=True,
            stream_options={"include_usage": True},
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        self.assertTrue(request.stream_options["include_usage"])
        self.assertEqual(ChatMessage(role="assistant").content, None)

    def test_chat_message_tool_fields_are_role_scoped(self):
        from pydantic import ValidationError

        from sparseengine.entrypoints.openai.api_server import ChatMessage

        with self.assertRaisesRegex(ValidationError, "tool_calls is only valid"):
            ChatMessage(role="user", content="question", tool_calls=[])
        with self.assertRaisesRegex(ValidationError, "tool_call_id is only valid"):
            ChatMessage(role="assistant", content="answer", tool_call_id="call_1")
        with self.assertRaisesRegex(ValidationError, "require tool_call_id"):
            ChatMessage(role="tool", content="result")
        with self.assertRaisesRegex(ValidationError, "require content"):
            ChatMessage(role="tool", tool_call_id="call_1")
        with self.assertRaisesRegex(ValidationError, "non-empty id"):
            ChatMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            )

    def test_logprob_request_limits_match_openai_bounds(self):
        from pydantic import ValidationError

        from sparseengine.entrypoints.openai.api_server import ChatCompletionRequest, CompletionRequest

        with self.assertRaises(ValidationError):
            CompletionRequest(model="m", prompt="p", logprobs=6)
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "p"}],
                logprobs=True,
                top_logprobs=21,
            )

    def test_sampling_penalty_request_bounds_and_mapping(self):
        from pydantic import ValidationError

        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            CompletionRequest,
            ResponseRequest,
            _sampling_params_from_request,
            _sampling_params_from_response_request,
        )
        from sparseengine.sampling_params import DEFAULT_MAX_TOKENS

        completion_params = _sampling_params_from_request(
            CompletionRequest(
                model="m",
                prompt="p",
                presence_penalty=-0.5,
                repetition_penalty=0.8,
            )
        )
        chat_params = _sampling_params_from_request(
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "p"}],
                presence_penalty=0.75,
                repetition_penalty=1.2,
            )
        )
        response_params = _sampling_params_from_response_request(
            ResponseRequest(
                model="m",
                input="p",
                presence_penalty=2.0,
                repetition_penalty=1.5,
            )
        )

        self.assertEqual(completion_params.presence_penalty, -0.5)
        self.assertEqual(completion_params.repetition_penalty, 0.8)
        self.assertEqual(completion_params.max_tokens, DEFAULT_MAX_TOKENS)
        self.assertEqual(chat_params.presence_penalty, 0.75)
        self.assertEqual(chat_params.repetition_penalty, 1.2)
        self.assertEqual(chat_params.max_tokens, DEFAULT_MAX_TOKENS)
        self.assertEqual(response_params.presence_penalty, 2.0)
        self.assertEqual(response_params.repetition_penalty, 1.5)
        self.assertEqual(response_params.max_tokens, DEFAULT_MAX_TOKENS)

        for request_type, request_fields in (
            (CompletionRequest, {"prompt": "p"}),
            (
                ChatCompletionRequest,
                {"messages": [{"role": "user", "content": "p"}]},
            ),
            (ResponseRequest, {"input": "p"}),
        ):
            with self.subTest(request_type=request_type.__name__, field="presence"):
                with self.assertRaises(ValidationError):
                    request_type(
                        model="m",
                        presence_penalty=2.01,
                        **request_fields,
                    )
            with self.subTest(request_type=request_type.__name__, field="repetition"):
                with self.assertRaises(ValidationError):
                    request_type(
                        model="m",
                        repetition_penalty=0.0,
                        **request_fields,
                    )

    def test_stop_with_logprobs_fails_fast(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            CompletionRequest,
            _validate_chat_request,
            _validate_request,
        )

        with self.assertRaises(HTTPException):
            _validate_request(
                CompletionRequest(model="m", prompt="p", stop="END", logprobs=1),
                "m",
            )
        with self.assertRaises(HTTPException):
            _validate_chat_request(
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    stop="END",
                    logprobs=True,
                ),
                "m",
            )

    def test_chat_prompt_uses_fallback_without_chat_template(self):
        from sparseengine.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = None

            def apply_chat_template(self, *_args, **_kwargs):
                raise AssertionError("chat_template is not configured")

        prompt = _chat_prompt(Tokenizer(), [ChatMessage(role="user", content="hello")])

        self.assertEqual(prompt, "user: hello\nassistant:")

    def test_chat_append_prompt_renders_only_new_suffix(self):
        from sparseengine.entrypoints.openai.render import (
            _chat_request_append_prompt,
        )
        from sparseengine.entrypoints.openai.protocol.chat import (
            ChatCompletionRequest,
        )

        class Tokenizer:
            chat_template = "template"

            def apply_chat_template(
                self,
                chat,
                *,
                tokenize,
                add_generation_prompt,
                **_kwargs,
            ):
                self.assertFalse(tokenize)
                rendered = "".join(
                    f"<{message['role']}>{message.get('content') or ''}</{message['role']}>"
                    for message in chat
                )
                if add_generation_prompt:
                    rendered += "<assistant>"
                return rendered

            def assertFalse(self, value):
                if value:
                    raise AssertionError(value)

        request = ChatCompletionRequest(
            model="m",
            chain_id="chain-a",
            chain_append_start=2,
            messages=[
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
                {
                    "role": "tool",
                    "tool_call_id": "call-a",
                    "content": "result",
                },
            ],
        )

        assert _chat_request_append_prompt(Tokenizer(), request) == (
            "<tool>result</tool><assistant>"
        )

    def test_chat_prompt_maps_developer_and_text_parts_for_templates(self):
        from sparseengine.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None

            def apply_chat_template(self, chat, **_kwargs):
                self.chat = chat
                return "rendered"

        tokenizer = Tokenizer()
        prompt = _chat_prompt(
            tokenizer,
            [
                ChatMessage(
                    role="developer",
                    content=[
                        {"type": "text", "text": "policy"},
                        {"type": "text", "text": "details"},
                    ],
                )
            ],
        )

        self.assertEqual(prompt, "rendered")
        self.assertEqual(tokenizer.chat, [{"role": "system", "content": "policy\ndetails"}])

    def test_chat_prompt_preserves_multimodal_parts_for_model_processor(self):
        from sparseengine.entrypoints.openai.api_server import ChatMessage, _chat_prompt
        from sparseengine.multimodal import MultiModalPrompt

        prompt = _chat_prompt(
            object(),
            [
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "image_url", "image_url": {"url": "https://x/image.png"}},
                        {"type": "text", "text": "describe"},
                    ],
                )
            ],
        )

        self.assertIsInstance(prompt, MultiModalPrompt)
        self.assertEqual(prompt.messages[0]["content"][0]["type"], "image_url")
        self.assertEqual(prompt.messages[0]["content"][1]["text"], "describe")

    def test_chat_prompt_preserves_reasoning_content_for_templates(self):
        from sparseengine.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None

            def apply_chat_template(self, chat, **_kwargs):
                self.chat = chat
                return "rendered"

        tokenizer = Tokenizer()
        prompt = _chat_prompt(
            tokenizer,
            [
                ChatMessage(
                    role="assistant",
                    content="answer",
                    reasoning_content="reason",
                )
            ],
        )

        self.assertEqual(prompt, "rendered")
        self.assertEqual(
            tokenizer.chat,
            [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}],
        )

    def test_chat_prompt_accepts_reasoning_alias_for_templates(self):
        from sparseengine.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None

            def apply_chat_template(self, chat, **_kwargs):
                self.chat = chat
                return "rendered"

        tokenizer = Tokenizer()
        message = ChatMessage.model_validate(
            {"role": "assistant", "content": "answer", "reasoning": "reason"}
        )

        self.assertEqual(_chat_prompt(tokenizer, [message]), "rendered")
        self.assertEqual(message.reasoning_content, "reason")
        self.assertEqual(
            tokenizer.chat,
            [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}],
        )

    def test_reasoning_content_requires_assistant_role_and_chat_template(self):
        from pydantic import ValidationError

        from sparseengine.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        with self.assertRaisesRegex(ValidationError, "only valid for assistant"):
            ChatMessage(role="user", content="question", reasoning_content="reason")

        class Tokenizer:
            chat_template = None

        with self.assertRaisesRegex(ValueError, "requires a tokenizer chat_template"):
            _chat_prompt(
                Tokenizer(),
                [ChatMessage(role="assistant", content="answer", reasoning_content="reason")],
            )

    def test_chat_template_kwargs_enable_thinking_passes_to_tokenizer(self):
        from sparseengine.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.kwargs = None

            def apply_chat_template(self, _chat, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        tokenizer = Tokenizer()
        prompt = _chat_prompt(
            tokenizer,
            [ChatMessage(role="user", content="hello")],
            {"enable_thinking": False},
        )

        self.assertEqual(prompt, "rendered")
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_top_level_enable_thinking_resolves_to_chat_template_kwargs(self):
        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            resolve_chat_template_kwargs,
        )

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            enable_thinking=False,
        )

        self.assertEqual(resolve_chat_template_kwargs(request), {"enable_thinking": False})

    def test_top_level_enable_thinking_conflict_fails_fast(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import ChatCompletionRequest, _validate_chat_request

        class Tokenizer:
            chat_template = "template"

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": True},
        )

        with self.assertRaisesRegex(HTTPException, "conflicts") as ctx:
            _validate_chat_request(request, "m", Tokenizer())
        self.assertEqual(ctx.exception.status_code, 400)

    def test_vllm_template_kwargs_are_normalized(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _chat_request_prompt,
            _validate_chat_request,
            resolve_chat_template_kwargs,
        )

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.kwargs = None

            def apply_chat_template(self, _chat, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            preserve_thinking=True,
            chat_template_kwargs={"preserve_thinking": True, "custom_flag": "value"},
        )

        tokenizer = Tokenizer()
        _validate_chat_request(request, "m", tokenizer)
        self.assertEqual(
            resolve_chat_template_kwargs(request),
            {"preserve_thinking": True, "custom_flag": "value"},
        )
        self.assertEqual(_chat_request_prompt(tokenizer, request), "rendered")
        self.assertIs(tokenizer.kwargs["preserve_thinking"], True)
        self.assertEqual(tokenizer.kwargs["custom_flag"], "value")

        conflicting = request.model_copy(update={"preserve_thinking": False})
        with self.assertRaisesRegex(HTTPException, "conflicts"):
            _validate_chat_request(conflicting, "m", Tokenizer())

    def test_chat_reasoning_effort_controls_thinking(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _validate_chat_request,
            resolve_chat_template_kwargs,
        )

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            reasoning_effort="none",
        )
        self.assertEqual(resolve_chat_template_kwargs(request), {"enable_thinking": False})

        conflicting = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            reasoning_effort="high",
            enable_thinking=False,
        )
        with self.assertRaisesRegex(HTTPException, "conflicts"):
            _validate_chat_request(conflicting, "m")

    def test_chat_accepts_store_false_and_rejects_store_true(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import ChatCompletionRequest, _validate_chat_request

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            store=False,
        )
        _validate_chat_request(request, "m")

        with self.assertRaisesRegex(HTTPException, "store=true") as ctx:
            _validate_chat_request(request.model_copy(update={"store": True}), "m")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_chat_prompt_passes_tools_and_tool_history(self):
        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _chat_request_prompt,
        )

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None
                self.tools = None

            def apply_chat_template(self, chat, tools=None, **_kwargs):
                self.chat = chat
                self.tools = tools
                return "rendered"

        request = ChatCompletionRequest(
            model="m",
            messages=[
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "need a lookup",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        tokenizer = Tokenizer()

        self.assertEqual(_chat_request_prompt(tokenizer, request), "rendered")
        self.assertEqual(tokenizer.chat[1]["reasoning_content"], "need a lookup")
        self.assertEqual(tokenizer.chat[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(tokenizer.chat[1]["tool_calls"][0]["function"]["arguments"], {"city": "Paris"})
        self.assertEqual(tokenizer.chat[2]["tool_call_id"], "call_1")
        self.assertEqual(tokenizer.tools[0]["name"], "get_weather")

    def test_chat_prompt_adapts_tools_for_minimax_template(self):
        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _chat_request_prompt,
        )

        class Tokenizer:
            chat_template = "<minimax:tool_call>{{ tool.function }}"

            def __init__(self):
                self.tools = None

            def apply_chat_template(self, _chat, tools=None, **_kwargs):
                self.tools = tools
                return "rendered"

        tokenizer = Tokenizer()
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "weather"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

        self.assertEqual(_chat_request_prompt(tokenizer, request), "rendered")
        self.assertEqual(
            tokenizer.tools,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

    def test_multimodal_chat_adapts_tools_for_gemma4_template(self):
        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _chat_request_prompt,
        )
        from sparseengine.multimodal import MultiModalPrompt

        class Tokenizer:
            chat_template = """
                {% macro format_function_declaration(tool_data) %}
                {{ tool_data['function']['name'] }}
                {% endmacro %}
                {% for tool in tools %}{{ format_function_declaration(tool) }}{% endfor %}
            """

        request = ChatCompletionRequest(
            model="m",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x/image.png"},
                        },
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

        prompt = _chat_request_prompt(Tokenizer(), request)

        self.assertIsInstance(prompt, MultiModalPrompt)
        self.assertEqual(
            prompt.tools,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

    def test_chat_prompt_rejects_invalid_tool_history_arguments(self):
        from sparseengine.entrypoints.openai.api_server import ChatMessage, _chat_prompt

        class Tokenizer:
            chat_template = "template"

            def apply_chat_template(self, _chat, **_kwargs):
                raise AssertionError("invalid tool history must fail before rendering")

        message = ChatMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "not-json"},
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            _chat_prompt(Tokenizer(), [message])

    def test_chat_tool_controls_and_template_support_fail_fast(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _chat_request_prompt,
            _validate_chat_request,
        )

        tool = {"type": "function", "function": {"name": "search", "parameters": {}}}
        for request in (
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "p"}],
                tools=[tool],
                tool_choice="required",
            ),
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "p"}],
                tools=[tool],
                parallel_tool_calls=False,
            ),
        ):
            with self.assertRaises(HTTPException):
                _validate_chat_request(request, "m")

        class NoTemplateTokenizer:
            chat_template = None

        with self.assertRaisesRegex(ValueError, "tools requires"):
            _chat_request_prompt(
                NoTemplateTokenizer(),
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    tools=[tool],
                ),
            )

        class NoToolsTokenizer:
            chat_template = "template"

            def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
                del chat, tokenize, add_generation_prompt
                return "rendered"

        with self.assertRaisesRegex(ValueError, "does not support tools"):
            _chat_request_prompt(
                NoToolsTokenizer(),
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    tools=[tool],
                ),
            )

    def test_chat_template_kwargs_validation_is_explicit(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import ChatCompletionRequest, _validate_chat_request

        class Tokenizer:
            chat_template = "template"

        for kwargs in [
            {"enable_thinking": "false"},
            {"preserve_thinking": "true"},
        ]:
            with self.assertRaises(HTTPException) as type_ctx:
                _validate_chat_request(
                    ChatCompletionRequest(
                        model="m",
                        messages=[{"role": "user", "content": "p"}],
                        chat_template_kwargs=kwargs,
                    ),
                    "m",
                    Tokenizer(),
                )
            self.assertEqual(type_ctx.exception.status_code, 400)

        class NoTemplateTokenizer:
            chat_template = None

        with self.assertRaises(HTTPException) as template_ctx:
            _validate_chat_request(
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    chat_template_kwargs={"enable_thinking": False},
                ),
                "m",
                NoTemplateTokenizer(),
            )
        self.assertEqual(template_ctx.exception.status_code, 400)

    def test_chat_max_completion_tokens_maps_to_sampling_params(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            _sampling_params_from_request,
            _validate_chat_request,
        )

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            max_completion_tokens=32,
        )

        _validate_chat_request(request, "m")
        self.assertEqual(_sampling_params_from_request(request).max_tokens, 32)

        with self.assertRaises(HTTPException):
            _validate_chat_request(
                ChatCompletionRequest(
                    model="m",
                    messages=[{"role": "user", "content": "p"}],
                    max_tokens=16,
                    max_completion_tokens=32,
                ),
                "m",
            )

    def test_missing_non_bool_engine_arg_fails_fast(self):
        from sparseengine.entrypoints.openai.api_server import _parse_engine_kwargs

        with self.assertRaisesRegex(ValueError, "Missing value"):
            _parse_engine_kwargs(["--max-model-len"])

        with self.assertRaisesRegex(ValueError, "Unknown SparseEngine engine argument"):
            _parse_engine_kwargs(["--observation-layers", "0"])

        with self.assertRaisesRegex(ValueError, "Unknown SparseEngine engine argument"):
            _parse_engine_kwargs(["--obs-layer-ids", "0"])

        self.assertEqual(_parse_engine_kwargs(["--sparse-method", "snapkv"]), {"sparse_method": "snapkv"})

    def test_models_route_advertises_context_window(self):
        from sparseengine.entrypoints.openai.routes.models import models

        class Request:
            app = type(
                "App",
                (),
                {
                    "state": type(
                        "State",
                        (),
                        {
                            "served_model_name": "model",
                            "engine": type(
                                "Engine",
                                (),
                                {"config": type("Config", (), {"max_model_len": 128000})()},
                            )(),
                        },
                    )(),
                },
            )()

        payload = models(Request())

        self.assertEqual(payload["data"][0]["max_model_len"], 128000)
        self.assertNotIn("reasoning", payload["data"][0])

    async def test_cancel_during_admission_aborts_after_seq_id_exists(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.add_started = threading.Event()
                self.release_add = threading.Event()
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                self.add_started.set()
                self.release_add.wait(timeout=5)
                return 123

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def step(self):
                raise AssertionError("cancelled request should not be stepped")

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit("prompt", object(), 0)
            self.assertTrue(await asyncio.to_thread(engine.add_started.wait, 5))
            dispatcher.cancel(handle)
            engine.release_add.set()
            await asyncio.sleep(0.1)
            self.assertEqual(engine.aborted, [123])
        finally:
            dispatcher.close()

    async def test_prefix_cache_inspect_route_uses_dispatcher_control_queue(self):
        from sparseengine.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None

            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return [1, 2]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"sparse_method": ""})()

            def prefix_cache_inspect(self, token_ids, include_subtree=False):
                return {
                    "token_ids": list(token_ids),
                    "include_subtree": bool(include_subtree),
                    "thread": threading.current_thread().name,
                }

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/inspect")
        try:
            response = await endpoint(
                api_server.PrefixCacheInspectRequest(token_ids=[7, 8], include_subtree=True),
                _TestRequest(app),
            )
        finally:
            app.state.dispatcher.close()

        payload = json.loads(response.body)
        self.assertEqual(payload["token_ids"], [7, 8])
        self.assertTrue(payload["include_subtree"])
        self.assertEqual(payload["thread"], "sparseengine-openai-dispatcher")

    async def test_prefix_cache_prune_route_returns_async_job_and_status(self):
        from sparseengine.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None
            chat_template = "template"

            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return [1, 2, 3, 4]

            def apply_chat_template(self, chat, **_kwargs):
                return "|".join(
                    f"{item['role']}:{item['content']}" for item in chat
                )

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"sparse_method": ""})()

            def __init__(self):
                self.status = "queued"

            def prefix_cache_prune_start(self, **kwargs):
                self.kwargs = kwargs
                return {"prune_id": "job-1", "status": "queued"}

            def run_pending_prefix_prune(self):
                self.status = "completed"
                return True

            def prefix_cache_prune_status(self, prune_id):
                return {"prune_id": prune_id, "status": self.status}

            def exit(self):
                pass

        engine = Engine()
        app = api_server.create_app("/tmp/model", served_model_name="model", engine=engine)
        start_endpoint = _route_endpoint(app, "/v1/prefix_cache/prune")
        status_endpoint = _route_endpoint(app, "/v1/prefix_cache/prune/{prune_id}")
        try:
            response = await start_endpoint(
                api_server.PrefixCachePruneRequest(
                    chat={
                        "model": "model",
                        "messages": [{"role": "user", "content": "prune me"}],
                    },
                    range_start=0,
                    range_end=4,
                    keep_tokens=2,
                    policy="kvzip_global",
                ),
                _TestRequest(app),
            )
            status_response = await status_endpoint("job-1", _TestRequest(app))
        finally:
            app.state.dispatcher.close()

        self.assertEqual(response.status_code, 202)
        self.assertEqual(json.loads(response.body)["status"], "queued")
        self.assertEqual(json.loads(status_response.body)["status"], "completed")
        self.assertEqual(engine.kwargs["range_start"], 0)
        self.assertEqual(engine.kwargs["range_end"], 4)
        self.assertEqual(engine.kwargs["policy"], "kvzip_global")

    async def test_prefix_cache_match_accepts_chat_messages(self):
        from sparseengine.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None
            chat_template = "template"

            def apply_chat_template(self, chat, **_kwargs):
                return "|".join(f"{item['role']}:{item['content']}" for item in chat)

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(ch) for ch in text]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"sparse_method": ""})()

            def prefix_cache_match(self, token_ids):
                return {
                    "token_ids": list(token_ids),
                    "thread": threading.current_thread().name,
                    "supported": True,
                    "enabled": True,
                }

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/match")
        try:
            response = await endpoint(
                api_server.PrefixCacheMatchRequest(messages=[{"role": "user", "content": "hello"}]),
                _TestRequest(app),
            )
        finally:
            app.state.dispatcher.close()

        payload = json.loads(response.body)
        self.assertEqual(payload["token_ids"], [ord(ch) for ch in "user:hello"])
        self.assertEqual(payload["thread"], "sparseengine-openai-dispatcher")

    async def test_prefix_cache_match_accepts_full_chat_selector(self):
        from sparseengine.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None
            chat_template = "template"

            def apply_chat_template(
                self,
                chat,
                tools=None,
                enable_thinking=True,
                **_kwargs,
            ):
                tool_name = tools[0]["name"] if tools else "none"
                return (
                    f"thinking={enable_thinking}|tool={tool_name}|"
                    + "|".join(f"{item['role']}:{item['content']}" for item in chat)
                )

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(ch) for ch in text]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"sparse_method": ""})()

            def prefix_cache_match(self, token_ids):
                return {"token_ids": list(token_ids), "supported": True, "enabled": True}

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/match")
        chat = {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "none",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "search", "parameters": {}},
                }
            ],
        }
        try:
            response = await endpoint(
                api_server.PrefixCacheMatchRequest(chat=chat),
                _TestRequest(app),
            )
        finally:
            app.state.dispatcher.close()

        rendered = "thinking=False|tool=search|user:hello"
        self.assertEqual(
            json.loads(response.body)["token_ids"],
            [ord(ch) for ch in rendered],
        )

    async def test_worker_info_and_load_routes(self):
        from sparseengine.entrypoints.openai import api_server

        class Engine:
            tokenizer = object()
            config = type("Config", (), {"sparse_method": ""})()

            def worker_info(self, served_model_name=None, tags=None):
                return {"served_model_name": served_model_name, "tags": list(tags or [])}

            def worker_load(self):
                return {"active_requests": 3, "thread": threading.current_thread().name}

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        info_endpoint = _route_endpoint(app, "/v1/worker/info")
        load_endpoint = _route_endpoint(app, "/v1/worker/load")
        try:
            with patch.dict(os.environ, {"SPARSEENGINE_WORKER_TAGS": "dialog, omnikv"}):
                info_response = info_endpoint(_TestRequest(app))
            load_response = await load_endpoint(_TestRequest(app))
            app.state.dispatcher._failed_message = "OutOfMemoryError: CUDA out of memory"
            unavailable_info_response = info_endpoint(_TestRequest(app))
        finally:
            app.state.dispatcher.close()

        self.assertEqual(json.loads(info_response.body), {"served_model_name": "model", "tags": ["dialog", "omnikv"]})
        load_payload = json.loads(load_response.body)
        self.assertEqual(load_payload["active_requests"], 3)
        self.assertEqual(
            load_payload["thread"],
            "sparseengine-openai-dispatcher",
        )
        self.assertEqual(unavailable_info_response.status_code, 503)
        self.assertEqual(json.loads(unavailable_info_response.body)["reason"], "OutOfMemoryError")

    def test_worker_routing_load_does_not_collect_cache_stats(self):
        from sparseengine.engine.llm_engine import LLMEngine

        class RuntimeState:
            def __init__(self):
                self.calls = 0

            def free_slot_stats(self):
                self.calls += 1
                return {"free_slots": 123}

        runtime_state = RuntimeState()
        scheduler = type(
            "Scheduler",
            (),
            {
                "waiting": [1, 2],
                "decoding": [3],
                "total_preemptions": 4,
                "max_num_seqs_in_batch": 5,
                "max_decoding_seqs": 6,
                "config": type(
                    "Config",
                    (),
                    {"max_num_seqs_in_gpu": 7},
                )(),
            },
        )()
        engine = object.__new__(LLMEngine)
        engine.scheduler = scheduler
        engine.model_runner = type(
            "ModelRunner",
            (),
            {"runtime_state": runtime_state},
        )()

        routing_load = engine.worker_routing_load()

        self.assertEqual(routing_load["active_requests"], 3)
        self.assertNotIn("cache", routing_load)
        self.assertEqual(runtime_state.calls, 0)

        full_load = engine.worker_load()
        self.assertEqual(full_load["cache"]["free_slots"], 123)
        self.assertEqual(runtime_state.calls, 1)

    async def test_routing_probe_snapshots_do_not_wait_for_engine_step(self):
        from sparseengine.engine.prefix_cache import PrefixCacheRoutingSnapshot
        from sparseengine.entrypoints.openai import api_server

        tokenizer = _byte_level_tokenizer()
        step_started = threading.Event()
        release_step = threading.Event()

        class Engine:
            config = type("Config", (), {"sparse_method": ""})()

            def __init__(self):
                self.tokenizer = tokenizer

            def worker_routing_load(self):
                return {"active_requests": 1}

            def prefix_cache_routing_snapshot(self):
                return PrefixCacheRoutingSnapshot(
                    supported=True,
                    enabled=False,
                    method="",
                    reason="prefix cache is not enabled for this runtime.",
                )

            def add_request(self, _prompt, _sampling_params):
                return 17

            def step(self):
                step_started.set()
                release_step.wait(1)
                return [], 0

            def abort_request(self, _seq_id):
                pass

            def exit(self):
                pass

        app = api_server.create_app(
            "/tmp/model",
            served_model_name="model",
            engine=Engine(),
        )
        load_endpoint = _route_endpoint(
            app,
            "/v1/worker/routing_load",
        )
        match_endpoint = _route_endpoint(
            app,
            "/v1/prefix_cache/routing_match",
        )
        try:
            sampling = type("Sampling", (), {"max_tokens": 1})()
            await app.state.dispatcher.submit("prompt", sampling, 0)
            self.assertTrue(await asyncio.to_thread(step_started.wait, 1))

            load_response = await asyncio.wait_for(
                load_endpoint(_TestRequest(app)),
                timeout=0.2,
            )
            match_response = await asyncio.wait_for(
                match_endpoint(
                    api_server.PrefixCacheMatchRequest(token_ids=[1, 2]),
                    _TestRequest(app),
                ),
                timeout=0.2,
            )
        finally:
            release_step.set()
            app.state.dispatcher.close()

        self.assertTrue(json.loads(load_response.body)["snapshot"])
        match_payload = json.loads(match_response.body)
        self.assertTrue(match_payload["snapshot"])
        self.assertFalse(match_payload["enabled"])

    async def test_routing_snapshots_refresh_before_completion_events(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        completion_token_id = tokenizer.encode("a")[0]

        class Snapshot:
            def __init__(self, generation):
                self.generation = generation

            def match(self, _token_ids):
                return {
                    "supported": True,
                    "enabled": True,
                    "matched_tokens": 0,
                    "generation": self.generation,
                    "snapshot": True,
                }

        class Engine:
            last_step_token_outputs = []
            last_step_logprob_outputs = []

            def __init__(self):
                self.tokenizer = tokenizer
                self.active_requests = 0
                self.generation = 0

            def worker_routing_load(self):
                return {"active_requests": self.active_requests}

            def prefix_cache_routing_snapshot(self):
                return Snapshot(self.generation)

            def add_request(self, _prompt, _sampling_params):
                self.active_requests = 1
                return 17

            def step(self):
                self.active_requests = 0
                self.generation = 1
                return [(17, [completion_token_id], [None], [None])], 0

            def abort_request(self, _seq_id):
                pass

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        published_snapshot_states = []
        original_put = dispatcher._put

        def put_after_observing_snapshots(request, item):
            if item["type"] in {"token", "final"}:
                load = dispatcher.worker_routing_load_snapshot()
                match = dispatcher.prefix_cache_routing_match([1, 2])
                published_snapshot_states.append(
                    (
                        item["type"],
                        load["active_requests"],
                        match["generation"],
                    )
                )
            original_put(request, item)

        dispatcher._put = put_after_observing_snapshots
        try:
            handle = await dispatcher.submit(
                "prompt",
                type("Sampling", (), {"max_tokens": 1})(),
                0,
            )
            item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
            if item["type"] == "token":
                item = await asyncio.wait_for(
                    handle.output_queue.get(),
                    timeout=1,
                )
            self.assertEqual(item["type"], "final")
        finally:
            dispatcher.close()

        self.assertGreaterEqual(len(published_snapshot_states), 1)
        for _event_type, active_requests, generation in published_snapshot_states:
            self.assertEqual(active_requests, 0)
            self.assertEqual(generation, 1)

    async def test_prefix_cache_text_selector_tokenizes_server_side(self):
        from sparseengine.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = "<s>"

            def __init__(self):
                self.calls = []

            def encode(self, text, add_special_tokens=False):
                self.calls.append((text, add_special_tokens))
                return [0, 11] if add_special_tokens else [11]

        class Engine:
            config = type("Config", (), {"sparse_method": ""})()

            def __init__(self):
                self.tokenizer = Tokenizer()

            def prefix_cache_inspect(self, token_ids, include_subtree=False):
                del include_subtree
                return {"token_ids": list(token_ids), "calls": list(self.tokenizer.calls)}

            def exit(self):
                pass

        engine = Engine()
        app = api_server.create_app("/tmp/model", served_model_name="model", engine=engine)
        endpoint = _route_endpoint(app, "/v1/prefix_cache/inspect")
        try:
            response = await endpoint(api_server.PrefixCacheInspectRequest(text="hello"), _TestRequest(app))
        finally:
            app.state.dispatcher.close()

        payload = json.loads(response.body)
        self.assertEqual(payload["token_ids"], [0, 11])
        self.assertEqual(payload["calls"], [["hello", True]])

    async def test_prefix_cache_selector_rejects_both_or_neither(self):
        from fastapi import HTTPException
        from sparseengine.entrypoints.openai import api_server

        class Engine:
            tokenizer = object()
            config = type("Config", (), {"sparse_method": ""})()

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/inspect")
        try:
            with self.assertRaises(HTTPException):
                await endpoint(api_server.PrefixCacheInspectRequest(), _TestRequest(app))
            with self.assertRaises(HTTPException):
                await endpoint(api_server.PrefixCacheInspectRequest(token_ids=[1], text="x"), _TestRequest(app))
        finally:
            app.state.dispatcher.close()

    async def test_prefix_cache_disabled_error_is_explicit(self):
        from fastapi import HTTPException
        from sparseengine.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None

            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return [1]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"sparse_method": ""})()

            def prefix_cache_inspect(self, token_ids, include_subtree=False):
                del token_ids, include_subtree
                raise RuntimeError("prefix cache is not enabled or not supported by this cache manager.")

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/inspect")
        try:
            with self.assertRaises(HTTPException) as ctx:
                await endpoint(api_server.PrefixCacheInspectRequest(token_ids=[1]), _TestRequest(app))
        finally:
            app.state.dispatcher.close()

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("prefix cache is not enabled", ctx.exception.detail)

    async def test_prefix_cache_delete_and_priority_routes_are_synchronous_controls(self):
        from sparseengine.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(ch) for ch in text]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"sparse_method": ""})()

            def __init__(self):
                self.calls = []

            def prefix_cache_delete_subtree(self, token_ids):
                self.calls.append(("delete", list(token_ids), threading.current_thread().name))
                return {
                    "deleted_block_ids": ["aa"],
                    "deleted_block_count": 1,
                    "blocked_blocks": [{"block_id": "bb", "reason": "referenced"}],
                }

            def prefix_cache_set_eviction_priority(self, token_ids, priority):
                self.calls.append(("priority", list(token_ids), int(priority), threading.current_thread().name))
                return {
                    "matched": True,
                    "root_block_id": "aa",
                    "updated_block_count": 2,
                    "eviction_priority": int(priority),
                }

            def exit(self):
                pass

        engine = Engine()
        app = api_server.create_app("/tmp/model", served_model_name="model", engine=engine)
        delete_endpoint = _route_endpoint(app, "/v1/prefix_cache/delete_subtree")
        priority_endpoint = _route_endpoint(app, "/v1/prefix_cache/set_eviction_priority")
        try:
            delete_response = await delete_endpoint(
                api_server.PrefixCacheDeleteSubtreeRequest(text="ab"),
                _TestRequest(app),
            )
            priority_response = await priority_endpoint(
                api_server.PrefixCacheSetEvictionPriorityRequest(token_ids=[7, 8], priority=-5),
                _TestRequest(app),
            )
        finally:
            app.state.dispatcher.close()

        self.assertEqual(json.loads(delete_response.body)["blocked_blocks"][0]["reason"], "referenced")
        self.assertEqual(json.loads(priority_response.body)["eviction_priority"], -5)
        self.assertEqual(
            engine.calls,
            [
                ("delete", [97, 98], "sparseengine-openai-dispatcher"),
                ("priority", [7, 8], -5, "sparseengine-openai-dispatcher"),
            ],
        )

    async def test_step_failure_aborts_active_request(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        test_tokenizer = _byte_level_tokenizer()

        class Engine:
            tokenizer = test_tokenizer
            last_step_token_outputs = []
            last_step_logprob_outputs = []

            def __init__(self):
                self.step_started = threading.Event()
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                return 77

            def step(self):
                self.step_started.set()
                raise RuntimeError("boom")

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit("prompt", type("Sampling", (), {"max_tokens": 4})(), 0)
            self.assertTrue(await asyncio.to_thread(engine.step_started.wait, 2))
            item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
            self.assertEqual(item["type"], "error")
            self.assertEqual(engine.aborted, [77])
        finally:
            dispatcher.close()

    async def test_final_detokenizer_error_reaches_client(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        streamed_token_id = tokenizer.encode("a")[0]
        mismatched_final_id = tokenizer.encode("b")[0]

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                return 7

            def step(self):
                self.last_step_token_outputs = [(7, [streamed_token_id])]
                self.last_step_logprob_outputs = [(7, [None], [None])]
                return [(7, [mismatched_final_id], [None], [None])], 0

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit(
                "prompt",
                type("Sampling", (), {"max_tokens": 1})(),
                0,
            )
            token_item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
            error_item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(token_item["type"], "token")
        self.assertEqual(error_item["type"], "error")
        self.assertIn("token history mismatch", error_item["message"])
        self.assertEqual(engine.aborted, [7])

    async def test_stop_detokenizer_error_reaches_client(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        tokenizer = _byte_level_tokenizer()
        token_id = tokenizer.encode("a")[0]
        tokenizer.decode = lambda _ids, skip_special_tokens=True: "different"

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                return 7

            def step(self):
                self.last_step_token_outputs = [(7, [token_id])]
                self.last_step_logprob_outputs = [(7, [None], [None])]
                return [], 0

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            handle = await dispatcher.submit(
                "prompt",
                type("Sampling", (), {"max_tokens": 2})(),
                0,
                ["a"],
            )
            error_item = await asyncio.wait_for(handle.output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(error_item["type"], "error")
        self.assertIn("not a prefix", error_item["message"])
        self.assertEqual(engine.aborted, [7])

    async def test_dispatcher_close_times_out_blocked_step_and_exits_engine(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher

        test_tokenizer = _byte_level_tokenizer()

        class Engine:
            tokenizer = test_tokenizer

            def __init__(self):
                self.step_started = threading.Event()
                self.release_step = threading.Event()
                self.exited = threading.Event()
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []
                self.aborted = []

            def add_request(self, _prompt, _sampling_params):
                return 123

            def step(self):
                self.step_started.set()
                self.release_step.wait()
                return [], 0

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                self.exited.set()

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            await dispatcher.submit("prompt", type("Sampling", (), {"max_tokens": 1000})(), 0)
            self.assertTrue(await asyncio.to_thread(engine.step_started.wait, 2))
            with patch.dict(os.environ, {"SPARSEENGINE_OPENAI_SHUTDOWN_TIMEOUT_S": "0.05"}):
                started = time.perf_counter()
                dispatcher.close()
                elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 1.0)
            self.assertTrue(engine.exited.is_set())
            self.assertTrue(dispatcher._thread.is_alive())
        finally:
            engine.release_step.set()
            dispatcher._thread.join(timeout=1.0)

    async def test_completion_route_error_cancels_sibling_handles(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import RequestHandle, _completion_response

        queue_0 = asyncio.Queue()
        queue_1 = asyncio.Queue()
        handle_0 = RequestHandle(output_queue=queue_0, cancelled=threading.Event())
        handle_1 = RequestHandle(output_queue=queue_1, cancelled=threading.Event())
        await queue_0.put({"type": "error", "message": "failed"})

        with self.assertRaises(HTTPException):
            try:
                await _completion_response("cmpl-test", 123, "model-a", [handle_0, handle_1])
            except Exception:
                for handle in (handle_0, handle_1):
                    handle.cancelled.set()
                raise

        self.assertTrue(handle_0.cancelled.is_set())
        self.assertTrue(handle_1.cancelled.is_set())

    async def test_non_streaming_cancel_logs_request_cancel(self):
        from sparseengine.entrypoints.openai import api_server

        class Tokenizer:
            def encode(self, _prompt):
                return [1]

            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(str(token_id) for token_id in token_ids)

        class Engine:
            config = type("Config", (), {"sparse_method": ""})()

            def __init__(self):
                self.tokenizer = Tokenizer()
                self.last_step_token_outputs = []

            def add_request(self, _prompt, _sampling_params):
                return 1

            def abort_request(self, _seq_id):
                pass

            def step(self):
                return [], 0

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/completions")
        request = api_server.CompletionRequest(model="model", prompt="p")
        try:
            from sparseengine.entrypoints.openai.serving import completion as completion_serving

            with patch.object(
                completion_serving,
                "_completion_response",
                AsyncMock(side_effect=asyncio.CancelledError),
            ), patch.object(completion_serving.logger, "info") as log_info:
                with self.assertRaises(asyncio.CancelledError):
                    await endpoint(request, _TestRequest(app))
        finally:
            app.state.dispatcher.close()

        messages = [call.args[0] for call in log_info.call_args_list]
        self.assertIn("request_cancel id={} model={} stream=false elapsed_s={:.3f}", messages)

    async def test_dispatcher_streaming_delta_uses_cumulative_suffix(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("ab")

        class Engine:
            last_step_token_outputs = [(7, [token_ids[1]])]
            last_step_logprob_outputs = [(7, [None], [None])]

            def __init__(self):
                self.tokenizer = tokenizer

            def exit(self):
                pass

        dispatcher = AsyncEngineDispatcher(Engine())
        output_queue = asyncio.Queue()
        detokenizer = IncrementalDetokenizer(tokenizer)
        detokenizer.push([token_ids[0]])
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=2,
                stop=[],
                completion_token_ids=[token_ids[0]],
                completion_token_logprobs=[None],
                completion_top_logprobs=[None],
                detokenizer=detokenizer,
                emitted_text_len=1,
                emitted_raw_text_len=1,
            )
        }
        try:
            dispatcher._publish_token_deltas(active)
            item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(item["text"], "b")
        self.assertEqual(item["raw_text_delta"], "b")
        self.assertEqual(active[7].completion_token_ids, token_ids)

    async def test_dispatcher_strips_terminal_eos_from_parser_text(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer(special_tokens=["<eos>"])
        content_token_ids = tokenizer.encode("Paris")
        eos_token_id = tokenizer._tokenizer.token_to_id("<eos>")
        completion_token_ids = content_token_ids + [eos_token_id]

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        output_queue = asyncio.Queue()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(completion_token_ids),
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
                eos_token_ids=frozenset({eos_token_id}),
            )
        }
        try:
            engine.last_step_token_outputs = [(7, content_token_ids)]
            engine.last_step_logprob_outputs = [
                (
                    7,
                    [None] * len(content_token_ids),
                    [None] * len(content_token_ids),
                )
            ]
            dispatcher._publish_token_deltas(active)
            token_item = await asyncio.wait_for(output_queue.get(), timeout=1)

            engine.last_step_token_outputs = [(7, [eos_token_id])]
            engine.last_step_logprob_outputs = [(7, [None], [None])]
            dispatcher._publish_token_deltas(active)
            self.assertTrue(output_queue.empty())

            dispatcher._publish_finished(
                active,
                [
                    (
                        7,
                        completion_token_ids,
                        [None] * len(completion_token_ids),
                        [None] * len(completion_token_ids),
                    )
                ],
            )
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(token_item["raw_text_delta"], "Paris")
        self.assertEqual(final_item["raw_text"], "Paris")
        self.assertEqual(final_item["finish_reason"], "stop")
        self.assertEqual(final_item["token_ids"], completion_token_ids)
        self.assertEqual(
            final_item["completion_tokens"],
            len(completion_token_ids),
        )

    async def test_dispatcher_preserves_eos_when_ignored(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer(special_tokens=["<eos>"])
        content_token_ids = tokenizer.encode("Paris")
        eos_token_id = tokenizer._tokenizer.token_to_id("<eos>")
        completion_token_ids = content_token_ids + [eos_token_id]

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = [
                    (7, completion_token_ids)
                ]
                self.last_step_logprob_outputs = [
                    (
                        7,
                        [None] * len(completion_token_ids),
                        [None] * len(completion_token_ids),
                    )
                ]

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        output_queue = asyncio.Queue()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(completion_token_ids),
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
                eos_token_ids=frozenset({eos_token_id}),
                ignore_eos=True,
            )
        }
        try:
            dispatcher._publish_token_deltas(active)
            token_item = await asyncio.wait_for(output_queue.get(), timeout=1)
            dispatcher._publish_finished(
                active,
                [
                    (
                        7,
                        completion_token_ids,
                        [None] * len(completion_token_ids),
                        [None] * len(completion_token_ids),
                    )
                ],
            )
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(token_item["raw_text_delta"], "Paris<eos>")
        self.assertEqual(final_item["raw_text"], "Paris<eos>")
        self.assertEqual(final_item["finish_reason"], "length")

    async def test_dispatcher_streams_complete_unicode_with_pending_logprobs(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训练")

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        output_queue = asyncio.Queue()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(token_ids),
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
            )
        }
        token_logprobs = [-float(index + 1) for index in range(len(token_ids))]
        top_logprobs = [{token_id: value} for token_id, value in zip(token_ids, token_logprobs)]
        token_items = []
        try:
            for token_id, token_logprob, top_logprob in zip(
                token_ids,
                token_logprobs,
                top_logprobs,
            ):
                engine.last_step_token_outputs = [(7, [token_id])]
                engine.last_step_logprob_outputs = [(7, [token_logprob], [top_logprob])]
                dispatcher._publish_token_deltas(active)
                await asyncio.sleep(0)
                while not output_queue.empty():
                    token_items.append(output_queue.get_nowait())

            dispatcher._publish_finished(
                active,
                [(7, token_ids, token_logprobs, top_logprobs)],
            )
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual([item["text"] for item in token_items], ["训", "练"])
        self.assertEqual("".join(item["text"] for item in token_items), "训练")
        self.assertNotIn("�", "".join(item["text"] for item in token_items))
        self.assertEqual([len(item["token_ids"]) for item in token_items], [3, 3])
        self.assertEqual(
            [value for item in token_items for value in item["token_logprobs"]],
            token_logprobs,
        )
        self.assertEqual(final_item["text"], "训练")
        self.assertEqual(final_item["raw_text"], "训练")
        self.assertEqual(final_item["text_delta"], "")

    async def test_final_reconciliation_publishes_pending_logprobs(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训")[:2]
        token_logprobs = [-1.0, -2.0]

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        output_queue = asyncio.Queue()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(token_ids),
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
            )
        }
        try:
            for token_id, logprob in zip(token_ids, token_logprobs):
                engine.last_step_token_outputs = [(7, [token_id])]
                engine.last_step_logprob_outputs = [(7, [logprob], [None])]
                dispatcher._publish_token_deltas(active)
            self.assertTrue(output_queue.empty())

            dispatcher._publish_finished(
                active,
                [(7, token_ids, token_logprobs, [None, None])],
            )
            token_item = await asyncio.wait_for(output_queue.get(), timeout=1)
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(token_item["type"], "token")
        self.assertEqual(token_item["text"], "�")
        self.assertEqual(token_item["token_ids"], token_ids)
        self.assertEqual(token_item["token_logprobs"], token_logprobs)
        self.assertEqual(final_item["type"], "final")
        self.assertEqual(final_item["text"], "�")
        self.assertEqual(final_item["text_delta"], "")

    async def test_final_suffix_publishes_token_metadata(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("训练")
        token_logprobs = [-float(index + 1) for index in range(len(token_ids))]

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer

            def exit(self):
                pass

        dispatcher = AsyncEngineDispatcher(Engine())
        output_queue = asyncio.Queue()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(token_ids),
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
            )
        }
        try:
            dispatcher._publish_finished(
                active,
                [(7, token_ids, token_logprobs, [None] * len(token_ids))],
            )
            token_item = await asyncio.wait_for(output_queue.get(), timeout=1)
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(token_item["type"], "token")
        self.assertEqual(token_item["text"], "训练")
        self.assertEqual(token_item["token_ids"], token_ids)
        self.assertEqual(token_item["token_logprobs"], token_logprobs)
        self.assertEqual(final_item["type"], "final")
        self.assertEqual(final_item["text_delta"], "")

    async def test_stream_logprobs_include_raw_only_special_token(self):
        from sparseengine.entrypoints.openai.api_server import (
            RequestHandle,
            _chat_completion_stream,
            _completion_stream,
        )

        tokenizer = _byte_level_tokenizer(special_tokens=["<special>"])
        token_id = tokenizer._tokenizer.token_to_id("<special>")
        items = [
            {
                "type": "token",
                "index": 0,
                "text": "",
                "raw_text_delta": "<special>",
                "token_ids": [token_id],
                "token_logprobs": [-0.5],
                "top_logprobs": [None],
            },
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": "<special>",
                "text_delta": "",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [token_id],
                "token_logprobs": [-0.5],
                "top_logprobs": [None],
            },
        ]

        def handle():
            queue = asyncio.Queue()
            for item in copy.deepcopy(items):
                queue.put_nowait(item)
            return RequestHandle(output_queue=queue, cancelled=threading.Event())

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        completion_chunks = [
            chunk
            async for chunk in _completion_stream(
                Dispatcher(),
                "cmpl-test",
                123,
                "model",
                [handle()],
                tokenizer=tokenizer,
            )
        ]
        chat_chunks = [
            chunk
            async for chunk in _chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [handle()],
                tokenizer=tokenizer,
            )
        ]
        completion_payloads = [
            payload
            for event, payload in _response_sse_events(completion_chunks)
            if event != "[DONE]"
        ]
        chat_payloads = [
            payload
            for event, payload in _response_sse_events(chat_chunks)
            if event != "[DONE]"
        ]

        self.assertIsNotNone(completion_payloads[0]["choices"][0]["logprobs"])
        self.assertEqual(completion_payloads[0]["choices"][0]["text"], "")
        self.assertIsNotNone(chat_payloads[1]["choices"][0]["logprobs"])
        self.assertEqual(chat_payloads[1]["choices"][0]["delta"]["content"], "")

    async def test_unicode_streams_match_non_streaming_endpoints(self):
        from sparseengine.entrypoints.openai.api_server import (
            RequestHandle,
            ResponseRequest,
            _chat_completion_response,
            _chat_completion_stream,
            _completion_response,
            _completion_stream,
            _response_response,
            _response_stream,
        )

        text = "中文，日本語，한국어，café，🙂。"
        items = await _dispatcher_items_for_text(text)

        def handle():
            queue = asyncio.Queue()
            for item in copy.deepcopy(items):
                queue.put_nowait(item)
            return RequestHandle(output_queue=queue, cancelled=threading.Event())

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        completion_chunks = [
            chunk
            async for chunk in _completion_stream(
                Dispatcher(),
                "cmpl-test",
                123,
                "model",
                [handle()],
            )
        ]
        completion_text = "".join(
            payload["choices"][0]["text"]
            for event, payload in _response_sse_events(completion_chunks)
            if event != "[DONE]"
        )
        completion_final = await _completion_response("cmpl-test", 123, "model", [handle()])

        chat_chunks = [
            chunk
            async for chunk in _chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [handle()],
            )
        ]
        chat_text = "".join(
            (payload["choices"][0].get("delta") or {}).get("content", "")
            for event, payload in _response_sse_events(chat_chunks)
            if event != "[DONE]" and payload.get("choices")
        )
        chat_final = await _chat_completion_response("chatcmpl-test", 123, "model", [handle()])

        request = ResponseRequest(model="model", input="hello", stream=True)
        response_chunks = [
            chunk
            async for chunk in _response_stream(
                Dispatcher(),
                "resp_test",
                123,
                "model",
                handle(),
                time.perf_counter(),
                None,
                request,
                reasoning_parser_name=None,
            )
        ]
        response_events = _response_sse_events(response_chunks)
        response_text = "".join(
            payload["delta"]
            for event, payload in response_events
            if event == "response.output_text.delta"
        )
        response_final = await _response_response(
            "resp_test",
            123,
            "model",
            handle(),
            reasoning_parser_name=None,
        )

        self.assertEqual(completion_text, text)
        self.assertEqual(completion_text, completion_final["choices"][0]["text"])
        self.assertEqual(chat_text, text)
        self.assertEqual(chat_text, chat_final["choices"][0]["message"]["content"])
        self.assertEqual(response_text, text)
        self.assertEqual(response_text, response_final["output"][0]["content"][0]["text"])
        self.assertNotIn(b"\xef\xbf\xbd", "".join(response_chunks).encode("utf-8"))

    async def test_unicode_response_reasoning_and_tool_call_stream(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        raw_text = (
            '<think>中文推理</think><tool_call>{"name":"查询",'
            '"arguments":{"城市":"北京"}}</tool_call>'
        )
        items = await _dispatcher_items_for_text(raw_text)
        queue = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in _response_stream(
                Dispatcher(),
                "resp_test",
                123,
                "model",
                RequestHandle(output_queue=queue, cancelled=threading.Event()),
                time.perf_counter(),
                None,
                ResponseRequest(
                    model="model",
                    input="hello",
                    stream=True,
                    tools=[{"type": "function", "name": "查询", "parameters": {}}],
                ),
                reasoning_parser_name="qwen3",
                response_parser=_transformers_response_parser(),
            )
        ]
        events = _response_sse_events(chunks)
        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]

        self.assertEqual(completed["output"][0]["type"], "reasoning")
        self.assertEqual(completed["output"][0]["text"], "中文推理")
        self.assertEqual(completed["output"][1]["type"], "function_call")
        self.assertEqual(completed["output"][1]["name"], "查询")
        self.assertEqual(completed["output"][1]["arguments"], '{"城市":"北京"}')
        self.assertNotIn(b"\xef\xbf\xbd", "".join(chunks).encode("utf-8"))

    async def test_unicode_stop_is_not_exposed(self):
        items = await _dispatcher_items_for_text("回答训练结束忽略", stop=["结束"])
        streamed_text = "".join(item.get("text", "") for item in items if item["type"] == "token")
        final = [item for item in items if item["type"] == "final"][0]

        self.assertEqual(streamed_text, "回答训练")
        self.assertEqual(final["text"], "回答训练")
        self.assertNotIn("结束", streamed_text)

    async def test_dispatcher_stop_buffers_partial_stop_prefix(self):
        from sparseengine.entrypoints.openai.api_server import AsyncEngineDispatcher, _ActiveRequest
        from sparseengine.entrypoints.openai.detokenizer import IncrementalDetokenizer

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("abSTOP")

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = [(7, [token_ids[1]])]
                self.last_step_logprob_outputs = [(7, [None], [None])]
                self.aborted = []

            def abort_request(self, seq_id):
                self.aborted.append(seq_id)

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        output_queue = asyncio.Queue()
        detokenizer = IncrementalDetokenizer(tokenizer)
        detokenizer.push([token_ids[0]])
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[10],
                max_tokens=len(token_ids),
                stop=["bSTOP"],
                completion_token_ids=[token_ids[0]],
                completion_token_logprobs=[None],
                completion_top_logprobs=[None],
                detokenizer=detokenizer,
                emitted_text_len=0,
            )
        }
        try:
            dispatcher._publish_token_deltas(active)
            token_item = await asyncio.wait_for(output_queue.get(), timeout=1)
            self.assertEqual(token_item["text"], "a")
            self.assertEqual(token_item["raw_text_delta"], "a")
            engine.last_step_token_outputs = [(7, token_ids[2:])]
            engine.last_step_logprob_outputs = [
                (7, [None] * len(token_ids[2:]), [None] * len(token_ids[2:]))
            ]
            dispatcher._publish_token_deltas(active)
            final_item = await asyncio.wait_for(output_queue.get(), timeout=1)
        finally:
            dispatcher.close()

        self.assertEqual(final_item["type"], "final")
        self.assertEqual(final_item["text"], "a")
        self.assertEqual(final_item["raw_text"], "a")
        self.assertEqual(final_item["text_delta"], "")
        self.assertEqual(engine.aborted, [7])

    async def test_dispatcher_maps_visible_stop_across_special_tokens(self):
        tokenizer = _byte_level_tokenizer(
            special_tokens=["<special>"],
        )
        special_token_id = tokenizer._tokenizer.token_to_id("<special>")
        cases = [
            (
                [special_token_id] + tokenizer.encode("special"),
                "special",
                "",
                "<special>",
            ),
            (
                tokenizer.encode("okST")
                + [special_token_id]
                + tokenizer.encode("OP"),
                "STOP",
                "ok",
                "ok",
            ),
        ]

        for incremental in (True, False):
            for token_ids, stop, expected_text, expected_raw in cases:
                with self.subTest(
                    incremental=incremental,
                    stop=stop,
                ):
                    items = await _dispatcher_items_for_token_ids(
                        tokenizer,
                        token_ids,
                        stop=[stop],
                        incremental=incremental,
                    )
                    streamed_raw = "".join(
                        item["raw_text_delta"]
                        for item in items
                        if item["type"] == "token"
                    )
                    final = next(
                        item for item in items if item["type"] == "final"
                    )

                    self.assertEqual(streamed_raw, expected_raw)
                    self.assertEqual(final["text"], expected_text)
                    self.assertEqual(final["raw_text"], expected_raw)

    async def test_chat_completion_response_shape(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "hello",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
        )

        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(response["usage"], {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5})

    async def test_chain_metadata_is_exposed_by_all_response_shapes(self):
        from sparseengine.entrypoints.openai.api_server import (
            RequestHandle,
            _chat_completion_response,
            _completion_response,
        )
        from sparseengine.entrypoints.openai.serving.responses import _usage_from_final

        final = {
            "type": "final",
            "index": 0,
            "text": "hello",
            "finish_reason": "stop",
            "prompt_tokens": 6,
            "completion_tokens": 3,
            "token_ids": [1, 2, 3],
            "token_logprobs": [None, None, None],
            "top_logprobs": [None, None, None],
            "chain_id": "chain-1",
            "chain_status": "resumed",
            "reused_tokens": 4,
            "prefilled_tokens": 2,
        }

        def handle():
            queue = asyncio.Queue()
            queue.put_nowait(dict(final))
            return RequestHandle(
                output_queue=queue,
                cancelled=threading.Event(),
                chain_id="chain-1",
                chain_status="resumed",
                reused_tokens=4,
                prefilled_tokens=2,
            )

        chat = await _chat_completion_response(
            "chatcmpl-test", 123, "model", [handle()]
        )
        completion = await _completion_response(
            "cmpl-test", 123, "model", [handle()]
        )
        responses_usage = _usage_from_final(final)

        self.assertEqual(chat["chain_id"], "chain-1")
        self.assertEqual(chat["chain_status"], "resumed")
        self.assertEqual(chat["usage"]["reused_tokens"], 4)
        self.assertEqual(
            chat["usage"]["prompt_tokens_details"]["cached_tokens"], 4
        )
        self.assertEqual(completion["chain_id"], "chain-1")
        self.assertEqual(completion["chain_status"], "resumed")
        self.assertEqual(completion["usage"]["reused_tokens"], 4)
        self.assertEqual(
            completion["usage"]["prompt_tokens_details"]["cached_tokens"], 4
        )
        self.assertEqual(responses_usage["reused_tokens"], 4)
        self.assertEqual(
            responses_usage["input_tokens_details"]["cached_tokens"], 4
        )

    async def test_chain_worker_keeps_implicit_completion_batch_compatible(self):
        from sparseengine.entrypoints.openai.dispatcher import RequestHandle
        from sparseengine.entrypoints.openai.protocol.completion import (
            CompletionRequest,
        )
        from sparseengine.entrypoints.openai.serving.completion import (
            serve_completion,
        )

        class Dispatcher:
            admission_ack_enabled = True
            chain_cache_enabled = True

            def __init__(self):
                self.submitted = []

            async def submit(self, *_args, **_kwargs):
                raise AssertionError("submit_admitted must be used")

            async def submit_admitted(
                self,
                prompt,
                _sampling_params,
                index,
                _stop,
                *,
                stream=True,
            ):
                self.submitted.append((prompt, index))
                output_queue = asyncio.Queue()
                chain_id = f"chain-{index}"
                output_queue.put_nowait(
                    {
                        "type": "final",
                        "index": index,
                        "text": f"answer-{index}",
                        "finish_reason": "length",
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "token_ids": [index],
                        "token_logprobs": [None],
                        "top_logprobs": [None],
                        "chain_id": chain_id,
                        "chain_status": "created",
                        "reused_tokens": 0,
                        "prefilled_tokens": 1,
                    }
                )
                return RequestHandle(
                    output_queue=output_queue,
                    cancelled=threading.Event(),
                    seq_id=index,
                    chain_id=chain_id,
                    chain_status="created",
                    prefilled_tokens=1,
                )

        dispatcher = Dispatcher()
        response = await serve_completion(
            CompletionRequest(
                model="model",
                prompt=["first", "second"],
                max_tokens=1,
                temperature=0,
            ),
            dispatcher,
            tokenizer=None,
            served_model_name="model",
            request_log_path=None,
        )
        payload = json.loads(response.body)

        self.assertEqual(
            dispatcher.submitted,
            [("first", 0), ("second", 1)],
        )
        self.assertNotIn("chain_id", payload)
        self.assertEqual(
            [choice["chain_id"] for choice in payload["choices"]],
            ["chain-0", "chain-1"],
        )

    async def test_implicit_chain_batch_cancels_partial_admissions(self):
        from fastapi import HTTPException

        from sparseengine.engine.chain_cache import ChainCapacityError
        from sparseengine.entrypoints.openai.dispatcher import RequestHandle
        from sparseengine.entrypoints.openai.protocol.completion import (
            CompletionRequest,
        )
        from sparseengine.entrypoints.openai.serving.completion import (
            serve_completion,
        )

        class Dispatcher:
            admission_ack_enabled = True
            chain_cache_enabled = True

            def __init__(self):
                self.discarded = []

            async def submit(self, *_args, **_kwargs):
                raise AssertionError("submit_admitted must be used")

            async def submit_admitted(
                self,
                _prompt,
                _sampling_params,
                index,
                _stop,
                *,
                stream=True,
            ):
                if index == 1:
                    raise ChainCapacityError(
                        "no chain capacity",
                        chain_id="chain-1",
                    )
                return RequestHandle(
                    output_queue=asyncio.Queue(),
                    cancelled=threading.Event(),
                    seq_id=7,
                    chain_id="chain-0",
                    chain_status="created",
                )

            async def discard(self, handle):
                self.discarded.append(handle.chain_id)

        dispatcher = Dispatcher()
        with self.assertRaises(HTTPException) as ctx:
            await serve_completion(
                CompletionRequest(
                    model="model",
                    prompt=["first", "second"],
                    max_tokens=1,
                    temperature=0,
                ),
                dispatcher,
                tokenizer=None,
                served_model_name="model",
                request_log_path=None,
            )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(dispatcher.discarded, ["chain-0"])

    async def test_partial_admission_cleanup_attempts_every_handle(self):
        from sparseengine.entrypoints.openai.dispatcher import RequestHandle
        from sparseengine.entrypoints.openai.serving.completion import (
            _discard_partial_admissions,
        )

        class Dispatcher:
            def __init__(self):
                self.discarded = []

            async def discard(self, handle):
                self.discarded.append(handle.chain_id)
                if handle.chain_id == "chain-0":
                    raise RuntimeError("first cleanup failed")

        handles = [
            RequestHandle(
                output_queue=asyncio.Queue(),
                cancelled=threading.Event(),
                seq_id=index,
                chain_id=f"chain-{index}",
            )
            for index in range(2)
        ]
        dispatcher = Dispatcher()

        with self.assertRaisesRegex(
            RuntimeError,
            "2 partially admitted|1 cleanup error",
        ):
            await _discard_partial_admissions(dispatcher, handles)

        self.assertEqual(
            dispatcher.discarded,
            ["chain-0", "chain-1"],
        )

    async def test_chain_admission_error_precedes_streaming_http_200(self):
        from fastapi import HTTPException

        from sparseengine.engine.chain_cache import ChainBusyError
        from sparseengine.entrypoints.openai.api_server import ChatCompletionRequest
        from sparseengine.entrypoints.openai.serving.chat import (
            serve_chat_completion,
        )

        class Dispatcher:
            admission_ack_enabled = True

            async def submit(self, *_args, **_kwargs):
                raise AssertionError("submit_admitted must be used")

            async def submit_admitted(self, *_args, **_kwargs):
                raise ChainBusyError(
                    "already active",
                    chain_id="chain-1",
                )

        class Tokenizer:
            chat_template = None

        request = ChatCompletionRequest(
            model="model",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            chain_id="chain-1",
        )
        with self.assertRaises(HTTPException) as ctx:
            await serve_chat_completion(
                request,
                Dispatcher(),
                Tokenizer(),
                "model",
                None,
            )
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_dispatcher_shutdown_resolves_pending_admission(self):
        from sparseengine.entrypoints.openai.dispatcher import (
            AsyncEngineDispatcher,
            RequestHandle,
            _QueuedRequest,
        )

        loop = asyncio.get_running_loop()
        output_queue = asyncio.Queue()
        cancelled = threading.Event()
        future = loop.create_future()
        handle = RequestHandle(
            output_queue=output_queue,
            cancelled=cancelled,
            admission_future=future,
        )
        item = _QueuedRequest(
            prompt=[1],
            sampling_params=object(),
            index=0,
            stop=[],
            loop=loop,
            output_queue=output_queue,
            cancelled=cancelled,
            handle=handle,
            admission_future=future,
        )
        dispatcher = object.__new__(AsyncEngineDispatcher)
        dispatcher._closing = threading.Event()
        dispatcher._closing.set()
        dispatcher._failed_message = None

        dispatcher._admit(item, {})
        admitted = await asyncio.wait_for(future, timeout=1)

        self.assertIs(admitted, handle)
        self.assertIsInstance(handle.admission_error, RuntimeError)
        self.assertIn("shutting down", str(handle.admission_error))

    async def test_dispatcher_invalidates_chain_when_stop_is_found_at_finish(self):
        from sparseengine.entrypoints.openai.api_server import (
            AsyncEngineDispatcher,
            _ActiveRequest,
        )
        from sparseengine.entrypoints.openai.detokenizer import (
            IncrementalDetokenizer,
        )

        tokenizer = _byte_level_tokenizer()
        token_ids = tokenizer.encode("answerSTOP")

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []
                self.aborted = []

            def abort_request(self, seq_id, disposition="invalidate"):
                self.aborted.append((int(seq_id), str(disposition)))

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        output_queue = asyncio.Queue()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=output_queue,
                prompt_token_ids=[1],
                max_tokens=len(token_ids) + 1,
                stop=["STOP"],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
                chain_id="chain-1",
                chain_status="resumed",
            )
        }
        try:
            dispatcher._publish_finished(
                active,
                [
                    (
                        7,
                        token_ids,
                        [None] * len(token_ids),
                        [None] * len(token_ids),
                    )
                ],
            )
            items = []
            await asyncio.sleep(0)
            while not output_queue.empty():
                items.append(output_queue.get_nowait())
        finally:
            dispatcher.close()

        final = next(item for item in items if item["type"] == "final")
        self.assertEqual(final["text"], "answer")
        self.assertEqual(final["raw_text"], "answer")
        self.assertEqual(final["chain_status"], "invalidated")
        self.assertEqual(engine.aborted, [(7, "invalidate")])

    async def test_dispatcher_cleans_up_chain_when_post_admission_setup_fails(self):
        from sparseengine.entrypoints.openai.dispatcher import (
            AsyncEngineDispatcher,
            RequestHandle,
            _QueuedRequest,
        )

        base_tokenizer = _byte_level_tokenizer()

        class FailingTokenizer:
            is_fast = True
            backend_tokenizer = base_tokenizer.backend_tokenizer

            def encode(self, _text):
                raise RuntimeError("token accounting failed")

            def decode(self, token_ids, skip_special_tokens=True):
                return base_tokenizer.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                )

        class Admission:
            seq_id = 7
            chain_id = "chain-1"
            chain_status = "created"
            reused_tokens = 0
            prefilled_tokens = 1

        class Engine:
            tokenizer = FailingTokenizer()

            def __init__(self):
                self.aborted = []

            def admit_request(self, *_args, **_kwargs):
                return Admission()

            def abort_request(self, seq_id, disposition="invalidate"):
                self.aborted.append((int(seq_id), str(disposition)))

        loop = asyncio.get_running_loop()
        output_queue = asyncio.Queue()
        cancelled = threading.Event()
        future = loop.create_future()
        handle = RequestHandle(
            output_queue=output_queue,
            cancelled=cancelled,
            admission_future=future,
        )
        item = _QueuedRequest(
            prompt="hello",
            sampling_params=type("Params", (), {"max_tokens": 1})(),
            index=0,
            stop=[],
            loop=loop,
            output_queue=output_queue,
            cancelled=cancelled,
            handle=handle,
            admission_future=future,
        )
        dispatcher = object.__new__(AsyncEngineDispatcher)
        dispatcher.engine = Engine()
        dispatcher._closing = threading.Event()
        dispatcher._failed_message = None

        dispatcher._admit(item, {})
        await asyncio.wait_for(future, timeout=1)

        self.assertIsInstance(handle.admission_error, RuntimeError)
        self.assertEqual(
            dispatcher.engine.aborted,
            [(7, "invalidate")],
        )

    async def test_dispatcher_uses_admitted_prompt_count_for_usage(self):
        from sparseengine.entrypoints.openai.dispatcher import (
            AsyncEngineDispatcher,
            RequestHandle,
            _QueuedRequest,
        )

        tokenizer = _byte_level_tokenizer()

        class Admission:
            seq_id = 7
            chain_id = "chain-1"
            chain_status = "resumed"
            reused_tokens = 2
            prefilled_tokens = 1
            prompt_token_ids = [11, 12, 13]

        class Engine:
            config = type(
                "Config",
                (),
                {"eos_token_ids": (41, 42), "eos": -1},
            )()

            def __init__(self):
                self.tokenizer = tokenizer

            def admit_request(self, *_args, **_kwargs):
                return Admission()

        loop = asyncio.get_running_loop()
        output_queue = asyncio.Queue()
        future = loop.create_future()
        handle = RequestHandle(
            output_queue=output_queue,
            cancelled=threading.Event(),
            admission_future=future,
        )
        item = _QueuedRequest(
            prompt="hello",
            sampling_params=type("Params", (), {"max_tokens": 1})(),
            index=0,
            stop=[],
            loop=loop,
            output_queue=output_queue,
            cancelled=handle.cancelled,
            handle=handle,
            admission_future=future,
            chain_id="chain-1",
        )
        dispatcher = object.__new__(AsyncEngineDispatcher)
        dispatcher.engine = Engine()
        dispatcher._closing = threading.Event()
        dispatcher._failed_message = None
        active = {}

        dispatcher._admit(item, active)
        await asyncio.wait_for(future, timeout=1)
        self.assertEqual(active[7].eos_token_ids, frozenset({41, 42}))
        dispatcher._publish_finished(active, [(7, [], [], [])])
        final = await asyncio.wait_for(output_queue.get(), timeout=1)

        self.assertIsNone(handle.admission_error)
        self.assertEqual(handle.prompt_token_ids, [11, 12, 13])
        self.assertEqual(final["type"], "final")
        self.assertEqual(final["prompt_tokens"], 3)
        self.assertTrue(handle.terminal.is_set())
        self.assertEqual(dispatcher._latest_request_tokens, {})
        dispatcher.cancel(handle)

    async def test_dispatcher_wakes_to_invalidate_finished_chain_on_cancel(self):
        from sparseengine.entrypoints.openai.dispatcher import (
            AsyncEngineDispatcher,
            RequestHandle,
        )

        invalidated = threading.Event()

        class Engine:
            tokenizer = _byte_level_tokenizer()
            last_step_token_outputs = []
            last_step_logprob_outputs = []

            def invalidate_chain(self, chain_id):
                if chain_id != "chain-1":
                    raise AssertionError(chain_id)
                invalidated.set()

            def exit(self):
                pass

        dispatcher = AsyncEngineDispatcher(Engine())
        handle = RequestHandle(
            output_queue=asyncio.Queue(),
            cancelled=threading.Event(),
            seq_id=7,
            chain_id="chain-1",
        )
        try:
            dispatcher.cancel(handle)
            completed = await asyncio.to_thread(invalidated.wait, 1.0)
        finally:
            dispatcher.close()

        self.assertTrue(completed)

    async def test_dispatcher_ignores_stale_cancel_after_chain_seq_id_reuse(self):
        import queue

        from sparseengine.entrypoints.openai.dispatcher import (
            AsyncEngineDispatcher,
            RequestHandle,
            _ActiveRequest,
        )
        from sparseengine.entrypoints.openai.detokenizer import (
            IncrementalDetokenizer,
        )

        tokenizer = _byte_level_tokenizer()

        class Engine:
            def __init__(self):
                self.aborted = []

            def abort_request(self, seq_id, disposition="invalidate"):
                self.aborted.append((int(seq_id), str(disposition)))

        dispatcher = object.__new__(AsyncEngineDispatcher)
        dispatcher.engine = Engine()
        dispatcher._aborts = queue.Queue()
        dispatcher._pending = queue.Queue()
        dispatcher._latest_request_tokens = {7: "new-turn"}
        old_handle = RequestHandle(
            output_queue=asyncio.Queue(),
            cancelled=threading.Event(),
            seq_id=7,
            chain_id="chain-1",
            request_token="old-turn",
        )
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=asyncio.Queue(),
                prompt_token_ids=[1, 2],
                max_tokens=1,
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
                chain_id="chain-1",
                chain_status="resumed",
                request_token="new-turn",
            )
        }

        dispatcher.cancel(old_handle)
        changed = dispatcher._drain_aborts(active)

        self.assertFalse(changed)
        self.assertEqual(dispatcher.engine.aborted, [])
        self.assertIn(7, active)

        active.clear()
        dispatcher.cancel(old_handle)
        changed = dispatcher._drain_aborts(active)
        self.assertFalse(changed)
        self.assertEqual(dispatcher.engine.aborted, [])

        current_handle = RequestHandle(
            output_queue=asyncio.Queue(),
            cancelled=threading.Event(),
            seq_id=7,
            chain_id="chain-1",
            request_token="new-turn",
        )
        dispatcher.cancel(current_handle)
        changed = dispatcher._drain_aborts(active)
        self.assertTrue(changed)
        self.assertEqual(dispatcher.engine.aborted, [(7, "invalidate")])

    async def test_dispatcher_discard_invalidates_after_terminal_race(self):
        import queue
        from types import SimpleNamespace

        from sparseengine.entrypoints.openai.dispatcher import (
            AsyncEngineDispatcher,
            RequestHandle,
        )

        class Engine:
            def __init__(self):
                self.discarded = []

            def discard_chain(self, chain_id, *, expected_seq_id):
                self.discarded.append(
                    (str(chain_id), int(expected_seq_id))
                )
                return True

        dispatcher = object.__new__(AsyncEngineDispatcher)
        dispatcher.engine = Engine()
        dispatcher._aborts = queue.Queue()
        dispatcher._pending = queue.Queue()
        dispatcher._latest_request_tokens = {}
        dispatcher._state_lock = threading.Lock()
        dispatcher._closing = threading.Event()
        dispatcher._failed_message = None
        dispatcher._thread = SimpleNamespace(is_alive=lambda: True)
        handle = RequestHandle(
            output_queue=asyncio.Queue(),
            cancelled=threading.Event(),
            seq_id=7,
            chain_id="chain-1",
            request_token="finished-turn",
        )

        discard_task = asyncio.create_task(dispatcher.discard(handle))
        await asyncio.sleep(0)
        handle.terminal.set()
        changed = dispatcher._drain_aborts({})
        await asyncio.wait_for(discard_task, timeout=1)

        self.assertTrue(changed)
        self.assertEqual(
            dispatcher.engine.discarded,
            [("chain-1", 7)],
        )

    async def test_dispatcher_rejects_discard_after_shutdown(self):
        import queue
        from types import SimpleNamespace

        from sparseengine.entrypoints.openai.dispatcher import (
            AsyncEngineDispatcher,
            RequestHandle,
        )

        dispatcher = object.__new__(AsyncEngineDispatcher)
        dispatcher._aborts = queue.Queue()
        dispatcher._pending = queue.Queue()
        dispatcher._state_lock = threading.Lock()
        dispatcher._closing = threading.Event()
        dispatcher._closing.set()
        dispatcher._failed_message = None
        dispatcher._thread = SimpleNamespace(is_alive=lambda: False)
        handle = RequestHandle(
            output_queue=asyncio.Queue(),
            cancelled=threading.Event(),
            seq_id=7,
            chain_id="chain-1",
        )

        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            await asyncio.wait_for(dispatcher.discard(handle), timeout=1)
        self.assertTrue(dispatcher._aborts.empty())

    async def test_dispatcher_shutdown_resolves_queued_discard(self):
        import queue

        from sparseengine.entrypoints.openai.dispatcher import (
            AsyncEngineDispatcher,
            _AbortRequest,
        )

        output_queue = asyncio.Queue()
        abort = _AbortRequest(
            seq_id=7,
            disposition="invalidate",
            chain_id="chain-1",
            request_token="turn",
            force=True,
            loop=asyncio.get_running_loop(),
            output_queue=output_queue,
        )
        dispatcher = object.__new__(AsyncEngineDispatcher)
        dispatcher._aborts = queue.Queue()
        dispatcher._aborts.put(abort)

        dispatcher._fail_pending_aborts("dispatcher failed")
        result = await asyncio.wait_for(output_queue.get(), timeout=1)

        self.assertEqual(result["type"], "error")
        self.assertIn("dispatcher failed", result["message"])
        self.assertTrue(dispatcher._aborts.empty())

    async def test_dispatcher_abort_failure_resolves_discard_waiter(self):
        import queue

        from sparseengine.entrypoints.openai.detokenizer import (
            IncrementalDetokenizer,
        )
        from sparseengine.entrypoints.openai.dispatcher import (
            AsyncEngineDispatcher,
            _AbortRequest,
            _ActiveRequest,
        )

        class Engine:
            def abort_request(self, _seq_id, disposition="invalidate"):
                del disposition
                raise RuntimeError("abort failed")

        output_queue = asyncio.Queue()
        abort = _AbortRequest(
            seq_id=7,
            disposition="invalidate",
            chain_id="chain-1",
            request_token="turn",
            force=True,
            loop=asyncio.get_running_loop(),
            output_queue=output_queue,
        )
        dispatcher = object.__new__(AsyncEngineDispatcher)
        dispatcher.engine = Engine()
        dispatcher._aborts = queue.Queue()
        dispatcher._aborts.put(abort)
        dispatcher._latest_request_tokens = {7: "turn"}
        tokenizer = _byte_level_tokenizer()
        active = {
            7: _ActiveRequest(
                index=0,
                loop=asyncio.get_running_loop(),
                output_queue=asyncio.Queue(),
                prompt_token_ids=[1],
                max_tokens=1,
                stop=[],
                completion_token_ids=[],
                completion_token_logprobs=[],
                completion_top_logprobs=[],
                detokenizer=IncrementalDetokenizer(tokenizer),
                chain_id="chain-1",
                request_token="turn",
            )
        }

        with self.assertRaisesRegex(RuntimeError, "abort failed"):
            dispatcher._drain_aborts(active)
        result = await asyncio.wait_for(output_queue.get(), timeout=1)

        self.assertEqual(result["type"], "error")
        self.assertIn("abort failed", result["message"])

    async def test_chat_completion_parses_qwen3_reasoning(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "answer",
                "raw_text": "reason</think>\n\nanswer<|im_end|>",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "token_ids": [1, 2, 3],
                "token_logprobs": [None, None, None],
                "top_logprobs": [None, None, None],
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            prompt="<|im_start|>assistant\n<think>\n",
            reasoning_parser_name="qwen3",
            response_parser=_transformers_response_parser(),
        )

        choice = response["choices"][0]
        self.assertEqual(choice["message"]["reasoning_content"], "reason")
        self.assertEqual(choice["message"]["content"], "answer")
        self.assertEqual(choice["finish_reason"], "stop")

    async def test_chat_completion_parses_reasoning_and_tool_calls(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        raw_text = (
            "<think>need weather</think>"
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        )
        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": raw_text,
                "raw_text": raw_text,
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "token_ids": [1, 2, 3, 4, 5],
                "token_logprobs": [None] * 5,
                "top_logprobs": [None] * 5,
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            reasoning_parser_name="qwen3",
            parse_tools=True,
            response_parser=_transformers_response_parser(),
        )

        choice = response["choices"][0]
        message = choice["message"]
        self.assertEqual(message["reasoning_content"], "need weather")
        self.assertIsNone(message["content"])
        self.assertTrue(message["tool_calls"][0]["id"].startswith("call_"))
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"city":"Paris"}')
        self.assertEqual(choice["finish_reason"], "tool_calls")

    async def test_chat_completion_preserves_content_before_tool_calls(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        raw_text = (
            "THOUGHT: I should inspect the repository first.\n"
            '<tool_call>{"name":"bash","arguments":{"command":"ls -la"}}</tool_call>'
        )
        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": raw_text,
                "raw_text": raw_text,
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "token_ids": [1, 2, 3, 4, 5],
                "token_logprobs": [None] * 5,
                "top_logprobs": [None] * 5,
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            parse_tools=True,
            response_parser=_transformers_response_parser(),
        )

        choice = response["choices"][0]
        self.assertEqual(
            choice["message"]["content"],
            "THOUGHT: I should inspect the repository first.",
        )
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(choice["finish_reason"], "tool_calls")

    async def test_chat_completion_parses_multiple_tool_calls(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        text = (
            '<tool_call>{"name":"first","arguments":{"x":1}}</tool_call>'
            '<tool_call>{"name":"second","arguments":{"y":2}}</tool_call>'
        )
        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": text,
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            parse_tools=True,
            response_parser=_transformers_response_parser(),
        )

        calls = response["choices"][0]["message"]["tool_calls"]
        self.assertEqual([call["function"]["name"] for call in calls], ["first", "second"])

    async def test_chat_completion_parses_qwen_xml_tool_call_with_transformers(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        raw_text = (
            "reason</think>\n\nTHOUGHT: inspect\n\n"
            "<tool_call>\n<function=bash>\n"
            "<parameter=command>\nfind . -maxdepth 2\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": raw_text,
                "raw_text": raw_text,
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "token_ids": [1, 2, 3, 4, 5],
                "token_logprobs": [None] * 5,
                "top_logprobs": [None] * 5,
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            prompt="<|im_start|>assistant\n<think>\n",
            parse_tools=True,
            response_parser=_transformers_response_parser(xml_tools=True),
        )

        choice = response["choices"][0]
        self.assertEqual(choice["message"]["reasoning_content"], "reason")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"], {
            "name": "bash",
            "arguments": '{"command":"find . -maxdepth 2"}',
        })
        self.assertEqual(choice["finish_reason"], "tool_calls")

    async def test_chat_completion_reasoning_length_remains_explicit(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": "<think>partial",
                "finish_reason": "length",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        response = await _chat_completion_response(
            "chatcmpl-test",
            123,
            "model",
            [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            reasoning_parser_name="qwen3",
            response_parser=_transformers_response_parser(),
        )

        choice = response["choices"][0]
        self.assertEqual(choice["message"]["reasoning_content"], "partial")
        self.assertEqual(choice["message"]["content"], "")
        self.assertEqual(choice["finish_reason"], "length")

    async def test_chat_completion_transformers_controls_parse_outcomes(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        async def parse(text, *, reasoning_parser_name=None, parse_tools=False):
            queue = asyncio.Queue()
            await queue.put(
                {
                    "type": "final",
                    "index": 0,
                    "text": text,
                    "raw_text": text,
                    "finish_reason": "stop",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "token_ids": [1],
                    "token_logprobs": [None],
                    "top_logprobs": [None],
                }
            )
            return await _chat_completion_response(
                "chatcmpl-test",
                123,
                "model",
                [RequestHandle(output_queue=queue, cancelled=threading.Event())],
                reasoning_parser_name=reasoning_parser_name,
                parse_tools=parse_tools,
                response_parser=_transformers_response_parser(),
            )

        response = await parse("<think>partial", reasoning_parser_name="qwen3")
        self.assertEqual(response["choices"][0]["message"]["reasoning_content"], "partial")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")

        malformed = '<tool_call>{"name":</tool_call>'
        response = await parse(malformed, parse_tools=True)
        choice = response["choices"][0]
        self.assertEqual(choice["message"]["content"], malformed)
        self.assertNotIn("tool_calls", choice["message"])
        self.assertEqual(choice["finish_reason"], "stop")

    async def test_chat_completion_glm_empty_tool_name_preserves_output(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_response

        for text in (
            "</think><tool_call></tool_call>",
            "</think><tool_call>   </tool_call>",
            "</think><tool_call><arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>",
        ):
            with self.subTest(text=text):
                queue = asyncio.Queue()
                await queue.put({
                    "type": "final", "index": 0, "text": text, "raw_text": text,
                    "finish_reason": "length", "prompt_tokens": 12,
                    "completion_tokens": 3, "token_ids": [1, 2, 3],
                    "token_logprobs": [None] * 3, "top_logprobs": [None] * 3,
                })
                response = await _chat_completion_response(
                    "chatcmpl-test", 123, "model",
                    [RequestHandle(output_queue=queue, cancelled=threading.Event())],
                    prompt="[gMASK]<sop><|assistant|><think>",
                    parse_tools=True,
                    response_parser=_transformers_response_parser(glm_tools=True),
                )
                choice = response["choices"][0]
                self.assertEqual(choice["message"]["content"], text)
                self.assertNotIn("tool_calls", choice["message"])
                self.assertEqual(choice["finish_reason"], "length")
                self.assertEqual(response["usage"]["completion_tokens"], 3)
                self.assertEqual(response["usage"]["prompt_tokens"], 12)

    async def test_chat_completion_does_not_mask_parser_contract_errors(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import (
            RequestHandle,
            _chat_completion_response,
        )
        from sparseengine.entrypoints.openai.serving.response_parsing import (
            ResponseParseError,
        )

        class BrokenParser:
            def parse(self, *_args, **_kwargs):
                raise ResponseParseError("unexpected parsed response schema")

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "output",
                "raw_text": "output",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        with self.assertRaises(HTTPException) as raised:
            await _chat_completion_response(
                "chatcmpl-test",
                123,
                "model",
                [
                    RequestHandle(
                        output_queue=queue,
                        cancelled=threading.Event(),
                    )
                ],
                parse_tools=True,
                response_parser=BrokenParser(),
            )

        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("unexpected parsed response schema", raised.exception.detail)

    def test_chat_logprobs_reject_parsed_outputs(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import ChatCompletionRequest, _validate_chat_request

        base = {
            "model": "m",
            "messages": [{"role": "user", "content": "p"}],
            "logprobs": True,
        }
        with self.assertRaisesRegex(HTTPException, "cannot be aligned"):
            _validate_chat_request(
                ChatCompletionRequest(**base),
                "m",
                reasoning_parser_name="qwen3",
                response_parser=_transformers_response_parser(),
            )
        with self.assertRaisesRegex(HTTPException, "cannot be aligned"):
            _validate_chat_request(
                ChatCompletionRequest(
                    **base,
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "search", "parameters": {}},
                        }
                    ],
                ),
                "m",
                response_parser=_transformers_response_parser(),
            )

    async def test_chat_stream_starts_with_assistant_role(self):
        from sparseengine.entrypoints.openai import api_server

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "text_delta": "",
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 0,
                "token_ids": [],
                "token_logprobs": [],
                "top_logprobs": [],
            }
        )
        handle = api_server.RequestHandle(output_queue=queue, cancelled=threading.Event())

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in api_server._chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [handle],
            )
        ]

        first = json.loads(chunks[0].removeprefix("data: "))
        self.assertEqual(first["choices"][0]["delta"], {"role": "assistant"})

    async def test_chat_stream_parses_reasoning_and_tool_calls(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_stream

        raw_text = (
            "reason</think>"
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        )
        queue = asyncio.Queue()
        for delta in (
            "rea",
            "son</think><tool_",
            'call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
        ):
            await queue.put(
                {
                    "type": "token",
                    "index": 0,
                    "text": delta,
                    "raw_text_delta": delta,
                    "token_ids": [1],
                    "token_logprobs": [None],
                    "top_logprobs": [None],
                }
            )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": raw_text,
                "raw_text": raw_text,
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 3,
                "token_ids": [1, 2, 3],
                "token_logprobs": [None] * 3,
                "top_logprobs": [None] * 3,
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in _chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [RequestHandle(output_queue=queue, cancelled=threading.Event())],
                include_usage=True,
                prompt="<|im_start|>assistant\n<think>\n",
                reasoning_parser_name="qwen3",
                parse_tools=True,
                response_parser=_transformers_response_parser(),
            )
        ]
        payloads = [
            payload
            for event, payload in _response_sse_events(chunks)
            if event != "[DONE]"
        ]
        deltas = [choice["delta"] for payload in payloads for choice in payload.get("choices", [])]

        self.assertEqual(
            "".join(delta.get("reasoning_content", "") for delta in deltas),
            "reason",
        )
        tool_deltas = [delta["tool_calls"][0] for delta in deltas if delta.get("tool_calls")]
        self.assertEqual(tool_deltas[0]["index"], 0)
        self.assertTrue(tool_deltas[0]["id"].startswith("call_"))
        self.assertEqual(
            tool_deltas[0]["function"],
            {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        )
        final_choice = [
            choice
            for payload in payloads
            for choice in payload.get("choices", [])
            if choice["finish_reason"] is not None
        ][0]
        self.assertEqual(final_choice["finish_reason"], "tool_calls")
        usage = [payload["usage"] for payload in payloads if not payload.get("choices")][0]
        self.assertEqual(usage, {
            "prompt_tokens": 3,
            "completion_tokens": 3,
            "total_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 0},
        })

    async def test_multimodal_chat_stream_uses_admitted_parser_prefix(self):
        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            RequestHandle,
        )
        from sparseengine.entrypoints.openai.serving.chat import (
            serve_chat_completion,
        )

        raw_text = "inspect image</think>\n\nanswer<|im_end|>"
        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "",
                "raw_text_delta": raw_text,
                "token_ids": [20],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "answer",
                "raw_text": raw_text,
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "token_ids": [20],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        class Tokenizer(_TransformersResponseTokenizer):
            def decode(self, token_ids, *, skip_special_tokens):
                self.asserted_token_ids = list(token_ids)
                self.asserted_skip_special_tokens = skip_special_tokens
                return (
                    "<|im_start|>user\n<|vision_start|><|image_pad|>"
                    "<|vision_end|>describe<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n"
                )

        tokenizer = Tokenizer()
        handle = RequestHandle(
            output_queue=queue,
            cancelled=threading.Event(),
            prompt_token_ids=[10, 11, 12],
        )

        class Dispatcher:
            admission_ack_enabled = True

            async def submit(self, *_args, **_kwargs):
                raise AssertionError("submit_admitted must be used")

            async def submit_admitted(self, *_args, **_kwargs):
                return handle

            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        response = await serve_chat_completion(
            ChatCompletionRequest(
                model="model",
                stream=True,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://x/image.png"},
                            },
                            {"type": "text", "text": "describe"},
                        ],
                    }
                ],
            ),
            Dispatcher(),
            tokenizer,
            "model",
            None,
            reasoning_parser_name="qwen3",
            response_parser=_transformers_response_parser(),
        )
        chunks = [chunk async for chunk in response.body_iterator]
        payloads = [
            payload
            for event, payload in _response_sse_events(chunks)
            if event != "[DONE]"
        ]
        deltas = [
            choice["delta"]
            for payload in payloads
            for choice in payload.get("choices", [])
        ]

        self.assertEqual(tokenizer.asserted_token_ids, [10, 11, 12])
        self.assertFalse(tokenizer.asserted_skip_special_tokens)
        self.assertEqual(
            "".join(delta.get("reasoning_content", "") for delta in deltas),
            "inspect image",
        )
        self.assertEqual(
            "".join(delta.get("content", "") for delta in deltas),
            "\n\nanswer",
        )

    async def test_chat_stream_parser_disabled_preserves_raw_content(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_stream

        text = "<think>reason</think>answer"
        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": text,
                "raw_text_delta": text,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": text,
                "raw_text": text,
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in _chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [RequestHandle(output_queue=queue, cancelled=threading.Event())],
            )
        ]
        content = "".join(
            (payload["choices"][0]["delta"] or {}).get("content", "")
            for event, payload in _response_sse_events(chunks)
            if event != "[DONE]" and payload.get("choices")
        )
        self.assertEqual(content, text)

    def test_chat_stream_parser_indexes_multiple_tool_calls(self):
        parser = _transformers_response_parser().stream(prefix="", parse_tools=True)
        deltas = parser.feed(
            '<tool_call>{"name":"first","arguments":{"x":1}}</tool_call>'
        )
        deltas.extend(
            parser.feed(
                '<tool_call>{"name":"second","arguments":{"y":2}}</tool_call>'
            )
        )
        deltas.extend(parser.finish())

        starts = [
            delta["tool_calls"][0]
            for delta in deltas
            if delta.get("tool_calls") and "id" in delta["tool_calls"][0]
        ]
        self.assertEqual([start["index"] for start in starts], [0, 1])
        self.assertNotEqual(starts[0]["id"], starts[1]["id"])

    async def test_chat_stream_reasoning_length_finishes_explicitly(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": "<think>partial",
                "finish_reason": "length",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in _chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [RequestHandle(output_queue=queue, cancelled=threading.Event())],
                reasoning_parser_name="qwen3",
                response_parser=_transformers_response_parser(),
            )
        ]
        payloads = [
            payload
            for event, payload in _response_sse_events(chunks)
            if event != "[DONE]"
        ]
        self.assertEqual(
            "".join(
                choice["delta"].get("reasoning_content", "")
                for payload in payloads
                for choice in payload.get("choices", [])
            ),
            "partial",
        )
        self.assertEqual(
            [
                choice["finish_reason"]
                for payload in payloads
                for choice in payload.get("choices", [])
                if choice["finish_reason"] is not None
            ],
            ["length"],
        )

    async def test_chat_stream_parse_failure_is_visible_and_cancels(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": '<tool_call>{"name":</tool_call>',
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        handle = RequestHandle(output_queue=queue, cancelled=threading.Event())
        cancelled = threading.Event()

        class Dispatcher:
            def cancel(self, observed_handle):
                self_observed = observed_handle
                if self_observed is handle:
                    cancelled.set()

        chunks = [
            chunk
            async for chunk in _chat_completion_stream(
                Dispatcher(),
                "chatcmpl-test",
                123,
                "model",
                [handle],
                parse_tools=True,
                response_parser=_transformers_response_parser(),
            )
        ]
        events = _response_sse_events(chunks)
        errors = [payload for event, payload in events if event != "[DONE]" and payload.get("object") == "error"]
        self.assertIn("could not parse region as JSON", errors[0]["message"])
        self.assertEqual(events[-1], ("[DONE]", None))
        self.assertTrue(cancelled.is_set())

    async def test_chat_stream_cancel_releases_dispatcher_request(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_stream

        queue = asyncio.Queue()
        handle = RequestHandle(output_queue=queue, cancelled=threading.Event())
        cancelled = threading.Event()

        class Dispatcher:
            def cancel(self, observed_handle):
                if observed_handle is handle:
                    cancelled.set()

        stream = _chat_completion_stream(
            Dispatcher(),
            "chatcmpl-test",
            123,
            "model",
            [handle],
        )
        await stream.__anext__()
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

        self.assertTrue(cancelled.is_set())

    async def test_chat_stream_disconnect_releases_dispatcher_request(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _chat_completion_stream

        queue = asyncio.Queue()
        handle = RequestHandle(output_queue=queue, cancelled=threading.Event())
        cancelled = threading.Event()

        class Dispatcher:
            def cancel(self, observed_handle):
                if observed_handle is handle:
                    cancelled.set()

        async def is_disconnected():
            return True

        stream = _chat_completion_stream(
            Dispatcher(),
            "chatcmpl-test",
            123,
            "model",
            [handle],
            is_disconnected=is_disconnected,
        )
        await stream.__anext__()
        with self.assertRaises(asyncio.CancelledError):
            await stream.__anext__()

        self.assertTrue(cancelled.is_set())

    def test_completion_logprobs_serializes_sampled_tokens(self):
        from sparseengine.entrypoints.openai.api_server import _completion_logprobs

        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return {1: "a", 2: "b", 3: "c"}[token_ids[0]]

        logprobs = _completion_logprobs(
            Tokenizer(),
            [1, 2],
            [-0.1, -0.2],
            [{1: -0.1, 3: -1.0}, None],
        )

        self.assertEqual(logprobs["tokens"], ["a", "b"])
        self.assertEqual(logprobs["token_logprobs"], [-0.1, -0.2])
        self.assertEqual(logprobs["top_logprobs"][0], {"a": -0.1, "c": -1.0})

    def test_chat_logprobs_true_requests_sampled_logprobs(self):
        from sparseengine.entrypoints.openai.api_server import ChatCompletionRequest, _sampling_params_from_request

        request = ChatCompletionRequest(
            model="model",
            messages=[{"role": "user", "content": "hello"}],
            logprobs=True,
        )

        self.assertEqual(_sampling_params_from_request(request).logprobs, 0)

    def test_response_prompt_renders_string_input_and_instructions(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None
                self.kwargs = None

            def apply_chat_template(self, chat, **kwargs):
                self.chat = chat
                self.kwargs = kwargs
                return "rendered"

        tokenizer = Tokenizer()
        request = ResponseRequest(
            model="model",
            instructions="policy",
            input="hello",
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertEqual(_response_prompt(tokenizer, request), "rendered")
        self.assertEqual(
            tokenizer.chat,
            [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_response_prompt_rejects_unsupported_item_type(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = None

        with self.assertRaisesRegex(ValueError, "Unsupported responses input item type"):
            _response_prompt(
                Tokenizer(),
                ResponseRequest(model="model", input=[{"type": "image", "image_url": "x"}]),
            )

    def test_response_prompt_preserves_multimodal_parts_for_model_processor(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _response_prompt
        from sparseengine.multimodal import MultiModalPrompt

        request = ResponseRequest(
            model="model",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://x/image.png"},
                        {"type": "input_text", "text": "describe"},
                    ],
                }
            ],
        )

        prompt = _response_prompt(object(), request)

        self.assertIsInstance(prompt, MultiModalPrompt)
        self.assertEqual(prompt.messages[0]["content"][0]["type"], "input_image")
        self.assertEqual(prompt.messages[0]["content"][1], {"type": "text", "text": "describe"})

    def test_response_prompt_validates_multimodal_parts(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        cases = [
            ({"type": "input_image"}, "requires a non-empty image_url"),
            ({"type": "input_video", "video_url": ""}, "requires a non-empty video_url"),
            ({"type": "input_audio", "input_audio": {}}, "base64 data string"),
            (
                {
                    "type": "input_audio",
                    "input_audio": {"data": "AAAA", "format": "mp3"},
                },
                "Only WAV",
            ),
            (
                {"type": "input_image", "image_url": "https://x/image.png", "extra": 1},
                "unsupported fields",
            ),
        ]
        for part, error in cases:
            with self.subTest(part=part), self.assertRaisesRegex(ValueError, error):
                _response_prompt(
                    object(),
                    ResponseRequest(
                        model="model",
                        input=[
                            {
                                "type": "message",
                                "role": "user",
                                "content": [part],
                            }
                        ],
                    ),
                )

    async def test_multimodal_admission_errors_return_bad_request(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import (
            ChatCompletionRequest,
            ResponseRequest,
        )
        from sparseengine.entrypoints.openai.serving.chat import serve_chat_completion
        from sparseengine.entrypoints.openai.serving.responses import serve_response

        class Dispatcher:
            admission_ack_enabled = True

            async def submit(self, *_args, **_kwargs):
                raise AssertionError("submit_admitted must be used")

            async def submit_admitted(self, *_args, **_kwargs):
                raise NotImplementedError("checkpoint does not support audio")

        class Tokenizer:
            chat_template = "template"

            def apply_chat_template(self, *_args, **_kwargs):
                return "rendered"

        requests = [
            serve_chat_completion(
                ChatCompletionRequest(
                    model="model", messages=[{"role": "user", "content": "hello"}]
                ),
                Dispatcher(),
                Tokenizer(),
                "model",
                None,
            ),
            serve_response(
                ResponseRequest(model="model", input="hello"),
                Dispatcher(),
                Tokenizer(),
                "model",
                None,
                None,
            ),
        ]
        for request in requests:
            with self.assertRaises(HTTPException) as ctx:
                await request
            self.assertEqual(ctx.exception.status_code, 400)

    def test_response_prompt_passes_tools_and_tool_outputs(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = "template"

            def __init__(self):
                self.chat = None
                self.tools = None

            def apply_chat_template(self, chat, tools=None, **_kwargs):
                self.chat = chat
                self.tools = tools
                return "rendered"

        tokenizer = Tokenizer()
        request = ResponseRequest(
            model="model",
            input=[{"type": "function_call_output", "call_id": "call_1", "output": '{"ok":true}'}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        self.assertEqual(_response_prompt(tokenizer, request), "rendered")
        self.assertEqual(tokenizer.chat, [{"role": "tool", "content": '{"ok":true}', "tool_call_id": "call_1"}])
        self.assertEqual(tokenizer.tools[0]["name"], "get_weather")

    def test_response_prompt_adapts_minimax_tool_history(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = "<minimax:tool_call>{{ tool.function }}"

            def __init__(self):
                self.chat = None
                self.tools = None

            def apply_chat_template(self, chat, tools=None, **_kwargs):
                self.chat = chat
                self.tools = tools
                return "rendered"

        tokenizer = Tokenizer()
        request = ResponseRequest(
            model="model",
            input=[
                {"type": "message", "role": "user", "content": "weather"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        self.assertEqual(_response_prompt(tokenizer, request), "rendered")
        self.assertEqual(
            tokenizer.chat[1]["tool_calls"][0]["function"]["arguments"],
            {"city": "Paris"},
        )
        self.assertEqual(tokenizer.chat[2]["tool_call_id"], "call_1")
        self.assertEqual(tokenizer.tools[0]["function"]["name"], "get_weather")

    def test_response_prompt_rejects_tools_without_template_support(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = "template"

            def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
                del chat, tokenize, add_generation_prompt
                return "rendered"

        with self.assertRaisesRegex(ValueError, "does not support tools"):
            _response_prompt(
                Tokenizer(),
                ResponseRequest(
                    model="model",
                    input="hello",
                    tools=[{"type": "function", "name": "tool", "parameters": {}}],
                ),
            )

    def test_response_prompt_rejects_tool_history_without_template(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _response_prompt

        class Tokenizer:
            chat_template = None

        for item in [
            {"type": "function_call", "call_id": "call_1", "name": "tool", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
        ]:
            with self.assertRaisesRegex(ValueError, "tool-call history requires"):
                _response_prompt(Tokenizer(), ResponseRequest(model="model", input=[item]))

    def test_response_reasoning_effort_conflicts_fail_fast(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _validate_response_request

        with self.assertRaises(HTTPException) as ctx:
            _validate_response_request(
                ResponseRequest(
                    model="model",
                    input="hello",
                    reasoning={"effort": "none"},
                    chat_template_kwargs={"enable_thinking": True},
                ),
                "model",
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_response_unimplemented_control_fields_fail_fast(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _validate_response_request

        for request in [
            ResponseRequest(model="model", input="hello", tool_choice="required"),
            ResponseRequest(model="model", input="hello", parallel_tool_calls=False),
            ResponseRequest(model="model", input="hello", reasoning={"summary": "auto"}),
            ResponseRequest(model="model", input="hello", store=True),
        ]:
            with self.assertRaises(HTTPException) as ctx:
                _validate_response_request(request, "model")
            self.assertEqual(ctx.exception.status_code, 400)

    def test_response_accepts_opencode_compatibility_fields(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _response_prompt
        from sparseengine.entrypoints.openai.api_server import _validate_response_request

        class Tokenizer:
            chat_template = None

        request = ResponseRequest(
            model="model",
            input="hello",
            store=False,
            prompt_cache_key="ses_test",
        )

        _validate_response_request(request, "model")
        self.assertFalse(request.store)
        self.assertEqual(request.prompt_cache_key, "ses_test")
        self.assertEqual(_response_prompt(Tokenizer(), request), "user: hello\nassistant:")

    def test_response_max_output_tokens_maps_to_sampling_params(self):
        from sparseengine.entrypoints.openai.api_server import ResponseRequest, _sampling_params_from_response_request

        request = ResponseRequest(model="model", input="hello", max_output_tokens=7)

        self.assertEqual(_sampling_params_from_response_request(request).max_tokens, 7)

    async def test_response_response_shape_and_usage(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _response_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "hello",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 2,
            }
        )

        response = await _response_response(
            "resp_test",
            123,
            "model",
            RequestHandle(output_queue=queue, cancelled=threading.Event()),
            reasoning_parser_name=None,
        )

        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["type"], "message")
        self.assertEqual(response["output"][0]["content"][0]["text"], "hello")
        self.assertEqual(response["usage"], {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})

    async def test_response_reasoning_parser_uses_raw_text(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, _response_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "answer",
                "raw_text": "<think>reason</think>answer",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 3,
            }
        )

        response = await _response_response(
            "resp_test",
            123,
            "model",
            RequestHandle(output_queue=queue, cancelled=threading.Event()),
            reasoning_parser_name="qwen3",
            response_parser=_transformers_response_parser(),
        )

        self.assertEqual(response["output"][0]["type"], "reasoning")
        self.assertEqual(response["output"][0]["text"], "reason")
        self.assertEqual(response["output"][1]["content"][0]["text"], "answer")

    async def test_response_stream_true_returns_responses_sse(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest
        from sparseengine.entrypoints.openai.serving.responses import serve_response

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "hel",
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "hello",
                "finish_reason": "stop",
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        class Dispatcher:
            async def submit(self, prompt, _sampling_params, index, stop, *, stream=True):
                self.prompt = prompt
                self.index = index
                self.stop = stop
                return RequestHandle(output_queue=queue, cancelled=threading.Event())

            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        class Tokenizer:
            chat_template = None

        response = await serve_response(
            ResponseRequest(model="model", input="hello", stream=True),
            dispatcher=Dispatcher(),
            tokenizer=Tokenizer(),
            served_model_name="model",
            request_log_path=None,
            reasoning_parser_name=None,
        )
        chunks = [chunk async for chunk in response.body_iterator]
        events = _response_sse_events(chunks)
        event_types = [event for event, _payload in events]

        self.assertEqual(response.media_type, "text/event-stream")
        self.assertIn("response.created", event_types)
        self.assertIn("response.output_item.added", event_types)
        self.assertIn("response.content_part.added", event_types)
        self.assertIn("response.output_text.delta", event_types)
        self.assertIn("response.output_text.done", event_types)
        self.assertIn("response.content_part.done", event_types)
        self.assertIn("response.output_item.done", event_types)
        self.assertEqual(event_types[-2:], ["response.completed", "[DONE]"])
        completed = events[-2][1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["output"][0]["content"][0]["text"], "hello")
        self.assertEqual(completed["usage"], {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})

    async def test_response_stream_qwen3_reasoning_uses_raw_delta(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        for delta in ["<thi", "nk>rea", "son</thi", "nk>answer"]:
            await queue.put(
                {
                    "type": "token",
                    "index": 0,
                    "text": "",
                    "raw_text_delta": delta,
                    "token_ids": [1],
                    "token_logprobs": [None],
                    "top_logprobs": [None],
                }
            )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "answer",
                "raw_text": "<think>reason</think>answer",
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "token_ids": [1, 2, 3, 4],
                "token_logprobs": [None, None, None, None],
                "top_logprobs": [None, None, None, None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in _response_stream(
                Dispatcher(),
                "resp_test",
                123,
                "model",
                RequestHandle(output_queue=queue, cancelled=threading.Event()),
                time.perf_counter() - 1.0,
                None,
                ResponseRequest(model="model", input="hello", stream=True),
                reasoning_parser_name="qwen3",
                response_parser=_transformers_response_parser(),
            )
        ]

        events = _response_sse_events(chunks)
        reasoning_deltas = [
            payload["delta"]
            for event, payload in events
            if event == "response.reasoning_text.delta"
        ]
        output_deltas = [
            payload["delta"]
            for event, payload in events
            if event == "response.output_text.delta"
        ]
        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]

        self.assertEqual("".join(reasoning_deltas), "reason")
        self.assertEqual("".join(output_deltas), "answer")
        self.assertEqual(completed["output"][0]["type"], "reasoning")
        self.assertEqual(completed["output"][1]["content"][0]["text"], "answer")

    async def test_response_stream_parser_disabled_returns_raw_visible_text(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "<think>reason</think>answer",
                "raw_text_delta": "<think>reason</think>answer",
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "<think>reason</think>answer",
                "raw_text": "<think>reason</think>answer",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        chunks = [
            chunk
            async for chunk in _response_stream(
                Dispatcher(),
                "resp_test",
                123,
                "model",
                RequestHandle(output_queue=queue, cancelled=threading.Event()),
                time.perf_counter(),
                None,
                ResponseRequest(model="model", input="hello", stream=True),
                reasoning_parser_name=None,
            )
        ]

        output_deltas = [
            payload["delta"]
            for event, payload in _response_sse_events(chunks)
            if event == "response.output_text.delta"
        ]
        self.assertEqual("".join(output_deltas), "<think>reason</think>answer")

    async def test_response_stream_qwen3_thinking_off_streams_plain_answer(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "hel",
                "raw_text_delta": "hel",
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "hello",
                "raw_text": "hello",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(model="model", input="hello", stream=True, reasoning={"effort": "none"}),
                    reasoning_parser_name="qwen3",
                    response_parser=_transformers_response_parser(),
                )
            ]
        )

        event_types = [event for event, _payload in events]
        output_deltas = [
            payload["delta"]
            for event, payload in events
            if event == "response.output_text.delta"
        ]
        self.assertLess(
            event_types.index("response.output_text.delta"),
            event_types.index("response.completed"),
        )
        self.assertEqual("".join(output_deltas), "hello")

    async def test_response_stream_reasoning_length_finishes_incomplete(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": "",
                "raw_text_delta": "<think>partial",
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": "<think>partial",
                "finish_reason": "length",
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(model="model", input="hello", stream=True),
                    reasoning_parser_name="qwen3",
                    response_parser=_transformers_response_parser(),
                )
            ]
        )

        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]
        self.assertEqual(completed["status"], "incomplete")
        self.assertEqual(completed["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(completed["output"][0]["text"], "partial")

    async def test_response_stream_transformers_finalizes_unclosed_reasoning(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": "<think>partial",
                "finish_reason": "stop",
                "prompt_tokens": 2,
                "completion_tokens": 1,
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("completed response stream should not be cancelled")

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(model="model", input="hello", stream=True),
                    reasoning_parser_name="qwen3",
                    response_parser=_transformers_response_parser(),
                )
            ]
        )

        self.assertNotIn("response.failed", [event for event, _payload in events])
        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["output"][0]["text"], "partial")

    def test_response_route_returns_non_streaming_response(self):
        from fastapi.testclient import TestClient
        from sparseengine.entrypoints.openai import api_server

        test_tokenizer = _byte_level_tokenizer()
        completion_token_ids = test_tokenizer.encode("hello")

        class Engine:
            tokenizer = test_tokenizer
            config = type("Config", (), {"sparse_method": ""})()
            last_step_token_outputs = []
            last_step_logprob_outputs = []

            def add_request(self, prompt, sampling_params):
                self.prompt = prompt
                self.sampling_params = sampling_params
                return 1

            def step(self):
                return [
                    (
                        1,
                        completion_token_ids,
                        [None] * len(completion_token_ids),
                        [None] * len(completion_token_ids),
                    )
                ], 0

            def abort_request(self, _seq_id):
                pass

            def exit(self):
                pass

        engine = Engine()
        app = api_server.create_app("/tmp/model", served_model_name="model", engine=engine)
        try:
            response = TestClient(app).post(
                "/v1/responses",
                json={
                    "model": "model",
                    "input": "hello",
                    "max_output_tokens": 4,
                    "store": False,
                    "prompt_cache_key": "ses_test",
                },
            )
        finally:
            app.state.dispatcher.close()

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["output"][0]["content"][0]["text"], "hello")
        self.assertEqual(engine.sampling_params.max_tokens, 4)

    def test_qwen3_reasoning_parser_builds_reasoning_item(self):
        from sparseengine.entrypoints.openai.api_server import _response_output_items

        parsed = _transformers_response_parser().parse(
            "<think>reason</think>answer",
            prefix="",
            parse_tools=False,
        )
        output = _response_output_items(parsed)

        self.assertEqual(output[0]["type"], "reasoning")
        self.assertEqual(output[0]["text"], "reason")
        self.assertEqual(output[1]["content"][0]["text"], "answer")

    def test_qwen3_reasoning_parser_handles_template_opened_think(self):
        from sparseengine.entrypoints.openai.api_server import _response_output_items

        parsed = _transformers_response_parser().parse(
            "reason</think>\n\nanswer<|im_end|>",
            prefix="<|im_start|>assistant\n<think>\n",
            parse_tools=False,
        )
        output = _response_output_items(parsed)

        self.assertEqual(output[0]["type"], "reasoning")
        self.assertEqual(output[0]["text"], "reason")
        self.assertEqual(output[1]["content"][0]["text"], "answer")

    def test_multimodal_parser_prefix_restores_template_opened_think(self):
        from sparseengine.entrypoints.openai.dispatcher import RequestHandle
        from sparseengine.entrypoints.openai.serving.response_parsing import (
            response_parser_prefix,
        )
        from sparseengine.multimodal import MultiModalPrompt

        class Tokenizer:
            def decode(self, token_ids, *, skip_special_tokens):
                self.asserted_token_ids = list(token_ids)
                self.asserted_skip_special_tokens = skip_special_tokens
                return (
                    "<|im_start|>user\n<|vision_start|><|image_pad|>"
                    "<|vision_end|>describe<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n"
                )

        tokenizer = Tokenizer()
        handle = RequestHandle(
            output_queue=asyncio.Queue(),
            cancelled=threading.Event(),
            prompt_token_ids=[10, 11, 12],
        )
        prompt = MultiModalPrompt(
            [{"role": "user", "content": [{"type": "image"}]}]
        )
        prefix = response_parser_prefix(tokenizer, prompt, handle)
        parsed = _transformers_response_parser().parse(
            "reason</think>\n\nanswer<|im_end|>",
            prefix=prefix,
            parse_tools=False,
        )

        self.assertEqual(tokenizer.asserted_token_ids, [10, 11, 12])
        self.assertFalse(tokenizer.asserted_skip_special_tokens)
        self.assertEqual(parsed.reasoning_content, "reason")
        self.assertEqual(parsed.content, "answer")

    def test_qwen3_reasoning_parser_handles_unclosed_region(self):
        from sparseengine.entrypoints.openai.api_server import _response_output_items

        parsed = _transformers_response_parser().parse(
            "<think>partial",
            prefix="",
            parse_tools=False,
        )
        output = _response_output_items(parsed)

        self.assertEqual(output, [{"id": output[0]["id"], "type": "reasoning", "text": "partial", "summary": []}])

    def test_reasoning_parser_disabled_returns_raw_text(self):
        from sparseengine.entrypoints.openai.api_server import _response_output_items

        from sparseengine.entrypoints.openai.serving.response_parsing import ParsedModelResponse

        output = _response_output_items(
            ParsedModelResponse(None, "<think>reason</think>answer", [])
        )

        self.assertEqual(output[0]["content"][0]["text"], "<think>reason</think>answer")

    def test_tool_call_output_item_is_parsed(self):
        from sparseengine.entrypoints.openai.api_server import _response_output_items

        parsed = _transformers_response_parser().parse(
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
            prefix="",
            parse_tools=True,
        )
        output = _response_output_items(parsed)

        self.assertEqual(output[0]["type"], "function_call")
        self.assertEqual(output[0]["name"], "get_weather")
        self.assertEqual(output[0]["arguments"], '{"city":"Paris"}')

    def test_qwen_nonthinking_tool_call_parser_is_available(self):
        from sparseengine.entrypoints.openai.serving.response_parsing import (
            TransformersResponseParser,
        )

        tokenizer = _TransformersResponseTokenizer()
        tokenizer.chat_template = "<tool_call>"
        parser = TransformersResponseParser.from_tokenizer(tokenizer)

        self.assertIsNotNone(parser)
        parsed = parser.parse(
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
            prefix="",
            parse_tools=True,
        )
        self.assertEqual(parsed.reasoning_content, None)
        self.assertEqual(parsed.content, "")
        self.assertEqual(parsed.tool_calls[0]["function"], {
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        })

    def test_transformers_decides_xml_tool_delimiter_handling(self):
        parsed = _transformers_response_parser(xml_tools=True).parse(
            "reason</think>\n\n"
            "<tool_call><function=bash>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call></function>",
            prefix="<|im_start|>assistant\n<think>\n",
            parse_tools=True,
        )

        self.assertEqual(parsed.reasoning_content, "reason")
        self.assertEqual(parsed.content, "</function>")
        self.assertEqual(parsed.tool_calls[0]["function"], {
            "name": "bash",
            "arguments": '{"command":"pwd"}',
        })

    def test_qwen_xml_parser_normalizes_tool_openers(self):
        parser = _transformers_response_parser(xml_tools=True)
        raw = (
            "<tool_call><tool_call>\n\n<tool_call>\n\n<tool_call>\n"
            "[bash><parameter=command>pwd</parameter>"
            "</function></tool_call>"
        )

        parsed = parser.parse(raw, prefix="", parse_tools=True)
        stream = parser.stream(prefix="", parse_tools=True)
        deltas = []
        for chunk in (
            "<tool_call>",
            "<tool_call>\n\n",
            "<tool_call>\n\n",
            "<tool_call>\n[bash><parameter=command>pwd</parameter>",
            "</function></tool_call>",
        ):
            deltas.extend(stream.feed(chunk))
        deltas.extend(stream.finish())

        self.assertEqual(parsed.content, "")
        self.assertEqual(parsed.tool_calls[0]["function"]["name"], "bash")
        self.assertFalse(any("content" in delta for delta in deltas))
        self.assertEqual(deltas[0]["tool_calls"][0]["function"]["name"], "bash")

        parsed = parser.parse(
            "<tool_call><parameter=bash\n<parameter=command>pwd</parameter>"
            "</function></tool_call>",
            prefix="",
            parse_tools=True,
        )
        self.assertEqual(parsed.tool_calls[0]["function"]["name"], "bash")

    def test_glm_response_parser_handles_thinking_and_plain_content(self):
        parser = _transformers_response_parser(glm_tools=True)

        parsed = parser.parse(
            "先分析🙂</think>最终答案<|endoftext|>",
            prefix="[gMASK]<sop><|assistant|><think>",
            parse_tools=False,
        )
        nonthinking = parser.parse(
            "直接回答<|endoftext|>",
            prefix="[gMASK]<sop><|assistant|></think>",
            parse_tools=False,
        )

        self.assertEqual(parsed.reasoning_content, "先分析🙂")
        self.assertEqual(parsed.content, "最终答案")
        self.assertEqual(nonthinking.reasoning_content, None)
        self.assertEqual(nonthinking.content, "直接回答")

    def test_glm_response_parser_handles_parallel_tool_calls(self):
        parsed = _transformers_response_parser(glm_tools=True).parse(
            "分析</think>"
            "<tool_call>天气查询"
            "<arg_key>城市</arg_key><arg_value>北京</arg_value>"
            "<arg_key>天数</arg_key><arg_value>2</arg_value>"
            "</tool_call>"
            "<tool_call>echo"
            "<arg_key>text</arg_key><arg_value>你好🙂</arg_value>"
            "</tool_call><|endoftext|>",
            prefix="[gMASK]<sop><|assistant|><think>",
            parse_tools=True,
        )

        self.assertEqual(parsed.reasoning_content, "分析")
        self.assertEqual(parsed.content, "")
        self.assertEqual(
            [call["function"]["name"] for call in parsed.tool_calls],
            ["天气查询", "echo"],
        )
        self.assertEqual(
            parsed.tool_calls[0]["function"]["arguments"],
            '{"城市":"北京","天数":2}',
        )
        self.assertEqual(
            parsed.tool_calls[1]["function"]["arguments"],
            '{"text":"你好🙂"}',
        )

    def test_glm_response_stream_parser_handles_split_tags(self):
        parser = _transformers_response_parser(glm_tools=True).stream(
            prefix="[gMASK]<sop><|assistant|><think>",
            parse_tools=True,
        )
        deltas = []
        for chunk in [
            "推",
            "理</thi",
            "nk>答",
            "案<tool_",
            "call>天气<arg_key>城市</arg_key><arg_value>北",
            "京</arg_value></tool_",
            "call><|endoftext|>",
        ]:
            deltas.extend(parser.feed(chunk))
        deltas.extend(parser.finish())

        reasoning = "".join(
            delta.get("reasoning_content", "") for delta in deltas
        )
        content = "".join(delta.get("content", "") for delta in deltas)
        calls = [delta["tool_calls"][0] for delta in deltas if "tool_calls" in delta]
        self.assertEqual(reasoning, "推理")
        self.assertEqual(content, "答案")
        self.assertEqual(calls[0]["function"]["name"], "天气")
        self.assertEqual(calls[0]["function"]["arguments"], '{"城市":"北京"}')

    def test_glm_response_parser_rejects_empty_tool_name(self):
        from sparseengine.entrypoints.openai.serving.response_parsing import (
            ModelOutputParseError,
        )

        with self.assertRaisesRegex(ModelOutputParseError, "non-empty function name"):
            _transformers_response_parser(glm_tools=True).parse(
                "</think><tool_call></tool_call>",
                prefix="[gMASK]<sop><|assistant|><think>",
                parse_tools=True,
            )

    def test_minimax_tool_calls_parse_reasoning_and_parallel_invokes(self):
        from sparseengine.entrypoints.openai.api_server import _response_output_items

        parsed = _transformers_response_parser(minimax_tools=True).parse(
            "reason</think>\n"
            "<minimax:tool_call>"
            '<invoke name="get_weather">'
            '<parameter name="city">北京</parameter>'
            "</invoke>"
            '<invoke name="get_forecast">'
            '<parameter name="days">2</parameter>'
            "</invoke>"
            "</minimax:tool_call>[e~[",
            prefix="]~b]ai\n<think>\n",
            parse_tools=True,
        )
        output = _response_output_items(parsed)

        self.assertEqual(parsed.reasoning_content, "reason")
        self.assertEqual(parsed.content, "")
        self.assertEqual(
            [item["name"] for item in output[1:]],
            ["get_weather", "get_forecast"],
        )
        self.assertEqual(output[1]["arguments"], '{"city":"北京"}')
        self.assertEqual(output[2]["arguments"], '{"days":"2"}')

    def test_minimax_response_stream_parser_handles_split_invokes(self):
        parser = _transformers_response_parser(minimax_tools=True).stream(
            prefix="]~b]ai\n<think>\n",
            parse_tools=True,
        )
        deltas = []
        for chunk in [
            "rea",
            "son</think>\n<minimax:tool_",
            'call><invoke name="get_weather"><parameter name="city">北',
            "京</parameter></invoke>",
            '<invoke name="get_forecast"><parameter name="days">2</parameter>',
            "</invoke>",
            "</minimax:tool_call>",
        ]:
            deltas.extend(parser.feed(chunk))
        deltas.extend(parser.finish())

        reasoning = "".join(
            delta.get("reasoning_content", "")
            for delta in deltas
        )
        calls = [
            delta["tool_calls"][0]
            for delta in deltas
            if "tool_calls" in delta
        ]
        self.assertEqual(reasoning, "reason")
        self.assertEqual(
            [call["function"]["name"] for call in calls],
            ["get_weather", "get_forecast"],
        )
        self.assertEqual(calls[0]["function"]["arguments"], '{"city":"北京"}')
        self.assertEqual(calls[1]["function"]["arguments"], '{"days":"2"}')
        self.assertFalse(any(delta.get("content") for delta in deltas))

    def test_malformed_tool_call_json_fails_fast(self):
        from sparseengine.entrypoints.openai.serving.response_parsing import ModelOutputParseError

        with self.assertRaises(ModelOutputParseError):
            _transformers_response_parser().parse(
                '<tool_call>{"name":</tool_call>',
                prefix="",
                parse_tools=True,
            )

    def test_transformers_response_stream_parser_handles_split_regions(self):
        parser = _transformers_response_parser().stream(prefix="", parse_tools=True)
        deltas = []
        for chunk in [
            "<thi",
            "nk>reason</think><tool_",
            'call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
        ]:
            deltas.extend(parser.feed(chunk))
        deltas.extend(parser.finish())

        reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
        calls = [delta["tool_calls"][0] for delta in deltas if "tool_calls" in delta]
        self.assertEqual(reasoning, "reason")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(calls[0]["function"]["arguments"], '{"city":"Paris"}')

    async def test_response_stream_tool_call_outputs_function_events(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "token",
                "index": 0,
                "text": '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
                "raw_text_delta": '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "token_ids": [1],
                "token_logprobs": [None],
                "top_logprobs": [None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(
                        model="model",
                        input="hello",
                        stream=True,
                        tools=[{"type": "function", "name": "get_weather", "parameters": {}}],
                    ),
                    reasoning_parser_name=None,
                    response_parser=_transformers_response_parser(),
                )
            ]
        )
        event_types = [event for event, _payload in events]
        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]

        self.assertIn("response.function_call_arguments.delta", event_types)
        self.assertIn("response.function_call_arguments.done", event_types)
        self.assertEqual(completed["output"][0]["type"], "function_call")
        self.assertEqual(completed["output"][0]["name"], "get_weather")
        self.assertEqual(completed["output"][0]["arguments"], '{"city":"Paris"}')

    async def test_response_stream_reasoning_then_tool_call(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        raw = '<think>reason</think><tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        queue = asyncio.Queue()
        for delta in ["<think>reason</think><tool_", 'call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>']:
            await queue.put(
                {
                    "type": "token",
                    "index": 0,
                    "text": "",
                    "raw_text_delta": delta,
                    "token_ids": [1],
                    "token_logprobs": [None],
                    "top_logprobs": [None],
                }
            )
        await queue.put(
            {
                "type": "final",
                "index": 0,
                "text": "",
                "raw_text": raw,
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "token_ids": [1, 2],
                "token_logprobs": [None, None],
                "top_logprobs": [None, None],
            }
        )

        class Dispatcher:
            def cancel(self, _handle):
                raise AssertionError("finished response stream should not be cancelled")

        events = _response_sse_events(
            [
                chunk
                async for chunk in _response_stream(
                    Dispatcher(),
                    "resp_test",
                    123,
                    "model",
                    RequestHandle(output_queue=queue, cancelled=threading.Event()),
                    time.perf_counter(),
                    None,
                    ResponseRequest(
                        model="model",
                        input="hello",
                        stream=True,
                        tools=[{"type": "function", "name": "get_weather", "parameters": {}}],
                    ),
                    reasoning_parser_name="qwen3",
                    response_parser=_transformers_response_parser(),
                )
            ]
        )
        completed = [payload["response"] for event, payload in events if event == "response.completed"][0]

        self.assertEqual([item["type"] for item in completed["output"]], ["reasoning", "function_call"])
        self.assertEqual(completed["output"][0]["text"], "reason")
        self.assertEqual(completed["output"][1]["arguments"], '{"city":"Paris"}')

    async def test_response_stream_cancel_releases_dispatcher_request(self):
        from sparseengine.entrypoints.openai.api_server import RequestHandle, ResponseRequest, _response_stream

        queue = asyncio.Queue()
        handle = RequestHandle(output_queue=queue, cancelled=threading.Event())
        cancelled = threading.Event()
        outer = self

        class Dispatcher:
            def cancel(self, observed_handle):
                outer.assertIs(observed_handle, handle)
                cancelled.set()

        stream = _response_stream(
            Dispatcher(),
            "resp_test",
            123,
            "model",
            handle,
            time.perf_counter(),
            None,
            ResponseRequest(model="model", input="hello", stream=True),
            reasoning_parser_name=None,
        )

        first = await stream.__anext__()
        self.assertIn("response.created", first)
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

        self.assertTrue(cancelled.is_set())

    async def test_prefix_cache_match_accepts_response_selector(self):
        from sparseengine.entrypoints.openai import api_server

        class Tokenizer:
            bos_token = None
            chat_template = "template"

            def apply_chat_template(self, chat, **_kwargs):
                return "|".join(f"{item['role']}:{item['content']}" for item in chat)

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(ch) for ch in text]

        class Engine:
            tokenizer = Tokenizer()
            config = type("Config", (), {"sparse_method": ""})()

            def prefix_cache_match(self, token_ids):
                return {"token_ids": list(token_ids), "supported": True, "enabled": True}

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/match")
        try:
            response = await endpoint(
                api_server.PrefixCacheMatchRequest(
                    response={"model": "model", "instructions": "policy", "input": "hello"}
                ),
                _TestRequest(app),
            )
        finally:
            app.state.dispatcher.close()

        self.assertEqual(
            json.loads(response.body)["token_ids"],
            [ord(ch) for ch in "system:policy|user:hello"],
        )

    async def test_prefix_cache_match_rejects_multiple_selectors_with_response(self):
        from fastapi import HTTPException
        from sparseengine.entrypoints.openai import api_server

        class Engine:
            tokenizer = object()
            config = type("Config", (), {"sparse_method": ""})()

            def exit(self):
                pass

        app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
        endpoint = _route_endpoint(app, "/v1/prefix_cache/match")
        try:
            with self.assertRaises(HTTPException):
                await endpoint(
                    api_server.PrefixCacheMatchRequest(text="hello", response={"model": "model", "input": "hello"}),
                    _TestRequest(app),
                )
        finally:
            app.state.dispatcher.close()


    async def test_nonstream_suppresses_tokens_but_preserves_final_output(self):
        tokenizer = _byte_level_tokenizer(special_tokens=("<tool_call>",))
        for text, stop in [
            ("中文🙂 café", []),
            ("answer STOP hidden", ["STOP"]),
            ("<tool_call>tool arguments", []),
            ("partial ST", ["STOP"]),
        ]:
            for incremental in (True, False):
                with self.subTest(text=text, incremental=incremental):
                    token_ids = tokenizer.encode(text)
                    streamed = await _dispatcher_items_for_token_ids(
                        tokenizer, token_ids, stop=stop, incremental=incremental,
                    )
                    final_only = await _dispatcher_items_for_token_ids(
                        tokenizer, token_ids, stop=stop, incremental=incremental,
                        stream=False,
                    )
                    self.assertTrue(any(item["type"] == "token" for item in streamed))
                    self.assertEqual([item["type"] for item in final_only], ["final"])
                    self.assertEqual(final_only[0], streamed[-1])

    async def test_input_preparation_keeps_engine_running_and_submission_order(self):
        from sparseengine.entrypoints.openai.dispatcher import AsyncEngineDispatcher
        from sparseengine.sampling_params import SamplingParams

        tokenizer = _byte_level_tokenizer()
        original_encode = tokenizer.encode
        encoding = threading.Event()
        release = threading.Event()
        stepped_during_encode = threading.Event()
        worker_names = []

        def encode(text, add_special_tokens=False):
            worker_names.append(threading.current_thread().name)
            if text == "slow":
                encoding.set()
                if not release.wait(5):
                    raise RuntimeError("test preparation timed out")
            return original_encode(text, add_special_tokens=add_special_tokens)

        tokenizer.encode = encode

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.prompts = []
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def add_request(self, prompt, _sampling):
                self.prompts.append(prompt)
                return len(self.prompts)

            def step(self):
                if encoding.is_set() and not release.is_set():
                    stepped_during_encode.set()
                time.sleep(0.001)
                return [], 0

            def abort_request(self, _seq_id):
                pass

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        sampling = SamplingParams(max_tokens=8)
        tasks = []
        try:
            await dispatcher.submit_admitted([1], sampling, 0)
            slow = asyncio.create_task(dispatcher.submit_admitted("slow", sampling, 1))
            tasks.append(slow)
            self.assertTrue(await asyncio.to_thread(encoding.wait, 2))
            following = asyncio.create_task(dispatcher.submit_admitted([2], sampling, 2))
            tasks.append(following)
            self.assertTrue(await asyncio.to_thread(stepped_during_encode.wait, 2))
            self.assertEqual(engine.prompts, [[1]])
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
            self.assertEqual(engine.prompts, [[1], original_encode("slow"), [2]])
            self.assertTrue(all(name.startswith("sparseengine-input") for name in worker_names))
            self.assertIsNot(dispatcher._input_tokenizer, engine.tokenizer)
            self.assertIsNot(dispatcher._input_tokenizer.backend_tokenizer, engine.tokenizer.backend_tokenizer)
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            dispatcher.close()

    async def test_cancel_during_input_preparation_never_admits_request(self):
        from sparseengine.entrypoints.openai.dispatcher import AsyncEngineDispatcher
        from sparseengine.sampling_params import SamplingParams

        tokenizer = _byte_level_tokenizer()
        encoding = threading.Event()
        release = threading.Event()

        def encode(_text, add_special_tokens=False):
            encoding.set()
            if not release.wait(5):
                raise RuntimeError("test preparation timed out")
            return [7]

        tokenizer.encode = encode

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.prompts = []
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def add_request(self, prompt, _sampling):
                self.prompts.append(prompt)
                return len(self.prompts)

            def step(self):
                time.sleep(0.001)
                return [], 0

            def abort_request(self, _seq_id):
                pass

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        task = asyncio.create_task(dispatcher.submit_admitted("cancel me", SamplingParams(), 0))
        try:
            self.assertTrue(await asyncio.to_thread(encoding.wait, 2))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()
            await dispatcher.submit_admitted("next", SamplingParams(), 1)
            self.assertEqual(engine.prompts, [[7]])
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            dispatcher.close()

    async def test_input_preparation_preserves_bos_and_chain_suffix_rules(self):
        from sparseengine.entrypoints.openai.dispatcher import AsyncEngineDispatcher
        from sparseengine.sampling_params import SamplingParams

        tokenizer = _byte_level_tokenizer()
        tokenizer.bos_token = "<s>"
        calls = []

        def encode(text, add_special_tokens=False):
            calls.append((text, add_special_tokens))
            if text == "invalid":
                raise ValueError("invalid test input")
            return ([1] if add_special_tokens else []) + [2]

        tokenizer.encode = encode

        class Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.prompts = []
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def admit_request(self, prompt, _sampling, **kwargs):
                self.prompts.append((prompt, kwargs))
                return type("Admission", (), dict(
                    seq_id=len(self.prompts), chain_id=kwargs.get("chain_id"),
                    chain_status="disabled", reused_tokens=0,
                    prefilled_tokens=len(prompt), prompt_token_ids=prompt,
                ))()

            def step(self):
                time.sleep(0.001)
                return [], 0

            def abort_request(self, _seq_id):
                pass

            def exit(self):
                pass

        engine = Engine()
        dispatcher = AsyncEngineDispatcher(engine)
        try:
            await dispatcher.submit_admitted("hello", SamplingParams(), 0, stream=False)
            await dispatcher.submit_admitted("<s>hello", SamplingParams(), 1)
            await dispatcher.submit_admitted("suffix", SamplingParams(), 2,
                                             chain_id="chain", chain_append_only=True)
            with self.assertRaisesRegex(ValueError, "invalid test input"):
                await dispatcher.submit_admitted("invalid", SamplingParams(), 3)
            self.assertEqual(calls, [("hello", True), ("<s>hello", False),
                                     ("suffix", False), ("invalid", True)])
            self.assertEqual([prompt for prompt, _ in engine.prompts], [[1, 2], [2], [2]])
            self.assertEqual(engine.prompts[-1][1], dict(chain_id="chain", chain_append_only=True))
        finally:
            dispatcher.close()


    async def test_all_generation_endpoints_only_publish_tokens_for_streaming(self):
        from sparseengine.entrypoints.openai import api_server

        tokenizer = _byte_level_tokenizer()
        output_ids = tokenizer.encode("ok")

        class Engine:
            config = type("Config", (), {"sparse_method": ""})()

            def __init__(self):
                self.tokenizer = tokenizer
                self.last_step_token_outputs = []
                self.last_step_logprob_outputs = []

            def add_request(self, _prompt, _sampling):
                return 1

            def step(self):
                self.last_step_token_outputs = [(1, output_ids)]
                self.last_step_logprob_outputs = [(1, [-0.5] * len(output_ids), [None] * len(output_ids))]
                return [(1, output_ids, [-0.5] * len(output_ids), [None] * len(output_ids))], len(output_ids)

            def abort_request(self, _seq_id):
                pass

            def exit(self):
                pass

        for path, request_type, kwargs in [
            ("/v1/completions", api_server.CompletionRequest, {"prompt": "hello", "max_tokens": 2}),
            ("/v1/chat/completions", api_server.ChatCompletionRequest,
             {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2}),
            ("/v1/responses", api_server.ResponseRequest, {"input": "hello", "max_output_tokens": 2}),
        ]:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    app = api_server.create_app("/tmp/model", served_model_name="model", engine=Engine())
                    dispatcher = app.state.dispatcher
                    events = []
                    original_put = dispatcher._put

                    def record_put(request, item):
                        events.append(item["type"])
                        original_put(request, item)

                    dispatcher._put = record_put
                    try:
                        response = await _route_endpoint(app, path)(
                            request_type(model="model", stream=stream, **kwargs),
                            _TestRequest(app),
                        )
                        if stream:
                            chunks = [chunk async for chunk in response.body_iterator]
                            self.assertTrue(chunks)
                            self.assertIn("token", events)
                        else:
                            self.assertEqual(response.status_code, 200)
                            self.assertEqual(events, ["final"])
                        self.assertEqual(events[-1], "final")
                    finally:
                        dispatcher.close()


class OpenAIClientTest(unittest.TestCase):
    def test_stream_client_prints_text_without_sse_frames(self):
        client_path = Path(__file__).resolve().parents[1] / "src/sparseengine/entrypoints/openai/client.py"
        spec = importlib.util.spec_from_file_location("sparseengine_openai_client_test", client_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        lines = [
            'data: {"choices": [{"text": " local"}]}\n',
            "\n",
            'data: {"choices": [{"text": "_attention"}]}\n',
            "data: [DONE]\n",
        ]

        with patch("builtins.print") as mocked_print:
            output = module.print_stream_text(lines)

        self.assertEqual(output, " local_attention")
        self.assertEqual(mocked_print.call_args_list[0].args, (" local",))
        self.assertEqual(mocked_print.call_args_list[1].args, ("_attention",))


if __name__ == "__main__":
    unittest.main()
