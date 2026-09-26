import asyncio
import importlib.util
import json
import unittest
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch


@unittest.skipIf(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None,
    "OpenAI smart router dependencies are not installed",
)
class OpenAISmartRouterTest(unittest.TestCase):
    def test_prefix_prune_controls_route_to_fullest_cache_owner(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://busy-owner", "http://idle-worker"],
            request_timeout_s=1.0, overload_load_factor=1.5,
            load_abs_threshold=1, profiles={}, route_log_dir=None,
        )
        owner, idle = router.workers
        for worker in router.workers:
            worker.healthy = True
            worker.info = {"served_model_name": "model", "sparse_method": "omnikv",
                           "prefix_cache_enabled": True, "prefix_cache_mode": "radix"}
        router.refresh_worker_info = AsyncMock(return_value=None)
        chat = {"model": "model", "messages": [{"role": "tool", "content": "result"}]}
        router._probe_workers = AsyncMock(return_value=[
            smart_router.WorkerProbe(
                worker=owner, load={"active_requests": 6},
                match={"supported": True, "enabled": True,
                       "matched_tokens": 100, "match_ratio": 1.0},
            ),
            smart_router.WorkerProbe(
                worker=idle, load={"active_requests": 0},
                match={"supported": True, "enabled": True,
                       "matched_tokens": 20, "match_ratio": 0.2},
            ),
        ])
        for endpoint, payload in (
            ("/v1/prefix_cache/match", {"chat": chat}),
            ("/v1/prefix_cache/prune", {"token_ids": list(range(100)), "ranges": [[10, 20]]}),
        ):
            selected, _, route = asyncio.run(router.select_worker(endpoint, payload))
            self.assertIs(selected, owner)
            self.assertEqual(route["reason"], "prefix_owner")
        self.assertEqual(router._probe_workers.call_args_list[0].args[1], {"chat": chat})

    def test_empty_chain_id_creates_on_chain_capable_worker(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://chain-worker", "http://radix-worker"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        chain_worker, radix_worker = router.workers
        chain_worker.info = {
            "served_model_name": "model",
            "sparse_method": "snapkv",
            "prefix_cache_enabled": True,
            "prefix_cache_mode": "chain",
        }
        radix_worker.info = {
            "served_model_name": "model",
            "sparse_method": "omnikv",
            "prefix_cache_enabled": True,
            "prefix_cache_mode": "radix",
        }
        router.refresh_worker_info = AsyncMock(return_value=None)

        async def probe_workers(candidates, match_payload):
            self.assertEqual(candidates, [chain_worker])
            self.assertEqual(match_payload, {"text": "hello"})
            return [
                smart_router.WorkerProbe(
                    worker=chain_worker,
                    load={"active_requests": 0},
                    match={
                        "supported": False,
                        "enabled": False,
                        "matched_tokens": 0,
                        "match_ratio": 0.0,
                    },
                )
            ]

        router._probe_workers = probe_workers

        worker, payload, route = asyncio.run(
            router.select_worker(
                "/v1/completions",
                {
                    "model": "model",
                    "prompt": "hello",
                    "chain_id": "",
                },
            )
        )

        self.assertIs(worker, chain_worker)
        self.assertEqual(payload["chain_id"], "")
        self.assertEqual(route["reason"], "chain_create")
        self.assertTrue(route["chain_create"])
        self.assertEqual(
            route["chain_selection_reason"],
            "lowest_load_no_prefix_match",
        )

    def test_chain_affinity_requires_one_idle_owner(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a", "http://worker-b"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )

        def post_json(url, _payload, _timeout):
            if "worker-a" in url:
                return {
                    "present": True,
                    "state": "idle",
                    "chain_id": "chain-1",
                }
            return {"present": False, "tombstone": False}

        with patch.object(smart_router, "_post_json", side_effect=post_json):
            owner, matches = asyncio.run(
                router._select_chain_owner(router.workers, "chain-1")
            )
        self.assertEqual(owner.url, "http://worker-a")
        self.assertEqual(len(matches), 2)

        with patch.object(
            smart_router,
            "_post_json",
            return_value={
                "present": True,
                "state": "idle",
                "chain_id": "chain-1",
            },
        ):
            with self.assertRaises(HTTPException) as duplicate:
                asyncio.run(router._select_chain_owner(router.workers, "chain-1"))
        self.assertEqual(duplicate.exception.status_code, 500)

        with patch.object(
            smart_router,
            "_post_json",
            return_value={
                "present": True,
                "state": "active",
                "chain_id": "chain-1",
            },
        ):
            with self.assertRaises(HTTPException) as busy:
                asyncio.run(router._select_chain_owner(router.workers[:1], "chain-1"))
        self.assertEqual(busy.exception.status_code, 409)

    def test_chain_affinity_distinguishes_unknown_and_gone(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        with patch.object(
            smart_router,
            "_post_json",
            return_value={"present": False, "tombstone": False},
        ):
            with self.assertRaises(HTTPException) as unknown:
                asyncio.run(router._select_chain_owner(router.workers, "chain-1"))
        self.assertEqual(unknown.exception.status_code, 404)

        with patch.object(
            smart_router,
            "_post_json",
            return_value={"present": False, "tombstone": True},
        ):
            with self.assertRaises(HTTPException) as gone:
                asyncio.run(router._select_chain_owner(router.workers, "chain-1"))
        self.assertEqual(gone.exception.status_code, 410)
        self.assertEqual(gone.exception.detail["code"], "chain_gone")
        self.assertEqual(gone.exception.detail["chain_id"], "chain-1")
        self.assertIn("was evicted", gone.exception.detail["message"])

    def test_chain_affinity_fails_closed_on_probe_error_or_disabled_workers(self):
        from fastapi import HTTPException

        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a", "http://worker-b"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )

        def partial_failure(url, _payload, _timeout):
            if "worker-a" in url:
                return {"enabled": True, "present": True, "state": "idle"}
            raise RuntimeError("probe failed")

        with patch.object(
            smart_router,
            "_post_json",
            side_effect=partial_failure,
        ):
            with self.assertRaises(HTTPException) as failed:
                asyncio.run(
                    router._select_chain_owner(router.workers, "chain-1")
                )
        self.assertEqual(failed.exception.status_code, 503)

        with patch.object(
            smart_router,
            "_post_json",
            return_value={
                "enabled": False,
                "present": False,
                "tombstone": False,
            },
        ):
            with self.assertRaises(HTTPException) as disabled:
                asyncio.run(
                    router._select_chain_owner(router.workers, "chain-1")
                )
        self.assertEqual(disabled.exception.status_code, 400)

    def test_router_preserves_upstream_chain_header(self):
        from sparseengine.entrypoints.openai.smart_router import _content_headers

        self.assertEqual(
            _content_headers(
                {
                    "content-type": "text/event-stream",
                    "x-sparseengine-chain-id": "chain-1",
                }
            ),
            {
                "Content-Type": "text/event-stream",
                "X-SparseVLLM-Chain-ID": "chain-1",
            },
        )

    def test_route_headers_preserve_recreated_upstream_chain_id(self):
        from starlette.responses import Response

        from sparseengine.entrypoints.openai.smart_router import (
            _route_response_headers,
            _with_route_headers,
        )

        route = {
            "selected_worker_url": "http://worker-a",
            "reason": "chain_affinity",
            "selected_sparse_method": "snapkv",
            "chain_id": "old-chain",
        }
        upstream = {
            "content-type": "application/json",
            "x-sparseengine-chain-id": "new-chain",
        }

        self.assertEqual(
            _route_response_headers(upstream, route)[
                "X-SparseVLLM-Chain-ID"
            ],
            "new-chain",
        )
        response = Response(
            content=b"{}",
            headers={"X-SparseVLLM-Chain-ID": "new-chain"},
        )
        self.assertEqual(
            _with_route_headers(response, route).headers[
                "X-SparseVLLM-Chain-ID"
            ],
            "new-chain",
        )

    def test_worker_readiness_failure_is_removed_and_recovery_is_detected(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        ready = False
        urls = []
        timeouts = []

        def get_json(url, timeout):
            urls.append(url)
            timeouts.append(timeout)
            if not ready:
                raise RuntimeError("worker not ready")
            return {"served_model_name": "model", "max_model_len": 262144}

        with patch.object(smart_router, "_get_json", side_effect=get_json):
            asyncio.run(router.refresh_worker_info())
            self.assertFalse(router.workers[0].healthy)
            ready = True
            asyncio.run(router.refresh_worker_info())

        self.assertTrue(router.workers[0].healthy)
        self.assertEqual(router.workers[0].info["served_model_name"], "model")
        self.assertEqual(urls, ["http://worker-a/v1/worker/info"] * 2)
        self.assertEqual(timeouts, [5.0, 5.0])

    def test_router_health_returns_503_without_ready_workers(self):
        from sparseengine.entrypoints.openai import smart_router

        app = smart_router.create_app(["http://worker-a"])
        app.state.router.workers[0].healthy = False
        app.state.router.refresh_worker_info = AsyncMock(return_value=None)
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/health")

        response = asyncio.run(endpoint())

        self.assertEqual(response.status_code, 503)
        policy = json.loads(response.body)["router_policy"]
        revision = policy.pop("code_revision")
        self.assertEqual(
            policy,
            {
                "request_timeout_s": 30.0,
                "control_timeout_s": 5.0,
                "overload_load_factor": 1.5,
                "load_abs_threshold": 1,
                "profiles": {},
            },
        )
        self.assertTrue(revision["git_commit"] or revision["package_version"])

    def test_router_livez_stays_available_without_ready_workers(self):
        from sparseengine.entrypoints.openai import smart_router

        app = smart_router.create_app(["http://worker-a"])
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/livez")

        response = asyncio.run(endpoint())

        self.assertEqual(response.status_code, 200)

    def test_model_cards_use_smallest_worker_context(self):
        from sparseengine.entrypoints.openai.smart_router import _router_model_cards
        from sparseengine.entrypoints.openai.smart_router import WorkerState

        cards = _router_model_cards(
            [
                WorkerState(url="http://a", info={"served_model_name": "model", "max_model_len": 128000}),
                WorkerState(url="http://b", info={"served_model_name": "model", "max_model_len": 64000}),
                WorkerState(
                    url="http://unhealthy",
                    info={"served_model_name": "model", "max_model_len": 32000},
                    healthy=False,
                ),
            ],
            123,
        )

        self.assertEqual(cards[0]["max_model_len"], 64000)

    def test_choose_worker_prefers_prefix_match_when_load_is_close(self):
        from sparseengine.entrypoints.openai.smart_router import WorkerProbe, WorkerState, choose_worker

        cache_worker = WorkerState(url="http://worker-a", info={"sparse_method": "omnikv"})
        load_worker = WorkerState(url="http://worker-b", info={"sparse_method": "omnikv"})
        worker, reason = choose_worker(
            [
                WorkerProbe(
                    worker=cache_worker,
                    load={"active_requests": 1},
                    match={"supported": True, "enabled": True, "matched_tokens": 128, "match_ratio": 0.75},
                ),
                WorkerProbe(
                    worker=load_worker,
                    load={"active_requests": 0},
                    match={"supported": True, "enabled": True, "matched_tokens": 0, "match_ratio": 0.0},
                ),
            ],
            overload_load_factor=1.5,
            load_abs_threshold=1,
        )

        self.assertIs(worker, cache_worker)
        self.assertEqual(reason, "best_prefix_match")

    def test_route_headers_report_selected_probe_matched_tokens(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a", "http://worker-b"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        selected = router.workers[0]
        selected.info = {"sparse_method": "omnikv"}
        other = router.workers[1]
        other.info = {"sparse_method": "omnikv"}
        route = router._route_record(
            "/v1/completions",
            selected,
            "best_prefix_match",
            [
                smart_router.WorkerProbe(
                    worker=selected,
                    load={"active_requests": 1},
                    match={
                        "supported": True,
                        "enabled": True,
                        "matched_tokens": 128,
                        "match_ratio": 0.5,
                    },
                ),
                smart_router.WorkerProbe(
                    worker=other,
                    load={"active_requests": 0},
                    match={
                        "supported": True,
                        "enabled": True,
                        "matched_tokens": 0,
                        "match_ratio": 0.0,
                    },
                ),
            ],
            {},
        )

        self.assertEqual(route["selected_prefix_matched_tokens"], 128)
        self.assertEqual(
            smart_router.route_headers(route)[
                "X-SparseVLLM-Prefix-Matched-Tokens"
            ],
            "128",
        )

    def test_target_worker_route_does_not_claim_unprobed_prefix_match(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        route = router._route_record(
            "/v1/completions",
            router.workers[0],
            "target_worker",
            [],
            {"target_worker": "0"},
        )

        self.assertIsNone(route["selected_prefix_matched_tokens"])
        self.assertNotIn(
            "X-SparseVLLM-Prefix-Matched-Tokens",
            smart_router.route_headers(route),
        )

    def test_probe_uses_nonblocking_prefix_routing_snapshot(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=900.0,
            control_timeout_s=5.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        worker = router.workers[0]
        with (
            patch.object(
                smart_router,
                "_get_json",
                return_value={"active_requests": 1, "snapshot": True},
            ) as get_json,
            patch.object(
                smart_router,
                "_post_json",
                return_value={
                    "supported": True,
                    "enabled": True,
                    "matched_tokens": 16,
                    "match_ratio": 1.0,
                    "snapshot": True,
                },
            ) as post_json,
        ):
            probe = asyncio.run(
                router._probe_worker(
                    worker,
                    {"token_ids": [1, 2, 3]},
                )
            )

        get_json.assert_called_once_with(
            "http://worker-a/v1/worker/routing_load",
            5.0,
        )
        post_json.assert_called_once_with(
            "http://worker-a/v1/prefix_cache/routing_match",
            {"token_ids": [1, 2, 3]},
            5.0,
        )
        self.assertEqual(probe.matched_tokens, 16)

    def test_choose_worker_falls_back_to_lowest_load_when_match_worker_is_overloaded(self):
        from sparseengine.entrypoints.openai.smart_router import WorkerProbe, WorkerState, choose_worker

        cache_worker = WorkerState(url="http://worker-a", info={"sparse_method": "omnikv"})
        load_worker = WorkerState(url="http://worker-b", info={"sparse_method": "omnikv"})
        worker, reason = choose_worker(
            [
                WorkerProbe(
                    worker=cache_worker,
                    load={"active_requests": 10},
                    match={"supported": True, "enabled": True, "matched_tokens": 128, "match_ratio": 0.75},
                ),
                WorkerProbe(
                    worker=load_worker,
                    load={"active_requests": 0},
                    match={"supported": True, "enabled": True, "matched_tokens": 0, "match_ratio": 0.0},
                ),
            ],
            overload_load_factor=1.5,
            load_abs_threshold=1,
        )

        self.assertIs(worker, load_worker)
        self.assertEqual(reason, "prefix_match_overloaded_lowest_load")

    def test_route_profiles_filter_heterogeneous_workers_by_sparse_method(self):
        from sparseengine.entrypoints.openai.smart_router import SmartRouter

        router = SmartRouter(
            worker_urls=["http://omni", "http://snap"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={
                "conversation": {"methods": ["omnikv"]},
                "bulk": {"methods": ["snapkv"]},
            },
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv", "tags": ["dialog"]}
        router.workers[1].info = {"served_model_name": "model", "sparse_method": "snapkv", "tags": ["bulk"]}

        conversation = router._candidate_workers(
            "/v1/chat/completions",
            {"model": "model", "messages": [{"role": "user", "content": "x"}]},
            {"route_profile": "conversation"},
        )
        bulk = router._candidate_workers(
            "/v1/completions",
            {"model": "model", "prompt": "x"},
            {"route_profile": "bulk"},
        )

        self.assertEqual([worker.url for worker in conversation], ["http://omni"])
        self.assertEqual([worker.url for worker in bulk], ["http://snap"])

    def test_route_profiles_treat_empty_method_as_vanilla(self):
        from sparseengine.entrypoints.openai.smart_router import SmartRouter

        router = SmartRouter(
            worker_urls=["http://vanilla", "http://snap"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={"default": {"methods": ["vanilla"]}},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "", "tags": []}
        router.workers[1].info = {"served_model_name": "model", "sparse_method": "snapkv", "tags": []}

        candidates = router._candidate_workers(
            "/v1/completions",
            {"model": "model", "prompt": "x"},
            {},
        )

        self.assertEqual([worker.url for worker in candidates], ["http://vanilla"])

    def test_route_hints_are_stripped_before_forwarding(self):
        from sparseengine.entrypoints.openai.smart_router import strip_route_hints

        payload, hints = strip_route_hints(
            {
                "model": "model",
                "prompt": "hello",
                "sengine_route_profile": "bulk",
                "sengine_method_preference": ["snapkv"],
            }
        )

        self.assertEqual(payload, {"model": "model", "prompt": "hello"})
        self.assertEqual(hints["route_profile"], "bulk")
        self.assertEqual(hints["method_preference"], ["snapkv"])

    def test_match_payload_for_chat_and_completion_requests(self):
        from sparseengine.entrypoints.openai.smart_router import match_payload_for_request

        chat_payload = {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "search", "parameters": {}},
                }
            ],
            "reasoning_effort": "none",
        }
        self.assertEqual(
            match_payload_for_request(
                "/v1/chat/completions",
                chat_payload,
            ),
            {"chat": chat_payload},
        )
        self.assertEqual(
            match_payload_for_request("/v1/completions", {"prompt": [1, 2, 3]}),
            {"token_ids": [1, 2, 3]},
        )
        self.assertEqual(
            match_payload_for_request("/v1/completions", {"prompt": ["a", "b"]}),
            {"text": "a"},
        )
        response_payload = {"model": "model", "input": "hello", "reasoning": {"effort": "none"}}
        self.assertEqual(
            match_payload_for_request("/v1/responses", response_payload),
            {"response": response_payload},
        )

    def test_responses_route_profile_inference_stays_default(self):
        from sparseengine.entrypoints.openai.smart_router import infer_route_profile

        self.assertEqual(
            infer_route_profile(
                "/v1/responses",
                {"input": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
            ),
            "default",
        )

    def test_responses_route_forwards_through_router(self):
        from sparseengine.entrypoints.openai import smart_router

        class JsonRequest:
            async def json(self):
                return {"model": "model", "input": "hello"}

            async def is_disconnected(self):
                return False

        app = smart_router.create_app(["http://worker-a"])
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/responses")
        app.state.router.route_openai_request = AsyncMock(return_value="ok")

        request = JsonRequest()
        response = asyncio.run(endpoint(request))

        self.assertEqual(response, "ok")
        app.state.router.route_openai_request.assert_awaited_once_with(
            "/v1/responses",
            {"model": "model", "input": "hello"},
            is_disconnected=request.is_disconnected,
        )

    def test_responses_route_hints_are_stripped_before_forwarding(self):
        from fastapi.responses import Response
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv"}

        async def refresh_worker_info():
            return None

        router.refresh_worker_info = refresh_worker_info

        async def forward_json(
            _worker,
            _endpoint,
            payload,
            *,
            is_disconnected=None,
        ):
            self.assertEqual(payload, {"model": "model", "input": "hello"})
            self.assertIsNone(is_disconnected)
            return Response(content=b"{}", media_type="application/json")

        router.forward_json = forward_json
        response = asyncio.run(
            router.route_openai_request(
                "/v1/responses",
                {
                    "model": "model",
                    "input": "hello",
                    "sengine_target_worker": "0",
                    "sengine_method_preference": ["omnikv"],
                },
            )
        )

        self.assertEqual(response.headers["x-sparseengine-worker"], "http://worker-a")
        self.assertEqual(response.headers["x-sparseengine-route-reason"], "target_worker")

    def test_streaming_upstream_http_error_is_returned_before_sse_response(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv"}

        async def refresh_worker_info():
            return None

        router.refresh_worker_info = refresh_worker_info
        upstream_error = smart_router.UpstreamError(
            status=400,
            headers={"Content-Type": "application/json"},
            body=b'{"error":"bad request"}',
        )

        with patch.object(smart_router, "_open_stream_response", return_value=upstream_error):
            response = asyncio.run(
                router.route_openai_request(
                    "/v1/completions",
                    {
                        "model": "model",
                        "prompt": "hello",
                        "stream": True,
                        "sengine_target_worker": "0",
                    },
                )
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.body, b'{"error":"bad request"}')
        self.assertEqual(response.headers["x-sparseengine-worker"], "http://worker-a")
        self.assertEqual(response.headers["x-sparseengine-route-reason"], "target_worker")
        self.assertEqual(router.workers[0].local_inflight, 0)

    def test_responses_streaming_transparently_forwards_sse_bytes(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv"}

        async def refresh_worker_info():
            return None

        class Upstream:
            def __init__(self):
                self.chunks = [
                    b"event: response.created\n",
                    b'data: {"type":"response.created"}\n\n',
                    b"data: [DONE]\n\n",
                ]
                self.closed = False

            def read(self, _size):
                if not self.chunks:
                    return b""
                return self.chunks.pop(0)

            def close(self):
                self.closed = True

        upstream_response = Upstream()
        router.refresh_worker_info = refresh_worker_info

        async def run_request():
            with patch.object(
                smart_router,
                "_open_stream_response",
                return_value=smart_router.UpstreamStream(
                    response=upstream_response,
                    headers={"Content-Type": "text/event-stream"},
                ),
            ):
                response = await router.route_openai_request(
                    "/v1/responses",
                    {
                        "model": "model",
                        "input": "hello",
                        "stream": True,
                        "sengine_target_worker": "0",
                    },
                )
                chunks = [chunk async for chunk in response.body_iterator]
                return response, chunks

        response, chunks = asyncio.run(run_request())

        self.assertEqual(b"".join(chunks), b'event: response.created\ndata: {"type":"response.created"}\n\ndata: [DONE]\n\n')
        self.assertEqual(response.headers["x-sparseengine-worker"], "http://worker-a")
        self.assertEqual(response.headers["x-sparseengine-route-reason"], "target_worker")
        self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertTrue(upstream_response.closed)
        self.assertEqual(router.workers[0].local_inflight, 0)

    def test_streaming_client_disconnect_closes_upstream(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {
            "served_model_name": "model",
            "sparse_method": "omnikv",
        }

        async def refresh_worker_info():
            return None

        class Upstream:
            def __init__(self):
                self.closed = False

            def read(self, _size):
                raise AssertionError("disconnect must be checked before blocking on upstream")

            def close(self):
                self.closed = True

        async def is_disconnected():
            return True

        upstream_response = Upstream()
        router.refresh_worker_info = refresh_worker_info

        async def run_request():
            with patch.object(
                smart_router,
                "_open_stream_response",
                return_value=smart_router.UpstreamStream(
                    response=upstream_response,
                    headers={"Content-Type": "text/event-stream"},
                ),
            ):
                response = await router.route_openai_request(
                    "/v1/responses",
                    {
                        "model": "model",
                        "input": "hello",
                        "stream": True,
                        "sengine_target_worker": "0",
                    },
                    is_disconnected=is_disconnected,
                )
                await response.body_iterator.__anext__()

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run_request())

        self.assertTrue(upstream_response.closed)
        self.assertEqual(router.workers[0].local_inflight, 0)

    def test_non_streaming_client_disconnect_closes_upstream(self):
        import threading

        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        operation_started = threading.Event()
        operation_closed = threading.Event()

        class BlockingRequest:
            def execute(self):
                operation_started.set()
                if not operation_closed.wait(timeout=1.0):
                    raise AssertionError("upstream request was not closed")
                raise RuntimeError("upstream request closed")

            def close(self):
                operation_closed.set()

        async def is_disconnected():
            return operation_started.is_set()

        with patch.object(
            smart_router,
            "_CloseableByteRequest",
            return_value=BlockingRequest(),
        ):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(
                    router.forward_json(
                        router.workers[0],
                        "/v1/responses",
                        {"model": "model", "input": "hello"},
                        is_disconnected=is_disconnected,
                    )
                )

        self.assertTrue(operation_closed.is_set())
        self.assertEqual(router.workers[0].local_inflight, 0)

    def test_non_streaming_forward_returns_upstream_response(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        operation = Mock()
        operation.execute.return_value = (
            201,
            {"Content-Type": "application/json", "X-Upstream": "ignored"},
            b'{"ok":true}',
        )

        with patch.object(
            smart_router,
            "_CloseableByteRequest",
            return_value=operation,
        ):
            response = asyncio.run(
                router.forward_json(
                    router.workers[0],
                    "/v1/responses",
                    {"model": "model", "input": "hello"},
                )
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.body, b'{"ok":true}')
        self.assertEqual(response.headers["content-type"], "application/json")
        operation.close.assert_not_called()
        self.assertEqual(router.workers[0].local_inflight, 0)

    def test_closeable_request_interrupts_wait_for_response_headers(self):
        import socket
        import threading

        from sparseengine.entrypoints.openai import smart_router

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("127.0.0.1", 0))
        server_socket.listen(1)
        request_received = threading.Event()
        release_server = threading.Event()
        client_errors = []

        def serve_delayed_headers():
            connection, _ = server_socket.accept()
            with connection:
                connection.recv(8192)
                request_received.set()
                release_server.wait(timeout=2.0)
                try:
                    connection.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"
                    )
                except OSError:
                    pass

        server_thread = threading.Thread(target=serve_delayed_headers)
        server_thread.start()
        port = server_socket.getsockname()[1]
        operation = smart_router._CloseableByteRequest(
            f"http://127.0.0.1:{port}/v1/responses",
            "POST",
            {"model": "model", "input": "hello"},
            2.0,
        )

        def run_request():
            try:
                operation.execute()
            except BaseException as exc:
                client_errors.append(exc)

        client_thread = threading.Thread(target=run_request)
        client_thread.start()
        try:
            self.assertTrue(request_received.wait(timeout=1.0))
            operation.close()
            client_thread.join(timeout=0.5)
            self.assertFalse(
                client_thread.is_alive(),
                "closing the request did not interrupt response-header wait",
            )
        finally:
            release_server.set()
            server_thread.join(timeout=1.0)
            client_thread.join(timeout=1.0)
            server_socket.close()

        self.assertEqual(len(client_errors), 1)
        self.assertIsInstance(client_errors[0], RuntimeError)

    def test_streaming_open_exception_releases_local_inflight(self):
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        router.workers[0].info = {"served_model_name": "model", "sparse_method": "omnikv"}

        async def refresh_worker_info():
            return None

        router.refresh_worker_info = refresh_worker_info
        with patch.object(smart_router, "_open_stream_response", side_effect=ValueError("boom")):
            with self.assertRaisesRegex(ValueError, "boom"):
                asyncio.run(
                    router.route_openai_request(
                        "/v1/completions",
                        {
                            "model": "model",
                            "prompt": "hello",
                            "stream": True,
                            "sengine_target_worker": "0",
                        },
                    )
                )

        self.assertEqual(router.workers[0].local_inflight, 0)


if __name__ == "__main__":
    unittest.main()
