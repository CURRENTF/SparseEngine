import argparse
import asyncio
import http.client
import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.responses import StreamingResponse

from sparseengine.engine.chain_cache import ChainGoneError
from sparseengine.entrypoints.openai.serving.base import DisconnectChecker
from sparseengine.entrypoints.openai.serving.base import _chain_http_exception
from sparseengine.entrypoints.openai.serving.base import _wait_for_any_or_disconnect
from sparseengine.utils.code_revision import code_revision_info
from sparseengine.utils.log import logger


ROUTE_FIELDS = {
    "sengine_route_profile",
    "sengine_method_preference",
    "sengine_required_tags",
    "sengine_preferred_tags",
    "sengine_target_worker",
}


@dataclass
class WorkerState:
    url: str
    info: dict[str, Any]
    local_inflight: int = 0
    healthy: bool = True
    last_error: str | None = None


@dataclass(frozen=True)
class WorkerProbe:
    worker: WorkerState
    load: dict[str, Any]
    match: dict[str, Any]

    @property
    def load_value(self) -> int:
        return int(self.load.get("active_requests", 0) or 0) + int(self.worker.local_inflight)

    @property
    def match_ratio(self) -> float:
        if not (self.match.get("supported") and self.match.get("enabled")):
            return 0.0
        return float(self.match.get("match_ratio", 0.0) or 0.0)

    @property
    def matched_tokens(self) -> int:
        if not (self.match.get("supported") and self.match.get("enabled")):
            return 0
        return int(self.match.get("matched_tokens", 0) or 0)


@dataclass(frozen=True)
class UpstreamStream:
    response: Any
    headers: dict[str, str]


@dataclass(frozen=True)
class UpstreamError:
    status: int
    headers: dict[str, str]
    body: bytes


def create_app(
    worker_urls: list[str],
    *,
    request_timeout_s: float = 30.0,
    control_timeout_s: float = 5.0,
    overload_load_factor: float = 1.5,
    load_abs_threshold: int = 1,
    profiles: dict[str, Any] | None = None,
    route_log_dir: str | None = None,
) -> FastAPI:
    router = SmartRouter(
        worker_urls=worker_urls,
        request_timeout_s=request_timeout_s,
        control_timeout_s=control_timeout_s,
        overload_load_factor=overload_load_factor,
        load_abs_threshold=load_abs_threshold,
        profiles=profiles or {},
        route_log_dir=Path(route_log_dir) if route_log_dir else None,
    )
    app = FastAPI(title="SparseEngine OpenAI smart router")
    app.state.router = router

    @app.on_event("startup")
    async def _startup():
        await router.refresh_worker_info()

    async def _readiness_response():
        await router.refresh_worker_info()
        healthy = [worker.url for worker in router.workers if worker.healthy]
        return JSONResponse(
            {
                "status": "ok" if healthy else "unavailable",
                "healthy_workers": healthy,
                "router_policy": router.policy_info(),
            },
            status_code=200 if healthy else 503,
        )

    @app.get("/health")
    async def health():
        return await _readiness_response()

    @app.get("/readyz")
    async def readyz():
        return await _readiness_response()

    @app.get("/livez")
    async def livez():
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def models():
        await router.refresh_worker_info()
        created = int(time.time())
        return {
            "object": "list",
            "data": _router_model_cards(router.workers, created),
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await router.route_openai_request(
            "/v1/completions",
            await request.json(),
            is_disconnected=request.is_disconnected,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await router.route_openai_request(
            "/v1/chat/completions",
            await request.json(),
            is_disconnected=request.is_disconnected,
        )

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await router.route_openai_request(
            "/v1/responses",
            await request.json(),
            is_disconnected=request.is_disconnected,
        )

    @app.post("/v1/prefix_cache/inspect")
    async def prefix_cache_inspect(request: Request):
        payload = await request.json()
        all_workers = bool(payload.pop("all", False))
        if all_workers:
            return JSONResponse(await router.broadcast_json("/v1/prefix_cache/inspect", payload))
        worker, payload, route = await router.select_worker("/v1/prefix_cache/inspect", payload)
        response = await router.forward_json(worker, "/v1/prefix_cache/inspect", payload)
        return _with_route_headers(response, route)

    @app.post("/v1/prefix_cache/match")
    async def prefix_cache_match(request: Request):
        worker, payload, route = await router.select_worker(
            "/v1/prefix_cache/match", await request.json()
        )
        response = await router.forward_json(worker, "/v1/prefix_cache/match", payload)
        return _with_route_headers(response, route)

    @app.post("/v1/prefix_cache/delete_subtree")
    async def prefix_cache_delete_subtree(request: Request):
        return JSONResponse(await router.broadcast_json("/v1/prefix_cache/delete_subtree", await request.json()))

    @app.post("/v1/prefix_cache/set_eviction_priority")
    async def prefix_cache_set_eviction_priority(request: Request):
        return JSONResponse(await router.broadcast_json("/v1/prefix_cache/set_eviction_priority", await request.json()))

    @app.post("/v1/prefix_cache/prune")
    async def prefix_cache_prune(request: Request):
        payload = await request.json()
        worker, payload, route = await router.select_worker(
            "/v1/prefix_cache/prune", payload
        )
        response = await router.forward_json(
            worker, "/v1/prefix_cache/prune", payload
        )
        if response.status_code < 300:
            decoded = json.loads(bytes(response.body).decode("utf-8"))
            prune_id = str(decoded.get("prune_id") or "")
            if not prune_id:
                raise HTTPException(
                    status_code=502,
                    detail="Prefix prune worker returned no prune_id.",
                )
            router.prefix_prune_workers[prune_id] = worker
        return _with_route_headers(response, route)

    @app.get("/v1/prefix_cache/prune/{prune_id}")
    async def prefix_cache_prune_status(prune_id: str):
        worker = router.prefix_prune_workers.get(str(prune_id))
        if worker is None:
            raise HTTPException(status_code=404, detail="Unknown prefix prune id.")
        status, headers, body = await asyncio.to_thread(
            _request_bytes,
            f"{worker.url}/v1/prefix_cache/prune/{prune_id}",
            "GET",
            None,
            router.control_timeout_s,
        )
        return Response(
            content=body,
            status_code=status,
            headers=_content_headers(headers),
        )

    return app


def _router_model_cards(workers: list[WorkerState], created: int) -> list[dict[str, Any]]:
    model_workers: dict[str, list[WorkerState]] = {}
    for worker in workers:
        model_id = worker.info.get("served_model_name")
        if worker.healthy and model_id:
            model_workers.setdefault(str(model_id), []).append(worker)

    cards = []
    for model_id, serving_workers in sorted(model_workers.items()):
        card = {
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": "sparseengine-router",
        }
        max_model_lens = [int(worker.info.get("max_model_len", 0) or 0) for worker in serving_workers]
        if all(max_model_len > 0 for max_model_len in max_model_lens):
            card["max_model_len"] = min(max_model_lens)
        cards.append(card)
    return cards


class SmartRouter:
    def __init__(
        self,
        *,
        worker_urls: list[str],
        request_timeout_s: float,
        control_timeout_s: float = 5.0,
        overload_load_factor: float,
        load_abs_threshold: int,
        profiles: dict[str, Any],
        route_log_dir: Path | None,
    ):
        if not worker_urls:
            raise ValueError("At least one worker URL is required.")
        self.workers = [
            WorkerState(url=url.rstrip("/"), info={})
            for url in worker_urls
        ]
        self.request_timeout_s = float(request_timeout_s)
        self.control_timeout_s = float(control_timeout_s)
        if self.request_timeout_s <= 0 or self.control_timeout_s <= 0:
            raise ValueError("Router request and control timeouts must be positive.")
        self.overload_load_factor = float(overload_load_factor)
        self.load_abs_threshold = int(load_abs_threshold)
        self.profiles = profiles
        self.route_log_dir = route_log_dir
        self.prefix_prune_workers: dict[str, WorkerState] = {}
        if self.route_log_dir is not None:
            self.route_log_dir.mkdir(parents=True, exist_ok=True)

    def policy_info(self) -> dict[str, Any]:
        return {
            "request_timeout_s": self.request_timeout_s,
            "control_timeout_s": self.control_timeout_s,
            "overload_load_factor": self.overload_load_factor,
            "load_abs_threshold": self.load_abs_threshold,
            "profiles": self.profiles,
            "code_revision": code_revision_info(),
        }

    async def refresh_worker_info(self):
        results = await asyncio.gather(
            *[
                asyncio.to_thread(
                    _get_json,
                    f"{worker.url}/v1/worker/info",
                    self.control_timeout_s,
                )
                for worker in self.workers
            ],
            return_exceptions=True,
        )
        for worker, result in zip(self.workers, results):
            if isinstance(result, Exception):
                worker.healthy = False
                worker.last_error = f"{type(result).__name__}: {result}"
                continue
            worker.info = dict(result)
            worker.healthy = True
            worker.last_error = None

    async def route_openai_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        is_disconnected: DisconnectChecker | None = None,
    ) -> Response:
        worker, forward_payload, route = await self.select_worker(endpoint, payload)
        stream = bool(forward_payload.get("stream", False))
        if stream:
            return await self.forward_stream(
                worker,
                endpoint,
                forward_payload,
                route,
                is_disconnected=is_disconnected,
            )
        response = await self.forward_json(
            worker,
            endpoint,
            forward_payload,
            is_disconnected=is_disconnected,
        )
        return _with_route_headers(response, route)

    async def select_worker(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> tuple[WorkerState, dict[str, Any], dict[str, Any]]:
        await self.refresh_worker_info()
        forward_payload, route_hints = strip_route_hints(payload)
        chain_requested = "chain_id" in forward_payload
        chain_id = str(forward_payload.get("chain_id") or "").strip()
        if chain_id:
            requested_model = forward_payload.get("model")
            candidates = [
                worker
                for worker in self.workers
                if worker.healthy
                and (
                    requested_model is None
                    or worker.info.get("served_model_name") is None
                    or str(worker.info.get("served_model_name"))
                    == str(requested_model)
                )
            ]
            if not candidates:
                raise HTTPException(
                    status_code=503,
                    detail="No healthy SparseEngine worker matches this chain request.",
                )
            worker, chain_matches = await self._select_chain_owner(
                candidates, chain_id
            )
            route = self._route_record(
                endpoint,
                worker,
                "chain_affinity",
                [],
                route_hints,
            )
            route["chain_id"] = chain_id
            route["chain_affinity"] = True
            route["chain_probes"] = chain_matches
            self._write_route_log(route)
            return worker, forward_payload, route
        candidates = self._candidate_workers(
            endpoint, forward_payload, route_hints
        )
        if chain_requested:
            candidates = [
                worker
                for worker in candidates
                if bool(worker.info.get("prefix_cache_enabled"))
                and str(worker.info.get("prefix_cache_mode") or "")
                == "chain"
            ]
            if not candidates:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "No healthy chain-capable SparseEngine worker matches "
                        "this new-chain request."
                    ),
                )
        if not candidates:
            raise HTTPException(status_code=503, detail="No healthy SparseEngine worker matches this request.")
        if route_hints.get("target_worker"):
            target = str(route_hints["target_worker"])
            for worker in candidates:
                if worker.url == target or worker.url.endswith(target) or str(self.workers.index(worker)) == target:
                    return worker, forward_payload, self._route_record(endpoint, worker, "target_worker", [], route_hints)
            raise HTTPException(status_code=404, detail=f"Requested target worker {target!r} is not available.")

        match_payload = match_payload_for_request(endpoint, forward_payload)
        probes = await self._probe_workers(candidates, match_payload)
        if not probes:
            raise HTTPException(status_code=503, detail="No healthy SparseEngine worker responded to route probes.")
        if endpoint in {"/v1/prefix_cache/match", "/v1/prefix_cache/prune"}:
            # Control operations must address the cache owner even when busy.
            owner = min(probes, key=lambda probe: (
                -probe.matched_tokens, probe.load_value, probe.worker.url,
            ))
            worker, reason = owner.worker, "prefix_owner"
        else:
            worker, reason = choose_worker(
                probes,
                overload_load_factor=self.overload_load_factor,
                load_abs_threshold=self.load_abs_threshold,
            )
        route = self._route_record(
            endpoint,
            worker,
            "chain_create" if chain_requested else reason,
            probes,
            route_hints,
        )
        if chain_requested:
            route["chain_create"] = True
            route["chain_selection_reason"] = reason
        self._write_route_log(route)
        return worker, forward_payload, route

    async def _select_chain_owner(
        self,
        candidates: list[WorkerState],
        chain_id: str,
    ) -> tuple[WorkerState, list[dict[str, Any]]]:
        results = await asyncio.gather(
            *[
                asyncio.to_thread(
                    _post_json,
                    f"{worker.url}/v1/chain_cache/routing_match",
                    {"chain_id": chain_id},
                    self.control_timeout_s,
                )
                for worker in candidates
            ],
            return_exceptions=True,
        )
        matches: list[dict[str, Any]] = []
        owners: list[tuple[WorkerState, dict[str, Any]]] = []
        tombstoned = False
        probe_errors: list[str] = []
        enabled_workers = 0
        for worker, result in zip(candidates, results):
            if isinstance(result, Exception):
                worker.healthy = False
                worker.last_error = f"{type(result).__name__}: {result}"
                probe_errors.append(f"{worker.url}: {worker.last_error}")
                matches.append(
                    {
                        "worker_url": worker.url,
                        "error": worker.last_error,
                    }
                )
                continue
            match = dict(result)
            matches.append({"worker_url": worker.url, **match})
            enabled_workers += int(bool(match.get("enabled", True)))
            tombstoned = tombstoned or bool(match.get("tombstone"))
            if bool(match.get("present")):
                owners.append((worker, match))
        if probe_errors:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Cannot determine a unique chain owner because worker "
                    f"probes failed: {probe_errors}."
                ),
            )
        if enabled_workers == 0:
            raise HTTPException(
                status_code=400,
                detail="Chain prefix cache is not enabled on any matching worker.",
            )
        if len(owners) > 1:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"chain_id {chain_id!r} has duplicate owners: "
                    f"{[worker.url for worker, _ in owners]}."
                ),
            )
        if not owners:
            if tombstoned:
                raise _chain_http_exception(
                    ChainGoneError(
                        f"chain_id {chain_id!r} was evicted.",
                        chain_id=chain_id,
                    )
                )
            raise HTTPException(
                status_code=404,
                detail=f"Unknown chain_id {chain_id!r}.",
            )
        worker, match = owners[0]
        if str(match.get("state") or "") == "active":
            raise HTTPException(
                status_code=409,
                detail=f"chain_id {chain_id!r} already has an active writer.",
            )
        if str(match.get("state") or "") != "idle":
            raise HTTPException(
                status_code=500,
                detail=(
                    f"chain_id {chain_id!r} owner {worker.url} returned "
                    f"invalid state {match.get('state')!r}."
                ),
            )
        return worker, matches

    def _candidate_workers(
        self,
        endpoint: str,
        payload: dict[str, Any],
        route_hints: dict[str, Any],
    ) -> list[WorkerState]:
        profile_name = str(route_hints.get("route_profile") or infer_route_profile(endpoint, payload))
        profile = dict(self.profiles.get(profile_name, {}) or {})
        required_tags = set(_string_list(profile.get("required_tags")) + _string_list(route_hints.get("required_tags")))
        preferred_tags = set(_string_list(profile.get("preferred_tags")) + _string_list(route_hints.get("preferred_tags")))
        method_preference = _string_list(route_hints.get("method_preference")) or _string_list(profile.get("methods"))
        model = payload.get("model")
        candidates = []
        for worker in self.workers:
            if not worker.healthy:
                continue
            info = worker.info
            worker_model = info.get("served_model_name")
            if model is not None and worker_model is not None and str(worker_model) != str(model):
                continue
            tags = set(_string_list(info.get("tags")))
            if required_tags and not required_tags.issubset(tags):
                continue
            if method_preference:
                method = str(info.get("sparse_method", "") or "")
                if not _method_matches(method, method_preference):
                    continue
            candidates.append(worker)
        if preferred_tags:
            preferred = [
                worker
                for worker in candidates
                if preferred_tags.intersection(set(_string_list(worker.info.get("tags"))))
            ]
            if preferred:
                return preferred
        return candidates

    async def _probe_workers(
        self,
        candidates: list[WorkerState],
        match_payload: dict[str, Any] | None,
    ) -> list[WorkerProbe]:
        tasks = [
            self._probe_worker(worker, match_payload)
            for worker in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        probes = []
        for worker, result in zip(candidates, results):
            if isinstance(result, Exception):
                worker.healthy = False
                worker.last_error = f"{type(result).__name__}: {result}"
                continue
            probes.append(result)
        return probes

    async def _probe_worker(
        self,
        worker: WorkerState,
        match_payload: dict[str, Any] | None,
    ) -> WorkerProbe:
        load_task = asyncio.to_thread(
            _get_json,
            f"{worker.url}/v1/worker/routing_load",
            self.control_timeout_s,
        )
        if match_payload is None:
            load = await load_task
            match = {"supported": False, "enabled": False, "matched_tokens": 0, "match_ratio": 0.0}
        else:
            load, match = await asyncio.gather(
                load_task,
                asyncio.to_thread(
                    _post_json,
                    f"{worker.url}/v1/prefix_cache/routing_match",
                    match_payload,
                    self.control_timeout_s,
                ),
            )
        return WorkerProbe(worker=worker, load=dict(load), match=dict(match))

    async def forward_json(
        self,
        worker: WorkerState,
        endpoint: str,
        payload: dict[str, Any],
        *,
        is_disconnected: DisconnectChecker | None = None,
    ) -> Response:
        request = _CloseableByteRequest(
            f"{worker.url}{endpoint}",
            "POST",
            payload,
            self.request_timeout_s,
        )
        worker.local_inflight += 1
        request_task = asyncio.create_task(asyncio.to_thread(request.execute))
        try:
            done, _ = await _wait_for_any_or_disconnect(
                {request_task},
                is_disconnected,
            )
            status, headers, body = next(iter(done)).result()
        finally:
            if not request_task.done():
                request.close()
                request_task.cancel()
            worker.local_inflight -= 1
        return Response(content=body, status_code=status, headers=_content_headers(headers))

    async def forward_stream(
        self,
        worker: WorkerState,
        endpoint: str,
        payload: dict[str, Any],
        route: dict[str, Any],
        *,
        is_disconnected: DisconnectChecker | None = None,
    ) -> Response:
        worker.local_inflight += 1
        stream_handoff = False
        opened_response = None
        try:
            upstream = await asyncio.to_thread(
                _open_stream_response,
                f"{worker.url}{endpoint}",
                payload,
                self.request_timeout_s,
            )
            if isinstance(upstream, UpstreamError):
                return Response(
                    content=upstream.body,
                    status_code=upstream.status,
                    headers=_route_response_headers(upstream.headers, route),
                )
            opened_response = upstream.response
            response_headers = _route_response_headers(upstream.headers, route)

            async def _stream():
                try:
                    chunks = _stream_response_chunks(opened_response)
                    while True:
                        next_chunk = asyncio.create_task(chunks.__anext__())
                        try:
                            done, _ = await _wait_for_any_or_disconnect(
                                {next_chunk},
                                is_disconnected,
                            )
                            try:
                                chunk = next(iter(done)).result()
                            except StopAsyncIteration:
                                break
                        finally:
                            if not next_chunk.done():
                                next_chunk.cancel()
                        yield chunk
                finally:
                    try:
                        opened_response.close()
                    finally:
                        worker.local_inflight -= 1

            response = StreamingResponse(
                _stream(),
                media_type="text/event-stream",
                headers=response_headers,
            )
            stream_handoff = True
            return response
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            if not stream_handoff:
                if opened_response is not None:
                    try:
                        opened_response.close()
                    finally:
                        worker.local_inflight -= 1
                else:
                    worker.local_inflight -= 1

    async def broadcast_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        await self.refresh_worker_info()
        workers = [worker for worker in self.workers if worker.healthy]
        if not workers:
            raise HTTPException(status_code=503, detail="No healthy SparseEngine worker is available.")
        results = await asyncio.gather(
            *[self._broadcast_one(worker, endpoint, payload) for worker in workers],
            return_exceptions=True,
        )
        items = []
        for worker, result in zip(workers, results):
            if isinstance(result, Exception):
                items.append({"worker_url": worker.url, "ok": False, "error": f"{type(result).__name__}: {result}"})
            else:
                items.append({"worker_url": worker.url, "ok": True, "result": result})
        return {"workers": items}

    async def _broadcast_one(self, worker: WorkerState, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        status, _headers, body = await asyncio.to_thread(
            _request_bytes,
            f"{worker.url}{endpoint}",
            "POST",
            payload,
            self.request_timeout_s,
        )
        decoded = json.loads(body.decode("utf-8")) if body else {}
        if status >= 400:
            raise RuntimeError(json.dumps(decoded, ensure_ascii=False))
        return decoded

    def _route_record(
        self,
        endpoint: str,
        worker: WorkerState,
        reason: str,
        probes: list[WorkerProbe],
        route_hints: dict[str, Any],
    ) -> dict[str, Any]:
        selected_probe = next(
            (
                probe
                for probe in probes
                if probe.worker is worker
            ),
            None,
        )
        return {
            "id": f"route-{uuid.uuid4().hex}",
            "endpoint": endpoint,
            "selected_worker_url": worker.url,
            "selected_sparse_method": worker.info.get("sparse_method", ""),
            "selected_prefix_matched_tokens": (
                selected_probe.matched_tokens
                if selected_probe is not None
                else None
            ),
            "reason": reason,
            "hints": route_hints,
            "probes": [
                {
                    "worker_url": probe.worker.url,
                    "sparse_method": probe.worker.info.get("sparse_method", ""),
                    "load": probe.load_value,
                    "match_ratio": probe.match_ratio,
                    "matched_tokens": probe.matched_tokens,
                }
                for probe in probes
            ],
        }

    def _write_route_log(self, route: dict[str, Any]):
        if self.route_log_dir is None:
            return
        path = self.route_log_dir / f"{int(time.time() * 1000)}_{route['id']}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(route, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")


def choose_worker(
    probes: list[WorkerProbe],
    *,
    overload_load_factor: float,
    load_abs_threshold: int,
) -> tuple[WorkerState, str]:
    if not probes:
        raise ValueError("No worker probes supplied.")
    by_load = sorted(probes, key=lambda probe: (probe.load_value, probe.worker.url))
    best_load_probe = by_load[0]
    by_cache = sorted(
        probes,
        key=lambda probe: (-probe.matched_tokens, -probe.match_ratio, probe.load_value, probe.worker.url),
    )
    best_cache_probe = by_cache[0]
    if best_cache_probe.matched_tokens <= 0:
        return best_load_probe.worker, "lowest_load_no_prefix_match"
    min_load = best_load_probe.load_value
    best_load = best_cache_probe.load_value
    avg_load = sum(probe.load_value for probe in probes) / float(len(probes))
    overloaded = (
        best_load - min_load > int(load_abs_threshold)
        and best_load > max(avg_load * float(overload_load_factor), min_load)
    )
    if overloaded:
        return best_load_probe.worker, "prefix_match_overloaded_lowest_load"
    return best_cache_probe.worker, "best_prefix_match"


def strip_route_hints(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    forward_payload = dict(payload)
    hints = {
        "route_profile": forward_payload.pop("sengine_route_profile", None),
        "method_preference": forward_payload.pop("sengine_method_preference", None),
        "required_tags": forward_payload.pop("sengine_required_tags", None),
        "preferred_tags": forward_payload.pop("sengine_preferred_tags", None),
        "target_worker": forward_payload.pop("sengine_target_worker", None),
    }
    return forward_payload, {key: value for key, value in hints.items() if value is not None}


def infer_route_profile(endpoint: str, payload: dict[str, Any]) -> str:
    if endpoint == "/v1/chat/completions" and len(payload.get("messages") or []) > 2:
        return "conversation"
    return "default"


def match_payload_for_request(endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if endpoint == "/v1/prefix_cache/match":
        return {key: payload[key] for key in ("chat", "token_ids", "text") if key in payload}
    if endpoint == "/v1/responses":
        return {"response": payload}
    if endpoint == "/v1/chat/completions":
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            return {"chat": payload}
    if endpoint in {
        "/v1/completions",
        "/v1/prefix_cache/inspect",
        "/v1/prefix_cache/prune",
    }:
        prompt = payload.get("prompt")
        if "token_ids" in payload or "text" in payload:
            return {key: payload[key] for key in ("token_ids", "text") if key in payload}
        if isinstance(prompt, str):
            return {"text": prompt}
        if isinstance(prompt, list) and prompt:
            if all(isinstance(item, int) for item in prompt):
                return {"token_ids": prompt}
            if isinstance(prompt[0], str):
                return {"text": prompt[0]}
            if isinstance(prompt[0], list) and all(isinstance(item, int) for item in prompt[0]):
                return {"token_ids": prompt[0]}
    return None


def route_headers(route: dict[str, Any]) -> dict[str, str]:
    headers = {
        "X-SparseVLLM-Worker": str(route["selected_worker_url"]),
        "X-SparseVLLM-Route-Reason": str(route["reason"]),
        "X-SparseVLLM-Sparse-Method": str(route.get("selected_sparse_method", "")),
    }
    matched_tokens = route.get("selected_prefix_matched_tokens")
    if matched_tokens is not None:
        headers["X-SparseVLLM-Prefix-Matched-Tokens"] = str(matched_tokens)
    if route.get("chain_id"):
        headers["X-SparseVLLM-Chain-ID"] = str(route["chain_id"])
        headers["X-SparseVLLM-Chain-Affinity"] = "true"
    return headers


def _route_response_headers(
    upstream_headers: dict[str, str],
    route: dict[str, Any],
) -> dict[str, str]:
    content_headers = _content_headers(upstream_headers)
    headers = {**content_headers, **route_headers(route)}
    # The worker may replace a mismatched chain with a new identity. Its
    # response header is authoritative; route metadata only describes the
    # identity used to select the worker.
    returned_chain_id = content_headers.get("X-SparseVLLM-Chain-ID")
    if returned_chain_id:
        headers["X-SparseVLLM-Chain-ID"] = returned_chain_id
    return headers


def _with_route_headers(response: Response, route: dict[str, Any]) -> Response:
    returned_chain_id = response.headers.get("X-SparseVLLM-Chain-ID")
    for key, value in route_headers(route).items():
        response.headers[key] = value
    if returned_chain_id:
        response.headers["X-SparseVLLM-Chain-ID"] = returned_chain_id
    return response


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _canonical_method(value: str) -> str:
    method = str(value or "").strip()
    return "" if method == "vanilla" else method


def _method_matches(worker_method: str, preferences: list[str]) -> bool:
    worker = _canonical_method(worker_method)
    return any(worker == _canonical_method(item) for item in preferences)


def _get_json(url: str, timeout_s: float) -> dict[str, Any]:
    status, _headers, body = _request_bytes(url, "GET", None, timeout_s)
    if status >= 400:
        raise RuntimeError(body.decode("utf-8", errors="replace"))
    return json.loads(body.decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    status, _headers, body = _request_bytes(url, "POST", payload, timeout_s)
    if status >= 400:
        raise RuntimeError(body.decode("utf-8", errors="replace"))
    return json.loads(body.decode("utf-8"))


class _CloseableByteRequest:
    def __init__(
        self,
        url: str,
        method: str,
        payload: dict[str, Any] | None,
        timeout_s: float,
    ) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Unsupported upstream URL: {url!r}")
        connection_cls = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        self.connection = connection_cls(
            parsed.hostname,
            parsed.port,
            timeout=float(timeout_s),
        )
        self.method = method
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += f"?{parsed.query}"
        self.body = None if payload is None else json.dumps(payload).encode("utf-8")
        self.cancelled = threading.Event()
        self.response: http.client.HTTPResponse | None = None

    def close(self) -> None:
        self.cancelled.set()
        upstream_socket = self.connection.sock
        if upstream_socket is not None:
            try:
                upstream_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.response is not None:
            self.response.close()
        self.connection.close()

    def execute(self) -> tuple[int, dict[str, str], bytes]:
        if self.cancelled.is_set():
            raise RuntimeError("Upstream request was cancelled before it started.")
        try:
            self.connection.request(
                self.method,
                self.path,
                body=self.body,
                headers={"Content-Type": "application/json"},
            )
            if self.cancelled.is_set():
                raise RuntimeError("Upstream request was cancelled before its response.")
            response = self.connection.getresponse()
            self.response = response
            if self.cancelled.is_set():
                raise RuntimeError("Upstream request was cancelled before reading its response.")
            try:
                return (
                    int(response.status),
                    dict(response.headers.items()),
                    response.read(),
                )
            finally:
                response.close()
                self.response = None
        except OSError as exc:
            raise RuntimeError(str(exc)) from exc
        finally:
            self.connection.close()


def _request_bytes(
    url: str,
    method: str,
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> tuple[int, dict[str, str], bytes]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = UrlRequest(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return int(response.status), dict(response.headers.items()), response.read()
    except HTTPError as exc:
        return int(exc.code), dict(exc.headers.items()), exc.read()
    except URLError as exc:
        raise RuntimeError(str(exc)) from exc


def _open_stream_response(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> UpstreamStream | UpstreamError:
    data = json.dumps(payload).encode("utf-8")
    request = UrlRequest(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=timeout_s)
        return UpstreamStream(response=response, headers=dict(response.headers.items()))
    except HTTPError as exc:
        return UpstreamError(status=int(exc.code), headers=dict(exc.headers.items()), body=exc.read())
    except URLError as exc:
        raise RuntimeError(f"Failed to open upstream stream {url}: {type(exc).__name__}: {exc}") from exc


async def _stream_response_chunks(response: Any):
    while True:
        chunk = await asyncio.to_thread(response.read, 8192)
        if not chunk:
            break
        yield chunk


def _content_headers(headers: dict[str, str]) -> dict[str, str]:
    content_type = headers.get("Content-Type") or headers.get("content-type")
    result = {"Content-Type": content_type} if content_type else {}
    chain_id = (
        headers.get("X-SparseVLLM-Chain-ID")
        or headers.get("x-sparseengine-chain-id")
    )
    if chain_id:
        result["X-SparseVLLM-Chain-ID"] = chain_id
    return result


def _load_profiles(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SparseEngine OpenAI smart router")
    parser.add_argument("--worker-url", "--worker-urls", action="append", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--control-timeout-s", type=float, default=5.0)
    parser.add_argument("--overload-load-factor", type=float, default=1.5)
    parser.add_argument("--load-abs-threshold", type=int, default=1)
    parser.add_argument("--profiles-json", default=None, help="Inline JSON or a JSON file path for route profiles.")
    parser.add_argument("--route-log-dir", default=None)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    worker_urls = []
    for item in args.worker_url:
        worker_urls.extend(_string_list(item))
    app = create_app(
        worker_urls,
        request_timeout_s=args.request_timeout_s,
        control_timeout_s=args.control_timeout_s,
        overload_load_factor=args.overload_load_factor,
        load_abs_threshold=args.load_abs_threshold,
        profiles=_load_profiles(args.profiles_json),
        route_log_dir=args.route_log_dir,
    )
    logger.info("Starting SparseEngine smart router on {}:{} for workers={}", args.host, args.port, worker_urls)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
