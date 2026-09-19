from fastapi import APIRouter
from fastapi import Request

from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCacheDeleteSubtreeRequest
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCacheInspectRequest
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCacheMatchRequest
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCachePruneRequest
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCacheSetEvictionPriorityRequest
from sparseengine.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_delete_subtree
from sparseengine.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_inspect
from sparseengine.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_match
from sparseengine.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_prune
from sparseengine.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_prune_status
from sparseengine.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_routing_match
from sparseengine.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_set_eviction_priority


router = APIRouter()


@router.post("/v1/prefix_cache/inspect")
async def prefix_cache_inspect(body: PrefixCacheInspectRequest, request: Request):
    return await serve_prefix_cache_inspect(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
    )


@router.post("/v1/prefix_cache/match")
async def prefix_cache_match(body: PrefixCacheMatchRequest, request: Request):
    return await serve_prefix_cache_match(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
        request.app.state.reasoning_capabilities,
    )


@router.post("/v1/prefix_cache/routing_match")
async def prefix_cache_routing_match(body: PrefixCacheMatchRequest, request: Request):
    return await serve_prefix_cache_routing_match(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
        request.app.state.reasoning_capabilities,
    )


@router.post("/v1/prefix_cache/delete_subtree")
async def prefix_cache_delete_subtree(body: PrefixCacheDeleteSubtreeRequest, request: Request):
    return await serve_prefix_cache_delete_subtree(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
    )


@router.post("/v1/prefix_cache/set_eviction_priority")
async def prefix_cache_set_eviction_priority(body: PrefixCacheSetEvictionPriorityRequest, request: Request):
    return await serve_prefix_cache_set_eviction_priority(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
    )


@router.post("/v1/prefix_cache/prune")
async def prefix_cache_prune(body: PrefixCachePruneRequest, request: Request):
    return await serve_prefix_cache_prune(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
        request.app.state.reasoning_capabilities,
    )


@router.get("/v1/prefix_cache/prune/{prune_id}")
async def prefix_cache_prune_status(prune_id: str, request: Request):
    return await serve_prefix_cache_prune_status(
        prune_id,
        request.app.state.dispatcher,
    )
