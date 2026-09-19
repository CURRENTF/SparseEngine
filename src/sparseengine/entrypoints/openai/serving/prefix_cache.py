from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from sparseengine.entrypoints.openai.dispatcher import AsyncEngineDispatcher
from sparseengine.entrypoints.openai.protocol.chat import ChatCompletionRequest
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCacheDeleteSubtreeRequest
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCacheInspectRequest
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCacheMatchRequest
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCachePruneRequest
from sparseengine.entrypoints.openai.protocol.prefix_cache import PrefixCacheSetEvictionPriorityRequest
from sparseengine.entrypoints.openai.protocol.responses import ResponseRequest
from sparseengine.entrypoints.openai.reasoning import ReasoningCapabilities
from sparseengine.entrypoints.openai.render import _chat_prompt
from sparseengine.entrypoints.openai.render import _chat_request_prompt
from sparseengine.entrypoints.openai.render import _response_prompt


async def serve_prefix_cache_inspect(
    request: PrefixCacheInspectRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
):
    token_ids = _prefix_cache_token_ids_from_request(request, tokenizer)
    result = await _run_prefix_cache_control(
        dispatcher,
        "prefix_cache_inspect",
        token_ids=token_ids,
        include_subtree=bool(request.include_subtree),
    )
    return JSONResponse(result)


async def serve_prefix_cache_match(
    request: PrefixCacheMatchRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
    reasoning_capabilities: ReasoningCapabilities | None = None,
):
    token_ids = _prefix_cache_match_token_ids_from_request(
        request,
        tokenizer,
        reasoning_capabilities,
    )
    result = await _run_prefix_cache_control(
        dispatcher,
        "prefix_cache_match",
        token_ids=token_ids,
    )
    return JSONResponse(result)


async def serve_prefix_cache_routing_match(
    request: PrefixCacheMatchRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
    reasoning_capabilities: ReasoningCapabilities | None = None,
):
    token_ids = _prefix_cache_match_token_ids_from_request(
        request,
        tokenizer,
        reasoning_capabilities,
    )
    try:
        result = dispatcher.prefix_cache_routing_match(token_ids)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(result)


async def serve_prefix_cache_delete_subtree(
    request: PrefixCacheDeleteSubtreeRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
):
    token_ids = _prefix_cache_token_ids_from_request(request, tokenizer)
    result = await _run_prefix_cache_control(
        dispatcher,
        "prefix_cache_delete_subtree",
        token_ids=token_ids,
    )
    return JSONResponse(result)


async def serve_prefix_cache_set_eviction_priority(
    request: PrefixCacheSetEvictionPriorityRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
):
    token_ids = _prefix_cache_token_ids_from_request(request, tokenizer)
    result = await _run_prefix_cache_control(
        dispatcher,
        "prefix_cache_set_eviction_priority",
        token_ids=token_ids,
        priority=int(request.priority),
    )
    return JSONResponse(result)


async def serve_prefix_cache_prune(
    request: PrefixCachePruneRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
    reasoning_capabilities: ReasoningCapabilities | None = None,
):
    token_ids = _prefix_cache_token_ids_from_request(
        request,
        tokenizer,
        reasoning_capabilities,
    )
    result = await _run_prefix_cache_control(
        dispatcher,
        "prefix_cache_prune_start",
        token_ids=token_ids,
        range_start=int(request.range_start),
        range_end=int(request.range_end),
        keep_tokens=int(request.keep_tokens),
        policy=str(request.policy),
        allow_recompress=bool(request.allow_recompress),
        observation_tokens=int(request.observation_tokens),
        score_chunk_size=int(request.score_chunk_size),
        prev_postfix_size=int(request.prev_postfix_size),
    )
    return JSONResponse(result, status_code=202)


async def serve_prefix_cache_prune_status(
    prune_id: str,
    dispatcher: AsyncEngineDispatcher,
):
    result = await _run_prefix_cache_control(
        dispatcher,
        "prefix_cache_prune_status",
        prune_id=str(prune_id),
    )
    return JSONResponse(result)


def _prefix_cache_token_ids_from_request(
    request: PrefixCacheInspectRequest | PrefixCacheDeleteSubtreeRequest | PrefixCacheSetEvictionPriorityRequest | PrefixCachePruneRequest,
    tokenizer: Any,
    reasoning_capabilities: ReasoningCapabilities | None = None,
) -> list[int]:
    has_token_ids = request.token_ids is not None
    has_text = request.text is not None
    chat = getattr(request, "chat", None)
    has_chat = chat is not None
    if sum((has_token_ids, has_text, has_chat)) != 1:
        selectors = (
            "token_ids, text, or chat"
            if isinstance(request, PrefixCachePruneRequest)
            else "token_ids or text"
        )
        raise HTTPException(status_code=400, detail=f"Set exactly one of {selectors}.")
    if request.token_ids is not None:
        return [int(token_id) for token_id in request.token_ids]
    if chat is not None:
        try:
            chat_request = ChatCompletionRequest.model_validate(chat)
            prompt = _chat_request_prompt(
                tokenizer,
                chat_request,
                reasoning_capabilities,
            )
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _encode_prefix_cache_text(tokenizer, prompt)
    return _encode_prefix_cache_text(tokenizer, str(request.text))


def _prefix_cache_match_token_ids_from_request(
    request: PrefixCacheMatchRequest,
    tokenizer: Any,
    reasoning_capabilities: ReasoningCapabilities | None = None,
) -> list[int]:
    selectors = [
        request.token_ids is not None,
        request.text is not None,
        request.messages is not None,
        request.chat is not None,
        request.response is not None,
    ]
    if sum(1 for selected in selectors if selected) != 1:
        raise HTTPException(
            status_code=400,
            detail="Set exactly one of token_ids, text, messages, chat, or response.",
        )
    if request.token_ids is not None:
        return [int(token_id) for token_id in request.token_ids]
    if request.messages is not None:
        try:
            prompt = _chat_prompt(tokenizer, request.messages)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _encode_prefix_cache_text(tokenizer, prompt)
    if request.chat is not None:
        try:
            chat_request = ChatCompletionRequest.model_validate(request.chat)
            prompt = _chat_request_prompt(
                tokenizer,
                chat_request,
                reasoning_capabilities,
            )
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _encode_prefix_cache_text(tokenizer, prompt)
    if request.response is not None:
        try:
            response_request = ResponseRequest.model_validate(request.response)
            prompt = _response_prompt(
                tokenizer,
                response_request,
                reasoning_capabilities,
            )
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _encode_prefix_cache_text(tokenizer, prompt)
    return _encode_prefix_cache_text(tokenizer, str(request.text))


def _encode_prefix_cache_text(tokenizer: Any, text: str) -> list[int]:
    add_special_tokens = True
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token is None or text.startswith(str(bos_token)):
        add_special_tokens = False
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


async def _run_prefix_cache_control(
    dispatcher: AsyncEngineDispatcher,
    operation: str,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        result = await dispatcher.control(operation, **kwargs)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail=f"Prefix cache control returned non-object result: {type(result).__name__}.")
    return result
