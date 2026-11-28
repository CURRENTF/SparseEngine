import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from sparseengine.entrypoints.openai.dispatcher import AsyncEngineDispatcher
from sparseengine.entrypoints.openai.dispatcher import RequestHandle
from sparseengine.entrypoints.openai.protocol.completion import CompletionRequest
from sparseengine.entrypoints.openai.sampling import _normalize_prompts
from sparseengine.entrypoints.openai.sampling import _normalize_stop
from sparseengine.entrypoints.openai.sampling import _sampling_params_from_request
from sparseengine.entrypoints.openai.serving.base import (
    DisconnectChecker,
    _cancel_requests,
    _chain_headers,
    _chain_http_exception,
    _completion_logprobs,
    _json_response,
    _log_request_cancel,
    _log_request_finish,
    _log_request_payload,
    _request_submitter,
    _sse,
    _stream_response,
    _submit_request,
    _wait_final,
    _wait_for_any_or_disconnect,
)
from sparseengine.engine.chain_cache import ChainCacheError
from sparseengine.utils.log import logger


async def _discard_partial_admissions(
    dispatcher: AsyncEngineDispatcher,
    handles: list[RequestHandle],
) -> None:
    discard = getattr(dispatcher, "discard", None)
    errors: list[BaseException] = []
    for handle in handles:
        try:
            if callable(discard):
                await discard(handle)
            else:
                dispatcher.cancel(handle)
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(
            "Failed to discard one or more partially admitted requests: "
            f"{len(errors)} cleanup error(s)."
        ) from errors[0]


async def serve_completion(
    request: CompletionRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
    served_model_name: str,
    request_log_path: Path | None,
    *,
    is_disconnected: DisconnectChecker | None = None,
):
    _validate_request(request, served_model_name)
    request_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    started = time.perf_counter()
    endpoint = "/v1/completions"
    prompts = _normalize_prompts(request.prompt)
    logger.info(
        "request_start id={} model={} stream={} prompts={} max_tokens={} temperature={} top_p={} top_k={}",
        request_id,
        request.model,
        request.stream,
        len(prompts),
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.top_k,
    )
    sampling_params = _sampling_params_from_request(request)
    stop = _normalize_stop(request.stop)
    if request.stream:
        _log_request_payload(
            request_log_path, request, endpoint, request_id, "stream_started"
        )

    handles: list[RequestHandle] = []
    try:
        submit = _request_submitter(dispatcher)
        for index, prompt in enumerate(prompts):
            handles.append(await _submit_request(submit, request, prompt, sampling_params, index, stop))
    except ChainCacheError as exc:
        await _discard_partial_admissions(dispatcher, handles)
        raise _chain_http_exception(exc) from exc
    except BaseException:
        await _discard_partial_admissions(dispatcher, handles)
        raise
    headers = _chain_headers(handles)

    if request.stream:
        return _stream_response(
            _completion_stream(
                dispatcher,
                request_id,
                created,
                request.model,
                handles,
                started,
                tokenizer,
                is_disconnected=is_disconnected,
            ),
            headers,
        )

    return await _json_response(
        _completion_response(
            request_id,
            created,
            request.model,
            handles,
            tokenizer,
            is_disconnected=is_disconnected,
        ),
        dispatcher, handles, request, request_log_path, endpoint, request_id, started, headers,
    )


def _validate_request(request: CompletionRequest, served_model_name: str):
    if request.model != served_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {request.model!r}; this server is serving {served_model_name!r}.",
        )
    if request.n != 1:
        raise HTTPException(status_code=400, detail="SparseEngine completions currently supports n=1 only.")
    if request.stop and request.logprobs is not None:
        raise HTTPException(status_code=400, detail="stop with logprobs is not supported yet.")
    if request.chain_id and len(_normalize_prompts(request.prompt)) != 1:
        raise HTTPException(
            status_code=400,
            detail="chain mode completions requires exactly one prompt.",
        )


async def _completion_response(
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    tokenizer: Any | None = None,
    *,
    is_disconnected: DisconnectChecker | None = None,
) -> dict[str, Any]:
    choices = []
    prompt_tokens = 0
    completion_tokens = 0
    chain_status = None
    for handle in handles:
        final = await _wait_final(handle.output_queue, is_disconnected)
        chain_status = final.get("chain_status", chain_status)
        choice = {
            "text": final["text"],
            "index": final["index"],
            "logprobs": _completion_logprobs(
                tokenizer,
                final.get("token_ids", []),
                final.get("token_logprobs", []),
                final.get("top_logprobs", []),
            )
            if tokenizer is not None
            else None,
            "finish_reason": final["finish_reason"],
        }
        if final.get("chain_id") is not None:
            choice["chain_id"] = final["chain_id"]
            choice["chain_status"] = final.get("chain_status")
        choices.append(choice)
        prompt_tokens += final["prompt_tokens"]
        completion_tokens += final["completion_tokens"]

    choices.sort(key=lambda choice: choice["index"])
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": sum(
                int(getattr(handle, "reused_tokens", 0)) for handle in handles
            )
        },
    }
    if any(getattr(handle, "chain_id", None) for handle in handles):
        reused_tokens = sum(
            int(getattr(handle, "reused_tokens", 0))
            for handle in handles
        )
        usage["cached_tokens"] = reused_tokens
        usage["reused_tokens"] = reused_tokens
    response = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
    }
    chain_id = (
        getattr(handles[0], "chain_id", None)
        if len(handles) == 1
        else None
    )
    if chain_id is not None:
        response["chain_id"] = chain_id
        response["chain_status"] = chain_status
    return response


async def _completion_stream(
    dispatcher: AsyncEngineDispatcher,
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    started: float | None = None,
    tokenizer: Any | None = None,
    *,
    is_disconnected: DisconnectChecker | None = None,
):
    pending = {index: handle for index, handle in enumerate(handles)}
    prompt_tokens = 0
    completion_tokens = 0
    try:
        while pending:
            tasks = {
                asyncio.create_task(handle.output_queue.get()): index
                for index, handle in pending.items()
            }
            try:
                done, _ = await _wait_for_any_or_disconnect(
                    set(tasks),
                    is_disconnected,
                )
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
            for task in done:
                item = task.result()
                if item["type"] == "error":
                    yield _sse({"object": "error", "message": item["message"]})
                    pending.pop(tasks[task], None)
                    continue
                if item["type"] == "token":
                    completion_tokens += len(item["token_ids"])
                    logprobs = (
                        _completion_logprobs(
                            tokenizer,
                            item.get("token_ids", []),
                            item.get("token_logprobs", []),
                            item.get("top_logprobs", []),
                        )
                        if tokenizer is not None
                        else None
                    )
                    if not item["text"] and logprobs is None:
                        continue
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "text": item["text"],
                                "index": item["index"],
                                "logprobs": logprobs,
                                "finish_reason": None,
                            }
                        ],
                    }
                    if item.get("chain_id") is not None:
                        chunk["chain_id"] = item["chain_id"]
                        chunk["chain_status"] = item.get("chain_status")
                    yield _sse(
                        chunk
                    )
                elif item["type"] == "final":
                    prompt_tokens += item["prompt_tokens"]
                    completion_tokens = max(completion_tokens, item["completion_tokens"])
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "text": item.get("text_delta", ""),
                                "index": item["index"],
                                "logprobs": None,
                                "finish_reason": item["finish_reason"],
                            }
                        ],
                    }
                    if item.get("chain_id") is not None:
                        chunk["chain_id"] = item["chain_id"]
                        chunk["chain_status"] = item.get("chain_status")
                    yield _sse(chunk)
                    pending.pop(tasks[task], None)
        yield "data: [DONE]\n\n"
        if started is not None:
            _log_request_finish(
                request_id, model, time.perf_counter() - started,
                prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, stream=True,
            )
    except asyncio.CancelledError:
        _cancel_requests(dispatcher, pending.values())
        _log_request_cancel(
            request_id, model, time.perf_counter() - started if started is not None else 0.0,
            completion_tokens=completion_tokens,
        )
        raise
