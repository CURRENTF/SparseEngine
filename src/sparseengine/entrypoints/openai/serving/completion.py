import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from sparseengine.entrypoints.openai.dispatcher import AsyncEngineDispatcher
from sparseengine.entrypoints.openai.dispatcher import RequestHandle
from sparseengine.entrypoints.openai.protocol.completion import CompletionRequest
from sparseengine.entrypoints.openai.sampling import _normalize_prompts
from sparseengine.entrypoints.openai.sampling import _normalize_stop
from sparseengine.entrypoints.openai.sampling import _sampling_params_from_request
from sparseengine.entrypoints.openai.serving.base import _completion_logprobs
from sparseengine.entrypoints.openai.serving.base import _model_dump_json
from sparseengine.entrypoints.openai.serving.base import _sse
from sparseengine.entrypoints.openai.serving.base import _tokens_per_second
from sparseengine.entrypoints.openai.serving.base import DisconnectChecker
from sparseengine.entrypoints.openai.serving.base import _wait_for_any_or_disconnect
from sparseengine.entrypoints.openai.serving.base import _wait_final
from sparseengine.entrypoints.openai.serving.base import _write_request_log
from sparseengine.entrypoints.openai.serving.base import _chain_http_exception
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
        _write_request_log(
            request_log_path,
            {
                "status": "stream_started",
                "endpoint": "/v1/completions",
                "request_id": request_id,
                "request": _model_dump_json(request),
            },
        )

    handles: list[RequestHandle] = []
    try:
        submit = (
            getattr(dispatcher, "submit_admitted", dispatcher.submit)
            if bool(getattr(dispatcher, "admission_ack_enabled", False))
            else dispatcher.submit
        )
        for index, prompt in enumerate(prompts):
            handle = (
                await submit(prompt, sampling_params, index, stop, stream=request.stream)
                if request.chain_id is None
                else await submit(
                    prompt,
                    sampling_params,
                    index,
                    stop,
                    chain_id=request.chain_id,
                    stream=request.stream,
                )
            )
            handles.append(handle)
    except ChainCacheError as exc:
        await _discard_partial_admissions(dispatcher, handles)
        raise _chain_http_exception(exc) from exc
    except BaseException:
        await _discard_partial_admissions(dispatcher, handles)
        raise
    chain_id = (
        getattr(handles[0], "chain_id", None)
        if len(handles) == 1
        else None
    )
    headers = (
        {"X-SparseVLLM-Chain-ID": chain_id}
        if chain_id is not None
        else None
    )

    if request.stream:
        return StreamingResponse(
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
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        response = await _completion_response(
            request_id,
            created,
            request.model,
            handles,
            tokenizer,
            is_disconnected=is_disconnected,
        )
    except asyncio.CancelledError:
        for handle in handles:
            dispatcher.cancel(handle)
        logger.info(
            "request_cancel id={} model={} stream=false elapsed_s={:.3f}",
            request_id,
            request.model,
            time.perf_counter() - started,
        )
        raise
    except Exception:
        for handle in handles:
            dispatcher.cancel(handle)
        raise
    usage = response["usage"]
    elapsed_s = time.perf_counter() - started
    logger.info(
        "request_finish id={} model={} stream=false prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
        request_id,
        request.model,
        usage["prompt_tokens"],
        usage["completion_tokens"],
        usage["total_tokens"],
        elapsed_s,
        _tokens_per_second(usage["completion_tokens"], elapsed_s),
        _tokens_per_second(usage["total_tokens"], elapsed_s),
    )
    _write_request_log(
        request_log_path,
        {
            "status": "success",
            "endpoint": "/v1/completions",
            "request_id": request_id,
            "elapsed_s": elapsed_s,
            "request": _model_dump_json(request),
            "response": response,
        },
    )
    return JSONResponse(response, headers=headers)


def _validate_request(request: CompletionRequest, served_model_name: str):
    if request.model != served_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {request.model!r}; this server is serving {served_model_name!r}.",
        )
    if request.n != 1:
        raise HTTPException(status_code=400, detail="Sparse-Engine completions currently supports n=1 only.")
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
            elapsed_s = time.perf_counter() - started
            total_tokens = prompt_tokens + completion_tokens
            logger.info(
                "request_finish id={} model={} stream=true prompt_tokens={} completion_tokens={} total_tokens={} elapsed_s={:.3f} completion_tps={:.2f} total_tps={:.2f}",
                request_id,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                elapsed_s,
                _tokens_per_second(completion_tokens, elapsed_s),
                _tokens_per_second(total_tokens, elapsed_s),
            )
    except asyncio.CancelledError:
        for handle in pending.values():
            dispatcher.cancel(handle)
        logger.info(
            "request_cancel id={} model={} stream=true completion_tokens={} elapsed_s={:.3f}",
            request_id,
            model,
            completion_tokens,
            time.perf_counter() - started if started is not None else 0.0,
        )
        raise
