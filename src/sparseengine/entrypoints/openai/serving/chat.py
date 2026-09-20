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
from sparseengine.entrypoints.openai.protocol.chat import ChatCompletionRequest
from sparseengine.entrypoints.openai.reasoning import ReasoningCapabilities
from sparseengine.entrypoints.openai.render import _chat_request_prompt
from sparseengine.entrypoints.openai.render import _chat_request_append_prompt
from sparseengine.entrypoints.openai.render import resolve_chat_template_kwargs
from sparseengine.entrypoints.openai.render import resolve_chat_tools
from sparseengine.entrypoints.openai.sampling import _field_was_set
from sparseengine.entrypoints.openai.sampling import _normalize_stop
from sparseengine.entrypoints.openai.sampling import _sampling_params_from_request
from sparseengine.entrypoints.openai.serving.base import _chat_logprobs
from sparseengine.entrypoints.openai.serving.base import _model_dump_json
from sparseengine.entrypoints.openai.serving.base import _sse
from sparseengine.entrypoints.openai.serving.base import _tokens_per_second
from sparseengine.entrypoints.openai.serving.base import DisconnectChecker
from sparseengine.entrypoints.openai.serving.base import _queue_get_or_disconnect
from sparseengine.entrypoints.openai.serving.base import _wait_final
from sparseengine.entrypoints.openai.serving.base import _write_request_log
from sparseengine.entrypoints.openai.serving.base import _chain_http_exception
from sparseengine.engine.chain_cache import ChainCacheError
from sparseengine.entrypoints.openai.serving.response_parsing import ParsedModelResponse
from sparseengine.entrypoints.openai.serving.response_parsing import ModelOutputParseError
from sparseengine.entrypoints.openai.serving.response_parsing import ResponseParseError
from sparseengine.entrypoints.openai.serving.response_parsing import TransformersResponseParser
from sparseengine.entrypoints.openai.serving.response_parsing import response_parser_prefix
from sparseengine.utils.log import logger


async def serve_chat_completion(
    request: ChatCompletionRequest,
    dispatcher: AsyncEngineDispatcher,
    tokenizer: Any,
    served_model_name: str,
    request_log_path: Path | None,
    reasoning_parser_name: str | None = None,
    response_parser: TransformersResponseParser | None = None,
    reasoning_capabilities: ReasoningCapabilities | None = None,
    *,
    is_disconnected: DisconnectChecker | None = None,
):
    _validate_chat_request(
        request,
        served_model_name,
        tokenizer,
        reasoning_parser_name=reasoning_parser_name,
        response_parser=response_parser,
        reasoning_capabilities=reasoning_capabilities,
    )
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    started = time.perf_counter()
    sampling_params = _sampling_params_from_request(request)
    logger.info(
        "request_start id={} model={} endpoint=chat stream={} messages={} max_tokens={} temperature={} top_p={} top_k={}",
        request_id,
        request.model,
        request.stream,
        len(request.messages),
        sampling_params.max_tokens,
        request.temperature,
        request.top_p,
        request.top_k,
    )
    stop = _normalize_stop(request.stop)
    try:
        prompt = (
            _chat_request_append_prompt(tokenizer, request, reasoning_capabilities)
            if request.chain_append_start is not None
            else _chat_request_prompt(tokenizer, request, reasoning_capabilities)
        )
        chat_tools = resolve_chat_tools(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if request.stream:
        _write_request_log(
            request_log_path,
            {
                "status": "stream_started",
                "endpoint": "/v1/chat/completions",
                "request_id": request_id,
                "request": _model_dump_json(request),
            },
        )
    try:
        submit = (
            getattr(dispatcher, "submit_admitted", dispatcher.submit)
            if bool(getattr(dispatcher, "admission_ack_enabled", False))
            else dispatcher.submit
        )
        if request.chain_id is None:
            handle = await submit(prompt, sampling_params, 0, stop, stream=request.stream)
        else:
            submit_kwargs = {"chain_id": request.chain_id}
            if request.chain_append_start is not None:
                submit_kwargs["chain_append_only"] = True
            handle = await submit(
                prompt,
                sampling_params,
                0,
                stop,
                stream=request.stream,
                **submit_kwargs,
            )
    except ChainCacheError as exc:
        raise _chain_http_exception(exc) from exc
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    handles = [handle]
    should_parse = reasoning_parser_name is not None or bool(chat_tools)
    try:
        parser_prompt = (
            response_parser_prefix(tokenizer, prompt, handle)
            if should_parse
            else ""
        )
    except ResponseParseError as exc:
        dispatcher.cancel(handle)
        raise HTTPException(
            status_code=500,
            detail=f"Chat Completions parser setup failed: {exc}",
        ) from exc
    headers = (
        {"X-SparseVLLM-Chain-ID": getattr(handle, "chain_id", None)}
        if getattr(handle, "chain_id", None) is not None
        else None
    )

    if request.stream:
        return StreamingResponse(
            _chat_completion_stream(
                dispatcher,
                request_id,
                created,
                request.model,
                handles,
                started,
                tokenizer,
                _stream_include_usage(request.stream_options),
                prompt=parser_prompt,
                reasoning_parser_name=reasoning_parser_name,
                parse_tools=bool(chat_tools),
                response_parser=response_parser,
                is_disconnected=is_disconnected,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        response = await _chat_completion_response(
            request_id,
            created,
            request.model,
            handles,
            tokenizer,
            prompt=parser_prompt,
            reasoning_parser_name=reasoning_parser_name,
            parse_tools=bool(chat_tools),
            response_parser=response_parser,
            is_disconnected=is_disconnected,
        )
    except asyncio.CancelledError:
        dispatcher.cancel(handle)
        logger.info(
            "request_cancel id={} model={} stream=false elapsed_s={:.3f}",
            request_id,
            request.model,
            time.perf_counter() - started,
        )
        raise
    except Exception:
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
            "endpoint": "/v1/chat/completions",
            "request_id": request_id,
            "elapsed_s": elapsed_s,
            "request": _model_dump_json(request),
            "response": response,
        },
    )
    return JSONResponse(response, headers=headers)


def _validate_chat_request(
    request: ChatCompletionRequest,
    served_model_name: str,
    tokenizer: Any | None = None,
    *,
    reasoning_parser_name: str | None = None,
    response_parser: TransformersResponseParser | None = None,
    reasoning_capabilities: ReasoningCapabilities | None = None,
):
    if request.model != served_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {request.model!r}; this server is serving {served_model_name!r}.",
        )
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty.")
    if request.chain_append_start is not None and not str(
        request.chain_id or ""
    ).strip():
        raise HTTPException(
            status_code=400,
            detail="chain_append_start requires a non-empty chain_id.",
        )
    if request.n != 1:
        raise HTTPException(status_code=400, detail="SparseEngine chat completions currently supports n=1 only.")
    if request.store:
        raise HTTPException(status_code=400, detail="Chat Completions store=true is not supported; completions are not persisted.")
    if (
        request.max_completion_tokens is not None
        and _field_was_set(request, "max_tokens")
        and request.max_tokens != request.max_completion_tokens
    ):
        raise HTTPException(
            status_code=400,
            detail="max_tokens and max_completion_tokens disagree; set only one value.",
        )
    if request.top_logprobs is not None and not request.logprobs:
        raise HTTPException(status_code=400, detail="top_logprobs requires logprobs=true.")
    if request.stop and request.logprobs:
        raise HTTPException(status_code=400, detail="stop with logprobs is not supported yet.")
    try:
        chat_template_kwargs = resolve_chat_template_kwargs(
            request,
            reasoning_capabilities,
        )
        tools = resolve_chat_tools(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if chat_template_kwargs and tokenizer is not None and not getattr(tokenizer, "chat_template", None):
        raise HTTPException(status_code=400, detail="chat_template_kwargs requires a tokenizer chat_template.")
    parse_response = reasoning_parser_name is not None or bool(tools)
    if parse_response and response_parser is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "This tokenizer does not provide a compatible Transformers response template "
                "for parsed reasoning or tool calls."
            ),
        )
    if request.logprobs and parse_response:
        raise HTTPException(
            status_code=400,
            detail="Chat logprobs cannot be aligned with parsed reasoning or tool calls.",
        )


def _stream_include_usage(stream_options: dict[str, Any] | None) -> bool:
    if stream_options is None:
        return False
    return bool(stream_options.get("include_usage"))


async def _chat_completion_response(
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    tokenizer: Any | None = None,
    *,
    prompt: str = "",
    reasoning_parser_name: str | None = None,
    parse_tools: bool = False,
    response_parser: TransformersResponseParser | None = None,
    is_disconnected: DisconnectChecker | None = None,
) -> dict[str, Any]:
    if len(handles) != 1:
        raise HTTPException(status_code=500, detail="chat completions expects exactly one request handle.")
    final = await _wait_final(handles[0].output_queue, is_disconnected)
    should_parse = reasoning_parser_name is not None or parse_tools
    if should_parse:
        if response_parser is None:
            raise HTTPException(status_code=500, detail="Chat response parser is not configured.")
        try:
            parsed = response_parser.parse(
                final.get("raw_text", final["text"]),
                prefix=prompt,
                parse_tools=parse_tools,
            )
        except ModelOutputParseError as exc:
            # Malformed tool JSON is a model-format failure, not an inference
            # service failure. Return the generated text as plain assistant
            # content so tool-aware clients can record a FormatError without
            # retrying the identical request as an HTTP 500.
            logger.warning(
                "request_format_error id={} model={} stream=false error={}",
                request_id,
                model,
                exc,
            )
            parsed = ParsedModelResponse(None, final["text"], [])
        except ResponseParseError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Chat Completions parse failed: {exc}",
            ) from exc
    else:
        parsed = ParsedModelResponse(None, final["text"], [])
    message = _chat_message(parsed)
    usage = {
        "prompt_tokens": final["prompt_tokens"],
        "completion_tokens": final["completion_tokens"],
        "total_tokens": final["prompt_tokens"] + final["completion_tokens"],
    }
    if "reused_tokens" in final:
        usage["prompt_tokens_details"] = {
            "cached_tokens": int(final["reused_tokens"])
        }
    if final.get("chain_id"):
        usage["cached_tokens"] = int(final.get("reused_tokens", 0))
        usage["reused_tokens"] = int(final.get("reused_tokens", 0))
    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": final["index"],
                "message": message,
                "logprobs": _chat_logprobs(
                    tokenizer,
                    final.get("token_ids", []),
                    final.get("token_logprobs", []),
                    final.get("top_logprobs", []),
                )
                if tokenizer is not None
                else None,
                "finish_reason": (
                    "tool_calls"
                    if parsed.tool_calls and final["finish_reason"] == "stop"
                    else final["finish_reason"]
                ),
            }
        ],
        "usage": usage,
    }
    if final.get("chain_id") is not None:
        response["chain_id"] = final["chain_id"]
        response["chain_status"] = final.get("chain_status")
    return response


def _chat_message(parsed: ParsedModelResponse) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": parsed.content if parsed.content or not parsed.tool_calls else None,
    }
    if parsed.reasoning_content is not None:
        message["reasoning_content"] = parsed.reasoning_content
    if parsed.tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                },
            }
            for tool_call in parsed.tool_calls
        ]
    return message


async def _chat_completion_stream(
    dispatcher: AsyncEngineDispatcher,
    request_id: str,
    created: int,
    model: str,
    handles: list[RequestHandle],
    started: float | None = None,
    tokenizer: Any | None = None,
    include_usage: bool = False,
    *,
    prompt: str = "",
    reasoning_parser_name: str | None = None,
    parse_tools: bool = False,
    response_parser: TransformersResponseParser | None = None,
    is_disconnected: DisconnectChecker | None = None,
):
    if len(handles) != 1:
        raise HTTPException(status_code=500, detail="chat completions expects exactly one request handle.")
    pending = {index: handle for index, handle in enumerate(handles)}
    should_parse = reasoning_parser_name is not None or parse_tools
    if should_parse and response_parser is None:
        raise HTTPException(status_code=500, detail="Chat response parser is not configured.")
    parser = response_parser.stream(prefix=prompt, parse_tools=parse_tools) if should_parse else None
    prompt_tokens = 0
    completion_tokens = 0
    raw_text_len = 0
    visible_text_len = 0
    chain_id = getattr(handles[0], "chain_id", None)
    chain_status = getattr(handles[0], "chain_status", None)
    try:
        yield _chat_stream_chunk(
            request_id,
            created,
            model,
            0,
            {"role": "assistant"},
            chain_id=chain_id,
            chain_status=chain_status,
        )
        while pending:
            handle = next(iter(pending.values()))
            item = await _queue_get_or_disconnect(handle.output_queue, is_disconnected)
            if item["type"] == "error":
                yield _sse({"object": "error", "message": item["message"]})
                pending.clear()
                continue
            if item["type"] == "token":
                completion_tokens += len(item["token_ids"])
                logprobs = (
                    _chat_logprobs(
                        tokenizer,
                        item.get("token_ids", []),
                        item.get("token_logprobs", []),
                        item.get("top_logprobs", []),
                    )
                    if tokenizer is not None
                    else None
                )
                if parser is not None:
                    text_delta = item.get("raw_text_delta", item.get("text", ""))
                    raw_text_len += len(text_delta)
                    deltas = parser.feed(text_delta)
                else:
                    text_delta = item.get("text", "")
                    visible_text_len += len(text_delta)
                    deltas = [{"content": text_delta}] if text_delta else []
                if not deltas and logprobs is not None:
                    deltas = [{"content": ""}]
                for delta_index, delta in enumerate(deltas):
                    yield _chat_stream_chunk(
                        request_id,
                        created,
                        model,
                        item["index"],
                        delta,
                        logprobs=logprobs if delta_index == 0 else None,
                        chain_id=chain_id,
                        chain_status=item.get("chain_status", chain_status),
                    )
                continue
            if item["type"] == "final":
                chain_status = item.get("chain_status", chain_status)
                prompt_tokens += item["prompt_tokens"]
                completion_tokens = max(completion_tokens, item["completion_tokens"])
                if parser is not None:
                    parser_text = item.get("raw_text", item["text"])
                    deltas = parser.feed(parser_text[raw_text_len:])
                    deltas.extend(parser.finish())
                else:
                    parser_text = item["text"]
                    suffix = parser_text[visible_text_len:]
                    deltas = [{"content": suffix}] if suffix else []
                for delta in deltas:
                    yield _chat_stream_chunk(
                        request_id,
                        created,
                        model,
                        item["index"],
                        delta,
                        chain_id=chain_id,
                        chain_status=item.get("chain_status", chain_status),
                    )
                finish_reason = item["finish_reason"]
                if parser is not None and parser.tools_called and finish_reason == "stop":
                    finish_reason = "tool_calls"
                yield _chat_stream_chunk(
                    request_id,
                    created,
                    model,
                    item["index"],
                    {},
                    finish_reason=finish_reason,
                    chain_id=chain_id,
                    chain_status=item.get("chain_status", chain_status),
                )
                pending.clear()
        if include_usage:
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": int(getattr(handles[0], "reused_tokens", 0))
                },
            }
            if chain_id:
                reused_tokens = int(
                    getattr(handles[0], "reused_tokens", 0)
                )
                usage["cached_tokens"] = reused_tokens
                usage["reused_tokens"] = reused_tokens
            usage_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            }
            if chain_id is not None:
                usage_chunk["chain_id"] = chain_id
                usage_chunk["chain_status"] = chain_status
            yield _sse(usage_chunk)
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
    except ResponseParseError as exc:
        for handle in pending.values():
            dispatcher.cancel(handle)
        message = f"Chat Completions parse failed: {exc}"
        yield _sse({"object": "error", "message": message})
        yield "data: [DONE]\n\n"
        logger.info(
            "request_failure id={} model={} stream=true elapsed_s={:.3f} error={}",
            request_id,
            model,
            time.perf_counter() - started if started is not None else 0.0,
            message,
        )


def _chat_stream_chunk(
    request_id: str,
    created: int,
    model: str,
    index: int,
    delta: dict[str, Any],
    *,
    logprobs: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    chain_id: str | None = None,
    chain_status: str | None = None,
) -> str:
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "logprobs": logprobs,
                "finish_reason": finish_reason,
            }
        ],
    }
    if chain_id is not None:
        chunk["chain_id"] = chain_id
        chunk["chain_status"] = chain_status
    return _sse(chunk)
