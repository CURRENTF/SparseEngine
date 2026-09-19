import asyncio
import json
import time
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel
from sparseengine.engine.chain_cache import ChainCacheError


DisconnectChecker = Callable[[], Awaitable[bool]]


class ClientDisconnected(asyncio.CancelledError):
    """Raised when the ASGI client disappears while generation is active."""


def _chain_http_exception(exc: ChainCacheError) -> HTTPException:
    return HTTPException(
        status_code=int(exc.status_code),
        detail={
            "code": exc.error_code,
            "message": str(exc),
            "chain_id": exc.chain_id,
        },
    )


def _model_dump_json(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _write_request_log(request_log_dir: Path | None, payload: dict[str, Any]):
    if request_log_dir is None:
        return
    path = request_log_dir / f"{int(time.time() * 1000)}_{uuid.uuid4().hex}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=True)


def _completion_logprobs(
    tokenizer: Any,
    token_ids: list[int],
    token_logprobs: list[float | None],
    top_logprobs: list[dict[int, float] | None],
) -> dict[str, Any] | None:
    if not token_logprobs or all(value is None for value in token_logprobs):
        return None
    tokens = [_token_text(tokenizer, token_id) for token_id in token_ids]
    text_offsets = []
    offset = 0
    for token in tokens:
        text_offsets.append(offset)
        offset += len(token)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": [
            None
            if item is None
            else {_token_text(tokenizer, token_id): value for token_id, value in item.items()}
            for item in top_logprobs
        ],
        "text_offset": text_offsets,
    }


def _chat_logprobs(
    tokenizer: Any,
    token_ids: list[int],
    token_logprobs: list[float | None],
    top_logprobs: list[dict[int, float] | None],
) -> dict[str, Any] | None:
    if not token_logprobs or all(value is None for value in token_logprobs):
        return None
    content = []
    for token_id, logprob, top_items in zip(token_ids, token_logprobs, top_logprobs):
        token = _token_text(tokenizer, token_id)
        top = []
        if top_items is not None:
            top = [
                {"token": _token_text(tokenizer, top_token_id), "logprob": value, "bytes": None}
                for top_token_id, value in top_items.items()
            ]
        content.append({"token": token, "logprob": logprob, "bytes": None, "top_logprobs": top})
    return {"content": content}


async def _wait_for_any_or_disconnect(
    tasks: set[asyncio.Task],
    is_disconnected: DisconnectChecker | None = None,
) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
    if is_disconnected is None:
        return await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    while True:
        if await is_disconnected():
            raise ClientDisconnected
        done, pending = await asyncio.wait(
            tasks,
            timeout=0.25,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if done:
            return done, pending


async def _queue_get_or_disconnect(
    queue_item: asyncio.Queue,
    is_disconnected: DisconnectChecker | None = None,
) -> Any:
    task = asyncio.create_task(queue_item.get())
    try:
        done, _ = await _wait_for_any_or_disconnect({task}, is_disconnected)
        return next(iter(done)).result()
    finally:
        if not task.done():
            task.cancel()


async def _wait_final(
    queue_item: asyncio.Queue,
    is_disconnected: DisconnectChecker | None = None,
) -> dict[str, Any]:
    while True:
        item = await _queue_get_or_disconnect(queue_item, is_disconnected)
        if item["type"] == "error":
            raise HTTPException(status_code=500, detail=item["message"])
        if item["type"] == "final":
            return item


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _tokens_per_second(tokens: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return tokens / elapsed_s
