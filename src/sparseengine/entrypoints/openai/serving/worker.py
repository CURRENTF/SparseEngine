import os
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from sparseengine.entrypoints.openai.dispatcher import AsyncEngineDispatcher


def serve_worker_info(engine: Any, served_model_name: str):
    return JSONResponse(engine.worker_info(served_model_name=served_model_name, tags=_worker_tags()))


async def serve_worker_load(dispatcher: AsyncEngineDispatcher):
    try:
        result = await dispatcher.control("worker_load")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail=f"Worker load returned non-object result: {type(result).__name__}.")
    return JSONResponse(result)


async def serve_worker_routing_load(dispatcher: AsyncEngineDispatcher):
    try:
        result = dispatcher.worker_routing_load_snapshot()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                "Worker routing load returned non-object result: "
                f"{type(result).__name__}."
            ),
        )
    return JSONResponse(result)


def _worker_tags() -> list[str]:
    raw = os.getenv("SPARSEENGINE_WORKER_TAGS", "")
    return [tag.strip() for tag in raw.split(",") if tag.strip()]
