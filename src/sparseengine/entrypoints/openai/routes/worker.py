from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import JSONResponse

from sparseengine.entrypoints.openai.serving.worker import serve_worker_info
from sparseengine.entrypoints.openai.serving.worker import serve_worker_load
from sparseengine.entrypoints.openai.serving.worker import serve_worker_routing_load


router = APIRouter()


@router.get("/v1/worker/info")
def worker_info(request: Request):
    dispatcher = request.app.state.dispatcher
    if not dispatcher.is_ready:
        failure_message = dispatcher.failure_message
        reason = failure_message.split(":", 1)[0] if failure_message else "dispatcher_unavailable"
        return JSONResponse(
            {"status": "unavailable", "reason": reason},
            status_code=503,
        )
    return serve_worker_info(
        request.app.state.engine,
        request.app.state.served_model_name,
    )


@router.get("/v1/worker/load")
async def worker_load(request: Request):
    return await serve_worker_load(request.app.state.dispatcher)


@router.get("/v1/worker/routing_load")
async def worker_routing_load(request: Request):
    return await serve_worker_routing_load(
        request.app.state.dispatcher,
    )


@router.post("/v1/chain_cache/routing_match")
async def chain_cache_routing_match(request: Request):
    payload = await request.json()
    chain_id = str(payload.get("chain_id") or "").strip()
    if not chain_id:
        return JSONResponse(
            {"detail": "chain_id must be a non-empty string."},
            status_code=400,
        )
    try:
        result = request.app.state.dispatcher.chain_cache_routing_match(
            chain_id
        )
    except RuntimeError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=503)
    return JSONResponse(result)
