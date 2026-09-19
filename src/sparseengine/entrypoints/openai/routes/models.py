import time

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import JSONResponse


router = APIRouter()


@router.get("/health")
def health(request: Request):
    return _readiness_response(request)


@router.get("/readyz")
def ready(request: Request):
    return _readiness_response(request)


@router.get("/livez")
def live():
    return JSONResponse({"status": "ok"})


def _readiness_response(request: Request) -> JSONResponse:
    dispatcher = request.app.state.dispatcher
    if dispatcher.is_ready:
        return JSONResponse({"status": "ok"})
    failure_message = dispatcher.failure_message
    reason = failure_message.split(":", 1)[0] if failure_message else "dispatcher_unavailable"
    return JSONResponse(
        {"status": "unavailable", "reason": reason},
        status_code=503,
    )


@router.get("/v1/models")
def models(request: Request):
    served_model_name = request.app.state.served_model_name
    created = int(time.time())
    config = getattr(request.app.state.engine, "config", None)
    max_model_len = int(getattr(config, "max_model_len", 0) or 0)
    model = {
        "id": served_model_name,
        "object": "model",
        "created": created,
        "owned_by": "sparseengine",
    }
    if max_model_len > 0:
        model["max_model_len"] = max_model_len
    return {
        "object": "list",
        "data": [model],
    }
