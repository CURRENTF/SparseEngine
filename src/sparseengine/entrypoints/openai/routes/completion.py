from fastapi import APIRouter
from fastapi import Request

from sparseengine.entrypoints.openai.protocol.completion import CompletionRequest
from sparseengine.entrypoints.openai.serving.completion import serve_completion


router = APIRouter()


@router.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request):
    return await serve_completion(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
        request.app.state.served_model_name,
        request.app.state.request_log_dir,
        is_disconnected=request.is_disconnected,
    )
