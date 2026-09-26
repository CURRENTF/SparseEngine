from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from sparseengine.entrypoints.openai.protocol.chat import ChatCompletionRequest
from sparseengine.entrypoints.openai.protocol.completion import CompletionRequest
from sparseengine.entrypoints.openai.protocol.responses import ResponseRequest
from sparseengine.sampling_params import DEFAULT_MAX_TOKENS, SamplingParams


def _field_was_set(request: BaseModel, name: str) -> bool:
    return name in request.model_fields_set


def _sampling_params_from_request(request: CompletionRequest | ChatCompletionRequest) -> SamplingParams:
    logprobs = request.logprobs
    max_tokens = request.max_tokens
    if isinstance(request, ChatCompletionRequest):
        logprobs = (
            request.top_logprobs
            if request.top_logprobs is not None
            else 0
        ) if request.logprobs else None
        if request.max_completion_tokens is not None:
            max_tokens = request.max_completion_tokens
    return SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        repetition_penalty=request.repetition_penalty,
        max_tokens=max_tokens,
        ignore_eos=request.ignore_eos,
        logprobs=logprobs,
        benchmark_forced_token_ids=(request.benchmark_forced_token_ids
                                    if isinstance(request, ChatCompletionRequest) else None),
    )


def _sampling_params_from_response_request(request: ResponseRequest) -> SamplingParams:
    return SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        repetition_penalty=request.repetition_penalty,
        max_tokens=(
            DEFAULT_MAX_TOKENS
            if request.max_output_tokens is None
            else request.max_output_tokens
        ),
    )


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [item for item in stop if item]


def _normalize_prompts(prompt: str | list[int] | list[str] | list[list[int]]) -> list[str | list[int]]:
    if isinstance(prompt, str):
        return [prompt]
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty.")
    if all(isinstance(item, int) for item in prompt):
        return [prompt]
    if all(isinstance(item, str) for item in prompt):
        return list(prompt)
    if all(isinstance(item, list) and all(isinstance(token, int) for token in item) for item in prompt):
        return list(prompt)
    raise HTTPException(status_code=400, detail="prompt must be a string, token id list, or homogeneous prompt list.")


def _find_stop_index(text: str, stop: list[str]) -> int | None:
    matches = [text.find(item) for item in stop if item and text.find(item) >= 0]
    return min(matches) if matches else None


def _safe_stream_text_len(text: str, stop: list[str]) -> int:
    if not stop:
        return len(text)
    max_overlap = 0
    for item in stop:
        max_prefix = min(len(item) - 1, len(text))
        for overlap in range(max_prefix, 0, -1):
            if text.endswith(item[:overlap]):
                max_overlap = max(max_overlap, overlap)
                break
    return len(text) - max_overlap


def _coerce_cli_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
