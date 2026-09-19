import json
from typing import Any


def response_event(event_type: str, payload: dict[str, Any]) -> str:
    data = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def response_object(
    request_id: str,
    created_at: int,
    model: str,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
    error: dict[str, Any] | None = None,
    chain_id: str | None = None,
    chain_status: str | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": request_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
    }
    if chain_id is not None:
        response["chain_id"] = chain_id
        response["chain_status"] = chain_status
    if usage is not None:
        response["usage"] = usage
    if incomplete_reason is not None:
        response["incomplete_details"] = {"reason": incomplete_reason}
    if error is not None:
        response["error"] = error
    return response


def response_created(response: dict[str, Any]) -> str:
    return response_event("response.created", {"response": response})


def response_completed(response: dict[str, Any]) -> str:
    return response_event("response.completed", {"response": response})


def response_failed(response: dict[str, Any], error: dict[str, Any]) -> str:
    return response_event("response.failed", {"response": response, "error": error})


def output_item_added(output_index: int, item: dict[str, Any]) -> str:
    return response_event(
        "response.output_item.added",
        {"output_index": output_index, "item": item},
    )


def output_item_done(output_index: int, item: dict[str, Any]) -> str:
    return response_event(
        "response.output_item.done",
        {"output_index": output_index, "item": item},
    )


def content_part_added(
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict[str, Any],
) -> str:
    return response_event(
        "response.content_part.added",
        {
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        },
    )


def content_part_done(
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict[str, Any],
) -> str:
    return response_event(
        "response.content_part.done",
        {
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        },
    )


def reasoning_part_added(
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict[str, Any],
) -> str:
    return response_event(
        "response.reasoning_part.added",
        {
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        },
    )


def reasoning_part_done(
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict[str, Any],
) -> str:
    return response_event(
        "response.reasoning_part.done",
        {
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        },
    )


def output_text_delta(
    item_id: str,
    output_index: int,
    content_index: int,
    delta: str,
) -> str:
    return response_event(
        "response.output_text.delta",
        {
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
        },
    )


def output_text_done(
    item_id: str,
    output_index: int,
    content_index: int,
    text: str,
) -> str:
    return response_event(
        "response.output_text.done",
        {
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": text,
        },
    )


def reasoning_text_delta(
    item_id: str,
    output_index: int,
    content_index: int,
    delta: str,
) -> str:
    return response_event(
        "response.reasoning_text.delta",
        {
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
        },
    )


def reasoning_text_done(
    item_id: str,
    output_index: int,
    content_index: int,
    text: str,
) -> str:
    return response_event(
        "response.reasoning_text.done",
        {
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": text,
        },
    )


def function_call_arguments_delta(
    item_id: str,
    output_index: int,
    delta: str,
) -> str:
    return response_event(
        "response.function_call_arguments.delta",
        {"item_id": item_id, "output_index": output_index, "delta": delta},
    )


def function_call_arguments_done(
    item_id: str,
    output_index: int,
    arguments: str,
) -> str:
    return response_event(
        "response.function_call_arguments.done",
        {"item_id": item_id, "output_index": output_index, "arguments": arguments},
    )
