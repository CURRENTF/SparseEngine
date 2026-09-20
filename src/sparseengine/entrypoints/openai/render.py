from __future__ import annotations

import inspect
import json
from typing import Any, TYPE_CHECKING

from sparseengine.entrypoints.openai.protocol.chat import ChatContentPart
if TYPE_CHECKING:
    from sparseengine.multimodal import MultiModalPrompt
from sparseengine.entrypoints.openai.protocol.chat import ChatCompletionRequest
from sparseengine.entrypoints.openai.protocol.chat import ChatMessage
from sparseengine.entrypoints.openai.protocol.responses import ResponseRequest
from sparseengine.entrypoints.openai.reasoning import ReasoningCapabilities
from sparseengine.entrypoints.openai.reasoning import reasoning_template_kwargs


BOOLEAN_CHAT_TEMPLATE_KWARGS = {"enable_thinking", "preserve_thinking"}


def normalize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("tools entries must be JSON objects.")
        tool_type = tool.get("type")
        if tool_type != "function":
            raise ValueError(f"Unsupported tool type: {tool_type!r}.")
        function = tool.get("function")
        if function is not None:
            if not isinstance(function, dict):
                raise ValueError("function tool.function must be a JSON object.")
            name = function.get("name")
            description = function.get("description")
            parameters = function.get("parameters", {})
            strict = function.get("strict", tool.get("strict", False))
        else:
            name = tool.get("name")
            description = tool.get("description")
            parameters = tool.get("parameters", {})
            strict = tool.get("strict", False)

        if not isinstance(name, str) or not name:
            raise ValueError("function tool name must be a non-empty string.")
        if description is not None and not isinstance(description, str):
            raise ValueError("function tool description must be a string.")
        if not isinstance(parameters, dict):
            raise ValueError("function tool parameters must be a JSON object.")
        if not isinstance(strict, bool):
            raise ValueError("function tool strict must be a bool.")

        item = {
            "type": "function",
            "name": name,
            "parameters": parameters,
            "strict": strict,
        }
        if description is not None:
            item["description"] = description
        normalized.append(item)
    return normalized


def _chat_template_source(tokenizer: Any) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, dict):
        return "\n".join(str(value) for value in chat_template.values())
    return chat_template if isinstance(chat_template, str) else ""


def _uses_minimax_tool_format(tokenizer: Any) -> bool:
    chat_template = _chat_template_source(tokenizer)
    return "<minimax:tool_call>" in chat_template and "tool.function" in chat_template


def _uses_nested_tool_format(tokenizer: Any) -> bool:
    chat_template = _chat_template_source(tokenizer)
    if _uses_minimax_tool_format(tokenizer):
        return True
    return "format_function_declaration(tool)" in chat_template and any(
        marker in chat_template
        for marker in (
            "tool_data['function']",
            'tool_data["function"]',
        )
    )


def _tools_for_chat_template(
    tokenizer: Any,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not _uses_nested_tool_format(tokenizer):
        return tools

    nested = []
    for tool in tools:
        function = {
            "name": tool["name"],
            "parameters": tool["parameters"],
        }
        if "description" in tool:
            function["description"] = tool["description"]
        if tool.get("strict"):
            function["strict"] = True
        nested.append({"type": "function", "function": function})
    return nested


def _chat_template_role(role: str) -> str:
    return "system" if role == "developer" else role


def _chat_content_text(content: str | list[ChatContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text or "" for part in content)


def _has_multimodal_content(messages) -> bool:
    return any(
        isinstance(message.content, list)
        and any(part.type != "text" for part in message.content)
        for message in messages
    )


def validate_chat_template_kwargs(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("chat_template_kwargs must be a JSON object.")
    for name in BOOLEAN_CHAT_TEMPLATE_KWARGS:
        if name in value and not isinstance(value[name], bool):
            raise ValueError(f"chat_template_kwargs.{name} must be a bool.")
    return dict(value)


def resolve_chat_template_kwargs(
    request: ChatCompletionRequest,
    reasoning_capabilities: ReasoningCapabilities | None = None,
) -> dict[str, Any] | None:
    kwargs = validate_chat_template_kwargs(request.chat_template_kwargs) or {}
    _merge_chat_template_kwarg(kwargs, "preserve_thinking", request.preserve_thinking)
    _merge_chat_template_kwarg(kwargs, "enable_thinking", request.enable_thinking)
    for name, value in reasoning_template_kwargs(
        request.reasoning_effort,
        reasoning_capabilities,
    ).items():
        _merge_chat_template_kwarg(kwargs, name, value)
    return kwargs or None


def _merge_chat_template_kwarg(kwargs: dict[str, Any], name: str, value: Any):
    if value is None:
        return
    if name in kwargs and kwargs[name] != value:
        raise ValueError(f"{name} conflicts with chat_template_kwargs.{name}.")
    kwargs[name] = value


def resolve_chat_tools(request: ChatCompletionRequest) -> list[dict[str, Any]] | None:
    tools = normalize_tools(request.tools)
    if request.tool_choice not in (None, "auto", "none"):
        raise ValueError("Chat tool_choice only supports null, 'auto', or 'none' in this implementation.")
    if request.parallel_tool_calls not in (None, True):
        raise ValueError("Chat parallel_tool_calls=false is not implemented yet.")
    return None if request.tool_choice == "none" else tools


def _chat_request_prompt(
    tokenizer: Any,
    request: ChatCompletionRequest,
    reasoning_capabilities: ReasoningCapabilities | None = None,
) -> str:
    return _chat_prompt(
        tokenizer,
        request.messages,
        resolve_chat_template_kwargs(request, reasoning_capabilities),
        resolve_chat_tools(request),
    )


def _chat_request_append_prompt(
    tokenizer: Any,
    request: ChatCompletionRequest,
    reasoning_capabilities: ReasoningCapabilities | None = None,
) -> str:
    if _has_multimodal_content(request.messages):
        raise ValueError("Multimodal chat does not support chain append rendering.")
    append_start = request.chain_append_start
    if append_start is None:
        raise ValueError("chain_append_start is required for append rendering.")
    if append_start >= len(request.messages):
        raise ValueError(
            "chain_append_start must point to at least one new message."
        )
    previous_response = request.messages[append_start - 1]
    if previous_response.role != "assistant":
        raise ValueError(
            "The message before chain_append_start must be an assistant "
            "response."
        )
    context = [
        ChatMessage(role="user", content="chain append context"),
        previous_response,
    ]
    kwargs = resolve_chat_template_kwargs(request, reasoning_capabilities)
    prefix = _chat_prompt(
        tokenizer,
        context,
        kwargs,
        tools=None,
        add_generation_prompt=False,
    )
    combined = _chat_prompt(
        tokenizer,
        [*context, *request.messages[append_start:]],
        kwargs,
        tools=None,
        add_generation_prompt=True,
    )
    if not combined.startswith(prefix):
        raise ValueError(
            "Chat template cannot render a stable chain append suffix."
        )
    suffix = combined[len(prefix):]
    if not suffix:
        raise ValueError("Chat template rendered an empty chain append suffix.")
    return suffix


def resolve_response_chat_template_kwargs(
    request: ResponseRequest,
    reasoning_capabilities: ReasoningCapabilities | None = None,
) -> dict[str, Any] | None:
    kwargs = validate_chat_template_kwargs(request.chat_template_kwargs) or {}
    effort = request.reasoning.effort if request.reasoning is not None else None
    for name, value in reasoning_template_kwargs(effort, reasoning_capabilities).items():
        _merge_chat_template_kwarg(kwargs, name, value)
    return kwargs or None


def _chat_prompt(
    tokenizer: Any,
    messages: list[ChatMessage],
    chat_template_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    *,
    add_generation_prompt: bool = True,
) -> str | MultiModalPrompt:
    chat = []
    for message in messages:
        rendered_message = {
            "role": _chat_template_role(message.role),
            "content": (
                None
                if message.content is None
                else [part.model_dump(exclude_none=True) for part in message.content]
                if isinstance(message.content, list) and _has_multimodal_content(messages)
                else _chat_content_text(message.content)
            ),
        }
        if message.reasoning_content is not None:
            rendered_message["reasoning_content"] = message.reasoning_content
        if message.tool_calls is not None:
            rendered_message["tool_calls"] = _chat_template_tool_calls(message.tool_calls)
        if message.tool_call_id is not None:
            rendered_message["tool_call_id"] = message.tool_call_id
        chat.append(rendered_message)
    if _has_multimodal_content(messages):
        from sparseengine.multimodal import MultiModalPrompt

        return MultiModalPrompt(
            chat,
            chat_template_kwargs=chat_template_kwargs,
            tools=_tools_for_chat_template(tokenizer, tools) if tools else None,
            add_generation_prompt=add_generation_prompt,
        )
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": bool(add_generation_prompt),
        }
        kwargs.update(chat_template_kwargs or {})
        if tools:
            if not _supports_chat_template_kwarg(tokenizer, "tools"):
                raise ValueError("Tokenizer chat template does not support tools.")
            kwargs["tools"] = _tools_for_chat_template(tokenizer, tools)
        return tokenizer.apply_chat_template(chat, **kwargs)
    if any("reasoning_content" in message for message in chat):
        raise ValueError("reasoning_content requires a tokenizer chat_template.")
    if chat_template_kwargs:
        raise ValueError("chat_template_kwargs requires a tokenizer chat_template.")
    if tools:
        raise ValueError("tools requires a tokenizer chat_template with tools support.")
    if _messages_require_chat_template(chat):
        raise ValueError("Chat tool-call history requires a tokenizer chat_template.")

    rendered = []
    for message in chat:
        rendered.append(f"{message['role']}: {message['content'] or ''}")
    if add_generation_prompt:
        rendered.append("assistant:")
    return "\n".join(rendered)


def _chat_template_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered = []
    for tool_call in tool_calls:
        normalized = dict(tool_call)
        function = dict(normalized["function"])
        try:
            arguments = json.loads(function["arguments"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"assistant tool call {function['name']!r} arguments is not valid JSON: {exc}"
            ) from exc
        if not isinstance(arguments, dict):
            raise ValueError(
                f"assistant tool call {function['name']!r} arguments must decode to a JSON object."
            )
        function["arguments"] = arguments
        normalized["function"] = function
        rendered.append(normalized)
    return rendered


def _response_prompt(
    tokenizer: Any,
    request: ResponseRequest,
    reasoning_capabilities: ReasoningCapabilities | None = None,
) -> str | MultiModalPrompt:
    chat_template_kwargs = resolve_response_chat_template_kwargs(
        request,
        reasoning_capabilities,
    )
    tools = normalize_tools(request.tools) if request.tools else None
    messages = _response_messages(request)
    if any(
        isinstance(message.get("content"), list)
        and any(part.get("type") in {"input_image", "input_audio", "input_video"} for part in message["content"])
        for message in messages
    ):
        from sparseengine.multimodal import MultiModalPrompt

        return MultiModalPrompt(
            messages,
            chat_template_kwargs=chat_template_kwargs,
            tools=_tools_for_chat_template(tokenizer, tools) if tools else None,
        )

    has_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
    if has_template:
        if _uses_minimax_tool_format(tokenizer):
            messages = _minimax_response_messages(messages)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if chat_template_kwargs:
            kwargs.update(chat_template_kwargs)
        if tools:
            if not _supports_chat_template_kwarg(tokenizer, "tools"):
                raise ValueError("Tokenizer chat template does not support tools.")
            kwargs["tools"] = _tools_for_chat_template(tokenizer, tools)
        return tokenizer.apply_chat_template(messages, **kwargs)

    if chat_template_kwargs:
        raise ValueError("chat_template_kwargs requires a tokenizer chat_template.")
    if tools:
        raise ValueError("tools requires a tokenizer chat_template with tools support.")
    if _messages_require_chat_template(messages):
        raise ValueError("Responses tool-call history requires a tokenizer chat_template.")
    rendered = []
    for message in messages:
        rendered.append(f"{message['role']}: {message.get('content', '')}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def _response_messages(request: ResponseRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.instructions is not None:
        messages.append({"role": "system", "content": request.instructions})

    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
        return messages

    if not request.input:
        raise ValueError("responses input must not be empty.")
    for item in request.input:
        if not isinstance(item, dict):
            raise ValueError("responses input items must be JSON objects.")
        messages.extend(_response_input_item_messages(item))
    return messages


def _response_input_item_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    item_type = item.get("type")
    if item_type in (None, "message"):
        return [_response_message_item(item)]
    if item_type == "function_call_output":
        call_id = item.get("call_id")
        output = item.get("output")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("function_call_output.call_id must be a non-empty string.")
        if not isinstance(output, str):
            raise ValueError("function_call_output.output must be a string.")
        return [{"role": "tool", "content": output, "tool_call_id": call_id}]
    if item_type == "function_call":
        return [_response_function_call_item(item)]
    if item_type == "reasoning":
        return []
    raise ValueError(f"Unsupported responses input item type: {item_type!r}.")


def _response_message_item(item: dict[str, Any]) -> dict[str, Any]:
    role = item.get("role")
    if role not in {"developer", "system", "user", "assistant"}:
        raise ValueError("message.role must be one of developer, system, user, assistant.")
    return {
        "role": _chat_template_role(str(role)),
        "content": _response_content(item.get("content")),
    }


def _response_function_call_item(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("function_call.call_id must be a non-empty string.")
    if not isinstance(name, str) or not name:
        raise ValueError("function_call.name must be a non-empty string.")
    if not isinstance(arguments, str):
        raise ValueError("function_call.arguments must be a string.")
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _minimax_response_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapted = []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            message["tool_calls"] = _chat_template_tool_calls(message["tool_calls"])
        adapted.append(message)
    return adapted


def _response_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("message content parts must be JSON objects.")
            part_type = part.get("type")
            if part_type in {"input_image", "input_video"}:
                field = "image_url" if part_type == "input_image" else "video_url"
                if set(part) - {"type", field}:
                    raise ValueError(f"{part_type} contains unsupported fields.")
                value = part.get(field)
                url = value.get("url") if isinstance(value, dict) else value
                if not isinstance(url, str) or not url:
                    raise ValueError(f"{part_type} requires a non-empty {field}.")
                normalized.append({"type": part_type, field: value})
                continue
            if part_type == "input_audio":
                if set(part) - {"type", "input_audio"}:
                    raise ValueError("input_audio contains unsupported fields.")
                audio = part.get("input_audio")
                if not isinstance(audio, dict) or set(audio) - {"data", "format"}:
                    raise ValueError("input_audio requires data and optional format fields.")
                if not isinstance(audio.get("data"), str) or not audio["data"]:
                    raise ValueError("input_audio requires a non-empty base64 data string.")
                if "format" in audio and str(audio["format"]).lower() != "wav":
                    raise ValueError("Only WAV input_audio is supported.")
                normalized.append({"type": "input_audio", "input_audio": dict(audio)})
                continue
            if part_type not in {"text", "input_text", "output_text"}:
                raise ValueError(f"Unsupported message content part type: {part_type!r}.")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("message content text parts require a string text field.")
            normalized.append({"type": "text", "text": text})
        return normalized if any(part["type"] != "text" for part in normalized) else "\n".join(part["text"] for part in normalized)
    raise ValueError("message.content must be a string or a content part list.")


def _messages_require_chat_template(messages: list[dict[str, Any]]) -> bool:
    return any(message.get("role") == "tool" or message.get("tool_calls") for message in messages)


def _supports_chat_template_kwarg(tokenizer: Any, name: str) -> bool:
    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters
