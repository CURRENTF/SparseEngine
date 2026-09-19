import json
import uuid
from dataclasses import dataclass
from typing import Any


_QWEN_RESPONSE_TEMPLATE_BASE = {
    "version": 1,
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "thinking": {
            "open_pattern": r"<think>\s*",
            "close": "</think>",
            "content": "text",
            "optional": True,
        },
        "content": {
            "close_pattern": r"<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|\Z",
            "content": "text",
            "optional": True,
        },
    },
}

_QWEN_JSON_TOOL_FIELD = {
    "open": "<tool_call>",
    "close": "</tool_call>",
    "content": "json",
    "repeats": True,
    "optional": True,
}

_QWEN_XML_TOOL_FIELD = {
    "open_pattern": r"(?:<tool_call>\s*)+(?:<function=|\[|<parameter=)(?P<name>[^>\n]+)>?\s*",
    "close_pattern": r"\s*</function>\s*</tool_call>",
    "content": "xml-inline",
    "content_args": {
        "tag_pattern": r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    },
    "transform": {
        "type": "function",
        "function": {"name": "{name}", "arguments": "{content}"},
    },
    "repeats": True,
    "optional": True,
}

_GLM_RESPONSE_TEMPLATE = {
    "version": 1,
    "defaults": {"role": "assistant"},
    "start_anchor": "<|assistant|>",
    "fields": {
        "thinking": {
            # GLM writes <think> for thinking mode and a leading </think>
            # when thinking is disabled. The zero-width alternative consumes
            # that template-provided close without exposing it as content.
            "open_pattern": r"<think>\s*|(?=</think>)",
            "close": "</think>",
            "content": "text",
            "optional": True,
        },
        "tool_calls": {
            "open_pattern": (
                r"<tool_call>\s*(?P<name>[^<]*?)\s*"
                r"(?=<arg_key>|</tool_call>)"
            ),
            "close": "</tool_call>",
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r"<arg_key>\s*(?P<key>.*?)\s*</arg_key>\s*"
                    r"<arg_value>\s*(?P<value>.*?)\s*</arg_value>"
                ),
                "value_parser": {
                    "name": "json",
                    "args": {"allow_non_json": True},
                },
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
            "repeats": True,
            "optional": True,
        },
        "content": {
            "close_pattern": r"<\|endoftext\|>|<eop>|\Z",
            "content": "text",
            "optional": True,
        },
    },
}

_MINIMAX_RESPONSE_TEMPLATE_BASE = {
    "version": 1,
    "defaults": {"role": "assistant"},
    "start_anchor": "]~b]ai\n",
    "fields": {
        "thinking": {
            "open_pattern": r"<think>\s*",
            "close": "</think>",
            "content": "text",
            "optional": True,
        },
        "content": {
            "close_pattern": r"\[e~\[|\Z",
            "content": "text",
            "optional": True,
        },
    },
}

_MINIMAX_XML_TOOL_FIELD = {
    # MiniMax wraps one or more sibling <invoke> elements in a single
    # <minimax:tool_call> region. Keeping the outer tags optional on each
    # repeated field lets the shared parser emit every invocation separately.
    "open_pattern": (
        r"\s*(?:<minimax:tool_call>\s*)?"
        r"<invoke\s+name=[\"'](?P<name>[^\"']+)[\"']>\s*"
    ),
    "close_pattern": r"\s*</invoke>",
    "content": "xml-inline",
    "content_args": {
        "tag_pattern": (
            r"<parameter\s+name=[\"'](?P<key>[^\"']+)[\"']>\s*"
            r"(?P<value>.*?)\s*</parameter>"
        ),
    },
    "transform": {
        "type": "function",
        "function": {"name": "{name}", "arguments": "{content}"},
    },
    "repeats": True,
    "optional": True,
}

_MINIMAX_TOOL_END_FIELD = {
    "open_pattern": r"\s*</minimax:tool_call>\s*",
    "close_pattern": r"\[e~\[|\Z",
    "content": "text",
    "optional": True,
}


class ResponseParseError(ValueError):
    pass


class ModelOutputParseError(ResponseParseError):
    """The generated text is malformed, while the parser contract is valid."""


def response_parser_prefix(
    tokenizer: Any,
    prompt: Any,
    handle: Any,
) -> str:
    """Return the exact admitted prompt text needed to resume parser state."""
    if isinstance(prompt, str):
        return prompt
    prompt_token_ids = getattr(handle, "prompt_token_ids", None)
    if prompt_token_ids is None:
        raise ResponseParseError(
            "Multimodal response parsing requires admitted prompt token IDs."
        )
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise ResponseParseError(
            "Multimodal response parsing requires tokenizer.decode()."
        )
    try:
        prefix = decode(prompt_token_ids, skip_special_tokens=False)
    except (TypeError, ValueError) as exc:
        raise ResponseParseError(
            f"Failed to decode the multimodal parser prefix: {exc}"
        ) from exc
    if not isinstance(prefix, str):
        raise ResponseParseError(
            "Tokenizer decoded the multimodal parser prefix to a non-string value."
        )
    return prefix


def _caused_by_json_decode_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, json.JSONDecodeError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


@dataclass(frozen=True)
class ParsedModelResponse:
    reasoning_content: str | None
    content: str
    tool_calls: list[dict[str, Any]]


class TransformersResponseParser:
    """Thin OpenAI adapter around Transformers response parsing APIs."""

    def __init__(
        self,
        tokenizer: Any,
        response_template: dict[str, Any],
    ):
        if not callable(getattr(tokenizer, "parse_response", None)):
            raise TypeError("Transformers response parsing requires tokenizer.parse_response().")
        if not callable(getattr(tokenizer, "get_response_parser", None)):
            raise TypeError("Transformers response streaming requires tokenizer.get_response_parser().")
        self._tokenizer = tokenizer
        self._response_template = response_template

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "TransformersResponseParser | None":
        response_template = getattr(tokenizer, "response_template", None)
        if response_template is not None:
            return cls(tokenizer, response_template)

        chat_template = getattr(tokenizer, "chat_template", None)
        if isinstance(chat_template, dict):
            chat_template = "\n".join(str(value) for value in chat_template.values())
        if not isinstance(chat_template, str):
            return None
        if "<think>" not in chat_template and "<tool_call>" not in chat_template:
            return None

        if all(
            marker in chat_template
            for marker in ("<|assistant|>", "<arg_key>", "<arg_value>")
        ):
            return cls(tokenizer, _GLM_RESPONSE_TEMPLATE)

        if "<minimax:tool_call>" in chat_template and "<invoke name=" in chat_template:
            fields = dict(_MINIMAX_RESPONSE_TEMPLATE_BASE["fields"])
            content_field = fields.pop("content")
            fields["tool_calls"] = _MINIMAX_XML_TOOL_FIELD
            fields["tool_call_end"] = _MINIMAX_TOOL_END_FIELD
            fields["content"] = content_field
            response_template = {
                **_MINIMAX_RESPONSE_TEMPLATE_BASE,
                "fields": fields,
            }
            return cls(tokenizer, response_template)

        response_template = {
            **_QWEN_RESPONSE_TEMPLATE_BASE,
            "fields": dict(_QWEN_RESPONSE_TEMPLATE_BASE["fields"]),
        }
        if "<function=" in chat_template and "<parameter=" in chat_template:
            response_template["fields"]["tool_calls"] = _QWEN_XML_TOOL_FIELD
        elif "<tool_call>" in chat_template:
            response_template["fields"]["tool_calls"] = _QWEN_JSON_TOOL_FIELD
        return cls(tokenizer, response_template)

    def parse(
        self,
        text: str,
        *,
        prefix: str,
        parse_tools: bool,
    ) -> ParsedModelResponse:
        try:
            parsed = self._tokenizer.parse_response(
                text,
                self._response_template,
                prefix=prefix,
            )
            return _normalize_parsed_message(parsed, parse_tools=parse_tools)
        except ResponseParseError:
            raise
        except ValueError as exc:
            if _caused_by_json_decode_error(exc):
                raise ModelOutputParseError(str(exc)) from exc
            raise ResponseParseError(str(exc)) from exc
        except (AttributeError, KeyError, TypeError) as exc:
            raise ResponseParseError(str(exc)) from exc

    def stream(self, *, prefix: str, parse_tools: bool) -> "TransformersResponseStreamParser":
        try:
            parser = self._tokenizer.get_response_parser(
                self._response_template,
                prefix=prefix,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ResponseParseError(str(exc)) from exc
        return TransformersResponseStreamParser(
            parser,
            parse_tools=parse_tools,
        )


class TransformersResponseStreamParser:
    def __init__(
        self,
        parser: Any,
        *,
        parse_tools: bool,
    ):
        self._parser = parser
        self._parse_tools = parse_tools
        self._initial_events = list(parser.initial_events)
        self._tool_index = 0
        self.tools_called = False

    def feed(self, text_delta: str) -> list[dict[str, Any]]:
        events, self._initial_events = self._initial_events, []
        try:
            events.extend(self._parser.feed(text_delta))
            return self._events_to_deltas(events)
        except (KeyError, TypeError, ValueError) as exc:
            raise ResponseParseError(str(exc)) from exc

    def finish(self) -> list[dict[str, Any]]:
        try:
            _message, events = self._parser.finalize()
            return self._events_to_deltas(events)
        except (KeyError, TypeError, ValueError) as exc:
            raise ResponseParseError(str(exc)) from exc

    def _events_to_deltas(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deltas: list[dict[str, Any]] = []
        for event in events:
            event_type = event["type"]
            field = event["field"]
            if event_type == "region_chunk" and not event.get("dirty", False):
                if field in {"thinking", "reasoning", "reasoning_content"}:
                    if event["text"]:
                        deltas.append({"reasoning_content": event["text"]})
                elif field == "content" and event["text"]:
                    deltas.append({"content": event["text"]})
            elif event_type == "region_close" and field == "tool_calls" and self._parse_tools:
                tool_call = _normalize_tool_call(event["value"])
                call_id = f"call_{uuid.uuid4().hex}"
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": self._tool_index,
                                "id": call_id,
                                "type": "function",
                                "function": tool_call["function"],
                            }
                        ]
                    }
                )
                self._tool_index += 1
                self.tools_called = True
            elif event_type not in {"region_open", "region_chunk", "region_close"}:
                raise ResponseParseError(f"Unknown Transformers response event {event_type!r}.")
        return deltas


def _normalize_parsed_message(parsed: Any, *, parse_tools: bool) -> ParsedModelResponse:
    if not isinstance(parsed, dict):
        raise ResponseParseError(
            f"Transformers response parser returned {type(parsed).__name__}; expected a message object."
        )
    parsed = dict(parsed)
    role = parsed.pop("role", "assistant")
    if role != "assistant":
        raise ResponseParseError(f"Parsed response role must be 'assistant', got {role!r}.")
    reasoning = None
    for key in ("thinking", "reasoning_content", "reasoning"):
        if key in parsed:
            reasoning = parsed.pop(key)
            break
    content = parsed.pop("content", "")
    raw_tool_calls = parsed.pop("tool_calls", [])
    if raw_tool_calls and not parse_tools:
        raise ResponseParseError("Model returned tool calls when the request did not enable tools.")
    unexpected = {key: value for key, value in parsed.items() if value not in (None, "", [], {})}
    if unexpected:
        raise ResponseParseError(
            f"Unsupported parsed response fields: {sorted(unexpected)}."
        )
    if reasoning is not None and not isinstance(reasoning, str):
        raise ResponseParseError("Parsed reasoning content must be a string.")
    if not isinstance(content, str):
        raise ResponseParseError("Parsed assistant content must be a string.")
    if not isinstance(raw_tool_calls, list):
        raise ResponseParseError("Parsed tool_calls must be a list.")
    return ParsedModelResponse(
        reasoning_content=reasoning,
        content=content,
        tool_calls=[_normalize_tool_call(tool_call) for tool_call in raw_tool_calls],
    )


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise ResponseParseError("Parsed tool call must be an object.")
    function = tool_call.get("function", tool_call)
    if not isinstance(function, dict):
        raise ResponseParseError("Parsed tool call function must be an object.")
    name = function.get("name")
    arguments = function.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        raise ModelOutputParseError("Parsed tool call requires a non-empty function name.")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
