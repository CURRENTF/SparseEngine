"""Harness-owned tool-body spans; the engine receives only token IDs and ranges."""
from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4


def tool_result_ranges(
    chat: dict[str, Any],
    tokenizer: Any,
    render: Callable[[dict[str, Any]], str],
    *,
    block_size: int,
    usable_tokens: int,
    message_start: int = 0,
    message_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if block_size <= 0 or usable_tokens < 0:
        raise ValueError("Invalid prefix-cache block size or usable token count.")
    prompt = render(chat)
    if not 0 <= message_start <= len(chat["messages"]):
        raise ValueError("Invalid tool message cursor.")
    if message_indices is not None and any(
        index < message_start or index >= len(chat["messages"])
        for index in message_indices
    ):
        raise ValueError("Invalid selected tool message index.")
    selected_indices = set(message_indices) if message_indices is not None else None
    marked_chat = {**chat, "messages": list(chat["messages"])}
    markers = []
    nonce = uuid4().hex
    for index in range(message_start, len(marked_chat["messages"])):
        message = marked_chat["messages"][index]
        if message.get("role") != "tool" or (
            selected_indices is not None and index not in selected_indices
        ):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Tool-result pruning requires string tool content.")
        core = content.strip()
        if not core:
            continue
        left = len(content) - len(content.lstrip())
        right = len(content.rstrip())
        begin, end = f"__prune_{nonce}_{index}_begin__", f"__prune_{nonce}_{index}_end__"
        if begin in prompt or end in prompt:
            raise ValueError("Tool-span marker collision.")
        # Leave exterior whitespace outside the markers so template trim filters
        # retain their original behavior. It is conservatively left unpruned.
        marked_chat["messages"][index] = {
            **message, "content": content[:left] + begin + core + end + content[right:],
        }
        markers.append((begin, end))
    marked = render(marked_chat) if markers else prompt
    locations = []
    for begin, end in markers:
        if marked.count(begin) != 1 or marked.count(end) != 1:
            raise ValueError("Chat template does not preserve tool-body markers uniquely.")
        start, stop = marked.index(begin), marked.index(end)
        if stop < start + len(begin):
            raise ValueError("Chat template reordered tool-body markers.")
        locations.append((start, stop, begin, end))
    pieces, char_ranges = [], []
    cursor = clean_length = 0
    for start, stop, begin, end in sorted(locations):
        if start < cursor:
            raise ValueError("Chat template produced overlapping tool bodies.")
        prefix = marked[cursor:start]
        body = marked[start + len(begin):stop]
        pieces.extend((prefix, body))
        clean_length += len(prefix)
        char_ranges.append((clean_length, clean_length + len(body)))
        clean_length += len(body)
        cursor = stop + len(end)
    pieces.append(marked[cursor:])
    if "".join(pieces) != prompt:
        raise ValueError("Tool-span annotation changed the rendered prompt; refusing to prune.")
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is not None and not prompt.startswith(str(bos))
    encoded = tokenizer(prompt, add_special_tokens=add_special, return_offsets_mapping=True)
    token_ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(token_ids):
        raise ValueError("Tokenizer offsets do not match the rendered token sequence.")
    if usable_tokens > len(token_ids):
        raise ValueError("Local tokenizer is shorter than the server's usable prefix.")
    # Only tokens wholly inside one body qualify, including at BPE boundaries.
    runs = []
    # Fast tokenizer offsets can include zero-width special tokens; preserve
    # their order and inspect only the suffix containing newly annotated bodies.
    first_char = char_ranges[0][0] if char_ranges else len(prompt)
    first_token = next((i for i, (_, end) in enumerate(offsets) if end > first_char), len(offsets))
    span_index = 0
    for index in range(first_token, usable_tokens):
        start, stop = offsets[index]
        if stop <= start:
            continue
        while span_index < len(char_ranges) and start >= char_ranges[span_index][1]:
            span_index += 1
        if span_index == len(char_ranges):
            break
        left, right = char_ranges[span_index]
        if left <= start < stop <= right:
            if runs and runs[-1][1] == index:
                runs[-1] = (runs[-1][0], index + 1)
            else:
                runs.append((index, index + 1))
    ranges = []
    for left, right in runs:
        left = ((left + block_size - 1) // block_size) * block_size
        right = (right // block_size) * block_size
        if left < right:
            ranges.append((left, right))
    return {
        "token_ids": token_ids,
        "ranges": ranges,
        "tool_tokens": sum(right - left for left, right in runs),
        "eligible_tokens": sum(right - left for left, right in ranges),
    }


class ToolResultPruneSelector:
    def __init__(self, tokenizer_path: str):
        from transformers import AutoTokenizer
        from sparseengine.entrypoints.openai.reasoning import detect_reasoning_capabilities

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError("Tool-result pruning requires a fast tokenizer with offsets.")
        self.capabilities = detect_reasoning_capabilities(self.tokenizer)

    def _render(self, chat: dict[str, Any]) -> str:
        from sparseengine.entrypoints.openai.protocol.chat import ChatCompletionRequest
        from sparseengine.entrypoints.openai.render import _chat_request_prompt

        return _chat_request_prompt(
            self.tokenizer, ChatCompletionRequest.model_validate(chat), self.capabilities,
        )

    def select(self, chat: dict[str, Any], *, block_size: int, usable_tokens: int,
               message_start: int = 0, message_indices: tuple[int, ...] | None = None):
        return tool_result_ranges(
            chat, self.tokenizer, self._render,
            block_size=block_size, usable_tokens=usable_tokens,
            message_start=message_start, message_indices=message_indices,
        )
