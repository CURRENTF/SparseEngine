from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from tokenizers.decoders import DecodeStream


@dataclass(frozen=True)
class DecodedDelta:
    text: str
    raw_text: str


@dataclass(frozen=True)
class DecodedFinal:
    text: str
    raw_text: str
    text_delta: str
    raw_text_delta: str


@dataclass(frozen=True)
class _VisibleFragment:
    visible_start: int
    text: str
    raw_start: int
    raw_text: str

    @property
    def visible_end(self) -> int:
        return self.visible_start + len(self.text)

    def raw_offset(self, visible_offset: int) -> int:
        relative = visible_offset - self.visible_start
        if not 0 <= relative < len(self.text):
            raise ValueError(
                f"Visible offset {visible_offset} is outside fragment "
                f"[{self.visible_start}, {self.visible_end})."
            )
        if self.text == self.raw_text:
            return self.raw_start + relative
        if not self.raw_text:
            return self.raw_start

        first = self.raw_text.find(self.text)
        if first >= 0 and self.raw_text.find(self.text, first + 1) < 0:
            return self.raw_start + first + relative
        raise RuntimeError(
            "Visible and raw decode fragments cannot be aligned: "
            f"visible={self.text!r} raw={self.raw_text!r}."
        )


class IncrementalDetokenizer:
    def __init__(self, tokenizer: Any):
        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        if not getattr(tokenizer, "is_fast", False) or backend_tokenizer is None:
            raise TypeError(
                "OpenAI serving requires a fast tokenizer backend with "
                f"DecodeStream support; got {type(tokenizer).__name__}."
            )
        self.tokenizer = tokenizer
        self.backend_tokenizer = backend_tokenizer
        self.visible_stream = DecodeStream(skip_special_tokens=True)
        self.raw_stream = DecodeStream(skip_special_tokens=False)
        self.token_ids: list[int] = []
        self.text = ""
        self.raw_text = ""
        self._visible_fragments: list[_VisibleFragment] = []
        self._visible_fragment_ends: list[int] = []
        self.finished = False

    def _record_fragment(self, text: str, raw_text: str) -> None:
        if not text:
            return
        fragment = _VisibleFragment(
            visible_start=len(self.text),
            text=text,
            raw_start=len(self.raw_text),
            raw_text=raw_text,
        )
        self._visible_fragments.append(fragment)
        self._visible_fragment_ends.append(fragment.visible_end)

    def raw_offset_for_visible_prefix(
        self,
        visible_text_len: int,
        *,
        raw_text_limit: int | None = None,
    ) -> int:
        if not 0 <= visible_text_len <= len(self.text):
            raise ValueError(
                f"Visible prefix length {visible_text_len} is outside "
                f"[0, {len(self.text)}]."
            )
        if raw_text_limit is None:
            raw_text_limit = len(self.raw_text)
        if not 0 <= raw_text_limit <= len(self.raw_text):
            raise ValueError(
                f"Raw text limit {raw_text_limit} is outside "
                f"[0, {len(self.raw_text)}]."
            )

        fragment_index = bisect_right(
            self._visible_fragment_ends,
            visible_text_len,
        )
        if fragment_index == len(self._visible_fragments):
            return raw_text_limit
        raw_offset = self._visible_fragments[fragment_index].raw_offset(
            visible_text_len
        )
        return min(raw_offset, raw_text_limit)

    def push(self, token_ids: list[int]) -> DecodedDelta:
        if self.finished:
            raise RuntimeError("Cannot push token IDs after incremental detokenization finished.")

        text_parts: list[str] = []
        raw_text_parts: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            self.token_ids.append(token_id)
            text = (
                self.visible_stream.step(self.backend_tokenizer, token_id)
                or ""
            )
            raw_text = (
                self.raw_stream.step(self.backend_tokenizer, token_id)
                or ""
            )
            self._record_fragment(text, raw_text)
            if text:
                text_parts.append(text)
            if raw_text:
                raw_text_parts.append(raw_text)
            self.text += text
            self.raw_text += raw_text

        text_delta = "".join(text_parts)
        raw_text_delta = "".join(raw_text_parts)
        return DecodedDelta(text=text_delta, raw_text=raw_text_delta)

    def finish(self, token_ids: list[int]) -> DecodedFinal:
        if self.finished:
            raise RuntimeError("Incremental detokenization already finished.")

        final_token_ids = [int(token_id) for token_id in token_ids]
        observed = len(self.token_ids)
        if len(final_token_ids) < observed or final_token_ids[:observed] != self.token_ids:
            raise RuntimeError(
                "Incremental detokenization token history mismatch: "
                f"observed={self.token_ids!r} final={final_token_ids!r}."
            )
        pushed_text_delta = ""
        pushed_raw_text_delta = ""
        if len(final_token_ids) > observed:
            pushed = self.push(final_token_ids[observed:])
            pushed_text_delta = pushed.text
            pushed_raw_text_delta = pushed.raw_text

        final_text = self.tokenizer.decode(final_token_ids, skip_special_tokens=True)
        final_raw_text = self.tokenizer.decode(final_token_ids, skip_special_tokens=False)
        if not final_text.startswith(self.text):
            raise RuntimeError(
                "Incremental visible text is not a prefix of canonical final text: "
                f"incremental={self.text!r} final={final_text!r}."
            )
        if not final_raw_text.startswith(self.raw_text):
            raise RuntimeError(
                "Incremental raw text is not a prefix of canonical final text: "
                f"incremental={self.raw_text!r} final={final_raw_text!r}."
            )

        final_text_suffix = final_text[len(self.text):]
        final_raw_text_suffix = final_raw_text[len(self.raw_text):]
        self._record_fragment(final_text_suffix, final_raw_text_suffix)
        text_delta = pushed_text_delta + final_text_suffix
        raw_text_delta = pushed_raw_text_delta + final_raw_text_suffix
        self.text = final_text
        self.raw_text = final_raw_text
        self.finished = True
        return DecodedFinal(
            text=final_text,
            raw_text=final_raw_text,
            text_delta=text_delta,
            raw_text_delta=raw_text_delta,
        )
