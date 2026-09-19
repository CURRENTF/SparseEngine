from collections.abc import Iterable
from dataclasses import dataclass


DEFAULT_MAX_TOKENS = 8192


def _as_eos_token_id_set(
    value: int | Iterable[int] | None,
) -> frozenset[int]:
    if value is None:
        return frozenset()
    if isinstance(value, int):
        return frozenset({int(value)})
    return frozenset(int(token_id) for token_id in value)


def resolve_eos_token_ids(
    request_eos_token_ids: int | Iterable[int] | None = (),
    configured_eos_token_ids: int | Iterable[int] | None = (),
    *,
    fallback_eos_token_id: int | None = -1,
) -> frozenset[int]:
    requested = _as_eos_token_id_set(request_eos_token_ids)
    if requested:
        return requested
    configured = _as_eos_token_id_set(configured_eos_token_ids)
    if configured:
        return configured
    if (
        fallback_eos_token_id is not None
        and int(fallback_eos_token_id) >= 0
    ):
        return frozenset({int(fallback_eos_token_id)})
    return frozenset()


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    ignore_eos: bool = False
    eos_token_ids: int | list[int] | tuple[int, ...] | None = None
    logprobs: int | None = None

    def __post_init__(self):
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if not self.repetition_penalty > 0.0:
            raise ValueError("repetition_penalty must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.eos_token_ids is None:
            self.eos_token_ids = ()
        elif isinstance(self.eos_token_ids, int):
            self.eos_token_ids = (int(self.eos_token_ids),)
        else:
            self.eos_token_ids = tuple(dict.fromkeys(int(token_id) for token_id in self.eos_token_ids))
        if any(token_id < 0 for token_id in self.eos_token_ids):
            raise ValueError("eos_token_ids must contain non-negative token ids")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError("logprobs must be non-negative")
