from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict

from sparseengine.entrypoints.openai.protocol.chat import ChatMessage


class PrefixCacheInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int] | None = None
    text: str | None = None
    include_subtree: bool = False


class PrefixCacheMatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int] | None = None
    text: str | None = None
    messages: list[ChatMessage] | None = None
    chat: dict[str, Any] | None = None
    response: dict[str, Any] | None = None


class PrefixCacheDeleteSubtreeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int] | None = None
    text: str | None = None


class PrefixCacheSetEvictionPriorityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int] | None = None
    text: str | None = None
    priority: int


class PrefixCachePruneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int] | None = None
    text: str | None = None
    chat: dict[str, Any] | None = None
    range_start: int
    range_end: int
    keep_tokens: int
    policy: Literal["snapkv_global", "kvzip_global"]
    allow_recompress: bool = False
    observation_tokens: int = 64
    score_chunk_size: int = 2048
    prev_postfix_size: int = 64
