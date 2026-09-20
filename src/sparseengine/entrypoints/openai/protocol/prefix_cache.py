from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict, StrictInt, model_validator

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
    range_start: StrictInt | None = None
    range_end: StrictInt | None = None
    ranges: list[tuple[StrictInt, StrictInt]] | None = None
    keep_tokens: StrictInt
    policy: Literal["snapkv_global", "kvzip_global"]
    allow_recompress: bool = False
    observation_tokens: int = 64
    score_chunk_size: int = 2048
    prev_postfix_size: int = 64

    @model_validator(mode="after")
    def validate_range_selector(self):
        if self.ranges is not None:
            if not self.ranges or self.range_start is not None or self.range_end is not None:
                raise ValueError("provide non-empty ranges OR range_start/range_end, not both")
        elif self.range_start is None or self.range_end is None:
            raise ValueError("provide ranges or both range_start and range_end")
        return self
