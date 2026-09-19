from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from sparseengine.sampling_params import DEFAULT_MAX_TOKENS


class ResponseReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    summary: str | None = None


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    chain_id: str | None = None
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    stream: bool = False
    store: bool = False
    prompt_cache_key: str | None = Field(default=None, min_length=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: ResponseReasoning | None = None
    chat_template_kwargs: Any = None
