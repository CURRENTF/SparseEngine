from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from sparseengine.sampling_params import DEFAULT_MAX_TOKENS


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    chain_id: str | None = None
    prompt: str | list[int] | list[str] | list[list[int]]
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    ignore_eos: bool = False
    stop: str | list[str] | None = None
    logprobs: int | None = Field(default=None, ge=0, le=5)
