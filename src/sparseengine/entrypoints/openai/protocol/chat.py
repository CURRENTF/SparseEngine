from typing import Any
from typing import Literal

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from sparseengine.sampling_params import DEFAULT_MAX_TOKENS


class ChatContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image_url", "video_url", "input_audio"]
    text: str | None = None
    image_url: str | dict[str, Any] | None = None
    video_url: str | dict[str, Any] | None = None
    input_audio: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_content(self):
        value = getattr(self, self.type)
        if value is None:
            raise ValueError(f"{self.type} content requires its matching field.")
        return self


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: str | list[ChatContentPart] | None = None
    reasoning_content: str | None = Field(
        default=None,
        validation_alias=AliasChoices("reasoning_content", "reasoning"),
    )
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_role_specific_fields(self):
        if self.reasoning_content is not None and self.role != "assistant":
            raise ValueError("reasoning_content is only valid for assistant messages.")
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls is only valid for assistant messages.")
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError("tool_call_id is only valid for tool messages.")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id.")
        if self.role == "tool" and self.content is None:
            raise ValueError("tool messages require content.")
        for tool_call in self.tool_calls or []:
            _validate_message_tool_call(tool_call)
        return self


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    chain_id: str | None = None
    chain_append_start: int | None = Field(default=None, ge=1)
    messages: list[ChatMessage]
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    store: bool = False
    ignore_eos: bool = False
    benchmark_forced_token_ids: list[int] | None = None
    stop: str | list[str] | None = None
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    chat_template_kwargs: Any = None

    @model_validator(mode="after")
    def validate_benchmark_forced_tokens(self):
        if self.benchmark_forced_token_ids is not None and not self.ignore_eos:
            raise ValueError("benchmark_forced_token_ids requires ignore_eos=true")
        return self


def _validate_message_tool_call(tool_call: dict[str, Any]):
    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("assistant tool_calls entries require a non-empty id.")
    if tool_call.get("type") != "function":
        raise ValueError("assistant tool_calls only supports type='function'.")
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ValueError("assistant tool_calls entries require a function object.")
    if not isinstance(function.get("name"), str) or not function["name"]:
        raise ValueError("assistant tool_calls function.name must be a non-empty string.")
    if not isinstance(function.get("arguments"), str):
        raise ValueError("assistant tool_calls function.arguments must be a string.")
