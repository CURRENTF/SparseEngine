from dataclasses import dataclass
from typing import Any
from typing import Literal


ReasoningMode = Literal["none", "switchable", "required"]
EffortControl = Literal["none", "native"]
PROBE_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
PROBE_MESSAGES = [{"role": "user", "content": "Hello"}]


@dataclass(frozen=True)
class ReasoningCapabilities:
    mode: ReasoningMode = "none"
    effort_control: EffortControl = "none"
    supported_efforts: tuple[str, ...] = ()
    default_effort: str | None = None

NO_REASONING = ReasoningCapabilities()


def detect_reasoning_capabilities(tokenizer: Any) -> ReasoningCapabilities:
    if not getattr(tokenizer, "chat_template", None) or not hasattr(tokenizer, "apply_chat_template"):
        return NO_REASONING

    try:
        default_prompt = _render_probe(tokenizer)
    except Exception as exc:
        raise RuntimeError(f"Reasoning capability probe failed: {exc}") from exc

    enabled_prompt = _try_render_probe(tokenizer, enable_thinking=True)
    disabled_prompt = _try_render_probe(tokenizer, enable_thinking=False)
    if enabled_prompt is not None and disabled_prompt is not None and enabled_prompt != disabled_prompt:
        mode: ReasoningMode = "switchable"
    elif (
        enabled_prompt is not None and disabled_prompt is None
    ) or default_prompt.rstrip().endswith("<think>"):
        mode = "required"
    else:
        return NO_REASONING

    probe_kwargs = {"enable_thinking": True} if mode == "switchable" else {}
    rendered_efforts = {
        effort: prompt
        for effort in PROBE_EFFORTS
        if (prompt := _try_render_probe(tokenizer, reasoning_effort=effort, **probe_kwargs)) is not None
    }
    native_effort = len(set(rendered_efforts.values())) > 1
    if native_effort:
        supported_efforts = tuple(rendered_efforts)
        if mode == "switchable":
            supported_efforts = ("none", *supported_efforts)
        default_effort = next(
            (effort for effort, prompt in rendered_efforts.items() if prompt == default_prompt),
            None,
        )
        return ReasoningCapabilities(mode, "native", supported_efforts, default_effort)

    if mode == "switchable":
        default_effort = "none" if default_prompt == disabled_prompt else None
        if default_prompt == enabled_prompt:
            default_effort = "medium"
        return ReasoningCapabilities(mode, supported_efforts=("none", "medium"), default_effort=default_effort)
    return ReasoningCapabilities(mode)


def reasoning_template_kwargs(
    effort: str | None,
    capabilities: ReasoningCapabilities | None,
) -> dict[str, Any]:
    if effort is None:
        return {}
    if capabilities is None:
        return {"enable_thinking": effort != "none"}
    if effort not in capabilities.supported_efforts:
        detail = (
            f"supported values: {', '.join(capabilities.supported_efforts)}"
            if capabilities.supported_efforts
            else "the model does not accept an explicit reasoning effort"
        )
        raise ValueError(f"reasoning effort {effort!r} is not supported by this model; {detail}.")
    if effort == "none":
        return {"enable_thinking": False}
    if capabilities.effort_control == "native":
        kwargs = {"reasoning_effort": effort}
        if capabilities.mode == "switchable":
            kwargs["enable_thinking"] = True
        return kwargs
    return {"enable_thinking": True}


def _render_probe(tokenizer: Any, **kwargs: Any) -> str:
    return tokenizer.apply_chat_template(
        PROBE_MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def _try_render_probe(tokenizer: Any, **kwargs: Any) -> str | None:
    try:
        return _render_probe(tokenizer, **kwargs)
    except Exception:
        return None
