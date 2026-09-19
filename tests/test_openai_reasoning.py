import pytest

from sparseengine.entrypoints.openai.reasoning import detect_reasoning_capabilities
from sparseengine.entrypoints.openai.reasoning import reasoning_template_kwargs


class SwitchableTokenizer:
    chat_template = "template"

    def apply_chat_template(self, _messages, *, enable_thinking=True, **_kwargs):
        return f"thinking={enable_thinking}"


class RequiredNativeEffortTokenizer:
    chat_template = "template"

    def apply_chat_template(self, _messages, **kwargs):
        if kwargs.get("enable_thinking") is False:
            raise ValueError("thinking cannot be disabled")
        effort = kwargs.get("reasoning_effort", "xhigh")
        if effort not in {"low", "medium", "xhigh"}:
            raise ValueError("unsupported reasoning effort")
        return f"<think>{effort}"


class PlainTokenizer:
    chat_template = "template"

    def apply_chat_template(self, _messages, **_kwargs):
        return "assistant:"


def test_detects_switchable_reasoning_without_native_effort():
    capabilities = detect_reasoning_capabilities(SwitchableTokenizer())

    assert capabilities.mode == "switchable"
    assert capabilities.effort_control == "none"
    assert capabilities.supported_efforts == ("none", "medium")
    assert capabilities.default_effort == "medium"
    assert reasoning_template_kwargs("none", capabilities) == {"enable_thinking": False}
    assert reasoning_template_kwargs("medium", capabilities) == {"enable_thinking": True}
    with pytest.raises(ValueError, match="supported values: none, medium"):
        reasoning_template_kwargs("high", capabilities)


def test_detects_required_native_effort_from_template_behavior():
    capabilities = detect_reasoning_capabilities(RequiredNativeEffortTokenizer())

    assert capabilities.mode == "required"
    assert capabilities.effort_control == "native"
    assert capabilities.supported_efforts == ("low", "medium", "xhigh")
    assert capabilities.default_effort == "xhigh"
    assert reasoning_template_kwargs("low", capabilities) == {"reasoning_effort": "low"}
    with pytest.raises(ValueError, match="reasoning effort 'none'"):
        reasoning_template_kwargs("none", capabilities)


def test_ignores_unknown_template_kwargs_when_detecting_capabilities():
    capabilities = detect_reasoning_capabilities(PlainTokenizer())

    assert capabilities.mode == "none"
    with pytest.raises(ValueError, match="does not accept an explicit reasoning effort"):
        reasoning_template_kwargs("none", capabilities)
