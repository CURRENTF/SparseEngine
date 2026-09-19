from __future__ import annotations

__all__ = ["LLM", "MultiModalPrompt", "SamplingParams"]


def __getattr__(name: str):
    if name == "LLM":
        from sparseengine.llm import LLM

        return LLM
    if name == "SamplingParams":
        from sparseengine.sampling_params import SamplingParams

        return SamplingParams
    if name == "MultiModalPrompt":
        from sparseengine.multimodal import MultiModalPrompt

        return MultiModalPrompt
    raise AttributeError(f"module 'sparseengine' has no attribute {name!r}")
