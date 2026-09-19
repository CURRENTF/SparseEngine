from __future__ import annotations

import base64
import binascii
import hashlib
import io
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor


@dataclass(frozen=True)
class MultiModalPrompt:
    messages: list[dict[str, Any]]
    chat_template_kwargs: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    add_generation_prompt: bool = True


@dataclass(frozen=True)
class ProcessedMultiModalPrompt:
    token_ids: list[int]
    tensors: dict[str, torch.Tensor]
    digest: str


def is_multimodal_prompt(prompt: object) -> bool:
    return isinstance(prompt, MultiModalPrompt) or (
        isinstance(prompt, dict) and "messages" in prompt
    )


def _audio_part(part: dict[str, Any]) -> dict[str, Any]:
    audio = part.get("input_audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
        raise TypeError("input_audio requires a base64 data string.")
    if str(audio.get("format", "wav")).lower() != "wav":
        raise ValueError("Only WAV input_audio is supported.")
    try:
        encoded = base64.b64decode(audio["data"], validate=True)
        with wave.open(io.BytesIO(encoded), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise ValueError("Compressed WAV input_audio is unsupported.")
            channels, sample_width, sampling_rate = (
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getframerate(),
            )
            raw = wav.readframes(wav.getnframes())
    except (binascii.Error, wave.Error) as exc:
        raise ValueError("input_audio data must contain a valid base64 WAV file.") from exc
    if sample_width == 1:
        waveform = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif sample_width in {2, 4}:
        dtype = np.dtype(f"<i{sample_width}")
        waveform = np.frombuffer(raw, dtype).astype(np.float32) / float(1 << (8 * sample_width - 1))
    elif sample_width == 3:
        values = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        samples = values[:, 0] | values[:, 1] << 8 | values[:, 2] << 16
        waveform = ((samples ^ (1 << 23)) - (1 << 23)).astype(np.float32) / (1 << 23)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    return {
        "type": "audio",
        "audio": np.asarray(waveform, dtype=np.float32),
        "sampling_rate": int(sampling_rate),
    }


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("Multimodal messages must be dictionaries.")
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(dict(message))
            continue
        parts = []
        for raw_part in content:
            part = dict(raw_part)
            part_type = part.get("type")
            if part_type in {"image_url", "input_image"}:
                value = part.get("image_url")
                url = value.get("url") if isinstance(value, dict) else value
                if not isinstance(url, str):
                    raise TypeError("image_url requires a URL string.")
                part = {"type": "image", "image": url}
            elif part_type in {"video_url", "input_video"}:
                value = part.get("video_url")
                url = value.get("url") if isinstance(value, dict) else value
                if not isinstance(url, str):
                    raise TypeError("video_url requires a URL string.")
                part = {"type": "video", "video": url}
            elif part_type == "input_audio":
                part = _audio_part(part)
            parts.append(part)
        normalized.append({**message, "content": parts})
    return normalized


def _tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class MultiModalInputProcessor:
    def __init__(self, model_path: str) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

    def process(
        self, prompt: MultiModalPrompt | dict[str, Any]
    ) -> ProcessedMultiModalPrompt:
        messages = prompt.messages if isinstance(prompt, MultiModalPrompt) else prompt.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("A multimodal prompt requires non-empty messages.")
        outputs = self.processor.apply_chat_template(
            normalize_messages(messages),
            tokenize=True,
            add_generation_prompt=(
                prompt.add_generation_prompt
                if isinstance(prompt, MultiModalPrompt)
                else True
            ),
            return_dict=True,
            return_tensors="pt",
            **(
                {
                    **(prompt.chat_template_kwargs or {}),
                    **({"tools": prompt.tools} if prompt.tools else {}),
                }
                if isinstance(prompt, MultiModalPrompt)
                else {}
            ),
        )
        input_ids = outputs.pop("input_ids")
        outputs.pop("attention_mask", None)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"Multimodal processor must return one input sequence, got {tuple(input_ids.shape)}."
            )
        tensors = {
            str(name): value.detach().cpu().contiguous()
            for name, value in outputs.items()
            if isinstance(value, torch.Tensor)
        }
        if not tensors or "mm_token_type_ids" not in tensors:
            raise ValueError("The processor did not return multimodal tensors.")
        return ProcessedMultiModalPrompt(
            token_ids=[int(token_id) for token_id in input_ids[0].tolist()],
            tensors=tensors,
            digest=_tensor_digest(tensors),
        )


__all__ = [
    "MultiModalInputProcessor",
    "MultiModalPrompt",
    "ProcessedMultiModalPrompt",
    "is_multimodal_prompt",
    "normalize_messages",
]
