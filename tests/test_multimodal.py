import base64
import io
import pickle
import wave
from types import SimpleNamespace

import pytest
import torch

from sparseengine.engine.sequence import Sequence
from sparseengine.engine.llm_engine import LLMEngine
from sparseengine.multimodal.inputs import (
    MultiModalInputProcessor,
    ProcessedMultiModalPrompt,
    normalize_messages,
)
from sparseengine.multimodal.runtime import MultiModalRuntime, MultiModalState
from sparseengine.models.qwen3_5_multimodal import qwen35_mrope_positions
from sparseengine.operators.qwen35_mrope import Qwen35MRotaryEmbedding
from sparseengine.sampling_params import SamplingParams
from sparseengine.utils.context import get_context, set_context


def test_normalize_openai_multimodal_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://x/image.png"}},
                {"type": "video_url", "video_url": "https://x/video.mp4"},
            ],
        }
    ]
    assert normalize_messages(messages)[0]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image", "image": "https://x/image.png"},
        {"type": "video", "video": "https://x/video.mp4"},
    ]


def test_normalize_openai_wav_audio_without_optional_dependencies():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(torch.tensor([-32768, 0, 32767], dtype=torch.int16).numpy().tobytes())

    part = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "format": "wav",
                            "data": base64.b64encode(buffer.getvalue()).decode(),
                        },
                    }
                ],
            }
        ]
    )[0]["content"][0]

    assert part["type"] == "audio" and part["sampling_rate"] == 16_000
    torch.testing.assert_close(
        torch.from_numpy(part["audio"]),
        torch.tensor([-1.0, 0.0, 32767 / 32768]),
    )


def test_normalize_openai_audio_rejects_invalid_base64_wav():
    with pytest.raises(ValueError, match="valid base64 WAV"):
        normalize_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"format": "wav", "data": "not-base64"},
                        }
                    ],
                }
            ]
        )


def test_multimodal_processor_returns_stable_cpu_payload():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][1]["type"] == "image"
            assert kwargs["tokenize"] and kwargs["return_dict"]
            return {
                "input_ids": torch.tensor([[7, 8, 9]]),
                "attention_mask": torch.ones(1, 3),
                "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
                "pixel_values": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            }

    processor = object.__new__(MultiModalInputProcessor)
    processor.processor = Processor()
    prompt = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x"},
                    {"type": "image_url", "image_url": "https://x/image.png"},
                ],
            }
        ]
    }

    first = processor.process(prompt)
    second = processor.process(prompt)

    assert first.token_ids == [7, 8, 9]
    assert first.digest == second.digest
    assert set(first.tensors) == {"mm_token_type_ids", "pixel_values"}
    assert all(tensor.device.type == "cpu" and tensor.is_contiguous() for tensor in first.tensors.values())


def test_qwen35_mrope_positions_match_image_grid():
    positions, delta = qwen35_mrope_positions(
        torch.tensor([0, 0, 1, 1, 1, 1, 0]),
        torch.tensor([[1, 4, 4]]),
        None,
        spatial_merge_size=2,
    )

    assert positions.tolist() == [
        [0, 1, 2, 2, 2, 2, 4],
        [0, 1, 2, 2, 3, 3, 4],
        [0, 1, 2, 3, 2, 3, 4],
    ]
    assert delta == -2


def test_qwen35_multimodal_rope_matches_transformers():
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    torch.manual_seed(7)
    positions = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 6, 7]], dtype=torch.long
    )
    query = torch.randn(4, 2, 128)
    key = torch.randn(4, 1, 128)
    rope = Qwen35MRotaryEmbedding(128, 64, 32, 10_000, [11, 11, 10])

    actual_q, actual_k = rope(positions, query, key)
    inv_freq = 1 / 10_000 ** (torch.arange(0, 64, 2).float() / 64)
    freqs = positions[:, :, None].float() * inv_freq
    merged = freqs[0].clone()
    merged[:, 1:33:3] = freqs[1, :, 1:33:3]
    merged[:, 2:30:3] = freqs[2, :, 2:30:3]
    cos, sin = torch.cat((merged, merged), -1).cos(), torch.cat((merged, merged), -1).sin()
    expected_q, expected_k = apply_rotary_pos_emb(
        query.unsqueeze(0), key.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0), unsqueeze_dim=2
    )

    torch.testing.assert_close(actual_q, expected_q[0])
    torch.testing.assert_close(actual_k, expected_k[0])


def test_multimodal_runtime_replaces_chunk_features_and_tracks_vision_groups():
    type_ids = torch.tensor([0, 1, 1, 2, 2, 0])
    state = MultiModalState(
        type_ids=type_ids,
        embeddings={
            1: torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
            2: torch.tensor([[20.0, 21.0], [22.0, 23.0]]),
        },
        position_ids=torch.arange(18).reshape(3, 6),
        position_delta=-1,
    )

    class Model:
        multimodal_bidirectional = True

        def encode_multimodal(self, input_ids, tensors):
            assert input_ids == list(range(6)) and tensors == {}
            return state

        def embed_input_ids(self, input_ids):
            return input_ids.float().unsqueeze(1).expand(-1, 2).clone()

    runtime = MultiModalRuntime(Model(), torch.device("cpu"))
    assert runtime.register(3, list(range(6)), {}) == -1
    seq = Sequence(list(range(6)), SamplingParams(max_tokens=1))
    seq.seq_id = 3
    seq.current_chunk_size = 6
    set_context(True)

    embeds, positions, mask = runtime.prepare(
        [seq], torch.arange(6), torch.arange(6), is_prefill=True
    )

    assert embeds.tolist() == [
        [0.0, 0.0],
        [10.0, 11.0],
        [12.0, 13.0],
        [20.0, 21.0],
        [22.0, 23.0],
        [5.0, 5.0],
    ]
    assert positions.equal(state.position_ids)
    assert mask.tolist() == [False, True, True, True, True, False]
    assert get_context().multimodal_image_groups.tolist() == [0, 1, 1, 1, 1, 0]
    runtime.free(3)
    assert runtime.states == {}


def test_multimodal_sequence_state_preserves_decode_delta():
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=2))
    seq.multimodal_digest = "digest"
    seq.multimodal_position_delta = -4
    seq.multimodal_full_prefill = True
    restored = pickle.loads(pickle.dumps(seq))

    assert restored.multimodal_digest == "digest"
    assert restored.multimodal_full_prefill
    assert restored.decode_input_position == -2


def test_abort_queued_multimodal_request_releases_encoder_state_only():
    seq = Sequence([1], SamplingParams(max_tokens=1))
    seq.multimodal_digest = "digest"
    calls = []

    class Scheduler:
        waiting = [seq]
        decoding = []

        def abort(self, seq_id):
            self.waiting.clear()
            return False

    engine = object.__new__(LLMEngine)
    engine.scheduler = Scheduler()
    engine._active_chain_sequences = {}
    engine.model_runner = SimpleNamespace(
        runtime_state=SimpleNamespace(chain_cache_coordinator=None),
        call=lambda method, *args: calls.append((method, args)),
    )

    engine.abort_request(seq.seq_id)

    assert calls == [("free_multimodal", (seq.seq_id,))]


def test_multimodal_registration_error_survives_failed_rollback():
    calls = []

    class Runner:
        def call(self, method, *args):
            calls.append(method)
            if method == "register_multimodal_shared":
                raise ValueError("rank-local encoder failure")
            raise TimeoutError("rollback timeout")

    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        hf_config=SimpleNamespace(use_bidirectional_attention=None),
        max_model_len=16,
        resolved_prefix_cache_mode="disabled",
    )
    engine.multimodal_processor = SimpleNamespace(
        process=lambda _prompt: ProcessedMultiModalPrompt(
            token_ids=[1, 2],
            tensors={"mm_token_type_ids": torch.tensor([[0, 1]])},
            digest="digest",
        )
    )
    engine.model_runner = Runner()

    with pytest.raises(ValueError, match="rank-local encoder failure"):
        engine.admit_request(
            {"messages": [{"role": "user", "content": []}]},
            SamplingParams(max_tokens=1),
        )

    assert calls == ["register_multimodal_shared", "free_multimodal"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("score_ndim", [2, 3])
def test_gemma4_multimodal_context_attention_matches_reference(score_ndim):
    from sparseengine.kernels.triton.gemma4_multimodal_context_attention import (
        gemma4_multimodal_context_attention,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    # Keep the sequence longer than two windows so late query blocks encounter
    # fully masked early key blocks. Those blocks must not poison online
    # softmax state with -inf - -inf NaNs.
    length, num_heads, head_dim, window = 193, 2, 256, 64
    q = (torch.randn(length, num_heads, head_dim, device=device) / head_dim**0.5).bfloat16()
    k = torch.randn(length, 1, head_dim, device=device).bfloat16()
    v = torch.randn_like(k)
    output = torch.empty_like(q)
    attention_score = torch.zeros(
        (1, num_heads, length) if score_ndim == 3 else (1, length),
        device=device,
        dtype=torch.float32,
    )
    groups = torch.zeros(length, device=device, dtype=torch.int32)
    groups[20:51] = 1
    gemma4_multimodal_context_attention(
        q,
        k,
        v,
        output,
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.tensor([length], device=device, dtype=torch.int32),
        torch.tensor([0], device=device, dtype=torch.int32),
        length,
        torch.arange(length, device=device, dtype=torch.int32).unsqueeze(0),
        groups,
        sliding_window=window,
        attn_score=attention_score,
    )

    scores = torch.einsum("qhd,khd->hqk", q.float(), k.expand(-1, num_heads, -1).float())
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    same_group = (groups[:, None] == groups[None, :]) & (groups[:, None] > 0)
    visible = ((key <= query) | same_group) & (key > query - window)
    visible_scores = torch.where(visible.unsqueeze(0), scores, 0)
    if score_ndim == 3:
        expected_score = visible_scores.sum(1)
    else:
        expected_score = torch.stack(
            [
                visible_scores[:, start : start + 32].sum(1) / length
                for start in range(0, length, 32)
            ]
        ).amax((0, 1)).clamp_min_(0)
    scores.masked_fill_(~visible.unsqueeze(0), float("-inf"))
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.expand(-1, num_heads, -1).float())

    torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(attention_score[0], expected_score, atol=2e-2, rtol=2e-2)
