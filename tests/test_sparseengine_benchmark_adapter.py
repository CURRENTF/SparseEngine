import unittest
from unittest.mock import patch

from benchmark.model_adapters.sparseengine import get_sparseengine_generate_api


class _FakeLLM:
    instances = []

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.calls = []
        self.__class__.instances.append(self)

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [{"text": f"generated:{prompt}"} for prompt in prompts]


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class SparseVLLMBenchmarkAdapterTest(unittest.TestCase):
    def setUp(self):
        _FakeLLM.instances.clear()

    def test_constructs_native_engine_with_public_runtime_params(self):
        with (
            patch("sparseengine.LLM", _FakeLLM),
            patch("sparseengine.SamplingParams", _FakeSamplingParams),
        ):
            generate = get_sparseengine_generate_api(
                "/models/qwen",
                {"decode_keep_tokens": 64},
                sparse_method="deltakv",
                deltakv_checkpoint_path="/checkpoints/compressor",
            )

        llm = _FakeLLM.instances[0]
        self.assertEqual(llm.model_path, "/models/qwen")
        self.assertEqual(
            llm.kwargs,
            {
                "decode_keep_tokens": 64,
                "sparse_method": "deltakv",
                "deltakv_checkpoint_path": "/checkpoints/compressor",
            },
        )
        self.assertIs(generate._sparseengine_llm, llm)

    def test_generates_single_and_batched_text(self):
        with (
            patch("sparseengine.LLM", _FakeLLM),
            patch("sparseengine.SamplingParams", _FakeSamplingParams),
        ):
            generate = get_sparseengine_generate_api("/models/qwen", {})
            single = generate("one", max_new_tokens=3, do_sample=False)
            batch = generate(["one", "two"], max_tokens=4, temperature=0.7)

        self.assertEqual(single, "generated:one")
        self.assertEqual(batch, ["generated:one", "generated:two"])
        llm = _FakeLLM.instances[0]
        self.assertEqual(llm.calls[0][1].kwargs["temperature"], 0.0)
        self.assertEqual(llm.calls[0][1].kwargs["max_tokens"], 3)
        self.assertEqual(llm.calls[1][1].kwargs["temperature"], 0.7)
        self.assertEqual(llm.calls[1][1].kwargs["max_tokens"], 4)

    def test_forwards_final_prompt_token_ids_without_retokenizing(self):
        with (
            patch("sparseengine.LLM", _FakeLLM),
            patch("sparseengine.SamplingParams", _FakeSamplingParams),
        ):
            generate = get_sparseengine_generate_api("/models/qwen", {})
            batch = generate([[1, 2], [3, 4]], max_tokens=1)

        self.assertEqual(batch, ["generated:[1, 2]", "generated:[3, 4]"])
        self.assertEqual(_FakeLLM.instances[0].calls[0][0], [[1, 2], [3, 4]])

    def test_rejects_conflicting_or_unsupported_contracts(self):
        with (
            patch("sparseengine.LLM", _FakeLLM),
            patch("sparseengine.SamplingParams", _FakeSamplingParams),
        ):
            with self.assertRaisesRegex(ValueError, "Conflicting sparse_method"):
                get_sparseengine_generate_api(
                    "/models/qwen",
                    {"sparse_method": "omnikv"},
                    sparse_method="deltakv",
                )
            with self.assertRaisesRegex(ValueError, "use_cache=True"):
                get_sparseengine_generate_api("/models/qwen", {}, use_cache=False)

            generate = get_sparseengine_generate_api("/models/qwen", {})
            with self.assertRaisesRegex(ValueError, "external past_key_values"):
                generate("one", past_key_values=object())


if __name__ == "__main__":
    unittest.main()
