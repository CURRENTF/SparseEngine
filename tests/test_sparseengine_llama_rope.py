import types
import unittest

from sparseengine.models.rope import resolve_rope_scaling, resolve_rope_theta


class SparseVLLMLlamaRopeTest(unittest.TestCase):
    def test_llama3_rope_parameters_are_used(self):
        config = types.SimpleNamespace(
            rope_parameters={
                "rope_theta": 500000.0,
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            }
        )

        self.assertEqual(resolve_rope_theta(config), 500000.0)
        scaling = dict(resolve_rope_scaling(config, model_name="Llama"))
        self.assertEqual(scaling["rope_type"], "llama3")
        self.assertEqual(scaling["factor"], 8.0)
        self.assertEqual(scaling["original_max_position_embeddings"], 8192)


if __name__ == "__main__":
    unittest.main()
