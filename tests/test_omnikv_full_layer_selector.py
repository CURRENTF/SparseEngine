import random
import typing
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sparseengine.utils.select_omnikv_full_layers import (
    CalibrationPoint,
    add_topk_to_pair_scores,
    attention_layer_indices_from_config,
    compute_segment_scores,
    install_typing_compatibility,
    model_input_device,
    parse_args,
    prepare_fp8_transformers_config,
    sample_decode_points,
    select_full_layers_dp,
    selected_segment_breakdown,
)


class OmniKVFullLayerSelectorTest(unittest.TestCase):
    def test_cli_maps_legacy_token_flags_to_runtime_field_names(self):
        args = parse_args(
            [
                "--model-path",
                "/model",
                "--longbench-root",
                "/longbench",
                "--num-sink-tokens",
                "8",
                "--num-recent-tokens",
                "128",
            ]
        )

        self.assertEqual(args.sink_keep_tokens, 8)
        self.assertEqual(args.recent_keep_tokens, 128)

    def test_remote_model_code_can_be_disabled(self):
        args = parse_args(
            [
                "--model-path",
                "/model",
                "--longbench-root",
                "/longbench",
                "--no-trust-remote-code",
            ]
        )

        self.assertFalse(args.trust_remote_code)

    def test_python310_typing_compatibility_is_idempotent(self):
        had_unpack = hasattr(typing, "Unpack")

        first = install_typing_compatibility()
        second = install_typing_compatibility()

        self.assertTrue(hasattr(typing, "Unpack"))
        self.assertEqual(second, [])
        self.assertEqual(first, [] if had_unpack else ["typing.Unpack"])

    def test_model_input_device_uses_embedding_weight_device(self):
        class FakeModel:
            def get_input_embeddings(self):
                return SimpleNamespace(weight=torch.empty(1, device="cpu"))

        self.assertEqual(
            model_input_device(FakeModel(), torch.device("cuda:0")),
            torch.device("cpu"),
        )

    def test_add_topk_to_pair_scores_counts_forward_layer_intersections(self):
        pair_scores = np.zeros((4, 4), dtype=np.int64)
        add_topk_to_pair_scores(
            pair_scores,
            [
                [1, 2, 3],
                [2, 3, 4],
                [5, 6, 7],
                [1, 3, 7],
            ],
        )

        self.assertEqual(pair_scores[0, 1], 2)
        self.assertEqual(pair_scores[0, 2], 0)
        self.assertEqual(pair_scores[0, 3], 2)
        self.assertEqual(pair_scores[1, 3], 1)
        self.assertEqual(pair_scores[2, 3], 1)
        self.assertEqual(pair_scores[3, 0], 0)

    def test_dp_selects_best_policy_and_counts_final_segment(self):
        pair_scores = np.zeros((5, 5), dtype=np.int64)
        pair_scores[0, 1] = 1
        pair_scores[0, 2] = 1
        pair_scores[0, 3] = 1
        pair_scores[0, 4] = 1
        pair_scores[2, 3] = 10
        pair_scores[2, 4] = 10

        segment_scores = compute_segment_scores(pair_scores)
        selected, score = select_full_layers_dp(segment_scores, 2)

        self.assertEqual(selected, [0, 2])
        self.assertEqual(score, 21)

    def test_dp_tie_breaks_to_earlier_layers(self):
        segment_scores = np.zeros((5, 6), dtype=np.int64)
        selected, score = select_full_layers_dp(segment_scores, 3)

        self.assertEqual(score, 0)
        self.assertEqual(selected, [0, 1, 2])

    def test_sample_decode_points_adds_answer_boundary_after_random_points(self):
        points = sample_decode_points(
            sample_idx=3,
            prompt_token_ids=list(range(100)),
            answer_query_token_id=999,
            random_points_per_sample=4,
            rng=random.Random(7),
            sink_keep_tokens=0,
            recent_keep_tokens=32,
            min_prefix_tokens=1,
        )

        self.assertEqual(len(points), 5)
        self.assertIsInstance(points[-1], CalibrationPoint)
        self.assertEqual(points[-1].kind, "answer_boundary")
        self.assertEqual(points[-1].prefix_len, 100)
        self.assertEqual(points[-1].query_token_id, 999)
        self.assertTrue(all(point.prefix_len >= 32 for point in points if point.kind == "random"))
        self.assertEqual([point.prefix_len for point in points], sorted(point.prefix_len for point in points))

    def test_hybrid_config_exposes_only_full_attention_candidates(self):
        config = SimpleNamespace(
            text_config=SimpleNamespace(
                num_hidden_layers=8,
                layer_types=[
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "full_attention",
                ],
            )
        )

        self.assertEqual(attention_layer_indices_from_config(config), [1, 3, 5, 7])

    def test_segment_breakdown_maps_candidate_positions_to_physical_layers(self):
        pair_scores = np.zeros((4, 4), dtype=np.int64)
        pair_scores[0, 1:] = [2, 3, 5]
        segment_scores = compute_segment_scores(pair_scores)

        breakdown = selected_segment_breakdown(segment_scores, [0, 2], [3, 7, 11, 15])

        self.assertEqual(breakdown[0]["anchor"], 3)
        self.assertEqual(breakdown[0]["next_full_or_end"], 11)
        self.assertEqual(breakdown[0]["sparse_layers"], [7])
        self.assertEqual(breakdown[1]["anchor"], 11)
        self.assertEqual(breakdown[1]["sparse_layers"], [15])

    def test_global_calibration_excludes_truncated_sliding_attention_scores(self):
        from sparseengine.utils.select_omnikv_full_layers import topk_indices_from_decode_attentions

        # Sliding layers return only their local window; mixing their local
        # indices into full-history coverage can select the wrong observers.
        sliding = torch.tensor([[[[0.8, 0.2]]]])
        global_a = torch.tensor([[[[0.1, 0.2, 0.6, 0.1]]]])
        global_b = torch.tensor([[[[0.1, 0.1, 0.1, 0.7]]]])
        indices, k = topk_indices_from_decode_attentions(
            (sliding, global_a, sliding, global_b), topk=1,
            sink_keep_tokens=0, recent_keep_tokens=0, attention_layer_indices=[1, 3],
        )
        self.assertEqual(indices, [[2], [3]])
        self.assertEqual(k, 1)

    def test_fp8_config_removes_only_nonlinear_gate_exclusions(self):
        config = SimpleNamespace(
            quantization_config={
                "quant_method": "fp8",
                "modules_to_not_convert": [
                    "model.layers.0.mlp.gate",
                    "model.layers.0.mlp.shared_expert_gate",
                    "model.layers.0.mlp.gate_proj",
                    "lm_head",
                ],
            }
        )

        removed = prepare_fp8_transformers_config(config)

        self.assertEqual(
            removed,
            [
                "model.layers.0.mlp.gate",
                "model.layers.0.mlp.shared_expert_gate",
            ],
        )
        self.assertEqual(
            config.quantization_config["modules_to_not_convert"],
            ["model.layers.0.mlp.gate_proj", "lm_head"],
        )


if __name__ == "__main__":
    unittest.main()
