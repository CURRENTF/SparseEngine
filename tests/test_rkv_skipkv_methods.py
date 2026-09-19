from types import SimpleNamespace
import unittest

import torch

from sparseengine.config import RuntimeLayout
from sparseengine.engine.cache_manager.methods.rkv import RKVCacheManager
from sparseengine.engine.cache_manager.methods.skipkv import (
    SkipKVCacheManager,
    SkipKVSentence,
    SkipKVSequenceState,
)
from sparseengine.engine.activation_controller import ActivationController
from sparseengine.engine.sequence import Sequence


class RKVSkipKVMethodTest(unittest.TestCase):


    def test_attention_key_materializer_registration_is_idempotent_and_strict(self):
        manager = object.__new__(RKVCacheManager)
        manager.runtime_layout = RuntimeLayout.dense(1)
        first = lambda view: view.payload
        second = lambda view: view.payload

        manager.register_attention_key_materializer(0, first)
        manager.register_attention_key_materializer(0, first)

        self.assertTrue(manager.has_attention_key_materializer(0))
        with self.assertRaisesRegex(RuntimeError, "already bound"):
            manager.register_attention_key_materializer(0, second)

    def test_skipkv_segment_penalty_marks_older_similar_segment(self):
        keys = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.0, 1.0]],
            ]
        )
        penalty = SkipKVCacheManager.segment_redundancy_penalty(
            keys,
            segment_size=2,
            similarity_threshold=0.95,
        )
        self.assertGreater(float(penalty[0]), 0.9)
        self.assertGreater(float(penalty[1]), 0.9)
        self.assertEqual(float(penalty[2]), 0.0)
        self.assertEqual(float(penalty[-1]), 0.0)

    def test_skipkv_sentence_scoring_marks_older_redundant_sentence(self):
        manager = object.__new__(SkipKVCacheManager)
        manager.config = SimpleNamespace(
            skipkv_enable_sentence_scoring=True,
            skipkv_similarity_threshold=0.95,
            skipkv_sentence_min_tokens=1,
            skipkv_sentence_max_tokens=16,
            skipkv_max_tracked_sentences=16,
        )
        manager._skipkv_delimiter_token_ids = {99}
        manager._skipkv_non_execution_token_ids = set()
        manager._skipkv_seq_states = {}

        seq = Sequence([1])
        seq.num_prompt_tokens = 0
        for pos, token_id in enumerate([11, 12, 99, 21, 22, 99]):
            seq.num_tokens = pos + 1
            seq.last_token = token_id
            manager.record_skipkv_decode_hidden_states(
                [seq],
                torch.tensor([[1.0, 0.0]]),
            )

        state = manager._skipkv_seq_states[seq.seq_id]
        self.assertEqual(len(state.sentences), 2)
        self.assertGreater(state.sentences[0].redundancy, 0.95)
        self.assertEqual(state.redundant_sentence_count, 1)
        self.assertEqual(state.non_execution_count, 0)

    def test_skipkv_non_execution_marker_counts_completed_sentence(self):
        manager = object.__new__(SkipKVCacheManager)
        manager.config = SimpleNamespace(
            skipkv_enable_sentence_scoring=True,
            skipkv_similarity_threshold=0.95,
            skipkv_sentence_min_tokens=1,
            skipkv_sentence_max_tokens=16,
            skipkv_max_tracked_sentences=16,
        )
        manager._skipkv_delimiter_token_ids = {99}
        manager._skipkv_non_execution_token_ids = {42}
        manager._skipkv_seq_states = {}

        seq = Sequence([1])
        seq.num_prompt_tokens = 0
        for pos, token_id in enumerate([11, 42, 99]):
            seq.num_tokens = pos + 1
            seq.last_token = token_id
            manager.record_skipkv_decode_hidden_states(
                [seq],
                torch.tensor([[1.0, 0.0]]),
            )

        state = manager._skipkv_seq_states[seq.seq_id]
        self.assertEqual(len(state.sentences), 1)
        self.assertEqual(state.non_execution_count, 1)

    def test_skipkv_chain_turn_boundary_finalizes_open_sentence(self):
        manager = object.__new__(SkipKVCacheManager)
        manager.config = SimpleNamespace(
            skipkv_enable_sentence_scoring=True,
            skipkv_similarity_threshold=0.95,
            skipkv_sentence_min_tokens=1,
            skipkv_sentence_max_tokens=16,
            skipkv_max_tracked_sentences=16,
        )
        manager._skipkv_delimiter_token_ids = {99}
        manager._skipkv_non_execution_token_ids = set()
        manager._skipkv_seq_states = {}

        seq = Sequence([1])
        seq.num_prompt_tokens = 0
        seq.num_tokens = 1
        seq.last_token = 11
        manager.record_skipkv_decode_hidden_states(
            [seq],
            torch.tensor([[1.0, 0.0]]),
        )

        manager.on_chain_turn_finished(seq.seq_id, processed_token_count=1)

        seq.num_prompt_tokens = 2
        seq.num_tokens = 3
        seq.last_token = 99
        manager.record_skipkv_decode_hidden_states(
            [seq],
            torch.tensor([[0.0, 1.0]]),
        )

        state = manager._skipkv_seq_states[seq.seq_id]
        self.assertIsNone(state.open_start_gen)
        self.assertEqual(
            [(sentence.start_gen, sentence.end_gen) for sentence in state.sentences],
            [(0, 1), (2, 3)],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for activation controller buffers")
    def test_skipkv_activation_steering_uses_signed_non_execution_count(self):
        class FakeCacheManager:
            def __init__(self):
                self.delimiters = set()
                self.non_execution_markers = set()

            def set_skipkv_delimiter_token_ids(self, token_ids):
                self.delimiters = set(token_ids)

            def set_skipkv_non_execution_token_ids(self, token_ids):
                self.non_execution_markers = set(token_ids)

            def skipkv_non_execution_count(self, _seq_id):
                return 3

        config = SimpleNamespace(
            sparse_method="skipkv",
            hf_config=SimpleNamespace(num_hidden_layers=28, dtype=torch.float32, hidden_size=4),
            skipkv_sentence_embedding_layer=-1,
            skipkv_steering_layer=20,
            skipkv_steering_vector_path=None,
            skipkv_enable_activation_steering=True,
            skipkv_steering_alpha=-1.25,
            skipkv_steering_alpha_increment=-0.02,
            skipkv_steering_alpha_max=0.0,
            max_decoding_seqs=2,
        )
        controller = ActivationController.create(config, FakeCacheManager())
        controller._steering_vector = torch.ones(4, device="cuda")
        controller.set_tokenizer_metadata(delimiter_token_ids={99}, non_execution_token_ids={42})

        seq = Sequence([1])
        seq.num_prompt_tokens = 0
        seq.num_tokens = 2
        seq.last_token = 99
        controller.prepare_forward([seq], is_prefill=False)

        hidden = torch.zeros((1, 4), device="cuda")
        updated, _ = controller.apply_layer_hook(20, hidden, None, None)

        self.assertTrue(torch.allclose(updated.cpu(), torch.full((1, 4), -1.31)))

    def test_skipkv_sentence_penalty_uses_cache_range_mapping(self):
        manager = object.__new__(SkipKVCacheManager)
        manager.config = SimpleNamespace(
            skipkv_enable_sentence_scoring=True,
            skipkv_sentence_score_weight=1.0,
        )
        manager._skipkv_seq_states = {}
        manager._skipkv_row_gen_indices = [{0: [0, 1, 2, 3, 4, 5]}]
        seq = Sequence([1])
        seq.num_prompt_tokens = 0
        sentence = SkipKVSentence(
            start_gen=0,
            end_gen=3,
            embedding=torch.tensor([1.0, 0.0]),
            redundancy=0.97,
        )
        manager._skipkv_seq_states[seq.seq_id] = SkipKVSequenceState(
            num_prompt_tokens=0,
            sentences=[sentence],
        )

        penalty = manager._sentence_redundancy_penalty(
            0,
            seq,
            0,
            candidate_start=0,
            candidate_end=6,
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(penalty)
        self.assertGreater(float(penalty[0]), 0.9)
        self.assertGreater(float(penalty[2]), 0.9)
        self.assertEqual(float(penalty[3]), 0.0)

    def test_skipkv_batch_free_updates_gen_indices(self):
        manager = object.__new__(SkipKVCacheManager)
        manager.runtime_layout = RuntimeLayout.dense(1)
        manager.config = SimpleNamespace()
        manager.device = torch.device("cpu")
        seqs = [Sequence([1]), Sequence([2])]
        manager.seq_id_to_row = [{seqs[0].seq_id: 0, seqs[1].seq_id: 1}]
        manager.row_seq_lens = [torch.tensor([6, 6], dtype=torch.int32)]
        manager.buffer_req_to_token_slots = [torch.arange(12, dtype=torch.int32).view(2, 6)]
        manager.buffer_req_to_token_slots_tensor = None
        manager.free_slots_stack = [torch.empty(16, dtype=torch.int32)]
        manager.free_slots_stack_tensor = None
        manager._num_free_slots = [0]
        manager._uniform_decode_metadata = True
        manager._rkv_query_cache_enabled = False
        manager._skipkv_row_gen_indices = [{0: [10, 11, 12, 13, 14, 15], 1: [20, 21, 22, 23, 24, 25]}]
        manager._skipkv_seq_states = {}

        keep = torch.tensor([[0, 2, 5], [1, 3, 4]], dtype=torch.long)
        manager.free_part_slots_batch(0, seqs, keep)

        self.assertEqual(manager._skipkv_row_gen_indices[0][0], [10, 12, 15])
        self.assertEqual(manager._skipkv_row_gen_indices[0][1], [21, 23, 24])


    def test_skipkv_batch_selection_matches_single_selection_without_sentences(self):
        torch.manual_seed(29)
        manager = object.__new__(SkipKVCacheManager)
        manager.runtime_layout = RuntimeLayout.dense(1)
        manager.config = SimpleNamespace(
            sink_keep_tokens=1,
            recent_keep_tokens=1,
            skipkv_alpha=0.1,
            skipkv_similarity_threshold=0.95,
            skipkv_segment_size=2,
            skipkv_max_redundancy_tokens=16,
            skipkv_redundancy_window=4,
            rkv_similarity_threshold=0.8,
            rkv_recent_similar_keep=1,
        )
        seqs = [Sequence([1]), Sequence([2])]
        kv_len = 8
        manager.seq_id_to_row = [{seq.seq_id: idx for idx, seq in enumerate(seqs)}]
        manager.buffer_req_to_token_slots = [
            torch.arange(len(seqs) * kv_len, dtype=torch.int32).view(len(seqs), kv_len)
        ]
        manager.kv_cache = [
            (
                torch.randn(len(seqs) * kv_len, 1, 4),
                torch.randn(len(seqs) * kv_len, 1, 4),
            )
        ]
        manager._skipkv_seq_states = {}
        importance = torch.stack(
            [
                torch.linspace(0.0, 1.0, steps=kv_len),
                torch.linspace(1.0, 0.0, steps=kv_len),
            ],
            dim=0,
        )

        batch_keep = manager.select_skipkv_indices_batch(
            0,
            seqs,
            importance,
            [kv_len, kv_len],
            budget=5,
        )
        single_keep = torch.stack(
            [
                manager.select_skipkv_indices(
                    0,
                    seq,
                    importance[idx],
                    kv_len=kv_len,
                    budget=5,
                )
                for idx, seq in enumerate(seqs)
            ],
            dim=0,
        )

        self.assertIsNotNone(batch_keep)
        for row in range(len(seqs)):
            self.assertEqual(set(batch_keep[row].tolist()), set(single_keep[row].tolist()))


if __name__ == "__main__":
    unittest.main()
