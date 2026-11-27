import unittest

import torch

from sparseengine.utils.context import get_context, reset_context, set_context


class ForwardContextTest(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_next_step_clears_request_state_and_keeps_decode_workspace(self):
        context = get_context()
        workspace = torch.empty(2)
        logsumexp = torch.empty(1)
        context.decode_mid_o = workspace
        context.decode_mid_o_logexpsum = logsumexp
        context.sparse_controller = object()
        context.sparse_config = {"sparse_method": "quest"}
        context.multimodal_image_groups = (object(),)
        context.moe_token_capacity = 8
        context.moe_token_sizes = (2, 8)

        set_context(False)

        self.assertIs(get_context(), context)
        self.assertIsNone(context.sparse_controller)
        self.assertIsNone(context.sparse_config)
        self.assertIsNone(context.multimodal_image_groups)
        self.assertIsNone(context.moe_token_capacity)
        self.assertIsNone(context.moe_token_sizes)
        self.assertIs(context.decode_mid_o, workspace)
        self.assertIs(context.decode_mid_o_logexpsum, logsumexp)

        reset_context()
        self.assertIsNone(get_context().decode_mid_o)
        self.assertIsNone(get_context().decode_mid_o_logexpsum)

    def test_current_dp_transport_metadata_survives_step_setup(self):
        set_context(True, moe_token_metadata=(8, (2, 8)))
        self.assertEqual(get_context().moe_token_capacity, 8)
        self.assertEqual(get_context().moe_token_sizes, (2, 8))

        set_context(False, moe_token_metadata=(4, None))
        self.assertEqual(get_context().moe_token_capacity, 4)
        self.assertIsNone(get_context().moe_token_sizes)
