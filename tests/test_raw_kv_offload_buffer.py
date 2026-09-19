import unittest

import torch

from sparseengine.engine.cache_manager.raw_kv_offload import RawKVOffloadBuffer


class RawKVOffloadBufferTest(unittest.TestCase):
    def test_put_and_copy_prefix_to_for_all_modes(self):
        for mode in ("chunked", "contiguous"):
            with self.subTest(mode=mode):
                buffer = RawKVOffloadBuffer(pin_memory=False, mode=mode)
                buffer.ensure_entry(
                    layer_idx=3,
                    row_idx=1,
                    kind="sparse_pre_rope",
                    total_len=5,
                    k_shape_tail=(2, 4),
                    v_shape_tail=(2, 4),
                    dtype=torch.float32,
                )
                k0 = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
                v0 = k0 + 100
                k1 = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) + 24
                v1 = k1 + 100

                buffer.put_range(
                    layer_idx=3,
                    row_idx=1,
                    kind="sparse_pre_rope",
                    start=0,
                    k=k0,
                    v=v0,
                )
                buffer.put_range(
                    layer_idx=3,
                    row_idx=1,
                    kind="sparse_pre_rope",
                    start=3,
                    k=k1,
                    v=v1,
                )

                k_out = torch.empty((5, 2, 4), dtype=torch.float32)
                v_out = torch.empty((5, 2, 4), dtype=torch.float32)
                buffer.copy_prefix_to(
                    layer_idx=3,
                    row_idx=1,
                    kind="sparse_pre_rope",
                    end=5,
                    k_out=k_out,
                    v_out=v_out,
                )

                self.assertTrue(torch.equal(k_out, torch.cat([k0, k1], dim=0)))
                self.assertTrue(torch.equal(v_out, torch.cat([v0, v1], dim=0)))
                entry = buffer._entries[(3, 1, "sparse_pre_rope")]
                self.assertEqual(entry.producer_events, {})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for stream dependency coverage")
    def test_cuda_put_range_producer_event_orders_cross_stream_copy_prefix(self):
        device = torch.device("cuda")
        for mode in ("chunked", "contiguous"):
            with self.subTest(mode=mode):
                buffer = RawKVOffloadBuffer(pin_memory=True, mode=mode)
                buffer.ensure_entry(
                    layer_idx=0,
                    row_idx=0,
                    kind="sparse_pre_rope",
                    total_len=4,
                    k_shape_tail=(1, 2),
                    v_shape_tail=(1, 2),
                    dtype=torch.float32,
                )
                producer_stream = torch.cuda.Stream(device=device)
                consumer_stream = torch.cuda.Stream(device=device)
                expected_k = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
                expected_v = expected_k + 100

                with torch.cuda.stream(producer_stream):
                    k = expected_k.to(device=device)
                    v = expected_v.to(device=device)
                    buffer.put_range(
                        layer_idx=0,
                        row_idx=0,
                        kind="sparse_pre_rope",
                        start=0,
                        k=k,
                        v=v,
                    )

                entry = buffer._entries[(0, 0, "sparse_pre_rope")]
                self.assertIn(0, entry.producer_events)

                k_out = torch.empty((4, 1, 2), dtype=torch.float32, device=device)
                v_out = torch.empty((4, 1, 2), dtype=torch.float32, device=device)
                with torch.cuda.stream(consumer_stream):
                    buffer.copy_prefix_to(
                        layer_idx=0,
                        row_idx=0,
                        kind="sparse_pre_rope",
                        end=4,
                        k_out=k_out,
                        v_out=v_out,
                    )
                consumer_stream.synchronize()

                self.assertTrue(torch.equal(k_out.cpu(), expected_k))
                self.assertTrue(torch.equal(v_out.cpu(), expected_v))

    def test_chunked_mode_rejects_gaps(self):
        buffer = RawKVOffloadBuffer(pin_memory=False, mode="chunked")
        buffer.ensure_entry(
            layer_idx=0,
            row_idx=0,
            kind="full_post_rope",
            total_len=4,
            k_shape_tail=(1, 2),
            v_shape_tail=(1, 2),
            dtype=torch.float32,
        )

        with self.assertRaisesRegex(RuntimeError, "cannot leave a gap"):
            buffer.put_range(
                layer_idx=0,
                row_idx=0,
                kind="full_post_rope",
                start=2,
                k=torch.zeros((1, 1, 2), dtype=torch.float32),
                v=torch.zeros((1, 1, 2), dtype=torch.float32),
            )

    def test_copy_prefix_to_existing_destination(self):
        buffer = RawKVOffloadBuffer(pin_memory=False, mode="chunked")
        buffer.ensure_entry(
            layer_idx=2,
            row_idx=3,
            kind="full_post_rope",
            total_len=5,
            k_shape_tail=(1, 2),
            v_shape_tail=(1, 2),
            dtype=torch.float32,
        )
        k0 = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        v0 = k0 + 10
        k1 = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2) + 6
        v1 = k1 + 10
        buffer.put_range(layer_idx=2, row_idx=3, kind="full_post_rope", start=0, k=k0, v=v0)
        buffer.put_range(layer_idx=2, row_idx=3, kind="full_post_rope", start=3, k=k1, v=v1)

        k_out = torch.full((6, 1, 2), -1.0)
        v_out = torch.full((6, 1, 2), -1.0)
        buffer.copy_prefix_to(
            layer_idx=2,
            row_idx=3,
            kind="full_post_rope",
            end=5,
            k_out=k_out,
            v_out=v_out,
        )

        self.assertTrue(torch.equal(k_out[:5], torch.cat([k0, k1], dim=0)))
        self.assertTrue(torch.equal(v_out[:5], torch.cat([v0, v1], dim=0)))
        self.assertTrue(torch.equal(k_out[5:], torch.full((1, 1, 2), -1.0)))
        self.assertTrue(torch.equal(v_out[5:], torch.full((1, 1, 2), -1.0)))


if __name__ == "__main__":
    unittest.main()
