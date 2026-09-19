import os
import unittest

import torch

from sparseengine.kernels.triton.deltakv_kernels import (
    _validate_full_layer_kivi_decode_maps,
    deltakv_less_memory_reconstruct_writeback_quantized,
    deltakv_l2_topk_blockwise,
    deltakv_materialize_sparse_view,
    deltakv_reconstruct_writeback_grouped_heads,
    deltakv_static_decode_plan,
    full_layer_kivi_build_dense_decode_view,
    full_layer_kivi_flash_decode_stage1,
    full_layer_kivi_flash_decode_stage1_grouped,
    full_layer_kivi_flash_decode_stage1_token_map,
)
from sparseengine.kernels.triton.gqa_flash_decoding_stage1 import flash_decode_stage1 as gqa_flash_decode_stage1
from sparseengine.kernels.triton.gqa_flash_decoding_stage1 import (
    flash_decode_stage1_with_score as gqa_flash_decode_stage1_with_score,
)
from sparseengine.kernels.triton.paged_flash_decoding import (
    fixed_grid_flash_decode_stage2,
)
from sparseengine.kernels.triton.quant import (
    triton_dequantize_2d_int4_grouped,
    triton_quantize_and_pack_2d_int4_grouped,
    triton_quantize_and_pack_along_last_dim,
    unpack_quantized_to_16bit,
)


def test_full_layer_kivi_decode_map_validation_rejects_unmapped_valid_token():
    raw_slots_map = torch.tensor([[0, -1, -1]], dtype=torch.int32)
    kivi_block_slots_map = torch.tensor([[-1, -1, 2]], dtype=torch.int32)
    try:
        _validate_full_layer_kivi_decode_maps(
            raw_slots_map=raw_slots_map,
            kivi_block_slots_map=kivi_block_slots_map,
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([3], dtype=torch.int32),
            max_len_in_batch=3,
        )
    except RuntimeError as exc:
        assert "neither raw nor packed block slots" in str(exc)
    else:
        raise AssertionError("expected full-layer KIVI decode map validation to fail")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for DeltaKV Triton kernel tests.")
class DeltaKVLessMemoryKernelTest(unittest.TestCase):
    def test_residual_2d_int4_quant_matches_reference_for_float32(self):
        torch.manual_seed(19)
        device = "cuda"
        n = 37
        d = 512
        group_size = 32
        x = torch.randn(n, d, device=device, dtype=torch.float32)

        ref_packed, ref_scale, ref_mn = triton_quantize_and_pack_along_last_dim(
            x.unsqueeze(0).unsqueeze(0),
            group_size,
            4,
        )
        got_packed, got_scale, got_mn = triton_quantize_and_pack_2d_int4_grouped(x, group_size)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(got_packed, ref_packed.squeeze(0).squeeze(0).to(torch.int32)))
        self.assertTrue(torch.equal(got_scale, ref_scale.squeeze(0).squeeze(0)))
        self.assertTrue(torch.equal(got_mn, ref_mn.squeeze(0).squeeze(0)))

    def test_residual_2d_int4_dequant_matches_reference_for_float32(self):
        torch.manual_seed(23)
        device = "cuda"
        n = 41
        d = 512
        group_size = 32
        x = torch.randn(n, d, device=device, dtype=torch.float32)
        packed, scale, mn = triton_quantize_and_pack_2d_int4_grouped(x, group_size)

        ref = unpack_quantized_to_16bit(
            packed.unsqueeze(0).unsqueeze(0),
            scale.unsqueeze(0).unsqueeze(0),
            mn.unsqueeze(0).unsqueeze(0),
            group_size,
            4,
        ).squeeze(0).squeeze(0)
        got = triton_dequantize_2d_int4_grouped(packed, scale, mn, group_size, d)
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(got, ref, atol=1e-6, rtol=1e-6))

    def test_store_kvcache_chunks_large_launches(self):
        from sparseengine.kernels.triton.store_kvcache import store_kvcache

        old_value = os.environ.get("SPARSEENGINE_STORE_KVCACHE_CHUNK_TOKENS")
        os.environ["SPARSEENGINE_STORE_KVCACHE_CHUNK_TOKENS"] = "3"
        try:
            device = "cuda"
            dtype = torch.float16
            n = 8
            num_heads = 2
            head_dim = 4
            key = torch.arange(n * num_heads * head_dim, device=device, dtype=dtype).view(n, num_heads, head_dim)
            value = key + 1000
            k_cache = torch.full((10, num_heads, head_dim), -1, device=device, dtype=dtype)
            v_cache = torch.full_like(k_cache, -1)
            slot_mapping = torch.tensor([7, 0, 5, -1, 2, 4, 1, 6], device=device, dtype=torch.int32)

            store_kvcache(key, value, k_cache, v_cache, slot_mapping)
            torch.cuda.synchronize()

            for token_idx, slot in enumerate(slot_mapping.tolist()):
                if slot < 0:
                    continue
                self.assertTrue(torch.equal(k_cache[slot], key[token_idx]))
                self.assertTrue(torch.equal(v_cache[slot], value[token_idx]))
            self.assertTrue(torch.equal(k_cache[3], torch.full_like(k_cache[3], -1)))
        finally:
            if old_value is None:
                os.environ.pop("SPARSEENGINE_STORE_KVCACHE_CHUNK_TOKENS", None)
            else:
                os.environ["SPARSEENGINE_STORE_KVCACHE_CHUNK_TOKENS"] = old_value

    def test_l2_topk_blockwise_accepts_dynamic_center_positions(self):
        torch.manual_seed(3)
        device = "cuda"
        dtype = torch.float16
        n = 37
        d = 64
        m0 = 3
        k = 4
        new_center_rel = torch.tensor([0, 5, 11, 18, 27, 35], device=device, dtype=torch.int32)
        tokens = torch.randn((n, d), device=device, dtype=dtype)
        existing = torch.randn((m0, d), device=device, dtype=dtype)
        centers = torch.cat([existing, tokens.index_select(0, new_center_rel.to(torch.long))], dim=0)

        partial_scores, partial_idx = deltakv_l2_topk_blockwise(
            tokens=tokens,
            centers=centers,
            m0=m0,
            cluster_step=10,
            k=k,
            new_center_rel=new_center_rel,
            block_n=16,
            block_m=16,
            block_d=32,
        )
        nb, mb, bn, kk = partial_scores.shape
        cand_scores = partial_scores.permute(0, 2, 1, 3).reshape(nb * bn, mb * kk)[:n]
        cand_idx = partial_idx.permute(0, 2, 1, 3).reshape(nb * bn, mb * kk)[:n]
        merge_pos = cand_scores.topk(k=k, dim=1).indices
        got_idx = cand_idx.gather(1, merge_pos)

        scores = 2.0 * torch.matmul(tokens.float(), centers.float().T)
        scores -= centers.float().pow(2).sum(dim=1).view(1, -1)
        rows = torch.arange(n, device=device).view(n, 1)
        allow_existing = torch.ones((n, m0), device=device, dtype=torch.bool)
        allow_new = new_center_rel.view(1, -1).to(torch.long) <= rows
        scores = scores.masked_fill(~torch.cat([allow_existing, allow_new], dim=1), float("-inf"))
        expected_idx = scores.topk(k=k, dim=1).indices.to(torch.int32)

        self.assertTrue(torch.equal(got_idx, expected_idx))

    def test_streaming_cluster_helper_matches_whole_block_cluster(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        torch.manual_seed(11)
        device = "cuda"
        dtype = torch.float16
        n = 23
        d = 32
        existing_count = 2
        new_center_rel = torch.tensor([0, 4, 9, 13, 20], device=device, dtype=torch.long)

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = type(
            "Config",
            (),
            {
                "cluster_metric": "l2",
                "deltakv_neighbor_count": 3,
                "deltakv_cluster_gather_chunk_size": 5,
            },
        )()
        manager.deltakv_layer_to_idx = {7: 0}

        kv_states = torch.randn((1, n, d), device=device, dtype=dtype)
        existing_centers = torch.randn((1, existing_count, d), device=device, dtype=dtype)
        existing_slots = torch.arange(existing_count, device=device, dtype=torch.int32)

        def gather_existing(l_idx, slots, *, validate=True):
            del validate
            self.assertEqual(l_idx, 0)
            return existing_centers.squeeze(0).index_select(0, slots.to(torch.long))

        manager._gather_sparse_ref_raw_kv_by_slots = gather_existing
        ref_topk, ref_base = DeltaKVLessMemoryCacheManager._cluster_compress(
            manager,
            layer_idx=7,
            kv_states=kv_states,
            existing_center_slots=existing_slots,
            cluster_step=4,
            new_center_rel=new_center_rel,
        )

        all_centers = torch.cat([existing_centers, kv_states.index_select(1, new_center_rel)], dim=1)
        topk_chunks = []
        base_chunks = []
        for start in (0, 6, 12, 18):
            end = min(n, start + 6)
            topk, base = DeltaKVLessMemoryCacheManager._cluster_compress_against_centers(
                manager,
                kv_states=kv_states[:, start:end, :],
                all_centers=all_centers,
                existing_center_count=existing_count,
                new_center_rel=new_center_rel,
                row_start=start,
            )
            topk_chunks.append(topk)
            base_chunks.append(base)

        self.assertTrue(torch.equal(ref_topk, torch.cat(topk_chunks, dim=0)))
        self.assertTrue(torch.allclose(ref_base, torch.cat(base_chunks, dim=1), atol=1e-3, rtol=1e-3))

    def test_static_decode_plan_matches_reference(self):
        device = "cuda"
        batch_size = 2
        k_max = 4
        sink = 2
        max_buffer = 3
        max_s = sink + k_max + max_buffer

        raw_slots_map = torch.full((5, 12), -1, device=device, dtype=torch.int32)
        latent_slots_map = torch.full_like(raw_slots_map, -1)
        for row in range(raw_slots_map.shape[0]):
            for pos in range(raw_slots_map.shape[1]):
                raw_slots_map[row, pos] = row * 100 + pos
        raw_slots_map[1, 2] = -1
        latent_slots_map[1, 2] = 20
        raw_slots_map[1, 6] = -1
        latent_slots_map[1, 6] = 21
        raw_slots_map[3, 3] = -1
        latent_slots_map[3, 3] = 30
        raw_slots_map[3, 5] = -1
        latent_slots_map[3, 5] = -1

        active_compressed_indices = torch.tensor(
            [[0, 2, 4, 99], [1, -1, 3, 0]],
            device=device,
            dtype=torch.int32,
        )
        req_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
        context_lens = torch.tensor([10, 7], device=device, dtype=torch.int32)
        compressed_lens = torch.tensor([5, 2], device=device, dtype=torch.int32)
        temp_slots = torch.tensor(
            [[900, 901, 902, 903], [910, 911, 912, 913]],
            device=device,
            dtype=torch.int32,
        )

        rows = req_indices.to(torch.long)
        safe_slot = raw_slots_map[rows, 0:1].clamp_min(0)

        sink_pos = torch.arange(sink, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        sink_slots = raw_slots_map[rows[:, None], sink_pos]

        top_rel = active_compressed_indices
        top_pos = top_rel + sink
        valid_top = (
            (top_rel >= 0)
            & (top_rel < compressed_lens.unsqueeze(1))
            & (top_pos < context_lens.unsqueeze(1))
        )
        safe_top_pos = top_pos.clamp(0, raw_slots_map.shape[1] - 1).to(torch.long)
        top_raw_slots = raw_slots_map[rows[:, None], safe_top_pos]
        top_latent_slots = latent_slots_map[rows[:, None], safe_top_pos]
        top_len = compressed_lens.clamp(0, k_max)
        top_capacity = torch.arange(k_max, device=device, dtype=torch.int32).unsqueeze(0) < top_len.unsqueeze(1)
        valid_top = valid_top & top_capacity
        need_reconstruct = valid_top & (top_latent_slots >= 0)
        top_slots = torch.where(
            need_reconstruct,
            temp_slots,
            torch.where(valid_top, top_raw_slots.clamp_min(0), safe_slot.expand(-1, k_max)),
        )
        recon_pos_ref = torch.where(need_reconstruct, top_pos, torch.full_like(top_pos, -1)).reshape(-1)
        recon_latent_ref = torch.where(
            need_reconstruct,
            top_latent_slots,
            torch.full_like(top_latent_slots, -1),
        ).reshape(-1)
        recon_out_slot_ref = torch.where(need_reconstruct, temp_slots, torch.full_like(temp_slots, -1)).reshape(-1)

        buffer_start = sink + compressed_lens
        buffer_len = (context_lens - buffer_start).clamp(0, max_buffer)
        buffer_offsets = torch.arange(max_buffer, device=device, dtype=torch.int32).unsqueeze(0)
        buffer_pos = (buffer_start.unsqueeze(1) + buffer_offsets).clamp(0, raw_slots_map.shape[1] - 1).to(torch.long)
        buffer_valid = buffer_offsets < buffer_len.unsqueeze(1)
        buffer_slots_raw = raw_slots_map[rows[:, None], buffer_pos]
        buffer_slots = torch.where(buffer_valid, buffer_slots_raw.clamp_min(0), safe_slot.expand(-1, max_buffer))

        active_slots_ref = safe_slot.expand(-1, max_s).clone()
        active_pos_ref = torch.zeros((batch_size, max_s), device=device, dtype=torch.int32)
        active_slots_ref[:, :sink] = sink_slots
        active_pos_ref[:, :sink] = torch.arange(sink, device=device, dtype=torch.int32).unsqueeze(0)
        top_pos_ref = torch.where(valid_top, top_pos, torch.zeros_like(top_pos)).to(torch.int32)
        buffer_pos_ref = torch.where(buffer_valid, buffer_pos.to(torch.int32), torch.zeros_like(buffer_pos, dtype=torch.int32))
        for batch_idx in range(batch_size):
            visible_top = int(top_len[batch_idx].item())
            active_slots_ref[batch_idx, sink : sink + visible_top] = top_slots[batch_idx, :visible_top]
            active_pos_ref[batch_idx, sink : sink + visible_top] = top_pos_ref[batch_idx, :visible_top]
            buffer_dst = sink + visible_top
            active_slots_ref[batch_idx, buffer_dst : buffer_dst + max_buffer] = buffer_slots[batch_idx]
            active_pos_ref[batch_idx, buffer_dst : buffer_dst + max_buffer] = buffer_pos_ref[batch_idx]
        new_context_lens_ref = sink + top_len + buffer_len

        active_slots_out = torch.empty((batch_size, max_s), device=device, dtype=torch.int32)
        active_pos_out = torch.empty((batch_size, max_s), device=device, dtype=torch.int32)
        new_context_lens_out = torch.empty((batch_size,), device=device, dtype=torch.int32)
        recon_pos_out = torch.empty((batch_size * k_max,), device=device, dtype=torch.int32)
        recon_latent_out = torch.empty_like(recon_pos_out)
        recon_out_slot_out = torch.empty_like(recon_pos_out)
        deltakv_static_decode_plan(
            raw_slots_map=raw_slots_map,
            latent_slots_map=latent_slots_map,
            active_compressed_indices=active_compressed_indices,
            req_indices=req_indices,
            context_lens=context_lens,
            compressed_lens=compressed_lens,
            temp_slots=temp_slots,
            active_slots_out=active_slots_out,
            active_pos_out=active_pos_out,
            new_context_lens_out=new_context_lens_out,
            recon_pos_out=recon_pos_out,
            recon_latent_out=recon_latent_out,
            recon_out_slot_out=recon_out_slot_out,
            sink=sink,
            max_buffer=max_buffer,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(active_slots_out, active_slots_ref))
        self.assertTrue(torch.equal(active_pos_out, active_pos_ref))
        self.assertTrue(torch.equal(new_context_lens_out, new_context_lens_ref))
        self.assertTrue(torch.equal(recon_pos_out, recon_pos_ref))
        self.assertTrue(torch.equal(recon_latent_out, recon_latent_ref))
        self.assertTrue(torch.equal(recon_out_slot_out, recon_out_slot_ref))

    def test_materialize_sparse_view_writes_padded_rows(self):
        from sparseengine.layers.rotary_embedding import apply_rotary_emb

        torch.manual_seed(1)
        device = "cuda"
        dtype = torch.float16
        batch_size = 3
        width = 16
        num_slots = 64
        num_heads = 2
        head_dim = 8

        k_cache = torch.randn(num_slots, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        active_slots = torch.randint(1, num_slots, (batch_size, width), device=device, dtype=torch.int32)
        context_lens = torch.tensor([16, 7, 11], device=device, dtype=torch.int32)
        active_slots[1, 7:] = active_slots[1, 0]
        active_slots[2, 11:] = active_slots[2, 0]
        slot_to_pos = torch.arange(num_slots, device=device, dtype=torch.int32)
        postrope_mask = torch.zeros(num_slots, device=device, dtype=torch.bool)

        cos = torch.randn(128, head_dim // 2, device=device)
        sin = torch.randn(128, head_dim // 2, device=device)
        norm = torch.sqrt(cos * cos + sin * sin).clamp_min(1e-6)
        cos_sin = torch.cat([cos / norm, sin / norm], dim=-1).to(dtype).unsqueeze(1)

        out_k = torch.full((batch_size * width, num_heads, head_dim), float("nan"), device=device, dtype=dtype)
        out_v = torch.full_like(out_k, float("nan"))
        deltakv_materialize_sparse_view(
            active_slots=active_slots,
            context_lens=context_lens,
            slot_to_pos=slot_to_pos,
            postrope_mask=postrope_mask,
            k_cache=k_cache,
            v_cache=v_cache,
            out_k=out_k,
            out_v=out_v,
            cos_sin=cos_sin,
            block_tokens=8,
        )
        torch.cuda.synchronize()

        flat_slots = active_slots.reshape(-1).to(torch.long).clamp_min(0)
        raw_k = k_cache[flat_slots]
        raw_v = v_cache[flat_slots]
        pos = slot_to_pos[flat_slots].to(torch.long).clamp_min(0)
        cos_ref, sin_ref = cos_sin[pos].chunk(2, dim=-1)
        expected_k = apply_rotary_emb(raw_k, cos_ref, sin_ref)

        self.assertFalse(bool(torch.isnan(out_k).any()))
        self.assertFalse(bool(torch.isnan(out_v).any()))
        self.assertTrue(torch.equal(out_k, expected_k))
        self.assertTrue(torch.equal(out_v, raw_v))

    def test_fused_quantized_reconstruct_matches_unpack_then_reconstruct(self):
        torch.manual_seed(0)
        device = "cuda"
        dtype = torch.float16
        num_tokens = 5
        num_slots = 16
        num_heads = 2
        head_dim = 8
        kv_dim = num_heads * head_dim
        max_pos = 32

        angles = torch.randn(max_pos, head_dim // 2, device=device, dtype=torch.float32) * 0.1
        cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(dtype)

        father_slots = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
            device=device,
            dtype=torch.int32,
        )
        slot_to_pos = torch.arange(num_slots, device=device, dtype=torch.int32)
        out_slots = torch.tensor([8, 9, 10, 11, 12], device=device, dtype=torch.int32)
        out_pos = torch.tensor([7, 9, 11, 13, 15], device=device, dtype=torch.int32)

        residual_dim = 2 * kv_dim
        for quant_bits in (2, 4):
            for group_size in (residual_dim, 8):
                with self.subTest(quant_bits=quant_bits, group_size=group_size):
                    k_base = torch.randn(num_slots, num_heads, head_dim, device=device, dtype=dtype)
                    v_base = torch.randn_like(k_base)
                    k_ref = k_base.clone()
                    v_ref = v_base.clone()
                    k_fused = k_base.clone()
                    v_fused = v_base.clone()
                    k_norm_weight = torch.randn(head_dim, device=device, dtype=dtype)

                    residual = torch.randn(num_tokens, residual_dim, device=device, dtype=dtype) * 0.2
                    packed, scale, mn = triton_quantize_and_pack_along_last_dim(
                        residual.unsqueeze(0).unsqueeze(0),
                        group_size,
                        quant_bits,
                    )
                    packed = packed.squeeze(0).squeeze(0).contiguous()
                    scale = scale.squeeze(0).squeeze(0).contiguous()
                    mn = mn.squeeze(0).squeeze(0).contiguous()

                    latent_slots = torch.arange(num_tokens, device=device, dtype=torch.int32)
                    packed_cache = torch.empty((num_slots, packed.shape[1]), device=device, dtype=torch.int32)
                    scale_cache = torch.empty((num_slots, residual_dim // group_size), device=device, dtype=dtype)
                    mn_cache = torch.empty((num_slots, residual_dim // group_size), device=device, dtype=dtype)
                    packed_cache[latent_slots.long()] = packed
                    scale_cache[latent_slots.long()] = scale
                    mn_cache[latent_slots.long()] = mn

                    kv_delta = unpack_quantized_to_16bit(
                        packed.unsqueeze(0).unsqueeze(0),
                        scale.unsqueeze(0).unsqueeze(0),
                        mn.unsqueeze(0).unsqueeze(0),
                        group_size,
                        quant_bits,
                    ).squeeze(0).squeeze(0)
                    deltakv_reconstruct_writeback_grouped_heads(
                        kv_delta=kv_delta,
                        father_slots=father_slots,
                        slot_to_pos=slot_to_pos,
                        out_slots=out_slots,
                        out_pos=out_pos,
                        cos_sin=cos_sin,
                        k_cache=k_ref,
                        v_cache=v_ref,
                        heads_per_program=2,
                        k_norm_weight=k_norm_weight,
                        k_norm_eps=1e-6,
                    )
                    deltakv_less_memory_reconstruct_writeback_quantized(
                        packed_delta_cache=packed_cache,
                        scale_cache=scale_cache,
                        min_cache=mn_cache,
                        latent_slots=latent_slots,
                        father_slots=father_slots,
                        slot_to_pos=slot_to_pos,
                        out_slots=out_slots,
                        out_pos=out_pos,
                        cos_sin=cos_sin,
                        k_cache=k_fused,
                        v_cache=v_fused,
                        quant_bits=quant_bits,
                        group_size=group_size,
                        heads_per_program=2,
                        k_norm_weight=k_norm_weight,
                        k_norm_eps=1e-6,
                    )
                    torch.cuda.synchronize()

                    out = out_slots.long()
                    self.assertTrue(torch.allclose(k_ref[out], k_fused[out], atol=2e-3, rtol=2e-3))
                    self.assertTrue(torch.allclose(v_ref[out], v_fused[out], atol=4e-3, rtol=2e-3))

    def test_fused_quantized_reconstruct_can_write_raw_cache(self):
        torch.manual_seed(6)
        device = "cuda"
        dtype = torch.float16
        num_tokens = 5
        num_slots = 16
        num_heads = 2
        head_dim = 8
        kv_dim = num_heads * head_dim
        residual_dim = 2 * kv_dim
        max_pos = 32
        quant_bits = 4
        group_size = 8

        angles = torch.randn(max_pos, head_dim // 2, device=device, dtype=torch.float32) * 0.1
        cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(dtype)
        father_slots = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
            device=device,
            dtype=torch.int32,
        )
        slot_to_pos = torch.arange(num_slots, device=device, dtype=torch.int32)
        out_slots = torch.tensor([8, 9, 10, 11, 12], device=device, dtype=torch.int32)
        out_pos = torch.tensor([7, 9, 11, 13, 15], device=device, dtype=torch.int32)

        k_base = torch.randn(num_slots, num_heads, head_dim, device=device, dtype=dtype)
        v_base = torch.randn_like(k_base)
        k_ref = k_base.clone()
        v_ref = v_base.clone()
        k_fused = k_base.clone()
        v_fused = v_base.clone()

        residual = torch.randn(num_tokens, residual_dim, device=device, dtype=dtype) * 0.2
        packed, scale, mn = triton_quantize_and_pack_along_last_dim(
            residual.unsqueeze(0).unsqueeze(0),
            group_size,
            quant_bits,
        )
        packed = packed.squeeze(0).squeeze(0).contiguous()
        scale = scale.squeeze(0).squeeze(0).contiguous()
        mn = mn.squeeze(0).squeeze(0).contiguous()
        latent_slots = torch.arange(num_tokens, device=device, dtype=torch.int32)
        packed_cache = torch.empty((num_slots, packed.shape[1]), device=device, dtype=torch.int32)
        scale_cache = torch.empty((num_slots, residual_dim // group_size), device=device, dtype=dtype)
        mn_cache = torch.empty((num_slots, residual_dim // group_size), device=device, dtype=dtype)
        packed_cache[latent_slots.long()] = packed
        scale_cache[latent_slots.long()] = scale
        mn_cache[latent_slots.long()] = mn

        kv_delta = unpack_quantized_to_16bit(
            packed.unsqueeze(0).unsqueeze(0),
            scale.unsqueeze(0).unsqueeze(0),
            mn.unsqueeze(0).unsqueeze(0),
            group_size,
            quant_bits,
        ).squeeze(0).squeeze(0)
        deltakv_reconstruct_writeback_grouped_heads(
            kv_delta=kv_delta,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_ref,
            v_cache=v_ref,
            heads_per_program=2,
            raw_k_cache=True,
            store_raw_k=True,
        )
        deltakv_less_memory_reconstruct_writeback_quantized(
            packed_delta_cache=packed_cache,
            scale_cache=scale_cache,
            min_cache=mn_cache,
            latent_slots=latent_slots,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_fused,
            v_cache=v_fused,
            quant_bits=quant_bits,
            group_size=group_size,
            heads_per_program=2,
            raw_k_cache=True,
            store_raw_k=True,
        )
        torch.cuda.synchronize()

        out = out_slots.long()
        self.assertTrue(torch.allclose(k_ref[out], k_fused[out], atol=2e-3, rtol=2e-3))
        self.assertTrue(torch.allclose(v_ref[out], v_fused[out], atol=4e-3, rtol=2e-3))

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "torch.float8_e4m3fn is required.")
    def test_reconstruct_grouped_heads_accepts_pre_rope_refs_and_k_norm(self):
        torch.manual_seed(4)
        device = "cuda"
        dtype = torch.float16
        num_tokens = 6
        num_slots = 20
        num_heads = 2
        head_dim = 8
        kv_dim = num_heads * head_dim
        max_pos = 32

        angles = torch.randn(max_pos, head_dim // 2, device=device, dtype=torch.float32) * 0.1
        cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(dtype)
        father_slots = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
            device=device,
            dtype=torch.int32,
        )
        slot_to_pos = torch.arange(num_slots, device=device, dtype=torch.int32)
        out_slots = torch.tensor([10, 11, 12, 13, 14, 15], device=device, dtype=torch.int32)
        out_pos = torch.tensor([7, 9, 11, 13, 15, 17], device=device, dtype=torch.int32)

        kv_delta = torch.randn(num_tokens, 2 * kv_dim, device=device, dtype=dtype) * 0.2
        pre_rope_k_cache = torch.randn(num_slots, num_heads, head_dim, device=device, dtype=dtype)
        ref_v_cache = torch.randn_like(pre_rope_k_cache)
        pre_rope_k_cache = pre_rope_k_cache.to(torch.float8_e4m3fn).to(dtype)
        ref_v_cache = ref_v_cache.to(torch.float8_e4m3fn).to(dtype)

        k_cache = torch.randn_like(pre_rope_k_cache)
        v_cache = torch.randn_like(ref_v_cache)
        k_out = k_cache.clone()
        v_out = v_cache.clone()
        k_norm_weight = torch.randn(head_dim, device=device, dtype=dtype)

        father_i64 = father_slots.long()
        mean_k = pre_rope_k_cache[father_i64].float().mean(dim=1)
        mean_v = ref_v_cache[father_i64].float().mean(dim=1)
        raw_k = kv_delta[:, :kv_dim].view(num_tokens, num_heads, head_dim).float() + mean_k
        v_ref = kv_delta[:, kv_dim:].view(num_tokens, num_heads, head_dim).float() + mean_v
        raw_k = raw_k * torch.rsqrt(raw_k.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        raw_k = raw_k * k_norm_weight.float()
        cos, sin = cos_sin[out_pos.long()].float().chunk(2, dim=-1)
        k1, k2 = raw_k.chunk(2, dim=-1)
        k_ref = torch.cat(
            [
                k1 * cos[:, None, :] - k2 * sin[:, None, :],
                k2 * cos[:, None, :] + k1 * sin[:, None, :],
            ],
            dim=-1,
        )

        deltakv_reconstruct_writeback_grouped_heads(
            kv_delta=kv_delta,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_out,
            v_cache=v_out,
            heads_per_program=2,
            pre_rope_k_cache=pre_rope_k_cache,
            ref_v_cache=ref_v_cache,
            k_norm_weight=k_norm_weight,
            k_norm_eps=1e-6,
        )
        torch.cuda.synchronize()

        out = out_slots.long()
        self.assertTrue(torch.allclose(k_out[out], k_ref.to(dtype), atol=2e-3, rtol=2e-3))
        self.assertTrue(torch.allclose(v_out[out], v_ref.to(dtype), atol=2e-3, rtol=2e-3))

    def test_reconstruct_grouped_heads_can_write_raw_cache(self):
        torch.manual_seed(5)
        device = "cuda"
        dtype = torch.float16
        num_tokens = 5
        num_slots = 18
        num_heads = 2
        head_dim = 8
        kv_dim = num_heads * head_dim
        max_pos = 32

        angles = torch.randn(max_pos, head_dim // 2, device=device, dtype=torch.float32) * 0.1
        cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(dtype)
        father_slots = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
            device=device,
            dtype=torch.int32,
        )
        slot_to_pos = torch.arange(num_slots, device=device, dtype=torch.int32)
        out_slots = torch.tensor([10, 11, 12, 13, 14], device=device, dtype=torch.int32)
        out_pos = torch.tensor([7, 9, 11, 13, 15], device=device, dtype=torch.int32)

        kv_delta = torch.randn(num_tokens, 2 * kv_dim, device=device, dtype=dtype) * 0.2
        k_cache = torch.randn(num_slots, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        k_out = k_cache.clone()
        v_out = v_cache.clone()

        father_i64 = father_slots.long()
        mean_k = k_cache[father_i64].float().mean(dim=1)
        mean_v = v_cache[father_i64].float().mean(dim=1)
        k_ref = kv_delta[:, :kv_dim].view(num_tokens, num_heads, head_dim).float() + mean_k
        v_ref = kv_delta[:, kv_dim:].view(num_tokens, num_heads, head_dim).float() + mean_v

        deltakv_reconstruct_writeback_grouped_heads(
            kv_delta=kv_delta,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_out,
            v_cache=v_out,
            heads_per_program=2,
            raw_k_cache=True,
            store_raw_k=True,
        )
        torch.cuda.synchronize()

        out = out_slots.long()
        self.assertTrue(torch.allclose(k_out[out], k_ref.to(dtype), atol=2e-3, rtol=2e-3))
        self.assertTrue(torch.allclose(v_out[out], v_ref.to(dtype), atol=2e-3, rtol=2e-3))

    def test_full_layer_kivi_batched_store_matches_reference_pack(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        torch.manual_seed(7)
        device = "cuda"
        dtype = torch.float16
        num_blocks = 4
        num_slots = 7
        group_size = 32
        num_heads = 2
        head_dim = 64
        bits = 4

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = type("Config", (), {"full_layer_kv_quant_bits": bits, "full_layer_kivi_group_size": group_size})()
        manager.num_kv_heads = num_heads
        manager.head_dim = head_dim
        manager.full_layer_kivi_key_packed = torch.empty(
            1, num_slots, num_heads, head_dim, group_size // 8, device=device, dtype=torch.int32
        )
        manager.full_layer_kivi_key_scales = torch.empty(1, num_slots, num_heads, head_dim, device=device, dtype=dtype)
        manager.full_layer_kivi_key_mins = torch.empty_like(manager.full_layer_kivi_key_scales)
        manager.full_layer_kivi_value_packed = torch.empty(
            1, num_slots, num_heads, group_size, head_dim // 8, device=device, dtype=torch.int32
        )
        manager.full_layer_kivi_value_scales = torch.empty(
            1, num_slots, num_heads, group_size, head_dim // group_size, device=device, dtype=dtype
        )
        manager.full_layer_kivi_value_mins = torch.empty_like(manager.full_layer_kivi_value_scales)

        block_slots = torch.tensor([3, 0, 6, 2], device=device, dtype=torch.int32)
        key_blocks = torch.randn(num_blocks, group_size, num_heads, head_dim, device=device, dtype=dtype)
        value_blocks = torch.randn_like(key_blocks)

        DeltaKVLessMemoryCacheManager._store_full_layer_kivi_blocks(
            manager,
            l_idx=0,
            block_slots=block_slots,
            key_post_rope=key_blocks,
            value=value_blocks,
        )
        torch.cuda.synchronize()

        key_states = key_blocks.permute(0, 2, 3, 1).contiguous()
        packed_k, scale_k, mn_k = triton_quantize_and_pack_along_last_dim(key_states, group_size, bits)
        value_states = value_blocks.permute(0, 2, 1, 3).contiguous()
        packed_v, scale_v, mn_v = triton_quantize_and_pack_along_last_dim(value_states, group_size, bits)
        slots = block_slots.long()
        self.assertTrue(torch.equal(manager.full_layer_kivi_key_packed[0, slots], packed_k.to(torch.int32)))
        self.assertTrue(torch.equal(manager.full_layer_kivi_value_packed[0, slots], packed_v.to(torch.int32)))
        self.assertTrue(torch.allclose(manager.full_layer_kivi_key_scales[0, slots], scale_k.squeeze(-1)))
        self.assertTrue(torch.allclose(manager.full_layer_kivi_key_mins[0, slots], mn_k.squeeze(-1)))
        self.assertTrue(torch.allclose(manager.full_layer_kivi_value_scales[0, slots], scale_v))
        self.assertTrue(torch.allclose(manager.full_layer_kivi_value_mins[0, slots], mn_v))

    def test_full_prefill_kivi_materialize_chunks_block_store(self):
        from sparseengine.engine.cache_manager.methods.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        torch.manual_seed(13)
        device = "cuda"
        dtype = torch.float16
        group_size = 8
        num_blocks = 5
        num_heads = 2
        head_dim = 8
        total_tokens = num_blocks * group_size

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = type(
            "Config",
            (),
            {
                "full_layer_kivi_group_size": group_size,
                "mlp_chunk_size": 16,
            },
        )()
        manager.full_layer_to_idx = {3: 0}
        manager._full_layer_kivi_full_prefill_materialized_layers = set()
        manager._full_layer_kivi_full_prefill_plans = {
            0: {
                "keep_pos": torch.empty((0,), device=device, dtype=torch.int32),
                "keep_slots": torch.empty((0,), device=device, dtype=torch.int32),
                "block_start_pos": torch.arange(0, total_tokens, group_size, device=device, dtype=torch.int32),
                "block_slots": torch.arange(num_blocks, device=device, dtype=torch.int32),
            }
        }
        manager.deltakv_prefill_staging_kv_cache = torch.randn(
            2,
            total_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        manager.full_kv_cache = torch.empty(2, 1, 0, num_heads, head_dim, device=device, dtype=dtype)
        manager.num_kv_heads = num_heads
        manager.head_dim = head_dim
        chunk_sizes = []

        def record_store(l_idx, block_slots, key_post_rope, value):
            self.assertEqual(l_idx, 0)
            chunk_sizes.append(int(block_slots.numel()))
            expected_shape = (int(block_slots.numel()), group_size, num_heads, head_dim)
            self.assertEqual(tuple(key_post_rope.shape), expected_shape)
            self.assertEqual(tuple(value.shape), expected_shape)

        manager._store_full_layer_kivi_blocks = record_store
        DeltaKVLessMemoryCacheManager._full_layer_kivi_materialize_full_prefill_layer(manager, 3)

        self.assertEqual(chunk_sizes, [2, 2, 1])
        self.assertIn(3, manager._full_layer_kivi_full_prefill_materialized_layers)

    def test_full_layer_kivi_flash_decode_stage1_matches_dense_stage1(self):
        torch.manual_seed(2)
        device = "cuda"
        dtype = torch.float16
        batch = 2
        seq_len = 40
        sink = 8
        group_size = 32
        num_heads = 14
        num_kv_heads = 2
        head_dim = 64
        block_seq = 256
        bits = 4

        q = torch.randn(batch, num_heads, head_dim, device=device, dtype=dtype)
        full_k = torch.randn(batch, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        full_v = torch.randn_like(full_k)

        raw_slots_map = torch.full((batch, seq_len), -1, device=device, dtype=torch.int32)
        kivi_block_slots_map = torch.full_like(raw_slots_map, -1)
        kivi_block_start_pos = torch.full((batch,), -1, device=device, dtype=torch.int32)

        raw_k = torch.empty((batch * sink, num_kv_heads, head_dim), device=device, dtype=dtype)
        raw_v = torch.empty_like(raw_k)
        dense_k = torch.empty((batch * seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
        dense_v = torch.empty_like(dense_k)
        req_to_tokens = torch.empty((batch, seq_len), device=device, dtype=torch.int32)

        key_packed = torch.empty(
            batch,
            num_kv_heads,
            head_dim,
            group_size // 8,
            device=device,
            dtype=torch.int32,
        )
        key_scales = torch.empty((batch, num_kv_heads, head_dim), device=device, dtype=dtype)
        key_mins = torch.empty_like(key_scales)
        value_packed = torch.empty(
            batch,
            num_kv_heads,
            group_size,
            head_dim // 8,
            device=device,
            dtype=torch.int32,
        )
        value_scales = torch.empty(
            batch,
            num_kv_heads,
            group_size,
            head_dim // group_size,
            device=device,
            dtype=dtype,
        )
        value_mins = torch.empty_like(value_scales)

        for b in range(batch):
            for pos in range(seq_len):
                dense_slot = b * seq_len + pos
                req_to_tokens[b, pos] = dense_slot
            raw_start = b * sink
            raw_slots = torch.arange(raw_start, raw_start + sink, device=device, dtype=torch.int32)
            raw_slots_map[b, :sink] = raw_slots
            raw_k[raw_slots.long()] = full_k[b, :sink]
            raw_v[raw_slots.long()] = full_v[b, :sink]
            dense_k[b * seq_len: b * seq_len + sink] = full_k[b, :sink]
            dense_v[b * seq_len: b * seq_len + sink] = full_v[b, :sink]

            key_block = full_k[b, sink: sink + group_size]
            value_block = full_v[b, sink: sink + group_size]
            key_states = key_block.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
            packed_k, scale_k, mn_k = triton_quantize_and_pack_along_last_dim(key_states, group_size, bits)
            key_packed[b] = packed_k.squeeze(0)
            key_scales[b] = scale_k.squeeze(0).squeeze(-1)
            key_mins[b] = mn_k.squeeze(0).squeeze(-1)
            value_states = value_block.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            packed_v, scale_v, mn_v = triton_quantize_and_pack_along_last_dim(value_states, group_size, bits)
            value_packed[b] = packed_v.squeeze(0)
            value_scales[b] = scale_v.squeeze(0)
            value_mins[b] = mn_v.squeeze(0)

            key_dequant = unpack_quantized_to_16bit(
                key_packed[b].unsqueeze(0),
                key_scales[b].unsqueeze(0).unsqueeze(-1),
                key_mins[b].unsqueeze(0).unsqueeze(-1),
                group_size,
                bits,
            ).permute(0, 3, 1, 2).squeeze(0)
            value_dequant = unpack_quantized_to_16bit(
                value_packed[b].unsqueeze(0),
                value_scales[b].unsqueeze(0),
                value_mins[b].unsqueeze(0),
                group_size,
                bits,
            ).permute(0, 2, 1, 3).squeeze(0)
            dense_k[b * seq_len + sink: (b + 1) * seq_len] = key_dequant
            dense_v[b * seq_len + sink: (b + 1) * seq_len] = value_dequant
            kivi_block_slots_map[b, sink: sink + group_size] = b
            kivi_block_start_pos[b] = sink

        req_indices = torch.arange(batch, device=device, dtype=torch.int32)
        context_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
        row_kivi_quantized_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
        num_blocks = (seq_len + block_seq - 1) // block_seq
        mid_ref = torch.empty((batch, num_heads, num_blocks, head_dim), device=device, dtype=torch.float32)
        lse_ref = torch.empty((batch, num_heads, num_blocks), device=device, dtype=torch.float32)
        mid_fused = torch.empty_like(mid_ref)
        lse_fused = torch.empty_like(lse_ref)
        mid_grouped = torch.empty_like(mid_ref)
        lse_grouped = torch.empty_like(lse_ref)
        dense_k_view = torch.empty_like(dense_k)
        dense_v_view = torch.empty_like(dense_v)
        dense_req_to_tokens = torch.empty_like(req_to_tokens)
        score_ref = torch.full((batch, num_heads, seq_len), -1e20, device=device, dtype=torch.float32)
        score_fused = torch.full_like(score_ref, -1e20)
        score_grouped = torch.full_like(score_ref, -1e20)

        full_layer_kivi_build_dense_decode_view(
            raw_k=raw_k,
            raw_v=raw_v,
            raw_slots_map=raw_slots_map,
            kivi_block_slots_map=kivi_block_slots_map,
            kivi_block_start_pos=kivi_block_start_pos,
            key_packed=key_packed,
            key_scales=key_scales,
            key_mins=key_mins,
            value_packed=value_packed,
            value_scales=value_scales,
            value_mins=value_mins,
            row_kivi_quantized_lens=row_kivi_quantized_lens,
            req_indices=req_indices,
            context_lens=context_lens,
            max_len_in_batch=seq_len,
            dense_req_to_tokens=dense_req_to_tokens,
            dense_k=dense_k_view,
            dense_v=dense_v_view,
            group_size=group_size,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(dense_req_to_tokens, req_to_tokens))
        self.assertTrue(torch.allclose(dense_k_view, dense_k, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(dense_v_view, dense_v, atol=3e-2, rtol=3e-2))

        gqa_flash_decode_stage1(
            q,
            dense_k,
            dense_v,
            req_to_tokens,
            req_indices,
            context_lens,
            seq_len,
            mid_ref,
            lse_ref,
            block_seq,
        )
        full_layer_kivi_flash_decode_stage1(
            q=q,
            raw_k=raw_k,
            raw_v=raw_v,
            raw_slots_map=raw_slots_map,
            kivi_block_slots_map=kivi_block_slots_map,
            kivi_block_start_pos=kivi_block_start_pos,
            key_packed=key_packed,
            key_scales=key_scales,
            key_mins=key_mins,
            value_packed=value_packed,
            value_scales=value_scales,
            value_mins=value_mins,
            req_indices=req_indices,
            context_lens=context_lens,
            max_len_in_batch=seq_len,
            mid_out=mid_fused,
            mid_out_logsumexp=lse_fused,
            group_size=group_size,
            block_seq=block_seq,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(mid_fused, mid_ref, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(lse_fused, lse_ref, atol=3e-2, rtol=3e-2))
        full_layer_kivi_flash_decode_stage1_grouped(
            q=q,
            raw_k=raw_k,
            raw_v=raw_v,
            raw_slots_map=raw_slots_map,
            kivi_block_slots_map=kivi_block_slots_map,
            key_packed=key_packed,
            key_scales=key_scales,
            key_mins=key_mins,
            value_packed=value_packed,
            value_scales=value_scales,
            value_mins=value_mins,
            row_kivi_quantized_lens=row_kivi_quantized_lens,
            req_indices=req_indices,
            context_lens=context_lens,
            max_len_in_batch=seq_len,
            mid_out=mid_grouped,
            mid_out_logsumexp=lse_grouped,
            group_size=group_size,
            kivi_start=sink,
            block_seq=block_seq,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(mid_grouped, mid_ref, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(lse_grouped, lse_ref, atol=3e-2, rtol=3e-2))

        gqa_flash_decode_stage1_with_score(
            q,
            dense_k,
            dense_v,
            req_to_tokens,
            req_indices,
            context_lens,
            seq_len,
            mid_ref,
            lse_ref,
            score_ref,
            block_seq,
        )
        full_layer_kivi_flash_decode_stage1(
            q=q,
            raw_k=raw_k,
            raw_v=raw_v,
            raw_slots_map=raw_slots_map,
            kivi_block_slots_map=kivi_block_slots_map,
            kivi_block_start_pos=kivi_block_start_pos,
            key_packed=key_packed,
            key_scales=key_scales,
            key_mins=key_mins,
            value_packed=value_packed,
            value_scales=value_scales,
            value_mins=value_mins,
            req_indices=req_indices,
            context_lens=context_lens,
            max_len_in_batch=seq_len,
            mid_out=mid_fused,
            mid_out_logsumexp=lse_fused,
            group_size=group_size,
            block_seq=block_seq,
            attn_score=score_fused,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(mid_fused, mid_ref, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(lse_fused, lse_ref, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(score_fused, score_ref, atol=3e-2, rtol=3e-2))
        full_layer_kivi_flash_decode_stage1_grouped(
            q=q,
            raw_k=raw_k,
            raw_v=raw_v,
            raw_slots_map=raw_slots_map,
            kivi_block_slots_map=kivi_block_slots_map,
            key_packed=key_packed,
            key_scales=key_scales,
            key_mins=key_mins,
            value_packed=value_packed,
            value_scales=value_scales,
            value_mins=value_mins,
            row_kivi_quantized_lens=row_kivi_quantized_lens,
            req_indices=req_indices,
            context_lens=context_lens,
            max_len_in_batch=seq_len,
            mid_out=mid_grouped,
            mid_out_logsumexp=lse_grouped,
            group_size=group_size,
            kivi_start=sink,
            block_seq=block_seq,
            attn_score=score_grouped,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(mid_grouped, mid_ref, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(lse_grouped, lse_ref, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(score_grouped, score_ref, atol=3e-2, rtol=3e-2))

        max_kv_splits = 4
        target_tokens_per_split = 16
        mid_fixed = torch.empty(
            (batch, num_heads, max_kv_splits, head_dim),
            device=device,
            dtype=torch.float32,
        )
        lse_fixed = torch.empty(
            (batch, num_heads, max_kv_splits),
            device=device,
            dtype=torch.float32,
        )
        out_fixed = torch.empty_like(q)
        out_lse_fixed = torch.empty(
            (num_heads, batch),
            device=device,
            dtype=torch.float32,
        )
        score_fixed = torch.full_like(score_ref, -1e20)

        def run_fixed_grid():
            full_layer_kivi_flash_decode_stage1(
                q=q,
                raw_k=raw_k,
                raw_v=raw_v,
                raw_slots_map=raw_slots_map,
                kivi_block_slots_map=kivi_block_slots_map,
                kivi_block_start_pos=kivi_block_start_pos,
                key_packed=key_packed,
                key_scales=key_scales,
                key_mins=key_mins,
                value_packed=value_packed,
                value_scales=value_scales,
                value_mins=value_mins,
                req_indices=req_indices,
                context_lens=context_lens,
                max_len_in_batch=seq_len,
                mid_out=mid_fixed,
                mid_out_logsumexp=lse_fixed,
                group_size=group_size,
                block_seq=block_seq,
                attn_score=score_fixed,
                max_kv_splits=max_kv_splits,
                target_tokens_per_split=target_tokens_per_split,
            )
            fixed_grid_flash_decode_stage2(
                mid_fixed,
                lse_fixed,
                context_lens,
                out_fixed,
                out_lse_fixed,
                target_tokens_per_split=target_tokens_per_split,
            )

        run_fixed_grid()
        torch.cuda.synchronize()
        output_ptr = out_fixed.data_ptr()
        workspace_ptrs = (mid_fixed.data_ptr(), lse_fixed.data_ptr())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_fixed_grid()

        context_lens.copy_(
            torch.tensor([17, seq_len], device=device, dtype=torch.int32)
        )
        q.copy_(torch.randn_like(q))
        graph.replay()
        torch.cuda.synchronize()

        expected = torch.empty_like(q)
        for batch_idx, length in enumerate(context_lens.tolist()):
            keys = dense_k[
                batch_idx * seq_len : batch_idx * seq_len + length
            ].repeat_interleave(num_heads // num_kv_heads, dim=1)
            values = dense_v[
                batch_idx * seq_len : batch_idx * seq_len + length
            ].repeat_interleave(num_heads // num_kv_heads, dim=1)
            logits = torch.einsum(
                "hd,lhd->hl",
                q[batch_idx].float(),
                keys.float(),
            )
            expected[batch_idx] = torch.einsum(
                "hl,lhd->hd",
                (logits / head_dim**0.5).softmax(-1),
                values.float(),
            ).to(dtype)
            torch.testing.assert_close(
                score_fixed[batch_idx, :, :length],
                logits,
                atol=3e-2,
                rtol=3e-2,
            )

        torch.testing.assert_close(out_fixed, expected, atol=3e-2, rtol=3e-2)
        self.assertEqual(out_fixed.data_ptr(), output_ptr)
        self.assertEqual(
            (mid_fixed.data_ptr(), lse_fixed.data_ptr()),
            workspace_ptrs,
        )

    def test_full_layer_kivi_token_map_flash_decode_stage1_matches_dense_stage1(self):
        torch.manual_seed(8)
        device = "cuda"
        dtype = torch.float16
        batch = 2
        seq_len = 47
        group_size = 32
        num_heads = 14
        num_kv_heads = 2
        head_dim = 64
        block_seq = 256
        bits = 4
        num_kivi_slots = 5

        q = torch.randn(batch, num_heads, head_dim, device=device, dtype=dtype)
        raw_k = torch.randn(12, num_kv_heads, head_dim, device=device, dtype=dtype)
        raw_v = torch.randn_like(raw_k)
        dense_k = torch.empty((batch * seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
        dense_v = torch.empty_like(dense_k)
        req_to_tokens = torch.empty((batch, seq_len), device=device, dtype=torch.int32)
        raw_slots_map = torch.full((batch, seq_len), -1, device=device, dtype=torch.int32)
        kivi_token_slots_map = torch.full_like(raw_slots_map, -1)
        kivi_token_offsets_map = torch.full_like(raw_slots_map, -1)

        key_packed = torch.empty(
            num_kivi_slots,
            num_kv_heads,
            head_dim,
            group_size // 8,
            device=device,
            dtype=torch.int32,
        )
        key_scales = torch.empty((num_kivi_slots, num_kv_heads, head_dim), device=device, dtype=dtype)
        key_mins = torch.empty_like(key_scales)
        value_packed = torch.empty(
            num_kivi_slots,
            num_kv_heads,
            group_size,
            head_dim // 8,
            device=device,
            dtype=torch.int32,
        )
        value_scales = torch.empty(
            num_kivi_slots,
            num_kv_heads,
            group_size,
            head_dim // group_size,
            device=device,
            dtype=dtype,
        )
        value_mins = torch.empty_like(value_scales)
        key_dequant_blocks = []
        value_dequant_blocks = []
        for slot in range(num_kivi_slots):
            key_block = torch.randn(group_size, num_kv_heads, head_dim, device=device, dtype=dtype)
            value_block = torch.randn_like(key_block)
            key_states = key_block.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
            packed_k, scale_k, mn_k = triton_quantize_and_pack_along_last_dim(key_states, group_size, bits)
            key_packed[slot] = packed_k.squeeze(0)
            key_scales[slot] = scale_k.squeeze(0).squeeze(-1)
            key_mins[slot] = mn_k.squeeze(0).squeeze(-1)
            value_states = value_block.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            packed_v, scale_v, mn_v = triton_quantize_and_pack_along_last_dim(value_states, group_size, bits)
            value_packed[slot] = packed_v.squeeze(0)
            value_scales[slot] = scale_v.squeeze(0)
            value_mins[slot] = mn_v.squeeze(0)
            key_dequant_blocks.append(
                unpack_quantized_to_16bit(
                    key_packed[slot].unsqueeze(0),
                    key_scales[slot].unsqueeze(0).unsqueeze(-1),
                    key_mins[slot].unsqueeze(0).unsqueeze(-1),
                    group_size,
                    bits,
                ).permute(0, 3, 1, 2).squeeze(0)
            )
            value_dequant_blocks.append(
                unpack_quantized_to_16bit(
                    value_packed[slot].unsqueeze(0),
                    value_scales[slot].unsqueeze(0),
                    value_mins[slot].unsqueeze(0),
                    group_size,
                    bits,
                ).permute(0, 2, 1, 3).squeeze(0)
            )
        key_dequant_blocks = torch.stack(key_dequant_blocks, dim=0)
        value_dequant_blocks = torch.stack(value_dequant_blocks, dim=0)

        raw_positions = {
            0: {0: 0, 1: 1, 4: 2, 19: 3, 46: 4},
            1: {0: 5, 3: 6, 8: 7, 21: 8},
        }
        for b in range(batch):
            for pos in range(seq_len):
                dense_slot = b * seq_len + pos
                req_to_tokens[b, pos] = dense_slot
                raw_slot = raw_positions.get(b, {}).get(pos)
                if raw_slot is not None:
                    raw_slots_map[b, pos] = raw_slot
                    dense_k[dense_slot] = raw_k[raw_slot]
                    dense_v[dense_slot] = raw_v[raw_slot]
                    continue
                kivi_slot = (pos * 3 + b) % num_kivi_slots
                local_offset = (pos * 5 + b * 7) % group_size
                kivi_token_slots_map[b, pos] = kivi_slot
                kivi_token_offsets_map[b, pos] = local_offset
                dense_k[dense_slot] = key_dequant_blocks[kivi_slot, local_offset]
                dense_v[dense_slot] = value_dequant_blocks[kivi_slot, local_offset]

        req_indices = torch.arange(batch, device=device, dtype=torch.int32)
        context_lens = torch.tensor([seq_len, seq_len - 3], device=device, dtype=torch.int32)
        num_blocks = (seq_len + block_seq - 1) // block_seq
        mid_ref = torch.empty((batch, num_heads, num_blocks, head_dim), device=device, dtype=torch.float32)
        lse_ref = torch.empty((batch, num_heads, num_blocks), device=device, dtype=torch.float32)
        mid_fused = torch.empty_like(mid_ref)
        lse_fused = torch.empty_like(lse_ref)
        score_ref = torch.full((batch, num_heads, seq_len), -1e20, device=device, dtype=torch.float32)
        score_fused = torch.full_like(score_ref, -1e20)

        gqa_flash_decode_stage1(
            q,
            dense_k,
            dense_v,
            req_to_tokens,
            req_indices,
            context_lens,
            seq_len,
            mid_ref,
            lse_ref,
            block_seq,
        )
        full_layer_kivi_flash_decode_stage1_token_map(
            q=q,
            raw_k=raw_k,
            raw_v=raw_v,
            raw_slots_map=raw_slots_map,
            kivi_token_slots_map=kivi_token_slots_map,
            kivi_token_offsets_map=kivi_token_offsets_map,
            key_packed=key_packed,
            key_scales=key_scales,
            key_mins=key_mins,
            value_packed=value_packed,
            value_scales=value_scales,
            value_mins=value_mins,
            req_indices=req_indices,
            context_lens=context_lens,
            max_len_in_batch=seq_len,
            mid_out=mid_fused,
            mid_out_logsumexp=lse_fused,
            group_size=group_size,
            block_seq=block_seq,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(mid_fused, mid_ref, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(lse_fused, lse_ref, atol=3e-2, rtol=3e-2))

        gqa_flash_decode_stage1_with_score(
            q,
            dense_k,
            dense_v,
            req_to_tokens,
            req_indices,
            context_lens,
            seq_len,
            mid_ref,
            lse_ref,
            score_ref,
            block_seq,
        )
        full_layer_kivi_flash_decode_stage1_token_map(
            q=q,
            raw_k=raw_k,
            raw_v=raw_v,
            raw_slots_map=raw_slots_map,
            kivi_token_slots_map=kivi_token_slots_map,
            kivi_token_offsets_map=kivi_token_offsets_map,
            key_packed=key_packed,
            key_scales=key_scales,
            key_mins=key_mins,
            value_packed=value_packed,
            value_scales=value_scales,
            value_mins=value_mins,
            req_indices=req_indices,
            context_lens=context_lens,
            max_len_in_batch=seq_len,
            mid_out=mid_fused,
            mid_out_logsumexp=lse_fused,
            group_size=group_size,
            block_seq=block_seq,
            attn_score=score_fused,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(mid_fused, mid_ref, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(lse_fused, lse_ref, atol=3e-2, rtol=3e-2))
        valid_score = torch.arange(seq_len, device=device).unsqueeze(0) < context_lens.unsqueeze(1)
        valid_score = valid_score.unsqueeze(1).expand(batch, num_heads, seq_len)
        self.assertTrue(torch.allclose(score_fused[valid_score], score_ref[valid_score], atol=3e-2, rtol=3e-2))


if __name__ == "__main__":
    unittest.main()
