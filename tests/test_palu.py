"""Independent algebra and GPU oracles for Palu; catches orientation/slot/graph errors."""
from types import SimpleNamespace
import pytest
import torch

from sparseengine.models.palu import factorize_grouped, fuse_value_output


def test_grouped_factorization_and_output_fusion():
    torch.manual_seed(13)
    h, d, hidden, g, qh = 4, 8, 48, 2, 8
    weight = torch.randn(h*d, hidden)
    a, b = factorize_grouped(weight, g, d, g*d)
    x = torch.randn(5, hidden)
    latent = torch.einsum('ti,gri->tgr', x, a)
    reconstructed = torch.einsum('thr,hrd->thd', latent.repeat_interleave(g, dim=1), b)
    torch.testing.assert_close(reconstructed.flatten(1), x @ weight.T, atol=3e-5, rtol=1e-5)
    output = torch.randn(hidden, qh*d)
    z = torch.randn(5, qh, g*d)
    reference = torch.einsum('tqr,qrd->tqd', z, b.repeat_interleave(qh//h, dim=0)).flatten(1) @ output.T
    torch.testing.assert_close(z.flatten(1) @ fuse_value_output(output, b, qh).T, reference, atol=8e-5, rtol=1e-5)


def _rotate(x, pos, rope):
    a, b = x.float().chunk(2, -1)
    c, s = rope[pos].float().chunk(2, -1)
    return torch.cat((a*c-b*s, b*c+a*s), -1).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype,normalize,rk,rv,d', [(torch.bfloat16, True, 64, 64,128), (torch.float16, False, 96, 32,128), (torch.bfloat16,False,128,128,64), (torch.bfloat16,True,256,192,256)])
def test_fused_attention_reference_and_graph_slot_reuse(dtype, normalize, rk, rv,d):
    from sparseengine.engine.cache_manager.base import LowRankKVPayload, LowRankKVWrite
    from sparseengine.kernels.triton.palu import decode, materialize_prefill, rotate_query, store_latent
    torch.manual_seed(17)
    device = 'cuda'
    b, h, qh, g, capacity, slots_count, splits = 3, 4, 8, 2, 273, 900, 8
    zk = torch.randn(slots_count, h//g, rk, dtype=dtype, device=device) * .15
    zv = torch.randn(slots_count, h//g, rv, dtype=dtype, device=device)
    positions = torch.arange(slots_count, dtype=torch.int32, device=device) % 512
    payload = LowRankKVPayload(zk, zv, positions)
    key_up = torch.randn(h, rk, d, dtype=dtype, device=device) * .15
    norm = torch.randn(d, dtype=dtype, device=device).abs() if normalize else None
    angles = torch.randn(512, 1, d//2, device=device)
    rope = torch.cat((angles.cos(), angles.sin()), -1)
    table = torch.randperm(slots_count, device=device)[:b*capacity].reshape(b, capacity).int()
    rows = torch.tensor([2, 0, 1], dtype=torch.int32, device=device)
    lens = torch.tensor([17, 139, 0], dtype=torch.int32, device=device)
    q = torch.randn(b, qh, d, device=device, dtype=dtype)
    view = SimpleNamespace(payload=payload, meta=SimpleNamespace(active_slots=table, req_indices=rows, context_lens=lens))
    mid = torch.empty(b, qh, splits, rv, device=device)
    lse = torch.empty(b, qh, splits, device=device)
    output = torch.empty(b, qh, rv, dtype=dtype, device=device)
    eps = 1e-6

    def reconstruct(indices):
        latent = zk[indices].repeat_interleave(g, dim=1)
        k = torch.einsum('thr,hrd->thd', latent.float(), key_up.float()).to(dtype)
        if normalize:
            k = (k.float() * torch.rsqrt(k.float().square().mean(-1, keepdim=True) + eps) * norm.float()).to(dtype)
        return _rotate(k, positions[indices].long(), rope)

    def reference():
        expected = torch.zeros_like(output)
        for i, (row, length) in enumerate(zip(rows.tolist(), lens.tolist())):
            if length == 0:
                continue
            indices = table[row, :length].long()
            k = reconstruct(indices).repeat_interleave(qh//h, dim=1)
            v = zv[indices].repeat_interleave(g * qh//h, dim=1)
            scores = torch.einsum('hd,thd->ht', q[i].float(), k.float()) * d**-.5
            expected[i] = torch.einsum('ht,thr->hr', scores.softmax(-1), v.float()).to(dtype)
        return expected

    from sparseengine.operators.palu_attention import PaluSpec, TritonPaluDecode
    op = TritonPaluDecode(op_spec=PaluSpec(qh,h,d,g,rk,rv,dtype,b,capacity), device=torch.device('cuda'))
    output = op.output
    def run():
        return op.run(q, view, (key_up,norm,rope,eps))

    pos = positions[:b].long()
    torch.testing.assert_close(rotate_query(q, pos, rope), _rotate(q, pos, rope), atol=.02, rtol=.02)
    k, v = materialize_prefill(payload, key_up, norm, rope, table, 0, 139, g, eps)
    torch.testing.assert_close(k, reconstruct(table[0, :139].long()), atol=.02, rtol=.02)
    torch.testing.assert_close(v[..., :rv], zv[table[0, :139].long()].repeat_interleave(g, dim=1))
    one_view = SimpleNamespace(payload=payload, meta=SimpleNamespace(active_slots=table, req_indices=rows[:1], context_lens=lens[:1]))
    torch.testing.assert_close(op.run(q[:1],one_view,(key_up,norm,rope,eps)), reference()[:1], atol=.012, rtol=.025)
    run()
    torch.testing.assert_close(output, reference(), atol=.012 if dtype == torch.bfloat16 else .003, rtol=.025)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Replay changes lengths and row ownership, including a padded zero-length row.
    for lengths, row_ids in [([273, 65, 1], [1, 2, 0]), ([2, 0, 217], [0, 1, 2])]:
        lens.copy_(torch.tensor(lengths, device=device, dtype=torch.int32))
        rows.copy_(torch.tensor(row_ids, device=device, dtype=torch.int32))
        graph.replay()
        torch.testing.assert_close(output, reference(), atol=.012 if dtype == torch.bfloat16 else .003, rtol=.025)
    # Padded writes must not corrupt the last cache slot; active slot gets its position.
    before = zk[-1].clone()
    write = LowRankKVWrite(torch.zeros(2, h//g, rk, device=device, dtype=dtype),
                           torch.zeros(2, h//g, rv, device=device, dtype=dtype),
                           torch.tensor([123, 456], device=device))
    store_latent(write, torch.tensor([7, -1], device=device, dtype=torch.int32), payload)
    torch.testing.assert_close(zk[-1], before)
    assert zk[7].count_nonzero().item() == 0 and positions[7].item() == 123

@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_chunked_prefill_matches_full_causal_reference():
    from sparseengine.engine.cache_manager.base import LowRankKVPayload
    from sparseengine.operators.palu_attention import PaluSpec, FlashInferPaluPrefill
    torch.manual_seed(21)
    t, h, qh, g, d, r = 137, 2, 4, 1, 128, 96
    dtype, device = torch.bfloat16, torch.device('cuda')
    zk = torch.randn(t, h, r, device=device, dtype=dtype) * .1
    zv = torch.randn_like(zk)
    bk = torch.randn(h, r, d, device=device, dtype=dtype) * .1
    angles = torch.randn(t, 1, d//2, device=device)
    rope = torch.cat((angles.cos(), angles.sin()), -1)
    positions = torch.arange(t, device=device, dtype=torch.int32)
    q = _rotate(torch.randn(t, qh, d, device=device, dtype=dtype), positions.long(), rope)
    view = SimpleNamespace(payload=LowRankKVPayload(zk, zv, positions),
                           meta=SimpleNamespace(active_slots=positions[None]))
    op = FlashInferPaluPrefill(op_spec=PaluSpec(qh,h,d,g,r,r,dtype,1,t), device=device)
    weights = bk, None, rope, 0.
    full = op.run(q, view, weights, [(0,t,0,t)])
    chunked = torch.cat([op.run(q[:65], view, weights, [(0,65,0,65)]),
                         op.run(q[65:], view, weights, [(0,t-65,0,t)])])
    k = _rotate(torch.einsum('thr,hrd->thd',zk.float(),bk.float()).to(dtype), positions.long(), rope).repeat_interleave(qh//h,1)
    v = zv.repeat_interleave(qh//h,1)
    scores = torch.einsum('thd,shd->hts',q.float(),k.float()) * d**-.5
    scores.masked_fill_(torch.ones(t,t,device=device,dtype=torch.bool).triu(1), -float('inf'))
    expected = torch.einsum('hts,shr->thr',scores.softmax(-1),v.float()).to(dtype)
    torch.testing.assert_close(full, expected, atol=.015, rtol=.03)
    torch.testing.assert_close(chunked, full, atol=.015, rtol=.03)


def test_low_rank_storage_accounting_and_synthetic_history_isolation():
    from sparseengine.engine.cache_manager.storage.low_rank_kv import LowRankKVStorage
    manifest={'model':{'num_key_value_heads':4},'group_size':2,'ranks':[[32,48],[48,32],[32,48]]}
    storage=LowRankKVStorage(manifest,dtype=torch.bfloat16)
    storage.allocate(num_layers=3,num_slots=19,device='cpu')
    actual=sum(t.numel()*t.element_size() for t in storage.accounting_tensors())
    assert actual == 19*storage.bytes_per_slot()
    first=storage.layer_payload(0)
    first.key_latent.fill_(3)
    storage.copy_slots(0,torch.tensor([3,4]),torch.tensor([1,2]))
    torch.testing.assert_close(first.key_latent[[1,2]],first.key_latent[[3,4]])
    storage.allocate_shared_history(num_slots=19,device='cpu')
    assert storage.layer_payload(0) is storage.layer_payload(2)
    assert storage.layer_payload(0) is not storage.layer_payload(1)
    storage.layer_payload(0).key_latent[-1].fill_(2)
    assert storage.layer_payload(2).key_latent[:-1].count_nonzero() == 0


def test_factor_source_mismatch_fails_before_projection_mutation():
    from sparseengine.models.palu import PaluSelfAttention
    # Reject a wrong sidecar before touching projections, without GPU work.
    module=PaluSelfAttention.__new__(PaluSelfAttention)
    torch.nn.Module.__init__(module)
    module.q_heads,module.kv_heads,module.dim=2,1,16
    module.qkv_proj=torch.nn.Linear(32,64,bias=False)
    weight=module.qkv_proj.weight.clone()
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        module.prepare_weights({},0,{})
    torch.testing.assert_close(module.qkv_proj.weight,weight)


def test_whitening_minimizes_activation_weighted_reconstruction_error():
    torch.manual_seed(31)
    weight=torch.randn(16,32)
    scale=torch.diag(torch.logspace(-2,2,32))
    def reconstructed(r,whiten):
        a,b=factorize_grouped(weight,2,8,r,scale if whiten else None)
        return torch.einsum('hrd,hri->hdi',b,a.repeat_interleave(2,0)).flatten(0,1)
    full=reconstructed(16,True)
    # Whitening optimizes the scaled error; unscaling amplifies FP32 error in
    # this deliberately ill-conditioned covariance's almost-unused channels.
    assert torch.linalg.norm((full-weight)@scale) / torch.linalg.norm(weight@scale) < 1e-5
    plain_error=torch.linalg.norm((weight-reconstructed(8,False))@scale)
    whitened_error=torch.linalg.norm((weight-reconstructed(8,True))@scale)
    assert whitened_error < plain_error*.5


def test_manifest_rejects_wrong_architecture_and_out_of_bounds_rank(tmp_path):
    import json
    from sparseengine.configs.palu import read_palu_manifest
    hf=SimpleNamespace(model_type='llama',hidden_size=256,num_attention_heads=4,
                       num_key_value_heads=2,num_hidden_layers=1,head_dim=64)
    manifest={'format':'sparsevllm.palu.v1','model':vars(hf).copy(),'group_size':1,'ranks':[[32,32]]}
    (tmp_path/'palu.safetensors').touch()
    path=tmp_path/'palu.json'
    path.write_text(json.dumps(manifest))
    assert read_palu_manifest(tmp_path,hf)['ranks'] == [[32,32]]
    manifest['model']['hidden_size']=512
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='architecture mismatch'):
        read_palu_manifest(tmp_path,hf)
    manifest['model']['hidden_size']=256
    manifest['ranks']=[[80,32]]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='ranks must'):
        read_palu_manifest(tmp_path,hf)
