"""Real device transport tier. The strict validator treats a skip as failure."""
import pytest
import torch
from cache_contracts.cases import CHAIN_CASES, make_chain
from cache_contracts.ownership import ChainHarness


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA conformance requires a real GPU')
@pytest.mark.parametrize('case', [c for c in CHAIN_CASES if not c.shared], ids=lambda c: c.id)
def test_real_chain_transport_and_method_state(case):
    from test_chain_offload import round_trip
    m = make_chain(case, device='cuda:0')
    round_trip(m)
    torch.cuda.synchronize()
    ChainHarness(m).observe()
