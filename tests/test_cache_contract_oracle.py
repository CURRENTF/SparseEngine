"""Mutation tests of the independent oracle; runnable without the engine imports."""
import pytest
from cache_contracts.ownership import Pool, ContractViolation


def test_pool_counts_owners_not_logical_lengths():
    Pool('slots', 6, (0, 4), (('request', (1, 3)), ('shared-block', (2, 5)))).check()


@pytest.mark.parametrize('free,owners,reason', [
    ((0, 0), (('a', (1, 2)),), 'duplicate free'),
    ((0,), (('a', (1, 2)), ('b', (2,))), 'owned twice'),
    ((0, 1), (('a', (1, 2)),), 'free/live alias'),
    ((0,), (('a', (1,)),), 'missing='),
    ((0,), (('a', (1, 3)),), 'missing='),
])
def test_oracle_rejects_mutations(free, owners, reason):
    with pytest.raises(ContractViolation, match=reason):
        Pool('slots', 3, free, owners).check()


def test_compensating_counter_errors_cannot_fool_the_oracle():
    # "free + used == capacity" passes. Identity conservation still fails.
    free, owned = (0, 1), (1, 3)
    assert len(free) + len(owned) == 4
    with pytest.raises(ContractViolation):
        Pool('slots', 4, free, (('a', owned),)).check()
