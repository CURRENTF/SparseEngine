from unittest.mock import patch

import pytest

from sparseengine.distributed.parallel_context import ParallelContext, ParallelGroup
from sparseengine.utils.process_title import (
    engine_process_title,
    set_engine_process_title,
)


def _parallel_context(
    *,
    world_size: int,
    tp_size: int = 1,
    ep_size: int = 1,
    tp_rank: int = 0,
    ep_rank: int = 0,
    dp_rank: int = 0,
    dp_size: int = 1,
) -> ParallelContext:
    world = ParallelGroup(None, tuple(range(world_size)), 0, world_size)
    return ParallelContext(
        world=world,
        attn_tp=ParallelGroup(None, tuple(range(tp_size)), tp_rank, tp_size),
        moe_ep=ParallelGroup(None, tuple(range(ep_size)), ep_rank, ep_size),
        attn_dp=ParallelGroup(None, tuple(range(dp_size)), dp_rank, dp_size),
        moe_tp=ParallelGroup(None, tuple(range(tp_size)), tp_rank, tp_size),
    )


@pytest.mark.parametrize(
    ("parallel_context", "expected"),
    [
        (_parallel_context(world_size=1), "SENGINE_Engine"),
        (_parallel_context(world_size=2, tp_size=2, tp_rank=1), "SENGINE_TP1"),
        (
            _parallel_context(world_size=4, tp_size=2, ep_size=2, tp_rank=1, ep_rank=1),
            "SENGINE_TP1_EP1",
        ),
        (
            _parallel_context(
                world_size=8,
                tp_size=2,
                ep_size=2,
                tp_rank=1,
                ep_rank=1,
                dp_rank=1,
                dp_size=2,
            ),
            "SENGINE_TP1_DP1_EP1",
        ),
        (
            _parallel_context(world_size=2, ep_size=2, dp_size=2),
            "SENGINE_DP0_EP0",
        ),
        (
            _parallel_context(world_size=2, ep_size=2, dp_size=2, ep_rank=1, dp_rank=1),
            "SENGINE_DP1_EP1",
        ),
        (
            _parallel_context(world_size=8, tp_size=2, ep_size=2, dp_size=2),
            "SENGINE_TP0_DP0_EP0",
        ),
        (_parallel_context(world_size=2, ep_size=2), "SENGINE_EP0"),
        (_parallel_context(world_size=2, dp_size=2), "SENGINE_DP0"),
    ],
)
def test_engine_process_title(parallel_context, expected):
    assert engine_process_title(parallel_context) == expected


def test_set_engine_process_title_updates_os_title():
    parallel_context = _parallel_context(world_size=1)
    with patch("setproctitle.setproctitle") as setproctitle:
        set_engine_process_title(parallel_context)

    setproctitle.assert_called_once_with("SENGINE_Engine")
