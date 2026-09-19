from unittest.mock import Mock, patch

import pytest
import torch

from sparseengine.operators.quest_scoring import QuestPageScoreDispatch, QuestPageScoreSpec
from sparseengine.platforms.interface import DeviceCaps, PlatformEnum


def _provider():
    return QuestPageScoreDispatch(
        spec=QuestPageScoreSpec(torch.bfloat16, 32, 4, 128, True),
        caps=DeviceCaps(
            platform=PlatformEnum.CUDA, device_type="cuda", device_index=0,
            device_name="test device", compute_capability=(9, 0),
            supports_triton=True, supports_bfloat16=True,
            supports_graph_capture=True, multi_processor_count=132,
        ),
    )


def test_quest_score_dispatch_propagates_bound_kernel_failure():
    """A failed prepared matrix-product path must not silently retry scoring."""
    provider = _provider()
    provider.scalar.score = Mock()
    provider.tensorcore.score = Mock(side_effect=RuntimeError("score launch failed"))
    query = torch.empty(1, 32, 128, dtype=torch.bfloat16)
    metadata = torch.empty(1, 4, 128, dtype=torch.bfloat16)
    pages = torch.empty(1, 8320, dtype=torch.int32)
    with (
        patch("sparseengine.operators.quest_scoring.device_runtime.is_stream_capturing", return_value=False),
        pytest.raises(RuntimeError, match="score launch failed"),
    ):
        provider.score(query, metadata, metadata, pages)
    provider.scalar.score.assert_not_called()


def test_quest_score_rejects_reusing_provider_for_different_head_contract():
    """A scorer bound before Graph capture must reject a different model layout."""
    provider = _provider()
    provider.tensorcore.score = Mock()
    with pytest.raises(ValueError, match="bound head contract"):
        provider.score(
            torch.empty(1, 1, 576, dtype=torch.bfloat16),
            torch.empty(1, 1, 576, dtype=torch.bfloat16),
            torch.empty(1, 1, 576, dtype=torch.bfloat16),
            torch.empty(1, 8320, dtype=torch.int32),
        )
    provider.tensorcore.score.assert_not_called()


def test_quest_score_cost_estimate_cannot_override_capability_rejection():
    """Large estimated savings cannot enable unsupported matrix instructions."""
    provider = _provider()
    provider.tensorcore_supported = False
    provider.tensorcore.score = Mock(side_effect=AssertionError("unsupported kernel"))
    provider.scalar.score = Mock(return_value="scalar")
    q = torch.empty(8, 32, 128, dtype=torch.bfloat16)
    metadata = torch.empty(1, 4, 128, dtype=torch.bfloat16)
    with patch("sparseengine.operators.quest_scoring.device_runtime.is_stream_capturing", return_value=False):
        provider.score(q, metadata, metadata, torch.empty(8, 32768, dtype=torch.int32))
    provider.tensorcore.score.assert_not_called()
