import pytest

from sparseengine.utils.device_name import device_name_contains


@pytest.mark.parametrize(
    ("device_name", "keyword"),
    [
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA H100-SXM5-80GB", "h100"),
        ("NVIDIA H100 PCIe", "H100"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX PRO 6000"),
    ],
)
def test_device_name_contains_product_keyword(device_name: str, keyword: str) -> None:
    assert device_name_contains(device_name, keyword)


@pytest.mark.parametrize(
    ("device_name", "keyword"),
    [
        ("NVIDIA H20", "H100"),
        ("NVIDIA H1000", "H100"),
        ("unprofiled SM90 GPU", "H100"),
    ],
)
def test_device_name_keyword_does_not_match_partial_token(
    device_name: str,
    keyword: str,
) -> None:
    assert not device_name_contains(device_name, keyword)
