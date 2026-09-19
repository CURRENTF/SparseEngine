from __future__ import annotations

import re


def device_name_contains(device_name: str, keyword: str) -> bool:
    """Match a product keyword without interpreting the device taxonomy."""

    name = re.sub(r"[\s/_-]+", "_", str(device_name).strip().upper())
    target = re.sub(r"[\s/_-]+", "_", str(keyword).strip().upper())
    return bool(target) and f"_{target}_" in f"_{name}_"
