"""Shared split-KV launch and live-scratch accounting."""


def select_prefill_splits(*, heads, batch, max_q, max_k, sm_count):
    if min(heads, batch, sm_count) <= 0:
        raise ValueError("MLA prefill requires positive heads, batch and SM count.")
    if max_q <= 0 or max_k <= 0:
        return 1
    programs = ((max_q + 127) // 128) * heads * batch
    waves = (2 * sm_count + programs - 1) // programs
    key_chunks = (max_k + 255) // 256
    return min(64, 1 << (waves - 1).bit_length(), 1 << (key_chunks - 1).bit_length())


def split_workspace_bytes(*, tokens, heads, splits, value_dim):
    if splits == 1:
        return 0
    return splits * heads * tokens * (value_dim + 1) * 4
