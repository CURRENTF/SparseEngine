# SPDX-License-Identifier: Apache-2.0
"""Deterministic random traces for matched efficiency benchmarks."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any


TRACE_GENERATOR_VERSION = "random-varlen-v2"


@dataclass(frozen=True)
class RequestTrace:
    request_index: int
    prompt_token_ids: list[int]
    output_len: int
    trace_seed: int
    prompt_digest: str

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def metadata(self) -> dict[str, Any]:
        return {
            "request_index": self.request_index,
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "trace_seed": self.trace_seed,
            "prompt_digest": self.prompt_digest,
            "status": "success",
        }


def derive_trace_seed(
    base_seed: int,
    *,
    scenario: str,
    phase: str,
    nominal_prompt_len: int,
    nominal_output_len: int,
    concurrency: int,
    iteration: int,
) -> int:
    """Derive a process-independent seed shared by every engine."""
    payload = "|".join(
        (
            TRACE_GENERATOR_VERSION,
            str(int(base_seed)),
            scenario,
            phase,
            str(int(nominal_prompt_len)),
            str(int(nominal_output_len)),
            str(int(concurrency)),
            str(int(iteration)),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _jittered_lengths(
    rng: random.Random,
    *,
    target_len: int,
    count: int,
    jitter_fraction: float,
    vary: bool,
) -> list[int]:
    if target_len <= 0 or count <= 0:
        raise ValueError(
            f"target_len and count must be positive, got target_len={target_len}, count={count}."
        )
    if not 0.0 <= jitter_fraction < 1.0:
        raise ValueError(
            f"jitter_fraction must be in [0, 1), got {jitter_fraction}."
        )
    if not vary or count == 1 or jitter_fraction == 0:
        return [target_len] * count

    jitter_tokens = max(1, int(round(target_len * jitter_fraction)))
    lower = max(1, target_len - jitter_tokens)
    candidates = list(range(lower, target_len + 1))
    if len(candidates) == 1:
        raise ValueError(
            "Variable-length traces require at least two representable lengths; "
            f"target_len={target_len}, jitter_fraction={jitter_fraction}."
        )

    lengths: list[int] = []
    while len(lengths) < count:
        cycle = candidates.copy()
        rng.shuffle(cycle)
        lengths.extend(cycle)
    lengths = lengths[:count]
    if len(set(lengths)) < 2:
        lengths[-1] = candidates[0] if lengths[0] != candidates[0] else candidates[-1]
    rng.shuffle(lengths)
    return lengths


def build_request_trace(
    *,
    seed: int,
    request_count: int,
    nominal_prompt_len: int,
    nominal_output_len: int,
    vocab_size: int,
    prompt_jitter_fraction: float,
    output_jitter_fraction: float,
    vary_output_lengths: bool,
) -> list[RequestTrace]:
    """Build a reproducible, non-prefix-sharing random request trace.

    ``nominal_*_len`` are upper bounds. Lengths are sampled from a lower
    jittered bound through that target so a benchmark never silently exceeds
    the requested model-length envelope.
    """
    if request_count <= 0:
        raise ValueError(f"request_count must be positive, got {request_count}.")
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be greater than one, got {vocab_size}.")

    rng = random.Random(int(seed))
    prompt_lengths = _jittered_lengths(
        rng,
        target_len=nominal_prompt_len,
        count=request_count,
        jitter_fraction=prompt_jitter_fraction,
        vary=request_count > 1,
    )
    output_lengths = _jittered_lengths(
        rng,
        target_len=nominal_output_len,
        count=request_count,
        jitter_fraction=output_jitter_fraction,
        vary=vary_output_lengths and request_count > 1,
    )

    token_low = 100 if vocab_size > 101 else 0
    token_high_exclusive = min(vocab_size, 32_000)
    if token_high_exclusive <= token_low:
        token_low = 0
        token_high_exclusive = vocab_size

    traces: list[RequestTrace] = []
    observed_digests: set[str] = set()
    for request_index, (prompt_len, output_len) in enumerate(
        zip(prompt_lengths, output_lengths)
    ):
        request_seed = rng.getrandbits(64)
        request_rng = random.Random(request_seed)
        prompt_token_ids = [
            request_rng.randrange(token_low, token_high_exclusive)
            for _ in range(prompt_len)
        ]
        digest = hashlib.sha256(
            ",".join(map(str, prompt_token_ids)).encode("ascii")
        ).hexdigest()
        if digest in observed_digests:
            raise RuntimeError(
                "Random trace generated a duplicate prompt within one case; "
                f"request_index={request_index}, seed={seed}."
            )
        observed_digests.add(digest)
        traces.append(
            RequestTrace(
                request_index=request_index,
                prompt_token_ids=prompt_token_ids,
                output_len=output_len,
                trace_seed=request_seed,
                prompt_digest=digest,
            )
        )
    return traces


def trace_metadata(traces: list[RequestTrace], *, allow_duplicate_prompts: bool = False) -> dict[str, Any]:
    prompt_lengths = [trace.prompt_len for trace in traces]
    output_lengths = [trace.output_len for trace in traces]
    digests = [trace.prompt_digest for trace in traces]
    if not allow_duplicate_prompts and len(digests) != len(set(digests)):
        raise RuntimeError("Trace metadata contains duplicate prompt digests.")
    return {
        "generator_version": TRACE_GENERATOR_VERSION,
        "request_count": len(traces),
        "prompt_lengths": prompt_lengths,
        "output_lengths": output_lengths,
        "unique_prompt_lengths": len(set(prompt_lengths)),
        "unique_output_lengths": len(set(output_lengths)),
        "prompt_digests": digests,
        "requests": [trace.metadata() for trace in traces],
    }
