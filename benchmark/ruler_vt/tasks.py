# SPDX-License-Identifier: Apache-2.0
"""Self-contained generators for the SparseEngine RULER core regression set.

The task contracts, prompts, and default complexity settings follow NVIDIA
RULER (commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a).  This module deliberately
keeps only tasks that do not require downloaded essays or QA corpora.  Its
deterministic synthetic word pools replace RULER's optional ``wonderwords``
and large English-word assets; consequently these artifacts are regression
compatible, not official leaderboard datasets.
"""

from __future__ import annotations

import hashlib
import math
import random
import string
from dataclasses import dataclass, field
from typing import Any, Callable


SUPPORTED_TASKS = (
    "niah_single_1",
    "niah_multikey_2",
    "vt",
    "cwe",
    "fwe",
)

TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "niah_single_1": {
        "category": "retrieval",
        "tokens_to_generate": 128,
        "max_new_tokens": 128,
        "type_haystack": "noise",
    },
    "niah_multikey_2": {
        "category": "retrieval",
        "tokens_to_generate": 128,
        "max_new_tokens": 128,
        "type_haystack": "needle",
    },
    "vt": {
        "category": "multi_hop_tracing",
        "tokens_to_generate": 30,
        "max_new_tokens": 30,
        "num_chains": 1,
        "num_hops": 4,
    },
    "cwe": {
        "category": "aggregation",
        "tokens_to_generate": 120,
        "max_new_tokens": 120,
        "freq_cw": 30,
        "freq_ucw": 3,
        "num_cw": 10,
    },
    "fwe": {
        "category": "aggregation",
        "tokens_to_generate": 50,
        "max_new_tokens": 50,
        "alpha": 2.0,
    },
}

NIAH_NOISE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)


@dataclass
class RulerSample:
    index: int
    context_length: int
    input: str
    outputs: list[str]
    length: int
    answer_prefix: str
    query: str
    task: str = "vt"
    metadata: dict[str, Any] = field(default_factory=dict)


def canonical_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized == "ruler_vt":
        return "vt"
    if normalized not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported RULER task {task!r}; supported tasks: {SUPPORTED_TASKS}."
        )
    return normalized


def resolve_task_config(task: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    task = canonical_task(task)
    config = dict(TASK_DEFAULTS[task])
    if overrides:
        config.update(overrides)

    def require_positive_integer(key: str) -> None:
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"RULER task {task} {key} must be a positive integer.")

    for key in ("tokens_to_generate", "max_new_tokens"):
        require_positive_integer(key)
    if config["tokens_to_generate"] != config["max_new_tokens"]:
        raise ValueError(
            f"RULER task {task} tokens_to_generate must equal max_new_tokens."
        )
    if task == "vt":
        require_positive_integer("num_chains")
        require_positive_integer("num_hops")
    if task == "cwe":
        for key in ("freq_cw", "freq_ucw", "num_cw"):
            require_positive_integer(key)
        if config["freq_cw"] <= config["freq_ucw"]:
            raise ValueError("RULER task cwe requires freq_cw > freq_ucw.")
    if task == "fwe":
        alpha = config.get("alpha")
        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not math.isfinite(float(alpha))
            or float(alpha) <= 1.0
        ):
            raise ValueError("RULER task fwe alpha must be finite and greater than 1.0.")
    if task == "niah_single_1" and config.get("type_haystack") != "noise":
        raise ValueError("RULER task niah_single_1 requires type_haystack=noise.")
    if task == "niah_multikey_2" and config.get("type_haystack") != "needle":
        raise ValueError("RULER task niah_multikey_2 requires type_haystack=needle.")
    for key in ("minimum_vanilla_score", "maximum_score_loss"):
        if key not in config:
            continue
        value = config[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 100.0
        ):
            raise ValueError(f"RULER task {task} {key} must be in (0, 100].")
    return config


def _sample_seed(seed: int, task: str, context_length: int, sample_index: int) -> int:
    payload = f"{seed}:{task}:{context_length}:{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _base26(value: int, width: int) -> str:
    chars: list[str] = []
    for _ in range(width):
        value, remainder = divmod(value, 26)
        chars.append(string.ascii_lowercase[remainder])
    return "".join(reversed(chars))


def _word(index: int) -> str:
    return f"ruler-{_base26(index, 4)}"


def _coded_word(index: int) -> str:
    return _base26(index, 6)


def _number(rng: random.Random) -> str:
    return str(rng.randint(1_000_000, 9_999_999))


def _fit_repetition_count(
    *,
    target_length: int,
    tokens_to_generate: int,
    count_tokens: Callable[[str], int],
    build: Callable[[int], tuple[str, list[str], str, str, dict[str, Any]]],
) -> tuple[str, list[str], str, str, dict[str, Any], int]:
    def measured(count: int) -> tuple[int, tuple[str, list[str], str, str, dict[str, Any]]]:
        payload = build(count)
        prompt, _outputs, answer_prefix, _query, _metadata = payload
        return count_tokens(prompt + answer_prefix) + tokens_to_generate, payload

    minimum_length, minimum_payload = measured(1)
    if minimum_length > target_length:
        raise ValueError(
            "RULER task prompt does not fit the requested context length: "
            f"minimum={minimum_length} target={target_length}."
        )

    lower = 1
    upper = 2
    upper_length, _ = measured(upper)
    while upper_length <= target_length:
        lower = upper
        upper *= 2
        upper_length, _ = measured(upper)

    best_payload = minimum_payload
    best_length = minimum_length
    while lower <= upper:
        middle = (lower + upper) // 2
        length, payload = measured(middle)
        if length <= target_length:
            best_payload = payload
            best_length = length
            lower = middle + 1
        else:
            upper = middle - 1
    return (*best_payload, best_length)


def _niah_payload(
    *,
    task: str,
    count: int,
    sample_seed: int,
) -> tuple[str, list[str], str, str, dict[str, Any]]:
    rng = random.Random(sample_seed)
    query = _word(rng.randrange(26**4))
    answer = _number(rng)
    needle = f"One of the special magic numbers for {query} is: {answer}."

    if task == "niah_single_1":
        sentences = [NIAH_NOISE] * count
    elif task == "niah_multikey_2":
        sentences = []
        for distractor_index in range(count):
            distractor_key_index = (
                distractor_index + rng.randrange(26**4)
            ) % (26**4)
            distractor_key = _word(distractor_key_index)
            while distractor_key == query:
                distractor_key_index = (distractor_key_index + 1) % (26**4)
                distractor_key = _word(distractor_key_index)
            distractor_value = _number(rng)
            while distractor_value == answer:
                distractor_value = _number(rng)
            sentences.append(
                "One of the special magic numbers for "
                f"{distractor_key} is: {distractor_value}."
            )
    else:  # pragma: no cover - guarded by canonical_task
        raise ValueError(f"Unsupported NIAH task: {task}.")

    insertion_index = rng.randrange(len(sentences) + 1)
    sentences.insert(insertion_index, needle)
    context = "\n".join(sentences)
    prompt = (
        "A special magic number is hidden within the following text. Make sure to "
        "memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {query} mentioned in the provided text?"
    )
    answer_prefix = (
        f" The special magic number for {query} mentioned in the provided text is:"
    )
    return prompt, [answer], answer_prefix, query, {
        "category": "retrieval",
        "needle_position": insertion_index,
        "haystack_items": count,
    }


def _cwe_payload(
    *,
    count: int,
    sample_seed: int,
    freq_cw: int,
    freq_ucw: int,
    num_cw: int,
) -> tuple[str, list[str], str, str, dict[str, Any]]:
    rng = random.Random(sample_seed)
    minimum_distinct = num_cw + 1
    distinct_count = max(minimum_distinct, count)
    offset = rng.randrange(max(1, 26**4 - distinct_count))
    vocabulary = [_word(offset + index) for index in range(distinct_count)]
    common = vocabulary[:num_cw]
    uncommon = vocabulary[num_cw:]
    words = common * freq_cw + uncommon * freq_ucw
    rng.shuffle(words)
    context = " ".join(f"{index + 1}. {word}" for index, word in enumerate(words))
    prompt = (
        "Below is a numbered list of words. In these words, some appear more often "
        "than others. Memorize the ones that appear most often.\n"
        f"{context}\n"
        f"Question: What are the {num_cw} most common words in the above list?"
    )
    answer_prefix = (
        f" Answer: The top {num_cw} words that appear most often in the list are:"
    )
    return prompt, common, answer_prefix, "", {
        "category": "aggregation",
        "distinct_words": distinct_count,
        "freq_cw": freq_cw,
        "freq_ucw": freq_ucw,
    }


def _zeta_approx(alpha: float) -> float:
    cutoff = 10_000
    partial = sum(index**-alpha for index in range(1, cutoff + 1))
    return partial + (cutoff ** (1.0 - alpha)) / (alpha - 1.0)


def _fwe_payload(
    *,
    count: int,
    sample_seed: int,
    alpha: float,
    vocab_size: int,
) -> tuple[str, list[str], str, str, dict[str, Any]]:
    rng = random.Random(sample_seed)
    vocabulary = [_coded_word(index) for index in range(vocab_size)]
    vocabulary[0] = "..."
    normalizer = _zeta_approx(alpha)
    words: list[str] = []
    for rank, word in enumerate(vocabulary, start=1):
        words.extend([word] * int(count * (rank**-alpha) / normalizer))
    rng.shuffle(words)
    context = " ".join(words)
    prompt = (
        "Read the following coded text and track the frequency of each coded word. "
        "Find the three most frequently appeared coded words. "
        f"{context}\n"
        "Question: Do not provide any explanation. Please ignore the dots '....'. "
        "What are the three most frequently appeared words in the above coded text?"
    )
    answer_prefix = (
        " Answer: According to the coded text above, the three most frequently "
        "appeared words are:"
    )
    return prompt, vocabulary[1:4], answer_prefix, "", {
        "category": "aggregation",
        "alpha": alpha,
        "vocab_size": vocab_size,
        "sampled_words": len(words),
    }


def generate_non_vt_samples(
    *,
    task: str,
    tokenizer: Any,
    context_lengths: list[int],
    samples_per_length: int,
    seed: int,
    config: dict[str, Any],
) -> list[RulerSample]:
    task = canonical_task(task)
    if task == "vt":
        raise ValueError("VT generation remains owned by VariableTrackingGenerator.")
    resolved = resolve_task_config(task, config)
    tokens_to_generate = int(resolved["tokens_to_generate"])

    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    samples: list[RulerSample] = []
    sample_index = 0
    for context_length in context_lengths:
        for index_at_length in range(samples_per_length):
            local_seed = _sample_seed(seed, task, context_length, index_at_length)
            if task.startswith("niah_"):
                build = lambda count, local_seed=local_seed: _niah_payload(
                    task=task,
                    count=count,
                    sample_seed=local_seed,
                )
            elif task == "cwe":
                build = lambda count, local_seed=local_seed: _cwe_payload(
                    count=count,
                    sample_seed=local_seed,
                    freq_cw=int(resolved["freq_cw"]),
                    freq_ucw=int(resolved["freq_ucw"]),
                    num_cw=int(resolved["num_cw"]),
                )
            elif task == "fwe":
                vocab_size = max(64, min(2000, context_length // 50))
                build = lambda count, local_seed=local_seed, vocab_size=vocab_size: _fwe_payload(
                    count=count,
                    sample_seed=local_seed,
                    alpha=float(resolved["alpha"]),
                    vocab_size=vocab_size,
                )
            else:  # pragma: no cover - guarded by canonical_task
                raise ValueError(f"Unsupported RULER task: {task}.")

            prompt, outputs, answer_prefix, query, metadata, length = (
                _fit_repetition_count(
                    target_length=context_length,
                    tokens_to_generate=tokens_to_generate,
                    count_tokens=token_count,
                    build=build,
                )
            )
            samples.append(
                RulerSample(
                    index=sample_index,
                    context_length=context_length,
                    input=prompt,
                    outputs=outputs,
                    length=length,
                    answer_prefix=answer_prefix,
                    query=query,
                    task=task,
                    metadata=metadata,
                )
            )
            sample_index += 1
    return samples
