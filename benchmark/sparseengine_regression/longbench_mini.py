from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.long_bench.pred import build_chat


DEFAULT_LONGBENCH_MINI_TASKS = [
    "qasper",
    "hotpotqa",
    "multi_news",
    "trec",
    "passage_retrieval_en",
    "lcc",
]


@dataclass(frozen=True)
class SelectedSample:
    source_idx: int
    prompt_tokens: int
    row: dict[str, Any]


def prompt_token_length(
    *,
    tokenizer,
    dataset: str,
    prompt_format: str,
    row: dict[str, Any],
    no_chat_template: bool = False,
    thinking_mode: str = "off",
) -> int:
    prompt = prompt_format.format(**row)
    prompt = build_chat(
        tokenizer,
        prompt,
        dataset,
        no_chat_template=no_chat_template,
        thinking_mode=thinking_mode,
    )
    add_special_tokens = True
    if tokenizer.bos_token is None or prompt.startswith(tokenizer.bos_token):
        add_special_tokens = False
    return len(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))


def select_longbench_mini_samples(
    *,
    data: list[dict[str, Any]],
    tokenizer,
    dataset: str,
    prompt_format: str,
    min_prompt_tokens: int = 16_000,
    samples_per_task: int = 20,
    min_required_samples: int = 5,
    no_chat_template: bool = False,
    thinking_mode: str = "off",
) -> tuple[list[SelectedSample], dict[str, Any]]:
    if min_prompt_tokens < 0:
        raise ValueError(f"min_prompt_tokens must be >= 0, got {min_prompt_tokens}.")
    if samples_per_task <= 0:
        raise ValueError(f"samples_per_task must be > 0, got {samples_per_task}.")
    if min_required_samples <= 0:
        raise ValueError(f"min_required_samples must be > 0, got {min_required_samples}.")

    selected: list[SelectedSample] = []
    considered = 0
    for idx, row in enumerate(data):
        considered += 1
        prompt_tokens = prompt_token_length(
            tokenizer=tokenizer,
            dataset=dataset,
            prompt_format=prompt_format,
            row=row,
            no_chat_template=no_chat_template,
            thinking_mode=thinking_mode,
        )
        if prompt_tokens >= int(min_prompt_tokens):
            selected.append(SelectedSample(source_idx=idx, prompt_tokens=prompt_tokens, row=row))
        if len(selected) >= int(samples_per_task):
            break

    status = "success" if len(selected) >= int(min_required_samples) else "skipped_by_policy"
    meta = {
        "status": status,
        "dataset": dataset,
        "considered_rows": considered,
        "selected_rows": len(selected),
        "min_prompt_tokens": int(min_prompt_tokens),
        "samples_per_task": int(samples_per_task),
        "min_required_samples": int(min_required_samples),
        "selected_source_indices": [item.source_idx for item in selected],
        "selected_prompt_tokens": [item.prompt_tokens for item in selected],
    }
    return selected, meta

