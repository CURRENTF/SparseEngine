#!/usr/bin/env python3
"""Run a deterministic Deep Research-style workload through the smart router."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import Executor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from urllib.request import Request
from urllib.request import urlopen


VALID_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}
ROUTE_HEADERS = {
    "worker": "x-sparseengine-worker",
    "reason": "x-sparseengine-route-reason",
    "method": "x-sparseengine-sparse-method",
    "prefix_matched_tokens": "x-sparseengine-prefix-matched-tokens",
}
DEFAULT_ARTICLE_TOKEN_BUCKETS = (
    (60, 1_000, 8_000),
    (25, 8_001, 16_000),
    (10, 16_001, 32_000),
    (5, 32_001, 64_000),
)
DEFAULT_SUBAGENT_OUTPUT_TOKEN_BUCKETS = (
    (90, 100, 600),
    (10, 800, 1_500),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    base_url: str
    model: str
    output_dir: Path
    num_jobs: int = 1
    job_concurrency: int = 1
    rounds: int = 10
    articles_per_round: int = 20
    min_subagents_per_round: int | None = None
    max_subagents_per_round: int | None = None
    article_token_buckets: tuple[tuple[int, int, int], ...] = (
        DEFAULT_ARTICLE_TOKEN_BUCKETS
    )
    query_tokens: int = 64
    subagent_output_token_buckets: tuple[tuple[int, int, int], ...] = (
        DEFAULT_SUBAGENT_OUTPUT_TOKEN_BUCKETS
    )
    main_overhead_tokens: int = 128
    min_round_summary_tokens: int = 512
    max_round_summary_tokens: int = 1_024
    final_overhead_tokens: int = 128
    min_final_output_tokens: int = 1_000
    max_final_output_tokens: int = 2_000
    subagent_methods: tuple[str, ...] = ("snapkv",)
    main_agent_methods: tuple[str, ...] = ("omnikv", "vanilla")
    subagent_required_tags: tuple[str, ...] = ()
    main_agent_required_tags: tuple[str, ...] = ()
    synthetic_token_id_low: int = 100
    synthetic_token_id_high: int = 255
    request_timeout_s: float = 930.0
    router_timeout_margin_s: float = 30.0
    seed: int = 20260723
    require_router: bool = True
    min_healthy_workers: int = 2


@dataclass(frozen=True)
class RequestSpec:
    sample_id: str
    job_index: int
    phase: str
    round_index: int | None
    request_index: int | None
    prompt_tokens: int
    completion_tokens: int
    prompt_seed: int
    method_preferences: tuple[str, ...]
    required_tags: tuple[str, ...] = ()
    article_tokens: int | None = None
    prompt_token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class BenchmarkFailed(RuntimeError):
    """Raised after failure artifacts have been written."""

    def __init__(self, status: str, message: str):
        if status not in VALID_STATUSES or status == "success":
            raise ValueError(f"Invalid benchmark failure status: {status}")
        super().__init__(message)
        self.status = status


class PreflightParseError(ValueError):
    """Raised when a service preflight response violates its JSON contract."""


class ArtifactWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> ArtifactWriter:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for name in (
            "raw_outputs",
            "parsed_outputs",
            "per_sample_results",
            "round_metrics",
            "job_metrics",
        ):
            self._handles[name] = (self.output_dir / f"{name}.jsonl").open(
                "w",
                encoding="utf-8",
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        for handle in self._handles.values():
            handle.close()

    def write_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        handle = self._handles[name]
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()

    def write_json(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.output_dir / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_text(self, filename: str, content: str) -> None:
        (self.output_dir / filename).write_text(content, encoding="utf-8")


def _canonical_method(method: str | None) -> str:
    value = str(method or "").strip()
    return "vanilla" if not value or value == "vanilla" else value


def _parse_methods(value: str, flag: str) -> tuple[str, ...]:
    methods = tuple(_canonical_method(item) for item in value.split(",") if item.strip())
    if not methods:
        raise ValueError(f"{flag} must contain at least one method.")
    return methods


def _parse_tags(value: str, flag: str) -> tuple[str, ...]:
    tags = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(tags) != len(set(tags)):
        raise ValueError(f"{flag} must not contain duplicate tags.")
    return tags


def _parse_token_buckets(
    value: str,
    flag: str,
) -> tuple[tuple[int, int, int], ...]:
    buckets = []
    for raw_bucket in value.split(","):
        fields = raw_bucket.strip().split(":")
        if len(fields) != 3:
            raise ValueError(
                f"{flag} entries must use WEIGHT:MIN:MAX, got {raw_bucket!r}."
            )
        try:
            buckets.append(tuple(int(field) for field in fields))
        except ValueError as exc:
            raise ValueError(
                f"{flag} entries must contain integers, got {raw_bucket!r}."
            ) from exc
    parsed = tuple(buckets)
    _validate_token_buckets(flag, parsed)
    return parsed


def _validate_token_buckets(
    name: str,
    buckets: tuple[tuple[int, int, int], ...],
) -> None:
    if not buckets:
        raise ValueError(f"{name} must contain at least one token bucket.")
    for index, bucket in enumerate(buckets):
        if len(bucket) != 3:
            raise ValueError(
                f"{name}[{index}] must contain weight, minimum, and maximum."
            )
        weight, minimum, maximum = bucket
        if weight <= 0:
            raise ValueError(f"{name}[{index}] weight must be positive.")
        if minimum <= 0 or maximum <= 0:
            raise ValueError(f"{name}[{index}] token bounds must be positive.")
        if minimum > maximum:
            raise ValueError(
                f"{name}[{index}] minimum must not exceed maximum."
            )


def sample_bucketed_tokens(
    rng: random.Random,
    buckets: tuple[tuple[int, int, int], ...],
) -> int:
    selected = rng.randrange(sum(weight for weight, _, _ in buckets))
    cumulative = 0
    for weight, minimum, maximum in buckets:
        cumulative += weight
        if selected < cumulative:
            return rng.randint(minimum, maximum)
    raise AssertionError("Token bucket sampling did not select a bucket.")


def _max_bucket_tokens(
    buckets: tuple[tuple[int, int, int], ...],
) -> int:
    return max(maximum for _, _, maximum in buckets)


def _subagent_count_bounds(config: BenchmarkConfig) -> tuple[int, int]:
    if (
        config.min_subagents_per_round is None
        and config.max_subagents_per_round is None
    ):
        return config.articles_per_round, config.articles_per_round
    if (
        config.min_subagents_per_round is None
        or config.max_subagents_per_round is None
    ):
        raise ValueError(
            "min_subagents_per_round and max_subagents_per_round must "
            "be set together."
        )
    return (
        config.min_subagents_per_round,
        config.max_subagents_per_round,
    )


def _subagent_counts_for_job(
    config: BenchmarkConfig,
    job_index: int,
) -> tuple[int, ...]:
    minimum, maximum = _subagent_count_bounds(config)
    if minimum == maximum:
        return (minimum,) * config.rounds
    count_rng = random.Random(
        f"simulated-deep-research:{config.seed}:{job_index}:subagent-counts"
    )
    return tuple(
        count_rng.randint(minimum, maximum)
        for _ in range(config.rounds)
    )


def validate_config(config: BenchmarkConfig) -> None:
    if not config.base_url.startswith(("http://", "https://")):
        raise ValueError("--base-url must start with http:// or https://.")
    if not config.model:
        raise ValueError("--model must not be empty.")
    for name in (
        "num_jobs",
        "job_concurrency",
        "rounds",
        "articles_per_round",
        "query_tokens",
        "main_overhead_tokens",
        "min_round_summary_tokens",
        "max_round_summary_tokens",
        "final_overhead_tokens",
        "min_final_output_tokens",
        "max_final_output_tokens",
        "min_healthy_workers",
    ):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.job_concurrency > config.num_jobs:
        raise ValueError(
            "job_concurrency must not exceed num_jobs: "
            f"{config.job_concurrency} > {config.num_jobs}."
        )
    min_subagents, max_subagents = _subagent_count_bounds(config)
    if min_subagents <= 0:
        raise ValueError("min_subagents_per_round must be positive.")
    if min_subagents > max_subagents:
        raise ValueError(
            "min_subagents_per_round must not exceed "
            "max_subagents_per_round."
        )
    _validate_token_buckets(
        "article_token_buckets",
        config.article_token_buckets,
    )
    _validate_token_buckets(
        "subagent_output_token_buckets",
        config.subagent_output_token_buckets,
    )
    if config.min_round_summary_tokens > config.max_round_summary_tokens:
        raise ValueError(
            "min_round_summary_tokens must not exceed "
            "max_round_summary_tokens."
        )
    if config.min_final_output_tokens > config.max_final_output_tokens:
        raise ValueError(
            "min_final_output_tokens must not exceed max_final_output_tokens."
        )
    if config.synthetic_token_id_low < 0:
        raise ValueError("synthetic_token_id_low must be non-negative.")
    if config.synthetic_token_id_low > config.synthetic_token_id_high:
        raise ValueError(
            "synthetic_token_id_low must not exceed synthetic_token_id_high."
        )
    if config.request_timeout_s <= 0:
        raise ValueError("request_timeout_s must be positive.")
    if config.router_timeout_margin_s <= 0:
        raise ValueError("router_timeout_margin_s must be positive.")
    subagent_methods = {_canonical_method(item) for item in config.subagent_methods}
    main_agent_methods = {_canonical_method(item) for item in config.main_agent_methods}
    for name, tags in (
        ("subagent_required_tags", config.subagent_required_tags),
        ("main_agent_required_tags", config.main_agent_required_tags),
    ):
        if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError(f"{name} must contain non-empty strings.")
        if len(tags) != len(set(tags)):
            raise ValueError(f"{name} must not contain duplicate tags.")
    overlap = sorted(subagent_methods & main_agent_methods)
    if config.require_router and overlap:
        subagent_tags = set(config.subagent_required_tags)
        main_agent_tags = set(config.main_agent_required_tags)
        uses_role_tags = bool(subagent_tags or main_agent_tags)
        if uses_role_tags and (
            not subagent_tags
            or not main_agent_tags
            or subagent_tags & main_agent_tags
        ):
            raise ValueError(
                "When overlapping subagent and main-agent methods use role "
                "tags, both roles require non-empty, disjoint tags; "
                f"overlap={overlap}."
            )
    if config.require_router and config.min_healthy_workers < 2:
        raise ValueError("Router runs must require at least two healthy workers.")


def required_model_len_by_role(config: BenchmarkConfig) -> dict[str, int]:
    max_article_tokens = _max_bucket_tokens(config.article_token_buckets)
    max_subagent_output_tokens = _max_bucket_tokens(
        config.subagent_output_token_buckets
    )
    _, max_subagents_per_round = _subagent_count_bounds(config)
    max_subagent = (
        config.query_tokens
        + max_article_tokens
        + max_subagent_output_tokens
    )
    max_round_summary = (
        config.main_overhead_tokens
        + (config.rounds - 1) * config.max_round_summary_tokens
        + max_subagents_per_round * max_subagent_output_tokens
        + config.max_round_summary_tokens
    )
    max_final_summary = (
        config.main_overhead_tokens
        + config.rounds * config.max_round_summary_tokens
        + config.final_overhead_tokens
        + config.max_final_output_tokens
    )
    return {
        "subagent": max_subagent,
        "main agent": max(max_round_summary, max_final_summary),
    }


def _roles_require_distinct_workers(config: BenchmarkConfig) -> bool:
    subagent_methods = {
        _canonical_method(method)
        for method in config.subagent_methods
    }
    main_agent_methods = {
        _canonical_method(method)
        for method in config.main_agent_methods
    }
    return bool(
        subagent_methods.isdisjoint(main_agent_methods)
        or config.subagent_required_tags
        or config.main_agent_required_tags
    )


def required_model_len(config: BenchmarkConfig) -> int:
    return max(required_model_len_by_role(config).values())


def _guaranteed_main_agent_reusable_prefix_tokens(
    config: BenchmarkConfig,
) -> int:
    return (
        config.main_overhead_tokens
        + max(0, config.rounds - 1) * config.min_round_summary_tokens
    )


def _validate_code_revision(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise PreflightParseError(f"{label} is missing code_revision metadata.")
    required_fields = {
        "git_commit",
        "git_branch",
        "git_dirty",
        "package_version",
    }
    missing = sorted(required_fields - value.keys())
    if missing:
        raise PreflightParseError(
            f"{label} code_revision is missing fields: {missing}."
        )
    git_commit = value["git_commit"]
    package_version = value["package_version"]
    if git_commit is not None and not isinstance(git_commit, str):
        raise PreflightParseError(
            f"{label} code_revision.git_commit must be a string or null."
        )
    if package_version is not None and not isinstance(package_version, str):
        raise PreflightParseError(
            f"{label} code_revision.package_version must be a string or null."
        )
    if not git_commit and not package_version:
        raise PreflightParseError(
            f"{label} code revision has neither a Git commit nor package version."
        )
    git_dirty = value["git_dirty"]
    if git_dirty is not None and not isinstance(git_dirty, bool):
        raise PreflightParseError(
            f"{label} code_revision.git_dirty must be boolean or null."
        )
    if git_dirty is True:
        raise ValueError(
            f"{label} reports a dirty Git worktree; benchmark provenance "
            "requires committed router and worker code."
        )
    if git_dirty is None:
        if git_commit is not None:
            raise PreflightParseError(
                f"{label} code_revision.git_dirty must be false when "
                "git_commit is present."
            )
        if not package_version:
            raise PreflightParseError(
                f"{label} code_revision.git_dirty may be null only for an "
                "installed package with package_version metadata."
            )


def _health_url(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/health", "", ""))


def _worker_info_url(worker_url: str) -> str:
    parsed = urlsplit(worker_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"{path}/worker/info", "", "")
    )


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _completion_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/completions"


def _request_bytes(
    url: str,
    *,
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> HttpResponse:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return HttpResponse(
                status=int(response.status),
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read(),
            )
    except HTTPError as exc:
        return HttpResponse(
            status=int(exc.code),
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=exc.read(),
        )
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


async def get_json(url: str, timeout_s: float) -> HttpResponse:
    return await asyncio.to_thread(
        _request_bytes,
        url,
        payload=None,
        timeout_s=timeout_s,
    )


async def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
    *,
    executor: Executor | None = None,
) -> HttpResponse:
    request = partial(
        _request_bytes,
        url,
        payload=payload,
        timeout_s=timeout_s,
    )
    if executor is None:
        return await asyncio.to_thread(request)
    return await asyncio.get_running_loop().run_in_executor(executor, request)


def _decode_json(response: HttpResponse, label: str) -> dict[str, Any]:
    if response.status >= 400:
        body = response.body.decode("utf-8", errors="replace")
        raise RuntimeError(f"{label} failed with HTTP {response.status}: {body}")
    try:
        decoded = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightParseError(
            f"{label} returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise PreflightParseError(f"{label} must return a JSON object.")
    return decoded


async def preflight(
    config: BenchmarkConfig,
    *,
    get_fn=get_json,
) -> dict[str, Any]:
    models_response, health_response = await asyncio.gather(
        get_fn(_models_url(config.base_url), config.request_timeout_s),
        get_fn(_health_url(config.base_url), config.request_timeout_s),
    )
    models = _decode_json(models_response, "models preflight")
    health = _decode_json(health_response, "health preflight")
    cards = models.get("data")
    if not isinstance(cards, list):
        raise PreflightParseError(
            "/v1/models response is missing a data list."
        )
    model_card = next(
        (
            card
            for card in cards
            if isinstance(card, dict) and str(card.get("id")) == config.model
        ),
        None,
    )
    if model_card is None:
        available = [
            str(card.get("id"))
            for card in cards
            if isinstance(card, dict) and card.get("id") is not None
        ]
        raise ValueError(
            f"Model {config.model!r} is not served; available models={available}."
        )
    max_model_len_value = model_card.get("max_model_len")
    required_len = required_model_len(config)
    required_lens_by_role = required_model_len_by_role(config)
    if not config.require_router:
        if (
            isinstance(max_model_len_value, bool)
            or not isinstance(max_model_len_value, int)
        ):
            raise PreflightParseError(
                "The selected model card max_model_len must be an integer."
            )
        if max_model_len_value <= 0:
            raise ValueError(
                "The selected model does not advertise max_model_len; the "
                "synthetic workload cannot be validated safely."
            )
        if required_len > max_model_len_value:
            raise ValueError(
                f"Workload requires max_model_len>={required_len}, but the "
                f"server advertises {max_model_len_value}."
            )
    healthy_workers = health.get("healthy_workers")
    if config.require_router:
        if model_card.get("owned_by") != "sparseengine-router":
            raise ValueError(
                "--require-router was set, but /v1/models did not identify the "
                "Sparse-Engine smart router."
            )
        if not isinstance(healthy_workers, list):
            raise PreflightParseError(
                "Router health response is missing healthy_workers."
            )
        if not all(
            isinstance(worker_url, str) and worker_url
            for worker_url in healthy_workers
        ):
            raise PreflightParseError(
                "Router healthy_workers must contain non-empty worker URLs."
            )
        router_policy = health.get("router_policy")
        if not isinstance(router_policy, dict):
            raise PreflightParseError(
                "Router health response is missing router_policy."
            )
        required_policy_fields = {
            "request_timeout_s",
            "control_timeout_s",
            "overload_load_factor",
            "load_abs_threshold",
            "profiles",
            "code_revision",
        }
        missing_policy_fields = sorted(
            required_policy_fields - router_policy.keys()
        )
        if missing_policy_fields:
            raise PreflightParseError(
                "Router policy is missing fields: "
                f"{missing_policy_fields}."
            )
        router_timeout = router_policy["request_timeout_s"]
        if (
            isinstance(router_timeout, bool)
            or not isinstance(router_timeout, (int, float))
            or router_timeout <= 0
        ):
            raise PreflightParseError(
                "Router policy request_timeout_s must be positive."
            )
        if (
            config.request_timeout_s
            < router_timeout + config.router_timeout_margin_s
        ):
            raise ValueError(
                "Benchmark request timeout must exceed the router upstream "
                f"timeout by at least {config.router_timeout_margin_s}s: "
                f"router={router_timeout}s "
                f"benchmark={config.request_timeout_s}s."
            )
        control_timeout = router_policy["control_timeout_s"]
        if (
            isinstance(control_timeout, bool)
            or not isinstance(control_timeout, (int, float))
            or control_timeout <= 0
        ):
            raise PreflightParseError(
                "Router policy control_timeout_s must be positive."
            )
        overload_load_factor = router_policy["overload_load_factor"]
        if (
            isinstance(overload_load_factor, bool)
            or not isinstance(overload_load_factor, (int, float))
            or overload_load_factor <= 0
        ):
            raise PreflightParseError(
                "Router policy overload_load_factor must be positive."
            )
        load_abs_threshold = router_policy["load_abs_threshold"]
        if (
            isinstance(load_abs_threshold, bool)
            or not isinstance(load_abs_threshold, int)
            or load_abs_threshold < 0
        ):
            raise PreflightParseError(
                "Router policy load_abs_threshold must be a non-negative "
                "integer."
            )
        if not isinstance(router_policy["profiles"], dict):
            raise PreflightParseError(
                "Router policy profiles must be an object."
            )
        _validate_code_revision(
            router_policy["code_revision"],
            "Router policy",
        )
        worker_responses = await asyncio.gather(
            *[
                get_fn(
                    _worker_info_url(worker_url),
                    config.request_timeout_s,
                )
                for worker_url in healthy_workers
            ]
        )
        healthy_worker_info = [
            {
                "url": worker_url,
                "info": _decode_json(
                    response,
                    f"worker info preflight for {worker_url}",
                ),
            }
            for worker_url, response in zip(
                healthy_workers,
                worker_responses,
            )
        ]
        workers = [
            worker
            for worker in healthy_worker_info
            if str(worker["info"].get("served_model_name")) == config.model
        ]
    else:
        worker_url = config.base_url
        worker_response = await get_fn(
            _worker_info_url(worker_url),
            config.request_timeout_s,
        )
        workers = [
            {
                "url": worker_url,
                "info": _decode_json(
                    worker_response,
                    f"worker info preflight for {worker_url}",
                ),
            }
        ]
    required_worker_methods = {
        _canonical_method(method)
        for method in (
            config.subagent_methods
            + config.main_agent_methods
        )
    }
    benchmark_workers = []
    for worker in workers:
        info = worker["info"]
        if not isinstance(info.get("sparse_method"), str):
            raise PreflightParseError(
                "Worker info response is missing a string sparse_method: "
                f"worker={worker['url']}."
            )
        if (
            config.require_router
            and _canonical_method(info["sparse_method"])
            not in required_worker_methods
        ):
            continue
        benchmark_workers.append(worker)
        if not isinstance(info.get("benchmark_config"), dict):
            raise PreflightParseError(
                "Worker info response is missing benchmark_config: "
                f"worker={worker['url']}."
            )
        tags = info.get("tags", [])
        if (
            not isinstance(tags, list)
            or not all(isinstance(tag, str) and tag for tag in tags)
        ):
            raise PreflightParseError(
                "Worker info response must contain a list of non-empty string "
                f"tags: worker={worker['url']}."
            )
        info["tags"] = tags
        max_worker_len = info.get("max_model_len")
        if (
            isinstance(max_worker_len, bool)
            or not isinstance(max_worker_len, int)
            or max_worker_len <= 0
        ):
            raise PreflightParseError(
                "Worker info response is missing a positive integer "
                f"max_model_len: worker={worker['url']}."
            )
        vocab_size = info.get("vocab_size")
        if (
            isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int)
            or vocab_size <= 0
        ):
            raise PreflightParseError(
                "Worker info response is missing a positive integer "
                f"vocab_size: worker={worker['url']}."
            )
        _validate_code_revision(
            info.get("code_revision"),
            f"Worker {worker['url']}",
        )
        if str(info.get("served_model_name")) != config.model:
            raise ValueError(
                "Worker serves a different model: "
                f"worker={worker['url']} "
                f"served_model_name={info.get('served_model_name')!r} "
                f"expected={config.model!r}."
            )
    if config.require_router:
        if len(benchmark_workers) < config.min_healthy_workers:
            raise ValueError(
                f"Need at least {config.min_healthy_workers} healthy "
                f"benchmark-eligible workers for model {config.model!r}, "
                f"but the router reports {len(benchmark_workers)}."
            )
        advertised_methods = sorted(
            {
                _canonical_method(worker["info"]["sparse_method"])
                for worker in workers
            }
        )
        eligible_worker_urls_by_role: dict[str, set[str]] = {}
        for role, preferences, required_tags in (
            (
                "subagent",
                config.subagent_methods,
                config.subagent_required_tags,
            ),
            (
                "main agent",
                config.main_agent_methods,
                config.main_agent_required_tags,
            ),
        ):
            preferred_methods = {
                _canonical_method(method)
                for method in preferences
            }
            required_tag_set = set(required_tags)
            eligible_workers = [
                worker
                for worker in benchmark_workers
                if _canonical_method(worker["info"]["sparse_method"])
                in preferred_methods
                and required_tag_set.issubset(
                    set(worker["info"]["tags"])
                )
            ]
            if not eligible_workers:
                raise ValueError(
                    f"No healthy worker for model {config.model!r} matches "
                    f"the configured {role} methods {list(preferences)} and "
                    f"required tags {list(required_tags)}; "
                    f"advertised methods={advertised_methods}."
                )
            eligible_worker_urls_by_role[role] = {
                str(worker["url"])
                for worker in eligible_workers
            }
            required_role_len = required_lens_by_role[role]
            too_short = [
                {
                    "url": worker["url"],
                    "method": _canonical_method(
                        worker["info"]["sparse_method"]
                    ),
                    "max_model_len": worker["info"]["max_model_len"],
                }
                for worker in eligible_workers
                if worker["info"]["max_model_len"] < required_role_len
            ]
            if too_short:
                raise ValueError(
                    f"{role} workers cannot serve the required model length "
                    f"{required_role_len}: {too_short}."
                )
            too_small_vocab = [
                {
                    "url": worker["url"],
                    "method": _canonical_method(
                        worker["info"]["sparse_method"]
                    ),
                    "vocab_size": worker["info"]["vocab_size"],
                }
                for worker in eligible_workers
                if config.synthetic_token_id_high
                >= worker["info"]["vocab_size"]
            ]
            if too_small_vocab:
                raise ValueError(
                    "Synthetic token range exceeds an eligible worker "
                    f"vocabulary: high={config.synthetic_token_id_high} "
                    f"workers={too_small_vocab}."
                )
            if role == "main agent":
                without_prefix_cache = [
                    worker["url"]
                    for worker in eligible_workers
                    if worker["info"].get("prefix_cache_enabled") is not True
                ]
                if without_prefix_cache:
                    raise ValueError(
                        "All eligible main-agent workers must enable prefix "
                        f"caching: workers={without_prefix_cache}."
                    )
                invalid_block_sizes = [
                    {
                        "url": worker["url"],
                        "prefix_cache_block_size": worker["info"].get(
                            "prefix_cache_block_size"
                        ),
                    }
                    for worker in eligible_workers
                    if (
                        isinstance(
                            worker["info"].get("prefix_cache_block_size"),
                            bool,
                        )
                        or not isinstance(
                            worker["info"].get("prefix_cache_block_size"),
                            int,
                        )
                        or worker["info"]["prefix_cache_block_size"] <= 0
                    )
                ]
                if invalid_block_sizes:
                    raise PreflightParseError(
                        "All eligible main-agent workers must report a "
                        "positive integer prefix_cache_block_size: "
                        f"workers={invalid_block_sizes}."
                    )
                guaranteed_reusable_prefix = (
                    _guaranteed_main_agent_reusable_prefix_tokens(config)
                )
                oversized_block_sizes = [
                    {
                        "url": worker["url"],
                        "block_size": worker["info"][
                            "prefix_cache_block_size"
                        ],
                    }
                    for worker in eligible_workers
                    if worker["info"]["prefix_cache_block_size"]
                    > guaranteed_reusable_prefix
                ]
                if oversized_block_sizes:
                    details = ", ".join(
                        f"worker={worker['url']} "
                        f"block_size={worker['block_size']}"
                        for worker in oversized_block_sizes
                    )
                    raise ValueError(
                        "Eligible main-agent prefix-cache blocks cannot be "
                        "reused by the configured workload: "
                        f"{details}; guaranteed_reusable_prefix_tokens="
                        f"{guaranteed_reusable_prefix}. Increase --rounds or "
                        "--min-round-summary-tokens, or decrease the worker "
                        "prefix_cache_block_size."
                    )
        overlapping_methods = {
            _canonical_method(method)
            for method in config.subagent_methods
        } & {
            _canonical_method(method)
            for method in config.main_agent_methods
        }
        overlapping_role_workers = sorted(
            eligible_worker_urls_by_role["subagent"]
            & eligible_worker_urls_by_role["main agent"]
        )
        uses_role_tags = bool(
            config.subagent_required_tags
            or config.main_agent_required_tags
        )
        if (
            overlapping_methods
            and uses_role_tags
            and overlapping_role_workers
        ):
            raise ValueError(
                "Role tags did not isolate overlapping subagent and main-agent "
                "methods onto separate workers: "
                f"workers={overlapping_role_workers}."
            )
    else:
        worker = workers[0]
        if config.synthetic_token_id_high >= worker["info"]["vocab_size"]:
            raise ValueError(
                "Synthetic token range exceeds the direct worker vocabulary: "
                f"high={config.synthetic_token_id_high} "
                f"vocab_size={worker['info']['vocab_size']}."
            )
    return {
        "health": health,
        "model_card": model_card,
        "required_model_len": required_len,
        "required_model_len_by_role": required_lens_by_role,
        "workers": workers,
        "worker_prefix_cache_block_sizes": {
            str(worker["url"]): (
                int(worker["info"]["prefix_cache_block_size"])
                if (
                    not isinstance(
                        worker["info"].get("prefix_cache_block_size"),
                        bool,
                    )
                    and isinstance(
                        worker["info"].get("prefix_cache_block_size"),
                        int,
                    )
                    and worker["info"]["prefix_cache_block_size"] > 0
                )
                else None
            )
            for worker in workers
        },
    }


def synthetic_token_ids(
    length: int,
    *,
    seed: int,
    low: int,
    high: int,
) -> list[int]:
    if length <= 0:
        raise ValueError("Synthetic prompt length must be positive.")
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(length)]


def build_payload(spec: RequestSpec, config: BenchmarkConfig) -> dict[str, Any]:
    if spec.prompt_token_ids is None:
        prompt = synthetic_token_ids(
            spec.prompt_tokens,
            seed=spec.prompt_seed,
            low=config.synthetic_token_id_low,
            high=config.synthetic_token_id_high,
        )
    else:
        prompt = list(spec.prompt_token_ids)
        if len(prompt) != spec.prompt_tokens:
            raise ValueError(
                f"{spec.sample_id} prompt length mismatch: "
                f"declared={spec.prompt_tokens}, actual={len(prompt)}."
            )
    payload = {
        "model": config.model,
        "prompt": prompt,
        "max_tokens": spec.completion_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "ignore_eos": True,
        "stream": False,
    }
    if config.require_router:
        payload["sengine_method_preference"] = ",".join(spec.method_preferences)
        if spec.required_tags:
            payload["sengine_required_tags"] = list(spec.required_tags)
    return payload


def _route_values(headers: dict[str, str]) -> dict[str, str | None]:
    return {
        name: headers.get(header)
        for name, header in ROUTE_HEADERS.items()
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _common_prefix_tokens(
    previous: tuple[int, ...] | None,
    current: tuple[int, ...],
) -> int:
    if previous is None:
        return 0
    common = 0
    for previous_token, current_token in zip(previous, current):
        if previous_token != current_token:
            break
        common += 1
    return common


def _same_worker_prefix_expectation(
    current_prompt: tuple[int, ...],
    selected_worker: str,
    main_prompt_history_by_worker: dict[
        str,
        list[tuple[int, ...]],
    ],
    block_size: int,
) -> tuple[int, int, int]:
    if not current_prompt:
        raise ValueError("Current main-agent prompt must not be empty.")
    if block_size <= 0:
        raise ValueError("Prefix-cache block size must be positive.")
    prior_prompts = main_prompt_history_by_worker.get(
        selected_worker,
        [],
    )
    current_cacheable_tokens = (
        (len(current_prompt) - 1)
        // block_size
        * block_size
    )
    max_raw_common_prefix = 0
    max_block_aligned_prefix = 0
    for prior_prompt in prior_prompts:
        raw_common_prefix = _common_prefix_tokens(
            prior_prompt,
            current_prompt,
        )
        prior_materialized_tokens = (
            len(prior_prompt)
            // block_size
            * block_size
        )
        block_aligned_prefix = min(
            raw_common_prefix // block_size * block_size,
            current_cacheable_tokens,
            prior_materialized_tokens,
        )
        max_raw_common_prefix = max(
            max_raw_common_prefix,
            raw_common_prefix,
        )
        max_block_aligned_prefix = max(
            max_block_aligned_prefix,
            block_aligned_prefix,
        )
    return (
        max_raw_common_prefix,
        max_block_aligned_prefix,
        len(prior_prompts),
    )


async def run_request(
    spec: RequestSpec,
    config: BenchmarkConfig,
    writer: ArtifactWriter,
    *,
    worker_prefix_cache_block_sizes: dict[str, int | None],
    main_prompt_history_by_worker: dict[
        str,
        list[tuple[int, ...]],
    ],
    timeline_origin: float,
    post_fn=post_json,
) -> dict[str, Any]:
    payload = build_payload(spec, config)
    started = time.perf_counter()
    started_offset_s = started - timeline_origin
    http_status: int | None = None
    response_headers: dict[str, str] = {}
    response_body: Any = None
    status = "success"
    error: str | None = None
    parsed_text: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    route = _route_values(response_headers)
    actual_prefix_matched_tokens: int | None = None
    prefix_cache_block_size: int | None = None
    block_aligned_expected_reusable_prefix_tokens: int | None = None
    expected_reusable_prefix_tokens = 0
    same_worker_prior_prompt_count = 0
    selected_worker: str | None = None
    try:
        response = await post_fn(
            _completion_url(config.base_url),
            payload,
            config.request_timeout_s,
        )
        http_status = response.status
        response_headers = response.headers
        response_text = response.body.decode("utf-8", errors="replace")
        if response.status >= 400:
            try:
                response_body = json.loads(response_text)
            except json.JSONDecodeError:
                response_body = response_text
            status = "model_failed"
            error = f"HTTP {response.status}: {response_body}"
        else:
            try:
                response_body = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                status = "parse_failed"
                error = f"Invalid JSON response: {exc}"
                response_body = response_text
        if status == "success":
            if not isinstance(response_body, dict):
                status = "parse_failed"
                error = "Completion response must be a JSON object."
            else:
                choices = response_body.get("choices")
                usage_value = response_body.get("usage")
                if (
                    not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0], dict)
                    or not isinstance(usage_value, dict)
                ):
                    status = "parse_failed"
                    error = "Completion response is missing one choice or usage."
                else:
                    choice = choices[0]
                    text_value = choice.get("text")
                    finish_reason_value = choice.get("finish_reason")
                    prompt_tokens_value = usage_value.get("prompt_tokens")
                    completion_tokens_value = usage_value.get(
                        "completion_tokens"
                    )
                    if not isinstance(text_value, str):
                        status = "parse_failed"
                        error = "Completion choice text must be a string."
                    elif not isinstance(finish_reason_value, str):
                        status = "parse_failed"
                        error = "Completion choice finish_reason must be a string."
                    elif (
                        isinstance(prompt_tokens_value, bool)
                        or not isinstance(prompt_tokens_value, int)
                        or prompt_tokens_value < 0
                    ):
                        status = "parse_failed"
                        error = "usage.prompt_tokens must be a non-negative integer."
                    elif (
                        isinstance(completion_tokens_value, bool)
                        or not isinstance(completion_tokens_value, int)
                        or completion_tokens_value < 0
                    ):
                        status = "parse_failed"
                        error = (
                            "usage.completion_tokens must be a non-negative integer."
                        )
                    else:
                        parsed_text = text_value
                        finish_reason = finish_reason_value
                        usage = dict(usage_value)
        route = _route_values(response_headers)
        if status == "success" and config.require_router:
            missing_headers = [
                ROUTE_HEADERS[name]
                for name, value in route.items()
                if value is None
            ]
            if missing_headers:
                status = "metric_failed"
                error = f"Router response is missing route headers: {missing_headers}."
        if config.require_router and route.get("prefix_matched_tokens") is not None:
            try:
                actual_prefix_matched_tokens = int(
                    str(route["prefix_matched_tokens"])
                )
                if actual_prefix_matched_tokens < 0:
                    raise ValueError("value must be non-negative")
            except ValueError as exc:
                actual_prefix_matched_tokens = None
                if status == "success":
                    status = "metric_failed"
                    error = (
                        "Router response has invalid "
                        f"{ROUTE_HEADERS['prefix_matched_tokens']} header "
                        f"{route['prefix_matched_tokens']!r}: {exc}."
                    )
        if status == "success":
            expected_methods = {
                _canonical_method(item) for item in spec.method_preferences
            }
            actual_method = _canonical_method(route.get("method"))
            if config.require_router and actual_method not in expected_methods:
                status = "metric_failed"
                error = (
                    f"{spec.phase} request routed to method {actual_method!r}; "
                    f"expected one of {sorted(expected_methods)}."
                )
        if status == "success" and config.require_router:
            selected_worker = route.get("worker")
            if selected_worker not in worker_prefix_cache_block_sizes:
                status = "metric_failed"
                error = (
                    "Router selected a worker absent from preflight metadata: "
                    f"worker={selected_worker!r}."
                )
            else:
                prefix_cache_block_size = (
                    worker_prefix_cache_block_sizes[selected_worker]
                )
                if spec.phase == "subagent":
                    block_aligned_expected_reusable_prefix_tokens = 0
                elif spec.prompt_token_ids is None:
                    status = "metric_failed"
                    error = (
                        "Main-agent prefix auditing requires explicit prompt "
                        "token ids."
                    )
                elif prefix_cache_block_size is None:
                    status = "metric_failed"
                    error = (
                        "Selected worker did not report a usable "
                        "prefix_cache_block_size for expected prefix reuse: "
                        f"worker={selected_worker!r}."
                    )
                else:
                    (
                        expected_reusable_prefix_tokens,
                        block_aligned_expected_reusable_prefix_tokens,
                        same_worker_prior_prompt_count,
                    ) = _same_worker_prefix_expectation(
                        spec.prompt_token_ids,
                        selected_worker,
                        main_prompt_history_by_worker,
                        prefix_cache_block_size,
                    )
        if (
            status == "success"
            and config.require_router
            and actual_prefix_matched_tokens is not None
            and block_aligned_expected_reusable_prefix_tokens is not None
        ):
            if actual_prefix_matched_tokens > spec.prompt_tokens:
                status = "metric_failed"
                error = (
                    "Router reported more matched prefix tokens than the "
                    f"request prompt: matched={actual_prefix_matched_tokens}, "
                    f"prompt={spec.prompt_tokens}."
                )
            elif (
                actual_prefix_matched_tokens
                > block_aligned_expected_reusable_prefix_tokens
            ):
                status = "metric_failed"
                error = (
                    "Selected worker reported prefix reuse beyond this run's "
                    "cacheable expectation, indicating stale-cache "
                    "contamination: "
                    f"raw_expected={expected_reusable_prefix_tokens}, "
                    "block_aligned_expected="
                    f"{block_aligned_expected_reusable_prefix_tokens}, "
                    f"actual={actual_prefix_matched_tokens}, "
                    f"block_size={prefix_cache_block_size}."
                )
        if status == "success":
            actual_prompt = usage["prompt_tokens"]
            actual_completion = usage["completion_tokens"]
            if actual_prompt != spec.prompt_tokens:
                status = "metric_failed"
                error = (
                    f"Prompt token mismatch: target={spec.prompt_tokens}, "
                    f"actual={actual_prompt}."
                )
            elif actual_completion != spec.completion_tokens:
                status = "metric_failed"
                error = (
                    f"Completion token mismatch: target={spec.completion_tokens}, "
                    f"actual={actual_completion}."
                )
    except Exception as exc:
        status = "model_failed"
        error = f"{type(exc).__name__}: {exc}"
        route = _route_values(response_headers)

    finished = time.perf_counter()
    elapsed_s = finished - started
    finished_offset_s = finished - timeline_origin
    if status not in VALID_STATUSES:
        raise AssertionError(f"Unexpected request status: {status}")
    if (
        status == "success"
        and config.require_router
        and spec.phase != "subagent"
    ):
        if selected_worker is None or spec.prompt_token_ids is None:
            raise AssertionError(
                "Successful main-agent request is missing audited routing "
                "or prompt metadata."
            )
        main_prompt_history_by_worker.setdefault(
            selected_worker,
            [],
        ).append(spec.prompt_token_ids)
    raw_record = {
        "sample_id": spec.sample_id,
        "job_index": spec.job_index,
        "status": status,
        "error": error,
        "request": {
            "url": _completion_url(config.base_url),
            "timeout_s": config.request_timeout_s,
            "prompt_seed": spec.prompt_seed,
            "payload": payload,
        },
        "http_status": http_status,
        "response_headers": response_headers,
        "route_observation": {
            "worker": route.get("worker"),
            "reason": route.get("reason"),
            "method": route.get("method"),
            "prefix_cache_block_size": prefix_cache_block_size,
            "raw_expected_reusable_prefix_tokens": (
                expected_reusable_prefix_tokens
            ),
            "same_worker_prior_prompt_count": (
                same_worker_prior_prompt_count
            ),
            "block_aligned_expected_reusable_prefix_tokens": (
                block_aligned_expected_reusable_prefix_tokens
            ),
            "actual_prefix_matched_tokens": actual_prefix_matched_tokens,
        },
        "response": response_body,
        "started_offset_s": started_offset_s,
        "finished_offset_s": finished_offset_s,
    }
    parsed_record = {
        "sample_id": spec.sample_id,
        "job_index": spec.job_index,
        "status": status,
        "error": error,
        "text": parsed_text,
        "finish_reason": finish_reason,
        "usage": usage,
    }
    result = {
        "sample_id": spec.sample_id,
        "job_index": spec.job_index,
        "phase": spec.phase,
        "round_index": spec.round_index,
        "request_index": spec.request_index,
        "article_tokens": spec.article_tokens,
        "status": status,
        "error": error,
        "target_prompt_tokens": spec.prompt_tokens,
        "actual_prompt_tokens": usage.get("prompt_tokens"),
        "target_completion_tokens": spec.completion_tokens,
        "actual_completion_tokens": usage.get("completion_tokens"),
        "latency_s": elapsed_s,
        "started_offset_s": started_offset_s,
        "finished_offset_s": finished_offset_s,
        "route_worker": route.get("worker"),
        "route_reason": route.get("reason"),
        "route_method": (
            _canonical_method(route["method"])
            if route.get("method") is not None
            else None
        ),
        "method_preferences": list(spec.method_preferences),
        "required_tags": list(spec.required_tags),
        "expected_reusable_prefix_tokens": (
            expected_reusable_prefix_tokens
        ),
        "same_worker_prior_prompt_count": same_worker_prior_prompt_count,
        "prefix_cache_block_size": prefix_cache_block_size,
        "block_aligned_expected_reusable_prefix_tokens": (
            block_aligned_expected_reusable_prefix_tokens
        ),
        "actual_prefix_matched_tokens": actual_prefix_matched_tokens,
    }
    writer.write_jsonl("raw_outputs", raw_record)
    writer.write_jsonl("parsed_outputs", parsed_record)
    writer.write_jsonl("per_sample_results", result)
    return result


def _failure_status(records: list[dict[str, Any]]) -> str | None:
    return next(
        (
            status
            for status in (
                "model_failed",
                "parse_failed",
                "metric_failed",
                "invalid_input",
                "skipped_by_policy",
            )
            if any(record["status"] == status for record in records)
        ),
        None,
    )


def _require_success(records: list[dict[str, Any]], label: str) -> None:
    failed = [record for record in records if record["status"] != "success"]
    if failed:
        failures = [
            {
                "sample_id": record["sample_id"],
                "status": record["status"],
                "error": record["error"],
            }
            for record in failed
        ]
        failure_status = _failure_status(failed)
        if failure_status is None:
            raise AssertionError(f"Unknown failure statuses: {failures}")
        raise BenchmarkFailed(
            failure_status,
            f"{label} failed: {failures}",
        )


def _git_command(*args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_value(*args: str) -> str | None:
    result = _git_command(*args)
    if result is None:
        return None
    value = result.stdout.strip()
    return value or None


def _git_dirty() -> bool | None:
    result = _git_command("status", "--porcelain")
    if result is None:
        return None
    return bool(result.stdout.strip())


def _capture_client_code_state() -> tuple[dict[str, Any], str | None]:
    git_commit = _git_value("rev-parse", "HEAD")
    git_branch = _git_value("branch", "--show-current")
    status = _git_command("status", "--porcelain", "--untracked-files=all")
    if status is None:
        raise ValueError(
            "Cannot inspect the client source-tree Git status; benchmark "
            "provenance requires a readable Git worktree."
        )
    if not git_commit:
        raise ValueError(
            "Cannot resolve the client source-tree Git commit; benchmark "
            "provenance requires an exact source revision."
        )
    status_lines = [
        line
        for line in status.stdout.splitlines()
        if line
    ]
    if not status_lines:
        return (
            {
                "git_commit": git_commit,
                "git_branch": git_branch,
                "git_dirty": False,
                "worktree_patch": None,
            },
            None,
        )
    untracked = [
        line[3:]
        for line in status_lines
        if line.startswith("?? ")
    ]
    if untracked:
        raise ValueError(
            "Client worktree has untracked files that cannot be reproduced "
            f"by a Git patch; commit or remove them first: {untracked}."
        )
    diff = _git_command("diff", "--binary", "HEAD", "--")
    if diff is None or not diff.stdout:
        raise ValueError(
            "Client worktree is dirty but a reproducible Git patch could not "
            "be captured."
        )
    patch_sha256 = hashlib.sha256(diff.stdout.encode("utf-8")).hexdigest()
    return (
        {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "git_dirty": True,
            "worktree_patch": {
                "path": "client_worktree.patch",
                "sha256": patch_sha256,
                "format": "git diff --binary HEAD",
            },
        },
        diff.stdout,
    )


def _run_info(
    config: BenchmarkConfig,
    preflight_info: dict[str, Any] | None,
    *,
    status: str,
    error: str | None = None,
    client_code_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = asdict(config)
    resolved["output_dir"] = str(config.output_dir)
    resolved["subagent_methods"] = list(config.subagent_methods)
    resolved["main_agent_methods"] = list(config.main_agent_methods)
    resolved["subagent_required_tags"] = list(
        config.subagent_required_tags
    )
    resolved["main_agent_required_tags"] = list(
        config.main_agent_required_tags
    )
    if client_code_state is None:
        client_code_state = {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": _git_dirty(),
            "worktree_patch": None,
        }
    return {
        "benchmark": "simulated_deep_research",
        "status": status,
        "error": error,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": sys.argv,
        "config": resolved,
        "preflight": preflight_info,
        "git_commit": client_code_state["git_commit"],
        "git_branch": client_code_state["git_branch"],
        "git_dirty": client_code_state["git_dirty"],
        "client_worktree_patch": client_code_state["worktree_patch"],
        "environment": {
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")
            if key in os.environ
        },
    }


def _prefix_cache_metrics(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    observed_records = [
        record
        for record in records
        if record.get("actual_prefix_matched_tokens") is not None
    ]
    expected_hit_records = [
        record
        for record in records
        if int(
            record.get(
                "block_aligned_expected_reusable_prefix_tokens"
            )
            or 0
        )
        > 0
    ]
    return {
        "expected_reusable_prefix_tokens": sum(
            int(record.get("expected_reusable_prefix_tokens") or 0)
            for record in records
        ),
        "block_aligned_expected_reusable_prefix_tokens": sum(
            int(
                record.get(
                    "block_aligned_expected_reusable_prefix_tokens"
                )
                or 0
            )
            for record in records
        ),
        "actual_prefix_matched_tokens": sum(
            int(record["actual_prefix_matched_tokens"])
            for record in observed_records
        ),
        "observed_requests": len(observed_records),
        "expected_hit_requests": len(expected_hit_records),
        "actual_hit_requests": sum(
            int(record["actual_prefix_matched_tokens"]) > 0
            for record in observed_records
        ),
        "unexpected_zero_hit_requests": sum(
            int(
                record.get(
                    "block_aligned_expected_reusable_prefix_tokens"
                )
                or 0
            )
            > 0
            and record.get("actual_prefix_matched_tokens") == 0
            for record in records
        ),
        "partial_hit_requests": sum(
            0
            < int(record.get("actual_prefix_matched_tokens") or 0)
            < int(
                record.get(
                    "block_aligned_expected_reusable_prefix_tokens"
                )
                or 0
            )
            for record in records
        ),
        "unexpected_excess_hit_requests": sum(
            record.get("actual_prefix_matched_tokens") is not None
            and record.get(
                "block_aligned_expected_reusable_prefix_tokens"
            )
            is not None
            and (
                int(record["actual_prefix_matched_tokens"])
                > int(
                    record.get(
                        "block_aligned_expected_reusable_prefix_tokens"
                    )
                    or 0
                )
            )
            for record in records
        ),
    }


def _interval_union_duration(
    records: list[dict[str, Any]],
) -> float:
    intervals = sorted(
        (
            float(record["started_offset_s"]),
            float(record["finished_offset_s"]),
        )
        for record in records
    )
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _aggregate_metrics(
    config: BenchmarkConfig,
    records: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    elapsed_s: float,
    run_elapsed_s: float,
    preflight_elapsed_s: float,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    success_records = [record for record in records if record["status"] == "success"]
    status_counts = Counter(record["status"] for record in records)
    phase_counts = Counter(record["phase"] for record in records)
    route_worker_counts = Counter(
        str(record["route_worker"])
        for record in success_records
        if record.get("route_worker")
    )
    route_method_counts = Counter(
        _canonical_method(record["route_method"])
        for record in success_records
        if record.get("route_method") is not None
    )
    route_reason_counts = Counter(
        str(record["route_reason"])
        for record in success_records
        if record.get("route_reason")
    )
    latencies = [float(record["latency_s"]) for record in success_records]
    subagent_latencies = [
        float(record["latency_s"])
        for record in success_records
        if record["phase"] == "subagent"
    ]
    total_prompt_tokens = sum(
        int(record.get("actual_prompt_tokens") or 0)
        for record in success_records
    )
    total_completion_tokens = sum(
        int(record.get("actual_completion_tokens") or 0)
        for record in success_records
    )
    expected_requests_by_job = {
        str(job["job_index"]): int(job["expected_requests"])
        for job in jobs
    }
    per_job_expected_values = list(expected_requests_by_job.values())
    expected_requests_per_job = (
        per_job_expected_values[0]
        if per_job_expected_values
        and len(set(per_job_expected_values)) == 1
        else None
    )
    expected_requests = sum(per_job_expected_values)
    successful_jobs = [
        job for job in jobs if job["status"] == "success"
    ]
    job_status_counts = Counter(job["status"] for job in jobs)
    all_success = (
        status == "success"
        and len(success_records) == expected_requests
        and len(successful_jobs) == config.num_jobs
    )
    phase_metrics = {}
    for phase in sorted(phase_counts):
        phase_all_records = [
            record
            for record in records
            if record["phase"] == phase
        ]
        phase_records = [
            record
            for record in success_records
            if record["phase"] == phase
        ]
        phase_prompt_tokens = sum(
            int(record.get("actual_prompt_tokens") or 0)
            for record in phase_records
        )
        phase_completion_tokens = sum(
            int(record.get("actual_completion_tokens") or 0)
            for record in phase_records
        )
        phase_expected_reusable_prefix_tokens = sum(
            int(record.get("expected_reusable_prefix_tokens") or 0)
            for record in phase_records
        )
        phase_block_aligned_expected_reusable_prefix_tokens = sum(
            int(
                record.get(
                    "block_aligned_expected_reusable_prefix_tokens"
                )
                or 0
            )
            for record in phase_records
        )
        phase_latencies = [
            float(record["latency_s"])
            for record in phase_records
        ]
        phase_elapsed_s = _interval_union_duration(phase_records)
        phase_workers = Counter(
            str(record["route_worker"])
            for record in phase_records
            if record.get("route_worker")
        )
        phase_methods = Counter(
            _canonical_method(record["route_method"])
            for record in phase_records
            if record.get("route_method") is not None
        )
        phase_metrics[phase] = {
            "successful_requests": len(phase_records),
            "elapsed_s": phase_elapsed_s,
            "requests_per_s": (
                len(phase_records) / phase_elapsed_s
                if phase_elapsed_s > 0
                else 0.0
            ),
            "prompt_tokens": phase_prompt_tokens,
            "completion_tokens": phase_completion_tokens,
            "expected_reusable_prefix_tokens": (
                phase_expected_reusable_prefix_tokens
            ),
            "block_aligned_expected_reusable_prefix_tokens": (
                phase_block_aligned_expected_reusable_prefix_tokens
            ),
            "prefix_cache": _prefix_cache_metrics(phase_all_records),
            "total_tokens_per_s": (
                (phase_prompt_tokens + phase_completion_tokens)
                / phase_elapsed_s
                if phase_elapsed_s > 0
                else 0.0
            ),
            "latency_s": {
                "p50": _percentile(phase_latencies, 0.50),
                "p95": _percentile(phase_latencies, 0.95),
                "max": max(phase_latencies) if phase_latencies else None,
            },
            "route_worker_counts": dict(sorted(phase_workers.items())),
            "route_method_counts": dict(sorted(phase_methods.items())),
        }
    return {
        "benchmark": "simulated_deep_research",
        "status": status,
        "error": error,
        "elapsed_s": elapsed_s,
        "run_elapsed_s": run_elapsed_s,
        "preflight_elapsed_s": preflight_elapsed_s,
        "requested_research_jobs": config.num_jobs,
        "job_concurrency": config.job_concurrency,
        "completed_research_jobs": len(successful_jobs),
        "research_jobs_per_hour": (
            len(successful_jobs) * 3600.0 / elapsed_s
            if all_success and elapsed_s > 0
            else 0.0
        ),
        "expected_requests_per_job": expected_requests_per_job,
        "expected_requests_by_job": expected_requests_by_job,
        "expected_requests": expected_requests,
        "completed_requests": len(records),
        "successful_requests": len(success_records),
        "status_counts": dict(sorted(status_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "job_status_counts": dict(sorted(job_status_counts.items())),
        "job_latency_s": {
            "p50": _percentile(
                [float(job["elapsed_s"]) for job in successful_jobs],
                0.50,
            ),
            "p95": _percentile(
                [float(job["elapsed_s"]) for job in successful_jobs],
                0.95,
            ),
            "max": (
                max(float(job["elapsed_s"]) for job in successful_jobs)
                if successful_jobs
                else None
            ),
        },
        "phase_metrics": phase_metrics,
        "request_latency_s": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "subagent_latency_s": {
            "p50": _percentile(subagent_latencies, 0.50),
            "p95": _percentile(subagent_latencies, 0.95),
            "p99": _percentile(subagent_latencies, 0.99),
            "max": max(subagent_latencies) if subagent_latencies else None,
        },
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "total_tokens_per_s": (
            (total_prompt_tokens + total_completion_tokens) / elapsed_s
            if elapsed_s > 0
            else 0.0
        ),
        "route_worker_counts": dict(sorted(route_worker_counts.items())),
        "route_method_counts": dict(sorted(route_method_counts.items())),
        "route_reason_counts": dict(sorted(route_reason_counts.items())),
        "prefix_cache": _prefix_cache_metrics(records),
        "distinct_route_workers": len(route_worker_counts),
        "rounds_attempted": len(rounds),
        "rounds_completed": sum(
            str(round_row.get("status")) == "success"
            for round_row in rounds
        ),
        "round_metrics": rounds,
        "job_metrics": jobs,
        "artifact_paths": {
            "run_info": str(config.output_dir / "run_info.json"),
            "raw_outputs": str(config.output_dir / "raw_outputs.jsonl"),
            "parsed_outputs": str(config.output_dir / "parsed_outputs.jsonl"),
            "per_sample_results": str(
                config.output_dir / "per_sample_results.jsonl"
            ),
            "round_metrics": str(config.output_dir / "round_metrics.jsonl"),
            "job_metrics": str(config.output_dir / "job_metrics.jsonl"),
            "aggregate_metrics": str(
                config.output_dir / "aggregate_metrics.json"
            ),
        },
    }


def _exception_status(exc: Exception) -> str:
    if isinstance(exc, PreflightParseError):
        return "parse_failed"
    if isinstance(exc, ValueError):
        return "invalid_input"
    if isinstance(exc, BenchmarkFailed):
        return exc.status
    return "model_failed"


def _job_metrics_row(
    config: BenchmarkConfig,
    *,
    job_index: int,
    planned_subagent_counts: tuple[int, ...],
    status: str,
    error: str | None,
    records: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    started_offset_s: float,
    finished_offset_s: float,
) -> dict[str, Any]:
    successful_records = [
        record for record in records if record["status"] == "success"
    ]
    elapsed_s = finished_offset_s - started_offset_s
    prompt_tokens = sum(
        int(record.get("actual_prompt_tokens") or 0)
        for record in successful_records
    )
    completion_tokens = sum(
        int(record.get("actual_completion_tokens") or 0)
        for record in successful_records
    )
    expected_requests = (
        sum(planned_subagent_counts) + config.rounds + 1
    )
    route_worker_counts = Counter(
        str(record["route_worker"])
        for record in successful_records
        if record.get("route_worker")
    )
    return {
        "job_index": job_index,
        "status": status,
        "error": error,
        "started_offset_s": started_offset_s,
        "finished_offset_s": finished_offset_s,
        "elapsed_s": elapsed_s,
        "planned_subagent_requests_per_round": list(
            planned_subagent_counts
        ),
        "expected_requests": expected_requests,
        "completed_requests": len(records),
        "successful_requests": len(successful_records),
        "status_counts": dict(
            sorted(Counter(record["status"] for record in records).items())
        ),
        "rounds_attempted": len(rounds),
        "rounds_completed": sum(
            str(round_row.get("status")) == "success"
            for round_row in rounds
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "requests_per_s": (
            len(successful_records) / elapsed_s
            if elapsed_s > 0
            else 0.0
        ),
        "total_tokens_per_s": (
            (prompt_tokens + completion_tokens) / elapsed_s
            if elapsed_s > 0
            else 0.0
        ),
        "route_worker_counts": dict(sorted(route_worker_counts.items())),
        "prefix_cache": _prefix_cache_metrics(records),
    }


async def _run_one_job(
    config: BenchmarkConfig,
    *,
    job_index: int,
    writer: ArtifactWriter,
    worker_prefix_cache_block_sizes: dict[str, int | None],
    workload_started: float,
    post_fn=post_json,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    records: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    planned_subagent_counts = _subagent_counts_for_job(config, job_index)
    job_started = time.perf_counter()
    job_status = "model_failed"
    job_error: str | None = None
    try:
        rng = random.Random(config.seed + job_index)
        main_shared_prefix = tuple(
            synthetic_token_ids(
                config.main_overhead_tokens,
                seed=rng.getrandbits(63),
                low=config.synthetic_token_id_low,
                high=config.synthetic_token_id_high,
            )
        )
        round_summary_segments: list[tuple[int, ...]] = []
        main_prompt_history_by_worker: dict[
            str,
            list[tuple[int, ...]],
        ] = {}

        for round_index in range(config.rounds):
            round_started = time.perf_counter()
            specs = []
            sampled_subagent_requests = planned_subagent_counts[round_index]
            for request_index in range(sampled_subagent_requests):
                article_tokens = sample_bucketed_tokens(
                    rng,
                    config.article_token_buckets,
                )
                completion_tokens = sample_bucketed_tokens(
                    rng,
                    config.subagent_output_token_buckets,
                )
                specs.append(
                    RequestSpec(
                        sample_id=(
                            f"job-{job_index:04d}-round-{round_index:02d}-"
                            f"subagent-{request_index:03d}"
                        ),
                        job_index=job_index,
                        phase="subagent",
                        round_index=round_index,
                        request_index=request_index,
                        prompt_tokens=config.query_tokens + article_tokens,
                        completion_tokens=completion_tokens,
                        prompt_seed=rng.getrandbits(63),
                        method_preferences=config.subagent_methods,
                        required_tags=config.subagent_required_tags,
                        article_tokens=article_tokens,
                    )
                )

            barrier_started = time.perf_counter()
            subagent_results = await asyncio.gather(
                *[
                    run_request(
                        spec,
                        config,
                        writer,
                        worker_prefix_cache_block_sizes=(
                            worker_prefix_cache_block_sizes
                        ),
                        main_prompt_history_by_worker=(
                            main_prompt_history_by_worker
                        ),
                        timeline_origin=workload_started,
                        post_fn=post_fn,
                    )
                    for spec in specs
                ]
            )
            barrier_s = time.perf_counter() - barrier_started
            records.extend(subagent_results)
            subagent_failure = _failure_status(subagent_results)
            subagent_latencies = [
                float(record["latency_s"])
                for record in subagent_results
            ]
            subagent_prompt_tokens = sum(
                int(record.get("actual_prompt_tokens") or 0)
                for record in subagent_results
            )
            subagent_completion_tokens = sum(
                int(record.get("actual_completion_tokens") or 0)
                for record in subagent_results
            )
            round_rows.append(
                {
                    "job_index": job_index,
                    "round_index": round_index,
                    "status": subagent_failure or "model_failed",
                    "sampled_subagent_requests": (
                        sampled_subagent_requests
                    ),
                    "subagent_requests": len(subagent_results),
                    "subagent_successful_requests": sum(
                        record["status"] == "success"
                        for record in subagent_results
                    ),
                    "subagent_prompt_tokens": subagent_prompt_tokens,
                    "subagent_completion_tokens": subagent_completion_tokens,
                    "subagent_barrier_s": barrier_s,
                    "subagent_latency_p50_s": _percentile(
                        subagent_latencies,
                        0.50,
                    ),
                    "subagent_latency_p95_s": _percentile(
                        subagent_latencies,
                        0.95,
                    ),
                    "subagent_latency_max_s": max(subagent_latencies),
                    "straggler_gap_s": (
                        max(subagent_latencies)
                        - statistics.median(subagent_latencies)
                    ),
                    "main_agent_prompt_tokens": None,
                    "main_agent_completion_tokens": None,
                    "main_agent_expected_reusable_prefix_tokens": None,
                    "main_agent_block_aligned_expected_reusable_prefix_tokens": (
                        None
                    ),
                    "main_agent_actual_prefix_matched_tokens": None,
                    "main_agent_latency_s": None,
                    "round_elapsed_s": time.perf_counter() - round_started,
                }
            )
            _require_success(
                subagent_results,
                f"Job {job_index} round {round_index} subagents",
            )

            accumulated_summaries = tuple(
                token_id
                for segment in round_summary_segments
                for token_id in segment
            )
            current_answer_segment = tuple(
                token_id
                for record in subagent_results
                for token_id in synthetic_token_ids(
                    int(record["actual_completion_tokens"]),
                    seed=rng.getrandbits(63),
                    low=config.synthetic_token_id_low,
                    high=config.synthetic_token_id_high,
                )
            )
            main_prompt = (
                main_shared_prefix
                + accumulated_summaries
                + current_answer_segment
            )
            round_summary_tokens = rng.randint(
                config.min_round_summary_tokens,
                config.max_round_summary_tokens,
            )
            main_spec = RequestSpec(
                sample_id=(
                    f"job-{job_index:04d}-round-{round_index:02d}-main-agent"
                ),
                job_index=job_index,
                phase="round_summary",
                round_index=round_index,
                request_index=None,
                prompt_tokens=len(main_prompt),
                completion_tokens=round_summary_tokens,
                prompt_seed=rng.getrandbits(63),
                method_preferences=config.main_agent_methods,
                required_tags=config.main_agent_required_tags,
                prompt_token_ids=main_prompt,
            )
            main_result = await run_request(
                main_spec,
                config,
                writer,
                worker_prefix_cache_block_sizes=(
                    worker_prefix_cache_block_sizes
                ),
                main_prompt_history_by_worker=(
                    main_prompt_history_by_worker
                ),
                timeline_origin=workload_started,
                post_fn=post_fn,
            )
            records.append(main_result)
            round_rows[-1].update(
                {
                    "status": main_result["status"],
                    "main_agent_prompt_tokens": main_result[
                        "actual_prompt_tokens"
                    ],
                    "main_agent_completion_tokens": main_result[
                        "actual_completion_tokens"
                    ],
                    "main_agent_expected_reusable_prefix_tokens": (
                        main_result["expected_reusable_prefix_tokens"]
                    ),
                    "main_agent_block_aligned_expected_reusable_prefix_tokens": (
                        main_result[
                            "block_aligned_expected_reusable_prefix_tokens"
                        ]
                    ),
                    "main_agent_actual_prefix_matched_tokens": (
                        main_result["actual_prefix_matched_tokens"]
                    ),
                    "main_agent_latency_s": main_result["latency_s"],
                    "round_elapsed_s": time.perf_counter() - round_started,
                }
            )
            _require_success(
                [main_result],
                f"Job {job_index} round {round_index} main agent",
            )
            round_summary_segments.append(
                tuple(
                    synthetic_token_ids(
                        int(main_result["actual_completion_tokens"]),
                        seed=rng.getrandbits(63),
                        low=config.synthetic_token_id_low,
                        high=config.synthetic_token_id_high,
                    )
                )
            )
            round_rows[-1]["status"] = "success"
            round_rows[-1]["round_elapsed_s"] = (
                time.perf_counter() - round_started
            )

        final_overhead_segment = tuple(
            synthetic_token_ids(
                config.final_overhead_tokens,
                seed=rng.getrandbits(63),
                low=config.synthetic_token_id_low,
                high=config.synthetic_token_id_high,
            )
        )
        final_prompt = (
            main_shared_prefix
            + tuple(
                token_id
                for segment in round_summary_segments
                for token_id in segment
            )
            + final_overhead_segment
        )
        final_spec = RequestSpec(
            sample_id=f"job-{job_index:04d}-final-main-agent",
            job_index=job_index,
            phase="final_summary",
            round_index=None,
            request_index=None,
            prompt_tokens=len(final_prompt),
            completion_tokens=rng.randint(
                config.min_final_output_tokens,
                config.max_final_output_tokens,
            ),
            prompt_seed=rng.getrandbits(63),
            method_preferences=config.main_agent_methods,
            required_tags=config.main_agent_required_tags,
            prompt_token_ids=final_prompt,
        )
        final_result = await run_request(
            final_spec,
            config,
            writer,
            worker_prefix_cache_block_sizes=(
                worker_prefix_cache_block_sizes
            ),
            main_prompt_history_by_worker=(
                main_prompt_history_by_worker
            ),
            timeline_origin=workload_started,
            post_fn=post_fn,
        )
        records.append(final_result)
        _require_success([final_result], f"Job {job_index} final main agent")

        distinct_workers = {
            str(record["route_worker"])
            for record in records
            if record["status"] == "success" and record.get("route_worker")
        }
        if (
            config.require_router
            and _roles_require_distinct_workers(config)
            and len(distinct_workers) < 2
        ):
            raise BenchmarkFailed(
                "metric_failed",
                f"Job {job_index} successful requests exercised only "
                f"{len(distinct_workers)} distinct workers; expected at least "
                "2 role workers."
            )
        job_status = "success"
    except Exception as exc:
        job_status = _exception_status(exc)
        job_error = f"{type(exc).__name__}: {exc}"

    job_finished = time.perf_counter()
    job_row = _job_metrics_row(
        config,
        job_index=job_index,
        planned_subagent_counts=planned_subagent_counts,
        status=job_status,
        error=job_error,
        records=records,
        rounds=round_rows,
        started_offset_s=job_started - workload_started,
        finished_offset_s=job_finished - workload_started,
    )
    return job_row, records, round_rows


async def _run_benchmark_impl(
    config: BenchmarkConfig,
    *,
    get_fn=get_json,
    post_fn=post_json,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    preflight_info: dict[str, Any] | None = None
    run_error: str | None = None
    aggregate_status = "model_failed"
    run_started = time.perf_counter()
    workload_started: float | None = None
    preflight_elapsed_s = 0.0
    client_code_state_error: Exception | None = None
    try:
        client_code_state, client_worktree_patch = (
            _capture_client_code_state()
        )
    except Exception as exc:
        client_code_state_error = exc
        client_worktree_patch = None
        client_code_state = {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": _git_dirty(),
            "worktree_patch": None,
        }

    with ArtifactWriter(config.output_dir) as writer:
        if client_worktree_patch is not None:
            writer.write_text(
                "client_worktree.patch",
                client_worktree_patch,
            )
        writer.write_json(
            "run_info.json",
            _run_info(
                config,
                None,
                status="running",
                client_code_state=client_code_state,
            ),
        )
        try:
            if client_code_state_error is not None:
                raise client_code_state_error
            preflight_started = time.perf_counter()
            preflight_info = await preflight(config, get_fn=get_fn)
            preflight_elapsed_s = time.perf_counter() - preflight_started
            writer.write_json(
                "run_info.json",
                _run_info(
                    config,
                    preflight_info,
                    status="running",
                    client_code_state=client_code_state,
                ),
            )
            worker_prefix_cache_block_sizes = dict(
                preflight_info["worker_prefix_cache_block_sizes"]
            )
            workload_started = time.perf_counter()
            semaphore = asyncio.Semaphore(config.job_concurrency)

            async def run_scheduled_job(job_index: int):
                async with semaphore:
                    return await _run_one_job(
                        config,
                        job_index=job_index,
                        writer=writer,
                        worker_prefix_cache_block_sizes=(
                            worker_prefix_cache_block_sizes
                        ),
                        workload_started=workload_started,
                        post_fn=post_fn,
                    )

            outcomes = await asyncio.gather(
                *[
                    run_scheduled_job(job_index)
                    for job_index in range(config.num_jobs)
                ]
            )
            for job_row, job_records, job_round_rows in outcomes:
                job_rows.append(job_row)
                records.extend(job_records)
                round_rows.extend(job_round_rows)

            failed_jobs = [
                job for job in job_rows if job["status"] != "success"
            ]
            if failed_jobs:
                aggregate_status = (
                    _failure_status(failed_jobs) or "model_failed"
                )
                run_error = "; ".join(
                    f"job={job['job_index']} status={job['status']} "
                    f"error={job['error']}"
                    for job in failed_jobs
                )
            else:
                if config.require_router:
                    if not any(
                        record["status"] == "success"
                        and record["phase"] != "subagent"
                        and int(
                            record.get(
                                "block_aligned_expected_reusable_prefix_tokens"
                            )
                            or 0
                        )
                        > 0
                        and int(
                            record.get(
                                "actual_prefix_matched_tokens"
                            )
                            or 0
                        )
                        > 0
                        for record in records
                    ):
                        raise BenchmarkFailed(
                            "metric_failed",
                            "Run completed without a verified main-agent "
                            "prefix-cache hit; no main-agent record had "
                            "both positive block-aligned expected reuse "
                            "and positive actual matched tokens. Keep "
                            "reusable main prompts on a prefix-cache "
                            "worker or inspect routing and cache state.",
                        )
                    distinct_workers = {
                        str(record["route_worker"])
                        for record in records
                        if (
                            record["status"] == "success"
                            and record.get("route_worker")
                        )
                    }
                    if len(distinct_workers) < config.min_healthy_workers:
                        raise BenchmarkFailed(
                            "metric_failed",
                            "Run exercised only "
                            f"{len(distinct_workers)} distinct workers; "
                            f"expected at least {config.min_healthy_workers}.",
                        )
                aggregate_status = "success"
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"
            aggregate_status = _exception_status(exc)

        for round_row in round_rows:
            writer.write_jsonl("round_metrics", round_row)
        for job_row in job_rows:
            writer.write_jsonl("job_metrics", job_row)
        finished = time.perf_counter()
        run_elapsed_s = finished - run_started
        elapsed_s = (
            finished - workload_started
            if workload_started is not None
            else 0.0
        )
        aggregate = _aggregate_metrics(
            config,
            records,
            round_rows,
            job_rows,
            elapsed_s=elapsed_s,
            run_elapsed_s=run_elapsed_s,
            preflight_elapsed_s=preflight_elapsed_s,
            status=aggregate_status,
            error=run_error,
        )
        writer.write_json("aggregate_metrics.json", aggregate)
        writer.write_json(
            "run_info.json",
            _run_info(
                config,
                preflight_info,
                status=aggregate_status,
                error=run_error,
                client_code_state=client_code_state,
            ),
        )

    if aggregate_status != "success":
        raise BenchmarkFailed(
            aggregate_status,
            f"Simulated Deep Research benchmark failed with "
            f"status={aggregate_status}: {run_error}"
        )
    return aggregate


async def run_benchmark(
    config: BenchmarkConfig,
    *,
    get_fn=get_json,
    post_fn=post_json,
) -> dict[str, Any]:
    validate_config(config)
    if post_fn is not post_json:
        return await _run_benchmark_impl(
            config,
            get_fn=get_fn,
            post_fn=post_fn,
        )

    with ThreadPoolExecutor(
        max_workers=(
            config.job_concurrency * _subagent_count_bounds(config)[1]
        ),
        thread_name_prefix="simulated-deep-research",
    ) as executor:
        sized_post_fn = partial(post_json, executor=executor)
        return await _run_benchmark_impl(
            config,
            get_fn=get_fn,
            post_fn=sized_post_fn,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more synthetic Deep Research jobs through the "
            "Sparse-Engine smart router."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:18180/v1",
        help="Smart-router OpenAI base URL.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--job-concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--articles-per-round", type=int, default=20)
    parser.add_argument(
        "--min-subagents-per-round",
        type=int,
        default=None,
        help=(
            "Inclusive minimum subagent count sampled independently for each "
            "job round. Must be used with --max-subagents-per-round; when "
            "unset, --articles-per-round remains fixed."
        ),
    )
    parser.add_argument(
        "--max-subagents-per-round",
        type=int,
        default=None,
        help=(
            "Inclusive maximum subagent count sampled independently for each "
            "job round. Must be used with --min-subagents-per-round."
        ),
    )
    parser.add_argument(
        "--article-token-buckets",
        default="60:1000:8000,25:8001:16000,10:16001:32000,5:32001:64000",
        help="Comma-separated WEIGHT:MIN:MAX article-token buckets.",
    )
    parser.add_argument("--query-tokens", type=int, default=64)
    parser.add_argument(
        "--subagent-output-token-buckets",
        default="90:100:600,10:800:1500",
        help="Comma-separated WEIGHT:MIN:MAX subagent-output buckets.",
    )
    parser.add_argument("--main-overhead-tokens", type=int, default=128)
    parser.add_argument("--min-round-summary-tokens", type=int, default=512)
    parser.add_argument("--max-round-summary-tokens", type=int, default=1_024)
    parser.add_argument("--final-overhead-tokens", type=int, default=128)
    parser.add_argument("--min-final-output-tokens", type=int, default=1_000)
    parser.add_argument("--max-final-output-tokens", type=int, default=2_000)
    parser.add_argument(
        "--subagent-methods",
        default="snapkv",
        help="Comma-separated smart-router method preference for subagents.",
    )
    parser.add_argument(
        "--main-agent-methods",
        default="omnikv,vanilla",
        help="Comma-separated smart-router method preference for the main agent.",
    )
    parser.add_argument(
        "--subagent-required-tags",
        default="",
        help="Comma-separated worker tags required for subagent requests.",
    )
    parser.add_argument(
        "--main-agent-required-tags",
        default="",
        help="Comma-separated worker tags required for main-agent requests.",
    )
    parser.add_argument("--synthetic-token-id-low", type=int, default=100)
    parser.add_argument("--synthetic-token-id-high", type=int, default=255)
    parser.add_argument("--request-timeout-s", type=float, default=930.0)
    parser.add_argument(
        "--router-timeout-margin-s",
        type=float,
        default=30.0,
        help=(
            "Required margin between the client timeout and the router's "
            "advertised upstream timeout."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--min-healthy-workers", type=int, default=2)
    parser.add_argument(
        "--allow-direct-server",
        action="store_true",
        help=(
            "Disable router identity, route-header, method, and two-worker "
            "validation. Intended only for an explicit single-worker baseline."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        output_dir=args.output_dir.expanduser().resolve(),
        num_jobs=args.num_jobs,
        job_concurrency=args.job_concurrency,
        rounds=args.rounds,
        articles_per_round=args.articles_per_round,
        min_subagents_per_round=args.min_subagents_per_round,
        max_subagents_per_round=args.max_subagents_per_round,
        article_token_buckets=_parse_token_buckets(
            args.article_token_buckets,
            "--article-token-buckets",
        ),
        query_tokens=args.query_tokens,
        subagent_output_token_buckets=_parse_token_buckets(
            args.subagent_output_token_buckets,
            "--subagent-output-token-buckets",
        ),
        main_overhead_tokens=args.main_overhead_tokens,
        min_round_summary_tokens=args.min_round_summary_tokens,
        max_round_summary_tokens=args.max_round_summary_tokens,
        final_overhead_tokens=args.final_overhead_tokens,
        min_final_output_tokens=args.min_final_output_tokens,
        max_final_output_tokens=args.max_final_output_tokens,
        subagent_methods=_parse_methods(
            args.subagent_methods,
            "--subagent-methods",
        ),
        main_agent_methods=_parse_methods(
            args.main_agent_methods,
            "--main-agent-methods",
        ),
        subagent_required_tags=_parse_tags(
            args.subagent_required_tags,
            "--subagent-required-tags",
        ),
        main_agent_required_tags=_parse_tags(
            args.main_agent_required_tags,
            "--main-agent-required-tags",
        ),
        synthetic_token_id_low=args.synthetic_token_id_low,
        synthetic_token_id_high=args.synthetic_token_id_high,
        request_timeout_s=args.request_timeout_s,
        router_timeout_margin_s=args.router_timeout_margin_s,
        seed=args.seed,
        require_router=not args.allow_direct_server,
        min_healthy_workers=args.min_healthy_workers,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    aggregate = asyncio.run(run_benchmark(config))
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
