import asyncio
import json
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from benchmark.simulated_deep_research import run


CODE_REVISION = {
    "git_commit": "0123456789abcdef",
    "git_branch": "test",
    "git_dirty": False,
    "package_version": None,
}


def _clean_git_command(*args):
    outputs = {
        ("rev-parse", "HEAD"): "0123456789abcdef\n",
        ("branch", "--show-current"): "test\n",
        ("status", "--porcelain"): "",
        ("status", "--porcelain", "--untracked-files=all"): "",
    }
    return run.subprocess.CompletedProcess(
        ["git", *args],
        0,
        outputs[args],
        "",
    )


def _json_response(
    payload,
    *,
    status=200,
    headers=None,
):
    return run.HttpResponse(
        status=status,
        headers=headers or {},
        body=json.dumps(payload).encode("utf-8"),
    )


class FakeService:
    def __init__(
        self,
        *,
        fail_sample_number=None,
        wrong_subagent_method=False,
        zero_main_prefix_match=False,
        miss_first_reusable_main_prefix=False,
        partial_main_prefix_match=False,
        invalid_prefix_match_header=False,
        contaminate_first_main_prefix_match=False,
    ):
        self.active = 0
        self.max_active = 0
        self.completion_calls = 0
        self.payloads = []
        self.fail_sample_number = fail_sample_number
        self.wrong_subagent_method = wrong_subagent_method
        self.zero_main_prefix_match = zero_main_prefix_match
        self.miss_first_reusable_main_prefix = (
            miss_first_reusable_main_prefix
        )
        self.partial_main_prefix_match = partial_main_prefix_match
        self.invalid_prefix_match_header = invalid_prefix_match_header
        self.contaminate_first_main_prefix_match = (
            contaminate_first_main_prefix_match
        )
        self.previous_main_prompts = []
        self.prefix_cache_block_size = 2
        self.max_model_len = 65_536

    async def get(self, url, _timeout_s):
        if url.endswith("/models"):
            return _json_response(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "sim-model",
                            "owned_by": "sparseengine-router",
                            "max_model_len": self.max_model_len,
                        }
                    ],
                }
            )
        if url.endswith("/health"):
            return _json_response(
                {
                    "status": "ok",
                    "healthy_workers": [
                        "http://snap-worker",
                        "http://main-worker",
                    ],
                    "router_policy": {
                        "request_timeout_s": 900.0,
                        "control_timeout_s": 5.0,
                        "overload_load_factor": 1.0,
                        "load_abs_threshold": 0,
                        "profiles": {},
                        "code_revision": CODE_REVISION,
                    },
                }
            )
        if url.endswith("/v1/worker/info"):
            method = "snapkv" if "snap-worker" in url else "omnikv"
            return _json_response(
                {
                    "served_model_name": "sim-model",
                    "max_model_len": self.max_model_len,
                    "vocab_size": 32_000,
                    "sparse_method": method,
                    "tags": (
                        ["subagent"]
                        if method == "snapkv"
                        else ["main-agent"]
                    ),
                    "prefix_cache_enabled": method == "omnikv",
                    "prefix_cache_block_size": (
                        self.prefix_cache_block_size
                    ),
                    "code_revision": CODE_REVISION,
                    "benchmark_config": {
                        "gpu_memory_utilization": 0.9,
                        "prefill_schedule_policy": "all_chunked",
                        "engine_prefill_chunk_size": 8192,
                        "decode_graph": True,
                        "enable_prefix_caching": method == "omnikv",
                        "prefix_cache_block_size": (
                            self.prefix_cache_block_size
                        ),
                    },
                }
            )
        raise AssertionError(f"Unexpected GET URL: {url}")

    async def post(self, url, payload, _timeout_s):
        self.completion_calls += 1
        self.payloads.append(payload)
        call_number = self.completion_calls
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.001)
            methods = payload["sengine_method_preference"].split(",")
            if methods == ["snapkv"]:
                worker = "http://snap-worker"
                method = (
                    "omnikv"
                    if self.wrong_subagent_method
                    else "snapkv"
                )
            else:
                worker = "http://main-worker"
                method = "omnikv"
            matched_tokens = 0
            if methods != ["snapkv"]:
                prompt = tuple(payload["prompt"])
                raw_matched_tokens = max(
                    (
                        run._common_prefix_tokens(previous, prompt)
                        for previous in self.previous_main_prompts
                    ),
                    default=0,
                )
                matched_tokens = (
                    raw_matched_tokens
                    // self.prefix_cache_block_size
                    * self.prefix_cache_block_size
                )
                if (
                    not self.previous_main_prompts
                    and self.contaminate_first_main_prefix_match
                ):
                    matched_tokens = self.prefix_cache_block_size
                self.previous_main_prompts.append(prompt)
                if self.zero_main_prefix_match:
                    matched_tokens = 0
                elif (
                    self.miss_first_reusable_main_prefix
                    and matched_tokens > 0
                ):
                    matched_tokens = 0
                    self.miss_first_reusable_main_prefix = False
                elif (
                    self.partial_main_prefix_match
                    and matched_tokens > self.prefix_cache_block_size
                ):
                    matched_tokens -= self.prefix_cache_block_size
            matched_tokens_header = (
                "invalid"
                if self.invalid_prefix_match_header
                else str(matched_tokens)
            )
            headers = {
                "x-sparseengine-worker": worker,
                "x-sparseengine-route-reason": "lowest_load_no_prefix_match",
                "x-sparseengine-sparse-method": method,
                "x-sparseengine-prefix-matched-tokens": (
                    matched_tokens_header
                ),
            }
            if call_number == self.fail_sample_number:
                return _json_response(
                    {"error": "synthetic failure"},
                    status=500,
                    headers=headers,
                )
            prompt_tokens = len(payload["prompt"])
            completion_tokens = payload["max_tokens"]
            return _json_response(
                {
                    "id": f"cmpl-{call_number}",
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": 0,
                            "text": "synthetic output",
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
                headers=headers,
            )
        finally:
            self.active -= 1


class SimulatedDeepResearchTest(unittest.TestCase):
    def setUp(self):
        git_patch = patch.object(
            run,
            "_git_command",
            side_effect=_clean_git_command,
        )
        git_patch.start()
        self.addCleanup(git_patch.stop)

    def _config(self, output_dir):
        return run.BenchmarkConfig(
            base_url="http://router.test/v1",
            model="sim-model",
            output_dir=output_dir,
            rounds=2,
            articles_per_round=3,
            article_token_buckets=((1, 10, 20),),
            query_tokens=4,
            subagent_output_token_buckets=((1, 2, 5),),
            main_overhead_tokens=3,
            min_round_summary_tokens=4,
            max_round_summary_tokens=4,
            final_overhead_tokens=2,
            min_final_output_tokens=6,
            max_final_output_tokens=6,
            seed=7,
        )

    def test_required_model_len_covers_each_request_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            self.assertEqual(run.required_model_len(config), 29)

    def test_parses_random_subagent_count_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = run.config_from_args(
                run.build_arg_parser().parse_args(
                    [
                        "--model",
                        "sim-model",
                        "--output-dir",
                        str(Path(tmp) / "run"),
                        "--min-subagents-per-round",
                        "10",
                        "--max-subagents-per-round",
                        "40",
                    ]
                )
            )
            run.validate_config(config)
            self.assertEqual(config.min_subagents_per_round, 10)
            self.assertEqual(config.max_subagents_per_round, 40)
            self.assertEqual(run._subagent_count_bounds(config), (10, 40))

    def test_rejects_invalid_random_subagent_count_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._config(Path(tmp) / "run")
            invalid_cases = (
                (
                    {
                        "min_subagents_per_round": 2,
                        "max_subagents_per_round": None,
                    },
                    "must be set together",
                ),
                (
                    {
                        "min_subagents_per_round": 0,
                        "max_subagents_per_round": 2,
                    },
                    "must be positive",
                ),
                (
                    {
                        "min_subagents_per_round": 4,
                        "max_subagents_per_round": 3,
                    },
                    "must not exceed",
                ),
            )
            for overrides, message in invalid_cases:
                with self.subTest(overrides=overrides):
                    config = run.BenchmarkConfig(
                        **{
                            **base.__dict__,
                            **overrides,
                        }
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        run.validate_config(config)

    def test_rejects_invalid_token_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "article_token_buckets": ((0, 10, 20),),
                }
            )
            with self.assertRaisesRegex(ValueError, "weight must be positive"):
                run.validate_config(config)

    def test_required_model_len_includes_accumulated_main_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "rounds": 4,
                    "articles_per_round": 2,
                    "article_token_buckets": ((1, 1, 1),),
                    "query_tokens": 1,
                    "subagent_output_token_buckets": ((1, 1, 1),),
                    "main_overhead_tokens": 3,
                    "min_round_summary_tokens": 5,
                    "max_round_summary_tokens": 5,
                    "final_overhead_tokens": 2,
                    "min_final_output_tokens": 1,
                    "max_final_output_tokens": 1,
                }
            )
            self.assertEqual(run.required_model_len(config), 26)

    def test_required_model_len_uses_random_subagent_upper_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "min_subagents_per_round": 2,
                    "max_subagents_per_round": 5,
                }
            )
            self.assertEqual(run.required_model_len(config), 36)

    def test_allows_overlapping_methods_without_role_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "main_agent_methods": ("snapkv",),
                }
            )
            run.validate_config(config)
            self.assertFalse(
                run._roles_require_distinct_workers(config)
            )

    def test_rejects_partial_role_tags_for_overlapping_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "main_agent_methods": ("snapkv",),
                    "subagent_required_tags": ("subagent",),
                }
            )
            with self.assertRaisesRegex(ValueError, "disjoint tags"):
                run.validate_config(config)

    def test_allows_overlapping_methods_with_disjoint_role_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "subagent_methods": ("vanilla",),
                    "main_agent_methods": ("vanilla",),
                    "subagent_required_tags": ("subagent",),
                    "main_agent_required_tags": ("main-agent",),
                }
            )
            run.validate_config(config)
            self.assertTrue(
                run._roles_require_distinct_workers(config)
            )
            spec = run.RequestSpec(
                sample_id="job-0000-subagent",
                job_index=0,
                phase="subagent",
                round_index=0,
                request_index=0,
                prompt_tokens=2,
                completion_tokens=1,
                prompt_seed=1,
                method_preferences=config.subagent_methods,
                required_tags=config.subagent_required_tags,
            )
            payload = run.build_payload(spec, config)
            self.assertEqual(payload["sengine_method_preference"], "vanilla")
            self.assertEqual(
                payload["sengine_required_tags"],
                ["subagent"],
            )

    def test_preflight_accepts_two_tag_isolated_vanilla_workers(self):
        async def vanilla_get(url, _timeout_s):
            if url.endswith("/models"):
                return _json_response(
                    {
                        "data": [
                            {
                                "id": "sim-model",
                                "owned_by": "sparseengine-router",
                                "max_model_len": 65_536,
                            }
                        ]
                    }
                )
            if url.endswith("/health"):
                return _json_response(
                    {
                        "healthy_workers": [
                            "http://vanilla-subagent",
                            "http://vanilla-main",
                        ],
                        "router_policy": {
                            "request_timeout_s": 900.0,
                            "control_timeout_s": 5.0,
                            "overload_load_factor": 1.0,
                            "load_abs_threshold": 0,
                            "profiles": {},
                            "code_revision": CODE_REVISION,
                        },
                    }
                )
            if url.endswith("/v1/worker/info"):
                main_worker = "vanilla-main" in url
                return _json_response(
                    {
                        "served_model_name": "sim-model",
                        "max_model_len": 65_536,
                        "vocab_size": 32_000,
                        "sparse_method": "",
                        "tags": (
                            ["main-agent"]
                            if main_worker
                            else ["subagent"]
                        ),
                        "prefix_cache_enabled": main_worker,
                        "prefix_cache_block_size": (
                            2 if main_worker else None
                        ),
                        "code_revision": CODE_REVISION,
                        "benchmark_config": {
                            "decode_graph": True,
                            "enable_prefix_caching": main_worker,
                        },
                    }
                )
            raise AssertionError(f"Unexpected GET URL: {url}")

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "subagent_methods": ("vanilla",),
                    "main_agent_methods": ("vanilla",),
                    "subagent_required_tags": ("subagent",),
                    "main_agent_required_tags": ("main-agent",),
                }
            )
            result = asyncio.run(
                run.preflight(config, get_fn=vanilla_get)
            )
            self.assertEqual(len(result["workers"]), 2)
            self.assertEqual(
                result["worker_prefix_cache_block_sizes"],
                {
                    "http://vanilla-main": 2,
                    "http://vanilla-subagent": None,
                },
            )

    def test_preflight_accepts_balanced_untagged_vanilla_workers(self):
        async def vanilla_get(url, _timeout_s):
            if url.endswith("/models"):
                return _json_response(
                    {
                        "data": [
                            {
                                "id": "sim-model",
                                "owned_by": "sparseengine-router",
                                "max_model_len": 65_536,
                            }
                        ]
                    }
                )
            if url.endswith("/health"):
                return _json_response(
                    {
                        "healthy_workers": [
                            "http://vanilla-0",
                            "http://vanilla-1",
                        ],
                        "router_policy": {
                            "request_timeout_s": 900.0,
                            "control_timeout_s": 5.0,
                            "overload_load_factor": 1.0,
                            "load_abs_threshold": 0,
                            "profiles": {},
                            "code_revision": CODE_REVISION,
                        },
                    }
                )
            if url.endswith("/v1/worker/info"):
                return _json_response(
                    {
                        "served_model_name": "sim-model",
                        "max_model_len": 65_536,
                        "vocab_size": 32_000,
                        "sparse_method": "",
                        "tags": [],
                        "prefix_cache_enabled": True,
                        "prefix_cache_block_size": 2,
                        "code_revision": CODE_REVISION,
                        "benchmark_config": {
                            "decode_graph": True,
                            "enable_prefix_caching": True,
                        },
                    }
                )
            raise AssertionError(f"Unexpected GET URL: {url}")

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "subagent_methods": ("vanilla",),
                    "main_agent_methods": ("vanilla",),
                }
            )
            result = asyncio.run(
                run.preflight(config, get_fn=vanilla_get)
            )
            self.assertEqual(len(result["workers"]), 2)
            self.assertEqual(
                result["worker_prefix_cache_block_sizes"],
                {
                    "http://vanilla-0": 2,
                    "http://vanilla-1": 2,
                },
            )

    def test_rejects_job_concurrency_above_job_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "num_jobs": 2,
                    "job_concurrency": 3,
                }
            )
            with self.assertRaisesRegex(
                ValueError,
                "must not exceed num_jobs",
            ):
                run.validate_config(config)

    def test_runs_parallel_subagents_and_writes_auditable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService()

            aggregate = asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )

            self.assertEqual(aggregate["status"], "success")
            self.assertEqual(aggregate["expected_requests"], 9)
            self.assertEqual(aggregate["successful_requests"], 9)
            self.assertEqual(aggregate["phase_counts"]["subagent"], 6)
            self.assertEqual(aggregate["phase_counts"]["round_summary"], 2)
            self.assertEqual(aggregate["phase_counts"]["final_summary"], 1)
            self.assertEqual(aggregate["rounds_attempted"], 2)
            self.assertEqual(aggregate["rounds_completed"], 2)
            self.assertEqual(aggregate["requested_research_jobs"], 1)
            self.assertEqual(aggregate["completed_research_jobs"], 1)
            self.assertEqual(aggregate["job_concurrency"], 1)
            self.assertEqual(aggregate["job_status_counts"], {"success": 1})
            self.assertGreater(aggregate["research_jobs_per_hour"], 0.0)
            self.assertEqual(
                aggregate["route_method_counts"],
                {"omnikv": 3, "snapkv": 6},
            )
            self.assertEqual(
                aggregate["phase_metrics"]["subagent"][
                    "route_method_counts"
                ],
                {"snapkv": 6},
            )
            self.assertEqual(
                aggregate["phase_metrics"]["round_summary"][
                    "route_method_counts"
                ],
                {"omnikv": 2},
            )
            self.assertEqual(aggregate["distinct_route_workers"], 2)
            self.assertGreaterEqual(service.max_active, 3)
            main_payloads = [
                payload
                for payload in service.payloads
                if payload["sengine_method_preference"] == "omnikv,vanilla"
            ]
            self.assertEqual(len(main_payloads), 3)
            first_round_prompt = main_payloads[0]["prompt"]
            second_round_prompt = main_payloads[1]["prompt"]
            final_prompt = main_payloads[2]["prompt"]
            self.assertEqual(
                second_round_prompt[: config.main_overhead_tokens],
                first_round_prompt[: config.main_overhead_tokens],
            )
            reusable_before_final = (
                config.main_overhead_tokens
                + config.max_round_summary_tokens
            )
            self.assertEqual(
                final_prompt[:reusable_before_final],
                second_round_prompt[:reusable_before_final],
            )

            for name in (
                "run_info.json",
                "raw_outputs.jsonl",
                "parsed_outputs.jsonl",
                "per_sample_results.jsonl",
                "round_metrics.jsonl",
                "job_metrics.jsonl",
                "aggregate_metrics.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
            sample_rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(sample_rows), 9)
            self.assertTrue(all(row["status"] == "success" for row in sample_rows))
            self.assertTrue(
                all(
                    row["article_tokens"] is not None
                    for row in sample_rows
                    if row["phase"] == "subagent"
                )
            )
            self.assertEqual(
                {
                    row["route_method"]
                    for row in sample_rows
                    if row["phase"] == "subagent"
                },
                {"snapkv"},
            )
            self.assertEqual(
                {
                    row["route_method"]
                    for row in sample_rows
                    if row["phase"] != "subagent"
                },
                {"omnikv"},
            )
            main_rows = [
                row for row in sample_rows if row["phase"] != "subagent"
            ]
            self.assertEqual(
                [
                    row["expected_reusable_prefix_tokens"]
                    for row in main_rows
                ],
                [0, config.main_overhead_tokens, reusable_before_final],
            )
            self.assertEqual(
                [
                    row["actual_prefix_matched_tokens"]
                    for row in main_rows
                ],
                [0, 2, 6],
            )
            self.assertEqual(
                [
                    row[
                        "block_aligned_expected_reusable_prefix_tokens"
                    ]
                    for row in main_rows
                ],
                [0, 2, 6],
            )
            self.assertEqual(
                [
                    row["same_worker_prior_prompt_count"]
                    for row in main_rows
                ],
                [0, 1, 2],
            )
            expected_reuse = (
                config.main_overhead_tokens + reusable_before_final
            )
            self.assertEqual(
                aggregate["prefix_cache"],
                {
                    "expected_reusable_prefix_tokens": expected_reuse,
                    "block_aligned_expected_reusable_prefix_tokens": 8,
                    "actual_prefix_matched_tokens": 8,
                    "observed_requests": 9,
                    "expected_hit_requests": 2,
                    "actual_hit_requests": 2,
                    "unexpected_zero_hit_requests": 0,
                    "partial_hit_requests": 0,
                    "unexpected_excess_hit_requests": 0,
                },
            )
            self.assertEqual(
                aggregate["phase_metrics"]["round_summary"]["prefix_cache"][
                    "actual_hit_requests"
                ],
                1,
            )
            raw_rows = [
                json.loads(line)
                for line in (output_dir / "raw_outputs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [row["request"]["payload"] for row in raw_rows],
                service.payloads,
            )
            self.assertTrue(
                all(
                    isinstance(row["request"]["prompt_seed"], int)
                    for row in raw_rows
                )
            )
            self.assertTrue(
                all(
                    row["request"]["timeout_s"]
                    == config.request_timeout_s
                    for row in raw_rows
                )
            )
            self.assertEqual(
                [
                    row["route_observation"][
                        "actual_prefix_matched_tokens"
                    ]
                    for row in raw_rows
                    if row["route_observation"]["method"] == "omnikv"
                ],
                [0, 2, 6],
            )
            run_info = json.loads(
                (output_dir / "run_info.json").read_text(encoding="utf-8")
            )
            worker_configs = run_info["preflight"]["workers"]
            self.assertEqual(
                [worker["info"]["sparse_method"] for worker in worker_configs],
                ["snapkv", "omnikv"],
            )
            self.assertTrue(
                worker_configs[1]["info"]["benchmark_config"][
                    "enable_prefix_caching"
                ]
            )
            self.assertEqual(
                run_info["preflight"][
                    "worker_prefix_cache_block_sizes"
                ],
                {
                    "http://main-worker": 2,
                    "http://snap-worker": 2,
                },
            )
            self.assertEqual(
                run_info["preflight"]["health"]["router_policy"][
                    "request_timeout_s"
                ],
                900.0,
            )
            self.assertEqual(
                run_info["preflight"]["health"]["router_policy"][
                    "code_revision"
                ]["git_commit"],
                CODE_REVISION["git_commit"],
            )
            self.assertTrue(
                all(
                    worker["info"]["code_revision"]["git_commit"]
                    == CODE_REVISION["git_commit"]
                    for worker in worker_configs
                )
            )

    def test_runs_multiple_jobs_with_bounded_job_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "num_jobs": 3,
                    "job_concurrency": 2,
                }
            )
            service = FakeService()

            aggregate = asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )

            self.assertEqual(aggregate["status"], "success")
            self.assertEqual(aggregate["requested_research_jobs"], 3)
            self.assertEqual(aggregate["completed_research_jobs"], 3)
            self.assertEqual(aggregate["job_concurrency"], 2)
            self.assertEqual(aggregate["expected_requests_per_job"], 9)
            self.assertEqual(aggregate["expected_requests"], 27)
            self.assertEqual(aggregate["successful_requests"], 27)
            self.assertEqual(aggregate["job_status_counts"], {"success": 3})
            self.assertEqual(aggregate["rounds_completed"], 6)
            self.assertEqual(service.max_active, 6)
            job_rows = [
                json.loads(line)
                for line in (output_dir / "job_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [row["job_index"] for row in job_rows],
                [0, 1, 2],
            )
            self.assertTrue(
                all(row["status"] == "success" for row in job_rows)
            )
            self.assertTrue(
                all(
                    row["prefix_cache"]["actual_hit_requests"] >= 2
                    for row in job_rows
                )
            )
            self.assertLess(
                job_rows[1]["started_offset_s"],
                job_rows[0]["finished_offset_s"],
            )
            self.assertGreaterEqual(
                job_rows[2]["started_offset_s"],
                min(
                    job_rows[0]["finished_offset_s"],
                    job_rows[1]["finished_offset_s"],
                ),
            )
            sample_rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                Counter(row["job_index"] for row in sample_rows),
                Counter({0: 9, 1: 9, 2: 9}),
            )

    def test_samples_reproducible_subagent_counts_per_job_and_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "num_jobs": 2,
                    "job_concurrency": 2,
                    "rounds": 3,
                    "min_subagents_per_round": 2,
                    "max_subagents_per_round": 4,
                }
            )
            planned = {
                str(job_index): run._subagent_counts_for_job(
                    config,
                    job_index,
                )
                for job_index in range(config.num_jobs)
            }
            self.assertEqual(
                planned,
                {
                    str(job_index): run._subagent_counts_for_job(
                        config,
                        job_index,
                    )
                    for job_index in range(config.num_jobs)
                },
            )
            self.assertTrue(
                all(
                    2 <= count <= 4
                    for counts in planned.values()
                    for count in counts
                )
            )
            self.assertNotEqual(planned["0"], planned["1"])

            service = FakeService()
            aggregate = asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )

            expected_requests_by_job = {
                job_index: sum(counts) + config.rounds + 1
                for job_index, counts in planned.items()
            }
            self.assertEqual(
                aggregate["expected_requests_by_job"],
                expected_requests_by_job,
            )
            self.assertEqual(
                aggregate["expected_requests"],
                sum(expected_requests_by_job.values()),
            )
            self.assertEqual(
                aggregate["successful_requests"],
                aggregate["expected_requests"],
            )
            round_counts = {
                str(job_index): [
                    row["sampled_subagent_requests"]
                    for row in aggregate["round_metrics"]
                    if row["job_index"] == int(job_index)
                ]
                for job_index in planned
            }
            self.assertEqual(
                round_counts,
                {
                    job_index: list(counts)
                    for job_index, counts in planned.items()
                },
            )
            self.assertEqual(
                {
                    str(row["job_index"]): tuple(
                        row["planned_subagent_requests_per_round"]
                    )
                    for row in aggregate["job_metrics"]
                },
                planned,
            )

    def test_http_failure_is_recorded_before_run_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(fail_sample_number=2)

            with self.assertRaisesRegex(run.BenchmarkFailed, "model_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=service.get,
                        post_fn=service.post,
                    )
                )

            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "model_failed")
            self.assertEqual(aggregate["status_counts"]["model_failed"], 1)
            rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                sum(row["status"] == "model_failed" for row in rows),
                1,
            )
            round_rows = [
                json.loads(line)
                for line in (output_dir / "round_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(round_rows), 1)
            round_row = round_rows[0]
            self.assertEqual(round_row["status"], "model_failed")
            self.assertEqual(
                round_row["subagent_prompt_tokens"],
                sum(int(row["actual_prompt_tokens"] or 0) for row in rows),
            )
            self.assertEqual(
                round_row["subagent_completion_tokens"],
                sum(
                    int(row["actual_completion_tokens"] or 0)
                    for row in rows
                ),
            )
            self.assertGreater(round_row["subagent_latency_p50_s"], 0.0)
            self.assertGreaterEqual(
                round_row["subagent_latency_max_s"],
                round_row["subagent_latency_p95_s"],
            )
            self.assertGreaterEqual(round_row["straggler_gap_s"], 0.0)
            self.assertIsNone(round_row["main_agent_latency_s"])
            self.assertIsNone(round_row["main_agent_prompt_tokens"])

    def test_main_failure_keeps_attempted_subagent_barrier_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(fail_sample_number=4)

            with self.assertRaisesRegex(run.BenchmarkFailed, "model_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=service.get,
                        post_fn=service.post,
                    )
                )

            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["successful_requests"], 3)
            self.assertEqual(aggregate["rounds_attempted"], 1)
            self.assertEqual(aggregate["rounds_completed"], 0)
            self.assertGreater(
                aggregate["phase_metrics"]["subagent"]["elapsed_s"],
                0.0,
            )
            round_rows = [
                json.loads(line)
                for line in (output_dir / "round_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(round_rows), 1)
            self.assertEqual(round_rows[0]["status"], "model_failed")
            self.assertGreater(round_rows[0]["subagent_barrier_s"], 0.0)
            self.assertGreater(
                round_rows[0]["subagent_prompt_tokens"],
                0,
            )
            self.assertGreater(
                round_rows[0]["subagent_completion_tokens"],
                0,
            )
            self.assertGreater(
                round_rows[0]["subagent_latency_p50_s"],
                0.0,
            )
            self.assertGreaterEqual(
                round_rows[0]["subagent_latency_max_s"],
                round_rows[0]["subagent_latency_p95_s"],
            )
            self.assertIsNone(
                round_rows[0]["main_agent_prompt_tokens"]
            )
            self.assertIsNone(
                round_rows[0]["main_agent_completion_tokens"]
            )
            self.assertEqual(
                round_rows[0][
                    "main_agent_expected_reusable_prefix_tokens"
                ],
                0,
            )
            self.assertGreater(
                round_rows[0]["main_agent_latency_s"],
                0.0,
            )

    def test_non_json_http_error_is_model_failure(self):
        async def fail_post(_url, _payload, _timeout_s):
            return run.HttpResponse(
                status=502,
                headers={},
                body=b"<html>bad gateway</html>",
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "model_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=FakeService().get,
                        post_fn=fail_post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "model_failed")
            self.assertEqual(aggregate["status_counts"]["model_failed"], 3)

    def test_malformed_choice_text_is_parse_failure(self):
        async def malformed_post(_url, payload, _timeout_s):
            prompt_tokens = len(payload["prompt"])
            completion_tokens = payload["max_tokens"]
            return _json_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": None,
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
                headers={
                    "x-sparseengine-worker": "http://snap-worker",
                    "x-sparseengine-route-reason": "test",
                    "x-sparseengine-sparse-method": "snapkv",
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "parse_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=FakeService().get,
                        post_fn=malformed_post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "parse_failed")
            self.assertEqual(aggregate["status_counts"]["parse_failed"], 3)

    def test_malformed_usage_is_parse_failure(self):
        async def malformed_post(_url, payload, _timeout_s):
            return _json_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "synthetic output",
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": None,
                        "completion_tokens": payload["max_tokens"],
                    },
                },
                headers={
                    "x-sparseengine-worker": "http://snap-worker",
                    "x-sparseengine-route-reason": "test",
                    "x-sparseengine-sparse-method": "snapkv",
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "parse_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=FakeService().get,
                        post_fn=malformed_post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "parse_failed")
            self.assertEqual(aggregate["status_counts"]["parse_failed"], 3)

    def test_preflight_malformed_json_is_parse_failure(self):
        async def malformed_get(url, timeout_s):
            if url.endswith("/models"):
                return run.HttpResponse(
                    status=200,
                    headers={},
                    body=b"not-json",
                )
            return await FakeService().get(url, timeout_s)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "parse_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=malformed_get,
                        post_fn=FakeService().post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "parse_failed")

    def test_reusable_prefix_uses_actual_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "synthetic_token_id_low": 7,
                    "synthetic_token_id_high": 7,
                }
            )
            service = FakeService()
            asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )
            main_payloads = [
                payload
                for payload in service.payloads
                if payload["sengine_method_preference"] == "omnikv,vanilla"
            ]
            rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            main_rows = [
                row for row in rows if row["phase"] != "subagent"
            ]
            expected = [
                0,
                min(
                    len(main_payloads[0]["prompt"]),
                    len(main_payloads[1]["prompt"]),
                ),
                min(
                    len(main_payloads[1]["prompt"]),
                    len(main_payloads[2]["prompt"]),
                ),
            ]
            self.assertEqual(
                [
                    row["expected_reusable_prefix_tokens"]
                    for row in main_rows
                ],
                expected,
            )

    def test_older_same_worker_prompt_can_define_larger_expected_hit(self):
        current = (1, 2, 3, 4, 9, 9)
        history = {
            "worker-a": [
                (1, 2, 3, 4, 8, 8),
                (1, 2, 7, 7),
            ]
        }

        raw, aligned, prior_count = (
            run._same_worker_prefix_expectation(
                current,
                "worker-a",
                history,
                2,
            )
        )

        self.assertEqual(raw, 4)
        self.assertEqual(aligned, 4)
        self.assertEqual(prior_count, 2)

    def test_different_worker_prompt_does_not_raise_expected_hit(self):
        current = (1, 2, 3, 4, 9, 9)
        history = {
            "worker-a": [(1, 2, 3, 4, 8, 8)],
            "worker-b": [(1, 2, 7, 7)],
        }

        raw, aligned, prior_count = (
            run._same_worker_prefix_expectation(
                current,
                "worker-b",
                history,
                2,
            )
        )

        self.assertEqual(raw, 2)
        self.assertEqual(aligned, 2)
        self.assertEqual(prior_count, 1)

    def test_exact_block_prior_final_block_is_reusable(self):
        current = (1, 2, 3, 4, 9)
        history = {
            "worker-a": [(1, 2, 3, 4)],
        }

        raw, aligned, prior_count = (
            run._same_worker_prefix_expectation(
                current,
                "worker-a",
                history,
                2,
            )
        )

        self.assertEqual(raw, 4)
        self.assertEqual(aligned, 4)
        self.assertEqual(prior_count, 1)

    def test_single_evicted_expected_prefix_is_recorded_without_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(
                miss_first_reusable_main_prefix=True,
            )

            aggregate = asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )

            self.assertEqual(aggregate["status"], "success")
            self.assertEqual(
                aggregate["prefix_cache"]["unexpected_zero_hit_requests"],
                1,
            )
            self.assertGreater(
                aggregate["prefix_cache"]["actual_hit_requests"],
                0,
            )

    def test_no_actual_hit_for_any_expected_prefix_is_metric_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(zero_main_prefix_match=True)

            with self.assertRaisesRegex(
                run.BenchmarkFailed,
                "without a verified main-agent prefix-cache hit",
            ):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=service.get,
                        post_fn=service.post,
                    )
                )

            rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all(row["status"] == "success" for row in rows))
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "metric_failed")
            self.assertGreater(
                aggregate["prefix_cache"]["unexpected_zero_hit_requests"],
                0,
            )

    def test_partial_actual_prefix_hit_is_recorded_without_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(partial_main_prefix_match=True)

            aggregate = asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )

            self.assertEqual(aggregate["status"], "success")
            self.assertEqual(
                aggregate["prefix_cache"]["partial_hit_requests"],
                1,
            )
            self.assertEqual(
                aggregate["prefix_cache"]["unexpected_zero_hit_requests"],
                0,
            )

    def test_sub_block_prefix_allows_zero_actual_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "main_overhead_tokens": 1,
                }
            )
            service = FakeService()

            aggregate = asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )

            self.assertEqual(aggregate["status"], "success")
            rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            second_round = next(
                row
                for row in rows
                if row["sample_id"] == "job-0000-round-01-main-agent"
            )
            self.assertEqual(
                second_round["expected_reusable_prefix_tokens"],
                1,
            )
            self.assertEqual(
                second_round[
                    "block_aligned_expected_reusable_prefix_tokens"
                ],
                0,
            )
            self.assertEqual(
                second_round["actual_prefix_matched_tokens"],
                0,
            )

    def test_router_run_requires_a_verified_main_prefix_hit(self):
        class SpreadMainRequestsService(FakeService):
            def __init__(self):
                super().__init__()
                self.main_workers = [
                    "http://main-worker-0",
                    "http://main-worker-1",
                    "http://main-worker-2",
                ]
                self.main_request_index = 0

            async def get(self, url, timeout_s):
                response = await super().get(url, timeout_s)
                if url.endswith("/health"):
                    payload = json.loads(response.body)
                    payload["healthy_workers"] = [
                        "http://snap-worker",
                        *self.main_workers,
                    ]
                    return _json_response(payload)
                return response

            async def post(self, url, payload, timeout_s):
                response = await super().post(url, payload, timeout_s)
                if payload["sengine_method_preference"] == "snapkv":
                    return response
                headers = dict(response.headers)
                headers["x-sparseengine-worker"] = self.main_workers[
                    self.main_request_index
                ]
                headers["x-sparseengine-prefix-matched-tokens"] = "0"
                self.main_request_index += 1
                return run.HttpResponse(
                    status=response.status,
                    headers=headers,
                    body=response.body,
                )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = SpreadMainRequestsService()

            with self.assertRaisesRegex(
                run.BenchmarkFailed,
                "verified main-agent prefix-cache hit",
            ):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=service.get,
                        post_fn=service.post,
                    )
                )

            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "metric_failed")
            self.assertEqual(
                aggregate["prefix_cache"]["expected_hit_requests"],
                0,
            )
            self.assertEqual(
                aggregate["prefix_cache"]["actual_hit_requests"],
                0,
            )

    def test_block_multiple_prefix_retains_full_block_expectation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "main_overhead_tokens": 2,
                }
            )
            service = FakeService()

            aggregate = asyncio.run(
                run.run_benchmark(
                    config,
                    get_fn=service.get,
                    post_fn=service.post,
                )
            )

            self.assertEqual(aggregate["status"], "success")
            rows = [
                json.loads(line)
                for line in (output_dir / "per_sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            second_round = next(
                row
                for row in rows
                if row["sample_id"] == "job-0000-round-01-main-agent"
            )
            self.assertEqual(
                second_round["expected_reusable_prefix_tokens"],
                2,
            )
            self.assertEqual(
                second_round[
                    "block_aligned_expected_reusable_prefix_tokens"
                ],
                2,
            )
            self.assertEqual(
                second_round["actual_prefix_matched_tokens"],
                2,
            )

    def test_first_main_hit_is_rejected_as_stale_cache_contamination(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(
                contaminate_first_main_prefix_match=True
            )

            with self.assertRaisesRegex(
                run.BenchmarkFailed,
                "stale-cache contamination",
            ):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=service.get,
                        post_fn=service.post,
                    )
                )

            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                aggregate["prefix_cache"][
                    "unexpected_excess_hit_requests"
                ],
                1,
            )

    def test_invalid_prefix_match_header_is_metric_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(invalid_prefix_match_header=True)

            with self.assertRaisesRegex(
                run.BenchmarkFailed,
                "invalid x-sparseengine-prefix-matched-tokens",
            ):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=service.get,
                        post_fn=service.post,
                    )
                )

            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status_counts"]["metric_failed"], 3)
            self.assertEqual(
                aggregate["prefix_cache"]["observed_requests"],
                0,
            )

    def test_default_http_executor_reaches_configured_concurrency(self):
        concurrency = 40
        barrier = threading.Barrier(concurrency)
        lock = threading.Lock()
        active = 0
        max_active = 0
        previous_main_prompt = None
        prefix_cache_block_size = 2

        def fake_request(_url, *, payload, timeout_s):
            nonlocal active, max_active, previous_main_prompt
            self.assertGreater(timeout_s, 0)
            methods = payload["sengine_method_preference"].split(",")
            if methods == ["snapkv"]:
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    barrier.wait(timeout=5)
                finally:
                    with lock:
                        active -= 1
                worker = "http://snap-worker"
                method = "snapkv"
            else:
                worker = "http://main-worker"
                method = "omnikv"
            matched_tokens = 0
            if methods != ["snapkv"]:
                prompt = tuple(payload["prompt"])
                matched_tokens = run._common_prefix_tokens(
                    previous_main_prompt,
                    prompt,
                )
                matched_tokens = (
                    matched_tokens
                    // prefix_cache_block_size
                    * prefix_cache_block_size
                )
                previous_main_prompt = prompt
            prompt_tokens = len(payload["prompt"])
            completion_tokens = payload["max_tokens"]
            return _json_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "synthetic output",
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
                headers={
                    "x-sparseengine-worker": worker,
                    "x-sparseengine-route-reason": "test",
                    "x-sparseengine-sparse-method": method,
                    "x-sparseengine-prefix-matched-tokens": str(
                        matched_tokens
                    ),
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "rounds": 1,
                    "articles_per_round": concurrency,
                }
            )
            with patch.object(run, "_request_bytes", side_effect=fake_request):
                aggregate = asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=FakeService().get,
                    )
                )
            self.assertEqual(aggregate["status"], "success")
            self.assertEqual(max_active, concurrency)

    def test_wrong_subagent_method_is_metric_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            service = FakeService(wrong_subagent_method=True)

            with self.assertRaisesRegex(run.BenchmarkFailed, "metric_failed"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=service.get,
                        post_fn=service.post,
                    )
                )

            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "metric_failed")
            self.assertEqual(aggregate["status_counts"]["metric_failed"], 3)

    def test_fails_when_router_has_only_one_worker(self):
        async def one_worker_get(url, _timeout_s):
            if url.endswith("/models"):
                return _json_response(
                    {
                        "data": [
                            {
                                "id": "sim-model",
                                "owned_by": "sparseengine-router",
                                "max_model_len": 65536,
                            }
                        ]
                    }
                )
            if url.endswith("/v1/worker/info"):
                return _json_response(
                    {
                        "served_model_name": "sim-model",
                        "max_model_len": 65_536,
                        "vocab_size": 32_000,
                        "sparse_method": "snapkv",
                        "prefix_cache_enabled": False,
                        "code_revision": CODE_REVISION,
                        "benchmark_config": {},
                    }
                )
            return _json_response(
                {
                    "status": "ok",
                    "healthy_workers": ["http://only-worker"],
                    "router_policy": {
                        "request_timeout_s": 900.0,
                        "control_timeout_s": 5.0,
                        "overload_load_factor": 1.0,
                        "load_abs_threshold": 0,
                        "profiles": {},
                        "code_revision": CODE_REVISION,
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)
            with self.assertRaisesRegex(run.BenchmarkFailed, "invalid_input"):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=one_worker_get,
                        post_fn=FakeService().post,
                    )
                )
            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "invalid_input")

    def test_preflight_ignores_healthy_workers_for_other_models(self):
        service = FakeService()

        async def multi_model_get(url, timeout_s):
            if url.endswith("/health"):
                response = await service.get(url, timeout_s)
                payload = json.loads(response.body)
                payload["healthy_workers"].append("http://other-worker")
                return _json_response(payload)
            if url == "http://other-worker/v1/worker/info":
                return _json_response(
                    {
                        "served_model_name": "other-model",
                        "vocab_size": 32_000,
                        "benchmark_config": {},
                    }
                )
            return await service.get(url, timeout_s)

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            preflight_info = asyncio.run(
                run.preflight(config, get_fn=multi_model_get)
            )

        self.assertEqual(
            [worker["url"] for worker in preflight_info["workers"]],
            ["http://snap-worker", "http://main-worker"],
        )

    def test_preflight_ignores_unrelated_method_capacity_and_vocab(self):
        service = FakeService()

        async def heterogeneous_get(url, timeout_s):
            if url.endswith("/models"):
                response = await service.get(url, timeout_s)
                payload = json.loads(response.body)
                payload["data"][0]["max_model_len"] = 8
                return _json_response(payload)
            if url.endswith("/health"):
                response = await service.get(url, timeout_s)
                payload = json.loads(response.body)
                payload["healthy_workers"].append("http://quest-worker")
                return _json_response(payload)
            if url == "http://quest-worker/v1/worker/info":
                return _json_response(
                    {
                        "served_model_name": "sim-model",
                        "sparse_method": "quest",
                        "prefix_cache_enabled": False,
                    }
                )
            return await service.get(url, timeout_s)

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            preflight_info = asyncio.run(
                run.preflight(config, get_fn=heterogeneous_get)
            )

        self.assertEqual(preflight_info["required_model_len"], 29)
        self.assertEqual(
            [worker["url"] for worker in preflight_info["workers"]],
            [
                "http://snap-worker",
                "http://main-worker",
                "http://quest-worker",
            ],
        )

    def test_preflight_rejects_insufficient_client_timeout_margin(self):
        service = FakeService()

        async def insufficient_margin_get(url, timeout_s):
            response = await service.get(url, timeout_s)
            if url.endswith("/health"):
                payload = json.loads(response.body)
                payload["router_policy"]["request_timeout_s"] = 920.0
                return _json_response(payload)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with self.assertRaisesRegex(
                ValueError,
                "must exceed the router upstream timeout",
            ):
                asyncio.run(
                    run.preflight(config, get_fn=insufficient_margin_get)
                )

    def test_preflight_rejects_missing_main_agent_method(self):
        service = FakeService()

        async def snapkv_only_get(url, timeout_s):
            response = await service.get(url, timeout_s)
            if url.endswith("/v1/worker/info"):
                payload = json.loads(response.body)
                payload["sparse_method"] = "snapkv"
                return _json_response(payload)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with self.assertRaisesRegex(
                ValueError,
                "configured main agent methods",
            ):
                asyncio.run(
                    run.preflight(config, get_fn=snapkv_only_get)
                )

    def test_preflight_requires_prefix_cache_on_main_candidates(self):
        service = FakeService()

        async def no_prefix_cache_get(url, timeout_s):
            response = await service.get(url, timeout_s)
            if url == "http://main-worker/v1/worker/info":
                payload = json.loads(response.body)
                payload["prefix_cache_enabled"] = False
                payload["benchmark_config"]["enable_prefix_caching"] = False
                return _json_response(payload)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with self.assertRaisesRegex(
                ValueError,
                "must enable prefix caching",
            ):
                asyncio.run(
                    run.preflight(config, get_fn=no_prefix_cache_get)
                )

    def test_preflight_requires_main_worker_prefix_cache_block_size(self):
        service = FakeService()

        async def missing_block_size_get(url, timeout_s):
            response = await service.get(url, timeout_s)
            if url == "http://main-worker/v1/worker/info":
                payload = json.loads(response.body)
                payload.pop("prefix_cache_block_size")
                return _json_response(payload)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with self.assertRaisesRegex(
                run.PreflightParseError,
                "positive integer prefix_cache_block_size",
            ):
                asyncio.run(
                    run.preflight(
                        config,
                        get_fn=missing_block_size_get,
                    )
                )

    def test_preflight_accepts_guaranteed_reusable_prefix_block_size(self):
        service = FakeService()

        with tempfile.TemporaryDirectory() as tmp:
            config = run.BenchmarkConfig(
                base_url="http://router.test/v1",
                model="sim-model",
                output_dir=Path(tmp) / "run",
            )
            service.prefix_cache_block_size = 4_736
            service.max_model_len = run.required_model_len(config)

            info = asyncio.run(
                run.preflight(config, get_fn=service.get)
            )

        self.assertEqual(
            info["worker_prefix_cache_block_sizes"][
                "http://main-worker"
            ],
            4_736,
        )

    def test_preflight_rejects_dirty_router_revision(self):
        service = FakeService()

        async def dirty_router_get(url, timeout_s):
            response = await service.get(url, timeout_s)
            if url.endswith("/health"):
                payload = json.loads(response.body)
                payload["router_policy"]["code_revision"] = {
                    **CODE_REVISION,
                    "git_dirty": True,
                }
                return _json_response(payload)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with self.assertRaisesRegex(
                ValueError,
                "dirty Git worktree",
            ):
                asyncio.run(
                    run.preflight(config, get_fn=dirty_router_get)
                )

    def test_preflight_rejects_dirty_worker_revision(self):
        service = FakeService()

        async def dirty_worker_get(url, timeout_s):
            response = await service.get(url, timeout_s)
            if url == "http://main-worker/v1/worker/info":
                payload = json.loads(response.body)
                payload["code_revision"] = {
                    **CODE_REVISION,
                    "git_dirty": True,
                }
                return _json_response(payload)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with self.assertRaisesRegex(
                ValueError,
                "dirty Git worktree",
            ):
                asyncio.run(
                    run.preflight(config, get_fn=dirty_worker_get)
                )

    def test_installed_package_revision_may_have_unknown_git_dirty(self):
        run._validate_code_revision(
            {
                "git_commit": None,
                "git_branch": None,
                "git_dirty": None,
                "package_version": "1.2.3",
            },
            "installed package",
        )

    def test_source_revision_requires_explicit_clean_git_state(self):
        with self.assertRaisesRegex(
            run.PreflightParseError,
            "git_dirty must be false when git_commit is present",
        ):
            run._validate_code_revision(
                {
                    "git_commit": "abc123",
                    "git_branch": "main",
                    "git_dirty": None,
                    "package_version": "0.1.0",
                },
                "source revision",
            )

    def test_client_dirty_patch_is_archived_with_hash(self):
        patch_text = "diff --git a/a.py b/a.py\n"

        def git_command(*args):
            outputs = {
                ("rev-parse", "HEAD"): "abc123\n",
                ("branch", "--show-current"): "feature\n",
                (
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ): " M a.py\n",
                ("diff", "--binary", "HEAD", "--"): patch_text,
            }
            return run.subprocess.CompletedProcess(
                ["git", *args],
                0,
                outputs[args],
                "",
            )

        with patch.object(run, "_git_command", side_effect=git_command):
            state, captured_patch = run._capture_client_code_state()

        self.assertEqual(captured_patch, patch_text)
        self.assertTrue(state["git_dirty"])
        self.assertEqual(
            state["worktree_patch"]["path"],
            "client_worktree.patch",
        )
        self.assertEqual(
            state["worktree_patch"]["sha256"],
            run.hashlib.sha256(patch_text.encode("utf-8")).hexdigest(),
        )

    def test_benchmark_fails_when_client_git_status_cannot_be_inspected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = self._config(output_dir)

            with (
                patch.object(run, "_git_command", return_value=None),
                self.assertRaisesRegex(
                    run.BenchmarkFailed,
                    "Cannot inspect the client source-tree Git status",
                ),
            ):
                asyncio.run(
                    run.run_benchmark(
                        config,
                        get_fn=FakeService().get,
                        post_fn=FakeService().post,
                    )
                )

            aggregate = json.loads(
                (output_dir / "aggregate_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            run_info = json.loads(
                (output_dir / "run_info.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["status"], "invalid_input")
            self.assertIn(
                "Cannot inspect the client source-tree Git status",
                aggregate["error"],
            )
            self.assertIsNone(run_info["git_dirty"])

    def test_preflight_rejects_synthetic_ids_outside_vocab(self):
        service = FakeService()

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "synthetic_token_id_high": 32_000,
                }
            )
            with self.assertRaisesRegex(
                ValueError,
                "Synthetic token range exceeds",
            ):
                asyncio.run(
                    run.preflight(config, get_fn=service.get)
                )

    def test_git_dirty_is_unknown_when_probe_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            with patch.object(run, "_git_command", return_value=None):
                run_info = run._run_info(
                    config,
                    None,
                    status="running",
                )

        self.assertIsNone(run_info["git_dirty"])

    def test_direct_server_payload_omits_router_only_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp) / "run")
            config = run.BenchmarkConfig(
                **{
                    **config.__dict__,
                    "require_router": False,
                }
            )
            spec = run.RequestSpec(
                sample_id="direct",
                job_index=0,
                phase="subagent",
                round_index=0,
                request_index=0,
                prompt_tokens=8,
                completion_tokens=2,
                prompt_seed=1,
                method_preferences=("snapkv",),
            )
            payload = run.build_payload(spec, config)
            self.assertNotIn("sengine_method_preference", payload)


if __name__ == "__main__":
    unittest.main()
