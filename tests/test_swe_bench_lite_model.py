from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest


class FakeFormatError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.messages = (message,) if isinstance(message, dict) else ()


class FakeAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _response(
    chain_id,
    *,
    content="answer",
    reasoning_content=None,
    tool_calls=None,
    query_error=None,
    finish_reason="stop",
    chain_status="created",
):
    message = SimpleNamespace(
        model_dump=lambda mode="json": {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "provider_specific_fields": {"chain_id": chain_id},
        }
    )
    response = SimpleNamespace(
        chain_id=chain_id,
        chain_status=chain_status,
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )
        ],
        query_error=query_error,
    )
    response.model_dump = lambda mode="json": {
        "choices": [{"message": message.model_dump(), "finish_reason": finish_reason}],
        "chain_id": chain_id,
        "chain_status": chain_status,
    }
    return response


def _load_model_module(monkeypatch, responses):
    class FakeLitellmModel:
        def __init__(self, **kwargs):
            del kwargs
            self.config = SimpleNamespace(
                model_name="openai/test-model",
                model_kwargs={
                    "api_base": "http://127.0.0.1:18000/v1",
                    "extra_body": {"thinking": {"type": "disabled"}},
                },
            )
            self.calls = []

        def query(self, messages, **kwargs):
            response = self._query(
                self._prepare_messages_for_api(messages),
                **kwargs,
            )
            query_error = getattr(response, "query_error", None)
            if query_error is not None:
                if isinstance(query_error, FakeFormatError) and query_error.messages:
                    query_error.messages[0].setdefault("extra", {})["response"] = response.model_dump()
                raise query_error
            return {
                "role": "assistant",
                "content": "answer",
                "extra": {},
            }

        def _query(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def _prepare_messages_for_api(self, messages):
            return messages

    minisweagent = ModuleType("minisweagent")
    models = ModuleType("minisweagent.models")
    exceptions = ModuleType("minisweagent.exceptions")
    litellm_model = ModuleType("minisweagent.models.litellm_model")
    exceptions.FormatError = FakeFormatError
    litellm_model.BASH_TOOL = {
        "type": "function",
        "function": {"name": "bash", "parameters": {"type": "object"}},
    }
    litellm_model.LitellmModel = FakeLitellmModel
    monkeypatch.setitem(sys.modules, "minisweagent", minisweagent)
    monkeypatch.setitem(sys.modules, "minisweagent.models", models)
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions)
    monkeypatch.setitem(
        sys.modules,
        "minisweagent.models.litellm_model",
        litellm_model,
    )
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "swe_bench_lite"
        / "model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_sparseengine_swe_model",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chain_model_creates_then_resumes_one_chain(monkeypatch):
    responses = [
        _response(
            "chain-a",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"true"}',
                    },
                }
            ],
        ),
        _response("chain-a", content="done"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()

    first_messages = [{"role": "user", "content": "first"}]
    assistant = {
        "role": "assistant",
        "content": "answer",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"true"}',
                },
            }
        ],
        "provider_specific_fields": {"chain_id": "chain-a"},
    }
    model.query(first_messages)
    model.query(
        [
            *first_messages,
            assistant,
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "ok",
            },
        ]
    )

    first_extra = model.calls[0][1]["extra_body"]
    second_extra = model.calls[1][1]["extra_body"]
    assert first_extra == {
        "thinking": {"type": "disabled"},
        "chain_id": None,
        "preserve_thinking": True,
    }
    assert second_extra == {
        "thinking": {"type": "disabled"},
        "chain_id": "chain-a",
        "preserve_thinking": True,
        "chain_append_start": 2,
    }


def test_chain_model_fails_when_server_omits_chain_id(monkeypatch):
    module = _load_model_module(
        monkeypatch,
        [SimpleNamespace(chain_id=None, model_extra={}, choices=[])],
    )
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "true")
    model = module.SparseVLLMLitellmModel()

    with pytest.raises(RuntimeError, match="without a chain_id"):
        model.query([{"role": "user", "content": "first"}])


@pytest.mark.parametrize("source", ["config", "request"])
@pytest.mark.parametrize("extra_body", [
    {"preserve_thinking": False},
    {"chat_template_kwargs": {"preserve_thinking": False}},
    {"chat_template_kwargs": {"clear_thinking": True}},
])
def test_chain_rejects_reasoning_removal_before_api_call(monkeypatch, source, extra_body):
    """Extra YAML and per-query overrides cannot bypass the runner's validation."""
    module = _load_model_module(monkeypatch, [])
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    kwargs = {}
    if source == "config":
        model.config.model_kwargs["extra_body"] = extra_body
    else:
        kwargs["extra_body"] = extra_body
    with pytest.raises(ValueError, match="requires preserved thinking"):
        model._query([{"role": "user", "content": "first"}], **kwargs)
    assert not model.calls


def test_chain_runner_config_matches_adapter_thinking_parameters(monkeypatch, tmp_path):
    """Generated config and transmitted template controls describe the same run."""
    import yaml
    from benchmark.swe_bench_lite.run import (
        SweBenchLiteRunner, build_parser, render_mini_config,
    )

    args = build_parser().parse_args([
        "--stage", "summarize", "--run-dir", str(tmp_path), "--chain-cache",
    ])
    runner = SweBenchLiteRunner(args)
    config = yaml.safe_load(render_mini_config(
        step_limit=args.step_limit, cost_limit=args.cost_limit,
        wall_time_limit_seconds=args.wall_time_limit_seconds,
        cost_tracking=args.cost_tracking, max_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        preserve_thinking=runner.args.preserve_thinking, api_base=args.api_base,
    ))
    module = _load_model_module(monkeypatch, [_response("chain-a")])
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    model.config.model_kwargs = config["model"]["model_kwargs"]
    model._query([{"role": "user", "content": "first"}])
    sent = model.calls[0][1]["extra_body"]
    assert {k: v for k, v in sent.items() if k != "chain_id"} == config["model"]["model_kwargs"]["extra_body"]
    assert sent["chat_template_kwargs"]["clear_thinking"] is not sent["preserve_thinking"]

def test_non_chain_model_does_not_send_chain_id(monkeypatch):
    module = _load_model_module(
        monkeypatch,
        [SimpleNamespace(chain_id=None, choices=[])],
    )
    monkeypatch.delenv("SPARSEENGINE_CHAIN_CACHE", raising=False)
    model = module.SparseVLLMLitellmModel()

    model.query([{"role": "user", "content": "first"}])

    assert "extra_body" not in model.calls[0][1]


def test_non_chain_model_prunes_once_and_verifies_next_turn_reuse(
    monkeypatch,
    tmp_path,
):
    module = _load_model_module(
        monkeypatch,
        [SimpleNamespace(chain_id=None, choices=[]), SimpleNamespace(chain_id=None, choices=[])],
    )
    events = tmp_path / "prune.jsonl"
    monkeypatch.setenv("SPARSEENGINE_PREFIX_PRUNE_POLICY", "snapkv_global")
    monkeypatch.setenv("SPARSEENGINE_PREFIX_PRUNE_TRIGGER_TOKENS", "4096")
    monkeypatch.setenv("SPARSEENGINE_PREFIX_PRUNE_RANGE_START", "512")
    monkeypatch.setenv("SPARSEENGINE_PREFIX_PRUNE_RANGE_END", "4096")
    monkeypatch.setenv("SPARSEENGINE_PREFIX_PRUNE_KEEP_TOKENS", "1792")
    monkeypatch.setenv("SPARSEENGINE_PREFIX_PRUNE_EVENTS", str(events))
    model = module.SparseVLLMLitellmModel()
    matches = iter(
        [
            {"usable_tokens": 4096, "matched_tokens": 4096, "resident_kv_tokens": 4096},
            {"usable_tokens": 4096, "matched_tokens": 4096, "resident_kv_tokens": 2304},
            {"usable_tokens": 4200, "matched_tokens": 4096, "resident_kv_tokens": 2304},
        ]
    )
    monkeypatch.setattr(model, "_match_prefix", lambda _chat: next(matches))

    def request(method, path, body=None):
        if method == "POST":
            assert path == "/prefix_cache/prune"
            assert body["chat"]["model"] == "test-model"
            return {"prune_id": "job-1", "status": "queued"}
        assert path == "/prefix_cache/prune/job-1"
        return {
            "prune_id": "job-1",
            "status": "completed",
            "result": {"freed_device_slots": 1792, "quality_degraded": True},
        }

    monkeypatch.setattr(model, "_prefix_cache_request", request)
    first = [{"role": "user", "content": "first"}]
    model.query(first)
    model.query(
        [
            *first,
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ]
    )

    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["prune_completed", "reuse_verified"]
    assert all(row["freed_device_slots"] == 1792 for row in rows)


def test_chain_model_rejects_rewritten_history(monkeypatch):
    responses = [
        _response("chain-a", content="invalid response"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()

    model.query([{"role": "user", "content": "first"}])
    with pytest.raises(
        RuntimeError,
        match="previous assistant response changed",
    ):
        model.query(
            [
                {"role": "user", "content": "first"},
                {
                    "role": "user",
                    "content": "The prior response was invalid; retry.",
                },
            ]
        )

    assert model.calls[0][1]["extra_body"]["chain_id"] is None
    assert len(model.calls) == 1
    assert model._chain_id == "chain-a"


def test_chain_model_full_rerenders_after_length_finish(monkeypatch):
    responses = [
        _response("chain-a", finish_reason="length"),
        _response("chain-a", chain_status="resumed"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    first_messages = [{"role": "user", "content": "first"}]

    model.query(first_messages)
    model.query(
        [
            *first_messages,
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "continue"},
        ]
    )

    second_extra = model.calls[1][1]["extra_body"]
    assert second_extra["chain_id"] == "chain-a"
    assert "chain_append_start" not in second_extra


@pytest.mark.parametrize("chain_enabled", [False, True])
@pytest.mark.parametrize("content", [None, "partial answer"])
def test_length_format_error_retains_assistant_and_single_raw_response(
    monkeypatch, chain_enabled, content
):
    error = FakeFormatError({"role": "user", "content": "Output was cut off; retry.", "extra": {}})
    responses = [
        _response("chain-a", content=content, reasoning_content="unfinished reasoning",
                  finish_reason="length", query_error=error),
        _response("chain-a", chain_status="resumed"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1" if chain_enabled else "0")
    model = module.SparseVLLMLitellmModel()
    messages = [{"role": "user", "content": "first"}]

    with pytest.raises(FakeFormatError) as raised:
        model.query(messages)
    assert raised.value is error
    messages.extend(error.messages)  # DefaultAgent's actual recovery contract.
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[1]["reasoning_content"] == "unfinished reasoning"
    assert messages[1]["content"] == content
    assert sum("response" in message.get("extra", {}) for message in messages) == 1
    assert messages[2]["extra"]["truncated_response_preserved"] is True
    assert messages[2]["extra"]["response"]["choices"][0]["finish_reason"] == "length"
    model.query(messages)
    assert model.calls[1][0][1]["reasoning_content"] == "unfinished reasoning"
    if chain_enabled:
        extra = model.calls[1][1]["extra_body"]
        assert extra["chain_id"] == "chain-a"
        assert "chain_append_start" not in extra
        assert model._force_new_chain_reason is None


def test_length_recovery_rejects_dropping_preserved_response(monkeypatch):
    error = FakeFormatError({"role": "user", "content": "retry", "extra": {}})
    module = _load_model_module(monkeypatch, [
        _response("chain-a", content=None, reasoning_content="partial",
                  finish_reason="length", query_error=error),
    ])
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    first = [{"role": "user", "content": "first"}]
    with pytest.raises(FakeFormatError):
        model.query(first)
    with pytest.raises(RuntimeError, match="not append-only"):
        model.query([*first, error.messages[-1]])
    assert len(model.calls) == 1


def test_length_format_error_respects_server_invalidation(monkeypatch):
    error = FakeFormatError({"role": "user", "content": "retry", "extra": {}})
    module = _load_model_module(monkeypatch, [
        _response("chain-a", finish_reason="length", chain_status="invalidated",
                  query_error=error),
        _response("chain-b"),
    ])
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    first = [{"role": "user", "content": "first"}]
    with pytest.raises(FakeFormatError):
        model.query(first)
    model.query([*first, *error.messages])
    assert model.calls[1][1]["extra_body"]["chain_id"] is None


def test_length_format_error_does_not_insert_unmatched_tool_calls(monkeypatch):
    error = FakeFormatError({"role": "user", "content": "Invalid arguments; retry", "extra": {}})
    module = _load_model_module(monkeypatch, [
        _response("chain-a", finish_reason="length", query_error=error,
                  tool_calls=[{"id": "call-1", "type": "function",
                               "function": {"name": "bash", "arguments": '{"command":'}}]),
    ])
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    with pytest.raises(FakeFormatError):
        model.query([{"role": "user", "content": "first"}])
    assert len(error.messages) == 1
    assert error.messages[0]["role"] == "user"
    assert model._force_new_chain_reason == "format_error"


@pytest.mark.parametrize("chain_enabled", [False, True])
def test_length_format_error_records_unexecuted_tool_results(monkeypatch, chain_enabled):
    error = FakeFormatError({"role": "user", "content": "Missing command; retry", "extra": {}})
    calls = [{"id": f"call-{i}", "type": "function",
              "function": {"name": "bash", "arguments": "{}"}} for i in range(2)]
    module = _load_model_module(monkeypatch, [
        _response("chain-a", content=None, finish_reason="length", query_error=error,
                  tool_calls=calls),
        _response("chain-b", chain_status="recreated"),
    ])
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1" if chain_enabled else "0")
    model = module.SparseVLLMLitellmModel()
    first = [{"role": "user", "content": "first"}]
    with pytest.raises(FakeFormatError):
        model.query(first)
    assert [message["role"] for message in error.messages] == ["assistant", "tool", "tool", "user"]
    assert error.messages[0]["tool_calls"] == calls
    for call, result in zip(calls, error.messages[1:3]):
        assert result["tool_call_id"] == call["id"]
        assert "not executed" in result["content"]
    model.query([*first, *error.messages])
    if chain_enabled:
        assert model.calls[1][1]["extra_body"]["chain_id"] == "chain-a"
        assert "chain_append_start" not in model.calls[1][1]["extra_body"]
        assert model._chain_id == "chain-b"


def test_chain_model_starts_new_chain_after_invalidation(monkeypatch):
    responses = [
        _response("chain-a", chain_status="invalidated"),
        _response("chain-b"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    first_messages = [{"role": "user", "content": "first"}]

    model.query(first_messages)
    assert model._chain_id is None
    model.query(
        [
            *first_messages,
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "continue"},
        ]
    )

    assert model.calls[1][1]["extra_body"]["chain_id"] is None
    assert "chain_append_start" not in model.calls[1][1]["extra_body"]
    assert model._chain_id == "chain-b"


@pytest.mark.parametrize("error_source", ["worker", "router"])
def test_chain_model_recreates_evicted_chain_from_full_history(
    monkeypatch, error_source
):
    status_code = 410
    detail = {"code": "chain_gone"}
    if error_source == "router":
        fastapi = pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        from sparseengine.entrypoints.openai import smart_router

        router = smart_router.SmartRouter(
            worker_urls=["http://worker-a"],
            request_timeout_s=1.0,
            overload_load_factor=1.5,
            load_abs_threshold=1,
            profiles={},
            route_log_dir=None,
        )
        monkeypatch.setattr(
            smart_router,
            "_post_json",
            lambda *_args: {"present": False, "tombstone": True},
        )
        with pytest.raises(fastapi.HTTPException) as gone:
            asyncio.run(router._select_chain_owner(router.workers, "chain-a"))
        status_code = gone.value.status_code
        detail = gone.value.detail

    responses = [
        _response("chain-a"),
        FakeAPIError(
            status_code,
            f"Error code: {status_code} - {json.dumps({'detail': detail})}",
        ),
        _response("chain-b"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    first_messages = [{"role": "user", "content": "first"}]

    model.query(first_messages)
    recovered = model.query(
        [
            *first_messages,
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "continue"},
        ]
    )

    evicted_extra = model.calls[1][1]["extra_body"]
    recreated_extra = model.calls[2][1]["extra_body"]
    assert evicted_extra["chain_id"] == "chain-a"
    assert evicted_extra["chain_append_start"] == 2
    assert recreated_extra["chain_id"] is None
    assert "chain_append_start" not in recreated_extra
    assert len(model.calls) == 3
    assert model.calls[2][0] == model.calls[1][0]
    assert model._chain_id == "chain-b"
    assert recovered["extra"]["chain_reset_reason"] == "chain_gone"


def test_chain_model_commits_state_only_after_successful_query(monkeypatch):
    responses = [
        _response(
            "chain-a",
            content=None,
            query_error=FakeFormatError("missing tool call"),
        ),
        _response("chain-b", content="recovered"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEENGINE_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()

    first_messages = [{"role": "user", "content": "first"}]
    with pytest.raises(FakeFormatError, match="missing tool call"):
        model.query(first_messages)

    assert model._chain_id is None
    assert model._last_request_messages is None
    assert model._last_response_message is None
    assert model._force_new_chain_reason == "format_error"
    assert model._recovery_chain_id == "chain-a"

    recovered = model.query(
        [
            *first_messages,
            {
                "role": "user",
                "content": "The prior response had no tool call; retry.",
            },
        ]
    )

    assert model.calls[0][1]["extra_body"]["chain_id"] is None
    assert model.calls[1][1]["extra_body"] == {
        "thinking": {"type": "disabled"},
        "chain_id": "chain-a",
        "preserve_thinking": True,
    }
    assert model._chain_id == "chain-b"
    assert model._recovery_chain_id is None
    assert model._force_new_chain_reason is None
    assert recovered["extra"]["chain_reset_reason"] == "format_error"


@pytest.mark.parametrize("mismatch", [False, True])
def test_tool_result_pruning_passes_only_verified_token_ranges(monkeypatch, tmp_path, mismatch):
    module = _load_model_module(monkeypatch, [_response(None), _response(None)])
    for key, value in {
        "SPARSEENGINE_PREFIX_PRUNE_POLICY": "kvzip_global",
        "SPARSEENGINE_PREFIX_PRUNE_TARGET": "tool_results",
        "SPARSEENGINE_PREFIX_PRUNE_TOKENIZER": str(tmp_path),
        "SPARSEENGINE_PREFIX_PRUNE_KEEP_RATIO": "0.5",
        "SPARSEENGINE_PREFIX_PRUNE_TRIGGER_TOKENS": "8",
        "SPARSEENGINE_PREFIX_PRUNE_EVENTS": str(tmp_path / "events.jsonl"),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SPARSEENGINE_CHAIN_CACHE", raising=False)
    model = module.SparseVLLMLitellmModel()
    model._prune_tool_selector = SimpleNamespace(select=lambda *args, **kwargs: {
        "token_ids": list(range(11)), "ranges": [(1, 3), (5, 9)],
        "tool_tokens": 6, "eligible_tokens": 6,
    })
    full = {"block_size": 1, "prompt_tokens": 11, "usable_tokens": 10,
            "matched_tokens": 10, "resident_kv_tokens": 10, "last_block_id": "same-path"}
    matches = iter([full, {**full, "resident_kv_tokens": 7}, {**full, "resident_kv_tokens": 7}])
    monkeypatch.setattr(model, "_match_prefix", lambda _: next(matches))
    calls = []
    def request(method, path, body=None):
        calls.append((method, path, body))
        if path == "/prefix_cache/match":
            pruned = any(p == "/prefix_cache/prune" for _, p, _ in calls)
            return {**full, "resident_kv_tokens": 7 if pruned else 10,
                    "last_block_id": "wrong-path" if mismatch else "same-path"}
        if method == "POST":
            assert body["ranges"] == [(1, 3), (5, 9)]
            assert body["keep_tokens"] == 3
            assert "chat" not in body and "target" not in body
            return {"prune_id": "tools", "status": "queued"}
        return {"status": "completed", "result": {"freed_device_slots": 3, "quality_degraded": True}}
    monkeypatch.setattr(model, "_prefix_cache_request", request)
    messages = [{"role": "tool", "content": "result"}]
    if mismatch:
        with pytest.raises(RuntimeError, match="does not match"):
            model.query(messages)
        assert not any(path == "/prefix_cache/prune" for _, path, _ in calls)
    else:
        model.query(messages)
        model.query(messages)
        assert sum(path == "/prefix_cache/prune" for _, path, _ in calls) == 1
        records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in records] == ["prune_completed", "reuse_verified"]
        assert records[0]["ranges"] == [[1, 3], [5, 9]]


def test_tool_prune_advances_only_after_success_and_shares_new_turn_budget(monkeypatch, tmp_path):
    module = _load_model_module(monkeypatch, [])
    for key, value in {
        'SPARSEENGINE_PREFIX_PRUNE_POLICY': 'kvzip_global',
        'SPARSEENGINE_PREFIX_PRUNE_TARGET': 'tool_results',
        'SPARSEENGINE_PREFIX_PRUNE_TOKENIZER': str(tmp_path),
        'SPARSEENGINE_PREFIX_PRUNE_KEEP_RATIO': '0.2',
        'SPARSEENGINE_PREFIX_PRUNE_TRIGGER_TOKENS': '4096',
        'SPARSEENGINE_PREFIX_PRUNE_EVENTS': str(tmp_path/'events.jsonl'),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('SPARSEENGINE_CHAIN_CACHE', raising=False)
    model = module.SparseVLLMLitellmModel()
    state = {'dropped': 0, 'fail': False}
    selections, jobs = [], []

    def select(chat, *, message_start, **kwargs):
        selected = [i for i in range(message_start, len(chat['messages']))
                    if chat['messages'][i]['role'] == 'tool']
        selections.append(selected)
        ranges = [(i*10, (i+1)*10) for i in selected]
        return dict(token_ids=list(range(100)), ranges=ranges,
                    eligible_tokens=len(selected)*10, tool_tokens=len(selected)*10)

    def match(_):
        return dict(block_size=1, prompt_tokens=100, usable_tokens=100,
                    matched_tokens=100, resident_kv_tokens=100-state['dropped'], last_block_id='path')

    def request(method, path, body=None):
        if path == '/prefix_cache/match':
            return match(None)
        if path == '/prefix_cache/prune':
            jobs.append(body)
            if not state['fail']:
                state['freed'] = sum(r-l for l,r in body['ranges']) - body['keep_tokens']
                state['dropped'] += state['freed']
            return {'prune_id': 'job', 'status': 'queued'}
        if state['fail']:
            return {'status': 'failed'}
        return dict(status='completed', result=dict(freed_device_slots=state['freed'], quality_degraded=True))

    model._prune_tool_selector = SimpleNamespace(select=select)
    monkeypatch.setattr(model, '_match_prefix', match)
    monkeypatch.setattr(model, '_prefix_cache_request', request)
    messages = [{'role': 'user', 'content': 'keep'}]
    model._maybe_prune({'messages': messages})
    assert not selections and not jobs
    messages += [{'role': 'tool', 'content': 'one'}]
    model._maybe_prune({'messages': messages})
    messages += [{'role': 'assistant', 'content': 'think'},
                 {'role': 'tool', 'content': 'two'}, {'role': 'tool', 'content': 'three'}]
    state['fail'] = True
    with pytest.raises(RuntimeError, match='did not complete'):
        model._maybe_prune({'messages': messages})
    assert len(model._prune_processed_messages) == 2
    state['fail'] = False
    model._maybe_prune({'messages': messages})
    assert selections == [[1], [3,4], [3,4]]
    assert jobs[-1]['keep_tokens'] == 4  # Shared budget across two NEW bodies.
    assert state['dropped'] == 24  # First turn drops 8, second drops 16, never 8 again.
    model._maybe_prune({'messages': messages})  # Reuse verification only.
    assert len(jobs) == 3
    messages[1]['content'] = 'rewritten old result'
    with pytest.raises(RuntimeError, match='not append-only'):
        model._maybe_prune({'messages': messages})
