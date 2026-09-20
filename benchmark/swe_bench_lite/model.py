from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import BASH_TOOL
from minisweagent.models.litellm_model import LitellmModel


class SparseVLLMLitellmModel(LitellmModel):
    """Replay clean chat history and opt into per-instance SparseEngine chains."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chain_cache_enabled = os.getenv(
            "SPARSEENGINE_CHAIN_CACHE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._prune_policy = os.getenv("SPARSEENGINE_PREFIX_PRUNE_POLICY", "").strip()
        self._prune_finished = False
        self._prune_reuse_verified = False
        self._prune_freed_slots = 0
        if self._prune_policy:
            if self._prune_policy not in {"snapkv_global", "kvzip_global"}:
                raise ValueError(
                    "SPARSEENGINE_PREFIX_PRUNE_POLICY must be snapkv_global or "
                    f"kvzip_global, got {self._prune_policy!r}."
                )
            if self._chain_cache_enabled:
                raise ValueError("Prefix-tree pruning requires radix cache, not chain cache.")
            self._prune_range_start = self._required_env_int(
                "SPARSEENGINE_PREFIX_PRUNE_RANGE_START"
            )
            self._prune_range_end = self._required_env_int(
                "SPARSEENGINE_PREFIX_PRUNE_RANGE_END"
            )
            self._prune_keep_tokens = self._required_env_int(
                "SPARSEENGINE_PREFIX_PRUNE_KEEP_TOKENS"
            )
            self._prune_trigger_tokens = self._required_env_int(
                "SPARSEENGINE_PREFIX_PRUNE_TRIGGER_TOKENS"
            )
            if (
                self._prune_range_start < 0
                or self._prune_range_end <= self._prune_range_start
                or self._prune_range_start % 16
                or self._prune_range_end % 16
            ):
                raise ValueError("Prefix-prune range must be non-empty and 16-token aligned.")
            width = self._prune_range_end - self._prune_range_start
            if not 0 <= self._prune_keep_tokens < width:
                raise ValueError("Prefix-prune keep_tokens must be in [0, R-L).")
            if self._prune_trigger_tokens < self._prune_range_end:
                raise ValueError("Prefix-prune trigger must be at least range_end.")
            api_base = str(self.config.model_kwargs.get("api_base") or "").rstrip("/")
            if not api_base:
                raise ValueError("Prefix pruning requires model_kwargs.api_base.")
            self._prefix_cache_api_base = api_base
            model_name = str(self.config.model_name)
            self._served_model_name = (
                model_name.split("/", 1)[1] if model_name.startswith("openai/") else model_name
            )
            events = os.getenv("SPARSEENGINE_PREFIX_PRUNE_EVENTS", "").strip()
            if not events:
                raise ValueError("SPARSEENGINE_PREFIX_PRUNE_EVENTS is required.")
            self._prune_events_path = Path(events)
        self._chain_id: str | None = None
        self._last_request_messages: list[dict[str, Any]] | None = None
        self._last_response_message: dict[str, Any] | None = None
        self._last_response_append_safe = False
        self._recovery_chain_id: str | None = None
        self._pending_chain_state: tuple[
            str,
            list[dict[str, Any]],
            dict[str, Any],
            str | None,
            str,
            str,
        ] | None = None
        self._force_new_chain_reason: str | None = None

    @classmethod
    def _plain_value(cls, value: Any) -> Any:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            value = model_dump(mode="json")
        if isinstance(value, dict):
            return {
                str(key): cls._plain_value(item)
                for key, item in value.items()
                if key not in {"extra", "provider_specific_fields"}
                and item is not None
            }
        if isinstance(value, list):
            return [cls._plain_value(item) for item in value]
        return value

    @classmethod
    def _chain_message(cls, message: dict[str, Any]) -> dict[str, Any]:
        plain = cls._plain_value(message)
        return {
            key: plain[key]
            for key in (
                "role",
                "content",
                "reasoning_content",
                "tool_calls",
                "tool_call_id",
                "name",
            )
            if key in plain
        }

    @staticmethod
    def _value_digest(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @staticmethod
    def _required_env_int(name: str) -> int:
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(f"{name} is required when prefix pruning is enabled.")
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}.") from exc

    @staticmethod
    def _is_chain_gone_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        return status_code == 410 and "chain_gone" in str(exc)

    def _chat_selector(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        selector: dict[str, Any] = {
            "model": self._served_model_name,
            "messages": self._plain_value(messages),
            "tools": self._plain_value([BASH_TOOL]),
        }
        request_values = {**self.config.model_kwargs, **kwargs}
        for key in (
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "parallel_tool_calls",
            "tool_choice",
            "reasoning_effort",
            "chat_template_kwargs",
        ):
            value = request_values.get(key)
            if value is not None:
                selector[key] = self._plain_value(value)
        extra_body = request_values.get("extra_body") or {}
        if isinstance(extra_body, dict):
            selector.update(self._plain_value(extra_body))
        return selector

    def _prefix_cache_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self._prefix_cache_api_base + path,
            method=method,
            data=data,
            headers={
                "Authorization": "Bearer local-sparseengine",
                "Content-Type": "application/json",
            },
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=900) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Prefix-cache API failed: HTTP {exc.code} {method} {path}: {detail}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Prefix-cache API returned {type(payload).__name__}, expected object."
            )
        return payload

    def _record_prune_event(self, event: str, **values: Any) -> None:
        payload = {
            "event": event,
            "time": time.time(),
            "pid": os.getpid(),
            **values,
        }
        self._prune_events_path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        fd = os.open(
            self._prune_events_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(fd, line)
        finally:
            os.close(fd)

    def _match_prefix(self, chat: dict[str, Any]) -> dict[str, Any]:
        return self._prefix_cache_request(
            "POST",
            "/prefix_cache/match",
            {"chat": chat},
        )

    def _maybe_verify_prune_reuse(self, chat: dict[str, Any]) -> None:
        if not self._prune_finished or self._prune_reuse_verified:
            return
        match = self._match_prefix(chat)
        logical = int(match.get("matched_tokens") or 0)
        resident = int(match.get("resident_kv_tokens") or 0)
        if logical <= 0 or logical - resident < self._prune_freed_slots:
            raise RuntimeError(
                "Pruned prefix was not physically reused by the next MiniSWE turn: "
                f"matched={logical} resident={resident} expected_gap="
                f"{self._prune_freed_slots}."
            )
        self._prune_reuse_verified = True
        self._record_prune_event(
            "reuse_verified",
            policy=self._prune_policy,
            selector_digest=self._value_digest(chat["messages"][:2]),
            matched_tokens=logical,
            resident_kv_tokens=resident,
            freed_device_slots=self._prune_freed_slots,
        )

    def _maybe_prune(self, chat: dict[str, Any]) -> None:
        if self._prune_finished:
            return
        match_before = self._match_prefix(chat)
        usable = int(match_before.get("usable_tokens") or 0)
        if usable < self._prune_trigger_tokens:
            return
        matched = int(match_before.get("matched_tokens") or 0)
        if matched != usable:
            raise RuntimeError(
                "Completed MiniSWE turn is not fully available for pruning: "
                f"matched={matched} usable={usable}."
            )
        queued = self._prefix_cache_request(
            "POST",
            "/prefix_cache/prune",
            {
                "chat": chat,
                "range_start": self._prune_range_start,
                "range_end": self._prune_range_end,
                "keep_tokens": self._prune_keep_tokens,
                "policy": self._prune_policy,
                "observation_tokens": 64,
                "score_chunk_size": 1024,
                "prev_postfix_size": 32,
            },
        )
        prune_id = str(queued.get("prune_id") or "")
        if not prune_id:
            raise RuntimeError(f"Prefix prune returned no prune_id: {queued}.")
        status = queued
        for _ in range(9000):
            status = self._prefix_cache_request(
                "GET",
                f"/prefix_cache/prune/{prune_id}",
            )
            if status.get("status") in {"completed", "blocked", "failed"}:
                break
            time.sleep(0.1)
        if status.get("status") != "completed":
            raise RuntimeError(f"Prefix prune did not complete: {status}.")
        result = status.get("result") or {}
        freed = int(result.get("freed_device_slots") or 0)
        expected_freed = (
            self._prune_range_end
            - self._prune_range_start
            - self._prune_keep_tokens
        )
        if freed != expected_freed or result.get("quality_degraded") is not True:
            raise RuntimeError(
                "Prefix prune result violated physical accounting/tag contract: "
                f"freed={freed} expected={expected_freed} result={result}."
            )
        match_after = self._match_prefix(chat)
        after_matched = int(match_after.get("matched_tokens") or 0)
        after_resident = int(match_after.get("resident_kv_tokens") or 0)
        if after_matched != usable or after_matched - after_resident < freed:
            raise RuntimeError(
                "Prefix prune did not preserve the logical route or compact resident KV: "
                f"before={usable} after={after_matched} resident={after_resident}."
            )
        self._prune_finished = True
        self._prune_freed_slots = freed
        self._record_prune_event(
            "prune_completed",
            policy=self._prune_policy,
            selector_digest=self._value_digest(chat["messages"][:2]),
            prune_id=prune_id,
            usable_tokens=usable,
            resident_kv_tokens=after_resident,
            freed_device_slots=freed,
            range=[self._prune_range_start, self._prune_range_end],
            keep_tokens=self._prune_keep_tokens,
            quality_degraded=True,
        )

    def _continuation_error(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        if self._last_request_messages is None:
            return "missing previous request messages"
        if self._last_response_message is None:
            return "missing previous response message"
        current = [self._chain_message(message) for message in messages]
        previous = self._last_request_messages
        if len(current) <= len(previous):
            return (
                "message history did not grow: "
                f"current={len(current)} previous={len(previous)}"
            )
        for index, (expected, actual) in enumerate(
            zip(previous, current, strict=False)
        ):
            if expected == actual:
                continue
            differing_fields = sorted(
                key
                for key in set(expected) | set(actual)
                if expected.get(key) != actual.get(key)
            )
            return (
                "previous request prefix changed: "
                f"index={index} "
                f"expected_role={expected.get('role')!r} "
                f"actual_role={actual.get('role')!r} "
                f"differing_fields={differing_fields!r} "
                f"expected_digest={self._value_digest(expected)} "
                f"actual_digest={self._value_digest(actual)}"
            )
        appended_response = current[len(previous)]
        if appended_response != self._last_response_message:
            differing_fields = sorted(
                key
                for key in set(appended_response)
                | set(self._last_response_message)
                if appended_response.get(key)
                != self._last_response_message.get(key)
            )
            return (
                "previous assistant response changed: "
                f"index={len(previous)} "
                f"differing_fields={differing_fields!r} "
                "expected_digest="
                f"{self._value_digest(self._last_response_message)} "
                f"actual_digest={self._value_digest(appended_response)}"
            )
        return None

    def _query(self, messages: list[dict[str, str]], **kwargs):
        if not self._chain_cache_enabled:
            if not self._prune_policy:
                return super()._query(messages, **kwargs)
            chat = self._chat_selector(messages, kwargs)
            self._maybe_verify_prune_reuse(chat)
            response = super()._query(messages, **kwargs)
            self._maybe_prune(chat)
            return response

        self._pending_chain_state = None
        request_chain_id = self._recovery_chain_id or self._chain_id
        recovering_chain = self._force_new_chain_reason is not None
        if request_chain_id is not None and not recovering_chain:
            continuation_error = self._continuation_error(messages)
            if continuation_error is not None:
                raise RuntimeError(
                    "SparseEngine chain transcript is not append-only: "
                    f"{continuation_error}."
                )
        chain_append_start = (
            len(self._last_request_messages) + 1
            if request_chain_id is not None
            and not recovering_chain
            and self._last_response_append_safe
            and self._last_request_messages is not None
            else None
        )
        configured_extra_body = self.config.model_kwargs.get("extra_body") or {}
        request_extra_body = kwargs.pop("extra_body", None) or {}
        extra_body = {
            **configured_extra_body,
            **request_extra_body,
            "chain_id": request_chain_id,
        }
        template_kwargs = extra_body.get("chat_template_kwargs") or {}
        if (
            extra_body.get("preserve_thinking") is False
            or template_kwargs.get("preserve_thinking") is False
            or template_kwargs.get("clear_thinking") is True
        ):
            raise ValueError(
                "Chain cache requires preserved thinking for append-only prompts: "
                "preserve_thinking must not be false and clear_thinking must not be true."
            )
        extra_body["preserve_thinking"] = True
        if chain_append_start is not None:
            extra_body["chain_append_start"] = chain_append_start
        try:
            response = super()._query(
                messages,
                extra_body=extra_body,
                **kwargs,
            )
        except Exception as exc:
            if request_chain_id is None or not self._is_chain_gone_error(exc):
                raise
            self._chain_id = None
            self._recovery_chain_id = None
            self._force_new_chain_reason = "chain_gone"
            retry_extra_body = {**extra_body, "chain_id": None}
            retry_extra_body.pop("chain_append_start", None)
            response = super()._query(
                messages,
                extra_body=retry_extra_body,
                **kwargs,
            )
        chain_id = getattr(response, "chain_id", None)
        if chain_id is None:
            model_extra = getattr(response, "model_extra", None)
            if isinstance(model_extra, dict):
                chain_id = model_extra.get("chain_id")
        normalized_chain_id = str(chain_id or "").strip()
        if not normalized_chain_id:
            raise RuntimeError(
                "SparseEngine chain-cache request completed without a chain_id."
            )
        pending_request_messages = [
            self._chain_message(message) for message in messages
        ]
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError(
                "SparseEngine chain-cache response contained no choices."
            )
        response_message = getattr(choices[0], "message", None)
        if response_message is None:
            raise RuntimeError(
                "SparseEngine chain-cache response contained no assistant message."
            )
        pending_response_message = self._chain_message(response_message)
        finish_reason = str(
            getattr(choices[0], "finish_reason", "") or ""
        ).strip()
        if not finish_reason:
            raise RuntimeError(
                "SparseEngine chain-cache response omitted finish_reason."
            )
        chain_status = getattr(response, "chain_status", None)
        if chain_status is None:
            model_extra = getattr(response, "model_extra", None)
            if isinstance(model_extra, dict):
                chain_status = model_extra.get("chain_status")
        normalized_chain_status = str(chain_status or "").strip()
        if not normalized_chain_status:
            raise RuntimeError(
                "SparseEngine chain-cache response omitted chain_status."
            )
        self._pending_chain_state = (
            normalized_chain_id,
            pending_request_messages,
            pending_response_message,
            self._force_new_chain_reason,
            finish_reason,
            normalized_chain_status,
        )
        return response

    @classmethod
    def _preserve_truncated_response(cls, exc: FormatError) -> bool:
        # MiniSWE adds only the correction on FormatError. Keep the truncated
        # assistant turn too, so the next prompt does not discard cached output.
        corrections = getattr(exc, "messages", ())
        if not corrections or not isinstance(corrections[0], dict):
            return False
        response = corrections[0].get("extra", {}).get("response")
        if not isinstance(response, dict):
            return False
        choices = response.get("choices") or []
        if not choices or choices[0].get("finish_reason") != "length":
            return False
        assistant = choices[0].get("message") or {}
        if assistant.get("role") != "assistant":
            return False
        tool_errors = []
        for call in assistant.get("tool_calls") or []:
            # Only retain calls the server can render. Never execute partial
            # arguments; give every retained call an explicit failure result.
            try:
                arguments = json.loads(call["function"]["arguments"])
            except (KeyError, TypeError, json.JSONDecodeError):
                return False
            if not call.get("id") or not isinstance(arguments, dict):
                return False
            tool_errors.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": "Tool call was not executed because the response reached "
                "the output token limit. Retry with complete arguments.",
            })
        assistant = cls._chain_message(assistant)
        assistant.setdefault("content", None)
        corrections[0]["extra"]["truncated_response_preserved"] = True
        # Keep the raw response only on the correction, as upstream does, so
        # trajectory usage aggregation does not count this request twice.
        exc.messages = (assistant, *tool_errors, *corrections)
        return True

    def query(self, messages: list[dict[str, Any]], **kwargs):
        try:
            message = super().query(messages, **kwargs)
        except Exception as exc:
            preserved = (
                isinstance(exc, FormatError)
                and self._preserve_truncated_response(exc)
            )
            if not self._chain_cache_enabled:
                raise
            if self._pending_chain_state is not None:
                if preserved:
                    self._commit_chain_state()
                    raise
                self._recovery_chain_id = self._pending_chain_state[0]
                self._force_new_chain_reason = (
                    "format_error"
                    if isinstance(exc, FormatError)
                    else type(exc).__name__
                )
            self._pending_chain_state = None
            raise

        if not self._chain_cache_enabled:
            return message
        reset_reason = self._commit_chain_state()
        if reset_reason is not None and isinstance(message, dict):
            extra = message.setdefault("extra", {})
            extra["chain_reset_reason"] = reset_reason
        return message

    def _commit_chain_state(self) -> str | None:
        pending = self._pending_chain_state
        if pending is None:
            raise RuntimeError(
                "SparseEngine chain-cache query completed without pending "
                "chain state."
            )
        (
            self._chain_id,
            self._last_request_messages,
            self._last_response_message,
            reset_reason,
            finish_reason,
            chain_status,
        ) = pending
        chain_invalidated = chain_status == "invalidated"
        self._last_response_append_safe = (
            not chain_invalidated
            and finish_reason in {"stop", "tool_calls"}
        )
        if chain_invalidated:
            self._chain_id = None
        self._pending_chain_state = None
        self._recovery_chain_id = None
        self._force_new_chain_reason = None
        return reset_reason

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        cleaned = [
            {key: value for key, value in message.items() if key != "provider_specific_fields"}
            for message in messages
        ]
        return super()._prepare_messages_for_api(cleaned)
