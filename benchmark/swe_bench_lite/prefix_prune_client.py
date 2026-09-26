"""Shared client-side prefix pruning for live agents and recorded trace replay."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any


class PrefixPruneClient:
    def __init__(self, *, api_base: str, tokenizer_path: str, keep_ratio: float,
                 events_path: Path, trigger_tokens: int = 8192, tool_result_lag: int = 0):
        if not math.isfinite(keep_ratio) or not 0 <= keep_ratio < 1:
            raise ValueError("Prefix-prune keep ratio must be in [0, 1).")
        if trigger_tokens <= 0:
            raise ValueError("Prefix-prune trigger must be positive.")
        if tool_result_lag < 0:
            raise ValueError("Tool-result pruning lag must be non-negative.")
        self._prune_trigger_tokens = trigger_tokens
        self._prefix_cache_api_base = api_base.rstrip("/")
        self._prune_tokenizer_path = tokenizer_path
        self._prune_keep_ratio = keep_ratio
        self._prune_tool_result_lag = tool_result_lag
        self._prune_events_path = events_path
        self._prune_target = "tool_results"
        self._prune_policy = "kvzip_global"
        self._prune_finished = False
        self._prune_reuse_verified = False
        self._prune_freed_slots = 0
        self._prune_processed_messages = []
        self._prune_tool_selector = None
        self._prune_pending = None

    @staticmethod
    def _value_digest(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _prefix_cache_request(
        self, method: str, path: str, body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timing = getattr(self, "_prune_rpc_timing", None)
        if timing is None:
            return self._send_prefix_cache_request(method, path, body)
        key = "prune_poll" if method == "GET" else (
            "match_chat" if body and "chat" in body and path.endswith("/match")
            else "match_tokens" if path.endswith("/match") else "prune_submit")
        start = time.perf_counter()
        try:
            return self._send_prefix_cache_request(method, path, body)
        finally:
            row = timing.setdefault(key, {"calls": 0, "wall_s": 0.0})
            row["calls"] += 1
            row["wall_s"] += time.perf_counter() - start

    def _send_prefix_cache_request(
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
        if getattr(self, "_prune_rpc_timing", None) is not None:
            payload["client_rpc_timing"] = self._prune_rpc_timing
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

    def _maybe_verify_prune_reuse(self, chat: dict[str, Any], match=None) -> None:
        if not self._prune_finished or self._prune_reuse_verified:
            return
        if match is None:
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
        timing_enabled = os.getenv("SPARSEENGINE_PREFIX_PRUNE_TIMING", "0") == "1"
        self._prune_rpc_timing = {} if timing_enabled else None
        tool_mode = self._prune_target == "tool_results"
        if self._prune_finished and not tool_mode:
            return
        cursor = len(self._prune_processed_messages)
        new_tools = False
        selected_tool_index = None
        if tool_mode:
            messages = chat["messages"]
            seen = getattr(self, "_prune_seen_messages", None)
            if seen is None:
                seen = self._prune_processed_messages.copy()
            if len(messages) < len(seen) or any(
                old != messages[i] for i, old in enumerate(seen)
            ):
                raise RuntimeError("Tool-pruning transcript is not append-only; refusing to reuse its cursor.")
            old_seen_len = len(seen)
            seen.extend(deepcopy(messages[old_seen_len:]))
            self._prune_seen_messages = seen
            new_tools = any(m.get("role") == "tool" for m in messages[cursor:])
            lag = getattr(self, "_prune_tool_result_lag", 0)
            if lag:
                tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
                if len(tool_indices) <= lag:
                    return
                selected_tool_index = tool_indices[-lag - 1]
                if selected_tool_index < cursor:
                    return
                if sum(m.get("role") == "tool" for m in messages[old_seen_len:]) > 1:
                    raise RuntimeError(
                        "Delayed tool pruning requires one new tool result per model turn."
                    )
            if not new_tools and (not self._prune_finished or self._prune_reuse_verified):
                self._prune_processed_messages.extend(self._prune_seen_messages[cursor:])
                return
        match_before = self._match_prefix(chat)
        if tool_mode:
            self._maybe_verify_prune_reuse(chat, match_before)
            if not new_tools:
                self._prune_processed_messages.extend(self._prune_seen_messages[cursor:])
                return
        usable = int(match_before.get("usable_tokens") or 0)
        if not tool_mode and usable < self._prune_trigger_tokens:
            return
        matched = int(match_before.get("matched_tokens") or 0)
        if matched != usable:
            raise RuntimeError(
                "Completed MiniSWE turn is not fully available for pruning: "
                f"matched={matched} usable={usable}."
            )
        selection = None
        if self._prune_target == "tool_results":
            if self._prune_tool_selector is None:
                from benchmark.swe_bench_lite.tool_prune import ToolResultPruneSelector
                self._prune_tool_selector = ToolResultPruneSelector(self._prune_tokenizer_path)
            selection_start = time.perf_counter() if timing_enabled else None
            selector_kwargs = {"message_start": cursor}
            if selected_tool_index is not None:
                selector_kwargs["message_indices"] = (selected_tool_index,)
            selection = self._prune_tool_selector.select(
                chat, block_size=int(match_before["block_size"]), usable_tokens=usable,
                **selector_kwargs,
            )
            if selection_start is not None:
                self._prune_rpc_timing["tool_selection"] = {
                    "calls": 1, "wall_s": time.perf_counter() - selection_start,
                }
            ranges = selection["ranges"]
            if not ranges:
                self._record_prune_event(
                    "prune_skipped", status="skipped_by_policy", target="tool_results",
                    reason="no_block_aligned_tool_body", usable_tokens=usable,
                    tool_tokens=selection["tool_tokens"],
                )
                end = selected_tool_index + 1 if selected_tool_index is not None else len(messages)
                self._prune_processed_messages.extend(self._prune_seen_messages[cursor:end])
                return
            if selection["eligible_tokens"] < self._prune_trigger_tokens:
                # Commit the cursor only after pruning; pending bodies are selected
                # again on the next turn, without including previously pruned bodies.
                self._record_prune_event(
                    "prune_deferred", status="skipped_by_policy",
                    reason="below_candidate_threshold",
                    candidate_tokens=selection["eligible_tokens"],
                    trigger_tokens=self._prune_trigger_tokens,
                )
                return
            local_match = self._prefix_cache_request(
                "POST", "/prefix_cache/match", {"token_ids": selection["token_ids"]},
            )
            if not match_before.get("last_block_id") or any(
                local_match.get(key) != match_before.get(key)
                for key in ("last_block_id", "prompt_tokens", "usable_tokens", "matched_tokens", "block_size")
            ):
                raise RuntimeError("Tool-range tokenizer/template does not match the server's cached path.")
            keep_tokens = math.floor(selection["eligible_tokens"] * self._prune_keep_ratio)
            if str(match_before.get("method") or "") == "quest":
                block_size = int(match_before["block_size"])
                keep_tokens = (keep_tokens // block_size) * block_size
            selector = {"token_ids": selection["token_ids"], "ranges": ranges}
        else:
            ranges = [(self._prune_range_start, self._prune_range_end)]
            keep_tokens = self._prune_keep_tokens
            selector = {
                "chat": chat, "range_start": self._prune_range_start,
                "range_end": self._prune_range_end,
            }
        prune_body = {
            **selector, "keep_tokens": keep_tokens, "policy": self._prune_policy,
            "observation_tokens": 64, "score_chunk_size": 1024, "prev_postfix_size": 32,
        }
        request_digest = self._value_digest(prune_body)
        pending = getattr(self, "_prune_pending", None)
        if pending is not None:
            if pending["request_digest"] != request_digest:
                raise RuntimeError(
                    "A different prefix-prune request arrived while the previous job "
                    "still requires status recovery."
                )
            prune_id = str(pending["prune_id"])
            before_resident = int(pending["before_resident"])
            status = {"status": "queued"}
        else:
            queued = self._prefix_cache_request(
                "POST", "/prefix_cache/prune", prune_body,
            )
            prune_id = str(queued.get("prune_id") or "")
            if not prune_id:
                raise RuntimeError(f"Prefix prune returned no prune_id: {queued}.")
            before_resident = int(match_before.get("resident_kv_tokens") or 0)
            self._prune_pending = {
                "request_digest": request_digest,
                "prune_id": prune_id,
                "before_resident": before_resident,
            }
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
            if status.get("status") in {"blocked", "failed"}:
                self._prune_pending = None
            raise RuntimeError(f"Prefix prune did not complete: {status}.")
        result = status.get("result") or {}
        freed = int(result.get("freed_device_slots") or 0)
        expected_freed = sum(right - left for left, right in ranges) - keep_tokens
        if freed != expected_freed or result.get("quality_degraded") is not True:
            raise RuntimeError(
                "Prefix prune result violated physical accounting/tag contract: "
                f"freed={freed} expected={expected_freed} result={result}."
            )
        match_after = (
            self._prefix_cache_request("POST", "/prefix_cache/match", {"token_ids": selection["token_ids"]})
            if tool_mode else self._match_prefix(chat)
        )
        after_matched = int(match_after.get("matched_tokens") or 0)
        after_resident = int(match_after.get("resident_kv_tokens") or 0)
        if after_matched != usable or before_resident - after_resident != freed:
            raise RuntimeError(
                "Prefix prune did not preserve the logical route or compact resident KV: "
                f"before={usable} after={after_matched} resident={after_resident}."
            )
        self._prune_finished = True
        self._prune_reuse_verified = False
        self._prune_freed_slots = after_matched - after_resident
        self._prune_pending = None
        if tool_mode:
            end = selected_tool_index + 1 if selected_tool_index is not None else len(messages)
            self._prune_processed_messages.extend(self._prune_seen_messages[cursor:end])
        self._record_prune_event(
            "prune_completed",
            policy=self._prune_policy,
            selector_digest=self._value_digest(chat["messages"][:2]),
            prune_id=prune_id,
            usable_tokens=usable,
            resident_kv_tokens=after_resident,
            freed_device_slots=freed,
            target=self._prune_target,
            ranges=ranges,
            range=[ranges[0][0], ranges[-1][1]],
            keep_tokens=keep_tokens,
            tool_selection=({k: v for k, v in selection.items() if k != "token_ids"} if selection else None),
            message_start=cursor if tool_mode else None,
            message_end=(selected_tool_index + 1 if selected_tool_index is not None else len(chat["messages"])) if tool_mode else None,
            tool_result_lag=getattr(self, "_prune_tool_result_lag", 0) if tool_mode else None,
            quality_degraded=True,
            trigger_tokens=self._prune_trigger_tokens,
            scoring={key: result[key] for key in ("scoring_chunks", "max_scoring_batch", "scoring_jobs") if key in result},
            **({"server_queue_s": status["started_at"] - status["created_at"],
                "server_execution_s": status["finished_at"] - status["started_at"]}
               if timing_enabled else {}),
        )
