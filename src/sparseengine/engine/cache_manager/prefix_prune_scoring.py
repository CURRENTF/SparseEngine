from __future__ import annotations

from bisect import bisect_left

import torch

from sparseengine.kernels.triton.prefill_score import prefill_score_fwd

from .base import ExplicitKVPayload, PrefillComputeView, PrefillScoreRequest


class PrefixPruneScoringMixin:
    """Shared maintenance-forward scoring for radix cache managers."""

    def _prefix_prune_prefix_logical_positions(self, seq_id: int) -> list[int]:
        raise NotImplementedError

    def begin_prefix_prune_scoring(
        self,
        *,
        seq_id: int,
        candidate_start: int,
        query_start: int,
        query_end: int,
    ) -> None:
        if self._prefix_prune_scoring is not None:
            raise RuntimeError("another prefix-prune scoring forward is already active.")
        if not (0 <= candidate_start < query_start < query_end):
            raise ValueError(
                "invalid prefix-prune scoring ranges: "
                f"candidate_start={candidate_start} query=[{query_start}, {query_end})."
            )
        self._prefix_prune_scoring = {
            "seq_id": int(seq_id),
            "candidate_start": int(candidate_start),
            "query_start": int(query_start),
            "query_end": int(query_end),
            "score": None,
        }

    def begin_prefix_prune_scoring_batch(self, requests) -> None:
        if self._prefix_prune_scoring is not None:
            raise RuntimeError("another prefix-prune scoring forward is already active.")
        states = []
        for request in requests:
            self.begin_prefix_prune_scoring(**request)
            states.append(self._prefix_prune_scoring)
            self._prefix_prune_scoring = None
        self._prefix_prune_scoring = {"batch": states}

    def finish_prefix_prune_scoring_batch(self) -> list[torch.Tensor]:
        state = self._prefix_prune_scoring
        self._prefix_prune_scoring = None
        if state is None or "batch" not in state:
            raise RuntimeError("No prefix-prune scoring batch is active.")
        return [self._finish_prefix_prune_score(item) for item in state["batch"]]

    def abort_prefix_prune_scoring(self) -> None:
        self._prefix_prune_scoring = None

    def finish_prefix_prune_scoring(self) -> torch.Tensor:
        state = self._prefix_prune_scoring
        self._prefix_prune_scoring = None
        return self._finish_prefix_prune_score(state)

    @staticmethod
    def _finish_prefix_prune_score(state) -> torch.Tensor:
        if state is None or not isinstance(state.get("score"), torch.Tensor):
            raise RuntimeError("prefix-prune scoring forward produced no attention scores.")
        score = state["score"]
        positions = state.get("logical_positions")
        if positions is not None:
            logical_score = score.new_zeros(int(state["query_end"]))
            logical_score.index_copy_(0, positions, score)
            return logical_score
        return score

    def _prefix_prune_physical_score_window(self, state=None):
        if state is None:
            state = self._prefix_prune_scoring
        if state is None or "batch" in state:
            raise RuntimeError("No scalar prefix-prune scoring state is active.")
        if "physical_window" in state:
            return state["physical_window"]
        row = self.seq_id_to_row[int(state["seq_id"])]
        end = int(self.row_seq_lens[row])
        query_len = int(state["query_end"]) - int(state["query_start"])
        start = end - query_len
        candidate = int(state["candidate_start"])
        if end != int(state["query_end"]):
            positions = self._prefix_prune_prefix_logical_positions(int(state["seq_id"]))
            if len(positions) != start:
                raise RuntimeError("Prefix scoring logical/physical mapping length mismatch.")
            candidate = bisect_left(positions, candidate)
            positions.extend(range(int(state["query_start"]), int(state["query_end"])))
            state["logical_positions"] = torch.tensor(
                positions, dtype=torch.long, device=self.device,
            )
        state["physical_window"] = (start, end, candidate)
        return state["physical_window"]

    def prefill_score_request(self, layer_idx, seqs):
        del layer_idx
        state = self._prefix_prune_scoring
        if state is None:
            return None
        if "batch" in state:
            states = state["batch"]
            if [s.seq_id for s in seqs] != [s["seq_id"] for s in states]:
                raise RuntimeError("Prefix scoring batch order differs from model inputs.")
            windows = [self._prefix_prune_physical_score_window(s) for s in states]
            return PrefillScoreRequest(
                tuple((start, end) for start, end, _ in windows),
                "probability",
                candidate_ranges=tuple(
                    (candidate, start) for start, _, candidate in windows
                ),
            )
        start, end, candidate = self._prefix_prune_physical_score_window()
        return PrefillScoreRequest(
            ((start, end),), "probability", candidate, end - start,
        )

    @torch.no_grad()
    def collect_prefill_attention_score(
        self,
        layer_idx: int,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
        attention_lse: torch.Tensor | None = None,
    ):
        del layer_idx, b_start_loc, attention_lse
        state = self._prefix_prune_scoring
        if state is None:
            return None
        states = state.get("batch", [state])
        if int(chunk_lens.numel()) != len(states):
            raise RuntimeError("Prefix scoring metadata does not cover the maintenance batch.")
        if view.token_scores is None and not isinstance(view.payload, ExplicitKVPayload):
            raise TypeError("Prefix pruning requires explicit KV storage or fused token scores.")
        offset = 0
        for i, item in enumerate(states):
            start, end, candidate = self._prefix_prune_physical_score_window(item)
            length = end - start
            if view.token_scores is not None:
                if view.token_scores.shape[0] != len(states) or view.token_scores.shape[1] < end:
                    raise RuntimeError("Prefix-prune score shape does not match physical context.")
                score = view.token_scores[i, :end]
            else:
                step_score = torch.zeros((1, end), dtype=torch.float32, device=q.device)
                starts = torch.tensor([start], dtype=torch.int32, device=q.device)
                prefill_score_fwd(
                    q[offset:offset + length], view.payload.k_cache, step_score,
                    view.meta.req_indices[i:i + 1], torch.zeros_like(starts),
                    view.meta.context_lens[i:i + 1], starts, length,
                    view.meta.active_slots, starts,
                    torch.tensor([end], dtype=torch.int32, device=q.device),
                    candidate_start=candidate, recent_keep_tokens=length,
                    score_mode="probability",
                )
                score = step_score[0]
            accumulated = item.get("score")
            if accumulated is None:
                item["score"] = score.clone()
            else:
                torch.maximum(accumulated, score, out=accumulated)
            offset += length
        if offset != q.shape[0]:
            raise RuntimeError("Prefix-prune query window length mismatch.")
        return None
