from __future__ import annotations

from abc import ABC
from typing import Iterable

import torch

from sparseengine.config import Config
from sparseengine.engine.cache_manager import CacheManager
from sparseengine.engine.sequence import Sequence
from sparseengine.utils.log import logger


class ActivationController(ABC):
    """Method-owned hidden-state hooks coordinated by SparseController."""

    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager

    @staticmethod
    def create(config: Config, cache_manager: CacheManager) -> "ActivationController":
        if str(config.sparse_method or "") == "skipkv":
            return SkipKVActivationController(config, cache_manager)
        return ActivationController(config, cache_manager)

    def set_tokenizer_metadata(
        self,
        *,
        delimiter_token_ids: Iterable[int] | None = None,
        non_execution_token_ids: Iterable[int] | None = None,
    ):
        del delimiter_token_ids, non_execution_token_ids

    requires_committed_token_history = False

    def prepare_forward(self, seqs: list[Sequence], is_prefill: bool):
        del seqs, is_prefill

    def apply_layer_hook(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        context,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del layer_idx, context
        return hidden_states, residual

    def post_forward(self, seqs: list[Sequence], is_prefill: bool):
        del seqs, is_prefill

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        return []


class SkipKVActivationController(ActivationController):
    """SkipKV hidden-state capture and optional adaptive activation steering."""

    def __init__(self, config: Config, cache_manager: CacheManager):
        super().__init__(config, cache_manager)
        self._seqs: list[Sequence] = []
        self._is_prefill = False
        self._delimiter_token_ids: set[int] = set()
        self._non_execution_token_ids: set[int] = set()
        self._steering_vector: torch.Tensor | None = None
        self._steering_alpha: torch.Tensor | None = None
        self._hidden_capture: torch.Tensor | None = None
        self._active_real_batch = 0
        self._hidden_capture_layer = self._normalize_layer(
            int(getattr(config, "skipkv_sentence_embedding_layer", -1))
        )
        self._steering_layer = self._normalize_layer(
            int(getattr(config, "skipkv_steering_layer", -1))
        )
        self._load_steering_vector()

    def _normalize_layer(self, layer_idx: int) -> int:
        num_layers = int(getattr(self.config.hf_config, "num_hidden_layers", 0) or 0)
        if num_layers <= 0:
            return int(layer_idx)
        if int(layer_idx) < 0:
            return max(0, num_layers + int(layer_idx))
        return min(int(layer_idx), num_layers - 1)

    def _load_steering_vector(self):
        path = getattr(self.config, "skipkv_steering_vector_path", None)
        if not path:
            return
        vector = torch.load(str(path), map_location="cuda")
        if isinstance(vector, dict):
            for key in ("steering_vector", "vector", "direction"):
                if key in vector:
                    vector = vector[key]
                    break
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"skipkv steering vector at {path!r} did not contain a tensor.")
        vector = vector.detach().flatten().to(
            device="cuda", dtype=self.config.hf_config.dtype
        )
        hidden_size = int(getattr(self.config.hf_config, "hidden_size", vector.numel()))
        if int(vector.numel()) != hidden_size:
            raise ValueError(
                "skipkv steering vector hidden size mismatch: "
                f"vector={int(vector.numel())}, model={hidden_size}."
            )
        self._steering_vector = vector.to(dtype=self.config.hf_config.dtype)

    def _ensure_decode_buffers(self, hidden_size: int, device: torch.device, dtype: torch.dtype):
        capacity = max(1, int(getattr(self.config, "max_decoding_seqs", 1) or 1))
        if (
            self._hidden_capture is None
            or self._hidden_capture.device != device
            or self._hidden_capture.dtype != dtype
            or self._hidden_capture.shape[0] < capacity
            or self._hidden_capture.shape[1] != int(hidden_size)
        ):
            self._hidden_capture = torch.empty((capacity, int(hidden_size)), dtype=dtype, device=device)
        if (
            self._steering_alpha is None
            or self._steering_alpha.device != device
            or self._steering_alpha.shape[0] < capacity
        ):
            self._steering_alpha = torch.zeros(
                (capacity,),
                dtype=self.config.hf_config.dtype,
                device=device,
            )

    def set_tokenizer_metadata(
        self,
        *,
        delimiter_token_ids: Iterable[int] | None = None,
        non_execution_token_ids: Iterable[int] | None = None,
    ):
        if delimiter_token_ids is not None:
            self._delimiter_token_ids = {int(x) for x in delimiter_token_ids}
            setter = getattr(self.cache_manager, "set_skipkv_delimiter_token_ids", None)
            if setter is not None:
                setter(self._delimiter_token_ids)
        if non_execution_token_ids is not None:
            self._non_execution_token_ids = {int(x) for x in non_execution_token_ids}
            setter = getattr(self.cache_manager, "set_skipkv_non_execution_token_ids", None)
            if setter is not None:
                setter(self._non_execution_token_ids)

    def prepare_forward(self, seqs: list[Sequence], is_prefill: bool):
        self._seqs = list(seqs)
        self._is_prefill = bool(is_prefill)
        self._active_real_batch = len(seqs)
        if self._steering_alpha is None and torch.cuda.is_available():
            capacity = max(1, int(getattr(self.config, "max_decoding_seqs", 1) or 1))
            self._steering_alpha = torch.zeros(
                (capacity,),
                dtype=self.config.hf_config.dtype,
                device="cuda",
            )
        if is_prefill or self._steering_alpha is None:
            return
        self._steering_alpha.zero_()
        if not bool(getattr(self.config, "skipkv_enable_activation_steering", False)):
            return
        if self._steering_vector is None:
            return
        if not self._delimiter_token_ids:
            return

        alpha_base = float(getattr(self.config, "skipkv_steering_alpha", 0.0) or 0.0)
        alpha_step = float(getattr(self.config, "skipkv_steering_alpha_increment", 0.0) or 0.0)
        alpha_limit = float(getattr(self.config, "skipkv_steering_alpha_max", 0.0) or 0.0)
        values: list[float] = []
        for seq in seqs:
            token_id = seq.decode_input_token
            if seq.decode_input_position < int(seq.num_prompt_tokens) or token_id not in self._delimiter_token_ids:
                values.append(0.0)
                continue
            getter = getattr(self.cache_manager, "skipkv_non_execution_count", None)
            count = int(getter(seq.seq_id)) if getter is not None else 0
            alpha = alpha_base + alpha_step * count
            if alpha_limit != 0.0:
                if alpha_step < 0.0:
                    alpha = max(alpha_limit, alpha)
                elif alpha_step > 0.0:
                    alpha = min(alpha_limit, alpha)
            values.append(alpha)
        if values:
            self._steering_alpha[: len(values)].copy_(
                torch.tensor(values, dtype=self._steering_alpha.dtype, device=self._steering_alpha.device)
            )

    def apply_layer_hook(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        context,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del context
        if self._is_prefill or hidden_states.dim() != 2:
            return hidden_states, residual

        hidden_size = int(hidden_states.shape[-1])
        self._ensure_decode_buffers(hidden_size, hidden_states.device, hidden_states.dtype)

        if int(layer_idx) == self._steering_layer and self._steering_vector is not None:
            assert self._steering_alpha is not None
            vector = self._steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            alpha = self._steering_alpha[: hidden_states.shape[0]].to(dtype=hidden_states.dtype)
            hidden_states = hidden_states + alpha[:, None] * vector[None, :]

        if int(layer_idx) == self._hidden_capture_layer:
            assert self._hidden_capture is not None
            layer_output = hidden_states if residual is None else hidden_states + residual
            self._hidden_capture[: layer_output.shape[0]].copy_(layer_output.detach())
        return hidden_states, residual

    def post_forward(self, seqs: list[Sequence], is_prefill: bool):
        if is_prefill or self._hidden_capture is None:
            return
        updater = getattr(self.cache_manager, "record_skipkv_decode_hidden_states", None)
        if updater is None:
            return
        real_batch = min(len(seqs), int(self._active_real_batch), int(self._hidden_capture.shape[0]))
        if real_batch <= 0:
            return
        updater(seqs[:real_batch], self._hidden_capture[:real_batch])

    def decode_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        if self._steering_vector is not None:
            tensors.append(self._steering_vector)
        if self._steering_alpha is not None:
            tensors.append(self._steering_alpha)
        if self._hidden_capture is not None:
            tensors.append(self._hidden_capture)
        return tensors
