from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import typing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

NO_CHAT_TEMPLATE_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
DEFAULT_OUTPUT_ROOT = Path(os.getenv("SPARSEENGINE_OUTPUT_DIR", "outputs")) / "omnikv_full_layer_calibration"
DEFAULT_CONFIG_DIR = REPO_ROOT / "benchmark" / "long_bench" / "config"


@dataclass(frozen=True)
class CalibrationPoint:
    sample_idx: int
    point_idx: int
    kind: str
    prefix_len: int
    query_token_id: int


def build_chat(tokenizer, prompt: str, dataset: str, no_chat_template: bool, thinking_mode: str) -> str:
    if no_chat_template or dataset in NO_CHAT_TEMPLATE_DATASETS:
        return prompt
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return prompt
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking_mode != "off",
    )
    if thinking_mode == "off" and rendered.endswith("<think>\n"):
        rendered += "</think>\n"
    return rendered


def parse_int_list(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def require_path(path: str | Path, kind: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{kind} does not exist: {resolved}")
    return resolved


def install_typing_compatibility() -> list[str]:
    installed: list[str] = []
    if not hasattr(typing, "Unpack"):
        try:
            from typing_extensions import Unpack
        except ImportError as exc:
            raise RuntimeError(
                "Python < 3.11 requires typing_extensions.Unpack to load "
                "the MiniMax Transformers remote model code."
            ) from exc
        typing.Unpack = Unpack  # type: ignore[attr-defined]
        installed.append("typing.Unpack")
    return installed


def text_model_config(config):
    return getattr(config, "text_config", config)


def attention_layer_indices_from_config(config) -> list[int]:
    from sparseengine.models.layout import RuntimeLayout

    text_config = text_model_config(config)
    indices = list(RuntimeLayout.from_config(text_config).full_attention_layer_indices)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        indices = [i for i in indices if layer_types[i] != "sliding_attention"]
    if not indices:
        raise ValueError("Model layout does not contain any full-attention layer.")
    return indices


def prepare_fp8_transformers_config(config) -> list[str]:
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict) or quantization_config.get("quant_method") != "fp8":
        return []
    modules = quantization_config.get("modules_to_not_convert")
    if not isinstance(modules, list):
        return []
    removed = [
        name
        for name in modules
        if name.endswith(".mlp.gate") or name.endswith(".mlp.shared_expert_gate")
    ]
    if removed:
        quantization_config["modules_to_not_convert"] = [name for name in modules if name not in removed]
    return removed


def git_text(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl_prefix(path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError(f"num_samples must be > 0, got {count}.")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx >= count:
                break
            rows.append(json.loads(line))
    if len(rows) < count:
        raise ValueError(f"Requested {count} samples from {path}, but found only {len(rows)}.")
    return rows


def token_ids_for_prompt(tokenizer, prompt: str) -> list[int]:
    add_special_tokens = True
    if tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
        add_special_tokens = False
    ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if not ids:
        raise ValueError("Prompt tokenized to an empty sequence.")
    return ids


def first_answer_token_id(tokenizer, sample: dict[str, Any]) -> tuple[int, str]:
    answers = sample.get("answers")
    if isinstance(answers, list):
        if not answers:
            raise ValueError("Sample has an empty `answers` list; cannot build answer-boundary point.")
        answer = str(answers[0])
    elif answers is not None:
        answer = str(answers)
    else:
        raise ValueError("Sample has no `answers` field; cannot build answer-boundary point.")
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        raise ValueError(f"First answer tokenized to an empty sequence: {answer!r}")
    return int(ids[0]), answer


def build_longbench_prompt_and_ids(
    *,
    tokenizer,
    sample: dict[str, Any],
    dataset: str,
    prompt_format: str,
    max_length: int,
    no_chat_template: bool,
    thinking_mode: str,
) -> tuple[str, list[int], bool]:
    prompt = prompt_format.format(**sample)
    raw_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    truncated = False
    if len(raw_ids) > max_length:
        half = int(max_length / 2)
        if half <= 0:
            raise ValueError(f"max_length must be > 1 for middle truncation, got {max_length}.")
        prompt = tokenizer.decode(raw_ids[:half], skip_special_tokens=True) + tokenizer.decode(
            raw_ids[-half:], skip_special_tokens=True
        )
        truncated = True
    prompt = build_chat(
        tokenizer,
        prompt,
        dataset,
        no_chat_template=no_chat_template,
        thinking_mode=thinking_mode,
    )
    return prompt, token_ids_for_prompt(tokenizer, prompt), truncated


def sample_decode_points(
    *,
    sample_idx: int,
    prompt_token_ids: list[int],
    answer_query_token_id: int,
    random_points_per_sample: int,
    rng: random.Random,
    sink_keep_tokens: int,
    recent_keep_tokens: int,
    min_prefix_tokens: int,
) -> list[CalibrationPoint]:
    if random_points_per_sample < 0:
        raise ValueError(f"random_points_per_sample must be >= 0, got {random_points_per_sample}.")
    prompt_len = len(prompt_token_ids)
    min_prefix = max(int(min_prefix_tokens), int(sink_keep_tokens) + int(recent_keep_tokens))
    max_prefix = prompt_len - 1
    if max_prefix < min_prefix:
        raise ValueError(
            "Prompt is too short for decode-point sampling after sink/recent exclusion: "
            f"prompt_len={prompt_len}, min_prefix={min_prefix}."
        )

    candidates = list(range(min_prefix, max_prefix + 1))
    if random_points_per_sample > len(candidates):
        raise ValueError(
            "Requested more unique random decode points than available positions: "
            f"requested={random_points_per_sample}, available={len(candidates)}."
        )

    random_prefix_lens = sorted(rng.sample(candidates, random_points_per_sample))
    points: list[CalibrationPoint] = []
    for point_idx, prefix_len in enumerate(random_prefix_lens):
        points.append(
            CalibrationPoint(
                sample_idx=sample_idx,
                point_idx=point_idx,
                kind="random",
                prefix_len=prefix_len,
                query_token_id=int(prompt_token_ids[prefix_len]),
            )
        )

    points.append(
        CalibrationPoint(
            sample_idx=sample_idx,
            point_idx=len(points),
            kind="answer_boundary",
            prefix_len=prompt_len,
            query_token_id=int(answer_query_token_id),
        )
    )
    return sorted(points, key=lambda point: (point.prefix_len, point.kind != "random"))


def topk_indices_from_decode_attentions(
    attentions: tuple[torch.Tensor, ...],
    *,
    topk: int,
    sink_keep_tokens: int,
    recent_keep_tokens: int,
    attention_layer_indices: list[int] | None = None,
) -> tuple[list[list[int]], int]:
    if not attentions:
        raise RuntimeError("Model did not return attentions for the decode point.")
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}.")

    if attention_layer_indices is not None and len(attentions) != len(attention_layer_indices):
        if not attention_layer_indices or max(attention_layer_indices) >= len(attentions):
            raise ValueError("Returned attentions do not cover the configured global layers.")
        attentions = tuple(attentions[i] for i in attention_layer_indices)

    layer_topk: list[list[int]] = []
    k_eff: int | None = None
    for layer_idx, attn in enumerate(attentions):
        if attn is None:
            raise RuntimeError(f"Attention tensor for layer {layer_idx} is None.")
        if attn.dim() != 4 or attn.shape[0] != 1 or attn.shape[2] != 1:
            raise RuntimeError(
                "Expected decode attention shape (1, num_heads, 1, kv_len), "
                f"got layer {layer_idx} shape {tuple(attn.shape)}."
            )
        scores = attn[0, :, 0, :].detach().float().max(dim=0).values
        kv_len = int(scores.numel())
        search_start = int(sink_keep_tokens)
        search_end = kv_len - int(recent_keep_tokens)
        if search_end <= search_start:
            raise RuntimeError(
                "Decode point has no searchable history after sink/recent exclusion: "
                f"kv_len={kv_len}, sink_keep_tokens={sink_keep_tokens}, recent_keep_tokens={recent_keep_tokens}."
            )
        search_scores = scores[search_start:search_end]
        cur_k = min(int(topk), int(search_scores.numel()))
        if k_eff is None:
            k_eff = cur_k
        elif k_eff != cur_k:
            raise RuntimeError(f"Inconsistent top-k length across layers: first={k_eff}, layer{layer_idx}={cur_k}.")
        indices = torch.topk(search_scores, k=cur_k, dim=-1, sorted=False).indices + search_start
        layer_topk.append([int(x) for x in indices.cpu().tolist()])

    assert k_eff is not None
    return layer_topk, k_eff


def add_topk_to_pair_scores(pair_scores: np.ndarray, layer_topk: list[list[int]]) -> None:
    num_layers = len(layer_topk)
    if pair_scores.shape != (num_layers, num_layers):
        raise ValueError(f"pair_scores shape {pair_scores.shape} does not match {num_layers} layers.")
    sets = [set(indices) for indices in layer_topk]
    for anchor in range(num_layers):
        anchor_set = sets[anchor]
        for target in range(anchor + 1, num_layers):
            pair_scores[anchor, target] += len(anchor_set & sets[target])


def compute_segment_scores(pair_scores: np.ndarray) -> np.ndarray:
    if pair_scores.ndim != 2 or pair_scores.shape[0] != pair_scores.shape[1]:
        raise ValueError(f"pair_scores must be a square matrix, got shape {pair_scores.shape}.")
    num_layers = int(pair_scores.shape[0])
    segment_scores = np.zeros((num_layers, num_layers + 1), dtype=np.int64)
    for anchor in range(num_layers):
        running = 0
        for next_full in range(anchor + 1, num_layers + 1):
            target = next_full - 1
            if target > anchor:
                running += int(pair_scores[anchor, target])
            segment_scores[anchor, next_full] = running
    return segment_scores


def select_full_layers_dp(segment_scores: np.ndarray, num_full_layers: int) -> tuple[list[int], int]:
    if segment_scores.ndim != 2 or segment_scores.shape[1] != segment_scores.shape[0] + 1:
        raise ValueError(f"segment_scores must have shape (num_layers, num_layers + 1), got {segment_scores.shape}.")
    num_layers = int(segment_scores.shape[0])
    if num_full_layers <= 0 or num_full_layers > num_layers:
        raise ValueError(f"num_full_layers must be in [1, {num_layers}], got {num_full_layers}.")

    memo: dict[tuple[int, int, int], tuple[int, tuple[int, ...]]] = {}

    def best(prev_full: int, min_next: int, remaining: int) -> tuple[int, tuple[int, ...]]:
        key = (prev_full, min_next, remaining)
        if key in memo:
            return memo[key]
        if remaining == 0:
            result = (int(segment_scores[prev_full, num_layers]), ())
            memo[key] = result
            return result

        best_score: int | None = None
        best_suffix: tuple[int, ...] | None = None
        max_candidate = num_layers - remaining
        for candidate in range(min_next, max_candidate + 1):
            suffix_score, suffix = best(candidate, candidate + 1, remaining - 1)
            score = int(segment_scores[prev_full, candidate]) + suffix_score
            candidate_suffix = (candidate,) + suffix
            if best_score is None or score > best_score or (score == best_score and candidate_suffix < best_suffix):
                best_score = score
                best_suffix = candidate_suffix
        assert best_score is not None and best_suffix is not None
        result = (best_score, best_suffix)
        memo[key] = result
        return result

    score, suffix = best(0, 1, num_full_layers - 1)
    return [0, *suffix], int(score)


def selected_segment_breakdown(
    segment_scores: np.ndarray,
    selected_positions: list[int],
    attention_layer_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    num_layers = int(segment_scores.shape[0])
    layer_indices = attention_layer_indices or list(range(num_layers))
    if len(layer_indices) != num_layers:
        raise ValueError(
            f"attention_layer_indices has {len(layer_indices)} entries for {num_layers} score layers."
        )
    out = []
    for idx, anchor_position in enumerate(selected_positions):
        next_position = selected_positions[idx + 1] if idx + 1 < len(selected_positions) else num_layers
        out.append(
            {
                "anchor": int(layer_indices[anchor_position]),
                "next_full_or_end": (
                    int(layer_indices[next_position]) if next_position < num_layers else int(layer_indices[-1] + 1)
                ),
                "sparse_layers": [int(layer) for layer in layer_indices[anchor_position + 1 : next_position]],
                "score": int(segment_scores[anchor_position, next_position]),
            }
        )
    return out


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype {name!r}. Use bfloat16, float16, or float32.")


def move_inputs(token_ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def model_input_device(model, fallback: torch.device) -> torch.device:
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is None or weight.device.type == "meta":
        return fallback
    return weight.device


@torch.no_grad()
def advance_cache(model, past_key_values, token_ids: list[int], *, device: torch.device, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError(f"prefill_chunk_size must be > 0, got {chunk_size}.")
    past = past_key_values
    for start in range(0, len(token_ids), chunk_size):
        chunk = token_ids[start : start + chunk_size]
        if not chunk:
            continue
        outputs = model(
            input_ids=move_inputs(chunk, device),
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past = outputs.past_key_values
    return past


@torch.no_grad()
def collect_sample_topk(
    *,
    model,
    device: torch.device,
    prompt_token_ids: list[int],
    points: list[CalibrationPoint],
    topk: int,
    sink_keep_tokens: int,
    recent_keep_tokens: int,
    prefill_chunk_size: int,
    attention_layer_indices: list[int],
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    num_layers = len(attention_layer_indices)
    if num_layers <= 0:
        raise ValueError("attention_layer_indices must not be empty.")
    pair_scores = np.zeros((num_layers, num_layers), dtype=np.int64)
    point_records: list[dict[str, Any]] = []
    saveable_topk: list[dict[str, Any]] = []

    past = None
    processed = 0
    for point in points:
        if point.prefix_len < processed:
            raise RuntimeError(
                f"Calibration points must be nondecreasing by prefix_len; got {point.prefix_len} after {processed}."
            )
        past = advance_cache(
            model,
            past,
            prompt_token_ids[processed : point.prefix_len],
            device=device,
            chunk_size=prefill_chunk_size,
        )
        processed = point.prefix_len

        outputs = model(
            input_ids=move_inputs([point.query_token_id], device),
            past_key_values=past,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        processed += 1

        layer_topk, k_eff = topk_indices_from_decode_attentions(
            outputs.attentions,
            topk=topk,
            sink_keep_tokens=sink_keep_tokens,
            recent_keep_tokens=recent_keep_tokens,
            attention_layer_indices=attention_layer_indices,
        )
        add_topk_to_pair_scores(pair_scores, layer_topk)
        record = asdict(point)
        record.update({"status": "success", "effective_topk": int(k_eff)})
        point_records.append(record)
        saveable_topk.append({"point": record, "topk_indices_by_layer": layer_topk})

    return point_records, pair_scores, saveable_topk


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    model_path = require_path(args.model_path, "model path")
    longbench_root = require_path(args.longbench_root, "LongBench root")
    config_dir = require_path(args.config_dir, "LongBench config dir")
    data_path = require_path(longbench_root / "data" / f"{args.dataset}.jsonl", "LongBench dataset file")
    prompt_path = require_path(config_dir / "dataset2prompt.json", "LongBench prompt config")

    with prompt_path.open("r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    if args.dataset not in dataset2prompt:
        raise ValueError(f"Dataset {args.dataset!r} is missing from {prompt_path}.")
    prompt_format = dataset2prompt[args.dataset]
    typing_compatibility = (
        install_typing_compatibility() if args.trust_remote_code else []
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=args.trust_remote_code
    )
    base_config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=args.trust_remote_code
    )
    text_config = text_model_config(base_config)
    attention_layer_indices = attention_layer_indices_from_config(base_config)
    max_length = int(args.max_length or getattr(text_config, "max_position_embeddings", 32000))
    if max_length <= 0:
        raise ValueError(f"Resolved max_length must be > 0, got {max_length}.")

    dtype = torch_dtype_from_name(args.torch_dtype)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")
    if args.max_memory_per_device_gib <= 0:
        raise ValueError(
            "max_memory_per_device_gib must be > 0, "
            f"got {args.max_memory_per_device_gib}."
        )

    removed_fp8_exclusions = prepare_fp8_transformers_config(base_config)
    model_kwargs = {
        "config": base_config,
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": "eager",
    }
    if args.device_map == "auto":
        if requested_device.type != "cuda":
            raise ValueError("--device-map auto requires a CUDA --device.")
        visible_devices = torch.cuda.device_count()
        if visible_devices <= 0:
            raise RuntimeError("--device-map auto requires at least one visible CUDA device.")
        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = {
            index: f"{args.max_memory_per_device_gib}GiB"
            for index in range(visible_devices)
        }
    elif requested_device.type == "cuda":
        model_kwargs["device_map"] = {"": str(requested_device)}
    model_class = (
        AutoModelForImageTextToText
        if getattr(base_config, "model_type", "") == "gemma4"
        else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(
        str(model_path),
        **model_kwargs,
    )
    if args.device_map == "auto":
        hf_device_map = getattr(model, "hf_device_map", {})
        offloaded = sorted(
            {str(value) for value in hf_device_map.values() if str(value) in {"cpu", "disk"}}
        )
        if offloaded:
            raise RuntimeError(
                "Automatic device placement offloaded model modules to "
                f"{offloaded}; increase visible GPU memory instead of running a mixed CPU/disk calibration."
            )
    elif requested_device.type != "cuda":
        model.to(requested_device)
    device = model_input_device(model, requested_device)
    model.eval()

    num_hidden_layers = int(text_config.num_hidden_layers)
    num_attention_layers = len(attention_layer_indices)
    if args.num_full_layers > num_attention_layers:
        raise ValueError(
            f"num_full_layers={args.num_full_layers} exceeds full-attention candidates={num_attention_layers}."
        )

    samples = read_jsonl_prefix(data_path, args.num_samples)
    rng = random.Random(args.seed)
    total_pair_scores = np.zeros((num_attention_layers, num_attention_layers), dtype=np.int64)
    all_point_records: list[dict[str, Any]] = []
    all_topk: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []

    for sample_idx, sample in enumerate(tqdm(samples, desc=f"Calibrating {args.dataset}")):
        prompt, prompt_token_ids, truncated = build_longbench_prompt_and_ids(
            tokenizer=tokenizer,
            sample=sample,
            dataset=args.dataset,
            prompt_format=prompt_format,
            max_length=max_length,
            no_chat_template=bool(args.no_chat_template),
            thinking_mode=args.thinking_mode,
        )
        answer_token_id, answer_text = first_answer_token_id(tokenizer, sample)
        points = sample_decode_points(
            sample_idx=sample_idx,
            prompt_token_ids=prompt_token_ids,
            answer_query_token_id=answer_token_id,
            random_points_per_sample=args.random_decode_points_per_sample,
            rng=rng,
            sink_keep_tokens=args.sink_keep_tokens,
            recent_keep_tokens=args.recent_keep_tokens,
            min_prefix_tokens=args.min_prefix_tokens,
        )
        point_records, sample_pair_scores, sample_topk = collect_sample_topk(
            model=model,
            device=device,
            prompt_token_ids=prompt_token_ids,
            points=points,
            topk=args.topk,
            sink_keep_tokens=args.sink_keep_tokens,
            recent_keep_tokens=args.recent_keep_tokens,
            prefill_chunk_size=args.prefill_chunk_size,
            attention_layer_indices=attention_layer_indices,
        )
        total_pair_scores += sample_pair_scores
        all_point_records.extend(point_records)
        if args.save_topk:
            all_topk.extend(sample_topk)
        prompt_records.append(
            {
                "sample_idx": sample_idx,
                "status": "success",
                "prompt_token_length": len(prompt_token_ids),
                "truncated": truncated,
                "answer_boundary_token_id": answer_token_id,
                "answer_boundary_answer": answer_text,
                "point_count": len(points),
                "points": point_records,
            }
        )
        del prompt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    segment_scores = compute_segment_scores(total_pair_scores)
    selected_positions, best_score = select_full_layers_dp(segment_scores, args.num_full_layers)
    selected_layers = [attention_layer_indices[position] for position in selected_positions]
    full_layers_str = ",".join(str(layer) for layer in selected_layers)

    point_topk_sum = sum(int(record["effective_topk"]) for record in all_point_records)
    denominator = point_topk_sum * (num_attention_layers - len(selected_layers))
    normalized = float(best_score / denominator) if denominator else 0.0

    np.save(output_dir / "pair_scores.npy", total_pair_scores)
    np.save(output_dir / "segment_scores.npy", segment_scores)

    with (output_dir / "per_sample_points.jsonl").open("w", encoding="utf-8") as f:
        for record in prompt_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.save_topk:
        torch.save(all_topk, output_dir / "topk_indices.pt")

    selected_payload = {
        "selected_full_layers": selected_layers,
        "full_attention_layers": full_layers_str,
        "num_hidden_layers": num_hidden_layers,
        "attention_layer_indices": attention_layer_indices,
        "num_attention_layers": num_attention_layers,
        "num_full_layers": int(args.num_full_layers),
        "forced_full_layers": [attention_layer_indices[0]],
        "topk": int(args.topk),
        "sink_keep_tokens": int(args.sink_keep_tokens),
        "recent_keep_tokens": int(args.recent_keep_tokens),
        "dataset": args.dataset,
        "num_samples": int(args.num_samples),
        "random_decode_points_per_sample": int(args.random_decode_points_per_sample),
        "answer_boundary_points_per_sample": 1,
        "seed": int(args.seed),
        "token_coverage_score": int(best_score),
        "coverage_denominator": int(denominator),
        "normalized_token_coverage": normalized,
        "segment_breakdown": selected_segment_breakdown(
            segment_scores,
            selected_positions,
            attention_layer_indices,
        ),
    }
    json_dump(output_dir / "selected_full_layers.json", selected_payload)

    run_info = {
        "command": sys.argv,
        "created_at": datetime.now().isoformat(),
        "cwd": os.getcwd(),
        "model_path": str(model_path),
        "longbench_root": str(longbench_root),
        "data_path": str(data_path),
        "prompt_path": str(prompt_path),
        "output_dir": str(output_dir),
        "model_config_model_type": getattr(base_config, "model_type", None),
        "model_config_num_hidden_layers": num_hidden_layers,
        "attention_layer_indices": attention_layer_indices,
        "removed_fp8_modules_to_not_convert": removed_fp8_exclusions,
        "typing_compatibility": typing_compatibility,
        "trust_remote_code": bool(args.trust_remote_code),
        "attention_implementation": "eager",
        "torch_dtype": args.torch_dtype,
        "device": args.device,
        "device_map": args.device_map,
        "max_memory_per_device_gib": int(args.max_memory_per_device_gib),
        "resolved_input_device": str(device),
        "hf_device_map": {
            str(key): str(value)
            for key, value in getattr(model, "hf_device_map", {}).items()
        },
        "prefill_chunk_size": int(args.prefill_chunk_size),
        "max_length": int(max_length),
        "no_chat_template": bool(args.no_chat_template),
        "thinking_mode": args.thinking_mode,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "git_commit": git_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_text(["git", "status", "--short"]),
    }
    json_dump(output_dir / "run_info.json", run_info)
    return {"output_dir": str(output_dir), **selected_payload}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline OmniKV full-layer selector using decode-style token coverage.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--longbench-root", required=True)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--dataset", default="narrativeqa")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-full-layers", type=int, default=6)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--random-decode-points-per-sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-sink-tokens", dest="sink_keep_tokens", type=int, default=0
    )
    parser.add_argument(
        "--num-recent-tokens", dest="recent_keep_tokens", type=int, default=32
    )
    parser.add_argument("--min-prefix-tokens", type=int, default=1)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device-map", default="single", choices=("single", "auto"))
    parser.add_argument("--max-memory-per-device-gib", type=int, default=76)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--thinking-mode", default="off", choices=("off", "on"))
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--save-topk", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_calibration(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
