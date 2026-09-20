#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    raw = str(value).strip()
    if raw.startswith("@"):
        raw = Path(raw[1:]).expanduser().read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("JSON argument must decode to an object.")
    return payload


def _mode_suffix(mode: str) -> str:
    if mode == "scdq":
        return "_scdq"
    if mode == "multi_turn":
        return "_multi_turn"
    raise ValueError(f"unsupported SCBench mode {mode!r}")


def _cases_for_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["scdq", "multi_turn"]
    if mode in {"scdq", "multi_turn"}:
        return [mode]
    raise ValueError(f"unsupported SCBench mode {mode!r}")


def build_hyper_param(
    base_hyper_param: dict[str, Any],
    *,
    enable_prefix_caching: bool,
    prefix_cache_block_size: int,
    prefix_cache_salt: str,
) -> dict[str, Any]:
    hyper_param = dict(base_hyper_param)
    hyper_param["enable_prefix_caching"] = bool(enable_prefix_caching)
    hyper_param.setdefault("sparse_method", "vanilla")
    hyper_param["prefix_cache_block_size"] = int(prefix_cache_block_size)
    hyper_param["prefix_cache_salt"] = str(prefix_cache_salt)
    return hyper_param


def build_scbench_command(
    *,
    python: str,
    task: str,
    model_name_or_path: str,
    output_dir: Path,
    mode: str,
    num_eval_examples: int,
    max_turns: int,
    max_seq_length: int,
    tensor_parallel_size: int,
    hyper_param: dict[str, Any],
    trust_remote_code: bool,
    use_chat_template: bool,
    ws: int,
    extra_scbench_args: list[str],
) -> list[str]:
    cmd = [
        python,
        "-u",
        str(REPO_ROOT / "benchmark" / "scbench" / "run_scbench.py"),
        "--attn_type",
        "sparseengine",
        "--kv_type",
        "dense",
        "--task",
        task,
        "--model_name_or_path",
        model_name_or_path,
        "--output_dir",
        str(output_dir),
        "--num_eval_examples",
        str(int(num_eval_examples)),
        "--max_turns",
        str(int(max_turns)),
        "--max_seq_length",
        str(int(max_seq_length)),
        "--tensor_parallel_size",
        str(int(tensor_parallel_size)),
        "--ws",
        str(int(ws)),
        "--hyper_param",
        json.dumps(hyper_param, ensure_ascii=False, sort_keys=True),
    ]
    if mode == "scdq":
        cmd.append("--same_context_different_query")
    elif mode != "multi_turn":
        raise ValueError(f"unsupported SCBench mode {mode!r}")
    if trust_remote_code:
        cmd.append("--trust_remote_code")
    if use_chat_template:
        cmd.append("--use_chat_template")
    cmd.extend(extra_scbench_args)
    return cmd


def _find_single_summary(output_dir: Path, task: str, mode: str) -> Path:
    suffix = _mode_suffix(mode)
    matches = sorted(output_dir.rglob(f"prefix_cache_summary_{task}{suffix}.json"))
    if not matches:
        raise FileNotFoundError(
            f"No prefix-cache summary found under {output_dir} for task={task!r}, mode={mode!r}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected one prefix-cache summary under {output_dir}, found {len(matches)}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _run_variant(
    *,
    cmd: list[str],
    output_dir: Path,
    task: str,
    mode: str,
    env: dict[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, env=env)
    wall_time_s = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"SCBench variant failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )
    summary_path = _find_single_summary(output_dir, task, mode)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "success":
        raise RuntimeError(
            f"SCBench prefix-cache summary is not successful: {summary_path} "
            f"status={summary.get('status')!r}"
        )
    summary["summary_path"] = str(summary_path)
    summary["wall_time_s"] = wall_time_s
    summary["command"] = cmd
    return summary


def compare_summaries(off_summary: dict[str, Any], on_summary: dict[str, Any]) -> dict[str, Any]:
    off_request_elapsed = float(off_summary.get("request_elapsed_s", 0.0) or 0.0)
    on_request_elapsed = float(on_summary.get("request_elapsed_s", 0.0) or 0.0)
    off_wall = float(off_summary.get("wall_time_s", 0.0) or 0.0)
    on_wall = float(on_summary.get("wall_time_s", 0.0) or 0.0)
    return {
        "status": "success",
        "cache_off_summary_path": off_summary.get("summary_path"),
        "cache_on_summary_path": on_summary.get("summary_path"),
        "cache_off_wall_time_s": off_wall,
        "cache_on_wall_time_s": on_wall,
        "wall_time_speedup": off_wall / on_wall if on_wall > 0 else 0.0,
        "cache_off_request_elapsed_s": off_request_elapsed,
        "cache_on_request_elapsed_s": on_request_elapsed,
        "request_elapsed_speedup": (
            off_request_elapsed / on_request_elapsed if on_request_elapsed > 0 else 0.0
        ),
        "cache_on_total_cached_tokens": int(on_summary.get("total_cached_tokens", 0) or 0),
        "cache_on_total_cached_blocks": int(on_summary.get("total_cached_blocks", 0) or 0),
        "cache_on_hit_requests": int(on_summary.get("hit_requests", 0) or 0),
        "cache_on_cache_hit_rate": float(on_summary.get("cache_hit_rate", 0.0) or 0.0),
        "cache_on_eligible_cache_hit_rate": float(
            on_summary.get("eligible_cache_hit_rate", 0.0) or 0.0
        ),
        "cache_off_total_cached_tokens": int(off_summary.get("total_cached_tokens", 0) or 0),
        "cache_off_hit_requests": int(off_summary.get("hit_requests", 0) or 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SCBench with SparseEngine prefix caching off/on and summarize reuse and speed."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--task", default="scbench_kv")
    parser.add_argument("--mode", choices=("scdq", "multi_turn", "both"), default="both")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num_eval_examples", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_seq_length", type=int, default=131_072)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--ws", type=int, default=1)
    parser.add_argument("--prefix_cache_block_size", type=int, default=16)
    parser.add_argument("--prefix_cache_salt", default="scbench-prefix-cache-compare")
    parser.add_argument(
        "--master_port_base",
        type=int,
        default=25000,
        help="Base SPARSEENGINE_MASTER_PORT. Each off/on variant gets a distinct port.",
    )
    parser.add_argument("--base_hyper_param", default="{}")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument(
        "--extra_scbench_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed verbatim to run_scbench.py. Put this last.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else REPO_ROOT / "outputs" / "scbench_prefix_cache_compare" / _now_id()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    base_hyper_param = _load_json_arg(args.base_hyper_param)

    mode_results: dict[str, Any] = {}
    for mode_idx, mode in enumerate(_cases_for_modes(args.mode)):
        off_dir = output_root / mode / "prefix_cache_off"
        on_dir = output_root / mode / "prefix_cache_on"
        off_hyper = build_hyper_param(
            base_hyper_param,
            enable_prefix_caching=False,
            prefix_cache_block_size=args.prefix_cache_block_size,
            prefix_cache_salt=args.prefix_cache_salt,
        )
        on_hyper = build_hyper_param(
            base_hyper_param,
            enable_prefix_caching=True,
            prefix_cache_block_size=args.prefix_cache_block_size,
            prefix_cache_salt=args.prefix_cache_salt,
        )

        common = {
            "python": args.python,
            "task": args.task,
            "model_name_or_path": args.model_name_or_path,
            "mode": mode,
            "num_eval_examples": args.num_eval_examples,
            "max_turns": args.max_turns,
            "max_seq_length": args.max_seq_length,
            "tensor_parallel_size": args.tensor_parallel_size,
            "trust_remote_code": args.trust_remote_code,
            "use_chat_template": args.use_chat_template,
            "ws": args.ws,
            "extra_scbench_args": list(args.extra_scbench_args or []),
        }
        off_cmd = build_scbench_command(output_dir=off_dir, hyper_param=off_hyper, **common)
        on_cmd = build_scbench_command(output_dir=on_dir, hyper_param=on_hyper, **common)

        print(f"==== SCBench {mode} prefix cache off ====")
        print(" ".join(off_cmd))
        off_env = dict(os.environ)
        off_env["SPARSEENGINE_MASTER_PORT"] = str(int(args.master_port_base) + mode_idx * 2)
        off_summary = _run_variant(
            cmd=off_cmd,
            output_dir=off_dir,
            task=args.task,
            mode=mode,
            env=off_env,
        )
        print(f"==== SCBench {mode} prefix cache on ====")
        print(" ".join(on_cmd))
        on_env = dict(os.environ)
        on_env["SPARSEENGINE_MASTER_PORT"] = str(int(args.master_port_base) + mode_idx * 2 + 1)
        on_summary = _run_variant(
            cmd=on_cmd,
            output_dir=on_dir,
            task=args.task,
            mode=mode,
            env=on_env,
        )
        mode_results[mode] = {
            "cache_off": off_summary,
            "cache_on": on_summary,
            "comparison": compare_summaries(off_summary, on_summary),
        }

    aggregate = {
        "status": "success",
        "output_dir": str(output_root),
        "task": args.task,
        "model_name_or_path": args.model_name_or_path,
        "mode": args.mode,
        "args": vars(args),
        "results": mode_results,
    }
    (output_root / "comparison_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
