#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_ROOT = Path(os.getenv("SPARSEENGINE_MODEL_ROOT", PROJECT_ROOT / "models"))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("SPARSEENGINE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "multimodal" / "streamingbench_livevlm_table4_7b_vanilla"
DEFAULT_EXPECTED_MODEL_PATH = DEFAULT_MODEL_ROOT / "llava-onevision-qwen2-7b-ov-hf"
VALID_SAMPLE_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}
EXPECTED_VISIBLE_SUBTASKS = {
    "OP": ("Object Perception", 369),
    "CR": ("Causal Reasoning", 128),
    "CS": ("Clips Summarize", 317),
    "ATP": ("Attribute Perception", 312),
    "EU": ("Event Understanding", 159),
    "TR": ("Text-Rich Understanding", 321),
    "PR": ("Prospective Reasoning", 108),
    "SU": ("Spatial Understanding", 246),
    "ACP": ("Action Perception", 352),
    "CT": ("Counting", 188),
    "ER": ("Emotion Recognition", 250),
    "SCU": ("Scene Understanding", 250),
    "SD": ("Source Discrimination", 250),
    "MA": ("Multimodal Alignment", 250),
}
EXPECTED_EXTRA_SUBTASKS = {
    "ACU": ("Anomaly Context Understanding", 250),
    "MCU": ("Misleading Context Recognition", 250),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit a LLaVA-OneVision StreamingBench LiveVLM Table 4 run.")
    parser.add_argument("--metrics_path", default="", help="Path to vanilla_aggregate_metrics.json.")
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory containing vanilla_aggregate_metrics.json when --metrics_path is omitted.",
    )
    parser.add_argument("--json_out", default="", help="Optional path to write the audit summary JSON.")
    parser.add_argument(
        "--require_overall_delta_within_pct",
        type=float,
        default=None,
        help="Optional fail-fast threshold for |observed overall - expected overall| in percentage points.",
    )
    parser.add_argument(
        "--expected_model_path",
        default=str(DEFAULT_EXPECTED_MODEL_PATH),
        help="Expected LLaVA-OneVision-7B model path recorded in run_info.json.",
    )
    parser.add_argument(
        "--expected_attn_implementation",
        default="flash_attention_2",
        help="Expected attention backend recorded in run_info.json.",
    )
    parser.add_argument(
        "--expected_streamingbench_profile",
        default="livevlm_table4",
        help="Expected StreamingBench profile recorded in run_info.json.",
    )
    parser.add_argument(
        "--expected_context_seconds",
        type=float,
        default=-1.0,
        help="Expected context_seconds recorded in run_info.json.",
    )
    return parser.parse_args()


def resolve_metrics_path(metrics_path: str, output_dir: str) -> Path:
    if metrics_path:
        path = Path(metrics_path)
    else:
        path = Path(output_dir) / "vanilla_aggregate_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate metrics file: {path}")
    return path


def require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


def require_number(value, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    return float(value)


def require_file(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{name} is empty: {path}")
    return path


def load_json(path: Path, name: str):
    require_file(path, name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {name} JSON: {path}: {e}") from e


def load_jsonl(path: Path, name: str) -> list[dict]:
    require_file(path, name)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"{name} has an empty line at {path}:{line_number}")
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse {name} JSONL at {path}:{line_number}: {e}") from e
            if not isinstance(value, dict):
                raise TypeError(f"{name} row must be a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def weighted_accuracy_pct(items: list[dict]) -> float | None:
    total = sum(int(item.get("total", 0)) for item in items)
    if total <= 0:
        return None
    correct = sum(int(item.get("correct", 0)) for item in items)
    return 100.0 * correct / total


def expected_task_type_counts() -> dict[str, int]:
    counts = {}
    for task_type, count in EXPECTED_VISIBLE_SUBTASKS.values():
        counts[task_type] = count
    for task_type, count in EXPECTED_EXTRA_SUBTASKS.values():
        counts[task_type] = count
    return counts


def assert_equal(actual, expected, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} mismatch: got {actual!r}, expected {expected!r}")


def audit_subtask_items(items: list[dict], expected: dict[str, tuple[str, int]], group_name: str) -> None:
    abbrs = [str(item.get("abbr", "")) for item in items]
    duplicates = sorted({abbr for abbr in abbrs if abbrs.count(abbr) > 1})
    if duplicates:
        raise RuntimeError(f"{group_name} subtask mismatch: duplicate abbreviations={duplicates}")

    by_abbr = {abbr: item for abbr, item in zip(abbrs, items)}
    missing = [abbr for abbr in expected if abbr not in by_abbr]
    unexpected = sorted(abbr for abbr in by_abbr if abbr not in expected)
    if missing or unexpected:
        raise RuntimeError(f"{group_name} subtask mismatch: missing={missing}, unexpected={unexpected}")

    mismatched = {}
    for abbr, (task_type, expected_rows) in expected.items():
        item = by_abbr[abbr]
        got_task_type = item.get("task_type")
        got_rows = int(require_number(item.get("total"), f"{abbr}.total"))
        recorded_expected_rows = int(require_number(item.get("expected_rows"), f"{abbr}.expected_rows"))
        if (
            got_task_type != task_type
            or got_rows != expected_rows
            or recorded_expected_rows != expected_rows
            or item.get("matches_expected_rows") is not True
        ):
            mismatched[abbr] = {
                "task_type": got_task_type,
                "expected_task_type": task_type,
                "total": got_rows,
                "expected_rows": expected_rows,
                "recorded_expected_rows": recorded_expected_rows,
                "matches_expected_rows": item.get("matches_expected_rows"),
            }
    if mismatched:
        raise RuntimeError(f"{group_name} subtask row-count mismatch: {mismatched}")


def audit_run_info(
    run_info: dict,
    expected_model_path: str,
    expected_rows: int,
    expected_attn_implementation: str,
    expected_streamingbench_profile: str,
    expected_context_seconds: float,
) -> dict:
    required = {
        "model_path": expected_model_path,
        "methods": "vanilla",
        "streamingbench_profile": expected_streamingbench_profile,
        "tasks": "livevlm_table4",
        "num_video_frames": 32,
        "context_seconds": expected_context_seconds,
        "frame_sampling_backend": "decord",
        "choice_parse_mode": "official_first_char",
        "sample_start": 0,
        "num_samples_arg": -1,
        "evaluated_sample_count": expected_rows,
        "seed": 0,
    }
    for key, expected in required.items():
        assert_equal(run_info.get(key), expected, f"run_info.{key}")

    decoding = require_mapping(run_info.get("decoding"), "run_info.decoding")
    assert_equal(decoding.get("max_new_tokens"), 8, "run_info.decoding.max_new_tokens")
    assert_equal(decoding.get("do_sample"), False, "run_info.decoding.do_sample")
    assert_equal(decoding.get("torch_dtype"), "float16", "run_info.decoding.torch_dtype")
    assert_equal(
        decoding.get("attn_implementation"),
        expected_attn_implementation,
        "run_info.decoding.attn_implementation",
    )

    dataset_info = require_mapping(run_info.get("dataset_info"), "run_info.dataset_info")
    task_counts = require_mapping(dataset_info.get("evaluated_task_type_counts"), "run_info.dataset_info.evaluated_task_type_counts")
    expected_counts = expected_task_type_counts()
    mismatched_counts = {
        task_type: {"got": int(task_counts.get(task_type, 0)), "expected": expected}
        for task_type, expected in expected_counts.items()
        if int(task_counts.get(task_type, 0)) != expected
    }
    if mismatched_counts:
        raise RuntimeError(f"run_info dataset task-type counts mismatch: {mismatched_counts}")

    return {
        "model_path": run_info.get("model_path"),
        "command": run_info.get("command"),
        "git_commit": run_info.get("git_commit"),
        "dataset_dir": run_info.get("dataset_dir"),
        "video_dir": run_info.get("video_dir"),
    }


def audit_jsonl_artifacts(output_dir: Path, expected_rows: int) -> dict:
    raw_rows = load_jsonl(output_dir / "vanilla_raw_outputs.jsonl", "raw outputs")
    parsed_rows = load_jsonl(output_dir / "vanilla_parsed_outputs.jsonl", "parsed outputs")
    per_sample_rows = load_jsonl(output_dir / "vanilla_per_sample_results.jsonl", "per-sample results")
    assert_equal(len(raw_rows), expected_rows, "raw output row count")
    assert_equal(len(parsed_rows), expected_rows, "parsed output row count")
    assert_equal(len(per_sample_rows), expected_rows, "per-sample row count")

    raw_ids = set()
    for idx, row in enumerate(raw_rows):
        question_id = row.get("question_id")
        if question_id in raw_ids:
            raise RuntimeError(f"Duplicate raw question_id={question_id!r}")
        raw_ids.add(question_id)
        if "raw_prediction" not in row:
            raise RuntimeError(f"Raw output row {idx} is missing raw_prediction.")

    parsed_status_counts: dict[str, int] = {}
    parsed_ids = set()
    for idx, row in enumerate(parsed_rows):
        question_id = row.get("question_id")
        if question_id in parsed_ids:
            raise RuntimeError(f"Duplicate parsed question_id={question_id!r}")
        parsed_ids.add(question_id)
        status = row.get("status")
        if status not in VALID_SAMPLE_STATUSES:
            raise RuntimeError(f"Invalid parsed status at row {idx}: {status!r}")
        parsed_status_counts[status] = parsed_status_counts.get(status, 0) + 1
        if row.get("answer") not in {"A", "B", "C", "D"}:
            raise RuntimeError(f"Invalid parsed answer at row {idx}: {row.get('answer')!r}")

    per_sample_status_counts: dict[str, int] = {}
    task_type_counts: dict[str, int] = {}
    per_sample_ids = set()
    for idx, row in enumerate(per_sample_rows):
        question_id = row.get("question_id")
        if question_id in per_sample_ids:
            raise RuntimeError(f"Duplicate per-sample question_id={question_id!r}")
        per_sample_ids.add(question_id)
        status = row.get("status")
        if status not in VALID_SAMPLE_STATUSES:
            raise RuntimeError(f"Invalid per-sample status at row {idx}: {status!r}")
        per_sample_status_counts[status] = per_sample_status_counts.get(status, 0) + 1
        task_type = str(row.get("task_type", ""))
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        if row.get("answer") not in {"A", "B", "C", "D"}:
            raise RuntimeError(f"Invalid per-sample answer at row {idx}: {row.get('answer')!r}")
        if "raw_prediction" not in row or "parsed_text" not in row:
            raise RuntimeError(f"Per-sample row {idx} is missing raw/parsed prediction fields.")

    if raw_ids != parsed_ids or raw_ids != per_sample_ids:
        raise RuntimeError(
            "Artifact question_id sets differ across raw/parsed/per-sample files: "
            f"raw={len(raw_ids)}, parsed={len(parsed_ids)}, per_sample={len(per_sample_ids)}"
        )

    if parsed_status_counts != per_sample_status_counts:
        raise RuntimeError(
            "Parsed/per-sample status count mismatch: "
            f"parsed={parsed_status_counts}, per_sample={per_sample_status_counts}"
        )

    expected_counts = expected_task_type_counts()
    mismatched_task_types = {
        task_type: {"got": task_type_counts.get(task_type, 0), "expected": expected}
        for task_type, expected in expected_counts.items()
        if task_type_counts.get(task_type, 0) != expected
    }
    if mismatched_task_types:
        raise RuntimeError(f"Per-sample task-type counts mismatch: {mismatched_task_types}")

    return {
        "raw_output_rows": len(raw_rows),
        "parsed_output_rows": len(parsed_rows),
        "per_sample_rows": len(per_sample_rows),
        "status_counts_from_jsonl": per_sample_status_counts,
    }


def audit_output_artifacts(
    output_dir: Path,
    metrics: dict,
    expected_model_path: str,
    expected_attn_implementation: str,
    expected_streamingbench_profile: str,
    expected_context_seconds: float,
) -> dict:
    stats = require_mapping(metrics.get("livevlm_table4_stats"), "livevlm_table4_stats")
    expected_rows = int(require_number(stats.get("expected_overall_row_count"), "expected_overall_row_count"))
    run_info = require_mapping(load_json(output_dir / "run_info.json", "run_info"), "run_info")
    last_result = load_json(output_dir / "last_streamingbench_result.json", "last_streamingbench_result")
    if not isinstance(last_result, list) or len(last_result) != 1:
        raise RuntimeError("last_streamingbench_result.json must contain exactly one method result for baseline audit.")
    assert_equal(last_result[0].get("method"), "vanilla", "last_streamingbench_result[0].method")
    assert_equal(last_result[0].get("num_samples"), expected_rows, "last_streamingbench_result[0].num_samples")

    artifact_summary = audit_jsonl_artifacts(output_dir, expected_rows)
    metrics_status_counts = require_mapping(metrics.get("status_counts"), "aggregate_metrics.status_counts")
    overall = require_mapping(stats.get("overall"), "livevlm_table4_stats.overall")
    overall_status_counts = require_mapping(overall.get("status_counts"), "livevlm_table4_stats.overall.status_counts")
    status_counts_from_jsonl = artifact_summary["status_counts_from_jsonl"]
    assert_equal(
        {key: int(value) for key, value in metrics_status_counts.items()},
        status_counts_from_jsonl,
        "aggregate_metrics.status_counts",
    )
    assert_equal(
        {key: int(value) for key, value in overall_status_counts.items()},
        status_counts_from_jsonl,
        "livevlm_table4_stats.overall.status_counts",
    )
    run_info_summary = audit_run_info(
        run_info,
        expected_model_path,
        expected_rows,
        expected_attn_implementation,
        expected_streamingbench_profile,
        expected_context_seconds,
    )
    return {
        "output_dir": str(output_dir),
        **artifact_summary,
        "run_info": run_info_summary,
    }


def audit_metrics(metrics: dict, require_overall_delta_within_pct: float | None = None) -> dict:
    stats = require_mapping(metrics.get("livevlm_table4_stats"), "livevlm_table4_stats")
    overall = require_mapping(stats.get("overall"), "livevlm_table4_stats.overall")
    subtasks = stats.get("subtasks")
    extra_subtasks = stats.get("overall_extra_subtasks")
    if not isinstance(subtasks, list) or len(subtasks) != 14:
        raise RuntimeError(f"Expected 14 visible LiveVLM Table 4 subtasks, got {len(subtasks) if isinstance(subtasks, list) else type(subtasks).__name__}")
    if not isinstance(extra_subtasks, list) or len(extra_subtasks) != 2:
        raise RuntimeError(
            "Expected 2 overall-only LiveVLM Table 4 subtasks for ACU/MCU, "
            f"got {len(extra_subtasks) if isinstance(extra_subtasks, list) else type(extra_subtasks).__name__}"
        )
    audit_subtask_items(subtasks, EXPECTED_VISIBLE_SUBTASKS, "visible")
    audit_subtask_items(extra_subtasks, EXPECTED_EXTRA_SUBTASKS, "overall-only")

    expected_rows = int(require_number(stats.get("expected_overall_row_count"), "expected_overall_row_count"))
    overall_total = int(require_number(overall.get("total"), "overall.total"))
    if overall_total != expected_rows:
        raise RuntimeError(f"Overall row count mismatch: got {overall_total}, expected {expected_rows}")
    if overall.get("matches_expected_row_count") is not True:
        raise RuntimeError("overall.matches_expected_row_count must be true for a baseline audit")

    missing_visible = [item.get("abbr", item.get("task_type", "?")) for item in subtasks if int(item.get("total", 0)) <= 0]
    missing_extra = [item.get("abbr", item.get("task_type", "?")) for item in extra_subtasks if int(item.get("total", 0)) <= 0]
    if missing_visible or missing_extra:
        raise RuntimeError(f"Missing evaluated rows for subtasks: visible={missing_visible}, extra={missing_extra}")

    observed_overall = require_number(overall.get("accuracy_pct"), "overall.accuracy_pct")
    expected_overall = require_number(stats.get("expected_llava_onevision_7b_overall_pct"), "expected_llava_onevision_7b_overall_pct")
    overall_delta = observed_overall - expected_overall
    if require_overall_delta_within_pct is not None and abs(overall_delta) > require_overall_delta_within_pct:
        raise RuntimeError(
            f"Overall delta {overall_delta:.4f} pct exceeds threshold "
            f"{require_overall_delta_within_pct:.4f} pct"
        )

    visible_deltas = [
        require_number(item.get("delta_vs_expected_pct"), f"{item.get('abbr')}.delta_vs_expected_pct")
        for item in subtasks
    ]
    summary = {
        "method": metrics.get("method"),
        "num_samples": metrics.get("num_samples"),
        "overall_accuracy_pct": observed_overall,
        "expected_overall_accuracy_pct": expected_overall,
        "overall_delta_vs_expected_pct": overall_delta,
        "overall_total": overall_total,
        "expected_overall_row_count": expected_rows,
        "status_counts": overall.get("status_counts", {}),
        "visible_subtask_count": len(subtasks),
        "overall_extra_subtask_count": len(extra_subtasks),
        "max_abs_visible_subtask_delta_pct": max(abs(delta) for delta in visible_deltas),
        "expected_display_weighted_accuracy_pct": stats.get("expected_display_weighted_accuracy_pct"),
        "implied_expected_extra_subtasks_accuracy_pct": stats.get("implied_expected_extra_subtasks_accuracy_pct"),
        "observed_extra_subtasks_accuracy_pct": weighted_accuracy_pct(extra_subtasks),
        "subtasks": subtasks,
        "overall_extra_subtasks": extra_subtasks,
    }
    return summary


def print_summary(summary: dict) -> None:
    print(
        "overall "
        f"observed={summary['overall_accuracy_pct']:.4f} "
        f"expected={summary['expected_overall_accuracy_pct']:.4f} "
        f"delta={summary['overall_delta_vs_expected_pct']:.4f} "
        f"rows={summary['overall_total']}"
    )
    print(
        "extra "
        f"observed={summary['observed_extra_subtasks_accuracy_pct']:.4f} "
        f"implied_expected={summary['implied_expected_extra_subtasks_accuracy_pct']:.4f}"
    )
    for item in summary["subtasks"]:
        print(
            f"{item['abbr']:>3} {item['accuracy_pct']:7.3f} "
            f"expected={item['expected_llava_onevision_7b_pct']:7.3f} "
            f"delta={item['delta_vs_expected_pct']:7.3f} "
            f"total={item['total']:4d} {item['task_type']}"
        )
    for item in summary["overall_extra_subtasks"]:
        print(f"{item['abbr']:>3} {item['accuracy_pct']:7.3f} total={item['total']:4d} {item['task_type']}")


def main():
    args = parse_args()
    metrics_path = resolve_metrics_path(args.metrics_path, args.output_dir)
    metrics = require_mapping(load_json(metrics_path, "aggregate metrics"), str(metrics_path))
    summary = audit_metrics(metrics, require_overall_delta_within_pct=args.require_overall_delta_within_pct)
    artifact_summary = audit_output_artifacts(
        metrics_path.parent,
        metrics,
        args.expected_model_path,
        args.expected_attn_implementation,
        args.expected_streamingbench_profile,
        args.expected_context_seconds,
    )
    summary["artifact_audit"] = artifact_summary
    print_summary(summary)
    print(
        "artifacts "
        f"raw={artifact_summary['raw_output_rows']} "
        f"parsed={artifact_summary['parsed_output_rows']} "
        f"per_sample={artifact_summary['per_sample_rows']}"
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
